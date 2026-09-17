# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
OPSD + oracle-correction FSDP worker.

Loads a third frozen homologous model (oracle) from ``actor.oracle_model_path``.
Oracle shares the **student prompt** (no privileged hint). Distillation still uses
the OPSD teacher (hinted) as the top-k support; oracle only reweights that support.
"""

from typing import Optional, cast

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base.decorator import Dispatch, register
from ..utils.fsdp_utils import get_fsdp_wrap_policy, get_init_fn, load_fsdp_model, offload_fsdp_model
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.torch_dtypes import PrecisionType
from .fsdp_workers import FSDPWorker


class OPSDOracleFSDPWorker(FSDPWorker):
    """FSDP worker that colocates student + OPSD teacher (ref) + frozen oracle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_fsdp_module = None
        self._use_oracle_param_offload = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        oracle_path = getattr(self.config.actor, "oracle_model_path", None)
        if not oracle_path:
            return
        if not self._has_actor:
            return

        self._build_oracle_model(oracle_path)
        if hasattr(self, "actor") and self.actor is not None:
            self.actor.oracle_module = self.oracle_fsdp_module
            self.print_rank0(
                f"[OPSD-Oracle] Attached oracle module from {oracle_path}"
            )

    def _build_oracle_model(self, oracle_path: str) -> None:
        assert self.tokenizer is not None and self.model_config is not None, (
            "Actor must be initialized before the oracle model."
        )
        model_config = self.config.actor.model
        # Prefer ref FSDP settings; fall back to actor.
        fsdp_config = self.config.ref.fsdp if self._has_ref else self.config.actor.fsdp
        padding_free = self.config.ref.padding_free if self._has_ref else self.config.actor.padding_free

        oracle_hf_config = AutoConfig.from_pretrained(
            oracle_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )
        student_vocab = int(getattr(self.model_config, "vocab_size", -1))
        oracle_vocab = int(getattr(oracle_hf_config, "vocab_size", -1))
        assert_same_vocab = getattr(self.config.actor, "oracle_assert_same_vocab", True)
        if assert_same_vocab and student_vocab != oracle_vocab:
            raise ValueError(
                f"Oracle correction requires identical vocab_size. student={student_vocab}, "
                f"oracle={oracle_vocab} (path={oracle_path})."
            )
        self.print_rank0(
            f"[OPSD-Oracle] Loading oracle from {oracle_path} "
            f"(vocab={oracle_vocab}, hidden={getattr(oracle_hf_config, 'hidden_size', '?')}, "
            f"layers={getattr(oracle_hf_config, 'num_hidden_layers', '?')})"
        )

        if padding_free:
            apply_ulysses_patch(oracle_hf_config.model_type)
            self.print_rank0("[OPSD-Oracle] Ulysses patch applied for oracle.")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if type(oracle_hf_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                oracle_path,
                config=oracle_hf_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    oracle_hf_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)
        model.tie_weights()
        model = model.to(torch_dtype)
        model.requires_grad_(False)

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After OPSD oracle huggingface init")

        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"[OPSD-Oracle] Oracle FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            sharding_strategy = (
                ShardingStrategy.HYBRID_SHARD
                if fsdp_config.enable_full_shard
                else ShardingStrategy._HYBRID_SHARD_ZERO2
            )
        else:
            sharding_strategy = (
                ShardingStrategy.FULL_SHARD
                if fsdp_config.enable_full_shard
                else ShardingStrategy.SHARD_GRAD_OP
            )

        # Default: CPU-offload oracle params to leave room for student+teacher+vLLM.
        force_cpu_offload = getattr(self.config.actor, "oracle_enable_cpu_offload", True)
        enable_cpu_offload = bool(fsdp_config.enable_cpu_offload or force_cpu_offload)
        cpu_offload = CPUOffload(offload_params=True) if enable_cpu_offload else None
        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        self.oracle_fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        self.oracle_model_config = oracle_hf_config
        print_gpu_memory_usage("After OPSD oracle FSDP init")

        # Manual OffloadConfig path (in addition to FSDP CPUOffload).
        self._use_oracle_param_offload = bool(
            getattr(self.config.actor, "oracle_offload_params", False)
        )
        if self._use_oracle_param_offload:
            offload_fsdp_model(self.oracle_fsdp_module)
            print_gpu_memory_usage("After offload OPSD oracle during init")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        if self._use_oracle_param_offload and self.oracle_fsdp_module is not None:
            load_fsdp_model(self.oracle_fsdp_module)
        try:
            return super().update_actor(data)
        finally:
            if self._use_oracle_param_offload and self.oracle_fsdp_module is not None:
                offload_fsdp_model(self.oracle_fsdp_module)
