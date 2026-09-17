# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
OPD FSDP worker: colocated student + homologous teacher on the same GPUs.

Subclasses FSDPWorker; does not modify FSDPWorker source. Teacher is loaded from
``actor.teacher_model_path`` with its own AutoConfig (e.g. Qwen3-1.7B ← Qwen3-4B).
"""

from typing import Literal, Optional, cast

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
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import get_fsdp_wrap_policy, get_init_fn, offload_fsdp_model
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.torch_dtypes import PrecisionType
from .config import FSDPConfig, ModelConfig, OptimConfig
from .fsdp_workers import FSDPWorker


class OPDFSDPWorker(FSDPWorker):
    """FSDP worker that loads a separate homologous teacher as the ref module."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Init order: student → vLLM rollout → teacher.

        Building the larger teacher before vLLM makes memory profiling flaky when
        other jobs share the GPU (free memory can rise mid-profile).
        """
        from .actor.dp_opsd_actor import DataParallelOPSDActor
        from .actor.dp_actor import DataParallelPPOActor

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )
            # Teacher not loaded yet; attach after rollout init.
            self.actor = DataParallelOPSDActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
                teacher_module=None,
            )
            self.actor.tokenizer = self.tokenizer

        if self._has_rollout:
            self._build_rollout()

        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )
            if self._has_actor and getattr(self.config.actor, "freeze_teacher_model", False):
                self.actor.teacher_module = self.ref_fsdp_module
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
            )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        teacher_path = getattr(self.config.actor, "teacher_model_path", None)
        if role != "ref" or not teacher_path:
            return super()._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
                role=role,
            )

        assert self.tokenizer is not None and self.model_config is not None, (
            "Actor must be initialized before the OPD teacher (ref)."
        )

        teacher_hf_config = AutoConfig.from_pretrained(
            teacher_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )
        student_vocab = int(getattr(self.model_config, "vocab_size", -1))
        teacher_vocab = int(getattr(teacher_hf_config, "vocab_size", -1))
        assert_same_vocab = getattr(self.config.actor, "opd_assert_same_vocab", True)
        if assert_same_vocab and student_vocab != teacher_vocab:
            raise ValueError(
                f"OPD requires identical vocab_size. student={student_vocab}, "
                f"teacher={teacher_vocab} (path={teacher_path})."
            )
        self.print_rank0(
            f"[OPD] Loading homologous teacher from {teacher_path} "
            f"(vocab={teacher_vocab}, hidden={getattr(teacher_hf_config, 'hidden_size', '?')}, "
            f"layers={getattr(teacher_hf_config, 'num_hidden_layers', '?')})"
        )

        if padding_free:
            apply_ulysses_patch(teacher_hf_config.model_type)
            self.print_rank0("[OPD] Ulysses patch applied for teacher.")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if type(teacher_hf_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                teacher_path,
                config=teacher_hf_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    teacher_hf_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)
        model.tie_weights()
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.requires_grad_(False)

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After OPD teacher huggingface init")

        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"[OPD] Teacher FSDP wrap policy: {auto_wrap_policy}.")

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

        cpu_offload = CPUOffload(offload_params=True) if fsdp_config.enable_cpu_offload else None
        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        self.ref_fsdp_module = FSDP(
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
        print_gpu_memory_usage("After OPD teacher FSDP init")
        self.teacher_model_config = teacher_hf_config

        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_fsdp_module)
            print_gpu_memory_usage("After offload OPD teacher during init")
