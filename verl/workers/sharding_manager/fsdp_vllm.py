# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import re
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        # Record freed bytes to estimate memory usage correctly
        # https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
        self.freed_bytes = 0

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights

    @staticmethod
    def _materialize_tensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
        """Materialize FSDP/DTensor state-dict entries to a plain CPU tensor."""
        if isinstance(tensor, DTensor):
            # cpu_offload state dict → DTensor on CPU; full_tensor() needs CUDA collectives
            if tensor.device.type != "cuda":
                tensor = tensor.to("cuda")
            return tensor.full_tensor().detach().cpu()
        return tensor.detach().cpu()

    @staticmethod
    def _to_vllm_key(key: str) -> str:
        vllm_key = key.replace("base_model.model.", "model.", 1)
        if vllm_key.startswith("model.model."):
            vllm_key = vllm_key.replace("model.model.", "model.", 1)
        return vllm_key

    @classmethod
    def _convert_peft_weights_to_vllm(
        cls,
        actor_weights: dict[str, Union[torch.Tensor, DTensor]],
        *,
        lora_alpha: float,
        lora_r: int,
    ) -> dict[str, torch.Tensor]:
        """Merge LoRA deltas and strip PEFT prefixes for vLLM weight loading."""
        scale = float(lora_alpha) / float(lora_r)
        base_layers: dict[str, str] = {}
        lora_a: dict[str, str] = {}
        lora_b: dict[str, str] = {}
        passthrough: dict[str, str] = {}

        for key in actor_weights:
            if ".lora_A." in key:
                lora_a[key.split(".lora_A.")[0]] = key
            elif ".lora_B." in key:
                lora_b[key.split(".lora_B.")[0]] = key
            elif ".base_layer.weight" in key:
                base_layers[key.rsplit(".base_layer.weight", 1)[0]] = key
            elif "lora_" in key:
                continue
            else:
                passthrough[key] = key

        converted: dict[str, torch.Tensor] = {}
        for in_key in passthrough:
            if ".lm_head." in in_key:
                # Qwen3 ties lm_head to embed_tokens; vLLM skips top-level lm_head.* only.
                continue
            converted[cls._to_vllm_key(in_key)] = cls._materialize_tensor(actor_weights[in_key])

        for prefix, base_key in base_layers.items():
            vllm_key = cls._to_vllm_key(base_key.replace(".base_layer.weight", ".weight"))
            base_tensor = cls._materialize_tensor(actor_weights[base_key])
            if prefix in lora_a and prefix in lora_b:
                a = cls._materialize_tensor(actor_weights[lora_a[prefix]])
                b = cls._materialize_tensor(actor_weights[lora_b[prefix]])
                delta = b @ a
                converted[vllm_key] = base_tensor + delta.to(base_tensor.dtype) * scale
            else:
                converted[vllm_key] = base_tensor

        return converted

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            if isinstance(tensor, DTensor):
                # cpu_offload state dict → DTensor on CPU; full_tensor() needs CUDA collectives
                if tensor.device.type != "cuda":
                    tensor = tensor.to("cuda")
                tensor = tensor.full_tensor()
            elif isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                tensor = tensor.to("cuda", non_blocking=True)
            yield name, tensor

    def _gather_actor_weights(self):
        """Collect FSDP weights, preferably on CPU, so vLLM wake_up can reuse the GPU."""
        if self.use_param_offload:
            load_fsdp_model(self.module)

        actor_weights = get_model_state_dict(
            self.module,
            options=StateDictOptions(cpu_offload=True),
        )
        actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)
        print_gpu_memory_usage("After gather model weights in sharding manager")

        if self.use_param_offload:
            offload_fsdp_model(self.module)
            torch.cuda.empty_cache()
        return actor_weights

    def _sync_weight_to_vllm(
        self, actor_weights: Union[dict[str, Union[torch.Tensor, DTensor]], None] = None
    ):
        if actor_weights is None:
            actor_weights = self._gather_actor_weights()

        wrapped = self.module._fsdp_wrapped_module
        from peft import PeftModel

        if isinstance(wrapped, PeftModel):
            peft_cfg = wrapped.peft_config["default"]
            actor_weights = self._convert_peft_weights_to_vllm(
                actor_weights,
                lora_alpha=float(peft_cfg.lora_alpha),
                lora_r=int(peft_cfg.r),
            )

        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(self._make_weight_iterator(actor_weights))

        del actor_weights
        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in sharding manager")

    def load_vllm_and_sync_weights(self):
        """Load vllm engine and sync model weights to vllm model."""
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")

        # With param offload, gather *before* wake_up. Otherwise FSDP unshard + vLLM
        # weight buffers coexist and 32B models OOM (get_model_state_dict clone).
        actor_weights = None
        if self.use_param_offload:
            actor_weights = self._gather_actor_weights()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm(actor_weights)

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in sharding manager")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        """Offload vllm engine."""
        assert self.loaded is True, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()  # add empty cache after each compute

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]

        return data
