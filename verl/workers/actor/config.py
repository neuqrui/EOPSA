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
"""
Actor config
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ActorLoraConfig:
    """ThinkSafe-aligned LoRA (grpo_train.py defaults)."""
    enable: bool = False
    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    lora: ActorLoraConfig = field(default_factory=ActorLoraConfig)

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    lr_scheduler_type: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating actor"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.3
    """clip ratio in PPO & DAPO"""
    clip_ratio_dual: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    loss_type: str = "default"
    """loss type: `default`, `gspo`, `cispo`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    max_token_len_per_gpu: Optional[int] = None
    """max token length per GPU for dynamic batching. If None, use micro_batch_size * max_seq_len"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    """enable torch compile"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # Oracle correction (optional; set by opsd trainer / CLI when enable_oracle_correction)
    oracle_model_path: Optional[str] = None
    """HF path of frozen oracle; attached by OPSDOracleFSDPWorker."""
    oracle_assert_same_vocab: bool = True
    oracle_enable_cpu_offload: bool = True
    """If True, wrap oracle FSDP with CPUOffload(offload_params=True) (layer-wise H2D/D2H)."""
    oracle_offload_params: bool = False
    """If True, manually offload whole oracle between update_actor calls (extra slow)."""
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)
    opsd_enabled: bool = field(default=False, init=False)
    """Auto-set by trainer when OPSD mode is active."""
    freeze_teacher_model: bool = field(default=False, init=False)


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    max_token_len_per_gpu: Optional[int] = field(default=None, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
