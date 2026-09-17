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
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)

    # Mix-policy 配置
    # 启用后，对于有预采集轨迹的样本，使用 n-1 个在线生成 + 1 个离线轨迹
    # 对于没有预采集轨迹的样本，使用 n 个在线生成
    enable_mix_policy: bool = False

    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    train_response_length: int = field(default=-1, init=False)
    """Training rollout max tokens / default padding length."""
    response_length: int = field(default=-1, init=False)
    """Max of train/val response lengths; used for vLLM max_model_len budget."""
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
