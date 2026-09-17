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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig
from .opsd_config import OPSDConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    video_max_frames: int = 128
    max_prompt_length: int = 512
    max_response_length: int = 512
    val_max_response_length: Optional[int] = None
    """Validation rollout max tokens. Defaults to max_response_length when unset."""
    enable_dynamic_max_response_length: bool = False
    """If True, student rollout max tokens follow the 3-phase schedule below by global_step
    for ALL train samples (harmful + benign)."""
    dynamic_max_response_length_phase1_end: int = 100
    """Steps 1..phase1_end use phase1 length (inclusive)."""
    dynamic_max_response_length_phase1: int = 256
    dynamic_max_response_length_phase2_end: int = 300
    """Steps (phase1_end+1)..phase2_end use phase2 length (inclusive)."""
    dynamic_max_response_length_phase2: int = 512
    dynamic_max_response_length_phase3: int = 1024
    """Steps after phase2_end use phase3 length until training ends."""
    enable_adaptive_rollout: bool = False
    """After each validation, probe Teacher Rescue Rate (TRR) at fixed prefix lengths and set
    student rollout max tokens for ALL train samples (harmful + benign). Takes precedence
    over the static 3-phase schedule when enabled.
    """
    adaptive_rollout_prefix_lengths: Optional[list[int]] = None
    """Prefix lengths truncated from the student val-length generation for teacher rescue probe.
    Student probe rollout uses ``val_max_response_length`` (same as validation); only these
    prefixes are evaluated. Default: [128, 256, 1024]."""
    adaptive_rollout_min_student_fails: int = 10
    """If student defense fails on fewer than this many probe samples, skip teacher
    continuations and keep the current rollout length (student good enough)."""
    adaptive_rollout_trr_thresh: float = 0.30
    """Keep the longest probed L whose teacher DSR (safe rate over the full probe set)
    is still >= this threshold. Denominator is all probe samples, not the old
    student-unsafe ∩ teacher-alone-safe intersect."""
    adaptive_rollout_max_samples: int = 64
    """Max number of probe prompts (val or extra set) per ARS check."""
    adaptive_rollout_files: Optional[str] = None
    """Optional extra jsonl for the ARS probe. None → reuse data.val_files."""
    adaptive_rollout_monotonic: bool = True
    """If True, never decrease rollout length mid-training."""
    adaptive_rollout_teacher_max_tokens: int = 1024
    """Max new tokens for teacher continuation / alone generation during the probe."""
    adaptive_rollout_skip_first: bool = True
    """If True, skip ARS length-update when ``global_step == 0`` only
    (not the first mid-training val when val_before_train is false)."""
    adaptive_rollout_skip_last: bool = True
    """If True, skip ARS length-update when ``global_step == training_steps``."""
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    apply_chat_template_kwargs: Optional[dict] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    image_min_pixels: Optional[int] = None
    image_max_pixels: Optional[int] = None
    video_min_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None
    trajectory_key: Optional[str] = None
    """Key for large model trajectory in training data (for OPSD). None = no trajectory."""
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    use_preprocessed_videos: bool = True
    """whether to use preprocessed video files (.pt) if available"""
    preprocessed_video_dir: Optional[str] = None
    """directory containing preprocessed video files (.pt)"""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        self.preprocessed_video_dir = get_abs_path(self.preprocessed_video_dir, prompt="Preprocessed video directory")
        if self.image_min_pixels is None:
            self.image_min_pixels = self.min_pixels
        if self.image_max_pixels is None:
            self.image_max_pixels = self.max_pixels
        if self.video_min_pixels is None:
            self.video_min_pixels = self.min_pixels
        if self.video_max_pixels is None:
            self.video_max_pixels = self.max_pixels
        if self.enable_dynamic_max_response_length:
            if self.dynamic_max_response_length_phase1_end <= 0:
                raise ValueError(
                    "data.dynamic_max_response_length_phase1_end must be > 0 when dynamic schedule is enabled."
                )
            if self.dynamic_max_response_length_phase2_end <= self.dynamic_max_response_length_phase1_end:
                raise ValueError(
                    "data.dynamic_max_response_length_phase2_end must be > "
                    "data.dynamic_max_response_length_phase1_end."
                )
            for name in (
                "dynamic_max_response_length_phase1",
                "dynamic_max_response_length_phase2",
                "dynamic_max_response_length_phase3",
            ):
                if getattr(self, name) <= 0:
                    raise ValueError(f"data.{name} must be > 0.")
            peak = max(
                self.dynamic_max_response_length_phase1,
                self.dynamic_max_response_length_phase2,
                self.dynamic_max_response_length_phase3,
            )
            if self.max_response_length < peak:
                self.max_response_length = peak

        if self.adaptive_rollout_prefix_lengths is None:
            self.adaptive_rollout_prefix_lengths = [128, 256, 1024]
        else:
            self.adaptive_rollout_prefix_lengths = [int(x) for x in self.adaptive_rollout_prefix_lengths]
            if not self.adaptive_rollout_prefix_lengths:
                raise ValueError("data.adaptive_rollout_prefix_lengths must be non-empty when provided.")
        if self.enable_adaptive_rollout:
            if self.adaptive_rollout_min_student_fails < 0:
                raise ValueError("data.adaptive_rollout_min_student_fails must be >= 0.")
            if not (0.0 <= float(self.adaptive_rollout_trr_thresh) <= 1.0):
                raise ValueError("data.adaptive_rollout_trr_thresh must be in [0, 1].")
            if self.adaptive_rollout_max_samples <= 0:
                raise ValueError("data.adaptive_rollout_max_samples must be > 0.")
            peak = max(self.adaptive_rollout_prefix_lengths)
            if self.max_response_length < peak:
                self.max_response_length = peak
        self.adaptive_rollout_files = get_abs_path(self.adaptive_rollout_files, prompt="ARS probe data file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    pipeline_reward: bool = False
    """流水线 reward 计算：rollout 过程中边生成边提交 reward，减少等待时间"""
    enable_split_safety_advantage: bool = False
    """Split think/answer GRPO advantage for safety RLSD (requires think_safe/ans_safe rewards)."""
    split_safety_adv_eps: float = 1e-6
    """Epsilon for split safety GRPO normalization."""
    split_safety_adv_unified_scale: bool = False
    """Apply a unified batch normalization across all segments after per-segment GRPO normalization."""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    opsd: OPSDConfig = field(default_factory=OPSDConfig)

    def post_init(self):
        if self.data.enable_dynamic_max_response_length:
            train_response_length = max(
                self.data.dynamic_max_response_length_phase1,
                self.data.dynamic_max_response_length_phase2,
                self.data.dynamic_max_response_length_phase3,
            )
        elif self.data.enable_adaptive_rollout:
            train_response_length = max(
                self.data.max_response_length,
                max(self.data.adaptive_rollout_prefix_lengths or [128, 256, 1024]),
            )
        else:
            train_response_length = self.data.max_response_length
        val_response_length = (
            self.data.val_max_response_length
            if self.data.val_max_response_length is not None
            else self.data.max_response_length
        )
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.train_response_length = train_response_length
        self.worker.rollout.response_length = max(train_response_length, val_response_length)
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
