# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
OPD (On-Policy Distillation) configuration.

Homologous teacher/student (same tokenizer / vocab). Independent from OPSDConfig.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig
from .config import AlgorithmConfig, DataConfig, TrainerConfig, recursive_post_init


@dataclass
class OPDConfig:
    """Pure on-policy distillation with an external homologous teacher."""

    teacher_model_path: str = ""
    """HF path of the frozen teacher (same family / vocab as the student)."""

    distillation_loss_type: str = "kl"
    """kl | jsd | topk_jsd | topk_forward_kl | topk_reverse_kl."""

    opsd_kl_penalty: str = "kl"
    """Only 'kl' is supported for sampled-token PG OPD."""

    distillation_topk: Optional[int] = None
    """Top-k support size. None = sampled-token PG KL (no top-k logits)."""

    distillation_topk_source: str = "teacher"
    distillation_add_tail: bool = True
    distill_token_selection: str = "all"
    distill_token_keep_ratio: float = 1.0

    overlap_topk: int = 16
    enable_topk_overlap_metrics: bool = True

    student_rollout_n: int = 1
    """On-policy samples per prompt."""

    assert_same_vocab: bool = True
    """Fail fast if teacher/student vocab_size differ."""

    def post_init(self):
        if not self.teacher_model_path:
            raise ValueError("opd.teacher_model_path is required for OPD training.")
        resolved = get_abs_path(self.teacher_model_path, prompt="Teacher model")
        if not resolved:
            raise ValueError(f"opd.teacher_model_path not found: {self.teacher_model_path}")
        self.teacher_model_path = resolved
        if self.student_rollout_n < 1:
            raise ValueError("opd.student_rollout_n must be >= 1.")
        if self.distillation_loss_type not in {
            "kl",
            "jsd",
            "topk_jsd",
            "topk_forward_kl",
            "topk_reverse_kl",
        }:
            raise ValueError(
                f"Unsupported opd.distillation_loss_type={self.distillation_loss_type}."
            )


@dataclass
class OPDTrainConfig:
    """Top-level config for `python -m verl.trainer.opd_main` (does not use opsd:)."""

    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)

    def post_init(self):
        if self.data.enable_dynamic_max_response_length:
            train_response_length = max(
                self.data.dynamic_max_response_length_phase1,
                self.data.dynamic_max_response_length_phase2,
                self.data.dynamic_max_response_length_phase3,
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
