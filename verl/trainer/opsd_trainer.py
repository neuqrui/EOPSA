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
OPSD Trainer: Hint-Guided On-Policy Self-Distillation with Self-Evolving Navigator.

Extends RayPPOTrainer with:
  1. Multi-round generation: Student → Navigator → Teacher
  2. Navigator GRPO advantage from Teacher trajectory correctness
  3. OPSD forward batch assembly for joint training
  4. Curriculum learning for ground-truth dropping
"""

import hashlib
import json
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from .config import PPOConfig
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    filter_numeric_reward_metrics,
    reduce_metrics,
)
from .core_algos import AdvantageEstimator, compute_advantage_return
from .opsd_algos import (
    compute_curriculum_p_drop,
    resolve_dynamic_max_response_length,
    decide_adaptive_rollout_length,
    compute_trr_by_prefix,
    attach_token_prefix_to_raw_prompt_ids,
    compute_navigator_advantage,
    construct_navigator_prompts,
    construct_sdpo_teacher_prompts,
    construct_teacher_prompts,
    decode_navigator_hints,
    truncate_text_by_tokens,
    _has_dynamic_verifier_hint,
)
from .opsd_config import OPSDConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, apply_kl_penalty, compute_advantage
from ..utils.redacted_thinking_masks import build_redacted_think_answer_masks


def _concat_prompt_and_response(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate prompt and response following the rollout engine's position-id logic."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=1)

    response_length = response_ids.size(1)
    batch_size = prompt_ids.size(0)
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    if prompt_position_ids.ndim == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
            batch_size, prompt_position_ids.size(1), -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
    return input_ids, attention_mask, position_ids


def _mask_prefix_response_tokens(response_mask: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    """Zero out the first N valid response tokens in each sequence."""
    if prefix_tokens <= 0:
        return response_mask
    masked = response_mask.clone()
    batch_size = masked.shape[0]
    for idx in range(batch_size):
        valid_positions = torch.nonzero(masked[idx] > 0, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        cutoff = min(prefix_tokens, int(valid_positions.numel()))
        masked[idx, valid_positions[:cutoff]] = 0
    return masked


class RayOPSDTrainer(RayPPOTrainer):
    """OPSD Trainer with multi-round generation and joint Navigator + Reasoner update."""

    def __init__(self, config, *args, **kwargs):
        # OPSD does not rely on PPO/GRPO's rollout.n validation in parent trainer.
        # Temporarily set n=2 to bypass the parent check, then restore it.
        original_n = config.worker.rollout.n
        if original_n == 1:
            config.worker.rollout.n = 2  # bypass parent check
        super().__init__(config, *args, **kwargs)
        config.worker.rollout.n = original_n  # restore
        self.opsd_config: OPSDConfig = self.config.opsd

        # Set opsd_enabled flag on actor config so workers instantiate the right actor class
        self.config.worker.actor.opsd_enabled = True
        self.config.worker.actor.freeze_teacher_model = self.opsd_config.freeze_teacher_model
        if self.opsd_config.enable_oracle_correction:
            self.config.worker.actor.oracle_model_path = self.opsd_config.oracle_model_path
            self.config.worker.actor.oracle_assert_same_vocab = self.opsd_config.oracle_assert_same_vocab
            print("[OPSD] Oracle correction: ENABLED")
            print(f"[EOPSA]   - oracle_model_path: {self.opsd_config.oracle_model_path}")
            print(f"[EOPSA]   - oracle_mix: {self.opsd_config.oracle_mix}")
            print(f"[EOPSA]   - oracle_alpha: {self.opsd_config.oracle_alpha}")
            print(f"[EOPSA]   - oracle_lambda: {self.opsd_config.oracle_lambda}")
            print(f"[EOPSA]   - oracle_delta_clip: {self.opsd_config.oracle_delta_clip}")
            print(f"[EOPSA]   - oracle_residual_relu: {self.opsd_config.oracle_residual_relu}")
            print("[EOPSA]   - oracle prompt: student prompt (no privileged hint)")
            print("[EOPSA]   - mix support: teacher top-k only")
        else:
            print("[OPSD] Oracle correction: disabled")

        self._adaptive_rollout_len: Optional[int] = None
        # Filled by `_validate`; consumed by ARS probe to skip a second student rollout.
        self._ars_val_cache: Optional[list[dict[str, Any]]] = None
        self._last_ars_detail: Optional[dict[str, Any]] = None
        self._ars_dataloader = None
        self._ars_log_dir = Path(self.config.trainer.save_checkpoint_path) / "ars_logs"
        if getattr(self.config.data, "enable_adaptive_rollout", False):
            print("[EOPSA] Adaptive Rollout Scheduling rollout length: ENABLED")
            print(f"[EOPSA]   - probe lengths: {self.config.data.adaptive_rollout_prefix_lengths}")
            print(f"[EOPSA]   - min student fails to probe teacher: {self.config.data.adaptive_rollout_min_student_fails}")
            print(f"[EOPSA]   - TRR thresh: {self.config.data.adaptive_rollout_trr_thresh}")
            print(f"[EOPSA]   - max probe samples: {self.config.data.adaptive_rollout_max_samples}")
            print(f"[EOPSA]   - monotonic: {self.config.data.adaptive_rollout_monotonic}")
            print("[EOPSA]   - reuses validation student rollouts when adaptive_rollout_files is unset")
            print(f"[EOPSA]   - adaptive_rollout_files: {getattr(self.config.data, 'adaptive_rollout_files', None)}")
            print(
                f"[EOPSA]   - skip first/last val: "
                f"{getattr(self.config.data, 'adaptive_rollout_skip_first', True)}/"
                f"{getattr(self.config.data, 'adaptive_rollout_skip_last', True)}"
            )
            print(f"[EOPSA]   - logs -> {self._ars_log_dir}")

        # Load navigator and teacher templates
        self._nav_template_path = self.opsd_config.navigator_prompt_template
        self._teacher_template_path = self.opsd_config.teacher_hint_template
        self._sdpo_teacher_template_path = self.opsd_config.sdpo_teacher_template

        if self.opsd_config.enable_sdpo and self.opsd_config.use_gt_as_hint:
            raise ValueError("opsd.enable_sdpo and opsd.use_gt_as_hint cannot both be enabled.")

        if self.opsd_config.enable_sdpo:
            self.opsd_config.num_hints = 1
            self.opsd_config.lambda_nav = 0.0
            self.opsd_config.alpha = 1.0
            print("[OPSD] Mode: SDPO self-distillation")
            print("[EOPSA]   - Navigator rollout: SKIPPED")
            print(f"[EOPSA]   - candidate samples per prompt: {self.opsd_config.sdpo_num_candidates}")
            print("[EOPSA]   - num_hints forced to 1, lambda_nav forced to 0.0, alpha forced to 1.0")
        elif self.opsd_config.use_gt_as_hint:
            # Actually force the config values, not just print about it
            self.opsd_config.num_hints = 1
            self.opsd_config.lambda_nav = 0.0
            self.opsd_config.alpha = 1.0
            print("[OPSD] Mode: Self-Distilled Reasoner (use_gt_as_hint=True)")
            print("[EOPSA]   - Navigator rollout: SKIPPED (using GT answers as hints)")
            print("[EOPSA]   - num_hints forced to 1, lambda_nav forced to 0.0, alpha forced to 1.0")
        if self.opsd_config.enable_dynamic_verifier_hint:
            template_path = (
                self.opsd_config.teacher_dynamic_hint_template or self.opsd_config.teacher_hint_template
            )
            print("[OPSD] Dynamic verifier hint: ENABLED")
            print(f"[EOPSA]   - teacher template: {template_path}")
            print(
                "[EOPSA]   - reward must return verifier_feedback "
                "(see examples/safety_rl/reward/safety_reward.py)"
            )
            print("[EOPSA]   - p_drop forced to 0 (Teacher always sees GT answer)")
            if self.opsd_config.self_distill_negative_off_policy:
                print(
                    "[EOPSA]   - negative off-policy enabled: "
                    f"coef={self.opsd_config.self_distill_negative_off_policy_coef}"
                )
        else:
            print(f"[OPSD] Navigator hints per question (K): {self.opsd_config.num_hints}")
            print(f"[OPSD] Lambda_nav: {self.opsd_config.lambda_nav}")
        print(f"[OPSD] Alpha (on/off-policy weight): {self.opsd_config.alpha}")
        print(f"[OPSD] Distillation loss type: {self.opsd_config.distillation_loss_type}")
        print(f"[OPSD] Student rollout n: {self.opsd_config.student_rollout_n}")
        print(
            f"[OPSD] Navigator include student trajectory: "
            f"{self.opsd_config.navigator_include_student_trajectory}"
        )
        print(
            f"[OPSD] Navigator max trajectory length: "
            f"{self.opsd_config.navigator_max_trajectory_length}"
        )
        print(
            f"[OPSD] Teacher max trajectory length: "
            f"{self.opsd_config.teacher_max_trajectory_length}"
        )
        print(f"[OPSD] Teacher context mode: {self.opsd_config.teacher_context_mode}")
        print(
            f"[OPSD] Teacher reference context length: "
            f"{self.opsd_config.teacher_reference_context_length}"
        )
        print(
            f"[OPSD] Teacher student context length: "
            f"{self.opsd_config.teacher_student_context_length}"
        )
        print(f"[OPSD] Dynamic sample hint: {self.opsd_config.dynamic_sample_hint}")
        print(f"[OPSD] Teacher rollout n per hint: {self.opsd_config.teacher_rollout_n}")
        print(f"[OPSD] Distill token selection: {self.opsd_config.distill_token_selection}")
        print(f"[OPSD] Distill token keep ratio: {self.opsd_config.distill_token_keep_ratio}")
        benign_tf_mode = getattr(self.opsd_config, "token_filter_benign_mode", None)
        if benign_tf_mode:
            print(
                f"[OPSD] Token filter: {self.opsd_config.token_filter} "
                f"(harmful={self.opsd_config.token_filter_mode}, benign={benign_tf_mode})"
            )
        else:
            print(f"[OPSD] Token filter: {self.opsd_config.token_filter} (applies to all samples)")
        print(f"[OPSD] keep_mask (sparse topk/KL): {getattr(self.opsd_config, 'keep_mask', False)}")
        print(f"[OPSD] Token filter mode (harmful/default): {self.opsd_config.token_filter_mode}")
        print(f"[OPSD] Token filter action (harmful/default): {self.opsd_config.token_filter_action}")
        print(f"[OPSD] Token filter module: {self.opsd_config.token_filter_path}")
        if getattr(self.config.data, "enable_adaptive_rollout", False):
            print(
                "[EOPSA] Adaptive Rollout Scheduling student max_tokens: applies to harmful + benign"
            )
        elif getattr(self.config.data, "enable_dynamic_max_response_length", False):
            print(
                "[OPSD] Dynamic student max_tokens: applies to harmful + benign"
            )
        if self.opsd_config.token_filter_mode == "jsd_topn":
            print(f"[OPSD] Token filter top-n: {self.opsd_config.token_filter_top_n}")
        if self.opsd_config.token_filter_mode == "taxonomy_keep":
            print(f"[OPSD] Token filter keep categories: {self.opsd_config.token_filter_keep_categories}")
        if self.opsd_config.token_filter_mode == "taxonomy_drop":
            print(f"[OPSD] Token filter drop categories: {self.opsd_config.token_filter_drop_categories}")
        print(f"[OPSD] Token filter classifier: {self.opsd_config.token_filter_classifier}")
        if str(self.opsd_config.token_filter_mode).startswith("taxonomy") or str(
            self.opsd_config.token_filter_mode
        ) in {"drop", "same_only"}:
            print(
                f"[OPSD] Token filter student source: {self.opsd_config.token_filter_student_source} "
                f"(paired with teacher top-1)"
            )
        if benign_tf_mode:
            benign_action = (
                getattr(self.opsd_config, "token_filter_benign_action", None)
                or self.opsd_config.token_filter_action
            )
            benign_top_n = getattr(self.opsd_config, "token_filter_benign_top_n", None)
            if benign_top_n is None:
                benign_top_n = self.opsd_config.token_filter_top_n
            print(f"[OPSD] Token filter benign mode: {benign_tf_mode}")
            print(f"[OPSD] Token filter benign action: {benign_action}")
            if str(benign_tf_mode).strip().lower() in {"jsd_topn", "forward_kl_topn", "fwd_kl_topn"}:
                print(f"[OPSD] Token filter benign top-n: {benign_top_n} (ranked by forward KL)")
        benign_ratio = getattr(self.opsd_config, "token_filter_benign_ratio", None)
        if benign_ratio is not None:
            print(
                f"[OPSD] Token filter benign ratio: {benign_ratio} "
                "(keep first round(ratio × harmful keep) benign tokens in order; "
                "all-benign microbatches use EMA harmful-keep/row)"
            )
        print(
            f"[OPSD] Token filter log updated tokens: "
            f"{self.opsd_config.token_filter_log_updated} "
            f"(max={self.opsd_config.token_filter_log_max_tokens})"
        )
        print(f"[OPSD] Curriculum warmup steps: {self.opsd_config.curriculum_warmup_steps}")
        print(f"[OPSD] Off-policy include GT answer: {self.opsd_config.off_policy_include_gt_answer}")
        print(
            "[OPSD] Self-distill negative off-policy: "
            f"{self.opsd_config.self_distill_negative_off_policy}"
        )
        print(
            "[OPSD] Self-distill masked prefix tokens: "
            f"{self.opsd_config.self_distill_mask_prefix_tokens}"
        )
        print(f"[OPSD] KL penalty type: {self.opsd_config.opsd_kl_penalty}")
        if self.opsd_config.distillation_topk is not None:
            print(f"[OPSD] Top-k KL approximation: k={self.opsd_config.distillation_topk}")
            print(f"[OPSD] Top-k source: {self.opsd_config.distillation_topk_source}")
        print(
            f"[OPSD] Top-k overlap metrics: enabled={self.opsd_config.enable_topk_overlap_metrics}, "
            f"k={self.opsd_config.overlap_topk}"
        )

        self._nav_hint_log_samples = 4
        self._nav_hint_log_history_limit = 20
        self._nav_hint_log_history: list[dict[str, Any]] = []
        self._nav_hint_log_path = Path(self.config.trainer.save_checkpoint_path) / "navigator_hint_samples.json"
        self._student_generation_log_samples = 4
        self._student_generation_log_dir = Path(self.config.trainer.save_checkpoint_path) / "student_generation_samples"
        self._prompt_log_samples = 4
        self._prompt_log_dir = Path(self.config.trainer.save_checkpoint_path) / "prompt_logs"
        self._overlap_log_dir = Path(self.config.trainer.save_checkpoint_path) / "overlap_logs"
        self._stf_drop_log_dir = Path(self.config.trainer.save_checkpoint_path) / "stf_drop_logs"
        self._training_token_stats_path = (
            Path(self.config.trainer.save_checkpoint_path) / "training_token_stats.json"
        )
        self._training_token_stats: dict[str, float] = {
            "total_train_tokens": 0.0,
            "total_rollout_tokens": 0.0,
            "total_gen_time_s": 0.0,
            "total_train_time_s": 0.0,
            "total_steps": 0.0,
            "total_data_count": 0.0,
            "avg_train_tokens_per_data": 0.0,
        }

    def _extract_topk_overlap_actor_metrics(self, actor_output: DataProto) -> None:
        """Persist per-token-index overlap stats to disk; keep scalar for SwanLab."""
        if not self.opsd_config.enable_topk_overlap_metrics:
            return

        non_tensor = actor_output.non_tensor_batch
        ratio_key = "opsd/overlap_token_ratio_by_idx"
        count_key = "opsd/overlap_token_count_by_idx"
        topk_key = "opsd/overlap_topk"
        oracle_ratio_key = "opsd/oracle_overlap_token_ratio_by_idx"
        oracle_count_key = "opsd/oracle_overlap_token_count_by_idx"
        if ratio_key not in non_tensor and oracle_ratio_key not in non_tensor:
            return

        def _unpack_by_idx(ratio_raw, count_raw):
            ratio_list = ratio_raw[0] if isinstance(ratio_raw, np.ndarray) and ratio_raw.size > 0 else ratio_raw
            count_list = count_raw[0] if isinstance(count_raw, np.ndarray) and count_raw.size > 0 else count_raw
            by_idx = []
            if ratio_list is None:
                return by_idx
            for idx, ratio in enumerate(ratio_list):
                token_count = int(count_list[idx]) if count_list is not None and idx < len(count_list) else 0
                if token_count <= 0:
                    continue
                by_idx.append(
                    {
                        "token_idx": idx,
                        "overlap_ratio": float(ratio),
                        "token_count": token_count,
                    }
                )
            return by_idx

        overlap_by_idx = []
        if ratio_key in non_tensor:
            ratio_raw = non_tensor.pop(ratio_key)
            count_raw = non_tensor.pop(count_key, None)
            overlap_by_idx = _unpack_by_idx(ratio_raw, count_raw)

        topk_raw = non_tensor.pop(topk_key, None)
        overlap_topk = (
            float(topk_raw[0])
            if isinstance(topk_raw, np.ndarray) and topk_raw.size > 0
            else self.opsd_config.overlap_topk
        )

        oracle_overlap_by_idx = []
        if oracle_ratio_key in non_tensor:
            oracle_ratio_raw = non_tensor.pop(oracle_ratio_key)
            oracle_count_raw = non_tensor.pop(oracle_count_key, None)
            oracle_overlap_by_idx = _unpack_by_idx(oracle_ratio_raw, oracle_count_raw)

        self._overlap_log_dir.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "step": self.global_step,
            "overlap_topk": overlap_topk,
            "overlap_token_ratio": float(non_tensor.get("opsd/overlap_token_ratio", np.array([0.0]))[0])
            if "opsd/overlap_token_ratio" in non_tensor
            else None,
            "overlap_by_token_idx": overlap_by_idx,
            "oracle_overlap_token_ratio": float(
                non_tensor.get("opsd/oracle_overlap_token_ratio", np.array([0.0]))[0]
            )
            if "opsd/oracle_overlap_token_ratio" in non_tensor
            else None,
            "oracle_overlap_by_token_idx": oracle_overlap_by_idx,
        }
        out_path = self._overlap_log_dir / f"step_{self.global_step:06d}.json"
        out_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        jsonl_path = self._overlap_log_dir / "overlap_by_token_idx.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_payload, ensure_ascii=False) + "\n")

    def _extract_stf_drop_updated_token_logs(self, actor_output: DataProto) -> None:
        """Persist updated (kept) tokens under same_token_filter mode=drop/same_only."""
        key = "__stf_drop_updated_tokens__"
        non_tensor = actor_output.non_tensor_batch
        if key not in non_tensor:
            return

        raw = non_tensor.pop(key)
        if raw is None:
            return

        items = raw.tolist() if isinstance(raw, np.ndarray) else list(raw)
        all_tokens: list[dict[str, Any]] = []
        for item in items:
            if item is None:
                continue
            try:
                payload = json.loads(item) if isinstance(item, (str, bytes)) else item
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                toks = payload.get("tokens") or []
            elif isinstance(payload, list):
                toks = payload
            else:
                continue
            all_tokens.extend(toks)

        max_n = int(getattr(self.opsd_config, "token_filter_log_max_tokens", 50000) or 0)
        if max_n > 0 and len(all_tokens) > max_n:
            all_tokens = all_tokens[:max_n]

        # Decode token ids for offline taxonomy / wordcloud analysis.
        def _decode_id(tid: int) -> str:
            try:
                return self.tokenizer.decode([int(tid)], skip_special_tokens=False)
            except Exception:
                return ""

        for rec in all_tokens:
            if "y_str" not in rec and "y_id" in rec:
                rec["y_str"] = _decode_id(rec["y_id"])
            if "tea_top1_str" not in rec and "tea_top1_id" in rec:
                rec["tea_top1_str"] = _decode_id(rec["tea_top1_id"])
            if "stu_top1_str" not in rec and "stu_top1_id" in rec:
                rec["stu_top1_str"] = _decode_id(rec["stu_top1_id"])

        mode = str(getattr(self.opsd_config, "token_filter_mode", "taxonomy_drop"))
        self._stf_drop_log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.global_step,
            "mode": mode,
            "n_updated": len(all_tokens),
            "tokens": all_tokens,
        }
        out_path = self._stf_drop_log_dir / f"step_{self.global_step:06d}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        jsonl_path = self._stf_drop_log_dir / "updated_tokens.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for rec in all_tokens:
                f.write(json.dumps({"step": self.global_step, **rec}, ensure_ascii=False) + "\n")

        # Lightweight summary line for quick grepping.
        summary_path = self._stf_drop_log_dir / "updated_tokens_summary.jsonl"
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": self.global_step,
                        "n_updated": len(all_tokens),
                        "div_mean": float(np.mean([t.get("div", 0.0) for t in all_tokens])) if all_tokens else 0.0,
                        "div_sum": float(np.sum([t.get("div", 0.0) for t in all_tokens])) if all_tokens else 0.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _get_teacher_questions(self, batch: DataProto) -> np.ndarray:
        questions = batch.non_tensor_batch.get("teacher_question")
        if questions is None:
            questions = batch.non_tensor_batch["prompt_text"]
        return questions

    def _get_problem_types(self, batch: DataProto):
        return batch.non_tensor_batch.get("problem_type")

    def _get_data_types(self, batch: DataProto):
        return batch.non_tensor_batch.get("data_type")

    def _log_navigator_hints(self, batch: DataProto, hints: list[list[str]]) -> None:
        """Persist a small rolling window of navigator hint samples for inspection."""
        if not hints:
            return

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        ground_truth = batch.non_tensor_batch.get("ground_truth")
        uids = batch.non_tensor_batch.get("uid")

        sample_count = min(self._nav_hint_log_samples, len(hints))
        samples = []
        for idx in range(sample_count):
            question = "" if prompt_text is None else str(prompt_text[idx])
            samples.append(
                {
                    "uid": None if uids is None else str(uids[idx]),
                    "question": question[:500],
                    "ground_truth": None if ground_truth is None else str(ground_truth[idx]),
                    "hints": [str(h) for h in hints[idx]],
                }
            )

        self._nav_hint_log_history.append(
            {
                "step": self.global_step,
                "num_questions": len(hints),
                "num_hints_per_question": len(hints[0]) if hints else 0,
                "samples": samples,
            }
        )
        self._nav_hint_log_history = self._nav_hint_log_history[-self._nav_hint_log_history_limit :]

        self._nav_hint_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._nav_hint_log_path.write_text(
            json.dumps(self._nav_hint_log_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log_student_generations(self, batch: DataProto, student_output: DataProto) -> None:
        """Persist a few sampled student generations for the current step."""
        if len(student_output) == 0:
            return

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        ground_truth = batch.non_tensor_batch.get("ground_truth")
        uids = batch.non_tensor_batch.get("uid")
        problem_type = batch.non_tensor_batch.get("problem_type")

        sample_count = min(self._student_generation_log_samples, len(student_output))
        sampled_indices = np.random.choice(len(student_output), size=sample_count, replace=False)

        samples = []
        response_ids = student_output.batch["responses"]
        response_mask = student_output.batch["response_mask"]
        for idx in sampled_indices.tolist():
            cur_response_length = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            question = "" if prompt_text is None else str(prompt_text[idx])
            samples.append(
                {
                    "uid": None if uids is None else str(uids[idx]),
                    "problem_type": None if problem_type is None else str(problem_type[idx]),
                    "ground_truth": None if ground_truth is None else str(ground_truth[idx]),
                    "response_length": cur_response_length,
                    "question": question,
                    "response": response_text,
                }
            )

        payload = {
            "step": self.global_step,
            "num_samples_in_batch": len(student_output),
            "num_logged_samples": sample_count,
            "samples": samples,
        }

        self._student_generation_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._student_generation_log_dir / f"step_{self.global_step:06d}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_student_prompt_text(self, batch: DataProto, idx: int) -> str:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is not None and idx < len(raw_prompts):
            text = str(raw_prompts[idx])
            if text.strip():
                return text

        prompts = batch.batch.get("prompts")
        if prompts is not None:
            pad_id = self.tokenizer.pad_token_id
            ids = prompts[idx].tolist()
            while ids and ids[0] == pad_id:
                ids.pop(0)
            return self.tokenizer.decode(ids, skip_special_tokens=False)

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        if prompt_text is not None and idx < len(prompt_text):
            return str(prompt_text[idx])
        return ""

    def _decode_batch_response_text(self, batch: DataProto, idx: int) -> str:
        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        cur_response_length = int(response_mask[idx].sum().item())
        valid_response_ids = response_ids[idx][:cur_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

    @staticmethod
    def _compute_response_hash(response_ids: torch.Tensor, response_mask: torch.Tensor) -> str:
        valid_len = int(response_mask.sum().item())
        if valid_len <= 0:
            return ""
        token_ids = response_ids[:valid_len].tolist()
        payload = ",".join(str(int(tok)) for tok in token_ids)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _log_student_teacher_prompts(
        self,
        batch: DataProto,
        teacher_gen: DataProto,
        *,
        hints: Optional[list[list[str]]] = None,
        verifier_feedbacks: Optional[list[str]] = None,
        think_diagnoses: Optional[list[str]] = None,
        answer_diagnoses: Optional[list[str]] = None,
        reward_metrics: Optional[dict[str, list]] = None,
    ) -> None:
        """Persist student/teacher prompt pairs for manual inspection."""
        batch_size = len(batch)
        if batch_size == 0:
            return

        teacher_raw_prompts = teacher_gen.non_tensor_batch.get("raw_prompt")
        if teacher_raw_prompts is None or len(teacher_raw_prompts) != batch_size:
            return

        sample_count = min(self._prompt_log_samples, batch_size)
        sampled_indices = np.random.choice(batch_size, size=sample_count, replace=False)

        data_types = batch.non_tensor_batch.get("data_type")
        uids = batch.non_tensor_batch.get("uid")
        samples = []
        for idx in sampled_indices.tolist():
            hint_text = ""
            if hints is not None and idx < len(hints) and hints[idx]:
                hint_text = str(hints[idx][0])

            vf = verifier_feedbacks[idx] if verifier_feedbacks is not None and idx < len(verifier_feedbacks) else ""
            td = think_diagnoses[idx] if think_diagnoses is not None and idx < len(think_diagnoses) else ""
            ad = answer_diagnoses[idx] if answer_diagnoses is not None and idx < len(answer_diagnoses) else ""

            entry: dict[str, Any] = {
                "index": idx,
                "uid": None if uids is None else str(uids[idx]),
                "data_type": None if data_types is None else str(data_types[idx]),
                "student_prompt": self._get_student_prompt_text(batch, idx),
                "student_response": self._decode_batch_response_text(batch, idx),
                "teacher_prompt": str(teacher_raw_prompts[idx]),
                "static_hint": hint_text,
                "include_verifier_hint": _has_dynamic_verifier_hint(vf, td, ad),
                "think_diagnosis": str(td),
                "answer_diagnosis": str(ad),
                "verifier_feedback": str(vf),
            }
            if reward_metrics is not None:
                for key in ("overall", "accuracy", "format", "think_safe", "ans_safe"):
                    values = reward_metrics.get(key)
                    if values is not None and idx < len(values):
                        entry[f"reward_{key}"] = values[idx]
            samples.append(entry)

        payload = {
            "step": self.global_step,
            "batch_size": batch_size,
            "num_logged_samples": sample_count,
            "student_format_prompt": self.config.data.format_prompt,
            "teacher_template": self._resolve_teacher_template_path(),
            "samples": samples,
        }

        self._prompt_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._prompt_log_dir / f"step_{self.global_step:06d}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        jsonl_path = self._prompt_log_dir / "prompt_samples.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps({"step": self.global_step, **sample}, ensure_ascii=False) + "\n")

        print(f"[OPSD] Prompt samples saved -> {out_path}")

    def _decode_student_trajectories(
        self,
        student_output: DataProto,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Decode student rollout responses for optional Navigator conditioning."""
        trajectories: list[str] = []
        response_ids = student_output.batch["responses"]
        response_mask = student_output.batch["response_mask"]

        for idx in range(len(student_output)):
            cur_response_length = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            trajectories.append(truncate_text_by_tokens(response_text, self.tokenizer, max_tokens))

        return trajectories

    def _extract_teacher_context_snippet(
        self,
        text: Any,
        max_tokens: Optional[int],
        *,
        core_only: bool,
    ) -> str:
        if text is None:
            return ""
        text_str = str(text)
        if not text_str:
            return ""
        if not max_tokens or max_tokens <= 0:
            return ""
        if not core_only:
            return truncate_text_by_tokens(text_str, self.tokenizer, max_tokens)

        token_ids = self.tokenizer.encode(text_str, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text_str
        if max_tokens <= 32:
            return self.tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=False)

        head_len = max(1, max_tokens // 2)
        tail_len = max(1, max_tokens - head_len)
        head_text = self.tokenizer.decode(token_ids[:head_len], skip_special_tokens=False).strip()
        tail_text = self.tokenizer.decode(token_ids[-tail_len:], skip_special_tokens=False).strip()
        return (
            "[Earlier Key Step]\n"
            f"{head_text}\n\n"
            "[Conclusion / Final Segment]\n"
            f"{tail_text}"
        )

    def _build_teacher_context_trajectories(
        self,
        batch: DataProto,
        student_output: Optional[DataProto] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        mode = self.opsd_config.teacher_context_mode
        if mode == "hint":
            return None, None

        core_only = mode == "hint_plus_core_trajectories"
        ref_max_tokens = self.opsd_config.teacher_reference_context_length
        student_max_tokens = self.opsd_config.teacher_student_context_length

        reference_trajectories: list[str] = []
        raw_references = batch.non_tensor_batch.get("trajectory")
        for idx in range(len(batch)):
            raw_reference = "" if raw_references is None else raw_references[idx]
            reference_trajectories.append(
                self._extract_teacher_context_snippet(
                    raw_reference,
                    ref_max_tokens,
                    core_only=core_only,
                )
            )

        student_trajectories = None
        if student_output is not None:
            decoded_student_trajectories = self._decode_student_trajectories(student_output)
            student_trajectories = [
                self._extract_teacher_context_snippet(
                    trajectory,
                    student_max_tokens,
                    core_only=core_only,
                )
                for trajectory in decoded_student_trajectories
            ]

        return (
            np.array(reference_trajectories, dtype=object),
            None if student_trajectories is None else np.array(student_trajectories, dtype=object),
        )

    def _get_self_distill_hints(self, batch: DataProto) -> list[list[str]]:
        """Use trajectory as teacher context when available, otherwise fall back to GT."""
        trajectories = batch.non_tensor_batch.get("trajectory")
        hints = []
        for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
            trajectory = ""
            if trajectories is not None:
                trajectory = truncate_text_by_tokens(
                    trajectories[idx],
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
            hints.append([trajectory] if trajectory else [str(gt)])
        return hints

    @staticmethod
    def _filter_numeric_reward_metrics(reward_metrics: dict[str, list]) -> dict[str, list]:
        return filter_numeric_reward_metrics(reward_metrics)

    def _resolve_teacher_template_path(self) -> str:
        if self.opsd_config.enable_dynamic_verifier_hint and self.opsd_config.teacher_dynamic_hint_template:
            return self.opsd_config.teacher_dynamic_hint_template
        return self._teacher_template_path

    def _extract_verifier_hint_fields(
        self,
        reward_metrics: Optional[dict[str, list]],
        batch_size: int,
    ) -> tuple[Optional[list[str]], Optional[list[str]], Optional[list[str]]]:
        if not self.opsd_config.enable_dynamic_verifier_hint:
            return None, None, None
        if reward_metrics is None or "verifier_feedback" not in reward_metrics:
            raise ValueError(
                "opsd.enable_dynamic_verifier_hint=True requires reward key verifier_feedback. "
                "Use examples/safety_rl/reward/safety_reward.py or extend your reward function."
            )
        if len(reward_metrics["verifier_feedback"]) != batch_size:
            raise RuntimeError(
                f"verifier_feedback length mismatch: expected {batch_size}, "
                f"got {len(reward_metrics['verifier_feedback'])}"
            )

        max_len = self.opsd_config.verifier_feedback_max_length

        def _truncate(values: list[Any]) -> list[str]:
            truncated = []
            for value in values:
                text = "" if value is None else str(value)
                if max_len:
                    text = truncate_text_by_tokens(text, self.tokenizer, max_len)
                truncated.append(text)
            return truncated

        think_values = reward_metrics.get("think_diagnosis", [""] * batch_size)
        answer_values = reward_metrics.get("answer_diagnosis", [""] * batch_size)
        return (
            _truncate(reward_metrics["verifier_feedback"]),
            _truncate(think_values),
            _truncate(answer_values),
        )

    def _construct_teacher_fallback_prompts(
        self,
        batch: DataProto,
        repeat_k: int,
        student_output: Optional[DataProto] = None,
    ) -> tuple[DataProto, int]:
        """Build fallback teacher prompts using trajectory first and GT only as a final fallback."""
        trajectories = batch.non_tensor_batch.get("trajectory")
        fallback_hints: list[list[str]] = []
        fallback_gt_answers: list[str] = []
        trajectory_missing = 0

        for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
            trajectory = ""
            if trajectories is not None:
                trajectory = truncate_text_by_tokens(
                    trajectories[idx],
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )

            if trajectory:
                fallback_hints.append([trajectory] * repeat_k)
                fallback_gt_answers.append("")
            else:
                fallback_hints.append([""] * repeat_k)
                fallback_gt_answers.append(str(gt))
                trajectory_missing += 1

        reference_trajectories, student_trajectories = self._build_teacher_context_trajectories(
            batch,
            student_output=student_output,
        )
        fallback_teacher_gen = construct_teacher_prompts(
            questions=self._get_teacher_questions(batch),
            hints=fallback_hints,
            gt_answers=np.array(fallback_gt_answers, dtype=object),
            p_drop=0.0,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reference_trajectories=reference_trajectories,
            student_trajectories=student_trajectories,
            template_path=self._teacher_template_path,
            max_prompt_length=self.config.data.max_prompt_length,
            multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
            image_min_pixels=self.config.data.image_min_pixels,
            image_max_pixels=self.config.data.image_max_pixels,
            video_min_pixels=self.config.data.video_min_pixels,
            video_max_pixels=self.config.data.video_max_pixels,
            video_fps=self.config.data.video_fps,
            video_max_frames=self.config.data.video_max_frames,
            apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
            problem_types=self._get_problem_types(batch),
            data_types=self._get_data_types(batch),
        )
        return fallback_teacher_gen, trajectory_missing

    def _construct_on_policy_teacher_prompts(
        self,
        batch: DataProto,
        hints: list[list[str]],
        student_output: Optional[DataProto] = None,
    ) -> DataProto:
        """Build teacher prompts for on-policy distillation with GT always retained."""
        reference_trajectories, student_trajectories = self._build_teacher_context_trajectories(
            batch,
            student_output=student_output,
        )
        return construct_teacher_prompts(
            questions=self._get_teacher_questions(batch),
            hints=hints,
            gt_answers=batch.non_tensor_batch["ground_truth"],
            p_drop=0.0,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reference_trajectories=reference_trajectories,
            student_trajectories=student_trajectories,
            template_path=self._teacher_template_path,
            max_prompt_length=self.config.data.max_prompt_length,
            multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
            image_min_pixels=self.config.data.image_min_pixels,
            image_max_pixels=self.config.data.image_max_pixels,
            video_min_pixels=self.config.data.video_min_pixels,
            video_max_pixels=self.config.data.video_max_pixels,
            video_fps=self.config.data.video_fps,
            video_max_frames=self.config.data.video_max_frames,
            apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
            problem_types=self._get_problem_types(batch),
            data_types=self._get_data_types(batch),
        )

    def _select_dynamic_hint_targets(
        self,
        rollout_correctness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the best hint group and one correct rollout from it for each question."""
        batch_size, _, _ = rollout_correctness.shape
        device = rollout_correctness.device
        hint_group_scores = rollout_correctness.float().mean(dim=-1)
        selected_hint_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        selected_rollout_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        selected_has_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
        selected_group_scores = torch.zeros(batch_size, dtype=hint_group_scores.dtype, device=device)

        for i in range(batch_size):
            hint_scores = hint_group_scores[i]
            max_score = hint_scores.max()
            best_hint_candidates = torch.nonzero(torch.isclose(hint_scores, max_score), as_tuple=True)[0]
            chosen_hint = best_hint_candidates[np.random.randint(best_hint_candidates.numel())]
            selected_hint_indices[i] = chosen_hint
            selected_group_scores[i] = hint_scores[chosen_hint]

            correct_rollout_candidates = torch.nonzero(rollout_correctness[i, chosen_hint] > 0, as_tuple=True)[0]
            if correct_rollout_candidates.numel() > 0:
                selected_has_correct[i] = True
                selected_rollout_indices[i] = correct_rollout_candidates[
                    np.random.randint(correct_rollout_candidates.numel())
                ]

        return (
            hint_group_scores,
            selected_hint_indices,
            selected_rollout_indices,
            selected_has_correct,
            selected_group_scores,
        )

    def _build_student_gen_batch(self, batch: DataProto) -> DataProto:
        """Build rollout prompts from the current batch without removing prompt text fields."""
        student_gen = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "has_offline_trajectory",
                "offline_output",
            ],
            meta_info_keys=[
                "image_min_pixels", "image_max_pixels",
                "video_min_pixels", "video_max_pixels",
                "video_fps", "video_max_frames",
            ],
        )
        if "raw_prompt" in batch.non_tensor_batch:
            student_gen.non_tensor_batch["raw_prompt"] = batch.non_tensor_batch["raw_prompt"].copy()
        return student_gen

    def _compute_response_rewards(
        self,
        source_batch: DataProto,
        rollout_output: DataProto,
        repeat: int = 1,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        reward_batch = DataProto.from_dict(
            tensors={
                "responses": rollout_output.batch["responses"],
                "response_mask": rollout_output.batch["response_mask"],
            },
            non_tensors={},
        )
        for key, value in source_batch.non_tensor_batch.items():
            if key in ("multi_modal_data", "raw_prompt_ids"):
                continue
            reward_batch.non_tensor_batch[key] = np.repeat(value, repeat) if repeat > 1 else value
        return ray.get(self.reward_fn.compute_reward.remote(reward_batch))

    def _select_sdpo_feedback(
        self,
        candidate_output: DataProto,
        candidate_scores: torch.Tensor,
        batch_size: int,
    ) -> tuple[list[str], list[str], dict[str, float]]:
        candidate_n = self.opsd_config.sdpo_num_candidates
        response_ids = candidate_output.batch["responses"]
        response_mask = candidate_output.batch["response_mask"]
        response_lengths = response_mask.sum(dim=-1).to(torch.long)

        correct_solutions: list[str] = []
        incorrect_solutions: list[str] = []
        num_with_correct = 0
        num_with_incorrect = 0

        for i in range(batch_size):
            start = i * candidate_n
            end = start + candidate_n
            group_scores = candidate_scores[start:end]
            group_lengths = response_lengths[start:end]

            correct_idx = (group_scores > 0).nonzero(as_tuple=True)[0]
            incorrect_idx = (group_scores <= 0).nonzero(as_tuple=True)[0]

            selected_correct = ""
            if len(correct_idx) > 0:
                best_rel = correct_idx[group_lengths[correct_idx].argmin()].item()
                best_idx = start + best_rel
                cur_len = int(response_lengths[best_idx].item())
                selected_correct = self.tokenizer.decode(
                    response_ids[best_idx][:cur_len], skip_special_tokens=True
                ).strip()
                selected_correct = truncate_text_by_tokens(
                    selected_correct,
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
                num_with_correct += 1

            selected_incorrect = ""
            if len(incorrect_idx) > 0:
                best_rel = incorrect_idx[group_lengths[incorrect_idx].argmin()].item()
                best_idx = start + best_rel
                cur_len = int(response_lengths[best_idx].item())
                selected_incorrect = self.tokenizer.decode(
                    response_ids[best_idx][:cur_len], skip_special_tokens=True
                ).strip()
                selected_incorrect = truncate_text_by_tokens(
                    selected_incorrect,
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
                num_with_incorrect += 1

            correct_solutions.append(selected_correct)
            incorrect_solutions.append(selected_incorrect)

        metrics = {
            "sdpo/num_prompts_with_correct_candidate": float(num_with_correct),
            "sdpo/num_prompts_with_incorrect_candidate": float(num_with_incorrect),
            "sdpo/correct_candidate_coverage": float(num_with_correct / max(batch_size, 1)),
            "sdpo/incorrect_candidate_coverage": float(num_with_incorrect / max(batch_size, 1)),
        }
        return correct_solutions, incorrect_solutions, metrics

    def _get_fixed_max_response_length(self) -> int:
        """Legacy alias: same as student length (benign tracks harmful)."""
        return self._get_student_max_response_length()

    def _is_dynamic_student_length_enabled(self) -> bool:
        data = self.config.data
        return bool(
            getattr(data, "enable_adaptive_rollout", False)
            or getattr(data, "enable_dynamic_max_response_length", False)
        )

    def _get_student_max_response_length(self) -> int:
        """Dynamic / ARS-guided max_tokens for all train rollouts (harmful + benign)."""
        data = self.config.data
        if getattr(data, "enable_adaptive_rollout", False):
            if self._adaptive_rollout_len is not None:
                return int(self._adaptive_rollout_len)
            # Before the first probe, start at the shortest probe length.
            lengths = list(data.adaptive_rollout_prefix_lengths or [128, 256, 1024])
            return int(min(lengths))
        return resolve_dynamic_max_response_length(
            self.global_step,
            enabled=data.enable_dynamic_max_response_length,
            default=data.max_response_length,
            phase1_end=data.dynamic_max_response_length_phase1_end,
            phase1_len=data.dynamic_max_response_length_phase1,
            phase2_end=data.dynamic_max_response_length_phase2_end,
            phase2_len=data.dynamic_max_response_length_phase2,
            phase3_len=data.dynamic_max_response_length_phase3,
        )

    @staticmethod
    def _is_harmful_probe_sample(data_type: Any, problem_type: Any = None) -> bool:
        """True for harmful/safety samples; False for benign/overreject."""
        benign = {"overreject", "benign", "harmless", "harmless_queries"}
        for value in (data_type, problem_type):
            if value is None:
                continue
            if str(value).strip().lower() in benign:
                return False
        return True

    def _get_harmful_prompt_indices(self, batch: DataProto) -> list[int]:
        data_types = batch.non_tensor_batch.get("data_type")
        problem_types = batch.non_tensor_batch.get("problem_type")
        indices: list[int] = []
        for i in range(len(batch)):
            dt = None if data_types is None else data_types[i]
            pt = None if problem_types is None else problem_types[i]
            if self._is_harmful_probe_sample(dt, pt):
                indices.append(i)
        return indices

    @staticmethod
    def _pad_tensor_last_dim(tensor: torch.Tensor, target_len: int, pad_value: float = 0) -> torch.Tensor:
        cur = int(tensor.size(-1))
        if cur == target_len:
            return tensor
        if cur > target_len:
            return tensor.narrow(-1, 0, target_len)
        pad_shape = list(tensor.shape)
        pad_shape[-1] = target_len - cur
        pad = tensor.new_full(pad_shape, pad_value)
        return torch.cat([tensor, pad], dim=-1)

    def _merge_rollout_parts_by_prompt(
        self,
        parts: list[tuple[list[int], DataProto]],
        *,
        total_prompts: int,
        n: int,
    ) -> DataProto:
        """Merge subset rollouts into prompt order with n interleaved sequences."""
        if not parts:
            raise ValueError("parts must be non-empty")
        total = total_prompts * n
        ref = parts[0][1]
        tensor_keys = list(ref.batch.keys())
        max_lens = {k: 0 for k in tensor_keys}
        for _, out in parts:
            for k in tensor_keys:
                max_lens[k] = max(max_lens[k], int(out.batch[k].size(-1)))

        merged_tensors: dict[str, torch.Tensor] = {}
        for k in tensor_keys:
            t0 = ref.batch[k]
            shape = list(t0.shape)
            shape[0] = total
            shape[-1] = max_lens[k]
            merged_tensors[k] = t0.new_zeros(shape)

        non_tensor_keys = list(ref.non_tensor_batch.keys())
        merged_non: dict[str, np.ndarray] = {}
        for k in non_tensor_keys:
            sample = ref.non_tensor_batch[k]
            merged_non[k] = np.empty((total,), dtype=sample.dtype)

        for prompt_indices, out in parts:
            for local_i, prompt_i in enumerate(prompt_indices):
                for r in range(n):
                    src = local_i * n + r
                    dst = prompt_i * n + r
                    for k in tensor_keys:
                        pad_value = 0
                        if k == "attention_mask" or k == "response_mask":
                            pad_value = 0
                        merged_tensors[k][dst] = self._pad_tensor_last_dim(
                            out.batch[k][src], max_lens[k], pad_value=pad_value
                        )
                    for k in non_tensor_keys:
                        merged_non[k][dst] = out.non_tensor_batch[k][src]

        return DataProto.from_dict(
            tensors=merged_tensors,
            non_tensors=merged_non,
            meta_info=dict(ref.meta_info),
        )

    def _generate_sequences_with_max_tokens(
        self,
        gen_batch: DataProto,
        *,
        n: int,
        max_tokens: int,
    ) -> DataProto:
        gen = deepcopy(gen_batch)
        gen.meta_info.update(
            {
                "n": n,
                "temperature": self.config.worker.rollout.temperature,
                "top_p": self.config.worker.rollout.top_p,
                "max_tokens": int(max_tokens),
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            }
        )
        gen, pad_size = pad_dataproto_to_divisor(gen, self.actor_rollout_ref_wg.world_size)
        output = self.actor_rollout_ref_wg.generate_sequences(gen)
        return unpad_dataproto(output, pad_size=pad_size * n)

    def _generate_student_rollout_output(
        self,
        student_gen: DataProto,
        source_batch: DataProto,
        *,
        n: int,
        harmful_max_tokens: int,
        benign_max_tokens: int,
    ) -> DataProto:
        """Student rollout with dynamic/ARS max_tokens (harmful and benign use the same length).

        Token filter (distill) is unchanged and still applies to all samples.
        """
        if (not self._is_dynamic_student_length_enabled()) or harmful_max_tokens == benign_max_tokens:
            return self._generate_sequences_with_max_tokens(
                student_gen, n=n, max_tokens=harmful_max_tokens
            )

        harmful_idx = self._get_harmful_prompt_indices(source_batch)
        all_idx = list(range(len(source_batch)))
        benign_idx = [i for i in all_idx if i not in set(harmful_idx)]

        if not harmful_idx:
            return self._generate_sequences_with_max_tokens(
                student_gen, n=n, max_tokens=benign_max_tokens
            )
        if not benign_idx:
            return self._generate_sequences_with_max_tokens(
                student_gen, n=n, max_tokens=harmful_max_tokens
            )

        parts: list[tuple[list[int], DataProto]] = []
        harm_out = self._generate_sequences_with_max_tokens(
            student_gen.index_select(harmful_idx), n=n, max_tokens=harmful_max_tokens
        )
        parts.append((harmful_idx, harm_out))
        benign_out = self._generate_sequences_with_max_tokens(
            student_gen.index_select(benign_idx), n=n, max_tokens=benign_max_tokens
        )
        parts.append((benign_idx, benign_out))
        return self._merge_rollout_parts_by_prompt(
            parts, total_prompts=len(source_batch), n=n
        )

    def _score_response_texts_unsafe(
        self,
        source_batch: DataProto,
        response_texts: list[str],
    ) -> list[bool]:
        """Return per-sample unsafe flags via val_reward_fn (True = defense fail)."""
        assert self.val_reward_fn is not None
        n = len(response_texts)
        if n == 0:
            return []

        encoded = [self.tokenizer.encode(text or "", add_special_tokens=False) for text in response_texts]
        max_len = max((len(x) for x in encoded), default=1)
        max_len = max(1, min(max_len, int(self.config.worker.rollout.response_length)))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        responses = torch.full((n, max_len), fill_value=pad_id, dtype=torch.long)
        response_mask = torch.zeros((n, max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            ids = ids[:max_len]
            if not ids:
                continue
            responses[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            response_mask[i, : len(ids)] = 1

        reward_batch = DataProto.from_dict(
            tensors={"responses": responses, "response_mask": response_mask},
            non_tensors={},
        )
        for key, value in source_batch.non_tensor_batch.items():
            if key in ("multi_modal_data", "raw_prompt_ids", "raw_prompt"):
                continue
            reward_batch.non_tensor_batch[key] = value

        reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(reward_batch))
        scores = reward_tensor.sum(-1).cpu().tolist()
        return self._unsafe_flags_from_val_reward(
            reward_metrics, scores, batch_size=n
        )

    @staticmethod
    def _unsafe_flags_from_val_reward(
        reward_metrics: dict[str, list],
        scores: list[float],
        *,
        batch_size: int,
    ) -> list[bool]:
        """Per-sample unsafe flags aligned to the full batch.

        Note: llamaguard_reward only emits ``guard_safe`` for harmful rows, so that
        list is shorter than the batch. Prefer ``accuracy`` / ``scores`` which are
        present for every sample.
        """
        accuracy = reward_metrics.get("accuracy")
        if accuracy is not None and len(accuracy) == batch_size:
            return [float(v) < 0.5 for v in accuracy]
        guard_safe = reward_metrics.get("guard_safe")
        if guard_safe is not None and len(guard_safe) == batch_size:
            return [float(v) < 0.5 for v in guard_safe]
        if len(scores) == batch_size:
            return [float(s) < 0.5 for s in scores]
        # Defensive fallback: mark unknown as safe (not an ARS probe fail).
        flags = [False] * batch_size
        if guard_safe is not None and len(guard_safe) > 0:
            print(
                f"[EOPSA][ARS] WARN: guard_safe len={len(guard_safe)} "
                f"!= batch={batch_size}; falling back to scores/accuracy"
            )
        return flags

    def _append_ars_val_cache(
        self,
        cache: list[dict[str, Any]],
        test_batch: DataProto,
        reward_metrics: dict[str, list],
        scores: list[float],
        *,
        max_samples: int,
    ) -> None:
        """Store harmful val rollouts for ARS probe reuse."""
        if len(cache) >= max_samples:
            return
        data_types = test_batch.non_tensor_batch.get("data_type")
        problem_types = test_batch.non_tensor_batch.get("problem_type")
        questions = self._get_teacher_questions(test_batch)
        response_ids = test_batch.batch["responses"]
        response_mask = test_batch.batch["response_mask"]
        batch_size = len(test_batch)
        unsafe_flags = self._unsafe_flags_from_val_reward(
            reward_metrics, scores, batch_size=batch_size
        )
        for i in range(batch_size):
            if len(cache) >= max_samples:
                break
            dt = None if data_types is None else data_types[i]
            pt = None if problem_types is None else problem_types[i]
            if not self._is_harmful_probe_sample(dt, pt):
                continue
            cur_len = int(response_mask[i].sum().item())
            token_ids = response_ids[i, :cur_len].tolist()
            non_tensors: dict[str, Any] = {}
            for key, value in test_batch.non_tensor_batch.items():
                if key in ("multi_modal_data", "raw_prompt_ids", "raw_prompt"):
                    continue
                non_tensors[key] = value[i]
            cache.append(
                {
                    "question": str(questions[i]),
                    "student_token_ids": token_ids,
                    "student_token_len": cur_len,
                    "student_response": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                    "student_unsafe": bool(unsafe_flags[i]),
                    "non_tensors": non_tensors,
                }
            )

    def _load_ars_student_from_val_cache(
        self,
        *,
        max_samples: int,
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, list[Any]]], bool]:
        cache = self._ars_val_cache
        self._ars_val_cache = None  # consume once
        if not cache:
            return [], None, False
        records = cache[: max(0, int(max_samples))]
        aligned: dict[str, list[Any]] = defaultdict(list)
        for rec in records:
            for key, value in (rec.get("non_tensors") or {}).items():
                aligned[key].append(value)
        return records, aligned, True

    def _validate(self) -> dict[str, Any]:
        """Validation with optional ARS probe cache of harmful student rollouts."""
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        ars_cache: list[dict[str, Any]] = []
        max_ars = int(getattr(self.config.data, "adaptive_rollout_max_samples", 100) or 100)
        # Only cache val rollouts for ARS reuse when ARS will actually run on this step.
        cache_for_ars = (
            bool(getattr(self.config.data, "enable_adaptive_rollout", False))
            and (not self._uses_separate_adaptive_rollout_files())
            and (not self._should_skip_adaptive_rollout_probe()[0])
        )

        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            val_response_length = (
                self.config.data.val_max_response_length
                if self.config.data.val_max_response_length is not None
                else self.config.data.max_response_length
            )
            test_gen_batch.meta_info = dict(self.config.worker.rollout.val_override_config)
            test_gen_batch.meta_info.setdefault("max_tokens", val_response_length)
            test_gen_batch.meta_info["image_min_pixels"] = self.config.data.image_min_pixels
            test_gen_batch.meta_info["image_max_pixels"] = self.config.data.image_max_pixels
            test_gen_batch.meta_info["video_min_pixels"] = self.config.data.video_min_pixels
            test_gen_batch.meta_info["video_max_pixels"] = self.config.data.video_max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            test_gen_batch.meta_info["video_max_frames"] = self.config.data.video_max_frames

            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_ref_wg.world_size
            )
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch, pad_size=pad_size * repeat_times
            )

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            val_reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[
                    k for k in test_batch.non_tensor_batch if k != "multi_modal_data"
                ],
            )
            reward_tensor, reward_metrics = ray.get(
                self.val_reward_fn.compute_reward.remote(val_reward_batch)
            )

            input_ids = test_batch.batch["prompts"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            output_ids = test_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            if cache_for_ars:
                self._append_ars_val_cache(
                    ars_cache,
                    test_batch,
                    reward_metrics,
                    scores,
                    max_samples=max_ars,
                )

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._ars_val_cache = ars_cache if cache_for_ars else None
        if cache_for_ars:
            print(
                f"[EOPSA][ARS] cached {len(ars_cache)} harmful val rollouts for reuse"
            )

        self._maybe_log_val_generations(
            sample_inputs, sample_outputs, sample_labels, sample_scores
        )
        if self.config.trainer.val_generations_to_log > 0 and sample_inputs:
            print("Sample prompt (with template):", sample_inputs[0])
            print("Sample response:", sample_outputs[0])
            print("Sample ground_truth:", sample_labels[0])
            print("Sample reward:", sample_scores[0])
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {
            f"val/{key}_reward": value
            for key, value in reduce_metrics(
                filter_numeric_reward_metrics(reward_metrics_lst)
            ).items()
        }
        val_length_metrics = {
            f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()
        }
        print("Finish validation.")
        return {
            "val/reward_score": self.val_reward_score,
            **val_reward_metrics,
            **val_length_metrics,
        }

    def _uses_separate_adaptive_rollout_files(self) -> bool:
        path = getattr(self.config.data, "adaptive_rollout_files", None)
        return bool(path)

    def _get_ars_dataloader(self):
        """Val dataloader, or a dedicated dataloader when data.adaptive_rollout_files is set."""
        if not self._uses_separate_adaptive_rollout_files():
            return self.val_dataloader
        if self._ars_dataloader is not None:
            return self._ars_dataloader

        from torchdata.stateful_dataloader import StatefulDataLoader

        from ..utils.dataset import RLHFDataset, collate_fn

        cfg = self.config.data
        ds = RLHFDataset(
            data_path=cfg.adaptive_rollout_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=cfg.prompt_key,
            answer_key=cfg.answer_key,
            image_key=cfg.image_key,
            video_key=cfg.video_key,
            image_dir=cfg.image_dir,
            video_fps=cfg.video_fps,
            video_max_frames=cfg.video_max_frames,
            max_prompt_length=cfg.max_prompt_length,
            truncation="right",
            format_prompt=cfg.format_prompt,
            image_min_pixels=cfg.image_min_pixels,
            image_max_pixels=cfg.image_max_pixels,
            video_min_pixels=cfg.video_min_pixels,
            video_max_pixels=cfg.video_max_pixels,
            filter_overlong_prompts=cfg.filter_overlong_prompts,
            filter_overlong_prompts_workers=cfg.filter_overlong_prompts_workers,
            use_preprocessed_videos=cfg.use_preprocessed_videos,
            preprocessed_video_dir=cfg.preprocessed_video_dir,
            trajectory_key=cfg.trajectory_key,
            apply_chat_template_kwargs=cfg.apply_chat_template_kwargs,
        )
        if cfg.val_batch_size == -1:
            batch_size = len(ds)
        else:
            batch_size = cfg.val_batch_size
        self._ars_dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        print(
            f"[EOPSA][ARS] dedicated dataloader from {cfg.adaptive_rollout_files} "
            f"(n={len(ds)}, batch={batch_size})"
        )
        return self._ars_dataloader

    def _save_ars_log(self, payload: dict[str, Any]) -> None:
        self._ars_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._ars_log_dir / f"step_{self.global_step:06d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        jsonl_path = self._ars_log_dir / "ars_history.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[EOPSA][ARS] saved log -> {out_path}")

    def _collect_ars_batches(self) -> list[DataProto]:
        """Collect up to adaptive_rollout_max_samples from val or adaptive_rollout_files dataloader."""
        data_cfg = self.config.data
        max_samples = int(data_cfg.adaptive_rollout_max_samples)
        dataloader = self._get_ars_dataloader()
        batches: list[DataProto] = []
        n = 0
        meta_info = {
            "image_min_pixels": data_cfg.image_min_pixels,
            "image_max_pixels": data_cfg.image_max_pixels,
            "video_min_pixels": data_cfg.video_min_pixels,
            "video_max_pixels": data_cfg.video_max_pixels,
            "video_fps": data_cfg.video_fps,
            "video_max_frames": data_cfg.video_max_frames,
        }
        for batch_dict in dataloader:
            batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            remain = max_samples - n
            if remain <= 0:
                break
            if len(batch) > remain:
                batch = batch.slice_select(0, remain)
            batches.append(batch)
            n += len(batch)
            if n >= max_samples:
                break
        return batches


    def _run_adaptive_rollout_probe(self) -> dict[str, Any]:
        """After validation: probe teacher rescue at fixed prefixes and update rollout length.

        Prefer student trajectories cached by ``_validate`` (same ``val_max_response_length``
        generations + Guard scores) so ARS probe does **not** re-rollout the student.
        Teacher continuations only evaluate the configured prefix cuts (e.g. 128/256/1024).

        Early exit: if harmful student defense fails < min_student_fails, keep current length
        and skip teacher continuations.
        """
        data_cfg = self.config.data
        prefix_lengths = [int(x) for x in (data_cfg.adaptive_rollout_prefix_lengths or [128, 256, 1024])]
        prefix_lengths = sorted(set(prefix_lengths))
        # Student probe rollout matches validation length; only truncate those
        # generations to the configured prefix lengths for teacher continuation.
        student_max_tokens = (
            int(data_cfg.val_max_response_length)
            if data_cfg.val_max_response_length is not None
            else int(data_cfg.max_response_length)
        )
        if any(L > student_max_tokens for L in prefix_lengths):
            print(
                f"[EOPSA][ARS] WARN: some prefix lengths {prefix_lengths} exceed "
                f"student/val max_tokens={student_max_tokens}; those L will count as short-fail"
            )
        current_len = self._get_student_max_response_length()
        metrics: dict[str, Any] = {
            "ars/enabled": 1.0,
            "ars/current_length": float(current_len),
            "ars/student_max_tokens": float(student_max_tokens),
        }

        if self.val_reward_fn is None:
            print("[EOPSA][ARS] val_reward_fn missing; skip")
            metrics["ars/skipped"] = 1.0
            return metrics

        # Prefer student trajectories already produced by `_validate` (same data source).
        if self._uses_separate_adaptive_rollout_files():
            self._ars_val_cache = None
            student_records, aligned_non_tensors, reused_val = [], None, False
        else:
            student_records, aligned_non_tensors, reused_val = self._load_ars_student_from_val_cache(
                max_samples=int(data_cfg.adaptive_rollout_max_samples)
            )
        if reused_val:
            metrics["ars/reused_val_rollouts"] = 1.0
            print(
                f"[EOPSA][ARS] reuse {len(student_records)} validation student rollouts "
                "(skip student regen)"
            )
        else:
            metrics["ars/reused_val_rollouts"] = 0.0
            probe_batches = self._collect_ars_batches()
            if not probe_batches:
                print("[EOPSA][ARS] empty probe set; skip")
                metrics["ars/skipped"] = 1.0
                return metrics

            # ---- Student rollout (fallback when no val cache) ----
            student_records = []
            source_batches: list[DataProto] = []
            self.actor_rollout_ref_wg.prepare_rollout_engine()
            try:
                for batch in probe_batches:
                    batch = deepcopy(batch)
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
                    )
                    student_gen = self._build_student_gen_batch(batch)
                    student_gen.meta_info.update(
                        {
                            "n": 1,
                            "temperature": self.config.worker.rollout.temperature,
                            "top_p": self.config.worker.rollout.top_p,
                            "max_tokens": student_max_tokens,
                            "image_min_pixels": self.config.data.image_min_pixels,
                            "image_max_pixels": self.config.data.image_max_pixels,
                            "video_min_pixels": self.config.data.video_min_pixels,
                            "video_max_pixels": self.config.data.video_max_pixels,
                            "video_fps": self.config.data.video_fps,
                            "video_max_frames": self.config.data.video_max_frames,
                        }
                    )
                    student_gen, pad_size = pad_dataproto_to_divisor(
                        student_gen, self.actor_rollout_ref_wg.world_size
                    )
                    student_output = self.actor_rollout_ref_wg.generate_sequences(student_gen)
                    student_output = unpad_dataproto(student_output, pad_size=pad_size)

                    response_ids = student_output.batch["responses"]
                    response_mask = student_output.batch["response_mask"]
                    data_types = batch.non_tensor_batch.get("data_type")
                    problem_types = batch.non_tensor_batch.get("problem_type")
                    questions = self._get_teacher_questions(batch)

                    for i in range(len(batch)):
                        dt = None if data_types is None else data_types[i]
                        pt = None if problem_types is None else problem_types[i]
                        if not self._is_harmful_probe_sample(dt, pt):
                            continue
                        cur_len = int(response_mask[i].sum().item())
                        token_ids = response_ids[i, :cur_len].tolist()
                        student_records.append(
                            {
                                "question": str(questions[i]),
                                "student_token_ids": token_ids,
                                "student_token_len": cur_len,
                                "student_response": self.tokenizer.decode(
                                    token_ids, skip_special_tokens=True
                                ),
                                "batch_local_index": len(source_batches),
                                "row": i,
                            }
                        )
                    source_batches.append(batch)
            finally:
                self.actor_rollout_ref_wg.release_rollout_engine()

            aligned_non_tensors = defaultdict(list)
            response_texts = []
            for rec in student_records:
                b = source_batches[rec["batch_local_index"]]
                row = rec["row"]
                response_texts.append(rec["student_response"])
                for key, value in b.non_tensor_batch.items():
                    if key in ("multi_modal_data", "raw_prompt_ids", "raw_prompt"):
                        continue
                    aligned_non_tensors[key].append(value[row])

            aligned_batch = DataProto.from_dict(
                tensors={"dummy": torch.zeros(len(student_records), 1)},
                non_tensors={k: np.array(v, dtype=object) for k, v in aligned_non_tensors.items()},
            )
            student_unsafe = self._score_response_texts_unsafe(aligned_batch, response_texts)
            for rec, unsafe in zip(student_records, student_unsafe):
                rec["student_unsafe"] = bool(unsafe)

        metrics["ars/n_harmful"] = float(len(student_records))
        if not student_records:
            print("[EOPSA][ARS] no harmful samples in probe set; keep length")
            metrics["ars/n_student_fails"] = 0.0
            metrics["ars/skipped"] = 1.0
            return metrics

        student_unsafe = [bool(r.get("student_unsafe")) for r in student_records]
        if aligned_non_tensors is None:
            aligned_non_tensors = defaultdict(list)
            for rec in student_records:
                for key, value in (rec.get("non_tensors") or {}).items():
                    aligned_non_tensors[key].append(value)

        n_fails = sum(1 for u in student_unsafe if u)
        metrics["ars/n_student_fails"] = float(n_fails)
        print(
            f"[EOPSA][ARS] harmful={len(student_records)} student_fails={n_fails} "
            f"(threshold={data_cfg.adaptive_rollout_min_student_fails})"
        )

        if n_fails < int(data_cfg.adaptive_rollout_min_student_fails):
            new_len, info = decide_adaptive_rollout_length(
                n_student_fails=n_fails,
                trr_by_prefix={},
                prefix_lengths=prefix_lengths,
                min_student_fails=int(data_cfg.adaptive_rollout_min_student_fails),
                trr_thresh=float(data_cfg.adaptive_rollout_trr_thresh),
                current_length=current_len,
                monotonic=bool(data_cfg.adaptive_rollout_monotonic),
            )
            self._adaptive_rollout_len = new_len
            metrics["ars/skipped"] = 1.0
            metrics["ars/recommended_length"] = float(new_len)
            metrics["ars/selected_length"] = float(new_len)
            metrics["ars/reason_good_enough"] = 1.0
            metrics["ars/reason"] = str(info.get("reason", "student_good_enough"))
            print(
                f"[EOPSA][ARS] student good enough (fails<{data_cfg.adaptive_rollout_min_student_fails}); "
                f"keep rollout_len={new_len}, skip teacher continuations"
            )
            self._last_ars_detail = {
                "global_step": int(self.global_step),
                "current_length": int(current_len),
                "selected_length": int(new_len),
                "decision": info,
                "prefix_lengths": prefix_lengths,
                "n_harmful": int(len(student_records)),
                "n_student_fails": int(n_fails),
                "trr_by_prefix": {},
                "unsafe_samples": [],
                "note": "skipped teacher continuations (student fails below threshold)",
            }
            return metrics

        # Teacher probe on ALL harmful samples; DSR denominator = full probe set.
        probe_recs = list(student_records)
        probe_non_tensors = {
            k: np.array(list(aligned_non_tensors[k]), dtype=object)
            for k in aligned_non_tensors
        }
        if "ground_truth" not in probe_non_tensors:
            probe_non_tensors["ground_truth"] = np.array([""] * len(probe_recs), dtype=object)
        if "prompt_text" not in probe_non_tensors:
            probe_non_tensors["prompt_text"] = np.array(
                [r["question"] for r in probe_recs], dtype=object
            )
        probe_source = DataProto.from_dict(
            tensors={"dummy": torch.zeros(len(probe_recs), 1)},
            non_tensors=probe_non_tensors,
        )

        # ---- Teacher alone (L=0) + continuations @ prefix_lengths ----
        # Use frozen teacher/ref weights when available.
        use_ref = bool(self.opsd_config.freeze_teacher_model)
        teacher_max_new = int(data_cfg.adaptive_rollout_teacher_max_tokens)
        teacher_alone_safe: list[bool] = []
        teacher_full_safe_by_prefix: dict[int, list[Optional[bool]]] = {
            L: [None] * len(probe_recs) for L in prefix_lengths
        }
        try:
            if use_ref:
                self.actor_rollout_ref_wg.prepare_rollout_engine_from_ref()
            else:
                print("[EOPSA][ARS] WARN: freeze_teacher_model=False; using actor weights as teacher")
                self.actor_rollout_ref_wg.prepare_rollout_engine()

            hints = self._get_self_distill_hints(probe_source)
            questions = probe_source.non_tensor_batch.get("teacher_question")
            if questions is None:
                questions = probe_source.non_tensor_batch.get("prompt_text")
            if questions is None:
                questions = np.array([r["question"] for r in probe_recs], dtype=object)

            gt_answers = probe_source.non_tensor_batch.get("ground_truth")
            if gt_answers is None:
                gt_answers = np.array([""] * len(probe_recs), dtype=object)

            teacher_base = construct_teacher_prompts(
                questions=questions,
                hints=hints,
                gt_answers=gt_answers,
                p_drop=0.0,
                tokenizer=self.tokenizer,
                processor=self.processor,
                template_path=self._resolve_teacher_template_path(),
                max_prompt_length=self.config.data.max_prompt_length,
                apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                problem_types=probe_source.non_tensor_batch.get("problem_type"),
                data_types=probe_source.non_tensor_batch.get("data_type"),
            )
            # vLLM rollout requires these multimodal meta keys even for text-only.
            rollout_meta = {
                "n": 1,
                "max_tokens": teacher_max_new,
                "temperature": self.config.worker.rollout.temperature,
                "top_p": self.config.worker.rollout.top_p,
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            }

            # Teacher alone
            alone_gen = deepcopy(teacher_base)
            alone_gen.meta_info.update(rollout_meta)
            alone_gen, pad_size = pad_dataproto_to_divisor(alone_gen, self.actor_rollout_ref_wg.world_size)
            alone_out = self.actor_rollout_ref_wg.generate_sequences(alone_gen)
            alone_out = unpad_dataproto(alone_out, pad_size=pad_size)
            alone_texts = []
            for i in range(len(probe_recs)):
                cur_len = int(alone_out.batch["response_mask"][i].sum().item())
                ids = alone_out.batch["responses"][i, :cur_len]
                alone_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
            alone_unsafe = self._score_response_texts_unsafe(probe_source, alone_texts)
            teacher_alone_safe = [not u for u in alone_unsafe]
            n_probe = len(probe_recs)
            alone_dsr = (
                sum(1 for s in teacher_alone_safe if s) / n_probe if n_probe else 0.0
            )
            metrics["ars/n_probe"] = float(n_probe)
            metrics["ars/teacher_alone_dsr"] = float(alone_dsr)
            # Legacy key: previously meant |S_unsafe ∩ T_alone_safe|; now full-set alone DSR count.
            metrics["ars/n_teacher_alone_safe"] = float(sum(1 for s in teacher_alone_safe if s))
            print(
                f"[EOPSA][ARS] teacher_alone_dsr={alone_dsr:.2%} "
                f"({int(metrics['ars/n_teacher_alone_safe'])}/{n_probe})"
            )

            for L in prefix_lengths:
                # Use up to L student tokens for every sample (short replies use full text).
                keep_idx = list(range(len(probe_recs)))
                if not keep_idx:
                    continue

                subset_base = teacher_base.index_select(keep_idx)
                prefixes = [probe_recs[i]["student_token_ids"][:L] for i in keep_idx]
                cont_gen = attach_token_prefix_to_raw_prompt_ids(deepcopy(subset_base), prefixes)
                cont_gen.meta_info.update(rollout_meta)
                cont_gen, pad_size = pad_dataproto_to_divisor(
                    cont_gen, self.actor_rollout_ref_wg.world_size
                )
                cont_out = self.actor_rollout_ref_wg.generate_sequences(cont_gen)
                cont_out = unpad_dataproto(cont_out, pad_size=pad_size)

                subset_source_non = {
                    k: np.array([probe_non_tensors[k][j] for j in keep_idx], dtype=object)
                    for k in probe_non_tensors
                }
                subset_source = DataProto.from_dict(
                    tensors={"dummy": torch.zeros(len(keep_idx), 1)},
                    non_tensors=subset_source_non,
                )
                full_texts = []
                for local_i, global_i in enumerate(keep_idx):
                    prefix_text = self.tokenizer.decode(
                        probe_recs[global_i]["student_token_ids"][:L],
                        skip_special_tokens=True,
                    )
                    cur_len = int(cont_out.batch["response_mask"][local_i].sum().item())
                    cont_ids = cont_out.batch["responses"][local_i, :cur_len]
                    cont_text = self.tokenizer.decode(cont_ids, skip_special_tokens=True)
                    full_texts.append(prefix_text + cont_text)

                full_unsafe = self._score_response_texts_unsafe(subset_source, full_texts)
                for local_i, global_i in enumerate(keep_idx):
                    teacher_full_safe_by_prefix[L][global_i] = not bool(full_unsafe[local_i])

        finally:
            self.actor_rollout_ref_wg.release_rollout_engine()

        trr_stats = compute_trr_by_prefix(
            student_token_lens=[r["student_token_len"] for r in probe_recs],
            teacher_alone_safe=teacher_alone_safe,
            student_unsafe=student_unsafe,
            teacher_full_safe_by_prefix=teacher_full_safe_by_prefix,
            prefix_lengths=prefix_lengths,
            short_prefix_as_fail=False,  # short samples already got a filled flag above
        )
        trr_by_prefix = {L: float(trr_stats[L]["trr"]) for L in prefix_lengths}
        for L, rate in trr_by_prefix.items():
            metrics[f"ars/trr@{L}"] = rate
            print(
                f"[EOPSA][ARS] TRR@{L}={rate:.2%} "
                f"(n={int(trr_stats[L]['total'])}, "
                f"safe={int(trr_stats[L]['success'])})"
            )

        new_len, info = decide_adaptive_rollout_length(
            n_student_fails=n_fails,
            trr_by_prefix=trr_by_prefix,
            prefix_lengths=prefix_lengths,
            min_student_fails=int(data_cfg.adaptive_rollout_min_student_fails),
            trr_thresh=float(data_cfg.adaptive_rollout_trr_thresh),
            current_length=current_len,
            monotonic=bool(data_cfg.adaptive_rollout_monotonic),
        )
        self._adaptive_rollout_len = new_len
        metrics["ars/recommended_length"] = float(new_len)
        metrics["ars/selected_length"] = float(new_len)
        metrics["ars/skipped"] = 0.0
        metrics["ars/reason"] = str(info.get("reason", ""))
        print(
            f"[EOPSA][ARS] update student_max_response_length: "
            f"{current_len} -> {new_len} (reason={info.get('reason')})"
        )

        # Rich payload for on-disk inspection (prefixes / teacher DSR / chosen step).
        per_sample = []
        for i, rec in enumerate(probe_recs):
            per_sample.append(
                {
                    "question": rec.get("question"),
                    "student_token_len": rec.get("student_token_len"),
                    "student_unsafe": bool(student_unsafe[i]) if i < len(student_unsafe) else None,
                    "teacher_alone_safe": bool(teacher_alone_safe[i]) if i < len(teacher_alone_safe) else None,
                    "teacher_full_safe_by_prefix": {
                        str(L): (
                            None
                            if teacher_full_safe_by_prefix[L][i] is None
                            else bool(teacher_full_safe_by_prefix[L][i])
                        )
                        for L in prefix_lengths
                    },
                }
            )
        self._last_ars_detail = {
            "global_step": int(self.global_step),
            "current_length": int(current_len),
            "selected_length": int(new_len),
            "decision": info,
            "prefix_lengths": prefix_lengths,
            "n_harmful": int(len(student_records)),
            "n_student_fails": int(n_fails),
            "metric": "trr_over_full_probe_set",
            "trr_by_prefix": {str(k): v for k, v in trr_by_prefix.items()},
            "teacher_alone_dsr": float(metrics.get("ars/teacher_alone_dsr", 0.0)),
            "trr_stats": {
                str(k): {kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in d.items()}
                for k, d in trr_stats.items()
            },
            "samples": per_sample,
            "reused_val_rollouts": bool(metrics.get("ars/reused_val_rollouts", 0.0)),
        }
        return metrics

    def _persist_ars_log(self, ars_metrics: dict[str, Any]) -> None:
        """Save ARS probe summary + per-prefix teacher results under the experiment dir."""
        detail = getattr(self, "_last_ars_detail", None)
        payload = {
            "global_step": int(self.global_step),
            "metrics": {k: v for k, v in ars_metrics.items() if k.startswith("ars/")},
            "detail": detail,
            "selected_length": ars_metrics.get("ars/selected_length")
            or ars_metrics.get("ars/recommended_length"),
        }
        self._save_ars_log(payload)
        # Compact index for quick scanning across steps.
        try:
            index_path = self._ars_log_dir / "index.jsonl"
            summary = {
                "global_step": int(self.global_step),
                "selected_length": payload["selected_length"],
                "n_student_fails": ars_metrics.get("ars/n_student_fails"),
                "skipped": ars_metrics.get("ars/skipped"),
                "reason": ars_metrics.get("ars/reason"),
                **{
                    k.replace("ars/", ""): v
                    for k, v in ars_metrics.items()
                    if k.startswith("ars/rescue@")
                },
            }
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[EOPSA][ARS] WARN: failed to append index.jsonl: {e}")

    def _should_skip_adaptive_rollout_probe(self) -> tuple[bool, str]:
        """Skip ARS length-update only when global_step is 0 or max_steps.

        This is *not* "skip the first/last validation event". If
        ``val_before_train=False``, there is no step-0 val, and the first
        mid-training val (e.g. step 25) still runs ARS.
        """
        data = self.config.data
        step = int(self.global_step)
        max_step = int(self.training_steps)
        # Default: run ARS iff step not in {0, max_steps}.
        if getattr(data, "adaptive_rollout_skip_first", True) and step == 0:
            return True, "skip_step0"
        if getattr(data, "adaptive_rollout_skip_last", True) and step == max_step:
            return True, "skip_max_step"
        return False, ""

    def _maybe_run_adaptive_rollout_probe(self, metrics: dict[str, Any]) -> None:
        if not getattr(self.config.data, "enable_adaptive_rollout", False):
            return
        skip, reason = self._should_skip_adaptive_rollout_probe()
        if skip:
            # Drop any unused val cache so it cannot leak into a later probe.
            self._ars_val_cache = None
            skip_metrics = {
                "ars/enabled": 1.0,
                "ars/skipped": 1.0,
                "ars/reason": reason,
                "ars/current_length": float(self._get_student_max_response_length()),
            }
            metrics.update(skip_metrics)
            print(
                f"[EOPSA][ARS] skip at step={self.global_step} "
                f"({reason}); keep length={self._get_student_max_response_length()}"
            )
            self._last_ars_detail = {
                "global_step": int(self.global_step),
                "skipped": True,
                "reason": reason,
                "selected_length": int(self._get_student_max_response_length()),
            }
            self._persist_ars_log(skip_metrics)
            return
        ars_metrics = self._run_adaptive_rollout_probe()
        metrics.update(ars_metrics)
        self._persist_ars_log(ars_metrics)

    def _make_opsd_batch(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate a full OPSD batch with multi-round generation.

        Steps:
            1. Load data batch with trajectories
            2. Student on-policy rollout (n=opsd.student_rollout_n)
            3. Navigator hint generation (n=K)
            4. Teacher trajectory generation (n=1, B*K prompts)
            5. Evaluate Teacher trajectories for binary rewards
            6. Compute Navigator GRPO advantage
            7. Assemble OPSD forward pass batch

        Returns:
            dict with all data needed for joint update.
        """
        timing_raw = {}

        # === Step 0: Load data ===
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        batch_size = len(batch.batch)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        use_gt_as_hint = self.opsd_config.use_gt_as_hint
        enable_sdpo = self.opsd_config.enable_sdpo
        self_distill_negative_off_policy = (
            use_gt_as_hint and self.opsd_config.self_distill_negative_off_policy
        )
        student_rollout_n = int(self.opsd_config.student_rollout_n)
        dynamic_sample_hint = bool(self.opsd_config.dynamic_sample_hint)
        teacher_rollout_n = int(self.opsd_config.teacher_rollout_n)
        K = 1 if (use_gt_as_hint or enable_sdpo) else self.opsd_config.num_hints
        skip_nav_grpo = (self.opsd_config.lambda_nav == 0) or use_gt_as_hint or enable_sdpo
        skip_off_policy = (
            ((self.opsd_config.alpha >= 1.0) and (not self_distill_negative_off_policy))
            or (use_gt_as_hint and (not self_distill_negative_off_policy))
            or enable_sdpo
        )

        harmful_max_response_length = self._get_student_max_response_length()
        # Benign always tracks the same student max_tokens as harmful.
        benign_max_response_length = harmful_max_response_length
        metrics["data/student_max_response_length"] = float(harmful_max_response_length)
        metrics["data/harmful_max_response_length"] = float(harmful_max_response_length)
        metrics["data/benign_max_response_length"] = float(benign_max_response_length)
        if self.config.data.enable_dynamic_max_response_length:
            metrics["data/dynamic_max_response_length_enabled"] = 1.0
        if getattr(self.config.data, "enable_adaptive_rollout", False):
            metrics["data/adaptive_rollout_enabled"] = 1.0

        student_gen_base = self._build_student_gen_batch(batch)
        sdpo_correct_solutions: Optional[list[str]] = None
        sdpo_incorrect_solutions: Optional[list[str]] = None

        # === Step 1: Student On-Policy Rollout ===
        # Dynamic / ARS-guided max_tokens apply to harmful and benign alike.
        if enable_sdpo:
            candidate_n = self.opsd_config.sdpo_num_candidates
            with timer("sdpo_candidate_rollout", timing_raw):
                sdpo_candidate_output = self._generate_student_rollout_output(
                    deepcopy(student_gen_base),
                    batch,
                    n=candidate_n,
                    harmful_max_tokens=harmful_max_response_length,
                    benign_max_tokens=benign_max_response_length,
                )

            with timer("sdpo_candidate_reward", timing_raw):
                sdpo_candidate_reward_tensor, sdpo_candidate_reward_metrics = self._compute_response_rewards(
                    batch, sdpo_candidate_output, repeat=candidate_n
                )
                metrics.update(
                    {
                        f"sdpo_candidate_reward/{k}": v
                        for k, v in reduce_metrics(sdpo_candidate_reward_metrics).items()
                    }
                )
                sdpo_candidate_scores = sdpo_candidate_reward_tensor.sum(dim=-1)
                (
                    sdpo_correct_solutions,
                    sdpo_incorrect_solutions,
                    sdpo_selection_metrics,
                ) = self._select_sdpo_feedback(sdpo_candidate_output, sdpo_candidate_scores, batch_size)
                metrics.update(sdpo_selection_metrics)

            with timer("student_rollout", timing_raw):
                student_output = self._generate_student_rollout_output(
                    deepcopy(student_gen_base),
                    batch,
                    n=student_rollout_n,
                    harmful_max_tokens=harmful_max_response_length,
                    benign_max_tokens=benign_max_response_length,
                )
        else:
            with timer("student_rollout", timing_raw):
                student_output = self._generate_student_rollout_output(
                    deepcopy(student_gen_base),
                    batch,
                    n=student_rollout_n,
                    harmful_max_tokens=harmful_max_response_length,
                    benign_max_tokens=benign_max_response_length,
                )

        expected_student_bs = batch_size * student_rollout_n
        if len(student_output) != expected_student_bs:
            raise RuntimeError(
                f"Unexpected student rollout batch size. expected={expected_student_bs}, got={len(student_output)}. "
                f"base_batch_size={batch_size}, student_rollout_n={student_rollout_n}"
            )

        if student_rollout_n > 1:
            batch = batch.repeat(repeat_times=student_rollout_n, interleave=True)
            # Preserve uid as the prompt/group id so GRPO-style grouping remains aligned
            # with the original query. If a per-response unique id is needed, use sample_uid.
            batch.non_tensor_batch["sample_uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            if enable_sdpo:
                if sdpo_correct_solutions is not None:
                    sdpo_correct_solutions = np.repeat(
                        np.array(sdpo_correct_solutions, dtype=object), student_rollout_n
                    ).tolist()
                if sdpo_incorrect_solutions is not None:
                    sdpo_incorrect_solutions = np.repeat(
                        np.array(sdpo_incorrect_solutions, dtype=object), student_rollout_n
                    ).tolist()

        # Merge student output back
        batch = batch.union(student_output)
        batch_size = len(batch.batch)
        metrics["opsd/student_rollout_n"] = float(student_rollout_n)
        metrics["opsd/student_rollout_effective_batch"] = float(batch_size)
        self._log_student_generations(batch, student_output)

        print(
            f"[OPSD] Step 1: Student rollout complete. "
            f"base_batch_size={expected_student_bs // student_rollout_n}, "
            f"student_rollout_n={student_rollout_n}, effective_batch_size={batch_size}"
        )

        student_old_lp = None
        student_outcome_advantages = None
        student_outcome_response_mask = None
        student_distill_response_mask = None
        reward_tensor = None
        reward_metrics = None
        need_student_reward = (
            enable_sdpo
            or self.opsd_config.enable_dynamic_verifier_hint
            or self.opsd_config.outcome_ppo_coef > 0
        )
        if need_student_reward:
            with timer("student_reward", timing_raw):
                reward_tensor, reward_metrics = self._compute_response_rewards(batch, student_output)
                metrics.update(
                    {
                        f"student_reward/{k}": v
                        for k, v in reduce_metrics(self._filter_numeric_reward_metrics(reward_metrics)).items()
                    }
                )

            if self.opsd_config.outcome_ppo_coef > 0:
                with timer("student_old_logprobs", timing_raw):
                    student_logprob_batch = DataProto.from_dict(
                        tensors={
                            "input_ids": student_output.batch["input_ids"],
                            "attention_mask": student_output.batch["attention_mask"],
                            "position_ids": student_output.batch["position_ids"],
                            "responses": student_output.batch["responses"],
                        },
                        non_tensors={
                            "multi_modal_data": batch.non_tensor_batch.get(
                                "multi_modal_data",
                                np.array([{}] * student_output.batch["input_ids"].shape[0], dtype=object),
                            ),
                            "uid": batch.non_tensor_batch["uid"],
                        },
                        meta_info={
                            "temperature": self.config.worker.rollout.temperature,
                            "image_min_pixels": self.config.data.image_min_pixels,
                            "image_max_pixels": self.config.data.image_max_pixels,
                            "video_min_pixels": self.config.data.video_min_pixels,
                            "video_max_pixels": self.config.data.video_max_pixels,
                            "video_fps": self.config.data.video_fps,
                            "video_max_frames": self.config.data.video_max_frames,
                        },
                    )
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(student_logprob_batch)
                    student_old_lp = old_log_probs.batch["old_log_probs"]

                student_outcome_advantages, _ = compute_advantage_return(
                    AdvantageEstimator.REINFORCE_PLUS_PLUS,
                    token_level_rewards=reward_tensor,
                    response_mask=student_output.batch["response_mask"],
                    gamma=torch.as_tensor(
                        self.config.algorithm.gamma,
                        dtype=reward_tensor.dtype,
                        device=reward_tensor.device,
                    ),
                )
                student_outcome_response_mask = student_output.batch["response_mask"]
                student_distill_response_mask = student_output.batch["response_mask"]
                if self.opsd_config.outcome_ppo_incorrect_only:
                    if "accuracy" in reward_metrics:
                        accuracy = torch.as_tensor(
                            reward_metrics["accuracy"],
                            dtype=student_outcome_advantages.dtype,
                            device=student_outcome_advantages.device,
                        )
                        incorrect_mask = (accuracy <= 0).to(student_output.batch["response_mask"].dtype)
                    else:
                        outcome_scores = reward_tensor.sum(dim=-1)
                        incorrect_mask = (outcome_scores <= 0).to(student_output.batch["response_mask"].dtype)
                    student_outcome_response_mask = (
                        student_output.batch["response_mask"] * incorrect_mask.unsqueeze(-1)
                    )
                    student_distill_response_mask = (
                        student_output.batch["response_mask"] * (1 - incorrect_mask).unsqueeze(-1)
                    )
                    metrics["opsd/outcome_num_incorrect"] = float(incorrect_mask.sum().item())
                    metrics["opsd/outcome_num_correct"] = float(
                        incorrect_mask.numel() - incorrect_mask.sum().item()
                    )
            if use_gt_as_hint and self.opsd_config.self_distill_mask_prefix_tokens > 0:
                student_distill_response_mask = _mask_prefix_response_tokens(
                    student_distill_response_mask
                    if student_distill_response_mask is not None
                    else student_output.batch["response_mask"],
                    self.opsd_config.self_distill_mask_prefix_tokens,
                )
                masked_prefix_tokens = (
                    student_output.batch["response_mask"].sum() - student_distill_response_mask.sum()
                ).item()
                metrics["opsd/self_distill_masked_prefix_tokens"] = float(masked_prefix_tokens)
                metrics["opsd/self_distill_mask_prefix_tokens"] = float(
                    self.opsd_config.self_distill_mask_prefix_tokens
                )
            metrics["opsd/outcome_reward_mean"] = (
                (reward_tensor * student_output.batch["response_mask"]).sum()
                / student_output.batch["response_mask"].sum().clamp(min=1)
            ).item()

        # === Step 2: Navigator Hint Generation (or SDPO/GT-as-hint) ===
        if enable_sdpo:
            nav_output = None
            print(
                "[OPSD] Step 2: SDPO candidate selection complete. "
                f"correct_cov={metrics['sdpo/correct_candidate_coverage']:.3f}, "
                f"incorrect_cov={metrics['sdpo/incorrect_candidate_coverage']:.3f}"
            )
        elif use_gt_as_hint:
            # Self-Distilled Reasoner: skip Navigator rollout. Use the preprocessed
            # reference trajectory directly as teacher-side extra context; fall back
            # to GT only if the dataset does not provide one.
            trajectories = batch.non_tensor_batch.get("trajectory")
            hints = []
            used_trajectory = 0
            for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
                trajectory = ""
                if trajectories is not None:
                    trajectory = truncate_text_by_tokens(
                        trajectories[idx],
                        self.tokenizer,
                        self.opsd_config.teacher_max_trajectory_length,
                    )
                if trajectory:
                    hints.append([trajectory])
                    used_trajectory += 1
                else:
                    hints.append([str(gt)])
            nav_output = None
            print(
                f"[OPSD] Step 2: Using reference trajectories as teacher context where available "
                f"(fallback to GT for {batch_size - used_trajectory}/{batch_size}). B={batch_size}"
            )
        else:
            navigator_student_trajectories = None
            if self.opsd_config.navigator_include_student_trajectory:
                navigator_student_trajectories = self._decode_student_trajectories(
                    student_output,
                    max_tokens=self.opsd_config.navigator_max_trajectory_length,
                )

            with timer("navigator_rollout", timing_raw):
                nav_gen = construct_navigator_prompts(
                    questions=batch.non_tensor_batch["prompt_text"],
                    trajectories=batch.non_tensor_batch["trajectory"],
                    student_trajectories=navigator_student_trajectories,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._nav_template_path,
                    max_traj_len=self.opsd_config.navigator_max_trajectory_length,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
                nav_gen.meta_info.update({
                    "n": K,
                    "temperature": self.config.worker.rollout.temperature,
                    "top_p": self.config.worker.rollout.top_p,
                    "max_tokens": self.opsd_config.navigator_response_length,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                })

                nav_gen, nav_pad_size = pad_dataproto_to_divisor(
                    nav_gen, self.actor_rollout_ref_wg.world_size
                )
                nav_output = self.actor_rollout_ref_wg.generate_sequences(nav_gen)
                nav_output = unpad_dataproto(nav_output, pad_size=nav_pad_size * K)

            # Decode hints
            nav_output.meta_info["n"] = K
            hints = decode_navigator_hints(nav_output, self.tokenizer)  # B × K
            self._log_navigator_hints(batch, hints)

            print(f"[OPSD] Step 2: Navigator generated {len(hints)} × {K} hints")

        # === Step 3: Teacher ===
        if enable_sdpo:
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop

            with timer("teacher_prompt_tokenize", timing_raw):
                teacher_gen = construct_sdpo_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    correct_solutions=sdpo_correct_solutions or [""] * batch_size,
                    incorrect_solutions=sdpo_incorrect_solutions or [""] * batch_size,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._sdpo_teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
            teacher_output = None
            teacher_uid = batch.non_tensor_batch["uid"]
            hint_correct = torch.ones(batch_size)
            metrics["nav/hint_accuracy"] = 1.0
            print(f"[OPSD] Step 3: SDPO teacher prompt tokenized (no generation). B={batch_size}")
            print("[OPSD] Step 4: Skipped (SDPO, no Teacher trajectories)")
        elif use_gt_as_hint:
            # Self-Distilled Reasoner: skip Teacher generation entirely.
            # Only tokenize Teacher prompts (with GT answer as hint, p_drop=0).
            # The Teacher forward pass happens during actor update, not here.
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop

            with timer("teacher_prompt_tokenize", timing_raw):
                # In self-distillation mode, GT is already provided as `hint`.
                # Pass empty reference answers so the teacher prompt cannot
                # accidentally include a second copy of the ground truth.
                empty_gt_answers = np.array([""] * batch_size, dtype=object)
                verifier_feedbacks, think_diagnoses, answer_diagnoses = self._extract_verifier_hint_fields(
                    reward_metrics, batch_size
                )
                teacher_gen = construct_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    hints=hints,
                    gt_answers=empty_gt_answers,
                    p_drop=p_drop,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    verifier_feedbacks=verifier_feedbacks,
                    think_diagnoses=think_diagnoses,
                    answer_diagnoses=answer_diagnoses,
                    template_path=self._resolve_teacher_template_path(),
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                    problem_types=self._get_problem_types(batch),
                    data_types=self._get_data_types(batch),
                )
                self._log_student_teacher_prompts(
                    batch,
                    teacher_gen,
                    hints=hints,
                    verifier_feedbacks=verifier_feedbacks,
                    think_diagnoses=think_diagnoses,
                    answer_diagnoses=answer_diagnoses,
                    reward_metrics=reward_metrics,
                )
            teacher_on_policy_gen = teacher_gen
            teacher_output = None
            if self_distill_negative_off_policy:
                with timer("teacher_rollout", timing_raw):
                    teacher_gen.meta_info.update({
                        "n": 1,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    })
                    teacher_gen, teacher_pad_size = pad_dataproto_to_divisor(
                        teacher_gen, self.actor_rollout_ref_wg.world_size
                    )
                    teacher_output = self.actor_rollout_ref_wg.generate_sequences(teacher_gen)
                    teacher_output = unpad_dataproto(teacher_output, pad_size=teacher_pad_size)
                print(
                    "[OPSD] Step 3: Teacher rollout complete for self-distill negative off-policy. "
                    f"B={batch_size}"
                )
            else:
                print(f"[OPSD] Step 3: Teacher prompt tokenized (no generation). B={batch_size}")

            # Step 4: Skip Teacher evaluation. Self-distill does not reward/filter
            # Teacher trajectories, even when a negative off-policy branch is enabled.
            teacher_uid = batch.non_tensor_batch["uid"]
            hint_correct = torch.ones(batch_size)  # GT hint → assume correct
            metrics["nav/hint_accuracy"] = 1.0
            if self_distill_negative_off_policy:
                print("[OPSD] Step 4: Skipped (Self-Distilled Reasoner negative off-policy, no Teacher filtering)")
            else:
                print("[OPSD] Step 4: Skipped (Self-Distilled Reasoner, no Teacher trajectories)")

        else:
            # Full OPSD / Static Navigator: Teacher generates trajectories
            # Curriculum is temporarily disabled: always keep GT available to Teacher.
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop
            teacher_reference_trajectories, teacher_student_trajectories = (
                self._build_teacher_context_trajectories(batch, student_output=student_output)
            )
            rollout_gt_answers = (
                batch.non_tensor_batch["ground_truth"]
                if self.opsd_config.off_policy_include_gt_answer
                else np.array([""] * batch_size, dtype=object)
            )

            with timer("teacher_rollout", timing_raw):
                verifier_feedbacks, think_diagnoses, answer_diagnoses = self._extract_verifier_hint_fields(
                    reward_metrics, batch_size
                )
                teacher_gen = construct_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    hints=hints,
                    gt_answers=rollout_gt_answers,
                    p_drop=p_drop,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    reference_trajectories=teacher_reference_trajectories,
                    student_trajectories=teacher_student_trajectories,
                    verifier_feedbacks=verifier_feedbacks,
                    think_diagnoses=think_diagnoses,
                    answer_diagnoses=answer_diagnoses,
                    template_path=self._resolve_teacher_template_path(),
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                    problem_types=self._get_problem_types(batch),
                    data_types=self._get_data_types(batch),
                )
                teacher_gen.meta_info.update({
                    "n": teacher_rollout_n,
                    "temperature": self.config.worker.rollout.temperature,
                    "top_p": self.config.worker.rollout.top_p,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                })

                teacher_gen, teacher_pad_size = pad_dataproto_to_divisor(
                    teacher_gen, self.actor_rollout_ref_wg.world_size
                )
                teacher_output = self.actor_rollout_ref_wg.generate_sequences(teacher_gen)
                teacher_output = unpad_dataproto(
                    teacher_output, pad_size=teacher_pad_size * teacher_rollout_n
                )

            print(f"[OPSD] Step 3: Teacher rollout complete. B*K*M = {batch_size * K * teacher_rollout_n}")

            teacher_on_policy_gen = teacher_gen
            if not self.opsd_config.off_policy_include_gt_answer:
                with timer("teacher_on_policy_prompt_tokenize", timing_raw):
                    teacher_on_policy_gen = self._construct_on_policy_teacher_prompts(
                        batch,
                        hints,
                        student_output=student_output,
                    )

            # === Step 4: Evaluate Teacher Trajectories ===
            with timer("teacher_reward", timing_raw):
                teacher_repeat = K * teacher_rollout_n
                teacher_gt = np.repeat(batch.non_tensor_batch["ground_truth"], teacher_repeat)
                teacher_rollout_uid = np.repeat(batch.non_tensor_batch["uid"], teacher_repeat)

                teacher_reward_batch = DataProto.from_dict(
                    tensors={
                        "responses": teacher_output.batch["responses"],
                        "response_mask": teacher_output.batch["response_mask"],
                    },
                    non_tensors={
                        "ground_truth": teacher_gt,
                        "uid": teacher_rollout_uid,
                    },
                )
                for key in batch.non_tensor_batch:
                    if key not in teacher_reward_batch.non_tensor_batch and key not in ("multi_modal_data", "raw_prompt_ids"):
                        teacher_reward_batch.non_tensor_batch[key] = np.repeat(
                            batch.non_tensor_batch[key], teacher_repeat
                        )

                reward_tensor, reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(teacher_reward_batch)
                )

                teacher_rollout_rewards = reward_tensor.sum(dim=-1)
                if "accuracy" in reward_metrics:
                    rollout_correctness = torch.as_tensor(
                        reward_metrics["accuracy"],
                        dtype=teacher_rollout_rewards.dtype,
                        device=teacher_rollout_rewards.device,
                    )
                else:
                    rollout_correctness = (teacher_rollout_rewards > 0).to(teacher_rollout_rewards.dtype)

                rollout_correctness = rollout_correctness.view(batch_size, K, teacher_rollout_n)
                if dynamic_sample_hint:
                    (
                        hint_group_scores,
                        selected_hint_indices,
                        selected_rollout_indices,
                        selected_has_correct,
                        selected_group_scores,
                    ) = self._select_dynamic_hint_targets(rollout_correctness)
                    hint_correct = hint_group_scores.reshape(-1)
                    teacher_uid = np.repeat(batch.non_tensor_batch["uid"], K)
                else:
                    hint_correct = rollout_correctness.reshape(batch_size * K, teacher_rollout_n).mean(dim=-1)
                    teacher_uid = np.repeat(batch.non_tensor_batch["uid"], K)

                reward_metrics_reduced = {
                    f"teacher_reward/{k}": v
                    for k, v in reduce_metrics(reward_metrics).items()
                }
                metrics.update(reward_metrics_reduced)

            metrics["nav/hint_accuracy"] = hint_correct.mean().item()
            if dynamic_sample_hint:
                metrics["nav/best_hint_accuracy"] = selected_group_scores.mean().item()
                metrics["opsd/dynamic_selected_hint_has_correct_rate"] = (
                    selected_has_correct.float().mean().item()
                )
                print(
                    "[OPSD] Step 4: Teacher reward computed. "
                    f"hint_group_acc={metrics['nav/hint_accuracy']:.3f}, "
                    f"best_hint_acc={metrics['nav/best_hint_accuracy']:.3f}"
                )
            else:
                print(f"[OPSD] Step 4: Teacher reward computed. Hint accuracy: {metrics['nav/hint_accuracy']:.3f}")

        # === Step 5: Navigator GRPO Advantage ===
        if nav_output is not None:
            nav_resp_len = nav_output.batch["responses"].shape[1]
            nav_response_mask = nav_output.batch["response_mask"]
        else:
            # SDPO / use_gt_as_hint: no Navigator output, use minimal placeholders
            nav_resp_len = 1
            nav_response_mask = torch.zeros(batch_size * K, nav_resp_len)

        if not skip_nav_grpo:
            nav_advantages = compute_navigator_advantage(
                hint_rewards=hint_correct,
                hint_uids=teacher_uid,
                K=K,
            )
            nav_advantages_expanded = nav_advantages.unsqueeze(-1).expand(-1, nav_resp_len) * nav_response_mask
            metrics["nav/advantage_std"] = nav_advantages.std().item()
            metrics["nav/advantage_mean"] = nav_advantages.mean().item()
        else:
            # Placeholder: zero advantages (won't be used)
            nav_advantages_expanded = torch.zeros(batch_size * K, nav_resp_len)
            metrics["nav/advantage_std"] = 0.0
            metrics["nav/advantage_mean"] = 0.0

        # === Step 6: Compute Navigator old_log_probs ===
        if not skip_nav_grpo:
            with timer("nav_old_logprobs", timing_raw):
                nav_logprob_batch = DataProto.from_dict(
                    tensors={
                        "input_ids": nav_output.batch["input_ids"],
                        "attention_mask": nav_output.batch["attention_mask"],
                        "position_ids": nav_output.batch["position_ids"],
                        "responses": nav_output.batch["responses"],
                    },
                    non_tensors={
                        "multi_modal_data": nav_output.non_tensor_batch.get(
                            "multi_modal_data",
                            np.array([{}] * nav_output.batch["input_ids"].shape[0], dtype=object),
                        ),
                        "uid": np.repeat(batch.non_tensor_batch["uid"], K),
                    },
                    meta_info={
                        "temperature": self.config.worker.rollout.temperature,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    },
                )
                nav_old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(nav_logprob_batch)
                nav_old_lp = nav_old_log_probs.batch["old_log_probs"]
        else:
            # Placeholder: zeros (won't be used)
            nav_old_lp = torch.zeros(batch_size * K, nav_resp_len)
            if enable_sdpo:
                print("[OPSD] Skipping Navigator (enable_sdpo=True)")
            elif use_gt_as_hint:
                print("[OPSD] Skipping Navigator (use_gt_as_hint=True, Self-Distilled Reasoner)")
            else:
                print("[OPSD] Skipping Navigator old_log_probs (lambda_nav=0, static navigator)")

        # === Step 7: Assemble combined batch for joint update ===
        # We pack all data with prefix keys into a single DataProto

        # --- Navigator data (included even when static, but lambda_nav=0 skips gradient) ---
        if nav_output is not None:
            nav_tensors = {
                "nav_input_ids": nav_output.batch["input_ids"],
                "nav_attention_mask": nav_output.batch["attention_mask"],
                "nav_position_ids": nav_output.batch["position_ids"],
                "nav_responses": nav_output.batch["responses"],
                "nav_response_mask": nav_response_mask,
                "nav_old_log_probs": nav_old_lp,
                "nav_advantages": nav_advantages_expanded,
            }
        else:
            # use_gt_as_hint: no Navigator data, use minimal placeholders
            nav_tensors = {
                "nav_input_ids": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_attention_mask": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_position_ids": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_responses": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_response_mask": nav_response_mask,
                "nav_old_log_probs": nav_old_lp,
                "nav_advantages": nav_advantages_expanded,
            }

        if dynamic_sample_hint and teacher_output is not None:
            on_indices = torch.arange(batch_size, device=student_output.batch["responses"].device)
            on_question_indices = on_indices
            num_valid_on_policy = len(on_indices)

            student_input_ids_rep = student_output.batch["input_ids"]
            student_attention_mask_rep = student_output.batch["attention_mask"]
            student_position_ids_rep = student_output.batch["position_ids"]
            student_responses_rep = student_output.batch["responses"]
            student_response_mask_rep = student_output.batch["response_mask"]

            selected_hint_indices_dev = selected_hint_indices.to(device=on_indices.device)
            selected_has_correct_dev = selected_has_correct.to(device=on_indices.device)
            selected_rollout_indices_dev = selected_rollout_indices.to(device=on_indices.device)
            selected_prompt_rows = on_question_indices * K + selected_hint_indices_dev

            teacher_prompts = teacher_on_policy_gen.batch["input_ids"][selected_prompt_rows]
            teacher_prompt_mask = teacher_on_policy_gen.batch["attention_mask"][selected_prompt_rows]
            teacher_prompt_position_ids = teacher_on_policy_gen.batch["position_ids"][selected_prompt_rows]

            invalid_hint_mask = ~selected_has_correct_dev
            if torch.any(invalid_hint_mask):
                with timer("teacher_on_policy_fallback_prompt_tokenize", timing_raw):
                    fallback_teacher_gen, trajectory_missing = self._construct_teacher_fallback_prompts(
                        batch,
                        repeat_k=1,
                        student_output=student_output,
                    )

                fallback_prompts = fallback_teacher_gen.batch["input_ids"]
                fallback_prompt_mask = fallback_teacher_gen.batch["attention_mask"]
                fallback_prompt_position_ids = fallback_teacher_gen.batch["position_ids"]

                teacher_prompts = teacher_prompts.clone()
                teacher_prompt_mask = teacher_prompt_mask.clone()
                teacher_prompt_position_ids = teacher_prompt_position_ids.clone()
                teacher_prompts[invalid_hint_mask] = fallback_prompts[invalid_hint_mask]
                teacher_prompt_mask[invalid_hint_mask] = fallback_prompt_mask[invalid_hint_mask]
                teacher_prompt_position_ids[invalid_hint_mask] = fallback_prompt_position_ids[invalid_hint_mask]
                metrics["opsd/on_policy_trajectory_fallback_count"] = float(invalid_hint_mask.sum().item())
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = float(trajectory_missing)
                metrics["opsd/on_policy_gt_fallback_count"] = float(trajectory_missing)
            else:
                metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                metrics["opsd/on_policy_gt_fallback_count"] = 0.0

            on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = _concat_prompt_and_response(
                prompt_ids=teacher_prompts,
                prompt_attention_mask=teacher_prompt_mask,
                prompt_position_ids=teacher_prompt_position_ids,
                response_ids=student_responses_rep,
                response_mask=student_response_mask_rep,
            )
            on_policy_tensors = {
                "on_student_input_ids": student_input_ids_rep,
                "on_student_attention_mask": student_attention_mask_rep,
                "on_student_position_ids": student_position_ids_rep,
                "on_student_responses": student_responses_rep,
                "on_student_response_mask": student_response_mask_rep,
                "on_teacher_input_ids": on_teacher_input_ids,
                "on_teacher_attention_mask": on_teacher_attention_mask,
                "on_teacher_position_ids": on_teacher_position_ids,
                "on_teacher_responses": student_responses_rep,
                "on_teacher_response_mask": student_response_mask_rep,
                "on_student_distill_response_mask": (
                    student_distill_response_mask if student_distill_response_mask is not None else student_response_mask_rep
                ),
            }
            if student_old_lp is not None and student_outcome_advantages is not None:
                on_policy_tensors["on_student_old_log_probs"] = student_old_lp
                on_policy_tensors["on_student_outcome_advantages"] = student_outcome_advantages
                on_policy_tensors["on_student_outcome_response_mask"] = student_outcome_response_mask

            nonempty_on_mask = (
                (student_attention_mask_rep.sum(dim=-1) > 0)
                & (student_response_mask_rep.sum(dim=-1) > 0)
                & (on_teacher_attention_mask.sum(dim=-1) > 0)
            )
            if not torch.all(nonempty_on_mask):
                on_indices = on_indices[nonempty_on_mask]
                on_question_indices = on_question_indices[nonempty_on_mask]
                selected_has_correct_dev = selected_has_correct_dev[nonempty_on_mask]
                selected_rollout_indices_dev = selected_rollout_indices_dev[nonempty_on_mask]
                selected_hint_indices_dev = selected_hint_indices_dev[nonempty_on_mask]
                num_valid_on_policy = int(nonempty_on_mask.sum().item())
                on_policy_tensors = {
                    key: tensor[nonempty_on_mask] for key, tensor in on_policy_tensors.items()
                }

            on_teacher_attention_mask = on_policy_tensors["on_teacher_attention_mask"]
            off_policy_tensors = {}
            valid_question_indices = on_question_indices[selected_has_correct_dev]
            num_valid = len(valid_question_indices)
            if not skip_off_policy and num_valid > 0:
                valid_hint_rows = valid_question_indices * K + selected_hint_indices_dev[selected_has_correct_dev]
                valid_rollout_rows = valid_hint_rows * teacher_rollout_n + selected_rollout_indices_dev[selected_has_correct_dev]
                teacher_correct_responses = teacher_output.batch["responses"][valid_rollout_rows]
                teacher_correct_resp_mask = teacher_output.batch["response_mask"][valid_rollout_rows]

                student_prompts_valid = student_output.batch["prompts"][valid_question_indices]
                student_prompt_len = student_prompts_valid.shape[1]
                student_prompt_mask_valid = student_output.batch["attention_mask"][valid_question_indices, :student_prompt_len]
                student_prompt_position_ids_valid = student_output.batch["position_ids"][
                    valid_question_indices, ..., :student_prompt_len
                ]
                off_student_input_ids, off_student_attention_mask, off_student_position_ids = _concat_prompt_and_response(
                    prompt_ids=student_prompts_valid,
                    prompt_attention_mask=student_prompt_mask_valid,
                    prompt_position_ids=student_prompt_position_ids_valid,
                    response_ids=teacher_correct_responses,
                    response_mask=teacher_correct_resp_mask,
                )

                off_policy_tensors = {
                    "off_student_input_ids": off_student_input_ids,
                    "off_student_attention_mask": off_student_attention_mask,
                    "off_student_position_ids": off_student_position_ids,
                    "off_student_responses": teacher_correct_responses,
                    "off_student_response_mask": teacher_correct_resp_mask,
                    "off_teacher_input_ids": teacher_output.batch["input_ids"][valid_rollout_rows],
                    "off_teacher_attention_mask": teacher_output.batch["attention_mask"][valid_rollout_rows],
                    "off_teacher_position_ids": teacher_output.batch["position_ids"][valid_rollout_rows],
                    "off_teacher_responses": teacher_correct_responses,
                    "off_teacher_response_mask": teacher_correct_resp_mask,
                }
            elif skip_off_policy:
                print("[OPSD] Skipping off-policy batch assembly (alpha=1.0, on-policy only)")

            metrics["opsd/num_valid_hints"] = num_valid
            metrics["opsd/num_total_hints"] = batch_size * K
            metrics["opsd/dynamic_selected_hint_count"] = float(batch_size)
        else:
            on_indices = torch.arange(batch_size * K, device=student_output.batch["responses"].device)
            if use_gt_as_hint or enable_sdpo:
                on_positive_mask = torch.ones_like(on_indices, dtype=torch.bool)
            else:
                on_positive_mask = hint_correct.to(device=on_indices.device) > 0
            num_valid_on_policy = len(on_indices)

            student_input_ids_rep = student_output.batch["input_ids"].repeat_interleave(K, dim=0)[on_indices]
            student_attention_mask_rep = student_output.batch["attention_mask"].repeat_interleave(K, dim=0)[on_indices]
            student_position_ids_rep = student_output.batch["position_ids"].repeat_interleave(K, dim=0)[on_indices]
            student_responses_rep = student_output.batch["responses"].repeat_interleave(K, dim=0)[on_indices]
            student_response_mask_rep = student_output.batch["response_mask"].repeat_interleave(K, dim=0)[on_indices]
            on_question_indices = on_indices // K if num_valid_on_policy > 0 else on_indices

            on_policy_use_fallback_prompt = False
            if teacher_output is not None:
                teacher_prompts = teacher_on_policy_gen.batch["input_ids"][on_indices]
                teacher_prompt_mask = teacher_on_policy_gen.batch["attention_mask"][on_indices]
                teacher_prompt_position_ids = teacher_on_policy_gen.batch["position_ids"][on_indices]

                invalid_hint_mask = ~on_positive_mask
                if (not use_gt_as_hint) and (not enable_sdpo) and torch.any(invalid_hint_mask):
                    with timer("teacher_on_policy_fallback_prompt_tokenize", timing_raw):
                        fallback_teacher_gen, trajectory_missing = self._construct_teacher_fallback_prompts(
                            batch,
                            repeat_k=K,
                            student_output=student_output,
                        )

                    fallback_prompts = fallback_teacher_gen.batch["input_ids"][on_indices]
                    fallback_prompt_mask = fallback_teacher_gen.batch["attention_mask"][on_indices]
                    fallback_prompt_position_ids = fallback_teacher_gen.batch["position_ids"][on_indices]

                    teacher_prompts = teacher_prompts.clone()
                    teacher_prompt_mask = teacher_prompt_mask.clone()
                    teacher_prompt_position_ids = teacher_prompt_position_ids.clone()
                    teacher_prompts[invalid_hint_mask] = fallback_prompts[invalid_hint_mask]
                    teacher_prompt_mask[invalid_hint_mask] = fallback_prompt_mask[invalid_hint_mask]
                    teacher_prompt_position_ids[invalid_hint_mask] = fallback_prompt_position_ids[invalid_hint_mask]
                    on_policy_use_fallback_prompt = True
                    metrics["opsd/on_policy_trajectory_fallback_count"] = float(invalid_hint_mask.sum().item())
                    metrics["opsd/on_policy_fallback_missing_trajectory_count"] = float(trajectory_missing)
                    metrics["opsd/on_policy_gt_fallback_count"] = float(trajectory_missing)
                else:
                    metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                    metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                    metrics["opsd/on_policy_gt_fallback_count"] = 0.0
            else:
                teacher_prompts = teacher_gen.batch["input_ids"]
                teacher_prompt_mask = teacher_gen.batch["attention_mask"]
                teacher_prompt_position_ids = teacher_gen.batch["position_ids"]
                metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                metrics["opsd/on_policy_gt_fallback_count"] = 0.0

            student_resp_for_teacher = (
                student_responses_rep if teacher_output is not None else student_output.batch["responses"]
            )
            student_resp_mask_for_teacher = (
                student_response_mask_rep if teacher_output is not None else student_output.batch["response_mask"]
            )

            on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = _concat_prompt_and_response(
                prompt_ids=teacher_prompts,
                prompt_attention_mask=teacher_prompt_mask,
                prompt_position_ids=teacher_prompt_position_ids,
                response_ids=student_resp_for_teacher,
                response_mask=student_resp_mask_for_teacher,
            )
            on_teacher_responses = student_resp_for_teacher
            on_teacher_response_mask = student_resp_mask_for_teacher

            if student_distill_response_mask is not None:
                on_student_distill_response_mask = student_distill_response_mask[on_indices]
            else:
                if on_policy_use_fallback_prompt:
                    on_student_distill_response_mask = student_response_mask_rep
                else:
                    on_student_distill_response_mask = (
                        student_response_mask_rep
                        * on_positive_mask[on_indices].unsqueeze(-1).to(student_response_mask_rep.dtype)
                    )

            on_policy_tensors = {
                "on_student_input_ids": student_input_ids_rep,
                "on_student_attention_mask": student_attention_mask_rep,
                "on_student_position_ids": student_position_ids_rep,
                "on_student_responses": student_responses_rep,
                "on_student_response_mask": student_response_mask_rep,
                "on_teacher_input_ids": on_teacher_input_ids,
                "on_teacher_attention_mask": on_teacher_attention_mask,
                "on_teacher_position_ids": on_teacher_position_ids,
                "on_teacher_responses": on_teacher_responses,
                "on_teacher_response_mask": on_teacher_response_mask,
                "on_student_distill_response_mask": on_student_distill_response_mask,
            }
            if student_old_lp is not None and student_outcome_advantages is not None:
                on_policy_tensors["on_student_old_log_probs"] = student_old_lp[on_indices]
                on_policy_tensors["on_student_outcome_advantages"] = student_outcome_advantages[on_indices]
                on_policy_tensors["on_student_outcome_response_mask"] = student_outcome_response_mask[on_indices]

            if num_valid_on_policy > 0:
                nonempty_on_mask = (
                    (student_attention_mask_rep.sum(dim=-1) > 0)
                    & (student_response_mask_rep.sum(dim=-1) > 0)
                    & (on_teacher_attention_mask.sum(dim=-1) > 0)
                    & (on_teacher_response_mask.sum(dim=-1) > 0)
                )
                if not torch.all(nonempty_on_mask):
                    on_indices = on_indices[nonempty_on_mask]
                    on_question_indices = on_question_indices[nonempty_on_mask]
                    on_positive_mask = on_positive_mask[on_indices]
                    num_valid_on_policy = int(nonempty_on_mask.sum().item())
                    on_policy_tensors = {
                        key: tensor[nonempty_on_mask] for key, tensor in on_policy_tensors.items()
                    }
                else:
                    on_positive_mask = on_positive_mask[on_indices]

            on_teacher_attention_mask = on_policy_tensors["on_teacher_attention_mask"]
            valid_indices = on_indices[on_positive_mask]
            valid_question_indices = on_question_indices[on_positive_mask]

            off_policy_tensors = {}
            num_valid = 0
            if not skip_off_policy:
                if self_distill_negative_off_policy:
                    valid_indices = torch.arange(
                        batch_size,
                        device=student_output.batch["responses"].device,
                        dtype=torch.long,
                    )
                    valid_question_indices = valid_indices
                num_valid = len(valid_indices)

                if num_valid > 0:
                    teacher_correct_responses = teacher_output.batch["responses"][valid_indices]
                    teacher_correct_resp_mask = teacher_output.batch["response_mask"][valid_indices]

                    student_prompts_valid = student_output.batch["prompts"][valid_question_indices]
                    student_prompt_len = student_prompts_valid.shape[1]
                    student_prompt_mask_valid = student_output.batch["attention_mask"][valid_question_indices, :student_prompt_len]
                    student_prompt_position_ids_valid = student_output.batch["position_ids"][
                        valid_question_indices, ..., :student_prompt_len
                    ]
                    off_student_input_ids, off_student_attention_mask, off_student_position_ids = _concat_prompt_and_response(
                        prompt_ids=student_prompts_valid,
                        prompt_attention_mask=student_prompt_mask_valid,
                        prompt_position_ids=student_prompt_position_ids_valid,
                        response_ids=teacher_correct_responses,
                        response_mask=teacher_correct_resp_mask,
                    )

                    off_policy_tensors = {
                        "off_student_input_ids": off_student_input_ids,
                        "off_student_attention_mask": off_student_attention_mask,
                        "off_student_position_ids": off_student_position_ids,
                        "off_student_responses": teacher_correct_responses,
                        "off_student_response_mask": teacher_correct_resp_mask,
                        "off_teacher_input_ids": teacher_output.batch["input_ids"][valid_indices],
                        "off_teacher_attention_mask": teacher_output.batch["attention_mask"][valid_indices],
                        "off_teacher_position_ids": teacher_output.batch["position_ids"][valid_indices],
                        "off_teacher_responses": teacher_correct_responses,
                        "off_teacher_response_mask": teacher_correct_resp_mask,
                    }
            else:
                print("[OPSD] Skipping off-policy batch assembly (alpha=1.0, on-policy only)")

            metrics["opsd/num_valid_hints"] = num_valid
            metrics["opsd/num_total_hints"] = batch_size * K

        # Pad all tensors to same batch size (nav: B*K, on: B*K, off: num_valid)
        # We'll handle different sizes in the actor by processing each group separately
        # For now, pack them all into a single DataProto with separate key prefixes

        # Find the max sequence length across all groups to pad
        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)
        all_tensors.update(off_policy_tensors)

        # We need all tensors to have the same batch size for DataProto
        # Strategy: pad smaller batches to B*K (the largest group)
        target_bs = batch_size * K

        def pad_batch_to_size(tensor_dict, target_size, prefix=""):
            """Pad a dict of tensors along batch dim to target_size."""
            result = {}
            for key, tensor in tensor_dict.items():
                if tensor.shape[0] < target_size:
                    pad_size = target_size - tensor.shape[0]
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = pad_size
                    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                    result[key] = torch.cat([tensor, padding], dim=0)
                elif tensor.shape[0] > target_size:
                    result[key] = tensor[:target_size]
                else:
                    result[key] = tensor
            return result

        # Pad all smaller groups to B*K so DataProto sees a consistent batch
        # dimension. The actor uses num_valid_on_policy/num_valid_off_policy
        # to ignore the padded tail.
        if on_policy_tensors:
            on_policy_tensors = pad_batch_to_size(on_policy_tensors, target_bs)
        if off_policy_tensors:
            off_policy_tensors = pad_batch_to_size(off_policy_tensors, target_bs)

        # Rebuild all_tensors
        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)
        if off_policy_tensors:
            all_tensors.update(off_policy_tensors)

        # --- Multi-modal data for OPSD forward passes ---
        # All three roles (Nav, Student, Teacher) may need pixel_values because
        # the question text contains <image> tags. We pass multi_modal_data from
        # the original batch so that _process_multi_modal_inputs in fsdp_workers
        # can construct multi_modal_inputs on the worker side.
        #
        # Layout: on-policy Student/Teacher share the same image (repeated K times).
        # Nav and off-policy similarly reference the same original images.
        original_mmd = batch.non_tensor_batch.get("multi_modal_data")
        original_uid = batch.non_tensor_batch.get("uid")
        original_data_type = batch.non_tensor_batch.get("data_type")

        def _repeat_and_pad(arr, repeat_k, target_size, pad_value):
            """Repeat each element K times, then pad to target_size."""
            if arr is None:
                return np.array([pad_value] * target_size, dtype=object)
            repeated = np.repeat(arr, repeat_k)
            if len(repeated) < target_size:
                pad = np.array([pad_value] * (target_size - len(repeated)), dtype=object)
                repeated = np.concatenate([repeated, pad])
            return repeated[:target_size]

        nav_mmd = _repeat_and_pad(original_mmd, K, target_bs, {})
        nav_uid = _repeat_and_pad(original_uid, K, target_bs, "__nav_pad__")
        if num_valid_on_policy > 0:
            on_question_indices_np = on_question_indices.detach().cpu().numpy()
            on_mmd = original_mmd[on_question_indices_np] if original_mmd is not None else None
            on_uid = original_uid[on_question_indices_np] if original_uid is not None else None
            on_data_type = (
                original_data_type[on_question_indices_np]
                if original_data_type is not None
                else None
            )
        else:
            on_mmd = None
            on_uid = None
            on_data_type = None
        on_student_mmd = _repeat_and_pad(on_mmd, 1, target_bs, {})
        on_student_uid = _repeat_and_pad(on_uid, 1, target_bs, "__on_pad__")
        on_student_data_type = _repeat_and_pad(on_data_type, 1, target_bs, "__on_pad__")
        if num_valid > 0:
            valid_question_indices_np = valid_question_indices.detach().cpu().numpy()
            off_mmd = original_mmd[valid_question_indices_np] if original_mmd is not None else None
            off_uid = original_uid[valid_question_indices_np] if original_uid is not None else None
            off_data_type = (
                original_data_type[valid_question_indices_np]
                if original_data_type is not None
                else None
            )
        else:
            off_mmd = None
            off_uid = None
            off_data_type = None
        off_mmd = _repeat_and_pad(off_mmd, 1, target_bs, {})
        off_uid = _repeat_and_pad(off_uid, 1, target_bs, "__off_pad__")
        off_data_type = _repeat_and_pad(off_data_type, 1, target_bs, "__off_pad__")

        non_tensors = {
            "nav_multi_modal_data": nav_mmd,
            "nav_uid": nav_uid,
            "on_multi_modal_data": on_student_mmd,
            "on_uid": on_student_uid,
            "on_data_type": on_student_data_type,
            "off_multi_modal_data": off_mmd,
            "off_uid": off_uid,
            "off_data_type": off_data_type,
        }

        combined_batch = DataProto.from_dict(
            tensors=all_tensors,
            non_tensors=non_tensors,
            meta_info={
                "temperature": self.config.worker.rollout.temperature,
                "loss_type": "opsd",
                "opsd_config": {
                    "alpha": self.opsd_config.alpha,
                    "lambda_nav": self.opsd_config.lambda_nav,
                    "distillation_loss_type": self.opsd_config.distillation_loss_type,
                    "opsd_kl_penalty": self.opsd_config.opsd_kl_penalty,
                    "distillation_topk": self.opsd_config.distillation_topk,
                    "distillation_topk_source": self.opsd_config.distillation_topk_source,
                    "distillation_add_tail": self.opsd_config.distillation_add_tail,
                    "distill_token_selection": self.opsd_config.distill_token_selection,
                    "distill_token_keep_ratio": self.opsd_config.distill_token_keep_ratio,
                    "token_filter": self.opsd_config.token_filter,
                    "token_filter_mode": self.opsd_config.token_filter_mode,
                    "token_filter_action": self.opsd_config.token_filter_action,
                    "token_filter_top_n": self.opsd_config.token_filter_top_n,
                    "token_filter_log_updated": self.opsd_config.token_filter_log_updated,
                    "token_filter_log_max_tokens": self.opsd_config.token_filter_log_max_tokens,
                    "token_filter_path": self.opsd_config.token_filter_path,
                    "token_filter_keep_categories": self.opsd_config.token_filter_keep_categories,
                    "token_filter_drop_categories": self.opsd_config.token_filter_drop_categories,
                    "token_filter_classifier": self.opsd_config.token_filter_classifier,
                    "token_filter_student_source": self.opsd_config.token_filter_student_source,
                    "token_filter_benign_mode": self.opsd_config.token_filter_benign_mode,
                    "token_filter_benign_action": self.opsd_config.token_filter_benign_action,
                    "token_filter_benign_top_n": self.opsd_config.token_filter_benign_top_n,
                    "token_filter_benign_ratio": getattr(
                        self.opsd_config, "token_filter_benign_ratio", None
                    ),
                    "keep_mask": getattr(self.opsd_config, "keep_mask", False),
                    # legacy aliases for older actor code paths
                    "same_token_filter": self.opsd_config.token_filter,
                    "same_token_filter_mode": self.opsd_config.token_filter_mode,
                    "same_token_filter_top_n": self.opsd_config.token_filter_top_n,
                    "same_token_filter_log_updated": self.opsd_config.token_filter_log_updated,
                    "same_token_filter_log_max_tokens": self.opsd_config.token_filter_log_max_tokens,
                    "dynamic_sample_hint": self.opsd_config.dynamic_sample_hint,
                    "teacher_rollout_n": self.opsd_config.teacher_rollout_n,
                    "outcome_ppo_coef": self.opsd_config.outcome_ppo_coef,
                    "outcome_ppo_incorrect_only": self.opsd_config.outcome_ppo_incorrect_only,
                    "nav_clip_ratio_low": self.opsd_config.nav_clip_ratio_low,
                    "nav_clip_ratio_high": self.opsd_config.nav_clip_ratio_high,
                    "nav_clip_ratio_dual": self.opsd_config.nav_clip_ratio_dual,
                    "self_distill_negative_off_policy": self.opsd_config.self_distill_negative_off_policy,
                    "self_distill_negative_off_policy_coef": self.opsd_config.self_distill_negative_off_policy_coef,
                    "self_distill_mask_prefix_tokens": self.opsd_config.self_distill_mask_prefix_tokens,
                    "overlap_topk": self.opsd_config.overlap_topk,
                    "enable_topk_overlap_metrics": self.opsd_config.enable_topk_overlap_metrics,
                    "stgca_teacher_sync_interval": self.opsd_config.stgca_teacher_sync_interval,
                    "enable_oracle_correction": self.opsd_config.enable_oracle_correction,
                    "oracle_mix": self.opsd_config.oracle_mix,
                    "oracle_alpha": self.opsd_config.oracle_alpha,
                    "oracle_lambda": self.opsd_config.oracle_lambda,
                    "oracle_delta_clip": self.opsd_config.oracle_delta_clip,
                    "oracle_residual_relu": self.opsd_config.oracle_residual_relu,
                    "global_step": self.global_step,
                },
                "num_valid_off_policy": num_valid,
                "num_valid_on_policy": num_valid_on_policy,
                "global_token_num": torch.sum(
                    on_teacher_attention_mask, dim=-1
                ).tolist(),
                # Pixel config for _process_multi_modal_inputs
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            },
        )

        # Timing metrics
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        return combined_batch, batch, metrics

    def _load_training_token_stats(self) -> None:
        """Resume cumulative token accounting from ckpt dir if present."""
        path = self._training_token_stats_path
        if not path.exists():
            return
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[OPSD] Failed to load training_token_stats from {path}: {exc}")
            return
        for key in (
            "total_train_tokens",
            "total_rollout_tokens",
            "total_gen_time_s",
            "total_train_time_s",
            "total_steps",
            "total_data_count",
            "avg_train_tokens_per_data",
        ):
            if key in payload:
                self._training_token_stats[key] = float(payload[key])
        print(
            f"[OPSD] Loaded training_token_stats: "
            f"steps={int(self._training_token_stats['total_steps'])}, "
            f"train_tokens={self._training_token_stats['total_train_tokens']:.0f}, "
            f"rollout_tokens={self._training_token_stats['total_rollout_tokens']:.0f}, "
            f"gen_time={self._training_token_stats['total_gen_time_s']:.1f}s, "
            f"train_time={self._training_token_stats['total_train_time_s']:.1f}s, "
            f"data={int(self._training_token_stats['total_data_count'])}"
        )

    def _save_training_token_stats(self) -> None:
        """Persist cumulative token accounting under the checkpoint directory."""
        stats = self._training_token_stats
        data_count = float(stats["total_data_count"])
        stats["avg_train_tokens_per_data"] = (
            float(stats["total_train_tokens"]) / data_count if data_count > 0 else 0.0
        )
        total_gen_time_s = float(stats["total_gen_time_s"])
        total_train_time_s = float(stats["total_train_time_s"])
        payload = {
            "total_train_tokens": float(stats["total_train_tokens"]),
            "total_rollout_tokens": float(stats["total_rollout_tokens"]),
            "total_gen_time_s": total_gen_time_s,
            "total_train_time_s": total_train_time_s,
            "total_gen_time_h": round(total_gen_time_s / 3600.0, 2),
            "total_train_time_h": round(total_train_time_s / 3600.0, 2),
            "total_steps": int(stats["total_steps"]),
            "total_data_count": int(stats["total_data_count"]),
            "avg_train_tokens_per_data": float(stats["avg_train_tokens_per_data"]),
            "updated_at_step": int(self.global_step),
        }
        path = self._training_token_stats_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _update_training_token_stats(
        self,
        metrics: dict[str, Any],
        batch: Optional[DataProto] = None,
    ) -> None:
        """Accumulate train/rollout tokens, gen/train wall time, steps; write ckpt file."""
        # Actor already all_reduces these to global totals; reduce_metrics then means
        # identical per-rank values. Divide by ppo_epochs so multi-epoch updates do not
        # multiply unique tokens.
        ppo_epochs = int(getattr(self.config.worker.actor, "ppo_epochs", 1) or 1)
        ppo_epochs = max(ppo_epochs, 1)

        rollout_tokens = metrics.get("opsd/token_filter_tokens_before")
        if rollout_tokens is None:
            rollout_tokens = metrics.get("opsd/same_token_filter_tokens_before")
        if rollout_tokens is None and batch is not None:
            for key in ("response_mask", "on_student_response_mask"):
                if key in batch.batch:
                    rollout_tokens = float(batch.batch[key].sum().item())
                    break

        train_tokens = metrics.get("opsd/token_filter_tokens_trained")
        main_tokens = metrics.get("opsd/token_filter_tokens_main")
        if main_tokens is None:
            main_tokens = metrics.get("opsd/same_token_filter_tokens_after")
        # Compat: older actor builds left tokens_trained stuck at 0 while tokens_main was correct.
        if train_tokens is None or (
            float(train_tokens) <= 0 and main_tokens is not None and float(main_tokens) > 0
        ):
            train_tokens = main_tokens
        if train_tokens is None:
            train_tokens = rollout_tokens

        data_count = metrics.get("opsd/student_rollout_effective_batch")
        if data_count is None and batch is not None:
            data_count = float(len(batch))

        if rollout_tokens is not None:
            self._training_token_stats["total_rollout_tokens"] += float(rollout_tokens) / ppo_epochs
        if train_tokens is not None:
            self._training_token_stats["total_train_tokens"] += float(train_tokens) / ppo_epochs
        if data_count is not None:
            self._training_token_stats["total_data_count"] += float(data_count)
        self._training_token_stats["total_steps"] += 1.0

        gen_time = metrics.get("timing_s/gen")
        if gen_time is not None:
            self._training_token_stats["total_gen_time_s"] += float(gen_time)
        train_time = metrics.get("timing_s/update_actor")
        if train_time is not None:
            self._training_token_stats["total_train_time_s"] += float(train_time)

        data_total = float(self._training_token_stats["total_data_count"])
        avg = (
            float(self._training_token_stats["total_train_tokens"]) / data_total
            if data_total > 0
            else 0.0
        )
        self._training_token_stats["avg_train_tokens_per_data"] = avg

        metrics["train/total_train_tokens"] = float(self._training_token_stats["total_train_tokens"])
        metrics["train/total_rollout_tokens"] = float(self._training_token_stats["total_rollout_tokens"])
        metrics["train/total_gen_time_s"] = float(self._training_token_stats["total_gen_time_s"])
        metrics["train/total_train_time_s"] = float(self._training_token_stats["total_train_time_s"])
        metrics["train/total_gen_time_h"] = round(
            float(self._training_token_stats["total_gen_time_s"]) / 3600.0, 2
        )
        metrics["train/total_train_time_h"] = round(
            float(self._training_token_stats["total_train_time_s"]) / 3600.0, 2
        )
        metrics["train/total_steps"] = float(self._training_token_stats["total_steps"])
        metrics["train/total_data_count"] = float(self._training_token_stats["total_data_count"])
        metrics["train/avg_train_tokens_per_data"] = avg

        self._save_training_token_stats()

    def fit(self):
        """The OPSD training loop.

        Per step:
          1. Student rollout, Navigator hint generation, Teacher rollout
          2. Evaluate Teacher trajectories → binary rewards
          3. Navigator GRPO advantage
          4. Joint update: Navigator GRPO + OPSD distillation
        """
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="OPSD Training", position=0)
        val_metrics = None

        # Load checkpoint
        self._load_checkpoint()
        self._load_training_token_stats()
        main_tqdm.update(self.global_step)

        # Validation before training (val_freq==0 disables all validation).
        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_before_train
            and self.config.trainer.val_freq != 0
        ):
            val_metrics = self._validate()
            self._maybe_run_adaptive_rollout_probe(val_metrics)
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)

        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            rollout_len = self._get_student_max_response_length()
            metrics["train/rollout_length"] = float(rollout_len)
            metrics["data/student_max_response_length"] = float(rollout_len)
            metrics["data/harmful_max_response_length"] = float(rollout_len)
            metrics["data/benign_max_response_length"] = float(rollout_len)
            with timer("step", timing_raw):
                # === GENERATION PHASE ===
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    combined_batch, original_batch, gen_metrics = self._make_opsd_batch(metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                    metrics.update(gen_metrics)

                # === JOINT UPDATE ===
                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(combined_batch)

                self._extract_topk_overlap_actor_metrics(actor_output)
                self._extract_stf_drop_updated_token_logs(actor_output)
                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                # === VALIDATION ===
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                    with timer("ars", timing_raw):
                        self._maybe_run_adaptive_rollout_probe(metrics)

                # === CHECKPOINT ===
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # Collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_timing_metrics(batch=original_batch, timing_raw=timing_raw))

            self._update_training_token_stats(metrics, batch=original_batch)
            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # Final validation:
        #   val_freq > 0  → periodic (+ final if last step was not a val step)
        #   val_freq < 0  → final only (e.g. -1)
        #   val_freq == 0 → disabled (no val at all)
        if self.val_reward_fn is not None and self.config.trainer.val_freq != 0:
            if (
                val_metrics is None
                or self.config.trainer.val_freq < 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
                self._maybe_run_adaptive_rollout_probe(val_metrics)
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

    def _create_logger(self):
        """Create logger instance."""
        from ..utils.logger import Tracker
        return Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())


class RaySimpleGRPOOPSDTrainer(RayOPSDTrainer):
    """Separate minimal GRPO+OPSD trainer without Navigator logic."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("Simple GRPO+OPSD mode requires algorithm.adv_estimator=grpo.")
        if self.config.worker.rollout.n <= 1:
            raise ValueError("Simple GRPO+OPSD mode requires worker.rollout.n > 1.")
        if self.opsd_config.enable_grpo_opsd_joint:
            if self.opsd_config.distillation_loss_type != "kl":
                raise ValueError(
                    "GRPO+OPSD joint mode requires opsd.distillation_loss_type=kl."
                )
            print("[OPSD] Mode: GRPO+OPSD Joint (GRPO PG + sampled-token KL)")
            print(f"[EOPSA]   - rollout.n: {self.config.worker.rollout.n}")
            print(f"[EOPSA]   - opsd_loss_coef: {self.opsd_config.opsd_loss_coef}")
            print(f"[EOPSA]   - freeze_teacher_model: {self.opsd_config.freeze_teacher_model}")
            print(
                f"[EOPSA]   - teacher_sync_interval: "
                f"{self.opsd_config.stgca_teacher_sync_interval}"
            )
        else:
            print("[OPSD] Mode: Simple GRPO+RLSD (no Navigator)")
            print(f"[EOPSA]   - rollout.n: {self.config.worker.rollout.n}")
            print(f"[EOPSA]   - stgca_lambda: {self.opsd_config.stgca_lambda}")
            print(f"[EOPSA]   - stgca_lambda_warmup_steps: {self.opsd_config.stgca_lambda_warmup_steps}")
            print(f"[EOPSA]   - stgca_thought_only: {self.opsd_config.stgca_thought_only}")
            print(f"[EOPSA]   - stgca_teacher_sync_interval: {self.opsd_config.stgca_teacher_sync_interval}")
            print(f"[EOPSA]   - stgca_negative_only: {self.opsd_config.stgca_negative_only}")
            print(f"[EOPSA]   - stgca_lambda_decay_steps: {self.opsd_config.stgca_lambda_decay_steps}")
            print(
                f"[EOPSA]   - stgca_reweight_clip_range: "
                f"{self.opsd_config.stgca_reweight_clip_range}"
            )

    def _build_thought_span_mask(
        self, responses: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build a mask that is 1 only for tokens inside <thought>...</thought> spans."""
        thought_start_ids = self.tokenizer.encode("<thought>", add_special_tokens=False)
        thought_end_ids = self.tokenizer.encode("</thought>", add_special_tokens=False)
        start_len = len(thought_start_ids)
        end_len = len(thought_end_ids)

        batch_size, seq_len = responses.shape
        mask = torch.zeros_like(response_mask)

        for i in range(batch_size):
            tokens = responses[i].tolist()
            in_thought = False
            t = 0
            while t < seq_len:
                if not in_thought:
                    if t + start_len <= seq_len and tokens[t : t + start_len] == thought_start_ids:
                        in_thought = True
                        t += start_len
                        continue
                else:
                    if t + end_len <= seq_len and tokens[t : t + end_len] == thought_end_ids:
                        in_thought = False
                        t += end_len
                        continue
                    mask[i, t] = 1.0
                t += 1

        return mask * response_mask

    def _augment_batch_with_grpo_opsd(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        reward_metrics: Optional[dict[str, list]] = None,
    ) -> DataProto:
        """Attach teacher on-policy tensors aligned with the current GRPO batch."""
        with timer("grpo_opsd_teacher_prompt", timing_raw):
            hints = self._get_self_distill_hints(batch)
            verifier_feedbacks, think_diagnoses, answer_diagnoses = self._extract_verifier_hint_fields(
                reward_metrics, len(batch)
            )
            num_trajectory_hints = sum(1 for hint in hints if hint and str(hint[0]).strip() != str(batch.non_tensor_batch["ground_truth"][0]).strip())
            empty_gt_answers = np.array([""] * len(batch), dtype=object)
            teacher_gen = construct_teacher_prompts(
                questions=self._get_teacher_questions(batch),
                hints=hints,
                gt_answers=empty_gt_answers,
                p_drop=0.0,
                tokenizer=self.tokenizer,
                processor=self.processor,
                verifier_feedbacks=verifier_feedbacks,
                think_diagnoses=think_diagnoses,
                answer_diagnoses=answer_diagnoses,
                template_path=self._resolve_teacher_template_path(),
                max_prompt_length=self.config.data.max_prompt_length,
                multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                image_min_pixels=self.config.data.image_min_pixels,
                image_max_pixels=self.config.data.image_max_pixels,
                video_min_pixels=self.config.data.video_min_pixels,
                video_max_pixels=self.config.data.video_max_pixels,
                video_fps=self.config.data.video_fps,
                video_max_frames=self.config.data.video_max_frames,
                apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                problem_types=self._get_problem_types(batch),
                data_types=self._get_data_types(batch),
            )

            self._log_student_teacher_prompts(
                batch,
                teacher_gen,
                hints=hints,
                verifier_feedbacks=verifier_feedbacks,
                think_diagnoses=think_diagnoses,
                answer_diagnoses=answer_diagnoses,
                reward_metrics=reward_metrics,
            )

        teacher_input_ids, teacher_attention_mask, teacher_position_ids = _concat_prompt_and_response(
            prompt_ids=teacher_gen.batch["input_ids"],
            prompt_attention_mask=teacher_gen.batch["attention_mask"],
            prompt_position_ids=teacher_gen.batch["position_ids"],
            response_ids=batch.batch["responses"],
            response_mask=batch.batch["response_mask"],
        )

        batch.batch["grpo_opsd_teacher_input_ids"] = teacher_input_ids
        batch.batch["grpo_opsd_teacher_attention_mask"] = teacher_attention_mask
        batch.batch["grpo_opsd_teacher_position_ids"] = teacher_position_ids

        if self.opsd_config.stgca_thought_only and not self.opsd_config.enable_grpo_opsd_joint:
            batch.batch["stgca_thought_mask"] = self._build_thought_span_mask(
                batch.batch["responses"], batch.batch["response_mask"]
            )

        if self.opsd_config.enable_grpo_opsd_joint:
            batch.meta_info["loss_type"] = "grpo_opsd_joint"
            batch.meta_info["opsd_config"] = {
                "opsd_loss_coef": self.opsd_config.opsd_loss_coef,
                "opsd_kl_penalty": self.opsd_config.opsd_kl_penalty,
                "stgca_teacher_sync_interval": self.opsd_config.stgca_teacher_sync_interval,
                "global_step": self.global_step,
            }
            metrics["grpo_opsd_joint/opsd_loss_coef"] = float(self.opsd_config.opsd_loss_coef)
        else:
            warmup_steps = self.opsd_config.stgca_lambda_warmup_steps
            decay_steps = self.opsd_config.stgca_lambda_decay_steps
            target_lambda = self.opsd_config.stgca_lambda
            if warmup_steps > 0 and self.global_step < warmup_steps:
                effective_lambda = target_lambda * (self.global_step / warmup_steps)
            elif decay_steps > 0 and self.global_step >= warmup_steps:
                decay_progress = (self.global_step - warmup_steps) / decay_steps
                effective_lambda = target_lambda * max(1.0 - decay_progress, 0.0)
            else:
                effective_lambda = target_lambda

            batch.meta_info["loss_type"] = "grpo_opsd"
            batch.meta_info["opsd_config"] = {
                "stgca_lambda": effective_lambda,
                "stgca_reweight_clip_range": self.opsd_config.stgca_reweight_clip_range,
                "stgca_negative_only": self.opsd_config.stgca_negative_only,
                "stgca_thought_only": self.opsd_config.stgca_thought_only,
                "stgca_teacher_sync_interval": self.opsd_config.stgca_teacher_sync_interval,
                "global_step": self.global_step,
            }
            metrics["grpo_opsd/stgca_effective_lambda"] = effective_lambda
        metrics["grpo_opsd/teacher_context_from_trajectory"] = float(
            sum(
                1
                for idx, hint in enumerate(hints)
                if hint and str(hint[0]).strip() and str(hint[0]).strip() != str(batch.non_tensor_batch["ground_truth"][idx]).strip()
            )
        )
        if self.opsd_config.enable_dynamic_verifier_hint and verifier_feedbacks is not None:
            non_empty = sum(
                1
                for idx in range(len(verifier_feedbacks))
                if _has_dynamic_verifier_hint(
                    verifier_feedbacks[idx],
                    think_diagnoses[idx] if think_diagnoses is not None else "",
                    answer_diagnoses[idx] if answer_diagnoses is not None else "",
                )
            )
            metrics["grpo_opsd/teacher_dynamic_verifier_hint_ratio"] = float(non_empty) / max(len(verifier_feedbacks), 1)
        return batch

    @staticmethod
    def _reorder_reward_metric_lists(reward_metrics: dict[str, list], global_idx) -> dict[str, list]:
        if global_idx is None:
            return reward_metrics
        order = global_idx.detach().cpu().tolist() if torch.is_tensor(global_idx) else list(global_idx)
        return {key: [values[i] for i in order] for key, values in reward_metrics.items()}

    def _attach_split_safety_advantage_inputs(
        self,
        batch: DataProto,
        reward_metrics: dict[str, list],
        metrics: dict[str, Any],
    ) -> None:
        if "think_safe" not in reward_metrics or "ans_safe" not in reward_metrics:
            raise ValueError(
                "enable_split_safety_advantage=True requires reward keys think_safe and ans_safe. "
                "Update examples/safety_rl/reward/safety_reward.py."
            )

        think_mask, answer_mask = build_redacted_think_answer_masks(
            batch.batch["responses"],
            batch.batch["response_mask"],
            self.tokenizer,
        )
        device = batch.batch["responses"].device
        batch.batch["think_span_mask"] = think_mask
        batch.batch["answer_span_mask"] = answer_mask
        batch.batch["think_outcome_scores"] = torch.tensor(
            reward_metrics["think_safe"], dtype=torch.float32, device=device
        )
        batch.batch["answer_outcome_scores"] = torch.tensor(
            reward_metrics["ans_safe"], dtype=torch.float32, device=device
        )

        gap_mask = batch.batch["response_mask"] * (1.0 - think_mask) * (1.0 - answer_mask)
        response_tokens = batch.batch["response_mask"].sum(dim=-1).clamp(min=1.0)
        metrics["adv/think_token_ratio"] = (think_mask.sum(dim=-1) / response_tokens).mean().item()
        metrics["adv/answer_token_ratio"] = (answer_mask.sum(dim=-1) / response_tokens).mean().item()
        metrics["adv/gap_token_ratio"] = (gap_mask.sum(dim=-1) / response_tokens).mean().item()
        metrics["adv/missing_close_tag_ratio"] = (
            ((answer_mask.sum(dim=-1) == 0) & (think_mask.sum(dim=-1) > 0)).float().mean().item()
        )

    def _log_advantage_sample(
        self,
        batch: DataProto,
        reward_metrics: dict[str, list],
        global_step: int,
        stgca_samples: Optional[list[dict]] = None,
    ) -> None:
        """Log one sample's final normalized advantages plus same-group outcome rewards."""
        adv_log_dir = Path(self.config.trainer.save_checkpoint_path) / "prompt_logs"
        adv_log_dir.mkdir(parents=True, exist_ok=True)
        adv_log_path = adv_log_dir / "adv_samples.jsonl"

        sample_idx = 0
        batch_size = batch.batch["responses"].shape[0]
        target_uid = batch.non_tensor_batch["uid"][sample_idx] if "uid" in batch.non_tensor_batch else None
        group_indices = [
            i for i in range(batch_size)
            if target_uid is None or batch.non_tensor_batch["uid"][i] == target_uid
        ]
        if not group_indices:
            return

        response_ids = batch.batch["responses"][sample_idx]
        response_mask = batch.batch["response_mask"][sample_idx]
        advantages = batch.batch["advantages"][sample_idx]
        think_mask = batch.batch["think_span_mask"][sample_idx]
        answer_mask = batch.batch["answer_span_mask"][sample_idx]
        think_score = float(batch.batch["think_outcome_scores"][sample_idx].item())
        answer_score = float(batch.batch["answer_outcome_scores"][sample_idx].item())
        valid_len = int(response_mask.sum().item())
        if valid_len <= 0:
            return

        sample_response_hash = self._compute_response_hash(response_ids, response_mask)

        stgca_by_hash: dict[str, dict[str, Any]] = {}
        if stgca_samples is not None:
            for sample in stgca_samples:
                response_hash = sample.get("response_hash")
                if response_hash:
                    stgca_by_hash[response_hash] = sample

        group_think_scores: list[float] = []
        group_answer_scores: list[float] = []
        for idx in group_indices:
            think_score = float(batch.batch["think_outcome_scores"][idx].item())
            answer_score = float(batch.batch["answer_outcome_scores"][idx].item())
            group_think_scores.append(think_score)
            group_answer_scores.append(answer_score)

        response_text = self.tokenizer.decode(response_ids[:valid_len], skip_special_tokens=False)
        token_advantages: list[dict[str, Any]] = []
        for t in range(valid_len):
            token_id = int(response_ids[t].item())
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            post_norm_adv = float(advantages[t].item())

            if think_mask[t] > 0:
                pre_norm_adv = think_score
                segment = "think"
            elif answer_mask[t] > 0:
                pre_norm_adv = answer_score
                segment = "answer"
            else:
                pre_norm_adv = (think_score + answer_score) * 0.5
                segment = "gap"

            token_advantages.append(
                {
                    "t": t,
                    "token": token_str,
                    "token_id": token_id,
                    "segment": segment,
                    "pre_norm_adv": pre_norm_adv,
                    "post_norm_adv": post_norm_adv,
                }
            )

        reward_list = [
            float(reward_metrics.get("think_safe", [0.0])[sample_idx])
            if reward_metrics.get("think_safe") and sample_idx < len(reward_metrics["think_safe"])
            else 0.0,
            float(reward_metrics.get("ans_safe", [0.0])[sample_idx])
            if reward_metrics.get("ans_safe") and sample_idx < len(reward_metrics["ans_safe"])
            else 0.0,
        ]

        record = {
            "step": global_step,
            "group_uid": str(target_uid) if target_uid is not None else None,
            "group_size": len(group_indices),
            "prompt_text": self._get_student_prompt_text(batch, sample_idx),
            "group_think_scores": group_think_scores,
            "group_answer_scores": group_answer_scores,
            "response_hash": sample_response_hash,
            "response_text": response_text,
            "token_advantages": token_advantages,
            "reward_list": reward_list,
            "think_score": think_score,
            "answer_score": answer_score,
        }

        stgca_sample = stgca_by_hash.get(sample_response_hash)
        if stgca_sample is not None:
            record["stgca"] = {
                "lambda": stgca_sample["lambda"],
                "clip_range": stgca_sample["clip_range"],
                "negative_only": stgca_sample["negative_only"],
                "thought_only": stgca_sample["thought_only"],
                "token_delta": stgca_sample["token_delta"],
                "token_reweight": stgca_sample["token_reweight"],
                "token_adv": stgca_sample["token_stgca_adv"],
            }

        with open(adv_log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fit(self):
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        self._load_training_token_stats()
        main_tqdm.update(self.global_step)

        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_before_train
            and self.config.trainer.val_freq != 0
        ):
            val_metrics = self._validate()
            self._maybe_run_adaptive_rollout_probe(val_metrics)
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            rollout_len = self._get_student_max_response_length()
            metrics["train/rollout_length"] = float(rollout_len)
            metrics["data/student_max_response_length"] = float(rollout_len)
            metrics["data/harmful_max_response_length"] = float(rollout_len)
            metrics["data/benign_max_response_length"] = float(rollout_len)
            with timer("step", timing_raw):
                reward_futures = [] if self.config.algorithm.pipeline_reward else None
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics, reward_futures=reward_futures)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                global_idx = self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if "token_level_scores" not in batch.batch and not (reward_futures and len(reward_futures) > 0):
                    with timer("reward", timing_raw):
                        reward_batch = batch.select(
                            batch_keys=["responses", "response_mask"],
                            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                        )
                        reward_ref = self.reward_fn.compute_reward.remote(reward_batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                with timer("adv", timing_raw):
                    latest_reward_metrics: dict[str, list] = {}
                    if reward_futures and len(reward_futures) > 0:
                        with timer("reward", timing_raw):
                            all_reward_tensors = []
                            all_reward_metrics = defaultdict(list)
                            for ref, _mini_size in reward_futures:
                                reward_tensor_i, reward_metrics_i = ray.get(ref)
                                all_reward_tensors.append(reward_tensor_i)
                                for k, v in reward_metrics_i.items():
                                    all_reward_metrics[k].extend(v)
                            reward_tensor_all = torch.cat(all_reward_tensors, dim=0)
                            target_size = self.config.data.rollout_batch_size * self.config.worker.rollout.n
                            reward_tensor_all = reward_tensor_all[:target_size]
                            reward_tensor_all = reward_tensor_all[global_idx]
                            batch.batch["token_level_scores"] = reward_tensor_all
                            latest_reward_metrics = self._reorder_reward_metric_lists(all_reward_metrics, global_idx)
                            metrics.update(
                                {
                                    f"reward/{k}": v
                                    for k, v in reduce_metrics(
                                        self._filter_numeric_reward_metrics(latest_reward_metrics)
                                    ).items()
                                }
                            )
                    elif "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        latest_reward_metrics = reward_metrics
                        metrics.update(
                            {
                                f"reward/{k}": v
                                for k, v in reduce_metrics(
                                    self._filter_numeric_reward_metrics(reward_metrics)
                                ).items()
                            }
                        )

                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    if self.config.algorithm.enable_split_safety_advantage:
                        self._attach_split_safety_advantage_inputs(batch, latest_reward_metrics, metrics)

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        enable_split_safety_advantage=self.config.algorithm.enable_split_safety_advantage,
                        split_safety_adv_eps=self.config.algorithm.split_safety_adv_eps,
                        split_safety_adv_unified_scale=self.config.algorithm.split_safety_adv_unified_scale,
                    )

                    # Merge split-safety advantage monitoring metrics into the step metrics
                    if "adv_metrics" in batch.meta_info:
                        metrics.update(batch.meta_info["adv_metrics"])

                batch = self._augment_batch_with_grpo_opsd(
                    batch, metrics, timing_raw, reward_metrics=latest_reward_metrics
                )

                if "uid" in batch.non_tensor_batch and len(batch.non_tensor_batch["uid"]) > 0:
                    batch.meta_info["adv_log_target_uid"] = str(batch.non_tensor_batch["uid"][0])
                    batch.meta_info["adv_log_target_response_hash"] = self._compute_response_hash(
                        batch.batch["responses"][0], batch.batch["response_mask"][0]
                    )

                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                # Extract RLSD STGCA sample data before reduce_metrics consumes it.
                # Always strip these string keys — actor may emit them even when
                # enable_split_safety_advantage=False (adv_log_target_uid is set
                # unconditionally), and np.mean on Unicode JSON strings crashes.
                stgca_samples = None
                if "__stgca_group_samples__" in actor_output.non_tensor_batch:
                    raw_list = actor_output.non_tensor_batch["__stgca_group_samples__"]
                    if (
                        self.config.algorithm.enable_split_safety_advantage
                        and raw_list is not None
                        and len(raw_list) > 0
                    ):
                        stgca_samples = [json.loads(item) for item in raw_list]
                    del actor_output.non_tensor_batch["__stgca_group_samples__"]
                if "__stgca_sample__" in actor_output.non_tensor_batch:
                    raw_list = actor_output.non_tensor_batch["__stgca_sample__"]
                    if self.config.algorithm.enable_split_safety_advantage and raw_list:
                        stgca_samples = [json.loads(raw_list[0])]
                    del actor_output.non_tensor_batch["__stgca_sample__"]

                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                # Log one sample's per-token advantage values for inspection (now includes STGCA)
                if self.config.algorithm.enable_split_safety_advantage:
                    self._log_advantage_sample(batch, latest_reward_metrics, self.global_step, stgca_samples=stgca_samples)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                    with timer("ars", timing_raw):
                        self._maybe_run_adaptive_rollout_probe(metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self._update_training_token_stats(metrics, batch=batch)
            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None and self.config.trainer.val_freq != 0:
            if (
                val_metrics is None
                or self.config.trainer.val_freq < 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self._maybe_run_adaptive_rollout_probe(val_metrics)
                self.logger.log(data=val_metrics, step=self.global_step)

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
