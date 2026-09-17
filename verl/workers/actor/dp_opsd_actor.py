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
OPSD Actor: Extends DataParallelPPOActor with joint Navigator GRPO + Reasoner OPSD training.

Key additions:
  - update_policy_opsd(): Joint training loop for Navigator GRPO + OPSD distillation
  - Teacher forward passes in torch.no_grad() — gradients only flow through Student
"""

import hashlib
import json
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...trainer.opsd_algos import (
    DEFAULT_TEACHER_TOPK,
    TEACHER_TOPK_RENORM_LOSS_TYPES,
    accumulate_topk_overlap_stats,
    compute_opsd_jsd_loss,
    compute_opsd_kl_loss,
    compute_opsd_topk_jsd_loss,
    compute_opsd_topk_kl_loss,
    compute_teacher_topk_renorm_distill_loss,
    select_teacher_topk_renormalized_log_probs,
)
from ...trainer.token_filter_runtime import apply_token_filter
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .dp_actor import DataParallelPPOActor


def _compute_response_hash(response_ids: torch.Tensor, response_mask: torch.Tensor) -> str:
    valid_len = int(response_mask.sum().item())
    if valid_len <= 0:
        return ""
    token_ids = response_ids[:valid_len].tolist()
    payload = ",".join(str(int(tok)) for tok in token_ids)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelOPSDActor"]


class DataParallelOPSDActor(DataParallelPPOActor):
    """OPSD Actor with joint Navigator GRPO + Reasoner self-distillation training."""

    @staticmethod
    def _uses_logits_distillation(distillation_loss_type: str, distillation_topk: Optional[int]) -> bool:
        return (
            distillation_topk is not None
            or distillation_loss_type == "jsd"
            or distillation_loss_type in TEACHER_TOPK_RENORM_LOSS_TYPES
        )

    def __init__(
        self,
        config,
        actor_module,
        actor_optimizer=None,
        teacher_module: Optional[torch.nn.Module] = None,
        oracle_module: Optional[torch.nn.Module] = None,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.teacher_module = teacher_module
        self.oracle_module = oracle_module
        # EMA of harmful keep tokens / harmful row — used when a microbatch is
        # all-benign so TOKEN_FILTER_BENIGN_RATIO still has a harmful reference.
        self._tf_harmful_keep_per_row_ema: Optional[float] = None

    def _get_teacher_module(self):
        return self.teacher_module if self.teacher_module is not None else self.actor_module

    def _get_oracle_module(self, opsd_config: dict[str, Any]):
        if not opsd_config.get("enable_oracle_correction", False):
            return None
        return self.oracle_module

    @staticmethod
    def _oracle_mix_kwargs(opsd_config: dict[str, Any], oracle_logits: Optional[torch.Tensor]) -> dict[str, Any]:
        if oracle_logits is None:
            return {
                "oracle_logits": None,
                "oracle_mix": None,
            }
        return {
            "oracle_logits": oracle_logits,
            "oracle_mix": opsd_config.get("oracle_mix", "geometric"),
            "oracle_alpha": float(opsd_config.get("oracle_alpha", 0.2)),
            "oracle_lambda": float(opsd_config.get("oracle_lambda", 0.5)),
            "oracle_delta_clip": float(opsd_config.get("oracle_delta_clip", 2.0)),
            "oracle_residual_relu": bool(opsd_config.get("oracle_residual_relu", False)),
        }

    def _sync_teacher_from_actor(self):
        """Copy actor weights to teacher module for periodic teacher refresh."""
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            return
        with torch.no_grad():
            for t_param, a_param in zip(self.teacher_module.parameters(), self.actor_module.parameters()):
                t_param.data.copy_(a_param.data)

    def _maybe_sync_teacher(self, opsd_config: dict[str, Any], metrics: dict, metric_prefix: str) -> None:
        """Periodically copy actor weights to frozen teacher when sync interval > 0."""
        sync_interval = opsd_config.get("stgca_teacher_sync_interval", 0)
        if sync_interval <= 0:
            return
        global_step = opsd_config.get("global_step", 0)
        synced = global_step > 0 and global_step % sync_interval == 0
        if synced:
            self._sync_teacher_from_actor()
        append_to_dict(metrics, {f"{metric_prefix}/teacher_synced": float(synced)})

    @staticmethod
    def _masked_entropy_from_logits(
        logits: torch.Tensor,
        response_mask: torch.Tensor,
        loss_avg_mode: str,
    ) -> torch.Tensor:
        """Compute masked token entropy from full logits."""
        token_entropy = DataParallelOPSDActor._token_entropy_from_logits(logits)
        return average_loss(token_entropy, response_mask, mode=loss_avg_mode)

    @staticmethod
    def _token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    def _token_jsd_from_logits(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = student_log_probs.exp()
        mix_probs = 0.5 * (teacher_probs + student_probs)
        mix_log_probs = mix_probs.clamp_min(eps).log()
        return 0.5 * (
            (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
            + (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
        )

    @staticmethod
    def _select_token_mask_from_scores(
        scores: torch.Tensor,
        response_mask: torch.Tensor,
        keep_ratio: float,
        *,
        largest: bool,
    ) -> torch.Tensor:
        if keep_ratio >= 1.0:
            return response_mask

        selected_mask = torch.zeros_like(response_mask)
        batch_size = response_mask.shape[0]
        for i in range(batch_size):
            valid_idx = torch.nonzero(response_mask[i] > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            keep_count = max(1, int(valid_idx.numel() * keep_ratio + 0.999999))
            valid_scores = scores[i, valid_idx]
            topk_local = torch.topk(valid_scores, k=keep_count, largest=largest).indices
            chosen_idx = valid_idx[topk_local]
            selected_mask[i, chosen_idx] = 1
        return selected_mask

    @staticmethod
    def _select_token_mask_top_n(
        scores: torch.Tensor,
        response_mask: torch.Tensor,
        top_n: int,
        *,
        largest: bool = True,
    ) -> torch.Tensor:
        """Keep a fixed top-N tokens per sequence by score (within response_mask)."""
        if top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {top_n}")
        selected_mask = torch.zeros_like(response_mask)
        batch_size = response_mask.shape[0]
        for i in range(batch_size):
            valid_idx = torch.nonzero(response_mask[i] > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            keep_count = min(int(top_n), int(valid_idx.numel()))
            valid_scores = scores[i, valid_idx]
            topk_local = torch.topk(valid_scores, k=keep_count, largest=largest).indices
            selected_mask[i, valid_idx[topk_local]] = 1
        return selected_mask

    def _build_distill_token_mask(
        self,
        *,
        teacher_logits: Optional[torch.Tensor],
        student_logits: Optional[torch.Tensor],
        base_mask: torch.Tensor,
        selection_mode: str,
        keep_ratio: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if selection_mode == "all" or keep_ratio >= 1.0:
            return base_mask, {
                "selected_token_ratio": 1.0,
                "selected_token_count": float(base_mask.sum().item()),
            }

        if teacher_logits is None or student_logits is None:
            raise ValueError(
                "Token-selection distillation requires teacher/student logits. "
                "Use distillation_loss_type=jsd, topk_jsd, topk_forward_kl, topk_reverse_kl, "
                "or distillation_topk with logits-based distillation."
            )

        if selection_mode == "low_teacher_entropy":
            scores = self._token_entropy_from_logits(teacher_logits.detach())
            selected_mask = self._select_token_mask_from_scores(
                scores=scores,
                response_mask=base_mask,
                keep_ratio=keep_ratio,
                largest=False,
            )
        elif selection_mode == "high_teacher_student_gap":
            scores = self._token_jsd_from_logits(teacher_logits, student_logits).detach()
            selected_mask = self._select_token_mask_from_scores(
                scores=scores,
                response_mask=base_mask,
                keep_ratio=keep_ratio,
                largest=True,
            )
        else:
            raise ValueError(f"Unsupported distill token selection mode: {selection_mode}")

        selected_count = selected_mask.sum().item()
        total_count = base_mask.sum().item()
        selected_ratio = 0.0 if total_count <= 0 else float(selected_count / total_count)
        return selected_mask, {
            "selected_token_ratio": selected_ratio,
            "selected_token_count": float(selected_count),
        }

    @staticmethod
    def _per_token_training_signal(
        *,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        distillation_loss_type: str,
        distillation_topk: Optional[int],
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Per-token divergence used to measure KL/JSD training-signal mass (no grad)."""
        with torch.no_grad():
            use_topk = (
                distillation_loss_type in TEACHER_TOPK_RENORM_LOSS_TYPES
                or distillation_topk is not None
            )
            if use_topk:
                k = min(distillation_topk or DEFAULT_TEACHER_TOPK, teacher_logits.size(-1))
                teacher_topk_log_probs, student_topk_log_probs, _, _ = (
                    select_teacher_topk_renormalized_log_probs(
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        k=k,
                    )
                )
                if distillation_loss_type == "topk_reverse_kl":
                    student_probs = student_topk_log_probs.exp()
                    return (student_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)
                if distillation_loss_type in {"topk_jsd", "jsd"}:
                    teacher_probs = teacher_topk_log_probs.exp()
                    student_probs = student_topk_log_probs.exp()
                    mix_probs = 0.5 * (teacher_probs + student_probs)
                    mix_log_probs = mix_probs.clamp_min(eps).log()
                    return 0.5 * (
                        (teacher_probs * (teacher_topk_log_probs - mix_log_probs)).sum(dim=-1)
                        + (student_probs * (student_topk_log_probs - mix_log_probs)).sum(dim=-1)
                    )
                # Default / topk_forward_kl: teacher→student forward KL on top-k support.
                teacher_probs = teacher_topk_log_probs.exp()
                return (teacher_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)

            return DataParallelOPSDActor._token_jsd_from_logits(
                teacher_logits, student_logits, eps=eps
            )

    def _apply_same_token_filter(
        self,
        *,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        base_mask: torch.Tensor,
        distillation_loss_type: str,
        distillation_topk: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Legacy helper kept for compatibility; prefer ``_run_configured_token_filter``."""
        with torch.no_grad():
            teacher_top1 = teacher_logits.argmax(dim=-1)
            student_top1 = student_logits.argmax(dim=-1)
            same_top1 = (teacher_top1 == student_top1).to(dtype=base_mask.dtype)
            disagree_mask = base_mask * (1.0 - same_top1)
            same_mask = base_mask * same_top1

            token_div = self._per_token_training_signal(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                distillation_loss_type=distillation_loss_type,
                distillation_topk=distillation_topk,
            )
            tokens_before = float(base_mask.sum().item())
            tokens_after = float(disagree_mask.sum().item())
            total_kl = float((token_div * base_mask).sum().item())
            filtered_kl = float((token_div * same_mask).sum().item())
            kept_kl = float((token_div * disagree_mask).sum().item())

        return disagree_mask, same_mask, {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_filtered": tokens_before - tokens_after,
            "total_kl": total_kl,
            "filtered_kl": filtered_kl,
            "kept_kl": kept_kl,
        }

    def _make_token_decode_fn(self):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return None
        cache: dict[int, str] = {}

        def _decode(token_id: int) -> str:
            if token_id not in cache:
                cache[token_id] = tokenizer.decode([token_id], skip_special_tokens=True)
            return cache[token_id]

        return _decode

    def _run_configured_token_filter(
        self,
        *,
        mode: str,
        action: str,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        base_mask: torch.Tensor,
        responses: torch.Tensor,
        distillation_loss_type: str,
        distillation_topk: Optional[int],
        top_n: int,
        module_path: Optional[str],
        extra: Optional[dict[str, Any]] = None,
        score_loss_type: Optional[str] = None,
    ):
        """Dispatch to user-editable ``examples/safety_rl/token_filter/filters.py``.

        ``score_loss_type`` optionally overrides how ``token_div`` is scored
        (e.g. force ``topk_forward_kl`` for benign ``forward_kl_topn``).
        """
        score_type = (score_loss_type or distillation_loss_type or "").strip().lower()
        mode_key = (mode or "").strip().lower()
        # forward_kl_topn always ranks by teacher→student forward KL.
        if mode_key in {"forward_kl_topn", "fwd_kl_topn"}:
            score_type = "topk_forward_kl"
        with torch.no_grad():
            token_div = self._per_token_training_signal(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                distillation_loss_type=score_type,
                distillation_topk=distillation_topk,
            )
        return apply_token_filter(
            mode,
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            base_mask=base_mask,
            responses=responses,
            token_div=token_div,
            top_n=top_n,
            action=action,
            decode_id=self._make_token_decode_fn(),
            extra=extra or {},
            module_path=module_path,
        ), token_div

    # Taxonomy labels logged every step (0 if absent). Keep in sync with
    # ``TAXONOMY_METRIC_CATEGORIES`` in examples/safety_rl/token_filter/filters.py.
    _TOKEN_FILTER_CAT_METRIC_NAMES = (
        "pivot",
        "intent",
        "risk_lexicon",
        "risk_wo_same",
        "same",
        "function",
        "other",
        "think_filler",
        "comply_plan",
    )

    @staticmethod
    def _is_benign_data_type(data_type: Any) -> bool:
        if data_type is None:
            return False
        return str(data_type).strip().lower() in {
            "overreject",
            "benign",
            "harmless",
            "harmless_queries",
        }

    def _init_token_filter_cat_accums(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Zero tensors for per-taxonomy-category token counts (base + main)."""
        out: dict[str, torch.Tensor] = {}
        for name in self._TOKEN_FILTER_CAT_METRIC_NAMES:
            out[f"tokens_cat_{name}"] = torch.zeros((), device=device, dtype=torch.float32)
            out[f"tokens_main_cat_{name}"] = torch.zeros((), device=device, dtype=torch.float32)
        return out

    def _accumulate_token_filter_cat_stats(
        self,
        accums: Optional[dict[str, torch.Tensor]],
        stats: dict[str, float],
    ) -> None:
        if accums is None:
            return
        for key, tensor in accums.items():
            tensor += float(stats.get(key, 0.0))

    def _token_filter_main_by_data_split(
        self,
        keep_mask: torch.Tensor,
        data_types,
    ) -> tuple[float, float]:
        """Split kept (main-loss) tokens into harmful vs benign.

        Non-benign rows count as harmful. If ``data_types`` is missing, all kept
        tokens are attributed to harmful so harmful+benign still equals total.
        """
        tokens_main = float(keep_mask.sum().item())
        if data_types is None:
            return tokens_main, 0.0
        is_benign = torch.tensor(
            [self._is_benign_data_type(dt) for dt in data_types],
            device=keep_mask.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
        benign_row = is_benign.expand_as(keep_mask)
        benign_main = float((keep_mask * benign_row.to(dtype=keep_mask.dtype)).sum().item())
        harmful_main = float(tokens_main - benign_main)
        return harmful_main, benign_main

    def _rebuild_token_filter_stats(
        self,
        *,
        base_mask: torch.Tensor,
        keep_mask: torch.Tensor,
        action_mask: torch.Tensor,
        token_div: torch.Tensor,
        action: str,
    ) -> dict[str, float]:
        tokens_before = float(base_mask.sum().item())
        tokens_main = float(keep_mask.sum().item())
        if action == "sampled_kl":
            tokens_alt = float(action_mask.sum().item())
            tokens_dropped = float(
                (base_mask * (1.0 - keep_mask - action_mask)).sum().item()
            )
        else:
            tokens_alt = 0.0
            tokens_dropped = float((base_mask * (1.0 - keep_mask)).sum().item())
        tokens_trained = tokens_main + tokens_alt
        total_kl = float((token_div * base_mask).sum().item())
        kept_kl = float((token_div * keep_mask).sum().item())
        filtered_kl = float((token_div * (base_mask * (1.0 - keep_mask))).sum().item())
        return {
            "tokens_before": tokens_before,
            "tokens_after": tokens_main,
            "tokens_filtered": tokens_before - tokens_main,
            "total_kl": total_kl,
            "filtered_kl": filtered_kl,
            "kept_kl": kept_kl,
            "tokens_main": tokens_main,
            "tokens_alt": tokens_alt,
            "tokens_dropped": tokens_dropped,
            "tokens_trained": tokens_trained,
            "token_train_ratio": (tokens_trained / tokens_before) if tokens_before > 0 else 0.0,
        }

    def _run_token_filter_with_optional_benign(
        self,
        *,
        mode: str,
        action: str,
        top_n: int,
        benign_mode: Optional[str],
        benign_action: Optional[str],
        benign_top_n: Optional[int],
        benign_ratio: Optional[float] = None,
        data_types,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        base_mask: torch.Tensor,
        responses: torch.Tensor,
        distillation_loss_type: str,
        distillation_topk: Optional[int],
        module_path: Optional[str],
        extra: Optional[dict[str, Any]] = None,
    ):
        """Apply default filter; optionally override benign rows with another mode.

        ``benign_ratio`` is mode-agnostic: after benign keep candidates are chosen
        (taxonomy_keep / forward_kl_topn / …, or the shared default mode), keep
        only the first ``round(ratio * n_harmful_keep)`` of those candidates in
        their existing order (batch×time, no re-ranking).

        Pure-benign microbatches have no in-batch harmful reference; then we use
        an EMA of harmful-keep-per-row from recent mixed/harmful microbatches.
        """
        benign_mode_key = (benign_mode or "").strip().lower()
        is_benign_list = None
        # Benign top-n selection is always ranked by forward KL (teacher→student).
        benign_score_type = None
        if benign_mode_key in {"jsd_topn", "forward_kl_topn", "fwd_kl_topn"}:
            benign_mode_key = "forward_kl_topn"
            benign_score_type = "topk_forward_kl"
        if data_types is not None:
            is_benign_list = [self._is_benign_data_type(dt) for dt in data_types]

        ratio = None if benign_ratio is None else float(benign_ratio)
        has_benign = is_benign_list is not None and any(is_benign_list)
        all_benign = is_benign_list is not None and any(is_benign_list) and all(is_benign_list)
        want_ratio = ratio is not None and ratio >= 0.0 and has_benign

        if benign_mode_key and all_benign:
            filt_b, token_div = self._run_configured_token_filter(
                mode=benign_mode_key,
                action=(benign_action or action),
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                base_mask=base_mask,
                responses=responses,
                distillation_loss_type=distillation_loss_type,
                distillation_topk=distillation_topk,
                top_n=int(benign_top_n if benign_top_n is not None else top_n),
                module_path=module_path,
                extra=extra,
                score_loss_type=benign_score_type,
            )
            if not want_ratio:
                return filt_b
            benign_mask = torch.ones(
                (base_mask.shape[0], 1), device=base_mask.device, dtype=torch.bool
            )
            keep_mask, action_mask, ratio_stats = self._apply_benign_ratio_cap(
                keep_mask=filt_b.keep_mask,
                action_mask=filt_b.action_mask,
                benign_mask=benign_mask,
                ratio=ratio,
            )
            stats = self._rebuild_token_filter_stats(
                base_mask=base_mask,
                keep_mask=keep_mask,
                action_mask=action_mask,
                token_div=token_div,
                action=filt_b.action,
            )
            stats.update(ratio_stats)
            for k, v in filt_b.stats.items():
                if k.startswith("tokens_cat_"):
                    stats[k] = float(v)
            ResultCls = type(filt_b)
            return ResultCls(
                keep_mask=keep_mask,
                action_mask=action_mask,
                stats=stats,
                mode=filt_b.mode,
                action=filt_b.action,
                use_alt_kl_mix=filt_b.use_alt_kl_mix,
            )

        filt, token_div = self._run_configured_token_filter(
            mode=mode,
            action=action,
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            base_mask=base_mask,
            responses=responses,
            distillation_loss_type=distillation_loss_type,
            distillation_topk=distillation_topk,
            top_n=top_n,
            module_path=module_path,
            extra=extra,
        )

        keep_mask = filt.keep_mask
        action_mask = filt.action_mask
        merged_action = "sampled_kl" if filt.use_alt_kl_mix else "drop"
        use_alt = bool(filt.use_alt_kl_mix)
        mode_tag = filt.mode
        benign_mask = None

        if benign_mode_key and has_benign and not all_benign:
            filt_b, _ = self._run_configured_token_filter(
                mode=benign_mode_key,
                action=(benign_action or action),
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                base_mask=base_mask,
                responses=responses,
                distillation_loss_type=distillation_loss_type,
                distillation_topk=distillation_topk,
                top_n=int(benign_top_n if benign_top_n is not None else top_n),
                module_path=module_path,
                extra=extra,
                score_loss_type=benign_score_type,
            )
            benign_mask = torch.tensor(
                is_benign_list, device=base_mask.device, dtype=torch.bool
            ).unsqueeze(-1)
            keep_mask = torch.where(benign_mask, filt_b.keep_mask, filt.keep_mask)
            action_mask = torch.where(benign_mask, filt_b.action_mask, filt.action_mask)
            use_alt = bool(filt.use_alt_kl_mix or filt_b.use_alt_kl_mix)
            merged_action = "sampled_kl" if use_alt else "drop"
            mode_tag = f"{filt.mode}+{filt_b.mode}"
        elif want_ratio:
            # Shared mode for all rows; still cap benign rows by ratio.
            if all_benign:
                benign_mask = torch.ones(
                    (base_mask.shape[0], 1), device=base_mask.device, dtype=torch.bool
                )
            else:
                benign_mask = torch.tensor(
                    is_benign_list, device=base_mask.device, dtype=torch.bool
                ).unsqueeze(-1)
        elif not benign_mode_key:
            return filt

        ratio_stats: dict[str, float] = {}
        if want_ratio and benign_mask is not None:
            keep_mask, action_mask, ratio_stats = self._apply_benign_ratio_cap(
                keep_mask=keep_mask,
                action_mask=action_mask,
                benign_mask=benign_mask,
                ratio=ratio,
            )

        stats = self._rebuild_token_filter_stats(
            base_mask=base_mask,
            keep_mask=keep_mask,
            action_mask=action_mask,
            token_div=token_div,
            action=merged_action,
        )
        stats.update(ratio_stats)
        # Keep taxonomy base-category counts from the primary filter pass.
        for k, v in filt.stats.items():
            if k.startswith("tokens_cat_"):
                stats[k] = float(v)
        ResultCls = type(filt)
        return ResultCls(
            keep_mask=keep_mask,
            action_mask=action_mask,
            stats=stats,
            mode=mode_tag,
            action=merged_action,
            use_alt_kl_mix=use_alt,
        )

    def _update_harmful_keep_per_row_ema(self, keep_per_row: float) -> None:
        if keep_per_row < 0:
            return
        if self._tf_harmful_keep_per_row_ema is None:
            self._tf_harmful_keep_per_row_ema = float(keep_per_row)
        else:
            self._tf_harmful_keep_per_row_ema = (
                0.9 * float(self._tf_harmful_keep_per_row_ema) + 0.1 * float(keep_per_row)
            )

    def _apply_benign_ratio_cap(
        self,
        *,
        keep_mask: torch.Tensor,
        action_mask: torch.Tensor,
        benign_mask: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Cap benign keep to ``round(ratio * harmful_ref)`` in existing order.

        ``harmful_ref`` is in-batch harmful keep when available; otherwise
        ``EMA(harmful_keep_per_row) * n_benign_rows``.
        """
        dtype = keep_mask.dtype
        benign = benign_mask.to(dtype=torch.bool).expand_as(keep_mask)
        harmful = ~benign
        n_harmful_rows = int(harmful[:, 0].sum().item()) if harmful.ndim >= 2 else int((~benign_mask.reshape(-1)).sum().item())
        n_benign_rows = int(benign[:, 0].sum().item()) if benign.ndim >= 2 else int(benign_mask.reshape(-1).sum().item())
        n_harmful_keep = int((keep_mask * harmful.to(dtype=dtype)).sum().item())

        if n_harmful_rows > 0:
            self._update_harmful_keep_per_row_ema(n_harmful_keep / float(n_harmful_rows))
            harmful_ref = float(n_harmful_keep)
            ref_source = "batch"
        else:
            ema = self._tf_harmful_keep_per_row_ema
            if ema is None or n_benign_rows <= 0:
                # No harmful history yet — leave benign keep unchanged.
                return keep_mask, action_mask, {
                    "benign_ratio": float(ratio),
                    "benign_ratio_harmful_ref": 0.0,
                    "benign_ratio_target": -1.0,
                    "benign_ratio_kept": float(
                        (keep_mask * benign.to(dtype=dtype)).sum().item()
                    ),
                    "benign_ratio_ref_source": 0.0,  # none
                }
            harmful_ref = float(ema) * float(n_benign_rows)
            ref_source = "ema"

        keep_out, action_out = self._cap_benign_keep_by_harmful_ratio(
            keep_mask=keep_mask,
            action_mask=action_mask,
            benign_row_mask=benign_mask,
            ratio=ratio,
            harmful_ref=harmful_ref,
        )
        benign_kept = float((keep_out * benign.to(dtype=dtype)).sum().item())
        return keep_out, action_out, {
            "benign_ratio": float(ratio),
            "benign_ratio_harmful_ref": float(harmful_ref),
            "benign_ratio_target": float(round(ratio * harmful_ref)),
            "benign_ratio_kept": benign_kept,
            "benign_ratio_ref_source": 1.0 if ref_source == "batch" else 2.0,
        }

    @staticmethod
    def _cap_benign_keep_by_harmful_ratio(
        *,
        keep_mask: torch.Tensor,
        action_mask: torch.Tensor,
        benign_row_mask: torch.Tensor,
        ratio: float,
        harmful_ref: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Among benign keep candidates, keep the first ``round(ratio * n_h)`` in order.

        ``n_h`` defaults to in-batch harmful keep count; pass ``harmful_ref`` to
        override (e.g. EMA-based reference for all-benign microbatches). Selection
        follows existing batch×time order (no score re-ranking). Harmful rows are
        unchanged.
        """
        dtype = keep_mask.dtype
        benign = benign_row_mask.to(dtype=torch.bool).expand_as(keep_mask)
        harmful = ~benign
        if harmful_ref is None:
            n_harmful = int((keep_mask * harmful.to(dtype=dtype)).sum().item())
        else:
            n_harmful = int(round(float(harmful_ref)))
        if n_harmful <= 0:
            return keep_mask, action_mask

        target = int(round(float(ratio) * float(n_harmful)))
        benign_cand = keep_mask * benign.to(dtype=dtype)
        n_cand = int(benign_cand.sum().item())
        if n_cand <= 0:
            return keep_mask, action_mask

        keep_out = keep_mask.clone()
        action_out = action_mask.clone()
        # Drop all benign keep first, then restore the first-target in order.
        keep_out = torch.where(benign, torch.zeros_like(keep_out), keep_out)
        # Trimmed candidates leave main distill (do not move into action_mask).
        action_out = torch.where(benign, action_mask * (1.0 - benign_cand), action_out)

        if target <= 0:
            return keep_out, action_out

        # nonzero is row-major → preserves current sequence order across the batch.
        cand_idx = torch.nonzero(benign_cand.reshape(-1) > 0, as_tuple=False).squeeze(-1)
        k = min(target, int(cand_idx.numel()))
        if k <= 0:
            return keep_out, action_out
        selected = cand_idx[:k]
        flat_keep = keep_out.reshape(-1)
        flat_keep[selected] = 1
        return keep_out, action_out

    @staticmethod
    def _resolve_same_token_filter_masks(
        *,
        mode: str,
        disagree_mask: torch.Tensor,
        same_mask: torch.Tensor,
        stf_stats: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Pick train mask + stats for the configured same_token_filter_mode.

        Stats convention: tokens_after / kept_* = positions that receive the main
        distill loss; tokens_filtered / filtered_* = positions excluded from it.
        """
        if mode == "same_only":
            return same_mask, {
                "tokens_before": stf_stats["tokens_before"],
                "tokens_after": stf_stats["tokens_filtered"],
                "tokens_filtered": stf_stats["tokens_after"],
                "total_kl": stf_stats["total_kl"],
                "filtered_kl": stf_stats["kept_kl"],
                "kept_kl": stf_stats["filtered_kl"],
            }
        if mode in {"drop", "sampled_kl"}:
            return disagree_mask, stf_stats
        raise ValueError(
            f"Unsupported same_token_filter_mode={mode!r}. "
            "Expected one of: drop, same_only, sampled_kl, jsd_topn."
        )

    def _apply_jsd_topn_filter(
        self,
        *,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        base_mask: torch.Tensor,
        top_n: int,
        distillation_loss_type: str,
        distillation_topk: Optional[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Keep only top-N tokens per sequence by per-token training JSD/KL."""
        with torch.no_grad():
            token_div = self._per_token_training_signal(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                distillation_loss_type=distillation_loss_type,
                distillation_topk=distillation_topk,
            )
            keep_mask = self._select_token_mask_top_n(
                scores=token_div,
                response_mask=base_mask,
                top_n=top_n,
                largest=True,
            )
            drop_mask = base_mask * (1.0 - keep_mask)
            tokens_before = float(base_mask.sum().item())
            tokens_after = float(keep_mask.sum().item())
            total_kl = float((token_div * base_mask).sum().item())
            kept_kl = float((token_div * keep_mask).sum().item())
            filtered_kl = float((token_div * drop_mask).sum().item())

        return keep_mask, {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_filtered": tokens_before - tokens_after,
            "total_kl": total_kl,
            "filtered_kl": filtered_kl,
            "kept_kl": kept_kl,
        }

    def _collect_stf_drop_updated_records(
        self,
        *,
        records: list[dict[str, Any]],
        disagree_mask: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        responses: torch.Tensor,
        distillation_loss_type: str,
        distillation_topk: Optional[int],
        uids: Optional[Any],
        branch: str,
        max_records: int,
    ) -> None:
        """Append compact records for tokens that remain after drop filter (actually trained)."""
        if max_records > 0 and len(records) >= max_records:
            return
        with torch.no_grad():
            teacher_top1 = teacher_logits.argmax(dim=-1)
            student_top1 = student_logits.argmax(dim=-1)
            token_div = self._per_token_training_signal(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                distillation_loss_type=distillation_loss_type,
                distillation_topk=distillation_topk,
            )
            b_idx, t_idx = (disagree_mask > 0.5).nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                return
            b_list = b_idx.detach().cpu().tolist()
            t_list = t_idx.detach().cpu().tolist()
            resp_cpu = responses.detach().cpu()
            t1_cpu = teacher_top1.detach().cpu()
            s1_cpu = student_top1.detach().cpu()
            div_cpu = token_div.detach().float().cpu()
            for bi, ti in zip(b_list, t_list):
                if max_records > 0 and len(records) >= max_records:
                    break
                uid = None
                if uids is not None:
                    try:
                        uid = str(uids[bi])
                    except Exception:
                        uid = None
                records.append(
                    {
                        "branch": branch,
                        "uid": uid,
                        "pos": int(ti),
                        "y_id": int(resp_cpu[bi, ti].item()),
                        "tea_top1_id": int(t1_cpu[bi, ti].item()),
                        "stu_top1_id": int(s1_cpu[bi, ti].item()),
                        "div": float(div_cpu[bi, ti].item()),
                    }
                )

    @staticmethod
    def _token_log_probs_from_logits(logits: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Per-position log π(y_t) for the sampled response tokens."""
        return F.log_softmax(logits, dim=-1).gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)

    def _maybe_mix_same_token_sampled_kl(
        self,
        *,
        l_main: torch.Tensor,
        kl_main_mean: float,
        disagree_mask: torch.Tensor,
        same_mask: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        responses: torch.Tensor,
        same_token_filter: bool = False,
        same_token_filter_mode: str = "drop",
        token_filter: Optional[bool] = None,
        token_filter_action: Optional[str] = None,
        kl_penalty: str,
    ) -> tuple[torch.Tensor, float, torch.Tensor, dict[str, float]]:
        """Optionally mix sampled-token PG KL on action_mask into the main distill loss."""
        enabled = bool(token_filter) if token_filter is not None else bool(same_token_filter)
        action = (token_filter_action or same_token_filter_mode or "drop").strip().lower()
        # legacy: mode=sampled_kl meant action=sampled_kl
        use_alt = action == "sampled_kl" or same_token_filter_mode == "sampled_kl"
        extra: dict[str, float] = {}
        if not (enabled and use_alt):
            return l_main, kl_main_mean, disagree_mask, extra

        teacher_lp = self._token_log_probs_from_logits(teacher_logits.detach(), responses)
        student_lp = self._token_log_probs_from_logits(student_logits, responses)
        l_skl, skl_metrics = compute_opsd_kl_loss(
            teacher_log_probs=teacher_lp,
            student_log_probs=student_lp,
            response_mask=same_mask,
            kl_penalty=kl_penalty,
            loss_avg_mode=self.config.loss_avg_mode,
        )
        n_main = disagree_mask.sum()
        n_skl = same_mask.sum()
        n_tot = (n_main + n_skl).clamp(min=1.0)
        l_combined = (l_main * n_main + l_skl * n_skl) / n_tot
        scale_mask = disagree_mask + same_mask
        extra = {
            "opsd/same_token_sampled_kl_mean": float(skl_metrics.get("opsd/kl_mean", 0.0)),
            "opsd/same_token_sampled_kl_token_count": float(n_skl.detach().item()),
            "opsd/same_token_jsd_token_count": float(n_main.detach().item()),
            "opsd/token_filter_alt_kl_mean": float(skl_metrics.get("opsd/kl_mean", 0.0)),
            "opsd/token_filter_tokens_alt": float(n_skl.detach().item()),
            "opsd/token_filter_tokens_main": float(n_main.detach().item()),
        }
        return l_combined, kl_main_mean, scale_mask, extra

    @staticmethod
    def _select_topk_support(
        *,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        k: int,
        topk_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if topk_source == "teacher":
            source_log_softmax = F.log_softmax(teacher_logits.detach(), dim=-1)
            topk_log_probs, topk_indices = torch.topk(source_log_softmax, k=k, dim=-1)
            student_log_softmax = F.log_softmax(student_logits, dim=-1)
            student_selected_log_probs = student_log_softmax.gather(-1, topk_indices)
            return topk_log_probs.detach(), student_selected_log_probs, topk_indices

        if topk_source == "student":
            student_log_softmax = F.log_softmax(student_logits, dim=-1)
            student_selected_log_probs, topk_indices = torch.topk(student_log_softmax, k=k, dim=-1)
            teacher_log_softmax = F.log_softmax(teacher_logits.detach(), dim=-1)
            teacher_selected_log_probs = teacher_log_softmax.gather(-1, topk_indices)
            return teacher_selected_log_probs.detach(), student_selected_log_probs, topk_indices

        raise ValueError(
            f"Unsupported opsd.distillation_topk_source={topk_source}. "
            "Expected one of: teacher, student."
        )

    @staticmethod
    def _build_stgca_advantages(
        *,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        lam: float,
        clip_range: float,
        negative_only: bool = False,
        thought_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sign_adv = torch.sign(advantages)
        delta = (teacher_log_probs.detach() - student_log_probs.detach()) * response_mask
        weights = torch.exp(sign_adv * delta) * response_mask

        low_clip_mask = ((weights < (1.0 - clip_range)).to(response_mask.dtype)) * response_mask
        high_clip_mask = ((weights > (1.0 + clip_range)).to(response_mask.dtype)) * response_mask
        clipped_weights = torch.clamp(weights, min=1.0 - clip_range, max=1.0 + clip_range)
        reweight = ((1.0 - lam) + lam * clipped_weights) * response_mask

        if negative_only:
            seq_negative = (advantages.sum(dim=-1, keepdim=True) < 0).float()
            reweight = seq_negative * reweight + (1.0 - seq_negative) * response_mask

        if thought_mask is not None:
            # Only reweight tokens inside <thought>...</thought> spans.
            # Tokens outside (format tags, answer, etc.) keep reweight=1 (pure GRPO).
            reweight = thought_mask * reweight + (1.0 - thought_mask) * response_mask

        stgca_advantages = advantages * reweight.detach()

        neg_mask = response_mask * (advantages < 0).float()
        neg_token_count = neg_mask.sum().clamp(min=1.0)
        metrics = {
            "grpo_opsd/stgca_delta_mean": VF.masked_mean(delta.detach(), response_mask).item(),
            "grpo_opsd/stgca_weight_mean": VF.masked_mean(weights.detach(), response_mask).item(),
            "grpo_opsd/stgca_clipped_weight_mean": VF.masked_mean(clipped_weights.detach(), response_mask).item(),
            "grpo_opsd/stgca_clip_low_ratio": VF.masked_mean(low_clip_mask.detach(), response_mask).item(),
            "grpo_opsd/stgca_clip_high_ratio": VF.masked_mean(high_clip_mask.detach(), response_mask).item(),
            "grpo_opsd/stgca_adv_mean": VF.masked_mean(stgca_advantages.detach(), response_mask).item(),
            "grpo_opsd/stgca_adv_abs_mean": VF.masked_mean(
                stgca_advantages.detach().abs(), response_mask
            ).item(),
        }
        if negative_only:
            metrics["grpo_opsd/stgca_negative_only"] = 1.0
            metrics["grpo_opsd/stgca_neg_seq_ratio"] = (
                (neg_mask.sum(dim=-1) > 0).float().mean().item()
            )
            metrics["grpo_opsd/stgca_neg_delta_mean"] = (
                (delta.detach() * neg_mask).sum() / neg_token_count
            ).item()
        if thought_mask is not None:
            thought_token_count = thought_mask.sum().item()
            total_token_count = response_mask.sum().item()
            metrics["grpo_opsd/stgca_thought_only"] = 1.0
            metrics["grpo_opsd/stgca_thought_token_ratio"] = (
                thought_token_count / max(total_token_count, 1.0)
            )
        return stgca_advantages, metrics

    def _forward_micro_batch_with_module(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        module: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Equivalent to _forward_micro_batch, but allows selecting a different module."""
        forward_module = module if module is not None else self.actor_module
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )
            else:
                pad_size = 0

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            output = forward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
            return full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

        output = forward_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits = output.logits
        logits.div_(temperature)
        return self.log_probs_from_logits(logits=logits[:, -response_length - 1 : -1], labels=responses)

    def _forward_micro_batch_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        module: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Return sliced logits aligned with `responses` for top-k distillation."""
        forward_module = module if module is not None else self.actor_module
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
            else:
                pad_size = 0

            output = forward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)

            if self.config.ulysses_size > 1:
                logits_rmpad = gather_outputs_and_unpad(
                    logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )

            full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
            logits = full_logits[:, -response_length - 1 : -1, :]
        else:
            output = forward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]

        return logits

    def update_policy_grpo_opsd(self, data: DataProto) -> dict[str, Any]:
        """Joint update for simple GRPO+RLSD mode.

        This path keeps the standard PPO/GRPO policy loss and injects Teacher
        guidance through token-level advantage reweighting on the same sampled
        responses.
        """
        self.actor_module.train()
        teacher_module = self._get_teacher_module()
        if teacher_module is not self.actor_module:
            teacher_module.eval()

        temperature = data.meta_info["temperature"]
        opsd_config = data.meta_info["opsd_config"]
        stgca_lambda = opsd_config.get("stgca_lambda", 0.5)
        stgca_clip_range = opsd_config.get("stgca_reweight_clip_range")
        if stgca_clip_range is None:
            stgca_clip_range = float(self.config.clip_ratio_low)
        stgca_negative_only = opsd_config.get("stgca_negative_only", False)
        stgca_thought_only = opsd_config.get("stgca_thought_only", False)

        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "grpo_opsd_teacher_input_ids",
            "grpo_opsd_teacher_attention_mask",
            "grpo_opsd_teacher_position_ids",
        ]
        if "ref_log_probs" in data.batch:
            select_keys.append("ref_log_probs")
        if "stgca_thought_mask" in data.batch:
            select_keys.append("stgca_thought_mask")

        non_tensor_keys = ["multi_modal_inputs"]
        if "uid" in data.non_tensor_batch:
            non_tensor_keys.append("uid")
        mini_batches = data.select(select_keys, non_tensor_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        _logged_stgca_hashes: set[str] = set()
        _stgca_target_uid = opsd_config.get("adv_log_target_uid", data.meta_info.get("adv_log_target_uid"))
        _stgca_target_response_hash = opsd_config.get(
            "adv_log_target_response_hash", data.meta_info.get("adv_log_target_response_hash")
        )

        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    if self.config.max_token_len_per_gpu is not None:
                        max_token_len = self.config.max_token_len_per_gpu
                    else:
                        max_input_len = max(
                            mini_batch.batch["input_ids"].size(-1),
                            mini_batch.batch["grpo_opsd_teacher_input_ids"].size(-1),
                        )
                        max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    teacher_mb = {
                        "input_ids": micro_batch.batch["grpo_opsd_teacher_input_ids"],
                        "attention_mask": micro_batch.batch["grpo_opsd_teacher_attention_mask"],
                        "position_ids": micro_batch.batch["grpo_opsd_teacher_position_ids"],
                        "responses": micro_batch.batch["responses"],
                    }
                    if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                        teacher_mb["multi_modal_inputs"] = micro_batch.non_tensor_batch["multi_modal_inputs"]

                    # Memory-efficient RLSD path: only compute per-token log probs
                    # without materializing the full (batch, seq, vocab) log_softmax tensor.
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    teacher_was_training = teacher_module.training
                    teacher_module.eval()
                    with torch.no_grad():
                        teacher_lp = self._forward_micro_batch_with_module(
                            teacher_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    if teacher_was_training:
                        teacher_module.train()
                    thought_mask = model_inputs.get("stgca_thought_mask") if stgca_thought_only else None
                    advantages_for_pg, stgca_metrics = self._build_stgca_advantages(
                        teacher_log_probs=teacher_lp,
                        student_log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        lam=stgca_lambda,
                        clip_range=stgca_clip_range,
                        negative_only=stgca_negative_only,
                        thought_mask=thought_mask,
                    )

                    # ---- Record STGCA data for the exact target response ----
                    if _stgca_target_uid is not None and "uid" in micro_batch.non_tensor_batch:
                        for target_pos, uid in enumerate(micro_batch.non_tensor_batch["uid"]):
                            if str(uid) != str(_stgca_target_uid):
                                continue

                            valid_len = int(response_mask[target_pos].sum().item())
                            if valid_len <= 0:
                                continue

                            response_hash = _compute_response_hash(
                                micro_batch.batch["responses"][target_pos], response_mask[target_pos]
                            )
                            if (
                                _stgca_target_response_hash is not None
                                and str(response_hash) != str(_stgca_target_response_hash)
                            ):
                                continue
                            if not response_hash or response_hash in _logged_stgca_hashes:
                                continue

                            delta_t = (teacher_lp[target_pos].detach() - log_probs[target_pos].detach()) * response_mask[target_pos]
                            sign_adv = torch.sign(advantages[target_pos])
                            weights_t = torch.exp(sign_adv * delta_t) * response_mask[target_pos]
                            clip_min_val = 1.0 - stgca_clip_range
                            clip_max_val = 1.0 + stgca_clip_range
                            clipped_weights_t = torch.clamp(weights_t, min=clip_min_val, max=clip_max_val)
                            reweight_t = ((1.0 - stgca_lambda) + stgca_lambda * clipped_weights_t) * response_mask[target_pos]
                            stgca_adv_t = advantages_for_pg[target_pos].detach()

                            stgca_record = {
                                "uid": str(uid),
                                "response_hash": response_hash,
                                "lambda": float(stgca_lambda),
                                "clip_range": float(stgca_clip_range),
                                "negative_only": bool(stgca_negative_only),
                                "thought_only": bool(stgca_thought_only),
                                "token_delta": delta_t[:valid_len].tolist(),
                                "token_reweight": reweight_t[:valid_len].tolist(),
                                "token_stgca_adv": stgca_adv_t[:valid_len].tolist(),
                            }
                            metrics["__stgca_group_samples__"].append(json.dumps(stgca_record))
                            _logged_stgca_hashes.add(response_hash)
                    # ---- End STGCA recording ----

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages_for_pg,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    base_kl_loss = None
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        base_kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + base_kl_loss * self.config.kl_coef
                    else:
                        loss = pg_loss
                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    if base_kl_loss is not None:
                        batch_metrics["actor/kl_loss"] = base_kl_loss.detach().item()
                        batch_metrics["actor/kl_coef"] = self.config.kl_coef
                    batch_metrics.update(stgca_metrics)
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self._maybe_sync_teacher(opsd_config, metrics, metric_prefix="grpo_opsd")

        return metrics

    def update_policy_grpo_opsd_joint(self, data: DataProto) -> dict[str, Any]:
        """Joint update: standard GRPO PG loss + OPSD sampled-token KL distillation.

        L = L_grpo + opsd_loss_coef * L_opsd_kl, where L_opsd_kl uses the
        Thinking Machines sampled-token reverse-KL estimator (distillation_loss_type=kl).
        """
        self.actor_module.train()
        teacher_module = self._get_teacher_module()
        if teacher_module is not self.actor_module:
            teacher_module.eval()

        temperature = data.meta_info["temperature"]
        opsd_config = data.meta_info["opsd_config"]
        opsd_loss_coef = opsd_config.get("opsd_loss_coef", 1.0)
        kl_penalty = opsd_config.get("opsd_kl_penalty", "kl")

        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "grpo_opsd_teacher_input_ids",
            "grpo_opsd_teacher_attention_mask",
            "grpo_opsd_teacher_position_ids",
        ]
        if "ref_log_probs" in data.batch:
            select_keys.append("ref_log_probs")
        non_tensor_keys = ["multi_modal_inputs"]
        mini_batches = data.select(select_keys, non_tensor_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    if self.config.max_token_len_per_gpu is not None:
                        max_token_len = self.config.max_token_len_per_gpu
                    else:
                        max_input_len = max(
                            mini_batch.batch["input_ids"].size(-1),
                            mini_batch.batch["grpo_opsd_teacher_input_ids"].size(-1),
                        )
                        max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    teacher_mb = {
                        "input_ids": micro_batch.batch["grpo_opsd_teacher_input_ids"],
                        "attention_mask": micro_batch.batch["grpo_opsd_teacher_attention_mask"],
                        "position_ids": micro_batch.batch["grpo_opsd_teacher_position_ids"],
                        "responses": micro_batch.batch["responses"],
                    }
                    if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                        teacher_mb["multi_modal_inputs"] = micro_batch.non_tensor_batch["multi_modal_inputs"]

                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    teacher_was_training = teacher_module.training
                    teacher_module.eval()
                    with torch.no_grad():
                        teacher_lp = self._forward_micro_batch_with_module(
                            teacher_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    if teacher_was_training:
                        teacher_module.train()

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    opsd_loss, opsd_metrics = compute_opsd_kl_loss(
                        teacher_log_probs=teacher_lp,
                        student_log_probs=log_probs,
                        response_mask=response_mask,
                        kl_penalty=kl_penalty,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    base_kl_loss = None
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        base_kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        pg_loss = pg_loss + base_kl_loss * self.config.kl_coef

                    token_scale = torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss = (pg_loss + opsd_loss_coef * opsd_loss) * token_scale
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    batch_metrics["grpo_opsd_joint/opsd_loss"] = opsd_loss.detach().item()
                    batch_metrics["grpo_opsd_joint/opsd_loss_coef"] = float(opsd_loss_coef)
                    batch_metrics["grpo_opsd_joint/total_loss"] = loss.detach().item()
                    if base_kl_loss is not None:
                        batch_metrics["actor/kl_loss"] = base_kl_loss.detach().item()
                        batch_metrics["actor/kl_coef"] = self.config.kl_coef
                    batch_metrics.update({f"grpo_opsd_joint/{k}": v for k, v in opsd_metrics.items()})
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self._maybe_sync_teacher(opsd_config, metrics, metric_prefix="grpo_opsd_joint")

        return metrics

    def update_policy_opsd(self, data: DataProto) -> dict[str, Any]:
        """Joint update: Navigator GRPO loss + OPSD distillation loss.

        Data layout in data.batch:
            Navigator keys (prefixed with "nav_"):
                nav_input_ids, nav_attention_mask, nav_position_ids, nav_responses,
                nav_response_mask, nav_old_log_probs, nav_advantages

            OPSD on-policy keys (prefixed with "on_"):
                on_student_input_ids, on_student_attention_mask, on_student_position_ids,
                on_student_responses, on_student_response_mask,
                on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids,
                on_teacher_responses, on_teacher_response_mask

            OPSD off-policy keys (prefixed with "off_"):
                off_student_input_ids, off_student_attention_mask, off_student_position_ids,
                off_student_responses, off_student_response_mask,
                off_teacher_input_ids, off_teacher_attention_mask, off_teacher_position_ids,
                off_teacher_responses, off_teacher_response_mask

        Args:
            data: DataProto with all keys for joint training.

        Returns:
            metrics: dict of training metrics.
        """
        self.actor_module.train()
        teacher_module = self._get_teacher_module()
        if teacher_module is not self.actor_module:
            teacher_module.eval()

        temperature = data.meta_info["temperature"]
        opsd_config = data.meta_info["opsd_config"]
        alpha = opsd_config["alpha"]
        lambda_nav = opsd_config["lambda_nav"]
        distillation_loss_type = opsd_config.get("distillation_loss_type", "kl")
        kl_penalty = opsd_config["opsd_kl_penalty"]
        distillation_topk = opsd_config.get("distillation_topk")
        distillation_topk_source = opsd_config.get("distillation_topk_source", "teacher")
        distillation_add_tail = opsd_config.get("distillation_add_tail", True)
        distill_token_selection = opsd_config.get("distill_token_selection", "all")
        distill_token_keep_ratio = opsd_config.get("distill_token_keep_ratio", 1.0)
        token_filter = bool(
            opsd_config.get("token_filter", opsd_config.get("same_token_filter", False))
        )
        keep_mask = bool(opsd_config.get("keep_mask", False))
        token_filter_mode = str(
            opsd_config.get("token_filter_mode", opsd_config.get("same_token_filter_mode", "taxonomy_drop"))
        ).strip().lower()
        token_filter_action = str(opsd_config.get("token_filter_action", "drop")).strip().lower()
        # legacy combined mode
        if token_filter_mode == "sampled_kl":
            token_filter_mode = "drop"
            token_filter_action = "sampled_kl"
        token_filter_top_n = int(
            opsd_config.get("token_filter_top_n", opsd_config.get("same_token_filter_top_n", 16))
        )
        token_filter_log_updated = bool(
            opsd_config.get(
                "token_filter_log_updated",
                opsd_config.get("same_token_filter_log_updated", True),
            )
        )
        token_filter_log_max_tokens = int(
            opsd_config.get(
                "token_filter_log_max_tokens",
                opsd_config.get("same_token_filter_log_max_tokens", 50000),
            )
        )
        token_filter_path = opsd_config.get("token_filter_path")
        token_filter_extra = {
            "keep_categories": opsd_config.get("token_filter_keep_categories"),
            "drop_categories": opsd_config.get("token_filter_drop_categories"),
            "classifier": opsd_config.get("token_filter_classifier", "qwen_rubric"),
            "token_filter_classifier": opsd_config.get("token_filter_classifier", "qwen_rubric"),
            "student_token_source": opsd_config.get("token_filter_student_source", "top1"),
            "token_filter_student_source": opsd_config.get("token_filter_student_source", "top1"),
        }
        token_filter_benign_mode = opsd_config.get("token_filter_benign_mode")
        if token_filter_benign_mode is not None:
            token_filter_benign_mode = str(token_filter_benign_mode).strip().lower() or None
        token_filter_benign_action = opsd_config.get("token_filter_benign_action")
        if token_filter_benign_action is not None:
            token_filter_benign_action = str(token_filter_benign_action).strip().lower() or None
        token_filter_benign_top_n = opsd_config.get("token_filter_benign_top_n")
        if token_filter_benign_top_n is not None:
            token_filter_benign_top_n = int(token_filter_benign_top_n)
        token_filter_benign_ratio = opsd_config.get("token_filter_benign_ratio")
        if token_filter_benign_ratio is not None:
            token_filter_benign_ratio = float(token_filter_benign_ratio)
        # aliases used by remaining local code
        same_token_filter = token_filter
        same_token_filter_mode = token_filter_mode
        same_token_filter_top_n = token_filter_top_n
        same_token_filter_log_updated = token_filter_log_updated
        same_token_filter_log_max_tokens = token_filter_log_max_tokens
        log_stf_drop_tokens = (
            token_filter and token_filter_log_updated and token_filter_action != "sampled_kl"
        )
        outcome_ppo_coef = opsd_config.get("outcome_ppo_coef", 0.0)
        self_distill_negative_off_policy = opsd_config.get("self_distill_negative_off_policy", False)
        self_distill_negative_off_policy_coef = opsd_config.get("self_distill_negative_off_policy_coef", 0.0)
        enable_overlap_metrics = opsd_config.get("enable_topk_overlap_metrics", True)
        overlap_topk = int(opsd_config.get("overlap_topk", 16))
        oracle_module = self._get_oracle_module(opsd_config)
        if opsd_config.get("enable_oracle_correction", False) and oracle_module is None:
            raise RuntimeError(
                "opsd.enable_oracle_correction=True but actor.oracle_module is None. "
                "Ensure opsd_main uses OPSDOracleFSDPWorker and oracle_model_path is set."
            )
        if oracle_module is not None:
            oracle_module.eval()

        skip_nav_grpo = (lambda_nav == 0)
        skip_off_policy = (alpha >= 1.0) and (not self_distill_negative_off_policy)

        if self_distill_negative_off_policy and distillation_loss_type != "jsd":
            raise ValueError(
                "opsd.self_distill_negative_off_policy requires distillation_loss_type=jsd. "
                "Negative KL is intentionally disallowed because it is numerically unstable."
            )

        metrics = defaultdict(list)
        overlap_sum = None
        overlap_count = None
        overlap_pos_sum = None
        overlap_pos_count = None
        oracle_overlap_sum = None
        oracle_overlap_count = None
        oracle_overlap_pos_sum = None
        oracle_overlap_pos_count = None
        track_oracle_overlap = enable_overlap_metrics and oracle_module is not None
        stf_tokens_before = None
        stf_tokens_after = None
        stf_tokens_alt = None
        stf_tokens_trained = None
        stf_tokens_main_harmful = None
        stf_tokens_main_benign = None
        stf_tokens_cat = None
        stf_filtered_kl = None
        stf_kept_kl = None
        stf_total_kl = None
        stf_drop_records: list[dict[str, Any]] = []

        for epoch in range(self.config.ppo_epochs):
            # === Navigator GRPO Data (skip when lambda_nav=0) ===
            if not skip_nav_grpo:
                nav_data = data.select(
                    batch_keys=[
                        "nav_input_ids", "nav_attention_mask", "nav_position_ids",
                        "nav_responses", "nav_response_mask", "nav_old_log_probs", "nav_advantages",
                    ],
                    non_tensor_batch_keys=["nav_multi_modal_inputs"],
                )

            # === OPSD On-Policy Data ===
            on_keys = [
                "on_student_input_ids", "on_student_attention_mask", "on_student_position_ids",
                "on_student_responses", "on_student_response_mask",
                "on_teacher_input_ids", "on_teacher_attention_mask", "on_teacher_position_ids",
                "on_teacher_responses", "on_teacher_response_mask",
                "on_student_old_log_probs", "on_student_outcome_advantages",
                "on_student_outcome_response_mask", "on_student_distill_response_mask",
            ]
            on_data = data.select(
                batch_keys=[k for k in on_keys if k in data.batch],
                non_tensor_batch_keys=[
                    k
                    for k in ("on_multi_modal_inputs", "on_uid", "on_data_type")
                    if k in data.non_tensor_batch
                ],
            )

            # === OPSD Off-Policy Data (skip when alpha=1.0) ===
            has_off_policy = False
            if not skip_off_policy:
                off_keys = [
                    "off_student_input_ids", "off_student_attention_mask", "off_student_position_ids",
                    "off_student_responses", "off_student_response_mask",
                    "off_teacher_input_ids", "off_teacher_attention_mask", "off_teacher_position_ids",
                    "off_teacher_responses", "off_teacher_response_mask",
                ]
                off_data = data.select(
                    batch_keys=[k for k in off_keys if k in data.batch],
                    non_tensor_batch_keys=[
                        k
                        for k in ("off_multi_modal_inputs", "off_data_type")
                        if k in data.non_tensor_batch
                    ],
                )
                has_off_policy = data.meta_info.get("num_valid_off_policy", 0) > 0

            # Compute token counts for loss scaling.
            # Distillation should normalize over valid distillation tokens (H+) rather than all B*K tokens.
            device = (
                on_data.batch["on_student_response_mask"].device
                if "on_student_response_mask" in on_data.batch
                else torch.device("cuda", torch.cuda.current_device())
            )
            zero_tokens = torch.zeros((), device=device, dtype=torch.float32)
            on_distill_total_tokens = (
                torch.sum(on_data.batch["on_student_distill_response_mask"]).float()
                if "on_student_distill_response_mask" in on_data.batch
                else torch.sum(on_data.batch["on_student_response_mask"]).float()
                if "on_student_response_mask" in on_data.batch
                else zero_tokens.clone()
            )
            off_distill_total_tokens = (
                torch.sum(off_data.batch["off_teacher_response_mask"]).float()
                if (not skip_off_policy and "off_teacher_response_mask" in off_data.batch)
                else zero_tokens.clone()
            )
            nav_total_tokens = (
                torch.sum(nav_data.batch["nav_response_mask"]).float()
                if not skip_nav_grpo
                else zero_tokens.clone()
            )

            dist.all_reduce(on_distill_total_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(off_distill_total_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(nav_total_tokens, op=dist.ReduceOp.SUM)

            distill_total_tokens = (on_distill_total_tokens + off_distill_total_tokens).clamp(min=1.0)
            policy_total_tokens = (nav_total_tokens + on_distill_total_tokens).clamp(min=1.0)

            if enable_overlap_metrics and overlap_sum is None:
                max_resp_len = 0
                if "on_student_responses" in on_data.batch:
                    max_resp_len = max(max_resp_len, on_data.batch["on_student_responses"].size(-1))
                if not skip_off_policy and "off_student_responses" in off_data.batch:
                    max_resp_len = max(max_resp_len, off_data.batch["off_student_responses"].size(-1))
                max_resp_len = max(max_resp_len, 1)
                overlap_sum = torch.zeros((), device=device, dtype=torch.float32)
                overlap_count = torch.zeros((), device=device, dtype=torch.float32)
                overlap_pos_sum = torch.zeros(max_resp_len, device=device, dtype=torch.float32)
                overlap_pos_count = torch.zeros(max_resp_len, device=device, dtype=torch.float32)
                if track_oracle_overlap:
                    oracle_overlap_sum = torch.zeros((), device=device, dtype=torch.float32)
                    oracle_overlap_count = torch.zeros((), device=device, dtype=torch.float32)
                    oracle_overlap_pos_sum = torch.zeros(max_resp_len, device=device, dtype=torch.float32)
                    oracle_overlap_pos_count = torch.zeros(max_resp_len, device=device, dtype=torch.float32)

            if same_token_filter and stf_tokens_before is None:
                stf_tokens_before = torch.zeros((), device=device, dtype=torch.float32)
                stf_tokens_after = torch.zeros((), device=device, dtype=torch.float32)
                stf_tokens_alt = torch.zeros((), device=device, dtype=torch.float32)
                stf_tokens_trained = torch.zeros((), device=device, dtype=torch.float32)
                stf_tokens_main_harmful = torch.zeros((), device=device, dtype=torch.float32)
                stf_tokens_main_benign = torch.zeros((), device=device, dtype=torch.float32)
                stf_tokens_cat = self._init_token_filter_cat_accums(device)
                stf_filtered_kl = torch.zeros((), device=device, dtype=torch.float32)
                stf_kept_kl = torch.zeros((), device=device, dtype=torch.float32)
                stf_total_kl = torch.zeros((), device=device, dtype=torch.float32)

            # Batch sizes
            if "on_student_input_ids" in on_data.batch and "on_teacher_input_ids" in on_data.batch:
                on_batch_size = min(
                    data.meta_info.get("num_valid_on_policy", 0),
                    on_data.batch["on_student_input_ids"].shape[0],
                    on_data.batch["on_teacher_input_ids"].shape[0],
                )
            else:
                on_batch_size = 0
            if has_off_policy and "off_student_input_ids" in off_data.batch and "off_teacher_input_ids" in off_data.batch:
                off_batch_size = min(
                    data.meta_info.get("num_valid_off_policy", 0),
                    off_data.batch["off_student_input_ids"].shape[0],
                    off_data.batch["off_teacher_input_ids"].shape[0],
                )
            else:
                off_batch_size = 0

            # Keep loop counts consistent across ranks to avoid collective desync.
            on_batch_size_t = torch.tensor(on_batch_size, device=device, dtype=torch.long)
            dist.all_reduce(on_batch_size_t, op=dist.ReduceOp.MIN)
            on_batch_size = int(on_batch_size_t.item())
            if not skip_off_policy:
                off_batch_size_t = torch.tensor(off_batch_size, device=device, dtype=torch.long)
                dist.all_reduce(off_batch_size_t, op=dist.ReduceOp.MIN)
                off_batch_size = int(off_batch_size_t.item())

            micro_bs = self.config.micro_batch_size_per_device_for_update
            dummy_on_rows = None
            if on_batch_size > 0:
                dummy_on_mask = (
                    (on_data.batch["on_teacher_attention_mask"].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_attention_mask"].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_response_mask"].sum(dim=-1) > 0)
                )
                if torch.any(dummy_on_mask):
                    dummy_on_rows = dummy_on_mask.nonzero(as_tuple=True)[0][:1]

            def _build_sync_dummy_on_pair():
                """Build a always-valid dummy teacher/student pair for collective-sync backward."""
                if "on_teacher_input_ids" not in on_data.batch or on_data.batch["on_teacher_input_ids"].shape[0] == 0:
                    return None, None
                row = dummy_on_rows if dummy_on_rows is not None else torch.tensor(
                    [0], device=on_data.batch["on_teacher_input_ids"].device, dtype=torch.long
                )
                dummy_teacher_mb = {
                    "input_ids": on_data.batch["on_teacher_input_ids"][row],
                    "attention_mask": on_data.batch["on_teacher_attention_mask"][row].clone(),
                    "position_ids": on_data.batch["on_teacher_position_ids"][row],
                    "responses": on_data.batch["on_student_responses"][row].clone(),
                }
                dummy_student_mb = {
                    "input_ids": on_data.batch["on_student_input_ids"][row],
                    "attention_mask": on_data.batch["on_student_attention_mask"][row].clone(),
                    "position_ids": on_data.batch["on_student_position_ids"][row],
                    "responses": on_data.batch["on_student_responses"][row].clone(),
                }
                if "on_multi_modal_inputs" in on_data.non_tensor_batch:
                    dummy_mm = on_data.non_tensor_batch["on_multi_modal_inputs"][row.cpu().numpy()]
                    dummy_teacher_mb["multi_modal_inputs"] = dummy_mm
                    dummy_student_mb["multi_modal_inputs"] = dummy_mm

                # Prevent padding-free unpad_input() from seeing all-zero attention.
                if torch.sum(dummy_teacher_mb["attention_mask"]) <= 0:
                    dummy_teacher_mb["attention_mask"][:, -1] = 1
                if torch.sum(dummy_student_mb["attention_mask"]) <= 0:
                    dummy_student_mb["attention_mask"][:, -1] = 1
                return dummy_teacher_mb, dummy_student_mb

            # --- Navigator GRPO micro-batches (skipped when lambda_nav=0) ---
            if not skip_nav_grpo:
                nav_batch_size = nav_data.batch["nav_input_ids"].shape[0]
                for start in range(0, nav_batch_size, micro_bs):
                    end = min(start + micro_bs, nav_batch_size)
                    nav_mb = {
                        "input_ids": nav_data.batch["nav_input_ids"][start:end],
                        "attention_mask": nav_data.batch["nav_attention_mask"][start:end],
                        "position_ids": nav_data.batch["nav_position_ids"][start:end],
                        "responses": nav_data.batch["nav_responses"][start:end],
                    }
                    if "nav_multi_modal_inputs" in nav_data.non_tensor_batch:
                        nav_mb["multi_modal_inputs"] = nav_data.non_tensor_batch["nav_multi_modal_inputs"][start:end]
                    nav_response_mask = nav_data.batch["nav_response_mask"][start:end]
                    nav_old_log_probs = nav_data.batch["nav_old_log_probs"][start:end]
                    nav_advantages = nav_data.batch["nav_advantages"][start:end]

                    nav_log_probs = self._forward_micro_batch(nav_mb, temperature=temperature)

                    nav_loss, nav_pg_metrics = compute_policy_loss(
                        old_log_probs=nav_old_log_probs,
                        log_probs=nav_log_probs,
                        advantages=nav_advantages,
                        response_mask=nav_response_mask,
                        clip_ratio_low=opsd_config.get("nav_clip_ratio_low", self.config.clip_ratio_low),
                        clip_ratio_high=opsd_config.get("nav_clip_ratio_high", self.config.clip_ratio_high),
                        clip_ratio_dual=opsd_config.get("nav_clip_ratio_dual", self.config.clip_ratio_dual),
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    scaled_nav_loss = (
                        lambda_nav * nav_loss * torch.sum(nav_response_mask) * self.world_size / policy_total_tokens
                    )
                    scaled_nav_loss.backward()

                    batch_metrics = {f"nav/{k}": v for k, v in nav_pg_metrics.items()}
                    batch_metrics["nav/grpo_loss"] = nav_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

            # --- OPSD On-Policy micro-batches ---
            for start in range(0, max(on_batch_size, 1), micro_bs):
                if on_batch_size == 0:
                    break
                end = min(start + micro_bs, on_batch_size)
                if end <= start:
                    continue

                valid_rows = (
                    (on_data.batch["on_teacher_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_response_mask"][start:end].sum(dim=-1) > 0)
                )
                if not torch.any(valid_rows):
                    dummy_teacher_mb, dummy_student_mb = _build_sync_dummy_on_pair()
                    if dummy_teacher_mb is None:
                        continue
                    if self._uses_logits_distillation(distillation_loss_type, distillation_topk):
                        with torch.no_grad():
                            _ = self._forward_micro_batch_logits(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_logits = self._forward_micro_batch_logits(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_logits.sum() * 0.0).backward()
                    else:
                        with torch.no_grad():
                            _ = self._forward_micro_batch_with_module(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_lp = self._forward_micro_batch(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_lp.sum() * 0.0).backward()
                    continue
                valid_idx = valid_rows.nonzero(as_tuple=True)[0]

                # Teacher forward (no grad) — evaluates student's response
                # Teacher prompt is pure text (question+hint), but may contain <image>
                # placeholders from the original question, so it needs multi_modal_inputs.
                teacher_on_mb = {
                    "input_ids": on_data.batch["on_teacher_input_ids"][start:end][valid_idx],
                    "attention_mask": on_data.batch["on_teacher_attention_mask"][start:end][valid_idx],
                    "position_ids": on_data.batch["on_teacher_position_ids"][start:end][valid_idx],
                    "responses": on_data.batch["on_student_responses"][start:end][valid_idx],  # Teacher sees student response
                }
                if "on_multi_modal_inputs" in on_data.non_tensor_batch:
                    teacher_on_mb["multi_modal_inputs"] = on_data.non_tensor_batch["on_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if teacher_on_mb["input_ids"].shape[0] == 0:
                    continue

                # Student forward (with grad)
                student_on_mb = {
                    "input_ids": on_data.batch["on_student_input_ids"][start:end][valid_idx],
                    "attention_mask": on_data.batch["on_student_attention_mask"][start:end][valid_idx],
                    "position_ids": on_data.batch["on_student_position_ids"][start:end][valid_idx],
                    "responses": on_data.batch["on_student_responses"][start:end][valid_idx],
                }
                on_response_mask = on_data.batch["on_student_response_mask"][start:end][valid_idx]
                distill_response_mask = (
                    on_data.batch["on_student_distill_response_mask"][start:end][valid_idx]
                    if "on_student_distill_response_mask" in on_data.batch
                    else on_response_mask
                )
                if "on_multi_modal_inputs" in on_data.non_tensor_batch:
                    student_on_mb["multi_modal_inputs"] = on_data.non_tensor_batch["on_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if student_on_mb["input_ids"].shape[0] == 0:
                    continue

                stf_mix_metrics: dict[str, float] = {}
                if self._uses_logits_distillation(distillation_loss_type, distillation_topk):
                    with torch.no_grad():
                        teacher_on_logits = self._forward_micro_batch_logits(
                            teacher_on_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                        oracle_on_logits = None
                        if oracle_module is not None:
                            # Oracle uses the student prompt (no privileged hint).
                            oracle_on_logits = self._forward_micro_batch_logits(
                                student_on_mb,
                                temperature=temperature,
                                module=oracle_module,
                            )
                    student_on_logits = self._forward_micro_batch_logits(student_on_mb, temperature=temperature)
                    if enable_overlap_metrics and overlap_sum is not None:
                        accumulate_topk_overlap_stats(
                            teacher_on_logits,
                            student_on_logits,
                            distill_response_mask,
                            k=overlap_topk,
                            overlap_sum=overlap_sum,
                            overlap_count=overlap_count,
                            overlap_pos_sum=overlap_pos_sum,
                            overlap_pos_count=overlap_pos_count,
                        )
                        if (
                            track_oracle_overlap
                            and oracle_on_logits is not None
                            and oracle_overlap_sum is not None
                        ):
                            accumulate_topk_overlap_stats(
                                oracle_on_logits,
                                student_on_logits,
                                distill_response_mask,
                                k=overlap_topk,
                                overlap_sum=oracle_overlap_sum,
                                overlap_count=oracle_overlap_count,
                                overlap_pos_sum=oracle_overlap_pos_sum,
                                overlap_pos_count=oracle_overlap_pos_count,
                            )
                    selected_on_mask, selection_metrics = self._build_distill_token_mask(
                        teacher_logits=teacher_on_logits,
                        student_logits=student_on_logits,
                        base_mask=distill_response_mask,
                        selection_mode=distill_token_selection,
                        keep_ratio=distill_token_keep_ratio,
                    )
                    distill_mask = selected_on_mask
                    same_mask = torch.zeros_like(selected_on_mask)
                    if same_token_filter:
                        data_types_mb = None
                        if "on_data_type" in on_data.non_tensor_batch:
                            data_types_mb = on_data.non_tensor_batch["on_data_type"][start:end][
                                valid_idx.detach().cpu().numpy()
                            ]
                        filt = self._run_token_filter_with_optional_benign(
                            mode=token_filter_mode,
                            action=token_filter_action,
                            top_n=token_filter_top_n,
                            benign_mode=token_filter_benign_mode,
                            benign_action=token_filter_benign_action,
                            benign_top_n=token_filter_benign_top_n,
                            benign_ratio=token_filter_benign_ratio,
                            data_types=data_types_mb,
                            teacher_logits=teacher_on_logits,
                            student_logits=student_on_logits,
                            base_mask=distill_mask,
                            responses=student_on_mb["responses"],
                            distillation_loss_type=distillation_loss_type,
                            distillation_topk=distillation_topk,
                            module_path=token_filter_path,
                            extra=token_filter_extra,
                        )
                        distill_mask = filt.keep_mask
                        same_mask = filt.action_mask
                        stf_stats = filt.stats
                        if stf_tokens_before is not None:
                            stf_tokens_before += stf_stats["tokens_before"]
                            stf_tokens_after += stf_stats["tokens_after"]
                            tokens_alt = float(stf_stats.get("tokens_alt", 0.0))
                            tokens_trained = float(
                                stf_stats.get(
                                    "tokens_trained",
                                    float(stf_stats["tokens_after"]) + tokens_alt,
                                )
                            )
                            stf_tokens_alt += tokens_alt
                            stf_tokens_trained += tokens_trained
                            harmful_main, benign_main = self._token_filter_main_by_data_split(
                                distill_mask, data_types_mb
                            )
                            stf_tokens_main_harmful += harmful_main
                            stf_tokens_main_benign += benign_main
                            self._accumulate_token_filter_cat_stats(stf_tokens_cat, stf_stats)
                            stf_filtered_kl += stf_stats["filtered_kl"]
                            stf_kept_kl += stf_stats["kept_kl"]
                            stf_total_kl += stf_stats["total_kl"]
                        if log_stf_drop_tokens:
                            uid_mb = None
                            if "on_uid" in on_data.non_tensor_batch:
                                uid_mb = on_data.non_tensor_batch["on_uid"][start:end][
                                    valid_idx.detach().cpu().numpy()
                                ]
                            self._collect_stf_drop_updated_records(
                                records=stf_drop_records,
                                disagree_mask=distill_mask,
                                teacher_logits=teacher_on_logits,
                                student_logits=student_on_logits,
                                responses=student_on_mb["responses"],
                                distillation_loss_type=distillation_loss_type,
                                distillation_topk=distillation_topk,
                                uids=uid_mb,
                                branch="on",
                                max_records=same_token_filter_log_max_tokens,
                            )
                    teacher_on_token_entropy = self._token_entropy_from_logits(teacher_on_logits.detach())
                    on_entropy_all_tokens = average_loss(
                        self._token_entropy_from_logits(student_on_logits),
                        distill_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    on_selected_teacher_entropy = average_loss(
                        teacher_on_token_entropy,
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    if distillation_loss_type in TEACHER_TOPK_RENORM_LOSS_TYPES:
                        k = min(distillation_topk or DEFAULT_TEACHER_TOPK, teacher_on_logits.size(-1))
                        l_on, topk_metrics = compute_teacher_topk_renorm_distill_loss(
                            loss_type=distillation_loss_type,
                            teacher_logits=teacher_on_logits,
                            student_logits=student_on_logits,
                            response_mask=distill_mask,
                            k=k,
                            loss_avg_mode=self.config.loss_avg_mode,
                            keep_mask=keep_mask,
                            **self._oracle_mix_kwargs(opsd_config, oracle_on_logits),
                        )
                        kl_on_mean = next(iter(topk_metrics.values()))
                    elif distillation_topk is not None:
                        k = min(distillation_topk, teacher_on_logits.size(-1))
                        teacher_topk_log_probs, student_topk_log_probs, teacher_topk_indices = (
                            self._select_topk_support(
                                teacher_logits=teacher_on_logits,
                                student_logits=student_on_logits,
                                k=k,
                                topk_source=distillation_topk_source,
                            )
                        )
                        if distillation_loss_type == "jsd":
                            l_on, topk_metrics = compute_opsd_topk_jsd_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_on_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_on_mean = topk_metrics["opsd/topk_jsd_mean"]
                        else:
                            l_on, topk_metrics = compute_opsd_topk_kl_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_on_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_on_mean = topk_metrics["opsd/topk_kl_mean"]
                    else:
                        l_on, jsd_metrics = compute_opsd_jsd_loss(
                            teacher_logits=teacher_on_logits,
                            student_logits=student_on_logits,
                            response_mask=distill_mask,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        kl_on_mean = jsd_metrics["opsd/jsd_mean"]
                        topk_metrics = jsd_metrics
                    l_on, kl_on_mean, distill_mask, stf_mix_metrics = self._maybe_mix_same_token_sampled_kl(
                        l_main=l_on,
                        kl_main_mean=kl_on_mean if isinstance(kl_on_mean, float) else float(kl_on_mean),
                        disagree_mask=distill_mask,
                        same_mask=same_mask,
                        teacher_logits=teacher_on_logits,
                        student_logits=student_on_logits,
                        responses=student_on_mb["responses"],
                        same_token_filter=same_token_filter,
                        same_token_filter_mode=same_token_filter_mode,
                        token_filter=token_filter,
                        token_filter_action=token_filter_action,
                        kl_penalty=kl_penalty,
                    )
                    on_entropy = self._masked_entropy_from_logits(
                        student_on_logits,
                        distill_mask,
                        self.config.loss_avg_mode,
                    )
                    on_entropy_estimate = average_loss(
                        -F.log_softmax(student_on_logits, dim=-1)
                        .gather(dim=-1, index=student_on_mb["responses"].unsqueeze(-1))
                        .squeeze(-1),
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                else:
                    if distill_token_selection != "all":
                        raise ValueError(
                            "opsd.distill_token_selection requires logits-based distillation. "
                            "Set opsd.distillation_loss_type to jsd, topk_jsd, topk_forward_kl, "
                            "topk_reverse_kl, or set opsd.distillation_topk."
                        )
                    if same_token_filter:
                        raise ValueError(
                            "opsd.token_filter requires logits-based distillation. "
                            "Set opsd.distillation_loss_type to jsd, topk_jsd, topk_forward_kl, "
                            "topk_reverse_kl, or set opsd.distillation_topk."
                        )
                    with torch.no_grad():
                        teacher_on_lp = self._forward_micro_batch_with_module(
                            teacher_on_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    student_on_lp = self._forward_micro_batch(student_on_mb, temperature=temperature)
                    # Sampled-token PG OPD: A=sg(logπ_T-logπ_S), L=-A*logπ_S
                    l_on, kl_metrics = compute_opsd_kl_loss(
                        teacher_log_probs=teacher_on_lp,
                        student_log_probs=student_on_lp,
                        response_mask=distill_response_mask,
                        kl_penalty=kl_penalty,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    kl_on_mean = kl_metrics["opsd/kl_mean"]
                    on_entropy = None
                    on_entropy_estimate = average_loss(
                        -student_on_lp,
                        distill_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    distill_mask = distill_response_mask
                    selection_metrics = {
                        "selected_token_ratio": 1.0,
                        "selected_token_count": float(distill_response_mask.sum().item()),
                    }
                    on_entropy_all_tokens = None
                    on_selected_teacher_entropy = None

                scaled_on_loss = alpha * l_on * torch.sum(distill_mask) * self.world_size / distill_total_tokens
                scaled_on_loss.backward()

                append_to_dict(metrics, {
                    "opsd/on_policy_loss": l_on.detach().item(),
                    "opsd/on_policy_kl": kl_on_mean if isinstance(kl_on_mean, float) else kl_on_mean.item(),
                    "opsd/on_policy_entropy_estimate": on_entropy_estimate.detach().item(),
                    "opsd/on_policy_selected_token_ratio": selection_metrics["selected_token_ratio"],
                    "opsd/on_policy_selected_token_count": selection_metrics["selected_token_count"],
                })
                if distillation_loss_type in TEACHER_TOPK_RENORM_LOSS_TYPES:
                    append_to_dict(metrics, topk_metrics)
                if stf_mix_metrics:
                    append_to_dict(metrics, stf_mix_metrics)
                if on_entropy is not None:
                    append_to_dict(metrics, {
                        "opsd/on_policy_entropy": on_entropy.detach().item(),
                        "opsd/on_policy_entropy_all_tokens": on_entropy_all_tokens.detach().item(),
                        "opsd/on_policy_selected_teacher_entropy": on_selected_teacher_entropy.detach().item(),
                    })

                if (
                    outcome_ppo_coef > 0
                    and "on_student_old_log_probs" in on_data.batch
                    and "on_student_outcome_advantages" in on_data.batch
                    and "on_student_outcome_response_mask" in on_data.batch
                ):
                    student_on_lp = self._forward_micro_batch(student_on_mb, temperature=temperature)
                    outcome_old_log_probs = on_data.batch["on_student_old_log_probs"][start:end][valid_idx]
                    outcome_advantages = on_data.batch["on_student_outcome_advantages"][start:end][valid_idx]
                    outcome_response_mask = on_data.batch["on_student_outcome_response_mask"][start:end][valid_idx]
                    outcome_pg_loss, outcome_pg_metrics = compute_policy_loss(
                        old_log_probs=outcome_old_log_probs,
                        log_probs=student_on_lp,
                        advantages=outcome_advantages,
                        response_mask=outcome_response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    scaled_outcome_loss = (
                        outcome_ppo_coef
                        * outcome_pg_loss
                        * torch.sum(outcome_response_mask)
                        * self.world_size
                        / policy_total_tokens
                    )
                    scaled_outcome_loss.backward()

                    batch_metrics = {f"opsd/outcome_{k}": v for k, v in outcome_pg_metrics.items()}
                    batch_metrics["opsd/outcome_ppo_loss"] = outcome_pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

            # --- OPSD Off-Policy micro-batches (skipped when alpha=1.0) ---
            for start in range(0, max(off_batch_size, 1), micro_bs):
                if skip_off_policy or not has_off_policy or off_batch_size == 0:
                    break
                end = min(start + micro_bs, off_batch_size)
                if end <= start:
                    continue

                valid_rows = (
                    (off_data.batch["off_teacher_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (off_data.batch["off_student_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (off_data.batch["off_teacher_response_mask"][start:end].sum(dim=-1) > 0)
                )
                if not torch.any(valid_rows):
                    dummy_teacher_mb, dummy_student_mb = _build_sync_dummy_on_pair()
                    if dummy_teacher_mb is None:
                        continue

                    if self._uses_logits_distillation(distillation_loss_type, distillation_topk):
                        with torch.no_grad():
                            _ = self._forward_micro_batch_logits(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_logits = self._forward_micro_batch_logits(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_logits.sum() * 0.0).backward()
                    else:
                        with torch.no_grad():
                            _ = self._forward_micro_batch_with_module(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_lp = self._forward_micro_batch(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_lp.sum() * 0.0).backward()

                    append_to_dict(metrics, {
                        "opsd/off_policy_loss": 0.0,
                        "opsd/off_policy_kl": 0.0,
                    })
                    continue
                valid_idx = valid_rows.nonzero(as_tuple=True)[0]

                # Teacher forward (no grad) — evaluates correct teacher response
                teacher_off_mb = {
                    "input_ids": off_data.batch["off_teacher_input_ids"][start:end][valid_idx],
                    "attention_mask": off_data.batch["off_teacher_attention_mask"][start:end][valid_idx],
                    "position_ids": off_data.batch["off_teacher_position_ids"][start:end][valid_idx],
                    "responses": off_data.batch["off_teacher_responses"][start:end][valid_idx],
                }
                if "off_multi_modal_inputs" in off_data.non_tensor_batch:
                    teacher_off_mb["multi_modal_inputs"] = off_data.non_tensor_batch["off_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if teacher_off_mb["input_ids"].shape[0] == 0:
                    continue

                # Student forward (with grad) — student processes correct teacher response
                student_off_mb = {
                    "input_ids": off_data.batch["off_student_input_ids"][start:end][valid_idx],
                    "attention_mask": off_data.batch["off_student_attention_mask"][start:end][valid_idx],
                    "position_ids": off_data.batch["off_student_position_ids"][start:end][valid_idx],
                    "responses": off_data.batch["off_teacher_responses"][start:end][valid_idx],
                }
                off_response_mask = off_data.batch["off_teacher_response_mask"][start:end][valid_idx]
                if "off_multi_modal_inputs" in off_data.non_tensor_batch:
                    student_off_mb["multi_modal_inputs"] = off_data.non_tensor_batch["off_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if student_off_mb["input_ids"].shape[0] == 0:
                    continue

                stf_mix_metrics = {}
                if self._uses_logits_distillation(distillation_loss_type, distillation_topk):
                    with torch.no_grad():
                        teacher_off_logits = self._forward_micro_batch_logits(
                            teacher_off_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                        oracle_off_logits = None
                        if oracle_module is not None:
                            oracle_off_logits = self._forward_micro_batch_logits(
                                student_off_mb,
                                temperature=temperature,
                                module=oracle_module,
                            )
                    student_off_logits = self._forward_micro_batch_logits(student_off_mb, temperature=temperature)
                    if enable_overlap_metrics and overlap_sum is not None:
                        accumulate_topk_overlap_stats(
                            teacher_off_logits,
                            student_off_logits,
                            off_response_mask,
                            k=overlap_topk,
                            overlap_sum=overlap_sum,
                            overlap_count=overlap_count,
                            overlap_pos_sum=overlap_pos_sum,
                            overlap_pos_count=overlap_pos_count,
                        )
                        if (
                            track_oracle_overlap
                            and oracle_off_logits is not None
                            and oracle_overlap_sum is not None
                        ):
                            accumulate_topk_overlap_stats(
                                oracle_off_logits,
                                student_off_logits,
                                off_response_mask,
                                k=overlap_topk,
                                overlap_sum=oracle_overlap_sum,
                                overlap_count=oracle_overlap_count,
                                overlap_pos_sum=oracle_overlap_pos_sum,
                                overlap_pos_count=oracle_overlap_pos_count,
                            )
                    selected_off_mask, selection_metrics = self._build_distill_token_mask(
                        teacher_logits=teacher_off_logits,
                        student_logits=student_off_logits,
                        base_mask=off_response_mask,
                        selection_mode=distill_token_selection,
                        keep_ratio=distill_token_keep_ratio,
                    )
                    distill_mask = selected_off_mask
                    same_mask = torch.zeros_like(selected_off_mask)
                    if same_token_filter:
                        data_types_mb = None
                        if "off_data_type" in off_data.non_tensor_batch:
                            data_types_mb = off_data.non_tensor_batch["off_data_type"][start:end][
                                valid_idx.detach().cpu().numpy()
                            ]
                        filt = self._run_token_filter_with_optional_benign(
                            mode=token_filter_mode,
                            action=token_filter_action,
                            top_n=token_filter_top_n,
                            benign_mode=token_filter_benign_mode,
                            benign_action=token_filter_benign_action,
                            benign_top_n=token_filter_benign_top_n,
                            benign_ratio=token_filter_benign_ratio,
                            data_types=data_types_mb,
                            teacher_logits=teacher_off_logits,
                            student_logits=student_off_logits,
                            base_mask=distill_mask,
                            responses=student_off_mb["responses"],
                            distillation_loss_type=distillation_loss_type,
                            distillation_topk=distillation_topk,
                            module_path=token_filter_path,
                            extra=token_filter_extra,
                        )
                        distill_mask = filt.keep_mask
                        same_mask = filt.action_mask
                        stf_stats = filt.stats
                        if stf_tokens_before is not None:
                            stf_tokens_before += stf_stats["tokens_before"]
                            stf_tokens_after += stf_stats["tokens_after"]
                            tokens_alt = float(stf_stats.get("tokens_alt", 0.0))
                            tokens_trained = float(
                                stf_stats.get(
                                    "tokens_trained",
                                    float(stf_stats["tokens_after"]) + tokens_alt,
                                )
                            )
                            stf_tokens_alt += tokens_alt
                            stf_tokens_trained += tokens_trained
                            harmful_main, benign_main = self._token_filter_main_by_data_split(
                                distill_mask, data_types_mb
                            )
                            stf_tokens_main_harmful += harmful_main
                            stf_tokens_main_benign += benign_main
                            self._accumulate_token_filter_cat_stats(stf_tokens_cat, stf_stats)
                            stf_filtered_kl += stf_stats["filtered_kl"]
                            stf_kept_kl += stf_stats["kept_kl"]
                            stf_total_kl += stf_stats["total_kl"]
                    teacher_off_token_entropy = self._token_entropy_from_logits(teacher_off_logits.detach())
                    off_entropy_all_tokens = average_loss(
                        self._token_entropy_from_logits(student_off_logits),
                        off_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    off_selected_teacher_entropy = average_loss(
                        teacher_off_token_entropy,
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    if distillation_loss_type in TEACHER_TOPK_RENORM_LOSS_TYPES:
                        k = min(distillation_topk or DEFAULT_TEACHER_TOPK, teacher_off_logits.size(-1))
                        l_off, topk_metrics = compute_teacher_topk_renorm_distill_loss(
                            loss_type=distillation_loss_type,
                            teacher_logits=teacher_off_logits,
                            student_logits=student_off_logits,
                            response_mask=distill_mask,
                            k=k,
                            loss_avg_mode=self.config.loss_avg_mode,
                            keep_mask=keep_mask,
                            **self._oracle_mix_kwargs(opsd_config, oracle_off_logits),
                        )
                        kl_off_mean = next(iter(topk_metrics.values()))
                    elif distillation_topk is not None:
                        k = min(distillation_topk, teacher_off_logits.size(-1))
                        teacher_topk_log_probs, student_topk_log_probs, teacher_topk_indices = (
                            self._select_topk_support(
                                teacher_logits=teacher_off_logits,
                                student_logits=student_off_logits,
                                k=k,
                                topk_source=distillation_topk_source,
                            )
                        )
                        if distillation_loss_type == "jsd":
                            l_off, topk_metrics = compute_opsd_topk_jsd_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_off_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_off_mean = topk_metrics["opsd/topk_jsd_mean"]
                        else:
                            l_off, topk_metrics = compute_opsd_topk_kl_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_off_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_off_mean = topk_metrics["opsd/topk_kl_mean"]
                    else:
                        l_off, jsd_metrics = compute_opsd_jsd_loss(
                            teacher_logits=teacher_off_logits,
                            student_logits=student_off_logits,
                            response_mask=distill_mask,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        kl_off_mean = jsd_metrics["opsd/jsd_mean"]
                    l_off, kl_off_mean, distill_mask, stf_mix_metrics = self._maybe_mix_same_token_sampled_kl(
                        l_main=l_off,
                        kl_main_mean=kl_off_mean if isinstance(kl_off_mean, float) else float(kl_off_mean),
                        disagree_mask=distill_mask,
                        same_mask=same_mask,
                        teacher_logits=teacher_off_logits,
                        student_logits=student_off_logits,
                        responses=student_off_mb["responses"],
                        same_token_filter=same_token_filter,
                        same_token_filter_mode=same_token_filter_mode,
                        token_filter=token_filter,
                        token_filter_action=token_filter_action,
                        kl_penalty=kl_penalty,
                    )
                    off_entropy = self._masked_entropy_from_logits(
                        student_off_logits,
                        distill_mask,
                        self.config.loss_avg_mode,
                    )
                    off_entropy_estimate = average_loss(
                        -F.log_softmax(student_off_logits, dim=-1)
                        .gather(dim=-1, index=student_off_mb["responses"].unsqueeze(-1))
                        .squeeze(-1),
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                else:
                    if distill_token_selection != "all":
                        raise ValueError(
                            "opsd.distill_token_selection requires logits-based distillation. "
                            "Set opsd.distillation_loss_type to jsd, topk_jsd, topk_forward_kl, "
                            "topk_reverse_kl, or set opsd.distillation_topk."
                        )
                    if same_token_filter:
                        raise ValueError(
                            "opsd.token_filter requires logits-based distillation. "
                            "Set opsd.distillation_loss_type to jsd, topk_jsd, topk_forward_kl, "
                            "topk_reverse_kl, or set opsd.distillation_topk."
                        )
                    with torch.no_grad():
                        teacher_off_lp = self._forward_micro_batch_with_module(
                            teacher_off_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    student_off_lp = self._forward_micro_batch(student_off_mb, temperature=temperature)
                    # Sampled-token PG OPD: A=sg(logπ_T-logπ_S), L=-A*logπ_S
                    l_off, kl_metrics = compute_opsd_kl_loss(
                        teacher_log_probs=teacher_off_lp,
                        student_log_probs=student_off_lp,
                        response_mask=off_response_mask,
                        kl_penalty=kl_penalty,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    kl_off_mean = kl_metrics["opsd/kl_mean"]
                    off_entropy = None
                    off_entropy_estimate = average_loss(
                        -student_off_lp,
                        off_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    distill_mask = off_response_mask
                    selection_metrics = {
                        "selected_token_ratio": 1.0,
                        "selected_token_count": float(off_response_mask.sum().item()),
                    }
                    off_entropy_all_tokens = None
                    off_selected_teacher_entropy = None

                if self_distill_negative_off_policy:
                    # Treat negative off-policy as an auxiliary regularizer rather than
                    # another token-count-weighted branch of the main OPSD objective.
                    # This avoids shrinking it again by off_tokens / total_tokens.
                    off_policy_coef = -self_distill_negative_off_policy_coef
                    scaled_off_loss = off_policy_coef * l_off
                else:
                    off_policy_coef = (1 - alpha)
                    scaled_off_loss = (
                        off_policy_coef * l_off * torch.sum(distill_mask) * self.world_size / distill_total_tokens
                    )
                scaled_off_loss.backward()

                append_to_dict(metrics, {
                    "opsd/off_policy_loss": l_off.detach().item(),
                    "opsd/off_policy_signed_objective": (off_policy_coef * l_off.detach()).item(),
                    "opsd/off_policy_kl": kl_off_mean if isinstance(kl_off_mean, float) else kl_off_mean.item(),
                    "opsd/off_policy_entropy_estimate": off_entropy_estimate.detach().item(),
                    "opsd/off_policy_selected_token_ratio": selection_metrics["selected_token_ratio"],
                    "opsd/off_policy_selected_token_count": selection_metrics["selected_token_count"],
                })
                if distillation_loss_type in TEACHER_TOPK_RENORM_LOSS_TYPES:
                    append_to_dict(metrics, topk_metrics)
                if stf_mix_metrics:
                    append_to_dict(metrics, stf_mix_metrics)
                if off_entropy is not None:
                    append_to_dict(metrics, {
                        "opsd/off_policy_entropy": off_entropy.detach().item(),
                        "opsd/off_policy_entropy_all_tokens": off_entropy_all_tokens.detach().item(),
                        "opsd/off_policy_selected_teacher_entropy": off_selected_teacher_entropy.detach().item(),
                    })

            # Optimizer step
            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        if enable_overlap_metrics and overlap_sum is not None:
            dist.all_reduce(overlap_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(overlap_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(overlap_pos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(overlap_pos_count, op=dist.ReduceOp.SUM)

            overlap_token_ratio = (overlap_sum / overlap_count.clamp(min=1.0)).item()
            append_to_dict(metrics, {"opsd/overlap_token_ratio": overlap_token_ratio})

            valid_pos = overlap_pos_count > 0
            pos_ratio = torch.zeros_like(overlap_pos_sum)
            pos_ratio[valid_pos] = overlap_pos_sum[valid_pos] / overlap_pos_count[valid_pos]
            pos_ratio_list = pos_ratio.detach().cpu().tolist()
            pos_count_list = overlap_pos_count.detach().cpu().tolist()
            metrics["opsd/overlap_token_ratio_by_idx"] = [pos_ratio_list]
            metrics["opsd/overlap_token_count_by_idx"] = [pos_count_list]
            metrics["opsd/overlap_topk"] = [float(overlap_topk)]

        if same_token_filter and stf_tokens_before is not None:
            dist.all_reduce(stf_tokens_before, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_tokens_after, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_tokens_alt, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_tokens_trained, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_tokens_main_harmful, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_tokens_main_benign, op=dist.ReduceOp.SUM)
            if stf_tokens_cat is not None:
                for _cat_t in stf_tokens_cat.values():
                    dist.all_reduce(_cat_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_filtered_kl, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_kept_kl, op=dist.ReduceOp.SUM)
            dist.all_reduce(stf_total_kl, op=dist.ReduceOp.SUM)
            tokens_before = stf_tokens_before.item()
            tokens_after = stf_tokens_after.item()
            tokens_alt = stf_tokens_alt.item()
            tokens_trained = stf_tokens_trained.item()
            tokens_main_harmful = stf_tokens_main_harmful.item()
            tokens_main_benign = stf_tokens_main_benign.item()
            total_kl = max(stf_total_kl.item(), 1e-12)
            cat_metrics = {}
            if stf_tokens_cat is not None:
                for name in self._TOKEN_FILTER_CAT_METRIC_NAMES:
                    cat_metrics[f"opsd/token_filter_tokens_cat_{name}"] = float(
                        stf_tokens_cat[f"tokens_cat_{name}"].item()
                    )
                    cat_metrics[f"opsd/token_filter_tokens_main_cat_{name}"] = float(
                        stf_tokens_cat[f"tokens_main_cat_{name}"].item()
                    )
            append_to_dict(
                metrics,
                {
                    # legacy names
                    "opsd/same_token_filter_tokens_before": tokens_before,
                    "opsd/same_token_filter_tokens_after": tokens_after,
                    "opsd/same_token_filter_tokens_filtered": tokens_before - tokens_after,
                    "opsd/same_token_filter_token_keep_ratio": (
                        0.0 if tokens_before <= 0 else tokens_after / tokens_before
                    ),
                    "opsd/same_token_filter_filtered_kl_share": stf_filtered_kl.item() / total_kl,
                    "opsd/same_token_filter_kept_kl_share": stf_kept_kl.item() / total_kl,
                    "opsd/same_token_filter_top_n": float(same_token_filter_top_n),
                    # new accounting
                    "opsd/token_filter_tokens_before": tokens_before,
                    "opsd/token_filter_tokens_main": tokens_after,
                    "opsd/token_filter_tokens_main_harmful": tokens_main_harmful,
                    "opsd/token_filter_tokens_main_benign": tokens_main_benign,
                    "opsd/token_filter_tokens_alt": tokens_alt,
                    "opsd/token_filter_tokens_dropped": max(tokens_before - tokens_trained, 0.0),
                    "opsd/token_filter_tokens_trained": tokens_trained,
                    "opsd/token_filter_token_train_ratio": (
                        0.0 if tokens_before <= 0 else tokens_trained / tokens_before
                    ),
                    "opsd/token_filter_action_sampled_kl": float(token_filter_action == "sampled_kl"),
                    "opsd/token_filter_mode_jsd_topn": float(token_filter_mode == "jsd_topn"),
                    "opsd/token_filter_mode_taxonomy_drop": float(token_filter_mode == "taxonomy_drop"),
                    "opsd/token_filter_mode_taxonomy_keep": float(token_filter_mode == "taxonomy_keep"),
                    "opsd/token_filter_student_source_sampled": float(
                        str(token_filter_extra.get("student_token_source", "sampled")).strip().lower()
                        in {"sampled", "response", "sample"}
                    ),
                    "opsd/token_filter_student_source_top1": float(
                        str(token_filter_extra.get("student_token_source", "sampled")).strip().lower()
                        in {"top1", "argmax", "logit_top1"}
                    ),
                    "opsd/token_filter_benign_mode_jsd_topn": float(
                        (token_filter_benign_mode or "") in {"jsd_topn", "forward_kl_topn", "fwd_kl_topn"}
                    ),
                    "opsd/token_filter_benign_mode_forward_kl_topn": float(
                        (token_filter_benign_mode or "") in {"jsd_topn", "forward_kl_topn", "fwd_kl_topn"}
                    ),
                    "opsd/token_filter_top_n": float(token_filter_top_n),
                    "opsd/token_filter_benign_top_n": float(
                        token_filter_benign_top_n
                        if token_filter_benign_top_n is not None
                        else token_filter_top_n
                    ),
                    "opsd/token_filter_benign_ratio": float(
                        token_filter_benign_ratio
                        if token_filter_benign_ratio is not None
                        else -1.0
                    ),
                    **cat_metrics,
                },
            )

        if log_stf_drop_tokens:
            # One JSON blob per rank (possibly empty); trainer merges before reduce_metrics.
            metrics["__stf_drop_updated_tokens__"] = [
                json.dumps(
                    {
                        "rank": int(self.rank),
                        "n": len(stf_drop_records),
                        "tokens": stf_drop_records,
                    },
                    ensure_ascii=False,
                )
            ]

        if track_oracle_overlap and oracle_overlap_sum is not None:
            dist.all_reduce(oracle_overlap_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(oracle_overlap_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(oracle_overlap_pos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(oracle_overlap_pos_count, op=dist.ReduceOp.SUM)

            oracle_overlap_token_ratio = (
                oracle_overlap_sum / oracle_overlap_count.clamp(min=1.0)
            ).item()
            append_to_dict(metrics, {"opsd/oracle_overlap_token_ratio": oracle_overlap_token_ratio})

            oracle_valid_pos = oracle_overlap_pos_count > 0
            oracle_pos_ratio = torch.zeros_like(oracle_overlap_pos_sum)
            oracle_pos_ratio[oracle_valid_pos] = (
                oracle_overlap_pos_sum[oracle_valid_pos] / oracle_overlap_pos_count[oracle_valid_pos]
            )
            metrics["opsd/oracle_overlap_token_ratio_by_idx"] = [oracle_pos_ratio.detach().cpu().tolist()]
            metrics["opsd/oracle_overlap_token_count_by_idx"] = [
                oracle_overlap_pos_count.detach().cpu().tolist()
            ]

        self._maybe_sync_teacher(opsd_config, metrics, metric_prefix="opsd")

        return metrics
