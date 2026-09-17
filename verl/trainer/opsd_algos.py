from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from jinja2 import Template

from ..protocol import DataProto
from ..utils import torch_functional as VF
from ..utils.dataset import process_image, process_video
from .core_algos import average_loss

TEACHER_TOPK_RENORM_LOSS_TYPES = frozenset({"topk_jsd", "topk_forward_kl", "topk_reverse_kl"})
DEFAULT_TEACHER_TOPK = 512
ORACLE_MIX_TYPES = frozenset({"geometric", "convex", "residual"})


def _has_dynamic_verifier_hint(
    verifier_feedback: Any,
    think_diagnosis: Any,
    answer_diagnosis: Any,
) -> bool:
    return bool(
        str(verifier_feedback or "").strip()
        or str(think_diagnosis or "").strip()
        or str(answer_diagnosis or "").strip()
    )


def apply_oracle_mix_on_teacher_topk(
    teacher_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    k: int,
    mix: str = "geometric",
    alpha: float = 0.2,
    residual_lambda: float = 0.5,
    delta_clip: float = 2.0,
    residual_relu: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Mix oracle into teacher distribution on **teacher top-k** support only.

    Returns
    -------
    mixed_topk_log_probs : Tensor
        Corrected teacher target log-probs on the top-k support, shape [..., k].
    topk_indices : Tensor
        Teacher-defined support indices, shape [..., k].
    metrics : dict
    """
    if mix not in ORACLE_MIX_TYPES:
        raise ValueError(f"Unknown oracle mix={mix}. Expected one of: {sorted(ORACLE_MIX_TYPES)}.")

    teacher_logits_detached = teacher_logits.detach()
    oracle_logits_detached = oracle_logits.detach()
    teacher_log_softmax = F.log_softmax(teacher_logits_detached, dim=-1)
    _, topk_indices = torch.topk(teacher_log_softmax, k=k, dim=-1)

    teacher_topk_logits = teacher_logits_detached.gather(-1, topk_indices)
    oracle_topk_logits = oracle_logits_detached.gather(-1, topk_indices)
    teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
    oracle_topk_log_probs = F.log_softmax(oracle_topk_logits, dim=-1)

    if mix == "convex":
        mixed_probs = (1.0 - alpha) * teacher_topk_log_probs.exp() + alpha * oracle_topk_log_probs.exp()
        mixed_topk_log_probs = mixed_probs.clamp_min(eps).log()
        mixed_topk_log_probs = mixed_topk_log_probs - mixed_topk_log_probs.logsumexp(dim=-1, keepdim=True)
    elif mix == "geometric":
        mixed_topk_log_probs = (1.0 - alpha) * teacher_topk_log_probs + alpha * oracle_topk_log_probs
        mixed_topk_log_probs = mixed_topk_log_probs - mixed_topk_log_probs.logsumexp(dim=-1, keepdim=True)
    else:  # residual
        delta = oracle_topk_log_probs - teacher_topk_log_probs
        if residual_relu:
            delta = F.relu(delta)
        if delta_clip is not None and delta_clip >= 0:
            delta = delta.clamp(min=-float(delta_clip), max=float(delta_clip))
        mixed_topk_log_probs = teacher_topk_log_probs + residual_lambda * delta
        mixed_topk_log_probs = mixed_topk_log_probs - mixed_topk_log_probs.logsumexp(dim=-1, keepdim=True)

    with torch.no_grad():
        tv = 0.5 * (teacher_topk_log_probs.exp() - mixed_topk_log_probs.exp()).abs().sum(dim=-1)
        metrics = {
            "opsd/oracle_mix_tv_mean": float(tv.mean().item()),
            "opsd/oracle_alpha": float(alpha),
            "opsd/oracle_lambda": float(residual_lambda),
        }
    return mixed_topk_log_probs, topk_indices, metrics


def select_teacher_topk_renormalized_log_probs(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int,
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Select teacher top-k tokens, optionally mix oracle on that support, re-normalize."""
    mix_metrics: dict[str, float] = {}
    if oracle_logits is not None and oracle_mix:
        teacher_topk_log_probs, topk_indices, mix_metrics = apply_oracle_mix_on_teacher_topk(
            teacher_logits=teacher_logits,
            oracle_logits=oracle_logits,
            k=k,
            mix=oracle_mix,
            alpha=oracle_alpha,
            residual_lambda=oracle_lambda,
            delta_clip=oracle_delta_clip,
            residual_relu=oracle_residual_relu,
        )
        student_topk_logits = student_logits.gather(-1, topk_indices)
        student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        return teacher_topk_log_probs, student_topk_log_probs, topk_indices, mix_metrics

    teacher_logits_detached = teacher_logits.detach()
    teacher_log_softmax = F.log_softmax(teacher_logits_detached, dim=-1)
    _, topk_indices = torch.topk(teacher_log_softmax, k=k, dim=-1)

    teacher_topk_logits = teacher_logits_detached.gather(-1, topk_indices)
    student_topk_logits = student_logits.gather(-1, topk_indices)

    teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
    student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
    return teacher_topk_log_probs, student_topk_log_probs, topk_indices, mix_metrics


def _teacher_topk_renorm_token_scores(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int,
    divergence: Literal["forward_kl", "reverse_kl", "jsd"],
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-position teacher top-k renorm divergence; logits may be [..., V]."""
    teacher_topk_log_probs, student_topk_log_probs, _, mix_metrics = select_teacher_topk_renormalized_log_probs(
        teacher_logits,
        student_logits,
        k,
        oracle_logits=oracle_logits,
        oracle_mix=oracle_mix,
        oracle_alpha=oracle_alpha,
        oracle_lambda=oracle_lambda,
        oracle_delta_clip=oracle_delta_clip,
        oracle_residual_relu=oracle_residual_relu,
    )
    if divergence == "forward_kl":
        teacher_probs = teacher_topk_log_probs.exp()
        scores = (teacher_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
    elif divergence == "reverse_kl":
        student_probs = student_topk_log_probs.exp()
        scores = (student_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)
    elif divergence == "jsd":
        teacher_probs = teacher_topk_log_probs.exp()
        student_probs = student_topk_log_probs.exp()
        mix_probs = (0.5 * (teacher_probs + student_probs)).clamp_min(eps)
        mix_log_probs = mix_probs.log()
        scores = 0.5 * (
            (teacher_probs * (teacher_topk_log_probs - mix_log_probs)).sum(dim=-1)
            + (student_probs * (student_topk_log_probs - mix_log_probs)).sum(dim=-1)
        )
    else:
        raise ValueError(f"Unknown divergence={divergence!r}")
    if divergence == "jsd":
        return scores.clamp(min=0.0, max=10.0), mix_metrics
    return scores.clamp(min=-10.0, max=10.0), mix_metrics


def _masked_teacher_topk_renorm_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    k: int,
    divergence: Literal["forward_kl", "reverse_kl", "jsd"],
    loss_avg_mode: Literal["token", "seq"] = "token",
    keep_mask: bool = False,
    metric_key: str = "opsd/topk_renorm_forward_kl_mean",
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Top-k renorm distill loss; optionally compute topk/KL only on keep positions."""
    score_kwargs = dict(
        oracle_logits=oracle_logits,
        oracle_mix=oracle_mix,
        oracle_alpha=oracle_alpha,
        oracle_lambda=oracle_lambda,
        oracle_delta_clip=oracle_delta_clip,
        oracle_residual_relu=oracle_residual_relu,
        eps=eps,
    )
    if keep_mask:
        pos_mask = response_mask.bool()
        if not bool(pos_mask.any().item()):
            zero = student_logits.sum() * 0.0
            return zero, {metric_key: 0.0, "opsd/keep_mask_sparse": 1.0, "opsd/keep_mask_n": 0.0}
        teacher_keep = teacher_logits[pos_mask]
        student_keep = student_logits[pos_mask]
        oracle_keep = oracle_logits[pos_mask] if oracle_logits is not None else None
        scores_keep, mix_metrics = _teacher_topk_renorm_token_scores(
            teacher_keep,
            student_keep,
            k,
            divergence,
            **{**score_kwargs, "oracle_logits": oracle_keep},
        )
        scores = teacher_logits.new_zeros(response_mask.shape, dtype=scores_keep.dtype)
        scores = scores.masked_scatter(pos_mask, scores_keep)
        metrics = {
            metric_key: VF.masked_mean(scores.detach(), response_mask).item(),
            "opsd/keep_mask_sparse": 1.0,
            "opsd/keep_mask_n": float(pos_mask.sum().item()),
        }
        metrics.update(mix_metrics)
        loss = average_loss(scores, response_mask, mode=loss_avg_mode)
        return loss, metrics

    scores, mix_metrics = _teacher_topk_renorm_token_scores(
        teacher_logits,
        student_logits,
        k,
        divergence,
        **score_kwargs,
    )
    loss = average_loss(scores, response_mask, mode=loss_avg_mode)
    metrics = {
        metric_key: VF.masked_mean(scores.detach(), response_mask).item(),
        "opsd/keep_mask_sparse": 0.0,
    }
    metrics.update(mix_metrics)
    return loss, metrics


def compute_opsd_teacher_topk_forward_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    k: int,
    loss_avg_mode: Literal["token", "seq"] = "token",
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
    keep_mask: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    return _masked_teacher_topk_renorm_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        response_mask=response_mask,
        k=k,
        divergence="forward_kl",
        loss_avg_mode=loss_avg_mode,
        keep_mask=keep_mask,
        metric_key="opsd/topk_renorm_forward_kl_mean",
        oracle_logits=oracle_logits,
        oracle_mix=oracle_mix,
        oracle_alpha=oracle_alpha,
        oracle_lambda=oracle_lambda,
        oracle_delta_clip=oracle_delta_clip,
        oracle_residual_relu=oracle_residual_relu,
    )


def compute_opsd_teacher_topk_reverse_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    k: int,
    loss_avg_mode: Literal["token", "seq"] = "token",
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
    keep_mask: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    return _masked_teacher_topk_renorm_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        response_mask=response_mask,
        k=k,
        divergence="reverse_kl",
        loss_avg_mode=loss_avg_mode,
        keep_mask=keep_mask,
        metric_key="opsd/topk_renorm_reverse_kl_mean",
        oracle_logits=oracle_logits,
        oracle_mix=oracle_mix,
        oracle_alpha=oracle_alpha,
        oracle_lambda=oracle_lambda,
        oracle_delta_clip=oracle_delta_clip,
        oracle_residual_relu=oracle_residual_relu,
    )


def compute_opsd_teacher_topk_jsd_renorm_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    k: int,
    loss_avg_mode: Literal["token", "seq"] = "token",
    eps: float = 1e-10,
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
    keep_mask: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    return _masked_teacher_topk_renorm_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        response_mask=response_mask,
        k=k,
        divergence="jsd",
        loss_avg_mode=loss_avg_mode,
        keep_mask=keep_mask,
        metric_key="opsd/topk_renorm_jsd_mean",
        oracle_logits=oracle_logits,
        oracle_mix=oracle_mix,
        oracle_alpha=oracle_alpha,
        oracle_lambda=oracle_lambda,
        oracle_delta_clip=oracle_delta_clip,
        oracle_residual_relu=oracle_residual_relu,
        eps=eps,
    )


def compute_teacher_topk_renorm_distill_loss(
    loss_type: str,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    k: int,
    loss_avg_mode: Literal["token", "seq"] = "token",
    oracle_logits: Optional[torch.Tensor] = None,
    oracle_mix: Optional[str] = None,
    oracle_alpha: float = 0.2,
    oracle_lambda: float = 0.5,
    oracle_delta_clip: float = 2.0,
    oracle_residual_relu: bool = False,
    keep_mask: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    kwargs = dict(
        oracle_logits=oracle_logits,
        oracle_mix=oracle_mix,
        oracle_alpha=oracle_alpha,
        oracle_lambda=oracle_lambda,
        oracle_delta_clip=oracle_delta_clip,
        oracle_residual_relu=oracle_residual_relu,
        keep_mask=keep_mask,
    )
    if loss_type == "topk_forward_kl":
        return compute_opsd_teacher_topk_forward_kl_loss(
            teacher_logits, student_logits, response_mask, k, loss_avg_mode, **kwargs
        )
    if loss_type == "topk_reverse_kl":
        return compute_opsd_teacher_topk_reverse_kl_loss(
            teacher_logits, student_logits, response_mask, k, loss_avg_mode, **kwargs
        )
    if loss_type == "topk_jsd":
        return compute_opsd_teacher_topk_jsd_renorm_loss(
            teacher_logits, student_logits, response_mask, k, loss_avg_mode, **kwargs
        )
    raise ValueError(
        f"Unsupported teacher top-k renormalized distillation loss type: {loss_type}. "
        f"Expected one of: {sorted(TEACHER_TOPK_RENORM_LOSS_TYPES)}."
    )


def compute_opsd_kl_loss(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "kl",
    loss_avg_mode: Literal["token", "seq"] = "token",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sampled-token on-policy distillation (Thinking Machines / PG OPD).

    Uses the single-sample reverse-KL estimator as a *detached* advantage, then
    applies a policy-gradient update on the student's own tokens:

        A = sg(log π_T(y) - log π_S(y))
        L = -E[A * log π_S(y)]

    Directly minimizing ``log π_T - log π_S`` (without stop-grad on the
    estimator) is incorrect: gradients ignore the teacher and collapse into
    maximizing the student's own token likelihood.

    Only ``kl_penalty='kl'`` (k1 reverse-KL estimator) is supported.
    """
    if kl_penalty != "kl":
        raise ValueError(
            "Sampled-token PG distillation only supports opsd_kl_penalty='kl' "
            f"(reverse-KL estimator as reward), got {kl_penalty!r}."
        )

    # Stop-grad on the estimator so teacher preference survives in the gradient.
    advantage = (teacher_log_probs.detach() - student_log_probs).detach()
    pg_token_loss = -(advantage * student_log_probs)
    loss = average_loss(pg_token_loss, response_mask, mode=loss_avg_mode)
    advantage_mean = VF.masked_mean(advantage, response_mask).item()
    return loss, {
        "opsd/kl_mean": advantage_mean,
        "opsd/sampled_token_advantage_mean": advantage_mean,
    }


def compute_opsd_topk_kl_loss(
    teacher_topk_log_probs: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_all_logits: torch.Tensor,
    response_mask: torch.Tensor,
    add_tail: bool = True,
    loss_avg_mode: Literal["token", "seq"] = "token",
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_topk_probs = teacher_topk_log_probs.detach().exp()
    topk_kl = teacher_topk_probs * (teacher_topk_log_probs.detach() - student_topk_log_probs)
    topk_kl = topk_kl.sum(dim=-1)

    if add_tail:
        teacher_tail_prob = (1.0 - teacher_topk_probs.sum(dim=-1)).clamp(min=1e-10)
        teacher_tail_log_prob = teacher_tail_prob.log()

        student_log_softmax = F.log_softmax(student_all_logits, dim=-1)
        student_topk_log_probs_full = student_log_softmax.gather(dim=-1, index=teacher_topk_indices)
        student_topk_probs = student_topk_log_probs_full.exp()
        student_tail_prob = (1.0 - student_topk_probs.sum(dim=-1)).clamp(min=1e-10)
        student_tail_log_prob = student_tail_prob.log()

        tail_kl = teacher_tail_prob * (teacher_tail_log_prob - student_tail_log_prob)
        topk_kl = topk_kl + tail_kl

    topk_kl = topk_kl.clamp(min=-10.0, max=10.0)
    loss = average_loss(topk_kl, response_mask, mode=loss_avg_mode)
    kl_mean = VF.masked_mean(topk_kl.detach(), response_mask).item()
    return loss, {"opsd/topk_kl_mean": kl_mean}


def compute_opsd_jsd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_avg_mode: Literal["token", "seq"] = "token",
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_probs = student_log_probs.exp()

    mix_probs = 0.5 * (teacher_probs + student_probs)
    mix_log_probs = mix_probs.clamp_min(eps).log()

    jsd = 0.5 * (
        (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
        + (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
    )
    jsd = jsd.clamp(min=0.0, max=10.0)

    loss = average_loss(jsd, response_mask, mode=loss_avg_mode)
    jsd_mean = VF.masked_mean(jsd.detach(), response_mask).item()
    return loss, {"opsd/jsd_mean": jsd_mean}


def compute_opsd_topk_jsd_loss(
    teacher_topk_log_probs: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_all_logits: torch.Tensor,
    response_mask: torch.Tensor,
    add_tail: bool = True,
    loss_avg_mode: Literal["token", "seq"] = "token",
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_topk_probs = teacher_topk_log_probs.detach().exp()

    student_log_softmax = F.log_softmax(student_all_logits, dim=-1)
    student_topk_log_probs_full = student_log_softmax.gather(dim=-1, index=teacher_topk_indices)
    student_topk_probs = student_topk_log_probs_full.exp()

    if add_tail:
        teacher_tail_prob = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=eps)
        student_tail_prob = (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=eps)

        teacher_probs = torch.cat([teacher_topk_probs, teacher_tail_prob], dim=-1)
        student_probs = torch.cat([student_topk_probs, student_tail_prob], dim=-1)
    else:
        teacher_probs = teacher_topk_probs / teacher_topk_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
        student_probs = student_topk_probs / student_topk_probs.sum(dim=-1, keepdim=True).clamp(min=eps)

    teacher_log_probs = teacher_probs.clamp_min(eps).log()
    student_log_probs = student_probs.clamp_min(eps).log()
    mix_probs = 0.5 * (teacher_probs + student_probs)
    mix_log_probs = mix_probs.clamp_min(eps).log()

    jsd = 0.5 * (
        (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
        + (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
    )
    jsd = jsd.clamp(min=0.0, max=10.0)

    loss = average_loss(jsd, response_mask, mode=loss_avg_mode)
    jsd_mean = VF.masked_mean(jsd.detach(), response_mask).item()
    return loss, {"opsd/topk_jsd_mean": jsd_mean}


def compute_navigator_advantage(
    hint_rewards: torch.Tensor,
    hint_uids: np.ndarray | list[Any],
    K: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    del K
    scores = hint_rewards.float()
    id2score: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    id2mean: dict[str, torch.Tensor] = {}
    id2std: dict[str, torch.Tensor] = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        uid = hint_uids[i] if isinstance(hint_uids[i], str) else str(hint_uids[i])
        id2score[uid].append(scores[i])

    for uid in id2score:
        group_scores = torch.tensor(id2score[uid], device=scores.device, dtype=scores.dtype)
        id2mean[uid] = group_scores.mean()
        id2std[uid] = group_scores.std()

    advantages = torch.zeros_like(scores)
    for i in range(bsz):
        uid = hint_uids[i] if isinstance(hint_uids[i], str) else str(hint_uids[i])
        std = id2std[uid]
        if std < eps:
            advantages[i] = 0.0
            continue
        advantages[i] = (scores[i] - id2mean[uid]) / (std + eps)

    return advantages


def compute_curriculum_p_drop(global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(global_step / warmup_steps, 1.0)


def resolve_dynamic_max_response_length(
    global_step: int,
    *,
    enabled: bool,
    default: int,
    phase1_end: int,
    phase1_len: int,
    phase2_end: int,
    phase2_len: int,
    phase3_len: int,
) -> int:
    """Resolve student rollout max tokens for the current training step."""
    if not enabled:
        return default
    if global_step <= phase1_end:
        return phase1_len
    if global_step <= phase2_end:
        return phase2_len
    return phase3_len


def decide_adaptive_rollout_length(
    *,
    n_student_fails: int,
    trr_by_prefix: dict[int, float],
    prefix_lengths: list[int],
    min_student_fails: int,
    trr_thresh: float,
    current_length: int,
    monotonic: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Map Teacher Rescue Rate (TRR) at fixed prefix lengths to student rollout length.

    Early-exit: if the student fails on fewer than ``min_student_fails`` probe prompts,
    keep ``current_length`` (student is already good enough; skip length changes).

    Otherwise pick the longest probed prefix whose **TRR** (safe rate over the
    full probe set) is still >= ``trr_thresh``. If none qualify, fall back to the
    shortest probed length.
    """
    lengths = [int(x) for x in prefix_lengths]
    if not lengths:
        raise ValueError("prefix_lengths must be non-empty")

    info: dict[str, Any] = {
        "n_student_fails": int(n_student_fails),
        "min_student_fails": int(min_student_fails),
        "trr_thresh": float(trr_thresh),
        "current_length": int(current_length),
        "trr_by_prefix": {int(k): float(v) for k, v in trr_by_prefix.items()},
        "skipped": False,
        "reason": "",
    }

    if n_student_fails < min_student_fails:
        info["skipped"] = True
        info["reason"] = "student_good_enough"
        info["recommended_length"] = int(current_length)
        return int(current_length), info

    qualified = [L for L in lengths if float(trr_by_prefix.get(L, 0.0)) >= float(trr_thresh)]
    recommended = max(qualified) if qualified else min(lengths)
    if monotonic:
        recommended = max(int(recommended), int(current_length))

    info["qualified_lengths"] = qualified
    info["recommended_length"] = int(recommended)
    info["reason"] = "trr_threshold"
    return int(recommended), info


def compute_trr_by_prefix(
    *,
    student_token_lens: list[int],
    teacher_alone_safe: list[bool],
    student_unsafe: list[bool],
    teacher_full_safe_by_prefix: dict[int, list[Optional[bool]]],
    prefix_lengths: list[int],
    short_prefix_as_fail: bool = True,
) -> dict[int, dict[str, float]]:
    """Teacher DSR(L) over the full probe set (no student/teacher intersect filter).

    DSR(L) = (# samples where teacher continuation after student prefix L is safe) / N,
    where N is the number of probe samples (typically all harmful val rows).

    ``student_unsafe`` / ``teacher_alone_safe`` are kept for API compatibility / logging
    but are **not** used to restrict the denominator.
    Short prefixes (``student_token_len <= L``) still count as fail when
    ``short_prefix_as_fail`` is True and the per-sample flag is missing.
    """
    n = len(student_unsafe)
    if n == 0 and teacher_alone_safe:
        n = len(teacher_alone_safe)
    eval_idx = list(range(n))
    n_eval = len(eval_idx)
    out: dict[int, dict[str, float]] = {}
    for L in prefix_lengths:
        L = int(L)
        safe_flags = teacher_full_safe_by_prefix.get(L, [None] * n)
        success = 0
        n_short = 0
        n_missing = 0
        for i in eval_idx:
            if short_prefix_as_fail and i < len(student_token_lens) and int(student_token_lens[i]) <= L:
                # Prefer an explicit flag if the trainer already filled one (e.g. used
                # the full short student string as prefix); otherwise count as fail.
                flag = safe_flags[i] if i < len(safe_flags) else None
                if flag is None:
                    n_short += 1
                    continue
            else:
                flag = safe_flags[i] if i < len(safe_flags) else None
            if flag is None:
                n_missing += 1
                continue
            if bool(flag):
                success += 1
        dsr = (success / n_eval) if n_eval else 0.0
        out[L] = {
            "total": float(n_eval),
            "success": float(success),
            "n_short_as_fail": float(n_short),
            "n_missing_as_fail": float(n_missing),
            "dsr": float(dsr),
            "trr": float(dsr),  # alias for decide_adaptive_rollout_length
        }
    return out


def attach_token_prefix_to_raw_prompt_ids(
    gen_batch: DataProto,
    prefix_token_ids: list[list[int]],
) -> DataProto:
    """Append student prefix token ids onto teacher ``raw_prompt_ids`` for vLLM continuation."""
    if "raw_prompt_ids" not in gen_batch.non_tensor_batch:
        raise KeyError("gen_batch missing raw_prompt_ids")
    base_ids = gen_batch.non_tensor_batch["raw_prompt_ids"]
    if len(base_ids) != len(prefix_token_ids):
        raise ValueError(
            f"prefix batch size mismatch: prompts={len(base_ids)} prefixes={len(prefix_token_ids)}"
        )
    merged = [list(base_ids[i]) + list(prefix_token_ids[i]) for i in range(len(base_ids))]
    gen_batch.non_tensor_batch["raw_prompt_ids"] = np.array(merged, dtype=object)
    return gen_batch


def compute_topk_token_overlap_ratio(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Overlap ratio between teacher/student top-k vocab sets at each token position."""
    k = min(k, teacher_logits.size(-1))
    if k <= 0:
        raise ValueError(f"overlap top-k must be positive, got {k}.")

    teacher_topk = torch.topk(teacher_logits.detach(), k=k, dim=-1).indices
    student_topk = torch.topk(student_logits.detach(), k=k, dim=-1).indices
    teacher_in_student = (teacher_topk.unsqueeze(-1) == student_topk.unsqueeze(-2)).any(dim=-1).float()
    return teacher_in_student.sum(dim=-1) / float(k)


def accumulate_topk_overlap_stats(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    k: int,
    overlap_sum: torch.Tensor,
    overlap_count: torch.Tensor,
    overlap_pos_sum: torch.Tensor,
    overlap_pos_count: torch.Tensor,
) -> None:
    """Accumulate token-weighted overlap stats for one micro-batch."""
    per_token_overlap = compute_topk_token_overlap_ratio(teacher_logits, student_logits, k)
    mask = response_mask.to(dtype=per_token_overlap.dtype)
    overlap_sum.add_((per_token_overlap * mask).sum())
    overlap_count.add_(mask.sum())

    seq_len = min(per_token_overlap.size(1), overlap_pos_sum.size(0))
    if seq_len > 0:
        overlap_pos_sum[:seq_len].add_((per_token_overlap[:, :seq_len] * mask[:, :seq_len]).sum(dim=0))
        overlap_pos_count[:seq_len].add_(mask[:, :seq_len].sum(dim=0))


def _compute_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: dict[str, Any],
    processor: Any,
) -> torch.Tensor:
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
            attention_mask=attention_mask,
        )
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
    else:
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
    return position_ids


def _build_messages_from_prompt_text(
    prompt_text: str,
    multi_modal_item: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    multi_modal_item = multi_modal_item or {}

    has_images = "images" in multi_modal_item and len(multi_modal_item["images"]) > 0
    has_videos = any(
        key in multi_modal_item and multi_modal_item[key] is not None and len(multi_modal_item[key]) > 0
        for key in ("video", "videos")
    ) or "preprocessed_video_path" in multi_modal_item

    if has_images:
        content_list: list[dict[str, Any]] = []
        for i, content in enumerate(prompt_text.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})
            if content:
                content_list.append({"type": "text", "text": content})
        return [{"role": "user", "content": content_list}]

    if has_videos:
        content_list = []
        for i, content in enumerate(prompt_text.split("<video>")):
            if i != 0:
                content_list.append({"type": "video"})
            if content:
                content_list.append({"type": "text", "text": content})
        return [{"role": "user", "content": content_list}]

    return [{"role": "user", "content": prompt_text}]


def truncate_text_by_tokens(text: Any, tokenizer: Any, max_tokens: Optional[int]) -> str:
    """Truncate a text string by tokenizer token count instead of character count."""
    if text is None:
        return ""

    text = text if isinstance(text, str) else str(text)
    if max_tokens is None:
        return text
    if max_tokens <= 0:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text

    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def _tokenize_prompt(
    prompt: str,
    tokenizer: Any,
    processor: Any,
    multi_modal_item: Optional[dict[str, Any]],
    max_prompt_length: int,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], str]:
    multi_modal_item = multi_modal_item or {}

    if processor is not None and "images" in multi_modal_item and len(multi_modal_item["images"]) > 0:
        processed_images = [
            process_image(image, min_pixels=image_min_pixels, max_pixels=image_max_pixels)
            for image in multi_modal_item["images"]
        ]
        model_inputs = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "preprocessed_video_path" in multi_modal_item:
        preprocessed_data = torch.load(
            multi_modal_item["preprocessed_video_path"],
            map_location="cpu",
            weights_only=False,
        )
        model_inputs = processor(
            [prompt],
            [preprocessed_data["frames"]],
            add_special_tokens=False,
            video_metadata=[preprocessed_data["metadata"]],
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "video" in multi_modal_item and len(multi_modal_item["video"]) > 0:
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "videos": multi_modal_item["video"],
            "add_special_tokens": False,
            "return_tensors": "pt",
            "do_resize": False,
            "do_sample_frames": False,
        }
        video_metadatas = multi_modal_item.get("video_metadatas", None)
        if video_metadatas is not None and len(video_metadatas) > 0:
            processor_kwargs["video_metadata"] = video_metadatas
        model_inputs = processor(**processor_kwargs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "videos" in multi_modal_item and len(multi_modal_item["videos"]) > 0:
        processed_video_frames = []
        video_metadatas = []
        for video in multi_modal_item["videos"]:
            result = process_video(
                video,
                video_min_pixels if video_min_pixels else 4096,
                video_max_pixels if video_max_pixels else 65536,
                video_max_frames,
                video_fps,
                True,
            )
            if isinstance(result, tuple) and len(result) == 2:
                video_data, _ = result
                if isinstance(video_data, tuple) and len(video_data) == 2:
                    frames, metadata = video_data
                    processed_video_frames.append(frames)
                    video_metadatas.append(metadata)
                else:
                    processed_video_frames.append(video_data)
                    video_metadatas = None
                    break
            else:
                processed_video_frames.append(result)
                video_metadatas = None

        processor_kwargs = {
            "text": [prompt],
            "videos": processed_video_frames,
            "add_special_tokens": False,
            "return_tensors": "pt",
            "do_sample_frames": False,
        }
        if video_metadatas is not None and len(video_metadatas) > 0:
            processor_kwargs["video_metadata"] = video_metadatas
            processor_kwargs["do_resize"] = False
        model_inputs = processor(**processor_kwargs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    else:
        model_inputs = {}
        model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

    position_ids = _compute_position_ids(input_ids, attention_mask, model_inputs, processor)
    input_ids, attention_mask, position_ids = VF.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="left",
    )

    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]

    return input_ids, attention_mask, position_ids, raw_prompt_ids, prompt


def construct_navigator_prompts(
    questions: list[Any] | np.ndarray,
    trajectories: list[Any] | np.ndarray,
    tokenizer: Any,
    processor: Any,
    student_trajectories: Optional[list[Any] | np.ndarray] = None,
    template_path: Optional[str] = None,
    max_traj_len: int = 2048,
    max_prompt_length: int = 4096,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    if template_path:
        with open(template_path, encoding="utf-8") as f:
            template = Template(f.read().strip())
    else:
        template = Template(
            "You are a helpful navigator. Given a question and a reasoning trajectory from another model, "
            "extract a concise hint that captures the key insight for solving the problem.\n\n"
            "Question: {{ question }}\n\nTrajectory:\n{{ trajectory }}\n\n"
            "Provide a concise hint that captures the key reasoning insight. "
            "Output your hint within <hint> </hint> tags."
        )

    batch_size = len(questions)
    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []
    all_raw_prompt = []
    all_multi_modal = []

    for i in range(batch_size):
        question = questions[i] if isinstance(questions[i], str) else str(questions[i])
        trajectory = truncate_text_by_tokens(trajectories[i], tokenizer, max_traj_len)
        student_trajectory = ""
        if student_trajectories is not None:
            student_trajectory = truncate_text_by_tokens(student_trajectories[i], tokenizer, max_traj_len)

        prompt_text = template.render(
            question=question,
            trajectory=trajectory,
            student_trajectory=student_trajectory,
        )
        mm_item = multi_modal_data[i] if multi_modal_data is not None else None
        messages = _build_messages_from_prompt_text(prompt_text, mm_item)
        if processor is not None:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))
        else:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))

        input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = _tokenize_prompt(
            prompt,
            tokenizer,
            processor,
            mm_item,
            max_prompt_length,
            image_min_pixels,
            image_max_pixels,
            video_min_pixels,
            video_max_pixels,
            video_fps,
            video_max_frames,
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_raw_prompt_ids.append(raw_prompt_ids)
        all_raw_prompt.append(raw_prompt)
        all_multi_modal.append(mm_item if mm_item is not None else {})

    tensors = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "position_ids": torch.stack(all_position_ids),
    }
    non_tensors = {
        "raw_prompt_ids": np.array(all_raw_prompt_ids, dtype=object),
        "raw_prompt": np.array(all_raw_prompt, dtype=object),
        "multi_modal_data": np.array(all_multi_modal, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def construct_teacher_prompts(
    questions: list[Any] | np.ndarray,
    hints: list[list[str]] | np.ndarray,
    gt_answers: list[Any] | np.ndarray,
    p_drop: float,
    tokenizer: Any,
    processor: Any,
    reference_trajectories: Optional[list[Any] | np.ndarray] = None,
    student_trajectories: Optional[list[Any] | np.ndarray] = None,
    verifier_feedbacks: Optional[list[Any] | np.ndarray] = None,
    think_diagnoses: Optional[list[Any] | np.ndarray] = None,
    answer_diagnoses: Optional[list[Any] | np.ndarray] = None,
    template_path: Optional[str] = None,
    template_text: Optional[str] = None,
    format_prompt_path: Optional[str] = None,
    max_prompt_length: int = 4096,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
    problem_types: Optional[list[Any] | np.ndarray] = None,
    data_types: Optional[list[Any] | np.ndarray] = None,
) -> DataProto:
    del format_prompt_path
    if template_text is not None:
        hint_template = Template(template_text.strip())
    elif template_path:
        with open(template_path, encoding="utf-8") as f:
            hint_template = Template(f.read().strip())
    else:
        hint_template = Template(
            "{{ question }}\n\n[Hint from a reasoning navigator]\n{{ hint }}\n\n"
            "{% if gt_answer %}[Reference Answer]\n{{ gt_answer }}\n\n{% endif %}"
            "Using the above hint{% if gt_answer %} and reference answer{% endif %}, "
            "solve the problem step by step."
        )

    batch_size = len(questions)
    K = len(hints[0]) if batch_size > 0 else 0

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []
    all_raw_prompt = []
    all_multi_modal = []

    for i in range(batch_size):
        question = questions[i] if isinstance(questions[i], str) else str(questions[i])
        gt_answer = gt_answers[i] if isinstance(gt_answers[i], str) else str(gt_answers[i])
        reference_trajectory = ""
        if reference_trajectories is not None:
            reference_trajectory = (
                reference_trajectories[i]
                if isinstance(reference_trajectories[i], str)
                else str(reference_trajectories[i])
            )
        student_trajectory = ""
        if student_trajectories is not None:
            student_trajectory = (
                student_trajectories[i]
                if isinstance(student_trajectories[i], str)
                else str(student_trajectories[i])
            )
        verifier_feedback = ""
        if verifier_feedbacks is not None:
            verifier_feedback = (
                verifier_feedbacks[i]
                if isinstance(verifier_feedbacks[i], str)
                else str(verifier_feedbacks[i])
            )
        think_diagnosis = ""
        if think_diagnoses is not None:
            think_diagnosis = (
                think_diagnoses[i]
                if isinstance(think_diagnoses[i], str)
                else str(think_diagnoses[i])
            )
        answer_diagnosis = ""
        if answer_diagnoses is not None:
            answer_diagnosis = (
                answer_diagnoses[i]
                if isinstance(answer_diagnoses[i], str)
                else str(answer_diagnoses[i])
            )

        include_verifier_hint = _has_dynamic_verifier_hint(
            verifier_feedback, think_diagnosis, answer_diagnosis
        )

        include_gt = random.random() >= p_drop
        gt_for_prompt = gt_answer if include_gt else ""

        for k in range(K):
            hint = hints[i][k]
            problem_type = "safety"
            if problem_types is not None:
                problem_type = (
                    problem_types[i]
                    if isinstance(problem_types[i], str)
                    else str(problem_types[i])
                )
            data_type = ""
            if data_types is not None:
                data_type = data_types[i] if isinstance(data_types[i], str) else str(data_types[i])
            prompt_text = hint_template.render(
                question=question,
                problem=question,
                hint=hint,
                gt_answer=gt_for_prompt,
                reference_trajectory=reference_trajectory,
                student_trajectory=student_trajectory,
                verifier_feedback=verifier_feedback,
                think_diagnosis=think_diagnosis,
                answer_diagnosis=answer_diagnosis,
                include_verifier_hint=include_verifier_hint,
                problem_type=problem_type,
                data_type=data_type,
            )
            mm_item = multi_modal_data[i] if multi_modal_data is not None else None
            messages = _build_messages_from_prompt_text(prompt_text, mm_item)
            if processor is not None:
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))
            else:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))

            input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = _tokenize_prompt(
                prompt,
                tokenizer,
                processor,
                mm_item,
                max_prompt_length,
                image_min_pixels,
                image_max_pixels,
                video_min_pixels,
                video_max_pixels,
                video_fps,
                video_max_frames,
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_raw_prompt_ids.append(raw_prompt_ids)
            all_raw_prompt.append(raw_prompt)
            all_multi_modal.append(mm_item if mm_item is not None else {})

    tensors = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "position_ids": torch.stack(all_position_ids),
    }
    non_tensors = {
        "raw_prompt_ids": np.array(all_raw_prompt_ids, dtype=object),
        "raw_prompt": np.array(all_raw_prompt, dtype=object),
        "multi_modal_data": np.array(all_multi_modal, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def construct_sdpo_teacher_prompts(
    questions: list[Any] | np.ndarray,
    correct_solutions: list[str] | np.ndarray,
    incorrect_solutions: list[str] | np.ndarray,
    tokenizer: Any,
    processor: Any,
    template_path: Optional[str] = None,
    template_text: Optional[str] = None,
    max_prompt_length: int = 4096,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    if template_text is not None:
        teacher_template = Template(template_text.strip())
    elif template_path:
        with open(template_path, encoding="utf-8") as f:
            teacher_template = Template(f.read().strip())
    else:
        teacher_template = Template(
            "{{ question | trim }}\n\n"
            "{% if correct_solution %}Correct solution:\n{{ correct_solution | trim }}\n\n{% endif %}"
            "{% if incorrect_solution %}"
            "The following is feedback from your unsuccessful earlier attempt:\n"
            "{{ incorrect_solution | trim }}\n\n"
            "{% endif %}"
            "Correctly solve the original question."
        )

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []
    all_raw_prompt = []
    all_multi_modal = []

    for i in range(len(questions)):
        question = questions[i] if isinstance(questions[i], str) else str(questions[i])
        correct_solution = (
            correct_solutions[i] if isinstance(correct_solutions[i], str) else str(correct_solutions[i])
        )
        incorrect_solution = (
            incorrect_solutions[i] if isinstance(incorrect_solutions[i], str) else str(incorrect_solutions[i])
        )

        prompt_text = teacher_template.render(
            question=question,
            correct_solution=correct_solution,
            incorrect_solution=incorrect_solution,
        )
        mm_item = multi_modal_data[i] if multi_modal_data is not None else None
        messages = _build_messages_from_prompt_text(prompt_text, mm_item)
        if processor is not None:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))
        else:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))

        input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = _tokenize_prompt(
            prompt,
            tokenizer,
            processor,
            mm_item,
            max_prompt_length,
            image_min_pixels,
            image_max_pixels,
            video_min_pixels,
            video_max_pixels,
            video_fps,
            video_max_frames,
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_raw_prompt_ids.append(raw_prompt_ids)
        all_raw_prompt.append(raw_prompt)
        all_multi_modal.append(mm_item if mm_item is not None else {})

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask),
            "position_ids": torch.stack(all_position_ids),
        },
        non_tensors={
            "raw_prompt_ids": np.array(all_raw_prompt_ids, dtype=object),
            "raw_prompt": np.array(all_raw_prompt, dtype=object),
            "multi_modal_data": np.array(all_multi_modal, dtype=object),
        },
    )


def decode_navigator_hints(nav_output: DataProto, tokenizer: Any) -> list[list[str]]:
    responses = nav_output.batch["responses"]
    total = responses.shape[0]

    decoded = []
    for i in range(total):
        text = tokenizer.decode(responses[i], skip_special_tokens=True).strip()
        decoded.append(text)

    K = nav_output.meta_info.get("n", 1)
    B = total // K

    hints = []
    for b in range(B):
        hints_for_question = []
        for k in range(K):
            hints_for_question.append(decoded[b * K + k])
        hints.append(hints_for_question)

    return hints


def construct_opsd_forward_batch(
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    student_position_ids: torch.Tensor,
    student_responses: torch.Tensor,
    student_response_mask: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_position_ids: torch.Tensor,
    teacher_responses: torch.Tensor,
    teacher_response_mask: torch.Tensor,
    teacher_rewards: torch.Tensor,
    K: int,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    del student_input_ids, student_attention_mask, student_position_ids, tokenizer, max_seq_length
    B = student_responses.shape[0]
    student_resp_len = student_responses.shape[1]
    teacher_resp_len = teacher_responses.shape[1]
    del B, student_resp_len, teacher_resp_len

    student_responses_rep = student_responses.repeat_interleave(K, dim=0)
    student_response_mask_rep = student_response_mask.repeat_interleave(K, dim=0)

    valid_mask = teacher_rewards > 0
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    result = {
        "on_policy_student_responses": student_responses_rep,
        "on_policy_student_response_mask": student_response_mask_rep,
        "on_policy_teacher_input_ids": teacher_input_ids,
        "on_policy_teacher_attention_mask": teacher_attention_mask,
        "on_policy_teacher_position_ids": teacher_position_ids,
        "off_policy_teacher_responses": teacher_responses[valid_indices] if len(valid_indices) > 0 else teacher_responses[:0],
        "off_policy_teacher_response_mask": (
            teacher_response_mask[valid_indices] if len(valid_indices) > 0 else teacher_response_mask[:0]
        ),
        "off_policy_teacher_input_ids": teacher_input_ids[valid_indices] if len(valid_indices) > 0 else teacher_input_ids[:0],
        "off_policy_teacher_attention_mask": (
            teacher_attention_mask[valid_indices] if len(valid_indices) > 0 else teacher_attention_mask[:0]
        ),
        "off_policy_teacher_position_ids": (
            teacher_position_ids[valid_indices] if len(valid_indices) > 0 else teacher_position_ids[:0]
        ),
        "valid_hint_indices": valid_indices,
        "valid_hint_mask": valid_mask,
        "teacher_rewards": teacher_rewards,
        "K": K,
        "B": student_responses.shape[0],
    }
    return result
