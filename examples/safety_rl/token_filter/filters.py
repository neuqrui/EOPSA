#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EOPSA Selective Distillation — token filter (official).

Paper mapping
-------------
Safety-critical set ``K = {Pivot, Intent, Risk}`` is realized as taxonomy labels
``pivot``, ``intent``, ``risk_wo_same`` via the rubric classifier
(``filters_qwen_rubric.py``).

Only ``taxonomy_keep`` + action ``drop`` is supported in the official release:
main distillation trains tokens in ``K``; all other tokens are dropped.

``risk_wo_same`` (default Risk): teacher top-1 is in the risk lexicon and differs
from the student token — i.e. a genuine safety-vocabulary shift. Same-surface
risk hits fall into ``same`` and are not trained by default.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


def _load_sibling_module(filename: str):
    module_name = f"{__name__}_{filename.replace('.', '_')}"
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sibling token filter module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_QWEN_RUBRIC = _load_sibling_module("filters_qwen_rubric.py")

# Official classifier: rubric-extracted Qwen lexicons only.
_CLASSIFIERS = {
    "qwen_rubric": _QWEN_RUBRIC.classify_token_qwen_rubric,
}

# Paper K = {Pivot, Intent, Risk} ↔ risk_wo_same (see module docstring).
TAXONOMY_KEEP_CATEGORIES = {"pivot", "intent", "risk_wo_same"}
TAXONOMY_KEEP_CATEGORIES_QWEN_RUBRIC = _QWEN_RUBRIC.TAXONOMY_KEEP_CATEGORIES
TAXONOMY_DROP_CATEGORIES_QWEN_RUBRIC = _QWEN_RUBRIC.TAXONOMY_DROP_CATEGORIES

classify_token_qwen_rubric = _QWEN_RUBRIC.classify_token_qwen_rubric
classify_token_qwen = _QWEN_RUBRIC.classify_token_qwen_rubric


@dataclass
class TokenFilterContext:
    teacher_logits: torch.Tensor
    student_logits: torch.Tensor
    base_mask: torch.Tensor
    responses: torch.Tensor
    token_div: torch.Tensor
    top_n: int = 16
    decode_id: Optional[Callable[[int], str]] = None
    action: str = "drop"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenFilterResult:
    keep_mask: torch.Tensor
    action_mask: torch.Tensor
    stats: dict[str, float]
    mode: str
    action: str
    use_alt_kl_mix: bool = False

    @property
    def same_mask(self) -> torch.Tensor:
        return self.action_mask

    @property
    def use_sampled_kl_mix(self) -> bool:
        return self.use_alt_kl_mix


SelectFn = Callable[[TokenFilterContext], torch.Tensor]
SELECT_REGISTRY: dict[str, SelectFn] = {}


def register(name: str) -> Callable[[SelectFn], SelectFn]:
    key = name.strip().lower()

    def deco(fn: SelectFn) -> SelectFn:
        SELECT_REGISTRY[key] = fn
        return fn

    return deco


def list_filters() -> list[str]:
    return sorted(SELECT_REGISTRY.keys())


list_modes = list_filters
VALID_ACTIONS = ("drop",)


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def _build_stats(
    *,
    base_mask: torch.Tensor,
    main_mask: torch.Tensor,
    action_mask: torch.Tensor,
    token_div: torch.Tensor,
    action: str,
) -> dict[str, float]:
    del action_mask, action
    tokens_before = float(base_mask.sum().item())
    tokens_main = float(main_mask.sum().item())
    dropped = base_mask * (1.0 - main_mask)
    tokens_dropped = float(dropped.sum().item())
    total_kl = float((token_div * base_mask).sum().item()) + 1e-8
    return {
        "tokens_before": tokens_before,
        "tokens_main": tokens_main,
        "tokens_alt": 0.0,
        "tokens_dropped": tokens_dropped,
        "tokens_trained": tokens_main,
        "token_keep_ratio": tokens_main / max(tokens_before, 1.0),
        "token_train_ratio": tokens_main / max(tokens_before, 1.0),
        "filtered_kl_share": float((token_div * dropped).sum().item()) / total_kl,
        "kept_kl_share": float((token_div * main_mask).sum().item()) / total_kl,
    }


def _apply_action(
    *,
    base_mask: torch.Tensor,
    special_mask: torch.Tensor,
    token_div: torch.Tensor,
    mode: str,
    action: str,
) -> TokenFilterResult:
    action = (action or "drop").strip().lower()
    if action != "drop":
        raise ValueError(
            f"Official EOPSA token filter only supports action='drop', got {action!r}."
        )
    # special = not-in-K; main distillation trains ~special ∩ base = K
    main = base_mask * (1.0 - special_mask)
    stats = _build_stats(
        base_mask=base_mask,
        main_mask=main,
        action_mask=special_mask,
        token_div=token_div,
        action=action,
    )
    return TokenFilterResult(
        keep_mask=main,
        action_mask=special_mask,
        stats=stats,
        mode=mode,
        action=action,
        use_alt_kl_mix=False,
    )


def _normalize_taxonomy_category(name: str) -> str:
    c = str(name).strip().lower()
    # Paper "Risk" ↔ implementation label risk_wo_same
    if c == "risk":
        return "risk_wo_same"
    if c.endswith("_ds"):
        return c[:-3]
    return c


def _expand_match_cats(match_cats: set[str]) -> set[str]:
    out = {_normalize_taxonomy_category(c) for c in match_cats}
    # If user asks for risk_lexicon (all risk surface), include disagree subset.
    if "risk_lexicon" in out:
        out.add("risk_wo_same")
    if "same" in out:
        out.add("residual_same")
    if "residual_same" in out:
        out.add("same")
    return out


def _resolve_classifier(extra: Optional[dict[str, Any]] = None):
    extra = extra or {}
    classifier = str(
        extra.get("classifier") or extra.get("token_filter_classifier") or "qwen_rubric"
    ).strip().lower()
    # Accept legacy aliases by routing everything to the official rubric classifier.
    if classifier in {
        "qwen_rubric",
        "rubric",
        "qwen",
        "legacy",
        "eopsa",
        "qwen_eopsa",
        "qwen_ori",
        "original",
        "ds",
        "r1",
    }:
        return _QWEN_RUBRIC.classify_token_qwen_rubric
    raise ValueError(
        f"Unknown token_filter_classifier={classifier!r}. "
        "Official release only supports 'qwen_rubric'."
    )


def classify_token(
    stu: str,
    tea: str,
    *,
    stu_prob: Optional[float] = None,
    tea_prob: Optional[float] = None,
    forward_kl: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    del stu_prob, tea_prob, forward_kl
    return _resolve_classifier(extra or {})(stu, tea)


def _resolve_student_token_ids(ctx: TokenFilterContext) -> torch.Tensor:
    raw = ctx.extra.get("student_token_source")
    if raw is None:
        raw = ctx.extra.get("token_filter_student_source")
    source = str(raw or "sampled").strip().lower()
    if source in {"sampled", "response", "sample"}:
        if ctx.responses is None:
            raise RuntimeError(
                "taxonomy student_token_source=sampled requires ctx.responses"
            )
        return ctx.responses.to(device=ctx.base_mask.device, dtype=torch.long)
    if source in {"top1", "argmax", "logit_top1"}:
        return ctx.student_logits.argmax(dim=-1)
    raise ValueError(
        f"Unknown student_token_source={source!r}. Expected sampled|top1."
    )


TAXONOMY_METRIC_CATEGORIES = (
    "pivot",
    "intent",
    "risk_wo_same",
    "risk_lexicon",
    "same",
    "function",
    "other",
)


def _category_mask(ctx: TokenFilterContext, match_cats: set[str]) -> torch.Tensor:
    if ctx.decode_id is None:
        raise RuntimeError("taxonomy_keep requires decode_id (tokenizer).")
    match_cats = _expand_match_cats(match_cats)
    with torch.no_grad():
        tea_top1 = ctx.teacher_logits.argmax(dim=-1)
        stu_ids = _resolve_student_token_ids(ctx)
        tea_p = _softmax_probs(ctx.teacher_logits).gather(-1, tea_top1.unsqueeze(-1)).squeeze(-1)
        stu_p = (
            _softmax_probs(ctx.student_logits)
            .gather(-1, stu_ids.unsqueeze(-1))
            .squeeze(-1)
        )

        out = torch.zeros_like(ctx.base_mask)
        cat_ids = torch.full(
            ctx.base_mask.shape, -1, device=ctx.base_mask.device, dtype=torch.long
        )
        name_to_id: dict[str, int] = {}
        id_to_name: list[str] = []
        counts: dict[str, float] = {}
        id_cache: dict[int, str] = {}

        def _dec(i: int) -> str:
            if i not in id_cache:
                id_cache[i] = ctx.decode_id(i)
            return id_cache[i]

        bsz = ctx.base_mask.shape[0]
        for b in range(bsz):
            valid = torch.nonzero(ctx.base_mask[b] > 0, as_tuple=False).squeeze(-1)
            for t in valid.tolist():
                sid = int(stu_ids[b, t].item())
                tid = int(tea_top1[b, t].item())
                cat = classify_token(
                    _dec(sid),
                    _dec(tid),
                    stu_prob=float(stu_p[b, t].item()),
                    tea_prob=float(tea_p[b, t].item()),
                    forward_kl=float(ctx.token_div[b, t].item()),
                    extra=ctx.extra,
                )
                cat = str(cat).strip().lower() or "other"
                if cat not in name_to_id:
                    name_to_id[cat] = len(id_to_name)
                    id_to_name.append(cat)
                cid = name_to_id[cat]
                cat_ids[b, t] = cid
                counts[cat] = counts.get(cat, 0.0) + 1.0
                if cat in match_cats:
                    out[b, t] = 1.0

        ctx.extra["_taxonomy_cat_ids"] = cat_ids
        ctx.extra["_taxonomy_id_to_name"] = id_to_name
        ctx.extra["_taxonomy_cat_counts"] = counts
    return out


def _attach_taxonomy_category_stats(
    result: TokenFilterResult,
    ctx: TokenFilterContext,
) -> TokenFilterResult:
    counts = ctx.extra.pop("_taxonomy_cat_counts", None)
    cat_ids = ctx.extra.pop("_taxonomy_cat_ids", None)
    id_to_name = ctx.extra.pop("_taxonomy_id_to_name", None)
    if not counts:
        return result

    for name in TAXONOMY_METRIC_CATEGORIES:
        result.stats[f"tokens_cat_{name}"] = float(counts.get(name, 0.0))
    for name, n in counts.items():
        key = f"tokens_cat_{name}"
        if key not in result.stats:
            result.stats[key] = float(n)

    if cat_ids is not None and id_to_name is not None:
        keep = result.keep_mask
        dtype = keep.dtype
        main_counts: dict[str, float] = {n: 0.0 for n in TAXONOMY_METRIC_CATEGORIES}
        for i, name in enumerate(id_to_name):
            main_n = float(((cat_ids == i).to(dtype=dtype) * keep).sum().item())
            main_counts[name] = main_n
        for name in TAXONOMY_METRIC_CATEGORIES:
            result.stats[f"tokens_main_cat_{name}"] = float(main_counts.get(name, 0.0))
        for name, n in main_counts.items():
            key = f"tokens_main_cat_{name}"
            if key not in result.stats:
                result.stats[key] = float(n)
    return result


@register("taxonomy_keep")
def select_taxonomy_keep(ctx: TokenFilterContext) -> torch.Tensor:
    """S = tokens NOT in keep_categories (K); main loss trains K."""
    cats = ctx.extra.get("keep_categories") or TAXONOMY_KEEP_CATEGORIES
    keep_cats = {_normalize_taxonomy_category(c) for c in cats}
    in_keep = _category_mask(ctx, keep_cats)
    return ctx.base_mask * (1.0 - in_keep)


def apply_filter(
    mode: str,
    ctx: TokenFilterContext,
    action: Optional[str] = None,
) -> TokenFilterResult:
    key = (mode or "taxonomy_keep").strip().lower()
    if key != "taxonomy_keep":
        raise ValueError(
            f"Official EOPSA only supports token_filter_mode='taxonomy_keep', got {mode!r}."
        )
    act = (action if action is not None else getattr(ctx, "action", None) or "drop")
    act = str(act).strip().lower()
    with torch.no_grad():
        special = SELECT_REGISTRY[key](ctx)
        special = (special * ctx.base_mask).to(dtype=ctx.base_mask.dtype)
        result = _apply_action(
            base_mask=ctx.base_mask,
            special_mask=special,
            token_div=ctx.token_div,
            mode=key,
            action=act,
        )
        return _attach_taxonomy_category_stats(result, ctx)
