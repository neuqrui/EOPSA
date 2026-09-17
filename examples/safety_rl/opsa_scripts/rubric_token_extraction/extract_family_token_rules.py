#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubric-guided family-specific token rule extraction (standalone).

Pipeline
--------
1) N harmful samples → student rollout (max 256 tokens by default).
2) Top-16 forward-KL positions per sample → (student_top1, teacher_top1) pairs.
3) **One LLM call per sample** (gpt-4o-mini): rubric + query + response + 16 pairs.
4) ``consistent`` pre-labeled by surface-equal rule; LLM labels pivot/intent/risk/function/null.
5) Aggregate counts → PIVOT_STU/TEA, INTENT_STU/TEA, RISK, FUNC, _INTENT_PAIRS.
6) **One LLM call on the aggregated lexicons** (problem background + rubrics):
   drop query-specific / mislabeled tokens and intent pairs.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python extract_family_token_rules.py --tag qwen3-1.7b
  STEPS=judge,aggregate,refine python extract_family_token_rules.py --tag 200all
  STEPS=refine python extract_family_token_rules.py --tag 200all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OPSA_DIR = SCRIPT_DIR.parent
SAFETY_RL_DIR = OPSA_DIR.parent
PROJECT_DIR = SAFETY_RL_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SAFETY_RL_DIR))
sys.path.insert(0, str(OPSA_DIR))

from analyze_prefix_token_kl import (  # noqa: E402
    apply_chat_template,
    render_teacher_prompt,
    response_logits,
    run_student_rollouts,
    teacher_topk_divergences,
)
from reward.llm_api import call_openai_chat_async  # noqa: E402
from rubric_definitions import (  # noqa: E402
    CATEGORY_ALIASES,
    EXTRACTABLE_CATEGORIES,
    build_lexicon_audit_prompt,
    build_sample_batch_prompt,
    default_pair_judgment,
)

DEFAULT_MODEL = __import__("os").environ.get("MODEL_PATH", "Qwen/Qwen3-1.7B")
DEFAULT_DATA = (
    SAFETY_RL_DIR / "datasets" / "safety_ds_safechain_dsr100_h4400_b2200" / "train.jsonl"
)
DEFAULT_TEACHER_TEMPLATE = SAFETY_RL_DIR / "format_prompt" / "safety_teacher.jinja"
DEFAULT_OUT = SCRIPT_DIR / "outputs"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def _norm_token(s: str) -> str:
    s = (s or "").strip().lower().replace("\u2019", "'")
    if re.fullmatch(r"[\W_]+", s or ""):
        return s
    s2 = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s2 if s2 else s


def decode_tok(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], skip_special_tokens=False)


def load_harmful_samples(path: Path, n: int, seed: int, *, split: str) -> list[dict[str, Any]]:
    if split == "holdout":
        val_path = path.parent / "val.jsonl"
        if val_path.is_file():
            path = val_path
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("data_type") != "safety":
                continue
            rows.append(row)
    rng = np.random.default_rng(seed)
    if n > 0 and n < len(rows):
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def annotate_pairs_with_prelabels(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in pairs:
        row = dict(p)
        ns = row.get("student_norm") or _norm_token(row["student_top1"])
        nt = row.get("teacher_norm") or _norm_token(row["teacher_top1"])
        row["student_norm"] = ns
        row["teacher_norm"] = nt
        row["pre_category"] = "consistent" if ns == nt else None
        out.append(row)
    return out


@torch.no_grad()
def collect_top_kl_pairs(
    model_path: str,
    records: list[dict[str, Any]],
    teacher_template: Path,
    *,
    enable_thinking: bool,
    max_tokens: int,
    topk: int,
    top_kl_per_sample: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[KL] Loading {model_path} dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    samples_out: list[dict[str, Any]] = []
    t0 = time.time()

    for rec_idx, rec in enumerate(records):
        resp_ids = list(rec["student_token_ids"])[:max_tokens]
        if not resp_ids:
            continue
        q = str(rec["problem"])
        hint = str(rec.get("safe_reference") or "")
        student_response = str(rec.get("student_response") or "")

        stu_prompt = apply_chat_template(q, tokenizer, enable_thinking=enable_thinking, tokenize=True)
        tea_raw = render_teacher_prompt(teacher_template, q, hint)
        tea_prompt = apply_chat_template(tea_raw, tokenizer, enable_thinking=enable_thinking, tokenize=True)
        if not isinstance(stu_prompt, list):
            stu_prompt = list(stu_prompt)
        if not isinstance(tea_prompt, list):
            tea_prompt = list(tea_prompt)

        stu_logits = response_logits(model, stu_prompt, resp_ids, device)
        tea_logits = response_logits(model, tea_prompt, resp_ids, device)
        t_len = min(stu_logits.size(0), tea_logits.size(0), len(resp_ids))
        if t_len <= 0:
            continue

        div = teacher_topk_divergences(tea_logits[:t_len], stu_logits[:t_len], k=topk)
        fwd = div["forward_kl"].float().cpu().numpy()
        stu_top1 = stu_logits[:t_len].argmax(dim=-1).cpu().numpy()
        tea_top1 = tea_logits[:t_len].argmax(dim=-1).cpu().numpy()

        order = np.argsort(-fwd)[: min(top_kl_per_sample, t_len)]
        top_rows: list[dict[str, Any]] = []
        for rank, pos in enumerate(order):
            pos = int(pos)
            stu_tok = decode_tok(tokenizer, int(stu_top1[pos]))
            tea_tok = decode_tok(tokenizer, int(tea_top1[pos]))
            top_rows.append({
                "rank_in_sample": rank + 1,
                "position": pos,
                "student_top1": stu_tok,
                "teacher_top1": tea_tok,
                "student_norm": _norm_token(stu_tok),
                "teacher_norm": _norm_token(tea_tok),
                "forward_kl": float(fwd[pos]),
            })

        samples_out.append({
            "sample_idx": rec_idx,
            "problem": q,
            "student_response": student_response,
            "safe_reference": hint,
            "n_response_tokens": t_len,
            "top_kl_tokens": annotate_pairs_with_prelabels(top_rows),
        })

        if (rec_idx + 1) % 20 == 0:
            print(f"[KL] processed {rec_idx + 1}/{len(records)} samples")

    payload = {
        "model_path": model_path,
        "teacher_template": str(teacher_template),
        "max_tokens": max_tokens,
        "distillation_topk": topk,
        "top_kl_per_sample": top_kl_per_sample,
        "n_samples": len(samples_out),
        "samples": samples_out,
        "elapsed_sec": time.time() - t0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_llm_category(raw: Any) -> str:
    if raw is None:
        return "other"
    cat = str(raw).strip().lower()
    if cat in ("null", "none", "", "other"):
        return "other"
    if cat in CATEGORY_ALIASES:
        cat = CATEGORY_ALIASES[cat]
    if cat not in EXTRACTABLE_CATEGORIES:
        return "other"
    return cat


def _allowed_pair_norms(stu: str, tea: str) -> set[str]:
    return {_norm_token(stu), _norm_token(tea)}


def _pick_tokens_from_pair(
    raw_tokens: Any,
    stu: str,
    tea: str,
) -> list[str]:
    """Keep only tokens that match student_top1 or teacher_top1 (normalized)."""
    allowed = _allowed_pair_norms(stu, tea)
    out: list[str] = []
    if not isinstance(raw_tokens, list):
        return out
    for t in raw_tokens:
        nn = _norm_token(str(t))
        if nn and nn in allowed and nn not in out:
            out.append(nn)
    return out


def parse_sample_batch_response(raw: str, pairs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return rank -> {category, pivot_tokens, risk_tokens, function_tokens} from LLM JSON."""
    obj = _parse_json_obj(raw)
    rank_to_label: dict[int, dict[str, Any]] = {}
    if not obj:
        return rank_to_label
    labels = obj.get("pair_labels") or obj.get("pairs") or obj.get("classifications") or []
    if isinstance(labels, dict):
        labels = [{"rank": int(k), **(v if isinstance(v, dict) else {"category": v})} for k, v in labels.items()]

    pair_by_rank = {int(p["rank_in_sample"]): p for p in pairs}
    for item in labels:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank", item.get("rank_in_sample", item.get("idx")))
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            continue
        cat = _normalize_llm_category(item.get("category"))
        p = pair_by_rank.get(rank, {})
        stu = str(p.get("student_top1", ""))
        tea = str(p.get("teacher_top1", ""))
        pivot_tokens = _pick_tokens_from_pair(item.get("pivot_tokens"), stu, tea)
        risk_tokens = _pick_tokens_from_pair(item.get("risk_tokens"), stu, tea)
        function_tokens = _pick_tokens_from_pair(item.get("function_tokens"), stu, tea)
        if cat == "pivot" and not pivot_tokens:
            cat = "other"
        if cat == "risk" and not risk_tokens:
            cat = "other"
        if cat == "function" and not function_tokens:
            cat = "other"
        rank_to_label[rank] = {
            "category": cat,
            "pivot_tokens": pivot_tokens,
            "risk_tokens": risk_tokens,
            "function_tokens": function_tokens,
        }
    return rank_to_label


def heuristic_pair_label(stu: str, tea: str) -> dict[str, Any]:
    """Offline label + explicit risk/function token lists."""
    ns, nt = _norm_token(stu), _norm_token(tea)
    out = default_pair_judgment("other")
    if ns == nt:
        out["category"] = "consistent"
        out["confidence"] = 1.0
        return out

    pivot_tea = {"but", "however", "wait", "which", "maybe", "yet", "still"}
    pivot_stu = {"the", "i", "let", "they", "first", "also", "so", "wait", "my", "now", "okay", "alright"}
    risk_lex = {
        "harmful", "dangerous", "illegal", "unethical", "crime", "harm", "refuse", "reject", "avoid",
        "serious", "safety", "safe", "guidelines", "sensitive", "ethical", "risk", "risks",
        "threat", "violence", "weapon", "bomb",
    }
    func_lex = {
        ",", ".", ":", ";", "!", "?", "a", "an", "the", "to", "of", "in", "on", "for",
        "and", "or", "as", "by", "is", "are", "be", "that", "this", "with", "at", "from", "it", "'s", "n't",
    }
    intent_tea = {
        "recognize", "recognizing", "acknowledge", "core", "request", "intent", "asking", "user",
        "check", "consider", "considering", "remember",
    }
    intent_stu = {
        "understand", "outlining", "requirements", "key", "context", "example", "wants", "working",
        "needs", "goal", "main", "start", "think", "presented", "points", "use", "make", "should", "likely",
    }

    if nt in pivot_tea and ns != nt:
        pivot_tokens = [t for t in (ns, nt) if t in pivot_stu or t in pivot_tea]
        if not pivot_tokens:
            pivot_tokens = [nt]
        out.update(category="pivot", confidence=0.85, pivot_tokens=pivot_tokens)
        return out
    if ns in pivot_stu and nt in pivot_tea:
        pivot_tokens = [t for t in (ns, nt) if t in pivot_stu or t in pivot_tea]
        out.update(category="pivot", confidence=0.85, pivot_tokens=pivot_tokens)
        return out

    risk_tokens = [t for t in (ns, nt) if t in risk_lex]
    if risk_tokens:
        out.update(category="risk", confidence=0.9, risk_tokens=risk_tokens)
        return out

    if nt in intent_tea or ns in intent_stu:
        out.update(category="intent", confidence=0.75)
        return out

    func_tokens: list[str] = []
    if ns in func_lex:
        func_tokens.append(ns)
    if nt in func_lex and nt not in func_tokens:
        func_tokens.append(nt)
    if re.fullmatch(r"[\W_]+", (stu or "").strip()) and ns and ns not in func_tokens:
        func_tokens.append(ns)
    if re.fullmatch(r"[\W_]+", (tea or "").strip()) and nt and nt not in func_tokens:
        func_tokens.append(nt)
    if func_tokens:
        out.update(category="function", confidence=0.8, function_tokens=func_tokens)
        return out

    out["confidence"] = 0.5
    return out


def merge_sample_judgments(
    pairs: list[dict[str, Any]],
    rank_to_label: dict[int, dict[str, Any]],
    *,
    judge_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in pairs:
        rank = int(p["rank_in_sample"])
        if p.get("pre_category") == "consistent":
            judgment = default_pair_judgment("consistent")
            source = "rule"
        elif rank in rank_to_label:
            label = rank_to_label[rank]
            judgment = default_pair_judgment(str(label.get("category", "other")))
            judgment["pivot_tokens"] = list(label.get("pivot_tokens") or [])
            judgment["risk_tokens"] = list(label.get("risk_tokens") or [])
            judgment["function_tokens"] = list(label.get("function_tokens") or [])
            judgment["confidence"] = 0.9
            source = judge_source
        elif judge_source in ("heuristic", "heuristic_fallback"):
            judgment = heuristic_pair_label(p["student_top1"], p["teacher_top1"])
            source = judge_source
        else:
            judgment = default_pair_judgment("other")
            source = judge_source
        cat = str(judgment.get("category", "other"))
        rows.append({
            **p,
            "judgment": judgment,
            "judge_source": source,
            "runtime_category": "same" if cat == "consistent" else cat,
        })
    return rows


async def judge_samples_async(
    samples: list[dict[str, Any]],
    *,
    judge_mode: str,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    cache_path: Path,
) -> list[dict[str, Any]]:
    if cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("n_samples") == len(samples):
            print(f"[Judge] reuse cache {cache_path}")
            return cached["sample_judgments"]

    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[dict[str, Any] | None] = [None] * len(samples)
    done = 0

    async def _one(i: int, sample: dict[str, Any]) -> None:
        nonlocal done
        pairs = annotate_pairs_with_prelabels(sample["top_kl_tokens"])
        if judge_mode == "heuristic":
            pair_rows = merge_sample_judgments(pairs, {}, judge_source="heuristic")
            results[i] = {
                "sample_idx": sample.get("sample_idx", i),
                "problem": sample.get("problem", ""),
                "pair_judgments": pair_rows,
            }
            done += 1
            if done % 20 == 0 or done == len(samples):
                print(f"[Judge/heuristic] {done}/{len(samples)} samples")
            return

        prompt = build_sample_batch_prompt(
            query=str(sample.get("problem", "")),
            student_response=str(sample.get("student_response", "")),
            pairs=pairs,
        )
        async with sem:
            try:
                raw = await call_openai_chat_async(
                    [{"role": "user", "content": prompt}],
                    model_name=judge_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                rank_to_label = parse_sample_batch_response(raw, pairs)
                pair_rows = merge_sample_judgments(pairs, rank_to_label, judge_source="llm")
                err = None
            except Exception as e:
                raw = ""
                pair_rows = merge_sample_judgments(pairs, {}, judge_source="heuristic_fallback")
                err = str(e)

        results[i] = {
            "sample_idx": sample.get("sample_idx", i),
            "problem": sample.get("problem", ""),
            "pair_judgments": pair_rows,
            "llm_raw_preview": (raw or "")[:3000],
            "llm_error": err,
        }
        done += 1
        if done % 10 == 0 or done == len(samples):
            print(f"[Judge/llm] {done}/{len(samples)} samples")

    await asyncio.gather(*[_one(i, s) for i, s in enumerate(samples)])

    sample_judgments = [r for r in results if r is not None]
    payload = {
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "judge_api": "reward.llm_api.call_openai_chat_async",
        "n_samples": len(samples),
        "sample_judgments": sample_judgments,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Judge] saved {cache_path}")
    return sample_judgments


def flatten_pair_judgments(sample_judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for s in sample_judgments:
        for row in s.get("pair_judgments", []):
            flat.append({**row, "count": 1})
    return flat


def apply_category_to_counters(
    judgment: dict[str, Any],
    ns: str,
    nt: str,
    *,
    pivot_stu: Counter[str],
    pivot_tea: Counter[str],
    intent_stu: Counter[str],
    intent_tea: Counter[str],
    intent_pairs: Counter[tuple[str, str]],
    risk: Counter[str],
    func: Counter[str],
    weight: int = 1,
) -> None:
    cat = str(judgment.get("category", "other"))
    if cat == "pivot":
        # Only count tokens LLM marked as pivot cues; map to STU/TEA by which side.
        for tok in judgment.get("pivot_tokens") or []:
            nn = _norm_token(str(tok)) if tok else ""
            if not nn:
                continue
            if nn == ns:
                pivot_stu[nn] += weight
            if nn == nt:
                pivot_tea[nn] += weight
    elif cat == "intent":
        if ns:
            intent_stu[ns] += weight
        if nt:
            intent_tea[nt] += weight
        if ns and nt and ns != nt:
            intent_pairs[(ns, nt)] += weight
    elif cat == "risk":
        for tok in judgment.get("risk_tokens") or []:
            if tok:
                risk[str(tok)] += weight
    elif cat == "function":
        for tok in judgment.get("function_tokens") or []:
            if tok:
                func[str(tok)] += weight


def aggregate_rules(
    pair_rows: list[dict[str, Any]],
    *,
    min_token_count: int,
    min_pair_count: int,
) -> dict[str, Any]:
    pivot_stu: Counter[str] = Counter()
    pivot_tea: Counter[str] = Counter()
    intent_stu: Counter[str] = Counter()
    intent_tea: Counter[str] = Counter()
    intent_pairs: Counter[tuple[str, str]] = Counter()
    risk: Counter[str] = Counter()
    func: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()

    for row in pair_rows:
        judgment = row.get("judgment", {})
        cat = str(judgment.get("category", "other"))
        cat_counts[cat] += int(row.get("count", 1))
        if cat not in EXTRACTABLE_CATEGORIES:
            continue
        ns = row.get("student_norm") or _norm_token(row["student_top1"])
        nt = row.get("teacher_norm") or _norm_token(row["teacher_top1"])
        apply_category_to_counters(
            judgment, ns, nt,
            pivot_stu=pivot_stu, pivot_tea=pivot_tea,
            intent_stu=intent_stu, intent_tea=intent_tea,
            intent_pairs=intent_pairs, risk=risk, func=func,
            weight=int(row.get("count", 1)),
        )

    def _pick(counter: Counter[str], min_c: int) -> list[str]:
        return sorted(tok for tok, c in counter.items() if c >= min_c and tok)

    def _pick_pairs(counter: Counter[tuple[str, str]], min_c: int) -> list[list[str]]:
        return sorted([list(p) for p, c in counter.items() if c >= min_c])

    return {
        "PIVOT_STU": _pick(pivot_stu, min_token_count),
        "PIVOT_TEA": _pick(pivot_tea, min_token_count),
        "INTENT_STU": _pick(intent_stu, min_token_count),
        "INTENT_TEA": _pick(intent_tea, min_token_count),
        "_INTENT_PAIRS": _pick_pairs(intent_pairs, min_pair_count),
        "RISK": _pick(risk, min_token_count),
        "FUNC": _pick(func, min_token_count),
        "TAXONOMY_KEEP_CATEGORIES": ["pivot", "intent", "risk_wo_same"],
        "TAXONOMY_DROP_CATEGORIES": ["same", "function", "risk_lexicon", "other"],
        "category_counts": dict(cat_counts),
        "thresholds": {
            "min_token_count": min_token_count,
            "min_pair_count": min_pair_count,
        },
    }


def render_filter_module(rules: dict[str, Any], *, tag: str, source_meta: dict[str, Any]) -> str:
    def fmt_set(name: str, items: list[str]) -> str:
        return f"{name} = {{{', '.join(repr(x) for x in items)}}}"

    pairs = rules.get("_INTENT_PAIRS", [])
    pairs_lines = ",\n    ".join(f"({repr(a)}, {repr(b)})" for a, b in pairs)
    meta_json = json.dumps(source_meta, ensure_ascii=False, indent=2)
    return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-extracted token taxonomy for {tag}."""

from __future__ import annotations

import re

{fmt_set("PIVOT_TEA", rules["PIVOT_TEA"])}
{fmt_set("PIVOT_STU", rules["PIVOT_STU"])}
{fmt_set("INTENT_TEA", rules["INTENT_TEA"])}
{fmt_set("INTENT_STU", rules["INTENT_STU"])}
{fmt_set("RISK", rules["RISK"])}
{fmt_set("FUNC", rules["FUNC"])}

TAXONOMY_KEEP_CATEGORIES = {set(rules["TAXONOMY_KEEP_CATEGORIES"])}
TAXONOMY_DROP_CATEGORIES = {set(rules["TAXONOMY_DROP_CATEGORIES"])}

_INTENT_PAIRS = {{
    {pairs_lines}
}}


def _norm_token(s: str) -> str:
    s = (s or "").strip().lower().replace("\\u2019", "'")
    if re.fullmatch(r"[\\\\W_]+", s or ""):
        return s
    s2 = re.sub(r"^[^\\\\w]+|[^\\\\w]+$", "", s)
    return s2 if s2 else s


def classify_token(stu: str, tea: str) -> str:
    ns, nt = _norm_token(stu), _norm_token(tea)
    if nt in PIVOT_TEA and ns != nt:
        return "pivot"
    if ns in PIVOT_STU and nt in PIVOT_TEA:
        return "pivot"
    if ns in RISK or nt in RISK:
        return "risk_lexicon" if ns == nt else "risk_wo_same"
    if ns == nt:
        return "same"
    if (ns, nt) in _INTENT_PAIRS:
        return "intent"
    if nt in INTENT_TEA or ns in INTENT_STU:
        return "intent"
    if ns in FUNC and nt in FUNC:
        return "function"
    if re.fullmatch(r"[\\\\W_]+", (stu or "").strip()) or re.fullmatch(r"[\\\\W_]+", (tea or "").strip()):
        return "function"
    return "other"


classify = classify_token
'''


def compare_with_qwen_eopsa(rules: dict[str, Any]) -> dict[str, Any]:
    try:
        from token_filter.filters_qwen_eopsa import (  # noqa: WPS433
            FUNC as REF_FUNC,
            INTENT_STU as REF_IS,
            INTENT_TEA as REF_IT,
            PIVOT_STU as REF_PS,
            PIVOT_TEA as REF_PT,
            RISK as REF_RISK,
            _INTENT_PAIRS as REF_PAIRS,
        )
    except ImportError:
        return {}

    def _cmp(new: list[str], ref: set[str]) -> dict[str, Any]:
        new_s = set(new)
        return {
            "new_n": len(new_s),
            "ref_n": len(ref),
            "overlap": sorted(new_s & ref),
            "only_new": sorted(new_s - ref),
            "only_ref": sorted(ref - new_s),
        }

    return {
        "PIVOT_STU": _cmp(rules["PIVOT_STU"], REF_PS),
        "PIVOT_TEA": _cmp(rules["PIVOT_TEA"], REF_PT),
        "INTENT_STU": _cmp(rules["INTENT_STU"], REF_IS),
        "INTENT_TEA": _cmp(rules["INTENT_TEA"], REF_IT),
        "RISK": _cmp(rules["RISK"], REF_RISK),
        "FUNC": _cmp(rules["FUNC"], REF_FUNC),
        "INTENT_PAIRS_overlap": sorted(set(map(tuple, rules["_INTENT_PAIRS"])) & set(REF_PAIRS)),
    }


_LEXICON_KEYS = (
    "PIVOT_STU",
    "PIVOT_TEA",
    "INTENT_STU",
    "INTENT_TEA",
    "RISK",
    "FUNC",
)


def _normalize_drop_pairs(raw: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = _norm_token(str(item[0])), _norm_token(str(item[1]))
            if a and b:
                out.append([a, b])
        elif isinstance(item, str) and "->" in item:
            left, right = item.split("->", 1)
            a, b = _norm_token(left), _norm_token(right)
            if a and b:
                out.append([a, b])
    return out


def parse_lexicon_audit_response(raw: str, rules: dict[str, Any]) -> dict[str, Any]:
    """Parse one-shot lexicon audit JSON into drop sets constrained to candidates."""
    obj = _parse_json_obj(raw) or {}
    drop_obj = obj.get("drop") or obj.get("drops") or obj
    if not isinstance(drop_obj, dict):
        drop_obj = {}

    drops: dict[str, Any] = {k: [] for k in _LEXICON_KEYS}
    drops["_INTENT_PAIRS"] = []

    for key in _LEXICON_KEYS:
        cand = {_norm_token(x) for x in (rules.get(key) or [])}
        raw_list = drop_obj.get(key) or []
        if not isinstance(raw_list, list):
            continue
        kept: list[str] = []
        for t in raw_list:
            nn = _norm_token(str(t))
            if nn and nn in cand and nn not in kept:
                kept.append(nn)
        drops[key] = kept

    cand_pairs = {
        (_norm_token(a), _norm_token(b))
        for a, b in (rules.get("_INTENT_PAIRS") or [])
        if a and b
    }
    pair_drops: list[list[str]] = []
    for a, b in _normalize_drop_pairs(drop_obj.get("_INTENT_PAIRS")):
        if (a, b) in cand_pairs and [a, b] not in pair_drops:
            pair_drops.append([a, b])
    drops["_INTENT_PAIRS"] = pair_drops

    return {
        "drop": drops,
        "rationale_brief": str(obj.get("rationale_brief") or obj.get("rationale") or ""),
        "raw_preview": (raw or "")[:4000],
    }


def apply_lexicon_drops(rules: dict[str, Any], drops: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied rules dict with drop tokens/pairs removed."""
    out = dict(rules)
    for key in _LEXICON_KEYS:
        drop_set = {_norm_token(x) for x in (drops.get(key) or [])}
        out[key] = [t for t in (rules.get(key) or []) if _norm_token(t) not in drop_set]

    drop_pairs = {
        (_norm_token(a), _norm_token(b))
        for a, b in (drops.get("_INTENT_PAIRS") or [])
    }
    # Also drop pairs whose either side was removed from INTENT_* lexicons.
    intent_stu = {_norm_token(t) for t in out.get("INTENT_STU") or []}
    intent_tea = {_norm_token(t) for t in out.get("INTENT_TEA") or []}
    kept_pairs: list[list[str]] = []
    for pair in rules.get("_INTENT_PAIRS") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = _norm_token(str(pair[0])), _norm_token(str(pair[1]))
        if not a or not b:
            continue
        if (a, b) in drop_pairs:
            continue
        if a not in intent_stu or b not in intent_tea:
            continue
        kept_pairs.append([a, b])
    out["_INTENT_PAIRS"] = kept_pairs
    return out


async def refine_lexicons_async(
    rules: dict[str, Any],
    *,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    cache_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """One LLM call: audit aggregated lexicons; return refined rules + audit meta."""
    if cache_path.is_file() and not force:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("source_fingerprint") == _rules_fingerprint(rules):
            print(f"[Refine] reuse cache {cache_path}")
            return cached

    prompt = build_lexicon_audit_prompt(rules)
    print(f"[Refine] one LLM call model={judge_model} prompt_chars={len(prompt)}")
    try:
        raw = await call_openai_chat_async(
            [{"role": "user", "content": prompt}],
            model_name=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        err = None
    except Exception as e:
        raw = ""
        err = str(e)
        print(f"[Refine] LLM error: {err}; keeping all candidates")

    audit = parse_lexicon_audit_response(raw, rules) if raw else {
        "drop": {k: [] for k in (*_LEXICON_KEYS, "_INTENT_PAIRS")},
        "rationale_brief": f"llm_failed: {err}" if err else "empty_response",
        "raw_preview": "",
    }
    refined = apply_lexicon_drops(rules, audit["drop"])
    # Preserve non-lexicon fields from original, refresh comparison.
    for k in ("TAXONOMY_KEEP_CATEGORIES", "TAXONOMY_DROP_CATEGORIES", "category_counts", "thresholds"):
        if k in rules:
            refined[k] = rules[k]
    refined["comparison_with_qwen_eopsa"] = compare_with_qwen_eopsa(refined)

    drop_counts = {k: len(v) for k, v in audit["drop"].items()}
    payload = {
        "judge_model": judge_model,
        "source_fingerprint": _rules_fingerprint(rules),
        "drop_counts": drop_counts,
        "audit": audit,
        "rules_before": {k: rules.get(k) for k in (*_LEXICON_KEYS, "_INTENT_PAIRS")},
        "rules_after": {k: refined.get(k) for k in (*_LEXICON_KEYS, "_INTENT_PAIRS")},
        "refined_rules": refined,
        "llm_error": err,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Refine] saved {cache_path} drops={drop_counts}")
    return payload


def _rules_fingerprint(rules: dict[str, Any]) -> str:
    payload = {k: rules.get(k) for k in (*_LEXICON_KEYS, "_INTENT_PAIRS")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rubric-guided token rule extraction (1 LLM call / sample + 1 lexicon audit)")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--teacher_template", type=Path, default=DEFAULT_TEACHER_TEMPLATE)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_split", choices=("sample", "holdout"), default="sample")
    p.add_argument("--max_tokens", type=int, default=256,
                   help="Qwen3 student rollout length (response tokens)")
    p.add_argument("--distill_topk", type=int, default=512)
    p.add_argument("--top_kl_per_sample", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_rollout", action="store_true")
    p.add_argument("--steps", default="all",
                   help="Comma list: collect,judge,aggregate,refine  (all = all four)")
    p.add_argument("--judge_mode", choices=("llm", "heuristic"), default="llm")
    p.add_argument("--judge_model", default=__import__("os").environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_max_tokens", type=int, default=2048,
                   help="LLM judge max output tokens (gpt-4o-mini)")
    p.add_argument("--judge_concurrency", type=int, default=16)
    p.add_argument("--refine_max_tokens", type=int, default=4096,
                   help="Max output tokens for the one-shot lexicon audit call")
    p.add_argument("--force_refine", action="store_true",
                   help="Ignore lexicon audit cache and re-call LLM")
    p.add_argument("--min_token_count", type=int, default=2)
    p.add_argument("--min_pair_count", type=int, default=2)
    p.add_argument("--tag", default="qwen3-1.7b")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    if "all" in steps:
        steps = {"collect", "judge", "aggregate", "refine"}

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"topkl_pairs_n{args.n_samples}_max{args.max_tokens}.json"
    judge_path = out_dir / f"sample_judgments_{args.judge_mode}.json"
    rules_path = out_dir / "extracted_rules.json"
    rules_pre_path = out_dir / "extracted_rules_pre_refine.json"
    refine_cache_path = out_dir / "lexicon_audit_llm.json"
    module_path = out_dir / f"filters_extracted_{args.tag.replace('/', '_')}.py"

    print("=" * 72)
    print("  Rubric token rule extraction  (per-sample judge + 1 lexicon audit)")
    print(f"  tag={args.tag}  judge={args.judge_model}  steps={sorted(steps)}")
    print("=" * 72)

    pair_payload: dict[str, Any] | None = None
    need_pairs = bool(steps & {"collect", "judge", "aggregate"})
    if "collect" in steps:
        samples = load_harmful_samples(args.data, args.n_samples, args.seed, split=args.data_split)
        print(f"[Data] harmful n={len(samples)} from {args.data}")
        cache_rollout = out_dir / f"student_rollouts_n{len(samples)}_max{args.max_tokens}.json"
        records = run_student_rollouts(
            model_path=args.model_path,
            samples=samples,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            cache_path=cache_rollout,
            force=args.force_rollout,
        )
        pair_payload = collect_top_kl_pairs(
            args.model_path, records, args.teacher_template,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
            topk=args.distill_topk,
            top_kl_per_sample=args.top_kl_per_sample,
        )
        pair_payload.update(tag=args.tag, data=str(args.data), seed=args.seed)
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(pair_payload, f, ensure_ascii=False, indent=2)
        print(f"[Collect] saved {pairs_path} n_samples={pair_payload['n_samples']}")
    elif need_pairs and pairs_path.is_file():
        with open(pairs_path, "r", encoding="utf-8") as f:
            pair_payload = json.load(f)
        print(f"[Collect] loaded {pairs_path}")
    elif need_pairs:
        raise SystemExit(f"Missing {pairs_path}; run --steps collect first")

    sample_judgments: list[dict[str, Any]] = []
    if "judge" in steps:
        assert pair_payload is not None
        sample_judgments = asyncio.run(judge_samples_async(
            pair_payload["samples"],
            judge_mode=args.judge_mode,
            judge_model=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            concurrency=args.judge_concurrency,
            cache_path=judge_path,
        ))
    elif "aggregate" in steps:
        if judge_path.is_file():
            with open(judge_path, "r", encoding="utf-8") as f:
                sample_judgments = json.load(f)["sample_judgments"]
            print(f"[Judge] loaded {judge_path}")
        else:
            raise SystemExit(f"Missing {judge_path}; run --steps judge first")

    rules: dict[str, Any] | None = None
    if "aggregate" in steps:
        assert pair_payload is not None
        flat = flatten_pair_judgments(sample_judgments)
        rules = aggregate_rules(
            flat,
            min_token_count=args.min_token_count,
            min_pair_count=args.min_pair_count,
        )
        meta = {
            "tag": args.tag,
            "model_path": args.model_path,
            "judge_model": args.judge_model,
            "n_samples": pair_payload.get("n_samples"),
            "n_pair_labels": len(flat),
            "top_kl_per_sample": args.top_kl_per_sample,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "refined": False,
        }
        rules["meta"] = meta
        rules["comparison_with_qwen_eopsa"] = compare_with_qwen_eopsa(rules)
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        # Backup pre-refine snapshot whenever we re-aggregate.
        with open(rules_pre_path, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        module_path.write_text(render_filter_module(rules, tag=args.tag, source_meta=meta), encoding="utf-8")
        print(f"[Aggregate] {rules_path}")
        for k in ("PIVOT_STU", "PIVOT_TEA", "INTENT_STU", "INTENT_TEA", "RISK", "FUNC"):
            print(f"  {k}: {len(rules[k])}")
        print(f"  _INTENT_PAIRS: {len(rules['_INTENT_PAIRS'])}")
        print(f"  category_counts: {rules.get('category_counts')}")
    elif "refine" in steps:
        # Prefer pre-refine snapshot so re-running refine does not compound drops.
        load_path = rules_pre_path if rules_pre_path.is_file() else rules_path
        if not load_path.is_file():
            raise SystemExit(f"Missing {load_path}; run --steps aggregate first")
        with open(load_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        print(f"[Refine] loaded candidates from {load_path}")
        if not rules_pre_path.is_file():
            with open(rules_pre_path, "w", encoding="utf-8") as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)

    if "refine" in steps:
        if rules is None:
            raise SystemExit("No rules to refine; run aggregate first")
        refine_payload = asyncio.run(refine_lexicons_async(
            rules,
            judge_model=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.refine_max_tokens,
            cache_path=refine_cache_path,
            force=args.force_refine,
        ))
        refined = refine_payload["refined_rules"]
        meta = dict(refined.get("meta") or rules.get("meta") or {})
        meta.update({
            "refined": True,
            "refine_model": args.judge_model,
            "refine_drop_counts": refine_payload.get("drop_counts"),
            "refine_rationale": (refine_payload.get("audit") or {}).get("rationale_brief", ""),
            "refined_at": datetime.now().isoformat(timespec="seconds"),
        })
        refined["meta"] = meta
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump(refined, f, ensure_ascii=False, indent=2)
        module_path.write_text(
            render_filter_module(refined, tag=args.tag, source_meta=meta),
            encoding="utf-8",
        )
        print(f"[Refine] wrote {rules_path} (pre-refine kept at {rules_pre_path})")
        for k in ("PIVOT_STU", "PIVOT_TEA", "INTENT_STU", "INTENT_TEA", "RISK", "FUNC"):
            before_n = len((refine_payload.get("rules_before") or {}).get(k) or [])
            after_n = len(refined.get(k) or [])
            print(f"  {k}: {before_n} → {after_n}")
        bp = len((refine_payload.get("rules_before") or {}).get("_INTENT_PAIRS") or [])
        ap = len(refined.get("_INTENT_PAIRS") or [])
        print(f"  _INTENT_PAIRS: {bp} → {ap}")
        rationale = (refine_payload.get("audit") or {}).get("rationale_brief") or ""
        if rationale:
            print(f"  rationale: {rationale[:300]}")

    print("Done.")


if __name__ == "__main__":
    main()
