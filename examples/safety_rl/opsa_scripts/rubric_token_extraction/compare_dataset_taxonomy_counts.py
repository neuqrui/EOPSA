#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare taxonomy category token counts across datasets and classifiers.

For each dataset (safechain / star1 / mix):
  1) Sample N harmful rows from train.jsonl
  2) Student rollout (Qwen3-1.7B, max_tokens=256)
  3) Teacher forward on same response ids
  4) Classify every (student_top1, teacher_top1) with each requested classifier
  5) Aggregate category counts → summary tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SAFETY_RL_DIR = SCRIPT_DIR.parent.parent
PROJECT_DIR = SAFETY_RL_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SAFETY_RL_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from analyze_prefix_token_kl import (  # noqa: E402
    apply_chat_template,
    render_teacher_prompt,
    response_logits,
    run_student_rollouts,
)

DEFAULT_MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-1.7B")
DEFAULT_TEACHER = SAFETY_RL_DIR / "format_prompt" / "safety_teacher.jinja"
DEFAULT_OUT = SCRIPT_DIR / "outputs" / "dataset_taxonomy_compare"

DATASETS: dict[str, Path] = {
    "safechain": SAFETY_RL_DIR / "datasets/safety_ds_safechain_dsr100_h4400_b2200/train.jsonl",
    "star1": SAFETY_RL_DIR / "datasets/safety_ds_star1_h1000_b500/train.jsonl",
    "mix": SAFETY_RL_DIR / "datasets/opsd_mix_safechain_h4400_b1100_m1100/train.jsonl",
}

CATEGORIES = (
    "pivot",
    "intent",
    "risk_wo_same",
    "risk_lexicon",
    "same",
    "function",
    "other",
)

CLASSIFIERS: dict[str, tuple[str, Callable[[str, str], str]]] = {}


def _register_classifiers() -> None:
    if CLASSIFIERS:
        return
    from token_filter.filters_qwen_rubric import classify_token_qwen_rubric
    from token_filter.filters_qwen_eopsa import classify_token_qwen_eopsa_1p7b_overall

    CLASSIFIERS["qwen_rubric"] = ("filters_qwen_rubric", classify_token_qwen_rubric)
    CLASSIFIERS["qwen_eopsa"] = ("filters_qwen_eopsa", classify_token_qwen_eopsa_1p7b_overall)


def load_harmful(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
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


def decode_tok(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], skip_special_tokens=False)


@torch.no_grad()
def classify_dataset_records_multi(
    model_path: str,
    records: list[dict[str, Any]],
    teacher_template: Path,
    classifier_names: list[str],
    *,
    enable_thinking: bool,
    max_tokens: int,
) -> dict[str, dict[str, Any]]:
    _register_classifiers()
    fns = {name: CLASSIFIERS[name][1] for name in classifier_names}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[Classify] Loading {model_path} on {device} classifiers={classifier_names}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    cat_counts: dict[str, Counter[str]] = {n: Counter() for n in classifier_names}
    total_positions = 0
    n_samples = 0
    t0 = time.time()

    for rec_idx, rec in enumerate(records):
        resp_ids = list(rec.get("student_token_ids") or [])[:max_tokens]
        if not resp_ids:
            continue
        q = str(rec["problem"])
        hint = str(rec.get("safe_reference") or "")

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

        stu_top1 = stu_logits[:t_len].argmax(dim=-1).cpu().numpy()
        tea_top1 = tea_logits[:t_len].argmax(dim=-1).cpu().numpy()

        for pos in range(t_len):
            stu_tok = decode_tok(tokenizer, int(stu_top1[pos]))
            tea_tok = decode_tok(tokenizer, int(tea_top1[pos]))
            for cname, fn in fns.items():
                cat_counts[cname][fn(stu_tok, tea_tok)] += 1
            total_positions += 1
        n_samples += 1

        if (rec_idx + 1) % 20 == 0:
            print(f"[Classify] {rec_idx + 1}/{len(records)} samples, positions={total_positions}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    out: dict[str, dict[str, Any]] = {}
    for cname in classifier_names:
        mod_name = CLASSIFIERS[cname][0]
        out[cname] = {
            "classifier": cname,
            "classifier_module": mod_name,
            "n_samples": n_samples,
            "total_positions": total_positions,
            "category_counts": dict(cat_counts[cname]),
            "elapsed_sec": elapsed,
        }
    return out


def _keep_share(counts: dict[str, int], total: int) -> float:
    keep = counts.get("pivot", 0) + counts.get("intent", 0) + counts.get("risk_wo_same", 0)
    return 100.0 * keep / total if total else 0.0


def fmt_classifier_table(
    results: dict[str, dict[str, dict[str, Any]]],
    classifier: str,
    datasets: list[str],
) -> str:
    lines = [f"### {classifier}", ""]
    header = ["category", *datasets, "max/min"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for cat in CATEGORIES:
        vals = [results[ds][classifier]["category_counts"].get(cat, 0) for ds in datasets]
        vmax, vmin = max(vals), min(vals)
        ratio = f"{vmax / vmin:.2f}" if vmin > 0 else ("inf" if vmax > 0 else "—")
        lines.append("| " + " | ".join([cat, *[str(v) for v in vals], ratio]) + " |")

    totals = [results[ds][classifier]["total_positions"] for ds in datasets]
    vmax, vmin = max(totals), min(totals)
    ratio = f"{vmax / vmin:.2f}" if vmin > 0 else "—"
    lines.append("| " + " | ".join(["**total**", *[str(t) for t in totals], ratio]) + " |")
    lines.append("")
    lines.append("**Keep share (pivot + intent + risk_wo_same):**")
    for ds in datasets:
        r = results[ds][classifier]
        tot = r["total_positions"] or 1
        lines.append(f"- {ds}: {_keep_share(r['category_counts'], tot):.2f}%")
    return "\n".join(lines)


def fmt_cross_classifier_table(
    results: dict[str, dict[str, dict[str, Any]]],
    classifiers: list[str],
    datasets: list[str],
) -> str:
    """Wide table: rows=category, cols=dataset@classifier."""
    lines = ["### Cross-classifier comparison", ""]
    cols = [f"{ds}@{cls}" for ds in datasets for cls in classifiers]
    header = ["category", *cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for cat in CATEGORIES:
        vals = [
            results[ds][cls]["category_counts"].get(cat, 0)
            for ds in datasets for cls in classifiers
        ]
        lines.append("| " + " | ".join([cat, *[str(v) for v in vals]]) + " |")

    totals = [
        results[ds][cls]["total_positions"]
        for ds in datasets for cls in classifiers
    ]
    lines.append("| " + " | ".join(["**total**", *[str(t) for t in totals]]) + " |")
    lines.append("")
    lines.append("**Keep share (%):**")
    for ds in datasets:
        parts = []
        for cls in classifiers:
            r = results[ds][cls]
            tot = r["total_positions"] or 1
            parts.append(f"{cls}={_keep_share(r['category_counts'], tot):.2f}%")
        lines.append(f"- {ds}: " + ", ".join(parts))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare taxonomy counts across datasets and classifiers")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--teacher_template", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--datasets", default="safechain,star1,mix")
    p.add_argument("--classifiers", default="qwen_rubric,qwen_eopsa")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force_rollout", action="store_true")
    p.add_argument("--skip_rollout", action="store_true", help="Use cached rollouts only")
    p.add_argument("--skip_classify", action="store_true", help="Only rollout, skip HF classify")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _register_classifiers()
    ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    clf_names = [s.strip() for s in args.classifiers.split(",") if s.strip()]
    for c in clf_names:
        if c not in CLASSIFIERS:
            raise SystemExit(f"Unknown classifier {c!r}; choose from {list(CLASSIFIERS)}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # results[dataset][classifier]
    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for name in ds_names:
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
        data_path = DATASETS[name]
        if not data_path.is_file():
            raise SystemExit(f"Missing {data_path}")

        print("=" * 72)
        print(f"  Dataset: {name}  path={data_path}")
        print("=" * 72)

        samples = load_harmful(data_path, args.n_samples, args.seed)
        print(f"[Data] harmful n={len(samples)}")

        cache_rollout = out_dir / f"rollouts_{name}_n{len(samples)}_max{args.max_tokens}.json"
        if args.skip_rollout and cache_rollout.is_file():
            with open(cache_rollout, "r", encoding="utf-8") as f:
                records = json.load(f)
            print(f"[Rollout] loaded cache {cache_rollout}")
        else:
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

        if args.skip_classify:
            continue

        clf_results = classify_dataset_records_multi(
            args.model_path,
            records,
            args.teacher_template,
            clf_names,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
        )
        for cname, result in clf_results.items():
            result["dataset"] = name
            result["data_path"] = str(data_path)
            result["n_requested"] = args.n_samples
            out_json = out_dir / f"taxonomy_counts_{name}_{cname}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[Done] {name}/{cname} -> {out_json}")

        all_results[name] = clf_results

    if not all_results:
        print("[Info] skip_classify or no datasets processed.")
        return

    summary = {
        "model_path": args.model_path,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "classifiers": clf_names,
        "datasets": all_results,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = out_dir / "taxonomy_counts_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_parts = ["# Dataset taxonomy comparison\n"]
    for cls in clf_names:
        md_parts.append(fmt_classifier_table(all_results, cls, ds_names))
        md_parts.append("")
    md_parts.append(fmt_cross_classifier_table(all_results, clf_names, ds_names))
    md = "\n".join(md_parts) + "\n"

    md_path = out_dir / "taxonomy_counts_table.md"
    md_path.write_text(md, encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(md)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
