#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace harmful samples in a safety val.jsonl with wildjailbreak + wildchat mix.

Keeps total val size and benign (overreject) samples unchanged.
Sources:
  - wildjailbreak: local HF arrow cache / HuggingFace dataset
  - wildchat: CSV used by LLM-Safety-Eval
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SAFETY_RL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAFETY_RL_DIR.parent.parent
sys.path.insert(0, str(SAFETY_RL_DIR))

from prepare_safety_data import (  # noqa: E402
    EASY_HARMFUL_GUIDANCE,
    HARMFUL_GUIDANCE,
    interleave_by_ratio,
    write_jsonl,
)

DEFAULT_VAL = (
    SAFETY_RL_DIR
    / "datasets"
    / "safety_ds_safechain_dsr100_h4400_b2200"
    / "val.jsonl"
)
DEFAULT_TRAIN = DEFAULT_VAL.with_name("train.jsonl")
_EVAL_ROOT = Path(
    __import__("os").environ.get(
        "EVAL_LLM_SAFETY_DIR",
        str(SAFETY_RL_DIR.parent.parent.parent / "LLM-Safety-Eval"),
    )
)
DEFAULT_WJ_ARROW = (
    _EVAL_ROOT
    / "datasets"
    / "wildjailbreak_temp_hf_cache"
    / "allenai___wildjailbreak"
    / "eval"
    / "0.0.0"
    / "5ddc12a7894f842b0619b8e1c7ee496b198af009"
    / "wildjailbreak-train.arrow"
)
DEFAULT_WC_CSV = _EVAL_ROOT / "datasets" / "wildchat-100.csv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_wildjailbreak_harmful(arrow_path: Path) -> list[str]:
    from datasets import Dataset

    if not arrow_path.is_file():
        raise FileNotFoundError(f"wildjailbreak arrow not found: {arrow_path}")
    ds = Dataset.from_file(str(arrow_path))
    qs: list[str] = []
    for row in ds:
        if row.get("data_type") != "adversarial_harmful":
            continue
        q = (row.get("adversarial") or "").strip()
        if q:
            qs.append(q)
    return qs


def load_wildchat(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"wildchat csv not found: {csv_path}")
    qs: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").strip()
            if q:
                qs.append(q)
    return qs


def sample_unique(
    pool: list[str],
    n: int,
    *,
    seed: int,
    exclude: set[str],
) -> list[str]:
    rng = random.Random(seed)
    cand = []
    seen: set[str] = set()
    for q in pool:
        if q in exclude or q in seen:
            continue
        seen.add(q)
        cand.append(q)
    if len(cand) < n:
        raise SystemExit(
            f"[ERROR] need {n} unique prompts, only {len(cand)} after exclude "
            f"(pool={len(pool)}, exclude={len(exclude)})"
        )
    rng.shuffle(cand)
    return cand[:n]


def resolve_harmful_guidance(hint_mode: str) -> str:
    if hint_mode in ("easy",):
        return EASY_HARMFUL_GUIDANCE
    return HARMFUL_GUIDANCE


def make_harmful_row(problem: str, source: str, problem_id: int, harmful_guidance: str) -> dict[str, Any]:
    return {
        "problem": problem,
        "answer": "",
        "safe_reference": harmful_guidance,
        "data_type": "safety",
        "problem_type": "safety",
        "problem_reserved_text": problem,
        "problem_id": problem_id,
        "risk_summary": "",
        "difficulty": "",
        "safechain_label": source,
        "val_source": source,
        "hint_mode": "easy" if harmful_guidance == EASY_HARMFUL_GUIDANCE else "hard",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--wj_arrow", type=Path, default=DEFAULT_WJ_ARROW)
    parser.add_argument("--wc_csv", type=Path, default=DEFAULT_WC_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wj_ratio",
        type=float,
        default=0.5,
        help="Fraction of harmful val from wildjailbreak (rest from wildchat).",
    )
    parser.add_argument(
        "--n_harmful",
        type=int,
        default=None,
        help="Number of harmful (WJ+WC) rows. Default: keep count from --val.",
    )
    parser.add_argument(
        "--n_benign",
        type=int,
        default=None,
        help="Number of benign (overreject) rows kept from --val. Default: keep all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: overwrite --val (after backup to val_safechain_orig.jsonl).",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not write val_safechain_orig.jsonl when overwriting.",
    )
    parser.add_argument(
        "--hint_mode",
        choices=("hard", "full", "easy"),
        default="hard",
        help="Teacher PI for harmful rows: hard=long guidance, easy=short refuse sentence.",
    )
    args = parser.parse_args()
    hint_mode = "easy" if args.hint_mode == "easy" else "hard"
    harmful_guidance = resolve_harmful_guidance(hint_mode)

    val_rows = load_jsonl(args.val)
    benign = [r for r in val_rows if r.get("data_type") == "overreject"]
    old_harmful = [r for r in val_rows if r.get("data_type") == "safety"]
    n_harmful = int(args.n_harmful) if args.n_harmful is not None else len(old_harmful)
    n_benign = int(args.n_benign) if args.n_benign is not None else len(benign)
    if n_harmful <= 0:
        raise SystemExit("[ERROR] n_harmful must be > 0")
    if n_benign <= 0:
        raise SystemExit("[ERROR] n_benign must be > 0")
    if len(old_harmful) == 0 and args.n_harmful is None:
        raise SystemExit("[ERROR] no harmful (data_type=safety) rows in val")
    if len(benign) == 0:
        raise SystemExit("[ERROR] no benign (overreject) rows in val")
    if n_benign > len(benign):
        raise SystemExit(
            f"[ERROR] n_benign={n_benign} > available benign={len(benign)} in {args.val}"
        )

    n_wj = int(round(n_harmful * args.wj_ratio))
    n_wc = n_harmful - n_wj
    print(
        f"[INFO] val source={len(val_rows)} (harmful={len(old_harmful)} benign={len(benign)}) "
        f"-> target harmful={n_harmful} benign={n_benign} "
        f"WJ={n_wj} + WC={n_wc}  hint_mode={hint_mode}"
    )

    exclude: set[str] = set()
    if args.train.is_file():
        for r in load_jsonl(args.train):
            p = (r.get("problem") or "").strip()
            if p:
                exclude.add(p)
    for r in benign:
        p = (r.get("problem") or "").strip()
        if p:
            exclude.add(p)

    wj_pool = load_wildjailbreak_harmful(args.wj_arrow)
    wc_pool = load_wildchat(args.wc_csv)
    print(f"[INFO] pools: wildjailbreak_harmful={len(wj_pool)} wildchat={len(wc_pool)}")

    picked_wj = sample_unique(wj_pool, n_wj, seed=args.seed, exclude=exclude)
    exclude_wc = exclude | set(picked_wj)
    picked_wc = sample_unique(wc_pool, n_wc, seed=args.seed + 1, exclude=exclude_wc)

    harmful_rows = [
        make_harmful_row(q, "wildjailbreak", i, harmful_guidance) for i, q in enumerate(picked_wj)
    ] + [
        make_harmful_row(q, "wildchat", n_wj + i, harmful_guidance) for i, q in enumerate(picked_wc)
    ]
    # Keep benign fields; subsample if --n_benign is set.
    benign_rows = [dict(r) for r in benign]
    if n_benign < len(benign_rows):
        rng_b = random.Random(args.seed + 2)
        rng_b.shuffle(benign_rows)
        benign_rows = benign_rows[:n_benign]

    merged = interleave_by_ratio(harmful_rows, benign_rows)
    for i, row in enumerate(merged):
        row["problem_id"] = i

    out = args.output or args.val
    if out.resolve() == args.val.resolve() and not args.no_backup:
        bak = args.val.with_name("val_safechain_orig.jsonl")
        if not bak.exists():
            write_jsonl(bak, val_rows)
            print(f"[INFO] backed up original val -> {bak}")
        else:
            print(f"[INFO] backup already exists, skip: {bak}")

    write_jsonl(out, merged)

    meta_path = out.with_name(f"{out.stem}.meta.json")
    meta = {
        "output": str(out),
        "source_val": str(args.val),
        "n_total": len(merged),
        "n_harmful": n_harmful,
        "n_benign": n_benign,
        "n_wildjailbreak": n_wj,
        "n_wildchat": n_wc,
        "wj_ratio": args.wj_ratio,
        "seed": args.seed,
        "hint_mode": hint_mode,
        "harmful_guidance": harmful_guidance,
        "wj_arrow": str(args.wj_arrow),
        "wc_csv": str(args.wc_csv),
        "exclude_train": str(args.train) if args.train.is_file() else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out} ({len(merged)} rows)")
    print(f"[OK] meta {meta_path}")


if __name__ == "__main__":
    main()
