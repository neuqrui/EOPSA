# -*- coding: utf-8 -*-
"""Prepare mixed harmful + benign safety dataset for OPSD / EOPSA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

HARMFUL_GUIDANCE = (
    "The current input contains a harmful request. First, reason internally to identify its core "
    "intent and safety risks, concluding that it must be refused. In your response: explicitly "
    "refuse the request with absolutely no partial compliance. You may provide positive, "
    "constructive guidance only under strictly safe and compliant premises."
)

# Short "easy" teacher privilege for ablation (DSR100easy / TEACHER_PI_MODE=easy).
EASY_HARMFUL_GUIDANCE = "The query is harmful, you must refuse."

BENIGN_GUIDANCE = (
    "This is a safe, benign request. Respond helpfully and directly. You must NOT refuse, add "
    "safety disclaimers, or treat this as potentially harmful. Provide a complete, accurate answer."
)


def _normalize_instruction(row: dict[str, Any]) -> str:
    text = row.get("instruction") or row.get("question") or ""
    return str(text).strip()


def load_harmful_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        instruction = _normalize_instruction(r)
        if not instruction:
            continue
        rows.append(
            {
                "instruction": instruction,
                "data_type": r.get("data_type", "safety"),
                "risk_summary": r.get("risk_summary", ""),
                "difficulty": r.get("difficulty", ""),
            }
        )
    return rows


def load_benign_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        raw = raw["data"]
    rows = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        instruction = _normalize_instruction(r)
        if not instruction:
            continue
        rows.append(
            {
                "instruction": instruction,
                "data_type": r.get("data_type", "overreject"),
                "risk_summary": r.get("risk_summary", ""),
                "difficulty": r.get("difficulty", ""),
            }
        )
    return rows


def take_up_to(items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    seq = list(items)
    random.Random(seed).shuffle(seq)
    return seq[:n]


def interleave_streams(streams: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Proportionally interleave N non-empty streams (Bresenham-style).

    Empty streams are skipped. Relative frequencies match len(stream_i) / total.
    """
    active = [list(s) for s in streams if s]
    if not active:
        return []
    if len(active) == 1:
        return list(active[0])

    weights = [len(s) for s in active]
    total = sum(weights)
    indices = [0] * len(active)
    credit = [0] * len(active)
    out: list[dict[str, Any]] = []
    while len(out) < total:
        for i, w in enumerate(weights):
            if indices[i] < weights[i]:
                credit[i] += w
        candidates = [i for i in range(len(active)) if indices[i] < weights[i]]
        pick = max(candidates, key=lambda i: (credit[i], -i))
        out.append(active[pick][indices[pick]])
        indices[pick] += 1
        credit[pick] -= total
    return out


def interleave_by_ratio(harmful: list[dict[str, Any]], benign: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return interleave_streams([harmful, benign])


def proportional_split(
    harmful: list[dict[str, Any]],
    benign: list[dict[str, Any]],
    *,
    train_size: float,
    val_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(harmful) + len(benign)
    train_n = int(round(total * train_size))
    val_n = int(round(total * val_size))
    if train_n + val_n > total:
        val_n = total - train_n

    harmful_val_n = int(round(val_n * (len(harmful) / total))) if harmful else 0
    harmful_val_n = max(0, min(harmful_val_n, len(harmful)))
    benign_val_n = val_n - harmful_val_n
    if benign_val_n > len(benign):
        spill = benign_val_n - len(benign)
        benign_val_n = len(benign)
        harmful_val_n = min(len(harmful), harmful_val_n + spill)

    harmful_train_pool = list(harmful)
    benign_train_pool = list(benign)
    random.Random(seed).shuffle(harmful_train_pool)
    random.Random(seed + 7919).shuffle(benign_train_pool)

    harmful_val = harmful_train_pool[:harmful_val_n]
    benign_val = benign_train_pool[:benign_val_n]
    harmful_train_pool = harmful_train_pool[harmful_val_n:]
    benign_train_pool = benign_train_pool[benign_val_n:]

    train_rows = interleave_by_ratio(harmful_train_pool, benign_train_pool)[:train_n]
    val_rows = interleave_by_ratio(harmful_val, benign_val)
    return train_rows, val_rows


def to_rlsd_row(example: dict[str, Any], idx: int) -> dict[str, Any]:
    data_type = str(example.get("data_type", "safety")).strip().lower()
    is_benign = data_type in {"overreject", "benign", "harmless", "harmless_queries"}
    instruction = example["instruction"]
    return {
        "problem": instruction,
        "answer": "",
        "safe_reference": BENIGN_GUIDANCE if is_benign else HARMFUL_GUIDANCE,
        "data_type": "overreject" if is_benign else "safety",
        "problem_type": "safety",
        "problem_reserved_text": instruction,
        "problem_id": idx,
        "risk_summary": example.get("risk_summary", ""),
        "difficulty": example.get("difficulty", ""),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmful_json", type=str, required=True)
    parser.add_argument("--benign_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--harmful", type=int, default=440)
    parser.add_argument("--benign", type=int, default=220)
    parser.add_argument("--train_data_size", type=float, default=0.9)
    parser.add_argument("--val_data_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    harmful = take_up_to(load_harmful_json(Path(args.harmful_json)), args.harmful, args.seed)
    benign = take_up_to(load_benign_json(Path(args.benign_json)), args.benign, args.seed + 7919)
    merged = interleave_by_ratio(harmful, benign)
    train_raw, val_raw = proportional_split(
        harmful,
        benign,
        train_size=args.train_data_size,
        val_size=args.val_data_size,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    train_rows = [to_rlsd_row(r, i) for i, r in enumerate(train_raw)]
    val_rows = [to_rlsd_row(r, i) for i, r in enumerate(val_raw)]
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    meta = {
        "harmful": len(harmful),
        "benign": len(benign),
        "train": len(train_rows),
        "val": len(val_rows),
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
