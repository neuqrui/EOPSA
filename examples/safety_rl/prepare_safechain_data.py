# -*- coding: utf-8 -*-
"""Prepare SafeChain subset (harmful + benign) for EOPSA training."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from prepare_safety_data import (
    BENIGN_GUIDANCE,
    HARMFUL_GUIDANCE,
    interleave_by_ratio,
    proportional_split,
    take_up_to,
    write_jsonl,
)

DEFAULT_DATASET = "UWNSL/SafeChain"


def _ensure_hf_mirror() -> None:
    """Use hf-mirror when HF_ENDPOINT is unset (faster in CN)."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _is_benign_label(label: str) -> bool:
    text = str(label).strip().lower()
    if "benign" in text or "harmless" in text:
        return True
    if "harmful" in text:
        return False
    raise ValueError(f"Unrecognized SafeChain label: {label!r}")


def load_safechain_rows(
    *,
    dataset_name: str,
    local_path: str | None,
    hf_token: str | None,
) -> list[dict[str, Any]]:
    if local_path:
        path = Path(local_path)
        if path.is_dir():
            from datasets import load_dataset

            ds = load_dataset("parquet", data_files=str(path / "train-*.parquet"), split="train")
        elif path.suffix == ".parquet":
            from datasets import load_dataset

            ds = load_dataset("parquet", data_files=str(path), split="train")
        elif path.suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        else:
            raise ValueError(f"Unsupported local SafeChain path: {path}")
    else:
        from datasets import load_dataset

        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            ds = load_dataset(dataset_name, split="train", token=token)
        except Exception as exc:
            msg = str(exc)
            if "gated" in msg.lower() or "authorized" in msg.lower():
                raise SystemExit(
                    "[ERROR] SafeChain is a gated dataset. Steps:\n"
                    "  1) Visit https://huggingface.co/datasets/UWNSL/SafeChain and accept terms\n"
                    "  2) Ensure HF_TOKEN has access (export HF_TOKEN=...)\n"
                    "  3) HF mirror is already used via HF_ENDPOINT=https://hf-mirror.com\n"
                    "  4) Or download manually and pass --local_path /path/to/SafeChain/data\n"
                    f"Original error: {exc}"
                ) from exc
            raise

    rows: list[dict[str, Any]] = []
    for item in ds:
        instruction = str(item.get("instruction", "")).strip()
        label = str(item.get("label", "")).strip()
        if not instruction or not label:
            continue
        is_benign = _is_benign_label(label)
        rows.append(
            {
                "instruction": instruction,
                "label": label,
                "response": str(item.get("response", "")).strip(),
                "data_type": "overreject" if is_benign else "safety",
                "risk_summary": "",
                "difficulty": "",
            }
        )
    return rows


def split_harmful_benign(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    harmful = [r for r in rows if r["data_type"] == "safety"]
    benign = [r for r in rows if r["data_type"] == "overreject"]
    return harmful, benign


def to_rlsd_row(example: dict[str, Any], idx: int, *, use_response_hint: bool) -> dict[str, Any]:
    data_type = str(example.get("data_type", "safety")).strip().lower()
    is_benign = data_type in {"overreject", "benign", "harmless", "harmless_queries"}
    instruction = example["instruction"]
    if use_response_hint and example.get("response"):
        safe_reference = example["response"]
    else:
        safe_reference = BENIGN_GUIDANCE if is_benign else HARMFUL_GUIDANCE
    return {
        "problem": instruction,
        "answer": "",
        "safe_reference": safe_reference,
        "data_type": "overreject" if is_benign else "safety",
        "problem_type": "safety",
        "problem_reserved_text": instruction,
        "problem_id": idx,
        "risk_summary": example.get("risk_summary", ""),
        "difficulty": example.get("difficulty", ""),
        "safechain_label": example.get("label", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RLSD jsonl from SafeChain.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--local_path",
        type=str,
        default="",
        help="Local parquet dir/file or jsonl (skip Hub download).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--harmful", type=int, default=440, help="Max harmful samples (ignored with --use_all).")
    parser.add_argument("--benign", type=int, default=220, help="Max benign samples (ignored with --use_all).")
    parser.add_argument("--train_data_size", type=float, default=0.9)
    parser.add_argument("--val_data_size", type=float, default=0.1)
    parser.add_argument(
        "--use_all",
        action="store_true",
        help="Use all harmful and benign rows from SafeChain (no subsampling).",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Put all selected data into train.jsonl; write a 1-line val.jsonl stub for dataloader compat.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_response_hint",
        action="store_true",
        help="Use SafeChain `response` as teacher hint instead of fixed guidance text.",
    )
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    _ensure_hf_mirror()

    local_path = args.local_path.strip() or None
    hf_token = args.hf_token.strip() or None
    all_rows = load_safechain_rows(
        dataset_name=args.dataset,
        local_path=local_path,
        hf_token=hf_token,
    )
    harmful_pool, benign_pool = split_harmful_benign(all_rows)

    if args.train_only:
        args.train_data_size = 1.0
        args.val_data_size = 0.0

    if args.use_all:
        harmful = list(harmful_pool)
        benign = list(benign_pool)
    else:
        harmful = take_up_to(harmful_pool, args.harmful, args.seed)
        benign = take_up_to(benign_pool, args.benign, args.seed + 7919)
        if len(harmful) < args.harmful:
            raise SystemExit(
                f"[ERROR] Only {len(harmful)} harmful samples available, need {args.harmful}."
            )
        if len(benign) < args.benign:
            raise SystemExit(
                f"[ERROR] Only {len(benign)} benign samples available, need {args.benign}."
            )

    train_raw, val_raw = proportional_split(
        harmful,
        benign,
        train_size=args.train_data_size,
        val_size=args.val_data_size,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    train_rows = [
        to_rlsd_row(r, i, use_response_hint=args.use_response_hint) for i, r in enumerate(train_raw)
    ]
    val_rows = [
        to_rlsd_row(r, i, use_response_hint=args.use_response_hint) for i, r in enumerate(val_raw)
    ]
    if args.train_only and train_rows:
        val_rows = [train_rows[0]]
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)

    label_counts: dict[str, int] = {}
    for r in harmful + benign:
        label = str(r.get("label", ""))
        label_counts[label] = label_counts.get(label, 0) + 1

    meta = {
        "source": "safechain",
        "dataset": args.dataset,
        "local_path": local_path,
        "harmful": len(harmful),
        "benign": len(benign),
        "train": len(train_rows),
        "val": len(val_rows),
        "use_all": args.use_all,
        "train_only": args.train_only,
        "seed": args.seed,
        "use_response_hint": args.use_response_hint,
        "label_counts": label_counts,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "privileged_key": "safe_reference",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
