#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the paper SafeChain subset in veRL / EOPSA jsonl format.

Selects ``HARMFUL_N`` harmful + ``BENIGN_N`` benign rows from UWNSL/SafeChain
(or a local parquet/jsonl), converts them to the training schema, and writes:

  datasets/safechain-subset/{train,val}.jsonl
  datasets/safechain-subset/meta.json

The official release already ships this subset. Prefer the committed files;
only rebuild when you explicitly need a fresh sample (e.g. REBUILD_DATA=1).

Example:
  python3 prepare_safechain_subset.py
  python3 prepare_safechain_subset.py --local_path datasets/SafeChain_raw/data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prepare_safechain_data import (
    load_safechain_rows,
    split_harmful_benign,
    to_rlsd_row,
)
from prepare_safety_data import proportional_split, take_up_to, write_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "datasets" / "safechain-subset"
DEFAULT_DATASET = "UWNSL/SafeChain"
DEFAULT_HARMFUL = 4400
DEFAULT_BENIGN = 2200


def _ensure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample SafeChain into the EOPSA paper subset (veRL jsonl)."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--local_path",
        type=str,
        default="",
        help="Local SafeChain parquet dir/file or jsonl (skip Hub download).",
    )
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--harmful", type=int, default=DEFAULT_HARMFUL)
    parser.add_argument("--benign", type=int, default=DEFAULT_BENIGN)
    parser.add_argument("--train_data_size", type=float, default=0.98)
    parser.add_argument("--val_data_size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_response_hint",
        action="store_true",
        help="Use SafeChain response as teacher hint instead of fixed guidance.",
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
        to_rlsd_row(r, i, use_response_hint=args.use_response_hint)
        for i, r in enumerate(train_raw)
    ]
    val_rows = [
        to_rlsd_row(r, i, use_response_hint=args.use_response_hint)
        for i, r in enumerate(val_raw)
    ]
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)

    meta = {
        "source": "safechain-subset",
        "dataset": args.dataset,
        "local_path": local_path,
        "harmful": len(harmful),
        "benign": len(benign),
        "train": len(train_rows),
        "val": len(val_rows),
        "train_data_size": args.train_data_size,
        "val_data_size": args.val_data_size,
        "seed": args.seed,
        "use_response_hint": args.use_response_hint,
        "output_dir": str(out_dir),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "note": "Shipped paper subset lives under datasets/safechain-subset; "
        "rebuild only with REBUILD_DATA=1.",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {out_dir / 'train.jsonl'} and {out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
