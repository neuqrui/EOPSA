#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 Teacher DSR=100% 的 SafeChain OPSD 训练集。

流程:
  1. 从 SafeChain 取有害候选（默认全量有害池，可 --harmful_candidates 限制）
  2. Teacher（safety_teacher.jinja + HARMFUL_GUIDANCE）推理
  3. Llama-Guard 评测，仅保留 Guard=safe 的有害样本
  4. 取满 HARMFUL_N（默认 4400）；另抽 BENIGN_N（默认 2200）良性（不跑 DSR）
  5. 按训练相同逻辑 interleave + train/val split，写出 train.jsonl / val.jsonl

Usage:
  python3 build_opsd_dsr100_dataset.py --model qwen3-1.7b --harmful 4400 --benign 2200
  bash run_build_opsd_dsr100_dataset.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import jinja2
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
SAFETY_RL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SAFETY_RL_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_safechain_data import (  # noqa: E402
    load_safechain_rows,
    split_harmful_benign,
    to_rlsd_row,
)
from prepare_safety_data import (  # noqa: E402
    BENIGN_GUIDANCE,
    EASY_HARMFUL_GUIDANCE,
    HARMFUL_GUIDANCE,
    proportional_split,
    take_up_to,
    write_jsonl,
)
from config import (  # noqa: E402
    GUARD_MODEL_PATH,
    GUARD_SAMPLING_PARAMS,
    MODELS,
    TEACHER_TEMPLATE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    get_models,
)

DEFAULT_SAFECHAIN_LOCAL = SAFETY_RL_DIR / "datasets" / "SafeChain_raw" / "data"
DEFAULT_OUT_DIR = SAFETY_RL_DIR / "datasets" / "safety_ds_safechain_dsr100_h4400_b2200"


def render_teacher_prompt(question: str, hint: str, template_path: str) -> str:
    with open(template_path, encoding="utf-8") as f:
        tmpl = jinja2.Environment().from_string(f.read())
    return tmpl.render(question=question, problem=question, hint=hint)


def apply_chat_template(
    prompts: list[str],
    tokenizer: Any,
    *,
    enable_thinking: bool | None,
) -> list[str]:
    formatted = []
    for text in prompts:
        messages = [{"role": "user", "content": text}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
    return formatted


def teacher_infer_batch(
    *,
    model_path: str,
    rows: list[dict[str, Any]],
    template_path: str,
    harmful_guidance: str,
    use_response_hint: bool,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    hints = []
    raw_prompts = []
    for r in rows:
        if use_response_hint and r.get("response"):
            hint = str(r["response"])
        else:
            hint = harmful_guidance
        hints.append(hint)
        raw_prompts.append(render_teacher_prompt(r["instruction"], hint, template_path))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    final_prompts = apply_chat_template(raw_prompts, tokenizer, enable_thinking=enable_thinking)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    t0 = time.time()
    outputs = llm.generate(final_prompts, sp)
    print(f"[Teacher] inferred {len(rows)} in {time.time() - t0:.1f}s")

    results = []
    for r, hint, raw_p, final_p, out in zip(rows, hints, raw_prompts, final_prompts, outputs):
        resp = out.outputs[0].text.strip() if out.outputs else ""
        results.append(
            {
                **r,
                "hint": hint,
                "teacher_raw_prompt": raw_p,
                "teacher_final_prompt": final_p,
                "teacher_response": resp,
            }
        )

    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def guard_label_responses(
    *,
    items: list[dict[str, Any]],
    guard_model_path: str,
    response_key: str = "teacher_response",
) -> list[bool]:
    """Return list of is_unsafe aligned with items (True = unsafe / attack succeeded)."""
    print(f"[Guard] loading {guard_model_path}")
    guard_llm = LLM(model=guard_model_path, tensor_parallel_size=1, trust_remote_code=True)
    guard_tok = AutoTokenizer.from_pretrained(guard_model_path)
    sp = SamplingParams(
        temperature=GUARD_SAMPLING_PARAMS["temperature"],
        max_tokens=GUARD_SAMPLING_PARAMS["max_tokens"],
    )

    prompts = []
    for d in items:
        chat = [
            {"role": "user", "content": d["instruction"]},
            {"role": "assistant", "content": d[response_key]},
        ]
        prompts.append(
            guard_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    outputs = guard_llm.generate(prompts, sp)
    unsafe_flags = ["unsafe" in o.outputs[0].text.lower() for o in outputs]

    del guard_llm, guard_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return unsafe_flags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OPSD dataset with Teacher DSR=100% harmful")
    p.add_argument("--model", default="qwen3-1.7b")
    p.add_argument("--model_path", default=None)
    p.add_argument("--harmful", type=int, default=2200, help="Target harmful count after DSR filter")
    p.add_argument("--benign", type=int, default=2200, help="Benign count (no DSR filter; 1:1 with default harmful)")
    p.add_argument(
        "--harmful_candidates",
        type=int,
        default=0,
        help="Max harmful candidates to score (0=all harmful pool). Need >= --harmful after filter.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--safechain_local", default=str(DEFAULT_SAFECHAIN_LOCAL))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    # Defaults keep nearly all samples: val≈2% (≥100 when total≈6600), train=rest.
    p.add_argument("--train_data_size", type=float, default=0.98)
    p.add_argument("--val_data_size", type=float, default=0.02)
    p.add_argument("--use_response_hint", action="store_true")
    p.add_argument("--no_thinking", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument(
        "--hint_mode",
        choices=["full", "easy"],
        default="full",
        help="full/hard=HARMFUL_GUIDANCE; easy='The query is harmful, you must refuse.'",
    )
    p.add_argument(
        "--cache_infer",
        default="",
        help="Optional path to reuse teacher+guard cache json (skip re-infer if exists)",
    )
    return p.parse_args()


def resolve_harmful_guidance(hint_mode: str) -> str:
    if hint_mode == "easy":
        return EASY_HARMFUL_GUIDANCE
    return HARMFUL_GUIDANCE


def main() -> None:
    args = parse_args()
    enable_thinking = not args.no_thinking
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    harmful_guidance = resolve_harmful_guidance(args.hint_mode)

    model_name = args.model
    model_path = args.model_path or MODELS.get(model_name)
    if not model_path:
        # fallback when MODELS_TO_RUN filters it out
        model_path = MODELS.get(model_name) or os.environ.get("MODEL_PATH")
    if not model_path:
        raise SystemExit(f"[ERROR] unknown model {model_name!r}; pass --model_path")

    # ---- load SafeChain ----
    rows = load_safechain_rows(
        dataset_name="UWNSL/SafeChain",
        local_path=args.safechain_local,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    harmful_pool, benign_pool = split_harmful_benign(rows)
    print(
        f"[Data] harmful_pool={len(harmful_pool)} benign_pool={len(benign_pool)} "
        f"labels={dict(Counter(r['label'] for r in rows))}"
    )
    print(f"[Hint] mode={args.hint_mode} text={harmful_guidance!r}")

    cand_n = args.harmful_candidates if args.harmful_candidates > 0 else len(harmful_pool)
    cand_n = min(cand_n, len(harmful_pool))
    if cand_n < args.harmful:
        raise SystemExit(
            f"[ERROR] harmful_candidates={cand_n} < target harmful={args.harmful}"
        )
    candidates = take_up_to(harmful_pool, cand_n, args.seed)
    print(f"[Data] scoring {len(candidates)} harmful candidates for DSR=100% filter")

    cache_name = f"{model_name}_harmful_scored_{args.hint_mode}.json"
    cache_path = Path(args.cache_infer) if args.cache_infer else out_dir / cache_name
    if cache_path.is_file():
        print(f"[Cache] load {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            scored = json.load(f)
    else:
        scored = teacher_infer_batch(
            model_path=model_path,
            rows=candidates,
            template_path=TEACHER_TEMPLATE,
            harmful_guidance=harmful_guidance,
            use_response_hint=args.use_response_hint,
            enable_thinking=enable_thinking,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        unsafe_flags = guard_label_responses(
            items=scored,
            guard_model_path=GUARD_MODEL_PATH,
        )
        for item, unsafe in zip(scored, unsafe_flags):
            item["guard_unsafe"] = bool(unsafe)
            item["guard_safe"] = not bool(unsafe)
            item["hint_mode"] = args.hint_mode

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)
        print(f"[Cache] wrote {cache_path}")

    n_safe = sum(1 for x in scored if x.get("guard_safe"))
    n_unsafe = sum(1 for x in scored if x.get("guard_unsafe"))
    dsr_pool = n_safe / len(scored) if scored else 0.0
    print(f"[Filter] candidates={len(scored)} safe={n_safe} unsafe={n_unsafe} DSR={dsr_pool:.2%}")

    passed = [x for x in scored if x.get("guard_safe")]
    if len(passed) < args.harmful:
        raise SystemExit(
            f"[ERROR] Teacher DSR-safe harmful only {len(passed)} < needed {args.harmful}. "
            f"Increase --harmful_candidates or check teacher/guard."
        )

    # Stable take: already shuffled via take_up_to on candidates; keep order of passed
    harmful_keep = passed[: args.harmful]
    # Verify 100% DSR on kept set
    if any(x.get("guard_unsafe") for x in harmful_keep):
        raise SystemExit("[ERROR] internal: kept harmful still has unsafe")
    print(f"[Filter] kept harmful={len(harmful_keep)} (DSR=100% by construction)")

    # Benign: no DSR filter
    if len(benign_pool) < args.benign:
        raise SystemExit(f"[ERROR] benign pool {len(benign_pool)} < {args.benign}")
    benign_keep = take_up_to(benign_pool, args.benign, args.seed + 7919)
    print(f"[Filter] kept benign={len(benign_keep)}")

    # Convert to RLSD rows (same schema as prepare_safechain_data)
    harmful_raw = [
        {
            "instruction": x["instruction"],
            "label": x["label"],
            "response": x.get("response", ""),
            "data_type": "safety",
            "risk_summary": "",
            "difficulty": "",
            "teacher_response": x.get("teacher_response", ""),
            "guard_safe": True,
        }
        for x in harmful_keep
    ]
    benign_raw = [
        {
            "instruction": x["instruction"],
            "label": x["label"],
            "response": x.get("response", ""),
            "data_type": "overreject",
            "risk_summary": "",
            "difficulty": "",
        }
        for x in benign_keep
    ]

    train_raw, val_raw = proportional_split(
        harmful_raw,
        benign_raw,
        train_size=args.train_data_size,
        val_size=args.val_data_size,
        seed=args.seed,
    )

    def _to_row(r: dict[str, Any], i: int) -> dict[str, Any]:
        row = to_rlsd_row(r, i, use_response_hint=args.use_response_hint)
        # Force privileged hint text used at train time (must match filter-time hint).
        if row["data_type"] == "safety" and not (
            args.use_response_hint and r.get("response")
        ):
            row["safe_reference"] = harmful_guidance
        elif row["data_type"] == "overreject" and not (
            args.use_response_hint and r.get("response")
        ):
            row["safe_reference"] = BENIGN_GUIDANCE
        row["hint_mode"] = args.hint_mode
        return row

    train_rows = [_to_row(r, i) for i, r in enumerate(train_raw)]
    val_rows = [_to_row(r, i) for i, r in enumerate(val_raw)]

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)

    write_jsonl(
        out_dir / f"harmful_dsr100_{args.hint_mode}.jsonl",
        [
            {
                "instruction": x["instruction"],
                "label": x["label"],
                "teacher_response": x.get("teacher_response", ""),
                "hint": x.get("hint", harmful_guidance),
                "guard_safe": True,
                "hint_mode": args.hint_mode,
            }
            for x in harmful_keep
        ],
    )

    tag = "dsr100easy" if args.hint_mode == "easy" else "dsr100"
    meta = {
        "source": f"safechain_teacher_{tag}",
        "hint_mode": args.hint_mode,
        "harmful_guidance": harmful_guidance,
        "model": model_name,
        "model_path": model_path,
        "harmful_target": args.harmful,
        "benign_target": args.benign,
        "harmful_candidates_scored": len(scored),
        "harmful_passed_guard": n_safe,
        "candidate_pool_dsr": dsr_pool,
        "kept_harmful_dsr": 1.0,
        "train": len(train_rows),
        "val": len(val_rows),
        "train_harmful": sum(1 for r in train_rows if r["data_type"] == "safety"),
        "train_benign": sum(1 for r in train_rows if r["data_type"] == "overreject"),
        "val_harmful": sum(1 for r in val_rows if r["data_type"] == "safety"),
        "val_benign": sum(1 for r in val_rows if r["data_type"] == "overreject"),
        "seed": args.seed,
        "teacher_template": TEACHER_TEMPLATE,
        "use_response_hint": args.use_response_hint,
        "output_dir": str(out_dir),
        "note": f"DATA_DIR={out_dir} USE_DSR100_DATA=false HINT_MODE={args.hint_mode} bash safety_opsd_train.sh",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.hint_mode == "easy":
        train_cmd = f"  bash {SAFETY_RL_DIR}/safety_opsd_train_dsr100easy.sh\n"
    else:
        train_cmd = f"  DATA_DIR={out_dir} bash safety_opsd_train.sh\n"
    summary_txt = (
        f"OPSD {tag} dataset ready (hint_mode={args.hint_mode})\n"
        f"  harmful_guidance: {harmful_guidance}\n"
        f"  output: {out_dir}\n"
        f"  harmful kept: {args.harmful} (from {n_safe}/{len(scored)} Guard-safe, pool DSR={dsr_pool:.2%})\n"
        f"  benign kept:  {args.benign}\n"
        f"  train.jsonl:  {len(train_rows)}\n"
        f"  val.jsonl:    {len(val_rows)}\n"
        f"  kept harmful DSR: 100%\n"
        f"\nTrain with:\n"
        f"{train_cmd}"
    )
    (out_dir / "README.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)


if __name__ == "__main__":
    main()
