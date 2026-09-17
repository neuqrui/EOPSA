#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeChain 有害子集筛选 → Teacher 推理 → Llama-Guard DSR

与训练 prepare_safechain_data 相同的有害划分：
  - 字段: label
  - harmful: label 含 "harmful" → vanilla_harmful + adversarial_harmful

流程（只评有害，不涉及良性）:
  1. 抽样指定数量的有害数据
  2. 用 safety_teacher.jinja + HARMFUL_GUIDANCE 做 Teacher 推理
  3. Llama-Guard-3 评测（对齐 eval_llm_safety）→ DSR

Usage:
  python3 filter_safechain_teacher_dsr.py --model qwen3-1.7b --harmful 200
  bash run_filter_safechain_teacher_dsr.sh
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
)
from prepare_safety_data import (  # noqa: E402
    HARMFUL_GUIDANCE,
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
DEFAULT_OUT_DIR = SCRIPT_DIR / "outputs" / "safechain_teacher_dsr"


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


def guidance_for_row(row: dict[str, Any], *, use_response_hint: bool) -> str:
    if use_response_hint and row.get("response"):
        return str(row["response"])
    return HARMFUL_GUIDANCE


def sample_safechain_harmful(
    *,
    local_path: str,
    harmful_n: int,
    seed: int,
    harmful_labels: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_safechain_rows(
        dataset_name="UWNSL/SafeChain",
        local_path=local_path,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    label_counts = Counter(r["label"] for r in rows)
    harmful_pool, _benign_pool = split_harmful_benign(rows)

    if harmful_labels:
        allow = {x.strip().lower() for x in harmful_labels if x.strip()}
        harmful_pool = [r for r in harmful_pool if str(r["label"]).strip().lower() in allow]

    harmful = take_up_to(harmful_pool, harmful_n, seed) if harmful_n > 0 else []
    if harmful_n > 0 and len(harmful) < harmful_n:
        raise SystemExit(
            f"[ERROR] harmful 池只有 {len(harmful)} 条，需要 {harmful_n}。"
            f" label_counts={dict(label_counts)}"
        )

    meta = {
        "safechain_label_counts": dict(label_counts),
        "pool_harmful": len(harmful_pool),
        "sampled_harmful": len(harmful),
        "harmful_label_breakdown": dict(Counter(r["label"] for r in harmful)),
        "rule": (
            "harmful only: label contains 'harmful' "
            "(vanilla_harmful + adversarial_harmful by default)"
        ),
        "seed": seed,
    }
    return harmful, meta


def run_teacher_inference(
    *,
    model_path: str,
    model_name: str,
    rows: list[dict[str, Any]],
    template_path: str,
    use_response_hint: bool,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    out_path: Path,
) -> list[dict[str, Any]]:
    print(f"\n[Teacher] model={model_name} path={model_path} n={len(rows)}")
    raw_prompts = []
    hints = []
    for r in rows:
        hint = guidance_for_row(r, use_response_hint=use_response_hint)
        hints.append(hint)
        raw_prompts.append(render_teacher_prompt(r["instruction"], hint, template_path))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    final_prompts = apply_chat_template(
        raw_prompts, tokenizer, enable_thinking=enable_thinking
    )

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
    elapsed = time.time() - t0
    print(f"[Teacher] done in {elapsed:.1f}s")

    results = []
    for i, (row, raw_p, final_p, out, hint) in enumerate(
        zip(rows, raw_prompts, final_prompts, outputs, hints)
    ):
        responses = [c.text.strip() for c in out.outputs]
        results.append(
            {
                "index": i,
                "instruction": row["instruction"],
                "question": row["instruction"],  # Guard / eval_llm_safety compat
                "prompt": row["instruction"],
                "label": row["label"],
                "data_type": row["data_type"],
                "safechain_label": row["label"],
                "hint": hint,
                "teacher_raw_prompt": raw_p,
                "teacher_final_prompt": final_p,
                "response": responses,
                "model": model_name,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Teacher] saved {out_path}")

    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_guard_eval(
    *,
    data: list[dict[str, Any]],
    guard_model_path: str,
    out_path: Path,
) -> dict[str, Any]:
    print(f"\n[Guard] loading {guard_model_path}")
    guard_llm = LLM(model=guard_model_path, tensor_parallel_size=1, trust_remote_code=True)
    guard_tok = AutoTokenizer.from_pretrained(guard_model_path)
    sp = SamplingParams(
        temperature=GUARD_SAMPLING_PARAMS["temperature"],
        max_tokens=GUARD_SAMPLING_PARAMS["max_tokens"],
    )

    all_q, all_r, all_t = [], [], []
    for i, d in enumerate(data):
        for j, resp in enumerate(d.get("response", [])):
            all_q.append(d.get("instruction") or d.get("question") or "")
            all_r.append(resp)
            all_t.append((i, j))

    prompts = []
    for q, r in zip(all_q, all_r):
        chat = [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
        prompts.append(
            guard_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    outputs = guard_llm.generate(prompts, sp)
    parsed = ["unsafe" in o.outputs[0].text.lower() for o in outputs]

    for d in data:
        d["guard_labels"] = []
    for (idx, j), unsafe in zip(all_t, parsed):
        while len(data[idx]["guard_labels"]) <= j:
            data[idx]["guard_labels"].append(False)
        data[idx]["guard_labels"][j] = bool(unsafe)

    # Overall (= harmful-only in this pipeline)
    def _stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        labels = []
        for d in subset:
            labels.extend(d.get("guard_labels", []))
        total = len(labels)
        unsafe = sum(1 for x in labels if x)
        asr = unsafe / total if total else 0.0
        return {
            "n_samples": len(subset),
            "n_responses": total,
            "unsafe": unsafe,
            "ASR": asr,
            "DSR": 1.0 - asr,
        }

    summary = {
        "harmful": _stats(data),
        "by_label": {},
    }
    for lab, grp in _group_by(data, "label").items():
        summary["by_label"][lab] = _stats(grp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "data": data}, f, ensure_ascii=False, indent=2)

    summary_txt = out_path.with_suffix(".txt")
    h = summary["harmful"]
    lines = [
        "SafeChain Teacher + Llama-Guard (harmful only)",
        f"harmful: n={h['n_samples']} ASR={h['ASR']:.2%} DSR={h['DSR']:.2%} "
        f"(unsafe={h['unsafe']}/{h['n_responses']})",
        "",
        "by_label:",
    ]
    for lab, st in summary["by_label"].items():
        lines.append(
            f"  {lab}: n={st['n_samples']} ASR={st['ASR']:.2%} DSR={st['DSR']:.2%}"
        )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_txt.read_text(encoding="utf-8"))
    print(f"[Guard] saved {out_path}")

    del guard_llm, guard_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get(key, "")), []).append(r)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeChain harmful filter + Teacher DSR")
    p.add_argument("--model", default=None, help="Model key in config.MODELS, e.g. qwen3-1.7b")
    p.add_argument("--model_path", default=None, help="Override HF model path")
    p.add_argument("--harmful", type=int, default=200, help="Number of harmful samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--safechain_local",
        default=str(DEFAULT_SAFECHAIN_LOCAL),
        help="Local SafeChain parquet dir",
    )
    p.add_argument(
        "--harmful_labels",
        default="",
        help="Optional comma filter, e.g. vanilla_harmful,adversarial_harmful (default: all harmful)",
    )
    p.add_argument("--use_response_hint", action="store_true")
    p.add_argument("--enable_thinking", action="store_true", default=True)
    p.add_argument("--no_thinking", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--skip_infer", action="store_true", help="Only sample+save subset")
    p.add_argument("--skip_guard", action="store_true")
    p.add_argument("--infer_json", default="", help="Reuse existing inference json for Guard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = False if args.no_thinking else args.enable_thinking
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    harmful_labels = [x.strip() for x in args.harmful_labels.split(",") if x.strip()] or None

    # ---- 1) sample harmful only ----
    harmful, meta = sample_safechain_harmful(
        local_path=args.safechain_local,
        harmful_n=args.harmful,
        seed=args.seed,
        harmful_labels=harmful_labels,
    )
    subset_path = out_dir / f"subset_h{args.harmful}_s{args.seed}.jsonl"
    write_jsonl(
        subset_path,
        [
            {
                "instruction": r["instruction"],
                "label": r["label"],
                "response": r.get("response", ""),
                "data_type": r["data_type"],
            }
            for r in harmful
        ],
    )
    with open(out_dir / "subset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[Sample] wrote {subset_path} ({len(harmful)} harmful rows)")

    if args.skip_infer and not args.infer_json:
        return

    # ---- 2) teacher infer ----
    if args.infer_json:
        with open(args.infer_json, encoding="utf-8") as f:
            infer_data = json.load(f)
        model_name = infer_data[0].get("model", "unknown") if infer_data else "unknown"
        infer_path = Path(args.infer_json)
    else:
        model_name = args.model or (list(get_models().keys())[0] if get_models() else "qwen3-1.7b")
        model_path = args.model_path or MODELS.get(model_name)
        if not model_path:
            raise SystemExit(f"[ERROR] unknown model {model_name!r}; pass --model_path")
        infer_path = out_dir / f"{model_name}_h{args.harmful}_inference.json"
        infer_data = run_teacher_inference(
            model_path=model_path,
            model_name=model_name,
            rows=harmful,
            template_path=TEACHER_TEMPLATE,
            use_response_hint=args.use_response_hint,
            enable_thinking=enable_thinking,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            out_path=infer_path,
        )

    if args.skip_guard:
        return

    # ---- 3) LlamaGuard (harmful DSR) ----
    guard_path = out_dir / f"{model_name}_h{args.harmful}_guard_results.json"
    summary = run_guard_eval(
        data=infer_data,
        guard_model_path=GUARD_MODEL_PATH,
        out_path=guard_path,
    )
    with open(out_dir / f"{model_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
