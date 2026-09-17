#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本 1: Teacher Baseline 推理（仅推理，Guard 评测由独立脚本完成）

使用方式:
    bash run_teacher_baseline.sh
"""

from __future__ import annotations

import gc, json, os, sys, time
from datetime import datetime
from typing import Any

import jinja2, pandas as pd, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.environ.get("EVAL_LLM_SAFETY_DIR", "").strip() or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))),
    "LLM-Safety-Eval",
)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from config import (
    DATASET_PATH, DATASET_LIMIT,
    TEACHER_TEMPLATE, HARMFUL_GUIDANCE,
    TEACHER_SAMPLING_PARAMS, OUTPUT_DIR,
    VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_MODEL_LEN,
    print_config, get_models,
)

OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "teacher_baseline")
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)



def load_wildchat(n: int = DATASET_LIMIT) -> list[str]:
    df = pd.read_csv(DATASET_PATH)
    questions = df["question"].dropna().tolist()[:n]
    print(f"[数据] 加载 WildChat 数据集: {len(questions)} 条")
    return questions


def render_teacher_prompt(question: str, hint: str) -> str:
    env = jinja2.Environment()
    with open(TEACHER_TEMPLATE, "r", encoding="utf-8") as f:
        template_str = f.read()
    template = env.from_string(template_str)
    return template.render(question=question, hint=hint)


def apply_chat_template(prompts, tokenizer, enable_thinking=None, sys_template=""):
    formatted = []
    for prompt_text in prompts:
        messages = []
        if sys_template:
            messages.append({"role": "system", "content": sys_template})
        messages.append({"role": "user", "content": prompt_text})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
    return formatted


def run_model_inference(model_path, model_name, questions, hint, enable_thinking):
    print(f"\n{'='*60}")
    print(f"[推理] 模型: {model_name}  |  路径: {model_path}")
    print(f"{'='*60}")

    raw_prompts = [render_teacher_prompt(q, hint) for q in questions]
    print(f"  -> 渲染了 {len(raw_prompts)} 条 Teacher prompt")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    final_prompts = apply_chat_template(raw_prompts, tokenizer, enable_thinking=enable_thinking)
    print(f"  -> Chat template 应用完成 (enable_thinking={enable_thinking})")

    llm = LLM(
        model=model_path, tensor_parallel_size=1, trust_remote_code=True,
        max_model_len=VLLM_MAX_MODEL_LEN, gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )
    sp = SamplingParams(
        temperature=TEACHER_SAMPLING_PARAMS["temperature"],
        top_p=TEACHER_SAMPLING_PARAMS["top_p"],
        max_tokens=TEACHER_SAMPLING_PARAMS["max_tokens"],
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    t0 = time.time()
    outputs = llm.generate(final_prompts, sp)
    elapsed = time.time() - t0
    responses = [[comp.text.strip() for comp in out.outputs] for out in outputs]
    print(f"  -> 推理完成，耗时 {elapsed:.1f}s，共 {len(responses)} 条")

    results = []
    for i, (q, raw_p, final_p, resps) in enumerate(zip(questions, raw_prompts, final_prompts, responses)):
        results.append({
            "index": i, "question": q, "teacher_raw_prompt": raw_p,
            "teacher_final_prompt": final_p, "response": resps, "model": model_name,
        })

    save_path = os.path.join(OUTPUT_SUBDIR, f"{model_name}_inference.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> 推理结果已保存: {save_path}")
    print(f"  -> Guard 评测请稍后运行: python3 run_teacher_guard_eval.py")

    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    print("=" * 70)
    print("  脚本 1: Teacher Baseline 评测（仅推理）")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print_config()

    questions = load_wildchat(DATASET_LIMIT)

    for model_name, model_path in get_models().items():
        is_qwen3 = "qwen3" in model_name.lower()
        enable_thinking = True if is_qwen3 else None

        run_model_inference(
            model_path=model_path, model_name=model_name,
            questions=questions, hint=HARMFUL_GUIDANCE,
            enable_thinking=enable_thinking,
        )

    print(f"\nGuard 评测请单独运行: python3 run_teacher_guard_eval.py")


if __name__ == "__main__":
    main()
