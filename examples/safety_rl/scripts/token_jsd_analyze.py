#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student vs Teacher 按 token 位置的 Top-K JSD 分析。

与 safety_opsd_train.sh (distillation_loss_type=topk_jsd, topk=512) 一致：
  1) 用 Teacher logits 取 top-k 支撑集
  2) 在该支撑集上对 Teacher/Student logits 各自 softmax 重归一化
  3) 计算对称 Jensen–Shannon divergence

序列构造（与 OPSD 训练一致）：
  Student: chat_template(question) + student_response_ids
  Teacher: chat_template(safety_teacher.jinja(question, HARMFUL_GUIDANCE)) + student_response_ids
  在「学生 rollout 回复」的每个 token 位置上比较条件分布。

Student rollout 优先复用 teacher_prefix_continue 缓存；否则用 vLLM 推理一次。

使用:
    bash run_token_jsd.sh
    MODELS_TO_RUN=qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b DATASET_NAME=wildchat DISTILLATION_TOPK=512 bash run_token_jsd.sh
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import jinja2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import (
    DATASET_LIMIT,
    DATASET_NAME,
    EVAL_LLM_SAFETY_DIR,
    EVAL_LLM_SAFETY_CONFIG,
    TEACHER_TEMPLATE,
    HARMFUL_GUIDANCE,
    STUDENT_SAMPLING_PARAMS,
    OUTPUT_DIR,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    FORCE_STUDENT_ROLLOUT,
    print_config,
    get_models,
    _env_int,
)

DISTILLATION_TOPK = _env_int("DISTILLATION_TOPK", 512)
JSD_MAX_TOKENS = _env_int("JSD_MAX_TOKENS", 1024)  # 分析用回复截断；-1=不截断
JSD_MIN_COVERAGE = float(os.environ.get("JSD_MIN_COVERAGE", "0.2"))

OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "token_jsd", DATASET_NAME)
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
STUDENT_CACHE_DIR = os.path.join(OUTPUT_DIR, "teacher_prefix_continue", DATASET_NAME)

_JINJA_ENV = jinja2.Environment()


def load_eval_questions(n: int = DATASET_LIMIT) -> list[str]:
    eval_root = os.path.abspath(EVAL_LLM_SAFETY_DIR)
    cfg_path = os.path.abspath(EVAL_LLM_SAFETY_CONFIG)
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)
    from utils.data_handler import load_and_prep_data

    try:
        from omegaconf import OmegaConf
        full_cfg = OmegaConf.load(cfg_path)
        ds_cfg = OmegaConf.to_container(full_cfg.datasets[DATASET_NAME], resolve=True)
    except ImportError:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        ds_cfg = dict(full_cfg["datasets"][DATASET_NAME])

    assert isinstance(ds_cfg, dict)
    ds_cfg["limit_num"] = n
    prev = os.getcwd()
    os.chdir(eval_root)
    try:
        prompts, _ = load_and_prep_data(DATASET_NAME, ds_cfg)
    finally:
        os.chdir(prev)
    questions = [str(p) for p in prompts if p is not None and str(p).strip()]
    print(f"[数据] {DATASET_NAME}: {len(questions)} 条")
    return questions


def render_teacher_prompt(question: str, hint: str) -> str:
    with open(TEACHER_TEMPLATE, "r", encoding="utf-8") as f:
        template = _JINJA_ENV.from_string(f.read())
    return template.render(question=question, hint=hint, problem=question)


def apply_chat_template(
    text: str,
    tokenizer: Any,
    enable_thinking: bool | None = None,
    tokenize: bool = True,
) -> Any:
    messages = [{"role": "user", "content": text}]
    kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def teacher_topk_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    与 verl.trainer.opsd_algos.compute_opsd_teacher_topk_jsd_renorm_loss 同构的 per-token JSD。
    logits: [T, V]  ->  jsd: [T]
    """
    k = min(k, teacher_logits.size(-1))
    teacher_logits = teacher_logits.detach()
    teacher_log_softmax = F.log_softmax(teacher_logits, dim=-1)
    _, topk_indices = torch.topk(teacher_log_softmax, k=k, dim=-1)

    teacher_topk_logits = teacher_logits.gather(-1, topk_indices)
    student_topk_logits = student_logits.gather(-1, topk_indices)

    teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
    student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)

    teacher_probs = teacher_topk_log_probs.exp()
    student_probs = student_topk_log_probs.exp()
    mix_probs = 0.5 * (teacher_probs + student_probs)
    mix_log_probs = mix_probs.clamp_min(eps).log()

    jsd = 0.5 * (
        (teacher_probs * (teacher_topk_log_probs - mix_log_probs)).sum(dim=-1)
        + (student_probs * (student_topk_log_probs - mix_log_probs)).sum(dim=-1)
    )
    return jsd.clamp(min=0.0, max=10.0)


def student_cache_path(model_name: str) -> str:
    return os.path.join(STUDENT_CACHE_DIR, f"{model_name}_student_rollouts.json")


def jsd_result_path(model_name: str) -> str:
    return os.path.join(OUTPUT_SUBDIR, f"{model_name}_token_jsd.json")


def load_or_run_student_rollouts(
    model_name: str,
    model_path: str,
    questions: list[str],
    enable_thinking: bool | None,
) -> list[dict[str, Any]]:
    cache = student_cache_path(model_name)
    if not FORCE_STUDENT_ROLLOUT and os.path.isfile(cache):
        with open(cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_qs = [d["question"] for d in cached]
        # 允许用更少的问题子集匹配前缀
        if cached_qs[: len(questions)] == questions:
            print(f"[Student] 复用缓存: {cache} ({len(questions)}/{len(cached)} 条)")
            return cached[: len(questions)]
        if cached_qs == questions:
            print(f"[Student] 复用缓存: {cache}")
            return cached
        print("[Student] 缓存问题不一致，重新 rollout")

    from vllm import LLM, SamplingParams

    print(f"[Student] vLLM rollout max_tokens={STUDENT_SAMPLING_PARAMS['max_tokens']}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [
        apply_chat_template(q, tokenizer, enable_thinking=enable_thinking, tokenize=False)
        for q in questions
    ]
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )
    sp = SamplingParams(
        temperature=STUDENT_SAMPLING_PARAMS["temperature"],
        top_p=STUDENT_SAMPLING_PARAMS["top_p"],
        max_tokens=STUDENT_SAMPLING_PARAMS["max_tokens"],
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[Student] 完成 {time.time() - t0:.1f}s")

    records = []
    for i, (q, out) in enumerate(zip(questions, outputs)):
        comp = out.outputs[0]
        records.append({
            "index": i,
            "question": q,
            "student_response": comp.text,
            "student_token_ids": list(comp.token_ids),
            "student_token_len": len(comp.token_ids),
            "model": model_name,
        })
    os.makedirs(STUDENT_CACHE_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[Student] 已保存: {cache}")

    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


@torch.no_grad()
def response_logits(
    model: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    返回预测 response 每个 token 的 logits，形状 [resp_len, V]。
    Causal LM: 位置 prompt_len-1 + t 的 logit 预测 response[t]。
    """
    full_ids = prompt_ids + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0]  # [L, V]
    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)
    # logits[prompt_len-1 : prompt_len-1+resp_len]
    start = prompt_len - 1
    return logits[start : start + resp_len]


def compute_token_jsd_for_model(
    model_name: str,
    model_path: str,
    questions: list[str],
    student_records: list[dict[str, Any]],
    enable_thinking: bool | None,
    topk: int,
) -> dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"  Token JSD: {model_name}  topk={topk}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[HF] Loading {model_path} dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    sum_jsd: dict[int, float] = {}
    count_jsd: dict[int, int] = {}
    per_sample_lens: list[int] = []

    t0 = time.time()
    for rec in tqdm(student_records, desc=f"JSD-{model_name}"):
        q = rec["question"]
        resp_ids = list(rec["student_token_ids"])
        if JSD_MAX_TOKENS > 0:
            resp_ids = resp_ids[:JSD_MAX_TOKENS]
        if not resp_ids:
            continue

        stu_prompt = apply_chat_template(
            q, tokenizer, enable_thinking=enable_thinking, tokenize=True
        )
        tea_raw = render_teacher_prompt(q, HARMFUL_GUIDANCE)
        tea_prompt = apply_chat_template(
            tea_raw, tokenizer, enable_thinking=enable_thinking, tokenize=True
        )
        if not isinstance(stu_prompt, list):
            stu_prompt = list(stu_prompt)
        if not isinstance(tea_prompt, list):
            tea_prompt = list(tea_prompt)

        stu_logits = response_logits(model, stu_prompt, resp_ids, device)
        tea_logits = response_logits(model, tea_prompt, resp_ids, device)
        # align in case of slight length mismatch
        t_len = min(stu_logits.size(0), tea_logits.size(0), len(resp_ids))
        if t_len <= 0:
            continue
        jsd = teacher_topk_jsd(tea_logits[:t_len], stu_logits[:t_len], k=topk)
        jsd_cpu = jsd.float().cpu().numpy()
        per_sample_lens.append(t_len)
        for i, v in enumerate(jsd_cpu):
            sum_jsd[i] = sum_jsd.get(i, 0.0) + float(v)
            count_jsd[i] = count_jsd.get(i, 0) + 1

        del stu_logits, tea_logits, jsd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    max_pos = max(count_jsd.keys()) + 1 if count_jsd else 0
    n_samples = len(student_records)
    mean_jsd = []
    coverage = []
    for i in range(max_pos):
        c = count_jsd.get(i, 0)
        mean_jsd.append(sum_jsd[i] / c if c > 0 else float("nan"))
        coverage.append(c / n_samples if n_samples else 0.0)

    # 可选：覆盖率过低的尾部截断（绘图前还可再裁）
    result = {
        "model": model_name,
        "model_path": model_path,
        "dataset": DATASET_NAME,
        "topk": topk,
        "n_samples": n_samples,
        "jsd_max_tokens": JSD_MAX_TOKENS,
        "mean_response_len": float(np.mean(per_sample_lens)) if per_sample_lens else 0.0,
        "token_idx": list(range(max_pos)),
        "mean_jsd": mean_jsd,
        "coverage": coverage,
        "elapsed_sec": elapsed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    save_path = jsd_result_path(model_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → 已保存: {save_path}  ({elapsed:.1f}s, max_pos={max_pos})")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    print("=" * 70)
    print("  Student–Teacher Token-wise Top-K JSD")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print_config()
    print(f"  DISTILLATION_TOPK = {DISTILLATION_TOPK}")
    print(f"  JSD_MAX_TOKENS    = {JSD_MAX_TOKENS}")
    print(f"  OUTPUT_SUBDIR     = {OUTPUT_SUBDIR}")

    questions = load_eval_questions(DATASET_LIMIT)
    force_jsd = os.environ.get("FORCE_JSD", "0").strip() in ("1", "true", "True", "yes")

    for model_name, model_path in get_models().items():
        out_path = jsd_result_path(model_name)
        if not force_jsd and os.path.isfile(out_path):
            print(f"[skip] 已有结果 {out_path} (FORCE_JSD=1 可重算)")
            continue

        is_qwen3 = "qwen3" in model_name.lower()
        enable_thinking = True if is_qwen3 else None
        student_records = load_or_run_student_rollouts(
            model_name, model_path, questions, enable_thinking
        )
        compute_token_jsd_for_model(
            model_name=model_name,
            model_path=model_path,
            questions=questions,
            student_records=student_records,
            enable_thinking=enable_thinking,
            topk=DISTILLATION_TOPK,
        )

    print("\n完成。绘图: python3 plot/plot_token_jsd.py")


if __name__ == "__main__":
    main()
