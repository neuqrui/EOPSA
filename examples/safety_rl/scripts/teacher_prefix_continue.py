#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher-on-Student-Prefix 续写实验

流程（每个模型）:
  1. Student: 仅看到问题本身，rollout（max_tokens=8192）；结果按数据集落盘可复用
  2. LlamaGuard 评 Student；只保留 unsafe，抽样 TARGET_UNSAFE_N（默认 100）条
  3. 对每个前缀长度 L∈PREFIX_LENGTHS:
       若 student_token_len <= L：跳过（汇总时计为教师改写失败）
       否则 teacher 在学生前缀上续写
  4. Guard 评测 Teacher full / continuation；主指标为
       Rescue(L) = P(teacher full safe | student unsafe ∧ teacher alone safe)
       teacher alone = L=0；长度不够 / 未跑教师 → 计失败

数据集: 直接复用 eval_llm_safety（config.yaml + utils.data_handler.load_and_prep_data）
         DATASET_NAME=wildchat|wildjailbreak|...

使用方式:
    DATASET_NAME=wildchat bash run_teacher_prefix_continue.sh
    DATASET_NAME=wildjailbreak bash run_teacher_prefix_continue.sh
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

import jinja2
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import (
    DATASET_LIMIT,
    DATASET_NAME,
    EVAL_LLM_SAFETY_DIR,
    EVAL_LLM_SAFETY_CONFIG,
    TEACHER_TEMPLATE,
    HARMFUL_GUIDANCE,
    STUDENT_SAMPLING_PARAMS,
    TEACHER_SAMPLING_PARAMS,
    GUARD_MODEL_PATH,
    GUARD_SAMPLING_PARAMS,
    OUTPUT_DIR,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    PREFIX_LENGTHS,
    FORCE_STUDENT_ROLLOUT,
    TARGET_UNSAFE_N,
    SAMPLE_SEED,
    PREFIX_SOFT_MIN_LEN,
    print_config,
    get_models,
)

OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "teacher_prefix_continue", DATASET_NAME)
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

_JINJA_ENV = jinja2.Environment()


def cleanup_vllm() -> None:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def load_eval_questions(n: int = DATASET_LIMIT) -> list[str]:
    """
    直接调用 eval_llm_safety.utils.data_handler.load_and_prep_data，
    数据集定义与路径全部来自 eval_llm_safety/config.yaml。
    """
    eval_root = os.path.abspath(EVAL_LLM_SAFETY_DIR)
    cfg_path = os.path.abspath(EVAL_LLM_SAFETY_CONFIG)
    if not os.path.isdir(eval_root):
        raise FileNotFoundError(f"eval_llm_safety 目录不存在: {eval_root}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"eval_llm_safety config 不存在: {cfg_path}")

    # 保证可 import utils.data_handler
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)

    from utils.data_handler import load_and_prep_data

    try:
        from omegaconf import OmegaConf
        full_cfg = OmegaConf.load(cfg_path)
        datasets_cfg = full_cfg.datasets
        if DATASET_NAME not in datasets_cfg:
            available = list(datasets_cfg.keys())
            raise ValueError(
                f"DATASET_NAME={DATASET_NAME!r} 不在 eval_llm_safety config 中，可选: {available}"
            )
        ds_cfg = OmegaConf.to_container(datasets_cfg[DATASET_NAME], resolve=True)
    except ImportError:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        datasets_cfg = full_cfg.get("datasets") or {}
        if DATASET_NAME not in datasets_cfg:
            raise ValueError(
                f"DATASET_NAME={DATASET_NAME!r} 不在 eval_llm_safety config 中，"
                f"可选: {list(datasets_cfg.keys())}"
            )
        ds_cfg = dict(datasets_cfg[DATASET_NAME])

    assert isinstance(ds_cfg, dict)
    # 用本实验的 limit 覆盖 config 里的 limit_num（-1=全部）
    ds_cfg["limit_num"] = n

    # data_handler 里相对路径 / HF cache 都相对 eval_llm_safety 根目录
    prev_cwd = os.getcwd()
    os.chdir(eval_root)
    try:
        prompts, _metadata = load_and_prep_data(DATASET_NAME, ds_cfg)
    finally:
        os.chdir(prev_cwd)

    questions = [str(p) for p in prompts if p is not None and str(p).strip()]
    print(
        f"[数据] eval_llm_safety/{DATASET_NAME}: {len(questions)} 条  "
        f"limit={n}  config={cfg_path}"
    )
    return questions


def render_teacher_prompt(question: str, hint: str) -> str:
    with open(TEACHER_TEMPLATE, "r", encoding="utf-8") as f:
        template = _JINJA_ENV.from_string(f.read())
    return template.render(question=question, hint=hint, problem=question)


def apply_chat_template(
    prompts: list[str],
    tokenizer: Any,
    enable_thinking: bool | None = None,
    tokenize: bool = False,
) -> list[Any]:
    formatted = []
    for text in prompts:
        messages = [{"role": "user", "content": text}]
        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
    return formatted


def student_cache_path(model_name: str) -> str:
    return os.path.join(OUTPUT_SUBDIR, f"{model_name}_student_rollouts.json")


def pipeline_path(model_name: str) -> str:
    return os.path.join(OUTPUT_SUBDIR, f"{model_name}_prefix_pipeline.json")


def load_or_run_student(
    model_name: str,
    model_path: str,
    questions: list[str],
    enable_thinking: bool | None,
) -> tuple[list[dict[str, Any]], Any, Any]:
    """
    返回 (student_records, llm, tokenizer)。
    若缓存命中则 llm=None（需后续单独加载做教师推理）；缓存未命中则复用已加载的 llm。
    """
    cache_path = student_cache_path(model_name)
    if (
        not FORCE_STUDENT_ROLLOUT
        and os.path.isfile(cache_path)
    ):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_qs = [d["question"] for d in cached]
        if cached_qs == questions:
            print(f"[阶段 1] 复用已有 Student 结果: {cache_path}  ({len(cached)} 条)")
            return cached, None, None
        print(
            f"[阶段 1] 缓存问题与当前数据不一致 "
            f"(cache={len(cached_qs)} vs now={len(questions)})，重新推理"
        )

    print(f"[阶段 1] Student 推理 (max_tokens={STUDENT_SAMPLING_PARAMS['max_tokens']}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 学生只看到原始问题（对应 safety_student.jinja）
    student_prompts = apply_chat_template(
        list(questions), tokenizer, enable_thinking=enable_thinking, tokenize=False
    )

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
    outputs = llm.generate(student_prompts, sp)
    elapsed = time.time() - t0

    records: list[dict[str, Any]] = []
    for i, (q, out) in enumerate(zip(questions, outputs)):
        comp = out.outputs[0]
        token_ids = list(comp.token_ids)
        records.append({
            "index": i,
            "question": q,
            "student_response": comp.text,
            "student_token_ids": token_ids,
            "student_token_len": len(token_ids),
            "model": model_name,
        })

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  → Student 完成 {elapsed:.1f}s，已保存: {cache_path}")
    print(
        f"  → token_len 分布: "
        f"min={min(r['student_token_len'] for r in records)} "
        f"median={sorted(r['student_token_len'] for r in records)[len(records)//2]} "
        f"max={max(r['student_token_len'] for r in records)}"
    )
    return records, llm, tokenizer


def filter_by_soft_min_len(
    student_records: list[dict[str, Any]],
    min_len: int,
) -> list[dict[str, Any]]:
    if min_len <= 0:
        return list(student_records)
    filtered = [r for r in student_records if r["student_token_len"] > min_len]
    print(
        f"[筛选] student_token_len > {min_len} (soft): "
        f"{len(filtered)}/{len(student_records)} 条"
    )
    return filtered


def guard_student_records(student_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run LlamaGuard on student responses; attach student_guard_unsafe."""
    if not student_records:
        return student_records
    if all("student_guard_unsafe" in r for r in student_records):
        n_u = sum(1 for r in student_records if r["student_guard_unsafe"])
        print(f"[Guard-S] 已有 student_guard_unsafe，跳过 ({n_u}/{len(student_records)} unsafe)")
        return student_records

    print(f"[Guard-S] 加载 Guard: {GUARD_MODEL_PATH}")
    guard_llm = LLM(
        model=GUARD_MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    guard_tok = AutoTokenizer.from_pretrained(GUARD_MODEL_PATH)
    sp = SamplingParams(
        temperature=GUARD_SAMPLING_PARAMS["temperature"],
        max_tokens=GUARD_SAMPLING_PARAMS["max_tokens"],
    )
    prompts = []
    for r in student_records:
        chat = [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r.get("student_response", "")},
        ]
        prompts.append(
            guard_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )
    print(f"[Guard-S] 评测 Student {len(prompts)} 条 ...")
    outputs = guard_llm.generate(prompts, sp)
    for r, out in zip(student_records, outputs):
        r["student_guard_unsafe"] = "unsafe" in out.outputs[0].text.lower()

    del guard_llm, guard_tok
    gc.collect()
    cleanup_vllm()

    n_u = sum(1 for r in student_records if r["student_guard_unsafe"])
    print(f"[Guard-S] unsafe={n_u}/{len(student_records)} ({n_u/len(student_records):.1%})")
    return student_records


def sample_unsafe_records(
    student_records: list[dict[str, Any]],
    *,
    target_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    unsafe = [r for r in student_records if r.get("student_guard_unsafe")]
    print(f"[抽样] student unsafe 池: {len(unsafe)} 条，目标 n={target_n} (seed={seed})")
    if len(unsafe) < target_n:
        raise RuntimeError(
            f"student unsafe 仅 {len(unsafe)} < TARGET_UNSAFE_N={target_n}。"
            f"请增大 DATASET_LIMIT / 降低 PREFIX_SOFT_MIN_LEN，或减小 TARGET_UNSAFE_N。"
        )
    rng = random.Random(seed)
    picked = unsafe[:]
    rng.shuffle(picked)
    picked = picked[:target_n]
    # stable order by original index for reproducibility in logs
    picked.sort(key=lambda r: int(r.get("index", 0)))
    lens = [r["student_token_len"] for r in picked]
    print(
        f"[抽样] 取定 {len(picked)} 条  "
        f"token_len min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}"
    )
    return picked


def build_teacher_prefix_batch(
    filtered: list[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool | None,
    prefix_lengths: list[int],
) -> tuple[list[dict[str, Any]], list[SamplingParams], list[tuple[int, int]]]:
    """
    一次性构造所有 (sample, prefix_len) 的 teacher continuation 请求。

    拼接逻辑（与 OPSD 训练一致）:
      teacher_user = safety_teacher.jinja(question, HARMFUL_GUIDANCE)
      teacher_prompt_ids = chat_template(teacher_user, add_generation_prompt=True)
      prompt_token_ids = teacher_prompt_ids + student_token_ids[:L]

    若 student_token_len <= L：跳过该 (sample, L)，汇总时计为教师失败。
    """
    vllm_inputs: list[dict[str, Any]] = []
    sampling_list: list[SamplingParams] = []
    meta: list[tuple[int, int]] = []  # (sample_idx_in_filtered, prefix_len)
    n_skip_short = 0

    teacher_raws = [
        render_teacher_prompt(r["question"], HARMFUL_GUIDANCE) for r in filtered
    ]
    teacher_prompt_ids_list = apply_chat_template(
        teacher_raws, tokenizer, enable_thinking=enable_thinking, tokenize=True
    )

    for sample_i, (rec, teacher_ids) in enumerate(zip(filtered, teacher_prompt_ids_list)):
        if not isinstance(teacher_ids, list):
            teacher_ids = list(teacher_ids)
        student_ids = list(rec["student_token_ids"])
        stu_len = int(rec["student_token_len"])
        for L in prefix_lengths:
            if stu_len <= L:
                n_skip_short += 1
                continue
            prefix_ids = student_ids[:L]
            prompt_ids = list(teacher_ids) + prefix_ids
            # 剩余生成预算：不超过教师 max_tokens，且不超过上下文窗口
            remain_ctx = VLLM_MAX_MODEL_LEN - len(prompt_ids) - 8
            max_new = max(1, min(TEACHER_SAMPLING_PARAMS["max_tokens"], remain_ctx))
            if remain_ctx < 1:
                print(
                    f"  [WARN] sample={sample_i} L={L} prompt 过长 "
                    f"({len(prompt_ids)}), 跳过(计失败)"
                )
                continue
            vllm_inputs.append({"prompt_token_ids": prompt_ids})
            sampling_list.append(
                SamplingParams(
                    temperature=TEACHER_SAMPLING_PARAMS["temperature"],
                    top_p=TEACHER_SAMPLING_PARAMS["top_p"],
                    max_tokens=max_new,
                    stop_token_ids=(
                        [tokenizer.eos_token_id]
                        if tokenizer.eos_token_id is not None
                        else None
                    ),
                )
            )
            meta.append((sample_i, L))

    print(
        f"[阶段 2] 构造 Teacher 续写请求: {len(vllm_inputs)} 条 "
        f"({len(filtered)} samples × {len(prefix_lengths)} prefixes; "
        f"skip_short={n_skip_short})"
    )
    return vllm_inputs, sampling_list, meta


def run_teacher_continuations(
    llm: Any,
    tokenizer: Any,
    filtered: list[dict[str, Any]],
    enable_thinking: bool | None,
    prefix_lengths: list[int],
) -> list[dict[str, Any]]:
    vllm_inputs, sampling_list, meta = build_teacher_prefix_batch(
        filtered, tokenizer, enable_thinking, prefix_lengths
    )

    # 组装结果骨架（缺 L 的 teacher_by_prefix 表示长度不够/跳过 → 计失败）
    results: list[dict[str, Any]] = []
    for rec in filtered:
        entry = {
            "index": rec["index"],
            "question": rec["question"],
            "student_response": rec["student_response"],
            "student_token_len": rec["student_token_len"],
            "student_guard_unsafe": True,  # fixed unsafe eval set
            "teacher_by_prefix": {},
        }
        results.append(entry)

    if not vllm_inputs:
        print("  [WARN] 无 Teacher 续写请求（全部因长度不够被跳过？）")
        return results

    t0 = time.time()
    outputs = llm.generate(vllm_inputs, sampling_list)
    elapsed = time.time() - t0
    print(f"  → Teacher 续写完成 {elapsed:.1f}s")

    for (sample_i, L), out in zip(meta, outputs):
        comp = out.outputs[0]
        prefix_ids = list(filtered[sample_i]["student_token_ids"][:L])
        cont_ids = list(comp.token_ids)
        # 联合 decode，避免分段 decode 的边界伪影
        full_ids = prefix_ids + cont_ids
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        results[sample_i]["teacher_by_prefix"][str(L)] = {
            "prefix_len": L,
            "prefix_text": prefix_text,
            "continuation": comp.text,
            "continuation_token_len": len(cont_ids),
            "full_response": full_text,
            "prompt_token_len": len(out.prompt_token_ids) if out.prompt_token_ids else None,
        }

    return results


def run_pipeline_for_model(
    model_name: str,
    model_path: str,
    questions: list[str],
    enable_thinking: bool | None,
) -> dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"  Pipeline: {model_name}")
    print(f"{'=' * 70}")

    prefix_lengths = list(PREFIX_LENGTHS)
    if not prefix_lengths:
        raise ValueError("PREFIX_LENGTHS 为空")

    student_records, llm, tokenizer = load_or_run_student(
        model_name, model_path, questions, enable_thinking
    )
    # 释放 Student 模型，给 Guard / 后续 Teacher 腾显存
    if llm is not None:
        del llm, tokenizer
        cleanup_vllm()
        llm, tokenizer = None, None

    pool = filter_by_soft_min_len(student_records, PREFIX_SOFT_MIN_LEN)
    pool = guard_student_records(pool)
    # 写回 rollouts 缓存里的 guard 标记，便于复用
    guard_map = {int(r["index"]): bool(r["student_guard_unsafe"]) for r in pool}
    for r in student_records:
        if int(r["index"]) in guard_map:
            r["student_guard_unsafe"] = guard_map[int(r["index"])]
    cache_path = student_cache_path(model_name)
    # student cache may contain token_ids — rewrite only if file exists and we added flags
    try:
        if os.path.isfile(cache_path):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(student_records, f, ensure_ascii=False)
            print(f"[Guard-S] 已写回 student_guard_unsafe → {cache_path}")
    except Exception as e:
        print(f"[Guard-S] 写回 cache 失败（忽略）: {e}")

    try:
        filtered_src = sample_unsafe_records(
            pool, target_n=TARGET_UNSAFE_N, seed=SAMPLE_SEED
        )
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return {
            "model": model_name,
            "filtered": 0,
            "prefixes": prefix_lengths,
            "student_total": len(student_records),
        }

    print(f"[阶段 2] 加载模型用于 Teacher 续写: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )

    pipeline_results = run_teacher_continuations(
        llm, tokenizer, filtered_src, enable_thinking, prefix_lengths
    )

    del llm, tokenizer
    cleanup_vllm()

    save_obj = {
        "model": model_name,
        "dataset": DATASET_NAME,
        "prefix_lengths": prefix_lengths,
        "target_unsafe_n": TARGET_UNSAFE_N,
        "sample_seed": SAMPLE_SEED,
        "prefix_soft_min_len": PREFIX_SOFT_MIN_LEN,
        "short_prefix_as_fail": True,
        "metric": "trr",
        "metric_note": (
            "Rescue(L)=P(teacher_full_safe|student_unsafe ∧ teacher_alone_safe); "
            "teacher_alone=L=0; student_token_len<=L or missing teacher_by_prefix[L] counts as fail"
        ),
        "eval_set": "student_unsafe_and_teacher_alone_safe",
        "n_questions": len(questions),
        "n_student_rollouts": len(student_records),
        "n_filtered": len(pipeline_results),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples": pipeline_results,
    }
    save_path = pipeline_path(model_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)
    print(f"  → Pipeline 结果已保存: {save_path}")
    print(f"  → Guard 评测: python3 run_teacher_prefix_guard_eval.py")

    return {
        "model": model_name,
        "filtered": len(pipeline_results),
        "prefixes": prefix_lengths,
        "student_total": len(student_records),
    }


def main() -> None:
    print("=" * 70)
    print("  Teacher-on-Student-Prefix 续写实验")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print_config()

    questions = load_eval_questions(DATASET_LIMIT)
    summaries: list[dict[str, Any]] = []

    for model_name, model_path in get_models().items():
        is_qwen3 = "qwen3" in model_name.lower()
        enable_thinking = True if is_qwen3 else None
        summary = run_pipeline_for_model(
            model_name=model_name,
            model_path=model_path,
            questions=questions,
            enable_thinking=enable_thinking,
        )
        summaries.append(summary)

    print("\n" + "=" * 70)
    print("  汇总（推理阶段）")
    print("=" * 70)
    for s in summaries:
        print(
            f"  {s['model']:15s}  student={s.get('student_total', '?'):>4}  "
            f"filtered={s['filtered']:>4}  prefixes={s['prefixes']}"
        )
    print(f"\nGuard 评测请运行: python3 run_teacher_prefix_guard_eval.py")


if __name__ == "__main__":
    main()
