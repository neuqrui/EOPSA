#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 Student vs Teacher 在回复前 max_tokens 个位置上的 KL 随 token_idx 变化。

对齐 safety_opsd_train.sh (Pure OPSD)：
  Student: chat_template(problem)
  Teacher: chat_template(safety_teacher.jinja(problem, safe_reference))
  在 student rollout 的同一段 response ids 上比较条件分布。

默认指标：teacher top-k 支撑集上重归一化后的
  - forward_kl = KL(T || S)
  - reverse_kl = KL(S || T)
  - jsd（对称，便于对照训练的 topk_jsd）

用法:
  bash run_analyze_prefix_token_kl.sh
  N_SAMPLES=64 MAX_TOKENS=128 CUDA_VISIBLE_DEVICES=6 bash run_analyze_prefix_token_kl.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SAFETY_RL_DIR = SCRIPT_DIR.parent
DEFAULT_DATA = (
    SAFETY_RL_DIR
    / "datasets"
    / "safety_ds_safechain_dsr100_h4400_b2200"
    / "train.jsonl"
)
DEFAULT_TEACHER_TEMPLATE = SAFETY_RL_DIR / "format_prompt" / "safety_teacher.jinja"
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-1.7B")

_JINJA_ENV = jinja2.Environment()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def apply_chat_template(
    text: str,
    tokenizer: Any,
    enable_thinking: bool | None = True,
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


def render_teacher_prompt(template_path: Path, question: str, hint: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = _JINJA_ENV.from_string(f.read())
    return template.render(question=question, hint=hint, problem=question)


def load_samples(path: Path, n: int, seed: int, split_by_type: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rng = np.random.default_rng(seed)
    if not split_by_type:
        if n > 0 and n < len(rows):
            idx = rng.choice(len(rows), size=n, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        return rows

    harmful = [r for r in rows if r.get("data_type") == "safety"]
    benign = [r for r in rows if r.get("data_type") != "safety"]
    # n = 每类采样数；n<=0 时全取（可能很大，慎用）
    out: list[dict[str, Any]] = []
    for group in (harmful, benign):
        if n > 0 and n < len(group):
            idx = rng.choice(len(group), size=n, replace=False)
            out.extend(group[i] for i in sorted(idx))
        else:
            out.extend(group)
    return out


def teacher_topk_divergences(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int,
    eps: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Per-token divergences on teacher top-k renormalized support. logits: [T, V]."""
    k = min(k, teacher_logits.size(-1))
    teacher_logits = teacher_logits.detach()
    student_logits = student_logits.detach()

    teacher_log_full = F.log_softmax(teacher_logits, dim=-1)
    _, topk_indices = torch.topk(teacher_log_full, k=k, dim=-1)

    t_lp = F.log_softmax(teacher_logits.gather(-1, topk_indices), dim=-1)
    s_lp = F.log_softmax(student_logits.gather(-1, topk_indices), dim=-1)
    t_p = t_lp.exp()
    s_p = s_lp.exp()
    mix = (0.5 * (t_p + s_p)).clamp_min(eps)
    mix_lp = mix.log()

    forward_kl = (t_p * (t_lp - s_lp)).sum(dim=-1).clamp(min=0.0, max=20.0)
    reverse_kl = (s_p * (s_lp - t_lp)).sum(dim=-1).clamp(min=0.0, max=20.0)
    jsd = (
        0.5
        * (
            (t_p * (t_lp - mix_lp)).sum(dim=-1)
            + (s_p * (s_lp - mix_lp)).sum(dim=-1)
        )
    ).clamp(min=0.0, max=10.0)

    teacher_argmax = topk_indices[:, 0]
    student_on_teacher_argmax_logprob = s_lp[:, 0]
    return {
        "forward_kl": forward_kl,
        "reverse_kl": reverse_kl,
        "jsd": jsd,
        "teacher_argmax_id": teacher_argmax,
        "student_logp_on_teacher_argmax": student_on_teacher_argmax_logprob,
    }


@torch.no_grad()
def response_logits(
    model: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    full_ids = prompt_ids + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)
    start = prompt_len - 1
    return out.logits[0, start : start + resp_len]


def run_student_rollouts(
    model_path: str,
    samples: list[dict[str, Any]],
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    cache_path: Path,
    force: bool,
) -> list[dict[str, Any]]:
    if cache_path.is_file() and not force:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_qs = [d["problem"] for d in cached]
        want_qs = [s["problem"] for s in samples]
        if cached_qs == want_qs:
            print(f"[Student] 复用缓存: {cache_path}")
            return cached
        print("[Student] 缓存问题不一致，重新 rollout")

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [
        apply_chat_template(
            s["problem"], tokenizer, enable_thinking=enable_thinking, tokenize=False
        )
        for s in samples
    ]
    print(f"[Student] vLLM rollout n={len(prompts)} max_tokens={max_tokens}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")),
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.5")),
    )
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[Student] 完成 {time.time() - t0:.1f}s")

    records = []
    for i, (sample, out) in enumerate(zip(samples, outputs)):
        comp = out.outputs[0]
        records.append({
            "index": i,
            "problem": sample["problem"],
            "safe_reference": sample.get("safe_reference", ""),
            "data_type": sample.get("data_type", ""),
            "safechain_label": sample.get("safechain_label", ""),
            "student_response": comp.text,
            "student_token_ids": list(comp.token_ids),
            "student_token_len": len(comp.token_ids),
        })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[Student] 已保存: {cache_path}")

    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def accumulate_curve(
    curves: dict[str, dict[str, list[float]]],
    group: str,
    metric: str,
    values: np.ndarray,
) -> None:
    bucket = curves.setdefault(group, {}).setdefault(metric, {"sum": [], "count": []})
    # lazy expand
    while len(bucket["sum"]) < len(values):
        bucket["sum"].append(0.0)
        bucket["count"].append(0)
    for i, v in enumerate(values):
        bucket["sum"][i] += float(v)
        bucket["count"][i] += 1


def finalize_curves(curves: dict[str, dict[str, dict[str, list]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, metrics in curves.items():
        out[group] = {}
        for metric, bucket in metrics.items():
            mean = []
            coverage = []
            n = max(bucket["count"]) if bucket["count"] else 0
            for s, c in zip(bucket["sum"], bucket["count"]):
                mean.append(s / c if c > 0 else float("nan"))
                coverage.append(c / n if n else 0.0)
            out[group][metric] = {
                "token_idx": list(range(len(mean))),
                "mean": mean,
                "coverage": coverage,
                "n_max": int(n),
            }
    return out


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_kl_curves(
    curves: dict[str, Any],
    out_path: Path,
    title: str,
    max_tokens: int,
) -> None:
    apply_plot_style()
    metrics = ["forward_kl", "reverse_kl", "jsd"]
    metric_labels = {
        "forward_kl": r"Forward KL $D_{\mathrm{KL}}(T\|S)$",
        "reverse_kl": r"Reverse KL $D_{\mathrm{KL}}(S\|T)$",
        "jsd": "JSD",
    }
    group_style = {
        "all": {"color": "#4C78A8", "ls": "-"},
        "harmful": {"color": "#E45756", "ls": "-"},
        "benign": {"color": "#54A24B", "ls": "--"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
    for ax, metric in zip(axes, metrics):
        for group, style in group_style.items():
            if group not in curves or metric not in curves[group]:
                continue
            data = curves[group][metric]
            x = np.asarray(data["token_idx"][:max_tokens], dtype=float)
            y = np.asarray(data["mean"][:max_tokens], dtype=float)
            cov = np.asarray(data["coverage"][:max_tokens], dtype=float)
            mask = cov >= 0.2
            ax.plot(
                x[mask],
                y[mask],
                color=style["color"],
                linestyle=style["ls"],
                label=f"{group} (n≤{data['n_max']})",
            )
        ax.set_xlabel("Token index")
        ax.set_ylabel(metric_labels[metric])
        ax.set_xlim(0, max_tokens)
        ax.legend(loc="best", frameon=True, framealpha=0.95)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[Plot] {out_path}")


def decode_token(tokenizer: Any, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)], skip_special_tokens=False)
    except Exception:
        return f"<id:{tid}>"


@torch.no_grad()
def analyze(
    model_path: str,
    records: list[dict[str, Any]],
    teacher_template: Path,
    enable_thinking: bool,
    max_tokens: int,
    topk: int,
    high_kl_quantile: float,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[HF] Loading {model_path} dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    raw_curves: dict[str, dict[str, dict[str, list]]] = {}
    # (group, metric) -> Counter of student token string at high-KL positions
    high_kl_student_tokens: dict[str, Counter] = defaultdict(Counter)
    high_kl_teacher_argmax: dict[str, Counter] = defaultdict(Counter)
    # also track position-wise which student tokens appear most often when KL is high
    pos_high_kl_tokens: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))

    # collect all forward_kl for quantile threshold per group
    all_fwd: dict[str, list[float]] = defaultdict(list)
    per_sample_debug: list[dict[str, Any]] = []

    t0 = time.time()
    for rec in tqdm(records, desc="token-KL"):
        resp_ids = list(rec["student_token_ids"])[:max_tokens]
        if not resp_ids:
            continue
        q = rec["problem"]
        hint = rec.get("safe_reference") or ""
        group = "harmful" if rec.get("data_type") == "safety" else "benign"

        stu_prompt = apply_chat_template(q, tokenizer, enable_thinking=enable_thinking, tokenize=True)
        tea_raw = render_teacher_prompt(teacher_template, q, hint)
        tea_prompt = apply_chat_template(
            tea_raw, tokenizer, enable_thinking=enable_thinking, tokenize=True
        )
        if not isinstance(stu_prompt, list):
            stu_prompt = list(stu_prompt)
        if not isinstance(tea_prompt, list):
            tea_prompt = list(tea_prompt)

        stu_logits = response_logits(model, stu_prompt, resp_ids, device)
        tea_logits = response_logits(model, tea_prompt, resp_ids, device)
        t_len = min(stu_logits.size(0), tea_logits.size(0), len(resp_ids))
        if t_len <= 0:
            continue

        div = teacher_topk_divergences(tea_logits[:t_len], stu_logits[:t_len], k=topk)
        fwd = div["forward_kl"].float().cpu().numpy()
        rev = div["reverse_kl"].float().cpu().numpy()
        jsd = div["jsd"].float().cpu().numpy()
        tea_arg = div["teacher_argmax_id"].cpu().numpy()

        for g in ("all", group):
            accumulate_curve(raw_curves, g, "forward_kl", fwd)
            accumulate_curve(raw_curves, g, "reverse_kl", rev)
            accumulate_curve(raw_curves, g, "jsd", jsd)
            all_fwd[g].extend(fwd.tolist())

        # temporary store for second pass of high-KL token stats after threshold known
        per_sample_debug.append({
            "group": group,
            "resp_ids": resp_ids[:t_len],
            "fwd": fwd,
            "tea_arg": tea_arg,
        })

        del stu_logits, tea_logits, div
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    thresholds = {
        g: float(np.quantile(vals, high_kl_quantile)) if vals else 0.0
        for g, vals in all_fwd.items()
    }
    print(f"[High-KL] quantile={high_kl_quantile} thresholds={thresholds}")

    for item in per_sample_debug:
        group = item["group"]
        for i, (tid, fkl, ta) in enumerate(
            zip(item["resp_ids"], item["fwd"], item["tea_arg"])
        ):
            for g in ("all", group):
                thr = thresholds.get(g, 0.0)
                if fkl < thr:
                    continue
                stu_tok = decode_token(tokenizer, tid)
                tea_tok = decode_token(tokenizer, int(ta))
                high_kl_student_tokens[g][stu_tok] += 1
                high_kl_teacher_argmax[g][tea_tok] += 1
                pos_high_kl_tokens[g][i][stu_tok] += 1

    curves = finalize_curves(raw_curves)

    def top_counter(c: Counter, n: int = 30) -> list[dict[str, Any]]:
        return [{"token": t, "count": int(k)} for t, k in c.most_common(n)]

    high_kl_summary = {}
    for g in sorted(set(list(high_kl_student_tokens) + list(high_kl_teacher_argmax))):
        # top tokens at earliest positions that often trigger high KL
        early_pos = {}
        for pos in range(min(32, max_tokens)):
            if pos in pos_high_kl_tokens[g]:
                early_pos[str(pos)] = top_counter(pos_high_kl_tokens[g][pos], n=8)
        high_kl_summary[g] = {
            "forward_kl_threshold": thresholds.get(g, 0.0),
            "student_tokens_at_high_kl": top_counter(high_kl_student_tokens[g]),
            "teacher_argmax_at_high_kl": top_counter(high_kl_teacher_argmax[g]),
            "early_pos_student_tokens": early_pos,
        }

    result = {
        "model_path": model_path,
        "topk": topk,
        "max_tokens": max_tokens,
        "n_samples": len(records),
        "high_kl_quantile": high_kl_quantile,
        "curves": curves,
        "high_kl_tokens": high_kl_summary,
        "elapsed_sec": time.time() - t0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def print_high_kl_preview(result: dict[str, Any]) -> None:
    print("\n========== High-KL token preview (by forward KL) ==========")
    for group, info in result.get("high_kl_tokens", {}).items():
        print(f"\n[{group}] threshold={info['forward_kl_threshold']:.4f}")
        print("  student tokens @ high KL:")
        for row in info["student_tokens_at_high_kl"][:12]:
            print(f"    {row['count']:4d}  {row['token']!r}")
        print("  teacher argmax @ high KL:")
        for row in info["teacher_argmax_at_high_kl"][:12]:
            print(f"    {row['count']:4d}  {row['token']!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefix token-wise Student/Teacher KL analysis")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_PATH", str(DEFAULT_DATA))))
    p.add_argument("--teacher_template", type=Path, default=DEFAULT_TEACHER_TEMPLATE)
    p.add_argument("--n_samples", type=int, default=_env_int("N_SAMPLES", 32),
                   help="每类采样数（harmful/benign 各 n）；总样本约 2n")
    p.add_argument("--max_tokens", type=int, default=_env_int("MAX_TOKENS", 128))
    p.add_argument("--topk", type=int, default=_env_int("DISTILLATION_TOPK", 512))
    p.add_argument("--seed", type=int, default=_env_int("SEED", 42))
    p.add_argument("--temperature", type=float, default=_env_float("STUDENT_TEMPERATURE", 0.6))
    p.add_argument("--top_p", type=float, default=_env_float("STUDENT_TOP_P", 0.95))
    p.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_rollout", action="store_true",
                   default=os.environ.get("FORCE_STUDENT_ROLLOUT", "0") in ("1", "true", "True"))
    p.add_argument("--high_kl_quantile", type=float, default=_env_float("HIGH_KL_QUANTILE", 0.9))
    p.add_argument("--tag", default=os.environ.get("RUN_TAG", "qwen3-4b"))
    p.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "outputs")
    p.add_argument("--fig_dir", type=Path, default=SCRIPT_DIR / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("  Prefix Student–Teacher Token KL")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}={v}")

    samples = load_samples(args.data, args.n_samples, args.seed, split_by_type=True)
    n_h = sum(1 for s in samples if s.get("data_type") == "safety")
    n_b = len(samples) - n_h
    print(f"[Data] loaded {len(samples)} (harmful={n_h}, benign={n_b}) from {args.data}")

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"student_rollouts_max{args.max_tokens}_n{len(samples)}.json"

    records = run_student_rollouts(
        model_path=args.model_path,
        samples=samples,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cache_path=cache_path,
        force=args.force_rollout,
    )

    result = analyze(
        model_path=args.model_path,
        records=records,
        teacher_template=args.teacher_template,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        topk=args.topk,
        high_kl_quantile=args.high_kl_quantile,
    )
    result["tag"] = args.tag
    result["data"] = str(args.data)
    result["n_harmful"] = n_h
    result["n_benign"] = n_b

    result_path = out_dir / f"prefix_token_kl_max{args.max_tokens}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Save] {result_path}")

    fig_path = args.fig_dir / f"{args.tag}_prefix_kl_max{args.max_tokens}.png"
    plot_kl_curves(
        curves=result["curves"],
        out_path=fig_path,
        title=f"{args.tag}  topk={args.topk}  max_tokens={args.max_tokens}",
        max_tokens=args.max_tokens,
    )
    print_high_kl_preview(result)
    print("\n完成。")


if __name__ == "__main__":
    main()
