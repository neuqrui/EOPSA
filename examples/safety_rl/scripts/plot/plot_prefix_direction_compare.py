#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前缀续写绘图（一次运行出三类图）：

  1) 每模型方向对比图
       Tea←Stu vs Stu←Tea
       → prefix_direction_compare_{dataset}_{model}.{pdf,png}

  2) 学生前缀 → 教师续写：四模型同图
       Rescue = P(teacher full safe | student unsafe ∧ teacher alone safe)
       （长度不够 / 未跑教师 → 计失败；L=0 基线应为 100%）
       → prefix_tea_on_stu_{dataset}.{pdf,png}

  3) 教师前缀 → 学生续写：四模型同图
       Hold = P(student full safe | teacher safe ∧ student alone unsafe)
       （长度不够 → 计防守成功；未跑学生 → 计失败）
       → prefix_stu_on_tea_{dataset}.{pdf,png}

Usage:
    python3 plot/plot_prefix_direction_compare.py --dataset wildchat
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

MODEL_ORDER = ["qwen3-1.7b", "qwen3-4b", "ds-1.5b", "ds-7b"]
MODEL_LABELS = {
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "ds-1.5b": "R1-Distill-1.5B",
    "ds-7b": "R1-Distill-7B",
}
MODEL_STYLE = {
    "qwen3-1.7b": {"color": "#0072B2", "marker": "o"},
    "qwen3-4b":   {"color": "#D55E00", "marker": "s"},
    "ds-1.5b":    {"color": "#009E73", "marker": "^"},
    "ds-7b":      {"color": "#E69F00", "marker": "D"},
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def load_tea_on_stu(path: str) -> tuple[list[int], list[float]] | None:
    """Teacher←Student: Rescue(L)=P(T safe | S unsafe ∧ T alone safe).

    Prefer guard_summary.trr_by_prefix (short/missing counted as fail).
    Fallback: recompute from samples with the same rule.
    """
    with open(path) as f:
        data = json.load(f)
    summary = data.get("guard_summary") or {}

    rescue = summary.get("trr_by_prefix")
    if rescue and summary.get("eval_set") == "student_unsafe_and_teacher_alone_safe":
        prefixes, rates = [], []
        for pkey in sorted(rescue, key=lambda x: int(x)):
            prefixes.append(int(pkey))
            rates.append(float(rescue[pkey]["trr"]))
        return prefixes, rates

    # fallback from samples (S-unsafe ∩ T-alone-safe via L=0)
    samples = data.get("samples") or []

    def _tea_alone_safe(s: dict) -> bool:
        t0 = (s.get("teacher_by_prefix") or {}).get("0")
        if t0 is not None and "teacher_guard_unsafe" in t0:
            return not t0["teacher_guard_unsafe"]
        return s.get("teacher_guard_unsafe") is False

    eval_set = [
        s
        for s in samples
        if s.get("student_guard_unsafe") is True and _tea_alone_safe(s)
    ]
    if not eval_set:
        return None
    n = len(eval_set)
    prefix_lengths = data.get("prefix_lengths")
    if not prefix_lengths:
        keys: set[int] = set()
        for s in samples:
            keys.update(int(k) for k in (s.get("teacher_by_prefix") or {}))
        prefix_lengths = sorted(keys)
    prefixes, rates = [], []
    for L in prefix_lengths:
        pkey = str(L)
        success = 0
        for s in eval_set:
            if int(s.get("student_token_len", 0)) <= L:
                continue
            tres = (s.get("teacher_by_prefix") or {}).get(pkey)
            if tres is None or "teacher_guard_unsafe" not in tres:
                continue
            if not tres["teacher_guard_unsafe"]:
                success += 1
        prefixes.append(int(L))
        rates.append(success / n)
    return prefixes, rates


def load_stu_on_tea(path: str) -> tuple[list[int], list[float]] | None:
    """Student←Teacher: Hold(L)=P(S safe | T safe ∧ S alone unsafe)."""
    with open(path) as f:
        data = json.load(f)
    summary = data.get("guard_summary") or {}

    hold = summary.get("hold_by_prefix")
    if hold and summary.get("eval_set") == "teacher_safe_and_student_unsafe":
        # skip empty / invalid summaries
        any_n = any(int(v.get("total") or 0) > 0 for v in hold.values())
        if any_n:
            prefixes, rates = [], []
            for pkey in sorted(hold, key=lambda x: int(x)):
                prefixes.append(int(pkey))
                rates.append(float(hold[pkey]["hold_rate"]))
            return prefixes, rates

    samples = data.get("samples") or []
    eval_set = [
        s
        for s in samples
        if s.get("teacher_guard_unsafe") is False
        and s.get("student_guard_unsafe") is True
    ]
    if not eval_set:
        return None
    n = len(eval_set)
    prefix_lengths = data.get("prefix_lengths")
    if not prefix_lengths:
        keys: set[int] = set()
        for s in samples:
            keys.update(int(k) for k in (s.get("student_by_prefix") or {}))
        prefix_lengths = sorted(keys)
    prefixes, rates = [], []
    for L in prefix_lengths:
        pkey = str(L)
        success = 0
        for s in eval_set:
            if int(s.get("teacher_token_len", 0)) <= L:
                success += 1  # short teacher reply → count as hold success
                continue
            sres = (s.get("student_by_prefix") or {}).get(pkey)
            if sres is None or "student_guard_unsafe" not in sres:
                continue
            if not sres["student_guard_unsafe"]:
                success += 1
        prefixes.append(int(L))
        rates.append(success / n)
    return prefixes, rates


def _tick_fmt(prefixes: list[int]):
    labels = [str(p) for p in prefixes]

    def _fmt(x, _pos):
        i = int(round(x))
        return labels[i] if 0 <= i < len(labels) else ""

    return FuncFormatter(_fmt)


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pdf = out_path if out_path.endswith(".pdf") else out_path + ".pdf"
    fig.savefig(pdf)
    fig.savefig(pdf[:-4] + ".png")
    plt.close(fig)
    print(f"[save] {pdf}")
    print(f"[save] {pdf[:-4]}.png")


def plot_one_model_compare(
    model: str,
    tea_on_stu: tuple[list[int], list[float]],
    stu_on_tea: tuple[list[int], list[float]],
    out_path: str,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    p1, d1 = tea_on_stu
    p2, d2 = stu_on_tea
    base = p1 if p1 == p2 else sorted(set(p1) | set(p2))
    x = np.arange(len(base), dtype=float)
    m1, m2 = dict(zip(p1, d1)), dict(zip(p2, d2))
    y1 = [m1.get(p, np.nan) * 100 for p in base]
    y2 = [m2.get(p, np.nan) * 100 for p in base]

    ax.plot(
        x, y1, color="#0072B2", marker="o", markerfacecolor="white",
        markeredgewidth=1.4,
        label=r"Tea$\leftarrow$Stu  (rescue $|$ S unsafe $\wedge$ T safe)",
    )
    ax.plot(
        x, y2, color="#D55E00", marker="s", markerfacecolor="white",
        markeredgewidth=1.4,
        label=r"Stu$\leftarrow$Tea  (hold $|$ T safe $\wedge$ S fail)",
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Prefix length (tokens)")
    ax.set_ylabel(r"Success rate (\%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title(MODEL_LABELS.get(model, model))
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(_tick_fmt(base))
    ax.tick_params(axis="x", rotation=35, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.legend(loc="best", handlelength=2.2, borderpad=0.45)
    fig.tight_layout()
    _save(fig, out_path)


def plot_four_models(
    series: dict[str, tuple[list[int], list[float]]],
    out_path: str,
    ylabel: str,
    title: str | None = None,
    y_top_pad: float | None = None,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "best",
) -> None:
    """series: model -> (prefixes, rates_or_deltas in [0,1] fraction)

    y_top_pad: if set, extend ylim top by this many % points above data max
               so the legend does not cover curves.
    ylim: optional fixed (ymin, ymax) in percent units, e.g. (0, 100).
    legend_loc: matplotlib legend location.
    """
    if not series:
        print(f"[skip] empty series for {out_path}")
        return

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    # shared prefix axis from first model in MODEL_ORDER that exists
    base = None
    for m in MODEL_ORDER:
        if m in series:
            base = series[m][0]
            break
    if base is None:
        base = next(iter(series.values()))[0]

    x = np.arange(len(base), dtype=float)
    all_ys: list[float] = []
    for model in MODEL_ORDER:
        if model not in series:
            continue
        prefixes, deltas = series[model]
        mp = dict(zip(prefixes, deltas))
        ys = [mp.get(p, np.nan) * 100.0 for p in base]
        all_ys.extend(v for v in ys if np.isfinite(v))
        style = MODEL_STYLE.get(model, {"color": "#666666", "marker": "o"})
        ax.plot(
            x,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            label=MODEL_LABELS.get(model, model),
            markerfacecolor="white",
            markeredgewidth=1.4,
            markersize=7,
            zorder=3,
            clip_on=False,
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":", zorder=2)
    if 0 in base:
        ax.axvline(
            base.index(0), color="#AAAAAA", linewidth=0.8,
            linestyle="--", zorder=1, alpha=0.7,
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif y_top_pad is not None and all_ys:
        ymin, ymax = min(all_ys), max(all_ys)
        top = max(ymax, 0.0) + y_top_pad
        bottom = ymin - 0.05 * (top - ymin)
        ax.set_ylim(bottom, top)

    ax.set_xlabel("Prefix length (tokens)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(_tick_fmt(base))
    ax.tick_params(axis="x", rotation=35, pad=2)
    if ylim is not None and ylim[1] >= 100:
        ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    legend = ax.legend(loc=legend_loc, handlelength=2.2, borderpad=0.45, labelspacing=0.35)
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    _save(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wildchat")
    p.add_argument("--model", default=None, help="Only compare this model (per-model fig); 4-model figs still use all available")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-per-model", action="store_true", help="Skip per-model compare figs")
    p.add_argument("--no-four-model", action="store_true", help="Skip 4-model direction figs")
    args = p.parse_args()

    tea_dir = os.path.join(SCRIPTS_ROOT, "outputs", "teacher_prefix_continue", args.dataset)
    stu_dir = os.path.join(SCRIPTS_ROOT, "outputs", "student_prefix_continue", args.dataset)
    out_dir = args.output_dir or DEFAULT_OUT_DIR

    tea_files = {
        os.path.basename(f).replace("_prefix_pipeline.json", ""): f
        for f in glob.glob(os.path.join(tea_dir, "*_prefix_pipeline.json"))
    }
    stu_files = {
        os.path.basename(f).replace("_prefix_pipeline.json", ""): f
        for f in glob.glob(os.path.join(stu_dir, "*_prefix_pipeline.json"))
    }

    # ---- load all available ----
    tea_on_stu_all: dict[str, tuple[list[int], list[float]]] = {}
    for model, path in tea_files.items():
        data = load_tea_on_stu(path)
        if data is not None:
            tea_on_stu_all[model] = data
            print(f"[load] Tea←Stu  {model}")

    stu_on_tea_all: dict[str, tuple[list[int], list[float]]] = {}
    for model, path in stu_files.items():
        data = load_stu_on_tea(path)
        if data is not None:
            stu_on_tea_all[model] = data
            print(f"[load] Stu←Tea  {model}")

    # ---- 1) per-model direction compare ----
    if not args.no_per_model:
        both = sorted(set(tea_on_stu_all) & set(stu_on_tea_all))
        if args.model:
            both = [m for m in both if m == args.model]
        for model in both:
            out = os.path.join(out_dir, f"prefix_direction_compare_{args.dataset}_{model}")
            plot_one_model_compare(model, tea_on_stu_all[model], stu_on_tea_all[model], out)

    # ---- 2) four-model: student prefix (Tea←Stu) ----
    if not args.no_four_model:
        if tea_on_stu_all:
            plot_four_models(
                tea_on_stu_all,
                os.path.join(out_dir, f"prefix_tea_on_stu_{args.dataset}"),
                ylabel=r"Teacher rescue rate (\%)",
                title=None,  # caption: Tea←Stu, Rescue=P(T safe|S unsafe ∧ T alone safe)
                ylim=(0.0, 108.0),
                legend_loc="lower left",
            )
        else:
            print(f"[skip] no Tea←Stu results in {tea_dir}")

        # ---- 3) four-model: teacher prefix (Stu←Tea) ----
        if stu_on_tea_all:
            plot_four_models(
                stu_on_tea_all,
                os.path.join(out_dir, f"prefix_stu_on_tea_{args.dataset}"),
                ylabel=r"Student defense rate (\%)",
                title=None,
                ylim=(0.0, 108.0),
                legend_loc="lower right",
            )
        else:
            print(f"[skip] no Stu←Tea results in {stu_dir}")


if __name__ == "__main__":
    main()
