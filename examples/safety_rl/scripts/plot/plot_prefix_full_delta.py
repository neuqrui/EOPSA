#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Teacher-on-Student-Prefix results (full ΔDSR vs prefix length).

Usage:
    python3 plot/plot_prefix_full_delta.py --dataset wildchat
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

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

# Okabe–Ito colorblind-safe palette
MODEL_STYLE = {
    "qwen3-1.7b": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "qwen3-4b":   {"color": "#D55E00", "marker": "s", "linestyle": "-"},
    "ds-1.5b":    {"color": "#009E73", "marker": "^", "linestyle": "-"},
    "ds-7b":      {"color": "#E69F00", "marker": "D", "linestyle": "-"},
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
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


def load_prefix_delta_full(json_path: str) -> dict[str, Any] | None:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    summary = payload.get("guard_summary")
    if not summary:
        samples = payload.get("samples", [])
        if not samples or "student_guard_unsafe" not in samples[0]:
            return None
        n = len(samples)
        stu_unsafe = sum(1 for s in samples if s.get("student_guard_unsafe"))
        stu_dsr = 1.0 - stu_unsafe / n if n else 0.0
        prefix_keys = sorted(
            {k for s in samples for k in s.get("teacher_by_prefix", {})},
            key=lambda x: int(x),
        )
        prefixes, deltas = [], []
        for pkey in prefix_keys:
            unsafe = total = 0
            for s in samples:
                t = s.get("teacher_by_prefix", {}).get(pkey)
                if t is None or "teacher_guard_unsafe" not in t:
                    continue
                total += 1
                if t["teacher_guard_unsafe"]:
                    unsafe += 1
            if total == 0:
                continue
            tea_dsr = 1.0 - unsafe / total
            prefixes.append(int(pkey))
            deltas.append(tea_dsr - stu_dsr)
        model = payload.get("model") or os.path.basename(json_path).replace(
            "_prefix_pipeline.json", ""
        )
        return {
            "model": model,
            "student_dsr": stu_dsr,
            "n": n,
            "prefixes": prefixes,
            "delta_full": deltas,
        }

    stu_dsr = float(summary["student"]["DSR"])
    n = int(summary["student"]["total"])
    prefixes, deltas = [], []
    for pkey, t in summary.get("teacher_by_prefix", {}).items():
        prefixes.append(int(pkey))
        deltas.append(float(t["DSR"]) - stu_dsr)

    order = np.argsort(prefixes)
    prefixes = [prefixes[i] for i in order]
    deltas = [deltas[i] for i in order]

    model = payload.get("model") or os.path.basename(json_path).replace(
        "_prefix_pipeline.json", ""
    )
    return {
        "model": model,
        "student_dsr": stu_dsr,
        "n": n,
        "prefixes": prefixes,
        "delta_full": deltas,
    }


def collect_results(input_dir: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(input_dir, "*_prefix_pipeline.json")))
    results = []
    for path in files:
        item = load_prefix_delta_full(path)
        if item is None:
            print(f"[skip] no guard summary: {path}")
            continue
        results.append(item)
        print(
            f"[load] {item['model']:15s}  n={item['n']:3d}  "
            f"stu_dsr={item['student_dsr']:.2%}  prefixes={item['prefixes']}"
        )
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    results.sort(key=lambda r: rank.get(r["model"], 100))
    return results


def _prefix_tick_formatter(prefixes: list[int]):
    labels = [str(p) for p in prefixes]

    def _fmt(x, _pos):
        i = int(round(x))
        if 0 <= i < len(labels):
            return labels[i]
        return ""

    return FuncFormatter(_fmt)


def plot_delta_curves(
    results: list[dict[str, Any]],
    out_path: str,
    title: str | None = None,
) -> None:
    if not results:
        raise RuntimeError("No results to plot.")

    base_prefixes = results[0]["prefixes"]
    for r in results[1:]:
        if r["prefixes"] != base_prefixes:
            print(
                f"[warn] prefix mismatch: {r['model']} has {r['prefixes']}, "
                f"expected {base_prefixes}"
            )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    x = np.arange(len(base_prefixes), dtype=float)

    for r in results:
        model = r["model"]
        style = MODEL_STYLE.get(
            model,
            {"color": "#666666", "marker": "o", "linestyle": "-"},
        )
        prefix_to_delta = dict(zip(r["prefixes"], r["delta_full"]))
        ys_pct = [
            prefix_to_delta[p] * 100.0 if p in prefix_to_delta else np.nan
            for p in base_prefixes
        ]

        ax.plot(
            x,
            ys_pct,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=MODEL_LABELS.get(model, model),
            markerfacecolor="white",
            markeredgewidth=1.4,
            markersize=7,
            zorder=3,
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":", zorder=2)

    ax.set_xlabel("Student prefix length (tokens)")
    ax.set_ylabel(r"$\Delta\mathrm{DSR}$ (\%)")
    if title:
        ax.set_title(title)

    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(_prefix_tick_formatter(base_prefixes))
    ax.tick_params(axis="x", rotation=35, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    if 0 in base_prefixes:
        i0 = base_prefixes.index(0)
        ax.axvline(i0, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1, alpha=0.7)

    legend = ax.legend(
        loc="upper right",
        ncol=1,
        handlelength=2.2,
        borderpad=0.45,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pdf_path = out_path if out_path.endswith(".pdf") else out_path + ".pdf"
    png_path = pdf_path[:-4] + ".png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"[save] {pdf_path}")
    print(f"[save] {png_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot full ΔDSR vs prefix length")
    p.add_argument("--dataset", default="wildchat")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_dir:
        input_dir = args.input_dir
        dataset_tag = os.path.basename(os.path.abspath(input_dir.rstrip("/")))
    else:
        input_dir = os.path.join(
            SCRIPTS_ROOT, "outputs", "teacher_prefix_continue", args.dataset
        )
        dataset_tag = args.dataset

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    results = collect_results(input_dir)
    out = args.output or os.path.join(
        DEFAULT_OUT_DIR, f"prefix_full_delta_{dataset_tag}"
    )
    plot_delta_curves(results, out, title=args.title)


if __name__ == "__main__":
    main()
