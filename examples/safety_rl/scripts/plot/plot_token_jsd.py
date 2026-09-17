#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot token-wise Top-K JSD (Student vs Teacher).

Default: four models on one figure (Qwen3-1.7B/4B, R1-Distill-1.5B/7B).

Usage:
    python3 plot/plot_token_jsd.py --dataset wildchat
    python3 plot/plot_token_jsd.py --dataset wildchat --models qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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

# Soft AAAI-style pastel palette (readable on white, not neon).
MODEL_STYLE = {
    "qwen3-1.7b": {"color": "#6FA8DC", "linestyle": "-"},  # soft sky blue
    "qwen3-4b":   {"color": "#E8917D", "linestyle": "-"},  # soft coral
    "ds-1.5b":    {"color": "#7DBE8E", "linestyle": "-"},  # soft sage
    "ds-7b":      {"color": "#B89BC9", "linestyle": "-"},  # soft lilac
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.2,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def load_jsd(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    # preserve length with reflect padding
    pad = window // 2
    yp = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def plot_token_jsd(
    results: list[dict[str, Any]],
    out_path: str,
    min_coverage: float = 0.2,
    smooth_window: int = 1,
    max_tokens: int | None = None,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    for data in results:
        model = data["model"]
        style = MODEL_STYLE.get(model, {"color": "#555555", "linestyle": "-"})
        x = np.asarray(data["token_idx"], dtype=float)
        y = np.asarray(data["mean_jsd"], dtype=float)
        cov = np.asarray(data.get("coverage", [1.0] * len(y)), dtype=float)

        mask = (cov >= min_coverage) & np.isfinite(y)
        if max_tokens is not None:
            mask = mask & (x <= max_tokens)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            print(f"[warn] no points for {model} after filtering")
            continue

        if smooth_window > 1:
            y_plot = smooth_curve(y, smooth_window)
        else:
            y_plot = y

        topk = data.get("topk", "?")
        label = f"{MODEL_LABELS.get(model, model)} (k={topk})"
        ax.plot(
            x,
            y_plot,
            color=style["color"],
            linestyle=style["linestyle"],
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Token index")
    ax.set_ylabel(r"Top-$k$ JSD")
    if max_tokens is not None:
        ax.set_xlim(0, max_tokens)
    else:
        ax.set_xlim(left=0)

    legend = ax.legend(loc="best", handlelength=2.2, borderpad=0.45)
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
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wildchat")
    p.add_argument("--models", default="qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--min-coverage", type=float, default=0.2)
    p.add_argument("--smooth", type=int, default=1, help="Moving-average window (1=off)")
    p.add_argument("--max-tokens", type=int, default=512, help="Truncate x-axis to this token index (default: 512)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or os.path.join(
        SCRIPTS_ROOT, "outputs", "token_jsd", args.dataset
    )
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    # stable draw order
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    model_names = sorted(model_names, key=lambda m: rank.get(m, 100))

    results = []
    for name in model_names:
        path = os.path.join(input_dir, f"{name}_token_jsd.json")
        if not os.path.isfile(path):
            print(f"[skip] missing {path}")
            continue
        data = load_jsd(path)
        results.append(data)
        print(
            f"[load] {name}: n={data['n_samples']}  "
            f"max_pos={len(data['mean_jsd'])}  topk={data.get('topk')}"
        )

    if not results:
        raise FileNotFoundError(f"No JSD results in {input_dir} for {model_names}")

    out = args.output or os.path.join(DEFAULT_OUT_DIR, f"token_jsd_{args.dataset}")
    plot_token_jsd(
        results,
        out,
        min_coverage=args.min_coverage,
        smooth_window=args.smooth,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
