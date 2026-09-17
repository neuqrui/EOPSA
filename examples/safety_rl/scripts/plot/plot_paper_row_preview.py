#!/usr/bin/env python3
"""Render efficiency + safe-avg gap + KL mode figures side-by-side for layout check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from paper_figure_style import apply_paper_style

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
EFF_SCRIPT = SCRIPT_DIR / "plot_efficiency_bars.py"
GAP_SCRIPT = ROOT / "eval_llm_safety/figures/teacher_upper_bound/plot_safe_avg_gap_closed.py"
KL_SCRIPT = ROOT / "examples/safety_rl/ablations/kl_loss/plot_kl_mode_avg.py"
OUT_DIR = SCRIPT_DIR / "figures" / "efficiency"
EFF_PNG = OUT_DIR / "efficiency_bars_v6.png"
GAP_PNG = ROOT / "eval_llm_safety/figures/teacher_upper_bound/safe_avg_gap_closed.png"
KL_PNG = ROOT / "examples/safety_rl/ablations/kl_loss/figures/kl_mode_avg.png"
PREVIEW_PNG = OUT_DIR / "paper_row_preview.png"
PREVIEW_PDF = OUT_DIR / "paper_row_preview.pdf"
PYTHON = sys.executable


def _regenerate_sources() -> None:
    subprocess.run([PYTHON, str(EFF_SCRIPT)], check=True)
    subprocess.run([PYTHON, str(GAP_SCRIPT)], check=True)
    subprocess.run([PYTHON, str(KL_SCRIPT)], check=True)


def plot_preview() -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(21.6, 4.8))
    for ax, path, tag in (
        (axes[0], EFF_PNG, "(a) Efficiency"),
        (axes[1], GAP_PNG, "(b) Safe gain relative to teacher"),
        (axes[2], KL_PNG, "(c) KL mode ablation"),
    ):
        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
        ax.set_title(tag, fontsize=14, fontfamily="DejaVu Serif", pad=8)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.02, wspace=0.02)
    fig.savefig(PREVIEW_PNG, dpi=300)
    fig.savefig(PREVIEW_PDF, dpi=300)
    plt.close(fig)
    print(f"Saved: {PREVIEW_PNG}")
    print(f"Saved: {PREVIEW_PDF}")


def main() -> None:
    _regenerate_sources()
    plot_preview()


if __name__ == "__main__":
    main()
