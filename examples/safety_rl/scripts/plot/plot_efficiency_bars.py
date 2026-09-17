#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICLR-style efficiency figure for qwen3-1.7B.

Dual-axis broken chart; two comparison boxes (no drop arrows):
  - EOPSA vs OPSD
  - EOPSA vs OPSA

Usage:
    python plot_efficiency_bars.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from paper_figure_style import METHOD_LABEL_FONT, apply_paper_style

OUT_DIR = SCRIPT_DIR / "figures" / "efficiency"

COLOR_ROLLOUT = "#A8C8E8"
COLOR_TRAIN = "#5B8FDB"
COLOR_DATA = "#2E5FA3"
COLOR_ANNOT = "#1B3A6B"

METHODS = ["OPSA", "ThinkSafe", "OPSD", "EOPSA"]
ROLLOUT_H = np.array([4.05, np.nan, 0.51, 0.26], dtype=float)
TRAIN_H = np.array([7.51, 4.57, 0.29, 0.21], dtype=float)
EFF_DATA = np.array([119808, 120000, 6400, 6400], dtype=float)

TIME_BREAK = 0.70
TIME_BOT_YLIM = (0.0, TIME_BREAK)
TIME_HIGH = (3.0, 8.3)

DATA_BREAK = 7500.0
DATA_BOT_YLIM = (0.0, DATA_BREAK)
DATA_HIGH = (30000.0, 128000.0)

# White strip between panels (twin dashed break lines sit at its top/bottom edges)
BREAK_GAP = 0.014

BAR_WIDTH = 0.25


def apply_style() -> None:
    apply_paper_style()
    # Efficiency-only overrides (keep method x-labels from paper_figure_style)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
        }
    )


def _fmt_time(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _fmt_data(v: float) -> str:
    return f"{int(v):,}"


def _pct_drop(new: float, base: float) -> float:
    return 100.0 * (1.0 - new / base)


def _draw_break_decoration(fig: plt.Figure, ax_top: plt.Axes, ax_bot: plt.Axes) -> None:
    """Twin dashed break lines + two parallel // slashes crossing each y-axis spine."""
    pos_bot = ax_bot.get_position()
    pos_top = ax_top.get_position()
    x0, x1 = pos_bot.x0, pos_bot.x1
    y_bot_line = pos_bot.y1
    y_top_line = pos_top.y0
    y_break = 0.5 * (y_bot_line + y_top_line)

    dash_kw = dict(
        transform=fig.transFigure,
        color="k",
        lw=0.80,
        ls=(0, (2.8, 1.8)),
        clip_on=False,
        zorder=6,
        solid_capstyle="butt",
    )
    for y in (y_bot_line, y_top_line):
        fig.add_artist(Line2D([x0, x1], [y, y], **dash_kw))

    # Clockwise from 45°: flatter `/` (visual ~28° on 7.2×4.8 canvas).
    dx, dy = 0.0100, 0.0070
    # Parallel offset along the slash-normal; keep a visible gap between the pair.
    nlen = (dx ** 2 + dy ** 2) ** 0.5
    sep = 0.0075
    ox, oy = -dy / nlen * sep, dx / nlen * sep
    slash_kw = dict(
        transform=fig.transFigure,
        color="k",
        lw=0.95,
        clip_on=False,
        zorder=7,
        solid_capstyle="butt",
    )
    for x_spine in (x0, x1):
        for k in (-0.5, 0.5):
            xc = x_spine + k * ox
            yc = y_break + k * oy
            fig.add_artist(
                Line2D(
                    [xc - dx, xc + dx],
                    [yc - dy, yc + dy],
                    **slash_kw,
                )
            )


def _mask_for_panel(values: np.ndarray, panel: str, low_hi: float) -> np.ndarray:
    out = values.astype(float).copy()
    for i, v in enumerate(out):
        if np.isnan(v):
            continue
        if panel == "low":
            if v > low_hi:
                out[i] = low_hi
        else:
            if v <= low_hi:
                out[i] = np.nan
    return out


def _annotate_bars(
    ax,
    bars,
    values_plot,
    values_true,
    fmt,
    y_lo,
    y_hi,
    *,
    panel: str,
    low_hi: float,
    fontsize: float = 7.5,
    color: str = COLOR_ANNOT,
) -> None:
    span = y_hi - y_lo
    for bar, v_plot, v_true in zip(bars, values_plot, values_true):
        if np.isnan(v_true):
            continue
        if panel == "low" and v_true > low_hi:
            continue
        if panel == "high" and v_true <= low_hi:
            continue
        if np.isnan(v_plot) or v_plot <= 0:
            continue
        tip = min(v_true, y_hi) if panel == "high" else v_true
        if tip <= y_lo:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            tip + span * 0.03,
            fmt(v_true),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _draw_panel(ax_t, ax_d, x, width, panel: str) -> None:
    r_true, t_true, d_true = ROLLOUT_H, TRAIN_H, EFF_DATA
    r = _mask_for_panel(r_true, panel, TIME_BREAK)
    t = _mask_for_panel(t_true, panel, TIME_BREAK)
    d = _mask_for_panel(d_true, panel, DATA_BREAK)

    xs_r = np.full(len(x), np.nan)
    xs_tr = np.full(len(x), np.nan)
    xs_d = np.full(len(x), np.nan)
    for i, xi in enumerate(x):
        slots: list[str] = []
        if not np.isnan(r_true[i]):
            slots.append("r")
        if not np.isnan(t_true[i]):
            slots.append("t")
        if not np.isnan(d_true[i]):
            slots.append("d")
        if not slots:
            continue
        offsets = (np.arange(len(slots)) - (len(slots) - 1) / 2.0) * width
        pos = {name: xi + off for name, off in zip(slots, offsets)}
        if "r" in pos:
            xs_r[i] = pos["r"]
        if "t" in pos:
            xs_tr[i] = pos["t"]
        if "d" in pos:
            xs_d[i] = pos["d"]

    bars_r = ax_t.bar(
        np.nan_to_num(xs_r, nan=0.0),
        np.nan_to_num(r, nan=0.0),
        width,
        color=COLOR_ROLLOUT,
        edgecolor="none",
        zorder=3,
    )
    bars_tr = ax_t.bar(
        np.nan_to_num(xs_tr, nan=0.0),
        np.nan_to_num(t, nan=0.0),
        width,
        color=COLOR_TRAIN,
        edgecolor="none",
        zorder=3,
    )
    bars_d = ax_d.bar(
        np.nan_to_num(xs_d, nan=0.0),
        np.nan_to_num(d, nan=0.0),
        width,
        color=COLOR_DATA,
        edgecolor="none",
        zorder=3,
    )

    for bars, vals, xs in (
        (bars_r, r, xs_r),
        (bars_tr, t, xs_tr),
        (bars_d, d, xs_d),
    ):
        for bar, val, xv in zip(bars, vals, xs):
            if np.isnan(val) or np.isnan(xv):
                bar.set_height(0)
                bar.set_alpha(0)

    t_lo, t_hi = ax_t.get_ylim()
    d_lo, d_hi = ax_d.get_ylim()
    _annotate_bars(ax_t, bars_r, r, r_true, _fmt_time, t_lo, t_hi, panel=panel, low_hi=TIME_BREAK)
    _annotate_bars(ax_t, bars_tr, t, t_true, _fmt_time, t_lo, t_hi, panel=panel, low_hi=TIME_BREAK)
    _annotate_bars(
        ax_d, bars_d, d, d_true, _fmt_data, d_lo, d_hi, panel=panel, low_hi=DATA_BREAK, fontsize=7.0
    )


def _tighten_break_gap(
    fig: plt.Figure,
    ax_top: plt.Axes,
    ax_bot: plt.Axes,
    ax_top_r: plt.Axes,
    ax_bot_r: plt.Axes,
    *,
    gap: float = BREAK_GAP,
) -> None:
    """Manually stack panels with a tiny break strip (gridspec hspace alone is unreliable)."""
    fig.subplots_adjust(left=0.14, right=0.86, top=0.84, bottom=0.14)
    pos_bot = ax_bot.get_position()
    pos_top = ax_top.get_position()
    # Force identical horizontal extent so bar x positions match across the break
    x0, width = pos_bot.x0, pos_bot.width
    y_bot = pos_bot.y0
    y_top = y_bot + pos_bot.height + gap
    ax_bot.set_position([x0, y_bot, width, pos_bot.height])
    ax_top.set_position([x0, y_top, width, pos_top.height])
    for ax, ax_r in ((ax_top, ax_top_r), (ax_bot, ax_bot_r)):
        p = ax.get_position()
        ax_r.set_position([p.x0, p.y0, p.width, p.height])


def _add_comparison_boxes(ax_top: plt.Axes) -> None:
    idx = {m: i for i, m in enumerate(METHODS)}
    i_opsd, i_opsa, i_e = idx["OPSD"], idx["OPSA"], idx["EOPSA"]

    pct_r_opsd = _pct_drop(float(ROLLOUT_H[i_e]), float(ROLLOUT_H[i_opsd]))
    pct_t_opsd = _pct_drop(float(TRAIN_H[i_e]), float(TRAIN_H[i_opsd]))
    pct_r_opsa = _pct_drop(float(ROLLOUT_H[i_e]), float(ROLLOUT_H[i_opsa]))
    pct_t_opsa = _pct_drop(float(TRAIN_H[i_e]), float(TRAIN_H[i_opsa]))
    pct_d_opsa = _pct_drop(float(EFF_DATA[i_e]), float(EFF_DATA[i_opsa]))

    box_style = dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#7A9CC0",
        linewidth=0.7,
        alpha=0.96,
    )
    text_kw = dict(
        transform=ax_top.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLOR_ANNOT,
        linespacing=1.3,
        zorder=8,
    )

    ax_top.text(
        0.955,
        0.98,
        f"EOPSA vs OPSD\nrollout −{pct_r_opsd:.0f}%   train −{pct_t_opsd:.0f}%",
        bbox=box_style,
        **text_kw,
    )
    ax_top.text(
        0.955,
        0.68,
        (
            f"EOPSA vs OPSA\n"
            f"rollout −{pct_r_opsa:.0f}%   train −{pct_t_opsa:.0f}%\n"
            f"data −{pct_d_opsa:.0f}%"
        ),
        bbox=box_style,
        **text_kw,
    )


def plot_combined() -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.82, 0.70], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top_r = ax_top.twinx()
    ax_bot_r = ax_bot.twinx()

    x = np.arange(len(METHODS), dtype=float)
    width = BAR_WIDTH
    pad = 0.62
    xlim = (-pad, len(METHODS) - 1 + pad)

    ax_top.set_ylim(*TIME_HIGH)
    ax_bot.set_ylim(*TIME_BOT_YLIM)
    ax_top_r.set_ylim(*DATA_HIGH)
    ax_bot_r.set_ylim(*DATA_BOT_YLIM)
    ax_top.set_xlim(*xlim)
    ax_bot.set_xlim(*xlim)

    _draw_panel(ax_top, ax_top_r, x, width, "high")
    _draw_panel(ax_bot, ax_bot_r, x, width, "low")
    _add_comparison_boxes(ax_top)

    for ax in (ax_top, ax_top_r):
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for ax in (ax_bot, ax_bot_r):
        ax.spines["top"].set_visible(False)

    for ax in (ax_top, ax_bot):
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.95)
        ax.grid(False)
    for ax in (ax_top_r, ax_bot_r):
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_linewidth(0.95)
        ax.yaxis.set_ticks_position("right")
        ax.tick_params(axis="y", which="both", right=True, labelright=True)
        ax.grid(False)

    tick_fs = 9
    # Break-edge labels: keep only the value below the break (0.70 / 7.5).
    ax_bot.set_yticks([0.0, 0.35, 0.70])
    ax_bot.set_yticklabels(["0", "0.35", "0.70"])
    ax_top.set_yticks([4, 6, 8])
    ax_bot_r.set_yticks([0, 4000, 7500])
    ax_bot_r.set_yticklabels(["0", "4", "7.5"])
    ax_top_r.set_yticks([40000, 80000, 120000])
    ax_top_r.set_yticklabels(["40", "80", "120"])
    for ax in (ax_top, ax_bot, ax_top_r, ax_bot_r):
        ax.tick_params(axis="y", labelsize=tick_fs, pad=1.0, length=2.0, width=0.6)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(
        METHODS,
        rotation=0,
        ha="center",
        fontproperties=METHOD_LABEL_FONT,
    )
    ax_bot.tick_params(axis="x", pad=3)

    _tighten_break_gap(fig, ax_top, ax_bot, ax_top_r, ax_bot_r)
    _draw_break_decoration(fig, ax_top, ax_bot)

    pos_bot = ax_bot.get_position()
    pos_top = ax_top.get_position()
    y_mid = 0.5 * (pos_bot.y0 + pos_top.y1)
    fig.text(
        pos_bot.x0 - 0.058,
        y_mid,
        "Time (h)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=11,
    )
    fig.text(
        pos_bot.x1 + 0.058,
        y_mid,
        r"Effective data size ($\times 10^{3}$)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=11,
    )
    ax_top.set_ylabel("")
    ax_bot.set_ylabel("")
    ax_top_r.set_ylabel("")
    ax_bot_r.set_ylabel("")

    handles = [
        Patch(facecolor=COLOR_ROLLOUT, edgecolor="none", label="Rollout time"),
        Patch(facecolor=COLOR_TRAIN, edgecolor="none", label="Train time"),
        Patch(facecolor=COLOR_DATA, edgecolor="none", label="Effective data size"),
    ]
    ax_top.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.2,
    )
    return fig


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_combined()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"efficiency_bars.{ext}", dpi=300)
        fig.savefig(OUT_DIR / f"efficiency_bars_v6.{ext}", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'efficiency_bars_v6.pdf'}")
    print(f"Saved: {OUT_DIR / 'efficiency_bars_v6.png'}")
    print(f"Saved: {OUT_DIR / 'efficiency_bars.pdf'}")
    print(f"Saved: {OUT_DIR / 'efficiency_bars.png'}")


if __name__ == "__main__":
    main()
