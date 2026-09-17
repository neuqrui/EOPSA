"""Shared matplotlib style for paper figures (efficiency + safe-avg row)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont


def _font(size: float) -> FontProperties:
    """Bind the actual TTF file so PDF/PNG use identical glyph outlines."""
    path = findfont(FontProperties(family="DejaVu Serif"))
    return FontProperties(fname=path, size=size)


FONT_NAME = "DejaVu Serif"
METHOD_LABEL_FONT = _font(14)
BODY_FONT = _font(14)
ANNOT_FONT = _font(11)
LEGEND_FONT = _font(11)


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_NAME,
            "mathtext.fontset": "dejavuserif",
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            # Type 3 = glyph outlines; avoids PDF viewers substituting sans-serif
            # when embedded DejaVu Serif is ignored (e.g. Cursor PDF preview).
            "pdf.fonttype": 3,
            "ps.fonttype": 3,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def style_axis_text(ax) -> None:
    """Apply the shared font to every text element on an axis."""
    ax.xaxis.label.set_fontproperties(BODY_FONT)
    ax.yaxis.label.set_fontproperties(BODY_FONT)
    for label in ax.get_xticklabels():
        label.set_fontproperties(METHOD_LABEL_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(BODY_FONT)
