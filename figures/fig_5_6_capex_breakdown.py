#!/usr/bin/env python3
"""Thesis Figure 5.6: CapEx breakdown (Sep vs Other) at Q=10 and Q=100.

Shows CapEx-Sep and CapEx-Other for IX and SX. Uses V08 model.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tea_model_ga_thesis as tea

OUT_FIGURES = REPO_ROOT / "outputs" / "figures"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

CM_TO_IN = 1 / 2.54
FIG_W = 12 * CM_TO_IN
FIG_H = FIG_W * 0.78


def _set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def main():
    config = tea.DEFAULT_CONFIG
    bars = [("IX", 10.0), ("SX", 10.0), ("IX", 100.0), ("SX", 100.0)]
    x_labels = [f"{r}@{int(q)}" for r, q in bars]

    capex_sep = []
    capex_other = []
    for route, q in bars:
        cb = tea.calc_cost_breakdown(q, route=route, config=config)
        capex_sep.append(cb["CapEx-Sep"])
        capex_other.append(cb["CapEx-Other"])

    _set_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(bars))
    width = 0.6
    ax.bar(x - width/2, capex_sep, width, label="CapEx-Sep", color="#00549F", edgecolor="white")
    ax.bar(x - width/2, capex_other, width, bottom=capex_sep, label="CapEx-Other", color="#8EBAE5", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("CapEx [EUR m⁻³]")
    ax.set_title("CapEx breakdown (Q=10 vs 100)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT_FIGURES / "fig_5_6_capex_breakdown.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_5_6_capex_breakdown.svg")


if __name__ == "__main__":
    main()
