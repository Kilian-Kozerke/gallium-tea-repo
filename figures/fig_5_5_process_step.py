#!/usr/bin/env python3
"""Thesis Figure 5.5: LCOGa contribution by process step (stacked).

Shows cost contribution per process step for IX and SX at Q=10.
Uses V08 model.
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

STEP_LABELS = ["Filtration", "RO", "pH-Adj", "IX/SX", "Precip", "Leach", "EW", "Labour"]
STEP_COLORS = ["#B0B0B0", "#5B9BD5", "#8EBAE5", "#00549F", "#57AB27", "#7FBF7F", "#F6A800", "#808080"]

CM_TO_IN = 1 / 2.54
FIG_W = 12 * CM_TO_IN
FIG_H = FIG_W * 0.78


def _set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def main():
    config = tea.DEFAULT_CONFIG
    Q = 10.0
    costs_ix = tea.calc_total_costs(Q, route="IX", config=config)
    costs_sx = tea.calc_total_costs(Q, route="SX", config=config)

    step_keys_ix = ["Filtration", "RO_Split", "pH_Adjust", "IX", "Precipitation", "Selective_Leaching", "Electrowinning"]
    step_keys_sx = ["Filtration", "RO_Split", "pH_Adjust", "SX", "Precipitation", "Selective_Leaching", "Electrowinning"]

    def step_costs(costs, keys):
        return [
            costs[k]["CapEx"] + costs[k]["REP"] + costs[k]["OpEx"]
            for k in keys
        ]

    vals_ix = step_costs(costs_ix, step_keys_ix)
    vals_sx = step_costs(costs_sx, step_keys_sx)
    vals_ix.append(costs_ix["Total"]["Labour"])
    vals_sx.append(costs_sx["Total"]["Labour"])

    total_ix = sum(vals_ix)
    total_sx = sum(vals_sx)
    pct_ix = [v / total_ix * 100 for v in vals_ix]
    pct_sx = [v / total_sx * 100 for v in vals_sx]

    _set_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.array([0, 1])  # IX, SX
    width = 0.6
    bottom_ix, bottom_sx = 0.0, 0.0
    for i, (label, col) in enumerate(zip(STEP_LABELS, STEP_COLORS)):
        ax.bar(x, [pct_ix[i], pct_sx[i]], width, bottom=[bottom_ix, bottom_sx],
               color=col, label=label, edgecolor="white", linewidth=0.5)
        bottom_ix += pct_ix[i]
        bottom_sx += pct_sx[i]

    ax.set_xticks(x)
    ax.set_xticklabels(["IX", "SX"])
    ax.set_ylabel("Contribution to LCOGa [%]")
    ax.set_title("Process step contribution (Q=10)")
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.15), fontsize=7)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(OUT_FIGURES / "fig_5_5_process_step_contribution.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_5_5_process_step_contribution.svg")


if __name__ == "__main__":
    main()
