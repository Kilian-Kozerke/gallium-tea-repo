#!/usr/bin/env python3
"""Thesis Figure 5.7: Carbon load comparison IX vs SX.

Full-LCA carbon burden [kg CO₂-eq / kg Ga] at Q=10, 30, 50, 100.
Stacked by driver (Electricity, NaOH, Other). Uses V08 + carbon_burden.
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
from utils.carbon_burden import calc_carbon_burden_breakdown

OUT_FIGURES = REPO_ROOT / "outputs" / "figures"
OUT_CSV = REPO_ROOT / "outputs" / "csv"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_CSV.mkdir(parents=True, exist_ok=True)

Q_VALS = [10, 30, 50, 100]
COLORS = {"electricity": "#5B9BD5", "naoh": "#B37A3D", "other": "#9E9E9E"}

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


def _group_breakdown(breakdown):
    """Group GWP by Electricity, NaOH, Other."""
    elec = sum(v["gwp_per_kg_ga"] for k, v in breakdown.items() if "electricity" in k.lower() or k == "electricity")
    naoh = sum(v["gwp_per_kg_ga"] for k, v in breakdown.items() if "naoh" in k.lower() or "naoh" in k)
    other = sum(v["gwp_per_kg_ga"] for k, v in breakdown.items()
                if "electricity" not in k.lower() and k != "electricity" and "naoh" not in k.lower() and k != "naoh")
    return elec, naoh, other


def main():
    config = tea.DEFAULT_CONFIG
    data = []
    for Q in Q_VALS:
        for route in ["IX", "SX"]:
            bd = calc_carbon_burden_breakdown(Q, route=route, config=config)
            e, n, o = _group_breakdown(bd)
            data.append({"Q": Q, "route": route, "Electricity": e, "NaOH": n, "Other": o})

    # Plot: 4 groups (Q=10 IX, Q=10 SX, Q=100 IX, Q=100 SX) or 8 bars
    _set_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    labels = [f"{d['route']}@{d['Q']}" for d in data]
    x = np.arange(len(labels))
    width = 0.7

    elec = [d["Electricity"] for d in data]
    naoh = [d["NaOH"] for d in data]
    other = [d["Other"] for d in data]

    ax.bar(x, elec, width, label="Electricity", color=COLORS["electricity"], edgecolor="white")
    ax.bar(x, naoh, width, bottom=elec, label="NaOH", color=COLORS["naoh"], edgecolor="white")
    ax.bar(x, other, width, bottom=np.array(elec) + np.array(naoh), label="Other", color=COLORS["other"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Carbon load [kg CO$_2$-eq kg$^{-1}$ Ga]")
    ax.set_title("Full-LCA carbon burden by route and throughput")
    ax.legend(loc="upper right", frameon=False)
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(OUT_FIGURES / "fig_5_7_carbon_load_comparison.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_5_7_carbon_load_comparison.svg")


if __name__ == "__main__":
    main()
