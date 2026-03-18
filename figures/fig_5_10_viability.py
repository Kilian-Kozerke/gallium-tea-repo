"""Thesis figure: Figure 5.10 – IX/SX viability map (Q* vs feed concentration).

Break-even throughput (Q*) as a function of gallium feed concentration
for three market price scenarios (300, 423, 700 €/kg).

Outputs:
  - fig_5_10_viability_map.svg — figure
  - fig_5_10_viability_map_data.csv — Q* vs concentration data
  - fig_5_10_viability_map_notes.txt — analysis notes
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tea_model_ga_thesis as tea


# Output directories
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FIGURES = REPO_ROOT / "outputs" / "figures"
OUT_CSV = REPO_ROOT / "outputs" / "csv"
OUT_NOTES = REPO_ROOT / "outputs" / "notes"
for d in (OUT_FIGURES, OUT_CSV, OUT_NOTES):
    d.mkdir(parents=True, exist_ok=True)

# Constants
Q_MIN, Q_MAX = 1.0, 100.0
Q_GRID = np.linspace(Q_MIN, Q_MAX, 100)
CM_TO_IN = 1 / 2.54
FIG_W_MAP = 16 * CM_TO_IN
FIG_H_MAP = 9 * CM_TO_IN

# Color palette
PALETTE = {
    "IX": "#00549F",
    "SX": "#8EBAE5",
    "Labour": "#F6A800",
    "Grey": "#BEBEBE",
    "Grey_dark": "#808080",
}


def _set_style():
    """Apply thesis figure styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "svg.fonttype": "none",
    })


def _q_star(Q_grid, lco_arr, price):
    """Return Q* where LCOGa <= price (linear interpolation)."""
    if float(lco_arr[0]) <= float(price):
        return float(Q_grid[0])
    if float(lco_arr[-1]) > float(price):
        return None
    for i in range(len(Q_grid) - 1):
        y0, y1 = float(lco_arr[i]), float(lco_arr[i + 1])
        if (y0 - price) * (y1 - price) <= 0:
            q0, q1 = float(Q_grid[i]), float(Q_grid[i + 1])
            if y1 == y0:
                return q0
            return q0 + (price - y0) * (q1 - q0) / (y1 - y0)
    return None


def main():
    """Generate Figure 5.10 viability map."""
    config = tea.DEFAULT_CONFIG
    base_config = config

    C_grid = np.linspace(5.0, 125.0, 55)  # mg/L
    prices = [300.0, 423.0, 700.0]  # €/kg
    y_max = float(Q_GRID.max())
    y_min = float(Q_GRID.min())

    rows = []
    Qstars = {}
    for route in ["IX", "SX"]:
        Qstars[route] = {p: np.full_like(C_grid, np.nan, dtype=float) for p in prices}
        for i, C in enumerate(C_grid):
            cfg = config.replace(c_ga_feed_mg_L=float(C))
            lco_arr = np.array([tea.calc_lco_ga(float(q), route=route, config=cfg) for q in Q_GRID])
            for p in prices:
                qs = _q_star(Q_GRID, lco_arr, p)
                if qs is not None:
                    Qstars[route][p][i] = float(qs)
                rows.append({
                    "route": route,
                    "c_Ga_mg_L": C,
                    "market_price_eur_per_kg": p,
                    "Q_star_m3_per_day": qs if qs is not None else np.nan,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV / "fig_5_10_viability_map_data.csv", index=False)

    # Plot
    _set_style()
    fig, (ax_ix, ax_sx) = plt.subplots(1, 2, figsize=(FIG_W_MAP, FIG_H_MAP), sharex=True, sharey=True)

    for ax, route, color, color_light in [
        (ax_ix, "IX", PALETTE["IX"], "#8EBAE5"),
        (ax_sx, "SX", PALETTE["SX"], "#B8D4E8"),
    ]:
        y_low = Qstars[route][300.0]
        y_base = Qstars[route][423.0]
        y_high = Qstars[route][700.0]
        y_low_p = np.where(np.isfinite(y_low), y_low, y_max)
        y_base_p = np.where(np.isfinite(y_base), y_base, y_max)
        y_high_p = np.where(np.isfinite(y_high), y_high, y_max)
        mask = np.isfinite(y_low_p) & np.isfinite(y_high_p)
        ax.fill_between(C_grid[mask], y_high_p[mask], y_low_p[mask], color=color_light, alpha=0.25, label="300–700 €/kg")
        ax.plot(C_grid, y_base_p, color=color, linestyle="-", linewidth=2.0, label="423 €/kg")
        ax.axvline(x=34.6, color=PALETTE["Labour"], linestyle=":", linewidth=1.4, label=r"$c_{Ga}$ = 34.6 mg L$^{-1}$")
        ax.set_xlabel(r"$c_{Ga,feed}$ [mg L$^{-1}$]")
        ax.set_ylabel(r"Break-even throughput Q* [m$^3$ d$^{-1}$]")
        ax.set_title(f"{route} Route")
        ax.set_xlim(float(C_grid.min()), float(C_grid.max()))
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "fig_5_10_viability_map.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    notes = [
        "Figure 5.10 style: Q* vs feed concentration. Throughput domain 1–100 m³/d.",
        "Baseline line at 423 €/kg, price band 300–700 €/kg.",
        "Baseline point: c_Ga = 34.6 mg/L, market price = 423 €/kg.",
        "Q* = break-even throughput (LCOGa ≤ market price).",
        "Feed concentration varied via TEAConfig.c_ga_feed_mg_L.",
    ]
    (OUT_NOTES / "fig_5_10_viability_map_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    print("  -> fig_5_10_viability_map.svg, _data.csv, _notes.txt")


if __name__ == "__main__":
    main()
