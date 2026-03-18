"""Thesis figure: Figure 5.9 – P10/P50/P90 uncertainty bands of LCOGa.

Monte Carlo sensitivity analysis with triangular distributions. Generates
P10, P50, P90 percentile bands across throughput ranges for both routes.

Outputs:
  - fig_5_9_uncertainty_bands.svg — figure
  - fig_5_9_uncertainty_bands_data.csv — raw percentile data
  - fig_5_9_uncertainty_bands_notes.txt — analysis notes
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
N_DRAWS = 1500
CM_TO_IN = 1 / 2.54
FIG_W_STD = 16 * CM_TO_IN
FIG_H_STD = 7 * CM_TO_IN

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


def main():
    """Generate Figure 5.9 uncertainty bands via Monte Carlo."""
    config = tea.DEFAULT_CONFIG

    # Triangular distribution bounds (from tornado: local ranges)
    el_base = config.electricity_price
    el_low, el_high = el_base * 0.80, el_base * 1.20
    dists = {
        "electricity_price_eur_per_kwh": (el_low, el_base, el_high),
        "labour_multiplier": (0.90, 1.0, 1.10),
        "capex_multiplier": (0.80, 1.0, 1.20),
        "rep_multiplier": (0.70, 1.0, 1.30),
        "opex_multiplier": (0.80, 1.0, 1.20),
        "recovery_ix": (0.90, config.recoveries["ix_ga_to_eluat"], 0.98),
        "recovery_sx": (0.70, config.recoveries["sx_ga_to_loaded_organic"], 0.85),
        "co2_tax_multiplier": (0.5, 1.0, 2.0),
    }

    def sample_triangular(low, mode, high):
        return np.random.triangular(low, mode, high)

    Q_values = [10, 20, 50, 100]

    rows = []
    for route in ["IX", "SX"]:
        for Q in Q_values:
            lco_samples = []
            for _ in range(N_DRAWS):
                # Sample parameters
                el_price = sample_triangular(*dists["electricity_price_eur_per_kwh"])
                labour_mult = sample_triangular(*dists["labour_multiplier"])
                capex_mult = sample_triangular(*dists["capex_multiplier"])
                rep_mult = sample_triangular(*dists["rep_multiplier"])
                opex_mult = sample_triangular(*dists["opex_multiplier"])
                co2_mult = sample_triangular(*dists["co2_tax_multiplier"])

                # Route-specific recovery
                if route == "IX":
                    recovery_val = sample_triangular(*dists["recovery_ix"])
                    recoveries_dict = dict(config.recoveries)
                    recoveries_dict["ix_ga_to_eluat"] = recovery_val
                else:
                    recovery_val = sample_triangular(*dists["recovery_sx"])
                    recoveries_dict = dict(config.recoveries)
                    recoveries_dict["sx_ga_to_loaded_organic"] = recovery_val

                from types import MappingProxyType
                sample_config = config.replace(
                    electricity_price=el_price,
                    labour_multiplier=labour_mult,
                    capex_multiplier=capex_mult,
                    rep_multiplier=rep_mult,
                    opex_multiplier=opex_mult,
                    co2_tax_per_ton=60.0 * co2_mult,
                    recoveries=MappingProxyType(recoveries_dict),
                )

                lco = tea.calc_lco_ga(float(Q), route=route, config=sample_config)
                lco_samples.append(lco)

            lco_arr = np.array(lco_samples)
            p10, p50, p90 = np.percentile(lco_arr, [10, 50, 90])
            baseline_lco = tea.calc_lco_ga(float(Q), route=route, config=config)

            rows.append({
                "route": route,
                "Q_m3_per_day": Q,
                "P10": p10,
                "P50": p50,
                "P90": p90,
                "baseline_LCOGa": baseline_lco,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV / "fig_5_9_uncertainty_bands_data.csv", index=False)

    # Notes
    notes = [
        "=== Figure 5.9 Uncertainty Bands – Uncertainty Assumptions ===",
        "",
        "1. Monte Carlo setup:",
        "   - 1,500 random draws per (route, Q) combination",
        "   - 8 combinations: IX/SX × Q = 10, 20, 50, 100 m³/d",
        "   - Total: 12,000 draws",
        "",
        "2. Sampled parameters (all independent within each draw):",
        "   - electricity_price_eur_per_kwh",
        "   - labour_multiplier",
        "   - capex_multiplier",
        "   - rep_multiplier",
        "   - opex_multiplier",
        "   - co2_tax_multiplier",
        "   - route-specific recovery: ix_ga_to_eluat (IX) or sx_ga_to_loaded_organic (SX)",
        "",
        "3. Distribution: Triangular (low, mode, high) for all parameters.",
        "",
        "4. Parameter bounds (low, mode=baseline, high):",
        f"   - electricity_price_eur_per_kwh: {el_low:.4f}, {el_base:.4f}, {el_high:.4f} €/kWh",
        "   - labour_multiplier: 0.90, 1.0, 1.10",
        "   - capex_multiplier: 0.80, 1.0, 1.20",
        "   - rep_multiplier: 0.70, 1.0, 1.30",
        "   - opex_multiplier: 0.80, 1.0, 1.20",
        f"   - recovery_ix (ix_ga_to_eluat): 0.90, {config.recoveries['ix_ga_to_eluat']:.3f}, 0.98",
        f"   - recovery_sx (sx_ga_to_loaded_organic): 0.70, {config.recoveries['sx_ga_to_loaded_organic']:.3f}, 0.85",
        "   - co2_tax_multiplier: 0.5, 1.0, 2.0 (→ 30–120 €/t CO₂)",
        "",
        "5. Parameters held at baseline: RO, leach, electrowinning recoveries; precip_ga_to_cake; feed concentration.",
        "",
        "6. Independence: Parameters sampled independently within each draw.",
        "",
        "7. Same assumptions for all Q: Yes. Identical distribution bounds for Q = 10, 20, 50, 100.",
    ]
    (OUT_NOTES / "fig_5_9_uncertainty_bands_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    # Plot
    _set_style()
    fig, ax = plt.subplots(figsize=(FIG_W_STD, FIG_H_STD))

    for route, color in [("IX", PALETTE["IX"]), ("SX", PALETTE["SX"])]:
        sub = df[df["route"] == route]
        Q_vals = sub["Q_m3_per_day"].values
        ax.fill_between(Q_vals, sub["P10"], sub["P90"], color=color, alpha=0.25, label=f"{route} (P10–P90 band)")
        ax.plot(Q_vals, sub["P50"], color=color, linewidth=2, label=f"{route} (P50)")
        ax.plot(Q_vals, sub["baseline_LCOGa"], color=color, linestyle=":", linewidth=1.2, alpha=0.9, label=f"{route} baseline")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_xlabel(r"Throughput Q [m$^3$ d$^{-1}$]")
    ax.set_ylabel(r"LCOGa [€ kg$^{-1}$]")
    ax.set_title("P10/P50/P90 uncertainty bands of LCOGa versus throughput")
    ax.set_xlim(10, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "fig_5_9_uncertainty_bands.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    print("  -> fig_5_9_uncertainty_bands.svg, _data.csv, _notes.txt")


if __name__ == "__main__":
    main()
