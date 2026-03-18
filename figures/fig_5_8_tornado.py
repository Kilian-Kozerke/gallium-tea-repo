"""Thesis figure: Figure 5.8 – Tornado sensitivity of LCOGa at Q=50 m³/d.

Generates a tornado diagram showing the sensitivity of LCOGa to parameter
variations for both IX and SX routes at Q = 50 m³/d (local sensitivity).

Outputs:
  - fig_5_8_tornado_q50.svg — figure
  - fig_5_8_tornado_q50_data.csv — parameter sensitivity data
  - fig_5_8_tornado_q50_notes.txt — analysis notes
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
Q_TORNADO = 50.0
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


def _set_param(name: str, value: float, config: tea.TEAConfig) -> tea.TEAConfig:
    """Create a new config with a parameter changed."""
    if name == "C_Ga_feed_mg_L":
        return config.replace(c_ga_feed_mg_L=float(value))
    if name == "electricity_price_eur_per_kwh":
        return config.replace(electricity_price=float(value))
    if name == "capex_multiplier":
        return config.replace(capex_multiplier=float(value))
    if name == "rep_multiplier":
        return config.replace(rep_multiplier=float(value))
    if name == "opex_multiplier":
        return config.replace(opex_multiplier=float(value))
    if name == "labour_multiplier":
        return config.replace(labour_multiplier=float(value))
    if name == "co2_tax_multiplier":
        return config.replace(co2_tax_per_ton=60.0 * float(value))
    if name.startswith("recovery:"):
        key = name.split(":", 1)[1]
        recoveries_dict = dict(config.recoveries)
        recoveries_dict[key] = float(value)
        from types import MappingProxyType
        return config.replace(recoveries=MappingProxyType(recoveries_dict))
    raise ValueError(f"Unknown sensitivity parameter: {name}")


def _pretty_label(name: str) -> str:
    """Thesis-consistent parameter labels."""
    mapping = {
        "C_Ga_feed_mg_L": r"feed concentration $c_{Ga}$",
        "electricity_price_eur_per_kwh": "electricity price",
        "co2_tax_multiplier": r"CO$_2$ tax",
        "capex_multiplier": "CapEx",
        "rep_multiplier": "replacement",
        "opex_multiplier": "OpEx",
        "labour_multiplier": "labour",
        "recovery:ro_ga_to_concentrate": "RO Ga recovery to concentrate",
        "recovery:ix_ga_to_eluat": "IX separation recovery",
        "recovery:sx_ga_to_loaded_organic": "SX extraction recovery",
        "recovery:sx_ga_strip_to_aqueous": "SX strip recovery",
        "recovery:leach_ga_to_leachate": "leach recovery",
        "recovery:ew_ga_to_product": "electrowinning recovery",
    }
    return mapping.get(name, name)


def main():
    """Generate Figure 5.8 tornado sensitivity."""
    config = tea.DEFAULT_CONFIG

    # Parameter ranges (local uncertainty for V08)
    c_ga_base = 34.6  # FEED_BASELINE_TEMPLATE
    c_ga_low, c_ga_high = 25.0, 50.0
    el_base = config.electricity_price
    el_low, el_high = el_base * 0.80, el_base * 1.20
    param_ranges = {
        "C_Ga_feed_mg_L": (c_ga_low, c_ga_base, c_ga_high),
        "electricity_price_eur_per_kwh": (el_low, el_base, el_high),
        "co2_tax_multiplier": (0.5, 1.0, 2.0),
        "recovery:ro_ga_to_concentrate": (0.90, config.recoveries["ro_ga_to_concentrate"], 0.98),
        "recovery:ix_ga_to_eluat": (0.90, config.recoveries["ix_ga_to_eluat"], 0.98),
        "recovery:sx_ga_to_loaded_organic": (0.70, config.recoveries["sx_ga_to_loaded_organic"], 0.85),
        "recovery:sx_ga_strip_to_aqueous": (0.90, config.recoveries["sx_ga_strip_to_aqueous"], 0.98),
        "recovery:leach_ga_to_leachate": (0.90, config.recoveries["leach_ga_to_leachate"], 0.98),
        "recovery:ew_ga_to_product": (0.85, config.recoveries["ew_ga_to_product"], 0.95),
        "capex_multiplier": (0.80, 1.0, 1.20),
        "rep_multiplier": (0.70, 1.0, 1.30),
        "opex_multiplier": (0.80, 1.0, 1.20),
        "labour_multiplier": (0.90, 1.0, 1.10),
    }

    baseline_lco_ix = tea.calc_lco_ga(Q_TORNADO, route="IX", config=config)
    baseline_lco_sx = tea.calc_lco_ga(Q_TORNADO, route="SX", config=config)

    results = []
    for p, (low, base, high) in param_ranges.items():
        config_low = _set_param(p, low, config)
        config_high = _set_param(p, high, config)

        lco_ix_low = tea.calc_lco_ga(Q_TORNADO, route="IX", config=config_low)
        lco_sx_low = tea.calc_lco_ga(Q_TORNADO, route="SX", config=config_low)
        lco_ix_high = tea.calc_lco_ga(Q_TORNADO, route="IX", config=config_high)
        lco_sx_high = tea.calc_lco_ga(Q_TORNADO, route="SX", config=config_high)

        pct_ix_low = (lco_ix_low - baseline_lco_ix) / baseline_lco_ix * 100
        pct_ix_high = (lco_ix_high - baseline_lco_ix) / baseline_lco_ix * 100
        pct_sx_low = (lco_sx_low - baseline_lco_sx) / baseline_lco_sx * 100
        pct_sx_high = (lco_sx_high - baseline_lco_sx) / baseline_lco_sx * 100

        results.append({
            "param": p,
            "label": _pretty_label(p),
            "ix_pct_low": pct_ix_low,
            "ix_pct_high": pct_ix_high,
            "sx_pct_low": pct_sx_low,
            "sx_pct_high": pct_sx_high,
            "ix_span": max(abs(pct_ix_low), abs(pct_ix_high)),
            "sx_span": max(abs(pct_sx_low), abs(pct_sx_high)),
        })

    df = pd.DataFrame(results)
    df["max_span"] = df[["ix_span", "sx_span"]].max(axis=1)

    # Route-relevant params per panel
    ix_exclude = {"recovery:sx_ga_to_loaded_organic", "recovery:sx_ga_strip_to_aqueous"}
    sx_exclude = {"recovery:ix_ga_to_eluat"}
    df_ix = df[~df["param"].isin(ix_exclude)].copy()
    df_sx = df[~df["param"].isin(sx_exclude)].copy()
    df_ix = df_ix.sort_values("ix_span", ascending=True).reset_index(drop=True)
    df_sx = df_sx.sort_values("sx_span", ascending=True).reset_index(drop=True)

    # Export data
    export_df = df.sort_values("max_span", ascending=True)[["param", "label", "ix_pct_low", "ix_pct_high", "sx_pct_low", "sx_pct_high"]].copy()
    export_df.to_csv(OUT_CSV / "fig_5_8_tornado_data.csv", index=False)

    # Notes
    rank_ix = df.sort_values("ix_span", ascending=False).reset_index(drop=True)
    rank_sx = df.sort_values("sx_span", ascending=False).reset_index(drop=True)
    notes = [
        f"Baseline LCOGa IX @ Q=50: {baseline_lco_ix:.2f} €/kg",
        f"Baseline LCOGa SX @ Q=50: {baseline_lco_sx:.2f} €/kg",
        "",
        "IX sensitivity ranking (most to least):",
        *[f"  {i+1}. {row['label']}" for i, row in rank_ix.iterrows()],
        "",
        "SX sensitivity ranking (most to least):",
        *[f"  {i+1}. {row['label']}" for i, row in rank_sx.iterrows()],
    ]
    (OUT_NOTES / "fig_5_8_tornado_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    # X-axis limits
    ix_vals = np.concatenate([df_ix["ix_pct_low"].values, df_ix["ix_pct_high"].values])
    sx_vals = np.concatenate([df_sx["sx_pct_low"].values, df_sx["sx_pct_high"].values])
    margin = 2.0
    xlim_ix = (ix_vals.min() - margin, ix_vals.max() + margin)
    xlim_sx = (sx_vals.min() - margin, sx_vals.max() + margin)

    # Plot
    _set_style()
    fig, (ax_ix, ax_sx) = plt.subplots(1, 2, figsize=(FIG_W_STD, FIG_H_STD), sharey=False)
    fig.suptitle(r"Q = 50 m$^3$ d$^{-1}$ (local sensitivity; route-relevant parameters per panel)", fontsize=9, y=1.02)
    y_ix = np.arange(len(df_ix))
    y_sx = np.arange(len(df_sx))

    # IX panel
    ax_ix.barh(y_ix, df_ix["ix_pct_low"].values, color=PALETTE["IX"], alpha=0.6, label="Low")
    ax_ix.barh(y_ix, df_ix["ix_pct_high"].values, color=PALETTE["IX"], alpha=0.9, label="High")
    ax_ix.axvline(0, color=PALETTE["Grey_dark"], linewidth=1)
    ax_ix.set_title(f"IX (baseline LCOGa = {baseline_lco_ix:.0f} € kg$^{-1}$)")
    ax_ix.set_xlabel(r"$\Delta$LCOGa [%]")
    ax_ix.set_yticks(y_ix)
    ax_ix.set_yticklabels(df_ix["label"], fontsize=8)
    ax_ix.set_xlim(xlim_ix)
    ax_ix.legend(loc="lower right", fontsize=8)
    ax_ix.grid(True, alpha=0.3, axis="x")

    # SX panel
    ax_sx.barh(y_sx, df_sx["sx_pct_low"].values, color=PALETTE["SX"], alpha=0.6, label="Low")
    ax_sx.barh(y_sx, df_sx["sx_pct_high"].values, color=PALETTE["SX"], alpha=0.9, label="High")
    ax_sx.axvline(0, color=PALETTE["Grey_dark"], linewidth=1)
    ax_sx.set_title(f"SX (baseline LCOGa = {baseline_lco_sx:.0f} € kg$^{-1}$)")
    ax_sx.set_xlabel(r"$\Delta$LCOGa [%]")
    ax_sx.set_yticks(y_sx)
    ax_sx.set_yticklabels(df_sx["label"], fontsize=8)
    ax_sx.set_xlim(xlim_sx)
    ax_sx.legend(loc="lower right", fontsize=8)
    ax_sx.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "fig_5_8_tornado_sensitivity.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    print("  -> fig_5_8_tornado_sensitivity.svg, fig_5_8_tornado_data.csv, fig_5_8_tornado_notes.txt")


if __name__ == "__main__":
    main()
