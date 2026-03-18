"""Thesis table: Table 5.7 – Scenario matrix for route robustness.

Six scenarios at Q = 50 m³/d showing LCOGa, break-even throughput (Q*),
and dominant cost block for each route.

Scenarios:
  - Baseline: Reference case
  - S1: Conservative cost stress (+25% CapEx, +35% replacement, +10% labour)
  - S2: Operations stress (electricity 0.25 €/kWh, SX make-up +50%)
  - S3: Process optimisation (improved recovery + internal cost reduction)
  - S4: Decarbonised power (electricity 0.12 €/kWh, CO₂ tax 120 €/t)
  - S5: Multi-stage SX (sx_ga_to_loaded_organic ≈ 0.90, SX-CapEx × 1.4)

Outputs:
  - table_5_7_scenario_matrix.csv — scenario results
  - table_5_7_scenario_matrix.xlsx — Excel format (if openpyxl available)
  - table_5_7_scenario_matrix_notes.txt — analysis notes
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import numpy as np
import pandas as pd

import tea_model_ga_thesis as tea


# Output directory
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
Q_FEED = 50.0
Q_GRID = np.linspace(10, 100, 91)
MARKET_PRICE = 423.0


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


def _dominant_cost_block(Q_feed, route, config):
    """Return dominant cost block (CapEx-Sep or OpEx) at Q_feed for route."""
    cb = tea.calc_cost_breakdown(Q_feed, route=route, config=config)
    ga_out = config.recovery_rate_total_ix if route == "IX" else config.recovery_rate_total_sx
    # Note: V08 uses 0.5 mg/L Ga feed (hardcoded), adjust if needed
    capex_sep_per_kg = cb["CapEx-Sep"] / ga_out if ga_out > 0 else 0
    opex_per_kg = cb["OpEx"] / ga_out if ga_out > 0 else 0
    return "CapEx-Sep" if capex_sep_per_kg >= opex_per_kg else "OpEx"


def _run_scenario(name, config):
    """Run scenario with given config and compute metrics."""
    lco_ix = tea.calc_lco_ga(Q_FEED, route="IX", config=config)
    lco_sx = tea.calc_lco_ga(Q_FEED, route="SX", config=config)

    lco_ix_arr = np.array([tea.calc_lco_ga(float(q), route="IX", config=config) for q in Q_GRID])
    lco_sx_arr = np.array([tea.calc_lco_ga(float(q), route="SX", config=config) for q in Q_GRID])

    qstar_ix = _q_star(Q_GRID, lco_ix_arr, MARKET_PRICE)
    qstar_sx = _q_star(Q_GRID, lco_sx_arr, MARKET_PRICE)

    dom_ix = _dominant_cost_block(Q_FEED, "IX", config)
    dom_sx = _dominant_cost_block(Q_FEED, "SX", config)

    return {
        "name": name,
        "lco_ix": lco_ix,
        "lco_sx": lco_sx,
        "qstar_ix": qstar_ix,
        "qstar_sx": qstar_sx,
        "dom_ix": dom_ix,
        "dom_sx": dom_sx,
    }


def main():
    """Generate Table 5.7 scenario matrix."""
    config_base = tea.DEFAULT_CONFIG

    # Baseline
    baseline_config = config_base
    scenario_baseline = _run_scenario("Baseline", baseline_config)

    # S1: Conservative cost stress
    s1_config = config_base.replace(
        capex_multiplier=1.25,
        rep_multiplier=1.35,
        labour_multiplier=1.10,
    )
    scenario_s1 = _run_scenario("S1 Conservative cost stress", s1_config)

    # S2: Operations stress
    s2_config = config_base.replace(
        electricity_price=0.25,
        sx_makeup_rate_annual=0.15 * 1.5,  # +50% makeup
    )
    scenario_s2 = _run_scenario("S2 Operations stress", s2_config)

    # S3: Process optimisation (internal improvements only)
    recoveries_dict = dict(config_base.recoveries)
    recoveries_dict["ix_ga_to_eluat"] = 0.985
    recoveries_dict["sx_ga_to_loaded_organic"] = 0.82
    recoveries_dict["sx_ga_strip_to_aqueous"] = 0.985
    recoveries_dict["leach_ga_to_leachate"] = 0.985
    recoveries_dict["ew_ga_to_product"] = 0.92
    s3_config = config_base.replace(
        opex_multiplier=0.90,
        recoveries=MappingProxyType(recoveries_dict),
    )
    scenario_s3 = _run_scenario("S3 Process optimisation", s3_config)

    # S4: Decarbonised power context
    s4_config = config_base.replace(
        electricity_price=0.12,
        co2_tax_per_ton=120.0,
    )
    scenario_s4 = _run_scenario("S4 Decarbonised power context", s4_config)

    # S5: Multi-stage SX
    recoveries_dict = dict(config_base.recoveries)
    recoveries_dict["sx_ga_to_loaded_organic"] = 0.90
    s5_config = config_base.replace(
        sx_capex_multiplier=1.4,
        recoveries=MappingProxyType(recoveries_dict),
    )
    scenario_s5 = _run_scenario("S5 Multi-stage SX", s5_config)

    scenarios = [
        (scenario_baseline, "Reference case from 5.2.1"),
        (scenario_s1, "+25 % CapEx, +35 % replacement, +10 % labour"),
        (scenario_s2, "Electricity price = 0.25 € kWh⁻¹, SX make-up +50 %"),
        (scenario_s3, "Improved recovery chain + −10% chemicals/M&O (internal)"),
        (scenario_s4, "Electricity price = 0.12 € kWh⁻¹, CO₂ tax = 120 € t⁻¹"),
        (scenario_s5, "sx_ga_to_loaded_organic ≈ 0.90, SX-CapEx × 1.4"),
    ]

    # Build table
    table_rows = []
    for scenario, key_assumption in scenarios:
        qix = f"{scenario['qstar_ix']:.1f}" if scenario["qstar_ix"] is not None else "—"
        qsx = f"{scenario['qstar_sx']:.1f}" if scenario["qstar_sx"] is not None else "—"
        dom = f"IX: {scenario['dom_ix']}; SX: {scenario['dom_sx']}"
        table_rows.append({
            "Scenario": scenario["name"],
            "Key_assumption_set": key_assumption,
            "LCOGa_IX_50": scenario["lco_ix"],
            "LCOGa_SX_50": scenario["lco_sx"],
            "Q_star_IX": scenario["qstar_ix"],
            "Q_star_SX": scenario["qstar_sx"],
            "Dominant_cost_block_Q50": dom,
        })

    df = pd.DataFrame(table_rows)
    df.to_csv(OUT_DIR / "table_5_7_scenario_matrix.csv", index=False)

    try:
        df.to_excel(OUT_DIR / "table_5_7_scenario_matrix.xlsx", index=False)
    except ImportError:
        pass

    notes = [
        "S5: SX CapEx × 1.4 is a stylised engineering assumption for multi-stage mixer-settler capacity.",
        "Q* computed via linear interpolation where LCOGa crosses market price (423 €/kg).",
        "Dominant cost block: CapEx-Sep vs OpEx per kg Ga.",
    ]
    (OUT_DIR / "table_5_7_scenario_matrix_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    # Markdown table
    print("=" * 80)
    print("Table 5.7: Scenario matrix for route robustness (Q = 50 m³ d⁻¹)")
    print("=" * 80)
    print()
    print("| Scenario | Key assumption set | LCOGa IX (50) | LCOGa SX (50) | Q*_IX | Q*_SX | "
          "Dominant cost block at Q = 50 |")
    print("|---|---|---:|---:|---:|---:|---|")

    for scenario, _ in scenarios:
        qix = f"{scenario['qstar_ix']:.1f}" if scenario["qstar_ix"] is not None else "—"
        qsx = f"{scenario['qstar_sx']:.1f}" if scenario["qstar_sx"] is not None else "—"
        dom = f"IX: {scenario['dom_ix']}; SX: {scenario['dom_sx']}"
        print(
            f"| {scenario['name']} | [see CSV] | "
            f"{scenario['lco_ix']:.1f} | {scenario['lco_sx']:.1f} | {qix} | {qsx} | {dom} |"
        )

    print()
    print("  -> table_5_7_scenario_matrix.csv, .xlsx, _notes.txt")


if __name__ == "__main__":
    main()
