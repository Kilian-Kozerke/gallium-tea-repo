"""Thesis table: Table 5.8 – Q* vs feed concentration at market price 423 €/kg.

Break-even throughput (Q*) for IX and SX routes at selected gallium feed
concentrations (10, 20, 34.6, 50, 75, 100 mg/L) with market price = 423 €/kg.

Outputs:
  - table_5_8_qstar_concentration.csv
  - table_5_8_qstar_concentration_notes.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import tea_model_ga_thesis as tea


# Output directory
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
Q_MIN, Q_MAX = 1.0, 100.0
Q_GRID = np.linspace(Q_MIN, Q_MAX, 100)
MARKET_PRICE = 423.0
C_VALS = [10, 20, 34.6, 50, 75, 100]


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
    """Generate Table 5.8: Q* vs c_Ga at market price = 423 €/kg."""
    config_base = tea.DEFAULT_CONFIG

    rows = []
    for route in ["IX", "SX"]:
        for c in C_VALS:
            cfg = config_base.replace(c_ga_feed_mg_L=float(c))
            lco_arr = np.array([
                tea.calc_lco_ga(float(q), route=route, config=cfg) for q in Q_GRID
            ])
            qs = _q_star(Q_GRID, lco_arr, MARKET_PRICE)
            if qs is None:
                qs_str = f"not reached (Q ≤ {int(Q_MAX)})"
            else:
                qs_str = f"{qs:.1f}"
            rows.append({
                "route": route,
                "c_Ga_mg_L": c,
                "Q_star_m3_per_day": qs_str,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_5_8_qstar_concentration.csv", index=False)

    notes = [
        "Table 5.8: Q* at market price = 423 €/kg for c_Ga = 10, 20, 34.6, 50, 75, 100 mg/L.",
        f"'not reached' means LCOGa > price at Q_max = {int(Q_MAX)} m³/d (no extrapolation).",
        "Q* = 1.0 indicates break-even at or below model lower bound (1 m³/d).",
    ]
    (OUT_DIR / "table_5_8_qstar_concentration_notes.txt").write_text(
        "\n".join(notes), encoding="utf-8"
    )

    print("=" * 80)
    print("Table 5.8: Q* vs feed concentration (market price = 423 €/kg)")
    print("=" * 80)
    pivot = df.pivot(index="c_Ga_mg_L", columns="route", values="Q_star_m3_per_day")
    print(pivot.to_string())
    print()
    print("  -> table_5_8_qstar_concentration.csv, _notes.txt")


if __name__ == "__main__":
    main()
