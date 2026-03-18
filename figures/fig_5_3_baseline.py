#!/usr/bin/env python3
"""Thesis Figure 5.3: Annual production and LCOGa vs throughput (baseline).

Two panels: (a) Annual production [kg a⁻¹] vs Q, (b) LCOGa [€ kg⁻¹] vs Q.
Market price line at 423 €/kg. Uses V08 model.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tea_model_ga_thesis as tea

OUT_FIGURES = REPO_ROOT / "outputs" / "figures"
OUT_CSV = REPO_ROOT / "outputs" / "csv"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_CSV.mkdir(parents=True, exist_ok=True)


def _calc_q_star(route: str, market_price: float, config=None):
    """Break-even Q where LCOGa <= market_price (linear interpolation)."""
    config = config or tea.DEFAULT_CONFIG
    q_grid = list(range(config.q_star_grid_start, config.q_star_grid_end + 1, config.q_star_grid_step))
    lco = [tea.calc_lco_ga(float(q), route=route, config=config) for q in q_grid]
    if lco[0] <= market_price:
        return float(q_grid[0])
    if lco[-1] > market_price:
        return None
    for i in range(len(q_grid) - 1):
        y0, y1 = lco[i], lco[i + 1]
        if (y0 - market_price) * (y1 - market_price) <= 0:
            x0, x1 = float(q_grid[i]), float(q_grid[i + 1])
            if y1 == y0:
                return x0
            return x0 + (market_price - y0) * (x1 - x0) / (y1 - y0)
    return None


def main():
    config = tea.DEFAULT_CONFIG
    market_price = config.ga_market_price_base
    q_grid = list(range(1, 101))

    data = []
    for Q in q_grid:
        prod_ix = tea.calc_annual_production(Q, route="IX", config=config)
        prod_sx = tea.calc_annual_production(Q, route="SX", config=config)
        lco_ix = tea.calc_lco_ga(Q, route="IX", config=config)
        lco_sx = tea.calc_lco_ga(Q, route="SX", config=config)
        data.append({
            "Q": Q,
            "prod_IX": prod_ix,
            "prod_SX": prod_sx,
            "LCOGa_IX": lco_ix,
            "LCOGa_SX": lco_sx,
        })

    q_star_ix = _calc_q_star("IX", market_price, config)
    q_star_sx = _calc_q_star("SX", market_price, config)

    # CSV
    csv_path = OUT_CSV / "fig_5_3_full_grid.csv"
    with open(csv_path, "w") as f:
        f.write("Q_m3_per_d,annual_prod_IX_kg_per_yr,annual_prod_SX_kg_per_yr,LCOGa_IX_EUR_per_kg,LCOGa_SX_EUR_per_kg\n")
        for row in data:
            f.write(f"{row['Q']},{row['prod_IX']:.6f},{row['prod_SX']:.6f},{row['LCOGa_IX']:.6f},{row['LCOGa_SX']:.6f}\n")

    # SVG (thesis style: 840×360, two panels)
    w, h = 400, 280
    total_w, total_h = w * 2 + 40, h + 80
    color_ix, color_sx, color_market = "#00549F", "#8EBAE5", "#64748b"
    ymax_prod, ymax_lco = 1100, 800

    def scale_y_prod(y, ymax=ymax_prod):
        return h - 40 - (y / ymax) * (h - 80)

    def scale_y_lco(y, ymax=ymax_lco):
        return h - 40 - (y / ymax) * (h - 80)

    def path_pts(data, key_ix, key_sx, scale_fn, q_min=1, q_max=100):
        pts_ix = []
        pts_sx = []
        for r in data:
            if not (q_min <= r["Q"] <= q_max):
                continue
            x = 50 + (r["Q"] - q_min) / (q_max - q_min) * (w - 60)
            pts_ix.append(f"{x:.1f},{scale_fn(r[key_ix]):.1f}")
            pts_sx.append(f"{x:.1f},{scale_fn(r[key_sx]):.1f}")
        return " ".join(pts_ix), " ".join(pts_sx)

    pts_ix_a, pts_sx_a = path_pts(data, "prod_IX", "prod_SX", scale_y_prod)
    pts_ix_b, pts_sx_b = path_pts(data, "LCOGa_IX", "LCOGa_SX", scale_y_lco, q_min=10, q_max=100)
    y_market = scale_y_lco(market_price)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" width="{total_w}" height="{total_h}">
  <defs>
    <style>
      .axis {{ stroke: #475569; stroke-width: 1; fill: none; }}
      .grid {{ stroke: #e2e8f0; stroke-width: 0.5; stroke-dasharray: 2,2; }}
      .label {{ font-family: Arial, sans-serif; font-size: 8pt; fill: #334155; }}
      .title {{ font-family: Arial, sans-serif; font-size: 8pt; font-weight: bold; fill: #0f172a; }}
      .legend {{ font-family: Arial, sans-serif; font-size: 8pt; fill: #334155; }}
    </style>
  </defs>
  <rect width="{total_w}" height="{total_h}" fill="#fafafa"/>
  
  <!-- Panel A: Annual production -->
  <g transform="translate(20, 20)">
    <text x="0" y="-5" class="title">(a) Annual production</text>
    <text x="0" y="{h-20}" class="label">Q [m³ d⁻¹]</text>
    <text x="-140" y="15" class="label" transform="rotate(-90, -140, 15)">Production [kg a⁻¹]</text>
    <line x1="50" y1="40" x2="50" y2="{h-40}" class="axis"/>
    <line x1="50" y1="{h-40}" x2="{w-10}" y2="{h-40}" class="axis"/>
    <polyline points="{pts_ix_a}" fill="none" stroke="{color_ix}" stroke-width="2"/>
    <polyline points="{pts_sx_a}" fill="none" stroke="{color_sx}" stroke-width="2"/>
    <line x1="280" y1="25" x2="300" y2="25" stroke="{color_ix}" stroke-width="2"/>
    <text x="305" y="28" class="legend">IX</text>
    <line x1="280" y1="40" x2="300" y2="40" stroke="{color_sx}" stroke-width="2"/>
    <text x="305" y="43" class="legend">SX</text>
  </g>

  <!-- Panel B: LCOGa -->
  <g transform="translate({w+60}, 20)">
    <text x="0" y="-5" class="title">(b) LCOGa</text>
    <text x="0" y="{h-20}" class="label">Q [m³ d⁻¹]</text>
    <text x="-140" y="15" class="label" transform="rotate(-90, -140, 15)">LCOGa [€ kg⁻¹]</text>
    <line x1="50" y1="40" x2="50" y2="{h-40}" class="axis"/>
    <line x1="50" y1="{h-40}" x2="{w-10}" y2="{h-40}" class="axis"/>
    <line x1="50" y1="{y_market:.1f}" x2="{w-10}" y2="{y_market:.1f}" stroke="{color_market}" stroke-width="1" stroke-dasharray="4,2"/>
    <polyline points="{pts_ix_b}" fill="none" stroke="{color_ix}" stroke-width="2"/>
    <polyline points="{pts_sx_b}" fill="none" stroke="{color_sx}" stroke-width="2"/>
    <line x1="280" y1="25" x2="300" y2="25" stroke="{color_ix}" stroke-width="2"/>
    <text x="305" y="28" class="legend">IX</text>
    <line x1="280" y1="40" x2="300" y2="40" stroke="{color_sx}" stroke-width="2"/>
    <text x="305" y="43" class="legend">SX</text>
    <line x1="280" y1="55" x2="300" y2="55" stroke="{color_market}" stroke-width="1" stroke-dasharray="4,2"/>
    <text x="305" y="58" class="legend">423 €/kg</text>
  </g>
</svg>
'''

    svg_path = OUT_FIGURES / "fig_5_3_thesis_baseline.svg"
    with open(svg_path, "w") as f:
        f.write(svg)

    print(f"  -> {svg_path.name}, {csv_path.name}")
    if q_star_ix is not None:
        print(f"  Q*_IX = {q_star_ix:.2f} m³/d")
    if q_star_sx is not None:
        print(f"  Q*_SX = {q_star_sx:.2f} m³/d")


if __name__ == "__main__":
    main()
