"""
LCA aggregation: GWP [kg CO2-eq / kg Ga] from energy + material flows.
Thesis model adapter — uses the embedded `lca_factor_kg_co2eq_per_unit` values
already stored in each inventory entry of tea_model_ga_thesis.

Preserves the same public API as the V06 version:
  calc_lca_gwp_total_per_m3_feed(Q_feed, route) -> float
  calc_lca_gwp_total_per_kg_ga(Q_feed, route)   -> float
  calc_lca_gwp_per_kg_ga(Q_feed, route)          -> dict (total, by_process, gwp_per_m3)
  calc_lca_gwp_by_driver_per_kg_ga(Q_feed, route) -> dict per driver
  export_lca_csv(q_values, out_path)             -> Path

Impact factors are sourced from:
  - Electricity: 0.363 kg CO2-eq/kWh (UBA 2024 German grid average)
  - Reagents/resins/solvents: ReCiPe 2016 Midpoint (H), ecoinvent 3.5 cutoff
    (see lca_recipe_factors.py for full factor table)

Every inventory entry from calc_*_material_consumption() and
calc_*_energy_consumption() already carries its `lca_factor_kg_co2eq_per_unit`
so no manual factor look-up is needed here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Thesis model import
# ---------------------------------------------------------------------------
if "tea_model_ga_thesis" in sys.modules:
    tea = sys.modules["tea_model_ga_thesis"]
else:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import tea_model_ga_thesis as tea

# ---------------------------------------------------------------------------
# Process step order (for stacked-bar figures)
# ---------------------------------------------------------------------------
PROCESS_ORDER = [
    "Filtration",
    "RO_Split",
    "pH_Adjust",
    "Separation",
    "Precipitation",
    "Selective_Leaching",
    "Electrowinning",
]

# Driver aggregation (Luo 2024-comparable breakdown)
DRIVER_ORDER = ["Electricity", "NaOH", "HCl", "H2SO4", "Cyanex", "Kerosene", "Ti", "Resin"]

# Map V08 inventory flow keys -> DRIVER_ORDER entry
_FLOW_TO_DRIVER: dict[str, str] = {
    "electricity":            "Electricity",
    "hcl_32wt":              "HCl",
    "naoh_50wt":             "NaOH",
    "h2so4_01M":             "H2SO4",
    "ix_resin":              "Resin",
    "organic":               "Cyanex",
    "kerosene":              "Kerosene",
    "ti_cathode_replacement": "Ti",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ga_kg_per_m3(route: str, config=None) -> float:
    """kg Ga metal output per m3 feed (via V08 annual production function)."""
    config = config or tea.DEFAULT_CONFIG
    return tea.calc_annual_production(1.0, route=route, config=config) / config.operating_days


def _gwp_from_inventory(inv: dict) -> float:
    """Sum kg CO2-eq / m3 feed from a single inventory dict."""
    total = 0.0
    for entry in inv.values():
        if not isinstance(entry, dict):
            continue
        factor = entry.get("lca_factor_kg_co2eq_per_unit") or 0.0
        cons   = entry.get("consumption_per_m3") or 0.0
        total += factor * cons
    return total


def _gwp_by_driver_from_inventory(inv: dict) -> dict[str, float]:
    """Return {driver: kg CO2-eq / m3 feed} for a single inventory dict."""
    out = {d: 0.0 for d in DRIVER_ORDER}
    for flow_key, entry in inv.items():
        if not isinstance(entry, dict):
            continue
        factor = entry.get("lca_factor_kg_co2eq_per_unit") or 0.0
        cons   = entry.get("consumption_per_m3") or 0.0
        gwp    = factor * cons
        driver = _FLOW_TO_DRIVER.get(flow_key)
        if driver:
            out[driver] += gwp
    return out


# ---------------------------------------------------------------------------
# Per-process GWP collectors
# ---------------------------------------------------------------------------

def _collect_by_process(Q: float, route: str, config=None) -> dict[str, float]:
    """
    Return {process_step: kg CO2-eq / m3 feed} for the given route.

    Uses calc_*_material_consumption + calc_*_energy_consumption from V08
    and relies on the embedded lca_factor_kg_co2eq_per_unit in each entry.
    Flows without a characterised factor (wash_water, filterbags) contribute 0.
    """
    config = config or tea.DEFAULT_CONFIG
    route  = route.upper()

    result: dict[str, float] = {p: 0.0 for p in PROCESS_ORDER}

    # Shared pre-treatment steps
    result["Filtration"] = (
        _gwp_from_inventory(tea.calc_filtration_material_consumption(Q, config))
        + _gwp_from_inventory(tea.calc_filtration_energy_consumption(Q, config))
    )
    result["RO_Split"] = (
        _gwp_from_inventory(tea.calc_ro_split_material_consumption(Q, config))
        + _gwp_from_inventory(tea.calc_ro_split_energy_consumption(Q, config))
    )
    result["pH_Adjust"] = (
        _gwp_from_inventory(tea.calc_ph_adjust_material_consumption(Q, config))
        + _gwp_from_inventory(tea.calc_ph_adjust_energy_consumption(Q, config))
    )

    # Route-specific separation and downstream steps
    if route == "IX":
        result["Separation"] = (
            _gwp_from_inventory(tea.calc_ix_material_consumption(Q, config))
            + _gwp_from_inventory(tea.calc_ix_energy_consumption(Q, config))
        )
        result["Precipitation"] = (
            _gwp_from_inventory(tea.calc_precipitation_material_consumption(Q, route="IX", config=config))
            + _gwp_from_inventory(tea.calc_precipitation_energy_consumption(Q, route="IX", config=config))
        )
        result["Selective_Leaching"] = (
            _gwp_from_inventory(tea.calc_selective_leaching_material_consumption(Q, route="IX", config=config))
            + _gwp_from_inventory(tea.calc_selective_leaching_energy_consumption(Q, route="IX", config=config))
        )
        result["Electrowinning"] = (
            _gwp_from_inventory(tea.calc_electrowinning_material_consumption(Q, route="IX", config=config))
            + _gwp_from_inventory(tea.calc_electrowinning_energy_consumption(Q, route="IX", config=config))
        )
    elif route == "SX":
        result["Separation"] = (
            _gwp_from_inventory(tea.calc_sx_material_consumption(Q, config))
            + _gwp_from_inventory(tea.calc_sx_energy_consumption(Q, config))
        )
        result["Precipitation"] = (
            _gwp_from_inventory(tea.calc_precipitation_material_consumption(Q, route="SX", config=config))
            + _gwp_from_inventory(tea.calc_precipitation_energy_consumption(Q, route="SX", config=config))
        )
        result["Selective_Leaching"] = (
            _gwp_from_inventory(tea.calc_selective_leaching_material_consumption(Q, route="SX", config=config))
            + _gwp_from_inventory(tea.calc_selective_leaching_energy_consumption(Q, route="SX", config=config))
        )
        result["Electrowinning"] = (
            _gwp_from_inventory(tea.calc_electrowinning_material_consumption(Q, route="SX", config=config))
            + _gwp_from_inventory(tea.calc_electrowinning_energy_consumption(Q, route="SX", config=config))
        )
    else:
        raise ValueError(f"route must be 'IX' or 'SX', got {route!r}")

    return result


def _collect_by_driver(Q: float, route: str, config=None) -> dict[str, float]:
    """
    Return {driver: kg CO2-eq / m3 feed} for the given route.

    Aggregates across all process steps, attributing each flow to its chemical
    driver (Electricity, NaOH, HCl, H2SO4, Cyanex, Kerosene, Ti, Resin).
    """
    config = config or tea.DEFAULT_CONFIG
    route  = route.upper()

    totals = {d: 0.0 for d in DRIVER_ORDER}

    def _add(*inv_dicts):
        for inv in inv_dicts:
            for drv, v in _gwp_by_driver_from_inventory(inv).items():
                totals[drv] += v

    _add(
        tea.calc_filtration_material_consumption(Q, config),
        tea.calc_filtration_energy_consumption(Q, config),
        tea.calc_ro_split_material_consumption(Q, config),
        tea.calc_ro_split_energy_consumption(Q, config),
        tea.calc_ph_adjust_material_consumption(Q, config),
        tea.calc_ph_adjust_energy_consumption(Q, config),
    )

    if route == "IX":
        _add(
            tea.calc_ix_material_consumption(Q, config),
            tea.calc_ix_energy_consumption(Q, config),
            tea.calc_precipitation_material_consumption(Q, route="IX", config=config),
            tea.calc_precipitation_energy_consumption(Q, route="IX", config=config),
            tea.calc_selective_leaching_material_consumption(Q, route="IX", config=config),
            tea.calc_selective_leaching_energy_consumption(Q, route="IX", config=config),
            tea.calc_electrowinning_material_consumption(Q, route="IX", config=config),
            tea.calc_electrowinning_energy_consumption(Q, route="IX", config=config),
        )
    elif route == "SX":
        _add(
            tea.calc_sx_material_consumption(Q, config),
            tea.calc_sx_energy_consumption(Q, config),
            tea.calc_precipitation_material_consumption(Q, route="SX", config=config),
            tea.calc_precipitation_energy_consumption(Q, route="SX", config=config),
            tea.calc_selective_leaching_material_consumption(Q, route="SX", config=config),
            tea.calc_selective_leaching_energy_consumption(Q, route="SX", config=config),
            tea.calc_electrowinning_material_consumption(Q, route="SX", config=config),
            tea.calc_electrowinning_energy_consumption(Q, route="SX", config=config),
        )
    else:
        raise ValueError(f"route must be 'IX' or 'SX', got {route!r}")

    return totals


# ---------------------------------------------------------------------------
# Public API (identical signatures to V06 version)
# ---------------------------------------------------------------------------

def calc_lca_gwp_total_per_m3_feed(Q_feed: float, route: str, config=None) -> float:
    """Return the total modeled GWP in ``kg CO2-eq / m3 feed`` for one route."""
    return sum(_collect_by_process(Q_feed, route, config).values())


def calc_lca_gwp_total_per_kg_ga(Q_feed: float, route: str, config=None) -> float:
    """Return the total modeled GWP in ``kg CO2-eq / kg Ga`` for one route."""
    return calc_lca_gwp_per_kg_ga(Q_feed, route, config)["total"]


def calc_lca_gwp_per_kg_ga(Q_feed: float, route: str, config=None) -> dict[str, Any]:
    """
    Total and per-process GWP [kg CO2-eq / kg Ga] for given Q and route.

    Returns
    -------
    dict with keys:
      'total'      : float            -- total kg CO2-eq / kg Ga
      'by_process' : dict[str, float] -- per process step kg CO2-eq / kg Ga
      'gwp_per_m3' : float            -- total kg CO2-eq / m3 feed
    """
    config     = config or tea.DEFAULT_CONFIG
    by_proc_m3 = _collect_by_process(Q_feed, route, config)
    total_m3   = sum(by_proc_m3.values())
    ga         = _ga_kg_per_m3(route, config)

    total_per_kg  = total_m3 / ga if ga > 0 else 0.0
    by_process_kg = {k: (v / ga) if ga > 0 else 0.0 for k, v in by_proc_m3.items()}

    return {
        "total":      total_per_kg,
        "by_process": by_process_kg,
        "gwp_per_m3": total_m3,
    }


def calc_lca_gwp_by_driver_per_kg_ga(Q_feed: float, route: str, config=None) -> dict[str, float]:
    """
    GWP [kg CO2-eq / kg Ga] per chemical driver (Luo 2024-comparable).

    Drivers: Electricity, NaOH, HCl, H2SO4, Cyanex, Kerosene, Ti, Resin.
    Each flow is fully attributed to one driver -- no opaque 'Other' category.
    """
    config    = config or tea.DEFAULT_CONFIG
    by_drv_m3 = _collect_by_driver(Q_feed, route, config)
    ga        = _ga_kg_per_m3(route, config)
    return {d: (v / ga) if ga > 0 else 0.0 for d, v in by_drv_m3.items()}


def export_lca_csv(
    q_values: list | None = None,
    out_path: "Path | None" = None,
    config=None,
) -> Path:
    """
    Export LCA results (by process step) to CSV.

    Also appends the Luo (2024) benchmark total for comparison.

    Parameters
    ----------
    q_values : Feed flow rates to include (default: [10, 100]).
    out_path : Output CSV path (default: lca/outputs/lca_carbon_load.csv).
    config   : TEAConfig instance (default: tea.DEFAULT_CONFIG).
    """
    q_values = q_values or [10, 100]
    out_path = Path(out_path or (Path(__file__).resolve().parent / "outputs" / "lca_carbon_load.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for q in q_values:
        for route in ("IX", "SX"):
            res = calc_lca_gwp_per_kg_ga(q, route, config)
            for step in PROCESS_ORDER:
                rows.append({
                    "Q_m3_per_d":             q,
                    "route":                  route,
                    "process_step":           step,
                    "gwp_kgCO2eq_per_kgGa":   res["by_process"][step],
                    "total_kgCO2eq_per_kgGa": res["total"],
                })

    # Luo (2024) benchmark -- single total bar
    rows.append({
        "Q_m3_per_d":             None,
        "route":                  "Luo2024",
        "process_step":           "total",
        "gwp_kgCO2eq_per_kgGa":   282.0,
        "total_kgCO2eq_per_kgGa": 282.0,
    })

    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_path, index=False)
    except ImportError:
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return out_path


# ---------------------------------------------------------------------------
# Quick self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("LCA GWP [kg CO2-eq / kg Ga]  -- V08 lca_aggregation\n" + "=" * 60)
    print(f"{'Q':>5}  {'Route':>5}  {'Total':>8}  {'Elec':>7}  {'NaOH':>7}  {'HCl':>6}")
    print("-" * 60)

    abstract = {(10, "IX"): 236.8, (10, "SX"): 418.5, (100, "IX"): 67.9, (100, "SX"): 183.4}

    for Q in [10, 100]:
        for route in ["IX", "SX"]:
            total   = calc_lca_gwp_total_per_kg_ga(Q, route)
            drivers = calc_lca_gwp_by_driver_per_kg_ga(Q, route)
            ref     = abstract.get((Q, route), 0)
            delta   = total - ref
            ok      = "OK" if abs(delta) < 10 else "FAIL"
            print(
                f"Q={Q:>3}  {route:>5}  {total:>8.1f}  "
                f"{drivers['Electricity']:>7.1f}  {drivers['NaOH']:>7.1f}  "
                f"{drivers['HCl']:>6.1f}  (abstract={ref:.1f} delta={delta:+.1f}) {ok}"
            )

    print("\nBy process step at Q=10 IX:")
    bp = calc_lca_gwp_per_kg_ga(10, "IX")["by_process"]
    for step, val in bp.items():
        bar = "|" * max(1, int(val / 3))
        print(f"  {step:>20s}  {val:6.1f}  {bar}")

    print()
    p = export_lca_csv()
    print(f"CSV exported -> {p}")
