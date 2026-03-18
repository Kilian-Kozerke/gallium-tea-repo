"""
Carbon-burden calculation helper for the Gallium TEA model.

Provides a standalone full-LCA carbon-burden calculation (kg CO₂-eq / kg Ga)
that does not require the lca_aggregation module.  Uses the `lca_factor_kg_co2eq_per_unit`
values embedded in each material/energy inventory entry by tea_model_ga_thesis and
the ReCiPe 2016 Midpoint (H) GWP factors in lca/lca_recipe_factors.py.

Scope
-----
The *full_lca* carbon burden accounts for:
  - direct electricity use (UBA 2024 German grid: 0.363 kg CO₂-eq / kWh)
  - all characterised reagent/solvent/resin flows (ecoinvent 3.5 cutoff)

Flows that are not yet characterised in the ecoinvent dataset (e.g. cartridge
filters, polypropylene bags) are silently skipped.  Their contribution is
expected to be negligible (<1 %) relative to the dominant electricity and
NaOH/HCl reagent flows.

The *energy_only* carbon burden is a subset: it includes only electricity
flows and is equivalent to the CO₂ term used inside `calc_lco_ga`.

Usage
-----
>>> import sys, types, pathlib
>>> # From the repo root:
>>> sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
>>> from utils.carbon_burden import calc_carbon_burden_full_lca, calc_carbon_burden_energy_only
>>> calc_carbon_burden_full_lca(Q=10,  route='IX')   # ≈ 236 kg CO₂-eq/kg Ga
>>> calc_carbon_burden_full_lca(Q=10,  route='SX')   # ≈ 420 kg CO₂-eq/kg Ga
>>> calc_carbon_burden_full_lca(Q=100, route='IX')   # ≈  67 kg CO₂-eq/kg Ga
>>> calc_carbon_burden_full_lca(Q=100, route='SX')   # ≈ 185 kg CO₂-eq/kg Ga

Reference values (abstract, Kozerke 2025):
  Q=10  IX: 236.8   SX: 418.5   Q=100  IX: 67.9   SX: 183.4
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — support running from repo root or scripts/ subdirectory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tea_model_ga_thesis as tea

# ---------------------------------------------------------------------------
# Inventory collectors: one function per process step, per route
# ---------------------------------------------------------------------------

def _all_ix_inventories(Q: float, config: tea.TEAConfig):
    """Collect all material + energy inventory dicts for the IX route."""
    return [
        tea.calc_filtration_material_consumption(Q, config),
        tea.calc_filtration_energy_consumption(Q, config),
        tea.calc_ro_split_material_consumption(Q, config),
        tea.calc_ro_split_energy_consumption(Q, config),
        tea.calc_ph_adjust_material_consumption(Q, config),
        tea.calc_ph_adjust_energy_consumption(Q, config),
        tea.calc_ix_material_consumption(Q, config),
        tea.calc_ix_energy_consumption(Q, config),
        tea.calc_precipitation_material_consumption(Q, route='IX', config=config),
        tea.calc_precipitation_energy_consumption(Q, route='IX', config=config),
        tea.calc_selective_leaching_material_consumption(Q, route='IX', config=config),
        tea.calc_selective_leaching_energy_consumption(Q, route='IX', config=config),
        tea.calc_electrowinning_material_consumption(Q, route='IX', config=config),
        tea.calc_electrowinning_energy_consumption(Q, route='IX', config=config),
    ]


def _all_sx_inventories(Q: float, config: tea.TEAConfig):
    """Collect all material + energy inventory dicts for the SX route."""
    return [
        tea.calc_filtration_material_consumption(Q, config),
        tea.calc_filtration_energy_consumption(Q, config),
        tea.calc_ro_split_material_consumption(Q, config),
        tea.calc_ro_split_energy_consumption(Q, config),
        tea.calc_ph_adjust_material_consumption(Q, config),
        tea.calc_ph_adjust_energy_consumption(Q, config),
        tea.calc_sx_material_consumption(Q, config),
        tea.calc_sx_energy_consumption(Q, config),
        tea.calc_precipitation_material_consumption(Q, route='SX', config=config),
        tea.calc_precipitation_energy_consumption(Q, route='SX', config=config),
        tea.calc_selective_leaching_material_consumption(Q, route='SX', config=config),
        tea.calc_selective_leaching_energy_consumption(Q, route='SX', config=config),
        tea.calc_electrowinning_material_consumption(Q, route='SX', config=config),
        tea.calc_electrowinning_energy_consumption(Q, route='SX', config=config),
    ]


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def _sum_gwp_per_m3(inventories: list[dict], energy_only: bool = False) -> float:
    """
    Sum GWP contributions (kg CO₂-eq / m³ feed) across all inventory dicts.

    Parameters
    ----------
    inventories : list of dicts returned by calc_*_material/energy_consumption()
    energy_only : if True, count only flows with lca_flow_key == 'electricity'

    Returns
    -------
    float  Total kg CO₂-eq per m³ feed
    """
    total = 0.0
    for inv in inventories:
        for flow_name, entry in inv.items():
            if not isinstance(entry, dict):
                continue
            gwp_factor = entry.get('lca_factor_kg_co2eq_per_unit') or 0.0
            if gwp_factor == 0.0:
                continue
            if energy_only and entry.get('lca_flow_key') != 'electricity':
                continue
            consumption = entry.get('consumption_per_m3') or 0.0
            total += consumption * gwp_factor
    return total


def _ga_output_per_m3(route: str, config: tea.TEAConfig) -> float:
    """
    Return gallium metal output in kg per m³ feed.

    Mirrors the private `_ga_output_per_m3` in tea_model_ga_thesis by calling
    `calc_annual_production` and back-converting with operating_days and Q=1.
    Using Q=1 and dividing by config.operating_days gives kg Ga / (m³ feed × day)
    × day = kg Ga / m³ feed.
    """
    return tea.calc_annual_production(1.0, route=route, config=config) / config.operating_days


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calc_carbon_burden_full_lca(
    Q: float,
    route: str = 'IX',
    config: Optional[tea.TEAConfig] = None,
) -> float:
    """
    Return the full-LCA carbon burden in kg CO₂-eq per kg Ga.

    Scope: electricity (UBA 2024) + all characterised reagent/solvent/resin
    flows (ReCiPe 2016, ecoinvent 3.5 cutoff).

    Parameters
    ----------
    Q      : Feed flow rate in m³/day.
    route  : 'IX' or 'SX'.
    config : TEAConfig instance.  Defaults to tea.DEFAULT_CONFIG.

    Returns
    -------
    float  kg CO₂-eq per kg Ga produced.

    Examples
    --------
    >>> calc_carbon_burden_full_lca(Q=10,  route='IX')   # ≈ 236
    >>> calc_carbon_burden_full_lca(Q=100, route='SX')   # ≈ 185
    """
    config = config or tea.DEFAULT_CONFIG
    route = route.upper()

    if route == 'IX':
        inventories = _all_ix_inventories(Q, config)
    elif route == 'SX':
        inventories = _all_sx_inventories(Q, config)
    else:
        raise ValueError(f"route must be 'IX' or 'SX', got {route!r}")

    gwp_per_m3 = _sum_gwp_per_m3(inventories, energy_only=False)
    ga_per_m3 = _ga_output_per_m3(route, config)

    if ga_per_m3 <= 0:
        raise ValueError(f"Non-positive Ga output ({ga_per_m3}) — check route and config.")

    return gwp_per_m3 / ga_per_m3


def calc_carbon_burden_energy_only(
    Q: float,
    route: str = 'IX',
    config: Optional[tea.TEAConfig] = None,
) -> float:
    """
    Return the energy-only carbon burden in kg CO₂-eq per kg Ga.

    Scope: direct electricity emissions only (same scope as the CO₂ tax in
    LCO-Ga).  This is a *lower bound* on the full lifecycle impact.

    Parameters
    ----------
    Q      : Feed flow rate in m³/day.
    route  : 'IX' or 'SX'.
    config : TEAConfig instance.  Defaults to tea.DEFAULT_CONFIG.

    Returns
    -------
    float  kg CO₂-eq per kg Ga produced.
    """
    config = config or tea.DEFAULT_CONFIG
    route = route.upper()

    if route == 'IX':
        inventories = _all_ix_inventories(Q, config)
    elif route == 'SX':
        inventories = _all_sx_inventories(Q, config)
    else:
        raise ValueError(f"route must be 'IX' or 'SX', got {route!r}")

    gwp_per_m3 = _sum_gwp_per_m3(inventories, energy_only=True)
    ga_per_m3 = _ga_output_per_m3(route, config)

    if ga_per_m3 <= 0:
        raise ValueError(f"Non-positive Ga output ({ga_per_m3}) — check route and config.")

    return gwp_per_m3 / ga_per_m3


def calc_carbon_burden_breakdown(
    Q: float,
    route: str = 'IX',
    config: Optional[tea.TEAConfig] = None,
) -> dict:
    """
    Return a detailed breakdown of GWP contributions by flow (kg CO₂-eq / kg Ga).

    Useful for identifying dominant contributors and for verification.

    Returns
    -------
    dict  Mapping flow_name -> {'gwp_per_m3': float, 'gwp_per_kg_ga': float,
                                 'consumption_per_m3': float,
                                 'lca_factor_kg_co2eq_per_unit': float,
                                 'lca_flow_key': str}
    """
    config = config or tea.DEFAULT_CONFIG
    route = route.upper()

    if route == 'IX':
        inventories = _all_ix_inventories(Q, config)
    elif route == 'SX':
        inventories = _all_sx_inventories(Q, config)
    else:
        raise ValueError(f"route must be 'IX' or 'SX', got {route!r}")

    ga_per_m3 = _ga_output_per_m3(route, config)
    breakdown = {}

    for inv in inventories:
        for flow_name, entry in inv.items():
            if not isinstance(entry, dict):
                continue
            gwp_factor = entry.get('lca_factor_kg_co2eq_per_unit') or 0.0
            consumption = entry.get('consumption_per_m3') or 0.0
            gwp_per_m3 = consumption * gwp_factor

            key = flow_name
            if key in breakdown:
                breakdown[key]['gwp_per_m3'] += gwp_per_m3
                breakdown[key]['gwp_per_kg_ga'] += gwp_per_m3 / ga_per_m3
                breakdown[key]['consumption_per_m3'] += consumption
            else:
                breakdown[key] = {
                    'gwp_per_m3': gwp_per_m3,
                    'gwp_per_kg_ga': gwp_per_m3 / ga_per_m3 if ga_per_m3 > 0 else 0.0,
                    'consumption_per_m3': consumption,
                    'lca_factor_kg_co2eq_per_unit': gwp_factor,
                    'lca_flow_key': entry.get('lca_flow_key', ''),
                }

    return breakdown


if __name__ == '__main__':
    # Quick verification against thesis abstract values
    print("Full-LCA carbon burden (kg CO₂-eq / kg Ga)\n" + "=" * 48)
    print(f"{'Q':>6}  {'Route':>5}  {'Computed':>10}  {'Abstract':>10}  {'Δ':>7}")
    print("-" * 48)
    reference = {
        (10,  'IX'): 236.8,
        (10,  'SX'): 418.5,
        (100, 'IX'):  67.9,
        (100, 'SX'): 183.4,
    }
    for (Q, route), abstract_val in reference.items():
        computed = calc_carbon_burden_full_lca(Q=Q, route=route)
        delta = computed - abstract_val
        ok = '✓' if abs(delta) < 10 else '✗'
        print(f"{Q:>6}  {route:>5}  {computed:>10.1f}  {abstract_val:>10.1f}  {delta:>+7.1f}  {ok}")

    print("\nEnergy-only carbon burden (kg CO₂-eq / kg Ga)\n" + "=" * 48)
    for Q in [10, 30, 100]:
        for route in ['IX', 'SX']:
            val = calc_carbon_burden_energy_only(Q=Q, route=route)
            print(f"  Q={Q:>4}  {route}  {val:.1f}")
