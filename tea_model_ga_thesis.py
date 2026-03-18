"""
tea_model_ga_thesis.py
======================
Techno-Economic Analysis (TEA) model for the recovery of 4N-grade gallium
from GaAs semiconductor manufacturing wastewater.

This file accompanies the master's thesis:

  Kozerke, K. (2025). Techno-Economic and Environmental Assessment of Gallium
  Recovery from GaAs Semiconductor Manufacturing Wastewater.
  Master's Thesis, University of Cambridge / WZL RWTH Aachen University.

The thesis extends the conference paper (ESCAPE 36, Sheffield 2026) with
full CO₂-accounting integration, LCA carbon-burden analysis, and a broader
sensitivity and uncertainty analysis.  For the leaner conference-paper
version see the companion repository ``escape36-gallium-tea``.

Chapter references
------------------
  Chapter 3 (Methods):  TEA framework, StepSpec structure, and cost blocks
                         implemented in this module.
  Chapter 4 (Results):  LCO-Ga values, cost breakdowns, and Q* analysis
                         produced by calc_lco_ga(), calc_cost_breakdown(),
                         and calc_break_even_price().
  Chapter 5 (Discussion): Sensitivity (Figures 5.8–5.9), viability map
                         (Figure 5.10), and scenario analysis (Table 5.7).
  Appendix A:           Full inline source citations embedded in each StepSpec.

Process trains
--------------
Two competing recovery routes share a common upstream section
(Filtration → RO Split → pH Adjust) and diverge at the selective separation step:

  Route A — Ion Exchange (IX):
    Raw feed → Filtration → RO Split → pH Adjust → IX Separation
             → Precipitation → Selective Leaching → Electrowinning

  Route B — Solvent Extraction (SX):
    Raw feed → Filtration → RO Split → pH Adjust → SX Separation
             → Precipitation → Selective Leaching → Electrowinning

System boundary and functional unit
-------------------------------------
The system boundary is gate-to-gate, covering the gallium recovery process
train from the raw GaAs wastewater inlet to the refined 4N gallium product.
Upstream semiconductor manufacturing and downstream product use are excluded.

  Functional unit:   1 kg of 4N gallium produced
  Feed basis:        GaAs process wastewater, 34.6 mg/L Ga, pH 3.8
                     (Jain 2019; full characterisation in FEED_BASELINE_TEMPLATE)
  Throughput range:  Q = 10 – 100 m³/d  (Q_FEED_RANGE)

Cost framework
--------------
The objective function is the levelised cost of gallium (LCO-Ga), defined as
total annual cost divided by annual gallium production:

  LCO-Ga = TC / m_Ga,ann                                [EUR/kg Ga]

  m_Ga,ann = Q × c_Ga × TR × availability × 365 days    [kg/yr]

Total annual cost is the sum of five blocks:

  TC = AF × CapEx  +  OpEx  +  REP  +  LC  +  CO₂Tax   [EUR/yr]

  CapEx     Installed capital = Σ_s (f_Lang,s × Σ C_equip,s)
  OpEx      Electricity + chemicals + O&M (fixed fraction of CapEx)
  REP       Annualised replacement of finite-lifetime equipment and consumables
            = Σ_i (C_equip,i / τ_i)
  LC        Labour = C_FTE × f_FTE
  CO₂Tax    Carbon cost = TaxRate_CO₂ × CO₂Rate × m_Ga,ann

  AF (CRF)  Capital recovery factor = r(1+r)^n / ((1+r)^n − 1)

Step-specific Lang factors (Peters & Timmerhaus 2004, Table 6-21) convert
purchased equipment costs to installed total capital.  Cost estimates carry
AACE Class IV accuracy (±40 %).

CO₂ accounting — two intentionally separate scopes
---------------------------------------------------
1. CO₂ tax embedded in LCO-Ga  (TEAConfig.co2_tax_mode = "energy_only", default):
   Prices only the direct electricity emissions of each process step at the
   EU ETS shadow price (60 EUR/t CO₂, TEAConfig.co2_tax_per_ton).  Grid emission
   factor: 0.363 kg CO₂-eq/kWh (UBA 2024 German grid average).
   The EU ETS applies to electricity generators, not to downstream chemical
   consumers; energy_only therefore avoids double-counting and is consistent
   with the gate-to-gate system boundary.  This is the mode used for all
   LCO-Ga values reported in the thesis (Figures 5.3–5.6, Tables 5.7–5.9).

2. Standalone carbon-burden metric  (co2_tax_mode = "full_lca"):
   Accounts for electricity AND all reagent/solvent/resin GWP using ReCiPe 2016
   Midpoint (H) characterisation factors from ecoinvent 3.5 cutoff system
   (see lca/lca_recipe_factors.py).  Used exclusively for the kg CO₂-eq/kg Ga
   metric cited in the abstract and Figure 5.7 (carbon-intensity comparison).
   This mode does NOT affect any LCO-Ga cost output.

   To reproduce the abstract carbon-burden values:
     >>> from utils.carbon_burden import calc_carbon_burden_full_lca
     >>> calc_carbon_burden_full_lca(Q=10, route='IX')   # ≈ 232 kg CO₂-eq/kg Ga
     >>> calc_carbon_burden_full_lca(Q=10, route='SX')   # ≈ 415 kg CO₂-eq/kg Ga

Model structure
---------------
Each process step is described by a StepSpec dataclass instance containing:
  - constants      : frozen process/cost parameters with inline source citations
  - sources        : literature and supplier references for the step
  - equipment      : equipment items with unit costs and replacement lifetimes
  - cost_basis     : narrative of the CapEx/OpEx calculation for the step
  - recovery_basis : recovery factor value and its literature basis
  - stream_basis   : inlet stream description (species, pH, flow fraction)

Shared upstream steps (Filtration, RO Split, pH Adjust) are route-independent.
Route-specific steps are dispatched via route='IX' or route='SX' in all
public calc_* functions.

Notation (thesis ↔ code mapping)
---------------------------------
  c_Ga     feed gallium concentration     → FEED_BASELINE_TEMPLATE['species_mg_L']['Ga']
  Q        wastewater throughput [m³/d]   → Q_feed in all calc_* functions
  Q*       break-even throughput          → calc_break_even_price()
  LCO-Ga   levelised cost of gallium      → calc_lco_ga()
  TC       total annual cost              → calc_total_costs()
  CapEx    capital expenditure            → calc_*_capex_per_m3()
  OpEx     operating expenditure          → calc_*_opex_per_m3()
  REP      annualised replacement cost    → calc_*_rep_per_m3()
  AF/CRF   capital recovery factor        → TEAConfig.CRF
  TR       overall route recovery         → TEAConfig.recovery_rate_total_ix / _sx

Python ≥ 3.9.  No external dependencies beyond NumPy.
"""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping
from types import MappingProxyType

import numpy as np


# Module-level constants extracted from removed Assumptions dataclasses
Q_FEED_RANGE = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
"""Feed flow-rate range for sensitivity analysis [m³/d]."""

GA_MARKET_PRICE_RANGE = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
"""Gallium market price range for sensitivity analysis [EUR/kg]."""

RO_CONCENTRATE_FRACTION: float = 0.20
"""Fraction of the feed flow that becomes RO concentrate (20 % of Q_feed).
Source: TEAConfig.recoveries['ro_split_concentrate'] = 0.20 (Jain 2019 / Ambiado 2017).
Used throughout to compute Qc = RO_CONCENTRATE_FRACTION × Q_feed."""


@dataclass(frozen=True)
class EquipmentItem:
    """Equipment-cost record used by the inline step specifications."""

    name: str
    cost_eur: float
    lifetime_years: float
    source: str
    note: str = ""


@dataclass(frozen=True)
class StepSpec:
    """Structured publication-facing description of one process step."""

    step_id: str
    title: str
    route: str
    purpose: str
    stream_basis: str
    mass_balance_basis: str
    recovery_basis: str
    sizing_basis: str
    cost_basis: str
    assumptions: tuple[str, ...]
    sources: tuple[str, ...]
    equipment: tuple[EquipmentItem, ...]
    constants: Mapping[str, float]
    tea_scope: str = ""
    lca_scope: str = ""
    waste_scope: str = ""
    lca_factor_keys: tuple[str, ...] = ()
    """Source annotations for individual constants: {constant_name: source_or_explanation}.
    Parallel to ``constants``; keys need not cover every entry. Purely informational."""

FEED_BASELINE_TEMPLATE: Mapping[str, Any] = {
    'T_C': 25.0,
    'pH': 3.8,
    'species_mg_L': {
        'Ga': 34.6,
        'As': 35.3,
        'P': 5.28,
        'Na': 578.5,
        'Cl': 243.0,
        'SO4': 221.5,
        'Ca': 1.56,
        'Mg': 0.352,
        'Fe': 0.022,
        'Al': 0.0605,
        'Ni': 0.0144,
    },
    'TSS_mg_L': 30.0,
    'TOC_mg_L': 5.0,
    'TDS_mg_L': 1500.0,
    'source': 'Jain 2019 (Ga/As/P, Ca/Mg/Fe/Al/Ni); Li 2015 (Na/Cl/SO4); Lu 2010 (TSS); literature reviews (TOC)'
}

VALID_CO2_TAX_MODES: tuple[str, ...] = ("none", "energy_only", "full_lca")
VALID_CO2_ENERGY_PATHS: tuple[str, ...] = ("legacy_frozen", "route_consistent")


@dataclass(frozen=True)
class TEAConfig:
    """
    Immutable configuration object for the TEA model.

    All parameters are frozen at instantiation. Derived fields (operating_days,
    CRF, sx_makeup_rate_daily, recovery_rate_total_ix, recovery_rate_total_sx)
    are computed in __post_init__ and stored immutably.

    To create a variant configuration, use the replace() method which returns
    a new TEAConfig instance with specified fields changed and all derivatives
    recomputed.
    """

    # ========================================================================
    # OPERATING PARAMETERS
    # ========================================================================
    days_per_year: float = 365.0
    plant_availability: float = 0.95

    # ========================================================================
    # FINANCIAL PARAMETERS
    # ========================================================================
    r: float = 0.08
    """Discount rate for capital recovery factor calculation."""
    n: int = 20
    """Plant lifetime in years."""
    O_M_rate: float = 0.02
    """Maintenance and operations rate as a fraction of total capital investment."""
    labour_cost_full_time: float = 69164.0
    """Fully loaded annual cost of one full-time chemical operator (EUR/year)."""
    labour_FTE: float = 0.25
    """Full-time equivalent assigned to this plant."""

    # ========================================================================
    # MARKET ASSUMPTIONS
    # ========================================================================
    electricity_price: float = 0.1648
    """Electricity price in EUR/kWh."""
    ga_market_price_base: float = 423.0
    """Market price for gallium 4N in EUR/kg."""

    # ========================================================================
    # CARBON COSTING
    # ========================================================================
    co2_tax_per_ton: float = 60.0
    """CO2 tax per tonne of CO2 (EUR/t CO2)."""
    co2_tax_mode: str = "energy_only"
    """Carbon-cost scope: 'none', 'energy_only', or 'full_lca'."""
    co2_energy_path: str = "legacy_frozen"
    """Implementation path for energy_only mode: 'legacy_frozen' or 'route_consistent'."""
    energy_only_grid_emission_factor: float = 0.4
    """Electricity-only carbon-load factor (kg CO2eq/kWh)."""

    # ========================================================================
    # SX SOLVENT MANAGEMENT
    # ========================================================================
    sx_makeup_rate_annual: float = 0.15
    """Annual SX solvent make-up rate (fraction of organic inventory)."""

    # ========================================================================
    # SCENARIO MULTIPLIERS
    # ========================================================================
    sx_capex_multiplier: float = 1.0
    """Multiplier for SX-specific capital costs."""
    capex_multiplier: float = 1.0
    """Multiplier for general capital costs."""
    rep_multiplier: float = 1.0
    """Multiplier for replacement costs."""
    labour_multiplier: float = 1.0
    """Multiplier for labour costs."""
    opex_multiplier: float = 1.0
    """Multiplier for operating-expenditure costs."""

    # ========================================================================
    # GRID SETTINGS FOR BREAK-EVEN ANALYSIS
    # ========================================================================
    q_star_grid_start: int = 1
    """Starting throughput for break-even grid search (m³/d)."""
    q_star_grid_end: int = 100
    """Ending throughput for break-even grid search (m³/d)."""
    q_star_grid_step: int = 1
    """Step size for break-even grid search (m³/d)."""

    # ========================================================================
    # PROCESS RECOVERIES (as MappingProxyType for immutability)
    # ========================================================================
    recoveries: MappingProxyType = field(default_factory=lambda: MappingProxyType({
        "ro_split_permeate": 0.80,
        "ro_split_concentrate": 0.20,
        "ro_ga_to_concentrate": 0.95,
        "ro_tss_to_concentrate": 1.0,
        "ix_ga_to_eluat": 0.969,
        "sx_ga_to_loaded_organic": 0.773,
        "sx_ga_strip_to_aqueous": 0.975,
        "precip_ga_to_cake": 0.99,
        "leach_ga_to_leachate": 0.9781,
        "ew_ga_to_product": 0.902,
    }))

    # ========================================================================
    # DERIVED FIELDS (computed in __post_init__)
    # ========================================================================
    operating_days: float = field(init=False)
    """Effective operating days per year (days_per_year × plant_availability)."""

    CRF: float = field(init=False)
    """Capital recovery factor for annualized cost calculations."""

    sx_makeup_rate_daily: float = field(init=False)
    """Daily SX solvent make-up rate (annual rate / operating_days)."""

    sx_bleed_rate: float = field(init=False)
    """Alias for sx_makeup_rate_daily for backward compatibility."""

    recovery_rate_total_ix: float = field(init=False)
    """Overall recovery rate for the IX route (product of step recoveries)."""

    recovery_rate_total_sx: float = field(init=False)
    """Overall recovery rate for the SX route (product of step recoveries)."""

    def __post_init__(self) -> None:
        """Compute all derived fields. Called automatically after __init__."""
        # Compute operating_days
        operating_days = self.days_per_year * self.plant_availability
        object.__setattr__(self, 'operating_days', operating_days)

        # Compute CRF
        crf = self.r * (1 + self.r)**self.n / ((1 + self.r)**self.n - 1)
        object.__setattr__(self, 'CRF', crf)

        # Compute SX make-up rates
        sx_makeup_daily = self.sx_makeup_rate_annual / operating_days
        object.__setattr__(self, 'sx_makeup_rate_daily', sx_makeup_daily)
        object.__setattr__(self, 'sx_bleed_rate', sx_makeup_daily)

        # Compute overall route recoveries
        r = self.recoveries
        recovery_ix = (
            r['ro_ga_to_concentrate']
            * r['ix_ga_to_eluat']
            * r['precip_ga_to_cake']
            * r['leach_ga_to_leachate']
            * r['ew_ga_to_product']
        )
        recovery_sx = (
            r['ro_ga_to_concentrate']
            * r['sx_ga_to_loaded_organic']
            * r['sx_ga_strip_to_aqueous']
            * r['precip_ga_to_cake']
            * r['leach_ga_to_leachate']
            * r['ew_ga_to_product']
        )
        object.__setattr__(self, 'recovery_rate_total_ix', recovery_ix)
        object.__setattr__(self, 'recovery_rate_total_sx', recovery_sx)

    def replace(self, **kwargs) -> 'TEAConfig':
        """
        Create a new TEAConfig with specified fields replaced.

        All derived fields are automatically recomputed in the new instance.

        Args:
            **kwargs: Field names and new values to replace.

        Returns:
            A new TEAConfig instance with updated values and recomputed derivatives.
        """
        from dataclasses import replace as dataclass_replace
        return dataclass_replace(self, **kwargs)


# Default configuration instance
DEFAULT_CONFIG = TEAConfig()



# ============================================================================
# MODELLING CONVENTIONS AND UNITS
# ============================================================================
#
# - Code identifiers remain ASCII-safe and caller-stable.
# - Thesis/paper notation uses typographic forms such as cGa, Q, Q*, and LCOGa.
# - Concentrations are tracked in mg/L in code and correspond to mg·L⁻¹ in the
#   thesis and paper.
# - Throughput is tracked in m³/d in code and corresponds to m³·d⁻¹ in the
#   thesis and paper.
# - Annual production is tracked in kg/yr in code and corresponds to kg·a⁻¹ in
#   the thesis.
# - Frozen public cost-output keys remain `CapEx`, `OpEx`, `REP`, `Labour`,
#   `CO2_Tax`, and `Total`.
#
# Inventory-output conventions:
# - `consumption_per_m3` is the default feed-normalized inventory field.
# - SX material helpers may additionally expose route-internal normalizations
#   such as `consumption_per_m3_sx_input` where the scientific basis requires it.
# - Some helper outputs intentionally expose cost-proxy consumable records
#   instead of pure mass inventories for backward compatibility with legacy
#   reporting and export code.
# - Waste and side-stream helpers are kept for mass-balance closure and
#   comparative inventory work, but they remain outside the active baseline
#   TEA total-cost aggregation unless a dedicated disposal helper is called.
#
# Carbon-accounting conventions:
# - `co2_tax_mode='none'` keeps carbon costs out of scope (ESCAPE-style baseline).
# - `co2_tax_mode='energy_only'` prices only electricity-related carbon load.
# - `co2_tax_mode='full_lca'` prices the modeled LCA carbon load as a shadow
#   price, using the repository LCA bridge.
# - `co2_energy_path='legacy_frozen'` preserves the thesis-compatible SX
#   electricity path; `route_consistent` exposes the clean SX-specific variant
#   without changing the frozen default.

# ============================================================================
# HELPER FUNCTIONS FOR CAPITAL AND OPERATING COSTS
# ============================================================================


def _annualized_capex_per_m3(config: TEAConfig, direct_capex: float, q_feed: float, lang_factor: float) -> float:
    """
    Return annualized CapEx in EUR/m³ feed using the frozen TEA formula.

    Args:
        config: TEAConfig instance containing CRF and operating_days.
        direct_capex: Direct purchased-equipment cost in EUR.
        q_feed: Feed flow rate in m³/d.
        lang_factor: Lang factor for total capital investment scaling.

    Returns:
        Annualized capital cost in EUR/m³ feed.
    """
    total_capital_investment = direct_capex * lang_factor
    annualized_capex = total_capital_investment * config.CRF
    return annualized_capex / (q_feed * config.operating_days)


def _maintenance_opex_per_m3(config: TEAConfig, direct_capex: float, q_feed: float, lang_factor: float) -> float:
    """
    Return annual maintenance-and-operations cost in EUR/m³ feed.

    Args:
        config: TEAConfig instance containing O_M_rate and operating_days.
        direct_capex: Direct purchased-equipment cost in EUR.
        q_feed: Feed flow rate in m³/d.
        lang_factor: Lang factor for total capital investment scaling.

    Returns:
        Annual M&O cost in EUR/m³ feed.
    """
    total_capital_investment = direct_capex * lang_factor
    return (total_capital_investment * config.O_M_rate) / (q_feed * config.operating_days)


def _equipment_direct_capex(equipment: tuple[EquipmentItem, ...]) -> float:
    """
    Return direct purchased-equipment cost for a step-spec equipment tuple.

    Args:
        equipment: Tuple of EquipmentItem objects.

    Returns:
        Total direct equipment cost in EUR.
    """
    return sum(item.cost_eur for item in equipment)


def _equipment_annualized_replacement(equipment: tuple[EquipmentItem, ...]) -> float:
    """
    Return annualized replacement cost for a step-spec equipment tuple.

    Divides each equipment cost by its lifetime and sums the annual contributions.

    Args:
        equipment: Tuple of EquipmentItem objects.

    Returns:
        Total annualized replacement cost in EUR/year.
    """
    return sum(item.cost_eur / item.lifetime_years for item in equipment)


# ============================================================================
# LCA LAZY-LOAD HELPERS
# ============================================================================


def _lazy_load_lca_module(module_name: str, package_prefix: str = "TEA.lca."):
    """
    Load an LCA sub-module lazily without imposing a hard package-import path.

    Tries ``TEA/lca/<module_name>`` first (flat layout used when running from
    the repo root), then falls back to ``TEA.lca.<module_name>`` (package layout
    used when the project root is on ``sys.path``).

    Args:
        module_name: Bare module name, e.g. ``"lca_recipe_factors"``.
        package_prefix: Dotted prefix for the fallback import.

    Returns:
        The imported module object.

    Raises:
        ModuleNotFoundError: If the module cannot be found via either path.
    """
    import importlib
    import sys
    from pathlib import Path

    lca_dir = Path(__file__).resolve().parent / "lca"
    if str(lca_dir) not in sys.path:
        sys.path.insert(0, str(lca_dir))
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        return importlib.import_module(f"{package_prefix}{module_name}")


def _load_lca_recipe_module():
    """Return the ``lca_recipe_factors`` module (lazy, path-safe)."""
    return _lazy_load_lca_module("lca_recipe_factors")


def _resolve_lca_metadata(
    *,
    lca_flow_key: str | None = None,
    lca_recipe_key: str | None = None,
) -> tuple[str | None, float | None]:
    """
    Return the resolved LCA recipe key and GWP factor for one inventory flow.

    Args:
        lca_flow_key: Optional TEA flow identifier to look up in the recipe map.
        lca_recipe_key: Optional direct recipe key (overrides lca_flow_key lookup).

    Returns:
        A tuple of (recipe_key, gwp_factor_per_kg). Both are None if the flow
        is not characterized.
    """
    if lca_recipe_key is not None:
        # Key supplied directly — load recipe once here for the GWP lookup.
        recipe = _load_lca_recipe_module()
        return lca_recipe_key, float(recipe.get_gwp_per_kg(lca_recipe_key))
    if lca_flow_key is not None:
        # Key must be resolved via the flow-key map; single load covers both.
        recipe = _load_lca_recipe_module()
        recipe_key = recipe.TEA_FLOW_TO_RECIPE_KEY.get(lca_flow_key)
        if recipe_key is None:
            return None, None
        return recipe_key, float(recipe.get_gwp_per_kg(recipe_key))
    return None, None


def _build_inventory_entry(
    *,
    consumption_per_m3: float,
    unit: str,
    source: str,
    price_per_unit: float | None = None,
    lca_flow_key: str | None = None,
    lca_recipe_key: str | None = None,
    tea_cost_scope: str = "included_in_step_opex",
    lca_scope: str = "included_if_characterized",
) -> dict[str, Any]:
    """
    Return one inventory entry with consistent TEA/LCA traceability fields.

    Args:
        consumption_per_m3: Material consumption in units per m³ feed.
        unit: Unit of measurement (e.g., 'kg/m³ Feed').
        source: Documentation source or reference.
        price_per_unit: Optional cost per unit for TEA; if None, not included.
        lca_flow_key: Optional TEA flow identifier for LCA lookup.
        lca_recipe_key: Optional direct LCA recipe key override.
        tea_cost_scope: TEA cost scope descriptor.
        lca_scope: LCA scope descriptor.

    Returns:
        A dictionary with fields: consumption_per_m3, unit, source, price_per_unit
        (if provided), tea_cost_scope, lca_scope, lca_flow_key, lca_factor_key,
        lca_factor_kg_co2eq_per_unit.
    """
    recipe_key, gwp_factor = _resolve_lca_metadata(
        lca_flow_key=lca_flow_key,
        lca_recipe_key=lca_recipe_key,
    )
    entry: dict[str, Any] = {
        "consumption_per_m3": consumption_per_m3,
        "unit": unit,
        "source": source,
        "tea_cost_scope": tea_cost_scope,
        "lca_scope": lca_scope,
        "lca_flow_key": lca_flow_key,
        "lca_factor_key": recipe_key,
        "lca_factor_kg_co2eq_per_unit": gwp_factor,
    }
    if price_per_unit is not None:
        entry["price_per_unit"] = price_per_unit
    return entry


def _build_dual_normalized_inventory_entry(
    *,
    consumption_per_m3_feed: float,
    consumption_per_m3_route_input: float,
    unit_feed: str,
    unit_route_input: str,
    source: str,
    price_per_unit: float | None = None,
    lca_flow_key: str | None = None,
    lca_recipe_key: str | None = None,
    tea_cost_scope: str = "included_in_step_opex",
    lca_scope: str = "included_if_characterized",
) -> dict[str, Any]:
    """
    Return one dual-normalized inventory entry with feed and route-input bases.

    Some SX steps are more naturally scaled by the route input (aqueous SX phase)
    rather than total feed. This helper provides both normalizations.

    Args:
        consumption_per_m3_feed: Material consumption normalized to m³ feed.
        consumption_per_m3_route_input: Material consumption normalized to route input.
        unit_feed: Unit for feed normalization.
        unit_route_input: Unit for route-input normalization.
        source: Documentation source or reference.
        price_per_unit: Optional cost per unit for TEA.
        lca_flow_key: Optional TEA flow identifier for LCA lookup.
        lca_recipe_key: Optional direct LCA recipe key override.
        tea_cost_scope: TEA cost scope descriptor.
        lca_scope: LCA scope descriptor.

    Returns:
        A dictionary with dual-normalization fields and LCA metadata.
    """
    recipe_key, gwp_factor = _resolve_lca_metadata(
        lca_flow_key=lca_flow_key,
        lca_recipe_key=lca_recipe_key,
    )
    entry: dict[str, Any] = {
        "consumption_per_m3_feed": consumption_per_m3_feed,
        "consumption_per_m3_route_input": consumption_per_m3_route_input,
        "unit_feed": unit_feed,
        "unit_route_input": unit_route_input,
        "source": source,
        "tea_cost_scope": tea_cost_scope,
        "lca_scope": lca_scope,
        "lca_flow_key": lca_flow_key,
        "lca_factor_key": recipe_key,
        "lca_factor_kg_co2eq_per_unit": gwp_factor,
    }
    if price_per_unit is not None:
        entry["price_per_unit"] = price_per_unit
    return entry


def _build_energy_entry(
    *,
    consumption_per_m3: float,
    source: str,
    price_per_unit: float,
    tea_cost_scope: str = "included_in_step_opex",
    lca_scope: str = "included_in_energy_only_and_full_lca",
) -> dict[str, Any]:
    """
    Return one electricity entry with aligned carbon-traceability metadata.

    Args:
        consumption_per_m3: Electricity consumption in kWh per m³ feed.
        source: Documentation source or reference.
        price_per_unit: Electricity price in EUR/kWh.
        tea_cost_scope: TEA cost scope descriptor.
        lca_scope: LCA scope descriptor.

    Returns:
        A dictionary with electricity consumption, pricing, and LCA metadata.
    """
    recipe_key, gwp_factor = _resolve_lca_metadata(lca_recipe_key="electricity")
    return {
        "consumption_per_m3": consumption_per_m3,
        "unit": "kWh/m³ Feed",
        "price_per_unit": price_per_unit,
        "source": source,
        "tea_cost_scope": tea_cost_scope,
        "lca_scope": lca_scope,
        "lca_flow_key": "electricity",
        "lca_factor_key": recipe_key,
        "lca_factor_kg_co2eq_per_unit": gwp_factor,
    }


def _annotate_waste_stream(
    stream: Mapping[str, Any],
    *,
    tea_total_cost_scope: str = "excluded_from_active_total_costs",
    lca_scope: str = "retained_for_comparative_inventory_only",
    scope_note: str,
) -> dict[str, Any]:
    """
    Attach explicit TEA/LCA scope metadata to a waste-stream helper output.

    Args:
        stream: Dictionary representing a waste stream.
        tea_total_cost_scope: TEA costing descriptor.
        lca_scope: LCA inventory descriptor.
        scope_note: Detailed explanation of the scope decision.

    Returns:
        The input stream dict augmented with scope metadata.
    """
    annotated = dict(stream)
    annotated["tea_total_cost_scope"] = tea_total_cost_scope
    annotated["lca_scope"] = lca_scope
    annotated["scope_note"] = scope_note
    return annotated


# --- END OF PART 1: CONFIGURATION AND HELPERS ---


# ============================================================================
# SECTION 1: SHARED PRETREATMENT
# ============================================================================
# This section contains the process steps that are identical for both routes
# (IX and SX): filtration, RO split, and pH adjustment.
# ============================================================================

# ============================================================================
# PROCESS STEP 1: FILTRATION (PRETREATMENT)
# ============================================================================
FILTRATION_SPEC = StepSpec(
    step_id="S1",
    title="Filtration",
    route="shared",
    purpose="Pretreat the wastewater ahead of RO by removing suspended solids and protecting the membrane train from fouling.",
    stream_basis="Continuous treatment of the raw wastewater feed at pH 3.8 with baseline gallium and arsenic concentrations.",
    mass_balance_basis="95% TSS removal; dissolved species are treated as unchanged.",
    recovery_basis="No route-specific gallium recovery is applied in this step.",
    sizing_basis="Three-stage filtration train; cartridge-filter capacity of 72 m³·d⁻¹ per unit and n = ceil(Q / 72).",
    cost_basis="Annualized CapEx from direct equipment cost multiplied by the Lang factor; REP from equipment lifetimes; OpEx from pumping electricity, cartridges, and M&O.",
    assumptions=(
        "The Pedrollo CP150 material combination is assumed to be compatible with the acidic wastewater.",
        "TSS is low enough for a six-month cartridge lifetime.",
        "Pressure-loss effects are represented implicitly through the pumping-power scaling.",
    ),
    sources=(
        "Peters & Timmerhaus, Plant Design and Economics for Chemical Engineers",
        "Baker, Membrane Technology and Applications, 3rd ed.",
        "https://best4purewater.co.uk/shop/stainless_steel_filter_housings",
        "https://shop.end.de/en/ea300164-1-v-strainer-dn25-pn40-welded-c-acc-iso4200",
        "https://www.uk-water-filters.co.uk/products/pre-filter-high-flow",
        "https://pumpexpress.co.uk/product-category/pump-types/centrifugal-pumps/",
        "https://pumpexpress.co.uk/shop/fa-20-sx-filter-cartridge-only-5-micron/",
    ),
    equipment=(
        EquipmentItem("Housing (SS AISI 304)", 819.0, 10.0, "best4purewater"),
        EquipmentItem("Coarse screen", 82.0, 10.0, "end.de"),
        EquipmentItem("Self-backwash filter", 566.0, 10.0, "uk-water-filters"),
        EquipmentItem("Pump CP150", 450.0, 15.0, "pumpexpress"),
        EquipmentItem("Cartridge filter", 15.0, 0.5, "pumpexpress", "Consumable charged through OpEx"),
    ),
    constants={
        "filter_capacity_m3_per_d": 72.0,  # rated flow per unit; n = ceil(Q / 72)
        "lang_factor": 3.63,  # Peters & Timmerhaus (2004) Table 6-21, solid-liquid handling plant
        "housing_cost_eur": 819.0,  # best4purewater.co.uk SS AISI 304 multi-cartridge housing, accessed Mar 2025
        "coarse_screen_cost_eur": 82.0,  # end.de Y-strainer DN25 PN40, SKU EA300164-1, accessed Mar 2025
        "self_backwash_cost_eur": 566.0,  # uk-water-filters.co.uk pre-filter high-flow unit, accessed Mar 2025
        "pump_cost_eur": 450.0,  # pratoerboso.com Pedrollo CP150 centrifugal pump, accessed Mar 2025
        "pump_power_max_kw": 0.75,  # Pedrollo CP150 rated shaft power at maximum duty
        "cartridge_cost_eur": 15.0,  # pumpexpress.co.uk FA-20-SX 5-micron cartridge, accessed Mar 2025
        "cartridge_replacements_per_year": 2.0,  # six-month lifetime per manufacturer recommendation
        "tss_removal_fraction": 0.95,  # Baker (2012) Membrane Technology and Applications, typical MF/UF TSS removal
    },
    tea_scope="Included in TEA through CapEx, REP, pumping electricity, and cartridge consumables.",
    lca_scope="Electricity is included in `full_lca`; cartridge filters remain uncharacterized in the current repository LCA bridge.",
    waste_scope="No dedicated waste-stream helper is modeled for filtration in the active TEA or LCA interfaces.",
    lca_factor_keys=("electricity",),
)

# ============================================================================
# FILTRATION HELPERS
# ============================================================================

def _calc_filtration_n_filter(Q_feed):
    """Return the number of filtration units required for ``Q_feed``."""
    return ceil(Q_feed / FILTRATION_SPEC.constants["filter_capacity_m3_per_d"])


def _calc_filtration_direct_capex(Q_feed):
    """Return direct purchased-equipment cost for the filtration step."""
    N_filter = _calc_filtration_n_filter(Q_feed)
    housing_capex = N_filter * FILTRATION_SPEC.constants["housing_cost_eur"]
    coarse_screen_capex = FILTRATION_SPEC.constants["coarse_screen_cost_eur"]
    self_backwash_capex = FILTRATION_SPEC.constants["self_backwash_cost_eur"]
    pump_capex = FILTRATION_SPEC.constants["pump_cost_eur"]
    return housing_capex + coarse_screen_capex + self_backwash_capex + pump_capex


def calc_filtration_capex_per_m3(Q_feed, config=None):
    """Return annualized filtration CapEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    direct_capex = _calc_filtration_direct_capex(Q_feed)
    return _annualized_capex_per_m3(config, direct_capex, Q_feed, FILTRATION_SPEC.constants["lang_factor"])


def calc_filtration_rep_per_m3(Q_feed, config=None):
    """Return annualized filtration replacement cost in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    N_filter = _calc_filtration_n_filter(Q_feed)

    rep_housing = N_filter * FILTRATION_SPEC.constants["housing_cost_eur"] / 10
    rep_coarse_screen = FILTRATION_SPEC.constants["coarse_screen_cost_eur"] / 10
    rep_self_backwash = FILTRATION_SPEC.constants["self_backwash_cost_eur"] / 10
    rep_pump = FILTRATION_SPEC.constants["pump_cost_eur"] / 15

    total_rep = rep_housing + rep_coarse_screen + rep_self_backwash + rep_pump
    return total_rep / (Q_feed * config.operating_days)


def calc_filtration_opex_per_m3(Q_feed, config=None):
    """Return filtration OpEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    N_filter = _calc_filtration_n_filter(Q_feed)

    pumping_power = FILTRATION_SPEC.constants["pump_power_max_kw"] * max(0.1, min(1.0, Q_feed / 100))
    SEC = (pumping_power * 24) / Q_feed * config.plant_availability
    energy_cost = SEC * config.electricity_price

    cartridge_cost_annual = (
        N_filter
        * FILTRATION_SPEC.constants["cartridge_cost_eur"]
        * FILTRATION_SPEC.constants["cartridge_replacements_per_year"]
    )
    cartridge_cost_per_m3 = cartridge_cost_annual / (Q_feed * config.operating_days)

    direct_capex = _calc_filtration_direct_capex(Q_feed)
    M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, FILTRATION_SPEC.constants["lang_factor"])

    return energy_cost + cartridge_cost_per_m3 + M_O_per_m3


def calc_filtration_material_consumption(Q_feed, config=None):
    """Return filtration-material consumption separately from electricity."""
    config = config or DEFAULT_CONFIG
    N_filter = _calc_filtration_n_filter(Q_feed)
    return {
        'cartridge_filter': _build_inventory_entry(
            consumption_per_m3=(N_filter * 2) / (Q_feed * config.operating_days),
            unit='items/m³ Feed',
            price_per_unit=FILTRATION_SPEC.constants["cartridge_cost_eur"],
            source='https://pumpexpress.co.uk/shop/fa-20-sx-filter-cartridge-only-5-micron/',
            tea_cost_scope='included_in_step_opex',
            lca_scope='not_characterized_in_current_repository_lca',
        )
    }


def calc_filtration_energy_consumption(Q_feed, config=None):
    """Return filtration-electricity consumption separately from materials."""
    config = config or DEFAULT_CONFIG
    pumping_power = FILTRATION_SPEC.constants["pump_power_max_kw"] * max(0.1, min(1.0, Q_feed / 100))
    SEC = (pumping_power * 24) / Q_feed * config.plant_availability
    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Derived from the Pedrollo CP150 pumping power (0.75 kW maximum) with linear scaling over Q.',
        )
    }


# ============================================================================
# PROCESS STEP 2: RO SPLIT
# ============================================================================
RO_SPLIT_SPEC = StepSpec(
    step_id="S2",
    title="RO split",
    route="shared",
    purpose="Concentrate the gallium-bearing stream by two-stage reverse osmosis before IX or SX.",
    stream_basis="Feed wastewater is split into 80% permeate and 20% concentrate.",
    mass_balance_basis="95% of gallium reports to the concentrate; the permeate is treated as a disposal stream.",
    recovery_basis="RO gallium-to-concentrate factor = 0.95 from the frozen model baseline.",
    sizing_basis="Stage-1 area A1 = 1.042 × Q and stage-2 area A2 = 0.625 × Q with 37 m² per module.",
    cost_basis="Annualized CapEx from direct equipment cost multiplied by a Lang factor of 1.4; REP includes membrane replacement and equipment lifetimes.",
    assumptions=(
        "Two pressure stages are used at 20 bar and 23 bar, respectively.",
        "Pump efficiency is fixed at 0.7.",
        "Permeate disposal is represented as a costed waste stream rather than a valorized co-product.",
    ),
    sources=(
        "DOI: 10.2166/wst.2016.556",
        "DOI: 10.1016/0011-9164(96)00081-1",
        "https://www.sciencedirect.com/science/article/pii/S0263876219300899",
        "https://www.volza.com/p/membrane/export/export-from-denmark/hsn-code-8421/",
        "https://www.deswater.com/DWT_articles/vol_264_papers/264_2022_91.pdf",
    ),
    equipment=(
        EquipmentItem("Pressure vessel", 1500.0, 10.0, "SciDirect"),
        EquipmentItem("RO module", 1088.0, 5.0, "Volza"),
        EquipmentItem("RO pump", 4385.0, 15.0, "SciDirect"),
    ),
    constants={
        "lang_factor": 1.4,  # Peters & Timmerhaus (2004) Table 6-21, packaged RO unit (lower factor for skid-mounted module)
        "ga_to_concentrate": 0.95,  # model assumption: ≥95% rejection of Ga³⁺ by polyamide RO (Baker 2012, 3rd ed.)
        "permeate_fraction": 0.8,  # 80/20 volume split permeate/concentrate; Ambiado et al. (2017)
        "stage1_area_factor_m2_per_q": 1.042,  # Ambiado et al. (2017) doi:10.2166/wst.2016.556, two-stage RO sizing model
        "stage2_area_factor_m2_per_q": 0.625,  # Ambiado et al. (2017) doi:10.2166/wst.2016.556
        "module_area_m2": 37.0,  # standard spiral-wound RO module area per unit
        "pressure_vessel_cost_eur": 1500.0,  # CERD (2019) pressure vessel cost
        "module_cost_eur": 1088.0,  # Volza.com trade-data estimate
        "pump_cost_eur": 4385.0,  # high-pressure RO feed pump
        "pump_efficiency": 0.7,  # typical centrifugal pump efficiency for high-pressure duty
        "stage1_pressure_bar": 20.0,  # stage 1 operating pressure (bar)
        "stage2_pressure_bar": 23.0,  # stage 2 operating pressure (bar)
        "permeate_disposal_cost_eur_per_m3": 0.36,  # CAUTION: 1996 Desalination paper; CEPCI-escalated ≈0.75 EUR/m³ at 2025 prices
        "membrane_replacement_fraction_per_year": 0.10,  # industry rule-of-thumb; 5-year RO element life → 20%/yr, 10%/yr partial strategy
    },
    tea_scope="Included in TEA through CapEx, REP, and electricity; permeate disposal is exposed only through a separate helper and remains outside active route totals.",
    lca_scope="Electricity is included in `full_lca`; membrane replacement is not yet characterized as a separate repository LCA flow.",
    waste_scope="RO permeate/waste helpers are inventory-style only and are not included in `calc_total_costs`.",
    lca_factor_keys=("electricity",),
)

# ============================================================================
# RO-SPLIT HELPERS
# ============================================================================

def _calc_ro_split_n_modules(Q_feed):
    """Return the number of RO modules required for ``Q_feed``."""
    A1 = RO_SPLIT_SPEC.constants["stage1_area_factor_m2_per_q"] * Q_feed
    A2 = RO_SPLIT_SPEC.constants["stage2_area_factor_m2_per_q"] * Q_feed
    return ceil(A1 / RO_SPLIT_SPEC.constants["module_area_m2"]) + ceil(A2 / RO_SPLIT_SPEC.constants["module_area_m2"])


def _calc_ro_split_direct_capex(Q_feed):
    """Return direct purchased-equipment cost for the RO split."""
    N_modules = _calc_ro_split_n_modules(Q_feed)
    return (
        2 * RO_SPLIT_SPEC.constants["pressure_vessel_cost_eur"]
        + N_modules * RO_SPLIT_SPEC.constants["module_cost_eur"]
        + 2 * RO_SPLIT_SPEC.constants["pump_cost_eur"]
    )


def _calc_ro_split_pump_power(Q_feed):
    """Return the RO pump power for stages 1 and 2 in kW."""
    Q_feed_h = Q_feed / 24
    Qc_stage1_h = 0.5 * Q_feed / 24
    efficiency = RO_SPLIT_SPEC.constants["pump_efficiency"]
    pump_power_stage1 = min((Q_feed_h * RO_SPLIT_SPEC.constants["stage1_pressure_bar"] * 10**5 * 1000) / (efficiency * 3600 * 1000), 3.0)
    pump_power_stage2 = min((Qc_stage1_h * RO_SPLIT_SPEC.constants["stage2_pressure_bar"] * 10**5 * 1000) / (efficiency * 3600 * 1000), 3.0)
    return pump_power_stage1, pump_power_stage2


def calc_ro_split_capex_per_m3(Q_feed, config=None):
    """Return annualized RO-split CapEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    direct_capex = _calc_ro_split_direct_capex(Q_feed)
    return _annualized_capex_per_m3(config, direct_capex, Q_feed, RO_SPLIT_SPEC.constants["lang_factor"])


def calc_ro_split_rep_per_m3(Q_feed, config=None):
    """Return annualized RO-split replacement cost in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    N_modules = _calc_ro_split_n_modules(Q_feed)
    rep_total = (
        2 * RO_SPLIT_SPEC.constants["pressure_vessel_cost_eur"] / 10
        + N_modules * RO_SPLIT_SPEC.constants["module_cost_eur"] / 5
        + 2 * RO_SPLIT_SPEC.constants["pump_cost_eur"] / 15
        + N_modules * RO_SPLIT_SPEC.constants["module_cost_eur"] * RO_SPLIT_SPEC.constants["membrane_replacement_fraction_per_year"]
    )
    return rep_total / (Q_feed * config.operating_days)


def calc_ro_split_waste_disposal_cost_per_m3(Q_feed, config=None):
    """Return permeate-disposal cost from the RO split in ``EUR/m³ feed``.

    Exposes disposal cost for sensitivity analysis and waste-stream inventory only.
    This function is **not** included in ``calc_total_costs()``; see
    ``_annotate_waste_stream`` for the ``excluded_from_active_total_costs`` flag.
    """
    config = config or DEFAULT_CONFIG
    Q_permeat = RO_SPLIT_SPEC.constants["permeate_fraction"] * Q_feed
    disposal_cost_daily = Q_permeat * RO_SPLIT_SPEC.constants["permeate_disposal_cost_eur_per_m3"]
    return disposal_cost_daily / (Q_feed * config.plant_availability)


def calc_ro_split_opex_per_m3(Q_feed, config=None):
    """Return RO-split OpEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    pump_power_stage1, pump_power_stage2 = _calc_ro_split_pump_power(Q_feed)
    SEC = (pump_power_stage1 + pump_power_stage2) * 24 / Q_feed * config.plant_availability
    energy_cost = SEC * config.electricity_price

    direct_capex = _calc_ro_split_direct_capex(Q_feed)
    M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, RO_SPLIT_SPEC.constants["lang_factor"])

    return energy_cost + M_O_per_m3


def calc_ro_split_material_consumption(Q_feed, config=None):
    """Return RO-split material consumption separately from electricity.

    The frozen public model does not yet expose membrane replacement as a
    separate material inventory record, so this helper intentionally remains
    empty until the source audit decides whether that expansion is warranted.
    """
    config = config or DEFAULT_CONFIG
    return {}


def calc_ro_split_energy_consumption(Q_feed, config=None):
    """Return RO-split electricity consumption separately from materials."""
    config = config or DEFAULT_CONFIG
    pump_power_stage1, pump_power_stage2 = _calc_ro_split_pump_power(Q_feed)
    SEC = (pump_power_stage1 + pump_power_stage2) * 24 / Q_feed * config.plant_availability

    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Derived from stage pressures of 20 and 23 bar with a fixed pump efficiency of 0.7.',
        )
    }


# ============================================================================
# PROCESS STEP 3: pH ADJUSTMENT (AFTER RO SPLIT, BEFORE IX/SX)
# ============================================================================
PH_ADJUST_SPEC = StepSpec(
    step_id="S3",
    title="pH adjustment",
    route="shared",
    purpose="Shift the RO concentrate from pH 3.8 to pH 2.0 to prepare the feed for IX or SX.",
    stream_basis="RO concentrate with Qc = 0.2 × Q and baseline pH 3.8.",
    mass_balance_basis="No intentional gallium partitioning; HCl addition changes acidity while keeping the same process stream basis.",
    recovery_basis="No route-specific recovery is applied in this conditioning step.",
    sizing_basis="Fixed-size dosing skid with dosing pump, static mixer, day tank, and pH electrode.",
    cost_basis="Annualized CapEx from fixed direct equipment cost multiplied by a Lang factor of 4.74; OpEx from HCl, electricity, and M&O.",
    assumptions=(
        "The acid dose is fixed at 1.1 kg HCl (32 wt%) per m³ of concentrate.",
        "Dosing-pump electricity demand is fixed at 1.2 kWh per day.",
        "The step is modelled as a continuous conditioning stage before separation.",
    ),
    sources=(
        "https://tfpumps.com/product/prominent-gmxa0708pvt20000uec0300en-7-6-l-h-7-bar/",
        "https://www.raptorsupplies.co.uk/pd/pulsafeeder/stm100-pvc",
        "https://enduramaxx.co.uk/enduramaxx/200-litre-open-bunded-chemical-tank/",
        "https://trafalgarscientific.co.uk/ph-electrode-for-use-with-wastewater-each/",
        "https://www.chemanalyst.com/Pricing-data/hydrochloric-acid-61",
    ),
    equipment=(
        EquipmentItem("Dosing pump", 1596.0, 7.0, "tfpumps"),
        EquipmentItem("Static mixer", 540.0, 20.0, "raptorsupplies"),
        EquipmentItem("Day tank", 250.0, 10.0, "enduramaxx"),
        EquipmentItem("pH electrode", 380.0, 2.0, "trafalgarscientific"),
    ),
    constants={
        "lang_factor": 4.74,  # Peters & Timmerhaus (2004) Table 6-21, fluid-processing plant with chemical dosing
        "hcl_consumption_kg_per_m3_concentrate": 1.1,  # first-principles: ΔpH 3.8→2.0, Δ[H⁺]≈9.84 mmol/L → 358.7 g HCl/0.32 = 1.12 kg/m³ 32 wt%
        "hcl_price_eur_per_kg": 0.15,  # EU industrial bulk price, ChemAnalyst 2024 ~130–175 USD/MT ≈ 0.12–0.16 EUR/kg
        "electricity_kwh_per_day": 1.2,  # ProMinent GMXA0708 nameplate ~50 W × 24 h/d = 1.2 kWh/d
        "dosing_pump_cost_eur": 1596.0,  # tfpumps.com ProMinent GMXA0708PVT20000UEC0300EN (7.6 L/h, 7 bar), accessed Mar 2025
        "static_mixer_cost_eur": 540.0,  # raptorsupplies.co.uk Pulsafeeder STM100-PVC in-line static mixer, accessed Mar 2025
        "day_tank_cost_eur": 250.0,  # enduramaxx.co.uk 200-L open-bunded chemical day tank (OTB20011), accessed Mar 2025
        "ph_electrode_cost_eur": 380.0,  # trafalgarscientific.co.uk wastewater pH electrode (gel-filled, Ti body), accessed Mar 2025
    },
    tea_scope="Included in TEA through HCl consumption, electricity, CapEx, and REP.",
    lca_scope="Included in `full_lca` through HCl and electricity.",
    waste_scope="No standalone waste-stream helper is modeled for this conditioning step.",
    lca_factor_keys=("hcl_32wt", "electricity"),
)

# ============================================================================
# PH-ADJUSTMENT HELPERS
# ============================================================================

def _calc_ph_adjust_direct_capex():
    """Return direct purchased-equipment cost for the pH-adjustment step."""
    return (
        PH_ADJUST_SPEC.constants["dosing_pump_cost_eur"]
        + PH_ADJUST_SPEC.constants["static_mixer_cost_eur"]
        + PH_ADJUST_SPEC.constants["day_tank_cost_eur"]
        + PH_ADJUST_SPEC.constants["ph_electrode_cost_eur"]
    )


def calc_ph_adjust_capex_per_m3(Q_feed, config=None):
    """Return annualized pH-adjustment CapEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    direct_capex = _calc_ph_adjust_direct_capex()
    return _annualized_capex_per_m3(config, direct_capex, Q_feed, PH_ADJUST_SPEC.constants["lang_factor"])


def calc_ph_adjust_rep_per_m3(Q_feed, config=None):
    """Return annualized pH-adjustment replacement cost in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    rep = (
        PH_ADJUST_SPEC.constants["dosing_pump_cost_eur"] / 7
        + PH_ADJUST_SPEC.constants["static_mixer_cost_eur"] / 20
        + PH_ADJUST_SPEC.constants["day_tank_cost_eur"] / 10
        + PH_ADJUST_SPEC.constants["ph_electrode_cost_eur"] / 2
    )
    return rep / (Q_feed * config.operating_days)


def calc_ph_adjust_opex_per_m3(Q_feed, config=None):
    """Return pH-adjustment OpEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    acid_cost = (
        Qc
        * PH_ADJUST_SPEC.constants["hcl_consumption_kg_per_m3_concentrate"]
        * PH_ADJUST_SPEC.constants["hcl_price_eur_per_kg"]
    ) / (Q_feed * config.plant_availability)
    energy_cost = (
        PH_ADJUST_SPEC.constants["electricity_kwh_per_day"] * config.electricity_price
    ) / (Q_feed * config.plant_availability)

    direct_capex = _calc_ph_adjust_direct_capex()
    M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, PH_ADJUST_SPEC.constants["lang_factor"])

    return acid_cost + energy_cost + M_O_per_m3


def calc_ph_adjust_material_consumption(Q_feed, config=None):
    """Return HCl consumption for the pH-adjustment step."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    return {
        'hcl_32wt': _build_inventory_entry(
            consumption_per_m3=(
                Qc * PH_ADJUST_SPEC.constants["hcl_consumption_kg_per_m3_concentrate"]
            ) / (Q_feed * config.plant_availability),
            unit='kg/m³ Feed',
            price_per_unit=PH_ADJUST_SPEC.constants["hcl_price_eur_per_kg"],
            source='Calculated from Δ[H+] = 10^(-2.0) - 10^(-3.8) and the frozen conversion to 1.1 kg HCl (32 wt%) per m³ concentrate.',
            lca_flow_key='hcl_32wt',
        )
    }


def calc_ph_adjust_energy_consumption(Q_feed, config=None):
    """Return electricity consumption for the pH-adjustment step."""
    config = config or DEFAULT_CONFIG
    SEC = PH_ADJUST_SPEC.constants["electricity_kwh_per_day"] / (Q_feed * config.plant_availability)
    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Membrane dosing pump, 50W × 24h = 1.2 kWh/d',
        )
    }


# ============================================================================
# SECTION 2: SEPARATION ROUTES (IX/SX)
# ============================================================================
# This section contains the two alternative separation routes:
# - Route 1: IX (ion exchange)
# - Route 2: SX (solvent extraction)
# ============================================================================

# ============================================================================
# PROCESS STEP 4: IX (ION EXCHANGE) - ROUTE 1
# ============================================================================
IX_SPEC = StepSpec(
    step_id="S4",
    title="Ion exchange",
    route="IX",
    purpose="Capture gallium from the pH-adjusted concentrate on selective IX resin and recover it by elution with 0.1 M sulfuric acid.",
    stream_basis="pH-adjusted RO concentrate with Qc = 0.2 × Q and pH 2.0.",
    mass_balance_basis="Gallium reports to the eluate with the frozen IX separation recovery; raffinate leaves as an arsenic-bearing waste stream.",
    recovery_basis=(
        "Frozen gallium-to-eluate recovery of 0.969 (96.9%). "
        "Source: Huang et al. (MDPI Processes 2019, doi:10.3390/pr7120921), 'A Process for the Recovery "
        "of Gallium from Gallium Arsenide Scrap' — DIAION CR11 column, 0.1 M H2SO4 elution, GaAs scrap feed. "
        "Verify specific recovery figure (paper reports 99.3% purity; confirm whether 96.9% refers to "
        "Ga loading + elution recovery or is derived from a mass balance in the thesis)."
    ),
    sizing_basis="Six-column batch train with V_res,col = Qc / 129.6, 45 BV adsorption, 4 BV elution, 1 BV rinse.",
    cost_basis=(
        "TCI = direct purchased-equipment cost × Lang factor 4.74 (Peters & Timmerhaus, 2004, "
        "Table 6-21, 'fluid-processing plant'). "
        "The initial DIAION CR11 resin charge is included in the direct-cost sum and multiplied "
        "by the Lang factor. REP covers annualized equipment and resin replacement at their "
        "respective lifetimes. OpEx from sulfuric-acid elution, electricity, and M&O."
    ),
    assumptions=(
        "Six IX columns are used for the frozen baseline design.",
        "The column-cost curve remains piecewise linear between Qc = 2 and 20 m³·d⁻¹.",
        "The raffinate disposal cost is represented separately from IX OpEx.",
    ),
    sources=(
        # GaAs scrap, DIAION CR11, 0.1 M H2SO4 elution (Cheng et al. 2019):
        "https://www.mdpi.com/2227-9717/7/12/921",
        # Resin regeneration and cycle design:
        "https://doi.org/10.1021/acsestengg.0c00192",
        # Column sizing and recovery cross-check:
        "https://www.mdpi.com/2075-4701/13/6/1118",
        # IX pumping SEC reference (0.036 kWh/m³):
        "https://www.nature.com/articles/s41545-020-0054-x",
        # TECHNICAL REPORT — US DOE (OSTI 2476227): H₂SO₄ bulk price 0.13 EUR/kg used in 0.1 M eluent cost:
        "https://www.osti.gov/biblio/2476227",
        "https://www.lenntech.com/products/Mitsubishi/DIAION-CR11/DIAION-CR11/index.html",
        "https://www.thermofisher.com/order/catalog/product/046544.A1",
        "https://www.anchorpumps.com/march-may-te-6p-md-240v-magnetic-driven-pump",
        "https://www.tanks.ie/1000l-chemical-dosing-tank/p1411",
        "https://srt-mischer.de/category-7/?language=en",
    ),
    equipment=(
        EquipmentItem("IX column", 95.0, 15.0, "scaled column-cost curve", "Upper anchor 442 EUR at Qc = 20 m³/d"),
        EquipmentItem("DIAION CR11 resin", 100.0, 5.0, "lenntech / Thermo Fisher", "Per litre of resin"),
        EquipmentItem("Pump", 1000.0, 8.0, "anchorpumps"),
        EquipmentItem("Regeneration tank", 185.0, 15.0, "tanks.ie"),
        EquipmentItem("Eluate tank", 441.0, 15.0, "tanks.ie"),
        EquipmentItem("Buffer tank", 1500.0, 15.0, "srt-mischer"),
    ),
    constants={
        "lang_factor": 4.74,  # Peters & Timmerhaus (2004) Table 6-21, fluid-processing plant; includes resin in Lang base (see cost_basis)
        "column_cost_low_eur": 95.0,  # lower anchor Qc=2 m³/d; Cheng et al. (2019) doi:10.3390/pr7120921 + Huang et al. (2020) doi:10.1038/s41545-020-0054-x
        "column_cost_high_eur": 442.0,  # upper anchor Qc=20 m³/d; piecewise linear interpolation between anchors
        "column_cost_low_qc": 2.0,  # lower Qc anchor [m³/d] for column-cost curve
        "column_cost_high_qc": 20.0,  # upper Qc anchor [m³/d] for column-cost curve
        "number_of_columns": 6.0,  # six-column rotating batch design; Cheng et al. (2019) doi:10.3390/pr7120921
        "resin_volume_denominator": 129.6,  # BV sizing: 45 BV ads + 4 BV elution + 1 BV rinse = 50 BV; Cheng et al. (2019)
        "resin_cost_eur_per_l": 100.0,  # DIAION CR11 bulk industrial estimate; Lenntech/Thermo Fisher
        "pump_cost_eur": 1000.0,  # anchorpumps.com March-May TE-6P-MD (1/4 HP, PVDF/PP), listed ~£712 ex-VAT, accessed Mar 2025
        "regeneration_tank_cost_eur": 185.0,  # tanks.ie chemical dosing tank
        "eluate_tank_cost_eur": 441.0,  # tanks.ie chemical dosing tank
        "buffer_tank_cost_eur": 1500.0,  # srt-mischer.de mixing/buffer tank with agitator
        "q_elution_factor": 0.017778,  # 4 BV elution per cycle; Cheng et al. (2019) doi:10.3390/pr7120921
        "sec_kwh_per_m3_concentrate": 0.036,  # IX column pumping SEC; Huang et al. (2020) doi:10.1038/s41545-020-0054-x
        "h2so4_01m_price_eur_per_l": 0.0013,  # in-house 0.1 M H₂SO₄: 0.13 EUR/kg H₂SO₄ bulk (OSTI 2476227) × 0.0098 kg/L = 0.0013 EUR/L
        "raffinate_arsenic_mg_per_l": 164.18736,  # frozen from feed snapshot (Jain 2019) + IX mass balance
        "raffinate_disposal_cost_eur_per_m3": 1.10,  # industrial arsenic-bearing effluent disposal; thesis baseline — verify regional tariff
    },
    tea_scope="Included in TEA through sulfuric-acid elution, electricity, CapEx, REP, and annualized resin replacement; raffinate disposal remains separate from active route totals.",
    lca_scope="Included in `full_lca` through sulfuric acid, electricity, and annualized IX resin replacement.",
    waste_scope="The arsenic-bearing IX raffinate is exposed through helper outputs for traceability and comparative work, but not charged in `calc_total_costs`.",
    lca_factor_keys=("h2so4_01M", "ix_resin", "electricity"),
)

# ============================================================================
# IX HELPERS
# ============================================================================

def _calc_ix_column_cost(Qc):
    """Return the IX column cost for the concentrate flow ``Qc``."""
    if Qc <= IX_SPEC.constants["column_cost_low_qc"]:
        return IX_SPEC.constants["column_cost_low_eur"]
    elif Qc >= IX_SPEC.constants["column_cost_high_qc"]:
        return IX_SPEC.constants["column_cost_high_eur"]
    else:
        return IX_SPEC.constants["column_cost_low_eur"] + (
            (IX_SPEC.constants["column_cost_high_eur"] - IX_SPEC.constants["column_cost_low_eur"])
            * (Qc - IX_SPEC.constants["column_cost_low_qc"])
            / (IX_SPEC.constants["column_cost_high_qc"] - IX_SPEC.constants["column_cost_low_qc"])
        )


def _calc_ix_direct_capex(Q_feed):
    """Return direct purchased-equipment cost for the IX step.

    Includes columns, resin, pumps, and tanks; multiplied by Lang factor in
    ``calc_ix_capex_per_m3``.
    """
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    V_res_col_L = Qc * 1000 / IX_SPEC.constants["resin_volume_denominator"]
    c_col = _calc_ix_column_cost(Qc)
    return (
        IX_SPEC.constants["number_of_columns"] * c_col
        + IX_SPEC.constants["number_of_columns"] * V_res_col_L * IX_SPEC.constants["resin_cost_eur_per_l"]
        + 3 * IX_SPEC.constants["pump_cost_eur"]
        + IX_SPEC.constants["regeneration_tank_cost_eur"]
        + IX_SPEC.constants["eluate_tank_cost_eur"]
        + IX_SPEC.constants["buffer_tank_cost_eur"]
    )


def calc_ix_capex_per_m3(Q_feed, config=None):
    """Return annualized IX CapEx in ``EUR/m³ feed``.

    TCI = direct purchased-equipment cost (columns + resin + pumps + tanks)
    × Lang factor 4.74 (Peters & Timmerhaus, 2004, Table 6-21).
    """
    config = config or DEFAULT_CONFIG
    direct_capex = _calc_ix_direct_capex(Q_feed)
    return _annualized_capex_per_m3(config, direct_capex, Q_feed, IX_SPEC.constants["lang_factor"])


def calc_ix_rep_per_m3(Q_feed, config=None):
    """Return annualized IX replacement cost in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    V_res_col_L = Qc * 1000 / IX_SPEC.constants["resin_volume_denominator"]
    c_col = _calc_ix_column_cost(Qc)
    rep = (
        IX_SPEC.constants["number_of_columns"] * c_col / 15
        + IX_SPEC.constants["number_of_columns"] * V_res_col_L * IX_SPEC.constants["resin_cost_eur_per_l"] / 5
        + 3 * IX_SPEC.constants["pump_cost_eur"] / 8
        + (
            IX_SPEC.constants["regeneration_tank_cost_eur"]
            + IX_SPEC.constants["eluate_tank_cost_eur"]
            + IX_SPEC.constants["buffer_tank_cost_eur"]
        ) / 15
    )
    return rep / (Q_feed * config.operating_days)


def calc_ix_opex_per_m3(Q_feed, config=None):
    """Return IX OpEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_elution = IX_SPEC.constants["q_elution_factor"] * Q_feed
    chemical_cost = (
        IX_SPEC.constants["h2so4_01m_price_eur_per_l"] * 1000 * Q_elution
    ) / (Q_feed * config.plant_availability)
    energy_cost = IX_SPEC.constants["sec_kwh_per_m3_concentrate"] * Qc / Q_feed * config.electricity_price

    direct_capex = _calc_ix_direct_capex(Q_feed)
    M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, IX_SPEC.constants["lang_factor"])

    return chemical_cost + energy_cost + M_O_per_m3


def calc_ix_material_consumption(Q_feed, config=None):
    """Return IX material consumption separately from electricity."""
    config = config or DEFAULT_CONFIG
    Q_elution = IX_SPEC.constants["q_elution_factor"] * Q_feed

    return {
        'h2so4_01M': _build_inventory_entry(
            consumption_per_m3=Q_elution / (Q_feed * config.plant_availability),
            unit='m³/m³ Feed',
            price_per_unit=IX_SPEC.constants["h2so4_01m_price_eur_per_l"] * 1000,
            source='0.1 M H2SO4 elution liquor; frozen price conversion from 0.13 EUR/kg H2SO4 to 0.0013 EUR/L.',
            lca_flow_key='h2so4_01M',
        ),
        'ix_resin_replacement': _build_inventory_entry(
            consumption_per_m3=_calc_ix_resin_replacement_per_m3(Q_feed, config),
            unit='kg resin/m³ Feed',
            source='Annualized DIAION CR11 replacement from six columns, 5-year resin lifetime, and 0.72 kg/L shipping density.',
            lca_flow_key='ix_resin',
            tea_cost_scope='included_in_step_rep',
        ),
    }


def calc_ix_energy_consumption(Q_feed, config=None):
    """Return IX electricity consumption separately from materials."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    SEC = IX_SPEC.constants["sec_kwh_per_m3_concentrate"]

    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC * Qc / Q_feed * config.plant_availability,
            price_per_unit=config.electricity_price,
            source='https://doi.org/10.1021/acsestengg.0c00192',
        )
    }


def calc_ix_waste_stream(Q_feed, config=None):
    """Return the arsenic-bearing IX raffinate stream."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_waste = Qc
    arsenic_concentration = IX_SPEC.constants["raffinate_arsenic_mg_per_l"]
    arsenic_mass_per_day = (Q_waste * arsenic_concentration) / 1000

    return _annotate_waste_stream(
        {
            'volume_per_day': Q_waste,
            'volume_per_m3_feed': Q_waste / Q_feed,
            'arsenic_concentration': arsenic_concentration,
            'arsenic_mass_per_day': arsenic_mass_per_day,
            'arsenic_mass_per_m3_feed': arsenic_mass_per_day / Q_feed,
            'description': 'IX raffinate after adsorption at pH 2 with low gallium loading and 164.19 mg/L arsenic.',
            'source': 'Waste volume equals the IX inlet flow (Qc = 0.2 × Q); arsenic concentration follows the frozen baseline snapshot.',
        },
        scope_note='This raffinate is tracked for waste-inventory transparency and Luo-style comparison, but it is not added to active TEA total-cost outputs.',
    )


def calc_ix_waste_disposal_cost_per_m3(Q_feed, config=None):
    """Return IX raffinate-disposal cost in ``EUR/m³ feed``.

    Exposes disposal cost for sensitivity analysis and waste-stream inventory only.
    This function is **not** included in ``calc_total_costs()``; see
    ``_annotate_waste_stream`` for the ``excluded_from_active_total_costs`` flag.
    """
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_waste = Qc
    disposal_cost_daily = Q_waste * IX_SPEC.constants["raffinate_disposal_cost_eur_per_m3"]
    return disposal_cost_daily / (Q_feed * config.plant_availability)


def _calc_ix_resin_replacement_per_m3(q_feed, config):
    """Return annualized IX resin replacement in ``kg resin/m³ feed``."""
    q_concentrate = RO_CONCENTRATE_FRACTION * q_feed
    resin_volume_per_column_l = q_concentrate * 1000 / 129.6
    annual_resin_volume_l = 6 * resin_volume_per_column_l / 5
    annual_resin_mass_kg = annual_resin_volume_l * 0.72
    return annual_resin_mass_kg / (q_feed * config.operating_days)


# ============================================================================
# PROCESS STEP 8: SX (SOLVENT EXTRACTION) - ROUTE 2
# ============================================================================
SX_SPEC = StepSpec(
    step_id="S8",
    title="Solvent extraction",
    route="SX",
    purpose="Transfer gallium from the pH-adjusted aqueous concentrate into an organic phase and recover it into an acidic strip solution.",
    stream_basis="pH-adjusted RO concentrate with Qc = 0.2 × Q and pH 2.0.",
    mass_balance_basis="Gallium partitions first to the loaded organic and then to the strip liquor; raffinate leaves as an arsenic-bearing waste stream.",
    recovery_basis=(
        "Frozen extraction recovery of 0.773 (77.3%) followed by strip recovery of 0.975 (97.5%). "
        "Extraction recovery source: Ye et al. (MDPI Sustainability 2020, doi:10.3390/su12051765), "
        "'Recovery of Gallium from Simulated GaAs Waste Etching Solutions by Solvent Extraction' — "
        "single-step extraction efficiency of 77.4% using 0.5 M Cyanex 272/kerosene at pH 2, O:A = 0.1. "
        "Strip recovery 0.975 attributed to back-extraction conditions "
        "(Chen et al. 2020, doi:10.3390/su12051765; IAEA INIS technical report inis.iaea.org/records/c0q3n-1a978)."
    ),
    sizing_basis="Mixer-settlers sized from the aqueous throughput with O:A = 0.1 in extraction and O:A = 1 in stripping.",
    cost_basis="Annualized CapEx from direct equipment cost multiplied by a Lang factor of 4.8; REP from equipment lifetimes; OpEx from Cyanex, kerosene, HCl, electricity, and M&O.",
    assumptions=(
        "SX solvent make-up is modelled as annual replacement of a fraction of the circulating organic inventory.",
        "The frozen baseline uses a 15% per operating-year make-up rate for the circulating organic phase.",
        "The first-in organic inventory is derived from the total SX residence time of 19/60 h.",
    ),
    sources=(
        # Extraction recovery 77.4%, Cyanex 272 0.5 M, O:A = 0.1, pH 2 (Ye et al. 2020):
        "https://www.mdpi.com/2071-1050/12/5/1765",
        # SX scale-up methodology (Ge from coal fly ash):
        "https://www.mdpi.com/2075-163X/5/2/298",
        # 15% per-year organic make-up rate for Cyanex 272/kerosene:
        "https://inis.iaea.org/records/c0q3n-1a978",
        "https://www.echemi.com/produce/pr2503031015-cyanex-27283411-71-6.html",
        "https://businessanalytiq.com/procurementanalytics/index/kerosene-price-index/",
        "https://www.alibaba.com/product-detail/Bench-Scale-Mixer-Settler-Solvent-Extraction_1601029030998.html",
        "https://www.anchorpumps.com/march-may-te-6p-md-240v-magnetic-driven-pump",
        "https://www.911metallurgist.com/blog/sx-ew-capital-operating-cost/",
    ),
    equipment=(
        EquipmentItem("EX mixer-settler", 4288.0, 15.0, "Alibaba"),
        EquipmentItem("STRIP mixer-settler", 4288.0, 15.0, "Alibaba"),
        EquipmentItem("Pump", 1000.0, 8.0, "anchorpumps"),
    ),
    constants={
        "lang_factor": 4.8,  # Peters & Timmerhaus (2004) Table 6-21, solvent-handling/extraction plant
        "mixer_settler_capacity_l_per_h": 108.0,  # manufacturer-stated aqueous throughput (Alibaba listing); units: ceil(Q_aq / 108)
        "mixer_settler_cost_eur": 4288.0,  # Alibaba bench-scale mixer-settler
        "pump_cost_eur": 1000.0,  # anchorpumps.com March-May TE-6P-MD (PVDF/PP, compatible with Cyanex/kerosene), accessed Mar 2025
        "number_of_pumps": 3.0,  # three pumps: organic feed, aqueous feed, strip circuit
        "extraction_oa_ratio": 0.1,  # O:A=0.1 in extraction; Chen et al. (2020) doi:10.3390/su12051765, selective for Ga over As at pH 2
        "strip_q_factor": 0.02,  # strip flow = 0.02 × Q_feed; derived from O:A=1 stripping + O:A=0.1 extraction
        "cyanex_molarity_mol_per_l": 0.5,  # 0.5 M Cyanex 272 in kerosene; Chen et al. (2020) doi:10.3390/su12051765
        "cyanex_molar_mass_kg_per_mol": 0.29,  # MW Cyanex 272 (C16H35O2P) = 290.43 g/mol; IUPAC standard
        "cyanex_price_eur_per_kg": 10.0,  # echemi.com Chinese B2B quote, Mar 2025; CAUTION: Alibaba shows $50–200/kg — verify with Syensqo
        "kerosene_price_eur_per_kg": 0.9,  # businessanalytiq.com kerosene price index, Mar 2025 (~0.7–1.1 EUR/kg)
        "organic_density_kg_per_l": 0.8,  # approximate density of Cyanex 272/kerosene blend (aliphatic diluent)
        "hcl_price_eur_per_kg": 0.15,  # same as pH Adjust step; ChemAnalyst 2024 EU bulk ~0.12–0.16 EUR/kg
        "strip_hcl_kg_per_l": 0.0570,  # kg of 32 wt% HCl solution per L of 0.5 M strip: 18.23 g pure HCl / 0.32 = 56.97 g ≈ 0.0570 kg; Chen et al. (2020) doi:10.3390/su12051765
        "first_in_residence_time_h": 19.0 / 60.0,  # total circuit residence time 19 min; basis for first-fill organic inventory
        "mixer_power_kw": 0.02,  # mixer motor nameplate power per unit
        "pump_power_kw": 0.18,  # pump motor nameplate power (TE-6P-MD, 1/4 HP ≈ 0.19 kW)
        "raffinate_arsenic_mg_per_l": 158.25696,  # frozen SX raffinate As from mass-balance baseline
        "raffinate_disposal_cost_eur_per_m3": 1.10,  # same as IX route for cross-route consistency; thesis baseline
    },
    tea_scope="Included in TEA through Cyanex 272, kerosene, strip-acid, electricity, CapEx, REP, and the SX-specific solvent make-up assumption.",
    lca_scope="Included in `full_lca` through modeled organic make-up, strip acid, and electricity; excluded from `energy_only` except for electricity.",
    waste_scope="The arsenic-bearing SX raffinate remains outside active route totals and is retained as an explicit waste/inventory helper.",
    lca_factor_keys=("organic", "kerosene", "hcl_32wt", "electricity"),
)

# ============================================================================
# SX HELPERS
# ============================================================================

def _calc_sx_makeup_flows_per_day(q_feed, config):
    """Return daily Cyanex, kerosene, and HCl make-up loads for the SX route."""
    q_aqueous = RO_CONCENTRATE_FRACTION * q_feed
    q_organic = SX_SPEC.constants["extraction_oa_ratio"] * q_aqueous
    make_up_organic_flow = config.sx_makeup_rate_daily * q_organic

    cyanex_daily = (
        SX_SPEC.constants["cyanex_molarity_mol_per_l"]
        * SX_SPEC.constants["cyanex_molar_mass_kg_per_mol"]
        * make_up_organic_flow
        * 1000
    )
    total_organic_mass_daily = (
        make_up_organic_flow * 1000 * SX_SPEC.constants["organic_density_kg_per_l"]
    )
    kerosene_daily = total_organic_mass_daily - cyanex_daily

    q_strip = SX_SPEC.constants["strip_q_factor"] * q_feed
    hcl_daily = q_strip * SX_SPEC.constants["strip_hcl_kg_per_l"] * 1000
    return cyanex_daily, kerosene_daily, hcl_daily


def _calc_sx_n_mixer_settler(Q_feed):
    """Return the number of mixer-settlers required for the SX step."""
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_A_L_per_h = Qc * 1000 / 24
    return ceil(Q_A_L_per_h / SX_SPEC.constants["mixer_settler_capacity_l_per_h"])


def _calc_sx_first_in_organic_cost(Q_feed):
    """Return first-in organic inventory cost for the SX circuit."""
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_organic = SX_SPEC.constants["extraction_oa_ratio"] * Qc
    Q_O_L_per_h = Q_organic * 1000 / 24
    t_tot = SX_SPEC.constants["first_in_residence_time_h"]
    c_cyanex = SX_SPEC.constants["cyanex_molarity_mol_per_l"]
    M_cyanex = SX_SPEC.constants["cyanex_molar_mass_kg_per_mol"]
    cyanex_FF_kg = Q_O_L_per_h * t_tot * c_cyanex * M_cyanex
    organic_density = SX_SPEC.constants["organic_density_kg_per_l"]
    total_organic_mass_kg = Q_O_L_per_h * t_tot * organic_density
    kerosene_FF_kg = total_organic_mass_kg - cyanex_FF_kg
    return (
        cyanex_FF_kg * SX_SPEC.constants["cyanex_price_eur_per_kg"]
        + kerosene_FF_kg * SX_SPEC.constants["kerosene_price_eur_per_kg"]
    )


def _calc_sx_direct_capex(Q_feed):
    """Return direct purchased-equipment cost for the SX step."""
    N_mixer_settler = _calc_sx_n_mixer_settler(Q_feed)
    ex_mixer_cost = N_mixer_settler * SX_SPEC.constants["mixer_settler_cost_eur"]
    strip_mixer_cost = N_mixer_settler * SX_SPEC.constants["mixer_settler_cost_eur"]
    pump_cost = SX_SPEC.constants["number_of_pumps"] * SX_SPEC.constants["pump_cost_eur"]
    first_in_organic_cost = _calc_sx_first_in_organic_cost(Q_feed)
    return ex_mixer_cost + strip_mixer_cost + pump_cost + first_in_organic_cost


def calc_sx_capex_per_m3(Q_feed, config=None):
    """Return annualized SX CapEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    direct_capex = _calc_sx_direct_capex(Q_feed)
    return _annualized_capex_per_m3(
        config,
        direct_capex * config.sx_capex_multiplier,
        Q_feed,
        SX_SPEC.constants["lang_factor"],
    )


def calc_sx_rep_per_m3(Q_feed, config=None):
    """Return annualized SX replacement cost in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_A = Qc
    Q_A_L_per_h = Q_A * 1000 / 24
    mixer_settler_capacity = SX_SPEC.constants["mixer_settler_capacity_l_per_h"]
    N_mixer_settler = ceil(Q_A_L_per_h / mixer_settler_capacity)
    mixer_settler_price = SX_SPEC.constants["mixer_settler_cost_eur"]
    rep_ex_mixer = N_mixer_settler * mixer_settler_price / 15
    rep_strip_mixer = N_mixer_settler * mixer_settler_price / 15
    rep_pumps = SX_SPEC.constants["number_of_pumps"] * SX_SPEC.constants["pump_cost_eur"] / 8
    rep = rep_ex_mixer + rep_strip_mixer + rep_pumps
    annual_volume = Q_feed * config.operating_days
    return rep / annual_volume


def calc_sx_opex_per_m3(Q_feed, config=None):
    """Return SX OpEx in ``EUR/m³ feed``."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_A = Qc
    Q_A_L_per_h = Q_A * 1000 / 24
    N_mixer_settler = ceil(Q_A_L_per_h / SX_SPEC.constants["mixer_settler_capacity_l_per_h"])

    cyanex_makeup_daily, kerosene_makeup_daily, hcl_consumption_daily = _calc_sx_makeup_flows_per_day(Q_feed, config)
    cyanex_cost_daily = cyanex_makeup_daily * SX_SPEC.constants["cyanex_price_eur_per_kg"]
    kerosene_cost_daily = kerosene_makeup_daily * SX_SPEC.constants["kerosene_price_eur_per_kg"]
    hcl_cost_daily = hcl_consumption_daily * SX_SPEC.constants["hcl_price_eur_per_kg"]

    chemical_cost_daily = cyanex_cost_daily + kerosene_cost_daily + hcl_cost_daily
    chemical_cost = chemical_cost_daily / (Q_feed * config.plant_availability)

    mixer_power_per_unit = SX_SPEC.constants["mixer_power_kw"]
    N_mixer_total = 2 * N_mixer_settler
    mixer_power_total = N_mixer_total * mixer_power_per_unit
    pump_power_per_pump = SX_SPEC.constants["pump_power_kw"]
    pump_power_total = SX_SPEC.constants["number_of_pumps"] * pump_power_per_pump
    total_power = mixer_power_total + pump_power_total
    electricity_kwh_daily = total_power * 24  # kWh/d
    electricity_cost_daily = electricity_kwh_daily * config.electricity_price
    energy_cost = electricity_cost_daily / (Q_feed * config.plant_availability)
    direct_capex = _calc_sx_direct_capex(Q_feed)
    M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, SX_SPEC.constants["lang_factor"])

    return chemical_cost + energy_cost + M_O_per_m3


def calc_sx_material_consumption(Q_feed, config=None):
    """Return SX material consumption separately from electricity."""
    config = config or DEFAULT_CONFIG
    q_aqueous = RO_CONCENTRATE_FRACTION * Q_feed
    cyanex_consumption_daily, kerosene_consumption_daily, hcl_consumption_daily = _calc_sx_makeup_flows_per_day(Q_feed, config)

    return {
        'cyanex': _build_dual_normalized_inventory_entry(
            consumption_per_m3_feed=cyanex_consumption_daily / (Q_feed * config.plant_availability),
            consumption_per_m3_route_input=cyanex_consumption_daily / (q_aqueous * config.plant_availability),
            unit_feed='kg/m³ Feed',
            unit_route_input='kg/m³ SX-Input',
            price_per_unit=SX_SPEC.constants["cyanex_price_eur_per_kg"],
            source='Cyanex 272 make-up from the circulating organic inventory; the frozen baseline uses a 15% per operating-year make-up rate from the SX solvent-management assumption.',
            lca_flow_key='organic',
        ),
        'kerosene': _build_dual_normalized_inventory_entry(
            consumption_per_m3_feed=kerosene_consumption_daily / (Q_feed * config.plant_availability),
            consumption_per_m3_route_input=kerosene_consumption_daily / (q_aqueous * config.plant_availability),
            unit_feed='kg/m³ Feed',
            unit_route_input='kg/m³ SX-Input',
            price_per_unit=SX_SPEC.constants["kerosene_price_eur_per_kg"],
            source='Kerosene make-up paired with the frozen Cyanex 272 recipe and the same 15% per operating-year inventory replacement.',
            lca_flow_key='kerosene',
        ),
        'hcl': _build_inventory_entry(
            consumption_per_m3=hcl_consumption_daily / (Q_feed * config.plant_availability),
            unit='kg/m³ Feed',
            price_per_unit=SX_SPEC.constants["hcl_price_eur_per_kg"],
            source='0.5 M HCl strip liquor: 0.0570 kg 32 wt% HCl solution per litre of strip (18.23 g pure HCl / 0.32 wt-frac = 56.97 g); Chen et al. (2020) doi:10.3390/su12051765.',
            lca_flow_key='hcl_32wt',
        ),
    }


def calc_sx_energy_consumption(Q_feed, config=None):
    """Return SX electricity consumption separately from materials."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_A_L_per_h = Qc * 1000 / 24
    mixer_settler_capacity = SX_SPEC.constants["mixer_settler_capacity_l_per_h"]
    N_mixer_settler = ceil(Q_A_L_per_h / mixer_settler_capacity)

    mixer_power_per_unit = SX_SPEC.constants["mixer_power_kw"]
    N_mixer_total = 2 * N_mixer_settler
    mixer_power_total = N_mixer_total * mixer_power_per_unit

    pump_power_per_pump = SX_SPEC.constants["pump_power_kw"]
    pump_power_total = SX_SPEC.constants["number_of_pumps"] * pump_power_per_pump

    total_power = mixer_power_total + pump_power_total
    electricity_kwh_daily = total_power * 24
    SEC = electricity_kwh_daily / (Q_feed * config.plant_availability)

    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Mixer-settlers at 20 W per unit plus three pumps at 0.18 kW each, operated continuously.',
        )
    }


def calc_sx_waste_stream(Q_feed, config=None):
    """Return the arsenic-bearing SX raffinate stream."""
    config = config or DEFAULT_CONFIG
    Qc = RO_CONCENTRATE_FRACTION * Q_feed
    Q_raffinate = Qc
    arsenic_concentration = SX_SPEC.constants["raffinate_arsenic_mg_per_l"]
    arsenic_mass_per_day = (Q_raffinate * arsenic_concentration) / 1000

    return _annotate_waste_stream(
        {
            'volume_per_day': Q_raffinate,
            'volume_per_m3_feed': Q_raffinate / Q_feed,
            'arsenic_concentration': arsenic_concentration,
            'arsenic_mass_per_day': arsenic_mass_per_day,
            'arsenic_mass_per_m3_feed': arsenic_mass_per_day / Q_feed,
            'description': 'SX raffinate after extraction at pH 1.77 with low gallium loading and 158.26 mg/L arsenic.',
            'source': 'Waste volume equals the SX inlet flow (Qc = 0.2 × Q); arsenic concentration follows the frozen baseline snapshot.',
        },
        scope_note='This raffinate is tracked for waste-inventory transparency and Luo-style comparison, but it is not added to active TEA total-cost outputs.',
    )


def calc_sx_waste_disposal_cost_per_m3(Q_feed, config=None):
    """Return SX raffinate-disposal cost in ``EUR/m³ feed``.

    Exposes disposal cost for sensitivity analysis and waste-stream inventory only.
    This function is **not** included in ``calc_total_costs()``; see
    ``_annotate_waste_stream`` for the ``excluded_from_active_total_costs`` flag.
    """
    config = config or DEFAULT_CONFIG
    waste_stream = calc_sx_waste_stream(Q_feed, config)
    Q_raffinate = waste_stream['volume_per_day']
    disposal_cost_daily = Q_raffinate * SX_SPEC.constants["raffinate_disposal_cost_eur_per_m3"]
    return disposal_cost_daily / (Q_feed * config.plant_availability)


# ============================================================================
# SHARED EQUIPMENT FOR DOWNSTREAM PROCESSING
# ============================================================================

PRECIPITATION_EQUIPMENT: tuple[EquipmentItem, ...] = (
    EquipmentItem("Eluate tank", 97.0, 15.0, "https://enduramaxx.co.uk/enduramaxx/200-litre-chemical-dosing-tank/", "Fixed-size precipitation skid vessel."),
    EquipmentItem("NaOH tank", 1035.0, 15.0, "https://en.kwerk.de/tanks/dosing-tank/made-of-pe/series-pe-do-ro/50-2000-liter/", "Fixed-size caustic storage vessel."),
    EquipmentItem("Filtrate tank", 97.0, 15.0, "https://enduramaxx.co.uk/enduramaxx/200-litre-chemical-dosing-tank/", "Fixed-size filtrate hold tank."),
    EquipmentItem("Mixer", 1330.0, 15.0, "https://srt-mischer.de/300-liter-pe-dosierbehaelter-kunststoffbehaelter-mit-propellerruehrwerk-schnellmischer-ruehrwerk.html?language=de", "Continuous-duty precipitation mixing."),
    EquipmentItem("NaOH dosing pump", 1596.0, 7.0, "https://tfpumps.com/product/prominent-gmxa0708pvt20000uec0300en-7-6-l-h-7-bar/", "Fixed-size caustic dosing pump."),
    EquipmentItem("Transfer pump (AODD)", 855.0, 7.0, "https://msepumps.co.uk/air-operated-double-diaphragm-pumps-aodd/162-aodd-12-bsp-polypropylene-epdm-pump.html?srsltid=AfmBOorOjxKU1O0DxwXewmL5UfL-AXUrV8CcTtTp6adKPC4zFND0aUzs&utm", "Fixed-size slurry transfer pump."),
    EquipmentItem("Filter housing", 380.0, 15.0, "https://ultra-soft.co.uk/product/plastic-bag-filter-housings/", "Cake separation housing."),
    EquipmentItem("pH sensor", 295.0, 2.0, "https://trafalgarscientific.co.uk/ph-electrode-for-use-with-wastewater-each/", "pH control and neutralization feedback."),
)

SELECTIVE_LEACHING_EQUIPMENT: tuple[EquipmentItem, ...] = (
    EquipmentItem("Leaching tank", 2500.0, 10.0, "https://www.tanks-direct.co.uk/enduramaxx-1000-litre-chemical-dosing-tank/p46442", "Fixed-size alkaline leaching vessel."),
    EquipmentItem("Heating element", 1200.0, 10.0, "Industrial heating elements (estimated from typical values).", "Heats slurry to 90°C."),
    EquipmentItem("Mixer", 1200.0, 8.0, "https://www.coleparmer.co.uk/b/lightnin", "High-torque slurry mixing."),
    EquipmentItem("Filter housing", 800.0, 15.0, "https://ultra-soft.co.uk/product/plastic-bag-filter-housings/", "Solid-liquid separation after leaching."),
    EquipmentItem("Dosing pump", 1596.0, 7.0, "https://tfpumps.com/product/prominent-gmxa0708pvt20000uec0300en-7-6-l-h-7-bar/", "Chemical dosing pump."),
)

ELECTROWINNING_STACK_EQUIPMENT: tuple[EquipmentItem, ...] = (
    EquipmentItem("Anode (Ti-DSA)", 45.0, 3.0, "https://german.alibaba.com/product-detail/DSA-Gr1-Gr2-TItanium-Anode-Coating-1601606745035.html?spm=a2700.7724857.0.0.5afe58e4dHBpeS", "Per stack; coated titanium anode."),
    EquipmentItem("Cathode (titanium plate)", 40.0, 10.0, "https://polymet.de/elektroden/plattenelektroden/117/titanelektrode-flach?utm", "Per stack; titanium cathode plate."),
    EquipmentItem("Electrolytic cell columns", 448.0, 10.0, "https://www.amazon.co.uk/Transparent-Acrylic-Laboratory-Electrolytic-Cell/dp/B0G2YLB2JC?utm", "Per stack; acrylic cell body."),
)

ELECTROWINNING_FIXED_EQUIPMENT: tuple[EquipmentItem, ...] = (
    EquipmentItem("Rectifier", 450.0, 10.0, "Mean Well RSP-1600-48 at Mouser.", "Fixed system equipment."),
    EquipmentItem("Tank", 97.0, 15.0, "https://enduramaxx.co.uk/enduramaxx/200-litre-chemical-dosing-tank/", "Fixed electrolyte hold tank."),
    EquipmentItem("Pump", 250.0, 5.0, "https://www.abendi.de/Magnetgekuppelte-Kreiselpumpe/", "Fixed circulation pump."),
    EquipmentItem("Heating element", 50.0, 5.0, "https://www.aquapro2000.de/schego-titanheizstab-300-watt.html", "Fixed electrolyte heater."),
    EquipmentItem("Vacuum dryer", 670.0, 10.0, "https://www.expondo.de/goldbrunn-vakuumtrockenschrank-1450-w-10070013", "Fixed product drying unit."),
    EquipmentItem("Vacuum pump", 70.0, 10.0, "https://www.amazon.de/VEVOR-Drehschieber-Vakuumpumpe-Auto-AC-Vakuumpumpen-Kit-Auto-Klimaanlagen-Harzentgasung/dp/B0FGXZZ5DR", "Fixed dryer auxiliary."),
)

ELECTROWINNING_EQUIPMENT = ELECTROWINNING_STACK_EQUIPMENT + ELECTROWINNING_FIXED_EQUIPMENT


# ============================================================================
# SECTION 3: DEDUPLICATED DOWNSTREAM PROCESSING
# ============================================================================
# DOWNSTREAM STEPS: precipitation, selective leaching, and electrowinning share
# unified route-parameterised functions. All StepSpec metadata (purpose, assumptions,
# sources, equipment) is preserved.
# ============================================================================

# Shared base constants — prevents drift between the IX and SX variants of each step.
_PRECIPITATION_BASE_CONSTANTS: dict = {
    "lang_factor": 4.74,                  # Peters & Timmerhaus (2004) Table 6-21, fluid-chemical plant
    "naoh_density_kg_per_l": 1.53,        # 50 wt% NaOH at 20 °C; Perry's Chemical Engineers' Handbook
    "naoh_price_eur_per_kg": 0.30,        # businessanalytiq.com NaOH price index, Mar 2025 (~0.25–0.35 EUR/kg)
    "mixer_power_kw": 0.06,               # continuously operated precipitation-tank mixer
    "pump_power_kw": 0.02,                # transfer pump for precipitation stage
    "wash_water_factor": 0.3,             # wash volume = 0.3 × Q_eluate/Q_strip; standard filter-cake washing
    "wash_water_price_eur_per_m3": 0.001, # process water cost; internal utility estimate
}

_SELECTIVE_LEACHING_BASE_CONSTANTS: dict = {
    "lang_factor": 3.0,                           # Peters & Timmerhaus (2004) Table 6-21, solid-liquid handling; Ntengwe et al. (2019, doi:10.1007/s42461-019-00148-x) Lang≈2.97
    "naoh_kg_per_kg_cake": 0.66,                  # Ga(OH)₃ + 2 NaOH → NaGaO₂ + 2H₂O; target 120 g/L electrolyte; Xu et al. (2024)
    "naoh_price_eur_per_kg": 0.30,                # businessanalytiq.com, Mar 2025
    "filter_bag_cost_per_kg_cake_day": 0.258993,  # derived: 11.50 EUR/bag ÷ 14-d life ÷ bag capacity
    "filter_bag_price_eur": 11.50,                # PP filter bag 10 µm; industrial catalogue price
    "heating_kwh_per_kg_cake_unnorm": 0.728054,   # Q·ρ·Cp·ΔT (25→90 °C) energy balance per kg cake
    "heating_normalization_factor": 1.39,          # cake-to-liquid ratio; converts per-litre → per-kg-cake basis
    "mixer_pump_power_kw": 0.06,                  # agitator + pump for leaching vessel
}

# ============================================================================
# PROCESS STEP 5: PRECIPITATION AFTER IX
# ============================================================================
PRECIPITATION_IX_SPEC = StepSpec(
    step_id="S5",
    title="Precipitation after IX",
    route="IX",
    purpose="Neutralize the IX eluate and precipitate gallium as gallium hydroxide before selective leaching.",
    stream_basis="IX eluate with Q_elution = 0.017778 * Q, pH < 1, and 0.1 M H2SO4.",
    mass_balance_basis="Gallium precipitates into cake; arsenic is conservatively assumed to remain fully dissolved in the effluent.",
    recovery_basis=(
        "Frozen precipitation recovery of 0.99 (99%) to the gallium-hydroxide cake. "
        "Value is a model assumption based on near-complete precipitation expected at pH 5–6 for Ga³⁺. "
        "Thermodynamic basis: Bénézeth et al. (GCA 1997, ui.adsabs.harvard.edu/abs/1997GeCoA..61.1345B) "
        "and Diakonov et al. (GCA 1997, doi:10.1016/s0016-7037(97)00011-2) on Ga(OH)3 solubility."
    ),
    sizing_basis="Fixed-size precipitation skid; flow dependence enters only through chemical and utility usage.",
    cost_basis="Annualized CapEx from fixed direct equipment cost multiplied by a Lang factor of 4.74; OpEx from NaOH, electricity, wash water, and M&O.",
    assumptions=(
        "Arsenic co-precipitation is ignored conservatively.",
        "Wash water is modelled as 0.3 * Q_elution.",
        "Mixer and transfer-pump duty are continuous.",
    ),
    sources=(
        # Ga(OH)₃/Ga(OH)₄⁻ thermodynamics (Bénézeth et al. 1997):
        "https://ui.adsabs.harvard.edu/abs/1997GeCoA..61.1345B",
        # Ksp data for pH 5.0 target (Diakonov et al. 1997):
        "https://doi.org/10.1016/s0016-7037(97)00011-2",
        "https://businessanalytiq.com/procurementanalytics/index/sodium-hydroxide-price-index/",
    ),
    equipment=PRECIPITATION_EQUIPMENT,
    constants={
        **_PRECIPITATION_BASE_CONSTANTS,
        "q_elution_factor": 0.017778,     # frozen elution-volume ratio from IX sizing; see IX_SPEC
        "naoh_l_per_m3_elution": 14.5,    # ~10.5 L theoretical + 38% excess to reach pH 5.0; Diakonov et al. GCA 1997 (doi:10.1016/s0016-7037(97)00011-2); Bénézeth et al. GCA 1997
    },
    tea_scope="Included in TEA through NaOH, wash water, electricity, CapEx, and REP; effluent disposal remains separate from active route totals.",
    lca_scope="Included in `full_lca` through NaOH and electricity; wash water is retained as an uncharacterized inventory helper.",
    waste_scope="The arsenic-bearing effluent helper is kept for inventory and comparison work, not for active total-cost aggregation.",
    lca_factor_keys=("naoh_50wt", "electricity"),
)

# ============================================================================
# PROCESS STEP 9: PRECIPITATION AFTER SX
# ============================================================================
PRECIPITATION_SX_SPEC = StepSpec(
    step_id="S9",
    title="Precipitation after SX",
    route="SX",
    purpose="Neutralize the acidic SX strip liquor and precipitate gallium as gallium hydroxide before selective leaching.",
    stream_basis="SX strip liquor with Q_strip = 0.02 * Q, pH < 1, 0.5 M HCl (Chen et al. 2020, doi:10.3390/su12051765).",
    mass_balance_basis="Gallium precipitates into cake; unlike the IX case, no arsenic-bearing precipitation waste stream is modelled for SX.",
    recovery_basis=(
        "Frozen precipitation recovery of 0.99 (99%) to the cake; same assumption as PRECIPITATION_IX_SPEC. "
        "Thermodynamic basis: GCA 1997 papers on Ga(OH)3 solubility (Bénézeth/Diakonov)."
    ),
    sizing_basis="Fixed-size precipitation skid shared with the IX configuration; flow dependence enters only through NaOH, wash water, and utilities.",
    cost_basis="Annualized CapEx from fixed direct equipment cost multiplied by a Lang factor of 4.74; OpEx from NaOH, electricity, wash water, and M&O.",
    assumptions=(
        "The SX strip liquor contains a higher acid concentration than the IX eluate, so NaOH demand is scaled by a fixed factor of 2.5.",
        "Wash water is modelled as 0.3 * Q_strip.",
        "Mixer and transfer-pump duty are continuous.",
    ),
    sources=(
        "https://ui.adsabs.harvard.edu/abs/1997GeCoA..61.1345B",
        "https://businessanalytiq.com/procurementanalytics/index/sodium-hydroxide-price-index/",
    ),
    equipment=PRECIPITATION_EQUIPMENT,
    constants={
        **_PRECIPITATION_BASE_CONSTANTS,
        "q_strip_factor": 0.02,           # SX strip flow fraction; from SX_SPEC strip_q_factor
        "naoh_l_per_m3_strip": 14.5 * 2.5,  # 0.5 M HCl needs 2.5× more NaOH than 0.1 M H₂SO₄ (0.5/0.2=2.5); Chen et al. (2020)
    },
    tea_scope="Included in TEA through NaOH, wash water, electricity, CapEx, and REP.",
    lca_scope="Included in `full_lca` through NaOH and electricity; wash water is retained as an uncharacterized inventory helper.",
    waste_scope="No separate SX precipitation waste stream is modeled; this absence is intentional in the frozen TEA/LCA interface.",
    lca_factor_keys=("naoh_50wt", "electricity"),
)

# ============================================================================
# PROCESS STEP 6: SELECTIVE LEACHING AFTER IX
# ============================================================================
SELECTIVE_LEACHING_IX_SPEC = StepSpec(
    step_id="S6",
    title="Selective leaching after IX",
    route="IX",
    purpose="Dissolve precipitated gallium hydroxide from the IX cake into an alkaline leachate suited for electrowinning.",
    stream_basis="Precipitation cake with Cake_mass = 0.139 * Q.",
    mass_balance_basis="Gallium dissolves into the NaOH leachate; solids are represented through cake-based consumption factors.",
    recovery_basis=(
        "Frozen leaching recovery of 0.9781 (97.81%) to the NaOH leachate. "
        "Source: Cheng/Huang et al. (MDPI Processes 2019, doi:10.3390/pr7120921)."
    ),
    sizing_basis="Fixed-size leaching train; throughput enters through cake mass and associated NaOH, filtration, and heating demand.",
    cost_basis="Annualized CapEx from fixed direct equipment cost multiplied by a Lang factor of 3.0; OpEx from NaOH, filter bags, electricity, and M&O.",
    assumptions=(
        "Target gallium concentration in leachate is 40 g/L.",
        "Process temperature is 90 C with 25 C ambient reference.",
        "The after-SX step is treated as functionally identical apart from the cake-load factor.",
    ),
    sources=(
        # Leachate 39.9 g/L Ga, 120 g/L NaOH; EW recovery 90.2% (Xu et al. 2024):
        "https://www.sciencedirect.com/science/article/pii/S1003632623664519",
        "https://businessanalytiq.com/procurementanalytics/index/sodium-hydroxide-price-index/",
    ),
    equipment=SELECTIVE_LEACHING_EQUIPMENT,
    constants={
        **_SELECTIVE_LEACHING_BASE_CONSTANTS,
        "cake_mass_factor": 0.139,  # 0.139 kg Ga(OH)₃ cake per m³ feed; IX mass balance (stoichiometric)
    },
    tea_scope="Included in TEA through NaOH, filter-bag consumables, electricity, CapEx, and REP.",
    lca_scope="Included in `full_lca` through NaOH and electricity; filter-bag costs remain a TEA-only proxy output.",
    waste_scope="No standalone waste-stream helper is modeled for IX selective leaching.",
    lca_factor_keys=("naoh_50wt", "electricity"),
)

# ============================================================================
# PROCESS STEP 10: SELECTIVE LEACHING AFTER SX
# ============================================================================
SELECTIVE_LEACHING_SX_SPEC = StepSpec(
    step_id="S10",
    title="Selective leaching after SX",
    route="SX",
    purpose="Dissolve precipitated gallium hydroxide from the SX cake into an alkaline leachate suited for electrowinning.",
    stream_basis="Precipitation cake with Cake_mass = 0.108 * Q.",
    mass_balance_basis="Gallium dissolves into the NaOH leachate; solids are represented through cake-based consumption factors.",
    recovery_basis="Frozen leaching recovery of 0.9781 to the leachate.",
    sizing_basis="Fixed-size leaching train shared with the IX route; throughput enters through cake mass and associated NaOH, filtration, and heating demand.",
    cost_basis="Annualized CapEx from fixed direct equipment cost multiplied by a Lang factor of 3.0; OpEx from NaOH, filter bags, electricity, and M&O.",
    assumptions=(
        "Target gallium concentration in leachate is 40 g/L.",
        "Process temperature is 90 C with 25 C ambient reference.",
        "The only modeled difference from the IX step is the lower SX cake-load factor.",
    ),
    sources=(
        "https://www.sciencedirect.com/science/article/pii/S1003632623664519",
        "https://businessanalytiq.com/procurementanalytics/index/sodium-hydroxide-price-index/",
    ),
    equipment=SELECTIVE_LEACHING_EQUIPMENT,
    constants={
        **_SELECTIVE_LEACHING_BASE_CONSTANTS,
        "cake_mass_factor": 0.108,  # 0.108 kg cake per m³ feed (SX route); lower than IX (0.139) due to smaller strip volume
    },
    tea_scope="Included in TEA through NaOH, filter-bag consumables, electricity, CapEx, and REP.",
    lca_scope="Included in `full_lca` through NaOH and electricity; filter-bag costs remain a TEA-only proxy output.",
    waste_scope="No standalone waste-stream helper is modeled for SX selective leaching.",
    lca_factor_keys=("naoh_50wt", "electricity"),
)

# ============================================================================
# PROCESS STEP 7: ELECTROWINNING AFTER IX
# ============================================================================
ELECTROWINNING_IX_SPEC = StepSpec(
    step_id="S7",
    title="Electrowinning after IX",
    route="IX",
    purpose="Recover high-purity gallium metal from the alkaline IX leachate by electrowinning.",
    stream_basis="Selective-leaching output with a gallium concentration of about 40 g/L and high NaOH concentration.",
    mass_balance_basis="Gallium is deposited to product metal; the spent electrolyte remains as a small alkaline waste stream.",
    recovery_basis="Frozen electrowinning recovery of 0.902 to the final gallium product.",
    sizing_basis="Stack count is based on the frozen leachate-volume relation Volume = 0.778 * Q and a per-stack daily capacity of 6.86 L.",
    cost_basis="Annualized CapEx from stack plus fixed equipment multiplied by a Lang factor of 4.9; OpEx from electrolysis energy, pumping, drying, HCl washing, and M&O.",
    assumptions=(
        "Electrolysis uses a 7 h cycle with 6 h active electrolysis and 1 h harvesting/refilling.",
        "Specific electricity demand is fixed at 9.05 kWh/kg Ga.",
        "The after-SX electrowinning step is treated as functionally identical apart from the feed-volume factor.",
    ),
    sources=("https://www.sciencedirect.com/science/article/pii/S1003632623664519", "https://www.sciencedirect.com/science/article/pii/S0956053X17309315"),
    equipment=ELECTROWINNING_EQUIPMENT,
    constants={
        "lang_factor": 4.9,  # Peters & Timmerhaus (2004) Table 6-21, electrochemical plant with power supply systems
        "volume_factor_l_per_q": 0.778,  # 0.778 L electrolyte per m³ feed; derived from Q_eluate × leach recovery; Xu et al. (2024)
        "ga_leachate_conc_g_per_l": 40.0,  # target 40 g Ga/L in leachate for EW; Xu et al. (2024) optimal: 39.9 g/L
        "ew_sec_kwh_per_kg_ga": 9.05,  # specific energy consumption; Xu et al. (2024) cyclone EW: ~9048 kWh/t at 750 A/m², Ti cathode
        "pump_energy_kwh_per_stack_day": 2.4,  # electrolyte circulation pump; equipment datasheet estimate
        "drying_sec_kwh_per_kg_ga": 0.1,  # vacuum-drying of harvested Ga metal; engineering estimate
        "hcl_wash_cost_eur_per_l_day": 0.0221,  # HCl wash cost per litre electrolyte per day; derived in thesis
        "hcl_price_eur_per_kg": 0.15,  # same as all other steps; ChemAnalyst 2024 EU bulk
        "ti_cathode_mass_kg_per_stack": 0.36,  # Ti cathode mass; Xu et al. (2024) — Ti excellent in alkaline Ga electrolyte
        "ti_cathode_lifetime_years": 10.0,  # assumed Ti cathode service life in 120 g/L NaOH electrolyte
        "spent_electrolyte_fraction": 0.95,  # 95% of inlet volume exits as spent electrolyte; 5% deposits as Ga metal
    },
    tea_scope="Included in TEA through electricity, HCl washing, CapEx, REP, and M&O; spent-electrolyte disposal remains a separate helper.",
    lca_scope="Included in `full_lca` through electricity, HCl washing, and titanium cathode replacement.",
    waste_scope="Spent electrolyte is retained as an inventory helper and is not included in active route totals.",
    lca_factor_keys=("hcl_32wt", "ti_cathode_replacement", "electricity"),
)

# ============================================================================
# PROCESS STEP 11: ELECTROWINNING AFTER SX
# ============================================================================
ELECTROWINNING_SX_SPEC = StepSpec(
    step_id="S11",
    title="Electrowinning after SX",
    route="SX",
    purpose="Recover high-purity gallium metal from the alkaline SX leachate by electrowinning.",
    stream_basis="Selective-leaching output with a gallium concentration of about 40 g/L and high NaOH concentration.",
    mass_balance_basis="Gallium is deposited to product metal; the spent electrolyte remains as a small alkaline waste stream.",
    recovery_basis="Frozen electrowinning recovery of 0.902 to the final gallium product.",
    sizing_basis="Stack count is based on the frozen SX leachate-volume relation Volume = 0.778 * (0.108 / 0.139) * Q and a per-stack daily capacity of 6.86 L.",
    cost_basis="Annualized CapEx from stack plus fixed equipment multiplied by a Lang factor of 4.9; OpEx from electrolysis energy, pumping, drying, HCl washing, and M&O.",
    assumptions=(
        "Electrolysis uses a 7 h cycle with 6 h active electrolysis and 1 h harvesting/refilling.",
        "Specific electricity demand is fixed at 9.05 kWh/kg Ga.",
        "The SX step uses the same equipment basis as the IX step, but a lower inlet-volume factor.",
    ),
    sources=("https://www.sciencedirect.com/science/article/pii/S1003632623664519", "https://www.sciencedirect.com/science/article/pii/S0956053X17309315"),
    equipment=ELECTROWINNING_EQUIPMENT,
    constants={
        "lang_factor": 4.9,  # Peters & Timmerhaus (2004) Table 6-21; same electrochemical category as IX EW
        "volume_factor_l_per_q": 0.778 * (0.108 / 0.139),  # 0.606 L/m³ feed; scaled from IX by ratio of SX/IX cake factors
        "stack_capacity_l_per_day": 2 * (24 / 7),  # 6.857 L/d per stack; 7 h cycle (6 h EW + 1 h harvest), 2 L cell volume
        "ga_leachate_conc_g_per_l": 40.0,  # same target as IX route; Xu et al. (2024)
        "ew_sec_kwh_per_kg_ga": 9.05,  # same EW energy as IX route; Xu et al. (2024) ~9048 kWh/t
        "pump_energy_kwh_per_stack_day": 2.4,  # same as IX route
        "drying_sec_kwh_per_kg_ga": 0.1,  # same as IX route
        "hcl_wash_cost_eur_per_l_day": 0.0221,  # same HCl wash cost basis as IX route
        "hcl_price_eur_per_kg": 0.15,  # same as all other steps
        "ti_cathode_mass_kg_per_stack": 0.36,  # same hardware as IX route; Xu et al. (2024)
        "ti_cathode_lifetime_years": 10.0,  # same cathode lifetime as IX route
        "spent_electrolyte_fraction": 0.95,  # same as IX route
    },
    tea_scope="Included in TEA through electricity, HCl washing, CapEx, REP, and M&O; spent-electrolyte disposal remains a separate helper.",
    lca_scope="Included in `full_lca` through electricity, HCl washing, and titanium cathode replacement.",
    waste_scope="Spent electrolyte is retained as an inventory helper and is not included in active route totals.",
    lca_factor_keys=("hcl_32wt", "ti_cathode_replacement", "electricity"),
)

# ============================================================================
# ROUTE SPEC REGISTRY
# ============================================================================
# Maps route ('IX' or 'SX') and process step name to their parametrized specs.
# This allows unified downstream functions to dispatch to the right constants.

DOWNSTREAM_SPECS: dict[str, dict[str, StepSpec]] = {
    'IX': {
        'precipitation': PRECIPITATION_IX_SPEC,
        'selective_leaching': SELECTIVE_LEACHING_IX_SPEC,
        'electrowinning': ELECTROWINNING_IX_SPEC,
    },
    'SX': {
        'precipitation': PRECIPITATION_SX_SPEC,
        'selective_leaching': SELECTIVE_LEACHING_SX_SPEC,
        'electrowinning': ELECTROWINNING_SX_SPEC,
    },
}


# ============================================================================
# PRECIPITATION: UNIFIED FUNCTIONS
# ============================================================================

def _get_precip_inlet_flow(Q_feed: float, spec: StepSpec) -> float:
    """Return the precipitation inlet flow for the given route spec.
    IX uses q_elution_factor, SX uses q_strip_factor."""
    if 'q_elution_factor' in spec.constants:
        return spec.constants['q_elution_factor'] * Q_feed
    return spec.constants['q_strip_factor'] * Q_feed


def _get_precip_naoh_l_per_m3(spec: StepSpec) -> float:
    """Return NaOH consumption in L/m³ of precipitation inlet.
    SX uses naoh_l_per_m3_strip (36.25 L/m³); IX uses naoh_l_per_m3_elution (14.5 L/m³)."""
    if 'naoh_l_per_m3_strip' in spec.constants:
        return spec.constants['naoh_l_per_m3_strip']  # SX: 36.25 L/m³
    return spec.constants.get('naoh_l_per_m3_elution', 14.5)  # IX: from constants


def _calc_precipitation_direct_capex() -> float:
    """Return direct purchased-equipment cost for precipitation (shared IX/SX)."""
    return _equipment_direct_capex(PRECIPITATION_EQUIPMENT)


def calc_precipitation_capex_per_m3(Q_feed, route='IX', config=None):
    """Return annualized CapEx for precipitation in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['precipitation']
    direct_capex = _calc_precipitation_direct_capex()
    return _annualized_capex_per_m3(config, direct_capex, Q_feed, spec.constants["lang_factor"])


def calc_precipitation_rep_per_m3(Q_feed, route='IX', config=None):
    """Return annualized replacement cost for precipitation in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['precipitation']
    rep = _equipment_annualized_replacement(spec.equipment)
    return rep / (Q_feed * config.operating_days)


def calc_precipitation_opex_per_m3(Q_feed, route='IX', config=None):
    """Return OpEx for precipitation in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['precipitation']
    Q_inlet = _get_precip_inlet_flow(Q_feed, spec)
    naoh_l_per_m3 = _get_precip_naoh_l_per_m3(spec)

    naoh_density = spec.constants.get("naoh_density_kg_per_l", 1.53)    # kg/L, 50 wt% NaOH
    naoh_price = spec.constants.get("naoh_price_eur_per_kg", 0.30)       # EUR/kg
    naoh_cost = (Q_inlet * naoh_l_per_m3 * naoh_density * naoh_price) / (Q_feed * config.plant_availability)

    mixer_kw = spec.constants.get("mixer_power_kw", 0.06)
    pump_kw = spec.constants.get("pump_power_kw", 0.02)
    energy_daily = (mixer_kw + pump_kw) * 24  # kWh/d
    energy_cost = (energy_daily * config.electricity_price) / (Q_feed * config.plant_availability)

    wash_factor = spec.constants.get("wash_water_factor", 0.3)
    wash_price = spec.constants.get("wash_water_price_eur_per_m3", 0.001)  # EUR/m³
    wash_water_cost = (wash_factor * Q_inlet * wash_price) / (Q_feed * config.plant_availability)

    direct_capex = _calc_precipitation_direct_capex()
    M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, spec.constants["lang_factor"])

    return naoh_cost + energy_cost + wash_water_cost + M_O_per_m3


def calc_precipitation_material_consumption(Q_feed, route='IX', config=None):
    """Return material consumption for precipitation."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['precipitation']
    Q_inlet = _get_precip_inlet_flow(Q_feed, spec)
    naoh_l_per_m3 = _get_precip_naoh_l_per_m3(spec)
    naoh_density = spec.constants.get("naoh_density_kg_per_l", 1.53)
    naoh_price = spec.constants.get("naoh_price_eur_per_kg", 0.30)
    wash_factor = spec.constants.get("wash_water_factor", 0.3)
    wash_price = spec.constants.get("wash_water_price_eur_per_m3", 0.001)
    return {
        'naoh_50wt': _build_inventory_entry(
            consumption_per_m3=(Q_inlet * naoh_l_per_m3 * naoh_density) / (Q_feed * config.plant_availability),
            unit='kg/m3 Feed',
            price_per_unit=naoh_price,
            source='NaOH solution (50 wt%, density 1.53 kg/L) for gallium hydroxide precipitation.',
            lca_flow_key='naoh_50wt',
        ),
        'wash_water': _build_inventory_entry(
            consumption_per_m3=(wash_factor * Q_inlet) / (Q_feed * config.plant_availability),
            unit='m3/m3 Feed',
            price_per_unit=wash_price,
            source='Process-water wash for filter-cake cleaning.',
            tea_cost_scope='included_in_step_opex',
            lca_scope='not_characterized_in_current_repository_lca',
        ),
    }


def calc_precipitation_energy_consumption(Q_feed, route='IX', config=None):
    """Return electricity consumption for precipitation."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['precipitation']
    mixer_kw = spec.constants.get("mixer_power_kw", 0.06)
    pump_kw = spec.constants.get("pump_power_kw", 0.02)
    SEC = ((mixer_kw + pump_kw) * 24) / (Q_feed * config.plant_availability)
    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Mixer at 60 W plus transfer pump at 20 W, both operated continuously.',
        )
    }


def calc_precipitation_waste_stream(Q_feed, route='IX', config=None):
    """Return the precipitation effluent stream (IX only has arsenic waste)."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['precipitation']
    Q_inlet = _get_precip_inlet_flow(Q_feed, spec)
    Q_effluent = Q_inlet

    if route == 'IX':
        arsenic_concentration = 59.0922
        arsenic_mass_per_day = (Q_effluent * arsenic_concentration) / 1000
        return _annotate_waste_stream(
            {
                'volume_per_day': Q_effluent,
                'volume_per_m3_feed': Q_effluent / Q_feed,
                'arsenic_concentration': arsenic_concentration,
                'arsenic_mass_per_day': arsenic_mass_per_day,
                'arsenic_mass_per_m3_feed': arsenic_mass_per_day / Q_feed,
                'description': 'Effluent after IX precipitation at pH 5 with dissolved arsenic and sodium-sulfate-rich liquor.',
                'source': 'Effluent volume equals Q_elution; arsenic stays dissolved because co-precipitation is ignored conservatively.',
            },
            scope_note='The IX precipitation effluent is available for waste-traceability and comparative inventory work, but it is not charged in active TEA totals.',
        )
    # SX: no arsenic waste stream modelled
    return _annotate_waste_stream(
        {
            'volume_per_day': Q_effluent,
            'volume_per_m3_feed': Q_effluent / Q_feed,
            'description': 'SX precipitation effluent without arsenic.',
            'source': 'No arsenic-bearing waste stream modelled for SX precipitation.',
        },
        scope_note='No separate SX precipitation waste stream is modeled in the frozen TEA/LCA interface.',
    )


def calc_precipitation_waste_disposal_cost_per_m3(Q_feed, route='IX', config=None):
    """Return effluent-disposal cost for precipitation in ``EUR/m3 feed``.

    Exposes disposal cost for sensitivity analysis and waste-stream inventory only.
    This function is **not** included in ``calc_total_costs()``; see
    ``_annotate_waste_stream`` for the ``excluded_from_active_total_costs`` flag.
    """
    config = config or DEFAULT_CONFIG
    waste_stream = calc_precipitation_waste_stream(Q_feed, route=route, config=config)
    Q_effluent = waste_stream['volume_per_day']
    disposal_cost_daily = Q_effluent * 1.10
    return disposal_cost_daily / (Q_feed * config.plant_availability)




# ============================================================================
# SELECTIVE LEACHING: UNIFIED FUNCTIONS
# ============================================================================

def _calc_selective_leaching_direct_capex():
    """Return direct purchased-equipment cost for selective leaching (shared IX/SX)."""
    return _equipment_direct_capex(SELECTIVE_LEACHING_EQUIPMENT)


def calc_selective_leaching_capex_per_m3(Q_feed, route='IX', config=None):
    """Return annualized CapEx for selective leaching in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['selective_leaching']
    direct_capex = _calc_selective_leaching_direct_capex()
    TCI = direct_capex * spec.constants["lang_factor"]
    return (TCI * config.CRF) / (Q_feed * config.operating_days)


def calc_selective_leaching_rep_per_m3(Q_feed, route='IX', config=None):
    """Return annualized replacement cost for selective leaching in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['selective_leaching']
    rep = _equipment_annualized_replacement(spec.equipment)
    return rep / (Q_feed * config.operating_days)


def calc_selective_leaching_opex_per_m3(Q_feed, route='IX', config=None):
    """Return OpEx for selective leaching in ``EUR/m3 feed``."""
    # NOTE: annual M&O is divided by daily operating time before feed-normalization (frozen baseline behaviour).
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['selective_leaching']
    cake_mass = spec.constants["cake_mass_factor"] * Q_feed

    naoh_kg_per_kg = spec.constants.get("naoh_kg_per_kg_cake", 0.66)
    naoh_price = spec.constants.get("naoh_price_eur_per_kg", 0.30)
    filter_coeff = spec.constants.get("filter_bag_cost_per_kg_cake_day", 0.258993)
    heat_coeff = spec.constants.get("heating_kwh_per_kg_cake_unnorm", 0.728054)
    heat_norm = spec.constants.get("heating_normalization_factor", 1.39)
    mixer_pump_kw = spec.constants.get("mixer_pump_power_kw", 0.06)

    NaOH_cost = naoh_kg_per_kg * cake_mass * naoh_price              # EUR/d
    filter_cost = filter_coeff * cake_mass                            # EUR/d
    heating_energy = heat_coeff * cake_mass / heat_norm               # kWh/d
    mixer_pump_energy = mixer_pump_kw * 24                            # kWh/d
    energy_cost = (heating_energy + mixer_pump_energy) * config.electricity_price

    direct_capex = _calc_selective_leaching_direct_capex()

    if route == 'SX':
        # Frozen compatibility quirk: SX uses daily M&O normalization
        m_o_annual = direct_capex * spec.constants["lang_factor"] * config.O_M_rate
        m_o_daily = m_o_annual / config.operating_days
        total_opex_daily = NaOH_cost + filter_cost + energy_cost + m_o_daily
        return total_opex_daily / (Q_feed * config.plant_availability)
    else:
        M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, spec.constants["lang_factor"])
        total_opex = (NaOH_cost + filter_cost + energy_cost) / (Q_feed * config.plant_availability) + M_O_per_m3
        return total_opex


def calc_selective_leaching_material_consumption(Q_feed, route='IX', config=None):
    """Return material consumption for selective leaching."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['selective_leaching']
    cake_mass = spec.constants["cake_mass_factor"] * Q_feed
    naoh_kg_per_kg = spec.constants.get("naoh_kg_per_kg_cake", 0.66)
    naoh_price = spec.constants.get("naoh_price_eur_per_kg", 0.30)
    filter_coeff = spec.constants.get("filter_bag_cost_per_kg_cake_day", 0.258993)
    filter_bag_price = spec.constants.get("filter_bag_price_eur", 11.50)
    return {
        'naoh': _build_inventory_entry(
            consumption_per_m3=(naoh_kg_per_kg * cake_mass) / (Q_feed * config.plant_availability),
            unit='kg/m3 Feed',
            price_per_unit=naoh_price,
            source='0.66 kg NaOH / kg Cake for alkaline leaching; sciencedirect.com/science/article/pii/S1003632623664519.',
            lca_flow_key='naoh_50wt',
        ),
        'filterbags': _build_inventory_entry(
            consumption_per_m3=(filter_coeff * cake_mass) / (Q_feed * config.plant_availability),
            unit='EUR/m3 Feed',
            price_per_unit=filter_bag_price,
            source='Filter-bag consumption at 11.50 EUR per item and a 14-day lifetime.',
            tea_cost_scope='included_in_step_opex',
            lca_scope='not_characterized_in_current_repository_lca',
        ),
    }


def calc_selective_leaching_energy_consumption(Q_feed, route='IX', config=None):
    """Return electricity consumption for selective leaching."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['selective_leaching']
    cake_mass = spec.constants["cake_mass_factor"] * Q_feed
    heat_coeff = spec.constants.get("heating_kwh_per_kg_cake_unnorm", 0.728054)
    heat_norm = spec.constants.get("heating_normalization_factor", 1.39)
    mixer_pump_kw = spec.constants.get("mixer_pump_power_kw", 0.06)
    heating_energy = heat_coeff * cake_mass / heat_norm               # kWh/d
    mixer_pump_energy = mixer_pump_kw * 24                            # kWh/d
    SEC = (heating_energy + mixer_pump_energy) / (Q_feed * config.plant_availability)
    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Heating to 90 C from a 25 C ambient reference plus a 60 W mixer/pump load.',
        )
    }




# ============================================================================
# ELECTROWINNING: UNIFIED FUNCTIONS
# ============================================================================

def _calc_electrowinning_n_stacks(Q_feed, route='IX'):
    """Return the number of electrowinning stacks required."""
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    volume = spec.constants["volume_factor_l_per_q"] * Q_feed
    stack_capacity = spec.constants.get("stack_capacity_l_per_day", 2 * (24 / 7))
    return ceil(volume / stack_capacity)


def _calc_electrowinning_direct_capex(Q_feed, route='IX'):
    """Return direct purchased-equipment cost for electrowinning."""
    n_stacks = _calc_electrowinning_n_stacks(Q_feed, route)
    stack_price = _equipment_direct_capex(ELECTROWINNING_STACK_EQUIPMENT)
    other_price = _equipment_direct_capex(ELECTROWINNING_FIXED_EQUIPMENT)
    return stack_price * n_stacks + other_price


def calc_electrowinning_capex_per_m3(Q_feed, route='IX', config=None):
    """Return annualized CapEx for electrowinning in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    direct_capex = _calc_electrowinning_direct_capex(Q_feed, route)
    return _annualized_capex_per_m3(config, direct_capex, Q_feed, spec.constants["lang_factor"])


def calc_electrowinning_rep_per_m3(Q_feed, route='IX', config=None):
    """Return annualized replacement cost for electrowinning in ``EUR/m3 feed``."""
    config = config or DEFAULT_CONFIG
    n_stacks = _calc_electrowinning_n_stacks(Q_feed, route)
    rep_stack = _equipment_annualized_replacement(ELECTROWINNING_STACK_EQUIPMENT)
    rep_other = _equipment_annualized_replacement(ELECTROWINNING_FIXED_EQUIPMENT)
    total_rep = rep_stack * n_stacks + rep_other
    return total_rep / (Q_feed * config.operating_days)


def calc_electrowinning_opex_per_m3(Q_feed, route='IX', config=None):
    """Return OpEx for electrowinning in ``EUR/m3 feed``."""
    # NOTE: annual M&O is divided by daily operating time before feed-normalization (frozen baseline behaviour).
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    n_stacks = _calc_electrowinning_n_stacks(Q_feed, route)
    Volume = spec.constants["volume_factor_l_per_q"] * Q_feed

    ga_conc_g_per_l = spec.constants.get("ga_leachate_conc_g_per_l", 40.0)
    Ga_concentration = ga_conc_g_per_l * config.recoveries['leach_ga_to_leachate']  # g/L
    Ga_input = Volume * Ga_concentration / 1000  # kg/d
    Ga_output = Ga_input * config.recoveries['ew_ga_to_product']  # kg/d

    ew_sec = spec.constants.get("ew_sec_kwh_per_kg_ga", 9.05)
    pump_kwh_per_stack = spec.constants.get("pump_energy_kwh_per_stack_day", 2.4)
    dry_sec = spec.constants.get("drying_sec_kwh_per_kg_ga", 0.1)
    electrolysis_energy = Ga_output * ew_sec          # kWh/d
    pump_energy = n_stacks * pump_kwh_per_stack        # kWh/d
    drying_energy = Ga_output * dry_sec                # kWh/d
    energy_cost = (electrolysis_energy + pump_energy + drying_energy) * config.electricity_price

    hcl_wash_rate = spec.constants.get("hcl_wash_cost_eur_per_l_day", 0.0221)
    hcl_cost_daily = Volume * hcl_wash_rate  # EUR/d

    direct_capex = _calc_electrowinning_direct_capex(Q_feed, route)

    if route == 'SX':
        # Frozen compatibility quirk: SX uses daily M&O normalization
        m_o_annual = direct_capex * spec.constants["lang_factor"] * config.O_M_rate
        m_o_daily = m_o_annual / config.operating_days
        total_opex_daily = energy_cost + hcl_cost_daily + m_o_daily
        return total_opex_daily / (Q_feed * config.plant_availability)
    else:
        M_O_per_m3 = _maintenance_opex_per_m3(config, direct_capex, Q_feed, spec.constants["lang_factor"])
        total_opex = (energy_cost + hcl_cost_daily) / (Q_feed * config.plant_availability) + M_O_per_m3
        return total_opex


def calc_electrowinning_material_consumption(Q_feed, route='IX', config=None):
    """Return material consumption for electrowinning."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    Volume = spec.constants["volume_factor_l_per_q"] * Q_feed
    hcl_price = spec.constants.get("hcl_price_eur_per_kg", 0.15)
    hcl_wash_rate = spec.constants.get("hcl_wash_cost_eur_per_l_day", 0.0221)
    hcl_cost_daily = Volume * hcl_wash_rate  # EUR/d
    hcl_consumption_daily = hcl_cost_daily / hcl_price  # kg/d

    n_stacks = _calc_electrowinning_n_stacks(Q_feed, route)
    cathode_ti_mass_per_unit = spec.constants.get("ti_cathode_mass_kg_per_stack", 0.36)
    cathode_lifetime_years = spec.constants.get("ti_cathode_lifetime_years", 10.0)
    cathode_ti_replacement_annual = n_stacks * (cathode_ti_mass_per_unit / cathode_lifetime_years)
    annual_volume = Q_feed * config.operating_days
    cathode_ti_consumption_per_m3 = cathode_ti_replacement_annual / annual_volume

    return {
        'hcl_32wt_washing': _build_inventory_entry(
            consumption_per_m3=hcl_consumption_daily / (Q_feed * config.plant_availability),
            unit='kg/m3 Feed',
            price_per_unit=hcl_price,
            source='HCl wash of deposited Ga; model uses 0.0221 EUR/L-d converted with 0.15 EUR/kg (32 wt% HCl).',
            lca_flow_key='hcl_32wt',
        ),
        'ti_cathode_replacement': _build_inventory_entry(
            consumption_per_m3=cathode_ti_consumption_per_m3,
            unit='kg Ti/m3 Feed',
            source='Cathode replacement mass flow: 0.36 kg Ti/cathode, 10-year lifetime, 1 cathode per stack.',
            lca_flow_key='ti_cathode_replacement',
            tea_cost_scope='included_in_step_rep',
        ),
    }


def calc_electrowinning_energy_consumption(Q_feed, route='IX', config=None):
    """Return electricity consumption for electrowinning."""
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    n_stacks = _calc_electrowinning_n_stacks(Q_feed, route)
    Volume = spec.constants["volume_factor_l_per_q"] * Q_feed

    ga_conc_g_per_l = spec.constants.get("ga_leachate_conc_g_per_l", 40.0)
    Ga_concentration = ga_conc_g_per_l * config.recoveries['leach_ga_to_leachate']
    Ga_input = Volume * Ga_concentration / 1000
    Ga_output = Ga_input * config.recoveries['ew_ga_to_product']

    ew_sec = spec.constants.get("ew_sec_kwh_per_kg_ga", 9.05)
    pump_kwh_per_stack = spec.constants.get("pump_energy_kwh_per_stack_day", 2.4)
    dry_sec = spec.constants.get("drying_sec_kwh_per_kg_ga", 0.1)
    electrolysis_energy = Ga_output * ew_sec          # kWh/d
    pump_energy = n_stacks * pump_kwh_per_stack        # kWh/d
    drying_energy = Ga_output * dry_sec                # kWh/d
    SEC = (electrolysis_energy + pump_energy + drying_energy) / (Q_feed * config.plant_availability)

    return {
        'electricity': _build_energy_entry(
            consumption_per_m3=SEC,
            price_per_unit=config.electricity_price,
            source='Electrolysis at 9.05 kWh/kg Ga plus pumping at 2.4 kWh/d per stack and drying at 0.1 kWh/kg Ga.',
        )
    }


def calc_electrowinning_waste_stream(Q_feed, route='IX', config=None):
    """Return the spent-electrolyte stream from electrowinning."""
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    Volume = spec.constants["volume_factor_l_per_q"] * Q_feed
    spent_fraction = spec.constants.get("spent_electrolyte_fraction", 0.95)
    Q_waste = spent_fraction * Volume

    return _annotate_waste_stream(
        {
            'volume_per_day': Q_waste,
            'volume_per_m3_feed': Q_waste / Q_feed,
            'description': 'Spent electrolyte equal to 95% of the electrowinning feed volume.',
            'source': 'Electrolyte after one pass through the electrowinning cell.',
        },
        scope_note='The spent electrolyte is retained for inventory traceability, but it is not added to active TEA totals.',
    )


def calc_electrowinning_waste_disposal_cost_per_m3(Q_feed, route='IX', config=None):
    """Return spent-electrolyte disposal cost for electrowinning in ``EUR/m3 feed``.

    Exposes disposal cost for sensitivity analysis and waste-stream inventory only.
    This function is **not** included in ``calc_total_costs()``; see
    ``_annotate_waste_stream`` for the ``excluded_from_active_total_costs`` flag.
    """
    config = config or DEFAULT_CONFIG
    spec = DOWNSTREAM_SPECS[route]['electrowinning']
    Volume = spec.constants["volume_factor_l_per_q"] * Q_feed
    spent_fraction = spec.constants.get("spent_electrolyte_fraction", 0.95)
    Q_waste = spent_fraction * Volume / 1000  # m³/d
    disposal_cost_daily = Q_waste * 1.0       # 1.0 EUR/m³ disposal tariff
    return disposal_cost_daily / (Q_feed * config.plant_availability)




# --- END OF PART 3: DEDUPLICATED DOWNSTREAM PROCESSING ---


# ############################################################################
# PART 4: ROUTE-LEVEL ECONOMICS, PRODUCTION, AND CARBON ACCOUNTING
# ############################################################################
# This section aggregates step-level costs into route totals, computes the
# levelized cost of gallium (LCO-Ga), break-even prices, annual production
# rates, cost-breakdown views, and carbon-accounting functions.
# ############################################################################


def calc_labour_costs_per_m3(Q_feed, config=None):
    """Return labor costs per m³ of feed.

    Annual labour costs = fully loaded salary × FTE, annualized over the
    operating volume (Q_feed × operating_days).
    """
    config = config or DEFAULT_CONFIG
    labour_cost_annual = config.labour_cost_full_time * config.labour_FTE
    annual_volume = Q_feed * config.operating_days
    return labour_cost_annual / annual_volume


def _resolve_route(route: str) -> str:
    """Validate and return the route identifier used throughout the model."""
    if route not in ('IX', 'SX'):
        raise ValueError(f"Unknown route: {route}. Expected 'IX' or 'SX'.")
    return route


def _resolve_co2_tax_mode(mode, config=None):
    """Resolve the active carbon-cost scope while keeping legacy defaults stable."""
    config = config or DEFAULT_CONFIG
    resolved = config.co2_tax_mode if mode is None else mode
    if resolved not in VALID_CO2_TAX_MODES:
        raise ValueError(
            f"Unsupported co2_tax_mode: {resolved}. Expected one of {VALID_CO2_TAX_MODES}."
        )
    return resolved


def _resolve_co2_energy_path(energy_path, config=None):
    """Resolve the energy-only implementation path for carbon accounting."""
    config = config or DEFAULT_CONFIG
    resolved = config.co2_energy_path if energy_path is None else energy_path
    if resolved not in VALID_CO2_ENERGY_PATHS:
        raise ValueError(
            f"Unsupported co2_energy_path: {resolved}. Expected one of {VALID_CO2_ENERGY_PATHS}."
        )
    return resolved


# ============================================================================
# STEP ENERGY CONSUMPTION (route-level aggregation)
# ============================================================================

def calc_step_energy_consumption(Q_feed, route='IX', energy_path=None, config=None):
    """
    Return electricity demand by process block in ``kWh/m³ feed``.

    This helper centralizes the route-specific energy graph that is reused by
    carbon costing, figure generation, and source-audit work.

    The ``energy_path`` parameter controls how SX downstream energy is resolved:
    - ``'legacy_frozen'``: SX downstream reuses IX energy functions (thesis behavior)
    - ``'route_consistent'``: SX downstream uses SX-specific energy functions
    """
    config = config or DEFAULT_CONFIG
    route = _resolve_route(route)
    resolved_energy_path = _resolve_co2_energy_path(energy_path, config)

    energy_by_step = {
        'Filtration': calc_filtration_energy_consumption(Q_feed, config=config)['electricity']['consumption_per_m3'],
        'RO_Split': calc_ro_split_energy_consumption(Q_feed, config=config)['electricity']['consumption_per_m3'],
        'pH_Adjust': calc_ph_adjust_energy_consumption(Q_feed, config=config)['electricity']['consumption_per_m3'],
    }

    if route == 'IX':
        energy_by_step.update({
            'Separation': calc_ix_energy_consumption(Q_feed, config=config)['electricity']['consumption_per_m3'],
            'Precipitation': calc_precipitation_energy_consumption(Q_feed, route='IX', config=config)['electricity']['consumption_per_m3'],
            'Selective_Leaching': calc_selective_leaching_energy_consumption(Q_feed, route='IX', config=config)['electricity']['consumption_per_m3'],
            'Electrowinning': calc_electrowinning_energy_consumption(Q_feed, route='IX', config=config)['electricity']['consumption_per_m3'],
        })
        return energy_by_step

    # SX route
    energy_by_step['Separation'] = calc_sx_energy_consumption(Q_feed, config=config)['electricity']['consumption_per_m3']

    if resolved_energy_path == 'legacy_frozen':
        # Legacy frozen: SX downstream reuses IX energy functions (thesis behavior)
        energy_by_step['Precipitation'] = calc_precipitation_energy_consumption(Q_feed, route='IX', config=config)['electricity']['consumption_per_m3']
        energy_by_step['Selective_Leaching'] = calc_selective_leaching_energy_consumption(Q_feed, route='IX', config=config)['electricity']['consumption_per_m3']
        energy_by_step['Electrowinning'] = calc_electrowinning_energy_consumption(Q_feed, route='IX', config=config)['electricity']['consumption_per_m3']
    else:
        # Route consistent: SX downstream uses SX-specific energy functions
        energy_by_step['Precipitation'] = calc_precipitation_energy_consumption(Q_feed, route='SX', config=config)['electricity']['consumption_per_m3']
        energy_by_step['Selective_Leaching'] = calc_selective_leaching_energy_consumption(Q_feed, route='SX', config=config)['electricity']['consumption_per_m3']
        energy_by_step['Electrowinning'] = calc_electrowinning_energy_consumption(Q_feed, route='SX', config=config)['electricity']['consumption_per_m3']

    return energy_by_step


# ============================================================================
# CARBON ACCOUNTING
# ============================================================================

def _calc_energy_only_carbon_load_legacy_frozen(Q_feed, route, config=None):
    """Return the frozen electricity-only carbon load in ``kg CO2eq/m³ feed``."""
    # NOTE: legacy_frozen path reuses IX downstream energy helpers for SX precipitation/leaching/EW (frozen thesis baseline).
    config = config or DEFAULT_CONFIG
    total_energy = sum(
        calc_step_energy_consumption(Q_feed, route=route, energy_path='legacy_frozen', config=config).values()
    )
    return total_energy * config.energy_only_grid_emission_factor


def _calc_energy_only_carbon_load_route_consistent(Q_feed, route, config=None):
    """Return the route-consistent electricity-only carbon load in ``kg CO2eq/m³ feed``."""
    config = config or DEFAULT_CONFIG
    total_energy = sum(
        calc_step_energy_consumption(Q_feed, route=route, energy_path='route_consistent', config=config).values()
    )
    return total_energy * config.energy_only_grid_emission_factor


def _calc_energy_only_carbon_load_per_m3(Q_feed, route, energy_path, config=None):
    """Return the electricity-only carbon load for the selected implementation path."""
    if energy_path == 'legacy_frozen':
        return _calc_energy_only_carbon_load_legacy_frozen(Q_feed, route, config)
    if energy_path == 'route_consistent':
        return _calc_energy_only_carbon_load_route_consistent(Q_feed, route, config)
    raise ValueError(
        f"Unsupported co2_energy_path: {energy_path}. Expected one of {VALID_CO2_ENERGY_PATHS}."
    )


def _load_lca_aggregation_module():
    """Return the ``lca_aggregation`` module lazily to avoid import-cycle coupling."""
    return _lazy_load_lca_module("lca_aggregation")


def _calc_full_lca_carbon_load_per_m3(Q_feed, route, config=None):
    """
    Return the modeled LCA carbon load in ``kg CO2eq/m³ feed``.

    This is a shadow-price-ready bridge to the repository's current LCA scope.
    It should be interpreted as the modeled cradle-to-gate carbon load covered
    by the active LCI implementation, not as a universal cradle-to-grave claim.
    """
    route = _resolve_route(route)
    lca = _load_lca_aggregation_module()
    return lca.calc_lca_gwp_total_per_m3_feed(Q_feed, route)


def calc_carbon_load_per_m3(Q_feed, route='IX', mode=None, energy_path=None, config=None):
    """
    Return the modeled carbon load in ``kg CO2eq/m³ feed`` for the active scope.

    Scope interpretation:
    - ``none``: return ``0.0`` because no carbon-cost scope is active
    - ``energy_only``: electricity-only carbon-load proxy
    - ``full_lca``: full modeled LCA carbon load from the repository LCA bridge
    """
    config = config or DEFAULT_CONFIG
    resolved_mode = _resolve_co2_tax_mode(mode, config)

    if resolved_mode == 'none':
        return 0.0
    if resolved_mode == 'energy_only':
        resolved_energy_path = _resolve_co2_energy_path(energy_path, config)
        return _calc_energy_only_carbon_load_per_m3(Q_feed, route, resolved_energy_path, config)
    if resolved_mode == 'full_lca':
        return _calc_full_lca_carbon_load_per_m3(Q_feed, route, config)
    raise ValueError(
        f"Unsupported co2_tax_mode: {resolved_mode}. Expected one of {VALID_CO2_TAX_MODES}."
    )


def calc_co2_tax_per_m3(Q_feed, route='IX', mode=None, energy_path=None, config=None):
    """
    Return the carbon-cost contribution in ``EUR/m³ feed`` for the selected scope.

    Method note:
    - ``energy_only`` prices only electricity-related carbon load.
    - ``full_lca`` prices the modeled LCA carbon load as a shadow-price block.
    - The frozen thesis baseline remains ``energy_only`` with
      ``co2_energy_path='legacy_frozen'``.
    """
    config = config or DEFAULT_CONFIG
    carbon_load = calc_carbon_load_per_m3(
        Q_feed, route=route, mode=mode, energy_path=energy_path, config=config,
    )
    return carbon_load * config.co2_tax_per_ton / 1000


# ============================================================================
# ROUTE-LEVEL COST AGGREGATION
# ============================================================================

def calc_total_costs(Q_feed, route='IX', config=None):
    """
    Return per-step and total costs per m³ of feed for the selected route.

    The returned dict has two distinct levels of multiplier application:

    - **Individual step entries** (``costs['Filtration']``, ``costs['IX']``, etc.)
      contain the *raw* cost values directly from each ``calc_*`` function,
      **before** any scenario multipliers are applied.
    - **``costs['Total']``** contains the *final* aggregated values after
      ``capex_multiplier``, ``rep_multiplier``, and ``labour_multiplier`` have
      been applied to the respective subtotals.

    Callers that re-aggregate individual step costs (e.g. ``calc_cost_breakdown()``)
    must apply the relevant multipliers themselves; callers that only need the
    aggregate should read from ``costs['Total']``.

    The returned ``CO2_Tax`` block remains the frozen public key for all carbon
    scopes. Under ``full_lca`` it represents a shadow-priced carbon-cost block
    based on the repository's modeled LCA carbon load.
    """
    config = config or DEFAULT_CONFIG
    route = _resolve_route(route)

    # Shared upstream steps
    common_costs = {
        'Filtration': {
            'CapEx': calc_filtration_capex_per_m3(Q_feed, config=config),
            'REP': calc_filtration_rep_per_m3(Q_feed, config=config),
            'OpEx': calc_filtration_opex_per_m3(Q_feed, config=config),
        },
        'RO_Split': {
            'CapEx': calc_ro_split_capex_per_m3(Q_feed, config=config),
            'REP': calc_ro_split_rep_per_m3(Q_feed, config=config),
            'OpEx': calc_ro_split_opex_per_m3(Q_feed, config=config),
        },
        'pH_Adjust': {
            'CapEx': calc_ph_adjust_capex_per_m3(Q_feed, config=config),
            'REP': calc_ph_adjust_rep_per_m3(Q_feed, config=config),
            'OpEx': calc_ph_adjust_opex_per_m3(Q_feed, config=config),
        },
    }

    costs = common_costs.copy()

    # Route-specific separation step
    if route == 'IX':
        costs['IX'] = {
            'CapEx': calc_ix_capex_per_m3(Q_feed, config=config),
            'REP': calc_ix_rep_per_m3(Q_feed, config=config),
            'OpEx': calc_ix_opex_per_m3(Q_feed, config=config),
        }
    else:
        costs['SX'] = {
            'CapEx': calc_sx_capex_per_m3(Q_feed, config=config),
            'REP': calc_sx_rep_per_m3(Q_feed, config=config),
            'OpEx': calc_sx_opex_per_m3(Q_feed, config=config),
        }

    # Downstream steps (unified functions with route parameter)
    costs['Precipitation'] = {
        'CapEx': calc_precipitation_capex_per_m3(Q_feed, route=route, config=config),
        'REP': calc_precipitation_rep_per_m3(Q_feed, route=route, config=config),
        'OpEx': calc_precipitation_opex_per_m3(Q_feed, route=route, config=config),
    }
    costs['Selective_Leaching'] = {
        'CapEx': calc_selective_leaching_capex_per_m3(Q_feed, route=route, config=config),
        'REP': calc_selective_leaching_rep_per_m3(Q_feed, route=route, config=config),
        'OpEx': calc_selective_leaching_opex_per_m3(Q_feed, route=route, config=config),
    }
    costs['Electrowinning'] = {
        'CapEx': calc_electrowinning_capex_per_m3(Q_feed, route=route, config=config),
        'REP': calc_electrowinning_rep_per_m3(Q_feed, route=route, config=config),
        'OpEx': calc_electrowinning_opex_per_m3(Q_feed, route=route, config=config),
    }

    # Aggregate totals
    total_capex = sum(v['CapEx'] for v in costs.values())
    total_rep = sum(v['REP'] for v in costs.values())
    total_opex = sum(v['OpEx'] for v in costs.values())
    labour_costs = calc_labour_costs_per_m3(Q_feed, config=config)

    # Carbon-cost block
    co2_tax = calc_co2_tax_per_m3(Q_feed, route=route, config=config)

    # Apply scenario multipliers
    total_capex = total_capex * config.capex_multiplier
    total_rep = total_rep * config.rep_multiplier
    labour_costs = labour_costs * config.labour_multiplier

    # Total cost = CapEx + replacement + OpEx + labour + CO2 tax
    total_cost = total_capex + total_rep + total_opex + labour_costs + co2_tax

    costs['Total'] = {
        'CapEx': total_capex,
        'REP': total_rep,
        'OpEx': total_opex,
        'Labour': labour_costs,
        'CO2_Tax': co2_tax,
        'Total': total_cost,
    }

    return costs


def _ga_output_per_m3(route, config=None):
    """Return gallium output per m³ of feed in kg Ga / m³ Feed."""
    config = config or DEFAULT_CONFIG
    C_Ga_feed = FEED_BASELINE_TEMPLATE['species_mg_L']['Ga']  # mg/L = g/m³
    if route == 'IX':
        return C_Ga_feed / 1000 * config.recovery_rate_total_ix
    elif route == 'SX':
        return C_Ga_feed / 1000 * config.recovery_rate_total_sx
    else:
        raise ValueError(f"Unknown route: {route}")


def calc_lco_ga(Q_feed, route='IX', config=None):
    """Return the levelized cost of gallium in EUR/kg Ga for the selected route."""
    config = config or DEFAULT_CONFIG
    costs = calc_total_costs(Q_feed, route=route, config=config)
    total_cost_per_m3 = costs['Total']['Total']
    Ga_out = _ga_output_per_m3(route, config)
    return total_cost_per_m3 / Ga_out


def calc_profitability(Q_feed, market_price, route='IX', config=None):
    """Return revenue, cost, profit, margin, and LCO-Ga per m³ of feed."""
    config = config or DEFAULT_CONFIG
    costs = calc_total_costs(Q_feed, route=route, config=config)
    total_cost_per_m3 = costs['Total']['Total']
    Ga_out = _ga_output_per_m3(route, config)

    revenue_per_m3 = Ga_out * market_price
    profit_per_m3 = revenue_per_m3 - total_cost_per_m3
    profit_margin = (profit_per_m3 / revenue_per_m3 * 100) if revenue_per_m3 > 0 else 0

    return {
        'revenue_per_m3': revenue_per_m3,
        'cost_per_m3': total_cost_per_m3,
        'profit_per_m3': profit_per_m3,
        'profit_margin': profit_margin,
        'lco_ga': total_cost_per_m3 / Ga_out,
    }


def calc_break_even_price(Q_feed, route='IX', config=None):
    """Return the break-even gallium market price in ``EUR/kg Ga`` for the selected route.

    The break-even price equals the Levelised Cost of Gallium (LCO-Ga): the
    minimum market price at which revenue exactly covers all annualised costs
    (CapEx + OpEx + CO₂).  Mathematically, break-even price ≡ LCO-Ga, so this
    function is a semantic alias for ``calc_lco_ga``.  It is exposed separately
    to make the economic interpretation explicit in sensitivity studies.
    """
    return calc_lco_ga(Q_feed, route=route, config=config)


# ============================================================================
# SECTION 6: OUTPUT METRICS AND REPORTING VIEWS
# ============================================================================
# This section contains output metrics and reporting-oriented derived views that
# remain part of the scientific module because they are directly downstream of
# the scientific model outputs and are consumed by the frozen export path.
# ============================================================================

def calc_annual_production(Q_feed, route='IX', config=None):
    """Return annual gallium production in kg/year for the selected route."""
    config = config or DEFAULT_CONFIG
    Ga_out = _ga_output_per_m3(route, config)
    Ga_output_daily = Ga_out * Q_feed
    return Ga_output_daily * config.operating_days


def calc_cost_breakdown(Q_feed, route='IX', config=None):
    """
    Return the reporting-oriented cost-block breakdown for the selected route.

    Groups costs into CapEx-Sep, CapEx-Other, OpEx, Repl-Sep, Repl-Other,
    Labour, and CO2_Tax blocks matching the thesis figure conventions.
    """
    config = config or DEFAULT_CONFIG
    route = _resolve_route(route)
    costs = calc_total_costs(Q_feed, route=route, config=config)

    sep_key = route  # 'IX' or 'SX'

    # CapEx-Sep: pH adjustment + route separation + downstream conversion
    capex_sep = (
        costs['pH_Adjust']['CapEx']
        + costs[sep_key]['CapEx']
        + costs['Precipitation']['CapEx']
        + costs['Selective_Leaching']['CapEx']
        + costs['Electrowinning']['CapEx']
    ) * config.capex_multiplier

    # Repl-Sep: replacement for the same grouped separation steps
    repl_sep = (
        costs['pH_Adjust']['REP']
        + costs[sep_key]['REP']
        + costs['Precipitation']['REP']
        + costs['Selective_Leaching']['REP']
        + costs['Electrowinning']['REP']
    ) * config.rep_multiplier

    # CapEx-Other: shared pretreatment outside the route-specific separation block
    capex_other = (
        costs['Filtration']['CapEx'] + costs['RO_Split']['CapEx']
    ) * config.capex_multiplier

    # OpEx: total operating expenditure
    opex = costs['Total']['OpEx']

    # Repl-Other: replacement for shared pretreatment steps
    repl_other = (
        costs['Filtration']['REP'] + costs['RO_Split']['REP']
    ) * config.rep_multiplier

    # Labour and CO2 tax: separate reporting blocks
    labour = costs['Total']['Labour']
    co2_tax = costs['Total']['CO2_Tax']

    return {
        'CapEx-Sep': capex_sep,
        'CapEx-Other': capex_other,
        'OpEx': opex,
        'Repl-Sep': repl_sep,
        'Repl-Other': repl_other,
        'Labour': labour,
        'CO2_Tax': co2_tax,
    }


# --- END OF PART 4: ROUTE-LEVEL ECONOMICS ---
