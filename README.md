# Gallium TEA Model

Techno-Economic Analysis (TEA) model for the recovery of **4N-grade gallium** from GaAs
semiconductor manufacturing wastewater.  Developed as part of a master's thesis at the
University of Cambridge / WZL RWTH Aachen University.

## Process Overview

Two recovery routes share a common upstream section (Filtration → RO Split → pH Adjust)
and diverge at the selective separation step:

```
Route A — Ion Exchange (IX)
  Raw feed → Filtration → RO Split → pH Adjust → IX Separation
           → Precipitation → Selective Leaching → Electrowinning

Route B — Solvent Extraction (SX)
  Raw feed → Filtration → RO Split → pH Adjust → SX Separation
           → Precipitation → Selective Leaching → Electrowinning
```

**Feed:** GaAs process wastewater, **34.6 mg/L Ga**, pH 3.8 (Jain 2019). Full characterisation
(As, P, Na, Cl, SO₄, etc.) in `FEED_BASELINE_TEMPLATE` in `tea_model_ga_thesis.py`.

**Throughput range:** Q = 10 – 100 m³/d

## TEA Methodology

**System boundary:** Gate-to-gate, from raw GaAs wastewater inlet to refined 4N gallium product.
Upstream semiconductor manufacturing and downstream product use are excluded.

**Functional unit:** 1 kg of 4N gallium produced.

**Cost framework:** The levelised cost of gallium (LCO-Ga) is total annual cost divided by
annual gallium production:

  LCO-Ga = TC / m_Ga,ann   [EUR/kg Ga]

  m_Ga,ann = Q × c_Ga × TR × availability × 365 days   [kg/yr]

**Total annual cost (TC)** is the sum of five blocks:

| Block | Description |
|-------|-------------|
| CapEx | Installed capital = Σ_s (f_Lang,s × Σ C_equip,s) |
| OpEx | Electricity + chemicals + O&M (fixed fraction of CapEx) |
| REP | Annualised replacement of finite-lifetime equipment and consumables |
| LC | Labour = C_FTE × f_FTE |
| CO₂Tax | Carbon cost = TaxRate_CO₂ × CO₂Rate × m_Ga,ann |

**Capital recovery factor (CRF):** AF = r(1+r)^n / ((1+r)^n − 1), with discount rate `r` and
plant lifetime `n` years (default: r=0.08, n=20).

**CapEx estimation:** Step-specific Lang factors (Peters & Timmerhaus 2004, Table 6-21) convert
purchased equipment costs to installed total capital.

Each process step is defined by a `StepSpec` dataclass with `constants`, `sources`,
`equipment`, `stream_basis`, `recovery_basis`, and `cost_basis`.

## Key Outputs

| Q (m³/d) | Route | LCO-Ga (EUR/kg) | Annual Ga (kg/yr) |
|----------|-------|-----------------|-------------------|
| 10 | IX | 576 | 96 |
| 10 | SX | 720 | 75 |
| 30 | IX | 255 | 289 |
| 30 | SX | 318 | 225 |

LCO-Ga = Levelised Cost of Gallium (total annual cost / annual gallium production).

## CO₂ Accounting Methodology

Two intentionally separate scopes are used in this model:

**1. CO₂ tax embedded in LCO-Ga** (`co2_tax_mode = "energy_only"`, default)
Prices only the direct electricity emissions of each process step at the EU ETS
shadow price (60 EUR/t CO₂).  Grid emission factor: 0.363 kg CO₂-eq/kWh
(UBA 2024 German grid average).  This scope is consistent with current EU ETS
coverage of electricity generators (not downstream chemical consumers) and avoids
double-counting.  All LCO-Ga values in the thesis and in the table above are
computed with this mode.

**2. Full-LCA carbon burden** (standalone metric, `co2_tax_mode = "full_lca"`)
Includes electricity *and* all characterised reagent/solvent/resin GWP flows
using ReCiPe 2016 Midpoint (H) factors from ecoinvent 3.5 cutoff
(see `lca/lca_recipe_factors.py`).  Used exclusively for the carbon-intensity
comparison figure and the kg CO₂-eq/kg Ga metric cited in the abstract.
This mode does **not** affect any cost output.

To reproduce the abstract carbon-burden values:

```python
from utils.carbon_burden import calc_carbon_burden_full_lca
calc_carbon_burden_full_lca(Q=10,  route='IX')  # ≈ 236 kg CO₂-eq/kg Ga
calc_carbon_burden_full_lca(Q=10,  route='SX')  # ≈ 420 kg CO₂-eq/kg Ga
calc_carbon_burden_full_lca(Q=100, route='IX')  # ≈  67 kg CO₂-eq/kg Ga
calc_carbon_burden_full_lca(Q=100, route='SX')  # ≈ 185 kg CO₂-eq/kg Ga
```


## Requirements

- Python ≥ 3.9
- NumPy, Matplotlib, Pandas, openpyxl

```bash
pip install -r requirements.txt
```

## Usage

### Basic calculations

```python
import tea_model_ga_thesis as tea

Q = 30.0  # m³/d feed flow rate

# Levelised cost of gallium for each route
lco_ix = tea.calc_lco_ga(Q, route='IX')
lco_sx = tea.calc_lco_ga(Q, route='SX')
print(f"IX: {lco_ix:.1f} EUR/kg   SX: {lco_sx:.1f} EUR/kg")

# Annual gallium production
prod_ix = tea.calc_annual_production(Q, route='IX')
print(f"Annual Ga (IX): {prod_ix:.1f} kg/yr")

# Full cost breakdown (CapEx, REP, OpEx, Labour, CO2 tax)
bd = tea.calc_cost_breakdown(Q, route='IX')
```

### Adjusting model parameters

All operating assumptions are centralised in `TEAConfig` which uses an immutable pattern.
Use `r` (discount rate) and `n` (plant lifetime in years) — not `discount_rate`/`plant_lifetime_years`:

```python
import tea_model_ga_thesis as tea

custom_config = tea.DEFAULT_CONFIG.replace(
    n=15,
    r=0.08,
)
lco = tea.calc_lco_ga(Q, route='IX', config=custom_config)
```

**Feed concentration:** The baseline Ga feed concentration (34.6 mg/L) is defined in
`FEED_BASELINE_TEMPLATE['species_mg_L']['Ga']`, not in TEAConfig. For sensitivity
(e.g. tornado, viability), use `config.replace(c_ga_feed_mg_L=25.0)`.

### Scenario analysis with helpers

```python
from utils.baseline_config import create_scenario_config, create_recovery_variant

# Create a custom scenario
s1_config = create_scenario_config(
    capex_multiplier=1.25,
    rep_multiplier=1.35,
    labour_multiplier=1.10,
)
lco_s1_ix = tea.calc_lco_ga(50, route='IX', config=s1_config)

# Create a variant with improved recovery
improved_config = create_recovery_variant(
    ix_ga_to_eluat=0.985,
    sx_ga_to_loaded_organic=0.82,
)
lco_improved = tea.calc_lco_ga(50, route='IX', config=improved_config)
```

## Reproducing thesis figures and tables

Run from the **repo root** so `import tea_model_ga_thesis` works. Outputs are written to
`outputs/figures/`, `outputs/csv/`, `outputs/tables/`, `outputs/notes/`. These
directories are created on first run.

### Generate all figures at once

```bash
python figures/generate_all_figures.py
```

Generates Figures 5.3–5.10:
- **5.3** Baseline: LCOGa + annual production vs Q, Q* markers
- **5.4** Cost contribution (stacked bar: CapEx, OpEx, REP, Labour, CO₂ tax)
- **5.5** Process-step contribution (IX vs SX at Q=10)
- **5.6** CapEx breakdown (Sep vs Other)
- **5.7** Carbon load comparison (full LCA by route and Q)
- **5.8** Tornado sensitivity (local parameter sensitivity at Q=50)
- **5.9** Uncertainty bands (P10/P50/P90 Monte Carlo)
- **5.10** Viability map (Q* vs feed concentration)

### Generate all tables at once

```bash
python tables/generate_all_tables.py
```

Generates Tables 5.7–5.9:
- **5.7** Scenario matrix (6 scenarios at Q = 50 m³/d)
- **5.8** Q* vs feed concentration (at market price 423 €/kg)
- **5.9** Q* vs market price (selected c_Ga and prices 300/423/700 €/kg)

## File Structure

```
tea_model_ga_thesis.py         Core TEA model (thesis version)
README.md                   This file
LICENSE                     MIT licence
requirements.txt            Python dependencies

figures/
├── fig_5_3_baseline.py      Figure 5.3: LCOGa + production vs Q
├── fig_5_4_cost_contribution.py
├── fig_5_5_process_step.py
├── fig_5_6_capex_breakdown.py
├── fig_5_7_carbon_load.py
├── fig_5_8_tornado.py       Figure 5.8: tornado sensitivity
├── fig_5_9_uncertainty.py   Figure 5.9: uncertainty bands
├── fig_5_10_viability.py    Figure 5.10: viability map
└── generate_all_figures.py  Master runner (5.3 → 5.10)

tables/
├── table_5_7_scenarios.py   Table 5.7: scenario matrix
├── table_5_8_qstar_concentration.py
├── table_5_9_qstar_market.py
└── generate_all_tables.py   Master runner (5.7 → 5.9)

utils/
├── baseline_config.py       V08 baseline configuration helpers
└── carbon_burden.py         Full-LCA and energy-only carbon burden calculator

lca/
├── lca_recipe_factors.py    ReCiPe 2016 Midpoint (H) GWP factors (ecoinvent 3.5)
└── lca_aggregation.py       Full-LCA GWP aggregation

outputs/                     Generated on first run of generate_all_figures/tables
├── figures/                 SVG figures
├── csv/                     CSV data exports
├── tables/                  Table CSVs and notes
└── notes/                   Analysis notes
```

## Citation

If you use this model in academic work, please cite:

> Kozerke, K. (2025). *Techno-Economic Analysis of Gallium Recovery from GaAs
> Semiconductor Manufacturing Wastewater* (Master's Thesis).
> University of Cambridge / WZL RWTH Aachen University.
> doi: [ADD DOI WHEN AVAILABLE]  — repository: [ADD REPOSITORY URL WHEN AVAILABLE]

## Licence

MIT — see [LICENSE](LICENSE).
