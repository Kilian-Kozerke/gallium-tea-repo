#!/usr/bin/env python3
"""
Master runner for all thesis figures.

Generates:
  - Figure 5.3 (baseline: LCOGa + production vs Q)
  - Figure 5.4 (cost contribution)
  - Figure 5.5 (process step contribution)
  - Figure 5.6 (CapEx breakdown)
  - Figure 5.7 (carbon load comparison)
  - Figure 5.8 (tornado sensitivity)
  - Figure 5.9 (uncertainty bands)
  - Figure 5.10 (viability map)

All outputs are written to outputs/figures/, outputs/csv/, outputs/notes/
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import figure generation modules
import fig_5_3_baseline
import fig_5_4_cost_contribution
import fig_5_5_process_step
import fig_5_6_capex_breakdown
import fig_5_7_carbon_load
import fig_5_8_tornado
import fig_5_9_uncertainty
import fig_5_10_viability


def main():
    print("=" * 80)
    print("Generating thesis figures...")
    print("=" * 80)
    print()

    for name, mod in [
        ("5.3: Baseline (LCOGa + production)", fig_5_3_baseline),
        ("5.4: Cost contribution", fig_5_4_cost_contribution),
        ("5.5: Process step contribution", fig_5_5_process_step),
        ("5.6: CapEx breakdown", fig_5_6_capex_breakdown),
        ("5.7: Carbon load comparison", fig_5_7_carbon_load),
        ("5.8: Tornado sensitivity", fig_5_8_tornado),
        ("5.9: Uncertainty bands", fig_5_9_uncertainty),
        ("5.10: Viability map", fig_5_10_viability),
    ]:
        print(f"Figure {name}")
        try:
            mod.main()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()

    print("=" * 80)
    print("All figures generated successfully!")
    print("Outputs: outputs/figures/, outputs/csv/, outputs/notes/")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
