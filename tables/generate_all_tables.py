#!/usr/bin/env python3
"""
Master runner for all thesis tables.

Generates:
  - Table 5.7 (scenario matrix)
  - Table 5.8 (Q* vs concentration)
  - Table 5.9 (Q* vs market price)

All outputs are written to outputs/tables/
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import table generation modules
import table_5_7_scenarios
import table_5_8_qstar_concentration
import table_5_9_qstar_market


def main():
    print("=" * 80)
    print("Generating thesis tables...")
    print("=" * 80)
    print()

    for name, mod in [
        ("5.7: Scenario matrix", table_5_7_scenarios),
        ("5.8: Q* vs concentration", table_5_8_qstar_concentration),
        ("5.9: Q* vs market price", table_5_9_qstar_market),
    ]:
        print(f"Table {name}")
        try:
            mod.main()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()

    print("=" * 80)
    print("All tables generated successfully!")
    print("Outputs: outputs/tables/")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
