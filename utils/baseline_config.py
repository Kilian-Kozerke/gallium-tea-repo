"""V08 baseline configuration helpers for reproducible TEA model runs.

This module provides helpers to work with the immutable TEAConfig system
introduced in V08. The DEFAULT_CONFIG from tea_model_ga_thesis is frozen
and cannot be mutated in-place.

To create variants, use TEAConfig.replace() which returns a new instance
with specified fields updated and all derived fields recomputed.
"""

import sys
from pathlib import Path
from typing import Any, Mapping, Optional

# Import the tea model
import tea_model_ga_thesis as tea


# Export the baseline config from tea_model_ga_thesis
THESIS_BASELINE_CONFIG = tea.DEFAULT_CONFIG


def apply_config(config: Optional[tea.TEAConfig] = None) -> tea.TEAConfig:
    """
    Get a TEAConfig instance for use in model calculations.

    Args:
        config: TEAConfig instance. If None, uses THESIS_BASELINE_CONFIG.

    Returns:
        The TEAConfig instance to pass to calc_* functions.
    """
    return config or THESIS_BASELINE_CONFIG


def create_scenario_config(
    base_config: Optional[tea.TEAConfig] = None,
    **changes: Any
) -> tea.TEAConfig:
    """
    Create a new TEAConfig with specified changes applied.

    All derived fields (operating_days, CRF, recovery rates) are
    automatically recomputed.

    Args:
        base_config: TEAConfig to use as base. Defaults to THESIS_BASELINE_CONFIG.
        **changes: Field names and values to replace.

    Returns:
        A new TEAConfig instance with specified changes.

    Example:
        >>> custom = create_scenario_config(
        ...     r=0.08,
        ...     n=15,
        ...     capex_multiplier=1.25
        ... )
        >>> lco = tea.calc_lco_ga(50, route='IX', config=custom)
    """
    base = base_config or THESIS_BASELINE_CONFIG
    return base.replace(**changes)


def create_recovery_variant(
    base_config: Optional[tea.TEAConfig] = None,
    **recovery_changes: Any
) -> tea.TEAConfig:
    """
    Create a new TEAConfig with modified recovery rates.

    Args:
        base_config: TEAConfig to use as base. Defaults to THESIS_BASELINE_CONFIG.
        **recovery_changes: Recovery names and values to replace.

    Returns:
        A new TEAConfig instance with specified recovery rates changed.

    Example:
        >>> improved = create_recovery_variant(
        ...     ix_ga_to_eluat=0.985,
        ...     sx_ga_to_loaded_organic=0.82,
        ... )
    """
    base = base_config or THESIS_BASELINE_CONFIG

    # Create new recoveries dict with updates
    recoveries_dict = dict(base.recoveries)

    for key, value in recovery_changes.items():
        recoveries_dict[key] = value

    # Create new config with updated recoveries
    from types import MappingProxyType
    return base.replace(recoveries=MappingProxyType(recoveries_dict))
