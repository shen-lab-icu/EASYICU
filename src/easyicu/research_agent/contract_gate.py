"""Compatibility alias for :mod:`easyicu.research_agent.gates.contract`.

The module-object alias preserves collaborator monkeypatches made through the
legacy path; a star-import forwarding façade would not.
"""

from __future__ import annotations

import sys as _sys

from .gates import contract as _canonical
from .gates.contract import (
    _AGENT_OWNED_ROBUSTNESS_RESULT_METHODS,
    _AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS,
    _AUXILIARY_OUTPUT_KINDS,
    _authoritative_primary_robustness_contract,
    _closed_auxiliary_output_products,
    _cohort_definition_sensitivity_contract_findings,
    _declared_sensitivity_csv_paths,
    _is_cohort_definition_sensitivity_result_step,
    _method_head,
    _nonnegative_integral_value,
    _post_canonicalization_figure_findings,
    _read_locked_robustness_spec_dicts,
    _sensitivity_csv_rows,
    _step_deterministic_contract_findings,
)

__all__ = [
    "_step_deterministic_contract_findings",
    "_post_canonicalization_figure_findings",
    "_read_locked_robustness_spec_dicts",
    "_is_cohort_definition_sensitivity_result_step",
    "_authoritative_primary_robustness_contract",
    "_cohort_definition_sensitivity_contract_findings",
    "_closed_auxiliary_output_products",
    "_method_head",
    "_nonnegative_integral_value",
    "_declared_sensitivity_csv_paths",
    "_sensitivity_csv_rows",
    "_AGENT_OWNED_ROBUSTNESS_RESULT_METHODS",
    "_AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS",
    "_AUXILIARY_OUTPUT_KINDS",
]

_sys.modules[__name__] = _canonical
