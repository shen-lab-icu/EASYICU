"""Compatibility alias for :mod:`easyicu.research_agent.execution.figure_preparation`."""

from __future__ import annotations

import sys as _sys

from .execution import figure_preparation as _canonical
from .execution.figure_preparation import (
    _ensure_step_figure_contract,
    _family_has_deterministic_figure_renderer,
    _figure_contract_source_data_canonicalization_candidate,
    _infer_step_figure_panel_role,
    _install_figure_contract_source_data_canonicalization,
    _reader_label_from_stem,
    _step_has_figure_only_output_contract,
    _step_summary_paths,
)

__all__ = [
    "_step_has_figure_only_output_contract",
    "_reader_label_from_stem",
    "_infer_step_figure_panel_role",
    "_step_summary_paths",
    "_ensure_step_figure_contract",
    "_figure_contract_source_data_canonicalization_candidate",
    "_install_figure_contract_source_data_canonicalization",
    "_family_has_deterministic_figure_renderer",
]

_sys.modules[__name__] = _canonical
