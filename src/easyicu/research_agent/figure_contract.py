"""Compatibility shim for older/generated figure-contract imports.

Agent-generated code in older runs may still import
``easyicu.research_agent.figure_contract`` even though the canonical
implementation now lives in :mod:`easyicu.research_agent.publication_figures`.
Keep this module tiny and explicit so those runs continue to work.
"""

from __future__ import annotations

from .publication_figures import (
    FigureContract,
    PanelSpec,
    add_panel_label,
    apply_publication_style,
    audit_figure_contract,
    audit_publication_exports,
    make_figure_contract,
    save_publication_figure,
)

__all__ = [
    "FigureContract",
    "PanelSpec",
    "make_figure_contract",
    "audit_figure_contract",
    "apply_publication_style",
    "add_panel_label",
    "save_publication_figure",
    "audit_publication_exports",
]
