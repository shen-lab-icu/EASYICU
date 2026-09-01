"""A parent input must never be promoted to a figure's analytic source."""

from __future__ import annotations

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import deterministic_contract_repair


_PARENT_COPY_SCRIPT = '''
from pathlib import Path
import pandas as pd

OUT_DIR = Path(".")
parent = pd.DataFrame({"row_id": ["a"], "estimate": [1.25]})
source_name = "overview_source_data.csv"
parent.to_csv(OUT_DIR / source_name, index=False)

def make_figure_contract(**kwargs):
    return kwargs

contract = make_figure_contract(
    figure_id="figure:overview",
    source_data=source_name,
)
'''.lstrip()


def test_parent_table_is_not_a_repairable_panel_analytic_source() -> None:
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {
            "reason": "incomplete_source_lineage_coverage",
            "missing_bound_tables": ["parent.csv"],
            "missing_bound_statistics": [],
        },
    }

    assert (
        deterministic_contract_repair(code=_PARENT_COPY_SCRIPT, findings=[finding])
        is None
    )


def test_missing_panel_source_data_is_not_reconstructed_from_dataframe_objects() -> None:
    code = _PARENT_COPY_SCRIPT.replace(
        'source_data=source_name,',
        'source_data={"parent": parent.copy(deep=True)},',
    )
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {"reason": "missing_source_data"},
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None


def test_retired_parent_projection_repairs_are_automatically_denied() -> None:
    for repair_id in (
        "bound_figure_source_projection_v1",
        "bound_figure_source_projection_v2",
        "complete_bound_figure_source_bundle_v1",
        "direct_bound_figure_source_materialization_v1",
        "direct_bound_figure_source_projection_v1",
        "unavailable_figure_full_source_projection_v1",
    ):
        metadata = repair_metadata_for(repair_id)
        assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
        assert metadata.classification_source == "fallback:unknown_method_substitution"
        assert automatic_repair_allowed(repair_id) is False
