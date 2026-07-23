"""Regression tests for exact per-parent figure source-data projections."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import deterministic_contract_repair

_FINDING = {
    "validator": "figure_source_data",
    "severity": "error",
    "detail": {
        "reason": "incomplete_source_lineage_coverage",
        "missing_bound_tables": ["parent_a.csv", "parent_b.csv"],
        "missing_bound_statistics": [],
    },
}


def _script(out_dir: Path) -> str:
    return f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(out_dir)!r})
EXPECTED_INPUTS = ["table:parent_a", "table:parent_b"]

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def load_bound_table(input_key):
    frame = (
        pd.DataFrame({{"row_id": ["a"], "estimate": [1.25]}})
        if input_key.endswith("parent_a")
        else pd.DataFrame({{"row_id": ["b"], "count": [17]}})
    )
    return frame, {{"input_key": input_key}}, {{"product": input_key}}

def main():
    tables = {{}}
    input_bindings = []
    input_records = {{}}
    for input_key in EXPECTED_INPUTS:
        frame, binding, record = load_bound_table(input_key)
        tables[input_key] = frame
        input_bindings.append(binding)
        input_records[input_key] = record

    mixed = pd.DataFrame({{"source_row_index": [0], "value": [1.25]}})
    source_stem = "overview_source_data.csv"
    mixed.to_csv(OUT_DIR / source_stem, index=False)

    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data=source_stem,
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    step_summary = {{
        "figure_files": ["overview.png"],
        "source_data_files": [source_stem],
    }}
    return contract, step_summary
"""


def test_bound_figure_repair_projects_each_loaded_parent_without_renaming(
    tmp_path: Path,
) -> None:
    code = _script(tmp_path)

    repair = deterministic_contract_repair(code=code, findings=[_FINDING])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract, summary = namespace["main"]()

    declared = contract["source_data"]
    assert declared == summary["source_data_files"]
    assert len(declared) == 2
    assert "overview_source_data.csv" not in declared
    observed = [pd.read_csv(tmp_path / filename) for filename in declared]
    assert observed[0].to_dict(orient="records") == [{"row_id": "a", "estimate": 1.25}]
    assert observed[1].to_dict(orient="records") == [{"row_id": "b", "count": 17}]


def test_bound_figure_repair_requires_unambiguous_loader_and_bundle_contract(
    tmp_path: Path,
) -> None:
    code = _script(tmp_path).replace(
        "        tables[input_key] = frame\n",
        "        tables[input_key] = frame\n"
        "        shadow_tables[input_key] = frame\n",
    )

    assert deterministic_contract_repair(code=code, findings=[_FINDING]) is None
    assert (
        deterministic_contract_repair(
            code=_script(tmp_path),
            findings=[
                {
                    **_FINDING,
                    "detail": {
                        **_FINDING["detail"],
                        "missing_bound_statistics": ["statistic:primary_effect"],
                    },
                }
            ],
        )
        is None
    )


def test_bound_figure_source_projection_is_registered_structural() -> None:
    metadata = repair_metadata_for("bound_figure_source_projection_v1")

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)
