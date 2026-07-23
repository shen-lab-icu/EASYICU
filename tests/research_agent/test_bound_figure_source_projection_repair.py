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
    source_name = "parent_a.csv" if input_key.endswith("parent_a") else "parent_b.csv"
    return (
        frame,
        {{"input_key": input_key}},
        {{"product": input_key, "relative_path": f"evidence/{{source_name}}"}},
    )

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
    assert repair_id == "bound_figure_source_projection_v2"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract, summary = namespace["main"]()

    declared = contract["source_data"]
    assert declared == summary["source_data_files"]
    assert len(declared) == 2
    assert "overview_source_data.csv" not in declared
    observed = [pd.read_csv(tmp_path / filename) for filename in declared]
    assert observed[0].to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "parent_a.csv",
            "row_id": "a",
            "estimate": 1.25,
        }
    ]
    assert observed[1].to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "parent_b.csv",
            "row_id": "b",
            "count": 17,
        }
    ]


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


def test_direct_bound_figure_repair_projects_tables_that_also_bind_statistics(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def _export_figure_source_data(source_data):
    if isinstance(source_data, (list, tuple)) and all(
        isinstance(item, str) for item in source_data
    ):
        return list(source_data)
    exported = []
    for key, frame in source_data.items():
        name = f"{{key}}_figure_source.csv"
        frame.to_csv(OUT_DIR / name, index=False)
        exported.append(name)
    return exported

def make_figure_contract(**kwargs):
    kwargs["source_data"] = _export_figure_source_data(kwargs["source_data"])
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    loaded = {{
        "table:parent_a": (
            pd.DataFrame({{"row_id": ["a"], "primary_or": [1.25]}}),
            Path("parent_a.csv"),
        ),
        "table:parent_b": (
            pd.DataFrame({{"row_id": ["b"], "complete_case_n": [17]}}),
            Path("parent_b.csv"),
        ),
    }}
    parent_a = loaded["table:parent_a"][0].copy()
    parent_b = loaded["table:parent_b"][0].copy()
    mixed = pd.DataFrame({{"panel": ["A", "B"], "value": [1.25, 17]}})
    mixed.to_csv(OUT_DIR / "overview_source_data.csv", index=False)
    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data="overview_source_data.csv",
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    return contract
"""
    finding = {
        **_FINDING,
        "detail": {
            **_FINDING["detail"],
            "missing_bound_statistics": [
                "statistic:primary_or",
                "statistic:complete_case_n",
            ],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_parent_a_source_data.csv",
        "bound_001_parent_b_source_data.csv",
    ]
    assert pd.read_csv(tmp_path / "bound_000_parent_a_source_data.csv").to_dict(
        orient="records"
    ) == [
        {
            "row_id": "a",
            "primary_or": 1.25,
            "source_row_index": 0,
            "source_table": "parent_a.csv",
        }
    ]
    assert pd.read_csv(tmp_path / "bound_001_parent_b_source_data.csv").to_dict(
        orient="records"
    ) == [
        {
            "row_id": "b",
            "complete_case_n": 17,
            "source_row_index": 0,
            "source_table": "parent_b.csv",
        }
    ]


def test_direct_bound_figure_repair_materializes_prior_dataframe_dict_shape(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    parent_a = pd.DataFrame({{"row_id": ["a"], "primary_or": [1.25]}})
    parent_b = pd.DataFrame({{"row_id": ["b"], "complete_case_n": [17]}})
    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data={{
            "parent_a": parent_a.copy(deep=True).assign(
                source_row_index=range(len(parent_a)),
                source_table="parent_a.csv",
            ),
            "parent_b": parent_b.copy(deep=True).assign(
                source_row_index=range(len(parent_b)),
                source_table="parent_b.csv",
            ),
        }},
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    return contract
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {"reason": "missing_source_data"},
    }

    repair = deterministic_contract_repair(
        code=code,
        findings=[finding],
        previous_repair="direct_bound_figure_source_projection_v1",
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_materialization_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_parent_a_source_data.csv",
        "bound_001_parent_b_source_data.csv",
    ]
    assert (tmp_path / contract["source_data"][0]).is_file()
    assert (tmp_path / contract["source_data"][1]).is_file()


def test_bound_figure_source_projection_is_registered_structural() -> None:
    for repair_id in (
        "bound_figure_source_projection_v2",
        "direct_bound_figure_source_materialization_v1",
        "direct_bound_figure_source_projection_v1",
    ):
        metadata = repair_metadata_for(repair_id)
        assert metadata.classification_source == "exact"
        assert metadata.repair_class is RepairClass.STRUCTURAL
        assert metadata.introduces_numbers is False
        assert automatic_repair_allowed(metadata.repair_id)
