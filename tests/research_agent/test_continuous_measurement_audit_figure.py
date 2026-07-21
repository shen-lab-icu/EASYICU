"""Closed-contract tests for the three-table measurement-audit renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.figure_renderer import (
    _sealed_renderer_figure_step_matches_parent,
)
from easyicu.research_agent.figures.continuous_measurement_audit import (
    CONTROLLED_METHOD,
    REPAIR_ID,
    _continuous_measurement_audit_parent_digest_seal,
    prepare_continuous_measurement_audit_inputs,
    render_continuous_measurement_audit_bundle,
)
from easyicu.research_agent.pipeline import (
    _render_authorized_sealed_publication_bundle,
    deterministic_figure_repair_id_for_upstream,
)
from easyicu.research_agent.schema import AnalysisStep

PARENT_STEP = "03_marker_data_quality"
FIGURE_STEP = f"{PARENT_STEP}_figure"


def _write_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "steps" / PARENT_STEP / "outputs"
    parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "variable": "marker_peak",
                "metric": "distribution_summary",
                "unit": "units",
                "n": 8,
                "denominator": 10,
                "percentage": 80.0,
                "median": 2.5,
                "q25": 1.5,
                "q75": 4.0,
                "min": 0.5,
                "max": 8.0,
            },
            {
                "variable": "marker_peak",
                "metric": "observed_zero",
                "unit": "units",
                "n": 0,
                "denominator": 10,
                "percentage": 0.0,
                "median": None,
                "q25": None,
                "q75": None,
                "min": None,
                "max": None,
            },
        ]
    ).to_csv(parent / "marker_distribution.csv", index=False)
    pd.DataFrame(
        [
            {
                "variable": "marker_peak",
                "status": "authoritative value observed",
                "count": 8,
                "denominator": 10,
                "percentage": 80.0,
                "missing_n": 2,
                "missing_pct": 20.0,
            },
            {
                "variable": "marker_peak",
                "status": "authoritative value missing",
                "count": 2,
                "denominator": 10,
                "percentage": 20.0,
                "missing_n": 2,
                "missing_pct": 20.0,
            },
        ]
    ).to_csv(parent / "marker_missingness.csv", index=False)
    statuses = (
        ("valid observed", 8),
        ("no source", 2),
        ("measured/source present but summary missing", 0),
        ("contradictory/invalid", 0),
    )
    pd.DataFrame(
        [
            {
                "variable": "marker",
                "status": status,
                "count": count,
                "denominator": 10,
                "percentage": count * 10.0,
                "unit": "units",
                "measured_column": "marker_measured",
                "count_column": "marker_n",
                "summary_column": "marker_peak",
            }
            for status, count in statuses
        ]
    ).to_csv(parent / "marker_measurement_process.csv", index=False)
    summary = {
        "step": PARENT_STEP,
        "method": CONTROLLED_METHOD,
        "primary_exposure": "marker_peak",
        "unit": "units",
        "cohort_policy": {"final_cohort_n": 10},
        "output_files": {
            "table:marker_distribution": "marker_distribution.csv",
            "table:marker_missingness": "marker_missingness.csv",
            "table:marker_measurement_process": "marker_measurement_process.csv",
        },
    }
    (parent / "step_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    store = EvidenceStore(tmp_path)
    records = []
    for name in (
        "marker_distribution.csv",
        "marker_missingness.csv",
        "marker_measurement_process.csv",
    ):
        records.append(
            store.register_file(
                kind="table",
                description=name,
                source_path=parent / name,
                produced_by_step=PARENT_STEP,
                producer="coder",
                generation_mode="llm",
            )
        )
    summary_record = store.register_file(
        kind="statistic",
        description="Structured parent summary.",
        source_path=parent / "step_summary.json",
        produced_by_step=PARENT_STEP,
        producer="runner",
        generation_mode="llm",
    )
    records.append(summary_record)
    (tmp_path / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": PARENT_STEP,
                        "status": "ok",
                        "analysis_request": {
                            "step": {
                                "step_id": PARENT_STEP,
                                "method": CONTROLLED_METHOD,
                                "inputs": ["marker_peak"],
                                "expected_outputs": [
                                    "table:marker_distribution",
                                    "table:marker_missingness",
                                    "table:marker_measurement_process",
                                ],
                            }
                        },
                        "evidence_ids": [record.evidence_id for record in records],
                        "step_summary_evidence_id": summary_record.evidence_id,
                    }
                ],
                "evidence": [
                    record.model_dump(mode="json") for record in store.records()
                ],
            }
        ),
        encoding="utf-8",
    )
    return parent


def _figure_step() -> AnalysisStep:
    return AnalysisStep(
        step_id=FIGURE_STEP,
        intent="Render the three registered parent tables.",
        inputs=[
            "table:marker_distribution",
            "table:marker_missingness",
            "table:marker_measurement_process",
        ],
        expected_outputs=["figure:marker_distribution"],
        method="visualization",
    )


def test_closed_three_table_parent_routes_and_renders(tmp_path: Path) -> None:
    _write_parent(tmp_path)
    step = _figure_step()

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == (
        REPAIR_ID
    )
    assert _sealed_renderer_figure_step_matches_parent(tmp_path, step, REPAIR_ID)
    seal = _continuous_measurement_audit_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert set(seal or {}) == {
        "step_summary.json",
        "marker_distribution.csv",
        "marker_missingness.csv",
        "marker_measurement_process.csv",
    }

    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=REPAIR_ID,
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            parent_artifact_digests=seal or {},
        )
        == REPAIR_ID
    )
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (
            out / f"continuous_distribution_measurement_availability.{suffix}"
        ).is_file()
    rendered = json.loads((out / "step_summary.json").read_text())
    assert rendered["source_tables"] == [
        "marker_distribution.csv",
        "marker_missingness.csv",
        "marker_measurement_process.csv",
    ]


def test_renderer_rejects_summary_column_identity_swap(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    process = pd.read_csv(parent / "marker_measurement_process.csv")
    process["summary_column"] = "different_marker"
    process.to_csv(parent / "marker_measurement_process.csv", index=False)

    assert (
        _continuous_measurement_audit_parent_digest_seal(tmp_path, FIGURE_STEP) is None
    )
    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None


def test_renderer_requires_all_three_typed_child_edges(tmp_path: Path) -> None:
    _write_parent(tmp_path)
    incomplete = _figure_step().model_copy(
        update={
            "inputs": [
                "table:marker_distribution",
                "table:marker_missingness",
            ]
        }
    )

    assert not _sealed_renderer_figure_step_matches_parent(
        tmp_path, incomplete, REPAIR_ID
    )


def test_renderer_rejects_mutation_after_digest_seal(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    seal = _continuous_measurement_audit_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert seal is not None
    with (parent / "marker_missingness.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        render_continuous_measurement_audit_bundle(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            preverified_parent_artifacts={},
        )
        is None
    )
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=REPAIR_ID,
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            parent_artifact_digests=seal,
        )
        is None
    )


def test_input_validation_fails_closed_on_malformed_percentage(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    missingness = pd.read_csv(parent / "marker_missingness.csv")
    missingness.loc[0, "percentage"] = "not-a-number"
    missingness.to_csv(parent / "marker_missingness.csv", index=False)

    summary = json.loads((parent / "step_summary.json").read_text())
    names = (
        "marker_distribution.csv",
        "marker_missingness.csv",
        "marker_measurement_process.csv",
    )
    assert (
        prepare_continuous_measurement_audit_inputs(
            parent_out=parent,
            parent_summary=summary,
            planner_roles={
                "distribution": "marker_distribution",
                "missingness": "marker_missingness",
                "measurement_process": "marker_measurement_process",
            },
            preverified_table_bytes={
                name: (parent / name).read_bytes() for name in names
            },
        )
        is None
    )
