"""Additive v2 ordered-distribution + availability renderer contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.pipeline import (
    _render_authorized_sealed_publication_bundle,
    _sealed_renderer_figure_step_matches_parent,
    _sealed_renderer_parent_digest_seal,
    deterministic_figure_repair_id_for_upstream,
)
from easyicu.research_agent.schema import AnalysisStep

PARENT = "04_quality_protocol"
CHILD = f"{PARENT}_figure"
REPAIR_ID = "ordered_category_distribution_availability_publication_bundle_v2"


def _write_split_parent(run_dir: Path, *, mutation: str = "") -> Path:
    parent = run_dir / "steps" / PARENT / "outputs"
    parent.mkdir(parents=True)
    distribution = pd.DataFrame(
        [
            {
                "row_kind": "band",
                "variable": "severity_band",
                "level": level,
                "label": label,
                "count": count,
                "percentage": 100.0 * count / 100,
                "denominator": 100,
            }
            for level, label, count in (
                (0, "Low band", 60),
                (1, "Middle band", 30),
                (2, "High band", 10),
            )
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "concept": "severity",
                "status": status,
                "count": count,
                "percentage_of_locked_cohort": 100.0 * count / 104,
                "denominator": 104,
                "value_column": "severity_band",
            }
            for status, count in (
                ("valid observed", 100),
                ("no source", 4),
                ("source present but summary missing", 0),
                ("contradictory or invalid", 0),
            )
        ]
        + [
            {
                "concept": "unrelated",
                "status": "valid observed",
                "count": 80,
                "percentage_of_locked_cohort": 100.0,
                "denominator": 80,
                "value_column": "another_measure",
            }
        ]
    )
    distribution.to_csv(parent / "severity_distribution.csv", index=False)
    availability.to_csv(parent / "measurement_availability.csv", index=False)
    summary = {
        "method": "agent_selected_quality_protocol",
        "analysis_family": "association_study",
        "locked_cohort": {"n_rows": 104},
        "primary_exposure": {
            "variable": "severity_band",
            "scale": "ordinal",
            "declared_levels": [0, 1, 2],
            "missing_n": 4,
        },
        "output_files": {
            "table:severity_distribution": "severity_distribution.csv",
            "table:measurement_availability": "measurement_availability.csv",
        },
    }
    if mutation == "nonordinal":
        summary["primary_exposure"]["scale"] = "continuous"
    elif mutation == "level_mismatch":
        distribution.loc[distribution["level"].eq(2), "level"] = 3
    elif mutation == "distribution_denominator_mismatch":
        distribution.loc[0, "denominator"] = 99
    elif mutation == "availability_count_mismatch":
        availability.loc[
            availability["status"].eq("valid observed")
            & availability["value_column"].eq("severity_band"),
            "count",
        ] = 99
    elif mutation == "availability_denominator_mismatch":
        availability.loc[0, "denominator"] = 103
    elif mutation == "availability_percentage_missing":
        availability = availability.drop(columns=["percentage_of_locked_cohort"])
    elif mutation == "wrong_output_binding":
        summary["output_files"][
            "table:measurement_availability"
        ] = "severity_distribution.csv"
    elif mutation:
        raise AssertionError(f"Unknown fixture mutation: {mutation}")
    distribution.to_csv(parent / "severity_distribution.csv", index=False)
    availability.to_csv(parent / "measurement_availability.csv", index=False)
    (parent / "step_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    evidence = EvidenceStore(run_dir)
    records = []
    for evidence_id, kind, name in (
        ("split_distribution", "table", "severity_distribution.csv"),
        ("split_availability", "table", "measurement_availability.csv"),
        ("split_summary", "statistic", "step_summary.json"),
    ):
        records.append(
            evidence.register_file(
                kind=kind,
                description=f"Verified parent {name}.",
                source_path=parent / name,
                evidence_id=evidence_id,
                produced_by_step=PARENT,
                producer="coder" if kind == "table" else "runner",
                generation_mode="llm",
            )
        )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": PARENT,
                        "status": "ok",
                        "analysis_request": {
                            "step": {
                                "step_id": PARENT,
                                "method": "agent_selected_quality_protocol",
                                "expected_outputs": [
                                    "table:severity_distribution",
                                    "table:measurement_availability",
                                ],
                            },
                            "analysis_family": "association_study",
                        },
                        "evidence_ids": [record.evidence_id for record in records],
                        "step_summary_evidence_id": "split_summary",
                    }
                ],
                "evidence": [
                    record.model_dump(mode="json") for record in evidence.records()
                ],
            }
        ),
        encoding="utf-8",
    )
    return parent


def _child() -> AnalysisStep:
    return AnalysisStep(
        step_id=CHILD,
        intent="Render the Planner-owned availability panel.",
        inputs=[
            "table:severity_distribution",
            "table:measurement_availability",
        ],
        expected_outputs=["figure:measurement_availability_panel"],
        method="visualization",
    )


def test_v2_routes_by_typed_roles_and_digest_bound_schema_not_method(
    tmp_path: Path,
) -> None:
    _write_split_parent(tmp_path)
    assert deterministic_figure_repair_id_for_upstream(tmp_path, CHILD) == REPAIR_ID
    seal = _sealed_renderer_parent_digest_seal(tmp_path, CHILD, REPAIR_ID)
    assert set(seal or {}) == {
        "measurement_availability.csv",
        "severity_distribution.csv",
        "step_summary.json",
    }
    assert _sealed_renderer_figure_step_matches_parent(tmp_path, _child(), REPAIR_ID)

    out = tmp_path / "steps" / CHILD / "outputs"
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=REPAIR_ID,
            run_dir=tmp_path,
            current_step_id=CHILD,
            out_dir=out,
            parent_artifact_digests=seal or {},
        )
        == REPAIR_ID
    )
    source = pd.read_csv(out / "severity_distribution_source_data.csv")
    assert set(source.loc[source["panel_id"].eq("A"), "source_table"]) == {
        "severity_distribution.csv"
    }
    assert set(source.loc[source["panel_id"].eq("B"), "source_table"]) == {
        "measurement_availability.csv"
    }
    assert source.groupby("panel_id")["percentage"].sum().to_dict() == pytest.approx(
        {"A": 100.0, "B": 100.0}
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "nonordinal",
        "level_mismatch",
        "distribution_denominator_mismatch",
        "availability_count_mismatch",
        "availability_denominator_mismatch",
        "availability_percentage_missing",
        "wrong_output_binding",
    ),
)
def test_v2_schema_and_identity_ambiguity_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    _write_split_parent(tmp_path, mutation=mutation)
    assert deterministic_figure_repair_id_for_upstream(tmp_path, CHILD) is None


def test_v2_requires_exact_child_inputs_and_rejects_result_role(tmp_path: Path) -> None:
    _write_split_parent(tmp_path)
    missing_input = _child().model_copy(
        update={"inputs": ["table:severity_distribution"]}
    )
    assert not _sealed_renderer_figure_step_matches_parent(
        tmp_path, missing_input, REPAIR_ID
    )

    from easyicu.research_agent.contracts.declared_product import (
        authorize_declared_figure_product_slots,
    )

    with pytest.raises(ValueError):
        authorize_declared_figure_product_slots(
            declared_products=["figure:adjusted_effect"],
            renderer_repair_id=REPAIR_ID,
            planner_parent_anchors=[
                "table:severity_distribution",
                "table:measurement_availability",
            ],
        )


def test_v2_digest_tampering_is_rejected(tmp_path: Path) -> None:
    parent = _write_split_parent(tmp_path)
    seal = _sealed_renderer_parent_digest_seal(tmp_path, CHILD, REPAIR_ID)
    assert seal is not None
    path = parent / "measurement_availability.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=REPAIR_ID,
            run_dir=tmp_path,
            current_step_id=CHILD,
            out_dir=tmp_path / "tampered",
            parent_artifact_digests=seal,
        )
        is None
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() != seal[path.name]
