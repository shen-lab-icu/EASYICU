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


def _write_split_parent(
    run_dir: Path, *, mutation: str = "", with_audits: bool = False
) -> Path:
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
    if with_audits:
        missingness = pd.DataFrame(
            [
                {
                    "variable": variable,
                    "n_total": 104,
                    "n_nonmissing": 104 - missing_n,
                    "missing_n": missing_n,
                    "missing_pct": 100.0 * missing_n / 104,
                }
                for variable, missing_n in (
                    ("severity_band", 4),
                    ("secondary_measure", 24),
                    ("complete_measure", 0),
                )
            ]
        )
        structural = pd.DataFrame(
            [
                {
                    "variable": "secondary_measure",
                    "n_total": 104,
                    "missing_n": 24,
                    "missing_pct": 100.0 * 24 / 104,
                    "nonmissing_n": 80,
                    "nonmissing_unique_n": 2,
                    "structural_status": "partially_observed",
                }
            ]
        )
        missingness.to_csv(parent / "missingness_audit.csv", index=False)
        structural.to_csv(parent / "structural_missingness_audit.csv", index=False)
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
            **(
                {
                    "table:missingness_audit": "missingness_audit.csv",
                    "table:structural_missingness_audit": (
                        "structural_missingness_audit.csv"
                    ),
                }
                if with_audits
                else {}
            ),
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
    artifact_specs = [
        ("split_distribution", "table", "severity_distribution.csv"),
        ("split_availability", "table", "measurement_availability.csv"),
        ("split_summary", "statistic", "step_summary.json"),
    ]
    if with_audits:
        artifact_specs.extend(
            [
                ("split_missingness", "table", "missingness_audit.csv"),
                (
                    "split_structural_missingness",
                    "table",
                    "structural_missingness_audit.csv",
                ),
            ]
        )
    for evidence_id, kind, name in artifact_specs:
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
                                    *(
                                        [
                                            "table:missingness_audit",
                                            "table:structural_missingness_audit",
                                        ]
                                        if with_audits
                                        else []
                                    ),
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


def _child(*, with_audits: bool = False) -> AnalysisStep:
    return AnalysisStep(
        step_id=CHILD,
        intent="Render the Planner-owned availability panel.",
        inputs=[
            "table:severity_distribution",
            "table:measurement_availability",
            *(
                [
                    "table:missingness_audit",
                    "table:structural_missingness_audit",
                ]
                if with_audits
                else []
            ),
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


def test_v2_consumes_and_exports_every_declared_audit_table(tmp_path: Path) -> None:
    parent = _write_split_parent(tmp_path, with_audits=True)
    seal = _sealed_renderer_parent_digest_seal(tmp_path, CHILD, REPAIR_ID)
    assert set(seal or {}) == {
        "measurement_availability.csv",
        "missingness_audit.csv",
        "severity_distribution.csv",
        "step_summary.json",
        "structural_missingness_audit.csv",
    }
    assert _sealed_renderer_figure_step_matches_parent(
        tmp_path, _child(with_audits=True), REPAIR_ID
    )

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
    missingness_source = pd.read_csv(out / "missingness_audit_source_data.csv")
    structural_source = pd.read_csv(
        out / "structural_missingness_audit_source_data.csv"
    )
    assert missingness_source["source_table"].unique().tolist() == [
        "missingness_audit.csv"
    ]
    assert structural_source["source_table"].unique().tolist() == [
        "structural_missingness_audit.csv"
    ]
    assert missingness_source["source_step_id"].unique().tolist() == [PARENT]
    assert structural_source["source_step_id"].unique().tolist() == [PARENT]
    assert (
        missingness_source["variable"].tolist()
        == pd.read_csv(parent / "missingness_audit.csv")["variable"].tolist()
    )
    assert (
        structural_source["variable"].tolist()
        == pd.read_csv(parent / "structural_missingness_audit.csv")["variable"].tolist()
    )
    contract = json.loads(
        (out / "severity_distribution.figure_contract.json").read_text(encoding="utf-8")
    )
    assert {panel["panel_id"] for panel in contract["panels"]} == {"A", "B", "C"}
    assert set(contract["source_data"]) == {
        "severity_distribution_source_data.csv",
        "missingness_audit_source_data.csv",
        "structural_missingness_audit_source_data.csv",
    }
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert set(summary["source_tables"]) == {
        "severity_distribution.csv",
        "measurement_availability.csv",
        "missingness_audit.csv",
        "structural_missingness_audit.csv",
    }
    assert summary["denominator_contract"]["panel_c"] == "locked_analysis_cohort"


def test_v2_incomplete_or_malformed_audit_pair_fails_closed(tmp_path: Path) -> None:
    parent = _write_split_parent(tmp_path, with_audits=True)
    summary_path = parent / "step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_files"].pop("table:structural_missingness_audit")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert deterministic_figure_repair_id_for_upstream(tmp_path, CHILD) is None

    malformed = tmp_path / "malformed"
    parent = _write_split_parent(malformed, with_audits=True)
    missingness_path = parent / "missingness_audit.csv"
    missingness = pd.read_csv(missingness_path)
    missingness.loc[0, "n_total"] = 103
    missingness.to_csv(missingness_path, index=False)
    assert deterministic_figure_repair_id_for_upstream(malformed, CHILD) is None


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
    extra_input = _child().model_copy(
        update={
            "inputs": [
                "table:severity_distribution",
                "table:measurement_availability",
                "table:unconsumed_audit",
            ]
        }
    )
    assert not _sealed_renderer_figure_step_matches_parent(
        tmp_path, extra_input, REPAIR_ID
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
