"""Closed-contract tests for the incidence/prevalence figure renderer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.declared_product_contract import (
    authorize_declared_figure_product_slots,
    bind_declared_figure_products,
)
from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.figures.absolute_risk import (
    REPAIR_ID,
    prepare_absolute_risk_inputs,
    render_absolute_risk_bundle_from_prior_outputs,
)
from easyicu.research_agent.pipeline import (
    _absolute_risk_parent_digest_seal,
    _render_authorized_sealed_publication_bundle,
    _sealed_renderer_figure_step_matches_parent,
    deterministic_figure_repair_id_for_upstream,
)
from easyicu.research_agent.pipeline_execute import _sealed_parent_planner_anchors
from easyicu.research_agent.schema import AnalysisStep


PARENT_STEP = "04_incidence_and_absolute_risk"
FIGURE_STEP = f"{PARENT_STEP}_figure"
CONTROLLED_METHOD = "binary_outcome_incidence_and_absolute_risk"
SOURCE_STATUSES = (
    "valid observed",
    "no source",
    "measured/source present but summary missing",
    "contradictory/invalid",
)


def _wilson(event_n: int, denominator_n: int) -> tuple[float, float]:
    if denominator_n == 0:
        return (math.nan, math.nan)
    estimate = event_n / denominator_n
    z = 1.959963984540054
    denominator = 1.0 + z**2 / denominator_n
    centre = (estimate + z**2 / (2 * denominator_n)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / denominator_n
            + z**2 / (4 * denominator_n**2)
        )
        / denominator
    )
    return (centre - half_width, centre + half_width)


def _prevalence_row(
    *,
    group_type: str,
    group_value: str,
    n: int,
    event_n: int,
    denominator_n: int,
    source_status: str,
) -> dict[str, object]:
    fraction = n / denominator_n if denominator_n else math.nan
    return {
        "estimate_type": (
            "source_status_prevalence"
            if group_type == "source_status"
            else f"{group_type}_prevalence"
        ),
        "variable": (
            "marker_measurement_status"
            if group_type == "source_status"
            else group_type
        ),
        "group_type": group_type,
        "group_value": group_value,
        "group_label": group_value,
        "source_status": source_status,
        "denominator_type": (
            "locked_cohort" if group_type == "source_status" else "valid_observed"
        ),
        "denominator_n": denominator_n,
        "n": n,
        "event_n": event_n,
        "non_event_n": n - event_n,
        "percentage_of_denominator": 100.0 * fraction if n else math.nan,
        "fraction_of_denominator": fraction if n else math.nan,
        "percentage_of_locked_cohort": float(n),
        "fraction_of_locked_cohort": n / 100.0,
        "group_definition": (
            f"Pre-specified marker_value stratum: {group_value}."
            if group_type == "marker_group"
            else f"Pre-specified {group_type} stratum: {group_value}."
        ),
        "summary_status": "available",
    }


def _risk_row(
    *,
    group_type: str,
    group_value: str,
    n: int,
    event_n: int,
    source_status: str | None,
) -> dict[str, object]:
    risk = event_n / n if n else math.nan
    ci_low, ci_high = _wilson(event_n, n)
    return {
        "estimate_type": "outcome_risk",
        "outcome": "response_flag",
        "outcome_definition": "Pre-specified binary response in the locked cohort.",
        "group_type": group_type,
        "group_value": group_value,
        "source_status": source_status,
        "group_definition": (
            f"Pre-specified marker_value stratum: {group_value}."
            if group_type == "marker_group"
            else f"Pre-specified {group_type} stratum: {group_value}."
        ),
        "stratum_n": n,
        "denominator_type": "valid_response_within_stratum",
        "denominator_n": n,
        "n": n,
        "event_n": event_n,
        "non_event_n": n - event_n,
        "outcome_risk": risk,
        "outcome_risk_percentage": 100.0 * risk if n else math.nan,
        "outcome_risk_fraction": risk,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": "wilson_score" if n else None,
        "ci_alpha": 0.05 if n else math.nan,
        "percentage_of_locked_cohort": float(n) if n else math.nan,
        "fraction_of_locked_cohort": n / 100.0 if n else math.nan,
        "risk_status": "available" if n else "not_estimable_zero_denominator",
        "notes": "Unadjusted absolute response risk.",
    }


def _base_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    status_counts = {
        "valid observed": (60, 12),
        "no source": (40, 4),
        "measured/source present but summary missing": (0, 0),
        "contradictory/invalid": (0, 0),
    }
    prevalence = pd.DataFrame(
        [
            _prevalence_row(
                group_type="source_status",
                group_value=status,
                n=n,
                event_n=event_n,
                denominator_n=100,
                source_status=status,
            )
            for status, (n, event_n) in status_counts.items()
        ]
        + [
            {
                **_prevalence_row(
                    group_type="valid_observed_marker",
                    group_value="valid observed",
                    n=60,
                    event_n=12,
                    denominator_n=60,
                    source_status="valid observed",
                ),
                "estimate_type": "valid_observed_marker_distribution",
            }
        ]
        + [
            _prevalence_row(
                group_type="marker_group",
                group_value=label,
                n=n,
                event_n=event_n,
                denominator_n=60,
                source_status="valid observed",
            )
            for label, n, event_n in (
                ("lower", 30, 3),
                ("middle", 20, 5),
                ("upper", 10, 4),
            )
        ]
    )
    incidence = pd.DataFrame(
        [
            _risk_row(
                group_type="overall",
                group_value="overall",
                n=100,
                event_n=16,
                source_status=None,
            )
        ]
        + [
            _risk_row(
                group_type="source_status",
                group_value=status,
                n=n,
                event_n=event_n,
                source_status=status,
            )
            for status, (n, event_n) in status_counts.items()
        ]
        + [
            _risk_row(
                group_type="marker_group",
                group_value=label,
                n=n,
                event_n=event_n,
                source_status="valid observed",
            )
            for label, n, event_n in (
                ("lower", 30, 3),
                ("middle", 20, 5),
                ("upper", 10, 4),
            )
        ]
    )
    return prevalence, incidence


def _apply_schema_mutation(
    prevalence: pd.DataFrame,
    incidence: pd.DataFrame,
    mutation: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mutation == "unknown_status":
        for frame in (prevalence, incidence):
            mask = frame["source_status"].eq("contradictory/invalid")
            frame.loc[mask, ["group_value", "source_status"]] = "unrecognized"
    elif mutation == "missing_status":
        prevalence = prevalence.loc[
            ~prevalence["source_status"].eq("contradictory/invalid")
        ].reset_index(drop=True)
        incidence = incidence.loc[
            ~incidence["source_status"].eq("contradictory/invalid")
        ].reset_index(drop=True)
    elif mutation == "duplicate_status":
        prevalence = pd.concat(
            [
                prevalence,
                prevalence.loc[
                    prevalence["source_status"].eq("contradictory/invalid")
                ],
            ],
            ignore_index=True,
        )
        incidence = pd.concat(
            [
                incidence,
                incidence.loc[
                    incidence["source_status"].eq("contradictory/invalid")
                ],
            ],
            ignore_index=True,
        )
    elif mutation == "unknown_estimate_type":
        prevalence.loc[
            prevalence["group_type"].eq("valid_observed_marker"), "estimate_type"
        ] = "unrecognized_summary"
    elif mutation == "denominator_open":
        prevalence.loc[prevalence.index[0], "denominator_n"] = 99
    elif mutation == "event_partition_open":
        incidence.loc[incidence["group_value"].eq("lower"), "non_event_n"] = 26
    elif mutation == "risk_mismatch":
        incidence.loc[incidence["group_value"].eq("lower"), "outcome_risk"] = 0.2
    elif mutation == "percentage_mismatch":
        incidence.loc[
            incidence["group_value"].eq("lower"), "outcome_risk_percentage"
        ] = 20.0
    elif mutation == "ci_mismatch":
        incidence.loc[incidence["group_value"].eq("lower"), "ci_low"] = 0.2
    elif mutation == "variable_identity_drift":
        prevalence.loc[
            ~prevalence["group_type"].eq("source_status"), "variable"
        ] = "different_marker"
    elif mutation == "outcome_identity_drift":
        incidence["outcome"] = "different_response"
    elif mutation == "group_label_drift":
        mask = prevalence["group_type"].eq("marker_group")
        prevalence.loc[mask, "group_label"] = list(
            reversed(prevalence.loc[mask, "group_label"].tolist())
        )
    elif mutation == "group_definition_drift":
        prevalence.loc[
            prevalence["group_type"].eq("marker_group"), "group_definition"
        ] = "A different scientific grouping definition."
    elif mutation == "row_family_alias":
        prevalence.loc[
            prevalence["group_type"].eq("marker_group"), "estimate_type"
        ] = "fabricated_prevalence"
        prevalence.loc[
            prevalence["group_type"].eq("valid_observed_marker"),
            "estimate_type",
        ] = "unrelated_valid_observed_distribution"
    elif mutation == "zero_event_missing_ci":
        grouped = incidence.index[incidence["group_type"].eq("marker_group")]
        zero_index, receiver_index = grouped[:2]
        moved_events = int(incidence.at[zero_index, "event_n"])
        for row_index, event_n in (
            (zero_index, 0),
            (
                receiver_index,
                int(incidence.at[receiver_index, "event_n"]) + moved_events,
            ),
        ):
            n = int(incidence.at[row_index, "n"])
            risk = event_n / n
            incidence.at[row_index, "event_n"] = event_n
            incidence.at[row_index, "non_event_n"] = n - event_n
            incidence.at[row_index, "outcome_risk"] = risk
            incidence.at[row_index, "outcome_risk_percentage"] = 100.0 * risk
            incidence.at[row_index, "outcome_risk_fraction"] = risk
            if event_n == 0:
                incidence.at[row_index, "ci_low"] = math.nan
                incidence.at[row_index, "ci_high"] = math.nan
                incidence.at[row_index, "ci_method"] = None
                incidence.at[row_index, "ci_alpha"] = math.nan
            else:
                incidence.at[row_index, "ci_low"] = risk
                incidence.at[row_index, "ci_high"] = risk
            group_value = incidence.at[row_index, "group_value"]
            prevalence_index = prevalence.index[
                prevalence["group_type"].eq("marker_group")
                & prevalence["group_value"].eq(group_value)
            ][0]
            prevalence.at[prevalence_index, "event_n"] = event_n
            prevalence.at[prevalence_index, "non_event_n"] = n - event_n
    return prevalence, incidence


def _write_parent(
    run_dir: Path,
    *,
    planner_method: str = CONTROLLED_METHOD,
    planner_outputs: list[str] | None = None,
    schema_mutation: str | None = None,
    register_prevalence: bool = True,
    register_incidence: bool = True,
    extra_registered_table: bool = False,
) -> Path:
    parent = run_dir / "steps" / PARENT_STEP / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    prevalence, incidence = _apply_schema_mutation(
        *_base_tables(), schema_mutation
    )
    prevalence_path = parent / "exposure_prevalence.csv"
    incidence_path = parent / "outcome_incidence.csv"
    prevalence.to_csv(prevalence_path, index=False)
    incidence.to_csv(incidence_path, index=False)
    summary = {
        "step_id": PARENT_STEP,
        "method": CONTROLLED_METHOD,
        "analysis_family": "association_study",
        "analysis_status": "ok",
        "primary_exposure": "marker_value",
        "target_outcome": "response_flag",
        "locked_cohort_n": 100,
        "source_status_schema": list(SOURCE_STATUSES),
        "source_status_counts": {
            "valid observed": 60,
            "no source": 40,
            "measured/source present but summary missing": 0,
            "contradictory/invalid": 0,
        },
        "outcome_data_available": True,
        "source_status_available": True,
        "measurement_provenance_ok": True,
        "effect_output_authorized": False,
        "outcome_incidence_rows": json.loads(incidence.to_json(orient="records")),
        "exposure_prevalence_rows": json.loads(
            prevalence.to_json(orient="records")
        ),
        "output_files": ["outcome_incidence.csv", "exposure_prevalence.csv"],
    }
    if schema_mutation == "summary_fail_flags":
        summary["measurement_provenance_ok"] = False
        summary["source_status_available"] = False
        summary["outcome_data_available"] = False
    elif schema_mutation == "embedded_rows_conflict":
        summary["outcome_incidence_rows"][0]["event_n"] = 99
        summary["exposure_prevalence_rows"][0]["group_value"] = "not_the_csv_row"
    elif schema_mutation == "summary_exposure_identity_drift":
        summary["primary_exposure"] = "different_marker"
    summary_path = parent / "step_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = EvidenceStore(run_dir)
    records = []
    if register_incidence:
        records.append(
            evidence.register_file(
                kind="table",
                description="Planned binary-response incidence table.",
                source_path=incidence_path,
                evidence_id="outcome_incidence_table",
                produced_by_step=PARENT_STEP,
                producer="coder",
                generation_mode="llm",
            )
        )
    if register_prevalence:
        records.append(
            evidence.register_file(
                kind="table",
                description="Planned exposure prevalence table.",
                source_path=prevalence_path,
                evidence_id="exposure_prevalence_table",
                produced_by_step=PARENT_STEP,
                producer="coder",
                generation_mode="llm",
            )
        )
    if extra_registered_table:
        extra_path = parent / "unrelated_context.csv"
        pd.DataFrame({"row_id": [1], "value": [2]}).to_csv(extra_path, index=False)
        records.append(
            evidence.register_file(
                kind="table",
                description="Unrelated registered context.",
                source_path=extra_path,
                evidence_id="unrelated_context_table",
                produced_by_step=PARENT_STEP,
                producer="coder",
                generation_mode="llm",
            )
        )
    summary_record = evidence.register_file(
        kind="statistic",
        description="Structured parent summary.",
        source_path=summary_path,
        evidence_id="incidence_prevalence_summary",
        produced_by_step=PARENT_STEP,
        producer="runner",
        generation_mode="llm",
    )
    records.append(summary_record)
    manifest = {
        "per_step_records": [
            {
                "step_id": PARENT_STEP,
                "status": "ok",
                "analysis_request": {
                    "step": {
                        "step_id": PARENT_STEP,
                        "method": planner_method,
                        "inputs": [
                            "artifact:locked_cohort",
                            "artifact:adult_marker_value_complete_case",
                            "marker_value",
                            "response_flag",
                        ],
                        "expected_outputs": planner_outputs
                        or [
                            "table:outcome_incidence",
                            "table:exposure_prevalence",
                        ],
                    }
                },
                "evidence_ids": [record.evidence_id for record in records],
                "step_summary_evidence_id": summary_record.evidence_id,
            }
        ],
        "evidence": [record.model_dump(mode="json") for record in evidence.records()],
    }
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "research_context.json").write_text(
        json.dumps(
            {
                "primary_exposure": "marker_value",
                "target_outcome": "response_flag",
            }
        ),
        encoding="utf-8",
    )
    return parent


def _figure_step(*, inputs: list[str] | None = None) -> AnalysisStep:
    return AnalysisStep(
        step_id=FIGURE_STEP,
        intent="Render the direct parent's registered incidence and prevalence tables.",
        inputs=inputs
        or ["table:outcome_incidence", "table:exposure_prevalence"],
        expected_outputs=["figure:absolute_risk_by_marker_value"],
        method="visualization",
    )


def _assert_no_output(out_dir: Path) -> None:
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_verified_parent_selector_seal_and_authorized_render(tmp_path: Path) -> None:
    _write_parent(tmp_path, extra_registered_table=True)
    step = _figure_step()

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == REPAIR_ID
    assert _sealed_renderer_figure_step_matches_parent(tmp_path, step, REPAIR_ID)
    seal = _absolute_risk_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert seal is not None
    assert set(seal) == {
        "step_summary.json",
        "outcome_incidence.csv",
        "exposure_prevalence.csv",
    }

    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=REPAIR_ID,
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            parent_artifact_digests=seal,
        )
        == REPAIR_ID
    )
    assert bind_declared_figure_products(
        out_dir=out,
        declared_products=step.expected_outputs,
        authorized_product_slots=authorize_declared_figure_product_slots(
            declared_products=step.expected_outputs,
            renderer_repair_id=REPAIR_ID,
            planner_parent_anchors=_sealed_parent_planner_anchors(
                run_dir=tmp_path,
                figure_step_id=FIGURE_STEP,
            ),
            authoritative_display_subjects=["marker_value"],
        ),
        renderer_repair_id=REPAIR_ID,
        renderer_implementation_sha256="a" * 64,
        renderer_parent_digests=seal,
    )
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert len(list(out.glob(f"*.{suffix}"))) == 1
    contracts = list(out.glob("*.figure_contract.json"))
    assert len(contracts) == 1

    rendered_summary = json.loads((out / "step_summary.json").read_text("utf-8"))
    source_names = rendered_summary["source_data_files"]
    assert len(source_names) == 2
    for source_name in source_names:
        source = pd.read_csv(out / source_name)
        assert {"source_table", "source_row_index"} <= set(source.columns)
        assert source["source_row_index"].notna().all()

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out,
        run_dir=tmp_path,
        step_summary=rendered_summary,
    )
    assert findings == []


def test_direct_renderer_accepts_the_same_verified_three_file_seal(
    tmp_path: Path,
) -> None:
    _write_parent(tmp_path)
    seal = _absolute_risk_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert seal is not None
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert (
        render_absolute_risk_bundle_from_prior_outputs(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            preverified_parent_digests=seal,
        )
        == REPAIR_ID
    )


def test_host_owned_subject_identity_must_match_summary_and_tables(
    tmp_path: Path,
) -> None:
    parent = _write_parent(tmp_path)
    summary = json.loads((parent / "step_summary.json").read_text("utf-8"))
    outcome = parent / "outcome_incidence.csv"
    prevalence = parent / "exposure_prevalence.csv"

    assert prepare_absolute_risk_inputs(
        summary,
        outcome,
        prevalence,
        expected_primary_exposure="marker_value",
        expected_target_outcome="response_flag",
    ) is not None
    assert prepare_absolute_risk_inputs(
        summary,
        outcome,
        prevalence,
        expected_primary_exposure="different_marker",
        expected_target_outcome="response_flag",
    ) is None
    assert prepare_absolute_risk_inputs(
        summary,
        outcome,
        prevalence,
        expected_primary_exposure="marker_value",
        expected_target_outcome="different_response",
    ) is None


@pytest.mark.parametrize(
    "kwargs",
    (
        {"planner_method": "mixed_effects_regression"},
        {"planner_outputs": ["table:outcome_incidence"]},
        {"register_prevalence": False},
        {"register_incidence": False},
    ),
)
def test_wrong_planner_contract_or_unregistered_table_fails_closed(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    _write_parent(tmp_path, **kwargs)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None
    assert _absolute_risk_parent_digest_seal(tmp_path, FIGURE_STEP) is None
    _assert_no_output(out)


def test_child_requires_both_typed_parent_edges(tmp_path: Path) -> None:
    _write_parent(tmp_path)
    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == REPAIR_ID
    assert not _sealed_renderer_figure_step_matches_parent(
        tmp_path,
        _figure_step(inputs=["table:outcome_incidence"]),
        REPAIR_ID,
    )


def test_latest_parent_record_cannot_borrow_an_older_request(tmp_path: Path) -> None:
    _write_parent(tmp_path)
    manifest_path = tmp_path / "manifest_partial.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    previous = manifest["per_step_records"][0]
    manifest["per_step_records"].append(
        {
            "step_id": PARENT_STEP,
            "status": "ok",
            "analysis_request": None,
            "evidence_ids": previous["evidence_ids"],
            "step_summary_evidence_id": previous["step_summary_evidence_id"],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None
    assert _absolute_risk_parent_digest_seal(tmp_path, FIGURE_STEP) is None


def test_post_registration_table_tamper_fails_without_outputs(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    with (parent / "outcome_incidence.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None
    assert _absolute_risk_parent_digest_seal(tmp_path, FIGURE_STEP) is None
    _assert_no_output(out)


@pytest.mark.parametrize(
    "schema_mutation",
    (
        "unknown_status",
        "missing_status",
        "duplicate_status",
        "unknown_estimate_type",
        "denominator_open",
        "event_partition_open",
        "risk_mismatch",
        "percentage_mismatch",
        "ci_mismatch",
        "variable_identity_drift",
        "summary_exposure_identity_drift",
        "outcome_identity_drift",
        "group_label_drift",
        "group_definition_drift",
        "row_family_alias",
        "summary_fail_flags",
        "embedded_rows_conflict",
        "zero_event_missing_ci",
    ),
)
def test_malformed_closed_schema_fails_before_rendering(
    tmp_path: Path, schema_mutation: str
) -> None:
    _write_parent(tmp_path, schema_mutation=schema_mutation)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None
    assert _absolute_risk_parent_digest_seal(tmp_path, FIGURE_STEP) is None
    _assert_no_output(out)
