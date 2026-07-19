"""Closed-contract tests for the distribution/availability figure adapter."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.figures.distribution_availability import (
    REPAIR_ID,
    render_distribution_availability_bundle_from_prior_outputs,
)
from easyicu.research_agent.pipeline import (
    _distribution_availability_parent_digest_seal,
    _distribution_availability_figure_step_matches_parent,
    _render_publication_bundle_from_prior_outputs_for_step,
    _step_contract_findings,
    deterministic_figure_repair_id_for_upstream,
)
from easyicu.research_agent.schema import AnalysisStep
from easyicu.research_agent.schema import ResearchContext
from easyicu.research_agent.schema import ValidationFinding


PARENT_STEP = "02_marker_audit"
FIGURE_STEP = f"{PARENT_STEP}_figure"


def _research_context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe a planner-selected marker audit.",
        cohort={
            "cohort_name": "synthetic",
            "database": "synthetic",
            "n_patients": 10,
            "n_stays": 10,
        },
        variables=[
            {
                "name": "marker_value",
                "role": "lab",
                "dtype": "float64",
                "unit": "units",
            }
        ],
        primary_exposure="marker_value",
    )


def _write_parent(
    run_dir: Path,
    *,
    method: str = "exposure_distribution_and_missingness_audit",
    distribution_mutation: str | None = None,
    measurement_mutation: str | None = None,
    exposure_mutation: str | None = None,
    register_measurement: bool = True,
    extra_active_tables: bool = False,
    planner_method: str = "exposure_distribution_and_missingness_audit",
    planner_outputs: list[str] | None = None,
    distribution_table_name: str = "descriptive_distribution.csv",
    measurement_table_name: str = "source_availability.csv",
) -> Path:
    parent = run_dir / "steps" / PARENT_STEP / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    distribution = pd.DataFrame(
        [
            {
                "row_type": "marker_distribution",
                "variable": "marker_value",
                "category": "valid observed",
                "n": 8,
                "denominator_n": 10,
                "percentage": 80.0,
                "fraction": 0.8,
                "median_units": 2.5,
                "q25_units": 1.5,
                "q75_units": 4.0,
                "min_units": 0.5,
                "max_units": 8.0,
                "outcome_risk": None,
            },
            {
                "row_type": "outcome_by_availability",
                "variable": "response_flag",
                "category": "valid observed",
                "n": 8,
                "denominator_n": 10,
                "percentage": 80.0,
                "fraction": 0.8,
                "median_units": None,
                "q25_units": None,
                "q75_units": None,
                "min_units": None,
                "max_units": None,
                "outcome_risk": 0.375,
            },
        ]
    )
    if distribution_mutation in {
        "windowed_alias_conflict",
        "windowed_equivalent",
        "windowed_large_anchor_conflict",
        "windowed_large_alias_conflict",
        "windowed_malformed_category",
    }:
        distribution.loc[0, "row_type"] = "marker_value_distribution"
        distribution.loc[0, "category"] = None
        distribution.loc[0, "analysis_set"] = "valid_observed_marker_value"
        distribution.loc[0, "time_window"] = "first_24h"
        distribution.loc[0, "unit"] = "units"
        for metric in ("median", "q25", "q75"):
            distribution.loc[0, metric] = distribution.loc[0, f"{metric}_units"]
        distribution = distribution.rename(
            columns={"min_units": "min", "max_units": "max"}
        )
        if distribution_mutation == "windowed_alias_conflict":
            distribution.loc[0, "median"] = 9.5
        elif distribution_mutation == "windowed_large_alias_conflict":
            distribution.loc[0, "max_units"] = 1e12
            distribution.loc[0, "max"] = 1e12 + 999
        elif distribution_mutation == "windowed_large_anchor_conflict":
            distribution.loc[0, "max"] = 1e12 + 999
        elif distribution_mutation == "windowed_malformed_category":
            distribution.loc[0, "category"] = "!!!"
    elif distribution_mutation == "dimensionless_analysis_set":
        distribution.loc[0, "row_type"] = "marker_value_distribution"
        distribution.loc[0, "category"] = None
        distribution.loc[0, "analysis_set"] = "valid_observed_marker_value"
        distribution = distribution.rename(
            columns={
                "median_units": "median",
                "q25_units": "q25",
                "q75_units": "q75",
                "min_units": "min",
                "max_units": "max",
            }
        )
    elif distribution_mutation == "category_alias_equivalent":
        distribution.loc[0, "analysis_set"] = "valid_observed_marker_value"
    elif distribution_mutation == "explicit_window_conflict":
        distribution.loc[0, "time_window"] = "last_48h"
    elif distribution_mutation == "explicit_unit_conflict":
        distribution.loc[0, "unit"] = "wrong_units"
    elif distribution_mutation == "selector_alias_conflict":
        distribution.loc[0, "analysis_set"] = "all_finite_nonmissing"
    elif distribution_mutation == "secondary_after_valid":
        secondary = distribution.iloc[[0]].copy()
        secondary["category"] = "secondary descriptive"
        secondary["median_units"] = 9.5
        distribution = pd.concat([distribution, secondary], ignore_index=True)
    elif distribution_mutation == "ambiguous_rows":
        duplicate = distribution.iloc[[0]].copy()
        duplicate["median_units"] = 9.5
        distribution = pd.concat([distribution, duplicate], ignore_index=True)
    elif distribution_mutation == "identical_duplicate":
        distribution = pd.concat(
            [distribution, distribution.iloc[[0]].copy()], ignore_index=True
        )
    elif distribution_mutation == "alternate_first":
        alternate = distribution.iloc[[0]].copy()
        alternate["category"] = "all finite nonmissing"
        distribution = pd.concat([alternate, distribution], ignore_index=True)
    elif distribution_mutation == "misnamed_metric":
        distribution = distribution.rename(
            columns={"median_units": "median_response_risk"}
        )
    elif distribution_mutation == "metric_name_mismatch":
        distribution = distribution.rename(columns={"median_units": "median"})
    elif distribution_mutation == "metric_name_mismatch_with_unit":
        distribution = distribution.rename(columns={"median_units": "median"})
        distribution.loc[0, "unit"] = "units"
    elif distribution_mutation == "percentage_mismatch":
        distribution.loc[0, "percentage"] = 81.0
    distribution.to_csv(parent / distribution_table_name, index=False)

    observed_status_n = 7 if measurement_mutation == "observed_count_mismatch" else 8
    statuses = (
        ("valid observed", observed_status_n),
        ("no source", 10 - observed_status_n),
        ("source present but summary missing", 0),
        ("contradictory or invalid", 0),
    )
    measurement_rows = [
        {
            "row_type": "numeric_coercion_audit",
            "variable": "marker_value",
            "category": "",
            "source_status": "",
            "n": 10,
            "denominator_n": 10,
            "percentage": 100.0,
            "fraction": 1.0,
            "outcome_risk": None,
        }
    ]
    measurement_rows.extend(
        {
            "row_type": "source_status",
            "variable": "marker_source_status",
            "category": status,
            "source_status": status,
            "n": count,
            "denominator_n": 10,
            "percentage": 100.0 * count / 10,
            "fraction": count / 10,
            "outcome_risk": 0.25 if count else None,
        }
        for status, count in statuses
    )
    measurement = pd.DataFrame(measurement_rows)
    if measurement_mutation == "open_partition":
        measurement.loc[measurement["source_status"].eq("no source"), "n"] = 1
    elif measurement_mutation == "percentage_mismatch":
        measurement.loc[measurement["source_status"].eq("no source"), "percentage"] = (
            99.0
        )
    elif measurement_mutation == "extra_status":
        measurement = pd.concat(
            [
                measurement,
                pd.DataFrame(
                    [
                        {
                            "row_type": "source_status",
                            "variable": "marker_source_status",
                            "category": "unplanned",
                            "source_status": "unplanned",
                            "n": 0,
                            "denominator_n": 10,
                            "percentage": 0.0,
                            "fraction": 0.0,
                            "outcome_risk": None,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    elif measurement_mutation == "zero_rates_missing":
        zero = measurement["n"].eq(0)
        measurement.loc[zero, ["percentage", "fraction"]] = None
    elif measurement_mutation == "nonzero_rates_missing":
        nonzero = measurement["source_status"].eq("no source")
        measurement.loc[nonzero, ["percentage", "fraction"]] = None
    measurement.to_csv(parent / measurement_table_name, index=False)

    schema = [status for status, _count in statuses]
    summary_median_name = (
        "median_response_risk"
        if distribution_mutation == "misnamed_metric"
        else "median_units"
    )
    summary = {
        "step_id": PARENT_STEP,
        "analysis_family": "association_study",
        "method": method,
        "primary_exposure": {
            "column": "marker_value",
            "display_label": "Marker value",
            "unit": "units",
            "time_window": "first_24h",
            "authoritative": exposure_mutation != "not_authoritative",
            "role": (
                "alternate_exposure"
                if exposure_mutation == "wrong_role"
                else "authoritative_primary_exposure"
            ),
        },
        "distribution": {
            "table": distribution_table_name,
            "observed_n": 8,
            summary_median_name: 2.5,
            "q25_units": 1.5,
            "q75_units": 4.0,
            "min_units": 0.5,
            "max_units": 8.0,
        },
        "measurement_audit": {
            "table": measurement_table_name,
            "source_status_schema": schema,
            "source_status_counts": dict(statuses),
            "status_assignment_n": 10,
        },
    }
    if distribution_mutation in {
        "windowed_large_alias_conflict",
        "windowed_large_anchor_conflict",
    }:
        summary["distribution"]["max_units"] = 1e12
    elif distribution_mutation == "dimensionless_analysis_set":
        summary["primary_exposure"].pop("unit")
        summary["primary_exposure"].pop("time_window")
        summary["distribution"] = {
            "table": distribution_table_name,
            "observed_n": 8,
            "median": 2.5,
            "q25": 1.5,
            "q75": 4.0,
            "min": 0.5,
            "max": 8.0,
        }
    (parent / "step_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    evidence = EvidenceStore(run_dir)
    records = [
        evidence.register_file(
            kind="table",
            description="Planned descriptive distribution.",
            source_path=parent / distribution_table_name,
            evidence_id="descriptive_distribution_table",
            produced_by_step=PARENT_STEP,
            producer="coder",
            generation_mode="llm",
        )
    ]
    if register_measurement:
        records.append(
            evidence.register_file(
                kind="table",
                description="Planned source availability audit.",
                source_path=parent / measurement_table_name,
                evidence_id="source_availability_table",
                produced_by_step=PARENT_STEP,
                producer="coder",
                generation_mode="llm",
            )
        )
    if extra_active_tables:
        context_csv = parent / "context_table.csv"
        complete_parquet = parent / "complete_rows.parquet"
        pd.DataFrame({"context_id": [1], "label": ["auxiliary"]}).to_csv(
            context_csv, index=False
        )
        pd.DataFrame({"row_id": [1], "value": [3.0]}).to_parquet(
            complete_parquet, index=False
        )
        records.extend(
            [
                evidence.register_file(
                    kind="table",
                    description="Unrelated registered context table.",
                    source_path=context_csv,
                    evidence_id="context_table",
                    produced_by_step=PARENT_STEP,
                    producer="coder",
                    generation_mode="llm",
                ),
                evidence.register_file(
                    kind="table",
                    description="Unrelated registered row-level product.",
                    source_path=complete_parquet,
                    evidence_id="complete_rows",
                    produced_by_step=PARENT_STEP,
                    producer="coder",
                    generation_mode="llm",
                ),
            ]
        )
    summary_record = evidence.register_file(
        kind="statistic",
        description="Structured parent summary.",
        source_path=parent / "step_summary.json",
        evidence_id="distribution_audit_summary",
        produced_by_step=PARENT_STEP,
        producer="runner",
        generation_mode="llm",
    )
    records.append(summary_record)
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": PARENT_STEP,
                        "status": "ok",
                        "analysis_request": {
                            "step": {
                                "step_id": PARENT_STEP,
                                "method": planner_method,
                                "inputs": ["marker_value"],
                                "expected_outputs": planner_outputs
                                or [
                                    f"table:{Path(distribution_table_name).stem}",
                                    f"table:{Path(measurement_table_name).stem}",
                                ],
                            }
                        },
                        "evidence_ids": [record.evidence_id for record in records],
                        "step_summary_evidence_id": summary_record.evidence_id,
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


def test_verified_parent_contract_renders_without_outcome_products(
    tmp_path: Path,
) -> None:
    _write_parent(tmp_path)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert (
        deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == REPAIR_ID
    )
    assert (
        _render_publication_bundle_from_prior_outputs_for_step(
            run_dir=tmp_path, current_step_id=FIGURE_STEP, out_dir=out
        )
        == REPAIR_ID
    )
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out / f"distribution_availability.{suffix}").is_file()
    image = mpimg.imread(out / "distribution_availability.png")
    height, width = image.shape[:2]
    assert width / height >= 1.8
    svg = (out / "distribution_availability.svg").read_text("utf-8")
    assert "source present but summary missing" not in svg
    assert "source present but" in svg
    assert "summary missing" in svg
    contract_path = out / "distribution_availability.figure_contract.json"
    assert contract_path.is_file()
    assert (
        FigureContractQualityValidator().audit_contract_file(
            contract_path, manuscript_facing=True
        )
        == []
    )

    distribution_source = pd.read_csv(out / "distribution_panel_source_data.csv")
    availability_source = pd.read_csv(out / "availability_panel_source_data.csv")
    assert "outcome_risk" not in distribution_source.columns
    assert "outcome_risk" not in availability_source.columns
    assert distribution_source["row_type"].tolist() == ["marker_distribution"]
    assert availability_source["row_type"].eq("source_status").all()

    rendered_summary = json.loads((out / "step_summary.json").read_text("utf-8"))
    figure_step = AnalysisStep(
        step_id=FIGURE_STEP,
        intent="Render the direct parent's planned descriptive audit.",
        inputs=["table:descriptive_distribution", "table:source_availability"],
        expected_outputs=["figure:publication_figure"],
        method="publication_figure_generation",
    )
    findings = FigureSourceDataValidator().audit(
        step=figure_step,
        out_dir=out,
        run_dir=tmp_path,
        step_summary=rendered_summary,
    )
    assert findings == []

    clean_contract = json.loads(contract_path.read_text("utf-8"))
    for malformed_id in (
        "figure:",
        "figure:distribution:availability",
        "figure:../distribution_availability",
        "distribution:availability",
        "../distribution_availability",
    ):
        malformed_contract = {**clean_contract, "figure_id": malformed_id}
        contract_path.write_text(json.dumps(malformed_contract), encoding="utf-8")
        malformed_findings = FigureSourceDataValidator().audit(
            step=figure_step,
            out_dir=out,
            run_dir=tmp_path,
            step_summary=rendered_summary,
        )
        assert any(
            finding.detail.get("reason") == "figure_contract_export_mismatch"
            for finding in malformed_findings
        )


def test_controlled_parent_contract_accepts_closed_renderer_inputs(
    tmp_path: Path,
) -> None:
    parent_out = _write_parent(tmp_path)
    summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
    step = AnalysisStep(
        step_id=PARENT_STEP,
        intent="Audit a planner-selected exposure distribution and availability.",
        inputs=["marker_value"],
        expected_outputs=[
            "table:descriptive_distribution",
            "table:source_availability",
        ],
        method="exposure_distribution_and_missingness_audit",
    )

    findings = _step_contract_findings(
        step=step,
        step_summary=summary,
        context=_research_context(),
        out_dir=parent_out,
    )

    assert not any(
        finding.validator == "distribution_availability_parent_contract"
        for finding in findings
    )


def test_controlled_parent_contract_blocks_unrenderable_summary_before_sealing(
    tmp_path: Path,
) -> None:
    parent_out = _write_parent(tmp_path)
    summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
    summary["distribution"] = {
        "observed_n": summary["distribution"]["observed_n"],
        "median_units": 999.0,
    }
    step = AnalysisStep(
        step_id=PARENT_STEP,
        intent="Audit a planner-selected exposure distribution and availability.",
        inputs=["marker_value"],
        expected_outputs=[
            "table:descriptive_distribution",
            "table:source_availability",
        ],
        method="exposure_distribution_and_missingness_audit",
    )

    findings = _step_contract_findings(
        step=step,
        step_summary=summary,
        context=_research_context(),
        out_dir=parent_out,
    )

    parent_findings = [
        finding
        for finding in findings
        if finding.validator == "distribution_availability_parent_contract"
    ]
    assert len(parent_findings) == 1
    assert parent_findings[0].severity == "error"
    assert parent_findings[0].detail["reason"] == (
        "distribution_availability_parent_contract_invalid"
    )
    assert parent_findings[0].detail["kind"] == (
        "controlled_renderer_parent_contract_invalid"
    )
    assert parent_findings[0].detail["contract_issue"] == (
        "summary_table_roles_invalid"
    )
    assert parent_findings[0].detail["required_schema"]["primary_exposure"] == [
        "column",
        "authoritative=true",
        "role=authoritative_primary_exposure",
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    (
        ("wrong_exposure", "summary_primary_exposure_mismatch"),
        ("wrong_unit", "summary_primary_exposure_unit_mismatch"),
        ("missing_planner_input", "host_primary_exposure_not_planner_input"),
        ("missing_assignment_n", "closed_schema_rejected"),
    ),
)
def test_controlled_parent_contract_binds_host_exposure_and_accounting(
    tmp_path: Path,
    mutation: str,
    expected_issue: str,
) -> None:
    parent_out = _write_parent(tmp_path)
    summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
    step_inputs = ["marker_value"]
    if mutation == "wrong_exposure":
        summary["primary_exposure"]["column"] = "alternate_marker"
    elif mutation == "wrong_unit":
        summary["primary_exposure"]["unit"] = "other units"
    elif mutation == "missing_planner_input":
        step_inputs = ["alternate_marker"]
    elif mutation == "missing_assignment_n":
        summary["measurement_audit"].pop("status_assignment_n")
    step = AnalysisStep(
        step_id=PARENT_STEP,
        intent="Audit a planner-selected exposure distribution and availability.",
        inputs=step_inputs,
        expected_outputs=[
            "table:descriptive_distribution",
            "table:source_availability",
        ],
        method="exposure_distribution_and_missingness_audit",
    )

    findings = _step_contract_findings(
        step=step,
        step_summary=summary,
        context=_research_context(),
        out_dir=parent_out,
    )

    parent_finding = next(
        finding
        for finding in findings
        if finding.validator == "distribution_availability_parent_contract"
    )
    assert parent_finding.severity == "error"
    assert parent_finding.detail["contract_issue"] == expected_issue


def test_controlled_parent_contract_does_not_claim_unrelated_method_or_no_outputs(
    tmp_path: Path,
) -> None:
    parent_out = _write_parent(tmp_path)
    summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
    summary.pop("distribution")
    unrelated_step = AnalysisStep(
        step_id=PARENT_STEP,
        intent="Fit a separate adjusted model.",
        inputs=["marker_value"],
        expected_outputs=[
            "table:descriptive_distribution",
            "table:source_availability",
        ],
        method="mixed_effects_regression",
    )
    controlled_step = unrelated_step.model_copy(
        update={"method": "exposure_distribution_and_missingness_audit"}
    )

    unrelated_findings = _step_contract_findings(
        step=unrelated_step,
        step_summary=summary,
        context=_research_context(),
        out_dir=parent_out,
    )
    compatibility_findings = _step_contract_findings(
        step=controlled_step,
        step_summary=summary,
        context=_research_context(),
        out_dir=None,
    )

    assert not any(
        finding.validator == "distribution_availability_parent_contract"
        for finding in (*unrelated_findings, *compatibility_findings)
    )


def test_missingness_named_parent_routes_only_with_closed_schema_and_typed_edge(
    tmp_path: Path,
) -> None:
    _write_parent(
        tmp_path,
        distribution_table_name="marker_distribution.csv",
        measurement_table_name="marker_missingness.csv",
    )
    child = AnalysisStep(
        step_id=FIGURE_STEP,
        intent="Render the two direct-parent products.",
        inputs=["table:marker_distribution", "table:marker_missingness"],
        expected_outputs=["figure:publication_figure"],
        method="publication_figure_generation",
    )

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == (
        REPAIR_ID
    )
    assert _distribution_availability_figure_step_matches_parent(tmp_path, child)
    assert set(
        _distribution_availability_parent_digest_seal(tmp_path, FIGURE_STEP) or {}
    ) == {
        "step_summary.json",
        "marker_distribution.csv",
        "marker_missingness.csv",
    }


@pytest.mark.parametrize(
    ("planner_outputs", "role_detail", "expected_role_products"),
    (
        (
            [
                "table:descriptive_distribution",
                "table:alternate_distribution",
                "table:source_availability",
            ],
            "distribution_role_products",
            ["alternate_distribution", "descriptive_distribution"],
        ),
        (
            [
                "table:descriptive_distribution",
                "table:source_availability",
                "table:secondary_missingness",
            ],
            "availability_role_products",
            ["secondary_missingness", "source_availability"],
        ),
    ),
)
def test_duplicate_parent_role_fails_closed_before_renderer_routing(
    tmp_path: Path,
    planner_outputs: list[str],
    role_detail: str,
    expected_role_products: list[str],
) -> None:
    parent_out = _write_parent(tmp_path, planner_outputs=planner_outputs)
    summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
    step = AnalysisStep(
        step_id=PARENT_STEP,
        intent="Audit a planner-selected exposure distribution and availability.",
        inputs=["marker_value"],
        expected_outputs=planner_outputs,
        method="exposure_distribution_and_missingness_audit",
    )

    findings = _step_contract_findings(
        step=step,
        step_summary=summary,
        context=_research_context(),
        out_dir=parent_out,
    )

    parent_finding = next(
        finding
        for finding in findings
        if finding.validator == "distribution_availability_parent_contract"
    )
    assert parent_finding.detail["contract_issue"] == "planner_table_roles_ambiguous"
    assert parent_finding.detail[role_detail] == expected_role_products
    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None


def test_windowed_schema_equal_aliases_and_zero_count_blanks_renders(
    tmp_path: Path,
) -> None:
    _write_parent(
        tmp_path,
        distribution_mutation="windowed_equivalent",
        measurement_mutation="zero_rates_missing",
    )
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert (
        deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == REPAIR_ID
    )
    assert (
        _render_publication_bundle_from_prior_outputs_for_step(
            run_dir=tmp_path, current_step_id=FIGURE_STEP, out_dir=out
        )
        == REPAIR_ID
    )
    availability = pd.read_csv(out / "availability_panel_source_data.csv")
    zero_rows = availability["n"].eq(0)
    assert availability.loc[zero_rows, "percentage"].isna().all()
    assert availability.loc[zero_rows, "fraction"].isna().all()
    assert "0.0%" in (out / "distribution_availability.svg").read_text("utf-8")

    rendered_summary = json.loads((out / "step_summary.json").read_text("utf-8"))
    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id=FIGURE_STEP,
            intent="Render the direct parent's planned descriptive audit.",
            inputs=["table:descriptive_distribution", "table:source_availability"],
            expected_outputs=["figure:publication_figure"],
            method="publication_figure_generation",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary=rendered_summary,
    )
    assert findings == []


@pytest.mark.parametrize(
    "distribution_mutation",
    (
        "category_alias_equivalent",
        "dimensionless_analysis_set",
        "metric_name_mismatch_with_unit",
        "secondary_after_valid",
    ),
)
def test_closed_schema_equivalent_dialects_render(
    tmp_path: Path, distribution_mutation: str
) -> None:
    _write_parent(tmp_path, distribution_mutation=distribution_mutation)

    assert (
        deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) == REPAIR_ID
    )


@pytest.mark.parametrize(
    ("kwargs", "stale_file"),
    (
        ({"method": "mixed_effects_regression"}, None),
        ({"planner_method": "mixed_effects_regression"}, None),
        ({"planner_outputs": ["table:descriptive_distribution"]}, None),
        ({"distribution_mutation": "ambiguous_rows"}, None),
        ({"distribution_mutation": "identical_duplicate"}, None),
        ({"distribution_mutation": "alternate_first"}, None),
        ({"distribution_mutation": "misnamed_metric"}, None),
        ({"distribution_mutation": "metric_name_mismatch"}, None),
        ({"distribution_mutation": "windowed_alias_conflict"}, None),
        ({"distribution_mutation": "windowed_large_anchor_conflict"}, None),
        ({"distribution_mutation": "windowed_large_alias_conflict"}, None),
        ({"distribution_mutation": "windowed_malformed_category"}, None),
        ({"distribution_mutation": "explicit_window_conflict"}, None),
        ({"distribution_mutation": "explicit_unit_conflict"}, None),
        ({"distribution_mutation": "selector_alias_conflict"}, None),
        ({"distribution_mutation": "percentage_mismatch"}, None),
        ({"measurement_mutation": "open_partition"}, None),
        ({"measurement_mutation": "percentage_mismatch"}, None),
        ({"measurement_mutation": "extra_status"}, None),
        ({"measurement_mutation": "observed_count_mismatch"}, None),
        ({"measurement_mutation": "nonzero_rates_missing"}, None),
        ({"exposure_mutation": "not_authoritative"}, None),
        ({"exposure_mutation": "wrong_role"}, None),
        ({"register_measurement": False}, None),
        ({}, "descriptive_distribution.csv"),
    ),
)
def test_ambiguous_inconsistent_or_unverified_inputs_fail_closed(
    tmp_path: Path, kwargs: dict[str, object], stale_file: str | None
) -> None:
    parent = _write_parent(tmp_path, **kwargs)
    if stale_file is not None:
        with (parent / stale_file).open("a", encoding="utf-8") as handle:
            handle.write("\n")
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None
    assert (
        _render_publication_bundle_from_prior_outputs_for_step(
            run_dir=tmp_path, current_step_id=FIGURE_STEP, out_dir=out
        )
        is None
    )
    assert not out.exists()


def test_digest_seal_contains_only_summary_selected_inputs(tmp_path: Path) -> None:
    planner_outputs = [
        "table:descriptive_distribution",
        "table:source_availability",
        "table:context_table",
    ]
    parent_out = _write_parent(
        tmp_path,
        extra_active_tables=True,
        planner_outputs=planner_outputs,
    )
    summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
    findings = _step_contract_findings(
        step=AnalysisStep(
            step_id=PARENT_STEP,
            intent="Audit a planner-selected exposure distribution and availability.",
            inputs=["marker_value"],
            expected_outputs=planner_outputs,
            method="exposure_distribution_and_missingness_audit",
        ),
        step_summary=summary,
        context=_research_context(),
        out_dir=parent_out,
    )
    assert not any(
        finding.validator == "distribution_availability_parent_contract"
        for finding in findings
    )
    seal = _distribution_availability_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert seal is not None
    assert set(seal) == {
        "step_summary.json",
        "descriptive_distribution.csv",
        "source_availability.csv",
    }

    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        render_distribution_availability_bundle_from_prior_outputs(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            preverified_parent_digests=seal,
        )
        == REPAIR_ID
    )


def test_latest_parent_record_cannot_borrow_an_older_planner_contract(
    tmp_path: Path,
) -> None:
    _write_parent(tmp_path)
    manifest_path = tmp_path / "manifest_partial.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

    assert _distribution_availability_parent_digest_seal(tmp_path, FIGURE_STEP) is None
    assert deterministic_figure_repair_id_for_upstream(tmp_path, FIGURE_STEP) is None


def test_figure_child_requires_typed_edge_or_exact_legacy_split(tmp_path: Path) -> None:
    _write_parent(tmp_path)
    typed_child = AnalysisStep(
        step_id=FIGURE_STEP,
        intent="Render the registered parent tables.",
        inputs=["table:descriptive_distribution", "table:source_availability"],
        expected_outputs=["figure:publication_figure"],
        method="publication_figure_generation",
    )
    legacy_split = typed_child.model_copy(
        update={
            "inputs": ["marker_value"],
            "method": "exposure_distribution_and_missingness_audit",
        }
    )
    unbound = typed_child.model_copy(update={"inputs": ["marker_value"]})

    assert _distribution_availability_figure_step_matches_parent(tmp_path, typed_child)
    assert _distribution_availability_figure_step_matches_parent(tmp_path, legacy_split)
    assert not _distribution_availability_figure_step_matches_parent(tmp_path, unbound)


@pytest.mark.parametrize(
    "selected_name",
    (
        "descriptive_distribution.csv",
        "source_availability.csv",
        "step_summary.json",
    ),
)
def test_execution_digest_seal_rejects_selected_parent_mutation(
    tmp_path: Path, selected_name: str
) -> None:
    parent = _write_parent(tmp_path, extra_active_tables=True)
    seal = _distribution_availability_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert seal is not None
    with (parent / selected_name).open("a", encoding="utf-8") as handle:
        handle.write("\n")

    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        render_distribution_availability_bundle_from_prior_outputs(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            preverified_parent_digests=seal,
        )
        is None
    )
    assert not out.exists()


def test_execution_parses_the_same_bytes_that_crossed_the_digest_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _write_parent(tmp_path)
    seal = _distribution_availability_parent_digest_seal(tmp_path, FIGURE_STEP)
    assert seal is not None
    original_read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path == parent / "descriptive_distribution.csv":
            path.write_text("replaced after sealed read\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        render_distribution_availability_bundle_from_prior_outputs(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
            preverified_parent_digests=seal,
        )
        == REPAIR_ID
    )
    assert pd.read_csv(out / "distribution_panel_source_data.csv")["median_units"].iloc[
        0
    ] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("sealed_case", "figure_outputs", "visual_message", "expected_status"),
    [
        (
            "cosmetic_visual",
            [
                "figure:marker_distribution",
                "figure:marker_measurement_availability",
            ],
            "Overlapping text elements detected; adjust spacing between annotations.",
            "ok",
        ),
        (
            "hard_visual",
            [
                "figure:marker_distribution",
                "figure:marker_measurement_availability",
            ],
            "Rendered numeric annotations disagree with the expected source values.",
            "execution_failed",
        ),
        (
            "contract",
            [
                "figure:marker_distribution",
                "figure:marker_measurement_availability",
            ],
            None,
            "contract_failed",
        ),
        (
            "parent_receipt",
            [
                "figure:marker_distribution",
                "figure:marker_measurement_availability",
            ],
            None,
            "contract_failed",
        ),
        (
            "host_slot_denial",
            [
                "figure:marker_distribution",
                "figure:marker_measurement_availability",
                "figure:marker_third_role",
            ],
            None,
            "repair_failed",
        ),
    ],
)
def test_sealed_renderer_authority_and_failure_policy(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_case: str,
    figure_outputs: list[str],
    visual_message: str | None,
    expected_status: str,
) -> None:
    class PlannedAuditLLM:
        name = "planned-audit-llm"

        def __init__(self) -> None:
            self.code_calls = 0
            self.repair_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            user = next(
                (
                    message.content
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps(
                    {
                        "research_question": "Describe a planned marker audit.",
                        "steps": [
                            {
                                "step_id": PARENT_STEP,
                                "intent": "Audit the planned marker distribution and availability.",
                                "inputs": ["marker_value"],
                                "expected_outputs": [
                                    "table:descriptive_distribution",
                                    "table:marker_missingness",
                                ],
                                "method": "exposure_distribution_and_missingness_audit",
                                "icu_rule_refs": [],
                            },
                            {
                                "step_id": FIGURE_STEP,
                                "intent": "Render the direct parent's planned descriptive audit.",
                                "inputs": [
                                    "table:descriptive_distribution",
                                    "table:marker_missingness",
                                ],
                                "expected_outputs": figure_outputs,
                                "method": "publication_figure_generation",
                                "icu_rule_refs": [],
                            },
                        ],
                        "rationale": "The figure is a rendering-only child of the audit.",
                    }
                )
            if "WRITE THE PYTHON CODE" in upper:
                self.code_calls += 1
                return r"""
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
cohort = pd.read_parquet(os.environ["COHORT_PARQUET"])
values = pd.to_numeric(cohort["marker_value"], errors="coerce")
observed = values.dropna()
n_total = int(len(cohort))
n_observed = int(len(observed))
distribution = pd.DataFrame([{
    "row_type": "marker_distribution",
    "variable": "marker_value",
    "category": "valid observed",
    "n": n_observed,
    "denominator_n": n_total,
    "percentage": 100.0 * n_observed / n_total,
    "fraction": n_observed / n_total,
    "median_units": float(observed.median()),
    "q25_units": float(observed.quantile(0.25)),
    "q75_units": float(observed.quantile(0.75)),
    "min_units": float(observed.min()),
    "max_units": float(observed.max()),
}])
distribution.to_csv(os.path.join(out, "descriptive_distribution.csv"), index=False)
schema = [
    "valid observed",
    "no source",
    "source present but summary missing",
    "contradictory or invalid",
]
counts = {
    "valid observed": n_observed,
    "no source": n_total - n_observed,
    "source present but summary missing": 0,
    "contradictory or invalid": 0,
}
availability = pd.DataFrame([{
    "row_type": "source_status",
    "variable": "marker_source_status",
    "category": status,
    "source_status": status,
    "n": counts[status],
    "denominator_n": n_total,
    "percentage": 100.0 * counts[status] / n_total,
    "fraction": counts[status] / n_total,
} for status in schema])
availability.to_csv(os.path.join(out, "marker_missingness.csv"), index=False)
summary = {
    "method": "exposure_distribution_and_missingness_audit",
    "analysis_family": "association_study",
    "primary_exposure": {
        "column": "marker_value",
        "display_label": "Marker value",
        "unit": "units",
        "authoritative": True,
        "role": "authoritative_primary_exposure",
    },
    "distribution": {
        "table": "descriptive_distribution.csv",
        "observed_n": n_observed,
        "median_units": float(observed.median()),
        "q25_units": float(observed.quantile(0.25)),
        "q75_units": float(observed.quantile(0.75)),
        "min_units": float(observed.min()),
        "max_units": float(observed.max()),
    },
    "measurement_audit": {
        "table": "marker_missingness.csv",
        "source_status_schema": schema,
        "source_status_counts": counts,
        "status_assignment_n": n_total,
    },
    "output_files": {
        "table:descriptive_distribution": "descriptive_distribution.csv",
        "table:marker_missingness": "marker_missingness.csv",
    },
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
print(json.dumps(summary))
"""
            if "REPAIR THE PYTHON CODE" in upper:
                self.repair_calls += 1
                raise AssertionError("sealed renderer must not call coder repair")
            if "INTERPRET THE RESULTS" in upper:
                return "The planned descriptive audit completed."
            return "{}"

    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.gates import visual as visual_gate_module

    class ControlledVisualAuditor:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def audit_with_expected(self, **kwargs):
            del kwargs
            if visual_message is None:
                return []
            return [
                ValidationFinding(
                    validator="visual_qa",
                    severity="error",
                    message=visual_message,
                    detail=(
                        {"reason": "svg_text_overlap_spacing"}
                        if sealed_case == "cosmetic_visual"
                        else {}
                    ),
                )
            ]

        def audit(self, **kwargs):
            del kwargs
            return []

    monkeypatch.setattr(
        visual_gate_module,
        "VisualQAAuditor",
        ControlledVisualAuditor,
    )

    if sealed_case == "parent_receipt":
        original_snapshot_reader = pipeline_execute.read_digest_bound_artifact_snapshot
        host_snapshot_calls = 0

        def mutate_parent_at_host_receipt(**kwargs):
            nonlocal host_snapshot_calls
            host_snapshot_calls += 1
            if host_snapshot_calls == 1:
                parent_out = Path(kwargs["parent_out"])
                (parent_out / "descriptive_distribution.csv").write_text(
                    "changed after sealed child execution\n",
                    encoding="utf-8",
                )
            return original_snapshot_reader(**kwargs)

        monkeypatch.setattr(
            pipeline_execute,
            "read_digest_bound_artifact_snapshot",
            mutate_parent_at_host_receipt,
        )

    if sealed_case == "contract":

        class ControlledContractValidator:
            def audit(self, *, step, **kwargs):
                del kwargs
                if step.step_id != FIGURE_STEP:
                    return []
                return [
                    ValidationFinding(
                        validator="figure_contract",
                        severity="error",
                        message="Controlled hard figure-contract failure.",
                    )
                ]

        monkeypatch.setattr(
            pipeline_execute,
            "FigureContractQualityValidator",
            ControlledContractValidator,
        )
    llm = PlannedAuditLLM()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=True,
        enable_publication_figure_skill=False,
        enable_llm_concept_audit=False,
        enable_memory=False,
        enable_latex=False,
        enable_reviewer_round=False,
        enable_reporting_checklist=False,
        enable_fairness_subgroups=False,
        enable_causal_audit=False,
        enable_probe_step=False,
        enable_replanning=False,
        max_code_repair_attempts=1,
        runner_kind="subprocess",
    )
    cohort = pd.DataFrame(
        {
            "marker_value": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, None, None],
            "response_flag": [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        }
    )
    result = pipeline.run(
        question="Describe the planned marker audit.",
        cohort=cohort,
        cohort_name="planned_marker_audit",
        database="synthetic",
        target_outcome="response_flag",
        primary_exposure="marker_value",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    manifest = json.loads((run_dir / "manifest_partial.json").read_text("utf-8"))
    figure_record = next(
        record
        for record in manifest["per_step_records"]
        if record.get("step_id") == FIGURE_STEP
    )
    if sealed_case == "host_slot_denial":
        assert llm.code_calls == 2
        assert figure_record["status"] == expected_status
        assert "sealed_renderer_repair" not in figure_record
        assert "deterministic_code_fallback" not in figure_record
        return
    assert llm.code_calls == 1
    assert llm.repair_calls == 0
    assert figure_record["status"] == expected_status
    assert figure_record["runner_repair"] == REPAIR_ID
    assert figure_record["deterministic_code_fallback"] == (
        "publication_figure_parent_outputs_preflight"
    )
    assert figure_record["sealed_renderer_repair"] == REPAIR_ID
    assert figure_record["post_execution_mutation_policy"] == "audit_only"
    assert figure_record["code_repair_attempts"] == 0
    assert figure_record["generation_mode"] == "fallback"
    assert figure_record["llm_repair_used"] is False
    assert figure_record["sealed_renderer_executed_code_matches_authority"] is True
    assert figure_record["executed_code_sha256"] == (
        figure_record["sealed_renderer_authorized_code_sha256"]
    )
    assert len(figure_record["sealed_renderer_implementation_sha256"]) == 64
    assert "easyicu.research_agent.repair_registry" in (
        figure_record["sealed_renderer_source_digests"]
    )
    assert set(figure_record["sealed_renderer_parent_digests"]) == {
        "step_summary.json",
        "descriptive_distribution.csv",
        "marker_missingness.csv",
    }
    assert figure_record["sealed_renderer_authorized_product_slots"] == {
        "figure:marker_distribution": "distribution",
        "figure:marker_measurement_availability": "availability",
    }
    assert "repair_target_step_id" not in figure_record
    if sealed_case == "cosmetic_visual":
        assert figure_record["sealed_renderer_visual_repair_suppressed"] is True
        assert figure_record["visual_qa_demoted"] is True
        summary = figure_record["step_summary"]
        assert summary["output_files"]["figure:marker_distribution"] == (
            "distribution_availability.png"
        )
        assert summary["output_files"]["figure:marker_measurement_availability"] == (
            "distribution_availability.png"
        )
        assert summary["planner_product_slot_bindings"] == {
            "figure:marker_distribution": {
                "slot": "distribution",
                "panel_ids": ["A"],
            },
            "figure:marker_measurement_availability": {
                "slot": "availability",
                "panel_ids": ["B"],
            },
        }
        assert summary["sealed_renderer_implementation_sha256"] == (
            figure_record["sealed_renderer_implementation_sha256"]
        )
        assert summary["sealed_renderer_parent_digests"] == (
            figure_record["sealed_renderer_parent_digests"]
        )
        assert figure_record["sealed_renderer_parent_receipt_verified"] is True
    elif sealed_case == "hard_visual":
        assert figure_record["sealed_renderer_visual_repair_suppressed"] is True
        assert figure_record["sealed_renderer_terminal_reason"] == "visual_qa_failed"
    elif sealed_case in {"contract", "parent_receipt"}:
        assert figure_record["sealed_renderer_contract_repair_suppressed"] is True
        assert figure_record["sealed_renderer_terminal_reason"] == (
            "output_contract_failed"
        )
        if sealed_case == "parent_receipt":
            assert figure_record["sealed_renderer_parent_receipt_verified"] is False
    else:
        assert figure_record["sealed_renderer_runtime_repair_suppressed"] is True
        assert figure_record["sealed_renderer_terminal_reason"] == "runtime_failure"
