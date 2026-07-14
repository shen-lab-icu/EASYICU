"""The deterministic association forest-plot rescue must tolerate the OR/CI
column-name variants that free-model code emits (e.g. ``ci_lower``/``ci_upper``
rather than ``or_ci_low``/``or_ci_high``). Without this, a figure-only step
fails the whole run even though the parent step computed a valid odds ratio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.declared_product_contract import (
    bind_declared_figure_products,
)
from easyicu.research_agent.pipeline import (
    _context_axis_label,
    _render_authorized_sealed_publication_bundle,
    _render_cohort_overlap_publication_bundle_from_prior_outputs as cohort_overlap_rescue,
    _render_missingness_publication_bundle_from_prior_outputs as missingness_rescue,
    _render_publication_bundle_from_prior_outputs_for_step as routed_rescue,
    _render_association_publication_bundle_from_prior_outputs as rescue,
    _render_sensitivity_publication_bundle_from_prior_outputs as sensitivity_rescue,
    _resolve_upstream_analysis_family,
    _sealed_renderer_parent_digest_seal,
    deterministic_figure_repair_id_for_upstream,
    deterministic_figure_family_supported,
    deterministic_figure_family_supported_for_upstream,
)
from easyicu.research_agent.schema import AnalysisStep


def _make_parent_step(run_dir: Path, csv_name: str, columns: dict) -> None:
    out = run_dir / "steps" / "03_association_model" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(out / csv_name, index=False)


def test_planned_model_contract_seals_exact_association_renderer(
    tmp_path: Path, monkeypatch
) -> None:
    repair_id = "association_publication_bundle_from_planned_model_contract_v1"
    parent_step = "05_adjusted_association"
    figure_step = f"{parent_step}_figure"
    parent = tmp_path / "steps" / parent_step / "outputs"
    parent.mkdir(parents=True)
    estimates = pd.DataFrame(
        {
            "model_id": ["primary_model", "primary_model", "alternate_model"],
            "term": ["marker_max", "age", "marker_first"],
            "term_role": ["exposure", "adjustment", "exposure"],
            "source_variable": ["marker_max", "age", "marker_first"],
            "odds_ratio": [1.25, 1.02, 9.0],
            "ci_low": [1.10, 1.01, 8.0],
            "ci_high": [1.42, 1.03, 10.0],
        }
    )
    estimates.to_csv(parent / "adjusted_association_estimates.csv", index=False)
    summary = {
        "model_contracts": [
            {
                "model_id": "primary_model",
                "requirement_id": "primary_requirement",
                "analysis_role": "primary",
                "exposure_role": "primary",
                "exposure_source": "marker_max",
                "fit_status": "fitted",
            },
            {
                "model_id": "alternate_model",
                "requirement_id": "alternate_requirement",
                "analysis_role": "secondary",
                "exposure_role": "secondary",
                "exposure_source": "marker_first",
                "fit_status": "fitted",
            },
        ]
    }
    (parent / "step_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    seal = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            parent / "step_summary.json",
            parent / "adjusted_association_estimates.csv",
        )
    }
    request_step = {
        "method": "adjusted_association_models",
        "inputs": ["artifact:analysis_rows"],
        "expected_outputs": ["table:adjusted_association_estimates"],
        "model_requirements": [
            {
                "requirement_id": "primary_requirement",
                "analysis_role": "primary",
                "required_for_step_success": True,
            }
        ],
    }
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_artifact_digests",
        lambda run_dir, step_id: dict(seal),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: {"adjusted_association_estimates.csv"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_step",
        lambda run_dir, step_id: dict(request_step),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: {"step": dict(request_step)},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: "adjusted_association_models",
    )

    assert _sealed_renderer_parent_digest_seal(tmp_path, figure_step, repair_id) == seal
    assert deterministic_figure_repair_id_for_upstream(tmp_path, figure_step) == (
        repair_id
    )
    out = tmp_path / "steps" / figure_step / "outputs"
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=repair_id,
            run_dir=tmp_path,
            current_step_id=figure_step,
            out_dir=out,
            parent_artifact_digests=seal,
        )
        == repair_id
    )
    source = pd.read_csv(out / "publication_figure_source_data.csv")
    assert source["model_id"].tolist() == ["primary_model"]
    assert source["source_variable"].tolist() == ["marker_max"]
    assert source["source_row_index"].tolist() == [0]
    assert source["odds_ratio"].tolist() == pytest.approx([1.25])
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out / f"publication_figure.{suffix}").stat().st_size > 0
    contract = json.loads(
        (out / "publication_figure.figure_contract.json").read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "robustness",
    ]


def test_context_axis_label_wraps_metric_group_pairs():
    assert _context_axis_label("Death Risk", "Sepsis-3 Negative") == (
        "Sepsis-3 Negative\nDeath Risk"
    )
    assert _context_axis_label("Exposure prevalence", "Sepsis-3 prevalence") == (
        "Sepsis-3\nprevalence"
    )


def test_missingness_rescue_recomputes_percentages_from_counts(tmp_path: Path):
    parent = (
        tmp_path / "steps" / "02_baseline_characteristics_and_data_quality" / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "concept": ["resp", "lact", "sep3_sofa2"],
            "label": ["Respiratory rate", "Lactate", "Sepsis-3 source flag"],
            "n_total": [74829, 74829, 74829],
            "value_missing_n": [188, 30490, 0],
            "value_missing_pct": [0.2512394927, 40.7462347486, 0.0],
            "measured_one_n": [74641, 44339, 28229],
            "measured_one_pct": [99.7487605073, 59.2537652514, 37.7246789346],
            "value_present_but_measured_zero_n": [0, 0, 46600],
            "raw_indicator_one_n": [74641, 44339, 28229],
            "indicator_semantics": [
                "measurement_availability",
                "measurement_availability",
                "binary_event_presence",
            ],
            "event_count_column": ["", "", "sep3_sofa2_n"],
        }
    ).to_csv(parent / "missingness_measurement_audit.csv", index=False)
    out = (
        tmp_path
        / "steps"
        / "02_baseline_characteristics_and_data_quality_figure"
        / "outputs"
    )

    rid = missingness_rescue(
        run_dir=tmp_path,
        current_step_id="02_baseline_characteristics_and_data_quality_figure",
        out_dir=out,
    )

    assert rid == "missingness_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(out / "missingness_measurement_panel_source_data.csv")
    resp = source[source["variable"] == "resp"].iloc[0]
    source_flag = source[source["variable"] == "sep3_sofa2"].iloc[0]
    assert resp["missing_pct"] == pytest.approx(0.2512394927)
    assert resp["measured_pct"] == pytest.approx(99.7487605073)
    assert source_flag["measured_pct"] == pytest.approx(100.0)
    assert source_flag["indicator_semantics"] == "binary_event_presence"
    assert "analytic event status" in source_flag["display_label"].lower()
    contract = json.loads(
        (out / "missingness_measurement_panel.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(contract["panels"]) == 2
    assert contract["panels"][1]["title"] == "Analytic availability"
    assert "absence-as-negative" in contract["panels"][1]["claim"]


def test_missingness_split_figure_reads_only_its_direct_parent(tmp_path: Path):
    direct = tmp_path / "steps" / "05_agent_missingness" / "outputs"
    direct.mkdir(parents=True)
    pd.DataFrame(
        {
            "concept": ["direct_marker"],
            "n_total": [100],
            "missing_n": [10],
            "measured_n": [90],
        }
    ).to_csv(direct / "custom_status.csv", index=False)
    unrelated = tmp_path / "steps" / "01_missingness_measurement" / "outputs"
    unrelated.mkdir(parents=True)
    pd.DataFrame(
        {
            "concept": ["wrong_marker"],
            "n_total": [100],
            "missing_n": [99],
            "measured_n": [1],
        }
    ).to_csv(unrelated / "missingness_measurement_audit.csv", index=False)
    out = tmp_path / "steps" / "05_agent_missingness_figure" / "outputs"

    rid = missingness_rescue(
        run_dir=tmp_path,
        current_step_id="05_agent_missingness_figure",
        out_dir=out,
    )

    assert rid == "missingness_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(out / "missingness_measurement_panel_source_data.csv")
    assert set(source["variable"]) == {"direct_marker"}
    assert set(source["source_table"]) == {"custom_status.csv"}


def test_missingness_router_uses_manifest_method_and_standard_count_schema(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "03_missingness_audit" / "outputs"
    parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "variable": ["lactate", "creatinine"],
            "n": [100, 100],
            "n_missing": [10, 25],
            "fraction_missing": [0.10, 0.25],
        }
    ).to_csv(parent / "missingness.csv", index=False)
    (tmp_path / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": "03_missingness_audit",
                        "status": "ok",
                        "analysis_request": {
                            "step": {"method": "missingness"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    figure_step_id = "03_missingness_audit_figure"
    out = tmp_path / "steps" / figure_step_id / "outputs"

    # The legacy router remains unit-testable, but missingness table selection
    # is heuristic and therefore cannot preflight-replace the coder.
    assert not deterministic_figure_family_supported_for_upstream(
        tmp_path, figure_step_id
    )
    repair_id = routed_rescue(
        run_dir=tmp_path,
        current_step_id=figure_step_id,
        out_dir=out,
    )

    assert repair_id == "missingness_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(out / "missingness_measurement_panel_source_data.csv")
    assert source.set_index("variable")["missing_pct"].to_dict() == {
        "lactate": pytest.approx(10.0),
        "creatinine": pytest.approx(25.0),
    }
    assert (out / "missingness_measurement_panel.figure_contract.json").exists()


def test_missingness_rescue_prefers_rich_measurement_process_over_attrition(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "02_exposure_and_missingness_audit" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "table_section": "column_missingness",
            "variable": variable,
            "metric": "raw_missing",
            "category": None,
            "cohort": cohort,
            "n": raw_n,
            "denominator": denominator,
            "percentage": 100.0 * raw_n / denominator,
            "raw_missing_n": raw_n,
            "raw_missing_pct": 100.0 * raw_n / denominator,
            "analysis_unavailable_n": unavailable_n,
            "analysis_unavailable_pct": 100.0 * unavailable_n / denominator,
        }
        for variable, cohort, denominator, raw_n, unavailable_n in [
            ("marker", "export", 200, 5, 15),
            ("marker", "adult_analytic_cohort", 100, 10, 30),
            ("other", "export", 200, 15, 25),
            ("other", "adult_analytic_cohort", 100, 20, 40),
        ]
    ]
    status_categories = [
        "valid observed level/value",
        "no recorded source or observation",
        "measured or observed source with summary missing",
        "contradictory or invalid source-summary combinations",
    ]
    for cohort, denominator, counts in [
        ("export", 200, [185, 10, 3, 2]),
        ("adult_analytic_cohort", 100, [70, 25, 3, 2]),
    ]:
        for category, count in zip(status_categories, counts, strict=True):
            rows.append(
                {
                    "table_section": "source_status",
                    "variable": "marker",
                    "metric": "mutually_exclusive_source_status",
                    "category": category,
                    "cohort": cohort,
                    "n": count,
                    "denominator": denominator,
                    "percentage": 100.0 * count / denominator,
                    "raw_missing_n": None,
                    "raw_missing_pct": None,
                    "analysis_unavailable_n": None,
                    "analysis_unavailable_pct": None,
                }
            )
    pd.DataFrame(rows).to_csv(
        parent / "missingness_and_measurement_process.csv", index=False
    )
    pd.DataFrame(
        {
            "variable": ["marker"],
            "denominator": [100],
            "missing_n": [99],
            "missing_pct": [99.0],
        }
    ).to_csv(parent / "complete_case_attrition.csv", index=False)
    out = tmp_path / "steps" / "02_exposure_and_missingness_audit_figure" / "outputs"

    rid = missingness_rescue(
        run_dir=tmp_path,
        current_step_id="02_exposure_and_missingness_audit_figure",
        out_dir=out,
    )

    assert rid == "missingness_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(out / "missingness_measurement_panel_source_data.csv")
    assert set(source["variable_name"]) == {"marker", "other"}
    assert set(source["cohort_name"]) == {"export", "adult_analytic_cohort"}
    marker = source[
        (source["variable_name"] == "marker")
        & (source["cohort_name"] == "adult_analytic_cohort")
    ].iloc[0]
    assert marker["source_table"] == "missingness_and_measurement_process.csv"
    assert marker["source_row_filter"] == "column_missingness:raw_missing"
    assert marker["missing_pct"] == pytest.approx(10.0)
    assert marker["measured_pct"] == pytest.approx(70.0)
    assert marker["analysis_unavailable_pct"] == pytest.approx(30.0)
    status_path = out / "missingness_status_matrix_source_data.csv"
    status = pd.read_csv(status_path)
    assert set(status["status_category"]) == set(status_categories)
    assert set(status["cohort_name"]) == {"export", "adult_analytic_cohort"}
    contract = json.loads(
        (out / "missingness_measurement_panel.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert {panel["metadata"]["chart_type"] for panel in contract["panels"]} == {
        "missingness_matrix",
        "availability_panel",
    }

    step = AnalysisStep(
        step_id="02_exposure_and_missingness_audit_figure",
        intent="Render a source-status missingness matrix and availability panel.",
        expected_outputs=["figure:missingness_measurement_panel"],
    )
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out,
        run_dir=tmp_path,
        step_summary=summary,
    )
    assert not [finding for finding in findings if finding.severity == "error"]

    status.loc[0, "n"] += 1
    status.to_csv(status_path, index=False)
    tampered = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out,
        run_dir=tmp_path,
        step_summary=summary,
    )
    assert any(finding.severity == "error" for finding in tampered)


def test_missingness_rescue_accepts_scope_section_rich_schema(tmp_path: Path):
    parent = tmp_path / "steps" / "02_exposure_and_missingness_audit" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope, denominator, missing in [
        ("export", 200, 80),
        ("adult_analytic_cohort", 100, 30),
    ]:
        rows.append(
            {
                "scope": scope,
                "section": "column_missingness",
                "exposure_or_variable": "marker_first",
                "metric": "raw_missing",
                "category": None,
                "denominator_n": denominator,
                "n": missing,
                "percentage": 100.0 * missing / denominator,
            }
        )
        for category, count in [
            ("valid observed level/value", denominator - missing),
            ("no recorded source or observation", missing),
            ("measured or observed source with summary missing", 0),
            ("contradictory or invalid source-summary combinations", 0),
        ]:
            rows.append(
                {
                    "scope": scope,
                    "section": "source_status",
                    "exposure_or_variable": "marker_first",
                    "metric": "source_status",
                    "category": category,
                    "denominator_n": denominator,
                    "n": count,
                    "percentage": 100.0 * count / denominator,
                }
            )
    pd.DataFrame(rows).to_csv(
        parent / "missingness_and_measurement_process.csv", index=False
    )
    pd.DataFrame(
        {
            "required_variable": ["marker_first"],
            "n_full": [100],
            "missing_n": [99],
            "missing_percentage": [99.0],
        }
    ).to_csv(parent / "complete_case_attrition.csv", index=False)
    out = tmp_path / "steps" / "02_exposure_and_missingness_audit_figure" / "outputs"

    rid = missingness_rescue(
        run_dir=tmp_path,
        current_step_id="02_exposure_and_missingness_audit_figure",
        out_dir=out,
    )

    assert rid == "missingness_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(out / "missingness_measurement_panel_source_data.csv")
    assert set(source["cohort_name"]) == {"export", "adult_analytic_cohort"}
    analytic = source[source["cohort_name"] == "adult_analytic_cohort"].iloc[0]
    assert analytic["variable_name"] == "marker_first"
    assert analytic["missing_pct"] == pytest.approx(30.0)
    assert analytic["measured_pct"] == pytest.approx(70.0)
    assert analytic["source_table"] == "missingness_and_measurement_process.csv"
    status = pd.read_csv(out / "missingness_status_matrix_source_data.csv")
    assert len(status) == 8

    step = AnalysisStep(
        step_id="02_exposure_and_missingness_audit_figure",
        intent="Render a missingness matrix and availability panel.",
        expected_outputs=["figure:missingness_measurement_panel"],
    )
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out,
        run_dir=tmp_path,
        step_summary=summary,
    )
    assert not [finding for finding in findings if finding.severity == "error"]


def test_e3_ordered_stage_figure_step_is_deterministically_claimed(tmp_path: Path):
    # E3 regression: the deterministic ordinal runner emits a perfect
    # dose_response.csv, but the planner named the primary figure step
    # ``04_primary_ordered_stage_analysis_figure`` — which matched NO token group
    # (``ordered`` != ``ordinal``), so the forest fell to the LLM coder and
    # crashed, leaving primary_pub_fig_contracts=0 and failing the run closed.
    # The gate must now claim it AND route it to the association forest renderer.
    step_id = "04_primary_ordered_stage_analysis_figure"
    assert deterministic_figure_family_supported(step_id) is False

    parent = tmp_path / "steps" / "04_primary_ordered_stage_analysis" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    _write_parent_summary(tmp_path, "04_primary_ordered_stage_analysis", "association")
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 19593, 3621],
            "n_events": [2143, 1380, 2672, 1188],
            "event_rate": [0.0572, 0.0981, 0.1364, 0.3281],
            "is_reference": [True, False, False, False],
            "odds_ratio": [1.0, 1.587, 2.119, 5.766],
            "or_ci_low": [1.0, 1.477, 1.993, 5.289],
            "or_ci_high": [1.0, 1.705, 2.253, 6.287],
        }
    ).to_csv(parent / "dose_response.csv", index=False)
    out = tmp_path / "steps" / step_id / "outputs"

    rid = routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out)
    # A non-None id proves the deterministic association/ordinal renderer claimed
    # the step (instead of the crashing LLM coder) and emitted a figure bundle.
    assert rid is not None


def test_survival_figure_name_alone_is_not_routing_evidence(tmp_path: Path):
    assert not deterministic_figure_family_supported(
        "05_survival_by_disease_stage_figure"
    )


def _write_parent_summary(run_dir: Path, parent_id: str, family: str) -> None:
    out = run_dir / "steps" / parent_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(
        json.dumps({"step": parent_id, "status": "ok", "analysis_family": family}),
        encoding="utf-8",
    )


def test_upstream_family_routes_token_free_primary_figure(tmp_path: Path):
    # E3 (2026-07-08) real-run regression: the planner named the primary forest
    # figure ``05_primary_stage_outcome_analysis_figure`` — it matches NO family
    # token (no association/ordinal/ordered/trend/gradient), so id-token routing
    # returned None and the forest fell to the LLM coder, which failed. But its
    # PARENT analysis step recorded analysis_family='association' and produced a
    # canonical dose_response.csv. Routing by the parent's PROVEN family must
    # claim the step and render the forest deterministically.
    step_id = "05_primary_stage_outcome_analysis_figure"
    parent_id = "05_primary_stage_outcome_analysis"

    # The id itself carries no family token -> the token gate is False ...
    assert deterministic_figure_family_supported(step_id) is False

    _write_parent_summary(tmp_path, parent_id, "association")
    parent = tmp_path / "steps" / parent_id / "outputs"
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 19593, 3621],
            "n_events": [2143, 1380, 2672, 1188],
            "event_rate": [0.0572, 0.0981, 0.1364, 0.3281],
            "is_reference": [True, False, False, False],
            "odds_ratio": [1.0, 1.587, 2.119, 5.766],
            "or_ci_low": [1.0, 1.477, 1.993, 5.289],
            "or_ci_high": [1.0, 1.705, 2.253, 6.287],
        }
    ).to_csv(parent / "dose_response.csv", index=False)

    # The legacy router can still be called explicitly, but an association
    # family alone is not sufficient automatic-repair authority because the
    # renderer may have multiple effect tables/models to choose from.
    assert _resolve_upstream_analysis_family(tmp_path, step_id) == "association"
    assert (
        deterministic_figure_family_supported_for_upstream(tmp_path, step_id) is False
    )

    out = tmp_path / "steps" / step_id / "outputs"
    rid = routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out)
    assert rid is not None  # association forest renderer claimed + drew it


def test_structural_parent_family_outranks_ambiguous_figure_name(tmp_path: Path):
    step_id = "05_prediction_by_cohort_figure"
    parent_id = "05_prediction_by_cohort"
    _write_parent_summary(tmp_path, parent_id, "prediction")
    parent = tmp_path / "steps" / parent_id / "outputs"
    pd.DataFrame(
        {
            "auroc": [0.78],
            "brier_score": [0.16],
            "calibration_slope": [0.96],
            "calibration_intercept": [0.02],
        }
    ).to_csv(parent / "model_performance.csv", index=False)

    out = tmp_path / "steps" / step_id / "outputs"
    rid = routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out)

    assert rid == "prediction_publication_bundle_from_parent_outputs_v1"


def test_descriptive_parent_supported_but_guarded_against_empty_figure(tmp_path: Path):
    # A descriptive/table-one renderer exists for explicit/manual use, but a
    # family label alone cannot authorize automatic table selection.
    step_id = "03_baseline_context_figure"
    _write_parent_summary(tmp_path, "03_baseline_context", "descriptive")
    assert _resolve_upstream_analysis_family(tmp_path, step_id) == "descriptive"
    assert not deterministic_figure_family_supported_for_upstream(tmp_path, step_id)
    # No table-one output under the parent -> the strict guard declines (None).
    out = tmp_path / "steps" / step_id / "outputs"
    assert routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out) is None
    # No parent summary at all -> also unsupported (no crash).
    assert (
        deterministic_figure_family_supported_for_upstream(tmp_path, "99_x_figure")
        is False
    )


def test_rescue_handles_ci_lower_upper_variant(tmp_path: Path):
    # free-model style column names: odds_ratio + ci_lower/ci_upper
    _make_parent_step(
        tmp_path,
        "adjusted_odds_ratios.csv",
        {
            "variable": ["const", "sepsis3", "age"],
            "odds_ratio": [0.01, 0.80, 1.03],
            "ci_lower": [0.0, 0.74, 1.01],
            "ci_upper": [0.1, 0.86, 1.05],
        },
    )
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(
        run_dir=tmp_path, current_step_id="03_association_model_figure", out_dir=out
    )
    assert rid is not None
    figs = {p.suffix for p in out.iterdir()}
    assert ".png" in figs and ".svg" in figs
    contract_path = out / "publication_figure.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B"]
    assert (out / "publication_figure_source_data.csv").exists()
    findings = FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )
    assert not any(f.severity == "error" for f in findings), findings
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_rescue_handles_canonical_or_ci_columns(tmp_path: Path):
    # our deterministic fallback style: or_ci_low/or_ci_high
    _make_parent_step(
        tmp_path,
        "association_results.csv",
        {
            "variable": ["sepsis3"],
            "odds_ratio": [0.80],
            "or_ci_low": [0.74],
            "or_ci_high": [0.86],
        },
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out)
    assert rid is not None


def test_rescue_promotes_prevalence_and_absolute_risk_context(tmp_path: Path):
    parent = tmp_path / "steps" / "03_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variable": ["const", "exposed", "age"],
            "odds_ratio": [0.10, 1.20, 1.04],
            "ci_lower": [0.02, 1.05, 1.02],
            "ci_upper": [0.50, 1.40, 1.06],
        }
    ).to_csv(parent / "adjusted_odds_ratios.csv", index=False)
    pd.DataFrame(
        {
            "exposure": ["exposed"],
            "definition": ["binary exposure"],
            "n_denominator": [1000],
            "n_positive": [320],
            "prevalence": [0.32],
            "prevalence_pct": [32.0],
            "ci_low": [0.291],
            "ci_high": [0.350],
            "ci_low_pct": [29.1],
            "ci_high_pct": [35.0],
        }
    ).to_csv(parent / "exposure_prevalence.csv", index=False)
    pd.DataFrame(
        {
            "exposure_label": ["Exposure negative", "Exposure positive"],
            "n": [680, 320],
            "event_n": [61, 48],
            "outcome_risk": [0.0897, 0.1500],
            "outcome_risk_pct": [8.97, 15.0],
            "ci_low": [0.071, 0.115],
            "ci_high": [0.111, 0.193],
            "ci_low_pct": [7.1, 11.5],
            "ci_high_pct": [11.1, 19.3],
        }
    ).to_csv(parent / "outcome_by_exposure.csv", index=False)
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = rescue(
        run_dir=tmp_path,
        current_step_id="03_association_model_figure",
        out_dir=out,
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v3"
    contract = json.loads(
        (out / "publication_figure.figure_contract.json").read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "descriptive_result",
        "primary_estimand",
    ]
    assert (
        contract["panels"][0]["metadata"]["chart_type"] == "dot_interval_absolute_risk"
    )
    assert (out / "publication_figure_prevalence_source_data.csv").exists()
    assert (out / "publication_figure_absolute_risk_source_data.csv").exists()
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_rescue_uses_primary_summary_and_semantic_binary_risk_labels(tmp_path: Path):
    parent = tmp_path / "steps" / "03_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": ["primary_adjusted"],
            "outcome": ["death"],
            "exposure": ["treated"],
            "effect_scale": ["odds_ratio"],
            "point_estimate": [1.18],
            "ci_low": [1.04],
            "ci_high": [1.34],
        }
    ).to_csv(parent / "adjusted_association_death.csv", index=False)
    pd.DataFrame(
        {
            "term": ["treated", "age", "lab_missing"],
            "odds_ratio": [1.18, 1.03, 2.10],
            "ci_lower": [1.04, 1.01, 0.80],
            "ci_upper": [1.34, 1.05, 5.50],
        }
    ).to_csv(parent / "adjusted_association_death_full_coefficients.csv", index=False)
    pd.DataFrame(
        {
            "exposure": ["treated"],
            "n_denominator": [1000],
            "n_positive": [320],
            "prevalence_pct": [32.0],
            "ci_low_pct": [29.1],
            "ci_high_pct": [35.0],
        }
    ).to_csv(parent / "exposure_prevalence.csv", index=False)
    pd.DataFrame(
        {
            "treated": [0, 1, "risk_difference_1_minus_0"],
            "n_total": [680, 320, 1000],
            "death_events": [61, 48, 109],
            "death_risk": [0.0897, 0.1500, 0.0603],
            "death_risk_ci_low": [0.071, 0.115, None],
            "death_risk_ci_high": [0.111, 0.193, None],
        }
    ).to_csv(parent / "outcome_by_exposure.csv", index=False)
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = rescue(
        run_dir=tmp_path,
        current_step_id="03_association_model_figure",
        out_dir=out,
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v3"
    source = pd.read_csv(out / "publication_figure_source_data.csv")
    assert source["source_table"].tolist() == ["adjusted_association_death.csv"]
    assert source["exposure"].tolist() == ["treated"]
    absolute = pd.read_csv(out / "publication_figure_absolute_risk_source_data.csv")
    assert absolute["plot_group_label"].tolist() == [
        "Treated Negative",
        "Treated Positive",
    ]
    assert absolute["plot_ci_low_pct"].tolist() == pytest.approx([7.1, 11.5])
    contract = json.loads(
        (out / "publication_figure.figure_contract.json").read_text(encoding="utf-8")
    )
    assert contract["panels"][1]["title"] == "Primary adjusted association"
    assert contract["panels"][1]["metadata"]["chart_type"] == "dot_interval"
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_routed_rescue_prioritizes_primary_association_over_missingness(
    tmp_path: Path,
):
    parent = (
        tmp_path
        / "steps"
        / "03_primary_prevalence_and_adjusted_association"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    _write_parent_summary(
        tmp_path, "03_primary_prevalence_and_adjusted_association", "association"
    )
    pd.DataFrame(
        {
            "term": ["const", "exposed"],
            "effect_scale": ["odds_ratio", "odds_ratio"],
            "estimate": [0.10, 1.20],
            "ci_low": [0.02, 1.05],
            "ci_high": [0.50, 1.40],
        }
    ).to_csv(parent / "adjusted_association_death.csv", index=False)
    pd.DataFrame(
        {
            "exposure": ["exposed"],
            "n_denominator": [1000],
            "n_positive": [320],
            "prevalence_pct": [32.0],
            "ci_low_pct": [29.1],
            "ci_high_pct": [35.0],
        }
    ).to_csv(parent / "sepsis3_prevalence.csv", index=False)
    pd.DataFrame(
        {
            "sepsis3_label": ["Exposure negative", "Exposure positive"],
            "n": [680, 320],
            "death_n": [61, 48],
            "death_risk_pct": [8.97, 15.0],
            "ci_low_pct": [7.1, 11.5],
            "ci_high_pct": [11.1, 19.3],
        }
    ).to_csv(parent / "outcome_by_sepsis3.csv", index=False)

    missingness_parent = (
        tmp_path / "steps" / "02_baseline_characteristics_and_data_quality" / "outputs"
    )
    missingness_parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "concept": ["lactate"],
            "label": ["Lactate"],
            "n_total": [1000],
            "value_missing_n": [400],
            "value_missing_pct": [40.0],
            "measured_one_n": [600],
            "measured_one_pct": [60.0],
        }
    ).to_csv(missingness_parent / "missingness_measurement_audit.csv", index=False)

    out = (
        tmp_path
        / "steps"
        / "03_primary_prevalence_and_adjusted_association_figure"
        / "outputs"
    )

    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="03_primary_prevalence_and_adjusted_association_figure",
        out_dir=out,
        step_text="Render primary result figure with missingness/data-quality context.",
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v3"
    assert (out / "publication_figure.png").exists()
    assert not (out / "missingness_measurement_panel.png").exists()


def test_routed_association_prefers_complete_direct_parent_bundle_and_copies_contract_closure(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "05_primary_missingness_aware_association" / "outputs"
    parent.mkdir(parents=True)
    _write_parent_summary(
        tmp_path, "05_primary_missingness_aware_association", "association"
    )
    coefficient_rows = pd.DataFrame(
        {
            "model_id": ["bili_primary", "bili_primary", "sofa_secondary"],
            "term": ["bili_log1p", "age", "sofa_level_1"],
            "term_role": ["exposure", "adjustment", "exposure"],
            "source_variable": ["bili_max", "age", "sofa2_liver_max"],
            "analysis_role": ["primary", "primary", "secondary"],
            "analysis_set": ["source_aware", "source_aware", "source_aware"],
            "odds_ratio": [1.93, 1.03, 1.68],
            "ci_low": [1.86, 1.02, 1.56],
            "ci_high": [2.00, 1.04, 1.81],
        }
    )
    coefficient_rows.to_csv(parent / "coefficients.csv", index=False)
    figure_source = coefficient_rows.iloc[[0]].copy()
    figure_source["source_table"] = "coefficients.csv"
    figure_source.to_csv(parent / "figure_source_data.csv", index=False)
    pd.DataFrame({"model_id": ["bili_primary"], "n": [94458]}).to_csv(
        parent / "model_summaries.csv", index=False
    )
    pd.DataFrame({"source_status": ["observed"], "n": [41210]}).to_csv(
        parent / "source_status_summary.csv", index=False
    )
    (parent / "primary_adjusted_association_context.png").write_bytes(b"parent-png")
    (parent / "primary_adjusted_association_context.svg").write_text(
        "<svg><text>parent</text></svg>", encoding="utf-8"
    )
    contract = {
        "figure_id": "primary_adjusted_association_context",
        "core_claim": (
            "The primary bilirubin association is shown with absolute-risk "
            "context and a separately labelled sensitivity estimate."
        ),
        "source_data": "figure_source_data.csv",
        "statistics_note": (
            "All estimates and intervals are copied from registered parent-step tables."
        ),
        "panels": [
            {
                "panel_id": "A",
                "title": "Absolute-risk context",
                "role": "descriptive_result",
                "chart_type": "event_rate_panel",
                "claim": "Absolute outcome risk is shown before adjusted estimates.",
                "evidence_ids": ["source_status_summary.csv"],
            },
            {
                "panel_id": "B",
                "title": "Primary adjusted association",
                "role": "primary_estimand",
                "chart_type": "forest",
                "claim": "The locked primary adjusted odds ratio is shown with its 95% CI.",
                "evidence_ids": ["coefficients.csv", "model_summaries.csv"],
            },
        ],
    }
    (parent / "primary_adjusted_association_context.figure_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )

    out = (
        tmp_path
        / "steps"
        / "05_primary_missingness_aware_association_figure"
        / "outputs"
    )
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="05_primary_missingness_aware_association_figure",
        out_dir=out,
    )

    assert rid == "publication_bundle_promote_v1"
    assert (out / "publication_figure.png").read_bytes() == b"parent-png"
    assert (out / "publication_figure.svg").read_text(encoding="utf-8") == (
        "<svg><text>parent</text></svg>"
    )
    for filename in (
        "figure_source_data.csv",
        "source_status_summary.csv",
        "coefficients.csv",
        "model_summaries.csv",
    ):
        assert (out / filename).read_bytes() == (parent / filename).read_bytes()
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["publication_figure_rescue"]["mode"] == "promotion"
    assert summary["publication_figure_rescue"]["copied_trace_files"] == [
        "coefficients.csv",
        "figure_source_data.csv",
        "model_summaries.csv",
        "source_status_summary.csv",
    ]
    quality_findings = FigureContractQualityValidator().audit(
        step=AnalysisStep(
            step_id="05_primary_missingness_aware_association_figure",
            intent="Render the registered primary association figure.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert not [f for f in quality_findings if f.severity == "error"], quality_findings
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_primary_missingness_aware_association_figure",
            intent="Render the registered primary association figure.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_routed_association_does_not_promote_bundle_missing_declared_source_data(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "03_association_model" / "outputs"
    parent.mkdir(parents=True)
    _write_parent_summary(tmp_path, "03_association_model", "association")
    pd.DataFrame(
        {
            "variable": ["const", "exposed"],
            "odds_ratio": [0.10, 1.20],
            "ci_low": [0.02, 1.05],
            "ci_high": [0.50, 1.40],
        }
    ).to_csv(parent / "association_results.csv", index=False)
    (parent / "stale_primary.png").write_bytes(b"stale-parent")
    (parent / "stale_primary.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "stale_primary",
                "source_data": "missing_source_data.csv",
                "panels": [
                    {
                        "panel_id": "A",
                        "role": "primary_estimand",
                        "title": "Stale primary result",
                        "claim": "This bundle is incomplete and must not be promoted.",
                        "evidence_ids": ["association_results.csv"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="03_association_model_figure",
        out_dir=out,
    )

    assert rid in {
        "association_publication_bundle_from_parent_outputs_v2",
        "association_publication_bundle_from_parent_outputs_v3",
    }
    assert (out / "publication_figure.png").read_bytes() != b"stale-parent"
    assert not (out / "missing_source_data.csv").exists()


def test_association_renderer_keeps_primary_exposure_and_matching_sensitivity_with_composite_trace(
    tmp_path: Path,
):
    unrelated = tmp_path / "steps" / "01_unrelated_association" / "outputs"
    unrelated.mkdir(parents=True)
    pd.DataFrame(
        {
            "term": ["wrong_exposure"],
            "odds_ratio": [9.9],
            "ci_low": [8.0],
            "ci_high": [12.0],
        }
    ).to_csv(unrelated / "first_alphabetical_or_table.csv", index=False)

    parent = tmp_path / "steps" / "05_primary_missingness_aware_association" / "outputs"
    parent.mkdir(parents=True)
    coefficients = pd.DataFrame(
        [
            (
                "bili_full",
                "const",
                "intercept",
                None,
                "primary",
                "source_aware",
                0.01,
                0.005,
                0.02,
            ),
            (
                "bili_full",
                "bili_log1p",
                "exposure",
                "bili_max",
                "primary",
                "source_aware",
                1.93,
                1.86,
                2.00,
            ),
            (
                "bili_full",
                "bili_source",
                "availability",
                "bili_measured",
                "primary",
                "source_aware",
                1.40,
                1.32,
                1.48,
            ),
            (
                "bili_full",
                "age",
                "adjustment",
                "age",
                "primary",
                "source_aware",
                1.03,
                1.02,
                1.04,
            ),
            (
                "bili_cc",
                "const",
                "intercept",
                None,
                "sensitivity",
                "complete_case",
                0.02,
                0.01,
                0.03,
            ),
            (
                "bili_cc",
                "bili_log1p",
                "exposure",
                "bili_max",
                "sensitivity",
                "complete_case",
                1.88,
                1.81,
                1.95,
            ),
            (
                "bili_cc",
                "age",
                "adjustment",
                "age",
                "sensitivity",
                "complete_case",
                1.02,
                1.01,
                1.03,
            ),
            (
                "sofa_full",
                "const",
                "intercept",
                None,
                "secondary",
                "source_aware",
                0.02,
                0.01,
                0.03,
            ),
            (
                "sofa_full",
                "sofa_level_1",
                "exposure",
                "sofa2_liver_max",
                "secondary",
                "source_aware",
                1.68,
                1.56,
                1.81,
            ),
            (
                "sofa_full",
                "age",
                "adjustment",
                "age",
                "secondary",
                "source_aware",
                1.03,
                1.02,
                1.04,
            ),
            (
                "sofa_cc",
                "const",
                "intercept",
                None,
                "sensitivity",
                "complete_case",
                0.02,
                0.01,
                0.03,
            ),
            (
                "sofa_cc",
                "sofa_level_1",
                "exposure",
                "sofa2_liver_max",
                "sensitivity",
                "complete_case",
                1.64,
                1.53,
                1.77,
            ),
        ],
        columns=[
            "model_id",
            "term",
            "term_role",
            "source_variable",
            "analysis_role",
            "analysis_set",
            "odds_ratio",
            "ci_low",
            "ci_high",
        ],
    )
    coefficients.to_csv(parent / "coefficients.csv", index=False)
    contracts = [
        {
            "model_id": "bili_full",
            "exposure_source": "bili_max",
            "exposure_role": "primary",
            "analysis_role": "primary",
        },
        {
            "model_id": "bili_cc",
            "exposure_source": "bili_max",
            "exposure_role": "primary",
            "analysis_role": "sensitivity",
        },
        {
            "model_id": "sofa_full",
            "exposure_source": "sofa2_liver_max",
            "exposure_role": "secondary",
            "analysis_role": "secondary",
        },
        {
            "model_id": "sofa_cc",
            "exposure_source": "sofa2_liver_max",
            "exposure_role": "secondary",
            "analysis_role": "sensitivity",
        },
    ]
    (parent / "step_summary.json").write_text(
        json.dumps({"primary_model_id": "bili_full", "model_contracts": contracts}),
        encoding="utf-8",
    )

    out = (
        tmp_path
        / "steps"
        / "05_primary_missingness_aware_association_figure"
        / "outputs"
    )
    rid = rescue(
        run_dir=tmp_path,
        current_step_id="05_primary_missingness_aware_association_figure",
        out_dir=out,
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v2"
    source = pd.read_csv(out / "publication_figure_source_data.csv")
    assert source["model_id"].tolist() == ["bili_full", "bili_cc"]
    assert source["term"].tolist() == ["bili_log1p", "bili_log1p"]
    assert source["term_role"].tolist() == ["exposure", "exposure"]
    assert source["source_variable"].tolist() == ["bili_max", "bili_max"]
    assert source["analysis_role"].tolist() == ["primary", "sensitivity"]
    assert source["analysis_set"].tolist() == ["source_aware", "complete_case"]
    assert source["odds_ratio"].tolist() == pytest.approx([1.93, 1.88])
    assert source["ci_low"].tolist() == pytest.approx([1.86, 1.81])
    assert source["ci_high"].tolist() == pytest.approx([2.00, 1.95])
    assert "Source Aware" in source.loc[0, "plot_label"]
    assert "Complete Case" in source.loc[1, "plot_label"]
    result = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=out / "publication_figure_source_data.csv",
        upstream_path=parent / "coefficients.csv",
    )
    assert result.get("ok") is True, result
    assert result.get("key_column") == "model_id+term", result
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["publication_figure_repair"]["source_association_table"].endswith(
        "05_primary_missingness_aware_association/outputs/coefficients.csv"
    )


def test_rescue_returns_none_without_or_ci_table(tmp_path: Path):
    _make_parent_step(tmp_path, "prevalence.csv", {"group": ["a"], "rate": [0.3]})
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    assert rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out) is None


# --- ordinal dose-response figure steps must reach the deterministic renderer ---
# E3 regression: when the LLM names its primary figure step "..._stage_gradient_
# analysis_figure" / "..._dose_response_figure" (instead of "...association...")
# the deterministic figure family/router did not recognise it, so the step fell
# through to LLM code that produced a corrupted source_data table (ci_low filled
# with the cohort count) which the figure-trace gate then rejected. The ordinal
# dose-response family is an association forest and must route to the association
# bundle renderer, which reads dose_response.csv and emits stage-keyed source data.


@pytest.mark.parametrize(
    "step_id",
    [
        "04_primary_stage_gradient_analysis_figure",
        "04_primary_dose_response_figure",
        "04_ordinal_trend_analysis_figure",
    ],
)
def test_ordinal_figure_name_is_not_family_evidence(step_id: str):
    assert deterministic_figure_family_supported(step_id) is False


def test_graded_exposure_forest_keys_by_varying_level_not_constant_model(
    tmp_path: Path,
):
    # M1 regression: a single graded exposure keeps exposure_variable/model
    # CONSTANT across rows and varies by ordinal `level`. The renderer must label
    # and key rows by the varying `level`, not collapse every row to the constant
    # column (which drops the per-row trace key -> "no shared key").
    parent = tmp_path / "steps" / "04_primary_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    _write_parent_summary(tmp_path, "04_primary_association_model", "association")
    pd.DataFrame(
        {
            "model": ["adjusted"] * 5,
            "exposure_variable": ["sofa2_liver_cat"] * 5,
            "level": [0, 1, 2, 3, 4],
            "odds_ratio": [1.0, 1.2122, 1.3638, 1.6002, 3.9035],
            "ci_low": [1.0, 1.1075, 1.1985, 1.3627, 3.3520],
            "ci_high": [1.0, 1.3269, 1.5520, 1.8791, 4.5458],
        }
    ).to_csv(parent / "primary_adjusted_odds_ratios.csv", index=False)

    out = tmp_path / "steps" / "04_primary_association_model_figure" / "outputs"
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="04_primary_association_model_figure",
        out_dir=out,
        step_text="Adjusted odds ratio per SOFA-2 liver category level.",
    )
    assert rid is not None
    src = pd.read_csv(out / "publication_figure_source_data.csv")
    # keyed by the varying level, 5 distinct rows (not collapsed to one label)
    assert "level" in src.columns
    assert src["level"].nunique() == 5
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=src,
        source_path=out / "publication_figure_source_data.csv",
        upstream_path=parent / "primary_adjusted_odds_ratios.csv",
    )
    assert res.get("ok") is True, res
    assert res.get("key_column") == "level", res


def test_ordinal_stage_gradient_figure_routes_to_association_renderer(tmp_path: Path):
    parent = tmp_path / "steps" / "04_primary_stage_gradient_analysis" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    _write_parent_summary(tmp_path, "04_primary_stage_gradient_analysis", "association")
    # the deterministic ordinal runner's canonical dose_response.csv shape
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 5200, 2100],
            "n_events": [2143, 1380, 780, 500],
            "event_rate": [0.0572, 0.0981, 0.150, 0.238],
            "is_reference": [True, False, False, False],
            "odds_ratio": [1.0, 1.5871617453700098, 2.51, 4.02],
            "or_ci_low": [1.0, 1.4771205, 2.30, 3.60],
            "or_ci_high": [1.0, 1.7054007, 2.74, 4.49],
            "or_p_value": [None, 2.08e-36, 1e-40, 1e-50],
        }
    ).to_csv(parent / "dose_response.csv", index=False)

    out = tmp_path / "steps" / "04_primary_stage_gradient_analysis_figure" / "outputs"
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="04_primary_stage_gradient_analysis_figure",
        out_dir=out,
        step_text="Render the adjusted odds-ratio gradient per KDIGO stage.",
    )
    # routed to a real deterministic renderer (NOT None -> not LLM-coded fallback)
    assert rid is not None, "ordinal figure fell through to LLM code"
    assert (out / "publication_figure.png").exists()
    # and the emitted source data traces to dose_response.csv on the `stage` key
    src = pd.read_csv(out / "publication_figure_source_data.csv")
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=src,
        source_path=out / "publication_figure_source_data.csv",
        upstream_path=parent / "dose_response.csv",
    )
    assert res.get("ok") is True, res


def test_cohort_overlap_rescue_writes_traceable_multipanel_bundle(tmp_path: Path):
    parent = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "definition_id": ["primary", "relax_temp", "tight_los"],
            "definition_label": [
                "Primary cohort",
                "Relax temperature requirement",
                "Tighten ICU length-of-stay threshold",
            ],
            "definition_type": ["primary", "alternative", "alternative"],
            "criteria": ["primary", "no temp", "los >=2"],
            "n_included": [100, 112, 72],
            "n_excluded": [50, 38, 78],
            "included_pct_of_rows": [66.7, 74.7, 48.0],
            "overlap_with_primary_n": [100, 100, 72],
            "overlap_with_primary_pct_of_primary": [100.0, 100.0, 72.0],
            "overlap_with_primary_pct_of_definition": [100.0, 89.3, 100.0],
            "moved_in_vs_primary_n": [0, 12, 0],
            "moved_out_vs_primary_n": [0, 0, 28],
        }
    ).to_csv(parent / "alternative_cohort_attrition.csv", index=False)
    rows = []
    sizes = {"primary": 100, "relax_temp": 112, "tight_los": 72}
    intersections = {
        ("primary", "primary"): 100,
        ("primary", "relax_temp"): 100,
        ("primary", "tight_los"): 72,
        ("relax_temp", "primary"): 100,
        ("relax_temp", "relax_temp"): 112,
        ("relax_temp", "tight_los"): 72,
        ("tight_los", "primary"): 72,
        ("tight_los", "relax_temp"): 72,
        ("tight_los", "tight_los"): 72,
    }
    for definition_a, n_a in sizes.items():
        for definition_b, n_b in sizes.items():
            intersection = intersections[(definition_a, definition_b)]
            union = n_a + n_b - intersection
            rows.append(
                {
                    "definition_a": definition_a,
                    "definition_b": definition_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "intersection_n": intersection,
                    "union_n": union,
                    "jaccard": intersection / union,
                    "a_in_b_pct": intersection / n_a * 100,
                    "b_in_a_pct": intersection / n_b * 100,
                }
            )
    pd.DataFrame(rows).to_csv(parent / "cohort_overlap_matrix.csv", index=False)

    out = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    out.mkdir(parents=True, exist_ok=True)

    rid = cohort_overlap_rescue(
        run_dir=tmp_path,
        current_step_id="04_alternative_eligibility_definitions_and_overlap_figure",
        out_dir=out,
    )

    assert rid == "cohort_overlap_publication_bundle_from_parent_outputs_v1"
    assert (out / "publication_figure.png").exists()
    assert (out / "publication_figure.svg").exists()
    assert (out / "publication_figure_definition_source_data.csv").exists()
    assert (out / "publication_figure_overlap_source_data.csv").exists()
    contract_path = out / "publication_figure.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert (
        FigureContractQualityValidator().audit_contract_file(
            contract_path,
            manuscript_facing=True,
        )
        == []
    )
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            intent=(
                "Render the publication figure declared by step "
                "'04_alternative_eligibility_definitions_and_overlap'."
            ),
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_cohort_overlap_rescue_shortens_sepsis3_derivable_definition_labels(
    tmp_path: Path,
):
    parent = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    ids = [
        "primary_adult_los1_all_vitals_sepsis3_derivable",
        "alt_adult_no_los_all_vitals_sepsis3_derivable",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable",
        "alt_adult_los2_all_vitals_sepsis3_derivable",
    ]
    pd.DataFrame(
        {
            "definition_id": ids,
            "definition_label": [
                "Primary cohort",
                "Relax ICU length-of-stay threshold",
                "Relax vital completeness to >=3 of 4",
                "Relax temperature requirement",
                "Tighten ICU length-of-stay threshold",
            ],
            "definition_type": [
                "primary",
                "alternative",
                "alternative",
                "alternative",
                "alternative",
            ],
            "n_included": [100, 100, 112, 111, 70],
            "n_excluded": [20, 20, 8, 9, 50],
            "included_pct_of_rows": [83.3, 83.3, 93.3, 92.5, 58.3],
            "overlap_with_primary_n": [100, 100, 100, 100, 70],
            "overlap_with_primary_pct_of_primary": [100, 100, 100, 100, 70],
            "overlap_with_primary_pct_of_definition": [100, 100, 89.3, 90.1, 100],
            "moved_in_vs_primary_n": [0, 0, 12, 11, 0],
            "moved_out_vs_primary_n": [0, 0, 0, 0, 30],
        }
    ).to_csv(parent / "alternative_cohort_attrition.csv", index=False)
    rows = []
    for a in ids:
        for b in ids:
            rows.append(
                {
                    "definition_a": a,
                    "definition_b": b,
                    "n_a": 100,
                    "n_b": 100,
                    "intersection_n": 100 if a == b else 80,
                    "union_n": 100 if a == b else 120,
                    "jaccard": 1.0 if a == b else 2 / 3,
                }
            )
    pd.DataFrame(rows).to_csv(parent / "cohort_overlap_matrix.csv", index=False)
    out = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    out.mkdir(parents=True, exist_ok=True)

    assert (
        cohort_overlap_rescue(
            run_dir=tmp_path,
            current_step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            out_dir=out,
        )
        == "cohort_overlap_publication_bundle_from_parent_outputs_v1"
    )
    source = pd.read_csv(out / "publication_figure_definition_source_data.csv")
    assert source["display_label"].tolist() == [
        "Primary",
        "No LOS threshold",
        ">=3 of 4 vitals",
        "No temperature",
        "LOS >=2 d",
    ]


def test_sensitivity_rescue_writes_multipanel_contract_and_source_data(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "05_sensitivity_comparison" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spec_id": ["primary", "alt_cohort", "risk_difference"],
            "axis": ["cohort", "cohort", "outcome"],
            "display_label": [
                "Primary cohort",
                "Alternative cohort",
                "Risk difference",
            ],
            "effect_scale": ["OR", "OR", "RD"],
            "point_estimate": [1.12, 1.05, 0.03],
            "ci_low": [1.02, 0.95, 0.01],
            "ci_high": [1.24, 1.17, 0.05],
            "se": [0.05, 0.06, 0.01],
            "modeled_analytic_n": [1000, 920, 1000],
            "converged": [True, True, True],
        }
    ).to_csv(parent / "sensitivity_comparison.csv", index=False)
    out = tmp_path / "steps" / "05_sensitivity_comparison_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = sensitivity_rescue(
        run_dir=tmp_path,
        current_step_id="05_sensitivity_comparison_across_definitions_figure",
        out_dir=out,
    )

    assert rid == "sensitivity_publication_bundle_from_parent_outputs_v2"
    contract_path = out / "sensitivity_forest.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert (out / "sensitivity_forest_source_data.csv").exists()
    assert (
        FigureContractQualityValidator().audit_contract_file(
            contract_path,
            manuscript_facing=True,
        )
        == []
    )
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_sensitivity_comparison_figure",
            intent="Render the sensitivity figure declared by step '05_sensitivity_comparison'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []

    routed_out = (
        tmp_path
        / "steps"
        / "05_sensitivity_comparison_across_definitions_figure_routed"
        / "outputs"
    )
    routed_out.mkdir(parents=True, exist_ok=True)
    routed_id = routed_rescue(
        run_dir=tmp_path,
        current_step_id="05_sensitivity_comparison_across_definitions_figure",
        out_dir=routed_out,
    )
    assert routed_id is None
    assert not (routed_out / "sensitivity_forest_source_data.csv").exists()


def test_sensitivity_rescue_prefers_declared_summary_and_excludes_point_only_rows(
    tmp_path: Path,
):
    parent = (
        tmp_path / "steps" / "07_cohort_definition_sensitivity_comparison" / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spec_id": ["alt_binary", "alt_continuous"],
            "axis": ["outcome", "outcome"],
            "effect_scale": ["odds_ratio", "conditional_median_difference"],
            "point_estimate": [7.8, 1.9],
            "ci_low": [pd.NA, 1.8],
            "ci_high": [pd.NA, 2.0],
            "converged": [False, True],
            "penalized": [True, False],
            "reportable": [False, True],
            "n": [900, 880],
        }
    ).to_csv(parent / "robustness_summary.csv", index=False)
    # This competing model-level table also has the generic estimate schema.
    # The declared robustness product must retain source ownership.
    pd.DataFrame(
        {
            "spec_id": ["wrong_table"],
            "effect_scale": ["odds_ratio"],
            "point_estimate": [9.9],
            "ci_low": [9.0],
            "ci_high": [10.8],
            "converged": [True],
            "reportable": [True],
        }
    ).to_csv(parent / "model_fit_summary.csv", index=False)
    (parent / "step_summary.json").write_text(
        json.dumps(
            {
                "method": "cohort_definition_sensitivity",
                "output_files": [
                    "model_fit_summary.csv",
                    "robustness_summary.csv",
                ],
            }
        ),
        encoding="utf-8",
    )
    sealed_paths = [
        parent / "step_summary.json",
        parent / "robustness_summary.csv",
    ]
    seal = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sealed_paths
    }
    sealed_repair_id = "sensitivity_publication_bundle_from_locked_summary_v1"
    sealed_out = (
        tmp_path
        / "steps"
        / "07_cohort_definition_sensitivity_comparison_figure"
        / "sealed_outputs"
    )
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=sealed_repair_id,
            run_dir=tmp_path,
            current_step_id=("07_cohort_definition_sensitivity_comparison_figure"),
            out_dir=sealed_out,
            parent_artifact_digests=seal,
        )
        == sealed_repair_id
    )
    assert pd.read_csv(sealed_out / "sensitivity_forest_source_data.csv")[
        "spec_id"
    ].tolist() == ["alt_continuous"]
    out = (
        tmp_path
        / "steps"
        / "07_cohort_definition_sensitivity_comparison_figure"
        / "outputs"
    )

    rid = sensitivity_rescue(
        run_dir=tmp_path,
        current_step_id="07_cohort_definition_sensitivity_comparison_figure",
        out_dir=out,
    )

    assert rid == "sensitivity_publication_bundle_from_locked_summary_v1"
    plotted = pd.read_csv(out / "sensitivity_forest_source_data.csv")
    assert plotted["spec_id"].tolist() == ["alt_continuous"]
    assert plotted["source_table"].unique().tolist() == ["robustness_summary.csv"]
    assert plotted["modeled_analytic_n"].tolist() == [880]
    excluded = pd.read_csv(out / "sensitivity_estimability_source_data.csv")
    assert excluded["spec_id"].tolist() == ["alt_binary"]
    contract = json.loads(
        (out / "sensitivity_forest.figure_contract.json").read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["title"] == "Median-difference sensitivity"
    original_summary = json.loads(
        (out / "step_summary.json").read_text(encoding="utf-8")
    )
    assert original_summary["figure_contract"] == (
        "sensitivity_forest.figure_contract.json"
    )
    with pytest.raises(ValueError, match="host product-slot authorization"):
        bind_declared_figure_products(
            out_dir=out,
            declared_products=["figure:robustness_grid", "figure:robustness_plot"],
            authorized_product_slots={"figure:robustness_plot": "robustness_plot"},
            renderer_repair_id=(
                "sensitivity_publication_bundle_from_locked_summary_v1"
            ),
            renderer_implementation_sha256="d" * 64,
            renderer_parent_digests={
                "step_summary.json": "1" * 64,
                "robustness_summary.csv": "2" * 64,
            },
        )
    assert json.loads((out / "step_summary.json").read_text(encoding="utf-8")) == (
        original_summary
    )
    assert bind_declared_figure_products(
        out_dir=out,
        declared_products=["figure:robustness_plot"],
        authorized_product_slots={"figure:robustness_plot": "robustness_plot"},
        renderer_repair_id=("sensitivity_publication_bundle_from_locked_summary_v1"),
        renderer_implementation_sha256="d" * 64,
        renderer_parent_digests={
            "step_summary.json": "1" * 64,
            "robustness_summary.csv": "2" * 64,
        },
    )
    (parent / "robustness_summary.csv").write_text(
        "spec_id,effect_scale,point_estimate,ci_low,ci_high\n"
        "changed,odds_ratio,9,8,10\n",
        encoding="utf-8",
    )
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=sealed_repair_id,
            run_dir=tmp_path,
            current_step_id=("07_cohort_definition_sensitivity_comparison_figure"),
            out_dir=(
                tmp_path
                / "steps"
                / "07_cohort_definition_sensitivity_comparison_figure"
                / "mutated_outputs"
            ),
            parent_artifact_digests=seal,
        )
        is None
    )


def test_sensitivity_exact_method_authorizes_only_verified_summary(monkeypatch):
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: {"robustness_summary.csv"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: {
            "step": {
                "method": "cohort_definition_sensitivity",
                "expected_outputs": ["table:robustness_summary"],
            },
            "analysis_family": "association_study",
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: "cohort_definition_sensitivity",
    )
    assert (
        pipeline_module.deterministic_figure_repair_id_for_upstream(
            Path("/unused"), "07_sensitivity_figure"
        )
        == "sensitivity_publication_bundle_from_locked_summary_v1"
    )


@pytest.mark.parametrize(
    (
        "planner_method",
        "planner_outputs",
        "reported_method",
        "reported_family",
        "verified_tables",
    ),
    (
        (
            "kaplan_meier_estimation",
            ["table:kaplan_meier_curve_distribution"],
            "ordinal_exposure_derivation_and_quality_control",
            "survival",
            {"severity_distribution.csv"},
        ),
        (
            "mixed_effects_regression",
            ["table:robustness_summary"],
            "cohort_definition_sensitivity",
            "association_study",
            {"robustness_summary.csv"},
        ),
        (
            "survival_analysis",
            ["table:cohort_flow", "table:attrition"],
            "survival_analysis",
            "cohort_definition",
            {"cohort_flow.csv", "attrition.csv"},
        ),
    ),
)
def test_sealed_selector_cannot_be_overridden_by_coder_summary(
    monkeypatch,
    planner_method,
    planner_outputs,
    reported_method,
    reported_family,
    verified_tables,
):
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: verified_tables,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: {
            "step": {
                "method": planner_method,
                "expected_outputs": planner_outputs,
            },
            "analysis_family": "association_study",
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: reported_method,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_family",
        lambda run_dir, step_id: reported_family,
    )

    assert (
        pipeline_module.deterministic_figure_repair_id_for_upstream(
            Path("/unused"), "07_spoofed_figure"
        )
        is None
    )


def test_sealed_selector_requires_host_recorded_parent_request(monkeypatch):
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: {"robustness_summary.csv"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: "cohort_definition_sensitivity",
    )

    assert (
        pipeline_module.deterministic_figure_repair_id_for_upstream(
            Path("/unused"), "07_legacy_summary_only_figure"
        )
        is None
    )


def test_sensitivity_rescue_omits_empty_scale_and_separates_nonindependent_rows(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "06_robustness" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    exact_ci_high = 1.9991014484226914
    pd.DataFrame(
        {
            "spec_id": ["primary", "alt_cohort_los_ge_1d", "outcome_any"],
            "axis": ["primary", "cohort", "outcome"],
            "effect_scale": ["OR", "OR", "OR"],
            "point_estimate": [1.9259885602397633, 1.8685656161598152, 9.9],
            "ci_low": [1.8555496206064033, 1.7951476395205703, 9.0],
            "ci_high": [exact_ci_high, 1.9449862423723512, 10.8],
            "modeled_analytic_n": [94458, 74829, pd.NA],
            "event_n": [9466, 7397, pd.NA],
            "converged": [True, True, True],
            "independent_variant": [pd.NA, pd.NA, False],
            "notes": ["primary", "cohort replay", "same scalar outcome"],
        }
    ).to_csv(parent / "robustness_matrix.csv", index=False)
    (parent / "step_summary.json").write_text(
        json.dumps(
            {
                "output_files": {"robustness_matrix": "robustness_matrix.csv"},
                "aliases": {"sensitivity_comparison": "robustness_matrix.csv"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "steps" / "06_robustness_figure" / "outputs"

    rid = sensitivity_rescue(
        run_dir=tmp_path,
        current_step_id="06_robustness_figure",
        out_dir=out,
    )

    assert rid == "sensitivity_publication_bundle_from_parent_outputs_v2"
    contract = json.loads(
        (out / "sensitivity_forest.figure_contract.json").read_text(encoding="utf-8")
    )
    assert [panel["title"] for panel in contract["panels"]] == [
        "Ratio-scale sensitivity",
        "Model denominator audit",
    ]
    assert "Risk difference" not in (out / "sensitivity_forest.svg").read_text(
        encoding="utf-8"
    )
    plotted = pd.read_csv(
        out / "sensitivity_forest_source_data.csv",
        float_precision="round_trip",
    )
    assert plotted["spec_id"].tolist() == ["primary", "alt_cohort_los_ge_1d"]
    assert plotted.loc[0, "ci_high"] == exact_ci_high
    status = pd.read_csv(out / "sensitivity_estimability_source_data.csv")
    assert status["spec_id"].tolist() == ["outcome_any"]
    assert "modeled_analytic_n" not in status.columns
    assert "model_id" not in status.columns
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["n_rows_plotted"] == 2
    assert summary["n_denominator_rows"] == 2
    assert summary["n_non_independent_variants"] == 1


def test_structured_sensitivity_source_preserves_spec_model_trace_and_detects_tamper(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "06_robustness" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    script_sha = "a" * 64
    rows = [
        {
            "spec_id": "primary",
            "effect_scale": "OR",
            "point_estimate": 1.9,
            "ci_low": 1.8,
            "ci_high": 2.0,
            "modeled_analytic_n": 1000,
            "model_contract_n": 1000,
            "event_n": 100,
            "model_id": "primary_full",
            "source_model_id": "primary_full",
            "exposure_source": "exposure",
            "exposure_expression": "log1p(exposure)",
            "exposure_role": "primary",
            "analysis_role": "primary",
            "analysis_set": "source_aware",
            "baseline_missing_policy": "explicit_missing_category",
            "fit_status": "fitted",
            "fit_method": "registered_model",
            "replay_mode": "completed_primary_step_output",
            "coefficient_source_table": "coefficients.csv",
            "coefficient_term": "exposure_log1p",
            "model_contract_source": "step_summary.json:model_contracts",
            "source_script_sha256": script_sha,
            "axis": "primary",
            "converged": True,
            "estimability_status": "estimated",
            "membership_n": 1000,
            "independent_variant": pd.NA,
        },
        {
            "spec_id": "alt_cohort",
            "effect_scale": "OR",
            "point_estimate": 1.8,
            "ci_low": 1.7,
            "ci_high": 1.9,
            "modeled_analytic_n": 900,
            "model_contract_n": 900,
            "event_n": 85,
            "model_id": "primary_full",
            "source_model_id": "primary_full",
            "exposure_source": "exposure",
            "exposure_expression": "log1p(exposure)",
            "exposure_role": "primary",
            "analysis_role": "primary",
            "analysis_set": "source_aware",
            "baseline_missing_policy": "explicit_missing_category",
            "fit_status": "fitted",
            "fit_method": "registered_model",
            "replay_mode": "exact_registered_primary_model_code",
            "coefficient_source_table": "robustness_variant_coefficients.csv",
            "coefficient_term": "exposure_log1p",
            "model_contract_source": "step_summary.json:robustness_model_contracts",
            "source_script_sha256": script_sha,
            "axis": "cohort",
            "converged": True,
            "estimability_status": "estimated",
            "membership_n": 900,
            "independent_variant": pd.NA,
        },
        {
            "spec_id": "outcome_any",
            "effect_scale": "OR",
            "point_estimate": pd.NA,
            "ci_low": pd.NA,
            "ci_high": pd.NA,
            "modeled_analytic_n": pd.NA,
            "axis": "outcome",
            "converged": False,
            "estimability_status": "not_independent",
            "membership_n": 1000,
            "independent_variant": False,
            "notes": "Same stay-level scalar outcome.",
        },
    ]
    pd.DataFrame(rows).to_csv(parent / "robustness_matrix.csv", index=False)

    def contract(*, n: int, event_n: int, spec_id: str | None = None) -> dict:
        payload = {
            "model_id": "primary_full",
            "exposure_source": "exposure",
            "exposure_expression": "log1p(exposure)",
            "exposure_role": "primary",
            "analysis_role": "primary",
            "analysis_set": "source_aware",
            "baseline_missing_policy": "explicit_missing_category",
            "n": n,
            "event_n": event_n,
            "fit_status": "fitted",
            "converged": True,
            "separation_detected": False,
            "penalized": False,
            "fit_method": "registered_model",
        }
        if spec_id is not None:
            payload.update(
                {
                    "spec_id": spec_id,
                    "source_model_id": "primary_full",
                    "replay_mode": "exact_registered_primary_model_code",
                }
            )
        return payload

    coefficient = {
        "model_id": "primary_full",
        "term": "exposure_log1p",
        "term_role": "exposure",
        "source_variable": "exposure",
        "odds_ratio": 1.9,
        "ci_low": 1.8,
        "ci_high": 2.0,
    }
    pd.DataFrame([coefficient]).to_csv(parent / "coefficients.csv", index=False)
    pd.DataFrame([{**coefficient, "spec_id": "alt_cohort", "odds_ratio": 1.8}]).to_csv(
        parent / "robustness_variant_coefficients.csv", index=False
    )
    (parent / "step_summary.json").write_text(
        json.dumps(
            {
                "primary_model_id": "primary_full",
                "model_contracts": [contract(n=1000, event_n=100)],
                "robustness_model_contracts": [
                    contract(n=900, event_n=85, spec_id="alt_cohort")
                ],
                "output_files": {"robustness_matrix": "robustness_matrix.csv"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "steps" / "06_robustness_figure" / "outputs"
    assert (
        sensitivity_rescue(
            run_dir=tmp_path,
            current_step_id="06_robustness_figure",
            out_dir=out,
        )
        == "sensitivity_publication_bundle_from_parent_outputs_v2"
    )
    step = AnalysisStep(
        step_id="06_robustness_figure",
        intent="Render the figure declared by step '06_robustness'.",
    )
    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert (
        FigureSourceDataValidator().audit(
            step=step,
            out_dir=out,
            run_dir=tmp_path,
            step_summary=summary,
        )
        == []
    )

    figure_source = out / "sensitivity_forest_source_data.csv"
    tampered = pd.read_csv(figure_source)
    tampered.loc[tampered["spec_id"] == "alt_cohort", "model_id"] = "wrong_model"
    tampered.to_csv(figure_source, index=False)
    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out,
        run_dir=tmp_path,
        step_summary=summary,
    )
    assert findings
    assert any(
        "structured sensitivity-model trace" in finding.message
        or "not a traceable subset" in finding.message
        for finding in findings
    )
