from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.prediction_figure_executor import (
    PREDICTION_COMPOSITE_FIGURE_INPUTS,
    PREDICTION_FIGURE_ANALYSIS_KIND,
    prediction_composite_carries_validation,
    prediction_figure_executor_code,
    prediction_figure_executor_owns_step,
    run_prediction_figure,
)
from easyicu.research_agent.execution.runners.prediction_model_executor import (
    PREDICTION_MODEL_ANALYSIS_KIND,
    PREDICTION_SCORES_PRODUCT,
    prediction_model_executor_owns_step,
    run_prediction_model,
    run_prediction_robustness_specs,
    run_prediction_score_analysis,
)
from easyicu.research_agent.contracts.figure_plan import (
    STATIC_PREDICTION_FIGURE_PANELS,
    STATIC_PREDICTION_SPLIT_SURFACE_FIGURE_PANELS,
    STATIC_PREDICTION_VALIDATION_FIGURE_PANELS,
    STATIC_PREDICTION_VALIDATION_FIGURE_SUFFIX,
)
from easyicu.research_agent.contracts.prediction_execution import (
    static_prediction_model_columns,
)
from easyicu.research_agent.execution.figure_plan_binding import (
    validate_step_planned_figure_contract_binding,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.execution.final_validation import (
    _primary_runner_core_estimate_present,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.figure_plan_shaping import (
    bind_deterministic_figure_panels,
)
from easyicu.research_agent.reporting.readiness import (
    _deterministic_primary_estimate_bound,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(row_count: int) -> ResearchContext:
    return ResearchContext(
        research_question="Predict a binary outcome from prespecified variables.",
        cohort=CohortDescriptor(
            cohort_name="prediction_fixture",
            database="synthetic",
            n_stays=row_count,
            id_columns=["patient_stay_id"],
            outcome_columns=["death"],
            provenance={
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "a" * 64,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                }
            },
        ),
        variables=[
            ConceptDescriptor(name="age", dtype="float64"),
            ConceptDescriptor(
                name="sex",
                dtype="object",
                observed_domain={"n_unique": 2, "levels": ["F", "M"]},
            ),
            ConceptDescriptor(name="marker", dtype="float64"),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
        ],
        target_outcome="death",
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    subject_count = 90
    stays_per_subject = np.where(np.arange(subject_count) % 4 == 0, 2, 1)
    subjects = np.repeat(np.arange(subject_count), stays_per_subject)
    stay_number = np.concatenate([np.arange(count) + 1 for count in stays_per_subject])
    age = rng.normal(64, 12, len(subjects))
    marker = rng.normal(0, 1, len(subjects))
    sex = np.where(subjects % 2 == 0, "F", "M")
    logit = -2.0 + 0.025 * (age - 60) + 0.7 * marker + 0.25 * (sex == "M")
    probability = 1.0 / (1.0 + np.exp(-logit))
    death = rng.binomial(1, probability)
    # Keep both classes abundant so the one fixed group split is executable.
    death[:10] = np.arange(10) % 2
    frame = pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{subject}:s{stay}"
                for subject, stay in zip(subjects, stay_number, strict=True)
            ],
            "age": age,
            "sex": sex,
            "marker": marker,
            "death": death,
        }
    )
    frame.loc[frame.index[::13], "marker"] = np.nan
    frame.loc[frame.index[::17], "sex"] = None
    return frame


def _binding(key: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": str(path.relative_to(path.parents[1])),
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "produced_by_step": f"source_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
        "identity_row": {
            "input_key": key,
            "declared_kind": "table",
            "product": product,
            "evidence_id": f"evidence_{product}",
            "produced_by_step": f"source_{product}",
            "sha256": digest,
        },
    }


def _primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_model",
        planned_analysis_role="primary",
        intent="Fit the prespecified static mortality model.",
        inputs=["age", "sex", "marker", "death", "artifact:analysis_cohort"],
        expected_outputs=["table:prediction_scores", "table:model_performance"],
        method="logistic prediction model",
        scientific_action_id="prediction.discrimination_calibration",
    )


def test_prediction_owner_selects_only_exact_action_contract() -> None:
    step = _primary_step()
    assert prediction_model_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Predict mortality.", steps=[step]),
    )
    assert selection is not None
    assert selection.analysis_kind == PREDICTION_MODEL_ANALYSIS_KIND
    assert selection.consumed_input_keys == ("artifact:analysis_cohort",)
    widened = step.model_copy(
        update={"expected_outputs": [*step.expected_outputs, "table:extra"]}
    )
    assert not prediction_model_executor_owns_step(widened)


def test_prediction_model_roster_stops_at_typed_cohort_boundary() -> None:
    step = _primary_step().model_copy(
        update={
            "inputs": [
                "age",
                "sex",
                "marker",
                "death",
                "artifact:analysis_cohort",
                "marker_measured",
                "marker_n",
            ]
        }
    )
    assert prediction_model_executor_owns_step(step)
    assert static_prediction_model_columns(step) == ("age", "sex", "marker", "death")


@pytest.mark.parametrize(
    "separate_surface",
    [False, True],
    ids=["validation_on_composite", "reviewed_separate_surface"],
)
def test_prediction_workflow_is_group_safe_source_bound_and_renderable(
    tmp_path: Path, separate_surface: bool
) -> None:
    frame = _frame()
    (tmp_path / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    primary_dir = tmp_path / "primary"
    primary = run_prediction_model(
        frame=frame,
        declared_columns=("age", "sex", "marker", "death"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=primary_dir,
        run_dir=tmp_path,
        step_id="primary_model",
    )
    assert primary["authority_scope"] == "analysis_only"
    assert primary["predictor_roster"] == ["age", "sex", "marker"]
    assert primary["scientific_validation_contract"] == "PredictionValidationReceipt"
    assert primary["prediction_validation_receipt"]["paper_authorization"] is False
    assert _primary_runner_core_estimate_present(
        PREDICTION_MODEL_ANALYSIS_KIND, primary
    )
    assert _deterministic_primary_estimate_bound(
        [
            {
                "step_id": "primary_model",
                "deterministic_standard_analysis": PREDICTION_MODEL_ANALYSIS_KIND,
                "step_summary": primary,
            }
        ]
    )
    tampered = json.loads(json.dumps(primary))
    tampered["prediction_validation_receipt"]["result"]["summary"]["auroc"] = 0.5
    assert not _primary_runner_core_estimate_present(
        PREDICTION_MODEL_ANALYSIS_KIND, tampered
    )
    scores = pd.read_csv(primary_dir / "prediction_scores.csv")
    assert scores.groupby("subject_id")["split"].nunique().max() == 1
    assert set(scores["split"]) == {"development", "validation"}
    performance = pd.read_csv(primary_dir / "prediction_performance.csv").iloc[0]
    assert performance["predictor_n"] == 3
    assert performance["auroc_ci_low"] <= performance["auroc"]
    assert performance["auroc_ci_high"] >= performance["auroc"]
    assert performance["repeated_split_n"] == 10
    assert performance["repeated_split_auroc_sd"] >= 0
    assert performance["repeated_split_average_precision_sd"] >= 0
    assert performance["repeated_split_brier_sd"] >= 0
    repeats = json.loads(performance["repeated_split_results"])
    assert len(repeats) == 10
    assert [row["split_seed"] for row in repeats] == list(range(1730, 1740))
    assert all(row["patient_overlap_n"] == 0 for row in repeats)
    assert primary["resampling_validation"]["n_repeats"] == 10
    assert primary["resampling_validation"]["all_patient_overlap_zero"] is True
    assert primary["resampling_validation"]["external_validation_established"] is False

    score_binding = _binding(
        PREDICTION_SCORES_PRODUCT,
        scores,
        primary_dir / "prediction_scores.csv",
    )
    product_paths = {
        "table:prediction_scores": primary_dir / "prediction_scores.csv",
        "table:model_performance": primary_dir / "prediction_performance.csv",
    }
    for action, product, folder in (
        ("prediction.internal_validation", "table:validation", "validation"),
        ("prediction.calibration_metrics", "table:calibration", "calibration"),
        ("prediction.decision_curve", "table:clinical_utility", "utility"),
    ):
        out_dir = tmp_path / folder
        summary = run_prediction_score_analysis(
            action_id=action,
            out_dir=out_dir,
            run_dir=tmp_path,
            resolved_inputs={
                "step_id": folder,
                "inputs": {PREDICTION_SCORES_PRODUCT: score_binding},
            },
            step_id=folder,
        )
        assert summary["output_files"].keys() == {product}
        product_paths[product] = out_dir / next(iter(summary["output_files"].values()))

    figure_bindings = {}
    for key in PREDICTION_COMPOSITE_FIGURE_INPUTS:
        source = pd.read_csv(product_paths[key])
        figure_bindings[key] = _binding(key, source, product_paths[key])
    figure_step = AnalysisStep(
        step_id="prediction_figure",
        planned_analysis_role="auxiliary",
        intent="Render prediction performance and calibration.",
        inputs=[
            "table:prediction_scores",
            "table:model_performance",
            "table:calibration",
            "table:validation",
            "table:clinical_utility",
        ],
        # A plan reviewed before the composite carried validation declares the
        # renderer's separate repeated-split surface; a current plan does not.
        expected_outputs=[
            "figure:prediction_figure",
            *(
                ["figure:prediction_figure_validation_stability"]
                if separate_surface
                else []
            ),
        ],
        method="visualization",
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"}
            for key in PREDICTION_COMPOSITE_FIGURE_INPUTS
        ],
    )
    assert prediction_figure_executor_owns_step(
        figure_step, resolved_bindings=figure_bindings
    )
    selection = select_standard_executor(
        figure_step,
        plan=AnalysisPlan(research_question="Predict mortality.", steps=[figure_step]),
        resolved_bindings=figure_bindings,
    )
    assert selection is not None
    assert selection.analysis_kind == PREDICTION_FIGURE_ANALYSIS_KIND
    assert selection.host_sealed_renderer is True
    # The plan-time panel promise projected by shaping for this exact input
    # set is what the code generator reads and the host renderer implements,
    # or the end-of-execute join fails closed.
    shaped, panel_findings = bind_deterministic_figure_panels(
        plan=AnalysisPlan(research_question="Predict mortality.", steps=[figure_step])
    )
    planned = shaped.steps[0]
    assert prediction_composite_carries_validation(planned) is not separate_surface
    assert (
        f"validation_on_composite={not separate_surface!r}"
        in prediction_figure_executor_code(planned)
    )
    figure_dir = tmp_path / "figure"
    summary = run_prediction_figure(
        out_dir=figure_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": figure_step.step_id, "inputs": figure_bindings},
        step_id=figure_step.step_id,
        figure_product="prediction_figure",
        validation_on_composite=not separate_surface,
    )
    assert summary["status"] == "ok"
    assert summary["paper_authorization_allowed"] is False
    surfaces = ["prediction_figure"]
    if separate_surface:
        surfaces.append("prediction_figure_validation_stability")
    else:
        assert not list(figure_dir.glob("prediction_figure_validation_stability.*"))
    for suffix in ("png", "svg", "pdf", "tiff", "figure_contract.json"):
        for stem in (*surfaces, "prediction_figure_supplementary_decision_curve"):
            assert (figure_dir / f"{stem}.{suffix}").is_file()
    contract = json.loads(
        (figure_dir / "prediction_figure.figure_contract.json").read_text("utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "model_performance",
        "model_performance",
        "calibration",
        *([] if separate_surface else ["validation"]),
    ]
    assert {panel["metadata"]["placement"] for panel in contract["panels"]} == {"main"}
    # The composite is the figure the article leads with; on its own it must
    # satisfy the prediction strategy (calibration hero and three roles).
    from easyicu.research_agent.figures.skill import _contract_primary_strategy_ready

    assert _contract_primary_strategy_ready(_context(len(frame)), contract) is (
        not separate_surface
    )
    if separate_surface:
        validation_contract = json.loads(
            (
                figure_dir / "prediction_figure_validation_stability.figure_contract.json"
            ).read_text("utf-8")
        )
        assert [panel["role"] for panel in validation_contract["panels"]] == [
            "validation",
            "validation",
        ]
        assert validation_contract["panels"][0]["metadata"]["article_role"] == (
            "validation_design"
        )
        assert validation_contract["panels"][1]["metadata"]["chart_type"] == (
            "metric_dot_interval"
        )
    else:
        assert contract["panels"][3]["metadata"]["chart_type"] == "metric_dot_interval"
    assert summary["main_figure_paths"] == [f"{stem}.png" for stem in surfaces]
    assert set(summary["output_files"]) == {f"figure:{stem}" for stem in surfaces}
    supplementary_contract = json.loads(
        (
            figure_dir
            / "prediction_figure_supplementary_decision_curve.figure_contract.json"
        ).read_text("utf-8")
    )
    assert supplementary_contract["panels"][0]["role"] == "clinical_utility"
    assert supplementary_contract["panels"][0]["metadata"]["placement"] == (
        "supplementary"
    )
    decision_curve = pd.read_csv(figure_dir / "clinical_utility_source_data.csv")
    assert list(decision_curve.columns) == [
        "source_table",
        "source_step_id",
        "source_row_index",
        "threshold",
        "n",
        "net_benefit_model",
        "net_benefit_all",
        "net_benefit_none",
    ]
    assert decision_curve["source_table"].eq("clinical_utility.csv").all()
    assert decision_curve["source_step_id"].eq("source_clinical_utility").all()
    assert decision_curve["source_row_index"].tolist() == list(
        range(len(decision_curve))
    )
    assert decision_curve["threshold"].between(0.01, 0.50).all()
    assert "calibration" in contract["statistics_note"].lower()
    assert "clinical benefit" in contract["statistics_note"].lower()

    templates = (
        (*STATIC_PREDICTION_SPLIT_SURFACE_FIGURE_PANELS, *STATIC_PREDICTION_VALIDATION_FIGURE_PANELS)
        if separate_surface
        else STATIC_PREDICTION_FIGURE_PANELS
    )
    assert [
        (panel.panel_id, panel.article_role, panel.chart_type)
        for panel in planned.figure_panels
    ] == [
        (template.panel_id, template.article_role, template.chart_type)
        for template in templates
    ]
    # One exported image is one surface: a declared repeated-split surface
    # carries the validation panels, otherwise they join the composite, and
    # never both.
    assert {panel.figure_output for panel in planned.figure_panels} == {
        f"figure:{stem}" for stem in surfaces
    }
    assert [
        panel.figure_output
        for panel in planned.figure_panels
        if panel.article_role == "validation"
    ] == [
        "figure:prediction_figure_validation_stability"
        if separate_surface
        else "figure:prediction_figure"
    ]
    if separate_surface:
        assert {
            panel.article_role
            for panel in planned.figure_panels
            if panel.figure_output.endswith(STATIC_PREDICTION_VALIDATION_FIGURE_SUFFIX)
        } == {"validation_design", "validation"}
    assert [finding.detail["reason"] for finding in panel_findings] == [
        "deterministic_figure_panels_bound"
    ]
    assert (
        validate_step_planned_figure_contract_binding(
            step=planned, out_dir=figure_dir, step_summary=summary
        )
        == []
    )


def test_prediction_figure_shape_binds_registered_clinical_utility() -> None:
    core_inputs = [
        "table:prediction_scores",
        "table:model_performance",
        "table:calibration",
        "table:validation",
    ]
    utility = AnalysisStep(
        step_id="clinical_utility",
        planned_analysis_role="auxiliary",
        intent="Estimate net benefit over prespecified thresholds.",
        inputs=["table:prediction_scores"],
        expected_outputs=["table:clinical_utility"],
        scientific_action_id="prediction.decision_curve",
        method="decision_curve",
    )
    figure = AnalysisStep(
        step_id="prediction_figure",
        planned_analysis_role="auxiliary",
        intent="Render the complete validation evidence suite.",
        inputs=core_inputs,
        expected_outputs=["figure:prediction_figure"],
        method="visualization",
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"} for key in core_inputs
        ],
    )
    shaped, findings = bind_deterministic_figure_panels(
        plan=AnalysisPlan(
            research_question="Predict mortality.",
            steps=[utility, figure],
        )
    )

    shaped_figure = shaped.steps[1]
    assert shaped_figure.inputs == [*core_inputs, "table:clinical_utility"]
    assert [item.input_key for item in shaped_figure.input_consumption_contracts] == [
        *core_inputs,
        "table:clinical_utility",
    ]
    assert findings[0].detail["reason"] == ("prediction_figure_clinical_utility_bound")


def test_prediction_owner_executes_exact_complete_case_robustness_spec(
    tmp_path: Path,
) -> None:
    frame = _frame()
    (tmp_path / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    out_dir = tmp_path / "primary"
    run_prediction_model(
        frame=frame,
        declared_columns=("age", "sex", "marker", "death"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=out_dir,
        run_dir=tmp_path,
        step_id="primary_model",
    )
    scores = pd.read_csv(out_dir / "prediction_scores.csv")
    groups = frame["patient_stay_id"].str.split(":s", n=1).str[0]
    rows, results = run_prediction_robustness_specs(
        frame=frame,
        outcome=frame["death"],
        groups=groups,
        unit_ids=scores["unit_id"],
        split=scores["split"].to_numpy(),
        features=("age", "sex", "marker"),
        specs=[
            RobustnessSpec(
                spec_id="complete_case_declared_model",
                axis="missing",
                description="Complete-case refit of the exact model roster.",
                missing_override={
                    "strategy": "complete_case",
                    "variables": ["age", "sex", "marker", "death"],
                },
            )
        ],
    )
    assert len(rows) == len(results) == 1
    assert rows[0]["converged"] is True
    assert rows[0]["ci_low"] <= rows[0]["point_estimate"] <= rows[0]["ci_high"]
    assert results[0]["analysis"] == "complete_case_refit_same_patient_split"
    assert results[0]["authority_scope"] == "analysis_only"

    # The run-level panel reads the owner's rows from its step summary, so the
    # locked spec is executed -- not a blank row the finaliser must refuse.
    from easyicu.research_agent.robustness.panel import (
        build_robustness_panel_from_records,
        unexecuted_locked_spec_ids,
    )

    spec = RobustnessSpec(
        spec_id="complete_case_declared_model",
        axis="missing",
        description="Complete-case refit of the exact model roster.",
        missing_override={
            "strategy": "complete_case",
            "variables": ["age", "sex", "marker", "death"],
        },
    )
    panel = build_robustness_panel_from_records(
        specs=[spec],
        per_step_records=[
            {
                "step_id": "primary_model",
                "status": "ok",
                "step_summary": {"robustness_rows": rows},
                "step_summary_evidence_id": "primary_model_summary",
            }
        ],
    )
    assert unexecuted_locked_spec_ids(panel) == []
    executed = [row for row in panel.rows if row.spec_id == spec.spec_id]
    assert len(executed) == 1 and executed[0].converged
    assert executed[0].evidence_id == "primary_model_summary"

    # A narrower complete-case set is another analysis: the owner leaves it
    # unexecuted and the panel keeps it visible as a blank locked row.
    narrower = RobustnessSpec(
        spec_id="complete_case_narrower",
        axis="missing",
        description="Complete-case refit on a subset of the roster.",
        missing_override={"strategy": "complete_case", "variables": ["age", "death"]},
    )
    narrower_rows, _ = run_prediction_robustness_specs(
        frame=frame,
        outcome=frame["death"],
        groups=groups,
        unit_ids=scores["unit_id"],
        split=scores["split"].to_numpy(),
        features=("age", "sex", "marker"),
        specs=[narrower],
    )
    assert narrower_rows == []
    blank = build_robustness_panel_from_records(
        specs=[narrower],
        per_step_records=[
            {
                "step_id": "primary_model",
                "status": "ok",
                "step_summary": {"robustness_rows": narrower_rows},
                "step_summary_evidence_id": "primary_model_summary",
            }
        ],
    )
    assert unexecuted_locked_spec_ids(blank) == ["complete_case_narrower"]
