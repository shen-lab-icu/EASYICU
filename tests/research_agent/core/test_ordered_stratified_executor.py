"""Focused owner tests for the typed ordered-stratified adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.aggregate_row import (
    unlabelled_aggregate_row_findings,
)
from easyicu.research_agent.contracts.ordered_stratified import (
    ordered_stratified_numeric_findings,
    ordered_stratified_script_findings,
    ordered_stratified_structure_findings,
)
from easyicu.research_agent.execution.runners.ordered_stratified_executor import (
    ordered_stratified_executor_owns_step,
    ordered_stratified_spec_for_step,
    run_ordered_stratified_from_env,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "ordered_trend",
            "planned_analysis_role": "secondary",
            "intent": "Compare an ordered exposure across two declared outcomes.",
            "inputs": [
                "artifact:analysis_cohort",
                "table:adjusted_association_estimates",
                "severity",
                "death",
                "duration",
            ],
            "expected_outputs": [
                "table:ordinal_trend_dose_response",
            ],
            "method": "ordinal_stratified_descriptive_analysis",
            "scientific_action_id": "association.ordinal_trend",
        }
    )


def _plan(step: AnalysisStep | None = None) -> AnalysisPlan:
    child = step or _step()
    primary = AnalysisStep.model_validate(
        {
            "step_id": "primary_model",
            "planned_analysis_role": "primary",
            "intent": "Fit the declared adjusted primary model.",
            "inputs": ["artifact:analysis_cohort", "severity", "death"],
            "expected_outputs": ["table:adjusted_association_estimates"],
            "method": "adjusted_association_models",
            "model_requirements": [
                {
                    "requirement_id": "primary",
                    "outcome": "death",
                    "outcome_type": "binary",
                    "method_family": "statsmodels_logit_mle",
                    "exposure_source": "severity",
                    "analysis_role": "primary",
                    "analysis_set": "source_aware",
                    "covariates": [],
                    "model_terms": [
                        {
                            "name": "severity",
                            "role": "exposure",
                            "coding": "ordinal_linear",
                            "levels": ["0", "1", "2"],
                            "reference_level": None,
                            "transform": "declared_level_index",
                        }
                    ],
                }
            ],
        }
    )
    return AnalysisPlan(research_question="Test", steps=[primary, child])


def _bind(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "ordered_trend" / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    resolved = {
        "step_id": "ordered_trend",
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": "cohort.parquet",
                "sha256": digest,
                "declared_kind": "artifact",
                "product": "analysis_cohort",
                "evidence_id": "ev-cohort",
                "identity_row": {
                    "input_key": "artifact:analysis_cohort",
                    "declared_kind": "artifact",
                    "product": "analysis_cohort",
                    "evidence_id": "ev-cohort",
                    "sha256": digest,
                },
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": len(frame),
                },
            }
        },
    }
    resolved_path = run_dir / "resolved_inputs.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    return out_dir, resolved_path


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "severity": [0, 0, 1, 1, 2, 2, None],
            "death": [0, 1, 0, 1, 1, 1, 0],
            "duration": [1.0, 2.0, 2.0, 4.0, 5.0, 8.0, 9.0],
        }
    )


def test_typed_owner_executes_and_replays_without_coder(monkeypatch, tmp_path: Path) -> None:
    step = _step()
    plan = _plan(step)
    out_dir, resolved_path = _bind(tmp_path, _frame())
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(resolved_path.parent))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))

    assert ordered_stratified_executor_owns_step(step, plan=plan)
    selected = select_standard_executor(step, plan=plan)
    assert selected is not None
    assert selected.analysis_kind == "ordered_stratified_analysis"
    assert selected.consumed_input_keys == (
        "artifact:analysis_cohort",
        "table:adjusted_association_estimates",
    )

    spec = ordered_stratified_spec_for_step(step, plan=plan)
    assert spec is not None
    summary = run_ordered_stratified_from_env(
        spec_payload=spec.model_dump(mode="json"),
        typed_cohort_input="artifact:analysis_cohort",
        analysis_role="secondary",
    )
    assert not ordered_stratified_structure_findings(step=step, step_summary=summary)
    assert not ordered_stratified_numeric_findings(
        cohort_path=resolved_path.parent / "cohort.parquet",
        step=step,
        out_dir=out_dir,
        step_summary=summary,
    )
    assert not ordered_stratified_script_findings(
        step=step, script_text=selected.code
    )
    stratified = pd.read_csv(out_dir / "ordered_stratified_outcomes.csv")
    assert set(stratified["row_role"]) == {"exposure_level"}
    assert not unlabelled_aggregate_row_findings(
        step_id=step.step_id,
        out_dir=out_dir,
    )
    reporting = summary["reportable_secondary_results"]
    assert reporting["ordered_exposure"] == "severity"
    assert reporting["continuous_outcome"] == "duration"
    assert reporting["interpretation_ceiling"] == (
        "secondary_unadjusted_not_causal"
    )
    assert reporting["continuous_level_summaries"] == [
        {"level": 0, "n": 2, "median": 1.5, "q25": 1.25, "q75": 1.75},
        {"level": 1, "n": 2, "median": 3.0, "q25": 2.5, "q75": 3.5},
        {"level": 2, "n": 2, "median": 6.5, "q25": 5.75, "q75": 7.25},
    ]
    assert reporting["continuous_trend"]["test_id"] == (
        "jonckheere_terpstra"
    )
    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": step.step_id,
                "status": "ok",
                "step_summary": summary,
            }
        ]
    )
    assert '"reportable_secondary_results"' in digest
    assert '"continuous_outcome": "duration"' in digest

    canonical_summary = json.loads(json.dumps(summary))
    canonical_summary.pop("interpretation_class")
    canonical_summary.pop("interpretation_ceiling")
    canonical_reporting = canonical_summary["reportable_secondary_results"]
    canonical_reporting.pop("schema_version")
    canonical_reporting.pop("interpretation_ceiling")
    canonical_contract = canonical_summary["ordered_stratified_contract"]
    canonical_contract.pop("schema_version")
    canonical_contract.pop("execution_owner")
    writer_store = EvidenceStore(tmp_path / "writer_authority")
    canonical_record = {
        "step_id": step.step_id,
        "status": "ok",
        "generation_mode": "deterministic_standard",
        "deterministic_standard_analysis": "ordered_stratified_analysis",
        "writer_result_envelope_evidence_id": "envelope_fixture",
        "step_summary": canonical_summary,
    }
    canonical_digest = _render_writer_evidence_digest(
        [canonical_record],
        run_dir=writer_store.root,
        include_robustness_panel=False,
        evidence=writer_store,
    )
    assert '"reportable_secondary_results"' in canonical_digest
    assert '"continuous_outcome": "duration"' in canonical_digest
    assert '"deterministic_standard_analysis": "ordered_stratified_analysis"' in canonical_digest

    unowned_record = dict(canonical_record)
    unowned_record.pop("deterministic_standard_analysis")
    unowned_digest = _render_writer_evidence_digest(
        [unowned_record],
        run_dir=writer_store.root,
        include_robustness_panel=False,
        evidence=writer_store,
    )
    assert '"reportable_secondary_results"' not in unowned_digest


def test_typed_owner_fails_closed_on_undeclared_exposure_level(
    monkeypatch, tmp_path: Path
) -> None:
    frame = _frame()
    frame.loc[0, "severity"] = 9
    out_dir, resolved_path = _bind(tmp_path, frame)
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(resolved_path.parent))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))

    with pytest.raises(RuntimeError, match="outside the declared level set"):
        spec = ordered_stratified_spec_for_step(_step(), plan=_plan())
        assert spec is not None
        run_ordered_stratified_from_env(
            spec_payload=spec.model_dump(mode="json"),
            typed_cohort_input="artifact:analysis_cohort",
            analysis_role="secondary",
        )


def test_method_label_without_typed_spec_does_not_select_owner() -> None:
    step = _step()
    assert not ordered_stratified_executor_owns_step(
        step, plan=AnalysisPlan(research_question="Test", steps=[step])
    )


def test_typed_owner_accepts_declared_order_from_primary_treatment_contrasts() -> None:
    step = _step()
    plan = _plan(step)
    primary = plan.steps[0]
    requirement = primary.model_requirements[0]
    exposure = requirement.model_terms[0]
    categorical_exposure = exposure.model_copy(
        update={
            "coding": "categorical",
            "transform": "treatment_contrast",
            "reference_level": "0",
        }
    )
    categorical_requirement = requirement.model_copy(
        update={
            "model_terms": [categorical_exposure],
            "exposure_levels": ["0", "1", "2"],
            "exposure_reference_level": "0",
            "primary_contrast_level": "2",
        }
    )
    categorical_primary = primary.model_copy(
        update={"model_requirements": [categorical_requirement]}
    )
    categorical_plan = plan.model_copy(update={"steps": [categorical_primary, step]})

    spec = ordered_stratified_spec_for_step(step, plan=categorical_plan)

    assert spec is not None
    assert spec.ordered_levels == ["0", "1", "2"]
    assert ordered_stratified_executor_owns_step(step, plan=categorical_plan)
