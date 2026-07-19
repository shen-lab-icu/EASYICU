"""Fail-closed numeric integrity for agent-produced survival estimates.

The agent owns the method; this gate rejects impossible HR/CI shapes and event
counts without demanding a deterministic-runner fingerprint. It remains
case-neutral and ignores non-survival or prep-only steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.reporting.readiness import (
    primary_survival_estimate_integrity_errors,
)


def _plan(*steps: dict):
    return SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id=s.get("step_id"),
                method=s.get("method"),
                intent=s.get("intent"),
                expected_outputs=s.get("expected_outputs"),
            )
            for s in steps
        ]
    )


def _summary(run_dir: Path, step_id: str, payload: dict) -> None:
    out = run_dir / "steps" / step_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(json.dumps(payload))


_DET = {"fit_engine": "statsmodels.PHReg", "adjustment_source": "config"}


def test_impossible_event_count_is_flagged(tmp_path: Path):
    _summary(
        tmp_path,
        "01_survival_analysis",
        {"primary_model": {"hazard_ratio": 0.93}, "n": 73083, "events": 4635208},
    )
    errs = primary_survival_estimate_integrity_errors(
        _plan({"step_id": "01_survival_analysis", "method": "survival_analysis"}),
        tmp_path,
    )
    assert errs, "events greater than the analysis denominator must flag"
    assert "impossible event counts" in errs[0]


def test_deterministic_fingerprint_is_clean(tmp_path: Path):
    _summary(
        tmp_path,
        "01_survival_analysis",
        {**_DET, "hazard_ratio": 1.82, "n_events": 7063, "n_analysis": 74454},
    )
    assert (
        primary_survival_estimate_integrity_errors(
            _plan({"step_id": "01_survival_analysis", "method": "survival_analysis"}),
            tmp_path,
        )
        == []
    )


def test_deterministic_standard_analysis_marker_is_clean(tmp_path: Path):
    _summary(
        tmp_path,
        "01_survival_analysis",
        {
            "deterministic_standard_analysis": "survival_primary_cox",
            "hazard_ratio": 1.4,
        },
    )
    assert (
        primary_survival_estimate_integrity_errors(
            _plan({"step_id": "01_survival_analysis", "method": "survival_analysis"}),
            tmp_path,
        )
        == []
    )


def test_primary_deterministic_makes_secondary_llm_survival_step_clean(tmp_path: Path):
    # the design is anchored by a deterministic primary; a later LLM-coded
    # survival re-fit does not fail the whole run closed.
    _summary(tmp_path, "01_survival_analysis", {**_DET, "hazard_ratio": 1.82})
    _summary(tmp_path, "03_survival_robustness", {"hazard_ratio": 1.77})
    assert (
        primary_survival_estimate_integrity_errors(
            _plan(
                {"step_id": "01_survival_analysis", "method": "survival_analysis"},
                {"step_id": "03_survival_robustness", "method": "survival_analysis"},
            ),
            tmp_path,
        )
        == []
    )


def test_non_survival_question_is_never_touched(tmp_path: Path):
    # an association step that happens to report a ratio must not be judged by a
    # survival gate (case-neutral: no survival-method step -> no-op).
    _summary(tmp_path, "01_association", {"odds_ratio": 2.8, "primary_model": {}})
    assert (
        primary_survival_estimate_integrity_errors(
            _plan({"step_id": "01_association", "method": "association"}),
            tmp_path,
        )
        == []
    )


def test_valid_agent_survival_step_with_sensitivity_language_is_clean(
    tmp_path: Path,
):
    _summary(tmp_path, "01_survival_analysis", {"primary_model": {"hazard_ratio": 0.9}})
    assert (
        primary_survival_estimate_integrity_errors(
            _plan(
                {
                    "step_id": "01_survival_analysis",
                    "method": "survival_analysis",
                    "intent": "primary Cox with sensitivity across eligibility definitions",
                }
            ),
            tmp_path,
        )
        == []
    )


def test_nonpositive_hr_and_inverted_ci_are_flagged(tmp_path: Path):
    _summary(
        tmp_path,
        "01_survival_analysis",
        {
            "hazard_ratio": 0.0,
            "ci_low": 1.4,
            "ci_high": 0.8,
            "n_analysis": 100,
            "n_events": 20,
        },
    )

    errs = primary_survival_estimate_integrity_errors(
        _plan({"step_id": "01_survival_analysis", "method": "survival_analysis"}),
        tmp_path,
    )

    assert any("invalid hazard ratio" in error for error in errs)
    assert any("invalid HR confidence interval" in error for error in errs)


def test_cohort_sensitivity_step_without_a_cox_result_is_clean(tmp_path: Path):
    # a genuine cohort-definition-sensitivity step emits attrition/overlap, never
    # a primary Cox hazard_ratio/primary_model -> the result-key check filters it.
    _summary(
        tmp_path,
        "03_cohort_definition_sensitivity",
        {"alternative_cohort_attrition": [1, 2, 3], "cohort_overlap": 0.8},
    )
    assert (
        primary_survival_estimate_integrity_errors(
            _plan(
                {
                    "step_id": "03_cohort_definition_sensitivity",
                    "method": "survival_analysis",
                    "intent": "cohort definition sensitivity",
                }
            ),
            tmp_path,
        )
        == []
    )


def test_figure_step_is_ignored(tmp_path: Path):
    _summary(tmp_path, "01_survival_analysis_figure", {"primary_model": {"hr": 1.0}})
    assert (
        primary_survival_estimate_integrity_errors(
            _plan(
                {
                    "step_id": "01_survival_analysis_figure",
                    "method": "survival_analysis",
                }
            ),
            tmp_path,
        )
        == []
    )


def test_missing_summary_is_left_to_execution_gate(tmp_path: Path):
    # no step_summary on disk -> the execution gate handles missing/failed steps.
    assert (
        primary_survival_estimate_integrity_errors(
            _plan({"step_id": "01_survival_analysis", "method": "survival_analysis"}),
            tmp_path,
        )
        == []
    )


def test_none_inputs_are_clean(tmp_path: Path):
    assert primary_survival_estimate_integrity_errors(None, tmp_path) == []
    assert (
        primary_survival_estimate_integrity_errors(_plan(), None) == []  # type: ignore[arg-type]
    )
