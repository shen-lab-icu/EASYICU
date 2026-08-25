"""Tests for the C1 / C3 / E2 fixes derived from pilot run 20260515.

* C1 — replanner/planner step-count cap (``max_total_steps``)
* C3 — byte-identical validator finding dedupe (``dedupe_findings``)
* E2 — per-step numeric-claim cap (``max_leaves`` param)

The pilot triage is at ``pilot_runs/run_20260515T144753_597802/TRIAGE.md``;
each test below maps 1-to-1 to a problem documented there.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# -------------------------- C3: dedupe findings --------------------------

def test_dedupe_findings_collapses_identical(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(validator="v", severity="error", message="X"),
        schema.ValidationFinding(validator="v", severity="error", message="X"),
        schema.ValidationFinding(validator="v", severity="error", message="X"),
    ]
    out = dedupe_findings(raw)
    assert len(out) == 1
    assert out[0].detail["duplicate_count"] == 3


def test_dedupe_findings_keeps_different_severities_separate(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(validator="v", severity="error", message="X"),
        schema.ValidationFinding(validator="v", severity="warning", message="X"),
    ]
    out = dedupe_findings(raw)
    # Same validator + message but different severity → 2 distinct entries.
    assert len(out) == 2


def test_dedupe_findings_merges_evidence_ids(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(
            validator="v", severity="error", message="X",
            evidence_ids=["a", "b"],
        ),
        schema.ValidationFinding(
            validator="v", severity="error", message="X",
            evidence_ids=["b", "c"],
        ),
    ]
    out = dedupe_findings(raw)
    assert len(out) == 1
    assert out[0].evidence_ids == ["a", "b", "c"]


def test_dedupe_findings_preserves_order(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(validator="a", severity="error", message="first"),
        schema.ValidationFinding(validator="b", severity="error", message="second"),
        schema.ValidationFinding(validator="a", severity="error", message="first"),
    ]
    out = dedupe_findings(raw)
    assert [f.message for f in out] == ["first", "second"]


def test_dedupe_findings_keeps_different_step_owners_separate(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(
            validator="v",
            severity="error",
            message="same contract failure",
            detail={"step_id": "step_a"},
        ),
        schema.ValidationFinding(
            validator="v",
            severity="error",
            message="same contract failure",
            detail={"step_id": "step_b"},
        ),
    ]

    out = dedupe_findings(raw)

    assert len(out) == 2
    assert [finding.detail["step_id"] for finding in out] == ["step_a", "step_b"]


# -------------------------- E2: claim cap --------------------------

def test_register_step_summary_numerics_respects_max_leaves(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    summary = {f"x_{i}": float(i) * 0.1 for i in range(20)}
    claims = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary, max_leaves=5,
    )
    # First 5 keep, 15 dropped (plus 1 overflow sentinel).
    assert len(claims) == 5
    overflows = [
        c for c in store.numeric_claims()
        if "overflow" in c.source_field
    ]
    assert len(overflows) == 1
    assert overflows[0].canonical == 15.0  # dropped count


def test_numeric_claim_cap_keeps_headline_fields_ahead_of_nested_diagnostics(
    ra,
    tmp_path: Path,
):
    store = ra.EvidenceStore(tmp_path)
    summary = {
        "model_contracts": {f"term_{index}": float(index) for index in range(120)},
        "primary_ci_low": 0.72,
        "primary_ci_high": 1.31,
        "primary_estimate": 0.97,
        "n_total": 24_819,
    }

    claims = store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary=summary,
        max_leaves=100,
    )

    source_fields = [claim.source_field for claim in claims]
    assert source_fields[:4] == [
        "primary_estimate",
        "primary_ci_low",
        "primary_ci_high",
        "n_total",
    ]
    assert "model_contracts.term_0" in source_fields
    assert "model_contracts.term_96" not in source_fields
    overflows = [
        claim
        for claim in store.numeric_claims()
        if claim.source_field == "__easyicu_numeric_claim_overflow__"
    ]
    assert len(overflows) == 1
    assert overflows[0].canonical == 24.0


def test_numeric_claim_cap_keeps_nested_prediction_metrics(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    summary = {
        "prediction_validation_receipt": {
            "result": {
                "calibration_bins": [
                    {f"diagnostic_{index}": float(index) for index in range(120)}
                ],
                "summary": {
                    "auroc": 0.763,
                    "brier_score": 0.072,
                    "calibration_intercept": -0.068,
                    "calibration_slope": 0.981,
                },
            }
        }
    }

    claims = store.register_step_summary_numerics(
        step_id="prediction",
        evidence_id="prediction_summary",
        summary=summary,
        max_leaves=10,
    )

    fields = {claim.source_field for claim in claims}
    assert {
        "prediction_validation_receipt.result.summary.auroc",
        "prediction_validation_receipt.result.summary.brier_score",
        "prediction_validation_receipt.result.summary.calibration_intercept",
        "prediction_validation_receipt.result.summary.calibration_slope",
    } <= fields


def test_register_step_summary_numerics_no_cap_when_none(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    summary = {f"x_{i}": float(i) for i in range(10)}
    claims = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary, max_leaves=None,
    )
    assert len(claims) == 10
    overflows = [
        c for c in store.numeric_claims()
        if "overflow" in c.source_field
    ]
    assert overflows == []


def test_register_step_summary_numerics_no_cap_when_zero(ra, tmp_path: Path):
    """Cap of 0 is treated as 'disabled' (matches PipelineConfig semantics)."""
    store = ra.EvidenceStore(tmp_path)
    summary = {f"x_{i}": float(i) for i in range(10)}
    claims = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary, max_leaves=0,
    )
    assert len(claims) == 10


# -------------------------- C1: pipeline max_total_steps --------------------------

def test_pipeline_max_total_steps_default(ra, tmp_path: Path):
    """The default cap on plan size is exposed and non-zero."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
    )
    assert pipeline._max_total_steps > 0
    # Documented default in pipeline_config. Raised 12 -> 16 on 2026-07-25: 12
    # was sized for an association task and bound in normal operation on the
    # four-product families (prediction / survival / causal / trajectory), where
    # truncation silently drops planned steps while the run still completes and
    # scores. 16 is still far below the 30 the 20260515 pilot runaway reached,
    # so the guard this cap exists for is intact.
    assert pipeline._max_total_steps == 16


def test_pipeline_max_total_steps_can_be_disabled(ra, tmp_path: Path):
    """Setting max_total_steps=0 turns off the cap (back-compat)."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
        max_total_steps=0,
    )
    assert pipeline._max_total_steps == 0


# ---------------- replan convergence guards (E1 20260611) ----------------

def _two_step_plan(intent_a: str, *, method: str = "logit"):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    return AnalysisPlan(
        research_question="Does sepsis-3 predict in-hospital mortality?",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent=intent_a,
                expected_outputs=["cohort_table"],
                method="filter",
            ),
            AnalysisStep(
                step_id="02_association",
                intent="Fit the adjusted model.",
                expected_outputs=["or_table"],
                method=method,
            ),
        ],
    )


def test_plan_signature_normalizes_only_intent_case_and_whitespace():
    """Cosmetic casing/spacing is a no-op; semantic prose is still authority."""
    from easyicu.research_agent.execution.phase import _plan_signature

    base = _two_step_plan("Define the adult ICU cohort.")
    reworded = _two_step_plan("  DEFINE   THE ADULT ICU COHORT. ")
    assert base.model_dump() != reworded.model_dump()
    assert _plan_signature(base) == _plan_signature(reworded)


def test_plan_signature_detects_scientific_intent_change():
    from easyicu.research_agent.execution.phase import _plan_signature

    mortality = _two_step_plan(
        "Use ICU mortality after baseline lactate in the first 24 hours."
    )
    readmission = _two_step_plan(
        "Use 30-day readmission after baseline creatinine in the first 48 hours."
    )

    assert _plan_signature(mortality) != _plan_signature(readmission)


def test_plan_signature_detects_icu_rule_change():
    from easyicu.research_agent.execution.phase import _plan_signature

    base = _two_step_plan("Define the adult ICU cohort.")
    changed = base.model_copy(deep=True)
    changed.steps[0].icu_rule_refs = ["time_zero_before_exposure"]

    assert _plan_signature(base) != _plan_signature(changed)


def test_plan_signature_detects_substantive_change():
    """A changed method / outputs is a real revision."""
    from easyicu.research_agent.execution.phase import _plan_signature

    base = _two_step_plan("Define the cohort.", method="logit")
    changed_method = _two_step_plan("Define the cohort.", method="cox")
    assert _plan_signature(base) != _plan_signature(changed_method)


def test_plan_signature_detects_artifact_edge_or_source_input_change():
    """Repairing a scientific DAG edge/window input is a real revision."""
    from easyicu.research_agent.execution.phase import _plan_signature

    base = _two_step_plan("Define the cohort.")
    changed = base.model_copy(deep=True)
    changed.steps[1].inputs = ["artifact:trajectory_features", "window_h6_12"]

    assert _plan_signature(base) != _plan_signature(changed)


def test_plan_signature_detects_estimand_role_change():
    """Changing a model from primary to secondary is not prose-only."""
    from easyicu.research_agent.execution.phase import _plan_signature
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    primary = AnalysisPlan(
        research_question="Compare two representations.",
        steps=[
            AnalysisStep(
                step_id="02_association",
                intent="Fit the primary exposure model.",
                expected_outputs=["or_table"],
                method="logit",
            )
        ],
    )
    secondary = AnalysisPlan(
        research_question="Compare two representations.",
        steps=[
            AnalysisStep(
                step_id="02_association",
                intent="Fit the secondary exposure model.",
                expected_outputs=["or_table"],
                method="logit",
            )
        ],
    )

    assert _plan_signature(primary) != _plan_signature(secondary)


def test_pipeline_replan_guards_defaults(ra, tmp_path: Path):
    """No-op early-stop stays on; the hard budget defaults to 6 (2026-07-06).

    The legacy default of 0 (budget off) let a non-converging run churn the
    replanner indefinitely — an E3 canonical run replanned 9× over ~50 min and
    still failed. The balanced default gives 6 substantive revisions of repair
    headroom, then fails closed to diagnostic_only.
    """
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
    )
    assert pipeline._max_consecutive_noop_replans == 2
    assert pipeline._max_replans == 6


def test_pipeline_replan_guards_can_be_disabled(ra, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
        max_consecutive_noop_replans=0,
        max_replans=0,
    )
    assert pipeline._max_consecutive_noop_replans == 0
    assert pipeline._max_replans == 0


def test_pipeline_max_numeric_claims_per_step_default(ra, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
    )
    assert pipeline._max_numeric_claims_per_step == 100
