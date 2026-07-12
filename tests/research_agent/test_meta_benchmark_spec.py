"""Coverage lint for the meta-generalization benchmark spec.

This validates the SPEC (``benchmarks/meta_generalization/meta_benchmark.jsonl``),
not a run: that it is schema-valid, spans all six generalisation axes, includes
enough fail-closed probes to exercise the gap-report ladder, and does not simply
re-test canonical-9 coordinates. The point of the benchmark is to catch
overfitting to the 9 questions, so the spec itself must stay diverse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from easyicu.research_agent.capability_registry import get_capability
from easyicu.research_agent.study_design_playbook import StudyDesignFamily

_SPEC = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "meta_generalization"
    / "meta_benchmark.jsonl"
)

_REQUIRED_KEYS = {
    "id",
    "title",
    "question",
    "database",
    "analysis_family",
    "exposure",
    "outcome",
    "time_origin",
    "missingness",
    "novelty",
    "expected_behavior",
    "expected_runner",
    "gap_level",
    "expected_gap_reason",
    "feasibility",
    "rationale",
}

# canonical-9 MIMIC-IV exposures a positive probe must not simply re-test
_CANONICAL_MIIV_EXPOSURES = (
    "vasopressor",
    "kdigo",
    "aki stage",
    "mechanical ventilation",
)


def _items():
    return [json.loads(line) for line in _SPEC.read_text().splitlines() if line.strip()]


def test_spec_file_exists_and_parses():
    assert _SPEC.exists(), f"missing spec: {_SPEC}"
    items = _items()
    assert len(items) >= 10


def test_every_item_has_the_required_schema():
    for it in _items():
        missing = _REQUIRED_KEYS - set(it)
        assert not missing, f"{it.get('id')} missing keys: {missing}"


def test_ids_are_unique():
    ids = [it["id"] for it in _items()]
    assert len(ids) == len(set(ids))


def test_enums_are_valid():
    families = set(get_args(StudyDesignFamily))
    for it in _items():
        assert it["expected_behavior"] in {"bound_result", "fail_closed"}, it["id"]
        assert it["feasibility"] in {
            "runnable_now",
            "needs_universe",
            "needs_database",
        }, it["id"]
        assert it["analysis_family"] in families, it["id"]
        assert it["expected_runner"] is None, (
            f"{it['id']} assigns a primary scientific runner; primary methods "
            "must remain agent-owned"
        )
        # the family must be documented in the capability registry
        assert get_capability(it["analysis_family"]) is not None, it["id"]


def test_all_six_analysis_families_are_represented():
    families = {it["analysis_family"] for it in _items()}
    assert families == set(get_args(StudyDesignFamily)), families


def test_databases_span_beyond_mimic():
    dbs = {it["database"] for it in _items()}
    assert len(dbs) >= 4, dbs
    non_mimic = dbs - {"miiv"}
    assert len(non_mimic) >= 3, f"need >=3 non-MIMIC databases, got {non_mimic}"


def test_each_generalisation_axis_is_exercised():
    all_novelty = {n for it in _items() for n in it["novelty"]}
    for axis in (
        "unseen_exposure",
        "unseen_outcome",
        "unseen_time_origin",
        "unseen_database",
        "unseen_missingness",
    ):
        assert axis in all_novelty, f"no item exercises {axis}"


def test_enough_fail_closed_probes_with_surfaced_reasons():
    probes = [it for it in _items() if it["expected_behavior"] == "fail_closed"]
    assert len(probes) >= 4, "need >=4 fail-closed probes to exercise the gap ladder"
    for p in probes:
        assert p["expected_gap_reason"], f"{p['id']} fail-closed but no expected reason"
        assert p["gap_level"] in {
            "runner_block",
            "gate_diagnostic_only",
            "runner_block_or_gate",
        }, p["id"]


def test_positive_probes_do_not_retest_canonical_mimic_questions():
    for it in _items():
        if it["expected_behavior"] != "bound_result" or it["database"] != "miiv":
            continue
        exp = it["exposure"].lower()
        for canon in _CANONICAL_MIIV_EXPOSURES:
            assert canon not in exp, f"{it['id']} re-tests canonical exposure '{canon}'"


def test_includes_an_explicit_unsupported_capability_probe():
    # MG12-style: the benchmark must keep an honest record of a known gap.
    novelty = {n for it in _items() for n in it["novelty"]}
    assert "unsupported_family" in novelty


def test_behavior_probe_preserves_agent_method_owner_without_runner_injection():
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    context = ResearchContext(
        research_question="Estimate a generic time-to-event contrast.",
        cohort=CohortDescriptor(
            cohort_name="neutral", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[],
        target_outcome="event",
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            AnalysisStep(
                step_id="agent_time_to_event_model",
                intent="Fit the agent-selected time-to-event model.",
                method="cox_proportional_hazards",
                expected_outputs=["table:hazard_ratio"],
            )
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(
        plan=plan, context=context
    )

    assert [step.step_id for step in revised.steps] == ["agent_time_to_event_model"]
    assert revised.steps[0].method == "cox_proportional_hazards"


def test_behavior_probe_fails_closed_with_a_structured_owner_reason():
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    context = ResearchContext(
        research_question="Estimate a generic time-to-event contrast.",
        cohort=CohortDescriptor(
            cohort_name="neutral", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[],
        target_outcome="event",
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            AnalysisStep(
                step_id="agent_non_survival_model",
                intent="Fit a fixed-endpoint association model.",
                method="mixed_effects_regression",
                expected_outputs=["table:association_estimates"],
            )
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(
        plan=plan, context=context
    )

    assert revised == plan
    assert any(
        finding.detail.get("missing_structured_owner") is True
        and finding.detail.get("family") == "survival"
        for finding in findings
    )
