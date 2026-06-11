"""Tests for the §M1 five-dimension Tier-1 scorecard bridge.

Deterministic: every assertion is recomputed from synthetic readiness
artifacts, exercising the bridge the same way the manuscript's Fig.3
scorecard will be built.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from easyicu.research_agent import evaluation_scorecard as sc
from easyicu.research_agent.icu_agent_bench import (
    ICUAgentBenchGoldAnswer,
    ICUAgentBenchNumericBound,
    ICUAgentBenchTask,
)


def _gates(
    *,
    required=4,
    completed=4,
    failed=None,
    execution_complete=True,
    evidence_complete=True,
    numeric_verified=True,
    missing_evidence=0,
    manuscript_ready=True,
):
    return {
        "required_step_count": required,
        "completed_step_count": completed,
        "failed_steps": failed or [],
        "execution_complete": execution_complete,
        "evidence_complete": evidence_complete,
        "numeric_verified": numeric_verified,
        "missing_evidence_count": missing_evidence,
        "manuscript_ready": manuscript_ready,
    }


def _task(difficulty="intermediate", *, gold=None, outputs=None):
    return ICUAgentBenchTask(
        task_id="E1_demo",
        kind="cohort_extraction",
        title="demo",
        objective="demo",
        expected_outputs=outputs
        or ["table one", "forest plot figure", "component completeness audit"],
        difficulty=difficulty,
        gold_answer=gold,
        gold_answer_status="frozen" if gold else "planned",
    )


# ---------------------------------------------------------------------------
# bin_level
# ---------------------------------------------------------------------------


def test_bin_level_thresholds():
    assert sc.bin_level(1.0) == "Full"
    assert sc.bin_level(0.85) == "Full"
    assert sc.bin_level(0.6) == "Partial"
    assert sc.bin_level(0.3) == "Marginal"
    assert sc.bin_level(0.0) == "Fail"


# ---------------------------------------------------------------------------
# tristate
# ---------------------------------------------------------------------------


def test_tristate_gate_reportable():
    assert sc.compute_tristate(_gates()) == "gate_reportable"


def test_tristate_analysis_only_when_not_manuscript_ready():
    g = _gates(manuscript_ready=False)
    assert sc.compute_tristate(g) == "analysis_only"


def test_tristate_diagnostic_only_when_execution_incomplete():
    g = _gates(execution_complete=False, manuscript_ready=False)
    assert sc.compute_tristate(g) == "diagnostic_only"


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_plan_full_when_display_set_complete():
    dim = sc.score_plan(_task(), plan_steps=[{"intent": "x"}], gates=_gates())
    assert dim.level == "Full"
    assert dim.signals["has_table_one"] and dim.signals["has_audit_panel"]


def test_plan_fail_on_empty_plan():
    dim = sc.score_plan(_task(), plan_steps=[], gates=_gates(required=0))
    assert dim.level == "Fail"


def test_plan_fail_when_illegal():
    dim = sc.score_plan(
        _task(), plan_steps=[{"intent": "x"}], gates=_gates(), plan_illegal=True
    )
    assert dim.level == "Fail"


def test_plan_hard_task_needs_two_figures():
    # advanced task with only one figure hint -> figure requirement unmet
    task = _task(
        difficulty="advanced", outputs=["table one", "forest plot", "leakage audit"]
    )
    dim = sc.score_plan(task, plan_steps=[{"intent": "x"}], gates=_gates())
    assert dim.signals["min_result_figures"] == 2
    assert dim.signals["result_figure_count"] == 1
    assert dim.level != "Full"


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


def test_code_full_when_all_steps_ok():
    dim = sc.score_code(gates=_gates())
    assert dim.level == "Full"
    assert dim.subscore == 1.0


def test_code_partial_on_failed_step():
    g = _gates(
        completed=3,
        failed=[{"step_id": "03", "status": "failed"}],
        execution_complete=False,
    )
    dim = sc.score_code(gates=g)
    assert dim.level in {"Partial", "Marginal"}
    assert dim.level != "Full"


def test_code_fail_when_nothing_completed():
    g = _gates(
        completed=0,
        failed=[{"step_id": "01", "status": "failed"}],
        execution_complete=False,
    )
    dim = sc.score_code(gates=g)
    assert dim.level == "Fail"


# ---------------------------------------------------------------------------
# result_validity
# ---------------------------------------------------------------------------


def test_result_validity_unscored_without_locked_reference():
    dim = sc.score_result_validity(
        _task(), numeric_audit={"numeric_verified": True}, observed_metrics={"or": 0.8}
    )
    assert dim.subscore is None and dim.level is None


def test_result_validity_full_in_bound_and_verified():
    gold = ICUAgentBenchGoldAnswer(
        numeric_targets={"or": ICUAgentBenchNumericBound(lower=0.7, upper=0.9)}
    )
    dim = sc.score_result_validity(
        _task(gold=gold),
        numeric_audit={"numeric_verified": True, "numeric_error_count": 0},
        observed_metrics={"or": 0.8},
        locked_reference_frozen=True,
    )
    assert dim.level == "Full"
    assert dim.subscore == 1.0


def test_result_validity_fail_out_of_bound():
    gold = ICUAgentBenchGoldAnswer(
        numeric_targets={"or": ICUAgentBenchNumericBound(lower=0.7, upper=0.9)}
    )
    dim = sc.score_result_validity(
        _task(gold=gold),
        numeric_audit={"numeric_verified": True},
        observed_metrics={"or": 1.6},
        locked_reference_frozen=True,
    )
    assert dim.level == "Fail"


def test_result_validity_fails_on_validity_error_without_locked_reference():
    # An objective validity flaw (overadjustment) caps the dimension at Fail
    # even with no locked reference -- distinct from the honest *unscored*
    # state, so the gold-free score is failed only when warranted, not faked.
    dim = sc.score_result_validity(
        _task(),
        numeric_audit={"numeric_verified": True},
        observed_metrics={"or": 0.8},
        validity_errors=["overadjustment: adjusted for sofa_max"],
    )
    assert dim.level == "Fail"
    assert dim.subscore == 0.0
    assert dim.signals["validity_errors"] == ["overadjustment: adjusted for sofa_max"]


# ---------------------------------------------------------------------------
# evidence_binding
# ---------------------------------------------------------------------------


def test_evidence_binding_full_when_complete_and_kinds_present():
    dim = sc.score_evidence_binding(
        evidence_audit={
            "evidence_complete": True,
            "missing_evidence_count": 0,
            "kinds": {"table": 1, "figure": 1, "metric": 1, "cohort": 1, "model": 1},
        },
        numeric_audit={"numeric_verified": True},
        claim_rows=[{"status": "bound"}, {"status": "bound"}],
    )
    assert dim.level == "Full"


def test_evidence_binding_partial_when_unbound_demoted():
    dim = sc.score_evidence_binding(
        evidence_audit={
            "evidence_complete": False,
            "missing_evidence_count": 1,
            "kinds": {},
        },
        numeric_audit={"numeric_verified": True},
        claim_rows=[{"status": "bound"}, {"status": "demoted"}],
    )
    assert dim.level == "Partial"


def test_evidence_binding_low_when_unbound_not_demoted():
    dim = sc.score_evidence_binding(
        evidence_audit={
            "evidence_complete": False,
            "missing_evidence_count": 3,
            "kinds": {},
        },
        numeric_audit={"numeric_verified": False},
        claim_rows=[{"status": "unbound"}, {"status": "unbound"}],
    )
    assert dim.level in {"Fail", "Marginal"}


# ---------------------------------------------------------------------------
# audit_conclusion_safety
# ---------------------------------------------------------------------------


def test_audit_safety_full_when_hazard_hit_and_no_forbidden():
    gold = ICUAgentBenchGoldAnswer(
        required_warnings=["immortal time"],
        forbidden_outputs=["causal effect"],
    )
    dim = sc.score_audit_conclusion_safety(
        _task(gold=gold),
        observed_warnings=["beware immortal time bias"],
        observed_outputs=["association, not causal claim"],
        tristate="gate_reportable",
    )
    assert dim.level == "Full"


def test_audit_safety_fail_when_forbidden_leaks():
    gold = ICUAgentBenchGoldAnswer(
        required_warnings=["immortal time"],
        forbidden_outputs=["causal effect"],
    )
    dim = sc.score_audit_conclusion_safety(
        _task(gold=gold),
        observed_warnings=["beware immortal time bias"],
        observed_outputs=["we proved a causal effect of X on death"],
        tristate="gate_reportable",
    )
    assert dim.level == "Fail"
    assert dim.signals["forbidden_conclusion_leaked"] is True


def test_audit_safety_surfaces_cohort_hygiene_without_penalty():
    """Cohort-hygiene cautions are recorded but must not lower the subscore.

    Penalising them would punish a structural no-source export limitation
    (no patient id) or a defensible analytical choice (short-stay handling),
    which the impartiality rule forbids. They surface for the reader; the
    score-impact lands with the §M2 manuscript-engagement wiring.
    """
    gold = ICUAgentBenchGoldAnswer(required_warnings=["immortal time"])
    common = dict(
        observed_warnings=["beware immortal time bias"],
        tristate="gate_reportable",
    )
    baseline = sc.score_audit_conclusion_safety(_task(gold=gold), **common)
    with_cautions = sc.score_audit_conclusion_safety(
        _task(gold=gold),
        cohort_hygiene_cautions=[
            "Cohort is keyed at the ICU-stay level with no patient identifier...",
            "21% of stays have ICU length-of-stay <1 day...",
        ],
        **common,
    )
    # Same subscore/level — recorded, not penalised.
    assert with_cautions.subscore == baseline.subscore
    assert with_cautions.level == baseline.level
    assert len(with_cautions.signals["cohort_hygiene_cautions"]) == 2
    assert baseline.signals["cohort_hygiene_cautions"] == []


# ---------------------------------------------------------------------------
# score_run / score_run_from_dir
# ---------------------------------------------------------------------------


def test_score_run_produces_full_scorecard_and_source_row():
    card = sc.score_run(
        _task(),
        gates=_gates(),
        plan_steps=[{"intent": "table one + forest plot figure + completeness audit"}],
        evidence_audit={
            "evidence_complete": True,
            "missing_evidence_count": 0,
            "kinds": {"table": 1, "figure": 1, "metric": 1, "cohort": 1, "model": 1},
        },
        numeric_audit={"numeric_verified": True},
        claim_rows=[{"status": "bound"}],
        run_id="run_x",
    )
    assert card.tristate == "gate_reportable"
    assert card.code.level == "Full"
    assert card.evidence_binding.level == "Full"
    assert card.result_validity.level is None  # unscored without locked ref
    row = card.source_data_row()
    assert row["task_id"] == "E1_demo"
    assert row["code__level"] == "Full"
    assert row["tristate"] == "gate_reportable"
    assert "result_validity__subscore" in row


def test_score_run_from_dir_roundtrip(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": _gates()}), encoding="utf-8"
    )
    (run_dir / "analysis_plan.json").write_text(
        json.dumps(
            {"steps": [{"intent": "table one + forest figure + completeness audit"}]}
        ),
        encoding="utf-8",
    )
    (run_dir / "evidence_audit.json").write_text(
        json.dumps(
            {
                "evidence_complete": True,
                "missing_evidence_count": 0,
                "kinds": {
                    "table": 1,
                    "figure": 1,
                    "metric": 1,
                    "cohort": 1,
                    "model": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "numeric_audit.json").write_text(
        json.dumps({"numeric_verified": True}), encoding="utf-8"
    )
    with (run_dir / "claim_ledger.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim_id", "status"])
        w.writeheader()
        w.writerow({"claim_id": "c1", "status": "bound"})

    card = sc.score_run_from_dir(_task(), run_dir, run_id="run_dir_x")
    assert card.tristate == "gate_reportable"
    assert card.code.level == "Full"
    assert card.plan.level == "Full"


def test_score_run_from_dir_flags_overadjustment(tmp_path: Path):
    # E1-style: a regression that conditions on SOFA while studying Sepsis-3
    # (which is defined via SOFA) is overadjustment -> gold-free Fail.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": _gates()}), encoding="utf-8"
    )
    with (run_dir / "regression_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "coef"])
        w.writeheader()
        w.writerow({"variable": "const", "coef": "0.1"})
        w.writerow({"variable": "age", "coef": "0.02"})
        w.writerow({"variable": "sofa_max", "coef": "0.5"})

    card = sc.score_run_from_dir(
        _task(), run_dir, exposure_concept="sepsis3", run_id="oa"
    )
    assert card.result_validity.level == "Fail"
    assert card.result_validity.subscore == 0.0
    assert any("overadjustment" in n for n in card.result_validity.notes)

    # Without the exposure declared the check stays silent: the dimension
    # falls back to the honest unscored state, not a fabricated pass/fail.
    clean = sc.score_run_from_dir(_task(), run_dir, run_id="oa_silent")
    assert clean.result_validity.level is None


def test_score_run_from_dir_missing_files_degrade(tmp_path: Path):
    # Empty run dir -> diagnostic_only, low scores, no crash.
    card = sc.score_run_from_dir(_task(), tmp_path)
    assert card.tristate == "diagnostic_only"
    assert card.code.level == "Fail"


def _kind_task(kind: str):
    return ICUAgentBenchTask(
        task_id=f"{kind}_demo",
        kind=kind,
        title="demo",
        objective="demo",
        expected_outputs=["table one"],
        difficulty="intermediate",
        gold_answer_status="planned",
    )


def test_reporting_guideline_routing_is_kind_keyed():
    # Case-neutral routing: keyed on kind, not on any benchmark item.
    assert sc.reporting_guideline_for_kind("mortality_prediction") == "tripod"
    assert sc.reporting_guideline_for_kind("subphenotype_clustering") == "internal"
    assert (
        sc.reporting_guideline_for_kind("longitudinal_trajectory_analysis")
        == "internal"
    )
    assert sc.reporting_guideline_for_kind("descriptive_association") == "strobe"
    assert sc.reporting_guideline_for_kind("sepsis_onset") == "strobe"


def test_reporting_completeness_scores_strobe_coverage():
    checklist = {
        "summary": {
            "name": "STROBE",
            "n_total": 22,
            "n_addressed": 15,
            "n_partial": 0,
            "n_open": 7,
            "coverage": 0.682,
        }
    }
    dim = sc.score_reporting_completeness(
        _kind_task("sepsis_onset"), checklist=checklist
    )
    assert dim.subscore == pytest.approx(0.682, abs=1e-3)
    assert dim.level == "Partial"
    assert dim.signals["guideline"] == "strobe"
    assert dim.signals["n_open"] == 7


def test_reporting_completeness_unscored_for_clustering_without_checklist():
    # No EQUATOR guideline + no emitted internal core -> unscored, not penalised.
    dim = sc.score_reporting_completeness(
        _kind_task("subphenotype_clustering"), checklist={}
    )
    assert dim.subscore is None
    assert dim.level is None
    assert dim.signals["guideline"] == "internal"


def test_score_run_populates_reporting_completeness_and_six_dim_view():
    checklist = {
        "summary": {
            "n_total": 22,
            "n_addressed": 22,
            "n_partial": 0,
            "n_open": 0,
            "coverage": 1.0,
        }
    }
    card = sc.score_run(
        _kind_task("descriptive_association"),
        gates=_gates(),
        plan_steps=[{"intent": "table one"}],
        evidence_audit={"evidence_complete": True, "missing_evidence_count": 0},
        numeric_audit={"numeric_verified": True},
        claim_rows=[],
        reporting_checklist=checklist,
    )
    assert card.reporting_completeness is not None
    assert card.reporting_completeness.level == "Full"
    # Canonical Fig.3 column order stays at five; extended view adds the two
    # additive dimensions (reporting_completeness + fairness_subgroup).
    assert len(card.dimensions()) == 5
    assert len(card.all_dimensions()) == 7
    assert card.all_dimensions()[5].name == "reporting_completeness"
    assert card.all_dimensions()[6].name == "fairness_subgroup"


def test_score_run_reporting_unscored_when_no_checklist():
    card = sc.score_run(
        _kind_task("descriptive_association"),
        gates=_gates(),
        plan_steps=[{"intent": "table one"}],
        evidence_audit={"evidence_complete": True, "missing_evidence_count": 0},
        numeric_audit={"numeric_verified": True},
        claim_rows=[],
    )
    # Unscored reporting dim is still attached but excluded from the six-dim view.
    assert card.reporting_completeness is not None
    assert card.reporting_completeness.subscore is None
    assert len(card.all_dimensions()) == 7  # reporting + fairness both attached


def test_fairness_subgroup_open_when_no_subgroup_analysis():
    # STROBE 12b open (no subgroup analysis reported) -> fairness not addressed.
    checklist = {
        "items": [
            {
                "item_id": "12b",
                "status": "open",
                "statement": "Describe any methods used to examine subgroups and interactions.",
            },
            {
                "item_id": "7",
                "status": "addressed",
                "statement": "Define all outcomes.",
            },
        ]
    }
    dim = sc.score_fairness_subgroup(_kind_task("sepsis_onset"), checklist=checklist)
    assert dim.subscore == 0.0
    assert dim.level == "Fail"
    assert dim.signals["fairness_items"] == 1


def test_fairness_subgroup_addressed_when_subgroups_reported():
    checklist = {
        "items": [
            {
                "item_id": "12",
                "status": "addressed",
                "statement": "Fairness / subgroup performance plan.",
            },
            {
                "item_id": "18",
                "status": "addressed",
                "statement": "Subgroup / fairness results.",
            },
        ]
    }
    dim = sc.score_fairness_subgroup(
        _kind_task("mortality_prediction"), checklist=checklist
    )
    assert dim.subscore == 1.0
    assert dim.level == "Full"
    assert dim.signals["fairness_items"] == 2


def test_fairness_subgroup_unscored_when_no_fairness_item():
    dim = sc.score_fairness_subgroup(
        _kind_task("sepsis_onset"),
        checklist={
            "items": [
                {
                    "item_id": "7",
                    "status": "addressed",
                    "statement": "Define all outcomes.",
                },
            ]
        },
    )
    assert dim.subscore is None
    assert dim.level is None
