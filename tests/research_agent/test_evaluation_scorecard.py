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


def test_tristate_validity_fail_caps_gate_reportable_at_analysis_only():
    # A manuscript-ready run with a hard validity failure (e.g. overadjustment)
    # must NOT be reported as gate_reportable — it is demoted to analysis_only.
    assert (
        sc.compute_tristate(_gates(), result_validity_level="Fail") == "analysis_only"
    )


def test_tristate_validity_fail_does_not_promote_diagnostic_only():
    # The validity ceiling can only demote: a run that never executed stays
    # diagnostic_only regardless of the validity level.
    g = _gates(execution_complete=False, manuscript_ready=False)
    assert sc.compute_tristate(g, result_validity_level="Fail") == "diagnostic_only"


def test_tristate_validity_none_leaves_gate_verdict_unchanged():
    # Validity not scored for this task kind (None) must not change the verdict.
    assert (
        sc.compute_tristate(_gates(), result_validity_level=None) == "gate_reportable"
    )


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


def test_plan_clustering_kind_not_penalized_for_omitting_table_one():
    # H3/M3 regression: a phenotype-discovery kind (internal guideline) does not
    # need a STROBE/TRIPOD baseline "Table 1". A plan with figure + audit panel
    # but no Table 1 must still be Full (was wrongly capped at 0.8).
    task = ICUAgentBenchTask(
        task_id="H3",
        kind="longitudinal_trajectory_analysis",
        title="t",
        objective="t",
        expected_outputs=["trajectory plot figure", "stability audit panel"],
    )
    dim = sc.score_plan(task, plan_steps=[{"intent": "x"}], gates=_gates())
    assert dim.signals["table_one_expected"] is False
    assert dim.level == "Full"


def test_plan_credits_produced_publication_figure_when_not_declared():
    # E1/E3 regression: the publication-figure skill produces the result figure
    # outside declared plan steps (and the replanner can drop a declared figure
    # step). A plan with Table 1 + audit but no declared figure must still be
    # Full when the run delivered a publication-figure bundle.
    task = _task(outputs=["table one", "component completeness audit"])  # no figure
    gates = _gates()
    gates["publication_figure_bundle_ready"] = True
    dim = sc.score_plan(task, plan_steps=[{"intent": "x"}], gates=gates)
    assert dim.signals["result_figure_count"] == 0
    assert dim.signals["produced_publication_figure"] is True
    assert dim.level == "Full"


def test_plan_still_flags_missing_figure_when_none_declared_or_produced():
    # Impartiality: do not blanket-pass the figure component — a plan with no
    # declared figure AND no produced publication bundle still loses the figure.
    task = _task(outputs=["table one", "component completeness audit"])
    dim = sc.score_plan(task, plan_steps=[{"intent": "x"}], gates=_gates())
    assert dim.signals["produced_publication_figure"] is False
    assert any("figure" in n for n in dim.notes)
    assert dim.level != "Full"


def test_plan_hard_task_needs_two_figures():
    # advanced task with only one figure hint -> figure requirement unmet
    task = _task(
        difficulty="advanced", outputs=["table one", "forest plot", "leakage audit"]
    )
    dim = sc.score_plan(task, plan_steps=[{"intent": "x"}], gates=_gates())
    assert dim.signals["min_result_figures"] == 2
    assert dim.signals["result_figure_count"] == 1
    assert dim.level != "Full"


def test_declares_table_one_matches_diverse_baseline_table_names():
    # Real agent vocabulary for a Table 1 (baseline characteristics) — must match
    # despite diverse naming, via a cohort/sample noun + a summary descriptor.
    for name in [
        "table:cohort_summary",
        "table:covariate_summary_by_exposure",
        "table:adult_cohort_characteristics_by_lactate_measured",
        "table:patient_overview",
        "table_one",
    ]:
        assert sc._declares_table_one([name]), name


def test_declares_table_one_rejects_non_table_one_artifacts_and_prose():
    # Flow tables, single-variable distributions, missingness profiles, and a
    # generic results summary are NOT a Table 1 — and a prose step intent that
    # merely co-mentions "cohort" and a stats word must not count.
    for name in [
        "table:cohort_attrition",  # flow, not characteristics
        "table:stage_distribution_in_cohort",  # one-variable distribution
        "table:covariate_missingness_profile",  # missingness, not baseline
        "table:robustness_summary",  # no cohort/sample subject
        "table:final_results_summary",  # results, not baseline
        "Define the adult cohort and summarise its distribution before modelling",  # prose intent, no table: artifact
    ]:
        assert not sc._declares_table_one([name]), name


def test_score_run_from_dir_reads_latest_plan_revision(tmp_path):
    # The plan dimension must score the EXECUTED plan (latest revision), not the
    # initial analysis_plan.json. Here the base plan lacks a Table 1 but a later
    # revision adds one -> plan must see it.
    (tmp_path / "run_status.json").write_text(
        json.dumps(
            {
                "gates": {
                    "required_step_count": 1,
                    "completed_step_count": 1,
                    "execution_complete": True,
                    "manuscript_ready": True,
                }
            }
        ),
        encoding="utf-8",
    )
    base_steps = {"steps": [{"intent": "fit model", "expected_outputs": ["figure:x"]}]}
    revised = {
        "steps": [
            {
                "intent": "fit model",
                "expected_outputs": [
                    "figure:x",
                    "table:cohort_summary",
                    "figure:audit_panel",
                ],
            }
        ]
    }
    (tmp_path / "analysis_plan.json").write_text(json.dumps(base_steps), "utf-8")
    (tmp_path / "analysis_plan_revision_2.json").write_text(
        json.dumps(revised), "utf-8"
    )
    task = ICUAgentBenchTask(
        task_id="t", kind="causal_inference", title="t", objective="t"
    )
    card = sc.score_run_from_dir(task, tmp_path)
    assert card.plan.signals["has_table_one"] is True


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


def test_positive_validity_requires_execution_complete():
    # A run that never produced a result (execution_complete=False) must NOT get a
    # positive result_validity from an early-step check passing — there is no
    # result to validate. It stays honestly unscored (None), not the contradiction
    # of "fully valid" alongside a diagnostic_only tristate.
    sigs = [sc.ValiditySignal("patient_level_split_no_overlap", "pass", "overlap=0")]
    blocked = sc.score_result_validity(
        _task(),
        numeric_audit={"numeric_verified": False},
        positive_subscore=1.0,
        positive_signals=sigs,
        execution_complete=False,
    )
    assert blocked.subscore is None and blocked.level is None
    completed = sc.score_result_validity(
        _task(),
        numeric_audit={"numeric_verified": True},
        positive_subscore=1.0,
        positive_signals=sigs,
        execution_complete=True,
    )
    assert completed.subscore == 1.0


def test_single_assessable_signal_label_capped_below_full():
    # One assessable check that passes -> subscore 1.0 but the LABEL must not be
    # "Full": one check cannot establish full result validity. Numeric subscore is
    # unchanged; n_assessed is surfaced.
    sigs = [sc.ValiditySignal("patient_level_split_no_overlap", "pass", "overlap=0")]
    dim = sc.score_result_validity(
        _task(),
        numeric_audit={"numeric_verified": True},
        positive_subscore=1.0,
        positive_signals=sigs,
        execution_complete=True,
    )
    assert dim.subscore == 1.0
    assert dim.level == "Partial"
    assert dim.signals["n_assessed"] == 1


def test_two_assessable_signals_can_reach_full():
    # With >=2 assessable checks all passing, Full is legitimately awarded.
    sigs = [
        sc.ValiditySignal("covariate_balance_achieved", "pass", "max|SMD|=0.05"),
        sc.ValiditySignal("positivity_assessed", "pass", "overlap holds"),
    ]
    dim = sc.score_result_validity(
        _task(),
        numeric_audit={"numeric_verified": True},
        positive_subscore=1.0,
        positive_signals=sigs,
        execution_complete=True,
    )
    assert dim.level == "Full"
    assert dim.signals["n_assessed"] == 2


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


def test_audit_safety_floor_only_is_partial_not_full_without_hazard_key():
    # No per-task hazard key and no forbidden-output key: we can only confirm the
    # fail-closed floor (nothing forbidden leaked). That must NOT score Full —
    # hazard handling is unassessed — it caps at Partial with an explicit note.
    dim = sc.score_audit_conclusion_safety(
        _task(gold=None),
        observed_warnings=["some warning"],
        observed_outputs=["a plain association statement"],
        tristate="gate_reportable",
    )
    assert dim.level == "Partial"
    assert dim.signals["floor_only_no_hazard_key"] is True
    assert dim.signals["has_hazard_key"] is False
    assert dim.signals["forbidden_conclusion_leaked"] is False


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


def test_score_run_from_dir_prefers_current_manifest_primary_model_covariates(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": _gates()}), encoding="utf-8"
    )
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir()
    with (evidence_dir / "stale_adjusted_association.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["term", "effect_scale", "estimate"])
        w.writeheader()
        w.writerow({"term": "sepsis3", "effect_scale": "odds_ratio", "estimate": "1.2"})
        w.writerow({"term": "map_min", "effect_scale": "odds_ratio", "estimate": "0.9"})

    current_outputs = (
        run_dir
        / "steps"
        / "03_primary_prevalence_and_adjusted_association"
        / "outputs"
    )
    current_outputs.mkdir(parents=True)
    with (current_outputs / "adjusted_association_death.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["term", "effect_scale", "estimate"])
        w.writeheader()
        w.writerow({"term": "sepsis3", "effect_scale": "odds_ratio", "estimate": "1.1"})
        w.writerow({"term": "age", "effect_scale": "odds_ratio", "estimate": "1.02"})
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": "03_primary_prevalence_and_adjusted_association",
                        "status": "ok",
                        "step_summary": {"primary_model": {"exposure": "sepsis3"}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    card = sc.score_run_from_dir(
        _task(), run_dir, exposure_concept="sepsis3", run_id="current"
    )

    assert card.result_validity.level is None
    assert not any("overadjustment" in n for n in card.result_validity.notes)


def test_score_run_from_dir_flags_outcome_leakage(tmp_path: Path):
    # A model that conditions on its own declared outcome is target leakage ->
    # gold-free Fail, distinct from the honest unscored state.
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
        w.writerow({"variable": "age", "coef": "0.02"})
        w.writerow({"variable": "death_icu", "coef": "2.0"})

    card = sc.score_run_from_dir(
        _task(), run_dir, outcome_concept="death_icu", run_id="leak"
    )
    assert card.result_validity.level == "Fail"
    assert card.result_validity.subscore == 0.0
    assert any("outcome leakage" in n for n in card.result_validity.notes)

    # Without the outcome declared the self-leakage error stays silent.
    clean = sc.score_run_from_dir(_task(), run_dir, run_id="leak_silent")
    assert clean.result_validity.level is None


def test_score_run_from_dir_missing_files_degrade(tmp_path: Path):
    # Empty run dir -> diagnostic_only, low scores, no crash.
    card = sc.score_run_from_dir(_task(), tmp_path)
    assert card.tristate == "diagnostic_only"
    assert card.code.level == "Fail"


def _write_blocked_run(tmp_path: Path, *, n_rows: int, n_events: int, n_feats: int):
    """A run that did NOT complete and recorded a deliberate modeling block,
    plus a locked analysis cohort of the given size/event count."""
    import pandas as pd

    run_dir = tmp_path / "run"
    (run_dir / "steps" / "01_model_training" / "outputs").mkdir(parents=True)
    g = _gates(execution_complete=False, manuscript_ready=False)
    g["failed_steps"] = [{"step_id": "01_model_training", "status": "contract_failed"}]
    g["completed_step_count"] = 3
    (run_dir / "run_status.json").write_text(json.dumps({"gates": g}), encoding="utf-8")
    (
        run_dir / "steps" / "01_model_training" / "outputs" / "step_summary.json"
    ).write_text(
        json.dumps(
            {
                "step_id": "01_model_training",
                "execution_status": "blocked_non_execution",
                "modeling_blocked": True,
                "modeling_block_reason": "upstream viability gate said unusable",
            }
        ),
        encoding="utf-8",
    )
    death = [1] * n_events + [0] * (n_rows - n_events)
    data = {"death": death}
    for i in range(n_feats):
        data[f"f{i}_first"] = list(range(n_rows))  # fully populated
    pd.DataFrame(data).to_parquet(run_dir / "cohort_analysis.parquet", index=False)
    return run_dir


def test_self_inflicted_block_flagged_on_viable_cohort(tmp_path: Path):
    # Blocked its own model on a large, event-rich, well-populated cohort:
    # surfaced as a factual self-paralysis note WITHOUT changing the verdict
    # (execution genuinely did not complete -> still diagnostic_only).
    run_dir = _write_blocked_run(tmp_path, n_rows=2000, n_events=200, n_feats=8)
    card = sc.score_run_from_dir(
        _kind_task("mortality_prediction"), run_dir, outcome_concept="death"
    )
    assert card.tristate == "diagnostic_only"
    assert card.code.signals.get("self_inflicted_block") is True
    assert any("self-inflicted" in n for n in card.code.notes)


def test_self_inflicted_block_detected_with_manifest_present(tmp_path: Path):
    # Regression (2719ce4): real runs write a manifest ledger (manifest_partial/
    # manifest.json), and a self-inflicted modeling block is recorded there as a
    # NON-ok (contract_failed) step. _deliberate_block_reason must still see it;
    # previously it filtered to status==ok records in the manifest branch, so on
    # every real run the signal silently never fired. The other self-inflicted
    # tests here write NO manifest, so they only exercised the glob fallback.
    run_dir = _write_blocked_run(tmp_path, n_rows=2000, n_events=200, n_feats=8)
    blocked_summary = json.loads(
        (
            run_dir / "steps" / "01_model_training" / "outputs" / "step_summary.json"
        ).read_text(encoding="utf-8")
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": "00_cohort",
                        "status": "ok",
                        "step_summary": {"step_id": "00_cohort"},
                    },
                    {
                        "step_id": "01_model_training",
                        "status": "contract_failed",
                        "step_summary": blocked_summary,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(
        _kind_task("mortality_prediction"), run_dir, outcome_concept="death"
    )
    assert card.tristate == "diagnostic_only"
    assert card.code.signals.get("self_inflicted_block") is True
    assert any("self-inflicted" in n for n in card.code.notes)


def test_self_inflicted_block_silent_when_execution_completed(tmp_path: Path):
    run_dir = _write_blocked_run(tmp_path, n_rows=2000, n_events=200, n_feats=8)
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": _gates()}), encoding="utf-8"  # execution_complete=True
    )
    card = sc.score_run_from_dir(
        _kind_task("mortality_prediction"), run_dir, outcome_concept="death"
    )
    assert card.code.signals.get("self_inflicted_block") is None


def test_self_inflicted_block_silent_on_too_few_events(tmp_path: Path):
    # A genuinely event-poor cohort is NOT accused of a spurious block.
    run_dir = _write_blocked_run(tmp_path, n_rows=2000, n_events=3, n_feats=8)
    card = sc.score_run_from_dir(
        _kind_task("mortality_prediction"), run_dir, outcome_concept="death"
    )
    assert card.code.signals.get("self_inflicted_block") is None


def test_self_inflicted_block_silent_on_hard_crash(tmp_path: Path):
    # A run that failed without a *deliberate* block signal (a real crash) is
    # not labelled self-inflicted, even on a viable cohort.
    run_dir = _write_blocked_run(tmp_path, n_rows=2000, n_events=200, n_feats=8)
    summ = run_dir / "steps" / "01_model_training" / "outputs" / "step_summary.json"
    summ.write_text(json.dumps({"step_id": "01_model_training"}), encoding="utf-8")
    card = sc.score_run_from_dir(
        _kind_task("mortality_prediction"), run_dir, outcome_concept="death"
    )
    assert card.code.signals.get("self_inflicted_block") is None


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


# ---------------------------------------------------------------------------
# Phenotype (clustering / trajectory) reporting + validity (eval fix "b")
# ---------------------------------------------------------------------------


def test_clustering_reporting_completeness_scored_when_internal_core_emitted(
    ra, tmp_path
):
    # A clustering run that emits the internal phenotype core is now SCORED on
    # reporting completeness instead of being permanently unscored.
    report = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript="k-means clustering; silhouette and bootstrap stability",
    )
    (tmp_path / "reporting_checklist_internal_phenotype.json").write_text(
        json.dumps(report.to_json()), encoding="utf-8"
    )
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    assert card.reporting_completeness.subscore is not None
    assert card.reporting_completeness.signals["guideline"] == "internal"


def test_phenotype_validity_fails_on_objective_degeneracy(tmp_path):
    # Silhouette <= 0 means groups are no better than chance: an objective error
    # that caps result_validity at Fail even with no locked reference.
    (tmp_path / "cluster_validity.json").write_text(
        json.dumps({"silhouette": -0.02, "n_clusters": 3, "min_cluster_fraction": 0.2}),
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    assert card.result_validity.level == "Fail"
    assert any("silhouette" in n for n in card.result_validity.notes)


def test_phenotype_validity_flags_single_group_solution(tmp_path):
    (tmp_path / "cluster_validity.json").write_text(
        json.dumps({"silhouette": 0.3, "n_clusters": 1}), encoding="utf-8"
    )
    card = sc.score_run_from_dir(
        _kind_task("longitudinal_trajectory_analysis"), tmp_path
    )
    assert card.result_validity.level == "Fail"
    assert any("single-group" in n for n in card.result_validity.notes)


def test_phenotype_validity_healthy_solution_stays_unscored(tmp_path):
    # A non-degenerate solution is NOT given a fabricated pass: result_validity
    # stays honestly unscored (no locked reference). The dimension never imposes
    # a "good enough" silhouette threshold on a valid partition.
    (tmp_path / "cluster_validity.json").write_text(
        json.dumps({"silhouette": 0.41, "n_clusters": 3, "min_cluster_fraction": 0.18}),
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    assert card.result_validity.subscore is None
    assert card.result_validity.level is None


def test_phenotype_validity_unscored_without_metrics(tmp_path):
    # No metrics emitted -> honest NA, not a guess.
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    assert card.result_validity.level is None
    # And the objective-error check never fires for a non-phenotype kind.
    (tmp_path / "cluster_validity.json").write_text(
        json.dumps({"silhouette": -0.5, "n_clusters": 1}), encoding="utf-8"
    )
    card2 = sc.score_run_from_dir(_kind_task("sepsis_onset"), tmp_path)
    assert card2.result_validity.level is None


def test_phenotype_validity_reads_agent_emitted_artifacts(tmp_path):
    # The agent names its outputs itself (clustering_algorithm_details.json +
    # cluster_sizes.csv), not a fixed cluster_validity.json. The reader must still
    # recover silhouette / k / min-fraction so the degeneracy check is not blind.
    (tmp_path / "clustering_algorithm_details.json").write_text(
        json.dumps(
            {"algorithm": "KMeans", "selected_k": 2, "selected_silhouette_score": 0.80}
        ),
        encoding="utf-8",
    )
    (tmp_path / "cluster_sizes.csv").write_text(
        "cluster,n,percentage\n0,38584,99.48\n1,203,0.52\n", encoding="utf-8"
    )
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    # GENUINE degeneracy: one cluster holds 99.48% and the other is a 0.52%
    # outlier pocket. This is the "one dominant cluster" the guard is named for,
    # so the sub-1% group is an objective Fail regardless of the high silhouette
    # (which is a one-blob-plus-outlier artifact).
    assert card.result_validity.level == "Fail"
    assert any("near-empty" in n for n in card.result_validity.notes)


def test_small_cluster_in_balanced_partition_is_caution_not_fail(tmp_path):
    # M3-shaped real result: KMeans with k chosen by bootstrap-ARI stability gave
    # five substantial clusters (32/21/21/16/9 %) plus one rare 0.52% group. A
    # sub-1% cluster is NOT a degeneracy here — there is no dominant cluster
    # (largest is 32%), so it must not fail result_validity. It is a defensible
    # analytical outcome surfaced as a caution; the dimension stays honestly NA.
    (tmp_path / "clustering_algorithm_details.json").write_text(
        json.dumps(
            {"algorithm": "KMeans", "selected_k": 6, "selected_silhouette_score": 0.107}
        ),
        encoding="utf-8",
    )
    (tmp_path / "cluster_sizes.csv").write_text(
        "cluster,count,percentage\n"
        "1,12586,32.45\n2,8170,21.06\n3,8065,20.79\n"
        "4,6271,16.17\n5,3492,9.00\n6,203,0.52\n",
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    # No objective error -> honestly unscored (no locked reference), NOT a Fail.
    assert card.result_validity.level is None
    assert card.result_validity.subscore is None
    assert not any("near-empty" in n for n in card.result_validity.notes)
    # The rare group is still surfaced for human review as a caution.
    assert any("small cluster" in n for n in card.result_validity.notes)


def test_phenotype_model_based_negative_silhouette_is_caution_not_fail(tmp_path):
    # A GaussianMixture's fit criterion is the likelihood/BIC, not silhouette, so
    # a negative silhouette on a balanced, stable partition is surfaced as a
    # caution — NOT a fabricated Fail. result_validity stays honestly unscored.
    (tmp_path / "clustering_algorithm_details.json").write_text(
        json.dumps(
            {
                "algorithm": "GaussianMixture",
                "selected_k": 2,
                "selection_metrics": [
                    {
                        "k": 2,
                        "silhouette_score": -0.015,
                        "min_cluster_pct": 21.9,
                        "selected": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(
        _kind_task("longitudinal_trajectory_analysis"), tmp_path
    )
    assert card.result_validity.level is None  # not failed
    assert any("weak cluster separation" in n for n in card.result_validity.notes)


def test_phenotype_distance_based_negative_silhouette_still_fails(tmp_path):
    # For a distance/centroid method silhouette IS the right lens: <= 0 fails.
    (tmp_path / "clustering_algorithm_details.json").write_text(
        json.dumps(
            {"algorithm": "KMeans", "selected_k": 3, "selected_silhouette_score": -0.02}
        ),
        encoding="utf-8",
    )
    (tmp_path / "cluster_sizes.csv").write_text(
        "cluster,n,percentage\n0,40,33.3\n1,40,33.3\n2,40,33.4\n", encoding="utf-8"
    )
    card = sc.score_run_from_dir(_kind_task("subphenotype_clustering"), tmp_path)
    assert card.result_validity.level == "Fail"
    assert any("silhouette" in n for n in card.result_validity.notes)


def test_prediction_kind_routes_reporting_to_tripod_file(tmp_path):
    # M2 (mortality_prediction) routes reporting completeness to the TRIPOD
    # file, not STROBE -- closing the by-kind routing gap.
    assert sc.reporting_guideline_for_kind("mortality_prediction") == "tripod"
    (tmp_path / "reporting_checklist_tripod_ai.json").write_text(
        json.dumps(
            {
                "summary": {
                    "n_total": 27,
                    "n_addressed": 20,
                    "n_partial": 0,
                    "n_open": 7,
                    "coverage": 0.74,
                }
            }
        ),
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(_kind_task("mortality_prediction"), tmp_path)
    assert card.reporting_completeness.signals["guideline"] == "tripod"
    assert card.reporting_completeness.subscore == pytest.approx(0.74, abs=1e-2)


def test_unresolvable_derived_exposure_surfaces_caution_without_failing(tmp_path):
    # A callback-only composite exposure (news) whose constituents can't be
    # resolved: result_validity must stay honestly unscored (NOT Fail) but carry
    # the caution so the overadjustment risk is not silently passed.
    (tmp_path / "regression_results.csv").write_text(
        "variable,coef\nconst,1\nnews_score,0.5\nage,0.1\nsex,0.2\n",
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(
        _kind_task("descriptive_association"), tmp_path, exposure_concept="news"
    )
    assert card.result_validity.level is None
    assert card.result_validity.subscore is None
    assert card.result_validity.signals.get("validity_cautions")
    assert any("could not be checked" in n for n in card.result_validity.notes)


def test_resolvable_exposure_still_fails_not_cautions(tmp_path):
    # sofa adjusting for creatinine is a *resolvable* constituent -> the error
    # path Fails it; the caution path must not swallow it into a soft note.
    (tmp_path / "regression_results.csv").write_text(
        "variable,coef\nconst,1\nsofa,0.5\ncreatinine,0.1\nage,0.2\n",
        encoding="utf-8",
    )
    card = sc.score_run_from_dir(
        _kind_task("descriptive_association"), tmp_path, exposure_concept="sofa"
    )
    assert card.result_validity.level == "Fail"
