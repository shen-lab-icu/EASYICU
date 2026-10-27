"""A question about recorded, missing or timed measurements has an executable plan.

The primary descriptive summary binds the values of exactly one numeric
measurement, optionally within one declared categorical grouping, and its
executor raises on any other roster. Review refuses such a primary on every run
tier, before execution, and routes the repair to the Planner. Whether, how often
and when several measurements were recorded is the measurement audit's answer:
on an analysis-only first-result run that audit is the plan's whole result,
and it counts stays without any independent-patient inference. The article
tier keeps the registry's analysis-only ceiling for audits. Synthetic metadata
and rows only.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import DESCRIPTIVE_SUMMARY_CONTRACT
from easyicu.research_agent.contracts.claim_ceiling import bind_plan_tier
from easyicu.research_agent.contracts.cohort_summary import (
    is_primary_descriptive_summary_step,
    primary_descriptive_summary_scope_error,
)
from easyicu.research_agent.execution.runners import cohort_summary_executor
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    measurement_audit_answers_plan,
)
from easyicu.research_agent.schema import AnalysisStep, ConceptDescriptor, ResearchContext
from easyicu.webserver.research_plan_revision import PlanRevisionGuard

LABELS = {"lact": "Lactate", "crea": "Creatinine", "po2": "Arterial oxygen partial pressure",
          "map": "Mean arterial pressure", "hr": "Heart rate"}
MEASUREMENTS = tuple(LABELS)
SCOPE = "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"
REPEATED = {"REPEATED_STAY_METHOD_NOT_DECLARED", "REPEATED_STAY_IDENTITY_UNAVAILABLE",
            "REPEATED_STAY_DEDUP_UNDECLARED"}
COHORT = {"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}
AUDIT_OUTPUTS = [
    {"product_id": "table:measurement_missingness", "semantic_role": "measurement_missingness"},
    {"product_id": "table:measurement_process", "semantic_role": "measurement_process"},
    {"product_id": "table:event_timing", "semantic_role": "event_timing"},
]


def _context(*, patient_groups: bool = True) -> ResearchContext:
    provenance = {
        "analysis_unit": "icu_stay", "evidence_stage": "metadata_only_planning",
        "patient_rows_read": False,
    }
    if patient_groups:
        provenance["replacement_row_identity"] = {
            "output_identity_column": "stay_id", "mapping_file_sha256": "0" * 64,
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
        }
    return ResearchContext.model_validate({
        "research_question": (
            "For five first-day measurements, how many ICU stays have a record, how "
            "many lack one, and when were they taken?"
        ),
        "cohort": {"cohort_name": "synthetic", "database": "synthetic", "n_stays": 0,
                   "id_columns": ["stay_id"], "provenance": provenance},
        "variables": [
            {"name": "stay_id", "role": "id", "dtype": "string"},
            *({"name": name, "role": "lab" if name in {"lact", "crea", "po2"} else "vital",
               "dtype": "float64"} for name in MEASUREMENTS),
            {"name": "unit_type", "role": "other", "dtype": "object",
             "observed_domain": {"levels": ["A", "B"], "n_unique": 2}},
        ],
        "time_windows": [{"name": "first_day", "anchor": "icu_admission",
                          "start_hours": 0, "end_hours": 24}],
        "user_preferences": {"inferred_analysis_family": "descriptive_epidemiology"},
    })


def _design() -> dict:
    selected = {
        "design_id": "measurement_record_audit",
        "analysis_type": "descriptive_epidemiology",
        "estimand": "Recorded and missing ICU stays for each first-day measurement.",
        "time_zero": "ICU admission defines the study time origin.",
        "observation_window": "The host-bound first-day measurement window.",
        "primary_method": "Typed measurement audit counts",
        "required_variables": list(MEASUREMENTS),
        "assumptions": ["Each input row represents one eligible ICU stay."],
        "novelty_positioning": "Characterize measurement records before any modeling.",
        "figure_role": "Display recorded and missing counts per measurement.",
        "supports": "Recorded, missing and repeated measurement counts per stay.",
        "cannot_prove": "Clinical normality of unrecorded values or patient-level inference.",
        "reviewable_plan": [
            "Recommend all eligible ICU stays with an explicit denominator.",
            "Count records of each requested measurement in the bound window.",
            "No outcome or follow-up is needed for this measurement-record question.",
            "Use typed audit counts; no model or value summary is applicable.",
            "Report recorded and missing stays without imputing any value.",
            "Check the window bounds, duplicate stays and measurement indicators.",
        ],
        "disposition": "selected",
        "decision_reason": "The audit counts exactly what the question asks about.",
    }
    rejected = {
        **selected,
        "design_id": "value_distributions",
        "primary_method": "Empirical value distributions",
        "reviewable_plan": None,
        "disposition": "rejected",
        "decision_reason": "Value distributions do not answer a question about records.",
    }
    return {"candidates": [selected, rejected]}


def _outline(*, audit_role: str = "primary") -> ProgressivePlanOutline:
    return ProgressivePlanOutline.model_validate({
        "analysis_type": "descriptive_epidemiology",
        "cohort_objective": "Account for the ICU stays and their first-day measurement records.",
        "design_selection": _design(),
        "steps": [
            {"step_id": "cohort", "planned_analysis_role": "auxiliary",
             "module_id": "cohort_definition", "variable_names": ["stay_id"],
             "objective": "Account for the eligible ICU stay denominator."},
            {"step_id": "audit", "planned_analysis_role": audit_role,
             "module_id": "measurement_audit", "variable_names": list(MEASUREMENTS),
             "depends_on": ["cohort"],
             "objective": "Count recorded, missing and repeated records of each measurement."},
        ],
        "rationale": "The question asks whether measurements were recorded, not their values.",
    })


def _skeleton_step(step_id: str, *, role: str = "primary", columns=MEASUREMENTS) -> dict:
    if step_id == "cohort":
        return {"step_id": "cohort", "planned_analysis_role": "auxiliary",
                "module_id": "cohort_definition", "raw_inputs": ["stay_id"],
                "objective": "Account for the eligible ICU stay denominator."}
    if step_id == "audit":
        return {"step_id": "audit", "planned_analysis_role": role, "module_id": "measurement_audit",
                "objective": "Count recorded, missing and repeated records of each measurement.",
                "depends_on": ["cohort"], "raw_inputs": list(MEASUREMENTS),
                "product_inputs": [COHORT], "outputs": AUDIT_OUTPUTS}
    return {"step_id": "summary", "planned_analysis_role": "primary", "module_id": "custom_analysis",
            "scientific_action_id": "descriptive.descriptive_summary",
            "custom_method": "empirical_summary", "depends_on": ["cohort"],
            "objective": "Summarize the empirical distribution of the declared measurement.",
            "raw_inputs": list(columns), "product_inputs": [COHORT],
            "outputs": [{"product_id": "table:cohort_summary", "semantic_role": "custom"},
                        {"product_id": "table:distribution_summary", "semantic_role": "custom"}]}


def _plan(context, *steps: dict):
    outline = _outline()
    skeleton = ProgressivePlanSkeleton.model_validate({
        "analysis_type": "descriptive_epidemiology", "design_selection": outline.design_selection,
        "cohort": {"name": "primary", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": key, "value": value} for key, value in LABELS.items()],
        "steps": list(steps), "rationale": outline.rationale,
    })
    return compile_progressive_plan(skeleton=skeleton, context=context)[0]


def _review(context, plan, tier):
    return build_plan_scientific_review(
        context=context, plan=bind_plan_tier(plan, tier), require_reportable_capability=True,
    )


def _blockers(review) -> set[str]:
    return {finding.code for finding in review.findings if finding.severity == "blocker"}


@pytest.mark.parametrize("tier", ["first_result", "article"])
def test_a_multi_measurement_primary_summary_is_refused_on_every_tier(tier):
    context = _context()
    plan = _plan(context, _skeleton_step("cohort"), _skeleton_step("summary"),
                 _skeleton_step("audit", role="auxiliary"))
    summary = next(step for step in plan.steps if step.step_id == "summary")
    assert is_primary_descriptive_summary_step(summary)
    review = _review(context, plan, tier)
    assert not review.approval_allowed
    finding = next(item for item in review.findings if item.code == SCOPE)
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    assert "one measurement" in finding.message
    assert "measurement_audit" in finding.remediation


@pytest.mark.parametrize("tier", ["first_result", "article"])
def test_one_measurement_is_not_refused(tier):
    context = _context()
    plan = _plan(context, _skeleton_step("cohort"), _skeleton_step("summary", columns=("lact",)),
                 _skeleton_step("audit", role="auxiliary"))
    review = _review(context, plan, tier)
    assert SCOPE not in {finding.code for finding in review.findings}
    assert review.approval_allowed


@pytest.mark.parametrize("tier", ["first_result", "article"])
def test_one_typed_outcome_within_a_declared_grouping_is_not_refused(tier):
    # The grouped form follows the continuous-summary owner: a typed numeric
    # outcome within a closed categorical grouping.
    context = _context()
    context = context.model_copy(update={"variables": [*context.variables, ConceptDescriptor(
        name="lact_max", role="outcome", dtype="float64", description="First-day maximum lactate",
    )]})
    plan = _plan(context, _skeleton_step("cohort"), _skeleton_step("summary", columns=("lact",)))
    plan = plan.model_copy(update={"steps": [
        step.model_copy(update={"inputs": ["lact_max", "unit_type", "artifact:analysis_cohort"]})
        if step.step_id == "summary" else step for step in plan.steps
    ]})
    assert SCOPE not in {finding.code for finding in _review(context, plan, tier).findings}
    lone = plan.model_copy(update={"steps": [
        step.model_copy(update={"inputs": ["lact", "unit_type", "artifact:analysis_cohort"]})
        if step.step_id == "summary" else step for step in plan.steps
    ]})
    assert SCOPE in _blockers(_review(context, lone, tier))


def _summary_step(context, *columns: str) -> AnalysisStep:
    plan = _plan(context, _skeleton_step("cohort"), _skeleton_step("summary", columns=("lact",)))
    step = next(item for item in plan.steps if item.step_id == "summary")
    return step.model_copy(update={"inputs": [*columns, "artifact:analysis_cohort"]})


@pytest.mark.parametrize("columns", [MEASUREMENTS, ("lact", "crea"), ("unit_type",)])
def test_the_review_asks_the_executors_own_question(tmp_path, monkeypatch, columns):
    """A roster the review refuses, the executor refuses for the same reason."""

    context = _context()
    step = _summary_step(context, *columns)
    reason = primary_descriptive_summary_scope_error(step, context=context)
    assert reason
    frame = pd.DataFrame({name: ["A", "B"] if name == "unit_type" else [1.0, 2.0] for name in columns})
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    with pytest.raises(RuntimeError) as raised:
        cohort_summary_executor.run_cohort_summary_from_env(
            declared_columns=list(columns), typed_cohort_input="artifact:analysis_cohort",
            frame=frame, cohort_path=tmp_path / "synthetic.parquet", primary_summary=True,
        )
    assert str(raised.value) == reason


def test_only_an_exact_primary_summary_is_judged():
    context = _context()
    assert "known typed columns" in primary_descriptive_summary_scope_error(
        _summary_step(context, "unbound_measurement"), context=context,
    )
    assert primary_descriptive_summary_scope_error(_summary_step(context, "lact"), context=context) is None
    # A step this owner does not execute as a primary is not its to judge.
    secondary = _summary_step(context, *MEASUREMENTS).model_copy(update={"planned_analysis_role": "secondary"})
    assert primary_descriptive_summary_scope_error(secondary, context=context) is None


def test_the_contract_states_one_measurement_and_the_audit_route():
    text = DESCRIPTIVE_SUMMARY_CONTRACT
    assert "exactly one numeric measurement" in text
    assert "never list several measurements" in text
    assert "answered by measurement_audit" in text
    assert 'make that step the primary (planned_analysis_role="primary"' in text
    assert "plan no descriptive.descriptive_summary" in text
    assert "event_timing only for event-timed concepts" in text
    assert "cannot_prove" in text
    # A grouped continuous question: primary plus the answering secondary.
    assert "answering secondary" in text
    assert "the primary alone, grouped or not, does not" in text
    context = _context()
    families, variables, _actions, rows = ProgressivePlannerAgent._request_authorities(context)
    prompt = ProgressivePlannerAgent._user_prompt(
        context, analysis_types=families, variables=variables, action_rows=rows,
    )
    assert DESCRIPTIVE_SUMMARY_CONTRACT in prompt


def test_the_audit_led_outline_is_valid_on_a_first_result_run():
    context = _context()
    ProgressivePlannerAgent._validate_outline_authority(
        _outline(), analysis_types=["descriptive_epidemiology"],
        variable_names=[variable.name for variable in context.variables],
        allowed_literature_citation_keys=(), closed_domain_variables=("unit_type",),
        continuous_domain_variables=MEASUREMENTS, article_context=context,
        require_design_selection=True, execution_tier="first_result",
    )


@pytest.mark.parametrize("patient_groups", [True, False])
@pytest.mark.parametrize("audit_role", ["primary", "auxiliary"])
def test_an_audit_led_plan_is_approvable_on_first_result_and_analysis_only(patient_groups, audit_role):
    context = _context(patient_groups=patient_groups)
    plan = _plan(context, _skeleton_step("cohort"), _skeleton_step("audit", role=audit_role))
    assert measurement_audit_answers_plan(plan)
    first = _review(context, plan, "first_result")
    assert first.approval_allowed, _blockers(first)
    assert first.status == "analysis_only"
    assert not REPEATED & {finding.code for finding in first.findings}
    assert first.facts["scientific_capability"]["claim_ceiling"] == "analysis_only"
    # The registry keeps audits analysis-only: an article run is still refused.
    article = _review(context, plan, "article")
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" in _blockers(article)


def test_audit_counts_do_not_close_repeated_stays_for_another_analysis():
    context = _context()
    plan = _plan(context, _skeleton_step("cohort"), _skeleton_step("audit", role="secondary"))
    other = AnalysisStep(
        step_id="clusters", planned_analysis_role="primary", intent="Cluster the stays.",
        method="custom_cluster_assignment", inputs=["lact", "crea", "artifact:analysis_cohort"],
        expected_outputs=["table:cluster_assignments"],
    )
    plan = plan.model_copy(update={"steps": [*plan.steps, other]})
    assert not measurement_audit_answers_plan(plan)
    review = _review(context, plan, "first_result")
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" in _blockers(review)
    # Nor does a plan with no audit at all count as one answered by audit counts.
    cohort_only = plan.model_copy(update={"steps": plan.steps[:1]})
    assert not measurement_audit_answers_plan(cohort_only)
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" in _blockers(_review(context, cohort_only, "first_result"))


def test_removing_the_unrunnable_summary_is_an_admitted_repair():
    context = _context()
    source = bind_plan_tier(_plan(
        context, _skeleton_step("cohort"), _skeleton_step("summary"),
        _skeleton_step("audit", role="auxiliary"),
    ), "first_result")
    review = _review(context, source, "first_result")
    assert SCOPE in _blockers(review)
    guard = PlanRevisionGuard.from_review(plan=source.model_dump(mode="json"), review=review)
    revised = source.model_copy(update={"steps": [
        step for step in source.steps if step.step_id != "summary"
        and "table:distribution_summary" not in step.inputs
    ]})
    guard.validate_plan(plan=revised, context=context)
    assert _review(context, revised, "first_result").approval_allowed


def test_the_audit_led_plan_executes_through_its_typed_owner(tmp_path, monkeypatch):
    context = _context()
    plan = bind_plan_tier(_plan(context, _skeleton_step("cohort"), _skeleton_step("audit")), "first_result")
    audit = next(step for step in plan.steps if step.step_id == "audit")
    selection = select_standard_executor(audit, plan=plan)
    assert selection is not None and selection.analysis_kind
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frame = pd.DataFrame({"stay_id": [f"p{i // 2}:s{i}" for i in range(8)]})
    missing = {"lact": 3, "crea": 1, "po2": 5, "map": 0, "hr": 2}
    for name, absent in missing.items():
        frame[name] = [None] * absent + [float(i + 1) for i in range(8 - absent)]
        frame[f"{name}_measured"] = [0] * absent + [1] * (8 - absent)
        frame[f"{name}_n"] = [0] * absent + [2] * (8 - absent)
    cohort = run_dir / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    (run_dir / "research_context.json").write_text(context.model_dump_json())
    (run_dir / "analysis_plan.json").write_text(plan.model_dump_json())
    resolved = run_dir / "resolved_inputs.json"
    resolved.write_text(json.dumps({"inputs": {"artifact:analysis_cohort": {
        "relative_path": cohort.name, "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
    }}}))
    out_dir = run_dir / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_STEP_ID", audit.step_id)
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    try:
        exec(compile(selection.code, "<measurement-audit>", "exec"), {})
    except SystemExit:
        pass
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok", summary.get("blocking_reason")
    assert set(summary["output_files"]) == {output["product_id"] for output in AUDIT_OUTPUTS}
    table = pd.read_csv(out_dir / summary["output_files"]["table:measurement_missingness"])
    counts = {row.concept: (int(row.n_total), int(row.value_missing_n)) for row in table.itertuples()}
    assert counts == {name: (8, absent) for name, absent in missing.items()}


def test_a_one_measurement_record_question_owes_no_value_summary():
    """Its audit answers it, so the one-measurement primary summary is not missing."""

    context = _context()
    context = context.model_copy(update={"variables": [
        variable for variable in context.variables if variable.name in {"stay_id", "lact"}
    ]})
    plan = _plan(context, _skeleton_step("cohort"), {**_skeleton_step("audit"), "raw_inputs": ["lact"]})
    assert measurement_audit_answers_plan(plan)
    review = _review(context, plan, "first_result")
    assert "DESCRIPTIVE_PRIMARY_SUMMARY_MISSING" not in {finding.code for finding in review.findings}
    assert review.approval_allowed, _blockers(review)
