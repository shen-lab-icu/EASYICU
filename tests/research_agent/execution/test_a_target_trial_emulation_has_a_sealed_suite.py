"""A target trial emulation has its own sealed suite.

The signed target trial suite emulates a trial with a grace period by clone,
censor and weight (``methods.clone_censor_weight``): every stay alive and in
the ICU at time zero, with no start of the treatment before it, is cloned into
a strategy that starts the treatment within the grace period and one that does
not; each clone is censored when it deviates and weighted by models of starting
and of leaving the ICU.  The host signs the design a researcher confirmed,
compiles the plan's three owners, stops at the thresholds it prespecified,
states each estimate in one fixed sentence that names the emulation's
assumptions, and places those assumptions in the Limitations.

Synthetic, seeded rows only (``tests/support/target_trial.py``).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.authority.target_trial_runtime import (
    TargetTrialRuntimeAuthority,
)
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    TargetTrialDesign,
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.executor_stop import (
    EXECUTOR_STOP_RECORD_NAME,
    ExecutorStop,
    parse_executor_stop_record,
)
from easyicu.research_agent.contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from easyicu.research_agent.contracts.sealed_suite_robustness import (
    sealed_suite_prespecified_axes,
)
from easyicu.research_agent.contracts.step_families import effect_output_authorized
from easyicu.research_agent.execution.runners.target_trial_executor import (
    run_target_trial_suite,
    target_trial_executor_code,
    target_trial_executor_owns_step,
)
from easyicu.research_agent.execution.runners.target_trial_figure import (
    run_target_trial_figure,
    target_trial_figure_executor_code,
    target_trial_figure_executor_owns_step,
)
from easyicu.research_agent.figures.publication import audit_publication_exports
from easyicu.research_agent.methods import clone_censor_weight
from easyicu.research_agent.orchestration import scientific_runtime
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.scientific_review import timing_design_closed
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_tables import build_manuscript_tables
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)
from easyicu.research_agent.review.causal_audit import (
    scan_manuscript_for_causal_language,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRecord
from tests.support.target_trial import (
    OUTPUTS,
    synthetic_target_trial_cohort,
    target_trial_authority_body,
)

pytest.importorskip("lifelines")
pytest.importorskip("statsmodels")

STEP = "01_primary"
EVIDENCE = "statistic_step_summary_target_trial_suite"
#: Fewer resamples than the host's 500 keep the test short; the executed
#: design states the number the bootstrap ran.
RESAMPLES = 40


def _authority(**overrides) -> TargetTrialRuntimeAuthority:
    return build_current_case_scientific_runtime_authority(
        target_trial_authority_body(**overrides)
    )


def _draft() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": (
                "Does starting a vasopressor within six hours of time zero change "
                "28-day mortality?"
            ),
            "analysis_type": "causal_inference",
            "steps": [
                {
                    "step_id": STEP,
                    "planned_analysis_role": "primary",
                    "intent": "Draft primary step.",
                    "inputs": ["table:analysis_cohort"],
                    "expected_outputs": ["table:draft"],
                    "method": "draft_method",
                }
            ],
        }
    )


def _run(frame, out_dir, *, authority=None):
    authority = authority or _authority()
    summary = run_target_trial_suite(
        frame=frame,
        authority=authority.model_dump(mode="json"),
        runtime_projection_sha256="b" * 64,
        out_dir=out_dir,
        input_product="table:analysis_cohort",
        input_evidence_id="cohort_evidence",
        input_sha256="c" * 64,
    )
    return json.loads(json.dumps(summary, allow_nan=False))


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    """One run of the suite on the synthetic cohort, shared by the module."""

    original = clone_censor_weight.bootstrap_clone_censor_weight
    out_dir = tmp_path_factory.mktemp("target_trial") / "suite"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            clone_censor_weight,
            "bootstrap_clone_censor_weight",
            lambda estimate: original(estimate, resamples=RESAMPLES, seed=20261009),
        )
        summary = _run(synthetic_target_trial_cohort(), out_dir)
    return _authority(), summary, out_dir


def _numeric_leaves(value) -> int:
    if isinstance(value, dict):
        return sum(_numeric_leaves(child) for child in value.values())
    if isinstance(value, list):
        return sum(_numeric_leaves(child) for child in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def test_the_sealed_authority_closes_a_target_trial() -> None:
    authority = _authority()

    assert isinstance(authority, TargetTrialRuntimeAuthority)
    assert authority.required_columns == (
        "stay_id",
        "subject_id",
        "norepinephrine_onset_time",
        "vasopressin_onset_time",
        "mort_28d",
        "followup_days_28d",
        "death",
        "death_time",
        "los_icu",
        "age",
        "sex",
        "lactate_max",
        "map_min",
    )
    assert authority.plan_outputs == OUTPUTS
    assert authority.plan_rule_ref == (
        f"scientific_runtime_contract:{authority.execution_contract_sha256}"
    )
    assert authority.patient_group_requirement().group_source == "subject_id"
    assert (
        load_current_case_scientific_runtime_authority(
            authority.model_dump(mode="json")
        )
        == authority
    )
    tampered = {**authority.model_dump(mode="json"), "defer_label": "Late vasopressor"}
    with pytest.raises(ValueError, match="digest mismatch"):
        load_current_case_scientific_runtime_authority(tampered)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"time_zero_hours": 0}, "time zero is outside the host's menu"),
        ({"grace_period_hours": 25}, "grace period is outside the host's menu"),
        ({"treatment_onset_window_hours": [0.0, 11.0]}, "captured from ICU admission"),
        ({"host_policy_sha256": "d" * 64}, "another host policy"),
        (
            {
                "confirmation": {
                    "confirmed_by": "researcher",
                    "approval_event_id": "approval:card:0001",
                    "confirmed_compile_sha256": "e" * 64,
                    "n_lines_confirmed": 3,
                }
            },
            "confirmation is for another compile record",
        ),
        # The card showed one line fewer, or one more, than the record lists.
        *(
            (
                {
                    "confirmation": {
                        "confirmed_by": "researcher",
                        "approval_event_id": "approval:card:0001",
                        "confirmed_compile_sha256": "c" * 64,
                        "n_lines_confirmed": lines,
                    }
                },
                "another number of lines than its compile record lists",
            )
            for lines in (2, 4)
        ),
        ({"death_time_column": "deathtime"}, "host columns drifted"),
        ({"defer_label": "early vasopressor"}, "need different labels"),
        (
            {"initiate_label": "Vasopressor within 6 hours"},
            "String should match pattern",
        ),
        ({"treatment_onset_columns": []}, "one to four treatment onset columns"),
        ({"patient_group_column": "age"}, "patient group column has another role"),
        ({"resampling_unit": "icu_stay"}, "stay bootstrap declares no patient group"),
        ({"plan_outputs": [*OUTPUTS[1:], OUTPUTS[0]]}, "plan outputs must equal"),
    ],
)
def test_the_sealed_authority_refuses_an_open_contract(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _authority(**overrides)


def test_the_assumption_confirmation_is_a_researcher_s() -> None:
    confirmation = {
        "confirmed_by": "system",
        "approval_event_id": "approval:card:0001",
        "confirmed_compile_sha256": "c" * 64,
        "n_lines_confirmed": 3,
    }

    with pytest.raises(ValueError, match="researcher"):
        _authority(confirmation=confirmation)


def test_the_authority_binds_one_cohort_one_suite_and_one_figure_owner() -> None:
    authority = _authority()

    bound = authority.bind_plan(_draft())

    assert [step.method for step in bound.steps] == [
        "host_materialized_locked_cohort",
        "signed_target_trial_suite",
        "signed_target_trial_figure",
    ]
    suite_step, figure = bound.steps[1], bound.steps[2]
    assert suite_step.step_id == STEP
    assert tuple(suite_step.expected_outputs) == OUTPUTS[:-1]
    assert authority.plan_rule_ref in suite_step.icu_rule_refs
    assert suite_step.runtime_outcome_contract.outcomes == ("mort_28d",)
    assert [panel.chart_type for panel in figure.figure_panels] == [
        "timeline_diagram",
        "effect_curve",
        "love_plot",
        "trimming_panel",
    ]
    authority.validate_plan(bound)
    assert authority.governed_step(bound) == suite_step
    assert authority.governed_figure_step(bound) == figure
    assert target_trial_executor_owns_step(suite_step, plan=bound, authority=authority)
    assert target_trial_figure_executor_owns_step(
        figure, plan=bound, authority=authority
    )
    assert "run_target_trial_suite" in target_trial_executor_code(
        suite_step, authority=authority, runtime_projection_sha256="b" * 64
    )
    assert "run_target_trial_figure" in target_trial_figure_executor_code(
        figure, authority=authority
    )
    drifted = bound.model_copy(
        update={
            "steps": [
                bound.steps[0],
                suite_step.model_copy(update={"intent": "Another intent."}),
                figure,
            ]
        }
    )
    with pytest.raises(ValueError, match="drifted from signed authority: intent"):
        authority.validate_plan(drifted)


def test_the_host_compiles_the_plan_and_binds_the_run_inputs() -> None:
    authority = _authority()

    bound, spec = scientific_runtime._compile_current_case_plan(authority, _draft())

    assert spec.reason_code == "target_trial_suite_host_compiled"
    assert bound == authority.bind_plan(_draft())
    assert (
        scientific_runtime._compile_current_case_plan(
            authority, _draft(), development_execution_only=True
        )
        is None
    )
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    endpoint, exposure, _preferences = authorities.bind_run_inputs(
        endpoint=None, primary_exposure="norepinephrine", user_preferences=None
    )
    assert (endpoint.kind, endpoint.event_column, endpoint.time_column) == (
        "time_to_event",
        "mort_28d",
        "followup_days_28d",
    )
    assert exposure == "norepinephrine"


def test_the_review_reads_the_signed_suite_as_a_closed_temporal_owner() -> None:
    authority = _authority()
    bound = authority.bind_plan(_draft())
    suite_step = bound.steps[1]
    record = {
        "deterministic_standard_analysis": "signed_target_trial_suite",
        "deterministic_standard_selection_reason": (
            "signed_target_trial_suite_contract_preflight"
        ),
        "standard_executor_candidates": {"claimed_by": "signed_target_trial_suite"},
    }

    assert timing_design_closed(bound)
    unsigned = bound.model_copy(
        update={
            "steps": [
                bound.steps[0],
                suite_step.model_copy(update={"icu_rule_refs": []}),
                bound.steps[2],
            ]
        }
    )
    assert not timing_design_closed(unsigned)
    assert effect_output_authorized(suite_step, step_record=record)
    assert not effect_output_authorized(
        suite_step,
        step_record={
            **record,
            "deterministic_standard_analysis": "signed_landmark_survival_suite",
        },
    )
    assert sealed_suite_prespecified_axes(
        method=suite_step.method, rule_refs=suite_step.icu_rule_refs
    ) == ("model_specification",)


def test_the_suite_estimates_both_strategies_under_the_signed_design(suite) -> None:
    authority, summary, out_dir = suite

    assert summary["status"] == "ok"
    assert summary["analysis_family"] == "causal_inference"
    assert summary["paper_authorization_allowed"] is False
    assert summary["analysis_only"] is True
    envelope = summary["reportable_target_trial_results"]
    assert envelope["schema_version"] == "easyicu.target_trial_reporting/1"
    initiate, defer = envelope["arms"]["initiate"], envelope["arms"]["defer"]
    assert initiate["risk_percent"]["estimate"] - defer["risk_percent"]["estimate"] == (
        pytest.approx(envelope["risk_difference_percentage_points"]["estimate"])
    )
    assert envelope["bootstrap"]["resamples"] == RESAMPLES
    assert envelope["bootstrap"]["interval_method"] == "bootstrap_percentile"
    assert envelope["weight_truncation_percentiles"] == [1.0, 99.0]
    receipt = summary["scientific_runtime_receipt"]
    assert receipt["approval_event_id"] == "approval:card:0001"
    assert receipt["deaths_timed_by_calendar_day"] > 0
    assert list(receipt["eligibility_counts"]) == [
        "source_rows",
        "endpoint_observed",
        "alive_at_time_zero",
        "in_icu_at_time_zero",
        "no_treatment_start_before_time_zero",
        "baseline_covariates_usable",
    ]
    counts = list(receipt["eligibility_counts"].values())
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == summary["n_eligible"]
    assert sorted(summary["output_files"]) == sorted(OUTPUTS[:-1])
    for name in summary["output_files"].values():
        assert (out_dir / name).is_file()
    effects = pd.read_csv(out_dir / summary["output_files"][authority.effect_product])
    assert set(effects["weighting"]) == {
        "stabilized",
        "stabilized_truncated",
        "unweighted",
    }
    protocol = pd.read_csv(
        out_dir / summary["output_files"][authority.protocol_product]
    )
    assert len(protocol) == 9
    # The design and the estimates bind before the receipt under the cap.
    keys = list(summary)
    assert keys.index(EXECUTED_METHOD_DESIGN_KEY) < keys.index(
        "reportable_target_trial_results"
    )
    assert keys.index("reportable_target_trial_results") < keys.index(
        "scientific_runtime_receipt"
    )
    assert (
        _numeric_leaves(
            {
                key: summary[key]
                for key in (
                    EXECUTED_METHOD_DESIGN_KEY,
                    "reportable_target_trial_results",
                )
            }
        )
        <= 100
    )


def test_each_estimate_is_one_claim_in_the_fixed_template(suite) -> None:
    _authority_, summary, _out = suite

    drafts = derive_scientific_claim_drafts(summary)

    assert [(draft.claim_id, draft.analysis_role) for draft in drafts] == [
        ("strategy_risk_difference", "primary"),
        ("strategy_risk_ratio", "primary"),
        ("initiate_strategy_risk", "primary"),
        ("defer_strategy_risk", "primary"),
        ("truncated_weight_risk_difference", "sensitivity"),
        ("truncated_weight_risk_ratio", "sensitivity"),
    ]
    assert {draft.claim_type for draft in drafts} == {"target_trial_estimate"}
    assert {draft.schema_version for draft in drafts} == {"easyicu.scientific_claim/5"}
    claims = [
        ScientificClaim.model_validate(
            {**draft.model_dump(mode="json"), "step_id": STEP, "evidence_id": EVIDENCE}
        )
        for draft in drafts
    ]
    sentences = [claim.render_reader_text() for claim in claims] + [
        claim.render_reader_text(include_estimate=False) for claim in claims
    ]
    for sentence in sentences:
        assert sentence.startswith(
            "Under the emulation's assumptions of no unmeasured confounding, "
            "positivity, correctly specified weight models and censoring at ICU "
            "exit that the baseline covariates explain, the emulated target trial"
        )
    assert "with each strategy's weights truncated at the 1st and 99th percentiles" in (
        claims[4].render_reader_text()
    )
    assert (
        "Early vasopressor minus No early vasopressor" in claims[0].render_reader_text()
    )
    assert (
        scan_manuscript_for_causal_language(
            bound_manuscript=" ".join(sentences), effect_labels=[]
        )
        == []
    )
    difference = claims[0]
    assert difference.direction == (
        "positive"
        if difference.interval_lower > 0
        else "negative"
        if difference.interval_upper < 0
        else "no_clear_association"
    )


def test_the_envelope_s_contrasts_are_its_strategies_risks(suite) -> None:
    _authority_, summary, _out = suite
    envelope = summary["reportable_target_trial_results"]

    def claims_from(**changes):
        return derive_scientific_claim_drafts(
            {**summary, "reportable_target_trial_results": {**envelope, **changes}}
        )

    assert claims_from()
    for key in ("risk_ratio", "truncated_risk_ratio"):
        shifted = {**envelope[key], "estimate": envelope[key]["estimate"] * 1.000001}
        with pytest.raises(ValueError, match="a risk ratio is the strategies' ratio"):
            claims_from(**{key: shifted})
    with pytest.raises(ValueError, match="placed at the ICU exit was timed by its day"):
        claims_from(
            deaths_placed_at_icu_exit=envelope["deaths_timed_by_calendar_day"] + 1
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"weight_truncation_percentiles": [2.5, 97.5]}, "percentiles are whole"),
        (
            {"deaths_timed_by_calendar_day": 0, "deaths_placed_at_icu_exit": 1},
            "placed at the ICU exit was timed by its day",
        ),
    ],
)
def test_the_executed_design_is_closed(suite, changes, message) -> None:
    _authority_, summary, _out = suite

    with pytest.raises(ValueError, match=message):
        validate_executed_method_design(
            {**summary[EXECUTED_METHOD_DESIGN_KEY], **changes}
        )


def test_the_tables_and_the_writer_digest_carry_the_suite(suite, tmp_path) -> None:
    authority, summary, _out = suite

    tables = validate_manuscript_table_declarations(summary[MANUSCRIPT_TABLES_KEY])

    assert [table.body.layout for table in tables] == [
        "protocol_rows",
        "stage_flow",
        "grouped_summary",
    ]
    assert tables[0].body.item_labels["time_zero"]
    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": STEP,
                "status": "ok",
                "generation_mode": "deterministic_standard",
                "step_summary": summary,
            }
        ],
        run_dir=tmp_path,
        evidence=None,
    )
    lines = digest.splitlines()
    head = next(
        index for index, line in enumerate(lines) if line.startswith(f"- {STEP} [")
    )
    row = json.loads(lines[head + 1])
    assert (
        row["reportable_target_trial_results"]["risk_difference_percentage_points"]
        == (
            summary["reportable_target_trial_results"][
                "risk_difference_percentage_points"
            ]
        )
    )


def _register(run_dir, evidence_id, kind, payload: bytes, name: str) -> EvidenceRecord:
    target = run_dir / "evidence" / f"{evidence_id}__{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        description=evidence_id,
        relative_path=f"evidence/{target.name}",
        sha256=hashlib.sha256(payload).hexdigest(),
        produced_by_step=STEP,
        producer="runner",
        generation_mode="deterministic_standard",
    )


def test_the_protocol_the_flow_and_table_one_print_from_the_run_s_evidence(
    suite, tmp_path
) -> None:
    authority, summary, out_dir = suite
    run_dir = tmp_path / "tables"
    records = [
        _register(
            run_dir,
            f"statistic_step_summary_{STEP}",
            "statistic",
            json.dumps(summary).encode(),
            "step_summary.json",
        )
    ]
    for product, name in summary["output_files"].items():
        if product.startswith("table:"):
            records.append(
                _register(
                    run_dir,
                    f"table_{STEP}_{product.split(':', 1)[1]}",
                    "table",
                    (out_dir / name).read_bytes(),
                    name,
                )
            )
    plan = AnalysisPlan(
        research_question="Does an early vasopressor start change 28-day mortality?",
        steps=[
            AnalysisStep(
                step_id=STEP,
                intent="Execute the signed target trial suite",
                inputs=["stay_id"],
                expected_outputs=[
                    authority.protocol_product,
                    authority.eligibility_product,
                    authority.table_one_product,
                ],
                method="signed_target_trial_suite",
            )
        ],
        display_labels={
            "age": "Age",
            "sex": "Sex",
            "lactate_max": "Lactate",
            "map_min": "Mean arterial pressure",
        },
    )

    protocol, flow, table_one = build_manuscript_tables(
        plan=plan, evidence_records=records, run_dir=run_dir
    )

    assert protocol.columns == ("Protocol element", "Specification")
    assert [row[0] for row in protocol.rows] == [
        "Eligibility",
        "Treatment strategies",
        "Assignment",
        "Time zero",
        "Grace period",
        "Follow-up",
        "Outcome",
        "Contrast",
        "Statistical analysis",
    ]
    assert protocol.rows[3][1].startswith("6 hours after ICU admission")
    # The vital status at the horizon is known only after time zero.
    assert protocol.rows[0][1].endswith("and with a known vital status at day 28")
    assert any("the data cannot test" in note for note in protocol.notes)
    assert [row[0] for row in flow.rows] == [
        "Source cohort",
        "Endpoint observed",
        "Alive at time zero",
        "In the ICU at time zero",
        "No start of a vasopressor before time zero",
        "Baseline covariates usable",
    ]
    assert flow.rows[-1][1] == f"{summary['n_eligible']:,}" or flow.rows[-1][1] == str(
        summary["n_eligible"]
    )
    assert table_one.columns[0] == "Characteristic"
    assert table_one.rows[-1][0] == "Deaths by day 28, n (%)"


def _store(tmp_path, summary):
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic",
        description="Signed target trial suite summary",
        source_path=source,
        evidence_id=EVIDENCE,
        produced_by_step=STEP,
        producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP,
        evidence_id=EVIDENCE,
        summary=summary,
        max_leaves=100,
    )
    return store, [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]


def test_the_design_binds_in_methods_and_its_assumptions_in_limitations(
    suite, tmp_path
) -> None:
    _authority_, summary, _out = suite
    design = validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY])
    assert isinstance(design, TargetTrialDesign)
    assert design.bootstrap_resamples == RESAMPLES
    store, ledger = _store(tmp_path, summary)

    facts = {
        fact.section: fact
        for fact in store.manuscript_method_facts(ledger)
        if fact.source_field.startswith(f"{STEP}.")
    }

    methods, limitations = facts["variables"], facts["limitations"]
    assert methods.text.startswith("Executed target trial design: ")
    assert "6 hours after ICU admission" in methods.text
    assert "within a 6-hour grace period" in methods.text
    assert "at the 1st and 99th percentiles of the weights" in methods.text
    assert "deaths after hospital discharge were timed by calendar day" in methods.text
    assert "risk difference" not in methods.text
    assert "confidence interval" not in methods.text
    assert limitations.text.startswith("Target trial emulation assumptions: ")
    assert "no unmeasured confounding" in limitations.text
    assert (
        "eligibility also required a known vital status at day 28, which is "
        "learned only after time zero"
    ) in limitations.text
    scaffold = (
        f"## Methods\n\n### Variables\n\n{methods.scaffold}\n\n"
        f"## Limitations\n\n{limitations.scaffold}\n"
    )
    safe, _removed = store.enforce_evidence_bound_scaffold(
        scaffold, per_step_records=ledger
    )
    assert methods.scaffold in safe and limitations.scaffold in safe
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced
    # A fact is admitted only in its own section: in Methods the assumptions
    # line has no authority, so the strict store refuses the scaffold.
    misplaced = f"## Methods\n\n### Variables\n\n{limitations.scaffold}\n"
    with pytest.raises(EvidenceEnforcementError, match="lacks deterministic evidence"):
        store.enforce_evidence_bound_scaffold(misplaced, per_step_records=ledger)


def test_the_figure_draws_the_signed_tables(suite, tmp_path) -> None:
    authority, summary, out_dir = suite
    paths = {
        product: out_dir / name for product, name in summary["output_files"].items()
    }

    result = run_target_trial_figure(
        protocol=pd.read_csv(paths[authority.protocol_product]),
        eligibility=pd.read_csv(paths[authority.eligibility_product]),
        risk_curves=pd.read_csv(paths[authority.risk_curve_product]),
        effects=pd.read_csv(paths[authority.effect_product]),
        balance=pd.read_csv(paths[authority.balance_product]),
        weights=pd.read_csv(paths[authority.weight_product]),
        source_paths={
            product: paths[product] for product in authority.figure_input_products
        },
        authority=authority.model_dump(mode="json"),
        out_dir=tmp_path / "figure",
    )

    assert result["output_files"] == {
        "figure:target_trial_emulation": "target_trial_emulation.svg"
    }
    assert audit_publication_exports(tmp_path / "figure") == []


def _stopped(tmp_path, frame) -> tuple[ExecutorStop, tuple[str, str | None]]:
    out_dir = tmp_path / "stopped"
    with pytest.raises(ExecutorStop) as raised:
        _run(frame, out_dir)
    record = parse_executor_stop_record(
        (out_dir / EXECUTOR_STOP_RECORD_NAME).read_bytes(),
        expected_owner="signed_target_trial_suite",
    )
    return raised.value, (record.reason_code, record.cause_code)


def test_too_few_eligible_stays_stop_the_suite(tmp_path) -> None:
    stop, record = _stopped(tmp_path, synthetic_target_trial_cohort(n=90, seed=3))

    assert record == ("target_trial_sample_insufficient", None)
    assert str(stop).startswith("target_trial_sample_insufficient: ")


def test_a_strategy_nobody_follows_stops_the_suite(tmp_path) -> None:
    frame = synthetic_target_trial_cohort(n=600, seed=5).assign(
        norepinephrine_onset_time=np.nan, vasopressin_onset_time=np.nan
    )

    _stop, record = _stopped(tmp_path, frame)

    assert record == ("target_trial_strategy_unobserved", None)


def test_a_declared_level_nobody_has_stops_the_weight_model(tmp_path) -> None:
    frame = synthetic_target_trial_cohort(n=600, seed=5).assign(sex="male")

    _stop, record = _stopped(tmp_path, frame)

    assert record == ("target_trial_weight_model_not_estimable", "singular_design")


def test_frequent_icu_exits_in_the_grace_period_stop_the_suite(tmp_path) -> None:
    frame = synthetic_target_trial_cohort(n=1500, seed=5)
    onset = frame[["norepinephrine_onset_time", "vasopressin_onset_time"]].min(axis=1)
    alive = frame["death_time"].isna() | (frame["death_time"] > 8.0)
    leaving = (onset.isna() | (onset > 8.0)) & alive & (frame.index % 4 == 0)
    frame.loc[leaving, "los_icu"] = 8.0 / 24.0

    _stop, record = _stopped(tmp_path, frame)

    assert record == ("target_trial_icu_exit_excessive", None)


def test_a_death_time_the_follow_up_contradicts_is_refused(tmp_path) -> None:
    frame = synthetic_target_trial_cohort(n=300, seed=5)
    died = frame.index[frame["death"] == 1][0]
    frame.loc[died, "death_time"] = 40.0 * 24.0

    with pytest.raises(ValueError, match="death"):
        _run(frame, tmp_path / "contradiction")


def _day_in_grace(monkeypatch):
    """Sixteen stays that leave the ICU and then die, without starting.

    A time zero 12 hours after admission and an 18-hour grace period put hour
    24 inside the grace period.  The stays never start the vasopressor; eight
    leave the ICU at hour 40, after the grace period, and eight at hour 26,
    within it.
    """

    original = clone_censor_weight.bootstrap_clone_censor_weight
    monkeypatch.setattr(
        clone_censor_weight,
        "bootstrap_clone_censor_weight",
        lambda estimate: original(estimate, resamples=RESAMPLES, seed=20261009),
    )
    authority = _authority(
        time_zero_hours=12,
        grace_period_hours=18,
        treatment_onset_window_hours=[0.0, 30.0],
    )
    frame = synthetic_target_trial_cohort(time_zero=12, grace=18)
    onset = frame[["norepinephrine_onset_time", "vasopressin_onset_time"]]
    never = frame.index[onset.isna().all(axis=1)]
    exit_hours = pd.Series([40.0] * 8 + [26.0] * 8, index=never[:16])
    frame.loc[exit_hours.index, "los_icu"] = exit_hours / 24.0
    frame.loc[exit_hours.index, "mort_28d"] = 1
    return authority, frame, exit_hours


def _died(frame, exit_hours, *, in_hospital, hours_after_exit=None):
    """The stays die in hospital or after it, at a recorded hour or on day 1."""

    rows = exit_hours.index
    died = frame.copy()
    died.loc[rows, "death"] = int(in_hospital)
    if hours_after_exit is None:
        died.loc[rows, "death_time"] = np.nan
        died.loc[rows, "followup_days_28d"] = 1.0
    else:
        died.loc[rows, "death_time"] = exit_hours + hours_after_exit
        died.loc[rows, "followup_days_28d"] = (exit_hours + hours_after_exit) / 24.0
    return died


def _events(run):
    arms = run["reportable_target_trial_results"]["arms"]
    return {arm: arms[arm]["n_events"] for arm in ("initiate", "defer")}


def _bound_methods(tmp_path, summary):
    """The executed design's Methods sentence, with every number it binds."""

    store, ledger = _store(tmp_path, summary)
    methods = next(
        fact
        for fact in store.manuscript_method_facts(ledger)
        if fact.section == "variables" and fact.source_field.startswith(f"{STEP}.")
    )
    scaffold = f"## Methods\n\n### Variables\n\n{methods.scaffold}\n"
    safe, _removed = store.enforce_evidence_bound_scaffold(
        scaffold, per_step_records=ledger
    )
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced
    return methods.text


def test_a_death_known_by_its_day_before_the_icu_exit_is_placed_at_the_exit(
    tmp_path, monkeypatch
) -> None:
    authority, frame, exit_hours = _day_in_grace(monkeypatch)
    timed = _run(
        _died(frame, exit_hours, in_hospital=True, hours_after_exit=1.0),
        tmp_path / "timed",
        authority=authority,
    )
    by_day = _run(
        _died(frame, exit_hours, in_hospital=False),
        tmp_path / "by_day",
        authority=authority,
    )
    moved = len(exit_hours)

    receipt, timed_receipt = (
        run["scientific_runtime_receipt"] for run in (by_day, timed)
    )
    assert receipt["deaths_placed_at_icu_exit"] == moved
    assert receipt["other_deaths_timed_by_calendar_day"] == 0
    assert timed_receipt["deaths_placed_at_icu_exit"] == 0
    assert by_day["reportable_target_trial_results"]["deaths_placed_at_icu_exit"] == (
        moved
    )
    # Placed by its day, inside the grace period, each death would be one
    # more death of the starting strategy; placed at the exit, it follows the
    # exit, so the starting strategy censors the stay when it leaves the ICU
    # or the grace period ends, as when the hospital timed the death.
    assert receipt["eligibility_counts"] == timed_receipt["eligibility_counts"]
    assert _events(by_day) == _events(timed)

    design = validate_executed_method_design(by_day[EXECUTED_METHOD_DESIGN_KEY])
    assert design.deaths_placed_at_icu_exit == moved
    assert (
        f"; {receipt['deaths_timed_by_calendar_day']} deaths after hospital "
        f"discharge were timed by calendar day, {moved} of them placed at the ICU "
        "exit because their day fell before it"
    ) in _bound_methods(tmp_path / "by_day_methods", by_day)


def test_a_hospital_death_without_a_time_is_placed_at_the_exit_as_a_death_there(
    tmp_path, monkeypatch
) -> None:
    authority, frame, exit_hours = _day_in_grace(monkeypatch)
    at_exit = _run(
        _died(frame, exit_hours, in_hospital=True, hours_after_exit=0.0),
        tmp_path / "at_exit",
        authority=authority,
    )
    untimed = _run(
        _died(frame, exit_hours, in_hospital=True),
        tmp_path / "untimed",
        authority=authority,
    )
    after = _run(
        _died(frame, exit_hours, in_hospital=True, hours_after_exit=1.0),
        tmp_path / "after",
        authority=authority,
    )
    moved = len(exit_hours)

    # A hospital death the input does not time may have been the ICU exit
    # itself: placed there, it is a death in the ICU, as one recorded there.
    receipt, at_exit_receipt = (
        run["scientific_runtime_receipt"] for run in (untimed, at_exit)
    )
    assert receipt["other_deaths_timed_by_calendar_day"] == moved
    assert receipt["deaths_placed_at_icu_exit"] == moved
    after_discharge = "deaths_timed_by_calendar_day"
    assert receipt[after_discharge] == at_exit_receipt[after_discharge]
    assert _events(untimed) == _events(at_exit)
    # As deaths in the ICU, the eight that leave within the grace period count
    # for the starting strategy too.
    assert _events(untimed)["initiate"] == _events(after)["initiate"] + 8
    assert _events(untimed)["defer"] == _events(after)["defer"]
    assert (
        f"; {receipt['deaths_timed_by_calendar_day']} deaths after hospital "
        f"discharge and {moved} other deaths without a recorded time were timed "
        f"by calendar day, {moved} of them placed at the ICU exit because their "
        "day fell before it"
    ) in _bound_methods(tmp_path / "untimed_methods", untimed)


def test_a_death_known_by_its_day_a_full_day_before_the_icu_exit_is_refused(
    tmp_path,
) -> None:
    frame = synthetic_target_trial_cohort(n=300, seed=5)
    by_day = frame.index[(frame["mort_28d"] == 1) & (frame["death"] == 0)][:3]
    # The day of death ended a day before the stay left the ICU: the death
    # happened in the ICU, where the hospital would have timed it.
    frame.loc[by_day, "los_icu"] = frame.loc[by_day, "followup_days_28d"] + 2.0

    with pytest.raises(
        ValueError, match="3 deaths known only by their day fell a full day before"
    ):
        _run(frame, tmp_path / "contradiction")
