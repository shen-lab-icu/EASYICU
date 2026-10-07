"""A continuous exposure has its own sealed landmark survival suite.

The signed landmark survival suite contrasts one binary incident exposure with
its comparator, and every other layer refused a continuous exposure: a survival
question about a laboratory value or a vital sign had no executable owner.  The
continuous suite models one window summary recorded by the landmark per unit of
its source's scale.  It shares the binary suite's landmark risk set, endpoint,
PH rule and interval model, describes the risk set by exposure tertile, checks
the linear term against a restricted cubic spline, reports its estimates as
host claims, and states its executed design as one Methods fact.

Synthetic, seeded rows only (a laboratory value and 28-day mortality).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.landmark_continuous_survival_runtime import (
    LandmarkContinuousSurvivalRuntimeAuthority,
)
from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    LandmarkContinuousSurvivalDesign,
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from easyicu.research_agent.contracts.sealed_suite_robustness import (
    sealed_suite_prespecified_axes,
)
from easyicu.research_agent.contracts.step_families import effect_output_authorized
from easyicu.research_agent.execution.runners.landmark_continuous_survival_executor import (
    landmark_continuous_survival_executor_code,
    landmark_continuous_survival_executor_owns_step,
    run_landmark_continuous_survival_suite,
)
from easyicu.research_agent.execution.runners.landmark_continuous_survival_figure import (
    landmark_continuous_survival_figure_executor_code,
    landmark_continuous_survival_figure_executor_owns_step,
    run_landmark_continuous_survival_figure,
)
from easyicu.research_agent.figures.publication import FigureContract, audit_publication_exports
from easyicu.research_agent.methods.time_varying_cox import (
    TimeVaryingCoxError,
    fit_piecewise_time_varying_cox,
)
from easyicu.research_agent.orchestration import scientific_runtime
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.scientific_review import timing_design_closed
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest
from easyicu.research_agent.schema import AnalysisPlan
from tests.support.continuous_survival import OUTPUTS, continuous_authority_body

pytest.importorskip("lifelines")

STEP = "01_primary"
EVIDENCE = "statistic_step_summary_continuous_survival_suite"


def _authority(**overrides) -> LandmarkContinuousSurvivalRuntimeAuthority:
    return build_current_case_scientific_runtime_authority(continuous_authority_body(**overrides))


def _rows(n: int = 1500, *, seed: int = 7, crossing: bool = False, exposure=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(65.0, 12.0, n)
    sex = rng.choice(["F", "M"], n)
    lab = np.exp(rng.normal(0.7, 0.5, n)) if exposure is None else exposure(rng, n)
    lab[rng.random(n) < 0.08] = np.nan
    rate = np.exp(-4.2 + 0.25 * np.nan_to_num(lab, nan=2.0) + 0.02 * (age - 65.0))
    time = rng.exponential(1.0 / rate)
    if crossing:
        # The exposure raises the hazard early only; after day 10 it is gone.
        time = np.where(time > 10.0, rng.exponential(1.0 / np.exp(-4.0), n) + 10.0, time)
    time = np.where(rng.random(n) < 0.05, rng.uniform(0.0, 1.0, n), time)
    return pd.DataFrame(
        {
            "lab_max": lab,
            "mort_28d": (time <= 28.0).astype(int),
            "followup_days_28d": np.minimum(time, 28.0),
            "age": age,
            "sex": sex,
        }
    )


def _draft() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Is the first-day laboratory value associated with 28-day mortality?",
            "analysis_type": "survival",
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


def _run(tmp_path, frame, *, authority=None, name="suite"):
    authority = authority or _authority()
    summary = run_landmark_continuous_survival_suite(
        frame=frame,
        authority=authority.model_dump(mode="json"),
        runtime_projection_sha256="b" * 64,
        out_dir=tmp_path / name,
        input_product="table:analysis_cohort",
        input_evidence_id="cohort_evidence",
        input_sha256="c" * 64,
    )
    return authority, json.loads(json.dumps(summary, allow_nan=False))


def _numeric_leaves(value) -> int:
    if isinstance(value, dict):
        return sum(_numeric_leaves(child) for child in value.values())
    if isinstance(value, list):
        return sum(_numeric_leaves(child) for child in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def test_the_sealed_authority_closes_a_continuous_suite() -> None:
    authority = _authority()

    assert isinstance(authority, LandmarkContinuousSurvivalRuntimeAuthority)
    assert authority.required_columns == ("lab_max", "mort_28d", "followup_days_28d", "age", "sex")
    assert authority.plan_rule_ref == f"scientific_runtime_contract:{authority.execution_contract_sha256}"
    assert load_current_case_scientific_runtime_authority(authority.model_dump(mode="json")) == authority
    tampered = {**authority.model_dump(mode="json"), "exposure_label": "Another value"}
    with pytest.raises(ValueError, match="digest mismatch"):
        load_current_case_scientific_runtime_authority(tampered)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"exposure_window_hours": [0.0, 36.0]}, "window must close by the landmark"),
        ({"exposure_window_hours": [-6.0, 24.0]}, "window must close by the landmark"),
        ({"time_varying_interval_cutpoints_days": []}, "intervals must be increasing"),
        ({"time_varying_interval_cutpoints_days": [7.0, 27.5]}, "intervals must be increasing"),
        ({"spline_knot_quantiles": [0.05, 0.5, 0.95]}, "frozen 10/50/90 knots"),
        ({"adjustment_columns": ["age", "lab_max"], "table_one_columns": ["age"],
          "categorical_adjustment_columns": []}, "source columns must be unique"),
        ({"table_one_columns": ["age", "bmi"]}, "Table 1 columns"),
        ({"derived_time_column": "age"}, "derived columns"),
        ({"plan_outputs": [*OUTPUTS[1:], OUTPUTS[0]]}, "plan outputs must equal"),
    ],
)
def test_the_sealed_authority_refuses_an_open_contract(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _authority(**overrides)


def test_the_authority_binds_one_cohort_one_suite_and_one_figure_owner() -> None:
    authority = _authority()

    bound = authority.bind_plan(_draft())

    assert [step.method for step in bound.steps] == [
        "host_materialized_locked_cohort",
        "signed_landmark_continuous_survival_suite",
        "signed_landmark_continuous_survival_figure",
    ]
    suite, figure = bound.steps[1], bound.steps[2]
    assert suite.step_id == STEP
    assert tuple(suite.expected_outputs) == OUTPUTS[:-1]
    assert authority.plan_rule_ref in suite.icu_rule_refs
    assert suite.runtime_outcome_contract.outcomes == ("mort_28d",)
    assert [panel.chart_type for panel in figure.figure_panels] == [
        "kaplan_meier_curve", "hazard_ratio_curve", "cohort_flow", "schoenfeld_plot",
    ]
    assert figure.figure_panels[1].policy_alternative_chart_types == [
        "time_varying_hazard_ratio_forest"
    ]
    authority.validate_plan(bound)
    assert authority.governed_step(bound) == suite
    assert authority.governed_figure_step(bound) == figure
    assert landmark_continuous_survival_executor_owns_step(suite, plan=bound, authority=authority)
    assert landmark_continuous_survival_figure_executor_owns_step(figure, plan=bound, authority=authority)
    assert "run_landmark_continuous_survival_suite" in landmark_continuous_survival_executor_code(
        suite, authority=authority, runtime_projection_sha256="b" * 64
    )
    assert "run_landmark_continuous_survival_figure" in landmark_continuous_survival_figure_executor_code(
        figure, authority=authority
    )
    drifted = bound.model_copy(
        update={"steps": [bound.steps[0], suite.model_copy(update={"intent": "Another intent."}), figure]}
    )
    with pytest.raises(ValueError, match="drifted from signed authority: intent"):
        authority.validate_plan(drifted)


def test_the_host_compiles_the_plan_and_binds_the_run_inputs() -> None:
    authority = _authority()

    bound, spec = scientific_runtime._compile_current_case_plan(authority, _draft())

    assert spec.reason_code == "landmark_continuous_survival_suite_host_compiled"
    assert bound == authority.bind_plan(_draft())
    assert scientific_runtime._compile_current_case_plan(
        authority, _draft(), development_execution_only=True
    ) is None
    development = _authority(development_execution_only_allowed=True)
    projected, _spec = scientific_runtime._compile_current_case_plan(
        development, _draft(), development_execution_only=True
    )
    development.validate_plan(projected)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    endpoint, exposure, _preferences = authorities.bind_run_inputs(
        endpoint=None, primary_exposure=None, user_preferences=None
    )
    assert (endpoint.kind, endpoint.event_column, endpoint.time_column) == (
        "time_to_event", "mort_28d", "followup_days_28d",
    )
    assert exposure == "lab_max"
    with pytest.raises(ValueError, match="primary exposure conflicts"):
        authorities.bind_run_inputs(
            endpoint=None, primary_exposure="lab_mean", user_preferences=None
        )


def test_the_review_reads_the_signed_suite_as_a_closed_temporal_owner() -> None:
    authority = _authority()
    bound = authority.bind_plan(_draft())
    suite = bound.steps[1]
    record = {
        "deterministic_standard_analysis": "signed_landmark_continuous_survival_suite",
        "deterministic_standard_selection_reason": (
            "signed_landmark_continuous_survival_suite_contract_preflight"
        ),
        "standard_executor_candidates": {
            "claimed_by": "signed_landmark_continuous_survival_suite"
        },
    }

    assert timing_design_closed(bound)
    assert effect_output_authorized(suite, step_record=record)
    assert not effect_output_authorized(
        suite, step_record={**record, "deterministic_standard_analysis": "signed_landmark_survival_suite"}
    )
    assert sealed_suite_prespecified_axes(
        method=suite.method, rule_refs=suite.icu_rule_refs
    ) == ("timing", "model_specification")
    unsigned = suite.model_copy(update={"icu_rule_refs": []})
    assert sealed_suite_prespecified_axes(method=unsigned.method, rule_refs=[]) == ()


def test_the_suite_reports_one_per_unit_hazard_ratio_when_ph_holds(tmp_path) -> None:
    authority, summary = _run(tmp_path, _rows())

    envelope = summary["reportable_survival_results"]
    assert envelope["schema_version"] == "easyicu.continuous_survival_reporting/1"
    assert envelope["proportional_hazards_test"]["disposition"] == "assumption_not_rejected"
    assert envelope["constant_hazard_ratio_authorized"] is True
    per_unit = envelope["adjusted_hazard_ratio_per_unit"]
    assert per_unit["ci_low"] > 1.0
    assert len(envelope["time_varying_adjusted_association"]["intervals"]) == 3
    assert envelope["functional_form"]["status"] == "estimated"
    assert envelope["functional_form"]["knot_percentiles"] == [10.0, 50.0, 90.0]
    claims = derive_scientific_claim_drafts(summary)
    assert [(claim.claim_id, claim.analysis_role) for claim in claims] == [
        ("adjusted_hazard_ratio_per_unit", "primary"),
        ("interval_1_adjusted_hazard_ratio_per_unit", "secondary"),
        ("interval_2_adjusted_hazard_ratio_per_unit", "secondary"),
        ("interval_3_adjusted_hazard_ratio_per_unit", "secondary"),
        ("proportional_hazards_rule", "primary"),
    ]
    assert claims[0].estimand == (
        "adjusted hazard ratio per 1 mmol/L increase in the exposure over the "
        "post-landmark follow-up"
    )
    assert claims[0].direction == "positive"
    assert claims[0].adjusted_for == ["age", "sex"]
    assert claims[0].point_estimate == pytest.approx(per_unit["hazard_ratio"])
    # Headline estimates and the design come before the audit under the cap.
    keys = list(summary)
    assert keys.index(EXECUTED_METHOD_DESIGN_KEY) < keys.index("reportable_survival_results")
    assert keys.index("reportable_survival_results") < keys.index("missingness_measurement_audit")
    assert _numeric_leaves(summary) <= 100
    tables = validate_manuscript_table_declarations(summary[MANUSCRIPT_TABLES_KEY])
    table_one = tables[0].body
    assert [group.prefix for group in table_one.groups] == ["t1", "t2", "t3"]
    assert sum(group.n for group in table_one.groups) == summary["n_landmark_population"]
    assert tables[1].body.stage_labels["landmark_analysis_population"].endswith("value")
    # The Writer reads the authorized constant estimate inside the envelope.
    row = _digest_row(tmp_path, summary)
    assert row["reportable_survival_results"]["adjusted_hazard_ratio_per_unit"] == per_unit


def test_a_rejected_ph_test_withholds_the_constant_per_unit_estimate(tmp_path) -> None:
    _authority_, summary = _run(tmp_path, _rows(crossing=True, seed=11))

    envelope = summary["reportable_survival_results"]
    assert envelope["proportional_hazards_test"]["disposition"] == "assumption_rejected"
    assert envelope["constant_hazard_ratio_authorized"] is False
    assert "adjusted_hazard_ratio_per_unit" not in envelope
    assert summary["proportional_hazards_status"] == "violation_block_paper_authorization"
    claims = derive_scientific_claim_drafts(summary)
    assert [claim.claim_id for claim in claims if claim.analysis_role == "primary"] == [
        "interval_1_adjusted_hazard_ratio_per_unit",
        "interval_2_adjusted_hazard_ratio_per_unit",
        "interval_3_adjusted_hazard_ratio_per_unit",
        "proportional_hazards_rule",
    ]
    abstract = [
        claim["scientific_claim_id"]
        for claim in envelope["manuscript_projection"]["claims"]
        if "scientific_claim_id" in claim
    ]
    assert abstract == [
        "interval_1_adjusted_hazard_ratio_per_unit",
        "interval_2_adjusted_hazard_ratio_per_unit",
        "interval_3_adjusted_hazard_ratio_per_unit",
    ]
    # The interval block is the result, so no generic effect key is a headline.
    row = _digest_row(tmp_path, summary)
    assert not {"estimate", "effect_estimate", "hazard_ratio", "ci_low", "ci_high"} & set(row)
    assert "adjusted_hazard_ratio_per_unit" not in row["reportable_survival_results"]
    assert len(row["reportable_survival_results"]["time_varying_adjusted_association"]["intervals"]) == 3


def _digest_row(tmp_path, summary) -> dict:
    digest = _render_writer_evidence_digest(
        [{"step_id": STEP, "status": "ok", "generation_mode": "deterministic_standard", "step_summary": summary}],
        run_dir=tmp_path, evidence=None,
    )
    lines = digest.splitlines()
    head = next(index for index, line in enumerate(lines) if line.startswith(f"- {STEP} ["))
    return json.loads(lines[head + 1])


def test_a_spline_check_without_a_result_is_reported_without_one(tmp_path) -> None:
    def heaped(rng, n):
        # More than half the risk set at one value: the 10th and 50th
        # percentiles tie while the tertiles stay distinct.
        return np.where(rng.random(n) < 0.6, 1.0, 1.0 + np.exp(rng.normal(0.3, 0.5, n)))

    _authority_, summary = _run(tmp_path, _rows(exposure=heaped))

    envelope = summary["reportable_survival_results"]
    assert envelope["functional_form"] == {
        "method": "restricted_cubic_spline_likelihood_ratio_test",
        "status": "not_estimable",
        "reason": "tied_knots",
        "knot_percentiles": [10.0, 50.0, 90.0],
    }
    assert all(
        claim["claim_id"] != "restricted_cubic_spline_check"
        for claim in envelope["manuscript_projection"]["claims"]
    )
    assert derive_scientific_claim_drafts(summary)[0].claim_id == "adjusted_hazard_ratio_per_unit"


def _store(tmp_path, summary, *, max_leaves=None):
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic", description="Signed continuous survival suite summary",
        source_path=source, evidence_id=EVIDENCE, produced_by_step=STEP,
        producer="runner", generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id=EVIDENCE, summary=summary, max_leaves=max_leaves,
    )
    return store, [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]


@pytest.mark.parametrize("max_leaves", [None, 12])
def test_the_executed_design_is_one_exact_bound_methods_fact(tmp_path, max_leaves) -> None:
    authority, summary = _run(tmp_path, _rows())
    design = validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY])
    assert isinstance(design, LandmarkContinuousSurvivalDesign)
    assert (design.exposure_window_start_hours, design.exposure_window_end_hours) == (0.0, 24.0)
    assert design.exposure_window_summary == "max"
    assert design.spline_knot_percentiles == [10.0, 50.0, 90.0]
    store, ledger = _store(tmp_path, summary, max_leaves=max_leaves)

    (fact,) = [
        fact for fact in store.manuscript_method_facts(ledger)
        if fact.source_field.endswith(EXECUTED_METHOD_DESIGN_KEY)
    ]

    assert fact.text.startswith("Executed survival design: ")
    assert "the highest value of the exposure source in the first 24 hours" in fact.text
    assert "per 1 unit of the exposure's recorded scale" in fact.text
    assert "split at days 7 and 14 after the landmark" in fact.text
    assert "knots at the 10th, 50th and 90th percentiles" in fact.text
    assert "hazard ratio" not in fact.text and "confidence interval" not in fact.text
    assert fact.executed_hour_spans == ((0.0, 24.0),)
    scaffold = f"## Methods\n\n### Variables\n\n{fact.scaffold}\n"
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold, per_step_records=ledger)
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced


@pytest.mark.parametrize("crossing", [False, True], ids=["ph_holds", "ph_rejected"])
def test_the_figure_draws_the_estimate_its_ph_decision_allows(tmp_path, crossing) -> None:
    authority, summary = _run(tmp_path, _rows(crossing=crossing, seed=11 if crossing else 7))
    out = tmp_path / "suite"
    paths = {product: out / name for product, name in summary["output_files"].items()}

    result = run_landmark_continuous_survival_figure(
        km_table=pd.read_csv(paths[authority.km_product]),
        spline_table=pd.read_csv(paths[authority.spline_product]),
        time_varying_table=pd.read_csv(paths[authority.time_varying_cox_product]),
        risk_flow=pd.read_csv(paths[authority.risk_set_product]),
        ph_table=pd.read_csv(paths[authority.ph_product]),
        source_paths={product: paths[product] for product in authority.figure_input_products},
        authority=authority.model_dump(mode="json"),
        out_dir=tmp_path / "figure",
    )

    assert result["output_files"] == {
        "figure:landmark_continuous_survival_suite": "landmark_continuous_survival_suite.svg"
    }
    assert audit_publication_exports(tmp_path / "figure") == []
    contract = FigureContract.model_validate_json(
        (tmp_path / "figure" / result["figure_assets"]["contract"]).read_text(encoding="utf-8")
    )
    panel_b = next(panel for panel in contract.panels if panel.panel_id == "b")
    assert panel_b.metadata["chart_type"] == (
        "time_varying_hazard_ratio_forest" if crossing else "hazard_ratio_curve"
    )
    caption = contract.reader_caption
    assert ("each follow-up interval" in caption) is crossing
    assert ("restricted cubic spline model" in caption) is not crossing


def test_an_interval_without_an_event_refuses_a_continuous_interval_model() -> None:
    frame = _rows().dropna()
    frame = frame.assign(time=frame["followup_days_28d"] - 1.0).loc[lambda rows: rows["time"] > 0]
    frame.loc[frame["time"] > 14.0, "mort_28d"] = 0

    with pytest.raises(TimeVaryingCoxError, match="has no event in interval 3"):
        fit_piecewise_time_varying_cox(
            frame,
            duration_col="time",
            event_col="mort_28d",
            covariates=["lab_max", "age"],
            interval_cutpoints=[7.0, 14.0],
            exposure_col="lab_max",
        )
