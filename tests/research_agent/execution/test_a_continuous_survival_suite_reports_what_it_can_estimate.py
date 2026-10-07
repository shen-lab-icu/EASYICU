"""A continuous survival suite reports what its data let it estimate.

The suite described its landmark risk set by exposure tertile and always
fitted its interval model, and either failing ended the whole run without a
result: a recorded value heaped at one level (a score at its ceiling, a
saturation or an oxygen fraction at its maximum) has no three tertiles, and a
follow-up interval without a death, or a sparse covariate level, leaves the
interval model without an estimate.  Now a risk set whose tertiles would leave
one empty is described whole, for the reason its cutpoints give, and an
interval model the data leave without an estimate is reported as not
estimable while the PH test lets the constant estimate stand.  When the test
rejects it, the interval estimates are the result, and without them the suite
still fails.  The table note, the figure legend and the Methods state the
same reason.

Synthetic, seeded rows only: a laboratory value or a bounded score, 28-day
mortality, every stay alive at the 24-hour landmark.
"""

from __future__ import annotations

import hashlib
import json
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.continuous_survival_scientific_claims import (
    ContinuousSurvivalReporting,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority import manuscript_method_facts
from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.contracts import executed_method_design
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from easyicu.research_agent.execution.runners import (
    landmark_continuous_survival_executor,
)
from easyicu.research_agent.execution.runners.landmark_continuous_survival_executor import (
    run_landmark_continuous_survival_suite,
)
from easyicu.research_agent.execution.runners.landmark_continuous_survival_figure import (
    run_landmark_continuous_survival_figure,
)
from easyicu.research_agent.figures.publication import (
    FigureContract,
    audit_publication_exports,
)
from easyicu.research_agent.methods import time_varying_cox
from easyicu.research_agent.methods.time_varying_cox import (
    TimeVaryingCoxError,
    fit_piecewise_time_varying_cox,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_tables import build_manuscript_tables
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRecord
from tests.support.continuous_survival import continuous_authority_body

pytest.importorskip("lifelines")

STEP = "01_primary"
EVIDENCE = "statistic_step_summary_continuous_survival_suite"
AUTHORITY = build_current_case_scientific_runtime_authority(continuous_authority_body())


def _rows(
    *, seed=1, exposure=None, quiet_from_day=None, crossing=False, n=1500
) -> pd.DataFrame:
    """Stays alive at the 24-hour landmark; deaths by day 28 from ICU admission.

    ``quiet_from_day`` drops every death that late after the landmark, so the
    last follow-up interval (days 14 to 27) has none.  ``crossing`` reverses
    the exposure's effect five days after the landmark.
    """

    rng = np.random.default_rng(seed)
    age = rng.normal(65.0, 12.0, n)
    sex = rng.choice(["F", "M"], n)
    value = np.exp(rng.normal(0.7, 0.5, n)) if exposure is None else exposure(rng, n)
    centred = value - float(np.mean(value))
    if crossing:
        early = rng.exponential(1.0 / (0.03 * np.exp(0.9 * centred)))
        late = 5.0 + rng.exponential(1.0 / (0.03 * np.exp(-0.9 * centred)))
        after = np.where(early < 5.0, early, late)
    else:
        after = rng.exponential(
            1.0 / (0.03 * np.exp(0.05 * centred + 0.02 * (age - 65.0)))
        )
    death = after <= 27.0
    if quiet_from_day is not None:
        death &= after < quiet_from_day
    return pd.DataFrame(
        {
            "lab_max": value,
            "mort_28d": death.astype(int),
            "followup_days_28d": np.where(death, 1.0 + after, 28.0),
            "age": age,
            "sex": sex,
        }
    )


def _score(share_at_ceiling: float):
    """A bounded score from 3 to 15 with this share of the stays at 15."""

    def draw(rng, n):
        return np.where(
            rng.random(n) < share_at_ceiling, 15.0, rng.integers(3, 15, n).astype(float)
        )

    return draw


def _two_values(rng, n):
    return rng.choice([1.0, 2.0], size=n)


def _run(tmp_path, frame, name="suite"):
    summary = run_landmark_continuous_survival_suite(
        frame=frame,
        authority=AUTHORITY.model_dump(mode="json"),
        runtime_projection_sha256="b" * 64,
        out_dir=tmp_path / name,
        input_product="table:analysis_cohort",
        input_evidence_id="cohort_evidence",
        input_sha256="c" * 64,
    )
    return json.loads(json.dumps(summary, allow_nan=False))


def _outputs(tmp_path, summary, name="suite"):
    return {
        product: tmp_path / name / file
        for product, file in summary["output_files"].items()
    }


def _figure(tmp_path, summary, *, name="suite", km_table=None):
    paths = _outputs(tmp_path, summary, name)
    result = run_landmark_continuous_survival_figure(
        km_table=pd.read_csv(paths[AUTHORITY.km_product])
        if km_table is None
        else km_table,
        spline_table=pd.read_csv(paths[AUTHORITY.spline_product]),
        time_varying_table=pd.read_csv(paths[AUTHORITY.time_varying_cox_product]),
        risk_flow=pd.read_csv(paths[AUTHORITY.risk_set_product]),
        ph_table=pd.read_csv(paths[AUTHORITY.ph_product]),
        source_paths={
            product: paths[product] for product in AUTHORITY.figure_input_products
        },
        authority=AUTHORITY.model_dump(mode="json"),
        out_dir=tmp_path / f"{name}_figure",
    )
    contract = FigureContract.model_validate_json(
        (tmp_path / f"{name}_figure" / result["figure_assets"]["contract"]).read_text(
            encoding="utf-8"
        )
    )
    return contract


def _methods_fact(tmp_path, summary):
    """The executed design as its Methods fact, bound to the registered summary."""

    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic",
        description="Signed continuous survival suite summary",
        source_path=source,
        evidence_id=EVIDENCE,
        produced_by_step=STEP,
        producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id=EVIDENCE, summary=summary
    )
    ledger = [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]
    (fact,) = [
        fact
        for fact in store.manuscript_method_facts(ledger)
        if fact.source_field.endswith(EXECUTED_METHOD_DESIGN_KEY)
    ]
    scaffold = f"## Methods\n\n### Variables\n\n{fact.scaffold}\n"
    safe, removed = store.enforce_evidence_bound_scaffold(
        scaffold, per_step_records=ledger
    )
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced
    return fact.text


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


def _rendered_table_one(tmp_path, summary, name="suite"):
    """Table 1 as the manuscript prints it from the run's registered evidence."""

    run_dir = tmp_path / f"{name}_tables"
    records = [
        _register(
            run_dir,
            f"statistic_step_summary_{STEP}",
            "statistic",
            json.dumps(summary).encode(),
            "step_summary.json",
        )
    ]
    for product, path in _outputs(tmp_path, summary, name).items():
        if product.startswith("table:"):
            records.append(
                _register(
                    run_dir,
                    f"table_{STEP}_{product.split(':', 1)[1]}",
                    "table",
                    path.read_bytes(),
                    path.name,
                )
            )
    plan = AnalysisPlan(
        research_question="Is the first-day value associated with 28-day mortality?",
        steps=[
            AnalysisStep(
                step_id=STEP,
                intent="Execute the signed continuous survival suite",
                inputs=["lab_max", "mort_28d"],
                expected_outputs=[AUTHORITY.table_one_product],
                method="signed_landmark_continuous_survival_suite",
            )
        ],
        display_labels={"age": "Age", "sex": "Sex"},
    )
    table_one, _flow = build_manuscript_tables(
        plan=plan, evidence_records=records, run_dir=run_dir
    )
    return table_one


@pytest.mark.parametrize(
    ("exposure", "reason"),
    [
        (_score(0.45), "upper_tertile_cutpoint_at_maximum"),
        (_score(0.80), "lower_tertile_cutpoint_at_maximum"),
        (_two_values, "upper_tertile_cutpoint_at_maximum"),
    ],
    ids=["ceiling_above_a_third", "ceiling_above_two_thirds", "two_values"],
)
def test_a_risk_set_that_leaves_a_tertile_empty_is_described_whole(
    tmp_path, exposure, reason
) -> None:
    summary = _run(tmp_path, _rows(exposure=exposure))

    design = validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY])
    assert (design.descriptive_grouping, design.descriptive_grouping_reason) == (
        "whole_risk_set",
        reason,
    )
    receipt = summary["scientific_runtime_receipt"]
    assert (
        receipt["descriptive_grouping"],
        receipt["descriptive_grouping_reason"],
    ) == (
        "whole_risk_set",
        reason,
    )
    (table_one, _flow) = validate_manuscript_table_declarations(
        summary[MANUSCRIPT_TABLES_KEY]
    )
    (group,) = table_one.body.groups
    assert (group.prefix, group.label, group.n) == (
        "all",
        "Landmark analysis cohort",
        summary["n_landmark_population"],
    )
    assert group.events == summary["n_events_landmark_population"]
    assert "tertile" not in table_one.caption
    words = executed_method_design.WHOLE_RISK_SET_REASON_WORDS[reason]
    assert table_one.notes[0] == (
        f"The table describes the whole cohort: exposure tertiles were not formed, because {words}."
    )
    assert not any("standardized mean difference" in note for note in table_one.notes)
    rows = pd.read_csv(_outputs(tmp_path, summary)[AUTHORITY.table_one_product])
    assert rows["standardized_mean_difference"].isna().all()
    km = pd.read_csv(_outputs(tmp_path, summary)[AUTHORITY.km_product])
    assert set(km["exposure_group"]) == {0}
    assert set(km["descriptive_grouping"]) == {"whole_risk_set"}
    assert set(km["descriptive_grouping_reason"]) == {reason}
    assert int(km["group_n"].iloc[0]) == summary["n_landmark_population"]
    # The estimate the description never entered is reported as before.
    claims = [claim.claim_id for claim in derive_scientific_claim_drafts(summary)]
    assert claims[0] == "adjusted_hazard_ratio_per_unit"
    assert (
        f"Kaplan-Meier curves described the whole risk set, without exposure tertiles, because {words}"
        in (_methods_fact(tmp_path, summary))
    )


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([1.0] * 1000 + [9.0] * 500, "no_value_between_tertile_cutpoints"),
        ([1.0] * 500 + [2.0] * 400 + [9.0] * 600, "upper_tertile_cutpoint_at_maximum"),
        ([1.0] * 400 + [9.0] * 1100, "lower_tertile_cutpoint_at_maximum"),
        ([1.0] * 500 + [2.0] * 500 + [3.0] * 500, None),
    ],
    ids=["no_value_between", "upper_at_maximum", "lower_at_maximum", "three_groups"],
)
def test_the_empty_tertile_names_its_reason(values, reason) -> None:
    groups, low, high, found = landmark_continuous_survival_executor._tertile_groups(
        pd.Series(values)
    )

    assert found == reason
    assert (groups is None) is (reason is not None)
    if reason == "no_value_between_tertile_cutpoints":
        # The cutpoints differ, yet no recorded value lies between them.
        assert (
            low < high
            and not ((pd.Series(values) > low) & (pd.Series(values) <= high)).any()
        )
    if groups is not None:
        assert sorted(groups.unique()) == [1, 2, 3]


def test_a_whole_risk_set_table_has_no_comparison_column(tmp_path) -> None:
    whole = _run(tmp_path, _rows(exposure=_score(0.45)), name="whole")
    tertiles = _run(tmp_path, _rows(), name="tertiles")

    one = _rendered_table_one(tmp_path, whole, name="whole")
    three = _rendered_table_one(tmp_path, tertiles, name="tertiles")

    n = whole["n_landmark_population"]
    assert one.columns == ("Characteristic", f"Landmark analysis cohort (n = {n})")
    assert all(len(row) == 2 for row in one.rows)
    assert three.columns[-1] == "SMD" and len(three.columns) == 5


def test_the_figure_draws_the_whole_risk_set_its_table_records(tmp_path) -> None:
    summary = _run(tmp_path, _rows(exposure=_score(0.45)))

    contract = _figure(tmp_path, summary)

    assert audit_publication_exports(tmp_path / "suite_figure") == []
    (panel_a,) = [panel for panel in contract.panels if panel.panel_id == "a"]
    assert panel_a.title == "Unadjusted Kaplan-Meier survival"
    assert "whole risk set" in panel_a.claim
    words = executed_method_design.WHOLE_RISK_SET_REASON_WORDS[
        "upper_tertile_cutpoint_at_maximum"
    ]
    assert contract.reader_caption.startswith(
        "(a) Unadjusted Kaplan-Meier survival after the landmark for the whole risk set, with "
        f"the number at risk below; exposure tertiles were not formed, because {words}."
    )
    assert "of the whole risk set" in contract.core_claim


@pytest.mark.parametrize(
    "mutation", ["grouping_without_reason", "tertiles_with_reason", "groups_contradict"]
)
def test_the_figure_refuses_a_km_table_that_misstates_its_groups(
    tmp_path, mutation
) -> None:
    whole = _run(tmp_path, _rows(exposure=_score(0.45)))
    km = pd.read_csv(_outputs(tmp_path, whole)[AUTHORITY.km_product])
    if mutation == "grouping_without_reason":
        km["descriptive_grouping_reason"] = np.nan
    elif mutation == "tertiles_with_reason":
        km["descriptive_grouping"] = "value_tertiles"
    else:
        km.loc[km.index[: len(km) // 2], "exposure_group"] = 1

    with pytest.raises(ValueError, match="KM table"):
        _figure(tmp_path, whole, km_table=km)


def test_an_interval_model_without_an_event_is_not_estimable_while_ph_holds(
    tmp_path,
) -> None:
    summary = _run(tmp_path, _rows(quiet_from_day=13.0))

    envelope = summary["reportable_survival_results"]
    assert (
        envelope["proportional_hazards_test"]["disposition"]
        == "assumption_not_rejected"
    )
    assert envelope["time_varying_adjusted_association"] == {
        "status": "not_estimable",
        "reason": "interval_without_event",
        "method": "piecewise_time_varying_cox",
        "adjustment_columns": ["age", "sex"],
    }
    assert [claim.claim_id for claim in derive_scientific_claim_drafts(summary)] == [
        "adjusted_hazard_ratio_per_unit",
        "proportional_hazards_rule",
    ]
    abstract = [
        claim["scientific_claim_id"]
        for claim in envelope["manuscript_projection"]["claims"]
        if "scientific_claim_id" in claim
    ]
    assert abstract == ["adjusted_hazard_ratio_per_unit"]
    assert (
        summary["scientific_runtime_receipt"]["time_varying_status"]
        == "interval_without_event"
    )
    table = pd.read_csv(_outputs(tmp_path, summary)[AUTHORITY.time_varying_cox_product])
    assert table[["term", "model_status", "not_estimable_reason"]].to_dict(
        "records"
    ) == [
        {
            "term": "lab_max",
            "model_status": "not_estimable",
            "not_estimable_reason": "interval_without_event",
        }
    ]
    # The Writer still reads the authorized constant estimate.
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
        row["reportable_survival_results"]["adjusted_hazard_ratio_per_unit"]
        == (envelope["adjusted_hazard_ratio_per_unit"])
    )
    assert (
        "a piecewise Cox model split at days 7 and 14 after the landmark was prespecified for "
        "interval-specific associations and was not estimable, because a follow-up interval had no event"
    ) in _methods_fact(tmp_path, summary)
    contract = _figure(tmp_path, summary)
    assert audit_publication_exports(tmp_path / "suite_figure") == []
    assert next(panel for panel in contract.panels if panel.panel_id == "b").metadata[
        "chart_type"
    ] == ("hazard_ratio_curve")


def test_a_rejected_ph_test_without_interval_estimates_has_no_result(tmp_path) -> None:
    with pytest.raises(
        ValueError, match="continuous_survival_interval_result_not_estimable"
    ) as caught:
        _run(tmp_path, _rows(crossing=True, quiet_from_day=13.0))

    assert isinstance(caught.value.__cause__, TimeVaryingCoxError)
    assert caught.value.__cause__.reason == "interval_without_event"
    # With its interval estimates the same rejected test reports them (control).
    summary = _run(tmp_path, _rows(crossing=True), name="estimated")
    envelope = summary["reportable_survival_results"]
    assert envelope["proportional_hazards_test"]["disposition"] == "assumption_rejected"
    assert envelope["time_varying_adjusted_association"]["status"] == "estimated"


def test_an_input_the_interval_model_refuses_is_not_read_as_not_estimable(
    tmp_path, monkeypatch
) -> None:
    def refuse(*_args, **_kwargs):
        raise TimeVaryingCoxError("time-varying Cox covariates must be unique")

    monkeypatch.setattr(time_varying_cox, "fit_piecewise_time_varying_cox", refuse)

    with pytest.raises(TimeVaryingCoxError, match="must be unique"):
        _run(tmp_path, _rows())


def _method_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20261007)
    n = 1_200
    exposure = rng.binomial(1, 0.4, n)
    level = rng.normal(0.0, 1.0, n)
    event_time = rng.exponential(1.0 / np.exp(-3.0 + 0.55 * exposure + 0.2 * level))
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, 27.0),
            "event": (event_time <= 27.0).astype(int),
            "exposure": exposure,
            "level": level,
        }
    )


def _refusal(frame, **overrides):
    arguments = dict(
        duration_col="time",
        event_col="event",
        covariates=["exposure", "level"],
        interval_cutpoints=[7.0, 14.0],
        exposure_col="exposure",
    )
    arguments.update(overrides)
    with pytest.raises(TimeVaryingCoxError) as caught:
        fit_piecewise_time_varying_cox(frame, **arguments)
    return caught.value.reason


def test_a_data_condition_names_its_reason_and_a_bad_input_does_not() -> None:
    frame = _method_frame()
    quiet = frame.assign(event=np.where(frame["time"] > 14.0, 0, frame["event"]))
    one_group_quiet = frame.assign(
        event=np.where(
            (frame["exposure"] == 1) & (frame["time"] > 14.0), 0, frame["event"]
        )
    )
    rng = np.random.default_rng(20261008)
    separated = frame.assign(
        marker=np.where(frame["event"] == 1, 0, rng.binomial(1, 0.5, len(frame)))
    )
    continuous = quiet.assign(
        exposure=frame["level"], level=rng.normal(0.0, 1.0, len(frame))
    )

    assert _refusal(quiet) == "interval_without_event"
    assert _refusal(one_group_quiet) == "interval_without_event"
    assert _refusal(continuous) == "interval_without_event"
    assert (
        _refusal(separated, covariates=["exposure", "level", "marker"])
        == "did_not_converge"
    )
    assert (
        _refusal(frame, interval_cutpoints=[7.0, 27.0])
        == "follow_up_ends_by_final_cutpoint"
    )
    assert _refusal(frame, covariates=["exposure", "exposure"]) == "invalid_input"
    assert _refusal(frame, covariates=["level"]) == "invalid_input"
    assert _refusal(frame, covariates=["exposure", "absent"]) == "invalid_input"
    assert _refusal(frame, interval_cutpoints=[14.0, 7.0]) == "invalid_input"
    assert _refusal(frame.assign(level=np.nan)) == "invalid_input"
    with pytest.raises(ValueError, match="unknown time-varying Cox refusal reason"):
        TimeVaryingCoxError("refused", reason="tied_tertiles")


def _with_interval_model(envelope: dict, **time_varying) -> dict:
    return {
        **envelope,
        "time_varying_adjusted_association": {
            "method": "piecewise_time_varying_cox",
            "adjustment_columns": ["age", "sex"],
            **time_varying,
        },
    }


def test_the_claims_envelope_refuses_an_interval_model_that_contradicts_itself(
    tmp_path,
) -> None:
    holds = _run(tmp_path, _rows(), name="holds")["reportable_survival_results"]
    rejected = _run(tmp_path, _rows(crossing=True), name="rejected")[
        "reportable_survival_results"
    ]
    assert (
        holds["proportional_hazards_test"]["disposition"] == "assumption_not_rejected"
    )
    assert rejected["proportional_hazards_test"]["disposition"] == "assumption_rejected"

    ContinuousSurvivalReporting.model_validate(
        _with_interval_model(holds, status="not_estimable", reason="did_not_converge")
    )
    with pytest.raises(ValueError, match="must exist"):
        ContinuousSurvivalReporting.model_validate(
            _with_interval_model(
                rejected, status="not_estimable", reason="did_not_converge"
            )
        )
    for broken in (
        _with_interval_model(holds, status="not_estimable", reason="tied_tertiles"),
        _with_interval_model(
            holds, status="not_estimable", reason="did_not_converge", intervals=[]
        ),
        _with_interval_model(holds, status="not_estimable"),
        _with_interval_model(
            holds, intervals=holds["time_varying_adjusted_association"]["intervals"]
        ),
    ):
        with pytest.raises(ValueError):
            ContinuousSurvivalReporting.model_validate(broken)


def _design(**overrides) -> dict:
    summary_design = {
        "schema_version": "easyicu.executed_method_design/1",
        "design_kind": "landmark_continuous_survival",
        "time_origin": "ICU admission",
        "landmark_hours": 24.0,
        "endpoint_horizon_days": 28.0,
        "exposure_window_start_hours": 0.0,
        "exposure_window_end_hours": 24.0,
        "exposure_window_summary": "max",
        "exposure_increment": 1.0,
        "exposure_unit": "mmol/L",
        "n_adjustment_covariates": 2,
        "effect_model": "cox_proportional_hazards_efron_ties",
        "interval_method": "wald_95_ci",
        "proportional_hazards_test": "schoenfeld_residuals",
        "proportional_hazards_alpha": 0.05,
        "time_varying_cutpoints_days": [7.0, 14.0],
        "spline_knot_percentiles": [10.0, 50.0, 90.0],
        "descriptive_grouping": "value_tertiles",
    }
    summary_design.update(overrides)
    return summary_design


def test_the_design_states_a_reason_exactly_when_it_needs_one(tmp_path) -> None:
    for broken in (
        _design(descriptive_grouping="whole_risk_set"),
        _design(descriptive_grouping_reason="upper_tertile_cutpoint_at_maximum"),
        _design(
            descriptive_grouping="whole_risk_set", descriptive_grouping_reason="tied"
        ),
        _design(interval_model_not_estimable_reason="tied_tertiles"),
    ):
        with pytest.raises(ValueError):
            validate_executed_method_design(broken)
    stated = validate_executed_method_design(
        _design(
            descriptive_grouping="whole_risk_set",
            descriptive_grouping_reason="no_value_between_tertile_cutpoints",
            interval_model_not_estimable_reason="non_finite_estimate",
        )
    )
    assert stated.model_dump(mode="json") == _design(
        descriptive_grouping="whole_risk_set",
        descriptive_grouping_reason="no_value_between_tertile_cutpoints",
        interval_model_not_estimable_reason="non_finite_estimate",
    )
    # A run whose tertiles and intervals were estimated writes the design it
    # wrote before: no field for a reason it does not have (control).
    summary = _run(tmp_path, _rows())
    assert summary[EXECUTED_METHOD_DESIGN_KEY] == _design()
    envelope = summary["reportable_survival_results"]
    assert envelope["time_varying_adjusted_association"]["status"] == "estimated"
    assert len(envelope["time_varying_adjusted_association"]["intervals"]) == 3
    assert [
        group.prefix
        for group in validate_manuscript_table_declarations(
            summary[MANUSCRIPT_TABLES_KEY]
        )[0].body.groups
    ] == ["t1", "t2", "t3"]


def test_every_reason_has_its_reader_words() -> None:
    assert set(executed_method_design.WHOLE_RISK_SET_REASON_WORDS) == set(
        get_args(executed_method_design.WholeRiskSetReason)
    )
    assert (
        set(manuscript_method_facts._INTERVAL_NOT_ESTIMABLE_WORDS)
        == time_varying_cox.TIME_VARYING_NOT_ESTIMABLE_REASONS
    )
