"""The signed survival suite tests its prevalence definition.

The suite excludes exposure first recorded at or before time zero as
prevalent.  Exposure present on arrival but first recorded hours later is
still counted as incident, and no single later hour separates the two.  A
suite signed now therefore re-fits its reported estimate at the hours a
sealed rule gives for the exposure window: whole hours at a quarter and at a
half of it.  Each re-fit removes the exposed records first recorded by that
hour; none is moved to the comparator group.  Each re-fit repeats the
primary estimand, and its own PH test is a disclosed diagnostic.  A fit the
restricted risk set cannot support is reported without an estimate.  An
outcome-blind table gives the exposed group's first-record hours.

A suite signed before the analysis existed keeps its digest, its products
and its words.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.survival_scientific_claims import (
    derive_survival_claim_payloads,
)
from easyicu.research_agent.contracts.executed_method_design import EXECUTED_METHOD_DESIGN_KEY
from easyicu.research_agent.contracts.manuscript_result_structure import (
    PRIMARY_RESULT_HEADINGS_BY_FAMILY,
)
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    _exposure_onset_hours_table,
    _prevalence_sensitivity_fit,
)
from easyicu.research_agent.contracts.sealed_suite_robustness import (
    EXPOSURE_ONSET_HOURS_PRODUCT,
    PREVALENCE_SENSITIVITY_PRODUCT,
    PREVALENCE_SENSITIVITY_RULE,
    prevalence_sensitivity_cutoffs_hours,
    sealed_suite_prespecified_axes,
)
from easyicu.research_agent.methods import ph_schoenfeld
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.family_spec import survival_template
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from tests.support.survival_proposal import survival_context, survival_request
from tests.support.survival_sealed import (
    bound_survival_plan,
    run_signed_suite,
    sealed_request,
    sealed_survival,
    synthetic_crossing_hazard_rows,
    synthetic_survival_rows,
)

SUITE = "signed_landmark_survival_suite"
SURVIVAL_RESULTS = {"kind": "markdown_heading", "label": PRIMARY_RESULT_HEADINGS_BY_FAMILY["survival"]}
PREVALENCE_FIELDS = (
    "prevalence_sensitivity_rule",
    "prevalence_sensitivity_cutoffs_hours",
    "prevalence_sensitivity_product",
    "exposure_onset_hours_product",
)


def _without_undeclared_products(body: dict) -> dict:
    """The body's plan outputs without a sensitivity table it no longer declares."""

    undeclared = {
        product
        for field, product in (
            ("prevalence_sensitivity_product", PREVALENCE_SENSITIVITY_PRODUCT),
            ("exposure_onset_hours_product", EXPOSURE_ONSET_HOURS_PRODUCT),
        )
        if body.get(field) is None
    }
    return {**body, "plan_outputs": [item for item in body["plan_outputs"] if item not in undeclared]}


def _signed_before_the_analysis(authority):
    """The same suite as signed before the prevalence sensitivity analysis existed."""

    body = authority.model_dump(
        mode="json", exclude={"execution_contract_sha256", *PREVALENCE_FIELDS}
    )
    return build_current_case_scientific_runtime_authority(_without_undeclared_products(body))


def _selected_design(context, authorities, *, question=None):
    request = sealed_request(context, authorities)
    if question is not None:
        request = request.model_copy(update={"research_question": question})
    spec = SimpleNamespace(labels={}, literature_design_decisions=[])
    selection = survival_template._design_selection(
        request, spec, method_keys=["strobe_2007"], roster=[]
    )
    selected = next(item for item in selection.candidates if item.disposition == "selected")
    return request, selected


def _run(tmp_path, authority, rows):
    summary = json.loads(json.dumps(run_signed_suite(authority, rows, tmp_path / "out")))
    table = pd.read_csv(tmp_path / "out" / "landmark_prevalence_sensitivity.csv")
    return summary, table


def _early_exposed(rows: pd.DataFrame, hours: float) -> int:
    """Exposed rows first recorded after time zero and by ``hours``, read off the rows."""

    onset = rows["rrt_onset_time"]
    return int((rows["rrt"].eq(1) & onset.gt(0.0) & onset.le(hours)).sum())


@pytest.mark.parametrize(
    ("window_end", "hours"),
    [
        (24.0, (6.0, 12.0)),
        (48.0, (12.0, 24.0)),
        (30.0, (7.0, 15.0)),
        (6.0, (1.0, 3.0)),
        (4.0, (1.0, 2.0)),
        (3.0, (1.0,)),
        (2.0, (1.0,)),
        (1.0, ()),
    ],
)
def test_the_rule_takes_whole_hours_at_a_quarter_and_a_half_of_the_window(window_end, hours):
    assert prevalence_sensitivity_cutoffs_hours(window_end) == hours


def test_a_suite_signed_now_declares_the_grid_and_its_tables(tmp_path):
    context, authority = sealed_survival(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)

    assert authority.prevalence_sensitivity_rule == PREVALENCE_SENSITIVITY_RULE
    assert authority.prevalence_sensitivity_cutoffs_hours == (6.0, 12.0)
    outputs = list(authority.plan_outputs)
    audit = outputs.index("table:landmark_measurement_audit")
    assert outputs[audit - 2 : audit] == [PREVALENCE_SENSITIVITY_PRODUCT, EXPOSURE_ONSET_HOURS_PRODUCT]
    assert outputs.index("table:landmark_time_varying_cox_summary") < audit - 2
    assert "prevalence_sensitivity_cutoffs_hours" in authorities.planning_contract_context()
    # The saved authority verifies on load with the analysis in its body.
    assert load_current_case_scientific_runtime_authority(authority.model_dump(mode="json")) == authority

    request, selected = _selected_design(context, authorities)
    assert request.sealed_suite.prevalence_sensitivity_cutoffs_hours == [6.0, 12.0]
    assert selected.reviewable_plan[5].endswith(
        "replace a constant hazard ratio when proportional hazards is rejected. Prespecified "
        "sensitivity analyses of the prevalence definition also exclude exposure first recorded "
        "as present at or before hour 6 or, separately, hour 12 and repeat the reported "
        "estimate; a descriptive table shows the exposed group's first-record hours."
    )
    _request, chinese = _selected_design(
        context, authorities, question="肾脏替代治疗与 90 天死亡率相关吗？"
    )
    assert chinese.reviewable_plan[5].endswith(
        "替代恒定的风险比。预先设定的现患定义敏感性分析另外排除首次阳性记录在第 6 小时或"
        "（另一项分析中）第 12 小时及以前的暴露，并重复报告的估计；描述性表格给出暴露组首次"
        "记录的小时分布。"
    )


def test_a_proposed_suite_declares_the_same_grid_before_signing():
    context = survival_context()
    proposed = survival_request(context).proposed_suite

    assert proposed is not None
    assert proposed.prevalence_sensitivity_cutoffs_hours == list(
        prevalence_sensitivity_cutoffs_hours(proposed.landmark_hours)
    )
    outputs = list(proposed.plan_outputs)
    audit = outputs.index("table:landmark_measurement_audit")
    assert outputs[audit - 2 : audit] == [PREVALENCE_SENSITIVITY_PRODUCT, EXPOSURE_ONSET_HOURS_PRODUCT]


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"prevalence_sensitivity_cutoffs_hours": [6.0, 18.0]}, "must follow their rule"),
        ({"prevalence_sensitivity_cutoffs_hours": [12.0]}, "must follow their rule"),
        ({"prevalence_sensitivity_cutoffs_hours": []}, "must follow their rule"),
        ({"prevalence_sensitivity_product": None}, "are declared together"),
        ({"exposure_onset_hours_product": None}, "are declared together"),
        ({"prevalence_sensitivity_rule": None}, "are declared together"),
        ({"prevalence_sensitivity_cutoffs_hours": None}, "are declared together"),
    ],
    ids=["hours_off_the_rule", "one_hour_dropped", "no_hours", "no_table",
         "no_hours_table", "no_rule", "no_grid"],
)
def test_the_authority_refuses_a_grid_its_rule_does_not_give(tmp_path, change, refusal):
    _context, authority = sealed_survival(tmp_path)
    body = authority.model_dump(mode="json", exclude={"execution_contract_sha256"})

    with pytest.raises(ValueError, match=refusal):
        build_current_case_scientific_runtime_authority(
            _without_undeclared_products({**body, **change})
        )


def test_each_cutoff_removes_the_early_exposed_records_and_keeps_the_comparator(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    rows = synthetic_survival_rows()
    # A record first made at the cutoff hour itself is excluded by it.
    rows["rrt_onset_time"] = rows["rrt_onset_time"].replace({3.0: 6.0})
    assert rows["rrt_onset_time"].eq(6.0).any()

    summary, table = _run(tmp_path, authority, rows)

    assert summary["status"] == "ok"
    assert table["analysis"].tolist() == ["primary", "sensitivity", "sensitivity"]
    assert table["prevalent_exposure_cutoff_hours"].tolist() == [0.0, 6.0, 12.0]
    assert table["estimand"].eq("adjusted_hazard_ratio").all()
    assert table["reported"].all()
    primary = table.iloc[0]
    for hours, row in zip((6.0, 12.0), table.iloc[1:].itertuples(index=False)):
        excluded = _early_exposed(rows, hours)
        assert excluded > 0
        assert row.n_excluded_early_exposed == excluded
        assert row.n_landmark_population == primary["n_landmark_population"] - excluded
    # Removed, not reclassified: every fit has the primary's comparator group.
    comparator = table["n_complete_case"] - table["n_exposed"]
    assert comparator.nunique() == 1
    assert table["n_exposed"].is_monotonic_decreasing

    hours = pd.read_csv(
        tmp_path / "out" / "landmark_exposure_onset_hours.csv", dtype={"exposed_records": str},
    )
    assert hours["summary_kind"].eq("descriptive_outcome_blind").all()
    assert hours["first_record_after_hour"].tolist() == [float(hour) for hour in range(24)]
    assert hours["first_record_by_hour"].tolist() == [float(hour) for hour in range(1, 25)]
    # No running total: it would recover a suppressed hour by subtraction.
    assert list(hours.columns) == [
        "first_record_after_hour", "first_record_by_hour", "exposed_records", "summary_kind",
    ]
    # These rows hold no count under the floor, so every hour is shown and
    # the hours up to each cutoff sum to the records it excludes.
    by_hour = hours.set_index("first_record_by_hour")["exposed_records"].astype(int)
    for hours_cut, row in zip((6.0, 12.0), table.iloc[1:].itertuples(index=False)):
        assert by_hour.loc[:hours_cut].sum() == row.n_excluded_early_exposed

    files = summary["output_files"]
    assert files[PREVALENCE_SENSITIVITY_PRODUCT] == "landmark_prevalence_sensitivity.csv"
    assert files[EXPOSURE_ONSET_HOURS_PRODUCT] == "landmark_exposure_onset_hours.csv"
    design = summary[EXECUTED_METHOD_DESIGN_KEY]
    assert design["prevalence_sensitivity_cutoffs_hours"] == [6.0, 12.0]


def test_the_hours_table_suppresses_small_counts_so_no_subtraction_recovers_them(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    counts = {1: 30, 2: 5, 3: 25, 5: 40, 6: 22, 7: 3, 8: 7, 9: 50, 13: 12, 20: 60}
    onset = [hour - 0.5 for hour, n in counts.items() for _ in range(n)]
    onset[-1] = 20.0  # a record at the hour itself counts in that hour
    analysis = pd.DataFrame({
        authority.derived_exposure_column: [1] * len(onset) + [0] * 40,
        authority.exposure_onset_column: onset + [float("nan")] * 40,
    })

    table = _exposure_onset_hours_table(authority, analysis)

    shown = dict(zip(table["first_record_by_hour"].astype(int), table["exposed_records"]))
    assert list(shown) == list(range(1, 25))
    hidden = {hour for hour, value in shown.items() if value == "suppressed"}
    # Under the floor of 20: hours 2, 7, 8 and 13.  Hours 6 and 20 are the
    # smallest other nonzero counts of the stretches that would otherwise
    # hide one (up to 6 h, and from 12 h to the window's end).
    assert hidden == {2, 6, 7, 8, 13, 20}
    assert all(shown[hour] == str(counts.get(hour, 0)) for hour in shown if hour not in hidden)
    for low, high in ((0, 6), (6, 12), (12, 24)):
        assert len([hour for hour in hidden if low < hour <= high]) != 1


def test_each_estimable_refit_is_claimed_as_a_sensitivity_analysis(tmp_path):
    _context, authority = sealed_survival(tmp_path)

    summary, table = _run(tmp_path, authority, synthetic_survival_rows())

    envelope = summary["reportable_survival_results"]
    sensitivity = envelope["prevalence_definition_sensitivity"]
    # Last, so the per-step numeric cap reaches the primary estimates first.
    assert list(envelope)[-1] == "prevalence_definition_sensitivity"
    assert sensitivity["axis"] == "prevalence_definition"
    claims = {claim["claim_id"]: claim for claim in derive_survival_claim_payloads(summary)}
    projection = {
        entry["scientific_claim_id"]: entry
        for entry in envelope["manuscript_projection"]["claims"]
        if entry.get("scientific_claim_id")
    }
    primary = claims["adjusted_hazard_ratio"]
    for fit, row in zip(sensitivity["fits"], table.iloc[1:].itertuples(index=False)):
        hours = fit["prevalent_exposure_cutoff_hours"]
        assert set(fit) == {"prevalent_exposure_cutoff_hours", "n_analysis", "n_events",
                            "adjusted_hazard_ratio"}
        assert (fit["n_analysis"], fit["n_events"]) == (row.n_complete_case, row.n_events)
        assert fit["adjusted_hazard_ratio"]["hazard_ratio"] == pytest.approx(row.hazard_ratio)
        claim = claims[f"prevalence_cutoff_{hours:g}h_adjusted_hazard_ratio"]
        assert claim["analysis_role"] == "sensitivity"
        assert claim["point_estimate"] == fit["adjusted_hazard_ratio"]["hazard_ratio"]
        assert claim["interval_lower"] == fit["adjusted_hazard_ratio"]["ci_low"]
        assert claim["population"] == (
            f"{primary['population']}, excluding exposed records first recorded at or before "
            f"hour {hours:g}"
        )
        assert claim["estimand"] == primary["estimand"]
        # Reported beside the estimates it re-fits, never as the answer.
        entry = projection[claim["claim_id"]]
        assert entry["claim_id"] == f"results_{claim['claim_id']}"
        assert entry["targets"] == [SURVIVAL_RESULTS]
    assert primary["analysis_role"] == "primary"


def test_a_rejected_ph_test_keeps_the_interval_estimand_in_every_refit(tmp_path):
    _context, authority = sealed_survival(tmp_path)

    summary, table = _run(tmp_path, authority, synthetic_crossing_hazard_rows())

    envelope = summary["reportable_survival_results"]
    assert envelope["constant_hazard_ratio_authorized"] is False
    intervals = len(authority.time_varying_interval_cutpoints_days) + 1
    assert table["estimand"].eq("interval_adjusted_hazard_ratio").all()
    assert table.groupby("prevalent_exposure_cutoff_hours").size().tolist() == [intervals] * 3
    for fit in envelope["prevalence_definition_sensitivity"]["fits"]:
        assert "adjusted_hazard_ratio" not in fit
        assert len(fit["interval_hazard_ratios"]) == intervals
    ids = [claim["claim_id"] for claim in derive_survival_claim_payloads(summary)]
    sensitivity_ids = [claim_id for claim_id in ids if claim_id.startswith("prevalence_cutoff_")]
    assert sensitivity_ids == [
        f"prevalence_cutoff_{hours}h_interval_{position}_adjusted_hazard_ratio"
        for hours in (6, 12)
        for position in range(1, intervals + 1)
    ]
    assert "adjusted_hazard_ratio" not in ids


def test_a_refit_without_an_exposed_record_is_reported_without_an_estimate(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    rows = synthetic_survival_rows()
    # Every incident exposure is now first recorded by hour 12.
    rows["rrt_onset_time"] = rows["rrt_onset_time"].replace({18.0: 9.0})

    summary, table = _run(tmp_path, authority, rows)

    assert summary["status"] == "ok"
    late = table.loc[table["prevalent_exposure_cutoff_hours"].eq(12.0)].iloc[0]
    assert late["n_exposed"] == 0
    assert not late["reported"]
    assert pd.isna(late["hazard_ratio"])
    assert late["not_reported_reason"] == "a model term is constant on the restricted risk set"
    fits = summary["reportable_survival_results"]["prevalence_definition_sensitivity"]["fits"]
    assert set(fits[1]) == {"prevalent_exposure_cutoff_hours", "n_analysis", "n_events"}
    assert "adjusted_hazard_ratio" in fits[0]
    ids = {claim["claim_id"] for claim in derive_survival_claim_payloads(summary)}
    assert "prevalence_cutoff_6h_adjusted_hazard_ratio" in ids
    assert not any(claim_id.startswith("prevalence_cutoff_12h") for claim_id in ids)


def test_a_constant_estimate_its_refit_ph_test_rejects_is_not_reported(tmp_path, monkeypatch):
    _context, authority = sealed_survival(tmp_path)
    real = ph_schoenfeld.ph_test

    def rejects_on_a_restricted_risk_set(df, *args, **kwargs):
        result = real(df, *args, **kwargs)
        # The primary complete-case frame has 545 rows; each re-fit has fewer.
        return result.assign(p_value=0.001) if len(df) < 500 else result

    monkeypatch.setattr(ph_schoenfeld, "ph_test", rejects_on_a_restricted_risk_set)

    summary, table = _run(tmp_path, authority, synthetic_survival_rows())

    envelope = summary["reportable_survival_results"]
    assert envelope["constant_hazard_ratio_authorized"] is True
    refits = table.loc[table["analysis"].eq("sensitivity")]
    assert refits["estimand"].eq("adjusted_hazard_ratio").all()
    assert not refits["reported"].any()
    assert refits["not_reported_reason"].eq(
        "proportional hazards rejected on the restricted risk set"
    ).all()
    assert refits["ph_exposure_p_value"].eq(0.001).all()
    for fit in envelope["prevalence_definition_sensitivity"]["fits"]:
        assert set(fit) == {"prevalent_exposure_cutoff_hours", "n_analysis", "n_events"}
    ids = {claim["claim_id"] for claim in derive_survival_claim_payloads(summary)}
    assert "adjusted_hazard_ratio" in ids
    assert not any(claim_id.startswith("prevalence_cutoff_") for claim_id in ids)


def test_a_refit_whose_ph_test_lacks_a_term_is_reported_without_an_estimate(tmp_path, monkeypatch):
    _context, authority = sealed_survival(tmp_path)
    real = ph_schoenfeld.ph_test

    def drops_the_global_row_on_a_restricted_risk_set(df, *args, **kwargs):
        result = real(df, *args, **kwargs)
        if len(df) >= 500:
            return result
        return result.loc[result["covariate"].astype(str).ne("global")]

    monkeypatch.setattr(ph_schoenfeld, "ph_test", drops_the_global_row_on_a_restricted_risk_set)

    summary, table = _run(tmp_path, authority, synthetic_survival_rows())

    assert summary["status"] == "ok"
    refits = table.loc[table["analysis"].eq("sensitivity")]
    assert not refits["reported"].any()
    assert refits["not_reported_reason"].eq("the PH test lacks a global or exposure result").all()


def test_a_refit_the_primary_gates_would_refuse_has_no_estimate(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    rng = np.random.default_rng(20261004)
    exposure = authority.derived_exposure_column

    def frame(n: int, events: int) -> pd.DataFrame:
        return pd.DataFrame({
            exposure: np.resize([0.0, 1.0], n),
            "age": rng.normal(64.0, 13.0, size=n),
            authority.derived_time_column: rng.uniform(1.0, 89.0, size=n),
            authority.derived_event_column: np.r_[np.ones(events), np.zeros(n - events)],
        })

    for n, events in ((99, 40), (300, 9)):
        fit = _prevalence_sensitivity_fit(
            frame(n, events), sealed=authority, covariates=[exposure, "age"], hours=6.0,
            constant_estimand=True,
        )
        assert fit["estimates"] == []
        assert fit["not_reported"] == "below the primary model's minimum rows or events"
    # The primary withheld its estimate: no re-fit reports one in its place.
    withheld = authority.model_copy(update={"time_varying_effect_method": None})
    fit = _prevalence_sensitivity_fit(
        frame(300, 60), sealed=withheld, covariates=[exposure, "age"], hours=6.0,
        constant_estimand=False,
    )
    assert fit["estimates"] == []
    assert fit["not_reported"] == "the primary estimate is withheld"


def test_the_envelope_refuses_a_refit_that_changes_the_estimand(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    constant, _ = _run(tmp_path / "constant", authority, synthetic_survival_rows())
    crossing, _ = _run(tmp_path / "crossing", authority, synthetic_crossing_hazard_rows())

    def as_intervals(fits):
        fits[0]["interval_hazard_ratios"] = [fits[0].pop("adjusted_hazard_ratio")] * 4

    def as_constant(fits):
        fits[0]["adjusted_hazard_ratio"] = fits[0].pop("interval_hazard_ratios")[0]

    def fewer_intervals(fits):
        fits[0]["interval_hazard_ratios"] = fits[0]["interval_hazard_ratios"][:2]

    def both(fits):
        fits[0]["interval_hazard_ratios"] = [fits[0]["adjusted_hazard_ratio"]] * 4

    def reversed_hours(fits):
        fits.reverse()

    for summary, change in (
        (constant, as_intervals), (crossing, as_constant), (crossing, fewer_intervals),
        (constant, both), (constant, reversed_hours),
    ):
        derive_survival_claim_payloads(summary)
        changed = copy.deepcopy(summary)
        change(changed["reportable_survival_results"]["prevalence_definition_sensitivity"]["fits"])
        with pytest.raises(ValueError):
            derive_survival_claim_payloads(changed)


def test_the_review_credits_the_prevalence_axis_only_to_a_suite_that_runs_it(tmp_path):
    sealed_ref = f"scientific_runtime_contract:{'a' * 64}"
    product = [PREVALENCE_SENSITIVITY_PRODUCT]

    assert sealed_suite_prespecified_axes(
        method=SUITE, rule_refs=[sealed_ref], expected_outputs=product
    ) == ("timing", "model_specification", "prevalence_definition")
    assert sealed_suite_prespecified_axes(method=SUITE, rule_refs=[sealed_ref]) == (
        "timing", "model_specification",
    )
    # An unsigned draft that spells the product earns nothing.
    assert sealed_suite_prespecified_axes(method=SUITE, rule_refs=[], expected_outputs=product) == ()
    assert sealed_suite_prespecified_axes(
        method="adjusted_association_models", rule_refs=[sealed_ref], expected_outputs=product
    ) == ()
    # Review vocabulary only: a user cannot declare it on a StudyContext spec.
    declarable = set(PrespecifiedSensitivitySpec.model_fields["axis"].annotation.__args__)
    assert "prevalence_definition" not in declarable

    context, authority = sealed_survival(tmp_path)
    for current, expected in (
        (authority, {"timing", "model_specification", "prevalence_definition"}),
        (_signed_before_the_analysis(authority), {"timing", "model_specification"}),
    ):
        authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=current)
        plan = bound_survival_plan(context, authorities)
        facts = build_plan_scientific_review(context=context, plan=plan).facts
        assert set(facts["sensitivity"]["sealed_suite_axes"]) == expected


def test_a_suite_signed_before_the_analysis_keeps_its_digest_products_and_words(tmp_path):
    context, signed = sealed_survival(tmp_path)
    legacy = _signed_before_the_analysis(signed)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=legacy)

    dumped = legacy.model_dump(mode="json")
    assert not set(PREVALENCE_FIELDS) & set(dumped)
    assert load_current_case_scientific_runtime_authority(dumped) == legacy
    assert legacy.execution_contract_sha256 != signed.execution_contract_sha256
    assert PREVALENCE_SENSITIVITY_PRODUCT not in legacy.plan_outputs
    assert EXPOSURE_ONSET_HOURS_PRODUCT not in legacy.plan_outputs
    assert "prevalence_sensitivity" not in authorities.planning_contract_context()
    request, selected = _selected_design(context, authorities)
    assert request.sealed_suite.prevalence_sensitivity_cutoffs_hours is None
    assert "prevalence_sensitivity_cutoffs_hours" not in request.model_dump(mode="json")["sealed_suite"]
    assert selected.reviewable_plan[5].endswith("when proportional hazards is rejected.")

    summary = json.loads(json.dumps(run_signed_suite(legacy, synthetic_survival_rows(), tmp_path / "out")))
    assert "prevalence_definition_sensitivity" not in summary["reportable_survival_results"]
    assert "prevalence_sensitivity_cutoffs_hours" not in summary[EXECUTED_METHOD_DESIGN_KEY]
    assert not (tmp_path / "out" / "landmark_prevalence_sensitivity.csv").exists()
    assert not (tmp_path / "out" / "landmark_exposure_onset_hours.csv").exists()
    ids = [claim["claim_id"] for claim in derive_survival_claim_payloads(summary)]
    assert not any(claim_id.startswith("prevalence_cutoff_") for claim_id in ids)
