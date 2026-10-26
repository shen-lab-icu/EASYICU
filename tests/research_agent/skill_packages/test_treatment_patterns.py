"""Public synthetic demo: one prepared, governed row per ICU stay."""

import json

import pytest


def test_public_demo_treatment_results():
    from easyicu.research_agent.skill_packages import (
        public_demo, summarize_treatment_patterns,
    )

    frame, arguments = public_demo()
    result = summarize_treatment_patterns(frame, **arguments)
    exposed = [r for r in result["exposure"] if r["category"] == "1"]
    assert [r["count"] for r in exposed] == [4, 2, 2]
    assert [r["denominator_n"] for r in exposed] == [8, 4, 4]
    assert [r["percentage"] for r in exposed] == [50, 50, 50]
    timing = {r["group"]: r for r in result["timing"]}
    assert timing["A"]["median"] == 3
    assert (timing["A"]["q25"], timing["A"]["q75"]) == (2.5, 3.5)
    assert timing["A"]["exposed_n"] == 2
    assert result["cumulative_incidence"]["A"]["n_competing"] == 2
    assert result["cumulative_incidence"]["A"]["cif"][-1] == .5
    assert result["claim_ceiling"] == "analysis_only"
    assert all("p_value" not in row for row in result["exposure"] + result["timing"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value,code", [
    ("treatment_class", "unregistered_drug_class", "treatment.class_unknown"),
    ("captured_agents", ["norepinephrine"], "treatment.class_capture_incomplete"),
    ("capture_confirmed", False, "treatment.capture_review_required"),
    ("source_receipt", "", "treatment.capture_review_required"),
])
def test_unreviewed_or_incomplete_capture_stops(field, value, code):
    from easyicu.research_agent.skill_packages import (
        TreatmentPatternError, public_demo, summarize_treatment_patterns,
    )

    frame, arguments = public_demo()
    arguments[field] = value
    with pytest.raises(TreatmentPatternError, match=code):
        summarize_treatment_patterns(frame, **arguments)


@pytest.mark.parametrize("column,index,value,code", [
    ("stay_id", 1, 0, "treatment.one_row_per_stay_required"),
    ("stay_id", 1, None, "treatment.one_row_per_stay_required"),
    ("exposed", 0, None, "treatment.ascertainment_required"),
    ("death", 0, 2, "treatment.ascertainment_required"),
    ("death", 0, 1, "treatment.ascertainment_required"),
    ("discharged_alive", 0, 0, "treatment.ascertainment_required"),
    ("discharged_alive", 0, None, "treatment.ascertainment_required"),
    ("group", 0, None, "treatment.ascertainment_required"),
    ("group", 0, "C", "treatment.summary_contract_invalid"),
    ("first_exposure_hours", 0, -1, "treatment.time_invalid"),
    ("first_exposure_hours", 0, 13, "treatment.time_invalid"),
    ("first_exposure_hours", 0, None, "treatment.time_invalid"),
    ("first_exposure_hours", 0, "invalid", "treatment.time_invalid"),
    ("first_exposure_hours", 2, 2, "treatment.time_invalid"),
    ("followup_hours", 0, float("inf"), "treatment.time_invalid"),
    ("followup_hours", 2, -1, "treatment.time_invalid"),
])
def test_invalid_stay_records_stop(column, index, value, code):
    from easyicu.research_agent.skill_packages import (
        TreatmentPatternError, public_demo, summarize_treatment_patterns,
    )

    frame, arguments = public_demo()
    frame[column] = frame[column].astype(object)
    frame.loc[index, column] = value
    with pytest.raises(TreatmentPatternError, match=code):
        summarize_treatment_patterns(frame, **arguments)


def test_tied_death_and_use_is_not_silently_ordered():
    from easyicu.research_agent.skill_packages import (
        TreatmentPatternError, public_demo, summarize_treatment_patterns,
    )

    frame, arguments = public_demo()
    frame.loc[0, ["followup_hours", "death", "discharged_alive"]] = [2, 1, 0]
    with pytest.raises(TreatmentPatternError, match="treatment.event_order_unresolved"):
        summarize_treatment_patterns(frame, **arguments)


def test_unexposed_group_and_cohort_have_no_fabricated_time_estimate():
    from easyicu.research_agent.skill_packages import public_demo, summarize_treatment_patterns

    frame, arguments = public_demo()
    frame.loc[frame.group.eq("A"), ["exposed", "first_exposure_hours"]] = [0, None]
    result = summarize_treatment_patterns(frame, **arguments)
    assert result["cumulative_incidence"]["A"]["reason_code"] == "treatment.no_exposures"
    timing = next(row for row in result["timing"] if row["group"] == "A")
    assert timing["exposed_n"] == 0 and timing["median"] is None
    frame.loc[:, ["exposed", "first_exposure_hours"]] = [0, None]
    result = summarize_treatment_patterns(frame, **arguments)
    assert result["timing"] == []
    assert result["timing_status"] == "treatment.no_exposures"
    json.dumps(result, allow_nan=False)


def test_reviewed_summary_and_required_columns_are_enforced():
    from easyicu.research_agent.skill_packages import (
        TreatmentPatternError, public_demo, summarize_treatment_patterns,
    )

    frame, arguments = public_demo()
    with pytest.raises(TreatmentPatternError, match="treatment.columns_required"):
        summarize_treatment_patterns(frame.drop(columns=["exposed"]), **arguments)
    arguments["table_spec"] = arguments["table_spec"].model_copy(update={"missing_group_policy": "exclude_and_report"})
    with pytest.raises(TreatmentPatternError, match="treatment.summary_contract_invalid"):
        summarize_treatment_patterns(frame, **arguments)


def test_existing_owners_receive_exact_groups_and_competing_events(monkeypatch):
    from easyicu.research_agent import skill_packages as owner

    frame, arguments = owner.public_demo()
    calls = []
    from easyicu.research_agent.methods import competing_risks

    original = competing_risks.estimate_cif

    def cif(durations, events, event_of_interest):
        calls.append((durations.tolist(), events.tolist(), event_of_interest))
        return original(durations, events, event_of_interest)

    monkeypatch.setattr(competing_risks, "estimate_cif", cif)
    result = owner.summarize_treatment_patterns(frame, **arguments)
    assert calls[0] == ([2., 4., 6., 18.], [1, 1, 2, 3], 1)
    assert calls[1] == ([1., 7., 5., 24.], [1, 1, 2, 3], 1)
    assert result["time_origin"] == "icu_admission"


def test_alive_discharge_precludes_later_in_icu_use():
    from easyicu.research_agent.skill_packages import public_demo, summarize_treatment_patterns

    frame, arguments = public_demo()
    frame["discharged_alive"] = [1, 1, 0, 1, 1, 1, 0, 1]
    frame.loc[3, "followup_hours"] = 1
    result = summarize_treatment_patterns(frame, **arguments)
    curve = result["cumulative_incidence"]["A"]
    assert curve["n_competing"] == 2
    assert curve["cif"][-1] == .5
