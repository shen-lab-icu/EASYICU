"""A ratio label narrows only the ratio's own numbers.

A results sentence that names a hazard, odds or risk ratio narrows the numbers
after the label to claims on that scale, so a ratio cannot bind to a same-valued
count or proportion.  The label governed every later number in the sentence,
though, and a survival estimate is stated with numbers that are not ratios:
the step of a per-unit estimate ("per 500 U/L"), the exposure values of a
contrast ("133.4 versus 139 mmol/L"), an interval's days ("days 120 to 365"),
the landmark hour, a measured value, a percentage or an event count.  Each was
narrowed to ratio claims, stayed untraced, and the strict binder refused the
sentence.  A number the prose ties to a step, a time coordinate, a unit, a
percentage or a count now binds to its own field; the ratio and its interval
keep their scale.

Synthetic summaries in the suites' reporting shapes; no patient data.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"
INTERVALS = "reportable_survival_results.time_varying_adjusted_association.intervals"


def _interval(start: float, end: float, hazard_ratio: float, low: float, high: float) -> dict:
    return {
        "start_days": start, "end_days": end, "hazard_ratio": hazard_ratio,
        "ci_low": low, "ci_high": high, "p_value": 0.0213,
    }


def _summary() -> dict:
    return {
        "status": "ok",
        "landmark_hours": 168.0,
        "prevalent_exposure_cutoff_hours": 144.0,
        "n_analysis_cohort": 1234,
        "n_events": 312,
        "mortality_exposed": 0.182,
        "mortality_unexposed": 0.11,
        "reportable_continuous_survival_results": {
            "exposure_unit": "U/L",
            "exposure_increment": 500.0,
            "adjusted_hazard_ratio_per_unit": {
                "hazard_ratio": 1.12, "ci_low": 1.05, "ci_high": 1.2,
            },
        },
        "spline_contrast": {
            "comparison_value": 133.4, "reference_value": 139.0,
            "hazard_ratio": 1.31, "ci_low": 1.1, "ci_high": 1.55,
        },
        "reportable_survival_results": {
            "time_varying_adjusted_association": {
                "intervals": [
                    _interval(0.0, 30.0, 1.44, 1.21, 1.71),
                    _interval(30.0, 120.0, 1.18, 1.02, 1.37),
                    _interval(120.0, 365.0, 0.91, 0.8, 1.04),
                ],
            },
        },
    }


def _bind(tmp_path, sentence: str, summary: dict | None = None, *, strict: bool = True):
    """Bind one cited Results sentence as the manuscript binder does."""

    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    summary = _summary() if summary is None else summary
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(
        run_dir,
        enforcement_mode=(
            EvidenceEnforcementMode.STRICT if strict else EvidenceEnforcementMode.SOFT
        ),
    )
    store.register_file(
        kind="statistic", description="Survival suite summary", source_path=source,
        evidence_id=EVIDENCE, produced_by_step=STEP, producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(step_id=STEP, evidence_id=EVIDENCE, summary=summary)
    records = [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]
    bound = store.bind_manuscript(
        f"## Results\n\n{sentence} {{evidence:{EVIDENCE}}}\n", per_step_records=records
    )
    _, binding, untraced = bind_numeric_values(bound, evidence=store, per_step_records=records)
    return [claim.source_field for claim in binding.values()], untraced


@pytest.mark.parametrize(
    ("step", "unit"), [("500", "U/L"), ("0.5", "mg/dL"), ("1,000", "IU/L"), ("0.5", None)]
)
def test_the_step_of_a_per_unit_estimate_binds_to_the_step(tmp_path, step, unit):
    summary = _summary()
    reporting = summary["reportable_continuous_survival_results"]
    reporting.update(exposure_unit=unit, exposure_increment=float(step.replace(",", "")))
    words = step if unit is None else f"{step} {unit}"

    fields, untraced = _bind(
        tmp_path,
        f"The adjusted hazard ratio per {words} increase in the exposure was "
        "1.12 (95% CI 1.05 to 1.20).",
        summary,
    )

    assert untraced == []
    estimate = "reportable_continuous_survival_results.adjusted_hazard_ratio_per_unit"
    assert fields == [
        "reportable_continuous_survival_results.exposure_increment",
        f"{estimate}.hazard_ratio", f"{estimate}.ci_low", f"{estimate}.ci_high",
    ]


@pytest.mark.parametrize("versus", ["versus", "vs."])
def test_the_exposure_values_of_a_contrast_bind_to_the_contrast(tmp_path, versus):
    fields, untraced = _bind(
        tmp_path,
        "The adjusted hazard ratio comparing the 10th percentile with the median "
        f"(133.4 {versus} 139 mmol/L) was 1.31 (95% CI 1.10 to 1.55).",
    )

    assert untraced == []
    assert fields[:3] == [
        "spline_contrast.comparison_value", "spline_contrast.reference_value",
        "spline_contrast.hazard_ratio",
    ]


def test_an_intervals_days_bind_to_the_interval_even_after_another_interval(tmp_path):
    fields, untraced = _bind(
        tmp_path,
        "The adjusted hazard ratio for days 120 to 365 after the landmark was 0.91 "
        "(95% CI 0.80 to 1.04), and for days 30 to 120 it was 1.18 (95% CI 1.02 to 1.37).",
    )

    assert untraced == []
    assert {
        f"{INTERVALS}[2].end_days", f"{INTERVALS}[2].hazard_ratio", f"{INTERVALS}[1].hazard_ratio",
    } <= set(fields)
    # 120 ends one interval and starts the next: either is the cited record's field.
    assert {f"{INTERVALS}[1].end_days", f"{INTERVALS}[2].start_days"} & set(fields)


@pytest.mark.parametrize(
    "estimand",
    [
        "adjusted hazard ratio for days 30 to 120 after the landmark",
        "adjusted hazard ratio per 500 U/L increase in the exposure for days 30 to 120 "
        "after the landmark",
        "per-500 U/L adjusted hazard ratio for days 30 to 120 after the landmark",
    ],
)
def test_a_claim_sentence_in_its_rendered_shape_binds_whole(tmp_path, estimand):
    # ScientificClaim.render_text: the estimand, its estimate after a comma, then the CI.
    fields, untraced = _bind(
        tmp_path,
        "After adjustment for age, the exposure was positively associated with death in the "
        "stays alive and under observation at the 168-hour landmark "
        f"({estimand}, 1.18; 95% CI, 1.02 to 1.37; analysis role: primary).",
    )

    assert untraced == []
    assert {
        "landmark_hours", f"{INTERVALS}[1].hazard_ratio", f"{INTERVALS}[1].ci_low",
        f"{INTERVALS}[1].ci_high",
    } <= set(fields)
    assert {f"{INTERVALS}[1].end_days", f"{INTERVALS}[2].start_days"} & set(fields)


def test_the_last_interval_ends_at_the_longest_follow_up(tmp_path):
    # The last interval ends at the longest observed follow-up, not at a cut point:
    # a 365-day endpoint cut at 7, 14, 28 and 90 days prints "days 90 to 364.958".
    summary = _summary()
    summary["reportable_survival_results"]["time_varying_adjusted_association"]["intervals"] = [
        _interval(0.0, 90.0, 1.44, 1.21, 1.71),
        _interval(90.0, 364.958, 0.91, 0.8, 1.04),
    ]

    fields, untraced = _bind(
        tmp_path,
        "The adjusted hazard ratio for days 90 to 364.958 after the landmark was 0.91 "
        "(95% CI 0.80 to 1.04).",
        summary,
    )

    assert untraced == []
    assert fields == [
        f"{INTERVALS}[1].end_days", f"{INTERVALS}[1].hazard_ratio",
        f"{INTERVALS}[1].ci_low", f"{INTERVALS}[1].ci_high",
    ]


def test_the_landmark_and_cutoff_hours_bind_to_their_fields(tmp_path):
    fields, untraced = _bind(
        tmp_path,
        "The adjusted hazard ratio was 1.31 (95% CI 1.10 to 1.55) among stays alive at "
        "the 168-hour landmark, excluding exposed records first recorded at or before "
        "hour 144.",
    )

    assert untraced == []
    assert fields[-2:] == ["landmark_hours", "prevalent_exposure_cutoff_hours"]


def test_a_percentage_and_a_count_after_the_label_bind_to_their_fields(tmp_path):
    fields, untraced = _bind(
        tmp_path,
        "The adjusted hazard ratio was 1.31 (95% CI 1.10 to 1.55; 312 events, n = 1,234); "
        "mortality was 18.2% among exposed and 11.0% among unexposed stays.",
    )

    assert untraced == []
    assert fields[3:] == [
        "n_events", "n_analysis_cohort", "mortality_exposed", "mortality_unexposed",
    ]


@pytest.mark.parametrize(
    ("quantity", "value"),
    [("2.5 mmol/L", 2.5), ("350 U/L", 350.0), ("118.5 mm Hg", 118.5), ("72.5 kg", 72.5),
     ("36.8 °C", 36.8), ("67.5 years", 67.5), ("0.5-unit", 0.5)],
)
def test_a_measured_value_after_the_label_binds_to_its_field(tmp_path, quantity, value):
    summary = {
        "status": "ok", "hazard_ratio": 1.31, "ci_low": 1.1, "ci_high": 1.55,
        "exposure_median": value,
    }

    fields, untraced = _bind(
        tmp_path,
        f"The adjusted hazard ratio was 1.31 (95% CI 1.10 to 1.55) at an exposure median of {quantity}.",
        summary,
    )

    assert untraced == []
    assert fields == ["hazard_ratio", "ci_low", "ci_high", "exposure_median"]


def _untyped(*values: float) -> dict:
    """Same-valued numbers with no ratio identity: the label must keep them out."""

    return {"status": "ok", **{f"lactate_{index}": value for index, value in enumerate(values)}}


@pytest.mark.parametrize(
    ("sentence", "ratio_values"),
    [
        # The step after the estimate does not free the estimate.
        ("The adjusted hazard ratio was 1.42 per 500 U/L (95% CI 1.10 to 1.83).",
         ["1.42", "1.10", "1.83"]),
        # A time word that is the unit of a number is no coordinate.
        ("The hazard ratio at 300 days 1.42 was imprecise.", ["1.42"]),
        ("The hazard ratio at 1 year 1.42 was imprecise.", ["1.42"]),
        ("The hazard ratio at 6 months 1.42 was imprecise.", ["1.42"]),
        # A letter that starts a word is no unit.
        ("Hazard ratios of 1.42 U-shaped in the exposure were estimated.", ["1.42"]),
        # Two ratios contrasted without a unit stay ratios.
        ("Hazard ratios of 1.10 versus 1.83 were estimated in the two analyses.",
         ["1.10", "1.83"]),
        ("The hazard ratio was 1.42, a 1.83-fold higher hazard.", ["1.42", "1.83"]),
    ],
)
def test_the_ratio_and_its_interval_keep_their_scale(tmp_path, sentence, ratio_values):
    summary = _untyped(1.42, 1.10, 1.83, 500.0, 300.0)

    with pytest.raises(EvidenceEnforcementError, match="not traceable"):
        _bind(tmp_path, sentence, summary)
    _fields, untraced = _bind(tmp_path / "soft", sentence, summary, strict=False)
    assert [value for value in untraced if value in ratio_values] == ratio_values
