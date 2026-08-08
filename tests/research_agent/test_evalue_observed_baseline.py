"""An E-value must be anchored to this run's observed event rate, or not reported.

Converting an odds ratio to a risk ratio needs the cohort's baseline event
rate. The host used to seed ``baseline_prev = 0.1`` and let every failure mode
fall through to it -- no outcome-rate product, an unreadable file, an
unparseable cell -- so a reported E-value could be computed at an invented 10%
rate. It carried an explanatory note, but a note does not make an invented
scientific input defensible, and nothing in the value itself reveals the
substitution.

The guessed rate is not a rounding detail. It moves the reported number:

    OR = 2.0, assumed rate 0.10  ->  E = 3.04
    OR = 2.0, observed rate 0.214 ->  E = 2.68

Both are "the E-value for OR=2.0". Only one of them is about this cohort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.post_analysis import EValueConversionSpec
from easyicu.research_agent.methods.sensitivity import (
    BaselinePrevalenceRequiredError,
    compute_e_value,
)


def _spec(**overrides: str) -> EValueConversionSpec:
    payload = {
        "baseline_risk_column": "outcome_rate",
        "population_column": "population",
        "baseline_population": "analysis_cohort",
    }
    payload.update(overrides)
    return EValueConversionSpec(**payload)


from easyicu.research_agent.orchestration.finalize import (
    _primary_association_evalue_rows,
    _write_primary_association_evalue_artifacts,
    resolve_observed_event_rate,
)


# ---------------------------------------------------------------------------
# The kernel must refuse, so no caller can reintroduce the guess
# ---------------------------------------------------------------------------


def test_an_odds_ratio_without_a_baseline_rate_is_refused():
    with pytest.raises(BaselinePrevalenceRequiredError):
        compute_e_value(estimate=2.0, estimate_type="or")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_a_baseline_rate_outside_the_unit_interval_is_refused(bad: float):
    """0 and 1 are refused too: neither has a risk ratio to convert into."""

    with pytest.raises(BaselinePrevalenceRequiredError):
        compute_e_value(estimate=2.0, estimate_type="or", baseline_prevalence=bad)


def test_rr_and_hr_need_no_baseline_rate():
    """Only the OR conversion needs it; refusing the others would be wrong."""

    assert compute_e_value(estimate=2.0, estimate_type="rr").e_value > 1.0
    assert compute_e_value(estimate=2.0, estimate_type="hr").e_value > 1.0


def test_typed_deterministic_association_output_reaches_evalue_finalization(
    tmp_path: Path,
) -> None:
    """Typed owner -> semantic evidence authority -> O23 registered artefact."""

    primary_path = (
        tmp_path
        / "steps"
        / "primary"
        / "outputs"
        / "adjusted_association_estimates.csv"
    )
    primary_path.parent.mkdir(parents=True)
    primary_path.write_text(
        "fit_status,estimate,ci_low,ci_high,effect_scale,exposure,contrast\n"
        "fitted,1.8,1.2,2.7,odds_ratio,sep3,\n",
        encoding="utf-8",
    )
    outcome_path = tmp_path / "steps" / "outcome" / "outputs" / "outcome_rate.csv"
    outcome_path.parent.mkdir(parents=True)
    outcome_path.write_text(
        "population,outcome_rate\nanalysis_cohort,0.2\n", encoding="utf-8"
    )
    evidence = EvidenceStore(tmp_path)
    primary = evidence.register_file(
        kind="table",
        description="Typed deterministic primary association",
        source_path=primary_path,
        evidence_id="primary_association",
        produced_by_step="primary",
        producer="adjusted_association_executor",
        generation_mode="system",
    )
    outcome = evidence.register_file(
        kind="table",
        description="Observed outcome rate",
        source_path=outcome_path,
        evidence_id="outcome_rate",
        produced_by_step="outcome",
        producer="deterministic_descriptive",
        generation_mode="system",
    )

    artifacts = _write_primary_association_evalue_artifacts(
        evidence=evidence,
        per_step_records=[
            {
                "step_id": "primary",
                "status": "ok",
                "evidence_ids": [primary.evidence_id],
            },
            {
                "step_id": "outcome",
                "status": "ok",
                "evidence_ids": [outcome.evidence_id],
            },
        ],
        run_dir=tmp_path,
        spec=_spec(),
    )

    assert artifacts is not None
    assert artifacts.csv_path == tmp_path / "e_values.csv"
    assert artifacts.csv_path.is_file()
    assert artifacts.markdown_path.is_file()
    assert artifacts.row_count == 1
    assert artifacts.baseline_population == "analysis_cohort"
    assert len(artifacts.conversion_spec_sha256) == 64
    assert evidence.get("e_values") is not None
    text = artifacts.csv_path.read_text(encoding="utf-8")
    assert "sep3" in text
    assert "1.8" in text


def test_evalue_population_binding_changes_plan_scientific_identity() -> None:
    from easyicu.research_agent.authority.plan_scope import (
        _plan_scientific_scope_signature,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    base = AnalysisPlan(
        research_question="q",
        steps=[],
        evalue_conversion_spec=_spec(),
    )
    changed = base.model_copy(
        update={
            "evalue_conversion_spec": _spec(baseline_population="unexposed_reference")
        }
    )

    assert _plan_scientific_scope_signature(base) != (
        _plan_scientific_scope_signature(changed)
    )


def test_typed_non_odds_ratio_is_not_reinterpreted_as_an_or(tmp_path: Path) -> None:
    path = tmp_path / "adjusted_association_estimates.csv"
    path.write_text(
        "estimate,ci_low,ci_high,effect_scale,exposure\n"
        "1.8,1.2,2.7,mean_difference,sep3\n",
        encoding="utf-8",
    )

    assert _primary_association_evalue_rows(path, baseline_prevalence=0.2) == []


def test_the_assumed_rate_really_did_move_the_reported_number():
    """The measurement that justifies refusing rather than noting.

    If the old default had been numerically harmless, a note would have been a
    defensible design. It was not.
    """

    assumed = compute_e_value(
        estimate=2.0, estimate_type="or", baseline_prevalence=0.1
    ).e_value
    observed = compute_e_value(
        estimate=2.0, estimate_type="or", baseline_prevalence=0.214
    ).e_value

    assert round(assumed, 2) == 3.04
    assert round(observed, 2) == 2.68
    # 0.36 apart. Not a rounding artefact, and in the direction that
    # OVERSTATES robustness to unmeasured confounding.
    assert abs(assumed - observed) == pytest.approx(0.358, abs=0.01)


def test_the_note_records_the_rate_the_conversion_actually_used():
    result = compute_e_value(
        estimate=2.0, estimate_type="or", baseline_prevalence=0.214
    )
    assert result.note is not None
    assert "0.2140" in result.note


# ---------------------------------------------------------------------------
# The resolver reads the run's own product, and refuses rather than reduces
# ---------------------------------------------------------------------------


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_single_declared_rate_resolves(tmp_path):
    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "population,outcome_rate\nanalysis_cohort,0.214\n",
    )
    resolved = resolve_observed_event_rate(path, _spec())

    assert resolved.value == pytest.approx(0.214)
    assert resolved.cause == ""
    assert resolved.source_column == "outcome_rate"


def test_a_missing_product_resolves_to_nothing_not_to_a_default(tmp_path):
    resolved = resolve_observed_event_rate(None, _spec())
    assert resolved.value is None
    assert resolved.cause == "no_outcome_rate_product"


def test_an_unreadable_product_resolves_to_nothing(tmp_path):
    resolved = resolve_observed_event_rate(tmp_path / "does_not_exist.csv", _spec())
    assert resolved.value is None
    assert resolved.cause == "outcome_rate_unreadable"


def test_a_product_with_no_usable_rate_resolves_to_nothing(tmp_path):
    # Present, parseable, and carries nothing that is a rate in (0, 1).
    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "population,n_events,n_total\nanalysis_cohort,2021,9445\n",
    )
    resolved = resolve_observed_event_rate(path, _spec())

    assert resolved.value is None
    assert resolved.cause == "baseline_population_rate_invalid"


def test_disagreeing_rates_are_refused_not_reduced(tmp_path):
    """The failure the old code could not even see.

    It kept the LAST matching cell, so a per-group product silently anchored
    the E-value to whichever row sorted last. First, last, and mean are three
    different scientific choices; none of them is stated to the reader.
    """

    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "population,outcome_rate\nanalysis_cohort,0.31\nanalysis_cohort,0.18\n",
    )
    resolved = resolve_observed_event_rate(path, _spec())

    assert resolved.value is None
    assert resolved.cause == "baseline_population_rate_ambiguous"
    assert resolved.candidates == (0.18, 0.31)
    assert "2 row(s)" in resolved.reason


def test_duplicate_population_rows_are_ambiguous_even_when_rates_agree(tmp_path):

    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "population,outcome_rate\nanalysis_cohort,0.214\nanalysis_cohort,0.214\n",
    )
    resolved = resolve_observed_event_rate(path, _spec())

    assert resolved.value is None
    assert resolved.cause == "baseline_population_rate_ambiguous"


def test_a_different_population_is_not_used_as_the_baseline(tmp_path):
    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "population,outcome_rate\nexposed,0.214\n",
    )
    resolved = resolve_observed_event_rate(path, _spec())
    assert resolved.value is None
    assert resolved.cause == "baseline_population_not_found"


def test_an_or_conversion_without_a_typed_population_contract_is_omitted(tmp_path):
    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "population,outcome_rate\nanalysis_cohort,0.214\n",
    )
    resolved = resolve_observed_event_rate(path, None)
    assert resolved.value is None
    assert resolved.cause == "evalue_conversion_spec_required"


def test_every_unresolved_cause_is_distinct(tmp_path):
    """A shared cause string would make the finding unactionable.

    "E-values were not computed" is only useful if it says which of the four
    things went wrong -- a reader has to know whether to add a product, fix a
    file, or declare which rate is the baseline.
    """

    causes = {
        resolve_observed_event_rate(None, _spec()).cause,
        resolve_observed_event_rate(tmp_path / "nope.csv", _spec()).cause,
        resolve_observed_event_rate(
            _write(
                tmp_path,
                "a.csv",
                "population,n\nanalysis_cohort,5\n",
            ),
            _spec(),
        ).cause,
        resolve_observed_event_rate(
            _write(
                tmp_path,
                "b.csv",
                "population,outcome_rate\nanalysis_cohort,0.1\nanalysis_cohort,0.9\n",
            ),
            _spec(),
        ).cause,
    }
    assert len(causes) == 4, causes


# ---------------------------------------------------------------------------
# Properties carried over from the deleted second implementation
#
# ``methods/evalue.py`` was a parallel E-value kernel, declared unreachable
# because it disagreed with this one on OR -> RR. It was deleted on 2026-08-07
# in favour of the observed-prevalence (Zhang-Yu) conversion here, which uses
# the cohort's own event rate instead of asking the caller to assert a rare- or
# common-outcome regime. These four properties were only covered by its tests,
# so they move here rather than disappearing with it.
# ---------------------------------------------------------------------------


def test_the_canonical_published_example_reproduces():
    """VanderWeele & Ding 2017's smoking/lung-cancer RR = 3.9 -> E = 7.26."""

    assert compute_e_value(estimate=3.9, estimate_type="rr").e_value == pytest.approx(
        7.26, abs=1e-2
    )


def test_the_evalue_is_symmetric_about_the_null():
    """An RR of r and its inverse carry the same confounding burden."""

    above = compute_e_value(estimate=2.0, estimate_type="rr").e_value
    below = compute_e_value(estimate=0.5, estimate_type="rr").e_value
    assert above == pytest.approx(below, abs=1e-9)
    assert above == pytest.approx(3.4142, abs=1e-3)


def test_an_estimate_at_the_null_needs_no_confounder():
    assert compute_e_value(estimate=1.0, estimate_type="rr").e_value == pytest.approx(
        1.0, abs=1e-9
    )


def test_an_interval_crossing_the_null_has_a_bound_evalue_of_one():
    crossing = compute_e_value(estimate=1.4, estimate_type="rr", ci=(0.9, 1.8))
    assert crossing.e_value_lower_bound == pytest.approx(1.0, abs=1e-9)
