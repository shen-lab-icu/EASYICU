"""A number binds only when its words name the fact it binds to (P1-1).

External review: with the value and quantity-kind filters applied, the binder
still accepted the single remaining candidate, so a real registered number
went out with a footnote that named the wrong field:

1. "12 patients" bound to ``n_deaths``.
2. A printed mean bound to a registered median.
3. A value in days bound to a field registered in hours.
4. A group-A value bound to the group-B field.
5. A percentage bound to a proportion of another denominator or field.

The v0.1.0 mitigation fails closed: when another registered fact of the same
quantity kind differs in field, statistic, unit or group, the bind stands only
when the prose names the chosen fact's own identity. These tests read the
production write-phase order (bind_manuscript -> STRICT sentence drop ->
bind_numeric_values) and the SOFT marker the reviewer sees. Synthetic values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.numeric_claim_identity import (
    NUMERIC_FACT_IDENTITY_NOT_NAMED,
    numeric_fact_identity_tokens,
)
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values,
    drop_untraceable_numeric_sentences,
)

SUMMARY = {
    "n": 500,
    "n_deaths": 12,
    "los_days": {"median": 3.1, "mean": 4.2},
    "los_hours": {"median": 72.0},
    "age_by_group": {"group_a": {"median": 65}, "group_b": {"median": 71}},
    "mortality_rate": 0.25,
    "mortality_rate_ventilated": 0.4,
    "sepsis_prevalence": 0.31,
}
EVIDENCE = "cohort_stats"
RECORDS = [
    {
        "step_id": "01_cohort",
        "status": "ok",
        "evidence_ids": [EVIDENCE],
        "step_summary": SUMMARY,
    }
]


@pytest.fixture
def store(tmp_path: Path) -> EvidenceStore:
    summary_path = tmp_path / "step_summary.json"
    summary_path.write_text(json.dumps(SUMMARY), encoding="utf-8")
    store = EvidenceStore(tmp_path / "run", enforcement_mode="strict")
    store.register_file(
        kind="statistic",
        description="Synthetic cohort descriptive statistics.",
        source_path=summary_path,
        produced_by_step="01_cohort",
        evidence_id=EVIDENCE,
        producer="runner",
    )
    store.register_step_summary_numerics(
        step_id="01_cohort", evidence_id=EVIDENCE, summary=SUMMARY
    )
    return store


def _write_phase(store: EvidenceStore, sentence: str):
    bound = store.bind_manuscript(f"## Results\n\n{sentence}\n", per_step_records=RECORDS)
    kept, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=RECORDS
    )
    final, bindings, untraced = bind_numeric_values(
        kept, evidence=store, per_step_records=RECORDS
    )
    return kept, removed, final, bindings, untraced


def _prose(sentence: str) -> str:
    return sentence.split("{evidence:")[0].strip()


REVIEWER_CASES = [
    # 1. 12 is n_deaths; "patients" does not name the deaths.
    ("The cohort included 12 patients {evidence:cohort_stats}.", "12", "n_deaths"),
    # 2. 3.1 is the registered median, printed as a mean.
    (
        "The mean length of stay was 3.1 days {evidence:cohort_stats}.",
        "3.1",
        "los_days.median",
    ),
    # 3. 72 is registered in hours, printed in days.
    (
        "The median length of stay was 72 days {evidence:cohort_stats}.",
        "72",
        "los_hours.median",
    ),
    # 4. 71 is group B's median age, printed for group A.
    (
        "In group A, the median age was 71 years {evidence:cohort_stats}.",
        "71",
        "age_by_group.group_b.median",
    ),
    # 5a. 25% is overall mortality, printed for the ventilated denominator.
    (
        "Mortality among ventilated patients was 25% {evidence:cohort_stats}.",
        "25",
        "mortality_rate",
    ),
    # 5b. 25% is mortality, printed as the sepsis prevalence.
    ("Sepsis prevalence was 25% {evidence:cohort_stats}.", "25", "mortality_rate"),
]


@pytest.mark.parametrize("sentence, value, wrong_field", REVIEWER_CASES)
def test_a_reviewer_wrong_field_bind_is_dropped_under_strict(
    store, sentence, value, wrong_field
):
    kept, removed, final, bindings, untraced = _write_phase(store, sentence)

    assert _prose(sentence) not in kept
    assert len(removed) == 1
    (detail,) = removed
    assert value in "".join(detail["untraced"])
    (unnamed,) = detail["unnamed_fact_identity"]
    assert unnamed["reason_code"] == NUMERIC_FACT_IDENTITY_NOT_NAMED
    assert unnamed["candidates"] == [f"01_cohort:{wrong_field}@{EVIDENCE}"]
    assert bindings == {} and untraced == []
    assert wrong_field not in final


@pytest.mark.parametrize("sentence, value, wrong_field", REVIEWER_CASES)
def test_a_reviewer_wrong_field_bind_is_marked_not_footnoted_in_soft_mode(
    store, sentence, value, wrong_field
):
    bound = store.bind_manuscript(f"## Results\n\n{sentence}\n", per_step_records=RECORDS)
    final, bindings, untraced = bind_numeric_values(
        bound,
        evidence=store,
        per_step_records=RECORDS,
        enforcement_mode=EvidenceEnforcementMode.SOFT,
    )

    assert bindings == {}
    assert len(untraced) == 1 and value in untraced[0]
    assert f"reason={NUMERIC_FACT_IDENTITY_NOT_NAMED}" in final
    assert "<!-- AMBIGUOUS:" in final
    assert "[^claim_" not in final


def test_strict_binding_reports_the_reason_code(store):
    bound = store.bind_manuscript(
        "## Results\n\nThe cohort included 12 patients {evidence:cohort_stats}.\n",
        per_step_records=RECORDS,
    )
    with pytest.raises(EvidenceEnforcementError) as raised:
        bind_numeric_values(bound, evidence=store, per_step_records=RECORDS)
    (unnamed,) = raised.value.detail["unnamed_fact_identity"]
    assert unnamed["reason_code"] == NUMERIC_FACT_IDENTITY_NOT_NAMED
    assert unnamed["not_ruled_out"] == ["01_cohort:n"]


POSITIVE_CONTROLS = [
    ("The cohort recorded 12 deaths {evidence:cohort_stats}.", ["n_deaths"]),
    ("The cohort included 500 patients {evidence:cohort_stats}.", ["n"]),
    (
        "The median length of stay was 3.1 days {evidence:cohort_stats}.",
        ["los_days.median"],
    ),
    (
        "The mean length of stay was 4.2 days {evidence:cohort_stats}.",
        ["los_days.mean"],
    ),
    (
        "The median length of stay was 72 hours {evidence:cohort_stats}.",
        ["los_hours.median"],
    ),
    (
        "In group B, the median age was 71 years {evidence:cohort_stats}.",
        ["age_by_group.group_b.median"],
    ),
    (
        "In group A, the median age was 65 years {evidence:cohort_stats}.",
        ["age_by_group.group_a.median"],
    ),
    ("Overall mortality was 25% {evidence:cohort_stats}.", ["mortality_rate"]),
    (
        "Mortality among ventilated patients was 40% {evidence:cohort_stats}.",
        ["mortality_rate_ventilated"],
    ),
    ("Sepsis prevalence was 31% {evidence:cohort_stats}.", ["sepsis_prevalence"]),
    (
        "Among the 500 patients there were 12 deaths {evidence:cohort_stats}.",
        ["n", "n_deaths"],
    ),
    (
        "The median length of stay was 3.1 days and the mean was 4.2 days "
        "{evidence:cohort_stats}.",
        ["los_days.median", "los_days.mean"],
    ),
]


@pytest.mark.parametrize("sentence, fields", POSITIVE_CONTROLS)
def test_an_explicitly_named_fact_still_binds(store, sentence, fields):
    kept, removed, final, bindings, untraced = _write_phase(store, sentence)

    assert removed == [] and untraced == []
    assert _prose(sentence) in kept
    assert [claim.source_field for claim in bindings.values()] == fields


def test_a_statistic_named_after_the_number_does_not_rescue_a_mean(store):
    # "median" appears in the sentence, but "mean" governs 3.1.
    sentence = (
        "The mean length of stay was 3.1 days (median 4.2) {evidence:cohort_stats}."
    )
    kept, removed, _final, bindings, _untraced = _write_phase(store, sentence)
    assert _prose(sentence) not in kept and len(removed) == 1
    assert bindings == {}


@pytest.mark.parametrize(
    "field, tokens",
    [
        ("n", set()),
        ("n_deaths", {"deaths"}),
        ("los_days.median", {"los", "days", "median"}),
        ("median_length_of_stay", {"median", "los"}),
        ("age_by_group.group_b.median", {"age", "b", "median"}),
        ("groups.non_survivors.age.median", {"nonsurvivors", "age", "median"}),
        ("mortality_rate_ventilated", {"mortality", "ventilated"}),
        ("rows[3].median", {"rows", "median"}),
    ],
)
def test_a_field_identity_is_read_from_its_name(field, tokens):
    assert numeric_fact_identity_tokens(field) == frozenset(tokens)
