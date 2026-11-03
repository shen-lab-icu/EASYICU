"""A manuscript number binds only to the registered fact its words name.

External review, reproduced through the production order of the write phase
(``enforce_evidence_bound_scaffold`` -> ``bind_manuscript`` -> STRICT
``drop_untraceable_numeric_sentences`` -> ``bind_numeric_values`` ->
``audit_manuscript_numeric_claims``) with one current successful step that
registered ``{"n": 120, "median_age": 60, "median_length_of_stay": 9,
"n_deaths": 12}``:

R01  Short integers escaped the gate. "The median age was 42 years", "The
     median length of stay was 7 days" and "N=42" were kept with zero
     bindings and no finding, while "n=42" and "42 patients" were refused:
     the recogniser only knew lowercase ``n=`` and counted nouns.

R02  Equal value was taken for identity. "The median age was 12.0 years"
     bound to ``n_deaths``, "The cohort included 60 patients" and "The median
     length of stay was 60.0 days" bound to ``median_age`` -- each with a
     footnote -- because a single equal-valued candidate was accepted outright.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.audits.manuscript_claims import (
    audit_manuscript_numeric_claims,
)
from easyicu.research_agent.authority.evidence_store import (
    _NUMERIC_IN_PROSE_RE,
    EvidenceStore,
    NumericClaim,
)
from easyicu.research_agent.authority.numeric_claim_identity import (
    NumericQuantityKind,
    prose_quantity_kind,
    quantity_kind_refuses_claim,
    source_field_quantity_kind,
)
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values,
    drop_untraceable_numeric_sentences,
)

SUMMARY = {"n": 120, "median_age": 60, "median_length_of_stay": 9, "n_deaths": 12}
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
        description="Cohort descriptive statistics.",
        source_path=summary_path,
        produced_by_step="01_cohort",
        evidence_id=EVIDENCE,
        producer="runner",
    )
    store.register_step_summary_numerics(
        step_id="01_cohort", evidence_id=EVIDENCE, summary=SUMMARY
    )
    return store


def _after_policy(store: EvidenceStore, scaffold: str):
    """The write phase from the bound scaffold on: drop, bind, audit."""

    bound = store.bind_manuscript(scaffold, per_step_records=RECORDS)
    kept, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=RECORDS
    )
    final, bindings, untraced = bind_numeric_values(
        kept, evidence=store, per_step_records=RECORDS
    )
    findings = audit_manuscript_numeric_claims(final, per_step_records=RECORDS)
    return kept, removed, bindings, untraced, findings


def _production_chain(store: EvidenceStore, sentence: str):
    scaffold, filtered = store.enforce_evidence_bound_scaffold(
        f"## Results\n\n{sentence}\n", per_step_records=RECORDS
    )
    assert filtered == []
    return _after_policy(store, scaffold)


def _prose(sentence: str) -> str:
    return sentence.split("{evidence:")[0].strip()


R01_UNREGISTERED_SHORT_INTEGERS = [
    "The median age was 42 years {evidence:cohort_stats}.",
    "The median length of stay was 7 days {evidence:cohort_stats}.",
    "N=42 {evidence:cohort_stats}.",
    "n=42 {evidence:cohort_stats}.",
]

R02_WRONG_FIELD_SAME_VALUE = [
    # 12 is n_deaths, not an age.
    "The median age was 12.0 years {evidence:cohort_stats}.",
    # 60 is median_age, not a count of patients.
    "The cohort included 60 patients {evidence:cohort_stats}.",
    # 60 is median_age, not a length of stay.
    "The median length of stay was 60.0 days {evidence:cohort_stats}.",
]


@pytest.mark.parametrize(
    "sentence", R01_UNREGISTERED_SHORT_INTEGERS + R02_WRONG_FIELD_SAME_VALUE
)
def test_a_number_no_named_fact_owns_is_dropped_under_strict(store, sentence):
    kept, removed, bindings, untraced, _findings = _production_chain(store, sentence)

    assert _prose(sentence) not in kept
    assert [item["sentence"] for item in removed] == [
        store.bind_manuscript(sentence, per_step_records=RECORDS).strip()
    ]
    assert bindings == {} and untraced == []


@pytest.mark.parametrize(
    "sentence, value, field",
    [
        ("The median age was 60 years {evidence:cohort_stats}.", "60", "median_age"),
        (
            "The median length of stay was 9 days {evidence:cohort_stats}.",
            "9",
            "median_length_of_stay",
        ),
        ("N=120 {evidence:cohort_stats}.", "120", "n"),
        ("The cohort recorded 12 deaths {evidence:cohort_stats}.", "12", "n_deaths"),
        ("The cohort included 120 patients {evidence:cohort_stats}.", "120", "n"),
    ],
)
def test_a_number_binds_to_the_field_its_words_name(store, sentence, value, field):
    kept, removed, bindings, untraced, findings = _production_chain(store, sentence)

    assert removed == [] and untraced == [] and findings == []
    assert _prose(sentence) in kept
    assert [(claim.value, claim.source_field) for claim in bindings.values()] == [
        (value, field)
    ]


def test_one_sentence_binds_each_number_to_its_own_kind(store):
    sentence = (
        "The cohort included 120 patients with a median age of 60 years and a "
        "median length of stay of 9 days {evidence:cohort_stats}."
    )
    kept, removed, bindings, untraced, _findings = _production_chain(store, sentence)

    assert removed == [] and untraced == []
    assert [claim.source_field for claim in bindings.values()] == [
        "n",
        "median_age",
        "median_length_of_stay",
    ]


def test_a_chinese_age_binds_only_when_its_value_is_the_registered_age(store):
    """The zh manuscript path through the same numeric gates.

    The scaffold policy's neutral-findings vocabulary is English, so this
    starts at the bound scaffold; the numeric gates under test are language
    neutral and run unchanged.
    """

    wrong = "## 结果\n\n年龄中位数为 42 岁 {evidence:cohort_stats}。\n"
    kept, removed, bindings, _untraced, _findings = _after_policy(store, wrong)
    assert "42 岁" not in kept and len(removed) == 1 and bindings == {}

    right = "## 结果\n\n年龄中位数为 60 岁 {evidence:cohort_stats}。\n"
    kept, removed, bindings, untraced, _findings = _after_policy(store, right)
    assert "60 岁" in kept and removed == [] and untraced == []
    assert [claim.source_field for claim in bindings.values()] == ["median_age"]

    swapped = "## 结果\n\n年龄中位数为 12 岁 {evidence:cohort_stats}。\n"
    kept, removed, bindings, _untraced, _findings = _after_policy(store, swapped)
    assert "12 岁" not in kept and len(removed) == 1 and bindings == {}


@pytest.mark.parametrize(
    "text",
    [
        "Sepsis was defined with SOFA-2 as described in Section 2.",
        "Effects are reported with 95% CI and 95 % CI.",
        "The primary outcome was day 28 mortality.",
        "The primary outcome was 28-day mortality, also written 1 year mortality.",
        "Laboratory values within 24 hours of ICU admission were used.",
        "Glucose over 0-24 h after ICU admission was summarised.",
        "Mortality at 28 days was the endpoint; see Figure 2 and Table 1.",
        "Stage 4 patients and group 2 patients were described separately.",
        "COVID-19 and Sepsis-3 cohorts were excluded.",
        "Median values were computed for 2 groups.",
        "between 8-12 events",
        "as reported in Figure 2 (3)",
        "第28天死亡率与入ICU 24小时内的数值。",
    ],
)
def test_labels_windows_and_levels_are_not_result_numbers(store, text):
    assert [m.group("value") for m in _NUMERIC_IN_PROSE_RE.finditer(text)] == []

    cited = f"## Methods\n\n{text[:-1] if text.endswith('.') else text} {{evidence:cohort_stats}}.\n"
    kept, removed, _bindings, untraced, _findings = _after_policy(store, cited)
    assert removed == [] and untraced == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The median age was 42 years.", ["42"]),
        ("The median length of stay was 7 days.", ["7"]),
        ("N=42", ["42"]),
        ("N = 42", ["42"]),
        ("(n:42)", ["42"]),
        ("the median age was 42", ["42"]),
        ("median 42 (IQR 30–55)", ["42", "30", "55"]),
        ("(IQR 3–7)", ["3", "7"]),
        ("aged 42–60 years", ["42", "60"]),
        ("the mean SOFA-2 score was 7", ["7"]),
        ("LOS was 3 d", ["3"]),
        ("the mean age was 42 (12)", ["42", "12"]),
        ("年龄中位数为 42 岁", ["42"]),
        ("年龄中位数为42", ["42"]),
        ("住院时间中位数为 9 天", ["9"]),
    ],
)
def test_a_short_integer_in_a_quantity_phrase_is_a_result_number(text, expected):
    assert [m.group("value") for m in _NUMERIC_IN_PROSE_RE.finditer(text)] == expected


@pytest.mark.parametrize(
    "field, kind",
    [
        ("n", NumericQuantityKind.COUNT),
        ("cohort.n_stays", NumericQuantityKind.COUNT),
        ("n_deaths", NumericQuantityKind.COUNT),
        ("complete_case_flow.n_complete_case", NumericQuantityKind.COUNT),
        ("median_age", NumericQuantityKind.AGE),
        ("table_one.age.median", NumericQuantityKind.AGE),
        ("median_length_of_stay", NumericQuantityKind.DURATION),
        ("los_days.median", NumericQuantityKind.DURATION),
        ("mortality_rate", NumericQuantityKind.PROPORTION),
        ("variable_groups.lact.missingness.max_fraction_missing",
         NumericQuantityKind.PROPORTION),
        # Unknown stays unknown: no kind is guessed.
        ("primary_or", None),
        ("auroc", None),
        ("rows[3].median", None),
        ("missingness.age", None),
        ("age_missing", None),
        ("scientific_runtime_receipt.spline_knot_quantiles[1]", None),
    ],
)
def test_a_field_kind_is_read_only_from_an_explicit_name(field, kind):
    assert source_field_quantity_kind(field) is kind


def _claim(field: str, canonical: float) -> NumericClaim:
    return NumericClaim(
        value=str(canonical),
        canonical=canonical,
        evidence_id=EVIDENCE,
        step_id="01_cohort",
        source_field=field,
    )


def test_unknown_on_either_side_stays_permissive_and_counts_are_whole():
    count = NumericQuantityKind.COUNT
    assert not quantity_kind_refuses_claim(None, _claim("median_age", 60.0))
    assert not quantity_kind_refuses_claim(count, _claim("threshold", 60.0))
    assert quantity_kind_refuses_claim(count, _claim("median_age", 60.0))
    # A count is a whole number whatever its field is called.
    assert quantity_kind_refuses_claim(count, _claim("threshold", 59.6))


def test_a_percentage_does_not_bind_a_count_of_the_same_digits(store):
    sentence = "Mortality was 12% {evidence:cohort_stats}."
    kept, removed, bindings, _untraced, _findings = _after_policy(
        store, f"## Results\n\n{sentence}\n"
    )
    assert "Mortality was 12%" not in kept and len(removed) == 1
    assert bindings == {}


@pytest.mark.parametrize(
    "text, kind",
    [
        ("The median age was 12.0 years", NumericQuantityKind.AGE),
        ("The cohort included 60 patients", NumericQuantityKind.COUNT),
        ("The median length of stay was 60.0 days", NumericQuantityKind.DURATION),
        ("Mortality was 12%", NumericQuantityKind.PROPORTION),
        ("(n = 12)", NumericQuantityKind.COUNT),
        ("median follow-up was 2 years", NumericQuantityKind.DURATION),
        ("年龄中位数为 42 岁", NumericQuantityKind.AGE),
        # Years alone could be an age or a follow-up: unknown.
        ("over 42 years", None),
        ("The AUROC was 0.81", None),
    ],
)
def test_the_prose_names_the_kind_of_its_own_number(text, kind):
    match = list(_NUMERIC_IN_PROSE_RE.finditer(text))[-1]
    assert (
        prose_quantity_kind(text, start=match.start("value"), end=match.end("value"))
        is kind
    )
