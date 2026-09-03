"""The step that owns a number gets cited beside the one the writer chose.

MEASURED, and it was the entire remaining distance on both manuscripts the
pipeline has produced. e1 sepsis (10/10 steps) and e3 KDIGO (11/11) each ended
with exactly two unbound numbers, all four the cohort size 94,458, all four a
sentence citing a step that registered no such value:

    e1  cited 00_probe   owned by 02_table_one_by_sepsis3_status, ...
    e3  cited table_one  owned by 02_table_one_by_kdigo_stage, ...

`writer.txt` already carries the rule with this exact number as its worked
example -- a sentence printing values from different steps must cite EVERY step
that owns one -- and the writer gets no repair pass, so the sentence is final
the moment it is written.

Replaying the real binder over the real evidence stores, appending the owning
citation takes both to zero: e1 2 repairs / +84 bytes / 13 bound / 0 markers,
e3 2 repairs / +78 bytes / 11 bound / 0 markers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from easyicu.research_agent.reporting.manuscript_post import (
    repair_miscited_numeric_citations,
)


class _FakeRecord:
    def __init__(self, evidence_id: str, inputs=()):
        self.evidence_id = evidence_id
        self.inputs = tuple(inputs)
        self.script_evidence_id = None


class _FakeStore:
    """Only the surface the repair uses: records, names, numeric claims."""

    def __init__(self, claims, names):
        self._claims = claims
        self._names = set(names)

    def records(self):
        return [_FakeRecord(name) for name in sorted(self._names)]

    def resolvable_names(self):
        return set(self._names)

    def numeric_claims(self):
        return list(self._claims)


def _store():
    from easyicu.research_agent.authority.evidence_store import NumericClaim

    def claim(step, field):
        return NumericClaim(
            value="94,458",
            canonical=94458.0,
            evidence_id=step,
            step_id=step,
            source_field=field,
            tolerance=0.0,
        )

    return _FakeStore(
        [
            claim("02_table_one_by_sepsis3_status", "cohort_n"),
            claim("04_measurement_missingness_audit", "n_total"),
        ],
        names={
            "00_probe",
            "02_table_one_by_sepsis3_status",
            "04_measurement_missingness_audit",
        },
    )


MISCITING = (
    "The operational denominator comprised 94,458 ICU stays represented in "
    "the supplied cohort definition {evidence:00_probe}.\n"
)


def test_multiple_denominator_owners_are_not_assumed_to_be_the_same_fact() -> None:
    repaired, repairs = repair_miscited_numeric_citations(
        MISCITING, evidence=_store()
    )
    assert repairs == []
    assert repaired == MISCITING


def test_unique_canonical_stay_count_owner_outranks_derived_row_counts() -> None:
    from easyicu.research_agent.authority.evidence_store import NumericClaim

    store = _FakeStore(
        [
            NumericClaim(
                value="94,458",
                canonical=94458.0,
                evidence_id="research_context",
                step_id="research_context",
                source_field="cohort.n_stays",
                tolerance=0.0,
            ),
            NumericClaim(
                value="94,458",
                canonical=94458.0,
                evidence_id="01_model",
                step_id="01_model",
                source_field="input_bindings[0].row_count",
                tolerance=0.0,
            ),
        ],
        names={"00_probe", "research_context", "01_model"},
    )

    repaired, repairs = repair_miscited_numeric_citations(MISCITING, evidence=store)

    assert repairs == [
        {"value": "94,458", "cited": "00_probe", "added": "research_context"}
    ]
    assert "{evidence:research_context}" in repaired


def test_the_unique_owning_citation_is_added() -> None:
    from easyicu.research_agent.authority.evidence_store import NumericClaim

    store = _FakeStore(
        [
            NumericClaim(
                value="94,458",
                canonical=94458.0,
                evidence_id="02_table_one_by_sepsis3_status",
                step_id="02_table_one_by_sepsis3_status",
                source_field="cohort_n",
                tolerance=0.0,
            )
        ],
        names={"00_probe", "02_table_one_by_sepsis3_status"},
    )

    repaired, repairs = repair_miscited_numeric_citations(MISCITING, evidence=store)

    assert repairs == [
        {
            "value": "94,458",
            "cited": "00_probe",
            "added": "02_table_one_by_sepsis3_status",
        }
    ]
    assert "{evidence:02_table_one_by_sepsis3_status}" in repaired


def test_the_writers_own_citation_is_never_removed() -> None:
    """A genuine attribution error must stay visible, not be rewritten away."""

    repaired, _ = repair_miscited_numeric_citations(MISCITING, evidence=_store())
    assert "{evidence:00_probe}" in repaired
    # Additive only: nothing of the original prose disappears.
    for word in ("operational", "denominator", "supplied cohort definition"):
        assert word in repaired


def test_a_correctly_cited_sentence_is_left_byte_identical() -> None:
    text = (
        "The cohort comprised 94,458 ICU stays "
        "{evidence:02_table_one_by_sepsis3_status}.\n"
    )
    repaired, repairs = repair_miscited_numeric_citations(text, evidence=_store())
    assert repairs == []
    assert repaired == text


def test_a_number_that_already_binds_is_not_touched() -> None:
    """Keying on the miscitation alone over-repaired.

    On e3 it fired on 6 sentences where 2 were blocked; the other 4 bound
    perfectly well and would have collected a citation they did not need.
    """

    # The real shape: the sentence carries only an INHERITED citation (the
    # previous sentence's), to a step that owns no 94,458. That is a
    # miscitation by the scoping test -- but since 067b7d4 an inherited
    # citation no longer vetoes, so the number binds on its own and needs no
    # repair. An earlier version of this test cited the owner instead, which
    # never reached the guard at all and let the mutation survive.
    from easyicu.research_agent.authority.evidence_store import NumericClaim

    single = _FakeStore(
        [
            NumericClaim(
                value="94,458",
                canonical=94458.0,
                evidence_id="02_table_one_by_sepsis3_status",
                step_id="02_table_one_by_sepsis3_status",
                source_field="cohort_n",
                tolerance=0.0,
            )
        ],
        names={"00_probe", "02_table_one_by_sepsis3_status"},
    )
    text = "{evidence:00_probe} The cohort comprised 94,458 stays.\n"
    repaired, repairs = repair_miscited_numeric_citations(text, evidence=single)
    assert repairs == [] and repaired == text


def test_nothing_is_added_when_no_owner_is_citable() -> None:
    """With no citable owner the gate must still refuse, not be papered over."""

    from easyicu.research_agent.authority.evidence_store import NumericClaim

    uncitable = _FakeStore(
        [
            NumericClaim(
                value="94,458",
                canonical=94458.0,
                evidence_id="02_table_one_by_sepsis3_status",
                step_id="02_table_one_by_sepsis3_status",
                source_field="cohort_n",
                tolerance=0.0,
            ),
            NumericClaim(
                value="94,458",
                canonical=94458.0,
                evidence_id="04_measurement_missingness_audit",
                step_id="04_measurement_missingness_audit",
                source_field="n_total",
                tolerance=0.0,
            ),
        ],
        # The owners exist as claims but are not citable names.
        names={"00_probe"},
    )
    repaired, repairs = repair_miscited_numeric_citations(
        MISCITING, evidence=uncitable
    )
    assert repairs == [] and repaired == MISCITING


@pytest.mark.parametrize(
    ("sentence", "claims"),
    [
        (
            "The mortality rate was 0.85 {evidence:00_probe}.\n",
            [("01_model", "auroc"), ("04_outcome", "mortality_rate")],
        ),
        (
            "The hazard ratio was 1.20 {evidence:00_probe}.\n",
            [("01_logistic", "odds_ratio"), ("04_survival", "hazard_ratio")],
        ),
        (
            "There were 120 deaths {evidence:00_probe}.\n",
            [("01_cohort", "n_total"), ("04_outcome", "n_deaths")],
        ),
        (
            "Mortality was 85% {evidence:00_probe}.\n",
            [("01_model", "auroc"), ("04_outcome", "mortality_rate")],
        ),
        (
            "The primary odds ratio was 1.20 {evidence:00_probe}.\n",
            [("01_primary", "primary_or"), ("04_sensitivity", "sensitivity_or")],
        ),
    ],
)
def test_same_value_different_semantics_is_never_auto_repaired(
    sentence: str,
    claims: list[tuple[str, str]],
) -> None:
    from easyicu.research_agent.authority.evidence_store import NumericClaim

    store = _FakeStore(
        [
            NumericClaim(
                value=("0.85" if "85" in sentence else "1.20" if "1.20" in sentence else "120"),
                canonical=(0.85 if "85" in sentence else 1.2 if "1.20" in sentence else 120.0),
                evidence_id=step,
                step_id=step,
                source_field=field,
                tolerance=0.0,
            )
            for step, field in claims
        ],
        names={"00_probe", *(step for step, _ in claims)},
    )

    repaired, repairs = repair_miscited_numeric_citations(sentence, evidence=store)

    assert repairs == []
    assert repaired == sentence


def test_same_value_different_semantics_remains_a_strict_block(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceEnforcementError,
        EvidenceStore,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    for evidence_id in ("00_probe", "01_model", "04_outcome"):
        source = tmp_path / f"{evidence_id}.json"
        source.write_text("{}", encoding="utf-8")
        store.register_file(
            kind="statistic",
            description=evidence_id,
            source_path=source,
            evidence_id=evidence_id,
            produced_by_step=evidence_id,
        )
    store.register_numeric_claim(
        value="0.85",
        canonical=0.85,
        evidence_id="01_model",
        step_id="01_model",
        source_field="auroc",
    )
    store.register_numeric_claim(
        value="0.85",
        canonical=0.85,
        evidence_id="04_outcome",
        step_id="04_outcome",
        source_field="mortality_rate",
    )
    sentence = "The mortality rate was 0.85 {evidence:00_probe}.\n"

    repaired, repairs = repair_miscited_numeric_citations(sentence, evidence=store)

    assert repairs == []
    assert repaired == sentence
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(repaired, evidence=store)


def test_the_repair_is_wired_into_the_write_phase() -> None:
    """A repair that is never called would pass every test above."""

    import inspect

    from easyicu.research_agent.reporting import write_phase

    source = inspect.getsource(write_phase)
    assert "repair_miscited_numeric_citations(" in source
    # It must be reported, not applied silently.
    assert "miscitation_repairs" in source
    assert re.search(r"miscitation_repairs\b.*\n?.*findings\.append", source) or (
        "detail={\"miscitation_repairs\": miscitation_repairs}" in source
    )
