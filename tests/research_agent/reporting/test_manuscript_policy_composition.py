"""Reader repairs must survive the adjacent, unchanged authority gates."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.manuscript_claim_policy import (
    filter_evidence_bound_scaffold,
)
from easyicu.research_agent.reporting.manuscript_quality import (
    repair_registered_display_callouts,
)


def _filter(text: str, evidence_ids: tuple[str, ...] = ("result",)):
    return filter_evidence_bound_scaffold(
        text,
        resolve_claim=lambda _ref: None,
        resolve_evidence=lambda ref: ref in evidence_ids,
    )


def test_host_display_repairs_survive_strict_findings_grammar() -> None:
    draft = "## Results\n\n### Cohort characteristics\n\n### Primary association\n"
    repaired, changes = repair_registered_display_callouts(
        draft, expected_display_labels=("Table 1", "Figure 1"),
    )

    filtered = _filter(repaired, ("table_one", "publication_figure_contract"))

    assert len(changes) == 2
    assert filtered.filtered_sentences == ()
    assert "Table 1" in filtered.scaffold and "Figure 1" in filtered.scaffold
    assert repair_registered_display_callouts(
        repaired, expected_display_labels=("Table 1", "Figure 1"),
    ) == (repaired, ())


def test_display_repairs_do_not_bypass_current_evidence_membership() -> None:
    repaired, _ = repair_registered_display_callouts(
        "## Results\n\n### Cohort characteristics\n",
        expected_display_labels=("Table 1",),
    )

    filtered = _filter(repaired, ())

    assert filtered.filtered_sentences
    assert "Table 1" not in filtered.scaffold


@pytest.mark.parametrize("unit", ("ICU", "intensive care", "intensive care unit"))
def test_equivalent_icu_unit_names_keep_only_cited_numeric_facts(unit: str) -> None:
    sentence = f"The cohort comprised 120 {unit} stays {{evidence:result}}."

    assert _filter("## Results\n\n" + sentence).filtered_sentences == ()
    assert _filter("## Results\n\n" + sentence, ()).filtered_sentences


def test_recorded_result_count_is_not_a_claim_of_success_or_stability() -> None:
    sentence = (
        "The recorded sensitivity analysis result count was 0 "
        "{evidence:result}."
    )

    assert _filter("## Results\n\n" + sentence).filtered_sentences == ()
    assert _filter("## Results\n\n" + sentence, ()).filtered_sentences


@pytest.mark.parametrize("sentence", (
    "The cohort comprised 120 intensive care stays with reduced mortality {evidence:result}.",
    "Intensive care improved mortality in 120 stays {evidence:result}.",
    "The recorded sensitivity analysis result count was 0 and validated stability {evidence:result}.",
    "No sensitivity analysis was needed because results were robust {evidence:result}.",
    "Sepsis was harmful in 120 stays {evidence:result}.",
))
def test_plain_vocabulary_never_authorizes_a_scientific_conclusion(sentence: str) -> None:
    assert _filter("## Results\n\n" + sentence).unsupported_scientific_claim_sentences

