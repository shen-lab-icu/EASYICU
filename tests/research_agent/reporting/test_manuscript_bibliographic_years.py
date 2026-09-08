from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError, EvidenceEnforcementMode, EvidenceStore,
)
from easyicu.research_agent.literature import (
    CitationRecord, LiteratureBundle, LiteratureScreeningDecision,
)
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values, drop_untraceable_numeric_sentences,
)


def _literature(**changes):
    record = {
        "key": "definition_2016", "year": "2016",
        "title": "Consensus definitions for Example-3",
        "doi": "10.0000/example", "authors": ["Example Author"],
    }
    record.update(changes)
    return LiteratureBundle(research_question="Example", citations=[CitationRecord(**record)])


@pytest.mark.parametrize("qualifier", ["source-bound", "recorded"])
@pytest.mark.parametrize("tail", [
    " [@definition_2016].", ". [@definition_2016]",
    ", with the source criteria retained for this analysis [@definition_2016].",
])
def test_explicit_definition_year_uses_cited_metadata_not_result_claim(tmp_path, tail, qualifier):
    text = f"We used the {qualifier} 2016 Example-3 definition" + tail
    store = EvidenceStore(tmp_path)
    bound, bindings, untraced = bind_numeric_values(
        text, evidence=store, enforcement_mode=EvidenceEnforcementMode.STRICT,
        literature=_literature(),
    )
    assert bound == text and not bindings and not untraced
    cleaned, removed = drop_untraceable_numeric_sentences(
        text, evidence=store, literature=_literature(),
    )
    assert cleaned == text and not removed


@pytest.mark.parametrize("changes", [
    {"year": "2015"}, {"key": "different"},
    {"title": "An unrelated definition"}, {"doi": None},
])
def test_definition_year_requires_matching_source_metadata(tmp_path, changes):
    text = "We used the 2016 Example-3 definition [@definition_2016]."
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            text, evidence=EvidenceStore(tmp_path),
            enforcement_mode=EvidenceEnforcementMode.STRICT,
            literature=_literature(**changes),
        )


@pytest.mark.parametrize("text", [
    "In 2016, patients met the Example-3 definition [@definition_2016].",
    "There were 2016 patients meeting the Example-3 definition [@definition_2016].",
    "We observed 2016 events [@definition_2016].",
    "We used a threshold of 2016 [@definition_2016].",
    "We used the 2016 Example-3 definition.\n\n[@definition_2016]",
    "We used the recorded 2016 Example-3 definition. Another assertion [@definition_2016].",
    "We used the recorded 2016 Example-3 definition;\nanother assertion [@definition_2016].",
    "We used the recorded 2016 Example-3 definition [@unrelated], then discussed [@definition_2016].",
])
def test_citation_cannot_exempt_study_numbers_or_cross_paragraphs(tmp_path, text):
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            text, evidence=EvidenceStore(tmp_path),
            enforcement_mode=EvidenceEnforcementMode.STRICT,
            literature=_literature(),
        )


def test_literal_key_year_without_run_bound_bundle_remains_untraced(tmp_path):
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            "We used the 2016 Example-3 definition [@definition_2016].",
            evidence=EvidenceStore(tmp_path), enforcement_mode=EvidenceEnforcementMode.STRICT,
        )


def test_a_supported_definition_year_does_not_hide_a_new_study_count(tmp_path):
    with pytest.raises(EvidenceEnforcementError) as caught:
        bind_numeric_values(
            "We used the 2016 Example-3 definition [@definition_2016] in 999 patients.",
            evidence=EvidenceStore(tmp_path), enforcement_mode=EvidenceEnforcementMode.STRICT,
            literature=_literature(),
        )
    assert caught.value.detail["untraced"] == ["999"]


def test_same_sentence_citation_does_not_exempt_intervening_study_numbers(tmp_path):
    with pytest.raises(EvidenceEnforcementError) as caught:
        bind_numeric_values(
            "We used the recorded 2016 Example-3 definition, with 999 patients [@definition_2016].",
            evidence=EvidenceStore(tmp_path), enforcement_mode=EvidenceEnforcementMode.STRICT,
            literature=_literature(),
        )
    assert caught.value.detail["untraced"] == ["999"]


@pytest.mark.parametrize("condition", ["excluded", "conflicting", "duplicate_key"])
def test_unusable_or_ambiguous_reference_cannot_authorize_a_year(tmp_path, condition):
    literature = _literature()
    if condition == "duplicate_key":
        literature.citations.append(literature.citations[0].model_copy(update={"year": "2015"}))
    else:
        literature.screening_decisions.append(LiteratureScreeningDecision(
            citation_key="definition_2016", source="test", disposition="exclude",
            evidence_role="definition", rationale="Not usable",
        ))
        if condition == "conflicting":
            literature.screening_decisions.append(LiteratureScreeningDecision(
                citation_key="definition_2016", source="test", disposition="include",
                evidence_role="definition", rationale="Conflicting decision",
            ))
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            "We used the 2016 Example-3 definition [@definition_2016].",
            evidence=EvidenceStore(tmp_path), enforcement_mode=EvidenceEnforcementMode.STRICT,
            literature=literature,
        )
