"""STRICT binding must retain the citations emitted by claim expansion."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.reporting.manuscript_post import (
    _cited_evidence_ids,
    _numeric_sentence_context,
    bind_numeric_values,
)


def _registered_sources(tmp_path, value, *, duplicate_fields=True):
    store = EvidenceStore(tmp_path)
    records = []
    for step, number in (("target", value), ("alternative", value), ("foreign", 9753)):
        # Repeated fields in one immutable summary are not competing sources.
        # Another summary with the same value must still require disambiguation.
        summary = {"field_a": number}
        if duplicate_fields:
            summary["field_b"] = number
        record = store.register_json(
            kind="statistic", description="Registered aggregate result",
            payload=summary, filename=f"{step}.json", evidence_id=step,
            produced_by_step=step,
        )
        store.register_step_summary_numerics(
            step_id=step, evidence_id=record.evidence_id, summary=summary,
        )
        records.append({
            "step_id": step, "status": "ok", "step_summary": summary,
            "step_summary_evidence_id": record.evidence_id,
            "evidence_ids": [record.evidence_id],
        })
    return store, records


def _bind(store, records, text):
    return bind_numeric_values(
        text, evidence=store, per_step_records=records,
        enforcement_mode=EvidenceEnforcementMode.STRICT,
    )


@pytest.mark.parametrize("value,display", [(2468, "2468"), (17.125, "17.125%")])
@pytest.mark.parametrize("citation", [
    "{evidence:target}",
    '[result](evidence/target__target.json "sha256=sealed")',
])
@pytest.mark.parametrize("spacing", [" ", "\n", "\r\n"])
def test_trailing_citation_disambiguates_verified_same_value_fields(
    tmp_path, value, display, citation, spacing,
):
    store, records = _registered_sources(tmp_path, value)
    _, bindings, untraced = _bind(
        store, records, f"The registered result was {display}.{spacing}{citation}",
    )
    assert not untraced
    assert len(bindings) == 1
    assert next(iter(bindings.values())).evidence_id == "target"


@pytest.mark.parametrize("citation", [
    "{evidence:target}",
    '[result](evidence/target__target.json "sha256=sealed")',
])
@pytest.mark.parametrize("separator", ["\n\n", "\n \t\n", "\r\n\r\n"])
@pytest.mark.parametrize("terminal", [".", ""])
def test_next_paragraph_cannot_disambiguate_same_value_sources(
    tmp_path, citation, separator, terminal,
):
    store, records = _registered_sources(tmp_path, 2468)
    text = f"The registered result was 2468{terminal}{separator}{citation} Another result follows."
    start = text.index("2468")
    context = _numeric_sentence_context(text, start=start, end=start + 4)
    assert not _cited_evidence_ids(context)
    with pytest.raises(EvidenceEnforcementError):
        _bind(store, records, text)


@pytest.mark.parametrize("citation", [
    "{evidence:alternative}",
    '[result](evidence/alternative__alternative.json "sha256=sealed")',
])
def test_citation_chain_cannot_replace_owner_from_next_paragraph(tmp_path, citation):
    store, records = _registered_sources(tmp_path, 2468)
    text = (
        f"The registered result was 2468. {citation}\n\n"
        "{evidence:target} Another result follows."
    )
    start = text.index("2468")
    context = _numeric_sentence_context(text, start=start, end=start + 4)
    assert _cited_evidence_ids(context) == {"alternative"}
    _, bindings, untraced = _bind(store, records, text)
    assert not untraced
    assert next(iter(bindings.values())).evidence_id == "alternative"


@pytest.mark.parametrize("text", [
    "The registered result was 2468.",  # Ambiguous without an owner.
    "The registered result was 2468. {evidence:foreign}",
    "The registered result was 9753. {evidence:target}",
    "The registered result was 9999. {evidence:target}",
    "The registered result was 2468. Another result follows. {evidence:target}",
    "The odds ratio was 2468. {evidence:target}",  # A count is not an OR.
])
def test_trailing_placeholder_does_not_relax_strict_provenance(tmp_path, text):
    store, records = _registered_sources(tmp_path, 2468)
    with pytest.raises(EvidenceEnforcementError):
        _bind(store, records, text)


def test_trailing_placeholder_rejects_an_otherwise_unique_foreign_number(tmp_path):
    store, records = _registered_sources(tmp_path, 2468, duplicate_fields=False)
    # 9753 exists only in the foreign record. Losing the target citation would
    # bind it successfully, so ambiguity cannot mask a missing scoping guard.
    with pytest.raises(EvidenceEnforcementError):
        _bind(store, records, "The registered result was 9753. {evidence:target}")


def test_mixed_citation_run_stops_before_following_prose():
    text = (
        "The registered result was 2468. {evidence:first} "
        '[second](evidence/second__summary.json "sha256=sealed") '
        "{evidence:third} Another result follows. {evidence:foreign}"
    )
    start = text.index("2468")
    context = _numeric_sentence_context(text, start=start, end=start + 4)
    assert _cited_evidence_ids(context) == {"first", "second", "third"}
    assert "Another result" not in context


def test_trailing_placeholder_cannot_admit_tampered_source(tmp_path):
    store, records = _registered_sources(tmp_path, 2468)
    (tmp_path / store.get("target").relative_path).write_text("{}")
    with pytest.raises(EvidenceEnforcementError):
        _bind(store, records, "The registered result was 2468. {evidence:target}")
