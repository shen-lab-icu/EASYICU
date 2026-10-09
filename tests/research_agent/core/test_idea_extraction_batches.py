"""Literature idea extraction runs in batches that leave receipts.

The yield scales with the corpus because extraction is batched, a small corpus
takes one batch, a failed batch is isolated and resumed from the receipts of
the others, and a tampered receipt fails closed.
"""

from __future__ import annotations

import json
from typing import Sequence

import pytest

from easyicu.research_agent.discovery.idea_mining import (
    SourceMaterial,
    extract_literature_ideas,
)
from easyicu.research_agent.literature import CitationRecord
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _batch_response(indices: Sequence[int]) -> str:
    return json.dumps(
        [
            {
                "citation_key": f"review_{index:02d}",
                "population": "adult ICU patients",
                "exposure_or_predictor": "serum lactate",
                "outcome": "in-hospital mortality",
                "rationale": "Open direction from the source.",
                "source_quote": " ".join(
                    _excerpt_material(index).source_text.split()[:6]
                ),
                "analysis_family": "association",
            }
            for index in indices
        ]
    )


def _batch_idea_llm(
    batches: Sequence[Sequence[int]],
    *,
    malformed_call: int | None = None,
) -> ScriptedMockLLMClient:
    responses = [
        (
            '[{"citation_key" "missing-colon"}]'
            if malformed_call == call_index
            else _batch_response(indices)
        )
        for call_index, indices in enumerate(batches, start=1)
    ]
    client = ScriptedMockLLMClient(responses)
    client.batch_sizes = [len(indices) for indices in batches]
    return client


def _excerpt_material(idx: int) -> SourceMaterial:
    return SourceMaterial(
        citation=CitationRecord(
            key=f"review_{idx:02d}",
            title=f"ICU review {idx}",
            year="2026",
            venue="Critical Care",
        ),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            f"The authors of review {idx} note that an unresolved question "
            "remains for future study in critically ill patients."
        ),
    )


def test_extract_literature_ideas_batches_so_yield_scales_with_corpus() -> None:
    # 7 articles with batch_size=3 must produce 3 calls (3+3+1) and 7 ideas,
    # proving the corpus is fully processed rather than capped by one call.
    materials = [_excerpt_material(i) for i in range(7)]
    llm = _batch_idea_llm([[0, 1, 2], [3, 4, 5], [6]])

    candidates = extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:batch",
        llm=llm,
        batch_size=3,
    )

    assert len(llm.calls) == 3
    assert llm.batch_sizes == [3, 3, 1]
    assert len(candidates) == 7
    assert {c.citation_key for c in candidates} == {f"review_{i:02d}" for i in range(7)}


def test_extract_literature_ideas_single_batch_when_corpus_small() -> None:
    materials = [_excerpt_material(i) for i in range(2)]
    llm = _batch_idea_llm([[0, 1]])

    candidates = extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:batch",
        llm=llm,
        batch_size=6,
    )

    assert len(llm.calls) == 1
    assert len(candidates) == 2


def test_extraction_batch_receipts_isolate_failure_and_resume_only_failed_batch(
    tmp_path,
) -> None:
    materials = [_excerpt_material(i) for i in range(6)]
    receipt_dir = tmp_path / "receipts"
    dropped: list[list[str]] = []
    first_llm = _batch_idea_llm([[0, 1], [2, 3], [4, 5]], malformed_call=2)

    first = extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:receipt-resume",
        llm=first_llm,
        batch_size=2,
        malformed_batch_policy="skip",
        dropped_malformed_batches=dropped,
        batch_receipt_dir=receipt_dir,
    )

    assert len(first_llm.calls) == 3
    assert {idea.citation_key for idea in first} == {
        "review_00",
        "review_01",
        "review_04",
        "review_05",
    }
    assert dropped == [["review_02", "review_03"]]
    assert len(list(receipt_dir.glob("*_parsed_*.json"))) == 2
    assert len(list(receipt_dir.glob("*_malformed_*.json"))) == 1

    resumed_llm = _batch_idea_llm([[2, 3]])
    resumed = extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:receipt-resume",
        llm=resumed_llm,
        batch_size=2,
        malformed_batch_policy="skip",
        dropped_malformed_batches=[],
        batch_receipt_dir=receipt_dir,
    )

    assert len(resumed_llm.calls) == 1
    assert resumed_llm.batch_sizes == [2]
    assert {idea.citation_key for idea in resumed} == {
        f"review_{idx:02d}" for idx in range(6)
    }


def test_extraction_batch_receipt_tampering_fails_closed(tmp_path) -> None:
    materials = [_excerpt_material(0)]
    receipt_dir = tmp_path / "receipts"
    first_llm = _batch_idea_llm([[0]])
    extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:receipt-tamper",
        llm=first_llm,
        batch_receipt_dir=receipt_dir,
    )
    receipt_path = next(receipt_dir.glob("*_parsed_*.json"))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["raw_response"] += " "
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="receipt digest mismatch"):
        extract_literature_ideas(
            materials=materials,
            source_snapshot_id="source-snapshot/sha256:receipt-tamper",
            llm=_batch_idea_llm([[0]]),
            batch_receipt_dir=receipt_dir,
        )
