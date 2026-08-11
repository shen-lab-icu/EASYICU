from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver.ideas.prior_art_receipt import (
    PriorArtReceiptError,
    build_prior_art_binding,
    load_bound_prior_art_literature,
)


def _write_receipt(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prior_art": {
                    "status": "searched",
                    "search_performed": True,
                    "searched_at": "2026-08-11T12:00:00+00:00",
                    "results": [
                        {
                            "pmid": "26903338",
                            "title": "The Third International Consensus Definitions for Sepsis and Septic Shock",
                            "journal": "JAMA",
                            "pubdate": "2016 Feb 23",
                            "query": '"Sepsis-3"[Title/Abstract] AND mortality[Title/Abstract]',
                        }
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_prior_art_receipt_binding_projects_exact_pubmed_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prior_art_check.json"
    _write_receipt(path)

    binding = build_prior_art_binding(path)
    assert binding is not None
    assert binding["prior_art_result_count"] == 1
    assert binding["prior_art_searched_at"] == "2026-08-11T12:00:00+00:00"

    bundle = load_bound_prior_art_literature(
        path,
        binding=binding,
        research_question="Is Sepsis-3 associated with mortality?",
    )

    assert bundle["search_provenance"]["search_conducted"] is True
    assert bundle["search_provenance"]["searched_at"] == ("2026-08-11T12:00:00+00:00")
    assert bundle["search_provenance"]["sources_enabled"] == ["idea_mining_pubmed"]
    assert bundle["citations"][0]["key"] == "idea_pubmed_26903338"
    assert bundle["citations"][0]["url"] == (
        "https://pubmed.ncbi.nlm.nih.gov/26903338/"
    )


def test_prior_art_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "prior_art_check.json"
    _write_receipt(path)
    binding = build_prior_art_binding(path)
    assert binding is not None
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(PriorArtReceiptError) as exc_info:
        load_bound_prior_art_literature(
            path,
            binding=binding,
            research_question="Question",
        )

    assert exc_info.value.code == "prior_art_binding_digest_mismatch"


def test_prior_art_receipt_without_search_timestamp_is_not_bindable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prior_art_check.json"
    _write_receipt(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prior_art"].pop("searched_at")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert build_prior_art_binding(path) is None
