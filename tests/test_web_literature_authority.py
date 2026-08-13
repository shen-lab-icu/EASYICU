from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import literature_authority
from easyicu.webserver import study_contexts


@pytest.fixture(autouse=True)
def _isolated_authority_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        literature_authority,
        "_AUTHORITY_ROOT",
        tmp_path / "literature-authorities",
    )


def _study() -> dict:
    return {
        "id": "study-e1",
        "question": "Is Sepsis-3 associated with in-hospital mortality?",
        "data_source": {"path": "/typed/export", "database": "miiv"},
        "cohort": {"label": "adult ICU stays"},
        "modules": ["sepsis3_sofa1", "outcome"],
        "outcome": "In-hospital mortality",
        "primary_exposure": "Canonical Sepsis-3",
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa1",
            "covariates": [],
        },
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "time_window": {"hours": 24, "anchor": "suspected_infection_onset"},
    }


def _discovered() -> dict:
    return {
        "status": "searched",
        "search_performed": True,
        "searched_at": "2026-08-12T12:00:00+00:00",
        "queries_to_run": [
            '("Sepsis-3"[Title/Abstract] AND "mortality"[Title/Abstract])'
        ],
        "network_calls": 2,
        "source_candidates": [
            {
                "title": "The Third International Consensus Definitions for Sepsis",
                "journal": "JAMA",
                "year": 2016,
                "doi": "10.1001/jama.2016.0287",
                "pmid": "26903338",
                "evidence_quote": "Sepsis was defined as life-threatening organ dysfunction.",
                "design_excerpt": (
                    "Adult ICU patients with Sepsis-3 were followed for hospital mortality."
                ),
                "publication_types": ["Observational Study"],
                "matched_queries": [
                    '("Sepsis-3"[Title/Abstract] AND "mortality"[Title/Abstract])'
                ],
                "matched_query_strata": ["broad_icu"],
            }
        ],
    }


def test_web_literature_receipt_round_trips_to_agent_bundle() -> None:
    study = _study()
    binding = literature_authority.persist_literature_authority(
        study=study,
        discovered=_discovered(),
    )
    bundle = literature_authority.load_bound_literature(
        study={**study, "literature_authority": binding},
        research_question=study["question"],
    )

    assert bundle is not None
    assert bundle["citations"][0]["pmid"] == "26903338"
    assert bundle["citations"][0]["publication_types"] == ["Observational Study"]
    assert bundle["citations"][0]["relevance"].startswith(
        "Study-design excerpt: Adult ICU patients"
    )
    assert bundle["search_provenance"]["search_queries"]["web_pubmed"] == (
        _discovered()["queries_to_run"]
    )
    assert bundle["search_provenance"]["record_queries"] == {
        "web_pubmed_26903338": _discovered()["queries_to_run"]
    }
    assert bundle["prisma"] == {
        "identified": 1,
        "duplicates_removed": 0,
        "screened": 1,
        "eligible": 0,
        "included": 0,
    }


def test_web_literature_receipt_fails_on_scope_drift_and_tamper() -> None:
    study = _study()
    binding = literature_authority.persist_literature_authority(
        study=study,
        discovered=_discovered(),
    )
    with pytest.raises(
        literature_authority.LiteratureAuthorityError,
        match="study changed",
    ) as drift:
        literature_authority.load_bound_literature(
            study={
                **study,
                "question": "Is lactate associated with mortality?",
                "literature_authority": binding,
            },
            research_question="Is lactate associated with mortality?",
        )
    assert drift.value.code == "literature_authority_scope_mismatch"

    receipt = (
        literature_authority._AUTHORITY_ROOT / f"{binding['receipt_id']}.json"
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["citations"][0]["title"] = "Changed after binding"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(literature_authority.LiteratureAuthorityError) as tampered:
        literature_authority.load_bound_literature(
            study={**study, "literature_authority": binding},
            research_question=study["question"],
        )
    assert tampered.value.code == "literature_authority_digest_mismatch"


def test_completed_no_hits_receipt_is_authoritative_not_a_fake_search() -> None:
    study = _study()
    discovered = {
        **_discovered(),
        "status": "searched_no_hits",
        "source_candidates": [],
    }
    binding = literature_authority.persist_literature_authority(
        study=study,
        discovered=discovered,
    )
    bundle = literature_authority.load_bound_literature(
        study={**study, "literature_authority": binding},
        research_question=study["question"],
    )

    assert binding["result_count"] == 0
    assert bundle is not None
    assert bundle["citations"] == []
    assert bundle["search_provenance"]["search_conducted"] is True


def test_receipt_scope_is_the_same_owner_digest_used_by_study_context() -> None:
    study = _study()
    binding = literature_authority.persist_literature_authority(
        study=study,
        discovered=_discovered(),
    )

    assert binding["study_configuration_sha256"] == (
        study_contexts.literature_search_scope_sha256(study)
    )
