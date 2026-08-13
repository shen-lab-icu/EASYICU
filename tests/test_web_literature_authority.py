from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.planning.preplan_literature import (
    prepare_preplan_literature,
)
from easyicu.research_agent.schema import ResearchContext
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
        "revision": 4,
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
                "direct_comparator_screen": {
                    "citation_key": "web_pubmed_26903338",
                    "source": "web_pubmed_retrieval",
                    "disposition": "exclude",
                    "evidence_role": "related_context",
                    "rationale": (
                        "The source-backed excerpt does not establish the declared "
                        "exposure as the studied variable."
                    ),
                    "query": None,
                    "population_match": True,
                    "exposure_match": False,
                    "outcome_match": True,
                    "design_excerpt_available": True,
                    "publication_type_eligible": True,
                },
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
    assert binding["study_context_id"] == study["id"]
    assert binding["study_context_revision"] == study["revision"]
    receipt = json.loads(
        (
            literature_authority._AUTHORITY_ROOT
            / f"{binding['receipt_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["citations"][0]["direct_comparator_screen"]["disposition"] == (
        "exclude"
    )
    assert "direct_comparator_screen" not in bundle["citations"][0]
    trace = bundle["authority_trace"]
    assert trace == {
        "schema_version": literature_authority.LITERATURE_AUTHORITY_SCHEMA_VERSION,
        "receipt_id": binding["receipt_id"],
        "receipt_sha256": binding["receipt_sha256"],
        "study_context_id": study["id"],
        "study_context_revision": study["revision"],
        "retrieval_scope_sha256": binding["study_configuration_sha256"],
    }


def test_v2_receipt_preserves_historical_scope_without_fake_source_trace() -> None:
    study = _study()
    schema_version = "easyicu.web-literature-authority/2"
    query = _discovered()["queries_to_run"][0]
    searched_at = _discovered()["searched_at"]
    scope_sha256 = study_contexts.literature_search_scope_sha256(
        study,
        schema_version=schema_version,
    )
    receipt_id = f"lit_{'d' * 24}"
    payload = {
        "schema_version": schema_version,
        "searched_at": searched_at,
        "status": "searched",
        "study_configuration_sha256": scope_sha256,
        "search": {
            "source": "web_pubmed",
            "queries": [query],
            "query_strata": [],
            "network_calls": 2,
            "result_count": 1,
        },
        "citations": [
            {
                "key": "web_pubmed_26903338",
                "title": "The Third International Consensus Definitions for Sepsis",
                "year": "2016",
                "venue": "JAMA",
                "relevance": "Study-design excerpt: adult ICU mortality follow-up.",
                "doi": "10.1001/jama.2016.0287",
                "url": "https://pubmed.ncbi.nlm.nih.gov/26903338/",
                "pmid": "26903338",
                "publication_types": ["Observational Study"],
                "matched_queries": [query],
                "matched_query_strata": ["broad_icu"],
            }
        ],
        "privacy": {
            "patient_rows_recorded": False,
            "full_text_recorded": False,
            "host_paths_recorded": False,
            "external_llm_calls": 0,
        },
        "receipt_id": receipt_id,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt_path = literature_authority._AUTHORITY_ROOT / f"{receipt_id}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(raw)
    binding = {
        "schema_version": schema_version,
        "receipt_id": receipt_id,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "searched",
        "result_count": 1,
        "searched_at": searched_at,
        "study_configuration_sha256": scope_sha256,
    }

    seed = literature_authority.load_bound_literature(
        study={**study, "literature_authority": binding},
        research_question=study["question"],
    )

    assert seed is not None
    assert seed["search_provenance"]["search_conducted"] is True
    assert seed["citations"][0]["pmid"] == "26903338"
    assert "authority_trace" not in seed
    assert LiteratureBundle.model_validate(seed).authority_trace is None
    with pytest.raises(literature_authority.LiteratureAuthorityError) as changed:
        literature_authority.load_bound_literature(
            study={
                **study,
                "time_window": {"hours": 48, "anchor": "ICU admission"},
                "literature_authority": binding,
            },
            research_question=study["question"],
        )
    assert changed.value.code == "literature_authority_scope_mismatch"


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

    with pytest.raises(literature_authority.LiteratureAuthorityError) as rebound:
        literature_authority.load_bound_literature(
            study={
                **study,
                "literature_authority": {
                    **binding,
                    "study_context_revision": binding["study_context_revision"] + 1,
                },
            },
            research_question=study["question"],
        )
    assert rebound.value.code == "literature_authority_study_coordinate_mismatch"

    with pytest.raises(literature_authority.LiteratureAuthorityError) as copied:
        literature_authority.load_bound_literature(
            study={
                **study,
                "id": "study-other",
                "literature_authority": binding,
            },
            research_question=study["question"],
        )
    assert copied.value.code == "literature_authority_study_coordinate_mismatch"

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


def test_planning_changes_preserve_retrieval_but_topic_changes_fail_closed() -> None:
    study = {**_study(), "revision": 7}
    binding = literature_authority.persist_literature_authority(
        study=study,
        discovered=_discovered(),
    )

    replanned = {
        **study,
        "revision": 11,
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "covariates": ["age", "sex"],
        "time_window": {"hours": 48, "anchor": "ICU admission"},
        "literature_authority": binding,
    }
    bundle = literature_authority.load_bound_literature(
        study=replanned,
        research_question=replanned["question"],
    )

    assert bundle is not None
    assert bundle["search_provenance"]["search_conducted"] is True
    assert bundle["authority_trace"]["study_context_revision"] == 7
    with pytest.raises(literature_authority.LiteratureAuthorityError) as changed:
        literature_authority.load_bound_literature(
            study={
                **replanned,
                "execution_concepts": {
                    **replanned["execution_concepts"],
                    "primary_exposure": "lactate",
                },
            },
            research_question=replanned["question"],
        )
    assert changed.value.code == "literature_authority_scope_mismatch"


def test_verified_web_receipt_becomes_the_persisted_preplan_bundle(
    tmp_path: Path,
) -> None:
    study = {**_study(), "revision": 7}
    binding = literature_authority.persist_literature_authority(
        study=study,
        discovered=_discovered(),
    )
    seed = literature_authority.load_bound_literature(
        study={
            **study,
            "analysis_design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": "cluster_robust",
                "cluster_unit": "patient",
            },
            "literature_authority": binding,
        },
        research_question=study["question"],
    )
    assert seed is not None
    assert seed["screening_decisions"] == []
    context = ResearchContext.model_validate(
        {
            "research_question": study["question"],
            "cohort": {
                "cohort_name": "adult ICU stays",
                "database": "miiv",
                "n_stays": 100,
                "n_patients": 80,
                "inclusion_criteria": ["adult ICU stays"],
            },
            "variables": [
                {
                    "name": "sep3_sofa1",
                    "dtype": "int64",
                    "role": "composite_score",
                    "source_concept": "sep3_sofa1",
                    "description": "Sepsis-3 exposure",
                },
                {
                    "name": "death",
                    "dtype": "int64",
                    "role": "outcome",
                    "source_concept": "death",
                    "description": "in-hospital mortality",
                },
            ],
            "primary_exposure": "sep3_sofa1",
            "target_outcome": "death",
        }
    )

    class Evidence:
        def __init__(self) -> None:
            self.rows: dict[str, Path] = {}

        def get(self, evidence_id: str) -> None:
            return None

        def register_file(
            self, *, evidence_id: str, source_path: Path, **_: object
        ) -> None:
            self.rows[evidence_id] = source_path

    evidence = Evidence()
    bundle = prepare_preplan_literature(
        context=context,
        run_dir=tmp_path,
        evidence=evidence,
        enable_pubmed=False,
        pubmed_email=None,
        pubmed_api_key=None,
        enable_tavily=False,
        tavily_api_key=None,
        tavily_retmax=5,
        tavily_include_domains=(),
        bound_seed=LiteratureBundle.model_validate(seed),
    )
    saved = LiteratureBundle.model_validate_json(
        (tmp_path / "preplan_literature_bundle.json").read_text(encoding="utf-8")
    )

    assert bundle.search_provenance is not None
    assert bundle.search_provenance.search_conducted is True
    assert bundle.authority_trace is not None
    assert bundle.authority_trace.receipt_id == binding["receipt_id"]
    assert saved.authority_trace == bundle.authority_trace
    assert set(bundle.authority_trace.model_dump(mode="json")) == {
        "schema_version",
        "receipt_id",
        "receipt_sha256",
        "study_context_id",
        "study_context_revision",
        "retrieval_scope_sha256",
    }
    assert "preplan_literature_bundle" in evidence.rows
    # Retrieval candidates are re-screened under the sealed ResearchContext;
    # the upstream Web receipt itself supplied no comparator/novelty decision.
    assert bundle.screening_decisions
