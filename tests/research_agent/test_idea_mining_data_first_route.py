"""Integration contracts for the standard data-first Idea Mining route."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.discovery.idea_mining import (
    freeze_source_snapshot,
    run_idea_mining_dry_run,
)
from easyicu.research_agent.discovery.idea_mining_data_first_route import (
    run_data_first_idea_mining_dry_run,
)
from easyicu.research_agent.discovery.idea_mining_schema import (
    IdeaMiningError,
    LiteratureIdeaCandidate,
    SourceMaterial,
)
from easyicu.research_agent.literature import CitationRecord


def _all_full(*, concepts, databases):
    del concepts
    return {"cross_database_feasibility": {db: "full" for db in databases}}


def _joint_feasible(*, concepts, **kwargs):
    del kwargs
    return {
        concepts[0]: {
            "joint_fraction_complete": 0.95,
            "n_joint_complete": 950,
            "denominator_n": 1000,
            "source": "test_prepared_cohort",
            "note": "fixture",
            "predictor_contrast_fraction": 0.4,
        }
    }


class _SparsePriorArt:
    def search_prior_art(self, query, *, max_results, idea=None):
        del query, max_results, idea
        return {"hit_count": 2, "pmids": [], "top_hits": [], "search_ok": True}


def test_data_first_route_emits_standard_ledger_without_llm(tmp_path):
    data_path = tmp_path / "prepared.parquet"
    data_path.write_bytes(b"frozen test cohort")

    result = run_data_first_idea_mining_dry_run(
        predictor_concepts=["lact"],
        outcome_concepts=["death"],
        available_concepts=["lact", "death"],
        outcome_determinability={"death": {"outcome": "death", "status": "known_0_1"}},
        output_dir=tmp_path / "out",
        data_path=data_path,
        prior_art_search_client=_SparsePriorArt(),
        databases=["db1", "db2", "db3", "db4"],
        cross_database_feasibility_fn=_all_full,
        feasibility_probe=_joint_feasible,
    )

    triage = json.loads((tmp_path / "out" / "candidate_triage_report.json").read_text())
    assert result.triage_report_path.endswith("candidate_triage_report.json")
    assert len(triage["discovery_ledger"]) == 1
    row = triage["discovery_ledger"][0]
    assert row["resolved_predictor_concept"] == "lact"
    assert row["resolved_outcome_concept"] == "death"
    assert row["requires_human_confirmation"] is True
    assert row["go_no_go"] in {"hold", "recommend"}

    route = json.loads(
        (tmp_path / "out" / "data_first_route_manifest.json").read_text()
    )
    assert route["prepared_data_sha256"]
    assert route["harmonized_databases_considered"] == [
        "db1",
        "db2",
        "db3",
        "db4",
    ]
    assert route["data_first_candidates"][0]["predictor"] == "lact"
    shortlist = json.loads(
        (tmp_path / "out" / "data_first_review_shortlist.json").read_text()
    )
    assert shortlist["candidates"][0]["review_route"] == (
        "cross_database_external_validation"
    )
    assert shortlist["candidates"][0]["review_candidate_id"].startswith("reviewidea_")
    assert shortlist["candidates"][0]["origin_executable_candidate_id"]
    assert shortlist["candidates"][0]["paper_authorized"] is False


class _MustNotCallLLM:
    def complete(self, *args, **kwargs):
        raise AssertionError("precomputed route called the LLM")


def test_precomputed_idea_must_bind_frozen_snapshot_and_quote(tmp_path):
    material = SourceMaterial(
        citation=CitationRecord(key="profile", title="Profile", year="2026"),
        source_adapter_level="user_supplied_excerpt",
        source_text="verbatim profile evidence",
        source_text_role="data_profile_evidence",
    )
    snapshot = freeze_source_snapshot([material])
    bad = LiteratureIdeaCandidate(
        source_snapshot_id=snapshot.source_snapshot_id,
        citation_key="profile",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU stays",
        exposure_or_predictor="lact",
        outcome="death",
        rationale="data-first candidate",
        source_quote="invented non-verbatim evidence",
    )

    with pytest.raises(IdeaMiningError, match="not verbatim"):
        run_idea_mining_dry_run(
            materials=[material],
            precomputed_literature_ideas=[bad],
            llm=_MustNotCallLLM(),
            available_concepts=["lact", "death"],
            outcome_determinability={
                "death": {"outcome": "death", "status": "known_0_1"}
            },
            output_dir=tmp_path / "bad",
        )


def test_prior_art_search_error_cannot_become_apparent_gap():
    from easyicu.research_agent.discovery.idea_mining_priorart import (
        assess_prior_art_for_idea,
    )

    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:test",
        citation_key="profile",
        source_adapter_level="metadata_only",
        population="adult ICU stays",
        exposure_or_predictor="lactate",
        outcome="mortality",
        rationale="test",
        source_quote="test",
    )

    class BrokenSearch:
        def search_prior_art(self, *args, **kwargs):
            return {"hit_count": 0, "search_error": "network unavailable"}

    assessment = assess_prior_art_for_idea(idea, search_client=BrokenSearch())
    assert assessment.novelty_label == "crowded_but_differentiable"
    assert "prior-art search unavailable" in assessment.same_topic_screen_status


def test_cross_database_family_does_not_hide_single_database_same_topic_work():
    from easyicu.research_agent.discovery.idea_mining_priorart import (
        build_prior_art_queries,
    )

    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:test",
        citation_key="profile",
        source_adapter_level="metadata_only",
        population="adult ICU stays",
        exposure_or_predictor="diastolic shock index",
        outcome="hospital mortality",
        rationale="cross-database transportability candidate",
        source_quote="test",
        analysis_family="cross_database_replication",
        exposure_literature_aliases=["DSI"],
        outcome_literature_aliases=["in-hospital mortality"],
    )

    queries = build_prior_art_queries(idea)
    assert "cross_database_replication" not in queries["exact"]
    assert "Cross-database replication / transportability" not in queries["exact"]
    assert '"diastolic shock index"[Title/Abstract]' in queries["exact"]
    assert '"hospital mortality"[Title/Abstract]' in queries["exact"]
    assert "DSI[Title/Abstract]" in queries["exact"]
    assert '"in-hospital mortality"[Title/Abstract]' in queries["exact"]


def test_exact_query_includes_host_curated_spelling_variants():
    from easyicu.research_agent.discovery.idea_mining_priorart import (
        build_prior_art_queries,
    )

    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:test",
        citation_key="profile",
        source_adapter_level="metadata_only",
        population="adult ICU stays",
        exposure_or_predictor="calcium corrected for albumin",
        outcome="hospital mortality",
        rationale="measurement-bias candidate",
        source_quote="test",
        exposure_literature_aliases=[
            "albumin-corrected calcium",
            "albumin corrected calcium",
        ],
        outcome_literature_aliases=["in-hospital mortality"],
    )

    exact = build_prior_art_queries(idea)["exact"]
    assert '"albumin-corrected calcium"[Title/Abstract]' in exact
    assert '"albumin corrected calcium"[Title/Abstract]' in exact
    assert '"in-hospital mortality"[Title/Abstract]' in exact


def test_data_first_shortlist_separates_validation_from_measurement_audit(tmp_path):
    data_path = tmp_path / "prepared.parquet"
    data_path.write_bytes(b"frozen test cohort")

    def mixed_probe(*, concepts, **kwargs):
        del kwargs
        complete = concepts[0] == "well_observed"
        n = 950 if complete else 500
        return {
            concepts[0]: {
                "joint_fraction_complete": n / 1000,
                "n_joint_complete": n,
                "denominator_n": 1000,
                "source": "test_prepared_cohort",
                "note": "fixture",
                "predictor_contrast_fraction": 0.4,
            }
        }

    class AssociationPriorArt:
        def search_prior_art(self, query, *, max_results, idea=None):
            del query, max_results
            if idea is not None and idea.exposure_or_predictor == "partly_observed":
                return {
                    "hit_count": 1,
                    "top_hits": [
                        {
                            "pmid": "99",
                            "title": "partly observed and death",
                            "abstract": "The association was evaluated in both populations.",
                            "direct_same_topic": True,
                        }
                    ],
                    "search_ok": True,
                }
            return _SparsePriorArt().search_prior_art(
                "ignored", max_results=1, idea=idea
            )

    run_data_first_idea_mining_dry_run(
        predictor_concepts=["well_observed", "partly_observed"],
        outcome_concepts=["death"],
        available_concepts=["well_observed", "partly_observed", "death"],
        outcome_determinability={"death": {"outcome": "death", "status": "known_0_1"}},
        output_dir=tmp_path / "out",
        data_path=data_path,
        prior_art_search_client=AssociationPriorArt(),
        databases=["db1", "db2", "db3", "db4"],
        cross_database_feasibility_fn=_all_full,
        feasibility_probe=mixed_probe,
    )

    shortlist = json.loads(
        (tmp_path / "out" / "data_first_review_shortlist.json").read_text()
    )
    assert [item["review_route"] for item in shortlist["candidates"]] == [
        "cross_database_external_validation",
        "cross_database_measurement_bias_audit",
    ]
    assert all(
        item["requires_human_confirmation"] and not item["paper_authorized"]
        for item in shortlist["candidates"]
    )
    audit = shortlist["candidates"][1]
    assert audit["exact_same_topic_hit_count"] > 0
    assert audit["candidate_topic"] == (
        "cross-database measurement/source-status audit of partly_observed"
    )
    assert audit["origin_candidate_topic"].startswith("partly_observed -> death")
    assert audit["review_candidate_id"] != audit["origin_executable_candidate_id"]
    assert audit["route_prior_art"]["search_ok"] is True
    assert "partly_observed" in audit["route_prior_art"]["query"]
    assert "measurement availability" in audit["route_prior_art"]["query"]


def test_measurement_audit_route_fails_closed_when_its_prior_art_screen_fails(tmp_path):
    data_path = tmp_path / "prepared.parquet"
    data_path.write_bytes(b"frozen test cohort")

    def partly_observed_probe(*, concepts, **kwargs):
        del kwargs
        return {
            concepts[0]: {
                "joint_fraction_complete": 0.5,
                "n_joint_complete": 500,
                "denominator_n": 1000,
                "source": "test_prepared_cohort",
                "note": "fixture",
                "predictor_contrast_fraction": 0.4,
            }
        }

    class RouteSearchFails:
        def search_prior_art(self, query, *, max_results, idea=None):
            del query, max_results
            if idea is None:
                return None
            return {
                "hit_count": 2,
                "pmids": [],
                "top_hits": [],
                "search_ok": True,
            }

    run_data_first_idea_mining_dry_run(
        predictor_concepts=["partly_observed"],
        outcome_concepts=["death"],
        available_concepts=["partly_observed", "death"],
        outcome_determinability={"death": {"outcome": "death", "status": "known_0_1"}},
        output_dir=tmp_path / "out",
        data_path=data_path,
        prior_art_search_client=RouteSearchFails(),
        databases=["db1", "db2", "db3", "db4"],
        cross_database_feasibility_fn=_all_full,
        feasibility_probe=partly_observed_probe,
    )

    shortlist = json.loads(
        (tmp_path / "out" / "data_first_review_shortlist.json").read_text()
    )
    assert shortlist["candidates"] == []
