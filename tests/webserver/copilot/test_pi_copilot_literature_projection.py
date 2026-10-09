"""Literature reaches a study only through its bound, bounded projection.

The plan's literature projection keeps only bundle-bound keys from the
digest-verified final plan, refuses an ineligible comparator, and stays
bounded; the search tool needs its own one-turn network grant and typed study
concepts, and routes an unplanned study to the planner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver.literature_projection import (
    literature_source_resource,
    project_run_literature,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.literature_tool_projection import (
    compile_literature_tool_projection,
)
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
from tests.webserver.copilot.research_workflow_fixtures import (
    complete_study as _complete_study,
)


def test_curated_literature_projection_is_honest_and_does_not_backfill_plan_links() -> (
    None
):
    payload = project_run_literature(
        run_id="run-literature-1",
        bundle={
            "research_question": "Does an ICU exposure predict mortality?",
            "citations": [
                {
                    "key": "strobe_2007",
                    "title": "STROBE statement",
                    "year": "2007",
                    "venue": "BMJ",
                    "relevance": "Observational reporting guidance.",
                }
            ],
            "prisma": None,
            "search_provenance": {
                "curated_seed_count": 1,
                "sources_enabled": [],
                "sources_returning": [],
                "search_conducted": False,
                "note": "No retrieval source was enabled.",
            },
        },
        plan={
            "steps": [
                {
                    "step_id": "01_primary",
                    "planned_analysis_role": "primary",
                    "intent": "Fit the prespecified model.",
                }
            ]
        },
    )

    assert payload["status"] == "curated_only"
    assert payload["search"]["search_conducted"] is False
    assert payload["search"]["prisma"] is None
    assert payload["mapping_status"] == "not_bound"
    assert payload["step_citation_map"][0]["citation_keys"] == []
    assert payload["integrity"]["patient_rows_returned"] is False


def test_plan_literature_projection_keeps_only_bundle_bound_keys() -> None:
    payload = project_run_literature(
        run_id="run-literature-2",
        bundle={
            "citations": [
                {
                    "key": "method_key",
                    "title": "A real method paper",
                    "pmid": "12345",
                }
            ],
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["pubmed"],
                "sources_returning": ["pubmed"],
                "search_conducted": True,
                "searched_at": "2026-08-11T12:00:00+00:00",
                "search_queries": {"pubmed": ["ICU AND exposure AND outcome"]},
            },
            "screening_decisions": [
                {
                    "citation_key": "method_key",
                    "source": "pubmed",
                    "disposition": "include",
                    "evidence_role": "direct_comparator",
                    "rationale": "P/E/O matched in the retained abstract.",
                    "population_match": True,
                    "exposure_match": True,
                    "outcome_match": True,
                    "design_excerpt_available": True,
                }
            ],
        },
        plan={
            "steps": [
                {
                    "step_id": "primary",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the primary association.",
                    "literature_citation_keys": ["method_key", "invented_key"],
                    "literature_design_bindings": [
                        {
                            "citation_key": "method_key",
                            "design_elements": ["estimand"],
                            "application": "Use the article to prespecify the estimand.",
                        }
                    ],
                },
                {
                    "step_id": "render",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the already-bound estimate.",
                },
            ]
        },
    )

    assert payload["status"] == "searched"
    assert payload["mapping_status"] == "partial"
    assert payload["scientific_mapping_status"] == "complete"
    assert payload["scientific_plan_step_count"] == 1
    assert payload["scientific_mapped_step_count"] == 1
    assert payload["search"]["searched_at"] == "2026-08-11T12:00:00+00:00"
    assert payload["search"]["queries"]["pubmed"] == ["ICU AND exposure AND outcome"]
    assert payload["direct_comparator_keys"] == ["method_key"]
    assert payload["citations"][0]["screening"]["population_match"] is True
    assert payload["citation_year_range"] == {"oldest": None, "newest": None}
    assert payload["step_citation_map"][0]["citation_keys"] == ["method_key"]
    assert (
        payload["step_citation_map"][0]["citation_bindings"][0]["evidence_role"]
        == "direct_comparator"
    )
    assert payload["integrity"]["unknown_citation_keys_removed"] == ["invented_key"]
    assert (
        payload["citations"][0]["source_url"]
        == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    )


def test_web_projection_refuses_ineligible_publication_type_as_comparator() -> None:
    payload = project_run_literature(
        run_id="run-literature-review",
        bundle={
            "citations": [
                {
                    "key": "review_key",
                    "title": "Systematic review of the same ICU question",
                    "year": "2025",
                    "pmid": "12346",
                    "publication_types": ["Systematic Review", "Review"],
                }
            ],
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["pubmed"],
                "sources_returning": ["pubmed"],
                "search_conducted": True,
                "search_queries": {"pubmed": ["ICU question"]},
            },
            "screening_decisions": [
                {
                    "citation_key": "review_key",
                    "source": "pubmed",
                    "disposition": "include",
                    "evidence_role": "direct_comparator",
                    "rationale": "Legacy decision before publication-type gate.",
                    "population_match": True,
                    "exposure_match": True,
                    "outcome_match": True,
                    "design_excerpt_available": True,
                    "publication_type_eligible": False,
                }
            ],
        },
        plan={"steps": []},
    )

    assert payload["direct_comparator_count"] == 0
    assert payload["citations"][0]["publication_types"] == [
        "Systematic Review",
        "Review",
    ]
    assert payload["citations"][0]["screening"]["publication_type_eligible"] is False


def test_loaded_literature_projection_uses_digest_verified_final_plan(
    tmp_path: Path,
) -> None:
    import hashlib

    from easyicu.webserver.literature_projection import load_run_literature_projection

    bundle = {
        "citations": [{"key": "method_key", "title": "A method paper"}],
        "search_provenance": {"search_conducted": False},
    }
    (tmp_path / "preplan_literature_bundle.json").write_text(
        json.dumps(bundle), encoding="utf-8"
    )
    initial = {
        "steps": [
            {
                "step_id": "primary",
                "planned_analysis_role": "primary",
                "literature_citation_keys": ["method_key"],
            }
        ]
    }
    (tmp_path / "analysis_plan.json").write_text(json.dumps(initial), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    final_path = evidence_dir / "analysis_plan_revision_2.json"
    final = {
        "steps": [
            {
                "step_id": "primary",
                "planned_analysis_role": "primary",
                "literature_citation_keys": [],
            }
        ]
    }
    final_raw = json.dumps(final).encode("utf-8")
    final_path.write_bytes(final_raw)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "current_plan_authority": {
                    "relative_path": "evidence/analysis_plan_revision_2.json",
                    "sha256": hashlib.sha256(final_raw).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    payload = load_run_literature_projection(
        run_dir=tmp_path,
        run_id="run-final-plan",
    )

    assert payload["scientific_mapping_status"] == "not_bound"
    assert payload["integrity"]["current_plan_authority_verified"] is True


def test_literature_search_tool_uses_separate_one_turn_network_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module, "_bound_context", lambda binding: _complete_study()
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        lambda body: {
            "status": "searched",
            "search_performed": True,
            "queries_to_run": ["ICU mortality"],
            "network_calls": 2,
            "source_candidates": [
                {
                    "citation_key": "paper_12345",
                    "title": "A source-backed ICU study",
                    "journal": "Critical Care",
                    "year": 2025,
                    "pmid": "12345",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                    "evidence_quote": "The abstract describes an ICU cohort.",
                }
            ],
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-literature"),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["code"] == "easyicu_literature_search_completed"
    assert result["details"]["literature_search"]["search_performed"] is True
    assert result["details"]["resource"]["kind"] == "literature_source"
    assert result["details"]["resource"]["pmid"] == "12345"
    methodology = result["details"]["literature_search"]["methodology"]
    assert methodology["schema_version"].startswith("easyicu.method_literature_pack/")
    assert len(methodology["sha256"]) == 64
    assert {
        "reporting_standard",
        "time_alignment",
        "dependence",
        "functional_form",
        "missing_data",
        "interpretation",
    } <= {row["layer"] for row in methodology["cards"]}
    assert any(row.get("pmid") == "17938396" for row in methodology["sources"])
    consumed = tool_module.execute_tool("easyicu_search_literature", {}, context)
    assert consumed["code"] == "pi_action_grant_consumed"


def test_direct_study_literature_search_requires_typed_exposure_and_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study.update(
        {
            "question": "Study an ICU exposure and outcome.",
            "primary_exposure": "",
            "outcome": "",
            "execution_concepts": {},
        }
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    search_called = False

    def discover(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal search_called
        search_called = True
        return {}

    monkeypatch.setattr(tool_module.idea_mining, "discover_literature", discover)
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-incomplete-literature"),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["status"] == "blocked"
    assert result["code"] == "literature_study_scope_incomplete"
    assert search_called is False
    assert "literature" in context.allowed_actions


def test_blocked_literature_routes_an_unplanned_study_to_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal here must name the owning next step, not dead-end the turn.

    Before a plan exists the exposure and outcome are empty *by design* -- the
    Planner chooses them.  A bare block made Pi conclude the plan itself was
    impossible and stop without ever calling the Planner, so the receipt has to
    say where the authority actually lives.
    """

    study = _complete_study()
    study.update(
        {
            "question": "Study an ICU exposure and outcome.",
            "primary_exposure": "",
            "outcome": "",
            "execution_concepts": {},
        }
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        lambda body: pytest.fail("literature must not run before a plan"),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-literature-routes-to-plan"),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["status"] == "blocked"
    assert result["code"] == "literature_study_scope_incomplete"
    assert result["details"]["plan_generation_ready"] is True
    assert result["details"]["next_action_code"] == "provider_ready_to_generate_plan"
    # The summary is what the model reads; it must point at the plan and must
    # not read as a data or permission failure.
    assert "Planner" in result["summary"]
    assert "not a data or permission failure" in result["summary"]
    # The one-turn grant is not spent on a refusal.
    assert "literature" in context.allowed_actions


def test_literature_search_compiles_query_from_typed_execution_concepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study.update(
        {
            "question": "Estimate a governed ICU association.",
            "primary_exposure": (
                "Canonical EasyICU Sepsis-3: suspected infection plus "
                "traditional SOFA >=2 point increase, anchored to onset"
            ),
            "outcome": "In-hospital mortality",
            "execution_concepts": {
                "primary_exposure": "sep3_sofa1",
                "outcome": "death",
                "covariates": [],
            },
        }
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    captured: dict[str, Any] = {}

    def discover(body: dict[str, Any]) -> dict[str, Any]:
        captured.update(body)
        return {
            "status": "searched_no_hits",
            "search_performed": True,
            "queries_to_run": [
                '("Sepsis-3"[Title/Abstract] AND "mortality"[Title/Abstract])'
            ],
            "network_calls": 1,
            "source_candidates": [],
        }

    monkeypatch.setattr(tool_module.idea_mining, "discover_literature", discover)

    result = tool_module.execute_tool(
        "easyicu_search_literature",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-typed-literature-query"),
            allowed_actions={"literature"},
        ),
    )

    assert result["status"] == "ok"
    assert captured["exposure_concept"] == "sep3_sofa1"
    assert captured["outcome_concept"] == "death"


def test_bound_literature_projection_stays_bounded_with_long_abstracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study["idea_handoff"] = {
        "run_id": "idea-run-bounded",
        "idea_id": "idea-bounded",
        "status": "accepted",
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    monkeypatch.setattr(
        tool_module.idea_mining,
        "check_prior_art",
        lambda body: {
            "prior_art": {
                "status": "searched",
                "search_performed": True,
                "network_calls": 2,
                "queries_to_run": ["sepsis AND mortality"],
                "results": [
                    {
                        "pmid": str(10000 + index),
                        "title": f"Sepsis paper {index}",
                        "journal": "Critical Care",
                        "year": 2025,
                        "abstract_excerpt": "organ dysfunction " * 180,
                    }
                    for index in range(12)
                ],
            }
        },
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "prior_art_receipt_binding",
        lambda run_id: {
            "prior_art_binding_schema_version": "easyicu.idea-prior-art-binding/2",
            "prior_art_sha256": "a" * 64,
            "prior_art_status": "searched",
            "prior_art_result_count": 12,
        },
    )

    result = tool_module.execute_tool(
        "easyicu_search_literature",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-bounded-literature"),
            allowed_actions={"literature"},
        ),
    )

    assert result["status"] == "ok"
    assert len(result["details"]["resources"]) == 5
    assert all(
        row.get("pmid") != "17938396" for row in result["details"]["resources"]
    )
    assert len(result["details"]["literature_search"]["articles"]) == 5
    assert all(
        len(row["evidence_excerpt"]) <= 360
        and len(row["abstract_excerpt"]) <= 900
        for row in result["details"]["literature_search"]["articles"]
    )
    assert len(json.dumps(result)) < 20_000


def test_literature_projection_preserves_a_late_null_result() -> None:
    abstract = (
        ("Background and methods. " * 45)
        + "Weekend hypotension was treated less often than weekday daytime. "
        + "No association between weekday daytime and weekday nighttime treatment was found."
    )

    projection = compile_literature_tool_projection(
        discovered={"status": "searched", "search_performed": True},
        candidates=[
            {
                "pmid": "26975737",
                "title": "ICU staffing and hypotension treatment",
                "abstract_excerpt": abstract,
            }
        ],
        idea_receipt_binding=None,
        study_authority_binding=None,
        bound_idea_run_id="idea-direct-fit",
    )

    excerpt = projection["literature_search"]["articles"][0]["abstract_excerpt"]
    assert len(excerpt) <= 900
    assert "Weekend hypotension was treated less often" in excerpt
    assert "No association between weekday daytime and weekday nighttime" in excerpt


def test_zero_topic_hits_do_not_project_method_sources_as_search_results() -> None:
    projection = compile_literature_tool_projection(
        discovered={"status": "searched_no_matches", "search_performed": True},
        candidates=[],
        idea_receipt_binding=None,
        study_authority_binding=None,
        bound_idea_run_id="idea-zero-topic-hits",
    )

    assert projection["resource"] is None
    assert projection["resources"] == []
    assert projection["literature_search"]["methodology"]["sources"]


def test_literature_source_resource_rejects_unverified_or_unsafe_links() -> None:
    assert (
        literature_source_resource({"title": "Unsafe", "url": "javascript:alert(1)"})
        is None
    )
    assert literature_source_resource({"title": "No identifier"}) is None
