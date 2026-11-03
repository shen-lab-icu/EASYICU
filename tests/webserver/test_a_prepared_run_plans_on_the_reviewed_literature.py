"""A prepared run plans again on the literature the researcher reviewed.

11-03 eICU walk: accepting the metadata-only candidate started data
preparation, and the package-bound Planner searched the literature again
(3.4 minutes, as long as the candidate's own search), so the plan it showed
could rest on other citations than the reviewed one.  The candidate's
checkpoint binds its plan handoff, which carries the exact pre-plan bundle;
the preparation run now seeds its Planner with that bundle and skips the live
search.  A tampered handoff refuses the grant; a checkpoint without one, or
a bundle for another question, seeds nothing.  Synthetic bundles only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.plan_review import HumanReviewRequest
from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.agent_pipeline_runs import ResearchPipelineRunError, _reviewed_candidate_literature

QUESTION = "Is the first-day maximum lactate associated with in-hospital death?"


def _checkpoint(literature: dict | None) -> dict:
    request = HumanReviewRequest.create(
        kind="scientific_stop", summary="Review the candidate plan.",
        authority_sha256="a" * 64, payload={},
    )
    handoff = {"allowed_literature_citation_keys": ["smith_2020"]}
    if literature is not None:
        handoff["preplan_literature"] = literature
    return HumanReviewCheckpoint.create(
        run_id="run_candidate", pipeline_config_sha256="b" * 64, environment_identity={},
        llm_signature_sha256="c" * 64, run_input_capsule_sha256="d" * 64,
        capability_activation_sha256="e" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=(request,), plan_handoff=handoff, execution_coordinates={},
    ).model_dump(mode="json")


def _bundle(question: str = QUESTION) -> dict:
    return LiteratureBundle(research_question=question, citations=[CitationRecord(
        key="smith_2020", title="Lactate and mortality", year="2020", pmid="1",
    )]).model_dump(mode="json")


def test_the_candidate_checkpoint_carries_its_reviewed_bundle():
    assert _reviewed_candidate_literature(_checkpoint(_bundle())) == _bundle()
    # An older or minimal checkpoint has nothing to seed; the run searches.
    assert _reviewed_candidate_literature(_checkpoint(None)) is None
    assert _reviewed_candidate_literature({"state": "pending"}) is None


def test_a_handoff_changed_after_review_refuses_the_grant():
    checkpoint = _checkpoint(_bundle())
    checkpoint["plan_handoff"]["preplan_literature"]["citations"][0]["title"] = "Another paper"
    with pytest.raises(ResearchPipelineRunError) as caught:
        _reviewed_candidate_literature(checkpoint)
    assert caught.value.code == "candidate_plan_materialization_authority_invalid"


def test_the_preparation_run_seeds_only_a_bundle_for_its_own_question():
    source = Path(agent_pipeline_runs.__file__).read_text(encoding="utf-8")
    launch = source.split("bound_preplan_literature = idea_mining.load_bound_prior_art_literature(", 1)[1]
    launch = launch.split("live_pubmed_requested = (", 1)[0]
    # Idea Mining and a study-bound search keep precedence; the reviewed
    # candidate bundle fills only an empty seed, for the same question.
    assert "if bound_preplan_literature is None and reviewed_literature is not None and str(" in launch
    assert 'reviewed_literature.get("research_question") or "").strip() == question.strip():' in launch
    assert launch.index("reviewed_literature = candidate_authority.preplan_literature") < launch.index(
        "if development_resume_literature is not None:")
    assert "preplan_literature=_reviewed_candidate_literature(checkpoint)," in source
