"""Every webserver reader of a run's research context uses the versioned owner (B34, S1746).

An executed or candidate run writes its typed context (V2/V3, with
``materialized_inputs``); the plain V1 model refuses it.  The append source
(B34), the multi-database review and the candidate population mode read it
through ``parse_research_context_json``.  The synthetic fixtures write V1, so
each reader's dispatch is pinned with a spy around the real owner.
"""

from __future__ import annotations

import hashlib
import json

from easyicu.research_agent.research_context import typed
from easyicu.webserver import research_launch_population, research_multi_review
from tests.support.result_append import approvable_review, completed_source


def _spy(module, monkeypatch):
    calls = []
    owner = typed.parse_research_context_json

    def versioned(raw):
        calls.append(raw)
        return owner(raw)

    monkeypatch.setattr(module, "parse_research_context_json", versioned)
    return calls


def test_the_multi_database_review_reads_the_versioned_context(tmp_path, monkeypatch) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan

    wrapper, run, plan, context, _ = completed_source(tmp_path)
    review = approvable_review(plan).model_copy(update={"context_sha256": canonical_sha256(context.model_dump(mode="json"))})
    evidence = EvidenceStore(run)
    evidence.register_json(kind="log", description="Synthetic scientific review", evidence_id="scientific_plan_review",
                           filename="scientific_plan_review.json", payload=review.model_dump(mode="json"))
    requests = human_review_requests_for_plan(findings=[], plan=plan, evidence=evidence, require_plan_review=True)
    checkpoint = HumanReviewCheckpoint.create(
        run_id="source", pipeline_config_sha256="d" * 64, environment_identity={}, llm_signature_sha256="e" * 64,
        run_input_capsule_sha256=sha256_of_file(run / "run_input_capsule.json"), capability_activation_sha256="f" * 64,
        runtime_capabilities=(), runtime_bundle=None, requests=requests,
        plan_handoff={"plan": plan.model_dump(mode="json")}, execution_coordinates={})
    write_checkpoint(run / "human_review_checkpoint.json", checkpoint)
    calls = _spy(research_multi_review, monkeypatch)

    _binding, restored, _expected, _, _ = research_multi_review.read_review(wrapper, "source", {"id": "study"})

    assert calls == [(run / "research_context.json").read_bytes()]
    assert restored == context


def test_the_candidate_population_mode_reads_the_versioned_context(tmp_path, monkeypatch) -> None:
    _wrapper, run, _plan, _context, _ = completed_source(tmp_path)
    raw = (run / "research_context.json").read_bytes()
    capsule = {"context_relative_path": "research_context.json", "context_sha256": hashlib.sha256(raw).hexdigest()}
    # The reader imports the owner when it runs; the spy wraps it there.
    calls = _spy(typed, monkeypatch)
    monkeypatch.setattr(
        "easyicu.research_agent.planning.cohort_predicate_domain.authorized_population_mode",
        lambda mode, context: ("authorized", mode, context.research_question),
    )

    mode = research_launch_population.sealed_candidate_population_mode(
        inner_run=run, capsule=capsule, stated_mode="predicate_filtered")

    assert calls == [raw]
    assert mode[0] == "authorized" and mode[2] == json.loads(raw)["research_question"]
