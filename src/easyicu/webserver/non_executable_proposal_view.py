"""Read-only host projection of an unapproved proposal for a bound run."""
from __future__ import annotations

from typing import Any
import hashlib

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.literature import LiteratureBundle, manuscript_citable_keys
from easyicu.research_agent.planning.non_executable_proposal import (
    ProposedDecision, ReadOnlyEvidenceRegistry, build_non_executable_proposal,
)
from easyicu.webserver import run_artifact_disclosure, run_step_trace, study_contexts
from easyicu.webserver.pi_copilot.run_authority import list_bound_run_history, research_pipeline_project_root


def project_proposal(*, service: Any, project_id: str, session_id: str, run_id: str) -> dict:
    """Resolve every coordinate from host owners; accept no client evidence or text."""
    project_id, session_id, run_id = (str(x or "").strip() for x in (project_id, session_id, run_id))
    study_id = service.project_store.resolve(project_id)
    if not study_id:
        raise ValueError("proposal_project_unbound")
    listing = service.list_sessions(project_id=project_id, limit=100)
    if not any(row.get("session_id") == session_id for row in listing.get("sessions", [])):
        raise ValueError("proposal_session_not_bound")
    snapshot = service.replay_store.snapshot(session_id=session_id, project_id=project_id, limit=100)
    if not any(row.get("run_id") == run_id for row in snapshot.get("child_jobs", [])):
        raise ValueError("proposal_run_not_in_session")
    history = list_bound_run_history(study_context_id=study_id,
        project_root=research_pipeline_project_root(study_id), limit=200)
    row = next((item for item in history if item.get("run_id") == run_id), None)
    if not row or row.get("study_id") != study_id:
        raise ValueError("proposal_run_not_in_project")
    study = study_contexts.get_context(study_id)
    if not study:
        raise ValueError("proposal_study_unavailable")
    scientific_sha = study_contexts.scientific_configuration_sha256(study)
    if scientific_sha != row.get("scientific_configuration_sha256"):
        raise ValueError("proposal_run_configuration_mismatch")
    wrapper = row.get("project_dir")
    if not wrapper:
        raise ValueError("proposal_run_directory_unavailable")
    run_dir = run_step_trace.pipeline_run_dir(str(wrapper), run_id)
    # Failed wrappers can refer to one differently named, audit-bound pipeline.
    binding = run_step_trace.failed_literature_preview_binding(
        run_dir, wrapper_run_id=run_id, study_id=study_id)
    if binding is not None:
        run_dir = binding[0]
    if not (run_dir / "evidence").is_dir():
        raise ValueError("proposal_evidence_store_unavailable")
    store = ReadOnlyEvidenceRegistry(run_dir)
    evidence = store.get("preplan_literature_bundle")
    literature = None
    if evidence and evidence.evidence_id == "preplan_literature_bundle":
        path = verified_run_evidence_path(run_dir, evidence)
        if path is None:
            raise ValueError("proposal_evidence_integrity_failed")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != evidence.sha256:
            raise ValueError("proposal_evidence_integrity_failed")
        try:
            bundle = LiteratureBundle.model_validate_json(content)
        except ValueError as exc:
            raise ValueError("proposal_literature_bundle_invalid") from exc
        literature = {"evidence_id": evidence.evidence_id, "sha256": evidence.sha256,
                      "retained_citation_keys": list(manuscript_citable_keys(bundle)),
                      "decision_bindings_verified": False}
    wording = service.replay_store.question_wording_context(session_id=session_id, project_id=project_id)
    turns = wording.get("user_turns") or []
    original = str(turns[0].get("text") or "")[:3000] if turns else ""
    decisions = []
    for key, label in (("question", "研究问题"), ("cohort", "研究人群"),
                       ("primary_exposure", "暴露"), ("outcome", "结局")):
        value = study.get(key)
        if key == "cohort" and isinstance(value, dict):
            # StudyContext represents cohort as a typed object. Preserve its
            # user-visible label and review text without inferring validity.
            value = "；".join(
                item.strip() for item in (value.get("label"), value.get("review"))
                if isinstance(item, str) and item.strip()
            )
        if isinstance(value, str) and value.strip():
            decisions.append(ProposedDecision(decision_id=key, text=f"{label}：{value.strip()[:1200]}"))
    if not decisions:
        raise ValueError("proposal_configuration_empty")
    proposal = build_non_executable_proposal(store=store, configuration=study,
        expected_configuration_sha256=canonical_sha256(study), original_question=original,
        decisions=tuple(decisions), unresolved=("source_mapping_unverified", "literature_bindings_unverified"))
    latest = study_contexts.get_context(study_id)
    if not latest or study_contexts.scientific_configuration_sha256(latest) != scientific_sha:
        raise ValueError("proposal_configuration_changed")
    # Recheck only the authority-bearing values: unrelated replay timestamps
    # and progress updates must not make a stable proposal stale.
    if service.project_store.resolve(project_id) != study_id:
        raise ValueError("proposal_project_changed")
    current_sessions = service.list_sessions(project_id=project_id, limit=100)
    if not any(item.get("session_id") == session_id for item in current_sessions.get("sessions", [])):
        raise ValueError("proposal_session_changed")
    current_replay = service.replay_store.snapshot(session_id=session_id, project_id=project_id, limit=100)
    if not any(item.get("run_id") == run_id for item in current_replay.get("child_jobs", [])):
        raise ValueError("proposal_session_run_changed")
    current_history = list_bound_run_history(study_context_id=study_id,
        project_root=research_pipeline_project_root(study_id), limit=200)
    current_run = next((item for item in current_history if item.get("run_id") == run_id), None)
    if (not current_run or current_run.get("study_id") != study_id
            or current_run.get("project_dir") != wrapper
            or current_run.get("scientific_configuration_sha256") != scientific_sha):
        raise ValueError("proposal_run_changed")
    current_wording = service.replay_store.question_wording_context(session_id=session_id, project_id=project_id)
    current_turns = current_wording.get("user_turns") or []
    current_original = str(current_turns[0].get("text") or "")[:3000] if current_turns else ""
    if current_original != original:
        raise ValueError("proposal_original_question_changed")
    if literature is not None:
        current = ReadOnlyEvidenceRegistry(run_dir).get("preplan_literature_bundle")
        if (not current or current.sha256 != literature["sha256"]
                or verified_run_evidence_path(run_dir, current) is None):
            raise ValueError("proposal_evidence_integrity_failed")
    payload = {"run_id": run_id, "scientific_configuration_sha256": scientific_sha,
               "proposal": proposal.model_dump(mode="json"), "literature_candidates": literature}
    if not run_artifact_disclosure.scan_browser_projection(payload).get("passed"):
        raise ValueError("proposal_privacy_scan_failed")
    return payload
