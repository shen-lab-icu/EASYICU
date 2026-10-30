"""Read-only host projection of an unapproved proposal for a bound run."""
from __future__ import annotations

from typing import Any
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.literature import LiteratureBundle, PubMedLiteratureClient, manuscript_citable_records
from easyicu.research_agent.planning.literature_design_authority import _source_excerpt
from easyicu.research_agent.planning.non_executable_proposal import (
    CandidateEvidenceRegistry, ProposedDecision, ProposedReference, ReadOnlyEvidenceRegistry,
    build_non_executable_proposal,
)
from easyicu.webserver import run_artifact_disclosure, run_step_trace, study_contexts
from easyicu.webserver.candidate_pubmed import register_pubmed_candidate
from easyicu.webserver.pi_copilot.run_authority import list_bound_run_history, research_pipeline_project_root


class _LiteratureDecisionReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str = Field(min_length=1, max_length=200)
    decision_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    citation_key: str = Field(min_length=1, max_length=200)
    source_evidence_id: str = Field(min_length=1, max_length=200)
    verbatim_excerpt: str = Field(min_length=12, max_length=240)
    evidence_scope: str | None = Field(default=None, max_length=500)


class _LiteratureDecisionReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    references: list[_LiteratureDecisionReference] = Field(max_length=100)


class CandidateLiteratureReference(BaseModel):
    """An untrusted, transient suggestion; it grants no evidence authority."""
    model_config = ConfigDict(extra="forbid")
    decision_id: str = Field(min_length=1, max_length=200)
    citation_key: str = Field(min_length=1, max_length=200)
    source_evidence_id: str = Field(min_length=1, max_length=200)
    verbatim_excerpt: str = Field(min_length=12, max_length=240)
    evidence_scope: str | None = Field(default=None, max_length=500)


def _candidate_evidence_root(study_id: str, session_id: str, run_id: str) -> Path:
    scope = canonical_sha256({'session_id': session_id, 'run_id': run_id})
    return research_pipeline_project_root(study_id) / 'candidate_evidence' / scope


def _proposed_reference(citation_key: str, source_evidence_id: str, excerpt: str, scope: str | None) -> ProposedReference:
    external = source_evidence_id.startswith('candidate_pubmed_')
    return ProposedReference(evidence_id=source_evidence_id if external else 'preplan_literature_bundle',
        kind='literature_candidate' if external else 'literature', citation_key=citation_key,
        source_evidence_id=source_evidence_id, verbatim_excerpt=excerpt, evidence_scope=scope)


def _register_pubmed_abstract(*, bundle: LiteratureBundle, citation_key: str, root: Path) -> tuple[str, str]:
    """Fetch a verified public abstract into a separate candidate evidence store."""
    records = [item for item in manuscript_citable_records(bundle) if item.key == citation_key]
    screens = [item for item in bundle.screening_decisions if item.citation_key == citation_key]
    cards = [item for item in bundle.design_evidence_cards if item.citation_key == citation_key]
    if len(records) != 1 or len(screens) != 1 or screens[0].disposition != 'include' or len(cards) != 1:
        raise ValueError('proposal_candidate_not_screened')
    record, card = records[0], cards[0]
    pmid = str(record.pmid or '')
    if not pmid.isascii() or not pmid.isdigit() or not 1 <= len(pmid) <= 12 or card.access_mode != 'abstract':
        raise ValueError('proposal_candidate_pubmed_unavailable')
    excerpt = _source_excerpt(record)
    if not excerpt or hashlib.sha256(excerpt.encode('utf-8')).hexdigest() != card.abstract_sha256:
        raise ValueError('proposal_candidate_source_excerpt_mismatch')
    client = PubMedLiteratureClient(timeout=15)
    params = client._with_etiquette({'db': 'pubmed', 'id': pmid, 'retmode': 'xml'})
    body = None
    for _ in range(3):
        body = client._http_get('efetch.fcgi', params)
        if body:
            break
    if not body or len(body) > 5_000_000:
        raise ValueError('proposal_pubmed_unavailable')
    try:
        document = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError('proposal_pubmed_invalid') from exc
    articles = document.findall('.//PubmedArticle')
    if len(articles) != 1:
        raise ValueError('proposal_pubmed_identity_mismatch')
    article = articles[0]
    fetched_pmid = ''.join(article.findtext('./MedlineCitation/PMID', default='').split())
    title_node = article.find('.//Article/ArticleTitle')
    fetched_title = ' '.join(''.join(title_node.itertext()).split()) if title_node is not None else ''
    dois = [''.join(node.itertext()).strip().lower() for node in article.findall('./PubmedData/ArticleIdList/ArticleId')
            if node.get('IdType') == 'doi']
    abstract = ' '.join(' '.join(''.join(node.itertext()).split())
                        for node in article.findall('.//Abstract/AbstractText')).strip()
    if (fetched_pmid != pmid or fetched_title.casefold().rstrip(' .') != ' '.join(record.title.split()).casefold().rstrip(' .')
            or (record.doi and record.doi.strip().lower() not in dois) or not abstract):
        raise ValueError('proposal_pubmed_identity_mismatch')
    preview = excerpt[:240]
    while len(preview) >= 12 and preview not in abstract:
        preview = preview[:-1]
    if len(preview) < 12:
        raise ValueError('proposal_candidate_source_excerpt_mismatch')
    # Do not assert that every extractive selection is contiguous: a candidate
    # excerpt must separately occur in both the selected text and the abstract.
    content = {'citation_key': record.key, 'pmid': record.pmid, 'doi': record.doi,
        'title': record.title, 'abstract': abstract, 'source_excerpt': excerpt,
        'source_url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        'retrieved_at': datetime.now(timezone.utc).isoformat()}
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True)
    eid = 'candidate_abstract_' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]
    store = EvidenceStore(root)
    store.register_text(kind='log', description='Candidate PubMed abstract; applicability not reviewed',
        text=encoded, filename='pubmed_abstract.json', evidence_id=eid)
    return eid, preview


def save_candidate_reference(*, service: Any, project_id: str, session_id: str, run_id: str,
                             proposed: CandidateLiteratureReference) -> dict:
    """Persist an explicitly suggested reference in a separate, unapproved evidence store."""
    preview = project_proposal(service=service, project_id=project_id, session_id=session_id,
                               run_id=run_id, proposed=proposed)
    decision = next(row for row in preview['proposal']['decisions'] if row['decision_id'] == proposed.decision_id)
    study_id = service.project_store.resolve(project_id)
    study = study_contexts.get_context(study_id)
    if not study or study_contexts.scientific_configuration_sha256(study) != preview['scientific_configuration_sha256']:
        raise ValueError('proposal_configuration_changed')
    manifest = {'schema_version': 'easyicu.candidate-literature-bindings/2',
        'configuration_sha256': study_contexts.scientific_configuration_sha256(study), 'references': [{
            'decision_id': proposed.decision_id,
            'decision_text_sha256': hashlib.sha256(decision['text'].encode('utf-8')).hexdigest(),
            'citation_key': proposed.citation_key, 'source_evidence_id': proposed.source_evidence_id,
            'verbatim_excerpt': proposed.verbatim_excerpt, 'evidence_scope': proposed.evidence_scope}]}
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    eid = 'candidate_literature_binding_' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]
    EvidenceStore(_candidate_evidence_root(study_id, session_id, run_id)).register_text(
        kind='log', description='Candidate literature association; applicability not reviewed',
        text=text, filename='candidate_binding.json', evidence_id=eid)
    result = project_proposal(service=service, project_id=project_id, session_id=session_id, run_id=run_id)
    result['candidate_binding_evidence_id'] = eid
    return result


def _literature_projection(bundle: LiteratureBundle, *, evidence_id: str, sha256: str) -> dict:
    """Present retained records and actual screening; no claim-to-paper inference."""
    retained = manuscript_citable_records(bundle)
    retained_keys = {item.key for item in retained}
    screenings = {item.citation_key: item for item in bundle.screening_decisions}
    cards = {item.citation_key: item for item in bundle.design_evidence_cards}
    records = []
    for citation in retained[:100]:
        screen = screenings.get(citation.key)
        card = cards.get(citation.key)
        records.append({
            "key": citation.key,
            "title": citation.title,
            "year": citation.year,
            "pmid": citation.pmid,
            "doi": citation.doi,
            "screening": screen.disposition if screen else None,
            "evidence_role": screen.evidence_role if screen else None,
            "screening_rationale": screen.rationale if screen else None,
            "outcome_match": screen.outcome_match if screen else None,
            "card_access_mode": card.access_mode if card else None,
            # Cards are retained run output; they are not independent review
            # and a listed dimension does not prove support for a decision.
            "card_dimensions": [item.dimension for item in card.evidence] if card else [],
        })
    included = [item for item in bundle.screening_decisions if item.disposition == "include"]
    return {
        "evidence_id": evidence_id,
        "sha256": sha256,
        "retained_citation_keys": [item.key for item in retained],
        "records": records,
        "records_total": len(retained),
        "screened": len(bundle.screening_decisions),
        "included": len(included),
        "excluded": sum(item.disposition == "exclude" for item in bundle.screening_decisions),
        "included_direct_comparators": len({item.citation_key for item in included
                                            if item.citation_key in retained_keys
                                            and item.evidence_role == "direct_comparator"}),
        "decision_applicability_verified": False,
    }


def project_proposal(*, service: Any, project_id: str, session_id: str, run_id: str,
                     proposed: CandidateLiteratureReference | None = None,
                     collect_citation_key: str | None = None,
                     collect_pmid: str | None = None,
                     include_source_candidate: bool = False) -> dict:
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
    candidate_root = _candidate_evidence_root(study_id, session_id, run_id)
    store = CandidateEvidenceRegistry(run_dir, candidate_root)
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
        literature = _literature_projection(bundle, evidence_id=evidence.evidence_id, sha256=evidence.sha256)
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
    # This is a review slot, not a claim that a model or method was selected.
    decisions.append(ProposedDecision(decision_id="method_background",
        text="方法背景：候选材料及其对具体分析方法的适用性待核验"))
    reference_record = store.get("candidate_literature_bindings")
    reference_records = ([reference_record] if reference_record else []) + ([
        row for row in store.candidate.records if row.evidence_id.startswith('candidate_literature_binding_')
    ] if store.candidate else [])
    groups: dict[str, list[ProposedReference]] = {}
    by_id = {decision.decision_id: decision for decision in decisions}
    for reference_record in reference_records:
        reference_path = store.path(reference_record)
        if reference_path is None:
            raise ValueError("proposal_binding_manifest_integrity_failed")
        reference_bytes = reference_path.read_bytes()
        if hashlib.sha256(reference_bytes).hexdigest() != reference_record.sha256:
            raise ValueError("proposal_binding_manifest_integrity_failed")
        try:
            manifest = _LiteratureDecisionReferences.model_validate_json(reference_bytes)
        except ValueError as exc:
            raise ValueError("proposal_binding_manifest_invalid") from exc
        if manifest.schema_version not in {"easyicu.candidate-literature-bindings/1", "easyicu.candidate-literature-bindings/2"}:
            raise ValueError("proposal_binding_manifest_configuration_mismatch")
        expected_binding_sha = (canonical_sha256(study) if manifest.schema_version.endswith('/1')
                                else scientific_sha)
        if manifest.configuration_sha256 != expected_binding_sha:
            # Historical candidate manifests remain immutable but cannot bind
            # a new configuration. A stale one should not hide the proposal.
            continue
        for ref in manifest.references:
            decision = by_id.get(ref.decision_id)
            if (not decision or hashlib.sha256(decision.text.encode("utf-8")).hexdigest()
                    != ref.decision_text_sha256):
                raise ValueError("proposal_binding_manifest_decision_mismatch")
            groups.setdefault(ref.decision_id, []).append(_proposed_reference(
                ref.citation_key, ref.source_evidence_id, ref.verbatim_excerpt, ref.evidence_scope))
    if groups:
        decisions = [decision.model_copy(update={"references": tuple(groups.get(decision.decision_id, ()))})
                     for decision in decisions]
    if proposed is not None:
        proposed = CandidateLiteratureReference.model_validate(proposed)
        target = next((item for item in decisions if item.decision_id == proposed.decision_id), None)
        if target is None:
            raise ValueError("proposal_candidate_decision_not_found")
        ref = _proposed_reference(proposed.citation_key, proposed.source_evidence_id, proposed.verbatim_excerpt,
                                  proposed.evidence_scope)
        decisions = [item.model_copy(update={"references": (*item.references, ref)}) if item is target else item
                     for item in decisions]
    proposal = build_non_executable_proposal(store=store, configuration=study,
        expected_configuration_sha256=canonical_sha256(study), original_question=original,
        decisions=tuple(decisions), unresolved=("source_mapping_unverified", "literature_bindings_unverified"))
    source_candidate = None
    if include_source_candidate:
        from easyicu.webserver.source_partition_candidates import candidate_from_bound_research_context
        from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
        try:
            source_candidate = {"status": "candidate", **candidate_from_bound_research_context(study=study, run_dir=run_dir)}
        except (ValueError, OSError, TypeError, KeyError, AuthorityFilesystemError):
            # The proposal remains visible, but never turn an unavailable or
            # conflicting mapping into an empty or approved source roster.
            source_candidate = {"status": "unavailable"}
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
    for reference_record in reference_records:
        current_store = store.reload()
        current = current_store.get(reference_record.evidence_id)
        if (not current or current.sha256 != reference_record.sha256
                or current_store.path(current) is None):
            raise ValueError("proposal_binding_manifest_integrity_failed")
    collected = None
    if collect_citation_key is not None:
        if literature is None:
            raise ValueError('proposal_literature_bundle_missing')
        collected = _register_pubmed_abstract(bundle=bundle, citation_key=collect_citation_key, root=candidate_root)
    candidate_paper = None
    if collect_pmid is not None:
        candidate_paper = register_pubmed_candidate(pmid=collect_pmid, root=candidate_root)
    payload = {"run_id": run_id, "scientific_configuration_sha256": scientific_sha,
               "proposal": proposal.model_dump(mode="json"), "literature_candidates": literature,
               "source_partition_candidate": source_candidate}
    if collected:
        payload['candidate_abstract_evidence_id'] = collected[0]
        payload['candidate_abstract_excerpt'] = collected[1]
        payload['candidate_abstract_citation_key'] = collect_citation_key
    if candidate_paper:
        payload['candidate_abstract_evidence_id'] = candidate_paper[0]
        payload['candidate_abstract_citation_key'] = candidate_paper[1]['citation_key']
        payload['candidate_pubmed_title'] = candidate_paper[1]['title']
        payload['candidate_pubmed_abstract'] = candidate_paper[1]['abstract']
    if not run_artifact_disclosure.scan_browser_projection(payload).get("passed"):
        raise ValueError("proposal_privacy_scan_failed")
    return payload
