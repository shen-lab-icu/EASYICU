"""Run readiness gates and report rendering.

Extracted from :mod:`easyicu.research_agent.pipeline` so the fail-closed
readiness logic (run_status.json, evidence_audit.json, numeric_audit.json,
claim_ledger.csv, author_review_note.md, manuscript_ready.md) and the
human-readable results_report.md rendering can be reasoned about,
tested, and re-used without pulling the full ``ResearchAgentPipeline``
class into memory.

Public entry points
-------------------
* :func:`write_readiness_artifacts` — compute the fail-closed gates,
  write the canonical audit JSON / CSV artefacts, register them on the
  :class:`EvidenceStore`, and return ``(gates, artifact_paths)``.
* :func:`render_report` — render the human-readable run summary
  ``results_report.md``.
* :func:`execution_gate_status` — pure helper that decides whether
  the analysis-execution phase reached completion (called both during
  the execute loop and during readiness derivation).

Internal helpers
----------------
``_count_missing_evidence_markers``, ``_publication_figure_bundle_ready``,
``_compute_readiness_gates``, ``_extract_claim_ledger_rows`` and
``_render_author_review_note`` keep leading underscores because they
have no external callers — they are implementation details of the two
public entry points above.

All functions are stateless: they take their inputs as keyword args
(``context``, ``plan``, ``findings``, ``per_step_records``,
``evidence``, ``run_dir``, ``manuscript_path`` ...) and write/read
the canonical run-directory layout. Nothing here touches the
``ResearchAgentPipeline`` instance.
"""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .article_contract import (
    article_contract_audit_payload,
    summarize_article_contract_coverage,
)
from .display_suite import summarize_display_suite_status
from .completion import (
    count_missing_evidence_markers as _count_missing_evidence_markers,
    count_writer_attempts,
    has_figure_only_output_contract,
    publication_authorized,
    run_completion_axes,
    step_completion_projection,
)
from ..authority.evidence_store import EvidenceStore, sha256_of_file
from ..authority.step_recovery import StepRecoverySignature
from ..planning.capability_registry import (
    assess_scientific_capability,
    families_without_deterministic_primary,
)
from ..planning.figure_strategy import summarize_article_figure_strategy_coverage
from ..planning.study_design import study_design_family_for_analysis_type
from ..research_context.cohort_granularity import format_patient_count
from ..figures.publication import PUBLICATION_FIGURE_SKILL_POLICY_VERSION
from ..plan_utils import _output_declares_figure, _parent_step_id_for_figure_step
from .review_artifacts import build_review_artifact_payloads
from .result_integrity import (
    primary_result_plausibility_errors,
    primary_survival_estimate_integrity_errors,
)
from .step_summaries import (
    authoritative_step_summaries as _authoritative_step_summaries,
    step_authority_records as _step_authority_records,
)
from ..authority.runtime_artifacts import (
    active_step_evidence_ids,
    capture_code_version,
    current_evidence_records,
    current_run_evidence_records,
    current_step_records,
    current_successful_step_ids,
    current_successful_step_records,
    load_run_artifact_authority,
    verified_run_evidence_path,
)
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding


def _figure_steps_satisfied_by_repair(
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]],
    *,
    plan: Optional[AnalysisPlan] = None,
) -> set:
    """Figure step_ids whose figure a successful rendering-only repair produced.

    A ``*_figure`` step can fail (its own runner emitted no exports) yet still be
    salvaged by a later ``*_figure_repair`` step that renders the figure into its
    OWN outputs dir. That repair step is not a required plan step, so the gate
    would otherwise still count the original figure step as ``execution_failed``.
    Credit requires an orchestrator-owned ``repair_target_step_id``, a current
    successful source checkpoint, and (for modern manifests) digest-bound
    evidence that names the source evidence it rendered.  A renderer's own
    ``parent_step``-style self-report is deliberately insufficient.
    """

    satisfied: set = set()
    steps_dir = run_dir / "steps"
    if not steps_dir.is_dir():
        return satisfied
    authority = (
        load_run_artifact_authority(run_dir) if per_step_records is not None else None
    )
    if per_step_records is not None and authority is None:
        # A modern outer ledger cannot gain repair credit from append-only
        # files when its evidence authority is absent.
        return satisfied
    strict_evidence_binding = per_step_records is not None
    evidence_by_id = {
        str(record.get("evidence_id") or ""): record
        for record in ((authority or {}).get("evidence") or [])
        if isinstance(record, Mapping) and str(record.get("evidence_id") or "")
    }
    if per_step_records is not None:
        candidates = [
            (
                str(record.get("step_id") or ""),
                record.get("step_summary"),
                record,
            )
            for record in current_successful_step_records(per_step_records)
        ]
    else:
        candidates = []
        for summary in steps_dir.glob("*/outputs/step_summary.json"):
            try:
                payload = json.loads(summary.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidates.append((summary.parents[1].name, payload, None))
    current_by_step = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(per_step_records or [])
        if str(record.get("step_id") or "")
    }
    plan_by_step = {
        str(step.step_id or ""): step
        for step in ((plan.steps if plan is not None else []) or [])
        if str(step.step_id or "")
    }
    for step_id, payload, ledger_record in candidates:
        if not isinstance(payload, dict):
            continue
        if (
            not step_id
            or step_id in {".", ".."}
            or Path(step_id).name != step_id
            or "/" in step_id
            or "\\" in step_id
        ):
            continue
        if (
            payload.get("rendering_only") is not True
            or str(payload.get("status") or "") != "ok"
        ):
            continue
        # A renderer may describe its own inputs, but that self-report is not
        # enough to supersede a failed plan step.  The orchestrator-owned outer
        # ledger must name the exact target, and the target must be the split
        # figure directly downstream of a currently successful source step.
        source_step_id = str(payload.get("source_step_id") or "").strip()
        target_step_id = str(
            (ledger_record or {}).get("repair_target_step_id") or ""
        ).strip()
        if not source_step_id or not target_step_id:
            continue
        if plan is not None:
            target_step = plan_by_step.get(target_step_id)
            if (
                target_step is None
                or not has_figure_only_output_contract(target_step)
                or _parent_step_id_for_figure_step(target_step) != source_step_id
            ):
                continue
        elif target_step_id != f"{source_step_id}_figure":
            continue
        source_record = current_by_step.get(source_step_id)
        target_record = current_by_step.get(target_step_id)
        target_status = (
            str(
                (target_record or {}).get("status")
                if isinstance(target_record, Mapping)
                else ""
            )
            .strip()
            .lower()
        )
        if not isinstance(target_record, Mapping) or target_status not in {
            "execution_failed",
            "contract_failed",
            "repair_failed",
        }:
            continue

        if strict_evidence_binding and (
            not isinstance(source_record, Mapping)
            or str(source_record.get("status") or "").strip().lower() != "ok"
        ):
            continue

        source_evidence_ids = {
            str(evidence_id)
            for evidence_id in (
                (source_record.get("evidence_ids") or [])
                if isinstance(source_record, Mapping)
                else []
            )
            if str(evidence_id).strip()
        }
        ledger_source_ids = {
            str(evidence_id)
            for evidence_id in ((ledger_record or {}).get("source_evidence_ids") or [])
            if str(evidence_id).strip()
        }
        if strict_evidence_binding and (
            not ledger_source_ids
            or not source_evidence_ids
            or not ledger_source_ids <= source_evidence_ids
        ):
            continue
        if strict_evidence_binding:
            source_evidence_verified = True
            for evidence_id in ledger_source_ids:
                source_evidence = evidence_by_id.get(evidence_id)
                if (
                    not isinstance(source_evidence, Mapping)
                    or str(source_evidence.get("produced_by_step") or "")
                    != source_step_id
                    or verified_run_evidence_path(run_dir, source_evidence) is None
                ):
                    source_evidence_verified = False
                    break
            if not source_evidence_verified:
                continue
        outputs_dir = steps_dir / step_id / "outputs"
        try:
            if outputs_dir.is_symlink():
                continue
            outputs_dir.resolve().relative_to(steps_dir.resolve())
        except ValueError:
            continue
        declared_paths: List[str] = []
        for key in ("figure_paths", "figure_files", "figure_path", "figure_file"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                declared_paths.extend(str(item) for item in value.values())
            elif isinstance(value, (list, tuple, set)):
                declared_paths.extend(str(item) for item in value)
            elif value:
                declared_paths.append(str(value))

        def _declared_figure_exists(candidate: str) -> bool:
            path = Path(candidate)
            if not path.is_absolute():
                # Step summaries conventionally declare export basenames (for
                # example ``publication_figure.png``), relative to their own
                # outputs directory.  Resolving those names from ``run_dir``
                # makes a real, ledgered repair look absent.  The containment
                # check below still rejects ``..`` traversal and symlinks that
                # resolve outside this repair's outputs directory.
                path = outputs_dir / path
            try:
                path.resolve().relative_to(outputs_dir.resolve())
            except ValueError:
                return False
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix.lower()
                not in {
                    ".png",
                    ".svg",
                    ".pdf",
                    ".tif",
                    ".tiff",
                }
            ):
                return False
            if not strict_evidence_binding:
                return True

            # A current manifest can prove more than file existence: the
            # successful outer record must actively bind a figure evidence
            # record produced by this repair, and both the output and evidence
            # copy must still match the registered digest.  This prevents an
            # old same-step file from gaining credit merely because a newer
            # summary repeats its basename.
            active_ids = {
                str(evidence_id)
                for evidence_id in ((ledger_record or {}).get("evidence_ids") or [])
                if str(evidence_id).strip()
            }
            output_digest = sha256_of_file(path)
            for evidence_id in active_ids:
                evidence_record = evidence_by_id.get(evidence_id)
                if not isinstance(evidence_record, Mapping):
                    continue
                if (
                    str(evidence_record.get("kind") or "") != "figure"
                    or str(evidence_record.get("produced_by_step") or "") != step_id
                    or str(evidence_record.get("sha256") or "") != output_digest
                ):
                    continue
                evidence_path = verified_run_evidence_path(run_dir, evidence_record)
                if evidence_path is None:
                    continue
                metadata = evidence_record.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                evidence_source_ids = {
                    str(value)
                    for value in (evidence_record.get("inputs") or [])
                    if str(value).strip()
                }
                raw_source_ids = metadata.get("source_evidence_ids")
                if isinstance(raw_source_ids, (list, tuple, set)):
                    evidence_source_ids.update(
                        str(value) for value in raw_source_ids if str(value).strip()
                    )
                else:
                    source_id = str(metadata.get("source_evidence_id") or "").strip()
                    if source_id:
                        evidence_source_ids.add(source_id)
                if not ledger_source_ids <= evidence_source_ids:
                    continue
                original_name = evidence_path.name.split("__", 1)[-1]
                if (
                    original_name == path.name
                    and evidence_path.is_file()
                    and sha256_of_file(evidence_path) == output_digest
                ):
                    return True
            return False

        has_rendered_figure = any(
            _declared_figure_exists(candidate) for candidate in declared_paths
        )
        if per_step_records is None and not declared_paths:
            # Historical summaries did not always declare exports. Filesystem
            # discovery is retained only for that explicit legacy path; a live
            # ledger may never gain credit from an append-only stale file.
            has_rendered_figure = any(
                any(outputs_dir.glob(f"*{ext}"))
                for ext in (".png", ".svg", ".pdf", ".tif", ".tiff")
            )
        if has_rendered_figure:
            satisfied.add(target_step_id)
    return satisfied


def execution_gate_status(
    *,
    plan: Optional[AnalysisPlan],
    per_step_records: Sequence[Dict[str, Any]],
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if plan is None:
        return {
            "execution_complete": False,
            "required_step_count": 0,
            "completed_step_count": 0,
            "missing_steps": [],
            "failed_steps": [{"step_id": "plan", "status": "not_available"}],
        }
    required_step_ids = [s.step_id for s in (plan.steps if plan is not None else [])]
    # 🔧 2026-05-16: previously this dict excluded `00_probe` records on the
    # assumption the planner never lists it. But the planner *does* sometimes
    # include `00_probe` in the published plan, and a deterministic probe
    # record is appended to per_step_records with status='ok' from
    # execution.phase._build_probe_summary. Excluding it here made the gate
    # mis-report `00_probe` as a permanently-missing required step. Surfacing
    # the deterministic probe record fixes the false negative.
    current_records = current_step_records(per_step_records)
    record_by_step = {
        str(record.get("step_id")): record
        for record in current_records
        if record.get("step_id")
    }
    status_by_step = {
        step_id: str(record.get("status") or "")
        for step_id, record in record_by_step.items()
    }
    missing_steps = [
        step_id for step_id in required_step_ids if step_id not in status_by_step
    ]
    # Credit ``*_figure`` steps whose figure a later rendering-only repair step
    # actually produced: the figure exists on disk, so a hard fail-close here
    # would be a false negative (the deliverable is present). Only figure steps
    # matched EXACTLY by a repair's ``parent_step`` are credited; every other
    # failure still blocks the gate. ``run_dir=None`` preserves legacy behaviour.
    repaired_figures: set = (
        _figure_steps_satisfied_by_repair(
            run_dir,
            per_step_records,
            plan=plan,
        )
        if run_dir is not None
        else set()
    )

    def _step_ok(step_id: str) -> bool:
        return status_by_step.get(step_id) == "ok" or (step_id in repaired_figures)

    failed_steps = [
        {"step_id": step_id, "status": status_by_step.get(step_id)}
        for step_id in required_step_ids
        if step_id in status_by_step and not _step_ok(step_id)
    ]
    completion = step_completion_projection(
        required_step_ids=required_step_ids,
        record_by_step=record_by_step,
        status_by_step=status_by_step,
        step_ok=_step_ok,
    )
    scientific_incomplete_steps = completion["scientific_incomplete_steps"]
    execution_complete = not missing_steps and not failed_steps
    return {
        "execution_complete": execution_complete,
        "step_scientific_requirements_complete": (
            execution_complete and not scientific_incomplete_steps
        ),
        "required_step_count": len(required_step_ids),
        "completed_step_count": sum(
            1 for step_id in required_step_ids if _step_ok(step_id)
        ),
        "missing_steps": missing_steps,
        "failed_steps": failed_steps,
        "scientific_incomplete_steps": scientific_incomplete_steps,
        "step_completion_states": completion["step_completion_states"],
    }


_OUTCOME_ENDPOINT_RE = re.compile(
    r"\b(?:death|mortality|survival|endpoint|event|outcome)s?\b",
    flags=re.IGNORECASE,
)
_SPECIFIC_OUTCOME_RE = re.compile(
    r"\b(?:death|mortality|survival)\b",
    flags=re.IGNORECASE,
)
_OUTCOME_INFERENCE_RE = re.compile(
    r"\b(?:association|contrast|comparison|difference|effect|estimate|"
    r"point estimate|odds ratio|hazard ratio|risk ratio|relative risk|"
    r"protective|harmful|near[- ]?null|equivalence|prognostic|"
    r"separation|vary|varies|varied|ranging)\b",
    flags=re.IGNORECASE,
)
_OUTCOME_GROUP_RE = re.compile(
    r"\b(?:death|mortality|outcome|endpoint|event)s?\b.{0,80}"
    r"\b(?:across|between|by)\b.{0,60}\b(?:group|subgroup|definition)s?\b",
    flags=re.IGNORECASE,
)
_OUTCOME_SAFE_BLOCK_RE = re.compile(
    r"\b(?:blocked|withheld|not authorized|not authorised|not performed|"
    r"not executed|not analysed|not analyzed|not inferred|cannot be inferred|"
    r"must not be inferred|no outcome claim|no death claim|no mortality claim|"
    r"before any|prior to any)\b",
    flags=re.IGNORECASE,
)


def _payload_mentions_outcome(payload: Any) -> bool:
    return bool(_OUTCOME_ENDPOINT_RE.search(json.dumps(payload, ensure_ascii=False)))


def _step_summary_blocks_outcome(payload: Dict[str, Any]) -> bool:
    if payload.get("grouped_death_analysis_executed") is False:
        return True
    if payload.get("exploratory_group_death_tabulation_authorized") is False:
        return True
    if payload.get(
        "primary_analysis_authorized"
    ) is False and _payload_mentions_outcome(payload):
        return True
    if payload.get("analysis_executed") is False:
        dumped = json.dumps(payload, ensure_ascii=False).lower()
        return "blocked" in dumped and _payload_mentions_outcome(payload)
    return False


def _evidence_record_path(run_dir: Path, record: Mapping[str, Any]) -> Optional[Path]:
    return verified_run_evidence_path(run_dir, record)


def _blocked_outcome_step_ids(
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[str]:
    per_step_records = _step_authority_records(run_dir, per_step_records)
    blocked: set[str] = set()
    for step_id, payload in _authoritative_step_summaries(run_dir, per_step_records):
        if _step_summary_blocks_outcome(dict(payload)):
            blocked.add(step_id)
    if per_step_records is None:
        gate_files = [
            (path.parents[1].name, path)
            for path in sorted(run_dir.glob("steps/*/outputs/*gate*.csv"))
        ]
    else:
        gate_files = []
        for record in (
            current_run_evidence_records(
                run_dir,
                per_step_records=per_step_records,
            )
            or []
        ):
            basename = Path(str(record.get("relative_path") or "")).name
            basename = basename.split("__", 1)[-1].lower()
            if "gate" not in basename or not basename.endswith(".csv"):
                continue
            path = _evidence_record_path(run_dir, record)
            step_id = str(record.get("produced_by_step") or "").strip()
            if path is None and step_id:
                # A current gate record whose registered bytes disappeared or
                # changed cannot authenticate an open outcome-analysis state.
                blocked.add(step_id)
            elif path is not None and step_id:
                gate_files.append((step_id, path))
    for step_id, path in gate_files:
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_text = json.dumps(row, ensure_ascii=False).lower()
            false_authorization = any(
                str(value).strip().lower() == "false"
                and any(token in str(key).lower() for token in ("author", "execut"))
                for key, value in row.items()
            )
            if (
                str(row.get("status", "")).lower() == "blocked"
                and _OUTCOME_ENDPOINT_RE.search(row_text)
                and (false_authorization or "blocked" in row_text)
            ):
                blocked.add(step_id)
                break
    return sorted(blocked)


def _blocked_outcome_manuscript_leaks(manuscript_text: str) -> List[str]:
    leaks: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", manuscript_text or ""):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        natural = re.sub(r"<!--.*?-->", "", sentence)
        natural = re.sub(r"\[[^\]]+\]\([^)]+\)", "", natural)
        natural = re.sub(r"\{\[[^\]]+\]\([^)]+\)\}", "", natural)
        natural = re.sub(r"\s+", " ", natural).strip()
        if _OUTCOME_SAFE_BLOCK_RE.search(natural):
            continue
        has_specific_endpoint = bool(_SPECIFIC_OUTCOME_RE.search(natural))
        has_generic_endpoint = bool(_OUTCOME_ENDPOINT_RE.search(natural))
        if not has_generic_endpoint:
            continue
        inference = bool(_OUTCOME_INFERENCE_RE.search(natural))
        grouped = bool(_OUTCOME_GROUP_RE.search(natural))
        if grouped or (has_specific_endpoint and inference):
            leaks.append(natural[:600])
    return leaks


def _record_artifact_basename(record: Any) -> str:
    return Path(str(record.relative_path)).name.split("__", 1)[-1]


def _source_fingerprints_match(
    evidence: EvidenceStore, metadata: Dict[str, Any]
) -> bool:
    source_ids = metadata.get("source_evidence_ids")
    if isinstance(source_ids, str):
        ids = [source_ids]
    elif isinstance(source_ids, (list, tuple, set)):
        ids = [str(eid) for eid in source_ids if str(eid)]
    else:
        ids = []
    single = metadata.get("source_evidence_id")
    if single and str(single) not in ids:
        ids.append(str(single))
    fingerprints = metadata.get("source_evidence_sha256")
    if not ids or not isinstance(fingerprints, dict) or not fingerprints:
        return False
    for evidence_id in ids:
        source = evidence.get(evidence_id)
        if source is None or fingerprints.get(evidence_id) != source.sha256:
            return False
    return True


def _publication_figure_policy_matches(metadata: Dict[str, Any]) -> bool:
    return (
        metadata.get("figure_skill_policy_version")
        == PUBLICATION_FIGURE_SKILL_POLICY_VERSION
    )


def _run_level_publication_skill_record(record: Any) -> bool:
    return record.producer == "publication_figure_skill" and _record_artifact_basename(
        record
    ).startswith("easyicu_publication_figure.")


_PUBLICATION_FIGURE_VISUAL_ERROR_VALIDATORS = {
    "publication_figure_export",
    "visual_qa",
    "vlm_visual_qa",
}
_COSMETIC_VISUAL_REASON = "svg_text_overlap_spacing"
_LEGACY_COSMETIC_VISUAL_MESSAGE = re.compile(
    r"^svg figure '[^']+' has overlapping text elements; "
    r"multi-panel labels, annotations or axis text need more spacing\.?$",
    re.IGNORECASE,
)
_HARD_VISUAL_MESSAGE = re.compile(
    r"\b(?:blank|clip(?:ped|ping)?|crop(?:ped|ping)?|missing|absent|"
    r"unreadable|overflow|truncat(?:ed|ion)|numeric|mismatch|disagree)\b",
    re.IGNORECASE,
)


def _is_cosmetic_visual_error(finding: ValidationFinding) -> bool:
    """A deterministic SVG text-overlap spacing warning is cosmetic, not a
    manuscript blocker.

    Mirrors ``execution.phase._is_cosmetic_visual_finding`` at the readiness
    layer: the step-level demotion runs during execution, but this exact finding
    is re-generated when the FINAL manuscript SVG is audited, after that pass, so
    it leaks into ``analysis_errors`` / the figure-bundle gate and blocks a run
    whose analysis and evidence are sound. A minor
    multi-panel label/annotation overlap is demoted; genuine visual_qa errors
    (blank/absent figure, wrong content) still block because they do not carry
    the deterministic "overlapping text elements … spacing" signature.
    """
    if finding.severity != "error" or finding.validator != "visual_qa":
        return False
    message = str(finding.message or "").strip()
    if _HARD_VISUAL_MESSAGE.search(message):
        return False
    detail = finding.detail if isinstance(finding.detail, Mapping) else {}
    if str(detail.get("reason") or "").strip() == _COSMETIC_VISUAL_REASON:
        return True
    return _LEGACY_COSMETIC_VISUAL_MESSAGE.fullmatch(message) is not None


def _publication_figure_bundle_ready(
    *,
    evidence: EvidenceStore,
    run_dir: Path,
    findings: Optional[Sequence[ValidationFinding]] = None,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    stems: Dict[str, set[str]] = {}
    source_ready = False
    strict_checkpoint = per_step_records is not None
    active_ids = active_step_evidence_ids(per_step_records or [])

    def _sources_belong_to_current_checkpoint(record: Any) -> bool:
        if not strict_checkpoint:
            return True
        metadata = record.metadata or {}
        raw_ids = metadata.get("source_evidence_ids")
        if isinstance(raw_ids, list):
            source_ids = {str(value) for value in raw_ids if str(value).strip()}
        else:
            value = str(metadata.get("source_evidence_id") or "").strip()
            source_ids = {value} if value else set()
        if source_ids:
            return source_ids <= active_ids
        # Step-produced records were already filtered by
        # current_evidence_records. A run-level publication artifact without
        # source ids cannot prove that it still belongs to this checkpoint.
        return bool(str(record.produced_by_step or "").strip()) or not (
            record.producer == "publication_figure_skill"
            or str(record.evidence_id).startswith("publication_figure_")
        )

    current_records = [
        record
        for record in current_evidence_records(evidence.records(), per_step_records)
        if _sources_belong_to_current_checkpoint(record)
        and verified_run_evidence_path(run_dir, record) is not None
    ]
    verified_ids = {str(record.evidence_id) for record in current_records}

    def _verified_source_ids(metadata: Mapping[str, Any]) -> set[str]:
        raw_ids = metadata.get("source_evidence_ids")
        if isinstance(raw_ids, (list, tuple, set)):
            source_ids = {str(value) for value in raw_ids if str(value).strip()}
        else:
            source_ids = set()
        single = str(metadata.get("source_evidence_id") or "").strip()
        if single:
            source_ids.add(single)
        return source_ids if source_ids and source_ids <= verified_ids else set()

    def _is_publication_contract(record: Any) -> bool:
        metadata = record.metadata or {}
        return (
            record.evidence_id == "publication_figure_contract"
            or str(record.evidence_id).startswith("publication_figure_contract_v")
            or metadata.get("artifact_role") == "figure_contract"
        )

    contract_ready = any(_is_publication_contract(record) for record in current_records)
    visual_errors = [
        finding
        for finding in (findings or [])
        if finding.severity == "error"
        and finding.validator in _PUBLICATION_FIGURE_VISUAL_ERROR_VALIDATORS
        and not _is_cosmetic_visual_error(finding)
    ]
    for record in current_records:
        metadata = record.metadata or {}
        if record.evidence_id.startswith("publication_figure_source_"):
            source_ready = True
        is_contract_record = _is_publication_contract(record)
        if record.kind != "figure":
            if is_contract_record and _verified_source_ids(metadata):
                source_ready = True
            continue
        role = str(metadata.get("figure_role") or "").lower()
        haystack = " ".join(
            [
                record.evidence_id,
                record.description,
                record.relative_path,
                role,
            ]
        ).lower()
        is_explicit_publication = (
            role == "publication_figure"
            or record.evidence_id.startswith("publication_figure_")
            or "publication_figure" in haystack
        )
        if not is_explicit_publication:
            continue
        if _run_level_publication_skill_record(record) and (
            not _source_fingerprints_match(evidence, metadata)
            or not _publication_figure_policy_matches(metadata)
            or not _verified_source_ids(metadata)
        ):
            continue
        if _verified_source_ids(metadata):
            source_ready = True
        path = verified_run_evidence_path(run_dir, record)
        if path is None:
            continue
        stem = path.with_suffix("").name.split("__", 1)[-1]
        stems.setdefault(stem, set()).add(path.suffix.lower().lstrip("."))
    ready_stems = [
        stem
        for stem, suffixes in stems.items()
        if {"svg", "png"} <= suffixes
        or {"svg", "png", "pdf", "tiff"} <= suffixes
        or {"svg", "png", "pdf", "tif"} <= suffixes
    ]
    visual_qa_passed = not visual_errors
    contract_complete = contract_ready and source_ready and visual_qa_passed
    if not contract_complete:
        ready_stems = []
    return {
        "publication_figure_bundle_ready": bool(ready_stems) and contract_complete,
        "publication_figure_stems": sorted(stems),
        "publication_ready_stems": sorted(ready_stems),
        "publication_figure_contract_ready": contract_ready,
        "publication_figure_source_data_ready": source_ready,
        "publication_figure_visual_qa_passed": visual_qa_passed,
        "publication_figure_visual_qa_error_count": len(visual_errors),
        "publication_figure_visual_qa_errors": [
            {"validator": finding.validator, "message": finding.message}
            for finding in visual_errors
        ],
    }


def _publication_provenance_ready(
    *,
    evidence: EvidenceStore,
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Verify every declared raw/cohort source has a digest before publication.

    Analysis may remain useful when a very large or temporarily unavailable raw
    source could not be hashed.  Publication readiness is stricter: a missing,
    unreadable, or size-capped source must not be represented by ``sha256=None``
    and silently treated as reproducible.
    """

    records = [
        record
        for record in current_evidence_records(evidence.records(), per_step_records)
        if str(record.evidence_id) == "provenance_sources"
    ]
    invalid_sources: List[Dict[str, Any]] = []
    if len(records) != 1:
        return {
            "publication_provenance_ready": False,
            "publication_provenance_invalid_sources": [],
            "publication_provenance_error": (
                "missing_provenance_evidence"
                if not records
                else "ambiguous_provenance_evidence"
            ),
        }
    path = verified_run_evidence_path(run_dir, records[0])
    if path is None:
        return {
            "publication_provenance_ready": False,
            "publication_provenance_invalid_sources": [],
            "publication_provenance_error": "unverified_provenance_evidence",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = None
    raw_sources = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(raw_sources, list) or not raw_sources:
        return {
            "publication_provenance_ready": False,
            "publication_provenance_invalid_sources": [],
            "publication_provenance_error": "invalid_provenance_payload",
        }
    for index, raw in enumerate(raw_sources):
        digest = str(raw.get("sha256") or "") if isinstance(raw, Mapping) else ""
        if isinstance(raw, Mapping) and re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        invalid_sources.append(
            {
                "index": index,
                "relative_path": (
                    str(raw.get("relative_path") or "")
                    if isinstance(raw, Mapping)
                    else ""
                ),
                "skipped_reason": (
                    str(raw.get("skipped_reason") or "missing_sha256")
                    if isinstance(raw, Mapping)
                    else "invalid_source_record"
                ),
            }
        )
    return {
        "publication_provenance_ready": not invalid_sources,
        "publication_provenance_invalid_sources": invalid_sources,
        "publication_provenance_error": (
            None if not invalid_sources else "unhashed_declared_sources"
        ),
    }


_STEP_ID_IN_MESSAGE_PATTERNS = (
    # Matches the in-message tokens written by every execution.phase
    # ValidationFinding site that references a specific step. See
    # ``_step_id_referenced_in_finding`` for the full taxonomy.
    re.compile(r"\bfor step\s+([A-Za-z0-9_./-]+)"),
    re.compile(r"\bstep\s+([A-Za-z0-9_./-]+)\s+(?:was|failed|skipped|blocked)"),
)


def _step_id_referenced_in_finding(finding: ValidationFinding) -> Optional[str]:
    """Return the step_id a finding ties to, if any.

    Priority order:

    1. ``finding.detail["step_id"]`` (preferred — set by new
       step-tied finding sites going forward).
    2. Regex scan of ``finding.message`` for the canonical
       ``"for step <id>"`` / ``"step <id> failed"`` phrasings used
       across :mod:`execution.phase`. This catches the existing
       ~14 historical ValidationFinding sites without requiring a
       sweeping rewrite at every emit point.

    Returns ``None`` when no step_id reference is found, which means
    the finding is global (e.g. "no manuscript generated") and should
    NOT be superseded by per-step success.
    """
    if finding.detail and isinstance(finding.detail.get("step_id"), str):
        return finding.detail["step_id"]
    message = finding.message or ""
    for pattern in _STEP_ID_IN_MESSAGE_PATTERNS:
        m = pattern.search(message)
        if m:
            return m.group(1)
    return None


_LEGACY_UNSCOPED_STEP_VALIDATOR_METHODS: Dict[str, frozenset[str]] = {
    # These validators are emitted exclusively by the exact, non-figure
    # cohort-definition-sensitivity result contract.  Older checkpoints did
    # not include ``detail.step_id``; retain this closed migration registry so
    # a repaired step can supersede those historical findings without turning
    # unrelated run-level robustness warnings into step-owned state.
    "robustness_spec_lock": frozenset({"cohort_definition_sensitivity"}),
    "robustness_executed_result": frozenset({"cohort_definition_sensitivity"}),
    "robustness_cohort_membership": frozenset({"cohort_definition_sensitivity"}),
}

# Older checkpoints persisted this per-step deterministic finding before the
# orchestrator attached ``detail.step_id`` / attempt coordinates. It is safe to
# retire only when every current step record is successful: modern findings
# are scoped at emission, and a partially failing run must keep the ambiguous
# legacy error active.
_LEGACY_UNSCOPED_ALL_STEPS_CLEAN_VALIDATORS = frozenset({"mechanical_code_preflight"})


def _normalised_method_head(method: Any) -> str:
    """Return the exact scientific owner from ``<head>_with_<rider>``."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(method or "").strip().lower()).strip(
        "_"
    )
    return normalized.split("_with_", 1)[0]


def _legacy_unscoped_finding_owner_step_id(
    finding: ValidationFinding,
    *,
    plan: Optional[AnalysisPlan],
) -> Optional[str]:
    """Resolve a legacy unscoped finding only when ownership is unambiguous.

    Modern step validators write ``detail.step_id`` directly.  This helper is
    deliberately a closed migration path for old persisted checkpoints: the
    validator must have one registered exact method owner and the current plan
    must contain exactly one matching non-figure result step.  Zero or multiple
    candidates remain active (fail closed).
    """

    if plan is None or _step_id_referenced_in_finding(finding):
        return None
    owner_methods = _LEGACY_UNSCOPED_STEP_VALIDATOR_METHODS.get(
        str(finding.validator or "")
    )
    if not owner_methods:
        return None
    candidates = {
        str(step.step_id)
        for step in (plan.steps or [])
        if str(step.step_id or "")
        and _normalised_method_head(step.method) in owner_methods
        # Mirror the source validator's ownership predicate: it never runs on
        # a step that declares any figure product, including mixed
        # table+figure contracts.
        and not any(
            _output_declares_figure(str(output or ""))
            for output in (step.expected_outputs or [])
        )
    }
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


def _successful_step_ids(per_step_records: Sequence[Dict[str, Any]]) -> set:
    """Step ids whose FINAL recorded status was ``"ok"``.

    Used to identify findings that have been *superseded* by a
    successful retry / resume / deterministic-fallback / replanned
    re-execution. The general rule: if a step ultimately finished
    cleanly, its earlier failure findings (e.g. a 502 from the
    original pre-resume invocation) should not count against the
    final readiness gates — they remain in the manifest for the
    audit trail but are not treated as errors at the report stage.
    """
    return current_successful_step_ids(per_step_records)


def _step_ids_in_records(per_step_records: Sequence[Dict[str, Any]]) -> set:
    """Every step_id that has any per_step_record entry (ok or not).

    A finding whose referenced step_id is NOT in this set must
    come from a step that was *replanned away* — i.e. the replanner
    dropped the failing step and substituted a different step that
    later ran instead. The original failure is no longer part of
    the plan-of-record and shouldn't drag the readiness gate down
    when the final-plan execution otherwise completed cleanly. This
    is the third recognised supersession axis (see
    :func:`_partition_findings_by_supersession`).
    """
    return {
        str(rec.get("step_id"))
        for rec in per_step_records
        if isinstance(rec, dict) and rec.get("step_id")
    }


_GATE_STATE_SUPERSESSION_PATTERNS = (
    # Common "we skipped X because gate Y did not pass" / "X was not
    # produced because Y failed" findings emitted when a gate was
    # transiently False during the run. If the gate is now True at
    # report time, the finding is stale and should not count.
    (
        "manuscript_gate",
        "execution gate did not pass",
        "execution_complete",
    ),
    (
        "manuscript_gate",
        "manuscript generation skipped",
        "execution_complete",
    ),
    (
        "evidence_bound_writer",
        "strict evidence enforcement blocked manuscript generation",
        "manuscript_bound_clean",
    ),
    (
        "manuscript_numeric_auditor",
        "strict evidence enforcement blocked manuscript generation",
        "manuscript_numeric_bound_clean",
    ),
    (
        "critic_agent",
        "criticagent marked manuscript",
        "manuscript_critique_passed",
    ),
    # A caveat-count finding cannot tell which writer pass it came
    # from, so an earlier pass's "cites records with unresolved
    # manifest caveats" error survives a later clean rewrite (e.g. a
    # resume whose new draft cites only caveat-free records). Gate it
    # on the CURRENT bound text: if the latest manuscript carries no
    # `<!-- warning|error: see manifest -->` comments, the finding is
    # stale; if the latest text still has caveats, it stays active.
    (
        "evidence_bound_writer",
        "unresolved manifest caveats",
        "manuscript_manifest_caveats_clean",
    ),
)


def _is_gate_state_superseded(
    finding: ValidationFinding,
    *,
    gate_state: Dict[str, bool],
) -> bool:
    """True when the finding documents a transient gate-state failure
    that the latest gate snapshot has now resolved.

    For example, ``manuscript_gate`` emits a finding when the writer
    refuses to draft a manuscript because the execution gate was False
    at that moment. If the writer later runs successfully on resume
    and execution_complete becomes True, the original "skipped because
    execution gate did not pass" message is stale: nothing skipped in
    the *final* state of the run.
    """
    validator = finding.validator or ""
    message = (finding.message or "").lower()
    for v_match, msg_substr, gate_key in _GATE_STATE_SUPERSESSION_PATTERNS:
        if v_match == validator and msg_substr.lower() in message:
            if gate_state.get(gate_key):
                return True
    return False


def _manuscript_numeric_bound_clean(manuscript_text: str) -> bool:
    """Return true when the latest bound manuscript has no numeric gap markers."""
    if not manuscript_text:
        return False
    if "Manuscript scaffold not generated" in manuscript_text[:300]:
        return False
    return (
        "<!-- UNTRACED:" not in manuscript_text
        and "<!-- AMBIGUOUS:" not in manuscript_text
        and not _MANIFEST_COMMENT_RE.search(manuscript_text)
    )


_WRITER_FAILURE_RE = re.compile(
    r"\b(?:writer failed|error code|invalid proxy api key|api key|"
    r"connection error|rate limit|authentication)\b",
    flags=re.IGNORECASE,
)

_MANIFEST_COMMENT_RE = re.compile(
    r"<!--\s*(?P<level>warning|error)\s*:\s*see manifest\s*-->",
    flags=re.IGNORECASE,
)


def _manuscript_text_status(manuscript_text: str) -> Dict[str, Any]:
    """Validate that a manuscript file contains a real evidence-bound draft.

    Readiness used to treat any non-placeholder manuscript path as generated.
    That allowed a one-line writer exception such as ``(writer failed: ...)`` or
    an empty bound file to pass downstream gates. The check intentionally stays
    case-neutral: it validates draft substance and evidence binding, not any
    particular benchmark topic, variable, or figure.
    """

    text = str(manuscript_text or "")
    stripped = text.strip()
    errors: List[str] = []
    if not stripped:
        errors.append("manuscript draft is empty")
    head = stripped[:600]
    if "Manuscript scaffold not generated" in head:
        errors.append("manuscript scaffold was not generated")
    if _WRITER_FAILURE_RE.search(head):
        errors.append("manuscript draft contains a writer/runtime failure message")
    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9-]*", stripped))
    if word_count < 8:
        errors.append("manuscript draft has too little prose content")
    if not re.search(r"(?:\]\(evidence/|\{evidence:|\[\^claim_)", stripped):
        errors.append("manuscript draft has no evidence-bound claim links")
    manifest_comment_counts = {"warning": 0, "error": 0}
    for match in _MANIFEST_COMMENT_RE.finditer(stripped):
        level = match.group("level").lower()
        manifest_comment_counts[level] = manifest_comment_counts.get(level, 0) + 1
    if manifest_comment_counts["error"]:
        errors.append(
            "manuscript draft contains "
            f"{manifest_comment_counts['error']} unresolved manifest error comment(s)"
        )
    if manifest_comment_counts["warning"]:
        errors.append(
            "manuscript draft contains "
            f"{manifest_comment_counts['warning']} unresolved manifest warning comment(s)"
        )
    return {
        "manuscript_text_ready": not errors,
        "manuscript_text_errors": errors,
        "manuscript_word_count": word_count,
        "manuscript_manifest_warning_count": manifest_comment_counts["warning"],
        "manuscript_manifest_error_count": manifest_comment_counts["error"],
    }


_PUBLICATION_FIGURE_SUMMARY_RE = re.compile(
    r"^publication_figure_skill_summary(?:_v(?P<version>\d+))?"
    r"__publication_figure_skill_summary\.json$"
)
_PUBLICATION_FIGURE_SUMMARY_ID_RE = re.compile(
    r"^publication_figure_skill_summary(?:_v(?P<version>\d+))?$"
)


def _publication_figure_summary_sort_key(path: Path) -> tuple[int, str]:
    match = _PUBLICATION_FIGURE_SUMMARY_RE.match(path.name)
    if not match:
        return (0, path.name)
    version = int(match.group("version") or "1")
    return (version, path.name)


def _publication_figure_summary_record_sort_key(record: Any) -> tuple[int, str]:
    evidence_id = str(getattr(record, "evidence_id", "") or "")
    match = _PUBLICATION_FIGURE_SUMMARY_ID_RE.match(evidence_id)
    if match:
        return (int(match.group("version") or "1"), evidence_id)
    return _publication_figure_summary_sort_key(
        Path(str(getattr(record, "relative_path", "") or ""))
    )


def _latest_publication_figure_audit_status(
    run_dir: Path,
    *,
    evidence: EvidenceStore,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the latest *authorised* publication-figure audit summary.

    The evidence directory is append-only, so a filename glob cannot establish
    current authority.  Only a registered, digest-valid record whose declared
    sources are still present in the current checkpoint may supersede an older
    visual finding.
    """

    all_records = list(evidence.records())
    summary_records = [
        record
        for record in all_records
        if _PUBLICATION_FIGURE_SUMMARY_ID_RE.match(str(record.evidence_id))
        or _PUBLICATION_FIGURE_SUMMARY_RE.match(Path(record.relative_path).name)
    ]
    if not summary_records:
        return None
    # Never fall back from a newer malformed/stale/tampered audit to an older
    # clean one.  The newest registered attempt owns the audit state and fails
    # closed when it cannot prove authority.
    latest_record = max(
        summary_records,
        key=_publication_figure_summary_record_sort_key,
    )

    current_records = [
        record
        for record in current_evidence_records(all_records, per_step_records)
        if verified_run_evidence_path(run_dir, record) is not None
    ]
    current_by_id = {str(record.evidence_id): record for record in current_records}
    current_ids = set(current_by_id)
    record = current_by_id.get(str(latest_record.evidence_id))
    path = verified_run_evidence_path(run_dir, latest_record)
    if (
        record is None
        or path is None
        or not _PUBLICATION_FIGURE_SUMMARY_RE.match(path.name)
    ):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    findings = payload.get("audit_findings")
    if not isinstance(findings, list):
        return None
    raw_source_ids = payload.get("source_evidence_ids")
    source_ids = (
        {str(value) for value in raw_source_ids if str(value).strip()}
        if isinstance(raw_source_ids, list)
        else set()
    )
    if not source_ids or not source_ids <= current_ids:
        return None
    contract_id = str(payload.get("contract_evidence_id") or "").strip()
    raw_figure_ids = payload.get("figure_evidence_ids")
    figure_ids = (
        {str(value) for value in raw_figure_ids if str(value).strip()}
        if isinstance(raw_figure_ids, list)
        else set()
    )
    contract_record = current_by_id.get(contract_id)
    if (
        payload.get("generated") is not True
        or not contract_id
        or not figure_ids
        or contract_record is None
        or not figure_ids <= current_ids
        or not (
            contract_id == "publication_figure_contract"
            or contract_id.startswith("publication_figure_contract_v")
            or (contract_record.metadata or {}).get("artifact_role")
            == "figure_contract"
        )
        or any(
            str(current_by_id[figure_id].kind) != "figure" for figure_id in figure_ids
        )
    ):
        return None
    errors = [
        item
        for item in findings
        if isinstance(item, dict) and item.get("severity") == "error"
    ]
    return {
        "path": str(path),
        "evidence_id": str(record.evidence_id),
        "error_count": len(errors),
        "errors": errors,
    }


def _finding_references_publication_figure(finding: ValidationFinding) -> bool:
    detail_text = ""
    if finding.detail:
        try:
            detail_text = json.dumps(finding.detail, ensure_ascii=False, default=str)
        except Exception:
            detail_text = str(finding.detail)
    haystack = " ".join(
        [
            finding.validator or "",
            finding.message or "",
            " ".join(finding.evidence_ids or []),
            detail_text,
        ]
    ).lower()
    return "publication_figure" in haystack or "easyicu_publication_figure" in haystack


def _is_publication_figure_audit_superseded(
    finding: ValidationFinding,
    *,
    latest_publication_audit: Optional[Dict[str, Any]],
) -> bool:
    """True when a newer publication-figure audit has resolved an older error."""
    if not latest_publication_audit:
        return False
    if latest_publication_audit.get("error_count") != 0:
        return False
    if finding.severity != "error":
        return False
    if finding.validator not in {
        "publication_figure_export",
        "figure_contract_quality",
        "visual_qa",
        "vlm_visual_qa",
    }:
        return False
    return _finding_references_publication_figure(finding)


# ---------------------------------------------------------------------------
# Primary-result plausibility gate ("table == reality").
#
# The value-level numeric auditor verifies manuscript-number == table-number,
# but never table-number == reality. A generated primary table can be
# internally consistent yet physically impossible — e.g. a positional column
# swap that makes the Cox "event" column the sum of ages, yielding 4.6M events
# for 73k stays and a 0h median follow-up. The manuscript then faithfully
# transcribes the garbage and every value-level gate passes. This gate flags
# ONLY values that cannot occur for ANY question (never a question-specific
# direction, threshold, or magnitude), keeping shared gates case-neutral.
# ---------------------------------------------------------------------------
def _partition_findings_by_supersession(
    findings: Sequence[ValidationFinding],
    *,
    success_step_ids: set,
    latest_attempt_ids: Optional[Dict[str, str]] = None,
    known_attempt_ids: Optional[Dict[str, set[str]]] = None,
    plan: Optional[AnalysisPlan] = None,
    known_step_ids: Optional[set] = None,
    gate_state: Optional[Dict[str, bool]] = None,
    latest_publication_audit: Optional[Dict[str, Any]] = None,
) -> tuple[List[ValidationFinding], List[ValidationFinding]]:
    """Split findings into (active, superseded).

    A finding is *superseded* when one of:

    1. It carries a step_id reference (via ``detail.step_id`` or a
       recognised message pattern) and that step_id is in
       ``success_step_ids`` — the step ultimately succeeded so its
       earlier failure findings are stale.
    2. It explicitly references a step_id that is no longer in the plan-of-
       record (i.e. not in ``known_step_ids``). This is the
       "replanned away" axis: the replanner dropped the failing
       step and the substitute step ran instead. The original
       failure is detached from the final plan and shouldn't count
       against the final gate. ``known_step_ids`` is the union of
       every step_id present in ``per_step_records`` (ok or not).
       When ``known_step_ids`` is None, this axis is skipped.
    3. It documents a transient gate-state failure (e.g.,
       ``manuscript_gate`` complaining the execution gate had not
       passed) and the corresponding gate is now True in
       ``gate_state``.
    4. It is an older publication-figure audit error and the latest
       versioned publication-figure skill summary has no current
       errors. This keeps repaired figure exports from being blocked
       by stale resume-era QA findings while preserving the historical
       finding in the audit trail.
    5. A legacy persisted finding lacks a step id, but its validator belongs to
       the closed migration registry and the current plan has exactly one
       matching non-figure owner step and no unresolved explicitly scoped error
       from the same validator.  It is then treated exactly like an explicitly
       scoped finding.  Ambiguous or currently failing ownership remains active.
    6. A pre-scope mechanical-code finding is unambiguously historical because
       every current step record is successful. Any partially failing current
       run keeps the unscoped finding active.

    The classification is purely deterministic — same inputs always
    yield the same partition. The superseded set is returned
    alongside the active set so the manifest can record both for
    audit traceability.
    """
    gate_state = gate_state or {}
    latest_attempt_ids = latest_attempt_ids or {}
    known_attempt_ids = known_attempt_ids or {}
    active: List[ValidationFinding] = []
    superseded: List[ValidationFinding] = []
    # Precompute this across the whole batch so legacy classification is
    # deterministic regardless of finding order.  A scoped error on a known,
    # not-yet-successful step is current authority for that validator family;
    # while it exists, do not retire an unscoped historical sibling merely
    # because another owner step succeeded.
    unresolved_scoped_error_validators = {
        str(f.validator or "")
        for f in findings
        if f.severity == "error"
        and (sid := _step_id_referenced_in_finding(f)) is not None
        and sid not in success_step_ids
        and (known_step_ids is None or sid in known_step_ids)
    }
    for f in findings:
        explicit_sid = _step_id_referenced_in_finding(f)
        if (
            not explicit_sid
            and f.severity == "error"
            and str(f.validator or "") in _LEGACY_UNSCOPED_ALL_STEPS_CLEAN_VALIDATORS
            and str(f.validator or "") not in unresolved_scoped_error_validators
            and known_step_ids
            and set(known_step_ids).issubset(success_step_ids)
        ):
            superseded.append(f)
            continue
        legacy_sid = None
        if (
            not explicit_sid
            and str(f.validator or "") not in unresolved_scoped_error_validators
        ):
            legacy_sid = _legacy_unscoped_finding_owner_step_id(f, plan=plan)
        sid = explicit_sid or legacy_sid
        if sid:
            if sid in success_step_ids:
                finding_attempt_id = str(
                    (f.detail or {}).get("attempt_id") or ""
                ).strip()
                latest_attempt_id = str(latest_attempt_ids.get(sid) or "").strip()
                if f.severity == "error" and finding_attempt_id:
                    # A current-attempt ERROR can never be hidden merely because
                    # an inconsistent outer record says ``ok``.  Retire only a
                    # finding whose attempt is a known, older ledger entry.
                    known_for_step = known_attempt_ids.get(sid, set())
                    if (
                        not latest_attempt_id
                        or finding_attempt_id == latest_attempt_id
                        or finding_attempt_id not in known_for_step
                    ):
                        active.append(f)
                        continue
                superseded.append(f)
                continue
            if (
                explicit_sid
                and known_step_ids is not None
                and sid not in known_step_ids
            ):
                # Step was replanned away — its failure is no longer
                # part of the plan-of-record.
                superseded.append(f)
                continue
        if _is_gate_state_superseded(f, gate_state=gate_state):
            superseded.append(f)
            continue
        if _is_publication_figure_audit_superseded(
            f,
            latest_publication_audit=latest_publication_audit,
        ):
            superseded.append(f)
            continue
        active.append(f)
    return active, superseded


# Primary scientific analyses are agent-owned.  The empty set remains for
# backward-compatible inspection of legacy records, not live dispatch.
_PRIMARY_DETERMINISTIC_RUNNERS: frozenset[str] = frozenset()


def _deterministic_primary_estimate_bound(per_step_records: Any) -> bool:
    """Inspect legacy records without granting retired runners primary ownership.

    The live capability registry has no deterministic primary runner.  This
    compatibility predicate therefore stays false even when an old run record
    carries a historical runner marker and a finite estimate.
    """
    from ..robustness.primary_effect import (
        _extract_primary_effect_payload_from_records,
    )

    payload = _extract_primary_effect_payload_from_records(per_step_records or [])
    if not payload or payload.get("primary_or") is None:
        return False
    step_id = str(payload.get("step_id") or "")
    for record in per_step_records or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("step_id") or "") == step_id:
            return (
                record.get("deterministic_standard_analysis")
                in _PRIMARY_DETERMINISTIC_RUNNERS
            )
    return False


def _replan_budget_demotes(
    *,
    hit: bool,
    execution_complete: bool,
    has_failed_steps: bool,
    has_base_errors: bool,
    evidence_complete: bool,
    numeric_verified: bool,
    primary_estimate_bound: bool,
    no_deterministic_primary_expected: bool = False,
) -> bool:
    """Outcome-aware replan-budget rule.

    Reaching the replan cap demotes the run to ``diagnostic_only`` ONLY if it did
    not otherwise converge. A run that reached ``execution_complete`` with zero
    failed steps, a clean + numeric-verified manuscript, and no other hard errors
    is churny-but-successful. A cap hit on an unresolved run still fails closed.

    ``no_deterministic_primary_expected`` is derived from the capability
    registry. Primary science is agent-owned, so a clean run is not penalized
    merely because no auxiliary renderer can bind an estimand. If family
    inference fails, the conservative default remains false.
    """
    if not hit:
        return False
    has_publishable_primary = (
        primary_estimate_bound or no_deterministic_primary_expected
    )
    converged_clean = (
        execution_complete
        and not has_failed_steps
        and not has_base_errors
        and evidence_complete
        and numeric_verified
        and has_publishable_primary
    )
    return not converged_clean


def _plan_truncation_status(
    findings: Sequence[ValidationFinding],
    *,
    plan: Optional[AnalysisPlan] = None,
) -> Dict[str, Any]:
    """Report whether the plan the run executed still lacks what it planned.

    The truncation finding names the expected outputs it dropped. Repeating
    them here keeps the reason a reader needs — "no calibration figure" — next
    to the gate it blocks, rather than buried in one warning among hundreds.

    The cap runs on the initial plan *and* on every replanner revision, and all
    of their findings accumulate in one list. Asking "was anything ever
    truncated" would therefore latch. Recovery is nevertheless step-bound: a
    product with the same display name on another step cannot prove that the
    dropped scientific role came back. New findings carry exact step/product
    contracts; legacy findings must at least recover every dropped step id and
    every named product.

    A truncation finding that named no outputs cannot be shown to have been
    repaired, and counts as unresolved. Being unable to prove a loss was
    recovered is not evidence that it was.
    """

    declared: set[str] = set()
    declared_by_step: dict[str, StepRecoverySignature] = {}
    for step in (plan.steps if plan is not None else None) or ():
        signature = StepRecoverySignature.from_step(step)
        declared.update(signature.expected_outputs)
        if signature.step_id:
            declared_by_step[signature.step_id] = signature

    unresolved: list[str] = []
    unresolved_step_ids: list[str] = []
    recorded = False
    unnamed = False
    for finding in findings:
        detail = getattr(finding, "detail", None) or {}
        if not isinstance(detail, Mapping) or not detail.get("plan_truncated"):
            continue
        recorded = True
        step_products = detail.get("dropped_step_products")
        if isinstance(step_products, list) and step_products:
            for contract in step_products:
                if not isinstance(contract, Mapping):
                    unnamed = True
                    continue
                step_id = str(contract.get("step_id") or "").strip()
                role = str(contract.get("planned_analysis_role") or "").strip()
                method = str(contract.get("method") or "").strip()
                expected = [
                    str(output).strip()
                    for output in contract.get("expected_outputs") or ()
                    if str(output).strip()
                ]
                current = declared_by_step.get(step_id)

                raw_signature = contract.get("recovery_signature")
                if raw_signature is not None:
                    try:
                        required_signature = StepRecoverySignature.model_validate(
                            raw_signature
                        )
                    except (TypeError, ValueError):
                        unnamed = True
                        required_signature = None
                    recorded_digest = str(
                        contract.get("recovery_signature_sha256") or ""
                    ).strip()
                    signature_valid = (
                        required_signature is not None
                        and recorded_digest == required_signature.canonical_digest()
                        and step_id == required_signature.step_id
                        and role == required_signature.planned_analysis_role
                        and method == required_signature.method
                        and tuple(sorted(expected))
                        == required_signature.expected_outputs
                    )
                    if not signature_valid or current != required_signature:
                        if step_id and step_id not in unresolved_step_ids:
                            unresolved_step_ids.append(step_id)
                        for text in expected:
                            if text not in unresolved:
                                unresolved.append(text)
                    continue

                # Legacy findings predate the structured recovery signature.
                # Retain the prior shell + method comparison so archived runs
                # remain readable; new findings always take the stricter path.
                role_matches = current is not None and (
                    not role or current.planned_analysis_role == role
                )
                method_matches = current is not None and (
                    not method or current.method == method
                )
                if not step_id or not expected:
                    unnamed = True
                if not role_matches or not method_matches:
                    if step_id and step_id not in unresolved_step_ids:
                        unresolved_step_ids.append(step_id)
                    for text in expected:
                        if text not in unresolved:
                            unresolved.append(text)
                    continue
                for text in expected:
                    if text not in current.expected_outputs and text not in unresolved:
                        unresolved.append(text)
            continue

        named = [
            str(output).strip()
            for output in detail.get("dropped_expected_outputs") or ()
            if str(output).strip()
        ]
        dropped_step_ids = [
            str(step_id).strip()
            for step_id in detail.get("dropped_step_ids") or ()
            if str(step_id).strip()
        ]
        missing_step_ids = [
            step_id for step_id in dropped_step_ids if step_id not in declared_by_step
        ]
        if not named or not dropped_step_ids:
            unnamed = True
        for step_id in missing_step_ids:
            if step_id not in unresolved_step_ids:
                unresolved_step_ids.append(step_id)
        for text in named:
            if (
                (missing_step_ids or text not in declared)
                and text not in unresolved
            ):
                unresolved.append(text)
    return {
        # Retained either way: the audit trail must show the run was capped
        # even when a later revision recovered every product.
        "plan_truncation_recorded": recorded,
        "plan_truncated": bool(unresolved or unresolved_step_ids) or unnamed,
        "plan_truncated_dropped_outputs": unresolved,
        "plan_truncated_dropped_step_ids": unresolved_step_ids,
    }


def _compute_readiness_gates(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    per_step_records: Sequence[Dict[str, Any]],
    findings: Sequence[ValidationFinding],
    evidence: EvidenceStore,
    run_dir: Path,
    manuscript_path: Path,
    stop_after_analysis: bool,
    writer_probe_mode: bool = False,
    writer_probe_failed_steps: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    execution = execution_gate_status(
        plan=plan, per_step_records=per_step_records, run_dir=run_dir
    )
    manuscript_text = ""
    if manuscript_path.exists():
        try:
            manuscript_text = manuscript_path.read_text(encoding="utf-8")
        except Exception:
            manuscript_text = ""
    missing_evidence_count = _count_missing_evidence_markers(manuscript_text)
    # General supersession rule: if a step eventually succeeded
    # (status="ok" in per_step_records), any earlier ValidationFinding
    # tied to that step_id (e.g. the original 502 / coder_failed
    # finding from before a successful resume) should not count
    # toward the readiness gates. The full finding list is preserved
    # in the manifest so the audit trail still shows the failure +
    # recovery sequence.
    success_step_ids = _successful_step_ids(per_step_records)
    known_step_ids = _step_ids_in_records(per_step_records)
    known_attempt_ids: Dict[str, set[str]] = {}
    for record in per_step_records:
        if not isinstance(record, dict):
            continue
        step_id = str(record.get("step_id") or "").strip()
        attempt_id = str(record.get("attempt_id") or "").strip()
        if step_id and attempt_id:
            known_attempt_ids.setdefault(step_id, set()).add(attempt_id)
    latest_attempt_ids = {
        str(record.get("step_id") or "")
        .strip(): str(record.get("attempt_id") or "")
        .strip()
        for record in current_step_records(per_step_records)
        if str(record.get("step_id") or "").strip()
        and str(record.get("attempt_id") or "").strip()
    }
    # Compute current gate state once so the gate-state supersession
    # rule sees the final values, not the transient mid-run snapshot
    # the finding was emitted under.
    current_gate_state: Dict[str, bool] = {
        "execution_complete": bool(execution.get("execution_complete")),
        "manuscript_bound_clean": bool(
            manuscript_text
            and "Manuscript scaffold not generated" not in manuscript_text[:300]
            and missing_evidence_count == 0
            and not stop_after_analysis
            and not writer_probe_mode
        ),
        "manuscript_numeric_bound_clean": bool(
            manuscript_text
            and _manuscript_numeric_bound_clean(manuscript_text)
            and not stop_after_analysis
            and not writer_probe_mode
        ),
        "manuscript_manifest_caveats_clean": bool(
            manuscript_text
            and "Manuscript scaffold not generated" not in manuscript_text[:300]
            and not _MANIFEST_COMMENT_RE.search(manuscript_text)
            and not stop_after_analysis
            and not writer_probe_mode
        ),
        "manuscript_critique_passed": False,
    }
    critique_path = run_dir / "manuscript_critique.json"
    if critique_path.exists():
        try:
            critique_payload = json.loads(critique_path.read_text(encoding="utf-8"))
        except Exception:
            critique_payload = {}
        current_gate_state["manuscript_critique_passed"] = (
            isinstance(critique_payload, dict)
            and critique_payload.get("status") == "pass"
        )
    latest_publication_audit = _latest_publication_figure_audit_status(
        run_dir,
        evidence=evidence,
        per_step_records=per_step_records,
    )
    active_findings, superseded_findings = _partition_findings_by_supersession(
        findings,
        success_step_ids=success_step_ids,
        latest_attempt_ids=latest_attempt_ids,
        known_attempt_ids=known_attempt_ids,
        plan=plan,
        known_step_ids=known_step_ids,
        gate_state=current_gate_state,
        latest_publication_audit=latest_publication_audit,
    )
    numeric_errors = [
        f.message
        for f in active_findings
        if f.severity == "error" and f.validator == "manuscript_numeric_auditor"
    ]
    evidence_errors = [
        f.message
        for f in active_findings
        if f.severity == "error"
        and f.validator in {"evidence_bound_writer", "critic_agent"}
    ]
    non_manuscript_errors = [
        f.message
        for f in active_findings
        if f.severity == "error"
        and not _is_cosmetic_visual_error(f)
        and f.validator
        not in {
            "manuscript_numeric_auditor",
            "evidence_bound_writer",
            "critic_agent",
            # Run-level replan-budget latch: collected separately (from the
            # full findings list, not active_findings) so a step-id-shaped
            # replan trigger can never let supersession drop it.
            "replan_budget",
        }
    ]
    # A run that exhausted its replan budget MAY be demoted to diagnostic_only --
    # but the rule is outcome-aware (see the convergence check below). Scan the
    # *full* findings list (not active_findings) so this run-level latch survives
    # step supersession.
    replan_budget_errors = [
        f.message
        for f in findings
        if getattr(f, "validator", "") == "replan_budget"
        and bool((getattr(f, "detail", None) or {}).get("replan_budget_exhausted"))
    ]
    replan_budget_hit = bool(replan_budget_errors)
    blocked_outcome_steps = _blocked_outcome_step_ids(
        run_dir,
        per_step_records,
    )
    blocked_outcome_leaks = (
        _blocked_outcome_manuscript_leaks(manuscript_text)
        if blocked_outcome_steps
        else []
    )
    blocked_outcome_errors = [
        "blocked outcome gate leaked into manuscript: " + leak
        for leak in blocked_outcome_leaks
    ]
    manuscript_text_gate = _manuscript_text_status(manuscript_text)
    manuscript_generated = (
        not writer_probe_mode
        and manuscript_path.exists()
        and manuscript_text_gate["manuscript_text_ready"]
        and not stop_after_analysis
    )
    if (
        manuscript_generated
        and not current_gate_state["manuscript_numeric_bound_clean"]
    ):
        numeric_errors.append(
            "Bound manuscript contains unresolved numeric provenance markers."
        )
    evidence_complete = (
        manuscript_generated and missing_evidence_count == 0 and not evidence_errors
    )
    numeric_verified = manuscript_generated and not numeric_errors
    # `table == reality` gate: a physically-impossible primary result (e.g. a
    # column-swapped Cox table with more events than patients) fails closed even
    # though the value-level numeric auditor — which only checks
    # manuscript-number == table-number — would pass it.
    plausibility_errors = primary_result_plausibility_errors(
        run_dir,
        per_step_records,
    )
    # Integrity: a survival PRIMARY estimand must come from the deterministic Cox
    # runner, never an LLM coder that may silently swap columns.
    survival_integrity_errors = primary_survival_estimate_integrity_errors(
        plan,
        run_dir,
        per_step_records,
    )
    # Reaching the cap demotes the run only when it did not otherwise converge.
    # Clean, complete, numerically verified agent-owned analyses treat the cap as
    # advisory; failed or unresolved runs still fail closed.
    capability_assessment = assess_scientific_capability(
        analysis_type=(plan.analysis_type if plan is not None else None),
        context=context,
    )
    scientific_capability_errors = (
        []
        if plan is None or capability_assessment.publication_eligible
        else [
            f"{capability_assessment.issue_code or 'scientific_capability_unavailable'}: "
            f"{capability_assessment.reason}"
        ]
    )
    base_analysis_errors = (
        non_manuscript_errors
        + blocked_outcome_errors
        + plausibility_errors
        + survival_integrity_errors
        + scientific_capability_errors
    )
    # Primary science is agent-owned across the registry. Fail safe to the strict
    # rule (False) if the family cannot be inferred.
    try:
        from ..planning.study_design import infer_study_design_family

        _no_det_primary_expected = (
            infer_study_design_family(context)
            in families_without_deterministic_primary()
        )
    except Exception:
        _no_det_primary_expected = False
    replan_budget_exhausted = _replan_budget_demotes(
        hit=replan_budget_hit,
        execution_complete=bool(execution["execution_complete"]),
        has_failed_steps=bool(execution["failed_steps"]),
        has_base_errors=bool(base_analysis_errors),
        evidence_complete=bool(evidence_complete),
        numeric_verified=bool(numeric_verified),
        primary_estimate_bound=_deterministic_primary_estimate_bound(per_step_records),
        no_deterministic_primary_expected=_no_det_primary_expected,
    )
    analysis_errors = base_analysis_errors + (
        replan_budget_errors if replan_budget_exhausted else []
    )
    analysis_validated = (
        execution["execution_complete"]
        and execution["step_scientific_requirements_complete"]
        and not analysis_errors
    )
    manuscript_ready = (
        execution["execution_complete"]
        and evidence_complete
        and numeric_verified
        and analysis_validated
    )
    publication = _publication_figure_bundle_ready(
        evidence=evidence,
        run_dir=run_dir,
        findings=active_findings,
        per_step_records=per_step_records,
    )
    publication_provenance = _publication_provenance_ready(
        evidence=evidence,
        run_dir=run_dir,
        per_step_records=per_step_records,
    )
    display_suite = summarize_display_suite_status(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        publication=publication,
        per_step_records=per_step_records,
    )
    article_contract = summarize_article_contract_coverage(
        context=context,
        plan=plan,
        evidence_records=evidence.records(),
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    figure_strategy = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=run_dir,
        per_step_records=per_step_records,
        analysis_family=(
            study_design_family_for_analysis_type(plan.analysis_type)
            if plan is not None and plan.analysis_type is not None
            else None
        ),
    )
    # Read the FULL findings list, not `active_findings`: supersession retires a
    # finding when its step later succeeds, and a dropped step never ran, so no
    # later success can speak for it. What *can* speak for it is a later plan
    # revision that declares the product again — which is why the final plan is
    # passed in rather than the question being answered from findings alone.
    plan_truncation = _plan_truncation_status(findings, plan=plan)
    publication_ready = publication_authorized(
        manuscript_ready=manuscript_ready,
        publication_figure_bundle_ready=publication["publication_figure_bundle_ready"],
        publication_provenance_ready=publication_provenance[
            "publication_provenance_ready"
        ],
        display_suite_complete=display_suite["display_suite_complete"],
        article_contract_complete=article_contract["article_contract_complete"],
        article_figure_strategy_complete=figure_strategy[
            "article_figure_strategy_complete"
        ],
        plan_not_truncated=not plan_truncation["plan_truncated"],
    )
    return {
        **execution,
        **run_completion_axes(
            execution_ok=execution["execution_complete"],
            artifact_valid=evidence_complete,
            scientific_requirement_complete=analysis_validated,
            paper_authorized=publication_ready,
        ),
        "evidence_complete": evidence_complete,
        "numeric_verified": numeric_verified,
        "analysis_validated": analysis_validated,
        "manuscript_ready": manuscript_ready,
        "scientific_capability": capability_assessment.to_dict(),
        "scientific_capability_reportable": capability_assessment.publication_eligible,
        **plan_truncation,
        "replan_budget_exhausted": replan_budget_exhausted,
        "replan_budget_hit": replan_budget_hit,
        "replan_budget_advisory": replan_budget_hit and not replan_budget_exhausted,
        "publication_ready": publication_ready,
        "manuscript_generated": manuscript_generated,
        **manuscript_text_gate,
        "writer_probe_mode": bool(writer_probe_mode),
        **publication_provenance,
        "writer_probe_failed_steps": list(writer_probe_failed_steps or []),
        "missing_evidence_count": missing_evidence_count,
        "numeric_error_count": len(numeric_errors),
        "evidence_error_count": len(evidence_errors),
        "analysis_error_count": len(analysis_errors),
        "numeric_errors": numeric_errors,
        "evidence_errors": evidence_errors,
        "analysis_errors": analysis_errors,
        "blocked_outcome_step_ids": blocked_outcome_steps,
        "blocked_outcome_not_leaked": not blocked_outcome_leaks,
        "blocked_outcome_leak_count": len(blocked_outcome_leaks),
        "blocked_outcome_leaks": blocked_outcome_leaks,
        # Audit-trail surface for the supersession rule (see
        # _partition_findings_by_supersession). Reviewers can inspect
        # which findings the readiness gate ignored because the
        # underlying step ultimately succeeded.
        "superseded_error_count": sum(
            1 for f in superseded_findings if f.severity == "error"
        ),
        "superseded_errors": [
            {"validator": f.validator, "message": f.message}
            for f in superseded_findings
            if f.severity == "error"
        ],
        **publication,
        **display_suite,
        **article_contract,
        **figure_strategy,
    }


def write_readiness_artifacts(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
    evidence: EvidenceStore,
    run_dir: Path,
    manuscript_path: Path,
    stop_after_analysis: bool,
    writer_probe_mode: bool = False,
    writer_probe_failed_steps: Optional[Sequence[str]] = None,
    force_diagnostic_only: bool = False,
    execution_paper_eligible: bool = False,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    gates = _compute_readiness_gates(
        context=context,
        plan=plan,
        per_step_records=per_step_records,
        findings=findings,
        evidence=evidence,
        run_dir=run_dir,
        manuscript_path=manuscript_path,
        stop_after_analysis=stop_after_analysis,
        writer_probe_mode=writer_probe_mode,
        writer_probe_failed_steps=writer_probe_failed_steps,
    )
    # Attempts-to-ready: how many writer passes this run needed before the
    # gates were satisfied. Derived from the always-on audit_log.jsonl event
    # stream (no separate event artifact) so the gate story is quantitative
    # — a run that took 4 writer passes is more fragile than one that took 1.
    gates["writer_attempt_count"] = count_writer_attempts(run_dir)
    gates["forced_diagnostic_only"] = bool(force_diagnostic_only)
    status = _readiness_status(gates)

    # The content gate and execution authority answer different questions.
    # Preserve the former under an explicit name, then bind final paper
    # authorization to the independently constructed execution identity.
    # The default is fail-closed so direct callers cannot accidentally grant
    # paper authority without supplying that identity verdict.
    publication_artifacts_ready = bool(gates.get("paper_authorized")) and (
        status == "publication_ready"
    )
    gates["publication_artifacts_ready"] = publication_artifacts_ready
    gates["execution_paper_eligible"] = bool(execution_paper_eligible)
    gates["paper_authorized"] = publication_artifacts_ready and bool(
        execution_paper_eligible
    )

    artifact_paths: Dict[str, str] = {}

    run_status_path = run_dir / "run_status.json"
    run_status_payload = {
        "schema_version": "easyicu.run_status/2",
        "status": status,
        "strict_fail_closed": True,
        "forced_diagnostic_only": bool(force_diagnostic_only),
        "writer_probe_mode": bool(writer_probe_mode),
        "writer_probe_failed_steps": list(writer_probe_failed_steps or []),
        "research_question": context.research_question,
        # Code identity for quick access without opening the full manifest;
        # the authoritative copy lives in manifest.json's ``code_version``.
        "code_version": capture_code_version(),
        "gates": gates,
        "canonical_outputs": {},
    }
    run_status_path.write_text(
        json.dumps(run_status_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["run_status"] = str(run_status_path.relative_to(run_dir))

    evidence_audit_path = run_dir / "evidence_audit.json"
    evidence_records = evidence.records()
    kinds: Dict[str, int] = {}
    for rec in evidence_records:
        kinds[rec.kind] = kinds.get(rec.kind, 0) + 1
    evidence_audit_payload = {
        "schema_version": "easyicu.evidence_audit/1",
        "evidence_count": len(evidence_records),
        "kinds": kinds,
        "missing_evidence_count": gates["missing_evidence_count"],
        "evidence_complete": gates["evidence_complete"],
        "manuscript_path": (
            str(manuscript_path.relative_to(run_dir))
            if manuscript_path.exists()
            else None
        ),
    }
    evidence_audit_path.write_text(
        json.dumps(evidence_audit_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["evidence_audit"] = str(evidence_audit_path.relative_to(run_dir))

    numeric_audit_path = run_dir / "numeric_audit.json"
    numeric_audit_payload = {
        "schema_version": "easyicu.numeric_audit/1",
        "numeric_verified": gates["numeric_verified"],
        "numeric_error_count": gates["numeric_error_count"],
        "numeric_errors": gates["numeric_errors"],
    }
    numeric_audit_path.write_text(
        json.dumps(numeric_audit_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["numeric_audit"] = str(numeric_audit_path.relative_to(run_dir))

    display_suite_path = run_dir / "display_suite_audit.json"
    display_suite_payload = {
        "schema_version": "easyicu.display_suite_audit/2",
        "display_suite_complete": gates["display_suite_complete"],
        "table_count": gates["display_table_count"],
        "figure_contract_count": gates["display_figure_contract_count"],
        "result_figure_contract_count": gates["display_result_figure_contract_count"],
        "primary_publication_figure_contract_count": gates[
            "display_primary_publication_figure_contract_count"
        ],
        "supporting_figure_contract_count": gates[
            "display_supporting_figure_contract_count"
        ],
        "other_figure_contract_count": gates["display_other_figure_contract_count"],
        "primary_publication_contract_paths": gates[
            "display_primary_publication_contract_paths"
        ],
        "supporting_figure_contract_paths": gates[
            "display_supporting_figure_contract_paths"
        ],
        "other_figure_contract_paths": gates["display_other_figure_contract_paths"],
        "contract_panel_count": gates["display_contract_panel_count"],
        "primary_publication_panel_count": gates[
            "display_primary_publication_panel_count"
        ],
        "supporting_panel_count": gates["display_supporting_panel_count"],
        "contract_role_count": gates["display_contract_role_count"],
        "primary_publication_role_count": gates[
            "display_primary_publication_role_count"
        ],
        "supporting_role_count": gates["display_supporting_role_count"],
        "chart_types": gates["display_chart_types"],
        "primary_publication_chart_types": gates[
            "display_primary_publication_chart_types"
        ],
        "supporting_chart_types": gates["display_supporting_chart_types"],
        "absolute_risk_visual_present": gates["display_absolute_risk_visual_present"],
        "primary_publication_absolute_risk_visual_present": gates[
            "display_primary_publication_absolute_risk_visual_present"
        ],
        "supporting_absolute_risk_visual_present": gates[
            "display_supporting_absolute_risk_visual_present"
        ],
        "primary_publication_result_figure_contract_count": gates[
            "display_primary_publication_result_figure_contract_count"
        ],
        "supporting_result_figure_contract_count": gates[
            "display_supporting_result_figure_contract_count"
        ],
        "categories": gates["display_categories"],
        "table_one_expected": gates["display_table_one_expected"],
        "table_one_present": gates["display_table_one_present"],
        "audit_context_present": gates["display_audit_context_present"],
        "errors": gates["display_suite_errors"],
    }
    display_suite_path.write_text(
        json.dumps(display_suite_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["display_suite_audit"] = str(display_suite_path.relative_to(run_dir))

    article_contract_path = run_dir / "article_contract_audit.json"
    article_contract_payload = article_contract_audit_payload(gates)
    article_contract_path.write_text(
        json.dumps(
            article_contract_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    artifact_paths["article_contract_audit"] = str(
        article_contract_path.relative_to(run_dir)
    )

    figure_strategy_path = run_dir / "article_figure_strategy_audit.json"
    figure_strategy_payload = {
        "schema_version": gates["article_figure_strategy_audit_schema_version"],
        "article_figure_strategy_complete": gates["article_figure_strategy_complete"],
        "analysis_family": gates["article_figure_strategy_family"],
        "archetype": gates["article_figure_strategy_archetype"],
        "hero_role": gates["article_figure_strategy_hero_role"],
        "required_roles": gates["article_figure_strategy_required_roles"],
        "covered_roles": gates["article_figure_strategy_covered_roles"],
        "missing_roles": gates["article_figure_strategy_missing_roles"],
        "chart_types": gates["article_figure_strategy_chart_types"],
        "primary_publication_roles": gates[
            "article_figure_strategy_primary_publication_roles"
        ],
        "primary_publication_chart_types": gates[
            "article_figure_strategy_primary_publication_chart_types"
        ],
        "primary_publication_panel_count": gates[
            "article_figure_strategy_primary_publication_panel_count"
        ],
        "primary_publication_minimum_required_role_count": gates[
            "article_figure_strategy_primary_publication_minimum_required_role_count"
        ],
        "primary_publication_role_panels": gates[
            "article_figure_strategy_primary_publication_role_panels"
        ],
        "minimum_distinct_chart_types": gates[
            "article_figure_strategy_minimum_distinct_chart_types"
        ],
        "role_panels": gates["article_figure_strategy_role_panels"],
        "errors": gates["article_figure_strategy_errors"],
        "strategy": gates["article_figure_strategy"],
    }
    figure_strategy_path.write_text(
        json.dumps(
            figure_strategy_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    artifact_paths["article_figure_strategy_audit"] = str(
        figure_strategy_path.relative_to(run_dir)
    )

    claim_ledger_path = run_dir / "claim_ledger.csv"
    claim_rows = _extract_claim_ledger_rows(
        manuscript_path=manuscript_path,
        gates=gates,
        evidence=evidence,
        per_step_records=per_step_records,
    )
    with claim_ledger_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "claim_id",
                "claim_text",
                "evidence_refs",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(claim_rows)
    artifact_paths["claim_ledger"] = str(claim_ledger_path.relative_to(run_dir))

    author_review_path = run_dir / "author_review_note.md"
    author_review_path.write_text(
        _render_author_review_note(
            status=status,
            gates=gates,
            findings=findings,
            per_step_records=per_step_records,
        ),
        encoding="utf-8",
    )
    artifact_paths["author_review_note"] = str(author_review_path.relative_to(run_dir))

    manuscript_ready_path = run_dir / "manuscript_ready.md"
    if gates["manuscript_ready"] and manuscript_path.exists():
        manuscript_ready_path.write_text(
            manuscript_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        artifact_paths["manuscript_ready"] = str(
            manuscript_ready_path.relative_to(run_dir)
        )
    elif manuscript_ready_path.exists():
        manuscript_ready_path.unlink()

    review_payload, figure_gallery_payload, canonical_figure_paths = (
        build_review_artifact_payloads(run_dir=run_dir, gates=gates)
    )
    review_artifacts_path = run_dir / "review_artifacts.json"
    review_artifacts_path.write_text(
        json.dumps(review_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["review_artifacts"] = str(review_artifacts_path.relative_to(run_dir))
    figure_gallery_path = run_dir / "figure_gallery.json"
    figure_gallery_path.write_text(
        json.dumps(
            figure_gallery_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    artifact_paths["figure_gallery"] = str(figure_gallery_path.relative_to(run_dir))

    run_status_payload["canonical_outputs"] = {
        **artifact_paths,
        **canonical_figure_paths,
    }
    run_status_path.write_text(
        json.dumps(run_status_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    registrations = [
        (
            "run_status",
            "log",
            "Fail-closed run readiness gate summary.",
            run_status_path,
        ),
        (
            "evidence_audit",
            "statistic",
            "Evidence completeness audit for manuscript gating.",
            evidence_audit_path,
        ),
        (
            "numeric_audit",
            "statistic",
            "Numeric-claim audit for manuscript gating.",
            numeric_audit_path,
        ),
        (
            "display_suite_audit",
            "statistic",
            "Article display-suite coverage audit for publication gating.",
            display_suite_path,
        ),
        (
            "claim_ledger",
            "table",
            "Ledger of manuscript claims and evidence links.",
            claim_ledger_path,
        ),
        (
            "article_figure_strategy_audit",
            "log",
            "Article figure-strategy audit for publication gating.",
            figure_strategy_path,
        ),
        (
            "review_artifacts",
            "log",
            "Reviewer-facing artifact manifest with primary and supporting figure tiers.",
            review_artifacts_path,
        ),
        (
            "figure_gallery",
            "log",
            "Reviewer-facing figure gallery with primary and supporting figure tiers.",
            figure_gallery_path,
        ),
        (
            "author_review_note",
            "log",
            "Human-readable fail-closed review note for the run.",
            author_review_path,
        ),
    ]
    if gates["manuscript_ready"] and manuscript_ready_path.exists():
        registrations.append(
            (
                "manuscript_ready",
                "log",
                "Formal manuscript-ready markdown, emitted only after readiness gates pass.",
                manuscript_ready_path,
            )
        )
    for evidence_id, kind, description, path in registrations:
        if evidence.get(evidence_id) is None:
            evidence.register_file(
                kind=kind,
                description=description,
                source_path=path,
                evidence_id=evidence_id,
                aliases=[evidence_id],
                producer="pipeline",
                generation_mode="system",
            )

    return gates, artifact_paths


_MARKDOWN_LINK_TARGET_RE = re.compile(
    r"\[[^\]\r\n]+\]\(\s*(?:<(?P<angle>[^>\r\n]+)>|"
    r"(?P<bare>[^\s)\r\n]+))(?:\s+(?:\"[^\"\r\n]*\"|"
    r"'[^'\r\n]*'|\([^()\r\n]*\)))?\s*\)"
)


def _current_evidence_href_owners(
    *,
    evidence: EvidenceStore,
    per_step_records: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, set[str]]:
    """Return exact href spellings mapped to current evidence owners.

    Markdown labels are presentation text and never carry authority.  A target
    can bind only through an exact current evidence ID, published alias, or
    stored relative path.  Sets deliberately retain collisions so callers can
    fail closed rather than silently apply direct-ID or first-write priority.
    """

    records = evidence.current_verified_records(per_step_records)
    current_ids = {record.evidence_id for record in records}
    owners: Dict[str, set[str]] = {}

    def add(target: str, evidence_id: str) -> None:
        normalized = str(target or "").strip()
        if normalized:
            owners.setdefault(normalized, set()).add(evidence_id)

    for record in records:
        add(record.evidence_id, record.evidence_id)
        add(f"evidence/{record.evidence_id}", record.evidence_id)
        add(record.relative_path, record.evidence_id)
    for alias, evidence_id in evidence.aliases().items():
        if evidence_id not in current_ids:
            continue
        add(alias, evidence_id)
        if "/" not in alias and "\\" not in alias:
            add(f"evidence/{alias}", evidence_id)
    return owners


def _markdown_link_targets(text: str) -> List[str]:
    return [
        str(match.group("angle") or match.group("bare") or "").strip()
        for match in _MARKDOWN_LINK_TARGET_RE.finditer(text)
    ]


def _extract_claim_ledger_rows(
    *,
    manuscript_path: Path,
    gates: Dict[str, Any],
    evidence: EvidenceStore,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, str]]:
    if not manuscript_path.exists():
        return [
            {
                "claim_id": "claim_000",
                "claim_text": "",
                "evidence_refs": "",
                "status": "not_generated",
                "note": "No manuscript file was produced.",
            }
        ]
    text = manuscript_path.read_text(encoding="utf-8", errors="replace")
    if "Manuscript scaffold not generated" in text[:300]:
        return [
            {
                "claim_id": "claim_000",
                "claim_text": "Formal manuscript was not generated.",
                "evidence_refs": "",
                "status": "diagnostic_only",
                "note": "Strict fail-closed gate blocked writer output.",
            }
        ]
    href_owners = _current_evidence_href_owners(
        evidence=evidence,
        per_step_records=per_step_records,
    )
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        link_targets = _markdown_link_targets(stripped)
        evidence_refs: List[str] = []
        unresolved_targets: List[str] = []
        for target in link_targets:
            owners = href_owners.get(target, set())
            if len(owners) == 1:
                evidence_refs.append(next(iter(owners)))
            else:
                unresolved_targets.append(target)
        missing = _count_missing_evidence_markers(stripped)
        if not link_targets and not missing:
            continue
        notes: List[str] = []
        if missing:
            notes.append("Unresolved evidence marker present.")
        if unresolved_targets:
            notes.append(
                "Unresolved or ambiguous evidence href(s): "
                + ";".join(unresolved_targets)
            )
        rows.append(
            {
                "claim_id": f"claim_{len(rows) + 1:03d}",
                "claim_text": re.sub(r"\s+", " ", stripped)[:1000],
                "evidence_refs": ";".join(evidence_refs),
                "status": (
                    "missing_evidence" if missing or unresolved_targets else "bound"
                ),
                "note": " ".join(notes),
            }
        )
    if not rows:
        rows.append(
            {
                "claim_id": "claim_000",
                "claim_text": "",
                "evidence_refs": "",
                "status": "empty",
                "note": (
                    "No evidence-bound claims were detected; manuscript_ready="
                    + str(bool(gates.get("manuscript_ready")))
                ),
            }
        )
    for leak in gates.get("blocked_outcome_leaks") or []:
        rows.append(
            {
                "claim_id": f"claim_{len(rows) + 1:03d}",
                "claim_text": re.sub(r"\s+", " ", str(leak))[:1000],
                "evidence_refs": "",
                "status": "blocked_outcome_leak",
                "note": "Outcome linkage was blocked by a run gate, so this manuscript inference is not authorized.",
            }
        )
    return rows


def _render_author_review_note(
    *,
    status: str,
    gates: Dict[str, Any],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "# Author review note",
        "",
        f"- Status: `{status}`",
        f"- execution_complete: `{gates['execution_complete']}`",
        f"- evidence_complete: `{gates['evidence_complete']}`",
        f"- numeric_verified: `{gates['numeric_verified']}`",
        f"- analysis_validated: `{gates['analysis_validated']}`",
        f"- manuscript_ready: `{gates['manuscript_ready']}`",
        f"- display_suite_complete: `{gates.get('display_suite_complete')}`",
        "- primary_publication_figure_contracts: "
        f"`{gates.get('display_primary_publication_figure_contract_count')}`",
        "- supporting_figure_contracts: "
        f"`{gates.get('display_supporting_figure_contract_count')}`",
        f"- publication_ready: `{gates['publication_ready']}`",
        "",
    ]
    superseded_error_keys = {
        (str(item.get("validator") or ""), str(item.get("message") or ""))
        for item in (gates.get("superseded_errors") or [])
        if isinstance(item, dict)
    }
    failed_steps = gates.get("failed_steps") or []
    missing_steps = gates.get("missing_steps") or []
    if failed_steps or missing_steps:
        lines.extend(["## Blocking step issues", ""])
        for item in failed_steps:
            lines.append(f"- `{item.get('step_id')}` status `{item.get('status')}`")
        for step_id in missing_steps:
            lines.append(f"- `{step_id}` missing execution record")
    if gates.get("blocked_outcome_step_ids") or gates.get("blocked_outcome_leaks"):
        lines.extend(["", "## Blocked outcome gate", ""])
        for step_id in gates.get("blocked_outcome_step_ids") or []:
            lines.append(f"- `{step_id}` blocked outcome linkage or tabulation.")
        for leak in gates.get("blocked_outcome_leaks") or []:
            lines.append(f"- Manuscript leak: {str(leak)[:240]}")
        lines.append("")
    if gates.get("display_suite_errors"):
        lines.extend(["", "## Display suite gate", ""])
        for error in gates.get("display_suite_errors") or []:
            lines.append(f"- {error}")
        lines.append("")
    if gates.get("manuscript_text_errors"):
        lines.extend(["", "## Manuscript text gate", ""])
        for error in gates.get("manuscript_text_errors") or []:
            lines.append(f"- {error}")
        lines.append("")
    error_findings = [
        f
        for f in findings
        if f.severity == "error"
        and (str(f.validator or ""), str(f.message or "")) not in superseded_error_keys
    ]
    superseded_error_findings = [
        f
        for f in findings
        if f.severity == "error"
        and (str(f.validator or ""), str(f.message or "")) in superseded_error_keys
    ]
    if error_findings:
        lines.extend(["## Blocking findings", ""])
        for finding in error_findings:
            lines.append(f"- `{finding.validator}`: {finding.message}")
        lines.append("")
    if superseded_error_findings:
        lines.extend(
            [
                "## Superseded findings",
                "",
                "These findings are retained in the manifest audit trail but do "
                "not block the current readiness gates.",
                "",
            ]
        )
        for finding in superseded_error_findings:
            lines.append(f"- `{finding.validator}`: {finding.message}")
        lines.append("")
    manuscript_text_errors = gates.get("manuscript_text_errors") or []
    if (
        not error_findings
        and not failed_steps
        and not missing_steps
        and not manuscript_text_errors
    ):
        lines.extend(
            [
                "## Review",
                "",
                "No blocking gate failures were detected. Use `manuscript_ready.md` "
                "as the formal draft if present.",
                "",
            ]
        )
    lines.extend(["## Step status", ""])
    for record in per_step_records:
        step_id = record.get("step_id", "")
        if step_id == "00_probe":
            continue
        lines.append(f"- `{step_id}`: `{record.get('status', '?')}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _readiness_status(gates: Mapping[str, Any]) -> str:
    """Return the one fail-closed status ladder shared by artifacts and reports."""

    if gates.get("forced_diagnostic_only") or gates.get("replan_budget_exhausted"):
        return "diagnostic_only"
    if gates.get("publication_ready"):
        return "publication_ready"
    if gates.get("manuscript_ready"):
        return "manuscript_ready"
    if gates.get("execution_complete"):
        return "analysis_only"
    return "diagnostic_only"


_READINESS_STATUS_LABELS = {
    "publication_ready": "PUBLICATION READY",
    "manuscript_ready": "MANUSCRIPT READY",
    "analysis_only": "ANALYSIS ONLY",
    "diagnostic_only": "DIAGNOSTIC ONLY",
}


def render_report(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
    evidence: EvidenceStore,
    aborted_reason: Optional[str] = None,
    paused_after_analysis: bool = False,
    readiness: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    parts.append("# Research-agent results report")
    parts.append("")
    parts.append(f"- Research question: {context.research_question}")
    parts.append(f"- Cohort: {context.cohort.cohort_name} ({context.cohort.database})")
    patient_count = format_patient_count(context.cohort.n_patients)
    parts.append(f"- Stays: {context.cohort.n_stays:,} / Patients: {patient_count}")
    if context.target_outcome:
        parts.append(f"- Target outcome: {context.target_outcome}")
    if context.cross_database_validation:
        parts.append(
            "- Cross-database replication: "
            + ", ".join(context.cross_database_validation)
        )
    parts.append("")

    if aborted_reason:
        parts.append(f"## Status: ABORTED ({aborted_reason})")
        parts.append("")
    elif paused_after_analysis:
        parts.append("## Status: PAUSED AFTER ANALYSIS")
        parts.append("")
        parts.append(
            "The run intentionally stopped before literature retrieval, "
            "manuscript drafting and LaTeX export. Review the registered "
            "tables, figures, statistics and findings before drafting the article."
        )
        parts.append("")
    elif readiness:
        status = _READINESS_STATUS_LABELS[_readiness_status(readiness)]
        parts.append(f"## Status: {status}")
        parts.append("")
        parts.append(
            "- Gates: execution_complete={execution_complete}, "
            "evidence_complete={evidence_complete}, "
            "numeric_verified={numeric_verified}, "
            "analysis_validated={analysis_validated}, "
            "manuscript_ready={manuscript_ready}, "
            "display_suite_complete={display_suite_complete}, "
            "publication_ready={publication_ready}".format(**readiness)
        )
        parts.append("")
        primary_contracts = (
            readiness.get("display_primary_publication_contract_paths") or []
        )
        supporting_contracts = (
            readiness.get("display_supporting_figure_contract_paths") or []
        )
        if primary_contracts or supporting_contracts:
            parts.append("## Figure display tiers")
            parts.append("")
            if primary_contracts:
                parts.append("- Primary publication contracts:")
                parts.extend(f"  - `{path}`" for path in primary_contracts)
            else:
                parts.append("- Primary publication contracts: none")
            if supporting_contracts:
                parts.append("- Supporting step contracts:")
                parts.extend(f"  - `{path}`" for path in supporting_contracts)
            parts.append("")

    if plan:
        parts.append("## Plan")
        parts.append("")
        for s in plan.steps:
            parts.append(f"- **{s.step_id}** — {s.intent}")
        parts.append("")

    if per_step_records:
        parts.append("## Step outcomes")
        parts.append("")
        for r in per_step_records:
            parts.append(
                f"- **{r['step_id']}** — status: `{r.get('status', '?')}`"
                + (f" (rc={r['returncode']})" if "returncode" in r else "")
            )
        parts.append("")

    parts.append("## Findings")
    parts.append("")
    if not findings:
        parts.append("- (no findings recorded)")
    else:
        for f in findings:
            parts.append(f"- `{f.severity}` [{f.validator}] {f.message}")
    parts.append("")

    parts.append("## Evidence (registered artefacts)")
    parts.append("")
    parts.append("| evidence_id | kind | description | sha256 (head) | path |")
    parts.append("|---|---|---|---|---|")
    pipe_escape = "\\|"
    for r in evidence.records():
        desc = r.description.replace("|", pipe_escape)
        parts.append(
            f"| `{r.evidence_id}` | {r.kind} | "
            f"{desc} | `{r.sha256[:10]}…` | `{r.relative_path}` |"
        )
    parts.append("")
    parts.append(textwrap.dedent("""
        ---
        Generated by `easyicu.research_agent.ResearchAgentPipeline`. Every entry
        in the Evidence table is reproducible: rerun the script identified by
        `script_evidence_id` in the manifest, hash the output, and confirm it
        matches the `sha256` recorded here.
    """).strip())
    return "\n".join(parts) + "\n"


__all__ = [
    "execution_gate_status",
    "render_report",
    "write_readiness_artifacts",
]
