#!/usr/bin/env python3
"""Build shadow StepResultEnvelope files from an archived EasyICU run.

The source run is read-only.  The required output directory must be outside the
run so this replay cannot change, retrofit, or promote existing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from easyicu.research_agent.audits.envelope_shadow import (
    compare_validator_shadow_inputs,
)
from easyicu.research_agent.audits.envelope_consumers import (
    CrossStepRegisteredOutputEnvelopeDualReader,
    StepSummaryFractionEnvelopeDualReader,
)
from easyicu.research_agent.audits.validators import (
    CrossStepRegisteredOutputValidator,
    StepSummaryFractionValidator,
)
from easyicu.research_agent.execution.result_envelope import (
    StepResultEnvelope,
    normalize_step_result_shadow,
    verify_step_result_envelope,
    write_shadow_step_result_envelope,
)
from easyicu.research_agent.schema import AnalysisStep


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _current_step_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("per_step_records")
    if not isinstance(records, list):
        return []
    current: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("step_id") or "").strip()
        if not step_id:
            continue
        if step_id not in current:
            order.append(step_id)
        current[step_id] = item
    return [current[step_id] for step_id in order]


def _evidence_records_by_id(
    authority: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records = authority.get("records")
    if not isinstance(records, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        if evidence_id in by_id:
            duplicates.add(evidence_id)
        by_id[evidence_id] = item
    for evidence_id in duplicates:
        by_id.pop(evidence_id, None)
    return by_id


def _authorized_path_refs(
    *,
    run_dir: Path,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    digest_cache: dict[Path, str],
) -> dict[str, str]:
    resolved_ids = record.get("resolved_input_evidence_ids")
    if not isinstance(resolved_ids, list):
        return {}
    refs: dict[str, str] = {}
    for raw_id in resolved_ids:
        evidence_id = str(raw_id or "").strip()
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        relative = str(evidence.get("relative_path") or "").strip()
        declared_sha256 = str(evidence.get("sha256") or "").strip()
        if not relative or len(declared_sha256) != 64:
            continue
        source = run_dir / relative
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(run_dir)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if source.is_symlink() or not resolved.is_file():
            continue
        observed_sha256 = digest_cache.get(resolved)
        if observed_sha256 is None:
            observed_sha256 = _sha256_bytes(resolved.read_bytes())
            digest_cache[resolved] = observed_sha256
        if observed_sha256 != declared_sha256:
            continue
        opaque_ref = f"evidence:{evidence_id}@sha256:{declared_sha256}"
        refs[str(resolved)] = opaque_ref
        refs[(PurePosixPath("/easyicu-run") / PurePosixPath(relative)).as_posix()] = (
            opaque_ref
        )
    return refs


def replay_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError("shadow output directory must be outside the source run")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_object(manifest_path)
    authority_path = run_dir / "evidence" / "evidence_authority.json"
    authority = _load_object(authority_path) if authority_path.is_file() else {}
    evidence_by_id = _evidence_records_by_id(authority)
    digest_cache: dict[Path, str] = {}
    records = _current_step_records(manifest)
    if not records:
        raise ValueError("run manifest has no current per-step records")

    index_rows: list[dict[str, Any]] = []
    completed_records: list[dict[str, Any]] = []
    completed_envelopes: dict[str, StepResultEnvelope] = {}
    registered_output_validator = CrossStepRegisteredOutputValidator()
    registered_output_dual_reader = CrossStepRegisteredOutputEnvelopeDualReader()
    fraction_validator = StepSummaryFractionValidator()
    fraction_dual_reader = StepSummaryFractionEnvelopeDualReader()
    for record in records:
        step_id = str(record["step_id"])
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            summary = {}
        record_bytes = _canonical_json_bytes(record)
        summary_bytes = _canonical_json_bytes(summary)
        raw_summary_path = run_dir / "steps" / step_id / "outputs" / "step_summary.json"
        raw_summary_bytes = (
            raw_summary_path.read_bytes() if raw_summary_path.is_file() else None
        )
        source_output_dir = raw_summary_path.parent
        envelope = normalize_step_result_shadow(
            step_id=step_id,
            step_summary=summary,
            output_dir=source_output_dir,
            status=(
                str(record.get("status")).strip()
                if record.get("status") is not None
                else None
            ),
            planned_analysis_role=(
                str(record.get("planned_analysis_role")).strip()
                if record.get("planned_analysis_role") is not None
                else None
            ),
            source_summary_bytes=summary_bytes,
            raw_summary_artifact_bytes=raw_summary_bytes,
            ledger_record_sha256=_sha256_bytes(record_bytes),
            authorized_path_refs=_authorized_path_refs(
                run_dir=run_dir,
                record=record,
                evidence_by_id=evidence_by_id,
                digest_cache=digest_cache,
            ),
        )
        if not verify_step_result_envelope(envelope):
            raise RuntimeError(f"invalid envelope digest for step {step_id}")
        comparison = compare_validator_shadow_inputs(
            step_summary=summary,
            envelope=envelope,
            current_status=(
                str(record.get("status")).strip()
                if record.get("status") is not None
                else None
            ),
        )
        replay_step = AnalysisStep(
            step_id=step_id,
            intent="Replay archived registered-output claims without changing decisions.",
        )
        legacy_registered_output_findings = registered_output_validator.audit(
            step=replay_step,
            step_summary=summary,
            completed_step_records=completed_records,
        )
        canonical_registered_output_findings = registered_output_dual_reader.audit(
            step=replay_step,
            step_summary=summary,
            completed_step_records=completed_records,
            completed_step_envelopes=completed_envelopes,
        )
        legacy_registered_output_payload = [
            finding.model_dump(mode="json")
            for finding in legacy_registered_output_findings
        ]
        canonical_registered_output_payload = [
            finding.model_dump(mode="json")
            for finding in canonical_registered_output_findings
        ]
        registered_output_shadow_exact = (
            legacy_registered_output_payload == canonical_registered_output_payload
        )
        legacy_fraction_findings = fraction_validator.audit(
            step=replay_step,
            step_summary=summary,
        )
        canonical_fraction_findings = fraction_dual_reader.audit(
            step=replay_step,
            step_summary=summary,
            envelope=envelope,
            current_status=(
                str(record.get("status")).strip()
                if record.get("status") is not None
                else None
            ),
        )
        fraction_shadow_exact = [
            finding.model_dump(mode="json") for finding in legacy_fraction_findings
        ] == [
            finding.model_dump(mode="json") for finding in canonical_fraction_findings
        ]
        target_path = output_dir / f"{step_id}.step_result.envelope.json"
        write_shadow_step_result_envelope(
            envelope,
            target_path,
            source_output_dir=source_output_dir,
        )
        index_rows.append(
            {
                "step_id": step_id,
                "status": envelope.status,
                "envelope_path": target_path.name,
                "content_sha256": envelope.content_sha256,
                "artifact_count": len(envelope.artifacts),
                "table_count": len(envelope.tables),
                "statistic_count": len(envelope.statistics),
                "model_diagnostic_count": len(envelope.model_diagnostics),
                "normalization_error_count": sum(
                    issue.severity == "error" for issue in envelope.normalization_issues
                ),
                "normalization_warning_count": sum(
                    issue.severity == "warning"
                    for issue in envelope.normalization_issues
                ),
                "raw_summary_matches_current_record": (
                    envelope.raw_summary_artifact_sha256
                    == envelope.source_summary_sha256
                    if envelope.raw_summary_artifact_sha256 is not None
                    else None
                ),
                "validator_shadow_exact": comparison.exact_match,
                "validator_shadow_mismatch_count": len(comparison.mismatches),
                "registered_output_claim_count": len(
                    registered_output_validator._availability_blocks(summary)
                ),
                "registered_output_legacy_finding_count": len(
                    legacy_registered_output_findings
                ),
                "registered_output_shadow_exact": registered_output_shadow_exact,
                "registered_output_shadow_mismatch_count": (
                    0 if registered_output_shadow_exact else 1
                ),
                "fraction_legacy_finding_count": len(legacy_fraction_findings),
                "fraction_shadow_exact": fraction_shadow_exact,
                "fraction_shadow_mismatch_count": (0 if fraction_shadow_exact else 1),
            }
        )
        completed_records.append(record)
        completed_envelopes[step_id] = envelope
    index = {
        "schema_version": "easyicu.shadow_step_result_index/4",
        "source_manifest_sha256": _sha256_bytes(manifest_bytes),
        "envelope_count": len(index_rows),
        "normalization_error_count": sum(
            row["normalization_error_count"] for row in index_rows
        ),
        "validator_shadow_mismatch_count": sum(
            row["validator_shadow_mismatch_count"] for row in index_rows
        ),
        "registered_output_claim_count": sum(
            row["registered_output_claim_count"] for row in index_rows
        ),
        "registered_output_shadow_mismatch_count": sum(
            row["registered_output_shadow_mismatch_count"] for row in index_rows
        ),
        "fraction_legacy_finding_count": sum(
            row["fraction_legacy_finding_count"] for row in index_rows
        ),
        "fraction_shadow_mismatch_count": sum(
            row["fraction_shadow_mismatch_count"] for row in index_rows
        ),
        "steps": index_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    index_path.write_bytes(_canonical_json_bytes(index))
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        index = replay_run(args.run_dir, args.output_dir)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(index, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
