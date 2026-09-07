"""Web-owned report-only revisions of approved, completed analysis.

Uses the normal submission/provider authority, but never constructs a pipeline
or probes Docker. Revisions and their failures live outside the sealed source
run; the original scientific gates are not rewritten or promoted.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopLimits,
    validate_provider_transport_reservation_capacity,
)
from easyicu.research_agent.reporting.registered_report_inputs import (
    bind_registered_report_numbers,
    build_registered_report_reader,
    prepare_registered_report_repair,
)
from easyicu.research_agent.reporting.writer_only_migration import (
    WriterOnlyMigrationError,
    publish_writer_only_failure,
    publish_writer_only_result,
    repair_writer_only,
    writer_only_preflight_payload,
)
from easyicu.research_agent.reporting.manuscript_quality import render_reader_manuscript
from easyicu.webserver import agent_pipeline_runs as pipeline_owner
from easyicu.webserver import provider_adapter, run_artifact_disclosure, study_contexts
from easyicu.webserver import dataio


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_SOURCE_SYMLINK",
                detail="Source contains a symbolic link.",
            )
        if path.is_file():
            relative = str(path.relative_to(root)).encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _report_only_limits(limits: ProviderHardStopLimits) -> ProviderHardStopLimits:
    """Narrow the approved budget; a report revision cannot expand it."""
    narrowed = replace(
        limits,
        max_provider_attempts_per_run=min(6, limits.max_provider_attempts_per_run),
        max_provider_attempts_per_batch=min(6, limits.max_provider_attempts_per_batch),
        max_wall_clock_seconds_per_task=min(600, limits.max_wall_clock_seconds_per_task),
    )
    # Preserve approved token ceilings: transports without an enforceable
    # output cap need the existing conservative completion reservation.
    validate_provider_transport_reservation_capacity(narrowed)
    return narrowed


def make_report_only_run_runner(
    *,
    study_context,
    project_root,
    provider,
    provider_environment,
    credential_source,
    execution_resume_source_run_id,
    budget_mode,
    export_path,
    **other: Any,
):
    """Resolve approved source before job creation; revalidate at job start."""

    if budget_mode != "full_reviewed" or any(other.values()):
        raise pipeline_owner.ResearchPipelineRunError(
            "report_only_scope_invalid",
            "Report repair requires an exact reviewed run and no planning amendments.",
        )
    limits = _report_only_limits(
        provider_adapter.web_research_agent_hard_stop_limits(budget_mode)
    )
    target = pipeline_owner._resolve_execution_resume_wrapper(
        study=study_context,
        project_root=project_root,
        source_run_id=execution_resume_source_run_id,
    )
    seed = pipeline_owner.load_recovery_seed(target.wrapper_dir)
    if seed is None:
        raise pipeline_owner.ResearchPipelineRunError(
            "report_only_recovery_required",
            "The exact approved configuration is unavailable.",
        )
    dataio.validate_research_pipeline_source(
        export_path,
        database=(study_context.get("data_source") or {}).get("database"),
        expected_binding=seed.prepared_package_binding,
    )
    approved_config = pipeline_owner._validated_execution_retry_config(
        current_config=None,
        target=target,
        recovery_seed=seed,
        current_scientific_digest=study_contexts.scientific_configuration_sha256(
            study_context
        ),
        prepared_package_binding=seed.prepared_package_binding,
    )
    run_dir = target.wrapper_dir / "pipeline" / target.pipeline_run_id
    prepare_registered_report_repair(run_dir)
    source_digest = _source_fingerprint(run_dir)

    def runner(job):
        from easyicu.research_agent.agents.reporting import WriterAgent
        from easyicu.research_agent.authority.provider_hard_stop import (
            ProviderHardStopLedger,
        )
        from easyicu.research_agent.providers.cost import CostMeter, MeteredClient
        from easyicu.research_agent.providers.hard_stop import HardStopClient

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", str(job.id)):
            raise ValueError("Invalid report revision identifier")
        current = study_contexts.get_context(study_context["id"])
        if (
            study_contexts.scientific_configuration_sha256(current)
            != seed.scientific_configuration_sha256
        ):
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_STUDY_CHANGED",
                detail="Scientific setup changed before repair.",
            )
        if _source_fingerprint(run_dir) != source_digest:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_SOURCE_CHANGED",
                detail="Sealed source changed before repair.",
            )
        prepared = prepare_registered_report_repair(run_dir)
        output = target.wrapper_dir / "report_revisions" / str(job.id)
        if output.parent.is_symlink():
            raise ValueError("Report revision parent must not be a symbolic link")
        output.mkdir(parents=True, exist_ok=False)
        pipeline_owner._write_json(
            output / "preflight.json", writer_only_preflight_payload(prepared)
        )
        ledger_path = output / "runtime" / "provider_hard_stop.json"
        ledger = ProviderHardStopLedger(
            path=ledger_path,
            task_ids=[str(job.id)],
            limits=limits,
            batch_id="report-" + str(job.id),
            declaration_sha256=seed.scientific_configuration_sha256,
        )
        task = ledger.start_task(str(job.id))
        meter = CostMeter(runtime_dir=output / "runtime")
        public_provider = {key: provider.get(key) for key in ("provider", "model")}
        try:
            pipeline_owner._progress(
                job,
                step="report_repair",
                label="Repairing only the manuscript from sealed aggregate evidence; no analysis is being executed",
            )
            client, _ = provider_adapter.build_research_agent_provider_client(
                dict(provider),
                request_timeout=task.cap_timeout(180),
                request_hard_timeout=task.cap_timeout(240),
                environ=provider_environment,
            )
            metered = MeteredClient(
                HardStopClient(client, role="writer", task=task),
                role="writer",
                meter=meter,
                model_override=provider.get("model"),
            )

            class RecordingWriter(WriterAgent):
                attempt = 0

                def _call_section(self, **kwargs):
                    self.attempt += 1
                    section = str(kwargs["section_name"])
                    pipeline_owner._progress(
                        job,
                        step="report_repair",
                        label=f"Repairing report section: {section}",
                    )
                    text = super()._call_section(**kwargs)
                    pipeline_owner._write_json(
                        output
                        / "runtime"
                        / f"writer_candidate_{self.attempt:02d}.json",
                        {"section": section, "instruction": kwargs["instruction"], "text": text},
                    )
                    return text

            writer = RecordingWriter(
                metered,
                language=approved_config.manuscript_language,
                nature_writing_enabled=approved_config.enable_nature_writing_skill,
            )
            result = repair_writer_only(prepared, writer=writer)
            numbered, binding_count = bind_registered_report_numbers(
                run_dir, result.manuscript
            )
            result = replace(
                result,
                manuscript=numbered,
                reader_manuscript=render_reader_manuscript(numbered),
            )
            if _source_fingerprint(run_dir) != source_digest:
                raise WriterOnlyMigrationError(
                    code="WRITER_ONLY_SOURCE_CHANGED",
                    detail="Source changed during repair.",
                )
            receipt = publish_writer_only_result(
                prepared,
                result,
                output_dir=output,
                provider=str(provider.get("provider") or ""),
                model=str(provider.get("model") or ""),
                provider_summary=meter.summary(
                    hard_stop_accounting=task.accounting_summary()
                ),
                provider_ledger=str(ledger_path),
            )
            revision = {
                "schema_version": "easyicu.web-report-revision/1",
                "revision_id": str(job.id),
                "source_run_id": target.pipeline_run_id,
                "status": "pass",
                "source_fingerprint": source_digest,
                "receipt_sha256": hashlib.sha256(
                    (output / "writer_only_migration_receipt.json").read_bytes()
                ).hexdigest(),
                "output_sha256": receipt["output_bound_manuscript_sha256"],
                "numeric_binding_count": binding_count,
                "analysis_steps_executed": 0,
                "claim_ceiling": "analysis_only",
                "publication_authorized": False,
            }
            provenance = build_registered_report_reader(
                run_dir, (output / "manuscript_bound.md").read_text(encoding="utf-8"),
            )
            pipeline_owner._write_json(output / "manuscript_provenance.json", provenance)
            projected = _project_revision(
                target,
                study_context,
                result.reader_manuscript,
                revision,
                public_provider,
                provenance=provenance,
            )
            task.finish(score={"report_quality": "pass", "publication_authorized": False})
            return projected
        except BaseException as exc:
            task.finish(error=type(exc).__name__)
            publish_writer_only_failure(
                prepared,
                output_dir=output,
                error=WriterOnlyMigrationError(
                    code=getattr(exc, "code", "WRITER_ONLY_REPAIR_FAILED"),
                    detail=type(exc).__name__,
                ),
                provider=str(provider.get("provider") or ""),
                model=str(provider.get("model") or ""),
                provider_summary=meter.summary(
                    hard_stop_accounting=task.accounting_summary()
                ),
                provider_ledger=str(ledger_path),
            )
            raise

    return runner


def _project_revision(
    target, study, reader: str, revision: dict, provider: dict, *, provenance: dict
) -> dict:
    """Replace only the mutable Web draft projection, never source run gates."""

    wrapper = target.wrapper_dir
    if (
        provenance.get("schema_version") != "easyicu.manuscript-provenance/1"
        or provenance.get("manuscript_sha256") != revision.get("output_sha256")
        or provenance.get("claim_ceiling") != "analysis_only"
        or provenance.get("publication_authorized") is not False
    ):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_READER_BINDING_FAILED", detail="Reader does not match this revision.",
        )
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text())
    provenance = {**provenance, "report_revision": revision}
    gallery = next((row for row in ledger["artifacts"] if row.get("name") == "figure_gallery.json"), None)
    if gallery:
        provenance["figure_gallery_artifact"] = {"name": "figure_gallery.json", "sha256": gallery["sha256"]}
    draft = json.loads((wrapper / "manuscript_draft.json").read_text())
    draft.update(
        status="report_revision_quality_pass_analysis_only",
        markdown_preview=reader,
        source="registered_report_only_revision",
        report_revision=revision,
        claims=[],
        sentences=[],
        reader=provenance,
    )
    payloads = {"manuscript_draft.json": draft, "manuscript_provenance.json": provenance}
    if not run_artifact_disclosure.scan_browser_projection(payloads)["passed"]:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_PROJECTION_PRIVACY_FAILED",
            detail="Report preview withheld.",
        )
    # Rebuild only reader projections; every scientific gate stays as it
    # was. The revision is a separately identified report, not a new analysis.
    for name, payload in payloads.items():
        pipeline_owner._write_json(wrapper / name, payload)
    ledger["artifacts"] = [
        row for row in ledger["artifacts"] if row.get("name") not in payloads
    ] + [
        pipeline_owner._artifact_record(wrapper / name) for name in payloads
    ]
    pipeline_owner._write_json(wrapper / "evidence_ledger.json", ledger)
    gate = json.loads((wrapper / "quality_gate.json").read_text())["gate"]
    return {
        "run_id": target.pipeline_run_id,
        "study_id": study["id"],
        "mode": "research_agent_pipeline",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "project_dir": str(wrapper),
        "gate": gate,
        "provider": provider,
        "report_revision": revision,
        "artifacts": [
            *ledger["artifacts"],
            pipeline_owner._artifact_record(wrapper / "evidence_ledger.json"),
        ],
        "human_review_pending": False,
        "pending_reviews": [],
        "uploads": 0,
    }
