"""[Layer 5: Evaluation & Submission Scaffold / Layer 4: Evidence & Provenance]
Package/finalise phase for the EasyICU research-agent pipeline.

This module owns run reports, workflow/replay artefacts, manifest writing,
readiness outputs, cost summaries, and final PipelineResult construction.
It follows the same free-function pattern as ``execution/phase.py`` and
``reporting.write_phase``: callers pass the pipeline instance first, and the
function reads existing collaborators from that instance.

Boundary contract: consumes ``_PlanPhaseResult`` + ``_ExecutePhaseResult`` +
``_WritePhaseResult`` and emits ``PipelineResult``. The phase-result
dataclasses live in ``contracts.py``; this module does not own their schema.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

from ..audits.validators import dedupe_findings
from ..cohort.schema import COHORT_LOCK_FILENAME
from ..concept_dict_audit import (
    CONCEPT_DICT_PACKAGE_PATH,
    SOFA2_DICT_PACKAGE_PATH,
    compute_concept_dict_fingerprint,
)
from ..contracts.runtime import (
    ValidationFinding,
    _ExecutePhaseResult,
    _PlanPhaseResult,
    _WritePhaseResult,
)
from ..providers.cost import CostMeter
from ..authority.evidence_store import EvidenceStore
from ..methods.multiple_testing import build_multiple_testing_report
from ..methods.sensitivity import compute_e_value
from ..replication.report import _literature_provenance_note
from ..reporting.readiness import render_report, write_readiness_artifacts
from ..providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files
from ..authority.runtime_artifacts import (
    AuditLogger,
    active_step_evidence_ids,
    build_execution_replay,
    build_workflow_graph,
    capture_code_version,
    current_evidence_records,
    render_workflow_graph_mermaid,
    verified_run_evidence_path,
    write_json_artifact,
    write_run_checkpoint,
)
from ..robustness.panel import PANEL_FILENAME, load_robustness_panel
from ..schema import AnalysisManifest, PipelineResult, ResearchContext
from ..learning.store import quarantine_run_lesson
from ..reporting.side_findings import collect_side_findings, write_side_findings

logger = logging.getLogger(__name__)


def _code_version_manifest_fields() -> Dict[str, Any]:
    """Return the ``code_version`` manifest field (git + package identity)."""
    return {"code_version": capture_code_version()}


def _concept_dictionary_manifest_fields() -> Dict[str, Any]:
    """Return the concept dictionary identity for run manifests."""

    try:
        fingerprint = compute_concept_dict_fingerprint()
    except FileNotFoundError:
        return {
            "concept_dict_path": CONCEPT_DICT_PACKAGE_PATH,
            "concept_dict_sha": None,
            "sofa2_dict_path": SOFA2_DICT_PACKAGE_PATH,
            "sofa2_dict_sha": None,
            "concept_dict_fingerprint": None,
        }
    return {
        "concept_dict_path": fingerprint.concept_dict_path,
        "concept_dict_sha": fingerprint.concept_dict_sha,
        "sofa2_dict_path": fingerprint.sofa2_dict_path,
        "sofa2_dict_sha": fingerprint.sofa2_dict_sha,
        "concept_dict_fingerprint": fingerprint.to_dict(),
    }


def _active_step_evidence_ids(
    per_step_records: List[Dict[str, Any]],
) -> set[str]:
    """Return evidence referenced by the latest checkpoint for each step.

    Evidence blobs are immutable, so a resumed step leaves its prior outputs in
    the store.  The ordered per-step ledger appends the resumed checkpoint; the
    last record for a step is therefore the current execution authority.
    """

    return active_step_evidence_ids(per_step_records)


def _current_verified_semantic_csv(
    *,
    evidence: EvidenceStore,
    per_step_records: List[Dict[str, Any]],
    run_dir: Path,
    semantic_id: str,
) -> Optional[tuple[Any, Path]]:
    """Resolve one current, step-produced CSV for a stable semantic name.

    Evidence aliases are first-write-wins, while a resumed step whose bytes
    change is registered as ``<id>_v2``.  Looking up the stable alias directly
    would therefore silently select the superseded file.  Resolve the semantic
    family first, filter it through the latest successful step ledger, and
    require one digest-valid evidence copy.  Missing, stale, or ambiguous
    authority fails closed.
    """

    all_records = list(evidence.records())
    alias_root = str(evidence.aliases().get(semantic_id) or "").strip()
    family_roots = {semantic_id}
    if alias_root:
        family_roots.add(alias_root)

    candidates: List[tuple[Any, Path]] = []
    for record in current_evidence_records(all_records, per_step_records):
        producer_step = str(getattr(record, "produced_by_step", None) or "").strip()
        if not producer_step:
            continue
        evidence_id = str(getattr(record, "evidence_id", "") or "").strip()
        metadata = getattr(record, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        supersedes = str(metadata.get("resume_supersedes") or "").strip()
        relative_path = Path(str(getattr(record, "relative_path", "") or ""))
        logical_name = relative_path.name.split("__", 1)[-1]
        logical_path = Path(logical_name)
        belongs_to_family = bool(
            evidence_id in family_roots
            or supersedes in family_roots
            or logical_path.stem == semantic_id
        )
        if not belongs_to_family or logical_path.suffix.lower() != ".csv":
            continue
        path = verified_run_evidence_path(run_dir, record)
        if path is not None:
            candidates.append((record, path))

    unique = {str(record.evidence_id): (record, path) for record, path in candidates}
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _register_multiple_testing_outputs(
    *,
    evidence: EvidenceStore,
    csv_path: Path,
    markdown_path: Path,
) -> tuple[str, str]:
    """Register the current O22 outputs, versioning changed resume content."""

    csv_record = evidence.register_file(
        kind="statistic",
        description=(
            "Family-scoped Benjamini–Hochberg and Bonferroni "
            "correction for auditable registered p-values (O22)."
        ),
        source_path=csv_path,
        evidence_id="multiple_testing_report",
        producer="pipeline",
        generation_mode="system",
        on_sha_change="new_id",
    )
    markdown_record = evidence.register_file(
        kind="log",
        description="Human-readable summary of multiple-testing correction (O22).",
        source_path=markdown_path,
        evidence_id="multiple_testing_summary",
        producer="pipeline",
        generation_mode="system",
        on_sha_change="new_id",
    )
    return csv_record.evidence_id, markdown_record.evidence_id


def finalise_success(
    pipeline,
    *,
    plan_result: _PlanPhaseResult,
    execute_result: _ExecutePhaseResult,
    write_result: _WritePhaseResult,
    run_id: str,
    run_dir: Path,
    cohort_path: Path,
    notes: Optional[str],
    database: str,
    target_outcome: Optional[str],
    stop_after_analysis: bool,
    cache_key: Optional[str],
    scientific_identity: Mapping[str, Any],
    experiment_spec_path: Optional[Path],
    audit_logger: Optional[AuditLogger],
    emit_progress: Callable[..., None],
) -> PipelineResult:
    """Write reports/manifests and persist run memory after all phases finish."""
    context = plan_result.context
    evidence = plan_result.evidence
    findings = plan_result.findings
    per_step_records = execute_result.per_step_records
    plan = execute_result.plan

    plan_order = {s.step_id: i for i, s in enumerate(plan.steps)}
    per_step_records.sort(
        key=lambda r: (
            -1
            if r.get("step_id") == "00_probe"
            else plan_order.get(r.get("step_id"), 10**9)
        )
    )
    report_path = run_dir / "results_report.md"
    report_path.write_text(
        render_report(
            context=context,
            plan=plan,
            findings=findings,
            per_step_records=per_step_records,
            evidence=evidence,
            paused_after_analysis=stop_after_analysis,
        ),
        encoding="utf-8",
    )

    workflow_graph = build_workflow_graph(
        run_id=run_id,
        context=context,
        plan=plan,
        per_step_records=per_step_records,
        paused_after_analysis=stop_after_analysis,
    )
    workflow_graph_path = write_json_artifact(
        run_dir / "workflow_graph.json",
        workflow_graph,
    )
    workflow_mermaid_path = run_dir / "workflow_graph.md"
    workflow_mermaid_path.write_text(
        render_workflow_graph_mermaid(workflow_graph),
        encoding="utf-8",
    )
    evidence.register_file(
        kind="log",
        description="Workflow graph JSON for this run.",
        source_path=workflow_graph_path,
        aliases=["workflow_graph"],
        producer="pipeline",
        generation_mode="system",
    )
    evidence.register_file(
        kind="log",
        description="Mermaid workflow graph for this run.",
        source_path=workflow_mermaid_path,
        aliases=["workflow_graph_mermaid"],
        producer="pipeline",
        generation_mode="system",
    )

    replay_bundle = build_execution_replay(
        run_id=run_id,
        cohort_path=cohort_path,
        context_path=str(plan_result.context_path.relative_to(run_dir)),
        plan_path=str(plan_result.plan_path.relative_to(run_dir)),
        llm_signature=plan_result.llm_signature,
        prompt_pack_version=plan_result.prompt_version,
        per_step_records=per_step_records,
        findings=findings,
        evidence_ids=[r.evidence_id for r in evidence.records()],
    )
    replay_path = write_json_artifact(
        run_dir / "execution_replay.json",
        replay_bundle,
    )
    evidence.register_file(
        kind="log",
        description="Deterministic execution replay bundle for this run.",
        source_path=replay_path,
        aliases=["execution_replay"],
        producer="pipeline",
        generation_mode="system",
    )

    audit_log_rel: Optional[str] = None
    if audit_logger is not None and audit_logger.path.exists():
        evidence.register_file(
            kind="log",
            description="RuntimeSupervisor audit log (JSONL).",
            source_path=audit_logger.path,
            aliases=["audit_log"],
            producer="pipeline",
            generation_mode="system",
        )
        audit_log_rel = str(audit_logger.path.relative_to(run_dir))

    cost_records_for_manifest = []
    if plan_result.cost_meter is not None:
        cost_records_for_manifest = list(plan_result.cost_meter.records)
        cost_json_path = run_dir / "cost_records.json"
        cost_json_path.write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in plan_result.cost_meter.records],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        cost_md_path = run_dir / "cost_summary.md"
        cost_md_path.write_text(
            _render_cost_summary(plan_result.cost_meter), encoding="utf-8"
        )
        # Machine-readable aggregate (token totals + estimated USD, by model)
        # so the bench scorer and Fig.3 source-data builder can read cost
        # without re-parsing the markdown or recomputing from raw records.
        cost_summary_json_path = run_dir / "cost_summary.json"
        cost_summary_json_path.write_text(
            json.dumps(
                plan_result.cost_meter.summary(),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        evidence.register_file(
            kind="log",
            description="Raw per-call LLM cost records (T3.2).",
            source_path=cost_json_path,
            evidence_id="cost_records",
            producer="pipeline",
            generation_mode="system",
            # Resume legitimately appends new cost records, so the
            # JSON sha changes between original and resumption; the
            # original record stays canonical, the new one lands
            # alongside as ``cost_records_v2`` for the audit trail.
            on_sha_change="new_id",
        )
        evidence.register_file(
            kind="log",
            description="Human-readable LLM cost summary (T3.2).",
            source_path=cost_md_path,
            evidence_id="cost_summary",
            producer="pipeline",
            generation_mode="system",
            on_sha_change="new_id",
        )

    reproducibility_summary: Optional[Dict[str, Any]] = None
    if plan_result.repro_envelope is not None:
        envelope_path = run_dir / "reproducibility_envelope.json"
        plan_result.repro_envelope.to_disk(envelope_path)
        # The envelope captures per-call prompt/response shas,
        # timestamps, and the env snapshot. On a resumed run the
        # content legitimately differs from the original (new
        # per-call records for the steps we re-executed), so a sha
        # collision is expected. Use ``on_sha_change="new_id"`` so
        # the original envelope keeps its canonical evidence id
        # (still resolvable for citations) and the resume's
        # envelope lands beside it as ``..._v2`` for auditability,
        # instead of crashing the whole run.
        evidence.register_file(
            kind="log",
            description=(
                "LLM reproducibility envelope (O20): per-call prompt/response "
                "sha256, requested seed, temperature, provider/model, and a "
                "PHI-safe environment snapshot."
            ),
            source_path=envelope_path,
            evidence_id="reproducibility_envelope",
            producer="pipeline",
            generation_mode="system",
            on_sha_change="new_id",
        )
        reproducibility_summary = plan_result.repro_envelope.to_manifest_summary()

    # O22 — Multiple-testing correction. Scan registered table /
    # statistic artefacts for auditable raw p-values and adjust them
    # within their declared (or defensible source-local) hypothesis
    # families. Writes a CSV + MD pair and registers both as evidence
    # so the manuscript can cite
    # ``{evidence:multiple_testing_report}``.
    if pipeline._enable_multiple_testing_correction:
        mt_report = build_multiple_testing_report(
            evidence_records=current_evidence_records(
                evidence.records(), per_step_records
            ),
            run_dir=run_dir,
            alpha=pipeline._multiple_testing_alpha,
            active_evidence_ids=_active_step_evidence_ids(per_step_records),
        )
        mt_csv = run_dir / "multiple_testing_report.csv"
        mt_md = run_dir / "multiple_testing_report.md"
        mt_report.write_csv(mt_csv)
        mt_report.write_markdown(mt_md)
        mt_evidence_id, _mt_summary_evidence_id = _register_multiple_testing_outputs(
            evidence=evidence,
            csv_path=mt_csv,
            markdown_path=mt_md,
        )
        summary = mt_report.summary()
        if summary["n_tests"] > 0:
            # Surface the raw → corrected gap as an info finding so
            # paper figures can include it without re-reading the
            # CSV.
            findings.append(
                ValidationFinding(
                    validator="multiple_testing",
                    severity="info",
                    message=(
                        f"Ran family-scoped BH-FDR across {summary['n_tests']} "
                        f"tests in {summary['n_families']} families at "
                        f"alpha={summary['alpha']:.3f}: "
                        f"{summary['n_significant_raw']} significant raw, "
                        f"{summary['n_significant_bh']} after BH, "
                        f"{summary['n_significant_bonferroni']} after Bonferroni."
                    ),
                    evidence_ids=[mt_evidence_id],
                    detail=summary,
                )
            )
            # If raw and BH disagree meaningfully, emit a warning
            # so the Discussion section has to engage with it.
            if summary["n_significant_raw"] > summary["n_significant_bh"]:
                findings.append(
                    ValidationFinding(
                        validator="multiple_testing",
                        severity="warning",
                        message=(
                            "Some raw-significant results did not survive "
                            "BH-FDR within their declared or source-local "
                            "hypothesis families. Revise the primary / "
                            "secondary endpoint distinction or report "
                            "family-scoped corrected p-values explicitly."
                        ),
                        evidence_ids=[mt_evidence_id],
                        detail={
                            "n_raw_only": (
                                summary["n_significant_raw"]
                                - summary["n_significant_bh"]
                            ),
                        },
                    )
                )
        else:
            # An empty, defensibly scoped report is still an audit result. Keep
            # it visible in the manifest instead of making downstream reviewers
            # infer that O22 never ran. Do not relax extraction by promoting an
            # untyped coefficient dump into an invented hypothesis family.
            findings.append(
                ValidationFinding(
                    validator="multiple_testing",
                    severity="info",
                    message=(
                        "Multiple-testing audit completed, but no raw p-values "
                        "with a defensible declared or source-local hypothesis "
                        "family were found; no adjustment was computed."
                    ),
                    evidence_ids=[mt_evidence_id],
                    detail=summary,
                )
            )

    # O23 — E-values. For every primary-association row, compute
    # VanderWeele–Ding E-value + lower-CI E-value. Writes
    # ``e_values.csv`` + ``e_values.md`` and registers both.
    # Baseline prevalence defaults to observed outcome rate when
    # an ``outcome_rate.csv`` was registered.
    try:
        primary_source = _current_verified_semantic_csv(
            evidence=evidence,
            per_step_records=per_step_records,
            run_dir=run_dir,
            semantic_id="primary_association",
        )
        if primary_source is not None:
            import csv as _csv

            primary_record, primary_path = primary_source
            baseline_prev = 0.1
            outcome_rate_source = _current_verified_semantic_csv(
                evidence=evidence,
                per_step_records=per_step_records,
                run_dir=run_dir,
                semantic_id="outcome_rate",
            )
            if outcome_rate_source is not None:
                try:
                    _outcome_rate_record, or_path = outcome_rate_source
                    with or_path.open("r", encoding="utf-8") as fh:
                        for row in _csv.DictReader(fh):
                            for key in (
                                "outcome_rate",
                                "rate",
                                "mortality_rate",
                                "event_rate",
                            ):
                                if key in row:
                                    try:
                                        cand = float(row[key])
                                        if 0 < cand < 1:
                                            baseline_prev = cand
                                    except (TypeError, ValueError):
                                        pass
                except Exception:
                    pass

            rows_out: List[Dict[str, Any]] = []
            with primary_path.open("r", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    # Accept OR or odds_ratio column; skip age / intercept etc.
                    or_val = None
                    for key in ("odds_ratio", "or", "OR"):
                        if key in row and row[key] not in (None, "", "nan"):
                            try:
                                or_val = float(row[key])
                                break
                            except (TypeError, ValueError):
                                continue
                    if or_val is None:
                        continue
                    try:
                        ci_lo = float(row.get("or_lower") or row.get("ci_lower") or 0.0)
                        ci_hi = float(row.get("or_upper") or row.get("ci_upper") or 0.0)
                        ci = (ci_lo, ci_hi) if ci_lo > 0 and ci_hi > 0 else None
                    except (TypeError, ValueError):
                        ci = None
                    ev = compute_e_value(
                        estimate=or_val,
                        ci=ci,
                        estimate_type="or",
                        baseline_prevalence=baseline_prev,
                    )
                    row_out = {
                        "term": row.get("term")
                        or row.get("variable")
                        or row.get("predictor")
                        or "",
                        "odds_ratio": or_val,
                        "ci_lower": ci[0] if ci else "",
                        "ci_upper": ci[1] if ci else "",
                        "baseline_prevalence": baseline_prev,
                        "e_value": ev.e_value,
                        "e_value_lower_bound": ev.e_value_lower_bound,
                        "note": ev.note or "",
                    }
                    rows_out.append(row_out)

            if rows_out:
                ev_csv = run_dir / "e_values.csv"
                ev_md = run_dir / "e_values.md"
                with ev_csv.open("w", newline="", encoding="utf-8") as fh:
                    writer = _csv.writer(fh)
                    writer.writerow(list(rows_out[0].keys()))
                    for row in rows_out:
                        writer.writerow([row[k] for k in rows_out[0].keys()])
                ev_md_lines = [
                    "# E-values for primary effects (O23)",
                    "",
                    f"Baseline event prevalence used: **{baseline_prev:.3f}**",
                    "",
                    "| Term | OR | 95% CI | E-value | E-value (CI bound) |",
                    "|---|---|---|---|---|",
                ]
                for row in rows_out:
                    ci_disp = (
                        f"{row['ci_lower']:.2f} – {row['ci_upper']:.2f}"
                        if row["ci_lower"] != "" and row["ci_upper"] != ""
                        else "—"
                    )
                    ev_md_lines.append(
                        "| {t} | {orv:.2f} | {ci} | {ev:.2f} | {evb} |".format(
                            t=str(row["term"])[:40],
                            orv=row["odds_ratio"],
                            ci=ci_disp,
                            ev=row["e_value"],
                            evb=(
                                f"{row['e_value_lower_bound']:.2f}"
                                if row["e_value_lower_bound"] is not None
                                else "—"
                            ),
                        )
                    )
                ev_md.write_text("\n".join(ev_md_lines) + "\n", encoding="utf-8")
                source_ids = [str(primary_record.evidence_id)]
                if outcome_rate_source is not None:
                    source_ids.append(str(outcome_rate_source[0].evidence_id))
                ev_record = evidence.register_file(
                    kind="statistic",
                    description=(
                        "VanderWeele–Ding E-values for every primary "
                        "effect row (O23)."
                    ),
                    source_path=ev_csv,
                    evidence_id="e_values",
                    inputs=source_ids,
                    producer="pipeline",
                    generation_mode="system",
                    on_sha_change="new_id",
                )
                evidence.register_file(
                    kind="log",
                    description="Human-readable E-value summary (O23).",
                    source_path=ev_md,
                    evidence_id="e_values_summary",
                    inputs=source_ids,
                    producer="pipeline",
                    generation_mode="system",
                    on_sha_change="new_id",
                )
                findings.append(
                    ValidationFinding(
                        validator="e_value",
                        severity="info",
                        message=(
                            f"Computed E-values for {len(rows_out)} primary "
                            f"effect row(s) (baseline prevalence={baseline_prev:.3f})."
                        ),
                        evidence_ids=[ev_record.evidence_id],
                    )
                )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="e_value",
                severity="warning",
                message=f"E-value computation failed: {type(exc).__name__}: {exc}",
            )
        )

    # O24 — Fairness / subgroup analysis. Runs when a
    # ``primary_association`` artefact exists and the cohort has
    # at least one of (``age``, ``sex``, ``sex_M``, ``race``,
    # ``insurance``). Pure numpy; no pandas-only helpers so we
    # stay consistent with the rest of the deterministic layer.
    if pipeline._enable_fairness_subgroups:
        try:
            primary_source = _current_verified_semantic_csv(
                evidence=evidence,
                per_step_records=per_step_records,
                run_dir=run_dir,
                semantic_id="primary_association",
            )
            if primary_source is not None:
                import csv as _csv

                primary_record, primary_path = primary_source
                predictor_name: Optional[str] = None
                outcome_name = context.target_outcome
                with primary_path.open("r", encoding="utf-8") as fh:
                    for row in _csv.DictReader(fh):
                        term = (
                            row.get("term")
                            or row.get("variable")
                            or row.get("predictor")
                            or ""
                        )
                        if term and term.lower() not in {
                            "intercept",
                            "const",
                            "age",
                            "sex_m",
                        }:
                            predictor_name = term
                            break
                cohort_df = pd.read_parquet(cohort_path)
                candidate_subgroups = [
                    col
                    for col in ("age", "sex", "sex_M", "race", "insurance")
                    if col in cohort_df.columns
                ]
                if (
                    predictor_name is not None
                    and outcome_name
                    and outcome_name in cohort_df.columns
                    and candidate_subgroups
                ):
                    from ..methods.fairness import run_subgroup_analysis

                    result = run_subgroup_analysis(
                        cohort_df=cohort_df,
                        predictor=predictor_name,
                        outcome=outcome_name,
                        subgroup_columns=candidate_subgroups,
                    )
                    fair_csv = run_dir / "fairness_subgroups.csv"
                    fair_md = run_dir / "fairness_subgroups.md"
                    result.write_csv(fair_csv)
                    result.write_markdown(fair_md)
                    fair_record = evidence.register_file(
                        kind="statistic",
                        description=(
                            "Subgroup / fairness analysis for the "
                            "primary effect (O24)."
                        ),
                        source_path=fair_csv,
                        evidence_id="fairness_subgroups",
                        inputs=[str(primary_record.evidence_id)],
                        producer="pipeline",
                        generation_mode="system",
                        on_sha_change="new_id",
                    )
                    evidence.register_file(
                        kind="log",
                        description=(
                            "Human-readable fairness / subgroup summary (O24)."
                        ),
                        source_path=fair_md,
                        evidence_id="fairness_subgroups_summary",
                        inputs=[str(primary_record.evidence_id)],
                        producer="pipeline",
                        generation_mode="system",
                        on_sha_change="new_id",
                    )
                    findings.append(
                        ValidationFinding(
                            validator="fairness_subgroups",
                            severity="info",
                            message=(
                                f"Subgroup analysis for {predictor_name} ~ "
                                f"{outcome_name} across "
                                f"{len(candidate_subgroups)} axis/axes."
                            ),
                            evidence_ids=[fair_record.evidence_id],
                            detail={
                                "predictor": predictor_name,
                                "outcome": outcome_name,
                                "subgroup_columns": candidate_subgroups,
                                "interaction_pvalues": result.interaction_pvalues,
                            },
                        )
                    )
                    # Escalate to warning if any interaction p < 0.05.
                    sig_cols = [
                        col for col, p in result.interaction_pvalues.items() if p < 0.05
                    ]
                    if sig_cols:
                        findings.append(
                            ValidationFinding(
                                validator="fairness_subgroups",
                                severity="warning",
                                message=(
                                    f"Interaction p < 0.05 on "
                                    f"{sig_cols}; subgroup heterogeneity "
                                    "must be discussed."
                                ),
                                evidence_ids=[fair_record.evidence_id],
                                detail={"significant_subgroups": sig_cols},
                            )
                        )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="fairness_subgroups",
                    severity="warning",
                    message=(
                        f"Subgroup analysis failed: " f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

    manifest_notes = notes
    if stop_after_analysis:
        suffix = "paused_after_analysis: manuscript generation skipped by user option."
        manifest_notes = f"{notes}\n\n{suffix}" if notes else suffix
    literature_provenance = _literature_provenance_note(
        enable_literature=pipeline._enable_literature,
        enable_pubmed=pipeline._enable_pubmed,
        enable_tavily=pipeline._enable_tavily,
    )
    manifest_notes = (
        f"{manifest_notes}\n\n{literature_provenance}"
        if manifest_notes
        else literature_provenance
    )
    robustness_panel_path = run_dir / PANEL_FILENAME
    cohort_locked_path = run_dir / COHORT_LOCK_FILENAME
    cohort_locked_sha = (
        hashlib.sha256(cohort_locked_path.read_bytes()).hexdigest()
        if cohort_locked_path.exists()
        else None
    )
    robustness_panel = load_robustness_panel(robustness_panel_path)
    robustness_panel_sha = (
        hashlib.sha256(robustness_panel_path.read_bytes()).hexdigest()
        if robustness_panel_path.exists()
        else None
    )
    side_findings = collect_side_findings(per_step_records)
    side_findings_path, side_findings_sha = write_side_findings(
        run_dir=run_dir,
        findings=side_findings,
        evidence=evidence,
        prompt_pack_version=plan_result.prompt_version,
    )

    # C3 (pilot 20260515 fix): byte-identical findings recorded
    # multiple times across steps that share the same flagged column
    # get rolled up into a single entry with a ``duplicate_count``
    # detail. Keeps the manifest, the bound report, and the reviewer
    # prompt readable without losing audit information.
    findings = dedupe_findings(findings)

    readiness, artifact_paths = write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=findings,
        per_step_records=per_step_records,
        evidence=evidence,
        run_dir=run_dir,
        manuscript_path=write_result.bound_path,
        stop_after_analysis=stop_after_analysis,
        writer_probe_mode=write_result.writer_probe_mode,
        writer_probe_failed_steps=write_result.writer_probe_failed_steps,
    )

    report_path.write_text(
        render_report(
            context=context,
            plan=plan,
            findings=findings,
            per_step_records=per_step_records,
            evidence=evidence,
            paused_after_analysis=stop_after_analysis,
            readiness=readiness,
        ),
        encoding="utf-8",
    )

    manifest = AnalysisManifest(
        run_id=run_id,
        research_question=context.research_question,
        started_at=plan_result.started_at,
        finished_at=datetime.now(timezone.utc),
        context_path=str(plan_result.context_path.relative_to(run_dir)),
        plan_path=str(plan_result.plan_path.relative_to(run_dir)),
        evidence=evidence.records(),
        findings=findings,
        per_step_records=per_step_records,
        cost_records=cost_records_for_manifest,
        reproducibility=reproducibility_summary,
        submission_profile_name=pipeline._submission_profile_name,
        submission_profile_version=pipeline._submission_profile_version,
        submission_profile_locked_at=pipeline._submission_profile_locked_at,
        **_concept_dictionary_manifest_fields(),
        **_code_version_manifest_fields(),
        readiness=readiness,
        artifact_paths=artifact_paths,
        robustness_panel_path=(
            str(robustness_panel_path.relative_to(run_dir))
            if robustness_panel_path.exists()
            else None
        ),
        robustness_panel_sha=robustness_panel_sha,
        robustness_n_variants=(
            robustness_panel.n_variants if robustness_panel is not None else None
        ),
        robustness_range_low=(
            robustness_panel.range_low if robustness_panel is not None else None
        ),
        robustness_range_high=(
            robustness_panel.range_high if robustness_panel is not None else None
        ),
        cohort_locked_path=(
            str(cohort_locked_path.relative_to(run_dir))
            if cohort_locked_path.exists()
            else None
        ),
        cohort_locked_sha=cohort_locked_sha,
        side_findings_path=str(side_findings_path.relative_to(run_dir)),
        side_findings_sha=side_findings_sha,
        side_findings_count=len(side_findings),
        writer_probe_mode=write_result.writer_probe_mode,
        writer_probe_failed_steps=list(write_result.writer_probe_failed_steps),
        report_path=str(report_path.relative_to(run_dir)),
        manuscript_path=str(write_result.bound_path.relative_to(run_dir)),
        audit_log_path=audit_log_rel,
        workflow_graph_path=str(workflow_graph_path.relative_to(run_dir)),
        execution_replay_path=str(replay_path.relative_to(run_dir)),
        experiment_spec_path=(
            str(experiment_spec_path.relative_to(run_dir))
            if experiment_spec_path is not None and experiment_spec_path.exists()
            else None
        ),
        llm_signature=plan_result.llm_signature,
        used_mock_llm=plan_result.used_mock_llm,
        prompt_pack_version=plan_result.prompt_version,
        prompt_pack_files=plan_result.prompt_files,
        notes=manifest_notes,
    )
    manifest_path = run_dir / "manifest.json"
    execute_result.flush_partial_manifest()
    write_run_checkpoint(manifest_path, manifest.model_dump(mode="json"))

    if pipeline._memory is not None:
        memory_record = pipeline._memory.record(
            run_id=run_id,
            research_question=context.research_question,
            database=database,
            target_outcome=target_outcome,
            findings=findings,
            workdir=run_dir,
        )
        if pipeline._permissioned_memory_store is not None:
            quarantine_run_lesson(
                pipeline._permissioned_memory_store,
                run_id=run_id,
                project=pipeline.workdir.name,
                payload=memory_record.to_dict(),
                created_at=memory_record.finished_at,
            )

    result = PipelineResult(
        run_id=run_id,
        workdir=str(run_dir),
        context_path=str(plan_result.context_path),
        plan_path=str(plan_result.plan_path),
        manifest_path=str(manifest_path),
        report_path=str(report_path),
        manuscript_path=str(write_result.bound_path),
        evidence_count=len(evidence.records()),
        findings_count=len(findings),
    )
    # Phase-1 experience-bank write-back (default off). Mines
    # cross-run hints from this run via the deterministic
    # reflector and persists them. Off the critical path —
    # exceptions are logged and swallowed so a flaky bank file
    # never breaks an otherwise-successful run.
    try:
        pipeline.reflect_and_persist_experience(
            run_dir=run_dir,
            context=context,
            database=database,
            cohort_name=str(getattr(context.cohort, "cohort_name", "") or ""),
        )
    except Exception as exc:  # pragma: no cover — defence in depth
        logger.warning("experience-bank write-back failed (non-fatal): %s", exc)
    if cache_key is not None:
        pipeline._cache.record_hit(
            cache_key,
            result,
            scientific_identity=scientific_identity,
        )
    emit_progress(
        "run",
        "Research-agent run complete.",
        status="complete",
        run_id=run_id,
        evidence_count=result.evidence_count,
        findings_count=result.findings_count,
        stop_after_analysis=stop_after_analysis,
    )
    return result


def finalise_aborted(
    pipeline,
    *,
    run_id: str,
    run_dir: Path,
    context: ResearchContext,
    context_path: Path,
    evidence: EvidenceStore,
    findings: List[ValidationFinding],
    reason: str,
) -> PipelineResult:
    # C3: dedupe before any output uses the findings list.
    findings = dedupe_findings(findings)
    report_path = run_dir / "results_report.md"
    report_path.write_text(
        render_report(
            context=context,
            plan=None,
            findings=findings,
            per_step_records=[],
            evidence=evidence,
            aborted_reason=reason,
        ),
        encoding="utf-8",
    )
    bound_path = run_dir / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        f"# Manuscript scaffold not generated\n\nPipeline aborted: {reason}.\n",
        encoding="utf-8",
    )
    readiness, artifact_paths = write_readiness_artifacts(
        context=context,
        plan=None,
        findings=findings,
        per_step_records=[],
        evidence=evidence,
        run_dir=run_dir,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )
    report_path.write_text(
        render_report(
            context=context,
            plan=None,
            findings=findings,
            per_step_records=[],
            evidence=evidence,
            aborted_reason=reason,
            readiness=readiness,
        ),
        encoding="utf-8",
    )
    manifest = AnalysisManifest(
        run_id=run_id,
        research_question=context.research_question,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        context_path=str(context_path.relative_to(run_dir)),
        evidence=evidence.records(),
        findings=findings,
        submission_profile_name=pipeline._submission_profile_name,
        submission_profile_version=pipeline._submission_profile_version,
        submission_profile_locked_at=pipeline._submission_profile_locked_at,
        **_concept_dictionary_manifest_fields(),
        **_code_version_manifest_fields(),
        report_path=str(report_path.relative_to(run_dir)),
        readiness=readiness,
        artifact_paths=artifact_paths,
        llm_signature=pipeline._llm_signature(pipeline._llm),
        used_mock_llm=any(True for _ in pipeline._iter_mock_clients(pipeline._llm)),
        prompt_pack_version=PROMPT_PACK_VERSION,
        prompt_pack_files=prompt_pack_files(),
        notes=f"aborted: {reason}",
    )
    manifest_path = run_dir / "manifest.json"
    write_run_checkpoint(manifest_path, manifest.model_dump(mode="json"))
    return PipelineResult(
        run_id=run_id,
        workdir=str(run_dir),
        context_path=str(context_path),
        plan_path="",
        manifest_path=str(manifest_path),
        report_path=str(report_path),
        manuscript_path=str(bound_path),
        evidence_count=len(evidence.records()),
        findings_count=len(findings),
    )


# ---------------------------------------------------------------------------
# T3.2 — cost summary renderer
# ---------------------------------------------------------------------------


def _render_cost_summary(meter: "CostMeter") -> str:
    """Render a markdown view of a :class:`CostMeter` for the run report.

    The output has three sections:

    * a one-line headline (``n_calls``, total tokens, total USD);
    * a per-role breakdown so paper authors can quote, e.g., that the
      planner is the most expensive role;
    * a per-model breakdown so a multi-model router run shows which
      checkpoint dominates spend.

    All numbers come from :meth:`CostMeter.summary` so the markdown
    here is purely presentational — the row-level
    ``cost_records.json`` is the source of truth.
    """
    summary = meter.summary()
    lines: List[str] = ["# LLM cost summary (T3.2)", ""]
    if summary["n_calls"] == 0:
        lines.append("_No LLM calls were recorded for this run._")
        return "\n".join(lines) + "\n"
    lines.append(
        f"- **{summary['n_calls']}** LLM calls — "
        f"{summary['total_prompt_tokens']:,} prompt + "
        f"{summary['total_completion_tokens']:,} completion = "
        f"**{summary['total_tokens']:,} total tokens**"
    )
    cost = summary["total_cost_usd"]
    if cost > 0:
        lines.append(f"- Estimated total cost: **${cost:.4f} USD**")
    if summary["any_heuristic"]:
        lines.append(
            "- ⚠️ At least one record relies on a `chars/4` token heuristic "
            "(client did not expose `last_usage`). Treat counts as "
            "approximate."
        )
    lines.append("")
    if summary["by_role"]:
        lines.append("## By role")
        lines.append("")
        lines.append("| role | calls | prompt | completion | total | cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for role, b in sorted(summary["by_role"].items()):
            lines.append(
                f"| `{role}` | {b['n_calls']} | {b['prompt_tokens']:,} | "
                f"{b['completion_tokens']:,} | {b['total_tokens']:,} | "
                f"${b['cost_usd']:.4f} |"
            )
        lines.append("")
    if summary["by_model"]:
        lines.append("## By model")
        lines.append("")
        lines.append("| model | calls | prompt | completion | total | cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model, b in sorted(summary["by_model"].items()):
            lines.append(
                f"| `{model}` | {b['n_calls']} | {b['prompt_tokens']:,} | "
                f"{b['completion_tokens']:,} | {b['total_tokens']:,} | "
                f"${b['cost_usd']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
