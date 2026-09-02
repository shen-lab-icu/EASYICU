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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
from ..contracts.post_analysis import EValueConversionSpec, SubgroupAnalysisSpec
from ..providers.cost import CostMeter
from ..providers.factory import provider_authorization_manifest
from ..authority.evidence_store import EvidenceStore
from ..authority.execution_identity import execution_identity_for_pipeline
from ..authority.plan_input_closure import resolve_registered_plan_authority
from ..methods.multiple_testing import build_multiple_testing_report
from ..methods.sensitivity import compute_e_value
from ..replication.report import _literature_provenance_note
from ..reporting.readiness import render_report, write_readiness_artifacts
from ..reporting.manuscript_state import (
    ManuscriptState,
    render_not_generated,
)
from ..reporting.supplement_inventory import write_supplement_inventory
from ..reporting.supplement_package import write_supplement_package
from ..providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files
from ..authority.runtime_artifacts import (
    AuditLogger,
    STEP_ATTEMPT_HISTORY_REF_SCHEMA,
    active_step_evidence_ids,
    build_execution_replay,
    build_workflow_graph,
    capture_code_version,
    current_evidence_records,
    encode_step_attempt_history_jsonl,
    render_workflow_graph_mermaid,
    verified_run_evidence_path,
    write_json_artifact,
    write_run_checkpoint,
)
from ..robustness.panel import PANEL_FILENAME, load_robustness_panel
from ..schema import AnalysisManifest, AnalysisPlan, PipelineResult, ResearchContext
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


class _EValueBaselineUnresolved(RuntimeError):
    """Internal: the E-value block has nothing real to convert an OR with."""

    def __init__(self, resolved: "ObservedEventRate") -> None:
        super().__init__(resolved.reason)
        self.resolved = resolved


@dataclass(frozen=True)
class ObservedEventRate:
    """The run's own event rate, or an explicit account of why there is none."""

    value: Optional[float]
    cause: str  # stable machine reason; "" when resolved
    reason: str  # one sentence for the finding message
    candidates: Tuple[float, ...] = ()
    source_column: str = ""
    population_column: str = ""
    baseline_population: str = ""


@dataclass(frozen=True)
class EValueArtifacts:
    """Digest-registered O23 outputs from current semantic authorities."""

    csv_path: Path
    markdown_path: Path
    evidence_id: str
    row_count: int
    baseline_prevalence: float
    baseline_source_column: str
    baseline_population: str
    conversion_spec_sha256: str


def _subgroup_spec_matches_primary_requirement(
    plan: AnalysisPlan,
    spec: SubgroupAnalysisSpec,
) -> bool:
    """Bind optional subgroup work to one exact Planner-owned primary model."""

    return any(
        requirement.requirement_id == spec.primary_model_requirement_id
        and requirement.exposure_source == spec.predictor
        and requirement.outcome == spec.outcome
        and requirement.outcome_type == "binary"
        for step in plan.steps
        if step.planned_analysis_role == "primary"
        for requirement in step.model_requirements
    )


def resolve_observed_event_rate(
    path: Optional[Path],
    spec: Optional[EValueConversionSpec],
) -> ObservedEventRate:
    """Read the one rate bound to the declared population and evidence schema.

    Every failure mode returns ``value=None`` with a cause. The previous
    version instead seeded ``baseline_prev = 0.1`` and let each failure fall
    through to it: a missing product, an unreadable file, an unparseable cell,
    every one of them silently produced an E-value computed at an invented
    10% event rate. It also kept the LAST matching cell it saw, so a product
    with one row per exposure group contributed whichever row happened to sort
    last.

    Disagreeing candidates are refused rather than reduced. Picking the first,
    the last, or the mean would each be a different scientific choice about
    which population the E-value is anchored to, and none of them is stated
    anywhere the reader can see.
    """

    if spec is None:
        return ObservedEventRate(
            value=None,
            cause="evalue_conversion_spec_required",
            reason=(
                "the plan declares no E-value conversion evidence/population "
                "contract."
            ),
        )
    if path is None:
        return ObservedEventRate(
            value=None,
            cause="no_outcome_rate_product",
            reason="this run registered no outcome-rate product.",
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return ObservedEventRate(
            value=None,
            cause="outcome_rate_unreadable",
            reason=f"its outcome-rate product could not be read ({error}).",
        )

    import csv as _csv
    import io

    candidates: list[float] = []
    matching_rows = 0
    for row in _csv.DictReader(io.StringIO(text)):
        if str(row.get(spec.population_column) or "").strip() != spec.baseline_population:
            continue
        matching_rows += 1
        try:
            value = float(row[spec.baseline_risk_column])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 < value < 1.0:
            candidates.append(value)

    if matching_rows == 0:
        return ObservedEventRate(
            value=None,
            cause="baseline_population_not_found",
            reason=(
                f"its declared population {spec.baseline_population!r} was not "
                f"found in column {spec.population_column!r}."
            ),
        )
    if matching_rows == 1 and not candidates:
        return ObservedEventRate(
            value=None,
            cause="baseline_population_rate_invalid",
            reason=(
                f"the declared baseline-risk cell {spec.baseline_risk_column!r} "
                "is absent, non-numeric, or outside (0, 1)."
            ),
        )
    if matching_rows != 1 or len(candidates) != 1:
        return ObservedEventRate(
            value=None,
            cause="baseline_population_rate_ambiguous",
            reason=(
                f"its declared population matched {matching_rows} row(s) and "
                f"yielded {len(candidates)} usable rate(s); exactly one is required."
            ),
            candidates=tuple(sorted(set(candidates))),
        )
    return ObservedEventRate(
        value=candidates[0],
        cause="",
        reason="",
        candidates=tuple(candidates),
        source_column=spec.baseline_risk_column,
        population_column=spec.population_column,
        baseline_population=spec.baseline_population,
    )


def _primary_association_evalue_rows(
    path: Path,
    *,
    baseline_prevalence: float,
) -> List[Dict[str, Any]]:
    """Convert the verified primary-association CSV into E-value rows.

    The deterministic adjusted-association owner publishes the typed columns
    ``estimate``, ``ci_low``, ``ci_high`` and ``effect_scale``.  Finalization
    previously read only legacy column aliases such as ``odds_ratio`` and
    therefore produced no E-value for the host-owned primary result.  A typed
    scale declaration takes precedence: only ``effect_scale=odds_ratio`` is
    converted; another declared scale is never guessed from its magnitude.
    Legacy alias-shaped agent outputs remain readable only when no typed scale
    is present.
    """

    import csv as _csv

    rows_out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            scale = str(row.get("effect_scale") or "").strip().casefold()
            scale = scale.replace("-", "_").replace(" ", "_")
            estimate_keys: Tuple[str, ...]
            low_keys: Tuple[str, ...]
            high_keys: Tuple[str, ...]
            if scale:
                if scale != "odds_ratio":
                    continue
                estimate_keys = ("estimate",)
                low_keys = ("ci_low",)
                high_keys = ("ci_high",)
            else:
                estimate_keys = ("odds_ratio", "or", "OR")
                low_keys = ("or_lower", "ci_lower")
                high_keys = ("or_upper", "ci_upper")

            def _first_finite(keys: Tuple[str, ...]) -> Optional[float]:
                for key in keys:
                    value = row.get(key)
                    if value in (None, "", "nan"):
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if number > 0.0 and number not in {float("inf"), float("-inf")}:
                        return number
                return None

            odds_ratio = _first_finite(estimate_keys)
            if odds_ratio is None:
                continue
            ci_low = _first_finite(low_keys)
            ci_high = _first_finite(high_keys)
            ci = (
                (ci_low, ci_high)
                if ci_low is not None and ci_high is not None and ci_low <= ci_high
                else None
            )
            result = compute_e_value(
                estimate=odds_ratio,
                ci=ci,
                estimate_type="or",
                baseline_prevalence=baseline_prevalence,
            )
            rows_out.append(
                {
                    "term": row.get("term")
                    or row.get("variable")
                    or row.get("predictor")
                    or row.get("contrast")
                    or row.get("exposure")
                    or "",
                    "odds_ratio": odds_ratio,
                    "ci_lower": ci[0] if ci else "",
                    "ci_upper": ci[1] if ci else "",
                    "baseline_prevalence": baseline_prevalence,
                    "e_value": result.e_value,
                    "e_value_lower_bound": result.e_value_lower_bound,
                    "note": result.note or "",
                }
            )
    return rows_out


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


def _write_primary_association_evalue_artifacts(
    *,
    evidence: EvidenceStore,
    per_step_records: List[Dict[str, Any]],
    run_dir: Path,
    spec: Optional[EValueConversionSpec],
) -> Optional[EValueArtifacts]:
    """Write O23 from the current primary and outcome-rate authorities.

    This is the finalization boundary, not merely a CSV parser: both inputs
    must resolve through the current successful-step ledger and pass evidence
    digest verification before the typed association schema can produce and
    register ``e_values.csv``.
    """

    primary_source = _current_verified_semantic_csv(
        evidence=evidence,
        per_step_records=per_step_records,
        run_dir=run_dir,
        semantic_id="primary_association",
    )
    if primary_source is None:
        return None
    primary_record, primary_path = primary_source
    outcome_rate_source = _current_verified_semantic_csv(
        evidence=evidence,
        per_step_records=per_step_records,
        run_dir=run_dir,
        semantic_id=(
            spec.baseline_risk_evidence_id if spec is not None else "outcome_rate"
        ),
    )
    resolved = resolve_observed_event_rate(
        None if outcome_rate_source is None else outcome_rate_source[1],
        spec,
    )
    baseline_prevalence = resolved.value
    if baseline_prevalence is None:
        raise _EValueBaselineUnresolved(resolved)

    rows = _primary_association_evalue_rows(
        primary_path,
        baseline_prevalence=baseline_prevalence,
    )
    if not rows:
        return None

    import csv as _csv

    csv_path = run_dir / "e_values.csv"
    markdown_path = run_dir / "e_values.md"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(list(rows[0].keys()))
        for row in rows:
            writer.writerow([row[key] for key in rows[0].keys()])

    markdown_lines = [
        "# E-values for primary effects (O23)",
        "",
        f"Baseline event prevalence used: **{baseline_prevalence:.4f}** "
        f"— read from `{resolved.source_column}` for "
        f"`{resolved.population_column}={resolved.baseline_population}` in the "
        "Planner-bound baseline-risk evidence.",
        "",
        "| Term | OR | 95% CI | E-value | E-value (CI bound) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        ci_display = (
            f"{row['ci_lower']:.2f} – {row['ci_upper']:.2f}"
            if row["ci_lower"] != "" and row["ci_upper"] != ""
            else "—"
        )
        markdown_lines.append(
            "| {term} | {odds_ratio:.2f} | {ci} | {e_value:.2f} | {bound} |".format(
                term=str(row["term"])[:40],
                odds_ratio=row["odds_ratio"],
                ci=ci_display,
                e_value=row["e_value"],
                bound=(
                    f"{row['e_value_lower_bound']:.2f}"
                    if row["e_value_lower_bound"] is not None
                    else "—"
                ),
            )
        )
    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    source_ids = [str(primary_record.evidence_id)]
    if outcome_rate_source is not None:
        source_ids.append(str(outcome_rate_source[0].evidence_id))
    assert spec is not None
    spec_payload = spec.model_dump(mode="json")
    spec_sha256 = hashlib.sha256(
        json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": "easyicu.evalue_conversion_receipt/1",
        "conversion_spec": spec_payload,
        "conversion_spec_sha256": spec_sha256,
        "oracle_scope": "RR formula only; OR conversion requires this receipt",
    }
    csv_record = evidence.register_file(
        kind="statistic",
        description="VanderWeele–Ding E-values for every primary effect row (O23).",
        source_path=csv_path,
        evidence_id="e_values",
        inputs=source_ids,
        producer="pipeline",
        generation_mode="system",
        metadata=metadata,
        on_sha_change="new_id",
    )
    evidence.register_file(
        kind="log",
        description="Human-readable E-value summary (O23).",
        source_path=markdown_path,
        evidence_id="e_values_summary",
        inputs=source_ids,
        producer="pipeline",
        generation_mode="system",
        metadata=metadata,
        on_sha_change="new_id",
    )
    return EValueArtifacts(
        csv_path=csv_path,
        markdown_path=markdown_path,
        evidence_id=str(csv_record.evidence_id),
        row_count=len(rows),
        baseline_prevalence=baseline_prevalence,
        baseline_source_column=resolved.source_column,
        baseline_population=resolved.baseline_population,
        conversion_spec_sha256=spec_sha256,
    )


def _write_subgroup_analysis_artifacts(
    *,
    evidence: EvidenceStore,
    per_step_records: List[Dict[str, Any]],
    run_dir: Path,
    cohort_path: Path,
    plan: AnalysisPlan,
    spec: SubgroupAnalysisSpec,
) -> tuple[str, Dict[str, float]]:
    """Run exactly the declared unadjusted subgroup contract and register it."""

    import csv as _csv

    primary_source = _current_verified_semantic_csv(
        evidence=evidence,
        per_step_records=per_step_records,
        run_dir=run_dir,
        semantic_id="primary_association",
    )
    if primary_source is None:
        raise ValueError("subgroup_primary_association_unresolved")
    primary_record, primary_path = primary_source
    if not _subgroup_spec_matches_primary_requirement(plan, spec):
        raise ValueError("subgroup_spec_not_bound_to_primary_model_requirement")
    with primary_path.open("r", encoding="utf-8") as handle:
        primary_rows = list(_csv.DictReader(handle))
    matching_primary = any(
        spec.predictor
        in {
            str(row.get(field) or "").strip()
            for field in ("exposure", "source_variable", "term", "variable", "predictor")
        }
        and str(row.get("term_role") or "exposure").strip().casefold()
        in {"", "exposure", "primary"}
        and str(row.get("effect_scale") or "").strip().casefold() == spec.effect_scale
        for row in primary_rows
    )
    if not matching_primary:
        raise ValueError("subgroup_predictor_not_bound_to_primary_association")

    cohort_df = pd.read_parquet(cohort_path)
    required_columns = {
        spec.predictor,
        spec.outcome,
        *spec.subgroup_columns,
    }
    missing = sorted(required_columns - set(cohort_df.columns))
    if missing:
        raise ValueError(f"subgroup_declared_columns_missing:{','.join(missing)}")

    from ..methods.fairness import run_subgroup_analysis

    result = run_subgroup_analysis(
        cohort_df=cohort_df,
        predictor=spec.predictor,
        outcome=spec.outcome,
        subgroup_columns=spec.subgroup_columns,
        continuous_buckets=spec.continuous_buckets,
        minimum_axis_n=spec.minimum_axis_n,
        minimum_stratum_n=spec.minimum_stratum_n,
        multiplicity_family_id=spec.multiplicity_family_id,
    )
    csv_path = run_dir / "fairness_subgroups.csv"
    markdown_path = run_dir / "fairness_subgroups.md"
    result.write_csv(csv_path)
    result.write_markdown(markdown_path)
    metadata = {
        "schema_version": "easyicu.subgroup_analysis_receipt/1",
        "spec": spec.model_dump(mode="json"),
        "spec_sha256": hashlib.sha256(
            json.dumps(
                spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "claim_ceiling": "analysis_only",
    }
    input_ids = [str(primary_record.evidence_id)]
    if evidence.get("run_input_capsule") is not None:
        input_ids.append("run_input_capsule")
    record = evidence.register_file(
        kind="statistic",
        description="Pre-specified unadjusted subgroup analysis (O24).",
        source_path=csv_path,
        evidence_id="fairness_subgroups",
        inputs=input_ids,
        producer="pipeline",
        generation_mode="system",
        metadata=metadata,
        on_sha_change="new_id",
    )
    evidence.register_file(
        kind="log",
        description="Human-readable pre-specified subgroup summary (O24).",
        source_path=markdown_path,
        evidence_id="fairness_subgroups_summary",
        inputs=input_ids,
        producer="pipeline",
        generation_mode="system",
        metadata=metadata,
        on_sha_change="new_id",
    )
    return str(record.evidence_id), dict(result.interaction_pvalues)


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
        hard_stop_accounting = None
        provider_hard_stop = getattr(pipeline, "_provider_hard_stop", None)
        accounting_summary = getattr(provider_hard_stop, "accounting_summary", None)
        if callable(accounting_summary):
            hard_stop_accounting = accounting_summary()
        cost_summary = plan_result.cost_meter.summary(
            hard_stop_accounting=hard_stop_accounting,
        )
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
            _render_cost_summary(
                plan_result.cost_meter,
                hard_stop_accounting=hard_stop_accounting,
            ),
            encoding="utf-8",
        )
        # Machine-readable aggregate (token totals + estimated USD, by model)
        # so the bench scorer and Fig.3 source-data builder can read cost
        # without re-parsing the markdown or recomputing from raw records.
        cost_summary_json_path = run_dir / "cost_summary.json"
        cost_summary_json_path.write_text(
            json.dumps(
                cost_summary,
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

    # O24 must run before O22 so every declared stratum/interaction p-value is
    # part of the family-scoped multiplicity denominator. It is opt-in twice:
    # the operator enables the feature and the Planner declares exact science.
    if pipeline._enable_fairness_subgroups:
        subgroup_spec = plan.subgroup_analysis_spec
        if subgroup_spec is None:
            findings.append(
                ValidationFinding(
                    validator="fairness_subgroups",
                    severity="warning",
                    message=(
                        "Subgroup analysis was enabled but not computed because "
                        "AnalysisPlan.subgroup_analysis_spec is absent; the host "
                        "will not choose a predictor, subgroup axes, binning, or "
                        "multiplicity family."
                    ),
                    detail={"reason": "subgroup_analysis_spec_required"},
                )
            )
        else:
            try:
                subgroup_evidence_id, interaction_pvalues = (
                    _write_subgroup_analysis_artifacts(
                        evidence=evidence,
                        per_step_records=per_step_records,
                        run_dir=run_dir,
                        cohort_path=cohort_path,
                        plan=plan,
                        spec=subgroup_spec,
                    )
                )
                findings.append(
                    ValidationFinding(
                        validator="fairness_subgroups",
                        severity="info",
                        message=(
                            "Computed the pre-specified, analysis-only subgroup "
                            f"contract across {len(subgroup_spec.subgroup_columns)} "
                            "axis/axes; raw p-values are delegated to O22."
                        ),
                        evidence_ids=[subgroup_evidence_id],
                        detail={
                            "spec": subgroup_spec.model_dump(mode="json"),
                            "interaction_pvalues": interaction_pvalues,
                            "claim_ceiling": "analysis_only",
                        },
                    )
                )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="fairness_subgroups",
                        severity="warning",
                        message=(
                            "Declared subgroup analysis was not computed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        detail={
                            "reason": "subgroup_analysis_contract_failed",
                            "cause": str(exc),
                        },
                    )
                )

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
    # For odds ratios, baseline risk is read only from the exact evidence,
    # column and population declared by EValueConversionSpec.
    try:
        evalue_artifacts = _write_primary_association_evalue_artifacts(
            evidence=evidence,
            per_step_records=per_step_records,
            run_dir=run_dir,
            spec=plan.evalue_conversion_spec,
        )
        if evalue_artifacts is not None:
            findings.append(
                ValidationFinding(
                    validator="e_value",
                    severity="info",
                    message=(
                        f"Computed E-values for {evalue_artifacts.row_count} primary "
                        "effect row(s) at this run's observed event rate "
                        f"{evalue_artifacts.baseline_prevalence:.4f} for declared "
                        f"population {evalue_artifacts.baseline_population!r}."
                    ),
                    evidence_ids=[evalue_artifacts.evidence_id],
                    detail={
                        "baseline_prevalence": (
                            evalue_artifacts.baseline_prevalence
                        ),
                        "baseline_prevalence_source": "observed_outcome_rate",
                        "baseline_prevalence_column": (
                            evalue_artifacts.baseline_source_column
                        ),
                        "baseline_population": evalue_artifacts.baseline_population,
                        "conversion_spec_sha256": (
                            evalue_artifacts.conversion_spec_sha256
                        ),
                    },
                )
            )
    except _EValueBaselineUnresolved as exc:
        # Keep the structured cause instead of collapsing it into the generic
        # exception handler below: no rate was available, and none was invented.
        resolved = exc.resolved
        findings.append(
            ValidationFinding(
                validator="e_value",
                severity="warning",
                message=(
                    "E-values were not computed: "
                    f"{resolved.reason} An odds ratio cannot be converted "
                    "to a risk ratio without this run's observed event rate, "
                    "and the host does not substitute one."
                ),
                detail={
                    "reason": "e_value_baseline_prevalence_unresolved",
                    "cause": resolved.cause,
                    "candidates": list(resolved.candidates),
                },
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
    supplement_inventory, supplement_findings = write_supplement_inventory(
        plan=plan,
        evidence=evidence,
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    findings.extend(supplement_findings)
    write_supplement_package(
        inventory=supplement_inventory,
        evidence=evidence,
        run_dir=run_dir,
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
    execution_identity = execution_identity_for_pipeline(pipeline)
    # Resolve the immutable scientific plan before writing run_status.json.
    # Otherwise a later authority failure can leave a durable
    # ``paper_authorized=true`` status even though no final manifest exists.
    current_plan_authority = resolve_registered_plan_authority(
        run_dir=run_dir,
        evidence=evidence,
        plan=plan,
        plan_path=plan_result.plan_path,
    )

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
        force_diagnostic_only=bool(getattr(pipeline, "_development_diagnostic", False)),
        execution_paper_eligible=execution_identity.paper_eligible,
        plan_authority_verified=True,
        plan_authority_sha256=current_plan_authority.sha256,
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

    # Flush first so the in-memory append-only attempt ledger includes the
    # final current snapshots before both durable manifests are serialized.
    execute_result.flush_partial_manifest()
    step_attempt_history = list(execute_result.step_attempt_history)
    step_attempt_history_ref = None
    if step_attempt_history:
        history_record = evidence.register_text(
            kind="log",
            description=(
                "Append-only execution and deterministic-revalidation history "
                "externalized from the finalized run manifest."
            ),
            text=encode_step_attempt_history_jsonl(step_attempt_history),
            filename="step_attempt_history.jsonl",
            evidence_id="step_attempt_history",
            producer="pipeline",
            generation_mode="system",
            publish_aliases=False,
            on_sha_change="new_id",
        )
        step_attempt_history_ref = {
            "schema_version": STEP_ATTEMPT_HISTORY_REF_SCHEMA,
            "format": "jsonl",
            "evidence_id": history_record.evidence_id,
            "relative_path": history_record.relative_path,
            "sha256": history_record.sha256,
            "record_count": len(step_attempt_history),
        }
    manifest = AnalysisManifest(
        run_id=run_id,
        research_question=context.research_question,
        started_at=plan_result.started_at,
        finished_at=datetime.now(timezone.utc),
        context_path=str(plan_result.context_path.relative_to(run_dir)),
        plan_path=current_plan_authority.relative_path,
        current_plan_authority=current_plan_authority.to_dict(),
        evidence=evidence.records(),
        findings=findings,
        per_step_records=per_step_records,
        step_attempt_history=[],
        step_attempt_history_ref=step_attempt_history_ref,
        cost_records=cost_records_for_manifest,
        reproducibility=reproducibility_summary,
        provider_authorization=provider_authorization_manifest(pipeline._llm),
        execution_identity=execution_identity.model_dump(mode="json"),
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
    write_run_checkpoint(manifest_path, manifest.model_dump(mode="json"))

    if pipeline._memory is not None:
        try:
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
        except Exception as exc:  # pragma: no cover - storage boundary
            logger.warning("legacy memory finalization failed (non-fatal): %s", exc)

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
        experience_records = pipeline.reflect_and_persist_experience(
            run_dir=run_dir,
            context=context,
            database=database,
            cohort_name=str(getattr(context.cohort, "cohort_name", "") or ""),
        )
        if pipeline._permissioned_memory_store is not None:
            for experience_record in experience_records:
                quarantine_run_lesson(
                    pipeline._permissioned_memory_store,
                    run_id=run_id,
                    project=pipeline.workdir.name,
                    payload=experience_record.to_dict(),
                    created_at=experience_record.produced_at,
                    producer="legacy_experience_bank",
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
        render_not_generated(
            ManuscriptState.blocked("pipeline_aborted"),
            f"Pipeline aborted: {reason}.",
        ),
        encoding="utf-8",
    )
    execution_identity = execution_identity_for_pipeline(pipeline)
    readiness, artifact_paths = write_readiness_artifacts(
        context=context,
        plan=None,
        findings=findings,
        per_step_records=[],
        evidence=evidence,
        run_dir=run_dir,
        manuscript_path=bound_path,
        stop_after_analysis=False,
        execution_paper_eligible=execution_identity.paper_eligible,
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
        provider_authorization=provider_authorization_manifest(pipeline._llm),
        execution_identity=execution_identity.model_dump(mode="json"),
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


def _render_cost_summary(
    meter: "CostMeter",
    *,
    hard_stop_accounting: Optional[Dict[str, Any]] = None,
) -> str:
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
    summary = meter.summary(hard_stop_accounting=hard_stop_accounting)
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
    accounting = summary.get("usage_accounting") or {}
    reported = accounting.get("provider_reported") or {}
    unknown = accounting.get("usage_unknown") or {}
    upper = accounting.get("conservative_upper_bound") or {}
    lines.extend(
        [
            (
                "- Provider-reported actual usage: "
                f"{int(reported.get('n_calls') or 0)} calls, "
                f"{int(reported.get('total_tokens') or 0):,} tokens, "
                f"${float(reported.get('estimated_cost_usd') or 0.0):.4f} USD"
            ),
            (
                "- Usage unknown: "
                f"{int(unknown.get('n_calls') or 0)} calls "
                f"({json.dumps(unknown.get('states') or {}, sort_keys=True)})"
            ),
            (
                "- Conservative upper bound: "
                f"{int(upper.get('total_tokens') or 0):,} tokens, "
                f"${float(upper.get('estimated_cost_usd') or 0.0):.4f} USD "
                f"(`{upper.get('source') or 'unavailable'}`)"
            ),
        ]
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
