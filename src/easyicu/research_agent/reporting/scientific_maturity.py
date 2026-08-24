"""Deterministic article-maturity audit for the final executed plan.

This owner answers one narrow question: may a scientifically valid run be
described as a strong journal submission?  It never changes the user's study,
chooses a covariate, or computes a result.  It projects existing typed context,
final-plan, literature, robustness, figure, and reviewer receipts into stable
findings.  A low-intensity user-authorized analysis can therefore still run and
remain useful without being mislabeled as article-grade evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..planning.scientific_review import (
    association_study as _plan_association_study,
    method_source_facts,
    model_covariates as _plan_model_covariates,
    patient_identity_available as _patient_identity_available,
    post_baseline_exposure as _plan_post_baseline_exposure,
    repeat_units_possible as _plan_repeat_units_possible,
    repeated_unit_design_closed as _plan_repeated_unit_design_closed,
    scientific_steps as _plan_scientific_steps,
    timing_design_closed as _plan_timing_design_closed,
)
from ..planning.novelty_contract import NOVELTY_REVIEW_DIMENSIONS
from ..figures.contracts import figure_contract_paths, figure_contract_tier
from ..research_context.temporal_semantics import (
    primary_exposure_time_anchor_alignment,
)
from ..schema import AnalysisPlan, ResearchContext
from .display_suite import panel_has_absolute_risk_context
from .novelty_positioning import novelty_authority_digests


Severity = Literal["blocker", "major", "minor"]

SCIENTIFIC_MATURITY_AUDIT_REGISTRATION = (
    "scientific_maturity_audit",
    "statistic",
    "Typed literature, ICU-design, robustness, figure, and review maturity audit.",
)


class ScientificMaturityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Severity
    dimension: str
    message: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    remediation: str
    requires_user_authorization: bool = False
    authorization_question: Optional[str] = None


class ScientificMaturityAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.scientific_maturity/1"] = (
        "easyicu.scientific_maturity/1"
    )
    status: Literal["article_grade", "major_revision", "analysis_only"]
    article_grade: bool
    score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int]
    findings: list[ScientificMaturityFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


def scientific_maturity_readiness_gates(
    audit: ScientificMaturityAudit,
) -> dict[str, Any]:
    """Return the stable readiness projection owned by this audit contract."""

    return {
        "scientific_maturity_article_grade": audit.article_grade,
        "scientific_maturity_status": audit.status,
        "scientific_maturity_score": audit.score,
        "scientific_maturity_dimension_scores": audit.dimension_scores,
        "scientific_maturity_findings": [
            finding.model_dump(mode="json") for finding in audit.findings
        ],
        "scientific_maturity_facts": audit.facts,
    }


def scientific_maturity_audit_from_gates(
    gates: Mapping[str, Any],
) -> ScientificMaturityAudit:
    """Reconstitute the canonical persisted audit from readiness gates.

    Validation happens in the owner model so a missing or malformed projection
    cannot be silently written as a scientific audit.
    """

    return ScientificMaturityAudit.model_validate(
        {
            "status": gates["scientific_maturity_status"],
            "article_grade": gates["scientific_maturity_article_grade"],
            "score": gates["scientific_maturity_score"],
            "dimension_scores": gates["scientific_maturity_dimension_scores"],
            "findings": gates["scientific_maturity_findings"],
            "facts": gates["scientific_maturity_facts"],
        }
    )


def _read_json(run_dir: Path, name: str) -> Mapping[str, Any]:
    try:
        path = run_dir / name
        if path.stat().st_size > 2 * 1024 * 1024:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _model_covariates(plan: Optional[AnalysisPlan]) -> tuple[str, ...]:
    return _plan_model_covariates(plan)


def _registered_figure_adjustment_authority(
    run_dir: Path,
    *,
    figure_evidence_ids: set[str],
) -> tuple[Optional[tuple[str, ...]], list[str]]:
    """Resolve a figure's executed adjustment roster from registered receipts.

    A host-owned executor may compile its model contract after the Planner plan
    is frozen. In that case the registered runtime receipt, not missing plan
    prose, is the authority for whether the rendered estimate is adjusted.
    Only digest-matching runner receipts produced by a step on the plotted
    evidence lineage are accepted. Figure-only steps commonly copy an upstream
    result table, so requiring the receipt to share the renderer's step id would
    discard the model authority that the evidence graph already binds.
    """

    authority = _read_json(run_dir / "evidence", "evidence_authority.json")
    records = [
        record
        for record in authority.get("records") or []
        if isinstance(record, Mapping)
    ]
    records_by_id = {
        str(record.get("evidence_id") or "").strip(): record
        for record in records
        if str(record.get("evidence_id") or "").strip()
    }
    related_steps: set[str] = set()
    pending_ids = list(figure_evidence_ids)
    visited_ids: set[str] = set()
    while pending_ids:
        evidence_id = pending_ids.pop()
        if evidence_id in visited_ids:
            continue
        visited_ids.add(evidence_id)
        record = records_by_id.get(evidence_id)
        if record is None:
            continue
        produced_by_step = str(record.get("produced_by_step") or "").strip()
        if produced_by_step:
            related_steps.add(produced_by_step)
        pending_ids.extend(
            str(value).strip()
            for value in record.get("inputs") or []
            if str(value).strip() and str(value).strip() not in visited_ids
        )
    rosters: dict[tuple[str, ...], list[str]] = {}
    root = run_dir.resolve()
    for record in records:
        if str(record.get("produced_by_step") or "").strip() not in related_steps:
            continue
        if record.get("producer") != "runner" or record.get("kind") != "log":
            continue
        relative_path = str(record.get("relative_path") or "").strip()
        if not relative_path.endswith("_runtime_receipt.json"):
            continue
        try:
            path = (run_dir / relative_path).resolve()
            path.relative_to(root)
            raw = path.read_bytes()
        except (OSError, ValueError):
            continue
        if hashlib.sha256(raw).hexdigest() != str(record.get("sha256") or ""):
            continue
        try:
            receipt = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(receipt, Mapping):
            continue
        schema_version = str(receipt.get("schema_version") or "").strip()
        if not re.fullmatch(r"easyicu\.[a-z0-9_]+_runtime_receipt/1", schema_version):
            continue
        adjustment_columns = receipt.get("adjustment_columns")
        if not isinstance(adjustment_columns, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in adjustment_columns
        ):
            continue
        roster = tuple(value.strip() for value in adjustment_columns)
        rosters.setdefault(roster, []).append(relative_path)
    if len(rosters) != 1:
        return None, []
    roster, refs = next(iter(rosters.items()))
    return roster, sorted(set(refs))


def _scientific_steps(plan: Optional[AnalysisPlan]) -> list[Any]:
    return list(_plan_scientific_steps(plan))


def _association_study(plan: Optional[AnalysisPlan]) -> bool:
    return _plan_association_study(plan)


def _post_baseline_exposure(context: ResearchContext) -> tuple[bool, Optional[str]]:
    return _plan_post_baseline_exposure(context)


def _timing_design_closed(plan: Optional[AnalysisPlan]) -> bool:
    return _plan_timing_design_closed(plan)


def _repeated_unit_design_closed(
    context: ResearchContext, plan: Optional[AnalysisPlan]
) -> bool:
    return _plan_repeated_unit_design_closed(context, plan)


def _repeat_units_possible(context: ResearchContext) -> bool:
    return _plan_repeat_units_possible(context)


def _novelty_facts(
    run_dir: Path,
    *,
    comparison_source_keys: list[str],
    expected_authority_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Read an optional independent novelty-positioning receipt.

    A retrieved comparator is necessary but not sufficient: novelty requires
    an explicit, source-bound comparison of population, time zero, exposure,
    outcome/estimand, and analysis.  The Agent may report that this has not yet
    been established; it must never manufacture a differentiator from the
    study result itself.
    """

    audit = _read_json(run_dir, "novelty_positioning_audit.json")
    status = str(audit.get("status") or "not_established").strip().casefold()
    direct_keys = {
        str(value).strip()
        for value in audit.get("direct_comparator_keys") or []
        if str(value).strip()
    }
    analogue_keys = {
        str(value).strip()
        for value in audit.get("design_analogue_keys") or []
        if str(value).strip()
    }
    comparator_keys = sorted(
        {
            *direct_keys,
            *analogue_keys,
        }
    )
    dimensions = audit.get("comparison_dimensions")
    dimensions = dimensions if isinstance(dimensions, Mapping) else {}
    required = set(NOVELTY_REVIEW_DIMENSIONS)
    complete_dimensions = {
        name
        for name in required
        if isinstance(dimensions.get(name), Mapping)
        and str(dimensions[name].get("study") or "").strip()
        and str(dimensions[name].get("comparator") or "").strip()
        and str(dimensions[name].get("difference") or "").strip()
    }
    digest_fields = ("context_sha256", "plan_sha256", "literature_sha256")
    digest_mismatches = [
        field
        for field in digest_fields
        if str(audit.get(field) or "").strip().casefold()
        != str(expected_authority_digests.get(field) or "").strip().casefold()
    ]
    digest_bound = not digest_mismatches
    supported = bool(
        status == "supported"
        and digest_bound
        and comparator_keys
        and set(comparator_keys) <= set(comparison_source_keys)
        and complete_dimensions == required
        and str(audit.get("review_disposition") or "").strip().casefold()
        in {"independent_pre_review_pass", "human_review_pass"}
    )
    return {
        "status": status,
        "supported": supported,
        "direct_comparator_keys": sorted(direct_keys),
        "design_analogue_keys": sorted(analogue_keys),
        "comparison_source_keys": comparator_keys,
        "complete_dimensions": sorted(complete_dimensions),
        "required_dimensions": sorted(required),
        "review_disposition": str(audit.get("review_disposition") or "not_available"),
        "digest_bound": digest_bound,
        "digest_mismatches": digest_mismatches,
    }


def _robustness_facts(run_dir: Path, plan: Optional[AnalysisPlan]) -> dict[str, Any]:
    panel = _read_json(run_dir, "robustness_panel.json")
    rows = [row for row in panel.get("rows") or [] if isinstance(row, Mapping)]
    primary_id = str(panel.get("primary_spec_id") or "primary")
    primary = next((row for row in rows if str(row.get("spec_id")) == primary_id), None)
    variants = [row for row in rows if str(row.get("spec_id")) != primary_id]
    declared_axes = {
        str(spec.axis)
        for spec in (plan.robustness_specs if plan is not None else ())
        if str(spec.axis)
    }
    model_grid = _registered_association_model_grid_facts(run_dir)
    family_robustness = _registered_family_robustness_facts(run_dir)
    axes = sorted(
        declared_axes
        | set(model_grid["axes"])
        | set(family_robustness["axes"])
    )

    def same_estimate(row: Mapping[str, Any]) -> bool:
        if primary is None:
            return False
        keys = ("point_estimate", "ci_low", "ci_high")
        try:
            return all(
                math.isclose(
                    float(row.get(key)),
                    float(primary.get(key)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for key in keys
            ) and int(row.get("n") or 0) == int(primary.get("n") or 0)
        except (TypeError, ValueError):
            return False

    return {
        "declared_axes": axes,
        "variant_count": (
            len(variants)
            + int(model_grid["variant_count"])
            + int(family_robustness["variant_count"])
        ),
        "all_variants_duplicate_primary": bool(
            variants
            or model_grid["variant_count"]
            or family_robustness["variant_count"]
        )
        and all(same_estimate(row) for row in variants)
        and (
            not model_grid["variant_count"]
            or bool(model_grid["all_variants_duplicate_primary"])
        )
        and (
            not family_robustness["variant_count"]
            or bool(family_robustness["all_variants_duplicate_primary"])
        ),
        "registered_robustness_evidence_refs": sorted(
            {
                *model_grid["evidence_refs"],
                *family_robustness["evidence_refs"],
            }
        ),
    }


def _registered_runner_step_summaries(
    run_dir: Path,
) -> list[tuple[str, Mapping[str, Any]]]:
    authority = _read_json(run_dir / "evidence", "evidence_authority.json")
    records = [
        record
        for record in authority.get("records") or []
        if isinstance(record, Mapping)
        and record.get("kind") == "statistic"
        and record.get("producer") == "runner"
    ]
    root = run_dir.resolve()
    summaries: list[tuple[str, Mapping[str, Any]]] = []
    for record in records:
        relative_path = str(record.get("relative_path") or "").strip()
        if not relative_path.endswith("__step_summary.json"):
            continue
        try:
            path = (run_dir / relative_path).resolve()
            path.relative_to(root)
            raw = path.read_bytes()
        except (OSError, ValueError):
            continue
        if hashlib.sha256(raw).hexdigest() != str(record.get("sha256") or ""):
            continue
        try:
            summary = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(summary, Mapping):
            continue
        summaries.append((relative_path, summary))
    return summaries


def _registered_association_model_grid_facts(run_dir: Path) -> dict[str, Any]:
    """Project typed model-grid evidence into article robustness axes.

    Some study families execute their prespecified sensitivities through a
    host-owned model-grid step rather than the generic robustness replay.  The
    result is still valid robustness evidence when its registered step summary
    is digest-bound and carries the exact model-grid runtime receipt.  Reading
    only ``robustness_panel.json`` would otherwise discard those completed
    timing, cohort, and functional-form stress tests.
    """

    axes: set[str] = set()
    variant_count = 0
    duplicate_flags: list[bool] = []
    evidence_refs: list[str] = []
    for relative_path, summary in _registered_runner_step_summaries(run_dir):
        receipt = summary.get("scientific_runtime_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("schema_version") != (
            "easyicu.association_model_grid_runtime_receipt/1"
        ):
            continue
        rows = summary.get("analysis_rows")
        reference_id = str(receipt.get("reference_variant_id") or "").strip()
        variant_ids = [
            str(value).strip()
            for value in receipt.get("variant_ids") or []
            if str(value).strip()
        ]
        if (
            not isinstance(rows, list)
            or not reference_id
            or len(variant_ids) < 2
            or len(variant_ids) != len(set(variant_ids))
        ):
            continue
        by_id = {
            str(row.get("analysis_id") or "").strip(): row
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("analysis_id") or "").strip()
        }
        if set(by_id) != set(variant_ids) or reference_id not in by_id:
            continue
        reference = by_id[reference_id]
        basis_receipts = summary.get("basis_receipts")
        basis_receipts = basis_receipts if isinstance(basis_receipts, Mapping) else {}
        for analysis_id in variant_ids:
            if analysis_id == reference_id:
                continue
            row = by_id[analysis_id]
            variant_count += 1
            row_axes: set[str] = set()
            if any(
                row.get(field) != reference.get(field)
                for field in (
                    "landmark_hours",
                    "alive_at_landmark_required",
                    "negative_event_times_excluded",
                )
            ):
                row_axes.add("timing")
            if (
                row.get("fitted_covariates") != reference.get("fitted_covariates")
                or bool(basis_receipts.get(analysis_id))
            ):
                row_axes.add("model")
            if (
                "timing" not in row_axes
                and "model" not in row_axes
                and (
                    row.get("readmission_restriction")
                    != reference.get("readmission_restriction")
                    or row.get("n_stays") != reference.get("n_stays")
                )
            ):
                row_axes.add("cohort")
            axes.update(row_axes)
            duplicate_flags.append(
                not row_axes
                and all(
                    row.get(field) == reference.get(field)
                    for field in ("n_stays", "estimate", "ci_low", "ci_high")
                )
            )
        evidence_refs.append(relative_path)
    return {
        "axes": sorted(axes),
        "variant_count": variant_count,
        "all_variants_duplicate_primary": bool(duplicate_flags)
        and all(duplicate_flags),
        "evidence_refs": sorted(set(evidence_refs)),
    }


def _registered_family_robustness_facts(run_dir: Path) -> dict[str, Any]:
    """Accept independent variants from a digest-bound family robustness receipt."""

    axes: set[str] = set()
    variant_count = 0
    duplicate_flags: list[bool] = []
    evidence_refs: list[str] = []
    for relative_path, summary in _registered_runner_step_summaries(run_dir):
        if summary.get("analysis_family") != (
            "robustness_sensitivity"
        ):
            continue
        rows = summary.get("robustness_rows")
        if not isinstance(rows, list):
            continue
        accepted = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("converged") is True
            and row.get("independent_variant") is True
            and str(row.get("axis") or "").strip() not in {"", "primary"}
            and str(row.get("spec_id") or "").strip()
            and str(row.get("evidence_id") or "").strip()
        ]
        if not accepted:
            continue
        for row in accepted:
            axes.add(str(row["axis"]).strip())
            variant_count += 1
            duplicate_flags.append(False)
        evidence_refs.append(relative_path)
    return {
        "axes": sorted(axes),
        "variant_count": variant_count,
        "all_variants_duplicate_primary": bool(duplicate_flags)
        and all(duplicate_flags),
        "evidence_refs": sorted(set(evidence_refs)),
    }


def _endpoint_semantic_facts(context: ResearchContext) -> dict[str, Any]:
    target = str(context.target_outcome or "").strip()
    descriptor = context.variable(target) if target else None
    if descriptor is None:
        return {
            "target": target or None,
            "resolved": False,
            "conflict": False,
            "description": None,
            "source_concept": None,
        }
    caveats = [str(value) for value in descriptor.clinical_caveats]
    text = " ".join(
        [str(descriptor.description or ""), str(descriptor.source_concept or "")]
    ).lower()
    unresolved = bool(
        not descriptor.description
        or "mortality_unspecified" in text
        or "confirm whether" in text
        or "declared_primary_outcome" in text
    )
    return {
        "target": target,
        "resolved": not unresolved,
        "conflict": any("Endpoint-definition conflict" in value for value in caveats),
        "description": descriptor.description,
        "source_concept": descriptor.source_concept,
    }


def _manuscript_section_word_counts(manuscript: str) -> dict[str, int]:
    """Count words under top-level article headings, including subheadings."""

    matches = list(
        re.finditer(
            r"^(?P<marks>#{1,3})\s+(?P<title>.+?)\s*$", manuscript, re.MULTILINE
        )
    )
    aliases = {
        "abstract": {"abstract"},
        "introduction": {"introduction", "background"},
        "methods": {"methods", "method", "materials and methods"},
        "results": {"results"},
        "discussion": {"discussion"},
    }
    output = {key: 0 for key in aliases}
    for index, match in enumerate(matches):
        normalized = " ".join(
            re.sub(
                r"[^a-z0-9\u4e00-\u9fff]+",
                " ",
                match.group("title").casefold(),
            ).split()
        )
        section = next(
            (key for key, values in aliases.items() if normalized in values),
            None,
        )
        if section is None:
            continue
        level = len(match.group("marks"))
        end = len(manuscript)
        for candidate in matches[index + 1 :]:
            if len(candidate.group("marks")) <= level:
                end = candidate.start()
                break
        output[section] = len(re.findall(r"\b[\w'-]+\b", manuscript[match.end() : end]))
    return output


def _manuscript_facts(run_dir: Path) -> dict[str, Any]:
    audit = _read_json(run_dir, "manuscript_literature_audit.json")
    manuscript_path = run_dir / "manuscript_scaffold_bound.md"
    try:
        manuscript = manuscript_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        manuscript = ""
    headings = {
        match.group(1).strip().lower()
        for match in re.finditer(r"^#{1,3}\s+(.+?)\s*$", manuscript, re.MULTILINE)
    }
    required_groups = (
        {"abstract"},
        {"introduction"},
        {"methods"},
        {"results"},
        {"discussion"},
        {"limitations"},
        {"conclusion", "conclusions"},
        {"data and code availability", "data availability"},
        {"funding"},
        {"conflicts of interest", "conflict of interest"},
    )
    missing = [sorted(group)[0] for group in required_groups if not (headings & group)]
    section_word_counts = _manuscript_section_word_counts(manuscript)
    # These are structural anti-stub floors, not a target-journal word limit.
    # A journal-ready manuscript may be much longer, but a core section below
    # these bounds cannot contain a reproducible design or a critical account
    # of the findings.
    section_word_floors = {
        "abstract": 80,
        "introduction": 180,
        "methods": 250,
        "results": 200,
        "discussion": 250,
    }
    thin_sections = [
        section
        for section, minimum in section_word_floors.items()
        if section_word_counts.get(section, 0) < minimum
    ]
    return {
        "word_count": len(re.findall(r"\b[\w'-]+\b", manuscript)),
        "section_word_counts": section_word_counts,
        "section_word_floors": section_word_floors,
        "thin_sections": thin_sections,
        "missing_sections": missing,
        "literature_audit_status": str(audit.get("status") or "missing"),
        "exact_literature_citations_present": bool(
            audit.get("exact_citations_present")
        ),
        "unknown_literature_keys": list(audit.get("unknown_keys") or []),
        "section_cited_keys": dict(audit.get("section_cited_keys") or {}),
        "missing_required_citation_sections": list(
            audit.get("missing_required_citation_sections") or []
        ),
        "direct_comparator_sections_missing": list(
            audit.get("direct_comparator_sections_missing") or []
        ),
        "methods_method_source_missing": bool(
            audit.get("methods_method_source_missing")
        ),
        "pdf_present": (run_dir / "manuscript_scaffold.pdf").is_file(),
        "pdf_receipt_present": (
            run_dir / "manuscript_pdf_render_receipt.json"
        ).is_file()
        or (run_dir / "manuscript_pdf_receipt.json").is_file(),
    }


def _primary_figure_facts(
    run_dir: Path,
    plan: Optional[AnalysisPlan],
) -> dict[str, Any]:
    """Compare the primary figure's scientific label with the executed plan."""

    primary_contracts = [
        path
        for path in figure_contract_paths(run_dir)
        if figure_contract_tier(path, run_dir) == "primary_publication"
    ]
    labels: list[str] = []
    roles: list[str] = []
    adjustment_labels: list[str] = []
    adjustment_roles: list[str] = []
    figure_evidence_ids: set[str] = set()
    absolute_risk_panel_present = False
    for path in primary_contracts:
        raw = _read_json(path.parent, path.name)
        for panel in raw.get("panels") or []:
            if not isinstance(panel, Mapping):
                continue
            absolute_risk_panel_present = (
                absolute_risk_panel_present or panel_has_absolute_risk_context(panel)
            )
            role = str(panel.get("role") or "").strip().casefold()
            roles.append(role)
            figure_evidence_ids.update(
                str(value).strip()
                for value in panel.get("evidence_ids") or []
                if str(value).strip()
            )
            label = " ".join(
                [
                    str(panel.get("title") or ""),
                    str(panel.get("claim") or ""),
                ]
            ).casefold()
            labels.append(label)
            if role in {
                "association",
                "association_forest",
                "effect",
                "primary_effect",
                "primary_estimand",
                "survival_effect",
            }:
                adjustment_labels.append(label)
                adjustment_roles.append(role)
    plan_covariates = _model_covariates(plan)
    executed_covariates, execution_refs = _registered_figure_adjustment_authority(
        run_dir,
        figure_evidence_ids=figure_evidence_ids,
    )
    covariates = (
        plan_covariates
        if plan_covariates
        else executed_covariates
        if executed_covariates is not None
        else ()
    )
    expected_label = "adjusted" if covariates else "unadjusted"
    labels_to_check = adjustment_labels or labels
    conflicting = (
        any(
            "adjusted" in label and "unadjusted" not in label
            for label in labels_to_check
        )
        if expected_label == "unadjusted"
        else any("unadjusted" in label for label in labels_to_check)
    )
    return {
        "expected_adjustment_label": expected_label,
        "primary_contract_paths": [
            str(path.relative_to(run_dir)) for path in primary_contracts
        ],
        "primary_panel_roles": roles,
        "adjustment_panel_roles": adjustment_roles,
        "adjustment_covariates": list(covariates),
        "adjustment_authority": (
            "plan"
            if plan_covariates
            else "runtime_receipt"
            if executed_covariates is not None
            else "not_established"
        ),
        "adjustment_authority_refs": execution_refs,
        "adjustment_label_conflict": conflicting,
        "absolute_risk_panel_present": absolute_risk_panel_present,
    }


def build_scientific_maturity_audit(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    run_dir: Path,
    display_suite: Optional[Mapping[str, Any]] = None,
    publication_bundle: Optional[Mapping[str, Any]] = None,
    reviewer_report: Optional[Mapping[str, Any]] = None,
) -> ScientificMaturityAudit:
    findings: list[ScientificMaturityFinding] = []
    literature = _read_json(run_dir, "preplan_literature_bundle.json")
    provenance = literature.get("search_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    searched = bool(provenance.get("search_conducted"))
    sources_returning = list(provenance.get("sources_returning") or [])
    search_queries = provenance.get("search_queries")
    search_queries = search_queries if isinstance(search_queries, Mapping) else {}
    screening_decisions = [
        decision
        for decision in literature.get("screening_decisions") or []
        if isinstance(decision, Mapping)
    ]
    direct_comparator_keys = sorted(
        {
            str(decision.get("citation_key"))
            for decision in screening_decisions
            if decision.get("disposition") == "include"
            and decision.get("evidence_role") == "direct_comparator"
            and decision.get("publication_type_eligible", True) is not False
            and str(decision.get("citation_key") or "").strip()
        }
    )
    design_analogue_keys = sorted(
        {
            str(decision.get("citation_key"))
            for decision in screening_decisions
            if decision.get("disposition") == "include"
            and decision.get("evidence_role") == "design_analogue"
            and decision.get("publication_type_eligible", True) is not False
            and str(decision.get("citation_key") or "").strip()
        }
    )
    comparison_source_keys = sorted(
        set(direct_comparator_keys)
        | (set(design_analogue_keys) if not context.primary_exposure else set())
    )
    direct_required = bool(context.primary_exposure)
    citation_by_key = {
        str(record.get("key") or record.get("citation_key")): record
        for record in literature.get("citations") or []
        if isinstance(record, Mapping)
        and str(record.get("key") or record.get("citation_key") or "").strip()
    }
    direct_comparator_years = sorted(
        {
            int(str(citation_by_key[key].get("year") or "").strip())
            for key in direct_comparator_keys
            if key in citation_by_key
            and str(citation_by_key[key].get("year") or "").strip().isdigit()
        }
    )
    comparison_source_years = sorted(
        {
            int(str(citation_by_key[key].get("year") or "").strip())
            for key in comparison_source_keys
            if key in citation_by_key
            and str(citation_by_key[key].get("year") or "").strip().isdigit()
        }
    )
    search_year = datetime.now(timezone.utc).year
    searched_at = str(provenance.get("searched_at") or "")
    searched_year_match = re.search(r"\b(20\d{2})\b", searched_at)
    if searched_year_match:
        search_year = int(searched_year_match.group(1))
    newest_direct_comparator_year = (
        direct_comparator_years[-1] if direct_comparator_years else None
    )
    newest_comparison_source_year = (
        comparison_source_years[-1] if comparison_source_years else None
    )
    scientific_steps = _scientific_steps(plan)
    unbound = [
        str(step.step_id)
        for step in scientific_steps
        if not step.literature_citation_keys
    ]
    association_study = _association_study(plan)
    preferences = context.user_preferences
    covariate_selection = (
        preferences.covariate_selection
        if preferences is not None
        else "planner_selectable"
    )
    covariate_rationales = dict(getattr(preferences, "covariate_rationales", {}) or {})
    covariate_temporal_roles = dict(
        getattr(preferences, "covariate_temporal_roles", {}) or {}
    )
    post_baseline, exposure_window = _post_baseline_exposure(context)
    time_anchor_alignment = primary_exposure_time_anchor_alignment(context)
    patient_identity = _patient_identity_available(context)
    repeat_units_possible = _repeat_units_possible(context)
    method_facts = (
        method_source_facts(plan, context)
        if plan is not None
        else {
            "method_source_gaps": [],
            "method_layers_by_step": {},
            "required_method_layers": [],
            "missing_method_layers": [],
            "unsupported_method_bindings": [],
        }
    )
    method_source_gaps = list(method_facts["method_source_gaps"])
    method_layers_by_step = dict(method_facts["method_layers_by_step"])
    required_method_layers = list(method_facts["required_method_layers"])
    missing_method_layers = list(method_facts["missing_method_layers"])
    novelty = _novelty_facts(
        run_dir,
        comparison_source_keys=comparison_source_keys,
        expected_authority_digests=(
            novelty_authority_digests(
                context=context,
                plan=plan,
                literature=literature,
            )
            if plan is not None
            else {}
        ),
    )
    robustness = _robustness_facts(run_dir, plan)
    display = (
        dict(display_suite)
        if isinstance(display_suite, Mapping)
        else _read_json(run_dir, "display_suite_audit.json")
    )
    publication = (
        dict(publication_bundle) if isinstance(publication_bundle, Mapping) else {}
    )
    reviewer = (
        dict(reviewer_report)
        if isinstance(reviewer_report, Mapping)
        else _read_json(run_dir, "reviewer_report.json")
    )
    reviewer_summary = reviewer.get("summary")
    reviewer_summary = reviewer_summary if isinstance(reviewer_summary, Mapping) else {}
    endpoint = _endpoint_semantic_facts(context)
    manuscript = _manuscript_facts(run_dir)
    primary_figure = _primary_figure_facts(run_dir, plan)
    covariates = tuple(primary_figure["adjustment_covariates"])

    if not searched or not sources_returning:
        findings.append(
            ScientificMaturityFinding(
                code="TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED",
                severity="blocker",
                dimension="literature",
                message=(
                    "The plan used curated seed references only; current, direct "
                    "prior art and comparable ICU studies were not retrieved."
                ),
                evidence_refs=["preplan_literature_bundle.json"],
                remediation=(
                    "Run a dated database search with screening/provenance and bind "
                    "the retained definition, design-method, and direct-comparator "
                    "records before treating the run as a publication candidate."
                ),
            )
        )
    elif not any(list(values or []) for values in search_queries.values()):
        findings.append(
            ScientificMaturityFinding(
                code="LITERATURE_SEARCH_QUERY_NOT_RECORDED",
                severity="blocker",
                dimension="literature",
                message=(
                    "The retrieval receipt does not record the exact query used, "
                    "so the claimed prior-art search cannot be reproduced."
                ),
                evidence_refs=["preplan_literature_bundle.json.search_provenance"],
                remediation=(
                    "Persist each normalized source query beside the retrieval "
                    "timestamp and returned records before article review."
                ),
            )
        )
    if searched and not comparison_source_keys:
        findings.append(
            ScientificMaturityFinding(
                code=(
                    "DIRECT_COMPARATOR_SCREENING_NOT_ESTABLISHED"
                    if direct_required
                    else "DESIGN_ANALOGUE_SCREENING_NOT_ESTABLISHED"
                ),
                severity="blocker",
                dimension="literature",
                message=(
                    "Retrieved records have no inspectable, included "
                    + ("direct-comparator" if direct_required else "design-analogue")
                    + " screening decision for this exact ICU question."
                ),
                evidence_refs=["preplan_literature_bundle.json.screening_decisions"],
                remediation=(
                    "Screen each retrieved record against the declared population, "
                    + (
                        "exposure, outcome, and estimand"
                        if direct_required
                        else "clinical topic and analysis-design intent"
                    )
                    + "; retain a record-level decision without conflating design "
                    "analogy with a direct effect comparator."
                ),
            )
        )
    if (
        searched
        and newest_comparison_source_year is not None
        and search_year - newest_comparison_source_year > 5
    ):
        findings.append(
            ScientificMaturityFinding(
                code=(
                    "RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED"
                    if direct_required
                    else "RECENT_DESIGN_ANALOGUE_NOT_ESTABLISHED"
                ),
                severity="major",
                dimension="literature",
                message=(
                    "The newest retained "
                    + ("direct comparator" if direct_required else "design analogue")
                    + " predates the search "
                    f"year by {search_year - newest_comparison_source_year} years. "
                    "Older canonical sources may remain valid, but current similar "
                    "work has not been established for a timeliness/novelty claim."
                ),
                evidence_refs=[
                    "preplan_literature_bundle.json.citations",
                    "preplan_literature_bundle.json.search_provenance",
                ],
                remediation=(
                    "Review the exact search and document whether no recent direct "
                    "comparator exists or whether retrieval/screening missed one; "
                    "do not discard canonical older methods papers merely for age."
                ),
            )
        )
    primary_citation_keys = {
        key
        for step in scientific_steps
        if step.planned_analysis_role == "primary"
        for key in step.literature_citation_keys
    }
    if comparison_source_keys and set(comparison_source_keys).isdisjoint(
        primary_citation_keys
    ):
        findings.append(
            ScientificMaturityFinding(
                code=(
                    "DIRECT_COMPARATOR_NOT_BOUND_TO_PRIMARY_PLAN"
                    if direct_required
                    else "DESIGN_ANALOGUE_NOT_BOUND_TO_PRIMARY_PLAN"
                ),
                severity="blocker",
                dimension="literature_to_plan",
                message=(
                    "A screened comparison source survived, but no primary "
                    "analysis step binds it as design/comparison context."
                ),
                evidence_refs=[
                    "preplan_literature_bundle.json.screening_decisions",
                    "manifest.json.current_plan_authority",
                ],
                remediation=(
                    "Bind a screened comparator or design analogue to the primary "
                    "step alongside the "
                    "relevant method source; never infer borrowing from bundle presence."
                ),
            )
        )
    if not novelty["supported"]:
        findings.append(
            ScientificMaturityFinding(
                code="NOVELTY_POSITIONING_NOT_ESTABLISHED",
                severity="blocker",
                dimension="novelty",
                message=(
                    "The run has no independent, source-bound comparison showing "
                    "how its population/setting, exposure/time zero, outcome/"
                    "estimand, analysis/robustness, data-source transportability, "
                    "and clinical or methodological contribution differ from "
                    "retained comparison sources."
                ),
                evidence_refs=[
                    "preplan_literature_bundle.json.screening_decisions",
                    "novelty_positioning_audit.json",
                ],
                remediation=(
                    "Create a comparator matrix from retained source excerpts, "
                    "record substantive differences on all six dimensions, and "
                    "obtain independent review. A new database/concept instantiation "
                    "alone is not a novelty claim."
                ),
            )
        )
    if not endpoint["resolved"]:
        findings.append(
            ScientificMaturityFinding(
                code="OUTCOME_DEFINITION_UNRESOLVED",
                severity="blocker",
                dimension="icu_clinical_design",
                message="The primary outcome lacks an owner-issued clinical definition.",
                evidence_refs=["research_context.json.variables"],
                remediation=(
                    "Bind the physical endpoint to a typed concept definition and "
                    "time horizon before planning or interpreting the analysis."
                ),
            )
        )
    if endpoint["conflict"]:
        findings.append(
            ScientificMaturityFinding(
                code="OUTCOME_DEFINITION_CONFLICT",
                severity="blocker",
                dimension="icu_clinical_design",
                message=(
                    "The requested endpoint conflicts with the owner-issued meaning "
                    "of the materialized outcome column."
                ),
                evidence_refs=["research_context.json.variables.clinical_caveats"],
                remediation=(
                    "Select the matching physical endpoint or revise the research "
                    "question; never relabel the existing column from prose."
                ),
            )
        )
    if time_anchor_alignment.status in {"mismatch", "declared_only"}:
        mismatch = time_anchor_alignment.status == "mismatch"
        findings.append(
            ScientificMaturityFinding(
                code=(
                    "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH"
                    if mismatch
                    else "PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED"
                ),
                severity="blocker",
                dimension="icu_clinical_design",
                message=(
                    "The executed primary exposure is not bound to the same "
                    "clinical time anchor as the sealed study definition."
                ),
                evidence_refs=[
                    "research_context.json.user_preferences.timing_and_design",
                    "research_context.json.variables.clinical_definition",
                    "research_context.json.variables.analysis_window",
                    "research_context.json.variables.analysis_window_role",
                ],
                remediation=(
                    "Supersede this run with a new StudyContext/concept-authority "
                    "version whose typed exposure definition and declared time zero "
                    "have the same owner-issued identity. Keep any outer physical "
                    "observation window as a separate coordinate."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Should the study adopt the owner-issued clinical-definition "
                    "anchor or issue a new exposure definition around the declared "
                    "clinical anchor?"
                ),
            )
        )
    if unbound:
        findings.append(
            ScientificMaturityFinding(
                code="FINAL_PLAN_LITERATURE_BINDING_INCOMPLETE",
                severity="blocker",
                dimension="literature_to_plan",
                message=(
                    "The final executed plan contains scientific steps without exact "
                    "run-bound citation keys: " + ", ".join(unbound)
                ),
                evidence_refs=[
                    "manifest.json.current_plan_authority",
                    "preplan_literature_bundle.json",
                ],
                remediation=(
                    "Reject any replan that drops the initial citation authority and "
                    "require every primary, secondary, and sensitivity step to retain "
                    "an exact supported key."
                ),
            )
        )
    if method_source_gaps:
        findings.append(
            ScientificMaturityFinding(
                code="SCIENTIFIC_STEP_METHOD_SOURCE_NOT_BOUND",
                severity="major",
                dimension="literature_to_plan",
                message=(
                    "Scientific steps cite no retained source that governs a "
                    "methodological decision: " + ", ".join(method_source_gaps)
                ),
                evidence_refs=[
                    "manifest.json.current_plan_authority",
                    "preplan_literature_bundle.json",
                    "method_literature_pack",
                ],
                remediation=(
                    "Bind each scientific step to a relevant method card, not merely "
                    "a disease definition or database paper, and retain the exact key."
                ),
            )
        )
    if method_facts["unsupported_method_bindings"]:
        rows = [
            f"{item['step_id']}:{item['citation_key']}="
            + ",".join(item["unsupported_design_elements"])
            for item in method_facts["unsupported_method_bindings"]
        ]
        findings.append(
            ScientificMaturityFinding(
                code="METHOD_SOURCE_DESIGN_ELEMENT_UNSUPPORTED",
                severity="major",
                dimension="literature_to_plan",
                message=(
                    "The executed plan credits method sources for decisions their "
                    "sealed cards do not support: " + "; ".join(rows)
                ),
                evidence_refs=[
                    "manifest.json.current_plan_authority",
                    "method_literature_pack",
                ],
                remediation=(
                    "Bind each executed design element to a method card that "
                    "explicitly supports it; citation presence alone is not credit."
                ),
            )
        )
    if missing_method_layers:
        findings.append(
            ScientificMaturityFinding(
                code="APPLICABLE_METHOD_LAYERS_NOT_BOUND",
                severity="major",
                dimension="literature_to_plan",
                message=(
                    "The final plan does not bind source-backed guidance for "
                    "case-applicable design layers: " + ", ".join(missing_method_layers)
                ),
                evidence_refs=[
                    "manifest.json.current_plan_authority",
                    "method_literature_pack",
                ],
                remediation=(
                    "Bind the relevant timing, dependence, missing-data, functional-"
                    "form, interpretation, and reporting sources before execution; "
                    "a generic reporting citation cannot stand in for every decision."
                ),
            )
        )
    if post_baseline and not _timing_design_closed(plan):
        findings.append(
            ScientificMaturityFinding(
                code="POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
                severity="blocker",
                dimension="icu_clinical_design",
                message=(
                    "Exposure status is ascertained after time zero, but the final "
                    "plan does not close exposure opportunity, early events, or a "
                    "landmark/time-varying alternative."
                ),
                evidence_refs=[
                    "research_context.json",
                    "manifest.json.current_plan_authority",
                ],
                remediation=(
                    "Pre-specify a non-overlapping follow-up strategy and early "
                    "death/discharge accounting, or explicitly retain the estimate "
                    "as descriptive and non-article-grade."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "The exposure is classified after ICU time zero. Should a new "
                    "study version use a prespecified landmark/follow-up design, or "
                    "should this run remain descriptive only?"
                ),
            )
        )
    if repeat_units_possible and not patient_identity:
        findings.append(
            ScientificMaturityFinding(
                code="REPEATED_STAY_DEPENDENCE_UNRESOLVED",
                severity="major",
                dimension="icu_clinical_design",
                message=(
                    "ICU stays are the analysis unit, but patient identity is "
                    "unavailable; repeated stays cannot be identified or clustered."
                ),
                evidence_refs=["research_context.json.cohort.provenance"],
                remediation=(
                    "Materialize the patient-level identifier and pre-specify one "
                    "stay per patient or clustered/mixed estimation; otherwise make "
                    "the dependence limitation explicit and keep paper authority off."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "May a new study version materialize patient identity and choose "
                    "first-stay or clustered estimation, or should all ICU stays "
                    "remain the authorized analysis unit?"
                ),
            )
        )
    elif repeat_units_possible and not _repeated_unit_design_closed(context, plan):
        findings.append(
            ScientificMaturityFinding(
                code="REPEATED_STAY_METHOD_NOT_DECLARED",
                severity="major",
                dimension="icu_clinical_design",
                message="Patient identity exists, but the plan does not address repeated stays.",
                evidence_refs=[
                    "research_context.json",
                    "manifest.json.current_plan_authority",
                ],
                remediation="Declare one-stay selection or clustered/mixed estimation.",
                requires_user_authorization=True,
                authorization_question=(
                    "Should a new study version use one ICU stay per patient or "
                    "retain repeated stays with clustered/mixed estimation?"
                ),
            )
        )
    if association_study and not covariates:
        findings.append(
            ScientificMaturityFinding(
                code="UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
                severity="major",
                dimension="statistical_design",
                message=(
                    "The exact user-authorized primary association is unadjusted. "
                    "It is valid as descriptive analysis, but not a top-journal "
                    "estimate of an independent association."
                ),
                evidence_refs=[
                    "manifest.json.current_plan_authority",
                    "research_context.json.user_preferences",
                ],
                remediation=(
                    "Ask the user to authorize a clinically timed covariate strategy "
                    "and preserve the unadjusted result as descriptive context; do "
                    "not silently add covariates."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "The requested estimate is explicitly unadjusted. Do you want "
                    "to keep this descriptive analysis, or authorize a new study "
                    "version with a clinically timed adjustment strategy?"
                ),
            )
        )
    if covariates and covariate_selection != "exact":
        findings.append(
            ScientificMaturityFinding(
                code="ADJUSTMENT_SET_NOT_USER_CONFIRMED",
                severity="major",
                dimension="statistical_design",
                message=(
                    "The executed primary model uses a covariate roster that the "
                    "user had supplied only as Planner-selectable candidates."
                ),
                evidence_refs=[
                    "manifest.json.current_plan_authority",
                    "research_context.json.user_preferences",
                ],
                remediation=(
                    "Create a new StudyContext revision that records the exact "
                    "adjustment roster, its clinical rationale, and its baseline "
                    "temporal role before treating the estimate as article-grade."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Do you approve the executed covariate roster and its clinical/"
                    "time-zero rationale in a new study version?"
                ),
            )
        )
    elif covariates and (
        set(covariate_rationales) != set(covariates)
        or set(covariate_temporal_roles) != set(covariates)
    ):
        findings.append(
            ScientificMaturityFinding(
                code="ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND",
                severity="major",
                dimension="statistical_design",
                message=(
                    "The exact executed adjustment roster lacks a complete "
                    "user-reviewed clinical rationale or pre-time-zero temporal role."
                ),
                evidence_refs=["research_context.json.user_preferences"],
                remediation=(
                    "Record one confounding rationale and one baseline temporal role "
                    "for every exact covariate in a new StudyContext revision."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Do you approve the clinical rationale and baseline timing for "
                    "every exact adjustment covariate in a new study version?"
                ),
            )
        )
    if len(robustness["declared_axes"]) < 2:
        findings.append(
            ScientificMaturityFinding(
                code="ROBUSTNESS_AXES_TOO_NARROW",
                severity="major",
                dimension="robustness",
                message=(
                    "The final plan tests fewer than two distinct, executable "
                    "robustness axes."
                ),
                evidence_refs=["robustness_specs_locked.json", "robustness_panel.json"],
                remediation=(
                    "Pre-specify task-supported definition/window, cohort, outcome, "
                    "or missing-data alternatives; never invent an unsupported variant."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Should a new study version add prespecified, source-supported "
                    "sensitivity axes, or should the current narrow analysis remain "
                    "analysis-only?"
                ),
            )
        )
    if robustness["all_variants_duplicate_primary"]:
        findings.append(
            ScientificMaturityFinding(
                code="ROBUSTNESS_REPLAY_DUPLICATES_PRIMARY",
                severity="major",
                dimension="robustness",
                message=(
                    "Every robustness variant reproduces the primary sample and "
                    "estimate exactly, so it adds no empirical stress test."
                ),
                evidence_refs=["robustness_panel.json"],
                remediation=(
                    "Label the replay as an equivalence audit and require at least "
                    "one genuinely distinct supported specification for article maturity."
                ),
            )
        )
    if not bool(display.get("display_suite_complete")):
        findings.append(
            ScientificMaturityFinding(
                code="ARTICLE_DISPLAY_SUITE_INCOMPLETE",
                severity="major",
                dimension="figures",
                message="The owner-issued article display suite audit is incomplete.",
                evidence_refs=["display_suite_audit.json"],
                remediation="Close the exact missing figure/table roles before submission review.",
            )
        )
    if not bool(publication.get("publication_figure_contract_ready")):
        findings.append(
            ScientificMaturityFinding(
                code="PUBLICATION_FIGURE_CONTRACT_NOT_VERIFIED",
                severity="major",
                dimension="figures",
                message=(
                    "No verified primary-publication figure contract binds the "
                    "figure claim and panels to the executed run."
                ),
                evidence_refs=["publication_figures/*.figure_contract.json"],
                remediation=(
                    "Generate the publication figure from a typed FigureContract "
                    "with reader-facing panel roles and digest-registered exports."
                ),
            )
        )
    if not bool(publication.get("publication_figure_source_data_ready")):
        findings.append(
            ScientificMaturityFinding(
                code="PUBLICATION_FIGURE_SOURCE_DATA_NOT_VERIFIED",
                severity="blocker",
                dimension="figures",
                message=(
                    "The primary publication figure has no verified registered "
                    "source-data chain for every plotted value and denominator."
                ),
                evidence_refs=[
                    "publication_figures/*.figure_contract.json",
                    "publication_figure_source_data",
                ],
                remediation=(
                    "Export exact plotted source data, units, denominators, and "
                    "evidence ids; register and digest-check them before publication."
                ),
            )
        )
    if not bool(publication.get("publication_figure_visual_qa_passed")):
        findings.append(
            ScientificMaturityFinding(
                code="PUBLICATION_FIGURE_VISUAL_QA_NOT_PASSED",
                severity="blocker",
                dimension="figures",
                message=(
                    "Deterministic/model visual QA has unresolved layout, label, "
                    "or export findings for the primary publication figure."
                ),
                evidence_refs=["visual_qa.json", "publication_figures"],
                remediation=(
                    "Repair the exact rendered figure and rerun visual/export QA; "
                    "a syntactically valid contract alone is not visual acceptance."
                ),
            )
        )
    if primary_figure["adjustment_label_conflict"]:
        findings.append(
            ScientificMaturityFinding(
                code="PRIMARY_FIGURE_ADJUSTMENT_LABEL_CONFLICT",
                severity="blocker",
                dimension="figures",
                message=(
                    "The primary publication figure's adjustment label conflicts "
                    "with the executed primary model covariate roster."
                ),
                evidence_refs=primary_figure["primary_contract_paths"],
                remediation=(
                    "Derive Adjusted/Unadjusted from the typed primary model "
                    "requirement; do not let prose or a generic filename decide it."
                ),
            )
        )
    if association_study and not primary_figure["absolute_risk_panel_present"]:
        findings.append(
            ScientificMaturityFinding(
                code="PRIMARY_FIGURE_ABSOLUTE_RISK_CONTEXT_MISSING",
                severity="major",
                dimension="figures",
                message=(
                    "The primary association figure does not include observed "
                    "absolute outcome risk/prevalence by exposure."
                ),
                evidence_refs=primary_figure["primary_contract_paths"],
                remediation=(
                    "Pair the relative association with an owner-issued absolute-"
                    "risk panel and denominators in the primary publication figure."
                ),
            )
        )
    if manuscript["literature_audit_status"] != "pass":
        findings.append(
            ScientificMaturityFinding(
                code="MANUSCRIPT_EXACT_LITERATURE_BINDING_INCOMPLETE",
                severity="blocker",
                dimension="manuscript",
                message=(
                    "The manuscript does not use only exact keys from the run-bound "
                    "literature bundle."
                ),
                evidence_refs=["manuscript_literature_audit.json"],
                remediation=(
                    "Regenerate the draft with exact [@key] citations and reject "
                    "unknown or aggregate-search-only support."
                ),
            )
        )
    if manuscript["missing_sections"]:
        findings.append(
            ScientificMaturityFinding(
                code="MANUSCRIPT_REQUIRED_SECTIONS_INCOMPLETE",
                severity="major",
                dimension="manuscript",
                message=(
                    "Required article sections are missing: "
                    + ", ".join(manuscript["missing_sections"])
                ),
                evidence_refs=["manuscript_scaffold_bound.md"],
                remediation="Regenerate a complete reporting-standard manuscript scaffold.",
            )
        )
    if manuscript["thin_sections"]:
        findings.append(
            ScientificMaturityFinding(
                code="MANUSCRIPT_CORE_SECTIONS_TOO_THIN",
                severity="major",
                dimension="manuscript",
                message=(
                    "Core manuscript sections remain stub-like: "
                    + ", ".join(manuscript["thin_sections"])
                    + "."
                ),
                evidence_refs=["manuscript_scaffold_bound.md"],
                remediation=(
                    "Expand only from run-bound evidence and exact literature keys: "
                    "state the clinical rationale, reproducible design, complete "
                    "results, interpretation, comparison and limitations. The "
                    "section floors are anti-stub checks, not journal word targets."
                ),
            )
        )
    if not manuscript["pdf_present"] or not manuscript["pdf_receipt_present"]:
        findings.append(
            ScientificMaturityFinding(
                code="MANUSCRIPT_RENDER_QA_INCOMPLETE",
                severity="major",
                dimension="manuscript",
                message="No complete digest-bound manuscript PDF render receipt is available.",
                evidence_refs=["manuscript_scaffold.pdf", "manuscript_pdf_receipt"],
                remediation="Render the exact bound manuscript and persist its verification receipt.",
            )
        )
    recommendation = str(reviewer_summary.get("aggregated_recommendation") or "")
    if not recommendation:
        findings.append(
            ScientificMaturityFinding(
                code="INDEPENDENT_SCIENTIFIC_REVIEW_NOT_AVAILABLE",
                severity="blocker",
                dimension="clinical_review",
                message="No owner-issued independent scientific review receipt is available.",
                evidence_refs=["reviewer_report.json"],
                remediation=(
                    "Generate the clinical/methodological reviewer receipt and keep "
                    "human sign-off separate from the Agent's own review."
                ),
            )
        )
    elif recommendation in {"major_revision", "reject"}:
        findings.append(
            ScientificMaturityFinding(
                code="INDEPENDENT_SCIENTIFIC_REVIEW_NOT_CLOSED",
                severity="blocker",
                dimension="clinical_review",
                message=f"The reviewer receipt remains {recommendation}.",
                evidence_refs=["reviewer_report.json"],
                remediation="Resolve and regenerate the independent scientific review receipt.",
            )
        )

    dimensions = {
        "literature": 100,
        "novelty": 100,
        "literature_to_plan": 100,
        "icu_clinical_design": 100,
        "statistical_design": 100,
        "robustness": 100,
        "figures": 100,
        "manuscript": 100,
        "clinical_review": 100,
    }
    penalty = {"blocker": 65, "major": 35, "minor": 15}
    for finding in findings:
        dimensions[finding.dimension] = max(
            0,
            dimensions.get(finding.dimension, 100) - penalty[finding.severity],
        )
    score = round(sum(dimensions.values()) / max(1, len(dimensions)))
    blocker = any(finding.severity == "blocker" for finding in findings)
    major = any(finding.severity == "major" for finding in findings)
    status = (
        "analysis_only" if blocker else ("major_revision" if major else "article_grade")
    )
    return ScientificMaturityAudit(
        status=status,
        article_grade=status == "article_grade",
        score=score,
        dimension_scores=dimensions,
        findings=findings,
        facts={
            "literature_search_conducted": searched,
            "literature_sources_returning": sources_returning,
            "literature_search_queries": {
                str(source): list(values or [])
                for source, values in search_queries.items()
            },
            "direct_comparator_keys": direct_comparator_keys,
            "design_analogue_keys": design_analogue_keys,
            "comparison_source_keys": comparison_source_keys,
            "direct_comparator_years": direct_comparator_years,
            "comparison_source_years": comparison_source_years,
            "newest_direct_comparator_year": newest_direct_comparator_year,
            "newest_comparison_source_year": newest_comparison_source_year,
            "literature_search_year": search_year,
            "primary_plan_citation_keys": sorted(primary_citation_keys),
            "endpoint_semantics": endpoint,
            "manuscript": manuscript,
            "primary_figure": primary_figure,
            "scientific_step_count": len(scientific_steps),
            "unbound_scientific_steps": unbound,
            "primary_covariates": list(covariates),
            "covariate_selection": covariate_selection,
            "covariate_rationales": covariate_rationales,
            "covariate_temporal_roles": covariate_temporal_roles,
            "association_study": association_study,
            "post_baseline_exposure": post_baseline,
            "exposure_window": exposure_window,
            "primary_exposure_time_anchor_alignment": (time_anchor_alignment.to_dict()),
            "patient_identity_available": patient_identity,
            "repeat_units_possible": repeat_units_possible,
            "method_source_gaps": method_source_gaps,
            "method_layers_by_step": method_layers_by_step,
            "required_method_layers": required_method_layers,
            "missing_method_layers": missing_method_layers,
            "unsupported_method_bindings": method_facts["unsupported_method_bindings"],
            "novelty": novelty,
            **robustness,
            "display_suite_complete": bool(display.get("display_suite_complete")),
            "publication_figure": {
                "bundle_ready": bool(
                    publication.get("publication_figure_bundle_ready")
                ),
                "contract_ready": bool(
                    publication.get("publication_figure_contract_ready")
                ),
                "source_data_ready": bool(
                    publication.get("publication_figure_source_data_ready")
                ),
                "visual_qa_passed": bool(
                    publication.get("publication_figure_visual_qa_passed")
                ),
                "visual_qa_errors": list(
                    publication.get("publication_figure_visual_qa_errors") or []
                )[:20],
            },
            "reviewer_recommendation": recommendation or "not_available",
        },
    )


__all__ = [
    "SCIENTIFIC_MATURITY_AUDIT_REGISTRATION",
    "ScientificMaturityAudit",
    "ScientificMaturityFinding",
    "build_scientific_maturity_audit",
    "scientific_maturity_audit_from_gates",
    "scientific_maturity_readiness_gates",
]
