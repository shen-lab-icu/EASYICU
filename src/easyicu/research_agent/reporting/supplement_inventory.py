"""Host-owned inventory for a study's supplementary analysis package.

The inventory does not invent tables or declare a paper complete. It maps
registered evidence and typed step outputs to study-family sections, records
which expected sections are absent, and gives the manuscript/reviewer one
stable place to audit the supplement boundary.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..authority.evidence_store import EvidenceStore
from ..contracts.runtime import ValidationFinding


_SECTION_TOKENS: dict[str, tuple[str, ...]] = {
    "cohort_accounting": ("cohort", "attrition", "risk_set", "flow"),
    "baseline_characteristics": ("table_one", "baseline", "characteristic"),
    "missingness_measurement": ("missing", "measurement", "completeness"),
    "primary_results": (
        "primary",
        "effect",
        "model",
        "association",
        "estimate",
        "feasibility",
    ),
    "robustness_sensitivity": ("robust", "sensitivity", "alternative", "complete_case"),
    "figure_source_provenance": ("figure", "source_data", "plot", "contract"),
    "reproducibility": ("receipt", "provenance", "environment", "code", "lock"),
    "calibration": ("calibration", "brier"),
    "discrimination": ("auroc", "roc", "precision_recall", "performance"),
    "validation": ("validation", "heldout", "held_out", "split"),
    "clinical_utility": ("decision_curve", "clinical_utility", "net_benefit"),
    "cluster_selection": ("cluster_selection", "candidate", "silhouette", "bic"),
    "cluster_stability": ("cluster_stability", "stability", "adjusted_rand"),
    "external_reproducibility": (
        "external_reproducibility",
        "cross_cohort_replication",
        "replication_cohort",
        "transportability_assessment",
    ),
    "external_validation": (
        "external_validation",
        "external_validation_cohort",
        "transport_validation",
    ),
    "resampling_validation": (
        "bootstrap_validation",
        "repeated_split",
        "cross_validation",
        "resampling_validation",
    ),
    "ph_diagnostics": ("schoenfeld", "proportional_hazards", "ph_diagnostic"),
    "non_ph_alternative": ("rmst", "time_varying", "extended_cox", "non_ph"),
    "trajectory_window_missingness": ("trajectory", "window", "missing"),
    "alternative_algorithm": ("alternative_algorithm", "algorithm_agreement", "gbtm"),
    "identified_comparator": (
        "verified_comparator",
        "positivity_supported",
        "identified_causal_contrast",
    ),
}

_COMMON_REQUIRED = (
    "cohort_accounting",
    "baseline_characteristics",
    "missingness_measurement",
    "primary_results",
    "robustness_sensitivity",
    "figure_source_provenance",
    "reproducibility",
)

_FAMILY_DEVELOPMENT_REQUIRED: dict[str, tuple[str, ...]] = {
    "prediction": (
        "calibration",
        "discrimination",
        "validation",
        "clinical_utility",
        "resampling_validation",
    ),
    "phenotyping": (
        "cluster_selection",
        "cluster_stability",
        "alternative_algorithm",
    ),
    "survival": ("ph_diagnostics", "non_ph_alternative"),
    "trajectory_clustering": (
        "cluster_selection",
        "cluster_stability",
        "trajectory_window_missingness",
        "alternative_algorithm",
    ),
}

_FAMILY_TOP_JOURNAL_EXTENSION: dict[str, tuple[str, ...]] = {
    "association_study": ("external_reproducibility",),
    "prediction": ("external_validation",),
    "phenotyping": ("external_reproducibility",),
    "trajectory_clustering": ("external_reproducibility",),
}

_TERMINAL_DEVELOPMENT_REQUIRED: dict[str, tuple[str, ...]] = {
    "source_feasibility_fail_closed": (
        "primary_results",
        "reproducibility",
    ),
    "prespecified_selection_no_solution": (
        "cohort_accounting",
        "missingness_measurement",
        "cluster_selection",
        "figure_source_provenance",
        "reproducibility",
    ),
}

_TERMINAL_TOP_JOURNAL_EXTENSION: dict[str, tuple[str, ...]] = {
    "source_feasibility_fail_closed": ("identified_comparator",),
    "prespecified_selection_no_solution": (
        "baseline_characteristics",
        "alternative_algorithm",
        "external_reproducibility",
    ),
}

_FAMILY_ALIASES = {
    "prediction_model": "prediction",
    "mortality_prediction": "prediction",
    "risk_prediction": "prediction",
    "subphenotyping": "phenotyping",
    "phenotype_clustering": "phenotyping",
    "sepsis_subphenotype": "phenotyping",
    "time_to_event": "survival",
}


def _analysis_family(plan: object) -> str:
    raw = str(getattr(plan, "analysis_type", "") or "").strip().casefold()
    return _FAMILY_ALIASES.get(raw, raw)


def _artifact_text(item: Mapping[str, Any]) -> str:
    return " ".join(str(value or "") for value in item.values()).casefold()


def _sections_for(item: Mapping[str, Any]) -> list[str]:
    text = _artifact_text(item)
    return [
        section
        for section, tokens in _SECTION_TOKENS.items()
        if any(token in text for token in tokens)
    ]


def _step_artifacts(
    per_step_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for record in per_step_records:
        step_id = str(record.get("step_id") or "")
        summary = record.get("step_summary")
        if not isinstance(summary, Mapping):
            continue
        output_files = summary.get("output_files")
        if not isinstance(output_files, Mapping):
            continue
        for product, filename in output_files.items():
            artifacts.append(
                {
                    "source": "typed_step_output",
                    "step_id": step_id,
                    "product": str(product),
                    "path": f"steps/{step_id}/outputs/{filename}",
                    "method": str(summary.get("method") or ""),
                    "analysis_kind": str(
                        summary.get("deterministic_standard_analysis") or ""
                    ),
                }
            )
        if isinstance(summary.get("missingness_measurement_audit"), Mapping):
            receipt_files = [
                filename
                for product, filename in output_files.items()
                if str(product).startswith("log:")
                and "receipt" in str(product).casefold()
            ]
            if len(receipt_files) == 1:
                artifacts.append(
                    {
                        "source": "typed_step_metadata",
                        "step_id": step_id,
                        "product": "audit:missingness_measurement",
                        "path": f"steps/{step_id}/outputs/{receipt_files[0]}",
                        "method": str(summary.get("method") or ""),
                        "analysis_kind": str(
                            summary.get("deterministic_standard_analysis") or ""
                        ),
                    }
                )
    return artifacts


def _terminal_disposition(
    per_step_records: Sequence[Mapping[str, Any]],
) -> str | None:
    summaries = [
        record.get("step_summary")
        for record in per_step_records
        if isinstance(record.get("step_summary"), Mapping)
    ]
    if any(
        summary.get("analysis_family") == "causal_feasibility"
        and summary.get("scientific_decision") == "blocked_by_source_authority"
        and summary.get("causal_contrast_authorized") is False
        and summary.get("effect_estimate") is None
        for summary in summaries
    ):
        return "source_feasibility_fail_closed"
    if any(
        summary.get("scientific_status") == "failed_closed"
        and summary.get("reportable_result")
        in {
            "no_interior_solution_in_prespecified_candidate_range",
            "no_stable_phenotype_solution",
        }
        and bool(summary.get("reason_code"))
        for summary in summaries
    ):
        return "prespecified_selection_no_solution"
    return None


def write_supplement_inventory(
    *,
    plan: object,
    evidence: EvidenceStore,
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> tuple[dict[str, Any], list[ValidationFinding]]:
    """Write and register one nonauthoritative supplement coverage inventory."""

    family = _analysis_family(plan)
    terminal_disposition = _terminal_disposition(per_step_records)
    development_required = list(
        _TERMINAL_DEVELOPMENT_REQUIRED.get(
            terminal_disposition or "",
            _COMMON_REQUIRED,
        )
    )
    if terminal_disposition is None:
        development_required.extend(_FAMILY_DEVELOPMENT_REQUIRED.get(family, ()))
    top_journal_required = list(development_required)
    top_journal_required.extend(
        _TERMINAL_TOP_JOURNAL_EXTENSION.get(
            terminal_disposition or "",
            _FAMILY_TOP_JOURNAL_EXTENSION.get(family, ()),
        )
    )
    artifacts = [
        {
            "source": "evidence_store",
            "evidence_id": record.evidence_id,
            "kind": record.kind,
            "path": record.relative_path,
            "description": record.description,
            "produced_by_step": record.produced_by_step,
            "sha256": record.sha256,
        }
        for record in evidence.verified_records()
    ]
    artifacts.extend(_step_artifacts(per_step_records))
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        for section in _sections_for(artifact):
            by_section[section].append(artifact)
    sections = {
        section: {
            "required": section in top_journal_required,
            "present": bool(by_section.get(section)),
            "artifact_count": len(by_section.get(section, [])),
            "artifacts": by_section.get(section, []),
        }
        for section in sorted(set(_SECTION_TOKENS) | set(top_journal_required))
    }
    missing_development = [
        section
        for section in development_required
        if not sections[section]["present"]
    ]
    missing_top_journal = [
        section
        for section in top_journal_required
        if not sections[section]["present"]
    ]
    payload = {
        "schema_version": "easyicu.supplement_inventory/1",
        "authority": "analysis_only_inventory",
        "analysis_family": family,
        "terminal_disposition": terminal_disposition,
        # Backward-compatible strict axis. Existing consumers that read these
        # fields continue to require the top-journal extension.
        "required_sections": top_journal_required,
        "missing_required_sections": missing_top_journal,
        "supplement_complete": not missing_top_journal,
        "development_required_sections": development_required,
        "missing_development_required_sections": missing_development,
        "development_supplement_complete": not missing_development,
        "top_journal_required_sections": top_journal_required,
        "missing_top_journal_required_sections": missing_top_journal,
        "top_journal_supplement_complete": not missing_top_journal,
        "claim_boundary": (
            "Coverage records presence, not scientific adequacy, external validity, "
            "human review, release status, or publication readiness."
        ),
        "sections": sections,
    }
    lines = [
        "# Supplement analysis inventory",
        "",
        f"- Analysis family: `{family or 'unresolved'}`",
        "- Dev9 coverage status: "
        f"`{'complete' if not missing_development else 'incomplete'}`",
        "- Top-journal extension status: "
        f"`{'complete' if not missing_top_journal else 'incomplete'}`",
        "- Authority: `analysis_only_inventory`",
        "",
        "This inventory records registered coverage only. It does not certify "
        "scientific adequacy, release, human review, or publication readiness.",
        "",
        "| Section | Required | Present | Artifacts |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {section} | {'yes' if value['required'] else 'no'} | "
        f"{'yes' if value['present'] else 'no'} | {value['artifact_count']} |"
        for section, value in sections.items()
    )
    if missing_development:
        lines.extend(
            [
                "",
                "Missing Dev9-required sections: "
                + ", ".join(missing_development)
                + ".",
            ]
        )
    if missing_top_journal:
        lines.extend(
            [
                "",
                "Missing top-journal sections: "
                + ", ".join(missing_top_journal)
                + ".",
            ]
        )

    json_record = evidence.register_json(
        kind="log",
        description="Machine-readable supplement analysis coverage inventory.",
        payload=payload,
        filename="supplement_inventory.json",
        evidence_id="supplement_inventory",
        producer="supplement_inventory",
        generation_mode="system",
        on_sha_change="new_id",
    )
    md_record = evidence.register_text(
        kind="log",
        description="Human-readable supplement analysis coverage inventory.",
        text="\n".join(lines) + "\n",
        filename="supplement_inventory.md",
        evidence_id="supplement_inventory_markdown",
        inputs=[json_record.evidence_id],
        producer="supplement_inventory",
        generation_mode="system",
        on_sha_change="new_id",
    )
    for record, alias in (
        (json_record, "supplement_inventory.json"),
        (md_record, "supplement_inventory.md"),
    ):
        source = Path(run_dir) / record.relative_path
        destination = Path(run_dir) / alias
        destination.write_bytes(source.read_bytes())

    findings: list[ValidationFinding] = []
    if missing_top_journal:
        findings.append(
            ValidationFinding(
                validator="supplement_inventory",
                severity="warning",
                message=(
                    "The study-family supplementary analysis inventory is "
                    "incomplete: " + ", ".join(missing_top_journal) + "."
                ),
                evidence_ids=[json_record.evidence_id, md_record.evidence_id],
                detail={
                    "analysis_family": family,
                    "missing_development_required_sections": missing_development,
                    "missing_top_journal_required_sections": missing_top_journal,
                    "owner": "reporting.supplement_inventory",
                },
            )
        )
    return payload, findings


__all__ = ["write_supplement_inventory"]
