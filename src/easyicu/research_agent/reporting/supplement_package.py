"""Assemble an evidence-bound supplementary appendix without inventing results."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..authority.evidence_store import EvidenceStore


_CLAIM_BOUNDARIES = {
    "cohort_accounting": "Supports reported denominators and attrition; not representativeness or external validity.",
    "baseline_characteristics": "Describes the analyzed population; not exchangeability or adequate confounding control.",
    "missingness_measurement": "Describes observed missingness and measurement opportunity; not missing-at-random assumptions.",
    "primary_results": "Supports the declared estimand under its model contract; not causal or external claims unless separately authorized.",
    "robustness_sensitivity": "Tests only the prespecified axes shown; not robustness to every plausible analytic choice.",
    "figure_source_provenance": "Binds displays to registered source artifacts; not independent scientific validation.",
    "reproducibility": "Supports rerunning the recorded workflow; not cross-site reproducibility.",
    "calibration": "Supports calibration only in the evaluated split or cohort.",
    "discrimination": "Supports ranking performance only in the evaluated split or cohort.",
    "validation": "Supports the declared internal or held-out validation design; not external transportability.",
    "clinical_utility": "Supports net benefit only for declared thresholds and assumptions.",
    "resampling_validation": "Supports internal resampling stability; not external validation.",
    "external_validation": "Supports transport only to the named independent cohort and setting.",
    "cluster_selection": "Supports the recorded candidate-model comparison; not biological reality of clusters.",
    "cluster_stability": "Supports assignment stability under the recorded perturbations; not cross-site reproducibility.",
    "alternative_algorithm": "Supports agreement or disagreement with the named alternative algorithm only.",
    "external_reproducibility": "Supports replication only in the named independent cohort and implementation.",
    "ph_diagnostics": "Audits the proportional-hazards assumption; passing does not prove the model is otherwise correct.",
    "non_ph_alternative": "Provides the named non-PH estimand; it does not repair unmeasured confounding.",
    "trajectory_window_missingness": "Describes temporal support and missingness; not latent trajectory truth.",
}


def _file_binding(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    relative_path = str(artifact.get("path") or "")
    path = run_dir / relative_path
    observed_sha256 = (
        hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    )
    declared_sha256 = artifact.get("sha256")
    return {
        "source": str(artifact.get("source") or ""),
        "evidence_id": artifact.get("evidence_id"),
        "product": artifact.get("product"),
        "step_id": artifact.get("step_id") or artifact.get("produced_by_step"),
        "path": relative_path,
        "declared_sha256": declared_sha256,
        "observed_sha256": observed_sha256,
        "file_present": path.is_file(),
        "digest_verified": bool(
            observed_sha256
            and (declared_sha256 is None or declared_sha256 == observed_sha256)
        ),
    }


def write_supplement_package(
    *,
    inventory: Mapping[str, Any],
    evidence: EvidenceStore,
    run_dir: Path,
) -> dict[str, Any]:
    """Write one appendix index whose every listed artifact is file-bound."""

    run_dir = Path(run_dir)
    required = list(inventory.get("top_journal_required_sections") or ())
    sections_source = inventory.get("sections")
    if not isinstance(sections_source, Mapping):
        raise ValueError("supplement package requires a structured inventory")
    section_payload: dict[str, Any] = {}
    for section in required:
        source = sections_source.get(section)
        if not isinstance(source, Mapping):
            raise ValueError(f"supplement inventory omitted section {section!r}")
        artifacts = source.get("artifacts") or []
        if not isinstance(artifacts, list):
            raise ValueError(f"supplement section {section!r} artifacts are invalid")
        bindings = [_file_binding(run_dir, item) for item in artifacts]
        section_payload[section] = {
            "development_required": section
            in set(inventory.get("development_required_sections") or ()),
            "top_journal_required": True,
            "present": bool(source.get("present")),
            "claim_boundary": _CLAIM_BOUNDARIES.get(
                section,
                "Presence supports only the named section and registered artifacts.",
            ),
            "artifact_bindings": bindings,
            "all_files_digest_verified": bool(bindings)
            and all(item["digest_verified"] for item in bindings),
        }

    payload = {
        "schema_version": "easyicu.supplement_evidence_manifest/1",
        "authority": "analysis_only_evidence_index",
        "analysis_family": inventory.get("analysis_family"),
        "development_supplement_complete": bool(
            inventory.get("development_supplement_complete")
        ),
        "top_journal_supplement_complete": bool(
            inventory.get("top_journal_supplement_complete")
        ),
        "missing_development_required_sections": list(
            inventory.get("missing_development_required_sections") or ()
        ),
        "missing_top_journal_required_sections": list(
            inventory.get("missing_top_journal_required_sections") or ()
        ),
        "claim_boundary": (
            "This appendix is an evidence-bound index. It does not create missing "
            "analyses, certify scientific adequacy, or authorize publication."
        ),
        "sections": section_payload,
    }
    manifest_record = evidence.register_json(
        kind="log",
        description="Evidence-bound supplementary appendix manifest.",
        payload=payload,
        filename="supplement_evidence_manifest.json",
        evidence_id="supplement_evidence_manifest",
        producer="supplement_package",
        generation_mode="system",
        on_sha_change="new_id",
    )

    lines = [
        "# Supplementary evidence appendix",
        "",
        "- Authority: `analysis_only_evidence_index`",
        f"- Analysis family: `{payload['analysis_family'] or 'unresolved'}`",
        "- Dev9 package: "
        + ("complete" if payload["development_supplement_complete"] else "incomplete"),
        "- Top-journal extension: "
        + ("complete" if payload["top_journal_supplement_complete"] else "incomplete"),
        "",
        str(payload["claim_boundary"]),
    ]
    for section, value in section_payload.items():
        lines.extend(["", f"## {section}", "", str(value["claim_boundary"]), ""])
        bindings = value["artifact_bindings"]
        if not bindings:
            lines.append("Not produced in this run; no placeholder result was inserted.")
            continue
        lines.extend(
            [
                "| Evidence/product | Step | Path | SHA-256 verified |",
                "|---|---|---|---:|",
            ]
        )
        for item in bindings:
            identity = item["evidence_id"] or item["product"] or "unresolved"
            lines.append(
                f"| `{identity}` | `{item['step_id'] or ''}` | "
                f"`{item['path']}` | {'yes' if item['digest_verified'] else 'no'} |"
            )
    markdown_record = evidence.register_text(
        kind="log",
        description="Human-readable evidence-bound supplementary appendix.",
        text="\n".join(lines) + "\n",
        filename="supplement_package.md",
        evidence_id="supplement_package",
        inputs=[manifest_record.evidence_id],
        producer="supplement_package",
        generation_mode="system",
        on_sha_change="new_id",
    )
    for record, filename in (
        (manifest_record, "supplement_evidence_manifest.json"),
        (markdown_record, "supplement_package.md"),
    ):
        (run_dir / filename).write_bytes((run_dir / record.relative_path).read_bytes())
    return payload


__all__ = ["write_supplement_package"]
