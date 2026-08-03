#!/usr/bin/env python3
"""QC-A03: Render the two provenance-bound EasyICU submission QC figures.

The 19-module distribution atlas produced by QC-A01 remains an Extended Data
diagnostic layer.  This renderer turns the lightweight QC-A01/QC-A02 evidence
tables into two claim-led main figures without rescanning raw Parquet files.

No run is called current implicitly.  ``--source-status validated_current`` is
accepted only when the native-v2 schema, content, row-grain, null-time,
metadata, sidecar and runtime provenance gates all pass.  Candidate outputs are
visibly labelled as non-current.  Synthetic inputs are allowed only for layout
QA and receive an in-figure watermark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


# Nature-figure backend contract: Python-only, editable SVG/PDF text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


DATABASES = ("aumc", "eicu", "hirid", "mimic", "miiv", "sic")
DATABASE_LABELS = {
    "aumc": "AUMCdb",
    "eicu": "eICU-CRD",
    "hirid": "HiRID",
    "mimic": "MIMIC-III",
    "miiv": "MIMIC-IV",
    "sic": "SICdb",
}
PLOT_KIND_LABELS = {
    "continuous": "Continuous",
    "binary": "Binary",
    "ordinal": "Ordinal",
    "categorical": "Categorical",
    "unavailable": "Unavailable",
}
PLOT_KIND_ORDER = tuple(PLOT_KIND_LABELS)
SOURCE_STATUSES = ("candidate", "validated_current", "synthetic_layout_qa")
EXPECTED_MODULE_COUNT = 19
EXPECTED_EXPORT_COUNT = EXPECTED_MODULE_COUNT * len(DATABASES)
WIDTH_MM = 183
FIGURE_1_HEIGHT_MM = 158
FIGURE_2_HEIGHT_MM = 170
MM_PER_INCH = 25.4
DEFAULT_DPI = 600
HETEROGENEITY_MIN_RECORDS = 100
HETEROGENEITY_REVIEW_RATIO = 10.0


COLORS = {
    "ink": "#272727",
    "dark_grey": "#5C5C5C",
    "mid_grey": "#9A9A9A",
    "light_grey": "#E5E5E5",
    "paper_grey": "#F5F5F3",
    "blue": "#315F8C",
    "blue_mid": "#6F9BC3",
    "blue_light": "#DCEAF4",
    "teal": "#4E8A8B",
    "orange": "#C97835",
    "orange_light": "#F1D8C1",
    "violet": "#776A9C",
    "white": "#FFFFFF",
}


FIGURE_ROLES = {
    "QC-A01": {
        "role": "extended_data_diagnostic_atlas",
        "claim_boundary": (
            "Record-level distributions expose review signals; they do not "
            "establish database equivalence."
        ),
    },
    "QC-A02": {
        "role": "audit_evidence_layer",
        "claim_boundary": (
            "Schema, row-grain, null-time, provenance and source trace are "
            "eligibility evidence, not a standalone manuscript figure."
        ),
    },
    "QC-A03_Fig1": {
        "role": "main_qc_observational_support",
        "core_conclusion": (
            "One typed 19-module surface is retained across six databases "
            "while database-specific observability remains explicit."
        ),
    },
    "QC-A03_Fig2": {
        "role": "main_qc_harmonization_reliability",
        "core_conclusion": (
            "Physical compatibility and provenance gates must pass together, "
            "and residual distribution heterogeneity must remain traceable."
        ),
    },
}


@dataclass(frozen=True)
class SourceLineage:
    source_run_id: str
    source_run_metadata_sha256: str


@dataclass(frozen=True)
class AuditPaths:
    run_metadata: Path
    qc_a01_manifest: Path
    variable_audit: Path
    cohort_denominators: Path
    qc_a02_summary: Path
    field_contract: Path
    module_schema: Path
    manifest_audit: Path
    concept_availability: Path
    distribution_flags: Path
    verified_issues: Path


@dataclass
class AuditBundle:
    lineage: SourceLineage
    run_metadata: dict[str, Any]
    qc_a01_manifest: dict[str, Any]
    qc_a02_summary: dict[str, Any]
    variable_audit: pd.DataFrame
    cohort_denominators: pd.DataFrame
    field_contract: pd.DataFrame
    module_schema: pd.DataFrame
    manifest_audit: pd.DataFrame
    concept_availability: pd.DataFrame
    distribution_flags: pd.DataFrame
    verified_issues: pd.DataFrame
    input_hashes: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-a01-root",
        type=Path,
        required=True,
        help="QC-A01 publication_qc directory containing audit/ outputs.",
    )
    parser.add_argument(
        "--qc-a02-dir",
        type=Path,
        required=True,
        help="QC-A02 reliability_audit output directory.",
    )
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-status",
        choices=SOURCE_STATUSES,
        required=True,
        help=(
            "Explicit currentness declaration. validated_current is fail-closed; "
            "synthetic_layout_qa is permanently non-publication-eligible."
        ),
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--top-heterogeneity", type=int, default=10)
    return parser.parse_args()


def apply_publication_style() -> None:
    """Apply the exclusive Python Nature-family rendering contract."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required QC artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required QC artifact: {path}")
    return pd.read_csv(path)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _boolean_series(series: pd.Series, *, label: str) -> pd.Series:
    """Parse a required boolean column without truthy-string coercion."""

    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"{label} contains missing boolean values")
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    # This is schema validation, not analytical row exclusion: missing values
    # are rejected explicitly below and no row leaves the figure source.
    non_missing = normalized[normalized.notna()]
    unexpected = sorted(set(non_missing) - set(mapping))
    if unexpected or normalized.isna().any():
        detail = ", ".join(unexpected) if unexpected else "missing value"
        raise ValueError(f"{label} contains invalid booleans: {detail}")
    return normalized.map(mapping).astype(bool)


def _build_paths(
    *,
    qc_a01_root: Path,
    qc_a02_dir: Path,
    run_metadata: Path,
) -> AuditPaths:
    return AuditPaths(
        run_metadata=run_metadata,
        qc_a01_manifest=qc_a01_root / "audit" / "run_manifest.json",
        variable_audit=qc_a01_root / "audit" / "variable_audit.csv",
        cohort_denominators=qc_a01_root / "audit" / "cohort_denominators.csv",
        qc_a02_summary=qc_a02_dir / "audit_summary.json",
        field_contract=qc_a02_dir / "field_contract_audit.csv",
        module_schema=qc_a02_dir / "module_schema_summary.csv",
        manifest_audit=qc_a02_dir / "manifest_audit.csv",
        concept_availability=qc_a02_dir / "concept_availability.csv",
        distribution_flags=qc_a02_dir / "distribution_anomaly_flags.csv",
        verified_issues=qc_a02_dir / "verified_issue_register.csv",
    )


def _lineage_from_run_metadata(path: Path) -> tuple[SourceLineage, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source run metadata: {path}")
    raw = path.read_bytes()
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise TypeError("run_metadata.json must contain a JSON object")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_metadata.json must declare a non-empty run_id")
    return (
        SourceLineage(
            source_run_id=run_id.strip(),
            source_run_metadata_sha256=hashlib.sha256(raw).hexdigest(),
        ),
        metadata,
    )


def _validate_lineage(
    lineage: SourceLineage,
    *,
    qc_a01_manifest: Mapping[str, Any],
    qc_a02_summary: Mapping[str, Any],
) -> None:
    expected = {
        "source_run_id": lineage.source_run_id,
        "source_run_metadata_sha256": lineage.source_run_metadata_sha256,
    }
    mismatches: list[str] = []
    for artifact_label, artifact in (
        ("QC-A01", qc_a01_manifest),
        ("QC-A02", qc_a02_summary),
    ):
        for field, expected_value in expected.items():
            observed = artifact.get(field)
            if observed != expected_value:
                mismatches.append(
                    f"{artifact_label}.{field}={observed!r}, expected {expected_value!r}"
                )
    if mismatches:
        raise ValueError("QC source lineage mismatch: " + "; ".join(mismatches))


def load_audit_bundle(paths: AuditPaths) -> AuditBundle:
    lineage, run_metadata = _lineage_from_run_metadata(paths.run_metadata)
    qc_a01_manifest = _read_json(paths.qc_a01_manifest)
    qc_a02_summary = _read_json(paths.qc_a02_summary)
    _validate_lineage(
        lineage,
        qc_a01_manifest=qc_a01_manifest,
        qc_a02_summary=qc_a02_summary,
    )

    bundle = AuditBundle(
        lineage=lineage,
        run_metadata=run_metadata,
        qc_a01_manifest=qc_a01_manifest,
        qc_a02_summary=qc_a02_summary,
        variable_audit=_read_csv(paths.variable_audit),
        cohort_denominators=_read_csv(paths.cohort_denominators),
        field_contract=_read_csv(paths.field_contract),
        module_schema=_read_csv(paths.module_schema),
        manifest_audit=_read_csv(paths.manifest_audit),
        concept_availability=_read_csv(paths.concept_availability),
        distribution_flags=_read_csv(paths.distribution_flags),
        verified_issues=_read_csv(paths.verified_issues),
        input_hashes={
            field: _sha256(getattr(paths, field))
            for field in paths.__dataclass_fields__
        },
    )
    validate_audit_bundle(bundle)
    return bundle


def validate_audit_bundle(bundle: AuditBundle) -> None:
    _require_columns(
        bundle.variable_audit,
        (
            "module",
            "variable",
            "database",
            "plot_kind",
            "non_null_or_finite",
            "median_sample",
            "minimum",
            "catalog_min",
            "unit",
        ),
        label="QC-A01 variable audit",
    )
    _require_columns(
        bundle.cohort_denominators,
        ("database", "cohort_stays"),
        label="QC-A01 cohort denominators",
    )
    _require_columns(
        bundle.field_contract,
        ("module", "database", "concept", "available", "unit"),
        label="QC-A02 field contract",
    )
    _require_columns(
        bundle.module_schema,
        (
            "module",
            "all_six_parquets_present",
            "same_concept_set",
            "same_concept_order",
            "same_full_physical_schema",
            "type_mismatch_field_count",
            "canonical_stay_id_all_six",
            "canonical_charttime_all_six",
        ),
        label="QC-A02 module schema",
    )
    _require_columns(
        bundle.manifest_audit,
        (
            "module",
            "database",
            "availability",
            "saved_path_exists",
            "native_v2",
            "manifest_schema_matches_parquet",
            "parquet_sha256_matches",
            "parquet_bytes_matches",
            "row_grain_contract_valid",
            "null_time_concept_contract_valid",
            "concept_metadata_complete",
            "structural_placeholder_valid",
            "sidecar_sha256_matches",
            "runtime_commit_matches_run",
            "runtime_git_dirty",
        ),
        label="QC-A02 manifest audit",
    )
    _require_columns(
        bundle.concept_availability,
        ("module", "variable", "plot_kind", "databases_available"),
        label="QC-A02 concept availability",
    )

    module_order = list(dict.fromkeys(bundle.module_schema["module"].astype(str)))
    if len(module_order) != EXPECTED_MODULE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODULE_COUNT} module rows, found {len(module_order)}"
        )
    if bundle.module_schema["module"].astype(str).duplicated().any():
        raise ValueError("module_schema_summary.csv has duplicate module rows")

    for label, frame in (
        ("field contract", bundle.field_contract),
        ("variable audit", bundle.variable_audit),
        ("manifest audit", bundle.manifest_audit),
        ("cohort denominators", bundle.cohort_denominators),
    ):
        observed = set(frame["database"].astype(str))
        if observed != set(DATABASES):
            raise ValueError(
                f"{label} database set mismatch: {sorted(observed)}"
            )

    if bundle.manifest_audit.shape[0] != EXPECTED_EXPORT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EXPORT_COUNT} database-module manifests, "
            f"found {bundle.manifest_audit.shape[0]}"
        )
    if bundle.manifest_audit.duplicated(["module", "database"]).any():
        raise ValueError("manifest_audit.csv has duplicate database-module rows")
    if bundle.cohort_denominators.duplicated("database").any():
        raise ValueError("cohort_denominators.csv has duplicate databases")
    if bundle.field_contract.duplicated(["module", "database", "concept"]).any():
        raise ValueError("field_contract_audit.csv has duplicate concept contracts")
    if bundle.variable_audit.duplicated(["module", "variable", "database"]).any():
        raise ValueError("variable_audit.csv has duplicate database-concept rows")

    availability_depth = pd.to_numeric(
        bundle.concept_availability["databases_available"], errors="coerce"
    )
    if availability_depth.isna().any() or not availability_depth.between(0, 6).all():
        raise ValueError("databases_available must be an integer from 0 through 6")


def _annotate_lineage(
    frame: pd.DataFrame,
    *,
    lineage: SourceLineage,
    source_status: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["source_status"] = source_status
    result["source_run_id"] = lineage.source_run_id
    result["source_run_metadata_sha256"] = lineage.source_run_metadata_sha256
    return result


def build_module_support(bundle: AuditBundle) -> pd.DataFrame:
    fields = bundle.field_contract.copy()
    fields["available"] = _boolean_series(
        fields["available"], label="field_contract.available"
    )
    support = (
        fields.groupby(["module", "database"], sort=False)
        .agg(
            concepts_nonempty=("available", "sum"),
            concepts_declared=("concept", "size"),
        )
        .reset_index()
    )
    manifest = bundle.manifest_audit[
        ["module", "database", "availability", "actual_parquet_row_count"]
    ].copy()
    support = support.merge(
        manifest,
        on=["module", "database"],
        how="left",
        validate="one_to_one",
    )
    denominator = bundle.cohort_denominators[["database", "cohort_stays"]].copy()
    support = support.merge(
        denominator,
        on="database",
        how="left",
        validate="many_to_one",
    )
    support["concepts_nonempty"] = support["concepts_nonempty"].astype(int)
    support["concepts_declared"] = support["concepts_declared"].astype(int)
    if (support["concepts_declared"] <= 0).any():
        raise ValueError("Every database-module cell must declare at least one concept")
    support["support_fraction"] = (
        support["concepts_nonempty"] / support["concepts_declared"]
    )
    support["structurally_unavailable"] = support["availability"].eq(
        "structurally_unavailable"
    )
    invalid_structural = support[
        support["structurally_unavailable"] & support["concepts_nonempty"].gt(0)
    ]
    if not invalid_structural.empty:
        raise ValueError(
            "A structurally unavailable module contains non-empty concepts: "
            + ", ".join(
                f"{row.module}/{row.database}"
                for row in invalid_structural.itertuples(index=False)
            )
        )
    support["availability_state"] = np.select(
        [
            support["structurally_unavailable"],
            support["concepts_nonempty"].eq(0),
        ],
        ["structurally_unavailable", "declared_but_empty"],
        default="observed",
    )
    support["cell_label"] = (
        support["concepts_nonempty"].astype(str)
        + "/"
        + support["concepts_declared"].astype(str)
    )
    support["database_label"] = support["database"].map(DATABASE_LABELS)
    return support


def build_support_depth(bundle: AuditBundle) -> pd.DataFrame:
    depth = bundle.concept_availability.copy()
    depth["databases_available"] = pd.to_numeric(
        depth["databases_available"], errors="raise"
    ).astype(int)
    depth["coverage_class"] = np.select(
        [
            depth["databases_available"].eq(6),
            depth["databases_available"].eq(0),
        ],
        ["all_six", "none"],
        default="one_to_five",
    )
    depth["plot_kind_label"] = depth["plot_kind"].map(PLOT_KIND_LABELS).fillna(
        depth["plot_kind"].astype(str)
    )
    return depth


def _canonical_unit_pass_by_module(fields: pd.DataFrame) -> dict[str, bool]:
    normalized = fields.copy()
    normalized["unit_normalized"] = (
        normalized["unit"].astype("string").fillna("<missing>").str.strip()
    )
    per_concept = (
        normalized.groupby(["module", "concept"], sort=False)["unit_normalized"]
        .nunique(dropna=False)
        .reset_index(name="unit_count")
    )
    return (
        per_concept.groupby("module", sort=False)["unit_count"]
        .apply(lambda values: bool(values.eq(1).all()))
        .to_dict()
    )


def build_contract_matrix(bundle: AuditBundle) -> pd.DataFrame:
    modules = bundle.module_schema.copy()
    unit_pass = _canonical_unit_pass_by_module(bundle.field_contract)
    contract_specs: tuple[tuple[str, str], ...] = (
        ("same_concept_set", "Concept set"),
        ("same_concept_order", "Column order"),
        ("dtype_aligned", "Arrow dtype"),
        ("canonical_unit_contract", "Canonical unit"),
        ("same_full_physical_schema", "Full schema"),
        ("canonical_stay_id_all_six", "stay_id"),
        ("canonical_charttime_all_six", "charttime"),
        ("all_six_parquets_present", "Six files"),
    )
    modules["dtype_aligned"] = pd.to_numeric(
        modules["type_mismatch_field_count"], errors="raise"
    ).eq(0)
    modules["canonical_unit_contract"] = modules["module"].map(unit_pass)
    rows: list[dict[str, Any]] = []
    for module_position, record in enumerate(modules.to_dict(orient="records")):
        for contract_position, (field, label) in enumerate(contract_specs):
            value = _boolean_series(
                pd.Series([record[field]]), label=f"module_schema.{field}"
            ).iloc[0]
            rows.append(
                {
                    "module": str(record["module"]),
                    "module_position": module_position,
                    "contract": field,
                    "contract_label": label,
                    "contract_position": contract_position,
                    "passed": bool(value),
                    "denominator_modules": EXPECTED_MODULE_COUNT,
                }
            )
    return pd.DataFrame(rows)


def build_release_gates(bundle: AuditBundle) -> pd.DataFrame:
    manifests = bundle.manifest_audit.copy()
    metadata_closed = _boolean_series(
        manifests["concept_metadata_complete"],
        label="manifest.concept_metadata_complete",
    ) | _boolean_series(
        manifests["structural_placeholder_valid"],
        label="manifest.structural_placeholder_valid",
    )
    runtime_clean = _boolean_series(
        manifests["runtime_commit_matches_run"],
        label="manifest.runtime_commit_matches_run",
    ) & ~_boolean_series(
        manifests["runtime_git_dirty"],
        label="manifest.runtime_git_dirty",
    )
    gate_series: tuple[tuple[str, str, pd.Series], ...] = (
        (
            "parquet_present",
            "Parquet present",
            _boolean_series(
                manifests["saved_path_exists"], label="manifest.saved_path_exists"
            ),
        ),
        (
            "native_v2",
            "native-v2",
            _boolean_series(manifests["native_v2"], label="manifest.native_v2"),
        ),
        (
            "manifest_schema",
            "Manifest schema",
            _boolean_series(
                manifests["manifest_schema_matches_parquet"],
                label="manifest.manifest_schema_matches_parquet",
            ),
        ),
        (
            "content_sha256",
            "Content SHA-256",
            _boolean_series(
                manifests["parquet_sha256_matches"],
                label="manifest.parquet_sha256_matches",
            ),
        ),
        (
            "byte_receipt",
            "Byte receipt",
            _boolean_series(
                manifests["parquet_bytes_matches"],
                label="manifest.parquet_bytes_matches",
            ),
        ),
        (
            "row_grain",
            "Row grain",
            _boolean_series(
                manifests["row_grain_contract_valid"],
                label="manifest.row_grain_contract_valid",
            ),
        ),
        (
            "null_time",
            "Null-time semantics",
            _boolean_series(
                manifests["null_time_concept_contract_valid"],
                label="manifest.null_time_concept_contract_valid",
            ),
        ),
        ("metadata", "Metadata closure", metadata_closed),
        (
            "sidecar_sha256",
            "Sidecar SHA-256",
            _boolean_series(
                manifests["sidecar_sha256_matches"],
                label="manifest.sidecar_sha256_matches",
            ),
        ),
        ("runtime", "Pinned clean runtime", runtime_clean),
    )
    rows = []
    for gate_position, (gate, label, values) in enumerate(gate_series):
        denominator = int(values.shape[0])
        passed = int(values.sum())
        rows.append(
            {
                "gate": gate,
                "gate_label": label,
                "gate_position": gate_position,
                "passed": passed,
                "denominator": denominator,
                "pass_fraction": passed / denominator if denominator else math.nan,
                "failed": denominator - passed,
            }
        )
    return pd.DataFrame(rows)


def _distribution_flag_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "module",
        "variable",
        "database",
        "flag",
        "severity",
        "adjudication_status",
        "adjudicated_origin",
    )
    if frame.empty:
        return pd.DataFrame(columns=required)
    _require_columns(frame, required, label="distribution anomaly flags")
    return frame.copy()


def build_heterogeneity_table(
    bundle: AuditBundle,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Build a complete inclusion audit and rank descriptive median ratios."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    audit = bundle.variable_audit.copy()
    numeric_columns = (
        "non_null_or_finite",
        "median_sample",
        "minimum",
        "catalog_min",
    )
    for column in numeric_columns:
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    continuous = audit[audit["plot_kind"].eq("continuous")].copy()
    flags = _distribution_flag_columns(bundle.distribution_flags)

    rows: list[dict[str, Any]] = []
    for (module, variable), group in continuous.groupby(
        ["module", "variable"], sort=False
    ):
        declared_signed = bool(group["catalog_min"].lt(0).fillna(False).any())
        observed_signed = bool(group["minimum"].lt(0).fillna(False).any())
        eligible = group[
            group["non_null_or_finite"].ge(HETEROGENEITY_MIN_RECORDS)
            & group["median_sample"].notna()
        ].copy()
        exclusion_reason = "included"
        if declared_signed or observed_signed:
            exclusion_reason = "signed_scale_ratio_not_meaningful"
        elif eligible.shape[0] < 2:
            exclusion_reason = "fewer_than_two_databases_with_n_ge_100"
        elif eligible["median_sample"].le(0).any():
            exclusion_reason = "nonpositive_median_ratio_not_defined"

        row: dict[str, Any] = {
            "module": str(module),
            "variable": str(variable),
            "unit": _single_display_value(group["unit"]),
            "continuous_database_rows": int(group.shape[0]),
            "eligible_database_count": int(eligible.shape[0]),
            "declared_signed": declared_signed,
            "observed_signed": observed_signed,
            "exclusion_reason": exclusion_reason,
            "included": exclusion_reason == "included",
            "displayed": False,
            "display_rank": pd.NA,
            "display_rule": (
                f"Top {top_n} max/min positive-median ratios among concepts with "
                f">={HETEROGENEITY_MIN_RECORDS} records in at least two databases; "
                "signed and non-positive scales excluded."
            ),
            "interval_definition": (
                "none; descriptive repeated-record diagnostic, not an independent-patient effect"
            ),
        }
        if exclusion_reason != "included":
            rows.append(row)
            continue

        lowest = eligible.loc[eligible["median_sample"].idxmin()]
        highest = eligible.loc[eligible["median_sample"].idxmax()]
        ratio = float(highest["median_sample"]) / float(lowest["median_sample"])
        pair = f"{lowest['database']} vs {highest['database']}"
        matching_flags = flags[
            flags["module"].astype(str).eq(str(module))
            & flags["variable"].astype(str).eq(str(variable))
            & flags["flag"].astype(str).eq("median_scale_shift")
            & flags["database"].astype(str).eq(pair)
        ]
        if matching_flags.shape[0] > 1:
            raise ValueError(
                f"Multiple median-scale flags match {module}/{variable}/{pair}"
            )
        if matching_flags.empty:
            adjudication_status = (
                "below_review_trigger"
                if ratio < HETEROGENEITY_REVIEW_RATIO
                else "unadjudicated_detector_gap"
            )
            adjudicated_origin = pd.NA
            flag_severity = pd.NA
        else:
            flag_row = matching_flags.iloc[0]
            adjudication_status = str(flag_row["adjudication_status"])
            adjudicated_origin = flag_row["adjudicated_origin"]
            flag_severity = flag_row["severity"]
        row.update(
            {
                "low_database": str(lowest["database"]),
                "high_database": str(highest["database"]),
                "low_median": float(lowest["median_sample"]),
                "high_median": float(highest["median_sample"]),
                "max_min_median_ratio": ratio,
                "log10_max_min_ratio": math.log10(ratio),
                "low_database_records": int(lowest["non_null_or_finite"]),
                "high_database_records": int(highest["non_null_or_finite"]),
                "eligible_records_total": int(
                    eligible["non_null_or_finite"].sum()
                ),
                "database_pair": pair,
                "adjudication_status": adjudication_status,
                "adjudicated_origin": adjudicated_origin,
                "flag_severity": flag_severity,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    included_index = result.index[result["included"].fillna(False)].tolist()
    ranked = sorted(
        included_index,
        key=lambda index: float(result.at[index, "max_min_median_ratio"]),
        reverse=True,
    )
    for rank, index in enumerate(ranked, start=1):
        result.at[index, "display_rank"] = rank
        result.at[index, "displayed"] = rank <= top_n
    return result


def _single_display_value(values: pd.Series) -> str:
    normalized = [
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    ]
    unique = list(dict.fromkeys(normalized))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return " | ".join(unique)


def build_anomaly_trace(bundle: AuditBundle) -> pd.DataFrame:
    flags = _distribution_flag_columns(bundle.distribution_flags)
    if flags.empty:
        return pd.DataFrame(
            columns=(
                "flag",
                "flag_label",
                "trace_status",
                "count",
                "total_for_flag",
            )
        )
    flags["trace_status"] = np.where(
        flags["adjudication_status"].astype(str).eq("source_trace_complete"),
        "source_trace_complete",
        "unadjudicated",
    )
    flag_labels = {
        "median_scale_shift": "Median scale",
        "signed_median_direction_shift": "Signed location",
        "below_catalog_range": "Below range",
        "above_catalog_range": "Above range",
    }
    grouped = (
        flags.groupby(["flag", "trace_status"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = grouped.groupby("flag")["count"].sum().rename("total_for_flag")
    grouped = grouped.merge(totals, on="flag", validate="many_to_one")
    grouped["flag_label"] = grouped["flag"].map(flag_labels).fillna(
        grouped["flag"].astype(str).str.replace("_", " ").str.title()
    )
    return grouped


def publication_gate_errors(
    bundle: AuditBundle,
    *,
    contract_matrix: pd.DataFrame,
    release_gates: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if bundle.qc_a01_manifest.get("status") != "passed":
        errors.append("QC-A01 status is not passed")
    if int(bundle.qc_a01_manifest.get("failure_count", -1)) != 0:
        errors.append("QC-A01 reports scan/render failures")
    if int(bundle.qc_a01_manifest.get("module_count", -1)) != EXPECTED_MODULE_COUNT:
        errors.append("QC-A01 module denominator is not 19")
    if not contract_matrix["passed"].all():
        failed = contract_matrix.loc[~contract_matrix["passed"]]
        errors.append(f"{failed.shape[0]} module-level physical contract cells fail")
    failed_gates = release_gates[release_gates["failed"].gt(0)]
    if not failed_gates.empty:
        errors.append(
            "release gates fail: "
            + ", ".join(
                f"{row.gate} {row.passed}/{row.denominator}"
                for row in failed_gates.itertuples(index=False)
            )
        )
    flags = _distribution_flag_columns(bundle.distribution_flags)
    if not flags.empty:
        high_untraced = flags[
            flags["severity"].astype(str).eq("high")
            & ~flags["adjudication_status"].astype(str).eq("source_trace_complete")
        ]
        if not high_untraced.empty:
            errors.append(
                f"{high_untraced.shape[0]} high-severity distribution flags lack source trace"
            )
    return errors


def _panel_label(ax: mpl.axes.Axes, label: str, *, x: float = -0.08) -> None:
    ax.text(
        x,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def _display_module(module: str) -> str:
    return textwrap.fill(module.replace("_", " "), width=19)


def _status_banner(
    fig: mpl.figure.Figure,
    *,
    source_status: str,
    lineage: SourceLineage,
) -> None:
    if source_status == "validated_current":
        label = "VALIDATED CURRENT"
        color = COLORS["blue"]
    elif source_status == "candidate":
        label = "CANDIDATE · NOT CURRENT"
        color = COLORS["orange"]
    else:
        label = "SYNTHETIC LAYOUT QA · NOT DATA"
        color = COLORS["violet"]
    fig.text(
        0.995,
        0.995,
        f"{label}  |  run {lineage.source_run_id}",
        ha="right",
        va="top",
        fontsize=5.2,
        color=color,
        fontweight="bold",
    )
    if source_status == "synthetic_layout_qa":
        fig.text(
            0.52,
            0.50,
            "SYNTHETIC LAYOUT QA — NOT DATA",
            ha="center",
            va="center",
            rotation=28,
            fontsize=22,
            color=COLORS["violet"],
            alpha=0.13,
            fontweight="bold",
            zorder=100,
        )


def _support_matrix_arrays(
    support: pd.DataFrame,
    module_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.full((len(module_order), len(DATABASES)), np.nan, dtype=float)
    labels = np.empty((len(module_order), len(DATABASES)), dtype=object)
    states = np.empty((len(module_order), len(DATABASES)), dtype=object)
    row_counts = np.full((len(module_order), len(DATABASES)), np.nan, dtype=float)
    for i, module in enumerate(module_order):
        for j, database in enumerate(DATABASES):
            match = support[
                support["module"].astype(str).eq(module)
                & support["database"].astype(str).eq(database)
            ]
            if match.shape[0] != 1:
                raise ValueError(f"Expected one support cell for {module}/{database}")
            row = match.iloc[0]
            fractions[i, j] = float(row["support_fraction"])
            labels[i, j] = str(row["cell_label"])
            states[i, j] = str(row["availability_state"])
            row_counts[i, j] = float(row["actual_parquet_row_count"])
    return fractions, labels, states, row_counts


def render_figure_1(
    *,
    support: pd.DataFrame,
    depth: pd.DataFrame,
    cohort_denominators: pd.DataFrame,
    module_order: list[str],
    lineage: SourceLineage,
    source_status: str,
) -> mpl.figure.Figure:
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, FIGURE_1_HEIGHT_MM / MM_PER_INCH)
    )
    grid = fig.add_gridspec(
        2,
        5,
        height_ratios=[3.25, 1.15],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.22],
        hspace=0.38,
        wspace=0.62,
        left=0.155,
        right=0.985,
        top=0.945,
        bottom=0.105,
    )
    ax_support = fig.add_subplot(grid[0, :4])
    ax_cohort = fig.add_subplot(grid[0, 4])
    ax_depth = fig.add_subplot(grid[1, :])

    fractions, labels, states, _ = _support_matrix_arrays(support, module_order)
    support_cmap = LinearSegmentedColormap.from_list(
        "easyicu_support", [COLORS["paper_grey"], COLORS["blue_light"], COLORS["blue"]]
    )
    image = ax_support.imshow(
        fractions,
        cmap=support_cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for i in range(fractions.shape[0]):
        for j in range(fractions.shape[1]):
            state = states[i, j]
            fraction = fractions[i, j]
            if state == "structurally_unavailable":
                ax_support.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=COLORS["light_grey"],
                        edgecolor=COLORS["mid_grey"],
                        hatch="////",
                        linewidth=0.4,
                        zorder=2,
                    )
                )
                text_color = COLORS["dark_grey"]
            elif state == "declared_but_empty":
                ax_support.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=COLORS["orange"],
                        linewidth=1.05,
                        zorder=3,
                    )
                )
                text_color = COLORS["orange"]
            else:
                text_color = COLORS["white"] if fraction >= 0.63 else COLORS["ink"]
            ax_support.text(
                j,
                i,
                labels[i, j],
                ha="center",
                va="center",
                fontsize=5.0,
                color=text_color,
                fontweight="bold" if fraction >= 0.90 else "normal",
                zorder=4,
            )
    ax_support.set_xticks(np.arange(len(DATABASES)))
    ax_support.set_xticklabels([DATABASE_LABELS[db] for db in DATABASES])
    ax_support.xaxis.tick_top()
    ax_support.tick_params(axis="x", length=0, pad=3)
    ax_support.set_yticks(np.arange(len(module_order)))
    ax_support.set_yticklabels([_display_module(module) for module in module_order])
    ax_support.tick_params(axis="y", length=0, pad=4)
    ax_support.set_title(
        "Module observability: non-empty concepts / declared concepts",
        loc="left",
        pad=21,
    )
    for spine in ax_support.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(
        image,
        ax=ax_support,
        orientation="horizontal",
        fraction=0.028,
        pad=0.035,
        aspect=35,
    )
    colorbar.set_ticks([0, 0.5, 1.0])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    colorbar.set_label("Share of declared concepts with ≥1 finite/non-null value", fontsize=5.5)
    colorbar.ax.tick_params(labelsize=5, length=2)
    structural_patch = Patch(
        facecolor=COLORS["light_grey"],
        edgecolor=COLORS["mid_grey"],
        hatch="////",
        label="Structurally unavailable",
    )
    empty_patch = Patch(
        facecolor="none",
        edgecolor=COLORS["orange"],
        label="Declared but empty",
    )
    ax_support.legend(
        handles=[structural_patch, empty_patch],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.105),
        ncol=2,
        handlelength=1.4,
        columnspacing=1.3,
    )
    _panel_label(ax_support, "a", x=-0.115)

    cohort = cohort_denominators.copy()
    cohort["database"] = pd.Categorical(
        cohort["database"], categories=DATABASES, ordered=True
    )
    cohort = cohort.sort_values("database")
    cohort["cohort_stays"] = pd.to_numeric(cohort["cohort_stays"], errors="raise")
    y = np.arange(cohort.shape[0])[::-1]
    values = cohort["cohort_stays"].to_numpy(dtype=float)
    baseline = max(1.0, float(values.min()) / 2.2)
    ax_cohort.hlines(y, baseline, values, color=COLORS["blue_light"], lw=2.5)
    ax_cohort.scatter(
        values,
        y,
        s=24,
        color=COLORS["blue"],
        edgecolor=COLORS["white"],
        linewidth=0.6,
        zorder=3,
    )
    for yi, value in zip(y, values):
        ax_cohort.annotate(
            f"n={int(value):,}",
            (value, yi),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.3,
            color=COLORS["ink"],
        )
    ax_cohort.set_xscale("log")
    ax_cohort.set_xlim(baseline, values.max() * 3.7)
    ax_cohort.set_yticks(y)
    ax_cohort.set_yticklabels(
        [DATABASE_LABELS[str(db)] for db in cohort["database"].astype(str)]
    )
    ax_cohort.set_xlabel("Cohort stays (log scale)")
    ax_cohort.set_title("Cohort denominator", loc="left", pad=8)
    ax_cohort.grid(axis="x", color=COLORS["light_grey"], linewidth=0.45)
    ax_cohort.tick_params(axis="y", length=0)
    _panel_label(ax_cohort, "b", x=-0.20)

    depth_summary = (
        depth.groupby(["plot_kind", "plot_kind_label", "coverage_class"], sort=False)
        .size()
        .rename("concept_count")
        .reset_index()
    )
    present_kinds = [kind for kind in PLOT_KIND_ORDER if kind in set(depth["plot_kind"])]
    y_kind = np.arange(len(present_kinds))[::-1]
    left = np.zeros(len(present_kinds), dtype=float)
    coverage_order = ("none", "one_to_five", "all_six")
    coverage_labels = {
        "none": "0 databases",
        "one_to_five": "1–5 databases",
        "all_six": "All 6 databases",
    }
    coverage_colors = {
        "none": COLORS["orange_light"],
        "one_to_five": COLORS["blue_light"],
        "all_six": COLORS["blue"],
    }
    for coverage_class in coverage_order:
        counts = np.array(
            [
                int(
                    depth_summary.loc[
                        depth_summary["plot_kind"].eq(kind)
                        & depth_summary["coverage_class"].eq(coverage_class),
                        "concept_count",
                    ].sum()
                )
                for kind in present_kinds
            ],
            dtype=float,
        )
        bars = ax_depth.barh(
            y_kind,
            counts,
            left=left,
            height=0.58,
            color=coverage_colors[coverage_class],
            edgecolor=COLORS["white"],
            linewidth=0.5,
            label=coverage_labels[coverage_class],
        )
        for bar, count in zip(bars, counts):
            if count > 0:
                ax_depth.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=(
                        COLORS["white"]
                        if coverage_class == "all_six"
                        else COLORS["ink"]
                    ),
                )
        left += counts
    ax_depth.set_yticks(y_kind)
    ax_depth.set_yticklabels([PLOT_KIND_LABELS[kind] for kind in present_kinds])
    ax_depth.set_xlabel("Configured concepts")
    ax_depth.set_title(
        "Support depth by variable type (exact 0–6 depth retained in Source Data)",
        loc="left",
        pad=7,
    )
    ax_depth.tick_params(axis="y", length=0)
    ax_depth.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.24),
        ncol=3,
        columnspacing=1.2,
        handlelength=1.5,
    )
    _panel_label(ax_depth, "c", x=-0.045)

    fig.text(
        0.01,
        0.018,
        (
            f"Denominators: 19 modules × 6 databases; {depth.shape[0]:,} configured "
            "concepts. Availability means ≥1 finite/non-null extracted value; it is "
            "not evidence of clinical equivalence."
        ),
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=COLORS["dark_grey"],
    )
    _status_banner(fig, source_status=source_status, lineage=lineage)
    return fig


def _contract_matrix_arrays(
    contract: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    modules = list(
        contract.sort_values("module_position")["module"].astype(str).drop_duplicates()
    )
    labels = list(
        contract.sort_values("contract_position")["contract_label"]
        .astype(str)
        .drop_duplicates()
    )
    matrix = np.zeros((len(modules), len(labels)), dtype=int)
    for i, module in enumerate(modules):
        for j, label in enumerate(labels):
            match = contract[
                contract["module"].astype(str).eq(module)
                & contract["contract_label"].astype(str).eq(label)
            ]
            if match.shape[0] != 1:
                raise ValueError(f"Expected one contract cell for {module}/{label}")
            matrix[i, j] = int(bool(match.iloc[0]["passed"]))
    return matrix, modules, labels


def _heterogeneity_status_color(status: str) -> str:
    if status == "source_trace_complete":
        return COLORS["blue"]
    if status in {"unadjudicated", "unadjudicated_detector_gap"}:
        return COLORS["orange"]
    return COLORS["mid_grey"]


def render_figure_2(
    *,
    contract: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    gates: pd.DataFrame,
    anomaly_trace: pd.DataFrame,
    lineage: SourceLineage,
    source_status: str,
) -> mpl.figure.Figure:
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, FIGURE_2_HEIGHT_MM / MM_PER_INCH)
    )
    grid = fig.add_gridspec(
        4,
        7,
        height_ratios=[2.25, 2.25, 1.35, 1.25],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.95, 0.95, 0.95],
        hspace=0.68,
        wspace=0.95,
        left=0.145,
        right=0.985,
        top=0.942,
        bottom=0.095,
    )
    ax_contract = fig.add_subplot(grid[:, :4])
    ax_heterogeneity = fig.add_subplot(grid[:2, 4:])
    ax_gates = fig.add_subplot(grid[2, 4:])
    ax_trace = fig.add_subplot(grid[3, 4:])

    matrix, modules, contract_labels = _contract_matrix_arrays(contract)
    contract_cmap = ListedColormap([COLORS["orange_light"], COLORS["blue"]])
    ax_contract.imshow(
        matrix,
        cmap=contract_cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            passed = bool(matrix[i, j])
            ax_contract.text(
                j,
                i,
                "✓" if passed else "×",
                ha="center",
                va="center",
                fontsize=6.2,
                color=COLORS["white"] if passed else COLORS["orange"],
                fontweight="bold",
            )
    ax_contract.set_xticks(np.arange(len(contract_labels)))
    ax_contract.set_xticklabels(
        [textwrap.fill(label, width=11) for label in contract_labels],
        rotation=38,
        ha="left",
        rotation_mode="anchor",
    )
    ax_contract.xaxis.tick_top()
    ax_contract.tick_params(axis="x", length=0, pad=5)
    ax_contract.set_yticks(np.arange(len(modules)))
    ax_contract.set_yticklabels([_display_module(module) for module in modules])
    ax_contract.tick_params(axis="y", length=0, pad=4)
    ax_contract.set_title(
        "Per-module harmonization contract",
        loc="left",
        pad=31,
    )
    for spine in ax_contract.spines.values():
        spine.set_visible(False)
    ax_contract.legend(
        handles=[
            Patch(facecolor=COLORS["blue"], label="Pass"),
            Patch(
                facecolor=COLORS["orange_light"],
                edgecolor=COLORS["orange"],
                label="Fail",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.028),
        ncol=2,
        handlelength=1.3,
    )
    _panel_label(ax_contract, "a", x=-0.115)

    displayed = heterogeneity[heterogeneity["displayed"].fillna(False)].copy()
    displayed = displayed.sort_values("display_rank", ascending=False)
    if displayed.empty:
        ax_heterogeneity.text(
            0.5,
            0.5,
            "No eligible positive-median\nheterogeneity summaries",
            transform=ax_heterogeneity.transAxes,
            ha="center",
            va="center",
            color=COLORS["dark_grey"],
            fontsize=7,
        )
        ax_heterogeneity.set_axis_off()
    else:
        y = np.arange(displayed.shape[0])
        ratios = displayed["max_min_median_ratio"].to_numpy(dtype=float)
        statuses = displayed["adjudication_status"].fillna("below_review_trigger")
        point_colors = [_heterogeneity_status_color(str(value)) for value in statuses]
        ax_heterogeneity.hlines(
            y,
            1.0,
            ratios,
            color=COLORS["light_grey"],
            linewidth=1.6,
            zorder=1,
        )
        ax_heterogeneity.scatter(
            ratios,
            y,
            s=24,
            c=point_colors,
            edgecolor=COLORS["white"],
            linewidth=0.55,
            zorder=3,
        )
        for yi, row in zip(y, displayed.itertuples(index=False)):
            ax_heterogeneity.annotate(
                f"×{row.max_min_median_ratio:.1f}",
                (row.max_min_median_ratio, yi),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=5.0,
                color=COLORS["ink"],
            )
        labels = [
            textwrap.fill(
                f"{row.module} · {row.variable}  (d={row.eligible_database_count})",
                width=25,
            )
            for row in displayed.itertuples(index=False)
        ]
        ax_heterogeneity.set_yticks(y)
        ax_heterogeneity.set_yticklabels(labels)
        ax_heterogeneity.set_xscale("log")
        maximum = float(np.nanmax(ratios))
        ax_heterogeneity.set_xlim(0.9, max(12.5, maximum * 1.9))
        ax_heterogeneity.axvline(
            HETEROGENEITY_REVIEW_RATIO,
            color=COLORS["orange"],
            linestyle="--",
            linewidth=0.8,
        )
        ax_heterogeneity.text(
            HETEROGENEITY_REVIEW_RATIO,
            displayed.shape[0] - 0.2,
            "10× review trigger",
            rotation=90,
            va="top",
            ha="right",
            fontsize=5.0,
            color=COLORS["orange"],
        )
        ax_heterogeneity.set_xlabel("Cross-database median ratio (max/min; log scale)")
        ax_heterogeneity.tick_params(axis="y", length=0, pad=3)
        ax_heterogeneity.grid(axis="x", color=COLORS["light_grey"], linewidth=0.45)
        ax_heterogeneity.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=COLORS["blue"],
                    markeredgecolor="none",
                    label="Source trace complete",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=COLORS["orange"],
                    markeredgecolor="none",
                    label="Trace required",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=COLORS["mid_grey"],
                    markeredgecolor="none",
                    label="Below trigger",
                ),
            ],
            loc="lower right",
            fontsize=5,
            handletextpad=0.25,
        )
    ax_heterogeneity.set_title(
        "Largest descriptive cross-database location shifts",
        loc="left",
        pad=7,
    )
    _panel_label(ax_heterogeneity, "b", x=-0.16)

    ordered_gates = gates.sort_values("gate_position", ascending=False)
    y_gate = np.arange(ordered_gates.shape[0])
    percentages = ordered_gates["pass_fraction"].to_numpy(dtype=float) * 100
    gate_colors = [
        COLORS["blue"] if row.failed == 0 else COLORS["orange"]
        for row in ordered_gates.itertuples(index=False)
    ]
    ax_gates.hlines(
        y_gate,
        0,
        percentages,
        color=COLORS["light_grey"],
        linewidth=1.6,
    )
    ax_gates.scatter(
        percentages,
        y_gate,
        s=18,
        c=gate_colors,
        edgecolor=COLORS["white"],
        linewidth=0.45,
        zorder=3,
    )
    for yi, row in zip(y_gate, ordered_gates.itertuples(index=False)):
        ax_gates.text(
            min(109.0, row.pass_fraction * 100 + 2.0),
            yi,
            f"{row.passed}/{row.denominator}",
            ha="left",
            va="center",
            fontsize=5.0,
            color=COLORS["ink"],
        )
    ax_gates.set_yticks(y_gate)
    ax_gates.set_yticklabels(ordered_gates["gate_label"])
    ax_gates.set_xlim(0, 114)
    ax_gates.set_xticks([0, 50, 100])
    ax_gates.set_xticklabels(["0%", "50%", "100%"])
    ax_gates.axvline(100, color=COLORS["blue"], linewidth=0.65, alpha=0.7)
    ax_gates.tick_params(axis="y", length=0, pad=3)
    ax_gates.set_title(
        f"Release gates across {EXPECTED_EXPORT_COUNT} database × module files",
        loc="left",
        pad=6,
    )
    _panel_label(ax_gates, "c", x=-0.16)

    if anomaly_trace.empty:
        ax_trace.text(
            0.5,
            0.5,
            "No distribution review triggers (n=0)",
            transform=ax_trace.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            color=COLORS["dark_grey"],
        )
        ax_trace.set_axis_off()
    else:
        flag_order = list(
            anomaly_trace.groupby(["flag", "flag_label"], sort=False)["total_for_flag"]
            .max()
            .sort_values(ascending=True)
            .index
        )
        y_flag = np.arange(len(flag_order))
        left = np.zeros(len(flag_order), dtype=float)
        for trace_status, color, label in (
            ("source_trace_complete", COLORS["blue"], "Source trace complete"),
            ("unadjudicated", COLORS["orange_light"], "Trace required"),
        ):
            counts = np.array(
                [
                    int(
                        anomaly_trace.loc[
                            anomaly_trace["flag"].eq(flag)
                            & anomaly_trace["trace_status"].eq(trace_status),
                            "count",
                        ].sum()
                    )
                    for flag, _ in flag_order
                ],
                dtype=float,
            )
            bars = ax_trace.barh(
                y_flag,
                counts,
                left=left,
                height=0.55,
                color=color,
                edgecolor=COLORS["white"],
                linewidth=0.45,
                label=label,
            )
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax_trace.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(count)),
                        ha="center",
                        va="center",
                        fontsize=5.0,
                        color=(
                            COLORS["white"]
                            if trace_status == "source_trace_complete"
                            else COLORS["ink"]
                        ),
                    )
            left += counts
        ax_trace.set_yticks(y_flag)
        ax_trace.set_yticklabels([label for _, label in flag_order])
        ax_trace.set_xlabel("Review-trigger flags")
        ax_trace.tick_params(axis="y", length=0, pad=3)
        ax_trace.legend(
            loc="lower right",
            ncol=2,
            fontsize=5.0,
            handlelength=1.2,
            columnspacing=0.8,
        )
    ax_trace.set_title(
        "Distribution flags by traceback status",
        loc="left",
        pad=5,
    )
    _panel_label(ax_trace, "d", x=-0.16)

    included_count = int(heterogeneity["included"].fillna(False).sum())
    displayed_count = int(heterogeneity["displayed"].fillna(False).sum())
    fig.text(
        0.01,
        0.016,
        (
            f"Contract denominator: 19 modules; release-gate denominator: {EXPECTED_EXPORT_COUNT} "
            f"files. Heterogeneity panel displays {displayed_count}/{included_count} eligible "
            "continuous concepts by a prespecified top-ratio rule. Ratios are descriptive; "
            "no patient-level CI or P value is implied. Unadjudicated ≠ conversion defect."
        ),
        ha="left",
        va="bottom",
        fontsize=5.1,
        color=COLORS["dark_grey"],
    )
    _status_banner(fig, source_status=source_status, lineage=lineage)
    return fig


def _save_figure_bundle(
    fig: mpl.figure.Figure,
    *,
    base: Path,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        base.with_suffix(".svg"),
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
        base.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], format="svg")
    fig.savefig(outputs[1], format="pdf")
    fig.savefig(outputs[2], format="png", dpi=dpi)
    fig.savefig(
        outputs[3],
        format="tiff",
        dpi=dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return outputs


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    lineage: SourceLineage,
    source_status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _annotate_lineage(
        frame,
        lineage=lineage,
        source_status=source_status,
    ).to_csv(path, index=False)


def _write_legends(
    path: Path,
    *,
    support: pd.DataFrame,
    depth: pd.DataFrame,
    contract: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    gates: pd.DataFrame,
    anomaly_trace: pd.DataFrame,
    lineage: SourceLineage,
) -> None:
    all_six = int(depth["databases_available"].eq(6).sum())
    partial = int(depth["databases_available"].between(1, 5).sum())
    none = int(depth["databases_available"].eq(0).sum())
    contract_pass = int(contract["passed"].sum())
    contract_total = int(contract.shape[0])
    displayed = int(heterogeneity["displayed"].fillna(False).sum())
    eligible = int(heterogeneity["included"].fillna(False).sum())
    anomaly_total = int(anomaly_trace.groupby("flag")["total_for_flag"].max().sum()) if not anomaly_trace.empty else 0
    text = f"""# QC-A03 figure legends

**Fig. QC1 | Cross-database observational support of the EasyICU module surface**
**a,** Fraction and exact count of non-empty concepts in each of {EXPECTED_MODULE_COUNT} modules across six databases (`k/n`, where `n` is the declared module concept contract). Hatched cells denote a structurally unavailable typed placeholder; orange outlines denote a declared module with zero non-empty concepts. **b,** EasyICU cohort stays in each database; exact `n` is printed and the axis is logarithmic. **c,** Configured concepts by variable type and database support depth. Of {depth.shape[0]:,} concepts, {all_six:,} are non-empty in all six databases, {partial:,} in one to five and {none:,} in none. Availability requires at least one finite/non-null value and does not establish clinical equivalence. Repeated records are not treated as independent patients. Source data are provided in the QC-A03 Source Data files.

**Fig. QC2 | Harmonization contracts, release gates and traceable cross-database heterogeneity**
**a,** Pass/fail matrix for {contract_total} module-by-contract checks ({contract_pass} pass), including concept set and order, Arrow dtype, canonical unit, full physical schema, `stay_id`, `charttime` and six-file presence. **b,** The {displayed} largest eligible max/min database median ratios among {eligible} continuous concepts with at least {HETEROGENEITY_MIN_RECORDS} records in at least two databases. Points are descriptive repeated-record diagnostics without patient-level confidence intervals or P values; signed scales and non-positive medians are excluded under the prespecified ratio contract. **c,** Exact pass counts among {EXPECTED_EXPORT_COUNT} database-by-module files for native-v2 content, row-grain, null-time, metadata, sidecar and runtime-provenance gates. **d,** {anomaly_total} automated distribution review triggers by detector and source-trace status. A completed source trace can attribute heterogeneity to source measurement or recording practice; an unadjudicated flag requires traceback and is not evidence of a conversion defect. Source data are provided in the QC-A03 Source Data files.

Source run: `{lineage.source_run_id}`. Source `run_metadata.json` SHA-256: `{lineage.source_run_metadata_sha256}`.
"""
    path.write_text(text, encoding="utf-8")


def _write_qa_notes(
    path: Path,
    *,
    source_status: str,
    publication_eligible: bool,
    gate_errors: list[str],
    heterogeneity: pd.DataFrame,
    dpi: int,
) -> None:
    exclusion_counts = (
        heterogeneity["exclusion_reason"]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict()
    )
    text = f"""# QC-A03 figure QA notes

- Backend: Python / matplotlib only.
- Final width: {WIDTH_MM} mm; figure heights: {FIGURE_1_HEIGHT_MM} mm and {FIGURE_2_HEIGHT_MM} mm.
- Raster resolution: {dpi} dpi PNG and LZW-compressed TIFF.
- Vector exports: SVG with editable text and PDF with TrueType text.
- Source status: `{source_status}`.
- Publication eligible: `{str(publication_eligible).lower()}`.
- Currentness gate errors: {json.dumps(gate_errors, ensure_ascii=False)}.
- QC-A01 role: Extended Data / diagnostic atlas, not a primary manuscript claim figure.
- QC-A02 role: audit evidence layer, not a primary manuscript figure.
- Missing data: unavailable and structurally unavailable states are retained; no cell is silently dropped.
- Heterogeneity inclusion audit (one row per continuous concept before display ranking): {json.dumps(exclusion_counts, ensure_ascii=False, sort_keys=True)}.
- Heterogeneity display rule: top positive-median max/min ratios after the declared exclusions; all eligible and excluded concepts remain in Source Data.
- Statistical boundary: record-level diagnostic medians; no independent-patient CI, test or P value is claimed.
- Image integrity: no microscopy, photographic adjustment, local contrast editing or raster compositing is used.
- Visual checks required after render: panel-label position, 5–7 pt text readability at final size, heatmap annotation contrast, status watermark/banner, no clipping, selectable SVG/PDF text and color/grayscale distinction.
"""
    path.write_text(text, encoding="utf-8")


def render_submission_bundle(
    *,
    bundle: AuditBundle,
    output_dir: Path,
    source_status: str,
    dpi: int = DEFAULT_DPI,
    top_heterogeneity: int = 10,
) -> dict[str, Any]:
    if source_status not in SOURCE_STATUSES:
        raise ValueError(f"Unknown source status: {source_status}")
    if dpi < DEFAULT_DPI and source_status != "synthetic_layout_qa":
        raise ValueError("Real/candidate formal outputs require at least 600 dpi")

    apply_publication_style()
    support = build_module_support(bundle)
    depth = build_support_depth(bundle)
    contract = build_contract_matrix(bundle)
    gates = build_release_gates(bundle)
    heterogeneity = build_heterogeneity_table(bundle, top_n=top_heterogeneity)
    anomaly_trace = build_anomaly_trace(bundle)
    gate_errors = publication_gate_errors(
        bundle,
        contract_matrix=contract,
        release_gates=gates,
    )
    if source_status == "validated_current" and gate_errors:
        raise ValueError(
            "Cannot label QC figures validated_current: " + "; ".join(gate_errors)
        )
    publication_eligible = source_status == "validated_current" and not gate_errors

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    module_order = list(bundle.module_schema["module"].astype(str))

    source_tables = {
        "QC_Fig1a_module_support.csv": support,
        "QC_Fig1b_cohort_denominators.csv": bundle.cohort_denominators.copy(),
        "QC_Fig1c_concept_support_depth.csv": depth,
        "QC_Fig2a_contract_matrix.csv": contract,
        "QC_Fig2b_heterogeneity_inclusion_audit.csv": heterogeneity,
        "QC_Fig2c_release_gates.csv": gates,
        "QC_Fig2d_anomaly_trace.csv": anomaly_trace,
    }
    for filename, frame in source_tables.items():
        _write_csv(
            frame,
            source_dir / filename,
            lineage=bundle.lineage,
            source_status=source_status,
        )

    fig1 = render_figure_1(
        support=support,
        depth=depth,
        cohort_denominators=bundle.cohort_denominators,
        module_order=module_order,
        lineage=bundle.lineage,
        source_status=source_status,
    )
    fig2 = render_figure_2(
        contract=contract,
        heterogeneity=heterogeneity,
        gates=gates,
        anomaly_trace=anomaly_trace,
        lineage=bundle.lineage,
        source_status=source_status,
    )
    outputs = [
        *_save_figure_bundle(
            fig1,
            base=figures_dir / "QC_Fig1_cross_database_observational_support",
            dpi=dpi,
        ),
        *_save_figure_bundle(
            fig2,
            base=figures_dir / "QC_Fig2_harmonization_reliability",
            dpi=dpi,
        ),
    ]

    _write_legends(
        output_dir / "QC-A03_figure_legends.md",
        support=support,
        depth=depth,
        contract=contract,
        heterogeneity=heterogeneity,
        gates=gates,
        anomaly_trace=anomaly_trace,
        lineage=bundle.lineage,
    )
    _write_qa_notes(
        output_dir / "QC-A03_qa_notes.md",
        source_status=source_status,
        publication_eligible=publication_eligible,
        gate_errors=gate_errors,
        heterogeneity=heterogeneity,
        dpi=dpi,
    )

    tracked_files = [
        *outputs,
        *(source_dir / filename for filename in source_tables),
        output_dir / "QC-A03_figure_legends.md",
        output_dir / "QC-A03_qa_notes.md",
    ]
    manifest = {
        "version": 1,
        "stable_entry": "QC-A03",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": "python",
        "source_status": source_status,
        "publication_eligible": publication_eligible,
        "source_run_id": bundle.lineage.source_run_id,
        "source_run_metadata_sha256": bundle.lineage.source_run_metadata_sha256,
        "input_sha256": bundle.input_hashes,
        "figure_roles": FIGURE_ROLES,
        "dimensions_mm": {
            "width": WIDTH_MM,
            "figure_1_height": FIGURE_1_HEIGHT_MM,
            "figure_2_height": FIGURE_2_HEIGHT_MM,
        },
        "raster_dpi": dpi,
        "export_formats": ["svg", "pdf", "png", "tiff"],
        "denominators": {
            "databases": len(DATABASES),
            "modules": EXPECTED_MODULE_COUNT,
            "database_module_files": EXPECTED_EXPORT_COUNT,
            "configured_concepts": int(depth.shape[0]),
            "database_concept_cells": int(bundle.field_contract.shape[0]),
        },
        "heterogeneity": {
            "minimum_records_per_database": HETEROGENEITY_MIN_RECORDS,
            "review_ratio": HETEROGENEITY_REVIEW_RATIO,
            "continuous_concepts_before_exclusion": int(heterogeneity.shape[0]),
            "eligible_concepts": int(heterogeneity["included"].fillna(False).sum()),
            "displayed_concepts": int(heterogeneity["displayed"].fillna(False).sum()),
            "exclusion_counts": heterogeneity["exclusion_reason"]
            .fillna("missing")
            .value_counts(dropna=False)
            .to_dict(),
            "interval_definition": (
                "none; repeated-record descriptive QC, not an independent-patient effect"
            ),
        },
        "currentness_gate_errors": gate_errors,
        "output_sha256": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in tracked_files
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    paths = _build_paths(
        qc_a01_root=args.qc_a01_root.resolve(),
        qc_a02_dir=args.qc_a02_dir.resolve(),
        run_metadata=args.run_metadata.resolve(),
    )
    bundle = load_audit_bundle(paths)
    manifest = render_submission_bundle(
        bundle=bundle,
        output_dir=args.output_dir.resolve(),
        source_status=args.source_status,
        dpi=args.dpi,
        top_heterogeneity=args.top_heterogeneity,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
