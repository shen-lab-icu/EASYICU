#!/usr/bin/env python3
"""QC-A02: Audit EasyICU exports for cross-database reliability.

The audit separates four concerns that distribution figures alone cannot:

1. physical Parquet schema compatibility;
2. concept availability and non-null coverage;
3. typed metadata/provenance completeness; and
4. manually verified extraction or source-data defects.

Outputs are lightweight CSV/JSON evidence only. Raw Parquet data remain outside
the Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DATABASES = ("aumc", "eicu", "hirid", "mimic", "miiv", "sic")
ID_COLUMNS = {
    "admissionid",
    "patientunitstayid",
    "patientid",
    "icustay_id",
    "stay_id",
    "CaseID",
}
INDEX_COLUMNS = ID_COLUMNS | {"charttime"}
NATIVE_SCHEMA_VERSION = "easyicu_native_export_v2"
CURRENT_QC_SOURCE_RUN_ID = "current_full6_native_v2_hirid_urine24_20260730"
CURRENT_QC_SOURCE_RUN_METADATA_SHA256 = (
    "62adfb6f29a05305d687802f0eaa1c98f0ba2c4b888bb122c7e29233b4663d04"
)


# Direct source traces for the review-only shifts retained by the sealed
# 2026-07-30 six-database package.  These records deliberately do not suppress
# the anomaly flags: they distinguish a verified source/recording difference
# from an untraced conversion defect while preserving the signal for
# database-stratified downstream sensitivity analyses.
DISTRIBUTION_ADJUDICATIONS: dict[
    tuple[str, str, str, str, str, str], dict[str, str]
] = {
    (
        CURRENT_QC_SOURCE_RUN_ID,
        CURRENT_QC_SOURCE_RUN_METADATA_SHA256,
        "chemistry",
        "bili_dir",
        "aumc vs miiv",
        "median_scale_shift",
    ): {
        "adjudication_status": "source_trace_complete",
        "adjudicated_origin": "source_measurement_distribution_and_sparsity",
        "adjudication_evidence": (
            "AUMC item 6812 has 375 numeric records with source median 0.62 "
            "umol and declared conversion x0.058467; this gives 0.0362495 "
            "mg/dL, exactly the 375-record export median."
        ),
        "required_action": (
            "Do not rescale or re-extract; retain the sparse-source flag and "
            "use database-stratified or availability-sensitive analyses."
        ),
    },
    (
        CURRENT_QC_SOURCE_RUN_ID,
        CURRENT_QC_SOURCE_RUN_METADATA_SHA256,
        "chemistry",
        "tri",
        "eicu vs mimic",
        "median_scale_shift",
    ): {
        "adjudication_status": "source_trace_complete",
        "adjudicated_origin": "source_assay_and_reporting_heterogeneity",
        "adjudication_evidence": (
            "Raw eICU Troponin-I rows declare ng/mL (192,317 numeric rows; "
            "overall median about 0.18), while raw MIMIC-III item 51002 "
            "declares ng/ml (5,526 rows; median 2.4); episode-bounded export "
            "medians are 0.2 and 2.8 without an added unit transform."
        ),
        "required_action": (
            "Do not force a scale correction; treat Troponin-I as "
            "database/assay heterogeneous and prespecify stratified sensitivity."
        ),
    },
    (
        CURRENT_QC_SOURCE_RUN_ID,
        CURRENT_QC_SOURCE_RUN_METADATA_SHA256,
        "hematology",
        "bnd",
        "mimic vs hirid",
        "median_scale_shift",
    ): {
        "adjudication_status": "source_trace_complete",
        "adjudicated_origin": "source_measurement_distribution_heterogeneity",
        "adjudication_evidence": (
            "The HiRID reference names variable 24000557 'Band form "
            "neutrophils/100 leukocytes in Blood' with unit %; its 40,560 raw "
            "values and current export both have median 20.5. MIMIC-III item "
            "51144 is also reported as % but has a substantially lower raw "
            "distribution."
        ),
        "required_action": (
            "Keep the canonical percent unit; report cross-database measurement "
            "heterogeneity and avoid pooled interpretation without sensitivity."
        ),
    },
    (
        CURRENT_QC_SOURCE_RUN_ID,
        CURRENT_QC_SOURCE_RUN_METADATA_SHA256,
        "vasopressors",
        "epi_dur",
        "aumc vs mimic",
        "median_scale_shift",
    ): {
        "adjudication_status": "source_trace_complete",
        "adjudicated_origin": "source_recording_and_treatment_duration_heterogeneity",
        "adjudication_evidence": (
            "AUMC item 6818 yields 2,715 order groups with floor-hour median "
            "1 h (export n=1,885, median 1 h). MIMIC-III CareVue/MetaVision "
            "epinephrine groups yield 3,615 non-negative durations with pooled "
            "median 9 h (export n=3,138, median 10 h); all paths use the "
            "declared floor-hour duration contract."
        ),
        "required_action": (
            "Do not rescale; use duration descriptively and account for "
            "database-specific infusion-record construction in pooled models."
        ),
    },
}


RESOLUTION_UPDATES: dict[str, dict[str, str]] = {
    "EICU-QC-P0-001": {
        "status": "fixed; verify non-empty MIMIC-III core modules in this rerun",
        "required_action": (
            "Keep the full-run non-null counts and per-module timings as regression evidence."
        ),
    },
    "EICU-QC-P0-002": {
        "status": "fixed by mandatory native-v2 physical projection",
        "required_action": (
            "Require exact schema equality, canonical stay_id/charttime and typed placeholders "
            "in every publication/Agent package."
        ),
    },
    "EICU-QC-P0-003": {
        "status": "fixed by pinned checkout and runtime provenance",
        "required_action": "Reject packages whose recorded commit/import path differs from the run contract.",
    },
    "EICU-QC-P0-004": {
        "status": "fixed; mixed-unit CRP conversion now retains the unit column",
        "required_action": "Review the rerun CRP medians and retain the mixed-unit regression test.",
    },
    "EICU-QC-P0-005": {
        "status": "mitigated; ambiguous AUMC/FEU D-dimer sources excluded",
        "required_action": "Do not restore these sources until FEU/DDU assay semantics are explicit.",
    },
    "EICU-QC-P0-006": {
        "status": "fixed; AUMC IFCC HbA1c converted to NGSP percent",
        "required_action": "Review the rerun cross-database HbA1c distribution.",
    },
    "EICU-QC-P0-007": {
        "status": "fixed; AUMC terlipressin removed from vasopressin rate",
        "required_action": "Keep terlipressin separate if it is added in a future catalog revision.",
    },
    "EICU-QC-P0-008": {
        "status": "mitigated; unreliable HiRID neutrophil/lymphocyte mappings removed",
        "required_action": "Add separate absolute/percentage concepts before restoring HiRID NLR/PLR.",
    },
    "EICU-QC-P0-009": {
        "status": "fixed; eICU ETCO2 post-aggregation double regex removed",
        "required_action": "Confirm non-empty ETCO2 and a physiologic median in this rerun.",
    },
    "EICU-QC-P1-010": {
        "status": "confirmed structural source absence for the ICU-linked cohort",
        "required_action": "Retain typed all-null output plus structural-availability metadata.",
    },
    "EICU-QC-P1-011": {
        "status": "fixed by the ICU-episode time-window contract",
        "required_action": (
            "Retain the 24-hour pre/post allowance, quarantine invalid source intervals "
            "before expansion and require zero time-axis violations in publication exports."
        ),
    },
    "EICU-QC-P1-012": {
        "status": "fixed and validated against the official HiRID rate semantics",
        "required_action": (
            "Keep OUTurine/h as mL/h, backfill each observed rate only over its preceding "
            "observation interval and require complete 6/12/24-hour coverage for KDIGO."
        ),
    },
}


# Findings below were verified against the prepared source tables and the
# current EasyICU source tree on 2026-07-29. They are deliberately explicit:
# this table is an issue register, not an automated anomaly detector.
VERIFIED_FINDINGS: tuple[dict[str, Any], ...] = (
    {
        "issue_id": "EICU-QC-P0-001",
        "severity": "critical",
        "classification": "old extraction-run failure",
        "database": "mimic",
        "module": "multiple",
        "concept": "multiple",
        "status": "fixed in current code; full rerun required",
        "evidence": (
            "Old full-six run has near-empty core MIMIC-III modules after a "
            "group worker exited -6. Prepared labevents contain 490,504 pCO2, "
            "530,657 pH, 490,522 pO2, 797,231 creatinine, 752,277 haemoglobin, "
            "778,163 platelet and 752,813 WBC numeric rows. Current source-tree "
            "loads recover these concepts."
        ),
        "root_cause": (
            "Forked workers inherited Arrow/DuckDB allocator and stale concept "
            "cache state; fallback validation checked file/schema existence but "
            "not source-to-output non-null completeness."
        ),
        "required_action": (
            "Rerun all six databases from a pinned current commit using spawned "
            "workers and fail the run when a declared source has raw values but "
            "the exported concept is empty."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-002",
        "severity": "critical",
        "classification": "data-contract defect",
        "database": "all",
        "module": "all",
        "concept": "all",
        "status": "open",
        "evidence": (
            "Only 15/19 modules share the same concept set, only 10/19 share "
            "the same concept order, and only 2/19 have no Arrow dtype mismatch. "
            "Native stay identifiers differ by database; old manifests have "
            "empty concept_meta and merge_keys."
        ),
        "root_cause": (
            "The legacy export preserves source-native identifiers and physical "
            "types, and it does not require typed native-export metadata."
        ),
        "required_action": (
            "Make native_export_v2 mandatory for publication/Agent exports; "
            "emit canonical stay_id/charttime, ordered null placeholders, stable "
            "Arrow dtypes, canonical units, time semantics and availability reasons."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-003",
        "severity": "critical",
        "classification": "environment/provenance risk",
        "database": "all",
        "module": "all",
        "concept": "all",
        "status": "open",
        "evidence": (
            "A bare server Python currently imports EasyICU from "
            "/home/zhuhb/project/ricu_to_python/pyricu (144 concepts), whereas "
            "PYTHONPATH=/home/zhuhb/workspace/phd-thesis/EASYICU/src imports the "
            "current repository (251 concepts)."
        ),
        "root_cause": "An older installation precedes the Git checkout on sys.path.",
        "required_action": (
            "Install the intended checkout in the run environment or pin PYTHONPATH; "
            "record easyicu.__file__, version, Git SHA and catalog checksum in every run."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-004",
        "severity": "high",
        "classification": "unit-conversion defect",
        "database": "mimic",
        "module": "chemistry",
        "concept": "crp",
        "status": "open",
        "evidence": (
            "Raw MIMIC-III CRP includes 5,334 mg/L rows (median 30.53) and "
            "1,055 mg/dL/MG/DL rows (median about 4.43), but current output has "
            "median 447 mg/L and only 3,326 values."
        ),
        "root_cause": (
            "Pooled DuckDB aggregation drops the unit column, then a unit-filtered "
            "x10 callback is treated as an implicit conversion for all rows."
        ),
        "required_action": (
            "Convert per raw row before pooling and preserve unit-aware raw/bounded "
            "counts; add a mixed-unit CRP regression test."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-005",
        "severity": "high",
        "classification": "unit/assay-definition defect",
        "database": "aumc",
        "module": "chemistry",
        "concept": "d_dimer",
        "status": "open",
        "evidence": (
            "AUMC item 10393 is recorded in mg/L (929 rows; median 3.24), but "
            "the catalog declares ng/mL and applies no conversion. The exported "
            "median is 3.275 versus 2,648 in MIMIC-III and 1,976.5 in MIMIC-IV."
        ),
        "root_cause": "Source unit and FEU/DDU assay semantics are not harmonized.",
        "required_action": (
            "Exclude AUMC D-dimer from cross-database output until mg/L-to-ng/mL "
            "and FEU/DDU policy is explicit and testable."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-006",
        "severity": "high",
        "classification": "mixed-unit pooling defect",
        "database": "aumc",
        "module": "hematology",
        "concept": "hba1c",
        "status": "open",
        "evidence": (
            "AUMC item 11812 contains percent values (298 rows; median 6.07) and "
            "item 16166 contains mmol/mol values (201 rows; median 42.11). They "
            "are pooled as percent, producing an exported median of 21.525."
        ),
        "root_cause": "NGSP percent and IFCC mmol/mol values are pooled without conversion.",
        "required_action": (
            "Convert IFCC mmol/mol to NGSP percent per row, or export separate "
            "typed variants before a canonical HbA1c concept is formed."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-007",
        "severity": "high",
        "classification": "semantic mapping defect",
        "database": "aumc",
        "module": "vasopressors",
        "concept": "adh_rate",
        "status": "open",
        "evidence": (
            "AUMC item 12467 is Terlipressine (Glypressin), a 0.5 mg bolus, but "
            "is mapped to vasopressin rate. Exported median is 1.50 U/min versus "
            "about 0.04 in eICU/MIMIC databases."
        ),
        "root_cause": "A terlipressin bolus is treated as continuous vasopressin infusion.",
        "required_action": "Remove the source from adh_rate or define a separate terlipressin concept.",
    },
    {
        "issue_id": "EICU-QC-P0-008",
        "severity": "high",
        "classification": "definition mismatch",
        "database": "hirid",
        "module": "hematology",
        "concept": "neut, lymph",
        "status": "open",
        "evidence": (
            "HiRID exports absolute counts (neutrophils median 9.145) while most "
            "other databases export percentages (medians about 72-78%). The "
            "catalog accepts both percent and 10^9/L under one concept."
        ),
        "root_cause": "Absolute counts and percentages share a single canonical name.",
        "required_action": (
            "Split neut_pct/neut_abs and lymph_pct/lymph_abs; compute NLR only "
            "from numerator and denominator with matching representations."
        ),
    },
    {
        "issue_id": "EICU-QC-P0-009",
        "severity": "high",
        "classification": "current extraction defect",
        "database": "eicu",
        "module": "ventilator",
        "concept": "etco2",
        "status": "open",
        "evidence": (
            "Raw respiratoryCharting has 10,288 numeric ETCO2 rows. Current "
            "source-tree pre-matches three labels and aggregates 9,765 rows, "
            "then returns an empty concept."
        ),
        "root_cause": (
            "After regex labels are converted into exact IDs and aggregated, "
            "the label column is gone; the post-aggregation regex is incorrectly "
            "applied to the renamed numeric value column."
        ),
        "required_action": (
            "Skip the second regex filter after successful regex pre-match, or "
            "retain an explicit selector column; add a non-empty real/synthetic regression test."
        ),
    },
    {
        "issue_id": "EICU-QC-P1-010",
        "severity": "medium",
        "classification": "source coverage limitation",
        "database": "miiv",
        "module": "chemistry",
        "concept": "tri",
        "status": "confirmed structural absence for ICU cohort",
        "evidence": (
            "MIMIC-IV item 52642 has 668 numeric rows, but none of their hadm_id "
            "values match an ICU stay. No output is therefore expected for the ICU cohort."
        ),
        "root_cause": "The declared source exists in the hospital table but not in ICU admissions.",
        "required_action": (
            "Report as source-unavailable for the ICU cohort, not extraction_failed; "
            "include raw and ICU-linked counts in availability metadata."
        ),
    },
    {
        "issue_id": "EICU-QC-P1-011",
        "severity": "medium",
        "classification": "time-validity defect",
        "database": "multiple",
        "module": "multiple",
        "concept": "charttime",
        "status": "open",
        "evidence": (
            "The old exports retain implausible offsets, including eICU minima "
            "near -876,308 h, MIMIC-IV minima near -8,775 h and AUMC maxima "
            "near 108,795 h."
        ),
        "root_cause": "Source sentinels/outliers survive without an explicit extraction window.",
        "required_action": (
            "Declare and enforce allowed pre-ICU lookback/follow-up windows; export "
            "excluded-row counts and preserve raw offsets only in audit evidence."
        ),
    },
    {
        "issue_id": "EICU-QC-P1-012",
        "severity": "medium",
        "classification": "derived-concept comparability risk",
        "database": "hirid",
        "module": "renal",
        "concept": "uo_6h, uo_12h, uo_24h",
        "status": "needs paired raw validation",
        "evidence": (
            "HiRID medians are about 3.47-3.76 mL/kg/h versus roughly 0.8-1.9 "
            "across the other available databases."
        ),
        "root_cause": (
            "The recent switch to hourly urine volume plus interval integration "
            "may interact with measurement frequency or interval semantics."
        ),
        "required_action": (
            "Validate a sampled set of stays against raw hourly urine values and "
            "hand-calculated 6/12/24 h windows before declaring comparability."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--figure-audit", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_run_lineage(
    run_metadata_path: Path,
    raw_metadata: bytes | None = None,
) -> dict[str, str]:
    """Return the exact source-run identity shared by QC-A01 and QC-A02."""

    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"Missing source run metadata: {run_metadata_path}")
    source_bytes = (
        raw_metadata if raw_metadata is not None else run_metadata_path.read_bytes()
    )
    metadata = json.loads(source_bytes)
    run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(
            f"Source run metadata has no non-empty run_id: {run_metadata_path}"
        )
    return {
        "source_run_id": run_id.strip(),
        "source_run_metadata_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _load_root_manifests(export_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for database in DATABASES:
        path = export_root / database / "_manifest.json"
        if not path.is_file():
            manifests[database] = {}
            continue
        manifests[database] = json.loads(path.read_text(encoding="utf-8"))
    return manifests


def _verify_source_manifest_hashes(
    export_root: Path,
    run_metadata: dict[str, Any],
) -> dict[str, str]:
    """Fail closed unless the six audited manifests match the run receipt."""

    expected = run_metadata.get("source_manifest_sha256")
    if not isinstance(expected, dict):
        raise ValueError(
            "run_metadata.source_manifest_sha256 must bind all six root manifests"
        )
    expected_databases = set(DATABASES)
    declared_databases = {str(value) for value in expected}
    if declared_databases != expected_databases:
        missing = sorted(expected_databases - declared_databases)
        unexpected = sorted(declared_databases - expected_databases)
        raise ValueError(
            "run_metadata.source_manifest_sha256 database mismatch: "
            f"missing={missing}; unexpected={unexpected}"
        )

    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for database in DATABASES:
        manifest_path = export_root / database / "_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
        actual_sha256 = _sha256(manifest_path)
        actual[database] = actual_sha256
        declared_sha256 = expected.get(database)
        if not isinstance(declared_sha256, str) or declared_sha256 != actual_sha256:
            mismatches.append(
                f"{database}: declared={declared_sha256!r}, actual={actual_sha256}"
            )
    if mismatches:
        raise ValueError("Source manifest SHA-256 mismatch: " + "; ".join(mismatches))
    return actual


def _expected_runtime_commit(
    run_metadata: dict[str, Any],
    database: str,
) -> str | None:
    """Return the declared source commit for one database package.

    A normal extraction has one ``easyicu_commit``. A curated publication
    package may replace one database after a source-specific correction, so it
    declares ``database_commits`` instead of pretending that all six exports
    came from one checkout.
    """

    database_commits = run_metadata.get("database_commits")
    if isinstance(database_commits, dict):
        value = database_commits.get(database)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = run_metadata.get("easyicu_commit")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _derive_module_concepts(
    export_root: Path,
    run_metadata: dict[str, Any],
) -> dict[str, list[str]]:
    configured = run_metadata.get("module_concepts")
    if isinstance(configured, dict) and configured:
        return {
            str(module): [str(value) for value in concepts]
            for module, concepts in configured.items()
        }
    module_names = sorted(
        {
            path.stem
            for database in DATABASES
            for path in (export_root / database).glob("*.parquet")
        }
    )
    result: dict[str, list[str]] = {}
    for module in module_names:
        ordered: list[str] = []
        for database in DATABASES:
            path = export_root / database / f"{module}.parquet"
            if not path.is_file():
                continue
            for name in pq.read_schema(path).names:
                if name not in INDEX_COLUMNS and name not in ordered:
                    ordered.append(name)
        result[module] = ordered
    return result


def _resolved_findings() -> pd.DataFrame:
    findings = pd.DataFrame(VERIFIED_FINDINGS)
    findings["legacy_status"] = findings["status"]
    for issue_id, updates in RESOLUTION_UPDATES.items():
        selector = findings["issue_id"] == issue_id
        for field, value in updates.items():
            findings.loc[selector, field] = value
    return findings


def _distribution_flags(audit: pd.DataFrame) -> pd.DataFrame:
    """Generate conservative, review-oriented cross-database anomaly signals."""
    rows: list[dict[str, Any]] = []
    numeric = audit[audit["plot_kind"] == "continuous"].copy()
    for (module, variable), group in numeric.groupby(["module", "variable"]):
        available = group[
            (group["non_null_or_finite"].fillna(0) >= 100)
            & group["median_sample"].notna()
        ].copy()
        for _, record in available.iterrows():
            lower = record.get("catalog_min")
            upper = record.get("catalog_max")
            minimum = record.get("minimum")
            maximum = record.get("maximum")
            if pd.notna(lower) and pd.notna(minimum) and float(minimum) < float(lower):
                rows.append(
                    {
                        "module": module,
                        "variable": variable,
                        "database": record["database"],
                        "flag": "below_catalog_range",
                        "severity": "review",
                        "evidence": f"minimum={minimum:g}; catalog_min={lower:g}",
                        "origin_classification": "conversion_or_source_outlier_requires_traceback",
                    }
                )
            if pd.notna(upper) and pd.notna(maximum) and float(maximum) > float(upper):
                rows.append(
                    {
                        "module": module,
                        "variable": variable,
                        "database": record["database"],
                        "flag": "above_catalog_range",
                        "severity": "review",
                        "evidence": f"maximum={maximum:g}; catalog_max={upper:g}",
                        "origin_classification": "conversion_or_source_outlier_requires_traceback",
                    }
                )
        positive = available[available["median_sample"] > 0]
        if len(positive) >= 2:
            lowest = positive.loc[positive["median_sample"].idxmin()]
            highest = positive.loc[positive["median_sample"].idxmax()]
            ratio = float(highest["median_sample"]) / float(lowest["median_sample"])
            if ratio >= 10:
                rows.append(
                    {
                        "module": module,
                        "variable": variable,
                        "database": f"{lowest['database']} vs {highest['database']}",
                        "flag": "median_scale_shift",
                        "severity": "high" if ratio >= 100 else "review",
                        "evidence": (
                            f"positive median ratio={ratio:.2f}; "
                            f"{lowest['database']}={lowest['median_sample']:.6g}; "
                            f"{highest['database']}={highest['median_sample']:.6g}"
                        ),
                        "origin_classification": (
                            "unit_or_definition_mismatch_candidate_not_proven_by_distribution_alone"
                        ),
                    }
                )
    columns = [
        "module",
        "variable",
        "database",
        "flag",
        "severity",
        "evidence",
        "origin_classification",
    ]
    return pd.DataFrame(rows, columns=columns)


def _adjudicate_distribution_flags(
    flags: pd.DataFrame,
    *,
    source_run_id: str | None,
    source_run_metadata_sha256: str | None,
) -> pd.DataFrame:
    """Attach source traces only to the exact sealed run and anomaly type."""

    result = flags.copy()
    result["adjudication_status"] = "unadjudicated"
    result["adjudication_source_run_id"] = pd.NA
    result["adjudication_source_run_metadata_sha256"] = pd.NA
    result["adjudicated_origin"] = pd.NA
    result["adjudication_evidence"] = pd.NA
    result["required_action"] = (
        "Trace the source field, unit and transformation before interpretation."
    )
    for index, row in result.iterrows():
        key = (
            str(source_run_id),
            str(source_run_metadata_sha256),
            str(row.get("module")),
            str(row.get("variable")),
            str(row.get("database")),
            str(row.get("flag")),
        )
        adjudication = DISTRIBUTION_ADJUDICATIONS.get(key)
        if adjudication is None:
            continue
        result.at[index, "adjudication_source_run_id"] = source_run_id
        result.at[index, "adjudication_source_run_metadata_sha256"] = (
            source_run_metadata_sha256
        )
        for field, value in adjudication.items():
            result.at[index, field] = value
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _concept_metadata_complete(entry: dict[str, Any]) -> bool:
    """Require one unique metadata binding for every selected source concept."""

    selected = _string_list(entry.get("concept_ids"))
    metadata_columns = _string_list(entry.get("column_metadata_columns"))
    return (
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) == set(metadata_columns)
        and len(metadata_columns) == len(set(metadata_columns))
    )


def _structural_placeholder_checks(
    *,
    module: str,
    entry: dict[str, Any],
    expected_concepts: list[str],
    parquet_names: list[str],
    parquet_types: dict[str, str],
    actual_row_count: int | None,
    manifest_schema_matches_parquet: bool,
) -> dict[str, bool]:
    """Validate that structural absence is a real typed zero-row contract."""

    expected_names = ["stay_id"]
    if module != "demographics":
        expected_names.append("charttime")
    expected_names.extend(expected_concepts)
    status = entry.get("concept_status")
    status_complete = (
        bool(expected_concepts)
        and isinstance(status, dict)
        and set(status) == set(expected_concepts)
        and all(
            isinstance(status.get(concept), dict)
            and status[concept].get("availability")
            == "structurally_unavailable_placeholder"
            and status[concept].get("non_null") == 0
            for concept in expected_concepts
        )
    )
    typed_schema = (
        parquet_names == expected_names
        and parquet_types.get("stay_id") == "int64"
        and (module == "demographics" or parquet_types.get("charttime") == "double")
        and all(
            concept in parquet_types and parquet_types[concept] != "null"
            for concept in expected_concepts
        )
        and manifest_schema_matches_parquet
    )
    selection_empty = (
        not _string_list(entry.get("concept_ids"))
        and not _string_list(entry.get("column_metadata_columns"))
        and entry.get("concepts") == 0
    )
    checks = {
        "structural_declared": entry.get("availability") == "structurally_unavailable",
        "structural_declared_zero_rows": entry.get("rows") == 0,
        "structural_actual_zero_rows": actual_row_count == 0,
        "structural_schema_typed": typed_schema,
        "structural_concept_status_complete": status_complete,
        "structural_selection_empty": selection_empty,
        "structural_physical_concepts_complete": _string_list(
            entry.get("physical_concept_ids")
        )
        == expected_concepts,
    }
    checks["structural_placeholder_valid"] = all(checks.values())
    return checks


def _metadata_gap_mask(manifests: pd.DataFrame) -> pd.Series:
    complete_metadata = (
        manifests["concept_metadata_complete"].fillna(False).astype(bool)
    )
    valid_structural = (
        manifests["structural_placeholder_valid"].fillna(False).astype(bool)
    )
    return ~(complete_metadata | valid_structural)


def _manifest_metadata_coverage(manifests: pd.DataFrame) -> dict[str, int]:
    """Separate selected-concept metadata from explicit structural absence.

    A zero-row structural placeholder is intentionally excluded from the
    selected concept index and therefore has no primary column-metadata
    binding.  It closes the contract only after its manifest status, declared
    and physical row counts, typed schema and per-concept statuses all pass.
    Reporting the two states separately prevents the valid 108 + 6 = 114
    contract from being mistaken for six undocumented metadata gaps.
    """

    has_metadata = manifests["concept_meta_count"].fillna(0).astype(int) > 0
    complete_metadata = (
        manifests["concept_metadata_complete"].fillna(False).astype(bool)
    )
    structural = manifests["availability"].eq("structurally_unavailable")
    valid_structural = (
        manifests["structural_placeholder_valid"].fillna(False).astype(bool)
    )
    covered = complete_metadata | valid_structural
    gap_count = int((~covered).sum())
    return {
        "manifest_rows_with_concept_meta": int(has_metadata.sum()),
        "manifest_rows_with_complete_concept_meta": int(complete_metadata.sum()),
        "manifest_structurally_unavailable_rows": int(structural.sum()),
        "manifest_valid_structural_placeholder_rows": int(valid_structural.sum()),
        "manifest_invalid_structural_placeholder_rows": int(
            (structural & ~valid_structural).sum()
        ),
        "manifest_rows_with_concept_meta_or_structural_status": int(covered.sum()),
        "manifest_metadata_contract_gap_rows": gap_count,
        "manifest_rows_missing_concept_meta_without_valid_structural_status": gap_count,
        # Backward-compatible key retained for existing report consumers.
        "manifest_rows_missing_concept_meta_without_structural_status": gap_count,
    }


def _raise_for_metadata_gaps(manifests: pd.DataFrame) -> None:
    gaps = manifests.loc[
        _metadata_gap_mask(manifests),
        [
            "database",
            "module",
            "availability",
            "concept_metadata_complete",
            "structural_placeholder_valid",
        ],
    ]
    if gaps.empty:
        return
    records = gaps.to_dict(orient="records")
    raise ValueError(
        "Manifest metadata contract gaps detected; audit failed closed: "
        + json.dumps(records, ensure_ascii=False, sort_keys=True)
    )


def _build_report_artifact(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    modules: pd.DataFrame,
    availability: pd.DataFrame,
    findings: pd.DataFrame,
) -> dict[str, Any]:
    """Build the bounded Data Analytics report payload used for handoff."""

    generated_at = datetime.now(UTC).isoformat()
    availability_rows: list[dict[str, Any]] = []
    kind_labels = {
        "continuous": "连续变量",
        "binary": "二元变量",
        "ordinal": "有序变量",
        "categorical": "分类变量",
        "unavailable": "完全不可用",
    }
    coverage_labels = (
        ("六库均可用", lambda x: x == 6),
        ("部分数据库可用", lambda x: (x > 0) & (x < 6)),
        ("六库均不可用", lambda x: x == 0),
    )
    for plot_kind, group in availability.groupby("plot_kind", sort=False):
        for coverage_label, selector in coverage_labels:
            availability_rows.append(
                {
                    "plot_kind": kind_labels.get(plot_kind, plot_kind),
                    "coverage_class": coverage_label,
                    "concept_count": int(selector(group["databases_available"]).sum()),
                    "total_in_kind": int(len(group)),
                    "all_six_count": int((group["databases_available"] == 6).sum()),
                    "partial_count": int(
                        (
                            (group["databases_available"] > 0)
                            & (group["databases_available"] < 6)
                        ).sum()
                    ),
                    "none_count": int((group["databases_available"] == 0).sum()),
                }
            )

    module_rows = []
    for row in modules.sort_values(
        ["missing_schema_slots", "type_mismatch_field_count"],
        ascending=[False, False],
    ).to_dict(orient="records"):
        module_rows.append(
            {
                "module": row["module"],
                "column_set": "一致" if row["same_concept_set"] else "不一致",
                "column_order": "一致" if row["same_concept_order"] else "不一致",
                "dtype_mismatch_fields": int(row["type_mismatch_field_count"]),
                "missing_schema_slots": int(row["missing_schema_slots"]),
                "all_six_parquets": ("是" if row["all_six_parquets_present"] else "否"),
            }
        )

    priority = {"critical": 1, "high": 2, "medium": 3, "low": 4}
    issue_rows = []
    for row in findings.to_dict(orient="records"):
        issue_rows.append(
            {
                "priority": priority.get(str(row["severity"]), 9),
                "severity": row["severity"],
                "issue_id": row["issue_id"],
                "database": row["database"],
                "module": row["module"],
                "concept": row["concept"],
                "classification": row["classification"],
                "status": row["status"],
                "required_action": row["required_action"],
            }
        )

    metric_row = {
        "module_total": summary["module_count"],
        "same_column_set": summary["modules_same_concept_set"],
        "same_column_order": summary["modules_same_concept_order"],
        "no_dtype_mismatch": summary["modules_without_type_mismatch"],
        "concept_total": summary["concept_panel_count"],
        "concept_all_six": summary["concepts_available_all_six"],
        "missing_schema_slots": summary["missing_schema_slots"],
    }

    sources = [
        {
            "id": "schema_audit",
            "label": "EasyICU module schema audit",
            "path": str(output_dir / "module_schema_summary.csv"),
            "query": {
                "language": "sql",
                "engine": "duckdb",
                "sql": (
                    "SELECT * FROM read_csv_auto('"
                    f"{output_dir / 'module_schema_summary.csv'}"
                    "', header = true);"
                ),
                "description": (
                    "PyArrow audit of physical Parquet columns, order, types and "
                    "expected module concepts across six databases."
                ),
                "tables_used": ["six-database EasyICU module Parquet exports"],
                "filters": [
                    "19 configured modules",
                    "AUMCdb, eICU-CRD, HiRID, MIMIC-III, MIMIC-IV and SICdb",
                ],
                "metric_definitions": [
                    "same_column_set: all six physical concept-column sets are identical",
                    "same_column_order: all six physical concept-column orders are identical",
                    "dtype_mismatch_fields: expected concepts with more than one observed Arrow dtype",
                    "missing_schema_slots: expected database-concept columns absent from physical schema",
                ],
            },
        },
        {
            "id": "availability_audit",
            "label": "EasyICU concept availability audit",
            "path": str(output_dir / "concept_availability.csv"),
            "query": {
                "language": "sql",
                "engine": "duckdb",
                "sql": (
                    "SELECT * FROM read_csv_auto('"
                    f"{output_dir / 'concept_availability.csv'}"
                    "', header = true);"
                ),
                "description": (
                    "Aggregated non-null/finite coverage from 1,686 "
                    "concept-by-database panels."
                ),
                "tables_used": ["cross-database variable_audit.csv"],
                "filters": [
                    "A concept is available in a database when at least one finite/non-null value exists",
                    "Availability does not by itself distinguish structural absence from extraction failure",
                ],
                "metric_definitions": [
                    "databases_available: number of six databases with at least one non-null/finite value",
                    "available_all_six: databases_available equals 6",
                ],
            },
        },
        {
            "id": "issue_register",
            "label": "Verified EasyICU reliability issue register",
            "path": str(output_dir / "verified_issue_register.csv"),
            "query": {
                "language": "sql",
                "engine": "duckdb",
                "sql": (
                    "SELECT * FROM read_csv_auto('"
                    f"{output_dir / 'verified_issue_register.csv'}"
                    "', header = true);"
                ),
                "description": (
                    "Curated findings verified against prepared source tables, "
                    "legacy exports, current EasyICU code and direct current-code loads."
                ),
                "tables_used": [
                    "prepared ICU source tables",
                    "legacy EasyICU full-six exports",
                    "current EasyICU concept catalog and extraction source",
                ],
                "filters": [
                    "Only issues with direct source or code evidence are included",
                    "Unresolved distribution anomalies are labelled needs validation rather than defects",
                ],
            },
        },
    ]

    title = "EasyICU 六库提取一致性与可靠性审计"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Publication- and Agent-readiness audit of 19 EasyICU modules across "
            "six ICU databases."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "card_schema_set",
                "dataset": "headline_metrics",
                "sourceId": "schema_audit",
                "description": "物理 Parquet 中六库概念列集合完全一致的模块数。",
                "metrics": [
                    {
                        "label": "列集合一致模块",
                        "field": "same_column_set",
                        "format": "number",
                    },
                    {"label": "总模块", "field": "module_total", "format": "number"},
                ],
            },
            {
                "id": "card_schema_order",
                "dataset": "headline_metrics",
                "sourceId": "schema_audit",
                "description": "物理 Parquet 中六库概念列顺序完全一致的模块数。",
                "metrics": [
                    {
                        "label": "列顺序一致模块",
                        "field": "same_column_order",
                        "format": "number",
                    },
                    {"label": "总模块", "field": "module_total", "format": "number"},
                ],
            },
            {
                "id": "card_dtype",
                "dataset": "headline_metrics",
                "sourceId": "schema_audit",
                "description": "没有任何跨库 Arrow 类型差异的模块数。",
                "metrics": [
                    {
                        "label": "类型完全一致模块",
                        "field": "no_dtype_mismatch",
                        "format": "number",
                    },
                    {
                        "label": "缺失 schema 槽位",
                        "field": "missing_schema_slots",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "card_coverage",
                "dataset": "headline_metrics",
                "sourceId": "availability_audit",
                "description": "在六个数据库均有至少一个非空/有限值的概念面板数。",
                "metrics": [
                    {
                        "label": "六库均非空概念",
                        "field": "concept_all_six",
                        "format": "number",
                    },
                    {
                        "label": "总概念面板",
                        "field": "concept_total",
                        "format": "number",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "availability_by_kind",
                "title": "六库概念可用性（按变量类型）",
                "subtitle": (
                    f"{summary['concepts_available_all_six']}/"
                    f"{summary['concept_panel_count']} 个概念在六库均非空；"
                    "部分覆盖需要结合 native-v2 可用性原因解释。"
                ),
                "intent": "composition",
                "question": "不同变量类型的六库完整覆盖、部分覆盖和完全缺失各有多少？",
                "rationale": (
                    "堆叠柱图同时保留每类变量的总量和覆盖组成，适合比较覆盖缺口。"
                ),
                "comparisonContext": {
                    "denominator": "Each configured concept panel within plot kind",
                    "grain": "concept",
                    "unit": "concepts",
                },
                "type": "stackedBar",
                "dataset": "availability_by_kind",
                "sourceId": "availability_audit",
                "encodings": {
                    "x": {
                        "field": "plot_kind",
                        "type": "nominal",
                        "label": "变量类型",
                    },
                    "y": {
                        "field": "concept_count",
                        "type": "quantitative",
                        "label": "概念数",
                    },
                    "color": {
                        "field": "coverage_class",
                        "type": "nominal",
                        "label": "覆盖等级",
                    },
                    "tooltip": [
                        {
                            "field": "total_in_kind",
                            "type": "quantitative",
                            "label": "该类型总概念",
                        },
                        {
                            "field": "all_six_count",
                            "type": "quantitative",
                            "label": "六库均可用",
                        },
                        {
                            "field": "partial_count",
                            "type": "quantitative",
                            "label": "部分数据库可用",
                        },
                        {
                            "field": "none_count",
                            "type": "quantitative",
                            "label": "六库均不可用",
                        },
                    ],
                },
                "combinationRationale": "颜色编码覆盖等级，堆叠高度保持每类概念总数。",
                "palette": {"kind": "categorical", "name": "coverage-status"},
                "legend": {
                    "position": "bottom",
                    "sort": "spec",
                    "title": "覆盖等级",
                },
                "settings": {
                    "groupMode": "stacked",
                    "orientation": "vertical",
                    "showValues": True,
                    "categoryLabelPolicy": "wrap",
                },
                "layout": "full",
            }
        ],
        "tables": [
            {
                "id": "module_schema_table",
                "title": "19 个模块的物理 schema 一致性",
                "subtitle": "按缺失 schema 槽位及类型不一致字段数降序。",
                "dataset": "module_schema",
                "sourceId": "schema_audit",
                "defaultSort": {"field": "missing_schema_slots", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "module", "label": "模块", "type": "text"},
                    {"field": "column_set", "label": "列集合", "type": "text"},
                    {"field": "column_order", "label": "列顺序", "type": "text"},
                    {
                        "field": "dtype_mismatch_fields",
                        "label": "类型不一致字段",
                        "type": "number",
                    },
                    {
                        "field": "missing_schema_slots",
                        "label": "缺失槽位",
                        "type": "number",
                    },
                    {
                        "field": "all_six_parquets",
                        "label": "六库均有文件",
                        "type": "text",
                    },
                ],
            },
            {
                "id": "issue_register_table",
                "title": "已核实问题与来源判定",
                "subtitle": "Critical/High/Medium；仅收录已有原始表或代码证据的问题。",
                "dataset": "issue_register",
                "sourceId": "issue_register",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "priority", "label": "优先级", "type": "number"},
                    {"field": "severity", "label": "严重度", "type": "text"},
                    {"field": "issue_id", "label": "问题 ID", "type": "text"},
                    {"field": "database", "label": "数据库", "type": "text"},
                    {"field": "module", "label": "模块", "type": "text"},
                    {"field": "concept", "label": "概念", "type": "text"},
                    {"field": "classification", "label": "来源判定", "type": "text"},
                    {"field": "status", "label": "状态", "type": "text"},
                    {"field": "required_action", "label": "要求动作", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## 技术结论\n\n"
                    f"本轮 native-v2 导出中，{summary['modules_same_full_physical_schema']}/"
                    f"{summary['module_count']} 个模块通过六库完整物理 schema 一致性检查，"
                    f"{summary['native_v2_manifest_rows']}/{summary['manifest_row_count']} 个"
                    "数据库×模块 manifest 条目声明 native-v2。只有 schema、manifest、provenance、"
                    "分布异常和已知问题复验同时通过的输出才可进入论文补充材料或 Agent 默认输入。\n\n"
                    "旧批次中确认的 MIMIC-III 运行故障、CRP/HbA1c 单位转换、AUMC vasopressin "
                    "语义映射和 eICU ETCO₂ 提取问题已在代码中修复或安全排除；本报告使用本轮重跑数据"
                    "复验这些修复，并把仍需原始表逐 stay 核验的问题保留为开放项。"
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "card_schema_set",
                    "card_schema_order",
                    "card_dtype",
                    "card_coverage",
                ],
            },
            {
                "id": "coverage_section",
                "type": "markdown",
                "body": (
                    "## 六库覆盖并不等于六库可靠\n\n"
                    f"{summary['concept_panel_count']} 个概念面板中，"
                    f"{summary['concepts_available_all_six']} 个在六库均有非空值。其余面板可能是数据库"
                    "真正没有相应源数据，也可能需要进一步源到输出核验。因而投稿时不能只放密度图，还必须附带"
                    "每个数据库的 `available / structurally_unavailable / extraction_failed / unit_suspect` "
                    "原因码。下图展示覆盖组成，不把“部分覆盖”自动解释成质量问题。"
                ),
                "sourceId": "availability_audit",
            },
            {
                "id": "availability_chart",
                "type": "chart",
                "chartId": "availability_by_kind",
                "layout": "full",
            },
            {
                "id": "schema_section",
                "type": "markdown",
                "body": (
                    "## native-v2 物理合同验收\n\n"
                    f"{summary['modules_same_concept_set']}/{summary['module_count']} 个模块概念列集合一致，"
                    f"{summary['modules_same_concept_order']}/{summary['module_count']} 个模块列顺序一致，"
                    f"{summary['modules_without_type_mismatch']}/{summary['module_count']} 个模块无 Arrow "
                    "类型差异。下游合同还要求统一 `stay_id:int64`、非人口学模块的相对小时 "
                    "`charttime:double`、有类型的全空占位、内容寻址元数据 sidecar 和固定 Git provenance。"
                ),
                "sourceId": "schema_audit",
            },
            {
                "id": "schema_table",
                "type": "table",
                "tableId": "module_schema_table",
                "layout": "full",
            },
            {
                "id": "root_cause_section",
                "type": "markdown",
                "body": (
                    "## 问题来源是混合的，但主要修复点在 EasyICU\n\n"
                    "已核实问题可以分为四类：旧批次运行故障、当前提取器/回调缺陷、概念与单位定义不一致、"
                    "以及源数据库的真实结构性缺失。对核心 MIMIC-III 实验室指标，原始 Parquet 中存在"
                    "数十万条数值，当前代码也能直接恢复，因此不能归咎于原始数据库。对 MIMIC-IV "
                    "肌钙蛋白 I，则原始医院记录无法链接到任何 ICU stay，应该明确标为源覆盖限制。"
                ),
                "sourceId": "issue_register",
            },
            {
                "id": "issue_table",
                "type": "table",
                "tableId": "issue_register_table",
                "layout": "full",
            },
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## 审计范围与可比性定义\n\n"
                    f"本次审计对象是批次 `{summary.get('run_id')}` 的 "
                    f"{summary['module_count']} 个模块、{summary['concept_panel_count']} 个概念和 "
                    f"{summary['database_concept_panel_count']} 个“概念×数据库”面板。"
                    "**统一格式**要求同一模块具有相同的"
                    "规范列名、顺序、Arrow 类型、缺失占位策略、`stay_id`、以 ICU 入科为零点的小时偏移、"
                    "规范单位及明确可用性原因。**跨库可靠**进一步要求单位/化验语义一致、主要分布合理，"
                    "并能从输出非空计数追溯到 ICU 可链接的原始源计数。"
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "body": (
                    "## 审计方法能区分源缺失与转换丢失\n\n"
                    "审计逐个读取六库 Parquet schema，比较期望概念集合、物理列顺序与 Arrow 类型；"
                    "利用已生成的变量审计表统计非空/有限值覆盖；再对异常概念回查 prepared source "
                    "tables、当前 concept catalog、DuckDB 聚合和当前源码直接加载结果。只有具备原始计数、"
                    "ICU 链接计数或明确代码路径证据的项目才进入“已核实问题”表。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 解释边界\n\n"
                    "记录级密度受同一 ICU stay 内测量频率影响，适合提取 QC 和数据漂移检查，不能解释为"
                    "独立患者分布或数据库等价性证明。严格 Arrow 类型审计用于定义 Agent 输入合同，"
                    "不表示所有分布差异都会造成统计偏倚。HiRID 尿量已按官方 mL/h 速率语义和每条记录"
                    "之前的观察区间完成源级复核；修复后仍存在的跨库差异应结合病例组合、记录实践和可观测性解释，"
                    "不能仅凭曲线再次宣称为转换错误。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 投稿和 Agent 使用前的 P0 验收门槛\n\n"
                    "1. 保留 eICU ETCO₂、MIMIC-III CRP、AUMC D-dimer/HbA1c/`adh_rate`、"
                    "HiRID 尿量/MCHC 和 neut/lymph 定义的跨平台回归测试。\n"
                    "2. 从固定 Git SHA 重新跑六库；运行 manifest 必须记录 `easyicu.__file__`、版本、"
                    "Git SHA、catalog checksum、操作系统、Python/Arrow/DuckDB 版本和峰值 RSS。\n"
                    "3. 发布强类型输出合同：统一 `stay_id`、`charttime`、列集合/顺序/类型、规范单位、"
                    "缺失占位和原因码；投稿/Agent 输出必须开启并验收 native-v2 typed metadata。\n"
                    "4. 增加 source-to-output 守门：声明的数据源若存在 ICU 可链接数值而输出为 0，"
                    "整轮失败；同时记录原始、链接、转换后、bounds 后和最终非空计数。\n"
                    "5. 只有新批次全部通过 schema、单位、范围、覆盖和 provenance 门槛后，"
                    "再重画 19 模块补充图。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "- HiRID 修复后的 KDIGO 尿量率仍略高于部分数据库时，病例组合、记录实践和体重"
                    "可观测性分别解释多少差异？\n"
                    "- AUMC 直接胆红素的源字典、量级和换算是否与其他数据库指向同一检验？\n"
                    "- native-v2 当前能提供 typed metadata，但是否还需要对物理 Parquet 做统一 ID、"
                    "列顺序和类型的二次封装？\n"
                    "- Windows、macOS 与 Linux 的 16 GB 运行是否能在相同输入上产生完全一致的 schema、"
                    "非空计数和抽样分位数？"
                ),
            },
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline_metrics": [metric_row],
            "availability_by_kind": availability_rows,
            "module_schema": module_rows,
            "issue_register": issue_rows,
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_run_metadata = args.run_metadata.read_bytes()
    run_metadata = json.loads(raw_run_metadata)
    lineage = _source_run_lineage(args.run_metadata, raw_run_metadata)
    source_manifest_sha256 = _verify_source_manifest_hashes(
        args.export_root,
        run_metadata,
    )
    module_concepts = _derive_module_concepts(args.export_root, run_metadata)
    if len(module_concepts) != 19:
        raise ValueError(f"Expected 19 module contracts, found {len(module_concepts)}")
    root_manifests = _load_root_manifests(args.export_root)
    audit = pd.read_csv(args.figure_audit)
    audit["available"] = audit["non_null_or_finite"].fillna(0).astype(int) > 0

    field_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for module, expected in module_concepts.items():
        schemas: dict[str, dict[str, str]] = {}
        orders: dict[str, list[str]] = {}
        id_columns: dict[str, list[str]] = {}
        charttime_presence: dict[str, bool] = {}
        physical_schema_signatures: dict[str, tuple[tuple[str, str], ...]] = {}

        for database in DATABASES:
            parquet = args.export_root / database / f"{module}.parquet"
            manifest_path = args.export_root / database / "_manifest.json"
            schema = pq.read_schema(parquet) if parquet.exists() else None
            actual_row_count = (
                int(pq.ParquetFile(parquet).metadata.num_rows)
                if parquet.is_file()
                else None
            )
            names = schema.names if schema is not None else []
            physical_schema_signatures[database] = (
                tuple((field.name, str(field.type)) for field in schema)
                if schema is not None
                else ()
            )
            types = (
                {field.name: str(field.type) for field in schema}
                if schema is not None
                else {}
            )
            concept_order = [name for name in names if name not in INDEX_COLUMNS]
            schemas[database] = {name: types[name] for name in concept_order}
            orders[database] = concept_order
            id_columns[database] = [name for name in names if name in ID_COLUMNS]
            charttime_presence[database] = "charttime" in names

            manifest = root_manifests[database]
            runtime = manifest.get("runtime_provenance") or {}
            expected_runtime_commit = _expected_runtime_commit(
                run_metadata,
                database,
            )
            manifest_files = {
                entry.get("module"): entry
                for entry in manifest.get("files", [])
                if isinstance(entry, dict)
            }
            entry = manifest_files.get(module) or {}
            recorded_schema = entry.get("physical_schema") or {}
            sidecar = manifest.get("column_metadata") or {}
            sidecar_name = sidecar.get("file")
            sidecar_path = (
                args.export_root / database / sidecar_name
                if isinstance(sidecar_name, str)
                else None
            )
            sidecar_exists = bool(sidecar_path and sidecar_path.is_file())
            recorded_sidecar_sha = sidecar.get("sha256")
            sidecar_sha_matches = bool(
                sidecar_exists
                and recorded_sidecar_sha
                and _sha256(sidecar_path) == recorded_sidecar_sha
            )
            manifest_schema_matches_parquet = recorded_schema == types
            concept_metadata_complete = _concept_metadata_complete(entry)
            structural_checks = _structural_placeholder_checks(
                module=module,
                entry=entry,
                expected_concepts=expected,
                parquet_names=names,
                parquet_types=types,
                actual_row_count=actual_row_count,
                manifest_schema_matches_parquet=manifest_schema_matches_parquet,
            )
            saved_path = str(parquet)
            manifest_rows.append(
                {
                    "module": module,
                    "database": database,
                    "manifest_exists": manifest_path.exists(),
                    "saved_path": saved_path,
                    "saved_path_exists": parquet.is_file(),
                    "schema_version": manifest.get("schema_version"),
                    "native_v2": manifest.get("schema_version")
                    == NATIVE_SCHEMA_VERSION,
                    "availability": entry.get("availability"),
                    "manifest_declared_row_count": entry.get("rows"),
                    "actual_parquet_row_count": actual_row_count,
                    "selected_concept_count": len(entry.get("concept_ids") or []),
                    "manifest_concept_count": len(
                        entry.get("physical_concept_ids") or []
                    ),
                    "concept_meta_count": len(
                        entry.get("column_metadata_columns") or []
                    ),
                    "concept_metadata_complete": concept_metadata_complete,
                    **structural_checks,
                    "merge_key_count": int("stay_id" in names)
                    + int("charttime" in names),
                    "manifest_schema_matches_parquet": manifest_schema_matches_parquet,
                    "runtime_commit": runtime.get("easyicu_git_commit"),
                    "expected_runtime_commit": expected_runtime_commit,
                    "runtime_commit_matches_run": bool(
                        runtime.get("easyicu_git_commit")
                        and expected_runtime_commit
                        and runtime.get("easyicu_git_commit") == expected_runtime_commit
                    ),
                    "runtime_git_dirty": runtime.get("easyicu_git_dirty"),
                    "sidecar_file": sidecar_name,
                    "sidecar_exists": sidecar_exists,
                    "sidecar_sha256_matches": sidecar_sha_matches,
                    "manifest_error_count": 0,
                    "manifest_warning_count": 0,
                }
            )

            audit_subset = audit[
                (audit["module"] == module) & (audit["database"] == database)
            ].set_index("variable")
            for position, concept in enumerate(expected):
                row = (
                    audit_subset.loc[concept] if concept in audit_subset.index else None
                )
                field_rows.append(
                    {
                        "module": module,
                        "database": database,
                        "concept": concept,
                        "expected_position": position,
                        "physical_position": (
                            names.index(concept) if concept in names else pd.NA
                        ),
                        "present": concept in names,
                        "arrow_type": types.get(concept),
                        "row_count": (
                            int(row["row_count"]) if row is not None else pd.NA
                        ),
                        "non_null_count": (
                            int(row["non_null_or_finite"]) if row is not None else pd.NA
                        ),
                        "available": (
                            bool(row["available"]) if row is not None else False
                        ),
                        "unit": row["unit"] if row is not None else None,
                    }
                )

        concept_sets = [set(schemas[db]) for db in DATABASES]
        concept_orders = [orders[db] for db in DATABASES]
        type_mismatch_fields = 0
        for concept in expected:
            observed = {
                schemas[db].get(concept)
                for db in DATABASES
                if schemas[db].get(concept) is not None
            }
            type_mismatch_fields += int(len(observed) > 1)

        missing_slots = sum(
            int(concept not in schemas[database])
            for database in DATABASES
            for concept in expected
        )
        module_rows.append(
            {
                "module": module,
                "expected_concept_count": len(expected),
                "all_six_parquets_present": all(
                    (args.export_root / db / f"{module}.parquet").exists()
                    for db in DATABASES
                ),
                "same_concept_set": len({tuple(sorted(x)) for x in concept_sets}) == 1,
                "same_concept_order": len({tuple(x) for x in concept_orders}) == 1,
                "same_full_physical_schema": len(
                    set(physical_schema_signatures.values())
                )
                == 1
                and all(physical_schema_signatures.values()),
                "type_mismatch_field_count": type_mismatch_fields,
                "missing_schema_slots": missing_slots,
                "canonical_stay_id_all_six": all(
                    id_columns[database] == ["stay_id"] for database in DATABASES
                ),
                "canonical_charttime_all_six": all(
                    charttime_presence[database] == (module != "demographics")
                    for database in DATABASES
                ),
                "native_id_columns": json.dumps(id_columns, ensure_ascii=False),
                "charttime_presence": json.dumps(
                    charttime_presence, ensure_ascii=False
                ),
            }
        )

    fields = pd.DataFrame(field_rows)
    modules = pd.DataFrame(module_rows)
    manifests = pd.DataFrame(manifest_rows)

    availability = (
        audit.groupby(
            ["module", "variable", "description", "unit", "plot_kind"],
            dropna=False,
        )
        .agg(
            databases_available=("available", "sum"),
            total_non_null=("non_null_or_finite", "sum"),
            total_rows=("row_count", "sum"),
        )
        .reset_index()
    )
    missing_map = (
        audit.loc[~audit["available"]]
        .groupby(["module", "variable"])["database"]
        .apply(lambda x: ",".join(sorted(x)))
        .rename("unavailable_databases")
        .reset_index()
    )
    availability = availability.merge(
        missing_map, on=["module", "variable"], how="left"
    )
    availability["available_all_six"] = availability["databases_available"] == 6

    fields.to_csv(args.output_dir / "field_contract_audit.csv", index=False)
    modules.to_csv(args.output_dir / "module_schema_summary.csv", index=False)
    manifests.to_csv(args.output_dir / "manifest_audit.csv", index=False)
    availability.to_csv(args.output_dir / "concept_availability.csv", index=False)
    findings = _resolved_findings()
    findings.to_csv(args.output_dir / "verified_issue_register.csv", index=False)
    distribution_flags = _adjudicate_distribution_flags(
        _distribution_flags(audit),
        source_run_id=lineage["source_run_id"],
        source_run_metadata_sha256=lineage["source_run_metadata_sha256"],
    )
    distribution_flags.to_csv(
        args.output_dir / "distribution_anomaly_flags.csv",
        index=False,
    )

    metadata_coverage = _manifest_metadata_coverage(manifests)

    summary = {
        "run_id": run_metadata.get("run_id"),
        **lineage,
        "audited_easyicu_commit": run_metadata.get("easyicu_commit"),
        "audited_easyicu_commits": {
            database: _expected_runtime_commit(run_metadata, database)
            for database in DATABASES
        },
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_sha256_verified_rows": len(source_manifest_sha256),
        "databases": list(DATABASES),
        "module_count": len(module_concepts),
        "concept_panel_count": int(len(availability)),
        "database_concept_panel_count": int(len(audit)),
        "modules_same_concept_set": int(modules["same_concept_set"].sum()),
        "modules_same_concept_order": int(modules["same_concept_order"].sum()),
        "modules_without_type_mismatch": int(
            (modules["type_mismatch_field_count"] == 0).sum()
        ),
        "modules_same_full_physical_schema": int(
            modules["same_full_physical_schema"].sum()
        ),
        "modules_with_canonical_stay_id": int(
            modules["canonical_stay_id_all_six"].sum()
        ),
        "modules_with_canonical_charttime": int(
            modules["canonical_charttime_all_six"].sum()
        ),
        "missing_schema_slots": int(modules["missing_schema_slots"].sum()),
        "concepts_available_all_six": int(availability["available_all_six"].sum()),
        "concepts_unavailable_all_six": int(
            (availability["databases_available"] == 0).sum()
        ),
        **metadata_coverage,
        "manifest_rows_with_merge_keys": int((manifests["merge_key_count"] > 0).sum()),
        "manifest_saved_paths_currently_resolve": int(
            manifests["saved_path_exists"].sum()
        ),
        "manifest_row_count": int(len(manifests)),
        "native_v2_manifest_rows": int(manifests["native_v2"].sum()),
        "manifest_schema_matches_parquet_rows": int(
            manifests["manifest_schema_matches_parquet"].sum()
        ),
        "sidecar_sha256_verified_rows": int(manifests["sidecar_sha256_matches"].sum()),
        "runtime_commits": sorted(
            {str(value) for value in manifests["runtime_commit"].dropna() if str(value)}
        ),
        "runtime_commit_matches_run_rows": int(
            manifests["runtime_commit_matches_run"].sum()
        ),
        "runtime_dirty_rows": int(
            manifests["runtime_git_dirty"]
            .map(lambda value: True if pd.isna(value) else bool(value))
            .sum()
        ),
        "distribution_anomaly_flag_count": int(len(distribution_flags)),
        "distribution_high_flag_count": int(
            (distribution_flags["severity"] == "high").sum()
        ),
        "distribution_adjudicated_flag_count": int(
            (distribution_flags["adjudication_status"] == "source_trace_complete").sum()
        ),
        "distribution_unadjudicated_flag_count": int(
            (distribution_flags["adjudication_status"] == "unadjudicated").sum()
        ),
        "verified_findings_by_severity": (
            findings["severity"].value_counts().to_dict()
        ),
    }
    _json_dump(args.output_dir / "audit_summary.json", summary)
    _json_dump(
        args.output_dir / "report_artifact.json",
        _build_report_artifact(
            output_dir=args.output_dir,
            summary=summary,
            modules=modules,
            availability=availability,
            findings=findings,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _raise_for_metadata_gaps(manifests)


if __name__ == "__main__":
    main()
