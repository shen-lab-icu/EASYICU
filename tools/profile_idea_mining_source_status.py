#!/usr/bin/env python3
"""Profile a data-first Idea Mining measurement audit on existing full6 data.

This command is read-only with respect to the prepared data.  It does not run
the six-database extractor.  The current built-in audit recomputes
albumin-corrected calcium from total calcium and albumin, compares it with the
materialized prepared column, and reports ionized-calcium availability as a
separate measured alternative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from easyicu.research_agent.discovery.idea_mining_source_status import (  # noqa: E402
    ComparisonSourceSpec,
    CrossDatabaseDerivedConceptProfile,
    MeasurementAuditCriteria,
    RowwiseDerivedConceptSpec,
    profile_rowwise_derived_concept,
)

DEFAULT_FULL6_ROOT = Path("/Volumes/外置硬盘/easyicu_data/full6_20260717")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "research_output" / "experiments" / "FIG5-DISC-010" / "source_status"
)
DEFAULT_DATABASES = ("aumc", "mimic", "eicu", "miiv", "sic", "hirid")


def _corrected_calcium_formula(
    columns: dict[str, pa.ChunkedArray],
) -> pa.Array | pa.ChunkedArray:
    calcium = pc.cast(columns["ca"], pa.float64(), safe=False)
    albumin = pc.cast(columns["alb"], pa.float64(), safe=False)
    return pc.add(calcium, pc.multiply(0.8, pc.subtract(4.0, albumin)))


def _corrected_calcium_spec() -> RowwiseDerivedConceptSpec:
    return RowwiseDerivedConceptSpec(
        concept_name="corrected_calcium",
        source_table="chemistry",
        component_columns=("ca", "alb"),
        formula_id="corrected_calcium_mgdl_v1",
        valid_range=(4.0, 16.0),
        materialized_column="corrected_calcium",
        comparison_source=ComparisonSourceSpec(
            table="blood_gas",
            column="cai",
            valid_range=(0.1, 5.0),
        ),
        formula_tolerance=1e-5,
        material_difference_threshold=0.1,
    )


def _markdown(report: CrossDatabaseDerivedConceptProfile) -> str:
    def count_rate(count: int, denominator: int) -> str:
        fraction = count / denominator if denominator else 0.0
        return f"{count:,} ({fraction:.1%})"

    lines = [
        "# Corrected-calcium source-status audit",
        "",
        "> Provider-free audit over existing prepared full6 data; no extraction and "
        "no scientific-analysis authorization.",
        "",
        f"- Prepared-data root: `{report.export_root}`",
        f"- Formula: `{report.concept_spec.formula_id}`",
        f"- Databases ready: {report.n_databases_ready}/{report.n_databases_profiled}",
        (
            "- Measurement-audit answerability: `"
            + (
                report.measurement_audit_answerability.status
                if report.measurement_audit_answerability
                else "not_evaluated"
            )
            + "`"
        ),
        f"- Analysis authorized: `{str(report.analysis_authorized).lower()}`",
        f"- Paper authorized: `{str(report.paper_authorized).lower()}`",
        "",
        "| Database | Denominator stays | Ca observed | Albumin observed | "
        "Same-row components | Valid recomputation | Materialized observed | "
        "Ionized Ca observed | Material formula differences (>0.1 mg/dL) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.databases:
        components = {item.column: item for item in row.component_coverage}
        materialized = row.materialized_coverage
        comparison = row.comparison_coverage
        agreement = row.formula_agreement
        lines.append(
            "| "
            + " | ".join(
                [
                    row.database,
                    f"{row.denominator_stays:,}",
                    count_rate(components["ca"].observed_stays, row.denominator_stays),
                    count_rate(components["alb"].observed_stays, row.denominator_stays),
                    count_rate(row.exact_component_stays, row.denominator_stays),
                    count_rate(row.recomputed_valid_stays, row.denominator_stays),
                    (
                        count_rate(materialized.observed_stays, row.denominator_stays)
                        if materialized
                        else "NA"
                    ),
                    (
                        count_rate(comparison.observed_stays, row.denominator_stays)
                        if comparison
                        else "NA"
                    ),
                    (
                        count_rate(
                            agreement.material_difference_rows,
                            agreement.comparable_rows,
                        )
                        if agreement
                        else "NA"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report distinguishes structural source absence from unmeasured or "
            "out-of-range observations. It does not test an association, establish "
            "novelty, or authorize a manuscript claim. Any downstream Idea Mining "
            "candidate still requires literature review, temporal protocol, and "
            "human confirmation.",
            "",
            "Every input parquet is bound by byte SHA-256 and schema SHA-256 in the "
            "JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_FULL6_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--databases",
        default=",".join(DEFAULT_DATABASES),
        help="Comma-separated canonical database keys.",
    )
    args = parser.parse_args(argv)
    databases = tuple(
        item.strip() for item in str(args.databases).split(",") if item.strip()
    )
    report = profile_rowwise_derived_concept(
        args.export_root,
        databases=databases,
        spec=_corrected_calcium_spec(),
        formula=_corrected_calcium_formula,
        measurement_audit_criteria=MeasurementAuditCriteria(
            min_databases_with_valid_observations=3,
            min_valid_stays_per_database=500,
            min_cross_database_coverage_range=0.20,
        ),
    )
    out_dir = args.out_dir.expanduser().resolve()
    json_path = out_dir / "corrected_calcium_source_status.json"
    markdown_path = out_dir / "corrected_calcium_source_status.md"
    _atomic_write_text(
        json_path,
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(markdown_path, _markdown(report))
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
