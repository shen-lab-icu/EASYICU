#!/usr/bin/env python3
"""Build v14 task cohorts from a real EasyICU concept export.

This script consumes an EasyICU concept export package (one parquet
per concept module, with the manifest already on disk) and produces
ten task-specific cohort parquet files plus one shared master cohort
parquet that the v14 benchmark items consume.

Design goals
------------

1. **Real, not synthetic.** Every cohort comes from the EasyICU export
   the user is studying (e.g. ``/Users/haibo/Documents/GitHub/miiv_20260420/``).
2. **Reproducible.** Aggregation windows, filters, and column names
   are written in this file so the cohorts are a function of (export
   directory, code version) only.
3. **Task-fit.** Each task gets exactly the columns it needs. A
   clustering task does not see the outcome label; a prediction task
   gets train/eval-friendly numeric columns; a sensitivity task keeps
   the missingness structure of the source data.

Usage::

    python tools/build_v14_task_cohorts.py \
        --export-dir /path/to/miiv_20260420 \
        --out-dir    research_output/v14_task_cohorts_20260508

Outputs::

    research_output/v14_task_cohorts_20260508/
        v14_master_cohort.parquet            # all variables, all stays
        v14_master_cohort.csv                # readable copy
        v14_master_cohort_manifest.json      # builder + concept sources
        t01_table_one_descriptive.parquet
        t02_outcome_incidence_strata.parquet
        ...
        t10_complete_case_robustness.parquet
        v14_task_cohorts_summary.json        # n_rows + columns per task

The script only reads the export and writes the output directory; it
never modifies the source export.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Bootstrap so this script works whether or not easyicu is pip-installed.
# ---------------------------------------------------------------------------


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    for candidate in (str(src_path), str(repo_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_bootstrap_imports()

import pandas as pd  # noqa: E402  (after path bootstrap)

from easyicu.research_agent.easyicu_case_builder import (  # noqa: E402
    ID_COL,
    TIME_COL,
    _aggregate_circ,
    _aggregate_lactate,
    _aggregate_map,
    _aggregate_sep3,
    _aggregate_vaso,
    _merge_left,
    _window,
    index_export_package,
    read_exported_concept,
)


# ---------------------------------------------------------------------------
# Master cohort builder.
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    key: str
    description: str
    columns: List[str]
    filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


def _aggregate_hr_sbp(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    keep = [c for c in ["hr", "sbp", "dbp", "resp", "temp", "spo2"] if c in work.columns]
    for col in keep:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    agg_dict = {}
    for col in keep:
        agg_dict[f"{col}_min_24h"] = (col, "min")
        agg_dict[f"{col}_max_24h"] = (col, "max")
        agg_dict[f"{col}_median_24h"] = (col, "median")
    if not agg_dict:
        return pd.DataFrame(columns=[ID_COL])
    return grouped.agg(**agg_dict).reset_index()


def _aggregate_sofa2(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    cols = [c for c in [
        "sofa2", "sofa2_resp", "sofa2_cardio", "sofa2_cns",
        "sofa2_renal", "sofa2_coag", "sofa2_liver",
    ] if c in work.columns]
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    spec = {f"{c}_max_24h": (c, "max") for c in cols}
    if not spec:
        return pd.DataFrame(columns=[ID_COL])
    return grouped.agg(**spec).reset_index()


def _aggregate_kdigo(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty or "aki_stage" not in work.columns:
        return pd.DataFrame(columns=[ID_COL])
    work["aki_stage"] = pd.to_numeric(work["aki_stage"], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    out = grouped.agg(
        kdigo_stage_max_24h=("aki_stage", "max"),
        kdigo_stage_first_24h=("aki_stage", "first"),
    ).reset_index()
    return out


def _aggregate_creat(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty or "crea" not in work.columns:
        return pd.DataFrame(columns=[ID_COL])
    work["crea"] = pd.to_numeric(work["crea"], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    return grouped.agg(
        creat_max_24h=("crea", "max"),
        creat_min_24h=("crea", "min"),
        creat_median_24h=("crea", "median"),
    ).reset_index()


def _aggregate_bili(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty or "bili" not in work.columns:
        return pd.DataFrame(columns=[ID_COL])
    work["bili"] = pd.to_numeric(work["bili"], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    return grouped.agg(
        bili_max_24h=("bili", "max"),
        bili_median_24h=("bili", "median"),
        bili_n_24h=("bili", "count"),
    ).reset_index()


def _aggregate_gcs(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty or "gcs" not in work.columns:
        return pd.DataFrame(columns=[ID_COL])
    work["gcs"] = pd.to_numeric(work["gcs"], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    return grouped.agg(
        gcs_min_24h=("gcs", "min"),
        gcs_median_24h=("gcs", "median"),
        gcs_n_24h=("gcs", "count"),
    ).reset_index()


def _aggregate_plt(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = _window(df, start, end)
    if work.empty or "plt" not in work.columns:
        return pd.DataFrame(columns=[ID_COL])
    work["plt"] = pd.to_numeric(work["plt"], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    return grouped.agg(
        plt_min_24h=("plt", "min"),
        plt_median_24h=("plt", "median"),
    ).reset_index()


def build_master_cohort(export_dir: Path, *, window: Tuple[float, float] = (0.0, 24.0)) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Build a one-row-per-stay master cohort with all v14 task variables."""
    start, end = window
    index = index_export_package(export_dir)

    required = ["age", "sex", "death"]
    missing = [c for c in required if c not in index]
    if missing:
        raise KeyError(f"EasyICU export is missing required concepts: {missing}")

    demo_extra = [c for c in ["sex", "adm", "bmi", "weight", "height"] if c in index]
    demo = read_exported_concept(export_dir, "age", extra_columns=demo_extra)
    base_cols = [c for c in [ID_COL, "age", *demo_extra] if c in demo.columns]
    base = demo[base_cols].drop_duplicates(subset=[ID_COL]).copy()

    outcome = read_exported_concept(export_dir, "death", extra_columns=[c for c in ["los_icu", "los_hosp"] if c in index])
    out_cols = [c for c in [ID_COL, "death", "los_icu", "los_hosp"] if c in outcome.columns]
    outcome_df = outcome[out_cols].drop_duplicates(subset=[ID_COL]).copy()
    if "death" in outcome_df.columns:
        outcome_df["death"] = outcome_df["death"].fillna(False).astype(bool).astype(int)

    aggregates: List[pd.DataFrame] = [outcome_df]
    sources: Dict[str, str] = {
        "age": index["age"]["file_name"],
        "sex": index.get("sex", {}).get("file_name", ""),
        "death": index["death"]["file_name"],
    }
    if "los_icu" in index:
        sources["los_icu"] = index["los_icu"]["file_name"]

    if "lact" in index:
        lact = read_exported_concept(export_dir, "lact")
        aggregates.append(_aggregate_lactate(lact, start, end))
        sources["lactate_*_24h"] = index["lact"]["file_name"]

    if "map" in index:
        vitals_map = read_exported_concept(export_dir, "map")
        aggregates.append(_aggregate_map(vitals_map, start, end))
        sources["map_*_24h"] = index["map"]["file_name"]

    if "vaso_ind" in index:
        vaso_extra = ["norepi_equiv"] if "norepi_equiv" in index else None
        vaso = read_exported_concept(export_dir, "vaso_ind", extra_columns=vaso_extra)
        aggregates.append(_aggregate_vaso(vaso, start, end))
        sources["vaso_*_24h"] = index["vaso_ind"]["file_name"]

    if "circ_failure" in index:
        circ = read_exported_concept(export_dir, "circ_failure", extra_columns=["circ_event"] if "circ_event" in index else None)
        aggregates.append(_aggregate_circ(circ, start, end))
        sources["circ_*_24h"] = index["circ_failure"]["file_name"]

    if "sep3_sofa2" in index:
        sep3 = read_exported_concept(export_dir, "sep3_sofa2")
        aggregates.append(_aggregate_sep3(sep3, start, end))
        sources["sep3_sofa2_any_24h"] = index["sep3_sofa2"]["file_name"]

    if "hr" in index:
        vitals_hr = read_exported_concept(export_dir, "hr", extra_columns=[c for c in ["sbp", "dbp", "resp", "temp", "spo2"] if c in index])
        aggregates.append(_aggregate_hr_sbp(vitals_hr, start, end))
        sources["vitals_*_24h"] = index["hr"]["file_name"]

    if "sofa2" in index:
        sofa2 = read_exported_concept(export_dir, "sofa2", extra_columns=[c for c in [
            "sofa2_resp", "sofa2_cardio", "sofa2_cns", "sofa2_renal", "sofa2_coag", "sofa2_liver",
        ] if c in index])
        aggregates.append(_aggregate_sofa2(sofa2, start, end))
        sources["sofa2_*_24h"] = index["sofa2"]["file_name"]

    if "aki_stage" in index:
        aki = read_exported_concept(export_dir, "aki_stage")
        aggregates.append(_aggregate_kdigo(aki, start, end))
        sources["kdigo_stage_*_24h"] = index["aki_stage"]["file_name"]

    if "crea" in index:
        creat = read_exported_concept(export_dir, "crea")
        aggregates.append(_aggregate_creat(creat, start, end))
        sources["creat_*_24h"] = index["crea"]["file_name"]

    if "bili" in index:
        bili = read_exported_concept(export_dir, "bili")
        aggregates.append(_aggregate_bili(bili, start, end))
        sources["bili_*_24h"] = index["bili"]["file_name"]

    if "gcs" in index:
        gcs = read_exported_concept(export_dir, "gcs")
        aggregates.append(_aggregate_gcs(gcs, start, end))
        sources["gcs_*_24h"] = index["gcs"]["file_name"]

    if "plt" in index:
        plt_df = read_exported_concept(export_dir, "plt")
        aggregates.append(_aggregate_plt(plt_df, start, end))
        sources["plt_*_24h"] = index["plt"]["file_name"]

    cohort = _merge_left(base, aggregates)

    # Default-zero columns where downstream tasks expect a 0/1 indicator.
    default_zero = {
        "vaso_any_24h": 0,
        "vaso_hours_24h": 0,
        "circ_failure_any_24h": 0,
        "sep3_sofa2_any_24h": 0,
        "lactate_measured_24h": 0,
    }
    for col, default in default_zero.items():
        if col in cohort.columns:
            cohort[col] = cohort[col].fillna(default)

    # Adult first-stay style filter; the export already reflects "first ICU stays".
    if "age" in cohort.columns:
        cohort = cohort[pd.to_numeric(cohort["age"], errors="coerce") >= 18].copy()

    return cohort, sources


# ---------------------------------------------------------------------------
# Task definitions.
# ---------------------------------------------------------------------------


def _filter_los_24h(df: pd.DataFrame) -> pd.DataFrame:
    if "los_icu" not in df.columns:
        return df
    return df[pd.to_numeric(df["los_icu"], errors="coerce").fillna(0) >= 1.0].copy()


def _filter_lactate_measured(df: pd.DataFrame) -> pd.DataFrame:
    if "lactate_measured_24h" not in df.columns:
        return df
    return df[df["lactate_measured_24h"] == 1].copy()


def _filter_has_sofa2(df: pd.DataFrame) -> pd.DataFrame:
    if "sofa2_max_24h" not in df.columns:
        return df
    return df[df["sofa2_max_24h"].notna()].copy()


def _filter_vaso_eligible(df: pd.DataFrame) -> pd.DataFrame:
    if "los_icu" not in df.columns:
        return df
    return df[pd.to_numeric(df["los_icu"], errors="coerce").fillna(0) >= 0.5].copy()


def task_specs() -> List[TaskSpec]:
    return [
        TaskSpec(
            key="t01_table_one_descriptive",
            description="Descriptive Table 1 of demographics + early severity in adult first ICU stays.",
            columns=[
                ID_COL, "age", "sex", "los_icu",
                "sofa2_max_24h", "sofa2_resp_max_24h", "sofa2_cardio_max_24h",
                "map_min_24h", "lactate_max_24h", "vaso_any_24h", "death",
            ],
            filter_fn=None,
        ),
        TaskSpec(
            key="t02_outcome_incidence_strata",
            description="In-hospital mortality across SOFA-2 severity strata (descriptive, no model).",
            columns=[ID_COL, "age", "sex", "sofa2_max_24h", "death"],
            filter_fn=_filter_has_sofa2,
        ),
        TaskSpec(
            key="t03_severity_score_correlation",
            description="Correlation between SOFA-2 components and total score; identify component-total collinearity.",
            columns=[
                ID_COL, "age", "sex",
                "sofa2_max_24h", "sofa2_resp_max_24h", "sofa2_cardio_max_24h",
                "sofa2_cns_max_24h", "sofa2_renal_max_24h",
                "sofa2_coag_max_24h", "sofa2_liver_max_24h",
            ],
            filter_fn=_filter_has_sofa2,
        ),
        TaskSpec(
            key="t04_lactate_mortality_association",
            description="Early lactate -> mortality with explicit missing-indicator and adjusted OR.",
            columns=[
                ID_COL, "age", "sex", "los_icu",
                "lactate_max_24h", "lactate_measured_24h",
                "map_min_24h", "vaso_any_24h", "death",
            ],
            filter_fn=_filter_los_24h,
        ),
        TaskSpec(
            key="t05_kdigo_renal_sensitivity",
            description="KDIGO stage -> mortality, complete-case vs reduced-variable sensitivity.",
            columns=[
                ID_COL, "age", "sex",
                "kdigo_stage_max_24h", "creat_max_24h", "creat_median_24h",
                "sofa2_renal_max_24h", "vaso_any_24h", "death",
            ],
            filter_fn=None,
        ),
        TaskSpec(
            key="t06_shock_phenotype_clustering",
            description="Unsupervised phenotype clustering on shock physiology (lactate/MAP/vaso/HR/SBP).",
            columns=[
                ID_COL, "age", "sex",
                "lactate_max_24h", "map_min_24h", "vaso_any_24h",
                "hr_max_24h", "hr_median_24h",
                "sbp_min_24h", "sbp_median_24h",
                "death",  # held out for post-hoc cluster mortality summary
            ],
            filter_fn=_filter_los_24h,
        ),
        TaskSpec(
            key="t07_mortality_prediction_auroc",
            description="Multivariable logistic / RF prediction of in-hospital mortality with 5-fold CV AUROC + calibration.",
            columns=[
                ID_COL, "age", "sex",
                "sofa2_max_24h", "sofa2_resp_max_24h", "sofa2_cardio_max_24h",
                "sofa2_cns_max_24h", "sofa2_renal_max_24h",
                "lactate_max_24h", "map_min_24h", "vaso_any_24h",
                "death",
            ],
            filter_fn=_filter_has_sofa2,
        ),
        TaskSpec(
            key="t08_vaso_selection_bias_audit",
            description="Vasopressor exposure -> mortality with explicit selection-bias and missingness audit; no causal language.",
            columns=[
                ID_COL, "age", "sex", "los_icu",
                "vaso_any_24h", "norepi_equiv_max_24h",
                "lactate_max_24h", "map_min_24h",
                "sofa2_cardio_max_24h", "death",
            ],
            filter_fn=_filter_vaso_eligible,
        ),
        TaskSpec(
            key="t09_sofa_zero_artefact_audit",
            description="Detect anomalous SOFA-2==0 stays with high lactate or vasopressor exposure; data-quality flag.",
            columns=[
                ID_COL, "age", "sex",
                "sofa2_max_24h", "sofa2_resp_max_24h", "sofa2_cardio_max_24h",
                "sofa2_cns_max_24h", "sofa2_renal_max_24h",
                "lactate_max_24h", "map_min_24h", "vaso_any_24h", "death",
            ],
            filter_fn=None,
        ),
        TaskSpec(
            key="t10_complete_case_robustness",
            description="Mortality model robustness across complete-case, missing-indicator, and reduced-variable strategies.",
            columns=[
                ID_COL, "age", "sex", "los_icu",
                "sofa2_max_24h", "lactate_max_24h", "lactate_measured_24h",
                "map_min_24h", "vaso_any_24h",
                "bili_max_24h", "bili_n_24h",
                "creat_max_24h", "death",
            ],
            filter_fn=_filter_los_24h,
        ),
        # ------------------------------------------------------------------
        # v15 ladder extension: 5 additional tasks (t11-t15) covering basic
        # descriptive ground (t11-t13) and intermediate clinical-association
        # studies (t14-t15) so the benchmark spans 5 basic + 5 intermediate +
        # 5 advanced tasks. All columns reuse the v14 master cohort and do
        # not require new EasyICU concept dictionaries or master rebuilds.
        # ------------------------------------------------------------------
        TaskSpec(
            key="t11_los_distribution_descriptive",
            description="ICU and hospital length-of-stay distribution by survival status; basic descriptive.",
            columns=[
                ID_COL, "age", "sex", "los_icu", "los_hosp", "death",
            ],
            filter_fn=None,
        ),
        TaskSpec(
            key="t12_age_stratified_mortality",
            description="Age-tertile stratified in-hospital mortality with proportions and 95% CI; basic descriptive.",
            columns=[
                ID_COL, "age", "sex", "los_icu", "death",
            ],
            filter_fn=None,
        ),
        TaskSpec(
            key="t13_admission_vital_summary",
            description="First-24-hour vital-sign distribution (HR, SBP, MAP) by survival status; basic descriptive.",
            columns=[
                ID_COL, "age", "sex",
                "hr_max_24h", "hr_median_24h",
                "sbp_min_24h", "sbp_median_24h",
                "map_min_24h", "map_median_24h",
                "death",
            ],
            filter_fn=_filter_los_24h,
        ),
        TaskSpec(
            key="t14_creatinine_trajectory_kdigo",
            description="Within-24-hour creatinine trajectory (max/median ratio) and KDIGO stage; intermediate association.",
            columns=[
                ID_COL, "age", "sex",
                "creat_max_24h", "creat_median_24h",
                "kdigo_stage_max_24h", "sofa2_renal_max_24h",
                "vaso_any_24h", "death",
            ],
            filter_fn=None,
        ),
        TaskSpec(
            key="t15_norepinephrine_dose_response",
            description="Norepinephrine-equivalent dose quartile -> mortality among vasopressor-exposed stays; intermediate dose-response.",
            columns=[
                ID_COL, "age", "sex",
                "vaso_any_24h", "norepi_equiv_max_24h",
                "map_min_24h", "lactate_max_24h",
                "sofa2_cardio_max_24h", "death",
            ],
            filter_fn=_filter_vaso_eligible,
        ),
    ]


def derive_task_cohort(master: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    available = [c for c in spec.columns if c in master.columns]
    df = master[available].copy()
    if spec.filter_fn is not None:
        df = spec.filter_fn(df)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        required=True,
        help="EasyICU concept export directory (parquet files + manifest).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for v14 task cohorts.",
    )
    parser.add_argument(
        "--window-start", type=float, default=0.0,
        help="Aggregation window start hour (default: 0).",
    )
    parser.add_argument(
        "--window-end", type=float, default=24.0,
        help="Aggregation window end hour (default: 24).",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v14-build] export_dir = {export_dir}")
    print(f"[v14-build] out_dir    = {out_dir}")
    print(f"[v14-build] window     = ({args.window_start}, {args.window_end})")

    master, sources = build_master_cohort(export_dir, window=(args.window_start, args.window_end))
    master_path = out_dir / "v14_master_cohort.parquet"
    master_csv = out_dir / "v14_master_cohort.csv"
    master.to_parquet(master_path, index=False)
    master.to_csv(master_csv, index=False)
    print(f"[v14-build] master cohort: n={len(master)}, cols={len(master.columns)}")

    summary = {
        "builder": "tools/build_v14_task_cohorts.py",
        "export_dir": str(export_dir),
        "window": [args.window_start, args.window_end],
        "master_cohort": {
            "path": str(master_path),
            "n_rows": int(len(master)),
            "columns": list(master.columns),
            "sources": sources,
        },
        "tasks": [],
    }

    for spec in task_specs():
        df = derive_task_cohort(master, spec)
        path = out_dir / f"{spec.key}.parquet"
        df.to_parquet(path, index=False)
        df.to_csv(out_dir / f"{spec.key}.csv", index=False)
        info = {
            "key": spec.key,
            "description": spec.description,
            "path": str(path),
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        }
        summary["tasks"].append(info)
        print(f"[v14-build] {spec.key}: n={len(df)}, cols={len(df.columns)}")

    summary_path = out_dir / "v14_task_cohorts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    master_manifest = out_dir / "v14_master_cohort_manifest.json"
    master_manifest.write_text(
        json.dumps(
            {
                "builder": "tools/build_v14_task_cohorts.py",
                "export_dir": str(export_dir),
                "window": [args.window_start, args.window_end],
                "n_rows": int(len(master)),
                "concept_sources": sources,
                "columns": list(master.columns),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[v14-build] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
