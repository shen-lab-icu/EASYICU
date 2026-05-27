"""Writer-side scalar / digest helpers extracted from pipeline.py.

These functions pull machine-readable scalars out of step_summary payloads
and CSV artefacts so the writer prompt receives a compact, deterministic
evidence digest instead of the raw run directory. Moved out of pipeline.py
on 2026-05-27 as part of the pipeline.py size-reduction effort; the
behaviour is unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .context import ResearchContext
from .evidence import EvidenceStore
from .scalar_utils import _first_present_scalar


__all__ = [
    "_resolve_writer_aux_path",
    "_summarise_table_one_rows",
    "_summarise_primary_association_table",
    "_summarise_sofa_zero_audit",
    "_preferred_writer_evidence_names",
    "_render_writer_evidence_digest",
]


def _resolve_writer_aux_path(
    *,
    run_dir: Path,
    step_id: str,
    candidate: Optional[Any],
) -> Optional[Path]:
    if not candidate:
        return None
    raw = Path(str(candidate))
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        run_dir / "steps" / step_id / "outputs" / raw.name,
        run_dir / "steps" / step_id / "outputs" / str(raw),
        run_dir / str(raw),
    ]
    return next((path for path in candidates if path.exists()), None)


def _summarise_table_one_rows(rows: Any) -> Dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    wanted = {
        "age": "age",
        "sofa2": "sofa2",
        "lact": "lact",
        "creat": "creat",
        "map": "map",
        "los_icu": "los_icu",
    }
    summary: Dict[str, Any] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        variable = str(item.get("variable") or "").strip().lower()
        if variable not in wanted:
            continue
        prefix = wanted[variable]
        for source_key, target_key in (
            ("n", f"{prefix}_n"),
            ("median", f"{prefix}_median"),
            ("q25", f"{prefix}_q25"),
            ("q75", f"{prefix}_q75"),
            ("most_common", f"{prefix}_most_common"),
            ("most_common_n", f"{prefix}_most_common_n"),
        ):
            scalar = _first_present_scalar(item, (source_key,))
            if scalar is not None:
                summary[target_key] = scalar
    return summary


def _summarise_primary_association_table(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    cols = {str(c).lower(): c for c in frame.columns}
    variable_col = cols.get("variable") or cols.get("term")
    odds_col = cols.get("odds_ratio") or cols.get("or")
    lower_col = cols.get("or_lower") or cols.get("ci_lower") or cols.get("lower")
    upper_col = cols.get("or_upper") or cols.get("ci_upper") or cols.get("upper")
    p_col = cols.get("p_value") or cols.get("p")
    if variable_col is None:
        return {}
    digest: Dict[str, Any] = {}
    for _, row in frame.iterrows():
        variable = str(row.get(variable_col) or "").strip()
        if not variable or variable.lower() == "intercept":
            continue
        key = variable.replace(" ", "_")
        if odds_col is not None:
            val = _first_present_scalar(row, (odds_col,))
            if val is not None:
                digest[f"{key}_or"] = val
        if lower_col is not None:
            val = _first_present_scalar(row, (lower_col,))
            if val is not None:
                digest[f"{key}_ci_low"] = val
        if upper_col is not None:
            val = _first_present_scalar(row, (upper_col,))
            if val is not None:
                digest[f"{key}_ci_high"] = val
        if p_col is not None:
            val = _first_present_scalar(row, (p_col,))
            if val is not None:
                digest[f"{key}_p_value"] = val
    return digest


def _summarise_sofa_zero_audit(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    cols = {str(c).lower(): c for c in frame.columns}
    sofa_col = cols.get("sofa2") or cols.get("score") or cols.get("stratum")
    rate_col = cols.get("death_rate") or cols.get("outcome_rate") or cols.get("mortality_rate")
    if sofa_col is None or rate_col is None:
        return {}
    digest: Dict[str, Any] = {}
    for level in (0, 1):
        try:
            row = frame.loc[pd.to_numeric(frame[sofa_col], errors="coerce") == level]
        except Exception:
            row = pd.DataFrame()
        if row.empty:
            continue
        value = _first_present_scalar(row.iloc[0], (rate_col,))
        if value is not None:
            digest[f"sofa2_{level}_death_rate"] = value
    return digest


def _preferred_writer_evidence_names(evidence: EvidenceStore) -> List[str]:
    aliases = evidence.aliases()
    preferred = [
        "table_one",
        "cohort_summary",
        "outcome_incidence",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
        "sofa_strata",
        "stratum_audit",
        "multiple_testing_report",
        "fairness_subgroups",
        "literature_prisma",
        "causal_audit_report",
        "causal_audit_summary",
        "reporting_checklist",
    ]
    out: List[str] = [name for name in preferred if name in aliases or evidence.get(name) is not None]
    step_aliases = [
        name for name in sorted(aliases)
        if re.match(r"^\d{2}[_-]", name)
    ]
    for name in step_aliases:
        if name not in out:
            out.append(name)
    return out or evidence.resolvable_names()


def _render_writer_evidence_digest(
    per_step_records: Sequence[Dict[str, Any]] | None = None,
    *,
    context: ResearchContext | None = None,
    run_dir: Path | None = None,
) -> str:
    lines: List[str] = []
    if context is not None:
        lines.append("RUN_CONTEXT")
        lines.append(
            "  "
            + json.dumps(
                {
                    "research_question": context.research_question,
                    "cohort_name": context.cohort.cohort_name,
                    "database": context.cohort.database,
                    "n_stays": context.cohort.n_stays,
                    "n_patients": context.cohort.n_patients,
                    "target_outcome": context.target_outcome,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    run_dir = Path(run_dir or ".")
    preferred_keys = (
        "sample_size",
        "n_total",
        "n_total_stays",
        "n_death",
        "n_complete",
        "n_complete_case",
        "complete_case_n",
        "outcome_rate",
        "overall_mortality_rate",
        "overall_ci_low",
        "overall_ci_high",
        "mortality_rate",
        "median_age",
        "estimate",
        "primary_or",
        "odds_ratio",
        "adjusted_or",
        "ci_lower",
        "ci_upper",
        "primary_ci_low",
        "primary_ci_high",
        "primary_or_ci",
        "p_value",
        "auroc",
        "statistic:auroc",
        "auc",
        "statistic:auc",
        "cv_auroc",
        "statistic:cv_auroc",
        "held_out_auroc",
        "statistic:held_out_auroc",
        "mean_auroc",
        "statistic:mean_auroc",
        "auroc_median",
        "statistic:auroc_ci_lower",
        "statistic:auroc_ci_upper",
        "brier_score",
        "statistic:brier_score",
        "held_out_brier",
        "statistic:held_out_brier",
        "brier_median",
        "calibration_slope",
        "statistic:calibration_slope",
        "calibration_slope_median",
        "calibration_intercept",
        "statistic:calibration_intercept",
        "calibration_intercept_median",
        "baseline_prevalence",
        "statistic:baseline_prevalence",
        "split_strategy",
        "statistic:split_strategy",
        "silhouette_score",
        "silhouette",
        "n_clusters",
        "cluster_count",
        "spearman_rho",
        "rho",
        "skipped",
        "error",
    )
    for record in per_step_records:
        step_id = str(record.get("step_id") or "unknown_step")
        status = str(record.get("status") or "unknown")
        lines.append(f"- {step_id} [{status}]")
        summary = record.get("step_summary")
        if not isinstance(summary, dict) or not summary:
            lines.append("  {}")
            continue
        digest_row: Dict[str, Any] = {}
        for key in preferred_keys:
            scalar = _first_present_scalar(summary, (key,))
            if scalar is not None:
                digest_row[key] = scalar
        if "primary_predictor" in summary:
            digest_row["primary_predictor"] = str(summary["primary_predictor"])
        elif "predictor" in summary:
            digest_row["primary_predictor"] = str(summary["predictor"])
        if "target_outcome" in summary:
            digest_row["target_outcome"] = str(summary["target_outcome"])
        elif "outcome" in summary:
            digest_row["target_outcome"] = str(summary["outcome"])
        if "primary_or_ci" in summary and isinstance(summary["primary_or_ci"], (list, tuple)):
            ci_values = list(summary["primary_or_ci"])
            if len(ci_values) == 2:
                digest_row.setdefault("primary_ci_low", ci_values[0])
                digest_row.setdefault("primary_ci_high", ci_values[1])
        digest_row.update(_summarise_table_one_rows(summary.get("table_one_rows")))
        primary_path = _resolve_writer_aux_path(
            run_dir=run_dir,
            step_id=step_id,
            candidate=summary.get("primary_association_path"),
        )
        digest_row.update(_summarise_primary_association_table(primary_path))
        strata_path = _resolve_writer_aux_path(
            run_dir=run_dir,
            step_id=step_id,
            candidate=summary.get("table") if "sofa_zero_audit" in step_id.lower() else None,
        )
        if strata_path is None and "sofa_zero_audit" in step_id.lower():
            strata_path = run_dir / "steps" / step_id / "outputs" / "sofa_strata.csv"
        digest_row.update(_summarise_sofa_zero_audit(strata_path))
        lines.append(
            "  " + json.dumps(digest_row, ensure_ascii=False, sort_keys=True, default=str)
        )
    return "\n".join(lines)
