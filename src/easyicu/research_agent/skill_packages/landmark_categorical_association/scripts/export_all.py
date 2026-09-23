"""Step 4 -- export every table, the headline metrics, the report and the manifest.

``export_all`` is the only writer of the deliverable directory.  It copies
numbers from the result tables into ``report.md`` (never recomputing them),
writes ``key_metrics.csv`` as the single-row source every downstream reader
must quote, and runs the export consistency gate before printing its
completion token.  A failed hard check raises; the token is never printed for a
directory whose tables disagree with each other.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ....authority.evidence_store import sha256_of_file
from ..spec import SKILL_VERSION, LandmarkCategoricalSpec
from .generate_all_plots import FigureArtifact
from .run_analysis import PRIMARY_VARIANT_ID, AnalysisResult

EXPORT_TOKEN = "=== Export Complete ==="
KEY_METRICS_FILENAME = "key_metrics.csv"
REPORT_FILENAME = "report.md"
MANIFEST_FILENAME = "manifest.json"


class ExportConsistencyError(RuntimeError):
    """A hard invariant between exported tables failed; nothing is certified."""


@dataclass(frozen=True)
class ExportReceipt:
    out_dir: str
    files: dict[str, dict[str, Any]]
    checks_passed: int
    checks_total: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    manifest_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "files": self.files,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "warnings": list(self.warnings),
            "manifest_sha256": self.manifest_sha256,
        }


# --------------------------------------------------------------------------
# formatting helpers (display only; never feed numbers back into analysis)
# --------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any, digits: int = 2) -> str:
    if _is_missing(value):
        return "not estimable"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if _is_missing(value):
        return "not estimable"
    return f"{100.0 * float(value):.{digits}f}%"


def _p(value: Any) -> str:
    if _is_missing(value):
        return "not estimable"
    number = float(value)
    if number < 1e-4:
        return "<0.0001"
    return f"{number:.4f}"


def _ci(estimate: Any, low: Any, high: Any, digits: int = 2) -> str:
    if _is_missing(estimate) or _is_missing(low) or _is_missing(high):
        return "not estimable"
    return f"{_num(estimate, digits)} (95% CI {_num(low, digits)}–{_num(high, digits)})"


def _int(value: Any) -> str:
    if _is_missing(value):
        return "—"
    return f"{int(value):,}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# caveat sentences: one deterministic sentence per raised flag
# --------------------------------------------------------------------------


def caveat_sentences(spec: LandmarkCategoricalSpec, flags: Mapping[str, Any]) -> list[str]:
    """Mandatory report sentences implied by the raised caveat flags."""

    sentences: list[str] = []
    if flags.get("epv_below_minimum"):
        sentences.append(
            f"The primary model has {_num(flags.get('epv'), 1)} events per parameter, below the "
            f"declared minimum of {_num(flags.get('epv_minimum'), 0)}; the adjusted odds ratios are "
            "potentially overfitted and their intervals should be read as unstable."
        )
    if flags.get("separation_detected"):
        sentences.append(
            "The primary logistic fit showed (quasi-)separation; the affected odds ratio and its "
            "interval are not reliable estimates."
        )
    if flags.get("complete_case_warning"):
        sentences.append(
            f"{_pct(flags.get('complete_case_share_dropped'))} of landmark rows with a known exposure "
            "were excluded from the primary model because of missing covariates (complete-case "
            f"analysis, threshold {_pct(flags.get('complete_case_warning_share'), 0)}); the estimate is "
            "subject to selection bias and no imputation was performed."
        )
    if flags.get("unknown_exposure_share_high"):
        sentences.append(
            f"{_pct(flags.get('unknown_exposure_share'))} of the landmark population had a "
            f"non-evaluable exposure (reported as '{spec.unknown_level_label}', threshold "
            f"{_pct(flags.get('unknown_exposure_warning_share'), 0)}); these rows are described but not "
            "modelled, so the adjusted estimates apply to the evaluable subset only."
        )
    if flags.get("small_reference_group"):
        sentences.append(
            f"The reference level {spec.reference_level} contains only "
            f"{_int(flags.get('reference_group_n'))} rows (below {_int(flags.get('small_reference_group_n'))}); "
            "contrasts against it are unstable."
        )
    sparse = flags.get("sparse_exposure_levels") or []
    if sparse:
        sentences.append(
            f"Exposure level(s) {', '.join(map(str, sparse))} have fewer than "
            f"{_int(spec.sparse_level_events)} events; their odds ratios are imprecise."
        )
    empty = flags.get("empty_exposure_levels") or []
    if empty:
        sentences.append(
            f"Declared exposure level(s) {', '.join(map(str, empty))} were not observed in the landmark "
            "population; the corresponding contrasts are absent, not zero."
        )
    if flags.get("sensitivity_direction_consistent") is False:
        sentences.append(
            "At least one prespecified sensitivity refit reversed the direction of the primary "
            "contrast; see robustness_summary.csv before interpreting the primary estimate."
        )
    failed = flags.get("failed_sensitivity_variants") or []
    if failed:
        sentences.append(
            "Sensitivity refit(s) that could not be estimated and remain in the denominator: "
            + ", ".join(map(str, failed))
            + "."
        )
    not_estimable = flags.get("functional_form_not_estimable") or []
    if not_estimable:
        sentences.append(
            "Functional-form check(s) not estimable: " + ", ".join(map(str, not_estimable)) + "."
        )
    sentences.append(
        f"All results are observational associations in a fixed {spec.landmark_hours:g} h landmark "
        f"population with claim ceiling '{spec.claim_ceiling}': no causal effect, no transportability "
        "beyond the source, and no novelty claim is made."
    )
    return sentences


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def build_report(
    result: AnalysisResult,
    figures: Mapping[str, FigureArtifact] | None,
    *,
    table_files: Mapping[str, str],
) -> str:
    spec = result.spec
    metrics = result.key_metrics
    flags = result.caveat_flags
    cohort = result.cohort
    lines: list[str] = []
    lines.append(f"# {spec.title}")
    lines.append("")
    lines.append(
        f"Skill `{spec.schema_version}` · claim ceiling **{spec.claim_ceiling}** · "
        "every number below is copied from the exported tables listed at the end."
    )
    if cohort.source_sha256:
        lines.append(f"Source cohort SHA-256: `{cohort.source_sha256}`")
    # Always stated: an undeclared source is a visible gap, not a blank line.
    lines.append(f"Data provenance: **{cohort.provenance}**")
    lines.append("")

    lines.append("## Design")
    lines.append("")
    covariates = ", ".join(
        f"{item.name} ({item.coding}" + (f", ref {item.reference_level})" if item.reference_level else ")")
        for item in spec.covariates
    )
    lines.append(
        f"- Exposure: `{spec.exposure}` ({spec.label(spec.exposure)}), declared levels "
        f"{', '.join(spec.exposure_levels)}; reference {spec.reference_level}; primary contrast "
        f"{spec.primary_contrast_level} vs {spec.reference_level}; non-evaluable exposure kept as "
        f"'{spec.unknown_level_label}' and never recoded."
    )
    lines.append(
        f"- Outcome: `{spec.outcome}` ({spec.label(spec.outcome)}), binary; event time "
        f"`{spec.event_time_column}`; observation `{spec.observation_duration_column}` "
        f"({spec.observation_duration_unit})."
    )
    lines.append(
        f"- Time zero: {spec.landmark_hours:g} h landmark; rows must be alive and under observation "
        "at the landmark (host `landmark_eligibility_mask`)."
    )
    lines.append(f"- Adjustment: {covariates}.")
    if spec.dependence is not None:
        lines.append(
            f"- Dependence: {spec.dependence.variance_estimator} covariance by "
            f"{spec.dependence.cluster_unit} (groups from `{spec.dependence.group_source}`)."
        )
    else:
        lines.append("- Dependence: model-based covariance (no cluster contract declared).")
    if spec.alternate_exposures:
        lines.append("- Alternate exposure definitions: " + ", ".join(f"`{c}`" for c in spec.alternate_exposures) + ".")
    if spec.first_stay_column:
        lines.append(f"- Repeated-stay sensitivity: restriction to `{spec.first_stay_column}`.")
    if spec.functional_form_covariates:
        lines.append(
            "- Functional form: restricted cubic splines "
            f"({spec.functional_form_knots} knots) for " + ", ".join(spec.functional_form_covariates) + "."
        )
    lines.append("")

    lines.append("## Cohort flow")
    lines.append("")
    rows = []
    for _, row in result.cohort_flow.iterrows():
        if row["action"] == "split_not_exclude":
            rows.append(
                [
                    str(row["predicate_kind"]),
                    "split",
                    _int(row["n_before"]),
                    f"known {_int(row.get('n_exposure_known'))} / unknown {_int(row.get('n_exposure_unknown'))}",
                    _int(row["n_remaining"]),
                ]
            )
        else:
            rows.append(
                [str(row["predicate_kind"]), str(row["action"]), _int(row["n_before"]), _int(row["n_excluded"]), _int(row["n_remaining"])]
            )
    lines.append(_md_table(["Step", "Action", "n before", "Excluded / split", "n remaining"], rows))
    lines.append("")

    lines.append("## Exposure ascertainment")
    lines.append("")
    audit = result.measurement_audit
    rows = []
    for _, row in audit.loc[audit["audit_item"].eq("exposure_ascertainment")].iterrows():
        rows.append(
            [
                f"`{row['column']}`",
                str(row["role"]).replace("_", " "),
                _int(row["n_known"]),
                _int(row["n_unknown"]),
                _pct(row["share_unknown"]),
                _pct(row["event_rate_known"]),
                _pct(row["event_rate_unknown"]),
            ]
        )
    lines.append(
        _md_table(
            ["Column", "Role", "Known", "Unknown", "Unknown share", "Event rate (known)", "Event rate (unknown)"],
            rows,
        )
    )
    lines.append("")

    lines.append("## Absolute risk by level")
    lines.append("")
    rows = []
    for _, row in result.absolute_risk.iterrows():
        rows.append(
            [
                str(row["level"]),
                "yes" if bool(row["in_primary_model"]) else "no (unknown)",
                _int(row["n"]),
                _int(row["n_events"]),
                _pct(row["risk"]),
                f"{_pct(row['risk_ci_low'])}–{_pct(row['risk_ci_high'])}" if not _is_missing(row["risk"]) else "not estimable",
            ]
        )
    lines.append(_md_table(["Level", "In primary model", "n", "Events", "Risk", "Wilson 95% CI"], rows))
    lines.append("")

    lines.append("## Primary adjusted model")
    lines.append("")
    lines.append(
        f"Logistic regression (`{result.primary_summary.get('estimator_kind')}`, "
        f"{metrics['variance_estimator']} covariance, {_int(metrics['cluster_count'])} clusters) on "
        f"{_int(metrics['n_fit'])} rows with {_int(metrics['n_events_fit'])} events; "
        f"{_int(metrics['n_dropped_missing_covariates'])} known-exposure rows dropped for missing covariates; "
        f"{_int(metrics['n_model_parameters'])} parameters, EPV {_num(metrics['epv'], 1)}."
    )
    lines.append("")
    rows = []
    for _, row in result.adjusted_association_estimates.iterrows():
        rows.append(
            [
                f"{row['exposure_level']} vs {row['reference_level']}" + (" **(primary)**" if bool(row["is_primary_contrast"]) else ""),
                _ci(row["estimate"], row["ci_low"], row["ci_high"]),
            ]
        )
    lines.append(_md_table(["Contrast", "Adjusted OR (95% CI)"], rows))
    lines.append("")
    trend = result.adjusted_trend.iloc[0] if len(result.adjusted_trend) else None
    if trend is not None and trend["fit_status"] == "fitted":
        lines.append(
            f"Adjusted OR per one-level increment (declared level index): "
            f"{_ci(trend['estimate'], trend['ci_low'], trend['ci_high'])}."
        )
    elif trend is not None:
        lines.append(f"Adjusted per-level trend not estimable: {trend['note']}.")
    lines.append("")

    lines.append("## Ordered trend tests")
    lines.append("")
    rows = []
    for _, row in result.ordinal_trend_tests.iterrows():
        rows.append(
            [
                str(row.get("test_name") or row["test_id"]),
                str(row["outcome"]),
                str(row["population"]).replace("_", " "),
                _int(row.get("n")),
                _p(row.get("p_value")),
                _p(row.get("p_value_holm")) if bool(row["in_holm_family"]) else "not in family",
            ]
        )
    lines.append(_md_table(["Test", "Outcome", "Population", "n", "p", "Holm p (family)"], rows))
    lines.append("")
    if not result.secondary_outcome_summary.empty:
        lines.append("### Secondary outcomes by level")
        lines.append("")
        rows = []
        for _, row in result.secondary_outcome_summary.loc[result.secondary_outcome_summary["is_unknown"].eq(False)].iterrows():
            rows.append(
                [
                    str(row["outcome"]),
                    str(row["population"]).replace("_", " "),
                    str(row["level"]),
                    _int(row["n_total"]),
                    _int(row["n_evaluable"]),
                    f"{_num(row['median'])} ({_num(row['q25'])}–{_num(row['q75'])})",
                ]
            )
        lines.append(_md_table(["Outcome", "Population", "Level", "n", "Evaluable", "Median (IQR)"], rows))
        lines.append("")

    lines.append("## Prespecified sensitivity analyses")
    lines.append("")
    rows = []
    for _, row in result.association_sensitivity_grid.iterrows():
        rows.append(
            [
                str(row["variant_id"]),
                str(row["axis"]).replace("_", " "),
                _int(row["n"]),
                _int(row["n_events"]),
                _ci(row["estimate"], row["ci_low"], row["ci_high"]),
                str(row["fit_status"]),
                ("yes" if row.get("direction_consistent_with_primary") is True else "no" if row.get("direction_consistent_with_primary") is False else "—"),
            ]
        )
    lines.append(_md_table(["Variant", "Axis", "n", "Events", "OR primary contrast (95% CI)", "Fit", "Same direction"], rows))
    lines.append("")
    if not result.functional_form_sensitivity.empty:
        rows = []
        for _, row in result.functional_form_sensitivity.iterrows():
            rows.append(
                [
                    str(row["covariate"]),
                    _int(row["n_knots"]),
                    _int(row["n"]),
                    _ci(row["primary_contrast_or"], row["primary_contrast_ci_low"], row["primary_contrast_ci_high"]),
                    _p(row["nonlinearity_p_value"]),
                    str(row["fit_status"]) + (f" ({row['note']})" if row["note"] else ""),
                ]
            )
        lines.append(
            _md_table(
                ["Spline covariate", "Knots", "n", "OR primary contrast (95% CI)", "Nonlinearity p (model-based)", "Fit"],
                rows,
            )
        )
        lines.append("")

    lines.append("## Caveats required by the raised flags")
    lines.append("")
    for sentence in caveat_sentences(spec, flags):
        lines.append(f"- {sentence}")
    lines.append("")

    lines.append("## Fixed limitations of this design")
    lines.append("")
    lines.append(
        "- Exposure is classified from information available before the landmark; later changes in "
        "exposure state are not modelled (misclassification toward the null is possible)."
    )
    lines.append(
        "- Rows that died or left observation before the landmark are excluded by design; the estimand "
        "conditions on surviving to the landmark."
    )
    lines.append(
        "- Covariates enter as declared; unmeasured confounding and the temporal role of coded "
        "comorbidity indices are not resolved by this analysis."
    )
    lines.append("")

    if figures:
        lines.append("## Figures")
        lines.append("")
        for artifact in figures.values():
            lines.append(f"### {artifact.figure_id}")
            lines.append("")
            lines.append(f"![{artifact.figure_id}]({artifact.png_path})")
            lines.append("")
            lines.append(f"*{artifact.reader_caption}*")
            lines.append("")

    lines.append("## Exported files")
    lines.append("")
    for name, filename in table_files.items():
        lines.append(f"- `{filename}` — {name.replace('_', ' ')}")
    lines.append(f"- `{KEY_METRICS_FILENAME}` — single-row headline metrics (the source for every quoted number)")
    lines.append("- `caveat_flags.json` — machine-readable flags behind the caveat sentences")
    lines.append("- `spec.json` — the specification this run executed")
    lines.append(f"- `{MANIFEST_FILENAME}` — SHA-256 of every exported file")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# consistency gate
# --------------------------------------------------------------------------


def _close(left: Any, right: Any, *, rel: float = 1e-9) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    if _is_missing(left) or _is_missing(right):
        return False
    return math.isclose(float(left), float(right), rel_tol=rel, abs_tol=1e-12)


def assert_export_consistency(
    result: AnalysisResult, out_dir: Path, *, report_text: str
) -> tuple[int, int, list[str]]:
    """Hard row/number invariants raise; soft checks are returned as warnings."""

    spec = result.spec
    metrics = result.key_metrics
    warnings: list[str] = []
    checks: list[tuple[str, bool]] = []

    estimates = result.adjusted_association_estimates
    primary_rows = estimates.loc[estimates["is_primary_contrast"].astype(bool)]
    checks.append(("exactly one primary contrast row", len(primary_rows) == 1))
    checks.append(
        (
            "one estimate row per non-reference level",
            len(estimates) == len(spec.non_reference_levels),
        )
    )
    if len(primary_rows) == 1:
        checks.append(
            (
                "key_metrics.primary_or equals the primary estimate row",
                _close(metrics["primary_or"], primary_rows.iloc[0]["estimate"]),
            )
        )
    checks.append(
        ("key_metrics.n_fit equals the kernel n", int(metrics["n_fit"]) == int(result.primary_summary["n_total"]))
    )

    risk = result.absolute_risk
    checks.append(("absolute_risk has one row per level plus unknown", len(risk) == len(spec.exposure_levels) + 1))
    checks.append(("absolute_risk rows sum to the landmark population", int(risk["n"].sum()) == result.cohort.n_landmark))
    checks.append(
        (
            "absolute_risk known rows sum to the known population",
            int(risk.loc[risk["in_primary_model"].astype(bool), "n"].sum()) == result.cohort.n_exposure_known,
        )
    )

    flow = result.cohort_flow
    exclusions = flow.loc[flow["action"].ne("split_not_exclude")]
    checks.append(
        ("cohort_flow ends at the landmark population", int(exclusions.iloc[-1]["n_remaining"]) == result.cohort.n_landmark)
    )
    checks.append(
        (
            "known plus unknown equals the landmark population",
            result.cohort.n_exposure_known + result.cohort.n_exposure_unknown == result.cohort.n_landmark,
        )
    )

    grid = result.association_sensitivity_grid
    expected_grid = 1 + len(spec.alternate_exposures) + (1 if spec.first_stay_column else 0)
    checks.append(("sensitivity grid has every declared variant", len(grid) == expected_grid))
    grid_primary = grid.loc[grid["variant_id"].eq(PRIMARY_VARIANT_ID)]
    checks.append(
        (
            "sensitivity grid primary row equals key_metrics.primary_or",
            len(grid_primary) == 1 and _close(grid_primary.iloc[0]["estimate"], metrics["primary_or"]),
        )
    )
    checks.append(
        (
            "functional-form table has one row per declared covariate",
            len(result.functional_form_sensitivity) == len(spec.functional_form_covariates),
        )
    )

    referenced = set(re.findall(r"`([A-Za-z0-9_\-.]+\.(?:csv|json|md|png|svg))`", report_text))
    referenced |= set(re.findall(r"\]\(([A-Za-z0-9_\-.]+\.(?:png|svg))\)", report_text))
    # The manifest is written after this gate (it hashes the gated files), so
    # it is the one referenced file that legitimately does not exist yet.
    referenced.discard(MANIFEST_FILENAME)
    missing_files = sorted(name for name in referenced if not (out_dir / name).is_file())
    checks.append(("every file referenced by report.md exists", not missing_files))

    key_metrics_path = out_dir / KEY_METRICS_FILENAME
    if key_metrics_path.is_file():
        reread = pd.read_csv(key_metrics_path)
        checks.append(("key_metrics.csv is a single row", len(reread) == 1))
        if len(reread) == 1:
            checks.append(
                (
                    "key_metrics.csv round-trips the primary estimate",
                    _close(reread.iloc[0]["primary_or"], metrics["primary_or"], rel=1e-9),
                )
            )
    else:
        checks.append(("key_metrics.csv was written", False))

    table_one = result.table_one
    if "group_missing_excluded_n" in table_one.columns and len(table_one):
        excluded = int(table_one["group_missing_excluded_n"].iloc[0])
        if excluded != result.cohort.n_exposure_unknown:
            warnings.append(
                f"Table 1 excluded {excluded} ungrouped rows but the landmark cohort has "
                f"{result.cohort.n_exposure_unknown} unknown-exposure rows"
            )
    for name, table in result.tables().items():
        if table.empty and name not in {"secondary_outcome_summary", "functional_form_sensitivity"}:
            warnings.append(f"exported table {name} is empty")

    failed = [name for name, ok in checks if not ok]
    if missing_files:
        failed.append("missing files: " + ", ".join(missing_files))
    if failed:
        raise ExportConsistencyError(
            "Export consistency check FAILED: " + "; ".join(failed)
        )
    return len(checks), len(checks), warnings


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def export_all(
    result: AnalysisResult,
    out_dir: str | Path,
    *,
    figures: Mapping[str, FigureArtifact] | None = None,
    verbose: bool = True,
) -> ExportReceipt:
    """Write tables, key metrics, flags, spec, report and manifest; then gate."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("\n=== Exporting results ===")
    table_files: dict[str, str] = {}
    for name, table in result.tables().items():
        filename = f"{name}.csv"
        table.to_csv(target / filename, index=False)
        table_files[name] = filename
        if verbose:
            print(f"   Saved: {filename} ({len(table)} rows)")

    key_metrics = pd.DataFrame([result.key_metrics])
    key_metrics.to_csv(target / KEY_METRICS_FILENAME, index=False)
    (target / "caveat_flags.json").write_text(
        json.dumps(result.caveat_flags, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (target / "spec.json").write_text(
        json.dumps(result.spec.to_json_dict(), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    if figures:
        (target / "figures.json").write_text(
            json.dumps(
                {figure_id: artifact.as_dict() for figure_id, artifact in figures.items()},
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    report_text = build_report(result, figures, table_files=table_files)
    (target / REPORT_FILENAME).write_text(report_text, encoding="utf-8")
    if verbose:
        print(f"   Saved: {KEY_METRICS_FILENAME}, caveat_flags.json, spec.json, {REPORT_FILENAME}")

    if verbose:
        print("\n=== Consistency Check ===")
    passed, total, warnings = assert_export_consistency(result, target, report_text=report_text)
    if verbose:
        for warning in warnings:
            print(f"  WARN: {warning}")
        print(f"  Consistency check: PASSED ({passed}/{total} hard checks)")

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(target.iterdir()):
        if path.is_file() and path.name != MANIFEST_FILENAME:
            files[path.name] = {"sha256": sha256_of_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": "easyicu.skill_export_manifest/1",
        "skill": SKILL_VERSION,
        "spec_sha256": hashlib.sha256(
            json.dumps(result.spec.to_json_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_sha256": result.cohort.source_sha256,
        "source_path": result.cohort.source_path,
        "data_provenance": result.cohort.provenance,
        "n_source": result.cohort.n_source,
        "n_landmark": result.cohort.n_landmark,
        "consistency_checks_passed": passed,
        "consistency_checks_total": total,
        "warnings": warnings,
        "files": files,
    }
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    (target / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    receipt = ExportReceipt(
        out_dir=str(target),
        files=files,
        checks_passed=passed,
        checks_total=total,
        warnings=tuple(warnings),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    if verbose:
        print(f"\n{EXPORT_TOKEN}")
        print(f"  {len(files)} files, manifest sha256 {receipt.manifest_sha256[:16]}…")
    return receipt


__all__ = [
    "EXPORT_TOKEN",
    "KEY_METRICS_FILENAME",
    "MANIFEST_FILENAME",
    "REPORT_FILENAME",
    "ExportConsistencyError",
    "ExportReceipt",
    "assert_export_consistency",
    "build_report",
    "caveat_sentences",
    "export_all",
]
