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
from .pipeline_report import _blocked_outcome_step_ids
from .robustness_panel import load_robustness_panel, worst_rows_by_axis
from .scalar_utils import _first_present_scalar


__all__ = [
    "_resolve_writer_aux_path",
    "_summarise_table_one_rows",
    "_summarise_primary_association_table",
    "_preferred_writer_evidence_names",
    "_render_writer_evidence_digest",
    "_render_writer_evidence_digest_v2",
    "WRITER_DIGEST_PREFERRED_KEYS",
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


def _preferred_writer_evidence_names(evidence: EvidenceStore) -> List[str]:
    aliases = evidence.aliases()
    def is_citable(name: str) -> bool:
        record = evidence.get(name)
        if record is None:
            return False
        return record.finding_severity not in {"warning", "error"}

    preferred = [
        "table_one",
        "cohort_summary",
        "outcome_incidence",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
        "multiple_testing_report",
        "fairness_subgroups",
        "literature_prisma",
        "causal_audit_report",
        "causal_audit_summary",
        "reporting_checklist",
    ]
    out: List[str] = [name for name in preferred if is_citable(name)]
    step_aliases = [
        name for name in sorted(aliases)
        if re.match(r"^\d{2}[_-]", name)
    ]
    for name in step_aliases:
        if name not in out and is_citable(name):
            out.append(name)
    if out:
        return out
    clean_names: List[str] = []
    seen: set[str] = set()
    for name in evidence.resolvable_names():
        if name in seen or not is_citable(name):
            continue
        seen.add(name)
        clean_names.append(name)
    return clean_names or evidence.resolvable_names()


# Module-level constant: the keys the "primary" writer digest pulls out
# of step_summary. v1 (``_render_writer_evidence_digest``) feeds *only*
# these to the writer. v2 (``_render_writer_evidence_digest_v2``,
# Phase-1 widening behind ``PipelineConfig.writer_digest_widened``) uses
# this same tuple as the primary block plus a secondary block sourced
# from the full ``EvidenceStore.numeric_claims()`` registry.
#
# Exposed at module level so:
#   1. v2 can reuse it without re-defining the literal list, and
#   2. tests can assert primary vs secondary partitioning without
#      hand-maintaining a parallel tuple.
WRITER_DIGEST_PREFERRED_KEYS: tuple[str, ...] = (
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
    "effect_estimate",
    "primary_or",
    "odds_ratio",
    "adjusted_or",
    "primary_hr",
    "hazard_ratio",
    "adjusted_hr",
    "primary_ate",
    "ate",
    "average_treatment_effect",
    "treatment_effect",
    "risk_difference",
    "mean_difference",
    "coef",
    "coefficient",
    "beta",
    "primary_beta",
    "ci_lower",
    "ci_upper",
    "ci_low",
    "ci_high",
    "primary_ci_low",
    "primary_ci_high",
    "primary_or_ci",
    "primary_hr_ci",
    "primary_effect_ci",
    "estimate_ci_low",
    "estimate_ci_high",
    "p_value",
    "median_los_icu",
    "mean_los_icu",
    "median_los_hosp",
    "mean_los_hosp",
    "median_los_hospital",
    "mean_los_hospital",
    "los_icu_median",
    "los_hosp_median",
    "icu_los_median",
    "hospital_los_median",
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

_PRIMARY_EFFECT_DIGEST_KEYS_WHEN_PANEL_PRESENT = {
    "estimate",
    "effect_estimate",
    "primary_or",
    "odds_ratio",
    "adjusted_or",
    "primary_hr",
    "hazard_ratio",
    "adjusted_hr",
    "primary_ate",
    "ate",
    "average_treatment_effect",
    "treatment_effect",
    "risk_difference",
    "mean_difference",
    "coef",
    "coefficient",
    "beta",
    "primary_beta",
    "ci_lower",
    "ci_upper",
    "ci_low",
    "ci_high",
    "primary_ci_low",
    "primary_ci_high",
    "primary_or_ci",
    "primary_hr_ci",
    "primary_effect_ci",
    "estimate_ci_low",
    "estimate_ci_high",
    "p_value",
}


def _render_writer_evidence_digest(
    per_step_records: Sequence[Dict[str, Any]] | None = None,
    *,
    context: ResearchContext | None = None,
    run_dir: Path | None = None,
    include_robustness_panel: bool = True,
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
    has_panel_primary = _robustness_panel_has_primary_effect(run_dir)
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
        "effect_estimate",
        "primary_or",
        "odds_ratio",
        "adjusted_or",
        "primary_hr",
        "hazard_ratio",
        "adjusted_hr",
        "primary_ate",
        "ate",
        "average_treatment_effect",
        "treatment_effect",
        "risk_difference",
        "mean_difference",
        "coef",
        "coefficient",
        "beta",
        "primary_beta",
        "ci_lower",
        "ci_upper",
        "ci_low",
        "ci_high",
        "primary_ci_low",
        "primary_ci_high",
        "primary_or_ci",
        "primary_hr_ci",
        "primary_effect_ci",
        "estimate_ci_low",
        "estimate_ci_high",
        "p_value",
        "median_los_icu",
        "mean_los_icu",
        "median_los_hosp",
        "mean_los_hosp",
        "median_los_hospital",
        "mean_los_hospital",
        "los_icu_median",
        "los_hosp_median",
        "icu_los_median",
        "hospital_los_median",
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
            if has_panel_primary and key in _PRIMARY_EFFECT_DIGEST_KEYS_WHEN_PANEL_PRESENT:
                continue
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
        if not has_panel_primary:
            digest_row.update(_summarise_primary_association_table(primary_path))
        lines.append(
            "  " + json.dumps(digest_row, ensure_ascii=False, sort_keys=True, default=str)
        )
    primary = "\n".join(lines)
    primary = _append_blocked_outcome_gate_block(primary, run_dir=run_dir)
    if not include_robustness_panel:
        return primary
    return _append_robustness_panel_block(primary, run_dir=run_dir)


# ---------------------------------------------------------------------------
# v2 digest — primary + secondary blocks (Phase 1 widening)
# ---------------------------------------------------------------------------
#
# The v1 digest above feeds the writer ONLY the curated
# ``WRITER_DIGEST_PREFERRED_KEYS`` subset of every step_summary. The
# manuscript binder (``manuscript_post.bind_numeric_values``) is
# already wider than that — it accepts any value present in the full
# ``EvidenceStore.numeric_claims()`` registry via tolerance-based
# fuzzy matching. So in practice the writer is BIASED toward primary
# keys (because that's what it sees), not RESTRICTED to them.
#
# v2 closes the gap by appending a "secondary numbers" block. The
# secondary block enumerates every NumericClaim whose ``source_field``
# is not already covered by the primary block, grouped by step_id and
# capped to keep the writer prompt from bloating on heavy runs.
#
# v2 is opt-in via ``PipelineConfig.writer_digest_widened`` (default
# False). v1's behaviour is byte-for-byte preserved for callers that
# don't flip the flag. See the writer-digest widening design note in the
# project's internal design docs §2.


def _claim_step_field_covered_by_primary(
    *,
    step_id: str,
    source_field: str,
    primary_step_field_keys: set[tuple[str, str]],
) -> bool:
    """Return True if (step_id, source_field) is already cited in the v1 primary block.

    Matching rules:

    1. Exact match on (step_id, source_field).
    2. ``source_field`` is one of ``WRITER_DIGEST_PREFERRED_KEYS`` and the
       step emitted ANY primary-block row (handled by the caller via
       ``primary_step_field_keys`` set construction).
    3. ``statistic:<name>`` and ``<name>`` are treated as the same key
       (matches the v1 ``_first_present_scalar`` flatten behaviour
       tested in ``test_render_writer_evidence_digest_flattens_nested_statistics``).
    """
    if (step_id, source_field) in primary_step_field_keys:
        return True
    if source_field.startswith("statistic:"):
        if (step_id, source_field[len("statistic:") :]) in primary_step_field_keys:
            return True
    else:
        if (step_id, f"statistic:{source_field}") in primary_step_field_keys:
            return True
    return False


def _render_writer_evidence_digest_v2(
    per_step_records: Sequence[Dict[str, Any]] | None = None,
    *,
    context: ResearchContext | None = None,
    run_dir: Path | None = None,
    evidence: EvidenceStore | None = None,
    secondary_cap_per_step: int = 20,
) -> str:
    """Phase-1 wider writer-evidence digest.

    Layout:

    ::

        RUN_CONTEXT
          {json}
        ## primary numbers (most likely to be cited)
        - <step_id> [<status>]
          {json of primary subset}
        ## secondary numbers (cite if relevant; binder will accept)
        - <step_id>
          source_field=<value> (canonical=<canonical>)
          ...

    The primary block is the byte-identical output of
    :func:`_render_writer_evidence_digest`. The secondary block reads
    ``evidence.numeric_claims()`` and shows fields not already in the
    primary block. ``secondary_cap_per_step`` caps how many secondary
    fields per step are emitted; the cap exists because
    ``register_step_summary_numerics`` can legitimately register up to
    ``PipelineConfig.max_numeric_claims_per_step`` (default 100) leaves
    per step.

    ``evidence`` is optional. When None, the secondary block falls
    back to enumerating fields directly from each record's
    ``step_summary`` — useful for callers that do not have a live
    ``EvidenceStore`` (tests, replay tooling). When ``evidence`` is
    supplied, the secondary block prefers it because the registry has
    the canonical literal+float pair plus an authoritative dedup.
    """
    primary = _render_writer_evidence_digest(
        per_step_records,
        context=context,
        run_dir=run_dir,
        include_robustness_panel=False,
    )
    records = list(per_step_records or [])
    if not records:
        return _append_robustness_panel_block(primary, run_dir=run_dir)
    primary_keys_lower = {k.lower() for k in WRITER_DIGEST_PREFERRED_KEYS}

    # Build the (step_id, source_field) coverage set that the primary
    # block emitted. We can't introspect the rendered string cheaply,
    # so we walk per_step_records the same way v1 walks them and
    # record every key v1 *would* include.
    primary_step_field_keys: set[tuple[str, str]] = set()
    for record in records:
        step_id = str(record.get("step_id") or "unknown_step")
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        for key in WRITER_DIGEST_PREFERRED_KEYS:
            if _first_present_scalar(summary, (key,)) is not None:
                primary_step_field_keys.add((step_id, key))

    secondary_lines: List[str] = []
    derived_lines: List[str] = []
    if evidence is not None:
        # Group claims by step_id, sorted for determinism.
        claims_by_step: Dict[str, List[Any]] = {}
        for claim in evidence.numeric_claims():
            if claim.source_field == "__easyicu_numeric_claim_overflow__":
                continue
            claims_by_step.setdefault(claim.step_id, []).append(claim)
        for step_id in sorted(claims_by_step.keys()):
            step_claims = claims_by_step[step_id]
            # Preserve registration order (which matches step_summary
            # leaf walk order) while partitioning the uncovered claims
            # into formula-derived and ordinary secondary values.
            uncovered = [
                c
                for c in step_claims
                if not _claim_step_field_covered_by_primary(
                    step_id=step_id,
                    source_field=c.source_field,
                    primary_step_field_keys=primary_step_field_keys,
                )
                and c.source_field.lower() not in primary_keys_lower
                and not _is_hidden_robustness_row_claim(step_id, c.source_field)
            ]
            derived_claims = [
                c for c in uncovered if getattr(c, "is_derived", False)
            ]
            secondary_claims = [
                c for c in uncovered if not getattr(c, "is_derived", False)
            ]
            if derived_claims:
                derived_total = len(derived_claims)
                derived_truncated = False
                cap = max(0, int(secondary_cap_per_step))
                if cap and derived_total > cap:
                    derived_claims = derived_claims[:cap]
                    derived_truncated = True
                derived_lines.append(f"- {step_id}")
                for c in derived_claims:
                    sources = ", ".join(
                        f"{src_step}.{src_field}"
                        for src_step, src_field in getattr(c, "derived_from", [])
                    )
                    derived_lines.append(f"  {c.source_field}={c.value}")
                    derived_lines.append(f"    formula={c.formula}")
                    if sources:
                        derived_lines.append(f"    sources={sources}")
                    if getattr(c, "explanation", None):
                        derived_lines.append(f"    explanation={c.explanation}")
                if derived_truncated:
                    derived_lines.append(
                        f"  ... ({derived_total - cap} more derived leaves omitted)"
                    )
            if not secondary_claims:
                continue
            truncated = False
            cap = max(0, int(secondary_cap_per_step))
            secondary_total = len(secondary_claims)
            if cap and secondary_total > cap:
                secondary_claims = secondary_claims[:cap]
                truncated = True
            secondary_lines.append(f"- {step_id}")
            for c in secondary_claims:
                secondary_lines.append(
                    f"  {c.source_field}={c.value} (canonical={c.canonical})"
                )
            if truncated:
                secondary_lines.append(
                    f"  ... ({secondary_total - cap} more leaves omitted; raise writer_digest_secondary_cap_per_step to see)"
                )
    else:
        # Fallback path: walk step_summary directly. Cheaper than the
        # registry path but lacks canonical floats.
        for record in records:
            step_id = str(record.get("step_id") or "unknown_step")
            summary = record.get("step_summary")
            if not isinstance(summary, dict) or not summary:
                continue
            uncovered_pairs: List[tuple[str, Any]] = []
            for key, value in summary.items():
                if not isinstance(value, (int, float, str)) and not (
                    isinstance(value, list) and value and isinstance(value[0], (int, float))
                ):
                    continue
                if (step_id, key) in primary_step_field_keys:
                    continue
                if key.lower() in primary_keys_lower:
                    continue
                uncovered_pairs.append((key, value))
            if not uncovered_pairs:
                continue
            truncated = False
            cap = max(0, int(secondary_cap_per_step))
            uncovered_pairs_total = len(uncovered_pairs)
            if cap and uncovered_pairs_total > cap:
                uncovered_pairs = uncovered_pairs[:cap]
                truncated = True
            secondary_lines.append(f"- {step_id}")
            for key, value in uncovered_pairs:
                secondary_lines.append(f"  {key}={value}")
            if truncated:
                secondary_lines.append(
                    f"  ... ({uncovered_pairs_total - cap} more leaves omitted; pass evidence= or raise the per-step cap to see)"
                )

    extra_blocks: List[str] = []
    robustness_lines = _render_robustness_panel_block(run_dir=run_dir)
    if robustness_lines:
        extra_blocks.extend(["", "## robustness panel", *robustness_lines])
    if derived_lines:
        extra_blocks.extend(
            [
                "",
                "## derived numbers (computed from registered claims; cite with explanation)",
                *derived_lines,
            ]
        )
    if secondary_lines:
        extra_blocks.extend(
            [
                "",
                "## secondary numbers (cite if relevant; binder will accept any registered claim)",
                *secondary_lines,
            ]
        )

    if not extra_blocks:
        # No additional bindable numbers found beyond the primary
        # block; emit only the primary block (no empty header) so the
        # writer doesn't see an "empty secondary block" prompt artifact.
        return primary

    return "\n".join([primary, *extra_blocks])


def _append_robustness_panel_block(text: str, *, run_dir: Path | None) -> str:
    robustness_lines = _render_robustness_panel_block(run_dir=run_dir)
    if not robustness_lines:
        return text
    return "\n".join([text, "", "## robustness panel", *robustness_lines])


def _append_blocked_outcome_gate_block(text: str, *, run_dir: Path | None) -> str:
    guard_lines = _render_blocked_outcome_gate_block(run_dir=run_dir)
    if not guard_lines:
        return text
    return "\n".join([text, "", "## blocked outcome gate", *guard_lines])


def _render_blocked_outcome_gate_block(*, run_dir: Path | None) -> List[str]:
    if run_dir is None:
        return []
    root = Path(run_dir)
    blocked_steps = _blocked_outcome_step_ids(root)
    if not blocked_steps:
        return []
    lines = [
        "BLOCKED OUTCOME GATE: one or more executed steps explicitly blocked "
        "outcome linkage/tabulation.",
        "Writer instruction: do not report outcome associations, effects, "
        "contrasts, near-null interpretations, or point-estimate ranges from "
        "these blocked steps. It is acceptable to state that the outcome "
        "analysis was blocked and why.",
        "Any robustness or publication-figure effect estimates derived from "
        "the blocked outcome linkage are not manuscript-facing.",
        "blocked_steps=" + ",".join(blocked_steps),
    ]
    for step_id in blocked_steps:
        note = _blocked_outcome_step_note(root, step_id)
        if note:
            lines.append(f"{step_id}: {note}")
    return lines


def _blocked_outcome_step_note(root: Path, step_id: str) -> str:
    step_out = root / "steps" / step_id / "outputs"
    notes: List[str] = []
    for path in sorted(step_out.glob("*gate*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for _, row in frame.iterrows():
            status = str(row.get("status", "")).lower()
            if status != "blocked":
                continue
            decision = str(row.get("blocking_decision", "")).strip()
            rerun = str(row.get("future_rerun_condition", "")).strip()
            if decision:
                notes.append(decision)
            if rerun:
                notes.append("Rerun condition: " + rerun)
            if notes:
                return " ".join(notes)[:500]
    summary_path = step_out / "step_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        policies = summary.get("named_blocking_policy")
        if isinstance(policies, list) and policies:
            return "Blocking policies: " + ", ".join(str(p) for p in policies[:6])
    return ""


def _is_hidden_robustness_row_claim(step_id: str, source_field: str) -> bool:
    """Hide row-level robustness claims from the writer digest.

    The numeric binder can still trace these registered values, but the writer
    should only see the fixed robustness summary block to avoid variant
    cherry-picking.
    """

    return step_id == "robustness_panel" and source_field.startswith("row_")


def _render_robustness_panel_block(*, run_dir: Path | None) -> List[str]:
    if run_dir is None:
        return []
    if _blocked_outcome_step_ids(Path(run_dir)):
        return []
    panel = load_robustness_panel(Path(run_dir) / "robustness_panel.json")
    if panel is None:
        return []
    primary = next(
        (row for row in panel.rows if row.spec_id == panel.primary_spec_id),
        None,
    )
    lines: List[str] = []
    if primary is not None:
        lines.append(
            "CANONICAL PRIMARY EFFECT SOURCE: use this robustness-panel "
            "primary row for the manuscript-facing primary effect. Do not "
            "mix it with generated per-step model estimates."
        )
        lines.append(
            "primary: "
            f"spec_id={primary.spec_id}, "
            f"point={_fmt_panel_number(primary.point_estimate)}, "
            f"CI=[{_fmt_panel_number(primary.ci_low)}, "
            f"{_fmt_panel_number(primary.ci_high)}], "
            f"n={primary.n}"
        )
    converged_variants = [
        row
        for row in panel.rows
        if row.spec_id != panel.primary_spec_id and row.converged
    ]
    if converged_variants:
        lines.append(
            "variants: "
            f"n_variants={panel.n_variants}, "
            "range across variants point "
            f"in [{_fmt_panel_number(panel.range_low)}, "
            f"{_fmt_panel_number(panel.range_high)}]"
        )
    else:
        lines.append(
            "variants: "
            f"n_variants={panel.n_variants}, "
            "no robustness variants converged "
            "(see robustness_panel.json for MVP boundary reasons)"
        )
    for axis, row in sorted(worst_rows_by_axis(panel).items()):
        lines.append(
            f"worst on {axis} axis: "
            f"spec_id={row.spec_id}, point={_fmt_panel_number(row.point_estimate)}"
        )
    return lines


def _robustness_panel_has_primary_effect(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    panel = load_robustness_panel(Path(run_dir) / "robustness_panel.json")
    if panel is None:
        return False
    primary = next(
        (row for row in panel.rows if row.spec_id == panel.primary_spec_id),
        None,
    )
    return (
        primary is not None
        and primary.converged
        and primary.point_estimate is not None
        and primary.ci_low is not None
        and primary.ci_high is not None
    )


def _fmt_panel_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
