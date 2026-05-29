"""Salvage helpers that backfill ``step_summary.json`` from raw artefacts.

Hosted / local coder models sometimes leave a step in a half-finished
state — they print a valid summary to stdout but skip writing the file,
or they write the CSV / figure outputs but leave ``step_summary.json``
as ``{}``. Treating that as a hard failure makes the run look unusable
even when the structured evidence is on disk.

These helpers are deliberately conservative: they only recover values
that can be read deterministically from existing artefacts, never
inventing numbers. They run before the validators look at the step.

Originally inline in :mod:`pipeline`; isolated here because they are
pure functions over (run_result | out_dir | step) → bool, with no
dependency on pipeline state. The companion deterministic-repair
helpers (which mutate the *code* in response to summary failures)
live in :mod:`.code_repair`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .runner import RunResult
from .schema import AnalysisStep


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    latest: Optional[Dict[str, Any]] = None
    for idx, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(value, dict):
            latest = value
    return latest


def _salvage_stdout_json_step_summary(run_result: RunResult) -> bool:
    """Persist a JSON object printed to stdout as step_summary.json.

    Hosted coder models sometimes compute the right summary and print it,
    but forget to write artefacts into ``STEP_OUT_DIR``. This preserves the
    agent-generated result without replacing the analysis with fixed code.
    """

    out_dir = run_result.out_dir
    summary_path = out_dir / "step_summary.json"
    if summary_path.exists():
        return False
    data = _extract_last_json_object(run_result.stdout or "")
    if not isinstance(data, dict) or not data:
        return False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        return False
    return True


def _salvage_named_json_step_summary(run_result: RunResult) -> bool:
    """Promote an agent-written summary JSON artefact to step_summary.json."""

    out_dir = run_result.out_dir
    summary_path = out_dir / "step_summary.json"
    if summary_path.exists():
        return False
    excluded = {
        "critique_report.json",
        "visual_qa.json",
        "figure_contract.json",
    }
    candidates = sorted(
        path
        for path in out_dir.glob("*.json")
        if "summary" in path.name.lower() and path.name.lower() not in excluded
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or not data:
            continue
        try:
            summary_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            return False
        return True
    return False


def _salvage_minimal_contract_step_summary(
    *,
    step: AnalysisStep,
    out_dir: Path,
) -> bool:
    """Backfill an empty ``step_summary.json`` from standard artefacts.

    Small local coder models sometimes write the main CSV/PDF/SVG outputs but
    leave ``step_summary.json`` as ``{}``. Treating that as a hard failure makes
    the run look unusable even though the structured evidence is present. This
    helper only fills a *minimal* machine-readable summary from artefacts that
    already exist on disk; it never invents numbers that are not recoverable
    deterministically from those artefacts.
    """
    summary_path = out_dir / "step_summary.json"
    if not summary_path.exists():
        return False
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(loaded, dict) or loaded:
        return False

    recovered: Dict[str, Any] = {}
    table_one_csv = out_dir / "table_one.csv"
    if table_one_csv.exists():
        try:
            frame = pd.read_csv(table_one_csv)
            recovered["table_one_path"] = table_one_csv.name
            recovered["table_one_rows"] = frame.to_dict(orient="records")
            recovered["n_rows"] = int(len(frame))
            recovered["variables_reported"] = sorted(
                {str(value) for value in frame.get("variable", pd.Series(dtype=str)).dropna().astype(str)}
            )
        except Exception:
            pass

    outcome_csv = out_dir / "outcome_incidence.csv"
    if outcome_csv.exists():
        try:
            frame = pd.read_csv(outcome_csv)
            recovered["outcome_incidence_path"] = outcome_csv.name
            for key in ("outcome_rate", "mortality_rate", "rate", "incidence"):
                if key in frame.columns and not frame.empty:
                    recovered["outcome_rate"] = float(frame[key].iloc[0])
                    break
        except Exception:
            pass

    assoc_csv = out_dir / "primary_association.csv"
    if assoc_csv.exists():
        try:
            frame = pd.read_csv(assoc_csv)
            recovered["primary_association_path"] = assoc_csv.name
            if not frame.empty:
                working = frame.copy()
                if "variable" in working.columns:
                    working = working.loc[
                        ~working["variable"].astype(str).str.lower().isin({"const", "intercept"})
                    ]
                row = working.iloc[0] if not working.empty else frame.iloc[0]
                predictor = row.get("variable")
                if predictor is not None:
                    recovered["predictor"] = str(predictor)
                for src, dst in (
                    ("odds_ratio", "primary_or"),
                    ("estimate", "primary_or"),
                    ("p_value", "primary_pvalue"),
                    ("p", "primary_pvalue"),
                    ("ci_low", "primary_ci_low"),
                    ("conf_low", "primary_ci_low"),
                    ("ci_high", "primary_ci_high"),
                    ("conf_high", "primary_ci_high"),
                ):
                    value = row.get(src)
                    if value is None:
                        continue
                    try:
                        recovered[dst] = float(value)
                    except Exception:
                        continue
        except Exception:
            pass

    perf_csv = out_dir / "model_performance.csv"
    if perf_csv.exists():
        try:
            frame = pd.read_csv(perf_csv)
            recovered["model_performance_path"] = perf_csv.name
            for src, dst in (
                ("auroc", "cv_auroc_mean"),
                ("auc", "cv_auroc_mean"),
                ("brier_score", "brier_score"),
                ("calibration_slope", "calibration_slope"),
                ("calibration_intercept", "calibration_intercept"),
            ):
                if src not in frame.columns:
                    continue
                series = pd.to_numeric(frame[src], errors="coerce").dropna()
                if series.empty:
                    continue
                recovered[dst] = float(series.mean())
        except Exception:
            pass

    figure_files = sorted(
        path.name
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".svg", ".pdf", ".tiff", ".tif"}
    )
    if figure_files:
        recovered["figure_files"] = figure_files
        recovered.setdefault("figure_path", figure_files[0])

    if not recovered:
        return False
    try:
        summary_path.write_text(
            json.dumps(recovered, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class SummarySalvageOutcome:
    """Describes which step-summary salvage fired, so the caller can record it.

    Keeping this separate from the recording side lets the salvage decision be
    unit-tested end-to-end (which salvage fired, with which repair id) without
    driving the whole execute phase. ``repair_id`` values are classified in
    :mod:`.repair_registry`.
    """

    repair_id: str
    trigger_reason: str
    transformation: str
    selection_rule: Optional[str] = None
    reset_artefacts: bool = False


def salvage_step_summary(
    run_result: RunResult, *, step: AnalysisStep
) -> Optional[SummarySalvageOutcome]:
    """Run step-summary salvage and report what (if anything) was salvaged.

    Behaviour matches the previous inline logic exactly:

    * if ``step_summary.json`` is absent, try stdout JSON then a named summary
      artefact (short-circuit), and signal that artefacts should be re-listed;
    * else (present but empty) backfill a minimal contract from on-disk CSVs.

    Returns ``None`` when no salvage was needed or possible. The caller is
    responsible for recording the returned outcome in the repair ledger.
    """

    summary_path = run_result.out_dir / "step_summary.json"
    if not summary_path.exists():
        if _salvage_stdout_json_step_summary(run_result):
            return SummarySalvageOutcome(
                repair_id="summary_salvage_stdout_json_v1",
                trigger_reason="step produced no step_summary.json",
                transformation=(
                    "Recovered step_summary.json from the step's own stdout JSON "
                    "without re-running analysis."
                ),
                reset_artefacts=True,
            )
        if _salvage_named_json_step_summary(run_result):
            return SummarySalvageOutcome(
                repair_id="summary_salvage_named_json_v1",
                trigger_reason="step produced no step_summary.json",
                transformation=(
                    "Promoted a named summary JSON artefact to step_summary.json."
                ),
                reset_artefacts=True,
            )
        return None
    if _salvage_minimal_contract_step_summary(step=step, out_dir=run_result.out_dir):
        return SummarySalvageOutcome(
            repair_id="summary_salvage_minimal_contract_v1",
            trigger_reason=(
                "empty step_summary.json backfilled from on-disk artefacts"
            ),
            transformation=(
                "Backfilled a minimal step_summary from existing CSV/figure "
                "artefacts; no numbers invented beyond deterministic extraction."
            ),
            selection_rule=(
                "first non-const/intercept association row; mean of "
                "model_performance rows; deterministic CSV-to-summary extraction"
            ),
        )
    return None


__all__ = [
    "_extract_last_json_object",
    "_salvage_stdout_json_step_summary",
    "_salvage_named_json_step_summary",
    "_salvage_minimal_contract_step_summary",
    "SummarySalvageOutcome",
    "salvage_step_summary",
]
