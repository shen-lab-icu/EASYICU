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
live in :mod:`.source`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from ..contracts.runtime import RunResult
from ..contracts.declared_product import (
    _descriptor_path_is_compatible,
    typed_product,
)
from ..schema import AnalysisStep

_OUTPUT_REGISTRY_CANONICALIZATION_REPAIR_ID = (
    "summary_output_registry_canonicalization_v1"
)


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
                {
                    str(value)
                    for value in frame.get("variable", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                }
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
                        ~working["variable"]
                        .astype(str)
                        .str.lower()
                        .isin({"const", "intercept"})
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
        if path.is_file()
        and path.suffix.lower() in {".png", ".svg", ".pdf", ".tiff", ".tif"}
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


def _canonicalize_exact_declared_output_registry(
    *,
    step: AnalysisStep,
    out_dir: Path,
) -> bool:
    """Promote one exact legacy ``outputs`` map to typed ``output_files``.

    This is a representation-only compatibility repair.  It is deliberately
    narrower than the declared-product validator: every declared product must
    have a unique bare product name, the legacy map must contain exactly those
    names, and every value must identify a distinct, regular, output-local file
    whose physical suffix is compatible with the Planner-declared kind.  The
    host never guesses a missing product, chooses among files, or changes any
    scientific value.

    Container runtimes often write an absolute path such as
    ``/easyicu-run/steps/.../outputs/result.csv`` into the summary.  The host
    binds only its basename and only when that exact regular file already
    exists in ``out_dir``.  Parent traversal, links, duplicate inodes, extra
    keys, and ambiguous same-name typed products all fail closed.
    """

    summary_path = out_dir / "step_summary.json"
    if not summary_path.exists() or summary_path.is_symlink():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(summary, dict) or "output_files" in summary:
        return False
    legacy_outputs = summary.get("outputs")
    if not isinstance(legacy_outputs, Mapping) or not legacy_outputs:
        return False

    declared = [
        product
        for raw in (step.expected_outputs or [])
        if (product := typed_product(raw)) is not None
    ]
    if not declared:
        return False
    declared_names = [name for _kind, name in declared]
    if len(set(declared)) != len(declared) or len(set(declared_names)) != len(
        declared_names
    ):
        return False
    if not all(isinstance(key, str) for key in legacy_outputs):
        return False
    if set(legacy_outputs) != set(declared_names):
        return False

    try:
        root = out_dir.resolve(strict=True)
    except OSError:
        return False
    output_files: dict[str, str] = {}
    file_identities: set[tuple[int, int]] = set()
    for kind, name in declared:
        raw_path = legacy_outputs.get(name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            return False
        supplied = Path(raw_path.strip())
        if ".." in supplied.parts or not supplied.name:
            return False
        candidate = out_dir / supplied.name
        try:
            if candidate.is_symlink():
                return False
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            stat = resolved.stat()
        except (OSError, ValueError):
            return False
        if not resolved.is_file() or not _descriptor_path_is_compatible(
            kind=kind,
            path=resolved.name,
        ):
            return False
        identity = (stat.st_dev, stat.st_ino)
        if identity in file_identities:
            return False
        file_identities.add(identity)
        output_files[f"{kind}:{name}"] = resolved.name

    summary["output_files"] = output_files
    summary["output_registry_repair"] = {
        "repair_id": _OUTPUT_REGISTRY_CANONICALIZATION_REPAIR_ID,
        "source_container": "outputs",
        "selection_rule": (
            "exact declared product-name bijection plus output-local regular "
            "file and physical-kind verification"
        ),
    }
    payload = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    temporary_fd = -1
    temporary_name = ""
    try:
        temporary_fd, temporary_name = tempfile.mkstemp(
            dir=out_dir,
            prefix=".step_summary.output_registry.",
            suffix=".tmp",
            text=True,
        )
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            temporary_fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, summary_path)
        temporary_name = ""
        directory_fd = os.open(out_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        return False
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
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

    If ``step_summary.json`` is absent, try stdout JSON then a named summary
    artefact (short-circuit), and signal that artefacts should be re-listed.
    An empty summary is deliberately not reconstructed from result tables:
    selecting a primary row or aggregating performance rows would make the
    deterministic layer choose the scientific headline.

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
    if _canonicalize_exact_declared_output_registry(
        step=step,
        out_dir=run_result.out_dir,
    ):
        return SummarySalvageOutcome(
            repair_id=_OUTPUT_REGISTRY_CANONICALIZATION_REPAIR_ID,
            trigger_reason=(
                "step_summary used an exact untyped outputs map for all "
                "Planner-declared products"
            ),
            transformation=(
                "Added a typed output_files registry from the step's exact "
                "legacy outputs map after verifying every output-local file."
            ),
            selection_rule=(
                "exact declared product-name bijection plus output-local regular "
                "file and physical-kind verification"
            ),
        )
    return None


__all__ = [
    "_extract_last_json_object",
    "_salvage_stdout_json_step_summary",
    "_salvage_named_json_step_summary",
    "_salvage_minimal_contract_step_summary",
    "_canonicalize_exact_declared_output_registry",
    "SummarySalvageOutcome",
    "salvage_step_summary",
]
