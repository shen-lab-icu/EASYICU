"""Filesystem-only output helpers shared by pipeline and execute layers."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict


def _has_figure_exports(out_dir: Path) -> bool:
    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    return any(
        path.is_file() and path.suffix.lower() in figure_suffixes
        for path in out_dir.iterdir()
    )


def _clear_output_dir(out_dir: Path) -> None:
    """Recreate a step output directory without following untrusted symlinks."""

    # Generated code may replace the output leaf itself with a symlink.  Using
    # ``exists``/``iterdir`` first would follow that link and could delete an
    # arbitrary host directory during repair.  Remove any non-directory leaf
    # lexically, then create the expected directory in its place.
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_symlink() or not child.is_dir():
            child.unlink(missing_ok=True)
        else:
            shutil.rmtree(child)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bind_registered_primary_or_statistic(
    payload: Dict[str, Any],
    outputs: Any,
    out_dir: Path,
) -> None:
    relative = (
        outputs.get("statistic:primary_or") if isinstance(outputs, dict) else None
    )
    if (
        not isinstance(relative, str)
        or Path(relative).name != relative
        or Path(relative).suffix.lower() != ".json"
    ):
        return
    source = out_dir / relative
    if not source.is_file() or source.is_symlink():
        return
    try:
        statistic = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(statistic, dict):
        return
    declared_name = str(
        statistic.get("name") or statistic.get("statistic") or ""
    ).strip()
    if declared_name != "primary_or":
        return
    estimates = [
        number
        for key in ("value", "estimate", "result")
        if key in statistic
        and (number := _finite_number(statistic.get(key))) is not None
    ]
    if not estimates or any(
        not math.isclose(value, estimates[0], rel_tol=1e-12, abs_tol=1e-12)
        for value in estimates[1:]
    ):
        return
    estimate = estimates[0]
    if estimate <= 0:
        return
    low = _finite_number(statistic.get("ci_low"))
    high = _finite_number(statistic.get("ci_high"))
    interval_declared = "ci_low" in statistic or "ci_high" in statistic
    if interval_declared and (
        low is None or high is None or not (0 < low <= estimate <= high)
    ):
        return
    payload.update(
        primary_estimate=estimate,
        primary_estimate_label="odds_ratio",
        primary_or=estimate,
    )
    if low is not None and high is not None:
        payload.update(
            primary_estimate_interval=[low, high],
            primary_or_ci=[low, high],
        )


def bind_primary_output(step_summary: Any, out_dir: Path) -> Dict[str, Any]:
    """Bind one registered adjusted-association row into canonical scalars."""

    payload = (
        dict(step_summary) if isinstance(step_summary, dict) else {"raw": step_summary}
    )
    outputs = payload.get("output_files")
    _bind_registered_primary_or_statistic(payload, outputs, out_dir)
    relative = (
        outputs.get("table:adjusted_association_estimates")
        if isinstance(outputs, dict)
        else None
    )
    if not isinstance(relative, str) or Path(relative).name != relative:
        return payload
    source = out_dir / relative
    if not source.is_file() or source.is_symlink():
        return payload
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0].get("fit_status") != "fitted":
            return payload
        row = rows[0]
        estimate, low, high = map(
            float, (row["estimate"], row["ci_low"], row["ci_high"])
        )
        if not all(math.isfinite(value) for value in (estimate, low, high)):
            return payload
    except (KeyError, OSError, TypeError, ValueError):
        return payload
    scale = str(row.get("effect_scale") or "estimate").strip()
    payload.update(
        primary_estimate=estimate,
        primary_estimate_label=scale,
        primary_estimate_interval=[low, high],
        primary_association_term=str(row.get("exposure") or "").strip() or None,
    )
    if scale == "odds_ratio":
        payload.update(primary_or=estimate, primary_or_ci=[low, high])
    return payload


__all__ = ["_clear_output_dir", "_has_figure_exports", "bind_primary_output"]
