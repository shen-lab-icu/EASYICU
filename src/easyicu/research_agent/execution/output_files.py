"""Filesystem-only output helpers shared by pipeline and execute layers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


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


def _contains_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_finite_number(item) for item in value)
    return False


def normalize_typed_statistic_sidecars(
    step_summary: Any,
    out_dir: Path,
) -> list[Dict[str, str]]:
    """Canonicalize exact typed-statistic JSON outputs once at the host edge.

    The generated analysis and all numeric values remain unchanged.  A missing
    in-payload product identity is added only when ``output_files`` binds one
    exact ``statistic:<name>`` to one safe local JSON file containing finite
    numeric data. Conflicting identities, symlinks, dynamic paths, invalid JSON,
    and nonnumeric payloads remain untouched so the downstream typed gate fails
    closed.
    """

    if not isinstance(step_summary, Mapping):
        return []
    output_files = step_summary.get("output_files")
    if not isinstance(output_files, Mapping):
        return []
    receipts: list[Dict[str, str]] = []
    for raw_product, raw_path in sorted(
        output_files.items(), key=lambda item: str(item[0])
    ):
        product = str(raw_product or "").strip()
        relative = str(raw_path or "").strip()
        if (
            not product.startswith("statistic:")
            or product.count(":") != 1
            or not product.split(":", 1)[1]
            or not relative
            or Path(relative).name != relative
            or Path(relative).suffix.lower() != ".json"
        ):
            continue
        statistic_name = product.split(":", 1)[1]
        source = out_dir / relative
        if not source.is_file() or source.is_symlink():
            continue
        try:
            before_bytes = source.read_bytes()
            payload = json.loads(before_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not _contains_finite_number(payload):
            continue
        declared_name = payload.get("name") or payload.get("statistic")
        if declared_name is not None:
            continue
        normalized = {"name": statistic_name, **payload}
        after_text = json.dumps(
            normalized,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        after_bytes = (after_text + "\n").encode("utf-8")
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=out_dir,
                prefix=f".{source.name}.",
                suffix=".normalize",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                handle.write(after_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, source)
            temporary_path = None
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)
        receipts.append(
            {
                "product": product,
                "path": relative,
                "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
                "after_sha256": hashlib.sha256(after_bytes).hexdigest(),
            }
        )
    return receipts


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


__all__ = [
    "_clear_output_dir",
    "_has_figure_exports",
    "bind_primary_output",
    "normalize_typed_statistic_sidecars",
]
