"""Compact registered descriptive aggregates for the completed-run answer.

The caller supplies the same identifier-free, digest-verified CSV snapshot
used for the display preview. Pivoting its named statistics preserves their
strings, units and observed denominators; it computes no scientific values.
Display row/column limits stay independent of this bounded, complete block.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

from easyicu.research_agent.execution.runners.typed_input_binding import contained_regular_file


SUMMARY_SCHEMA = "easyicu.web-descriptive-summary/1"
_COLUMNS = {"variable", "statistic", "level", "value", "numerator", "denominator",
            "percentage", "unit", "source_column"}
_STATISTICS = {"median": "median", "q1": "q25", "q3": "q75", "mean": "mean", "std": "sd"}
_MAX_SUMMARY_ROWS = 64


def bound_descriptive_variables(run_dir: Path, step_id: str, summary: dict) -> dict:
    """Project names and window metadata from this producer's bound context.

    The browser's concept catalog supplies localized names. Never copy context
    domains, patient identifiers, host paths or unverified window descriptions.
    """

    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", step_id):
        return {}
    root = run_dir.resolve()
    try:
        manifest_path = contained_regular_file(root / "resolved_inputs" / f"{step_id}.json", root)
        if manifest_path is None or manifest_path.stat().st_size > 1_500_000:
            return {}
        manifest = json.loads(manifest_path.read_bytes())
        if not isinstance(manifest, dict) or manifest.get("step_id") != step_id:
            return {}
        binding = manifest.get("context")
        if not isinstance(binding, dict):
            return {}
        relative, digest = binding.get("relative_path"), binding.get("sha256")
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
            return {}
        context_path = contained_regular_file(root / relative, root)
        if context_path is None or context_path.stat().st_size > 1_500_000:
            return {}
        raw = context_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            return {}
        context = json.loads(raw)
    except (OSError, ValueError):
        return {}
    variables = context.get("variables") if isinstance(context, dict) else None
    if not isinstance(variables, list):
        return {}
    names = {row.get(key) for row in summary["continuous"] + summary["categorical"]
             for key in ("variable", "group_variable")} - {None, ""}
    result = {}
    for variable in variables:
        name = variable.get("name") if isinstance(variable, dict) else None
        if not isinstance(name, str) or name not in names:
            continue
        if name in result:
            return {}
        result[name] = {key: variable[key] for key in (
            "source_concept", "description", "unit_normalization", "analysis_window")
            if isinstance(variable.get(key), str) and 0 < len(variable[key]) <= 500}
    return result


def _count(value: str) -> int | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if math.isfinite(number) and number.is_integer() and 0 <= number <= 2**53 - 1 else None


def _wide_summary(reader: csv.DictReader, count_column: str) -> dict | None:
    """Preserve the continuous-outcome producer's wide aggregates and units."""

    continuous = []
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            return None
        row = {key: value.strip() for key, value in raw.items()}
        if _count(row[count_column]) is None or any(len(value) > 500 for value in row.values()):
            return None
        continuous.append({
            "variable": row["variable"], "group": row["group"],
            "group_variable": row.get("group_variable", ""),
            "row_role": row.get("row_role") or "group",
            "n_nonmissing": row[count_column], "unit": row.get("unit", ""),
            **{key: row[key] for key in ("median", "q25", "q75", "mean", "sd") if key in row},
        })
        if len(continuous) > _MAX_SUMMARY_ROWS:
            return None
    return {"schema_version": SUMMARY_SCHEMA, "continuous": continuous, "categorical": []} if continuous else None


def descriptive_summary_projection(snapshot: str) -> dict | None:
    """Select complete per-variable/group aggregates, or omit an unsafe block.

    A frequency/distribution table is not a location summary. Only the
    producer's long-form or wide summary contract is accepted. Independent tables
    are never combined, even when they describe the same estimand.
    """

    reader = csv.DictReader(io.StringIO(snapshot, newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        return None
    count_column = next((key for key in ("n_nonmissing", "nonmissing_n", "n") if key in headers), "")
    if {"variable", "group"}.issubset(headers) and count_column and {"median", "mean"}.intersection(headers):
        return _wide_summary(reader, count_column)
    if not _COLUMNS.issubset(headers):
        return None
    partitions: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            return None
        row = {key: value.strip() for key, value in raw.items()}
        if row["variable"] == "__cohort__":
            continue
        group = row.get("group", "")
        role = row.get("row_role") or ("group" if "group" in row else "overall")
        key = (row["variable"], row.get("group_variable", ""), group, role)
        if not key[0] or any(len(value) > 500 for value in row.values()):
            return None
        partitions.setdefault(key, []).append(row)
        if len(partitions) > _MAX_SUMMARY_ROWS:
            return None

    continuous, categorical = [], []
    for (variable, group_variable, group, role), rows in partitions.items():
        identity = {"variable": variable, "group_variable": group_variable,
                    "group": group, "row_role": role}
        if len({row["unit"] for row in rows}) != 1:
            return None
        identity["unit"] = rows[0]["unit"]
        levels = [row for row in rows if row["statistic"] == "level_count"]
        statistics = [row for row in rows if row["statistic"] in _STATISTICS]
        if levels:
            if statistics or len({row["level"] for row in levels}) != len(levels):
                return None
            for row in levels:
                categorical.append({**identity, **{key: row[key] for key in (
                    "level", "numerator", "denominator", "percentage")}})
        elif statistics:
            counts = [row for row in rows if row["statistic"] == "nonmissing_n"]
            if len(counts) != 1 or _count(counts[0]["value"]) is None:
                return None
            count = _count(counts[0]["value"])
            if (len({row["statistic"] for row in statistics}) != len(statistics)
                    or any(_count(row["denominator"]) != count for row in statistics)):
                return None
            continuous.append({**identity, "n_nonmissing": counts[0]["value"],
                               **{_STATISTICS[row["statistic"]]: row["value"] for row in statistics}})
        if len(continuous) + len(categorical) > _MAX_SUMMARY_ROWS:
            return None
    return {"schema_version": SUMMARY_SCHEMA, "continuous": continuous, "categorical": categorical} if continuous or categorical else None
