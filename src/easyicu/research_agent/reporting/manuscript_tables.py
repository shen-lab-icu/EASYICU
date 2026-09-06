"""Reader-only projection of current, contract-bound Table 1 source rows.

This owner formats registered summaries; it never reads the cohort or computes
new statistics. The executed plan chooses the variables and summary family.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from ..authority.evidence_store import evidence_artifact_basename_stem
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..methods.table_one import table_one_spec_sha256
from ..schema import AnalysisPlan, EvidenceRecord


class ManuscriptTableProjectionError(ValueError):
    """A planned reader table has no exact, supported source."""


@dataclass(frozen=True)
class ManuscriptTable:
    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...] = ()


def _number(value: str, places: int = 2) -> str:
    if value == "":
        return "N/A"
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ManuscriptTableProjectionError("Non-numeric table summary") from exc
    if not number.is_finite():
        raise ManuscriptTableProjectionError("Non-finite table summary")
    return format(number, f".{places}f")


def _percent(value: str) -> str:
    formatted = _number(value, 1)
    if value and not 0 <= Decimal(value) <= 100:
        raise ManuscriptTableProjectionError("Table percentage is outside [0, 100]")
    if value and 0 < Decimal(value) < Decimal("0.1"):
        return "<0.1"
    if value and Decimal("99.9") < Decimal(value) < 100:
        return ">99.9"
    return formatted


def _count(value: str) -> str:
    formatted = _number(value, 0)
    if value and (Decimal(value) < 0 or Decimal(value) != Decimal(value).to_integral_value()):
        raise ManuscriptTableProjectionError("Table count is not a nonnegative integer")
    return formatted


def _count_percent(count: str, percentage: str) -> str:
    formatted = _percent(percentage)
    unit = "" if formatted == "N/A" else "%"
    return f"{_count(count)} ({formatted}{unit})"


def _p_value(value: str) -> str:
    formatted = _number(value, 3)
    if value and not 0 <= Decimal(value) <= 1:
        raise ManuscriptTableProjectionError("Table P value is outside [0, 1]")
    return "<0.001" if value and 0 <= Decimal(value) < Decimal("0.001") else formatted


def build_manuscript_tables(
    *,
    plan: AnalysisPlan,
    evidence_records: Sequence[EvidenceRecord],
    run_dir: Path,
) -> tuple[ManuscriptTable, ...]:
    """Project the exact Table 1 owner from the current verified record set."""

    tables: list[ManuscriptTable] = []
    for step in plan.steps:
        spec = step.table_one_spec
        if spec is None:
            continue
        records = [
            record for record in evidence_records
            if record.kind == "table" and record.produced_by_step == step.step_id
            and evidence_artifact_basename_stem(
                Path(record.relative_path), record.evidence_id,
            ) == "table_one"
        ]
        if len(records) != 1:
            raise ManuscriptTableProjectionError(
                f"Table 1 step {step.step_id!r} requires one current registered source"
            )
        record = records[0]
        path = verified_run_evidence_path(run_dir, record)
        if path is None or path.suffix.lower() != ".csv":
            raise ManuscriptTableProjectionError("Table 1 source is missing or has drifted")
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                source = list(csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ManuscriptTableProjectionError("Table 1 source could not be read") from exc
        if any(not isinstance(value, str) for row in source for value in row.values()):
            raise ManuscriptTableProjectionError("Table 1 CSV row width is invalid")
        expected_digest = table_one_spec_sha256(spec)
        if not source or any(
            row.get("schema_version") != "easyicu.table_one_result/3"
            or row.get("contract_sha256") != expected_digest for row in source
        ):
            raise ManuscriptTableProjectionError("Table 1 result/plan contract mismatch")
        variables = {item.name: item for item in spec.variables}
        if {row.get("variable") for row in source} != set(variables):
            raise ManuscriptTableProjectionError("Table 1 variable roster mismatch")
        summaries = {item.summary for item in spec.variables}
        header = {
            "median_iqr": "Median [Q1, Q3]", "mean_sd": "Mean (SD)",
            "count_percent": "n (%)",
        }.get(next(iter(summaries)), "Summary") if len(summaries) == 1 else "Summary"
        columns = ("Variable", "Group", "N", header, "Missing n (%)", "SMD")
        if spec.p_values_required:
            columns += ("P value",)
        rows: list[tuple[str, ...]] = []
        try:
            for row in source:
                variable = variables[row["variable"]]
                label = plan.display_labels.get(variable.name, variable.name)
                summary_parts = []
                if variable.summary == "count_percent":
                    label += ": " + row["category"]
                    summary_parts.append(
                        _count_percent(row["count"], row["percentage"])
                    )
                if variable.summary in {"mean_sd", "both"}:
                    summary_parts.append(f"{_number(row['mean'])} ({_number(row['sd'])})")
                if variable.summary in {"median_iqr", "both"}:
                    summary_parts.append(
                        f"{_number(row['median'])} [{_number(row['q25'])}, {_number(row['q75'])}]"
                    )
                cells = (
                    label, row["group"], _count(row["denominator_n"]),
                    "; ".join(summary_parts),
                    _count_percent(row["missing_n"], row["missing_pct"]),
                    _number(row["standardized_mean_difference"], 3),
                )
                if spec.p_values_required:
                    cells += (_p_value(row["p_value"]),)
                rows.append(cells)
        except KeyError as exc:
            raise ManuscriptTableProjectionError("Table 1 required source field is missing") from exc
        exclusions = {row.get("group_missing_excluded_n", "") for row in source}
        if len(exclusions) != 1 or "" in exclusions:
            raise ManuscriptTableProjectionError("Table 1 grouping exclusions are not explicit")
        notes = [
            f"Grouping variable: {spec.group_by}. Rows are the executed table population.",
            "Categorical percentages use non-missing observations; missing percentages use N.",
            "Numeric summaries follow the approved plan: mean (SD), median [Q1, Q3], or both.",
            "Signed SMD is comparison minus reference; it is not a significance test. N/A means unavailable.",
            f"Rows excluded for missing grouping value: {_count(next(iter(exclusions)))}.",
            f"Source SHA-256: {record.sha256}",
        ]
        if len(spec.group_levels) == 2:
            notes.append(f"SMD reference: {spec.group_levels[0]}; comparison: {spec.group_levels[1]}.")
        for level in spec.group_levels:
            label = plan.display_labels.get(f"{spec.group_by}={level}")
            if label:
                notes.append(f"Group {level}: {label}.")
        if not spec.p_values_required:
            notes.append("No inferential P values were planned or added by this reader.")
        tables.append(ManuscriptTable(
            caption="Baseline characteristics", columns=columns, rows=tuple(rows), notes=tuple(notes),
        ))
    return tuple(tables)
