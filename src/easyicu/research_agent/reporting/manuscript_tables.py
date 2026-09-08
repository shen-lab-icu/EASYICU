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


def _same(rows: Sequence[dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ManuscriptTableProjectionError(f"Table 1 inconsistent repeated {field}")
    return next(iter(values))


def _group_columns(plan, spec, source):
    """Pivot source identities, never human labels or recomputed statistics."""
    groups = ["Overall", *(str(level) for level in spec.group_levels)]
    if len(set(groups)) != len(groups) or {row["group"] for row in source} != set(groups):
        raise ManuscriptTableProjectionError("Table 1 group roster is ambiguous or incomplete")
    indexed = {}
    for row in source:
        key = (row["variable"], row["category"], row["group"])
        if key in indexed:
            raise ManuscriptTableProjectionError("Table 1 duplicate variable/category/group")
        indexed[key] = row
    # The source contract uses the same eligible population for every variable.
    # Reject disagreement rather than putting a misleading common N in the header.
    denominators = [
        _count(_same([r for r in source if r["group"] == group], "denominator_n"))
        for group in groups
    ]
    columns = ("Characteristic", *(plan.display_labels.get(
        f"{spec.group_by}={group}", group,
    ) for group in groups), "SMD")
    if spec.p_values_required:
        columns += ("P value",)
    blank = ("",) * (1 + int(spec.p_values_required))
    rows = [("N", *denominators, *blank)]
    for variable in spec.variables:
        name = variable.name
        label = plan.display_labels.get(name, name)
        categories = ([str(level) for level in variable.levels]
                      if variable.summary == "count_percent" else [""])
        expected = {(name, category, group) for category in categories for group in groups}
        if {key for key in indexed if key[0] == name} != expected:
            raise ManuscriptTableProjectionError("Table 1 category/group cells are incomplete")
        variable_rows = [row for row in source if row["variable"] == name]
        p = (_p_value(_same(variable_rows, "p_value")),) if spec.p_values_required else ()
        if variable.summary == "count_percent":
            rows.append((label + ", n (%)", *("" for _ in groups), "", *p))
        for category in categories:
            cells = [indexed[(name, category, group)] for group in groups]
            smd = _number(_same(cells, "standardized_mean_difference"), 3)
            if variable.summary == "count_percent":
                rows.append(("  " + category, *(_count_percent(r["count"], r["percentage"])
                                              for r in cells), smd,
                             *(("",) if spec.p_values_required else ())))
            else:
                if variable.summary in {"mean_sd", "both"}:
                    rows.append((label + ", mean (SD)", *(f"{_number(r['mean'])} ({_number(r['sd'])})"
                                                         for r in cells), smd, *p))
                if variable.summary in {"median_iqr", "both"}:
                    row_label = ("  Median [Q1, Q3]" if variable.summary == "both"
                                 else label + ", median [Q1, Q3]")
                    comparison = blank if variable.summary == "both" else (smd, *p)
                    rows.append((row_label, *(f"{_number(r['median'])} [{_number(r['q25'])}, {_number(r['q75'])}]"
                                             for r in cells), *comparison))
        missing = []
        for group in groups:
            group_rows = [r for r in variable_rows if r["group"] == group]
            missing.append(_count_percent(_same(group_rows, "missing_n"),
                                          _same(group_rows, "missing_pct")))
        rows.append(("  Missing, n (%)", *missing, *blank))
    return columns, rows


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
        try:
            columns, rows = _group_columns(plan, spec, source)
        except KeyError as exc:
            raise ManuscriptTableProjectionError("Table 1 required source field is missing") from exc
        exclusions = {row.get("group_missing_excluded_n", "") for row in source}
        if len(exclusions) != 1 or "" in exclusions:
            raise ManuscriptTableProjectionError("Table 1 grouping exclusions are not explicit")
        notes = [
            f"Grouping variable: {spec.group_by}. Columns use the executed table population.",
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
