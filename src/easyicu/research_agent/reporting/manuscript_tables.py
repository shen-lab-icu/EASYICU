"""Reader-only projection of current, contract-bound table source rows.

This owner formats registered summaries; it never reads the cohort or computes
new statistics. The executed plan chooses the Table 1 variables and summary
family; a signed owner declares its own reader tables in its step summary
(``contracts.manuscript_tables``), and only their recorded cells are shown.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..authority.evidence_store import evidence_artifact_basename_stem
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    GroupedSummaryLayout,
    ManuscriptTableDeclaration,
    StageFlowLayout,
    validate_manuscript_table_declarations,
)
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
    tables.extend(_declared_tables(plan=plan, evidence_records=evidence_records, run_dir=run_dir))
    return tuple(tables)


def _verified_bytes(run_dir: Path, record: EvidenceRecord) -> bytes:
    path = verified_run_evidence_path(run_dir, record)
    try:
        payload = None if path is None else path.read_bytes()
    except OSError as exc:
        raise ManuscriptTableProjectionError("A declared table source could not be read") from exc
    if payload is None or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ManuscriptTableProjectionError("A declared table source is missing or has drifted")
    return payload


def _step_summary(run_dir: Path, record: EvidenceRecord) -> dict:
    try:
        summary = json.loads(_verified_bytes(run_dir, record))
    except ValueError as exc:
        raise ManuscriptTableProjectionError("A step summary is not valid JSON") from exc
    return summary if isinstance(summary, dict) else {}


def _source_rows(run_dir: Path, record: EvidenceRecord) -> list[dict[str, str]]:
    if Path(record.relative_path).suffix.lower() != ".csv":
        raise ManuscriptTableProjectionError("A declared table source is not a CSV product")
    try:
        rows = list(csv.DictReader(_verified_bytes(run_dir, record).decode("utf-8").splitlines()))
    except (UnicodeError, csv.Error) as exc:
        raise ManuscriptTableProjectionError("A declared table source could not be read") from exc
    if not rows or any(
        not isinstance(key, str) or not isinstance(value, str)
        for row in rows for key, value in row.items()
    ):
        raise ManuscriptTableProjectionError("A declared table source has no well-formed rows")
    return rows


def _level_text(plan: AnalysisPlan, name: str, level: str) -> str:
    """A recorded level by the plan's name for it; an integral number reads as one."""

    text = level
    try:
        number = Decimal(level)
        if number.is_finite() and number == number.to_integral_value():
            text = str(int(number))
    except InvalidOperation:
        pass
    return plan.display_labels.get(f"{name}={text}") or text


def _grouped_summary(plan: AnalysisPlan, body: GroupedSummaryLayout, rows):
    groups = body.groups
    columns = ("Characteristic", *(f"{group.label} (n = {group.n})" for group in groups), "SMD")
    projected: list[tuple[str, ...]] = []
    opened: set[str] = set()
    for row in rows:
        name = row["variable"]
        label = plan.display_labels.get(name) or name.replace("_", " ")
        smd = _number(row["standardized_mean_difference"], 3)
        if row["summary_type"] == "categorical_n_percent":
            if name not in opened:
                projected.append((f"{label}, n (%)", *("" for _ in groups), ""))
                opened.add(name)
            projected.append((
                "  " + _level_text(plan, name, row["level"]),
                *(_count_percent(row[f"{g.prefix}_n"], row[f"{g.prefix}_percent"]) for g in groups),
                smd,
            ))
        elif row["summary_type"] == "continuous_mean_sd":
            projected.append((
                f"{label}, mean (SD)",
                *(f"{_number(row[f'{g.prefix}_mean'])} ({_number(row[f'{g.prefix}_sd'])})" for g in groups),
                smd,
            ))
            projected.append((
                "  Median [Q1, Q3]",
                *(
                    f"{_number(row[f'{g.prefix}_median'])} "
                    f"[{_number(row[f'{g.prefix}_q1'])}, {_number(row[f'{g.prefix}_q3'])}]"
                    for g in groups
                ),
                "",
            ))
        else:
            raise ManuscriptTableProjectionError("A declared summary row has an unknown summary type")
    if body.events_label is not None:
        projected.append((
            body.events_label,
            *(_count_percent(str(group.events), repr(group.events_percent)) for group in groups),
            "",
        ))
    return columns, projected


def _stage_flow(body: StageFlowLayout, rows):
    try:
        ordered = sorted(rows, key=lambda row: int(row["stage_order"]))
    except ValueError as exc:
        raise ManuscriptTableProjectionError("A declared stage order is not an integer") from exc
    if [int(row["stage_order"]) for row in ordered] != list(range(1, len(ordered) + 1)):
        raise ManuscriptTableProjectionError("Declared stages are not consecutive")
    projected = []
    for index, row in enumerate(ordered):
        label = body.stage_labels.get(row["stage"])
        if label is None:
            raise ManuscriptTableProjectionError("A recorded stage has no declared reader label")
        excluded = "" if index == 0 else _count(row["excluded_since_prior_stage"])
        projected.append((label, _count(row["count"]), excluded))
    return ("Stage", "Records", "Excluded"), projected


def _declared_table(
    plan: AnalysisPlan, declaration: ManuscriptTableDeclaration, source: EvidenceRecord, run_dir: Path,
) -> ManuscriptTable:
    rows = _source_rows(run_dir, source)
    try:
        if isinstance(declaration.body, GroupedSummaryLayout):
            columns, projected = _grouped_summary(plan, declaration.body, rows)
        else:
            columns, projected = _stage_flow(declaration.body, rows)
    except KeyError as exc:
        raise ManuscriptTableProjectionError("A declared table source lacks a required field") from exc
    return ManuscriptTable(
        caption=declaration.caption, columns=columns, rows=tuple(projected),
        notes=(*declaration.notes, f"Source SHA-256: {source.sha256}"),
    )


def _declared_tables(
    *,
    plan: AnalysisPlan,
    evidence_records: Sequence[EvidenceRecord],
    run_dir: Path,
) -> list[ManuscriptTable]:
    """Project each reader table a plan step's signed owner declared, in plan order."""

    tables: list[ManuscriptTable] = []
    for step in plan.steps:
        summaries = [
            _step_summary(run_dir, record) for record in evidence_records
            if record.kind == "statistic"
            and record.generation_mode == "deterministic_standard"
            and record.produced_by_step == step.step_id
            and Path(record.relative_path).name.endswith("step_summary.json")
        ]
        declaring = [summary for summary in summaries if summary.get(MANUSCRIPT_TABLES_KEY) is not None]
        if not declaring:
            continue
        # Read without its step ledger, a store also keeps a re-executed
        # step's earlier summary, and which one is current is unknown.
        if len(summaries) > 1:
            raise ManuscriptTableProjectionError(
                f"Step {step.step_id!r} declares reader tables but has more than one step summary"
            )
        summary = declaring[0]
        raw = summary[MANUSCRIPT_TABLES_KEY]
        try:
            declarations = validate_manuscript_table_declarations(raw)
        except ValueError as exc:
            raise ManuscriptTableProjectionError("A declared reader table is malformed") from exc
        output_files = summary.get("output_files")
        for declaration in declarations:
            filename = output_files.get(declaration.product) if isinstance(output_files, dict) else None
            if not isinstance(filename, str):
                raise ManuscriptTableProjectionError("A declared reader table is not an output of its step")
            sources = [
                record for record in evidence_records
                if record.kind == "table" and record.produced_by_step == step.step_id
                and evidence_artifact_basename_stem(
                    Path(record.relative_path), record.evidence_id,
                ) == Path(filename).stem
            ]
            if len(sources) != 1:
                raise ManuscriptTableProjectionError(
                    f"Declared table {declaration.product!r} requires one current registered source"
                )
            tables.append(_declared_table(plan, declaration, sources[0], run_dir))
    return tables
