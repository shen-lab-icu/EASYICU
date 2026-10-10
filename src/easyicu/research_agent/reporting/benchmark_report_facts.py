"""The host's sentences for a prediction compared with an existing score.

Owner
-----
A prediction study that compares its model with an existing score on the
same validation stays (``prediction.benchmark_comparison``, run by the static
prediction owner) records, per comparator, both AUROCs with their intervals,
their paired difference with its interval, and the Brier scores side by side
when calibration was compared.  This module states that record in fixed
Results sentences, so the comparison the question asked for does not depend
on the Writer quoting a table.

Every number a sentence prints is read twice: from the step's comparison
table (``benchmark_comparison.csv``) and from its summary
(``reportable_benchmark_comparison``), each verified against its registered
digest.  The two must be equal, or the report stops
(:data:`BENCHMARK_REPORT_SOURCE_MISMATCH`): the sentence never chooses
between them.  It cites the summary, whose leaves the numeric binder
registered, so STRICT binding checks every printed number.

No P value is printed: the interval states the inference, and a P value
below the smallest printable one could not be bound.  A comparison the
question asked for (its planning record's benchmark requirement is judged
covered by this step on the executed plan, ``planning.question_requirements``)
is stated in the Abstract's results as well.

When the comparator's value was computed over a window that does not end at
the model's prediction time, the comparison credits one side with
information the other does not use.  A fixed Limitations sentence says so
(:func:`benchmark_window_limitations`); the write phase places it with the
source facts and fails closed when it did not survive binding
(:func:`audit_bound_benchmark_limitations`).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import pandas as pd

from ..authority.evidence_store import evidence_artifact_basename_stem
from ..authority.runtime_artifacts import (
    current_successful_step_records,
    verified_run_evidence_path,
)
from ..contracts.manuscript_result_structure import RESULT_HEADINGS_BY_ROLE
from ..contracts.prediction_execution import PREDICTION_BENCHMARK_PRODUCT
from ..research_context.materialization_window import column_window_from_label
from ..schema import ValidationFinding
from .manuscript_method_facts import missing_bound_method_facts
from .report_fact import DescriptiveReportFact

BENCHMARK_REPORT_SOURCE_MISMATCH = "benchmark_report_source_mismatch"
BENCHMARK_REPORT_SOURCE_UNREADABLE = "benchmark_report_source_unreadable"
BENCHMARK_REPORT_LABEL_UNSAFE = "benchmark_report_label_unsafe"
#: The planning record and the executed plan disagree on whether the question's
#: benchmark is answered: the plan drifted after its requirements were judged.
BENCHMARK_REPORT_REQUIREMENT_DRIFT = "benchmark_report_requirement_drift"
BENCHMARK_LIMITATIONS_MISSING = "writer_benchmark_limitations_missing"
REPORTABLE_KEY = "reportable_benchmark_comparison"

#: The table's AUROC row and the summary field each of its values must equal.
_AUROC_FIELDS = (
    ("model_value", ("model", "auroc")),
    ("model_ci_low", ("model", "auroc_ci_low")),
    ("model_ci_high", ("model", "auroc_ci_high")),
    ("comparator_value", ("comparator", "auroc")),
    ("comparator_ci_low", ("comparator", "auroc_ci_low")),
    ("comparator_ci_high", ("comparator", "auroc_ci_high")),
    ("difference", ("auroc_difference",)),
    ("difference_ci_low", ("auroc_difference_ci_low",)),
    ("difference_ci_high", ("auroc_difference_ci_high",)),
)
_COUNT_FIELDS = ("validation_n", "comparator_missing_n", "comparison_n", "comparison_event_n")
_BRIER_FIELDS = (
    ("model_value", ("model", "brier_score")),
    ("comparator_value", ("comparator", "brier_score")),
)
#: What a reader label may not hold: markup, placeholders, or a number the
#: binder would read as a result.
_UNSAFE_LABEL = re.compile(r"[{}\[\]`<>\\]|\d{2,}|\d\.\d")


class BenchmarkReportError(ValueError):
    """The comparison cannot be stated; ``code`` says why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class _Comparison:
    step_id: str
    evidence_id: str
    source_sha256: str
    position: int
    summary: Mapping[str, Any]
    prediction_time_hours: float
    #: The table values each sentence prints, equal to the summary's.
    auroc: Mapping[str, float]
    counts: Mapping[str, int]
    brier: Optional[Mapping[str, float]]

    @property
    def prefix(self) -> str:
        return f"{REPORTABLE_KEY}.comparisons[{self.position}]"

    @property
    def column(self) -> str:
        return str(self.summary["comparator_column"])


# -- reading the verified record ----------------------------------------------


def _summary_value(comparison: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = comparison
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise BenchmarkReportError(
                BENCHMARK_REPORT_SOURCE_UNREADABLE,
                f"the comparison summary has no {'.'.join(path)!r}",
            )
        value = value[key]
    return value


def _equal_number(table: Any, summary: Any, *, integer: bool, what: str) -> Any:
    """The table's value, when the summary records exactly the same number."""

    if integer:
        if type(summary) is not int or isinstance(table, float) and not table.is_integer():
            raise BenchmarkReportError(
                BENCHMARK_REPORT_SOURCE_MISMATCH, f"{what} is not a recorded count"
            )
        recorded = int(table)
    else:
        if type(summary) not in {int, float} or isinstance(summary, bool):
            raise BenchmarkReportError(
                BENCHMARK_REPORT_SOURCE_MISMATCH, f"{what} is not a recorded number"
            )
        recorded = float(table)
        if not math.isfinite(recorded):
            raise BenchmarkReportError(
                BENCHMARK_REPORT_SOURCE_MISMATCH, f"{what} is not finite in the table"
            )
    if recorded != summary:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_MISMATCH,
            f"{what} is {recorded!r} in the table but {summary!r} in the summary",
        )
    return recorded


def _verified_bytes(root: Path, record: Any) -> bytes:
    path = verified_run_evidence_path(root, record)
    try:
        payload = None if path is None else Path(path).read_bytes()
    except OSError as exc:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, f"{record.evidence_id!r} became unreadable"
        ) from exc
    if payload is None or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE,
            f"{record.evidence_id!r} is missing or changed since it was registered",
        )
    return payload


def _verified_summary(evidence: Any, row: Mapping[str, Any]) -> tuple[Any, dict]:
    evidence_id = str(row.get("step_summary_evidence_id") or "")
    record = evidence.get(evidence_id)
    if (
        record is None
        or record.evidence_id != evidence_id
        or record.kind != "statistic"
        or record.produced_by_step != row.get("step_id")
    ):
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE,
            f"the comparison summary {evidence_id!r} is not its step's registered statistic",
        )
    try:
        summary = json.loads(_verified_bytes(evidence.root, record))
    except ValueError as exc:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, f"{evidence_id!r} is not JSON"
        ) from exc
    if not isinstance(summary, dict):
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, f"{evidence_id!r} is not a JSON object"
        )
    return record, summary


def _verified_table(evidence: Any, step_id: str, summary: Mapping[str, Any]) -> pd.DataFrame:
    """The step's comparison table, found as its registered table evidence."""

    outputs = summary.get("output_files")
    filename = outputs.get(PREDICTION_BENCHMARK_PRODUCT) if isinstance(outputs, Mapping) else None
    if not isinstance(filename, str) or not filename.strip():
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, "the comparison summary names no table"
        )
    sources = [
        record
        for record in evidence.records()
        if record.kind == "table"
        and record.produced_by_step == step_id
        and evidence_artifact_basename_stem(Path(record.relative_path), record.evidence_id)
        == Path(filename).stem
    ]
    if len(sources) != 1:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE,
            f"step {step_id!r} registers {len(sources)} comparison tables, not one",
        )
    try:
        # Read back exactly the floats the executor wrote.
        return pd.read_csv(
            io.BytesIO(_verified_bytes(evidence.root, sources[0])),
            float_precision="round_trip",
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, f"step {step_id!r}'s comparison table is not a CSV"
        ) from exc


def _one_row(table: pd.DataFrame, column: str, metric: str) -> Mapping[str, Any]:
    rows = table.loc[
        table["comparator_column"].astype(str).eq(column) & table["metric"].astype(str).eq(metric)
    ]
    if len(rows) != 1:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_MISMATCH,
            f"the table holds {len(rows)} {metric} rows for {column!r}, not one",
        )
    return {key: (value.item() if hasattr(value, "item") else value)
            for key, value in rows.iloc[0].items()}


def _comparisons(records: Sequence[Mapping[str, Any]], evidence: Any) -> Iterator[_Comparison]:
    """Each recorded comparison, with the table's values equal to its summary's."""

    for row in current_successful_step_records(records):
        projected = row.get("step_summary")
        if not isinstance(projected, Mapping) or REPORTABLE_KEY not in projected:
            continue
        step_id = str(row.get("step_id") or "")
        record, summary = _verified_summary(evidence, row)
        reportable = summary.get(REPORTABLE_KEY)
        comparisons = reportable.get("comparisons") if isinstance(reportable, Mapping) else None
        if not isinstance(comparisons, list) or not comparisons:
            raise BenchmarkReportError(
                BENCHMARK_REPORT_SOURCE_UNREADABLE,
                f"step {step_id!r} records no comparison",
            )
        prediction_time = reportable.get("prediction_time_hours")
        if type(prediction_time) not in {int, float} or not math.isfinite(prediction_time):
            raise BenchmarkReportError(
                BENCHMARK_REPORT_SOURCE_UNREADABLE,
                f"step {step_id!r} records no prediction time",
            )
        table = _verified_table(evidence, step_id, summary)
        for position, comparison in enumerate(comparisons):
            column = str(_summary_value(comparison, ("comparator_column",)))
            auroc_row = _one_row(table, column, "auroc")
            auroc = {
                field: _equal_number(
                    auroc_row[field], _summary_value(comparison, path),
                    integer=False, what=f"{column!r} {field}",
                )
                for field, path in _AUROC_FIELDS
            }
            counts = {
                field: _equal_number(
                    auroc_row[field], _summary_value(comparison, (field,)),
                    integer=True, what=f"{column!r} {field}",
                )
                for field in _COUNT_FIELDS
            }
            if counts["comparison_n"] + counts["comparator_missing_n"] != counts["validation_n"]:
                raise BenchmarkReportError(
                    BENCHMARK_REPORT_SOURCE_MISMATCH,
                    f"{column!r}'s compared and missing stays do not make its validation stays",
                )
            brier = None
            if _summary_value(comparison, ("calibration_status",)) == "compared":
                brier_row = _one_row(table, column, "brier_score")
                brier = {
                    field: _equal_number(
                        brier_row[field], _summary_value(comparison, path),
                        integer=False, what=f"{column!r} Brier {field}",
                    )
                    for field, path in _BRIER_FIELDS
                }
            yield _Comparison(
                step_id=step_id,
                evidence_id=record.evidence_id,
                source_sha256=record.sha256,
                position=position,
                summary=comparison,
                prediction_time_hours=float(prediction_time),
                auroc=auroc,
                counts=counts,
                brier=brier,
            )


# -- the Results and Abstract sentences ----------------------------------------


def _three(value: float) -> str:
    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def _reader_labels(
    context: Any, reader_display_labels: Mapping[str, str], language: str
) -> Mapping[str, str]:
    """The report's labels, with each unlabeled variable's English description."""

    from .manuscript_labels import source_bound_manuscript_labels

    return source_bound_manuscript_labels(
        context, reader_display_labels, language=language, include_unlabeled=True
    )


def _reader_label(labels: Mapping[str, str], column: str) -> str:
    label = " ".join(str(labels.get(column) or "").split())
    if not label or label == column or _UNSAFE_LABEL.search(label):
        raise BenchmarkReportError(
            BENCHMARK_REPORT_LABEL_UNSAFE,
            f"comparator {column!r} has no reader label a sentence can carry",
        )
    return label


def _question_named(evidence: Any, plan: Any) -> frozenset[tuple[str, str]]:
    """(step, comparator concept or column) pairs a covered benchmark requirement names."""

    from ..planning.question_requirements import (
        QUESTION_REQUIREMENTS_FILENAME,
        QuestionRequirementsError,
        judge_recorded_requirements,
    )

    source = Path(evidence.root) / QUESTION_REQUIREMENTS_FILENAME
    if plan is None or not source.is_file():
        return frozenset()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, "the planning record cannot be read"
        ) from exc
    try:
        judged = judge_recorded_requirements(record, plan=plan).judged
    except QuestionRequirementsError as exc:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE, "the planning record cannot be read"
        ) from exc
    planned = {
        str(row.get("id")): row.get("disposition")
        for row in record.get("judged") or ()
        if isinstance(row, Mapping)
    }
    named: set[tuple[str, str]] = set()
    for entry in judged:
        if entry.requirement.kind != "benchmark":
            continue
        # The executed plan is the one judged; a verdict it changes is drift,
        # never a silent sentence left out of the Abstract.
        was = planned.get(str(entry.requirement.id))
        if (was == "covered") != (entry.disposition == "covered"):
            raise BenchmarkReportError(
                BENCHMARK_REPORT_REQUIREMENT_DRIFT,
                f"benchmark requirement {entry.requirement.id!r} was {was!r} when "
                f"planned and is {entry.disposition!r} on the executed plan",
            )
        if entry.disposition != "covered":
            continue
        for step_id in entry.owner_step_ids:
            named.update((step_id, str(concept)) for concept in entry.requirement.concepts)
    return frozenset(named)


def _auroc_clause(comparison: _Comparison, label: str) -> str:
    value = comparison.auroc
    return (
        f"the model's AUROC was {_three(value['model_value'])} (95% CI "
        f"{_three(value['model_ci_low'])} to {_three(value['model_ci_high'])}) and the "
        f"AUROC of {label} was {_three(value['comparator_value'])} (95% CI "
        f"{_three(value['comparator_ci_low'])} to {_three(value['comparator_ci_high'])}); "
        # "AUROCs" keeps the difference from reading as a third AUROC.
        f"the difference between the two AUROCs (the model's minus that of {label}) was "
        f"{_three(value['difference'])} (95% CI {_three(value['difference_ci_low'])} to "
        f"{_three(value['difference_ci_high'])})"
    )


def _comparison_text(comparison: _Comparison, label: str) -> tuple[str, tuple[str, ...]]:
    counts = comparison.counts
    if counts["comparator_missing_n"] == 0:
        text = (
            f"In the {counts['validation_n']:,} validation stays "
            f"({counts['comparison_event_n']:,} events), " + _auroc_clause(comparison, label)
        )
        count_fields = ("validation_n", "comparison_event_n")
    else:
        text = (
            f"{label[:1].upper()}{label[1:]} was recorded for {counts['comparison_n']:,} of "
            f"the {counts['validation_n']:,} validation stays "
            f"({counts['comparison_event_n']:,} events among them), and the comparison used "
            "these stays: " + _auroc_clause(comparison, label)
        )
        count_fields = ("comparison_n", "validation_n", "comparison_event_n")
    fields = (
        *count_fields,
        *(".".join(path) for _field, path in _AUROC_FIELDS),
    )
    return text, tuple(f"{comparison.prefix}.{field}" for field in fields)


def compile_benchmark_report_facts(
    records: Sequence[Mapping[str, Any]],
    *,
    evidence: Any,
    reader_display_labels: Mapping[str, str],
    context: Any = None,
    manuscript_language: str = "en",
    plan: Any = None,
) -> tuple[DescriptiveReportFact, ...]:
    """The Results (and, when the question asked, Abstract) sentences of each comparison.

    ``plan`` is the executed plan the run's planning record is judged on;
    without one no comparison counts as asked.
    """

    named: Optional[frozenset[tuple[str, str]]] = None
    labels: Mapping[str, str] = {}
    facts: list[DescriptiveReportFact] = []
    for comparison in _comparisons(records, evidence):
        if named is None:
            named = _question_named(evidence, plan)
            labels = _reader_labels(context, reader_display_labels, manuscript_language)
        label = _reader_label(labels, comparison.column)
        role = _step_role(plan, comparison.step_id)
        subsection = RESULT_HEADINGS_BY_ROLE.get(role, RESULT_HEADINGS_BY_ROLE["secondary"])
        asked = any(
            (comparison.step_id, name) in named
            for name in (comparison.column, str(comparison.summary.get("comparator_concept") or ""))
        )
        text, fields = _comparison_text(comparison, label)
        facts.append(DescriptiveReportFact(
            subsection=subsection,
            text=text,
            evidence_id=comparison.evidence_id,
            source_sha256=comparison.source_sha256,
            source_fields=fields,
            required_result_sections=("Abstract", "Results") if asked else ("Results",),
        ))
        if comparison.brier is not None:
            facts.append(DescriptiveReportFact(
                subsection=subsection,
                text=(
                    f"On the same stays, the Brier score was "
                    f"{_three(comparison.brier['model_value'])} for the model and "
                    f"{_three(comparison.brier['comparator_value'])} for {label}"
                ),
                evidence_id=comparison.evidence_id,
                source_sha256=comparison.source_sha256,
                source_fields=tuple(
                    f"{comparison.prefix}.{'.'.join(path)}" for _field, path in _BRIER_FIELDS
                ),
                required_result_sections=("Results",),
            ))
    return tuple(facts)


def _step_role(plan: Any, step_id: str) -> str:
    for step in getattr(plan, "steps", None) or ():
        if getattr(step, "step_id", None) == step_id:
            return str(getattr(step, "planned_analysis_role", None) or "secondary")
    return "secondary"


# -- the Limitations sentence --------------------------------------------------


@dataclass(frozen=True)
class BenchmarkWindowLimitation:
    """The window a comparator was computed over, as Limitations states it."""

    text: str
    evidence_id: str
    source_sha256: str
    field: str
    section: str = "limitations"

    @property
    def source_field(self) -> str:
        return f"{self.evidence_id}.{self.field}"

    @property
    def scaffold(self) -> str:
        return f"{self.text} {{evidence:{self.evidence_id}}}."


def _hours(value: float) -> str:
    return f"{value:g}"


def _window_text(comparison: _Comparison, label: str) -> Optional[str]:
    relation = str(comparison.summary.get("information_window_relation") or "")
    if relation == "same":
        return None
    at = _hours(comparison.prediction_time_hours)
    if relation == "comparator_window_unstated":
        return (
            f"The window over which {label} is computed is not recorded, so whether the "
            f"comparison credits {label} or the model, whose prediction time is {at} h "
            "after ICU admission, with information the other does not use is unknown"
        )
    window = column_window_from_label(comparison.summary.get("comparator_information_window"))
    if window is None:
        raise BenchmarkReportError(
            BENCHMARK_REPORT_SOURCE_UNREADABLE,
            f"{comparison.column!r} records a window relation without its window",
        )
    until = _hours(window.end_hours)
    if relation == "comparator_ends_after_prediction_time":
        return (
            f"{label[:1].upper()}{label[1:]} is computed from data recorded up to {until} h "
            f"after ICU admission, while the model's prediction time is {at} h, so the "
            f"comparison credits {label} with information the model does not use"
        )
    if relation == "comparator_ends_before_prediction_time":
        return (
            f"{label[:1].upper()}{label[1:]} is computed from data recorded up to {until} h "
            f"after ICU admission, while the model's prediction time is {at} h, so the "
            f"comparison credits the model with information {label} does not use"
        )
    raise BenchmarkReportError(
        BENCHMARK_REPORT_SOURCE_UNREADABLE,
        f"{comparison.column!r} records an unknown window relation {relation!r}",
    )


def benchmark_window_limitations(
    records: Sequence[Mapping[str, Any]],
    *,
    evidence: Any,
    reader_display_labels: Mapping[str, str],
    context: Any = None,
    manuscript_language: str = "en",
) -> tuple[BenchmarkWindowLimitation, ...]:
    """One Limitations sentence per comparator whose window is not the model's."""

    limitations = []
    labels: Optional[Mapping[str, str]] = None
    for comparison in _comparisons(records, evidence):
        if labels is None:
            labels = _reader_labels(context, reader_display_labels, manuscript_language)
        label = _reader_label(labels, comparison.column)
        text = _window_text(comparison, label)
        if text is not None:
            limitations.append(BenchmarkWindowLimitation(
                text=text,
                evidence_id=comparison.evidence_id,
                source_sha256=comparison.source_sha256,
                field=f"{comparison.prefix}.information_window_relation",
            ))
    return tuple(limitations)


def audit_bound_benchmark_limitations(
    bound: str,
    *,
    evidence: Any,
    per_step_records: Sequence[Mapping[str, Any]],
    reader_display_labels: Mapping[str, str],
    context: Any = None,
    manuscript_language: str = "en",
) -> Optional[ValidationFinding]:
    """Fail closed when a comparator's window sentence did not survive binding."""

    missing = missing_bound_method_facts(
        bound,
        benchmark_window_limitations(
            per_step_records,
            evidence=evidence,
            reader_display_labels=reader_display_labels,
            context=context,
            manuscript_language=manuscript_language,
        ),
        lambda text: evidence.bind_manuscript(
            text, per_step_records=per_step_records, reader_labels=None
        ),
    )
    if not missing:
        return None
    return ValidationFinding(
        validator="evidence_bound_writer",
        severity="error",
        message=(
            "A comparator's information window, which the comparison's Limitations "
            "states, did not survive manuscript provenance validation."
        ),
        detail={"reason_code": BENCHMARK_LIMITATIONS_MISSING, "source_fields": list(missing)},
    )


def benchmark_limitations_audit(
    reader_display_labels: Mapping[str, str], context: Any, manuscript_language: str
) -> Callable[..., Optional[ValidationFinding]]:
    """:func:`audit_bound_benchmark_limitations` with the report's labels bound."""

    return partial(
        audit_bound_benchmark_limitations,
        reader_display_labels=reader_display_labels,
        context=context,
        manuscript_language=manuscript_language,
    )


__all__ = [
    "BENCHMARK_LIMITATIONS_MISSING",
    "BENCHMARK_REPORT_LABEL_UNSAFE",
    "BENCHMARK_REPORT_REQUIREMENT_DRIFT",
    "BENCHMARK_REPORT_SOURCE_MISMATCH",
    "BENCHMARK_REPORT_SOURCE_UNREADABLE",
    "BenchmarkReportError",
    "BenchmarkWindowLimitation",
    "audit_bound_benchmark_limitations",
    "benchmark_limitations_audit",
    "benchmark_window_limitations",
    "compile_benchmark_report_facts",
]
