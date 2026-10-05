"""List the small cells a run's publication products show, for sign-off review.

``gates.publication_disclosure`` owns the policy; this owner applies it to one
run and changes no product.  It reads three:

* the reader tables, through the counts their projection prints;
* the numbers the bound manuscript cites, through the numeric binding map.
  Every printed number is bound to the field it came from, so the counts in
  host-written sentences and claim text are read here too;
* the result tables the run registered (CSV, TSV or JSON table evidence),
  which the download bundle exports and the figures are drawn from, read by
  column or key name.

The study's profile is its strictest source: the cohort database and every
cross-database validation source.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from easyicu.databases.profiles import normalize_database_key
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..gates.publication_disclosure import (
    PROFILE_UNDECLARED_REASON,
    SMALL_CELL_BELOW,
    SUPPRESSION_UNAVAILABLE_REASON,
    PublicationDisclosureProfile,
    SourceDisclosure,
    count_name_key,
    is_subject_count_name,
    small_cell_value,
    source_disclosure,
    strictest_profile,
)
from ..schema import AnalysisPlan, EvidenceRecord, ResearchContext, ValidationFinding
from .manuscript_tables import ManuscriptTableProjectionError, manuscript_table_counts

PUBLICATION_DISCLOSURE_REVIEW_SCHEMA = "easyicu.publication_disclosure_review/1"
PUBLICATION_DISCLOSURE_REVIEW_FILENAME = "publication_disclosure_review.json"
PUBLICATION_DISCLOSURE_VALIDATOR = "publication_disclosure"

#: A result table larger than this is listed as unread rather than parsed.
_MAX_RESULT_TABLE_BYTES = 32 * 1024 * 1024
#: Rows of one small-cell column the review names; the rest are counted.
_ROWS_NAMED_PER_COLUMN = 5
_RESULT_TABLE_SUFFIXES = {".csv": ",", ".tsv": "\t", ".json": None}


@dataclass(frozen=True)
class PublicationDisclosureReview:
    """The review a run writes and the findings it adds to the run."""

    payload: dict[str, Any]
    findings: tuple[ValidationFinding, ...]


def study_disclosures(context: ResearchContext) -> tuple[SourceDisclosure, ...]:
    """The profile of every source the study reports from, in study order."""

    sources = [context.cohort.database, *(context.cross_database_validation or ())]
    disclosures: dict[str, SourceDisclosure] = {}
    for source in sources:
        try:
            key = normalize_database_key(str(source))
        except KeyError:
            key = str(source)
        disclosure = source_disclosure(key)
        disclosures.setdefault(disclosure.source, disclosure)
    return tuple(disclosures.values())


def _reader_table_cells(
    plan: Optional[AnalysisPlan], evidence_records: Sequence[EvidenceRecord], run_dir: Path,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if plan is None:
        return [], None
    try:
        counts = manuscript_table_counts(plan=plan, evidence_records=evidence_records, run_dir=run_dir)
    except ManuscriptTableProjectionError as exc:
        # The reader then prints no table; the projection error is reported
        # where the reader's tables are built.
        return [], str(exc)
    return [
        {
            "table": count.table, "caption": count.caption, "row": count.row,
            "column": count.column, "value": count.value, "evidence_id": count.evidence_id,
        }
        for count in counts
        if small_cell_value(count.value) is not None
    ], None


def _field_leaf(source_field: str) -> str:
    """The last named segment of a bound field path (``results.n_events`` -> ``n_events``)."""

    names = [part for part in re.split(r"[.\[\]/:]+", str(source_field)) if part and not part.isdigit()]
    return names[-1] if names else ""


def _cited_cells(numeric_binding_map: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for footnote, claim in numeric_binding_map.items():
        if getattr(claim, "derived_from", None):
            continue
        field = _field_leaf(getattr(claim, "source_field", ""))
        value = small_cell_value(getattr(claim, "canonical", None))
        if value is None or not is_subject_count_name(field):
            continue
        cells.append({
            "footnote": str(footnote), "step_id": str(getattr(claim, "step_id", "")),
            "field": field, "value": value, "evidence_id": str(getattr(claim, "evidence_id", "")),
        })
    return sorted(cells, key=lambda cell: (cell["evidence_id"], cell["field"], cell["footnote"]))


class _ColumnCells:
    """The small cells of one result-table column: how many, which values, where."""

    def __init__(self) -> None:
        self.cells = 0
        self.values: set[int] = set()
        self.locations: list[str] = []

    def add(self, value: int, location: str) -> None:
        self.cells += 1
        self.values.add(value)
        if len(self.locations) < _ROWS_NAMED_PER_COLUMN:
            self.locations.append(location)


def _delimited_cells(path: Path, delimiter: str) -> dict[str, _ColumnCells]:
    found: dict[str, _ColumnCells] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        header = next(reader, [])
        counted = [(index, name) for index, name in enumerate(header) if is_subject_count_name(name)]
        if not counted:
            return found
        for row in reader:
            for index, name in counted:
                value = small_cell_value(row[index]) if index < len(row) else None
                if value is not None:
                    found.setdefault(name, _ColumnCells()).add(value, f"line {reader.line_num}")
    return found


def _json_cells(path: Path) -> dict[str, _ColumnCells]:
    found: dict[str, _ColumnCells] = {}

    def walk(node: Any, where: str, name: Any) -> None:
        # A value is read under the nearest key that names it, so a list of
        # counts under ``at_risk`` is read as at-risk counts.
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{where}.{key}" if where else str(key), key)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{where}[{index}]", name)
        elif name is not None and is_subject_count_name(name):
            small = small_cell_value(node)
            if small is not None:
                found.setdefault(count_name_key(name), _ColumnCells()).add(small, where)

    walk(json.loads(path.read_text(encoding="utf-8")), "", None)
    return found


def _result_table_cells(
    evidence_records: Sequence[EvidenceRecord], run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    columns: list[dict[str, Any]] = []
    unread: list[dict[str, str]] = []
    for record in evidence_records:
        suffix = Path(record.relative_path).suffix.lower()
        if record.kind != "table" or suffix not in _RESULT_TABLE_SUFFIXES:
            continue
        path = verified_run_evidence_path(run_dir, record)
        if path is None:
            unread.append({"evidence_id": record.evidence_id, "reason": "missing or changed since registration"})
            continue
        delimiter = _RESULT_TABLE_SUFFIXES[suffix]
        try:
            if path.stat().st_size > _MAX_RESULT_TABLE_BYTES:
                unread.append({"evidence_id": record.evidence_id, "reason": "larger than 32 MiB"})
                continue
            found = _json_cells(path) if delimiter is None else _delimited_cells(path, delimiter)
        except (OSError, UnicodeError, ValueError, RecursionError, csv.Error) as exc:
            unread.append({"evidence_id": record.evidence_id, "reason": f"unreadable: {type(exc).__name__}"})
            continue
        columns.extend(
            {
                "evidence_id": record.evidence_id, "relative_path": record.relative_path,
                "column": name, "cells": cells.cells, "values": sorted(cells.values),
                "first_locations": cells.locations,
            }
            for name, cells in sorted(found.items())
        )
    return columns, unread


def review_publication_disclosure(
    *,
    context: Optional[ResearchContext],
    plan: Optional[AnalysisPlan],
    evidence_records: Sequence[EvidenceRecord],
    run_dir: Path,
    numeric_binding_map: Mapping[str, Any],
) -> Optional[PublicationDisclosureReview]:
    """Review the run's publication products, or ``None`` when nothing needs review.

    A study whose every source holds no patient data needs no review.
    """

    if context is None:
        return None
    disclosures = study_disclosures(context)
    profile = strictest_profile(disclosures)
    if profile is PublicationDisclosureProfile.REPORT:
        return None
    reader_tables, reader_table_error = _reader_table_cells(plan, evidence_records, run_dir)
    cited = _cited_cells(numeric_binding_map)
    result_tables, unread = _result_table_cells(evidence_records, run_dir)
    payload: dict[str, Any] = {
        "schema_version": PUBLICATION_DISCLOSURE_REVIEW_SCHEMA,
        "profile": None if profile is None else profile.value,
        "sources": [
            {"source": item.source, "profile": None if item.profile is None else item.profile.value,
             "licence": item.licence}
            for item in disclosures
        ],
        "small_cell": f"a subject count from 1 to {SMALL_CELL_BELOW - 1}",
        "reader_tables": reader_tables,
        "cited_numbers": cited,
        "result_tables": result_tables,
        "unread_result_tables": unread,
    }
    if reader_table_error is not None:
        payload["reader_table_error"] = reader_table_error
    listed = len(reader_tables) + len(cited) + sum(column["cells"] for column in result_tables)
    detail = {
        "review": PUBLICATION_DISCLOSURE_REVIEW_FILENAME,
        "reader_table_cells": len(reader_tables),
        "cited_numbers": len(cited),
        "result_table_cells": sum(column["cells"] for column in result_tables),
        "unread_result_tables": len(unread),
    }
    findings: list[ValidationFinding] = []
    if profile is None:
        undeclared = sorted(item.source for item in disclosures if item.profile is None)
        findings.append(ValidationFinding(
            validator=PUBLICATION_DISCLOSURE_VALIDATOR, severity="error",
            message=(
                f"{PROFILE_UNDECLARED_REASON}: no publication disclosure profile is declared "
                f"for data source(s) {', '.join(undeclared)}, so the report cannot be published."
            ),
            detail={**detail, "reason": PROFILE_UNDECLARED_REASON, "sources": undeclared},
        ))
    elif profile is PublicationDisclosureProfile.SUPPRESS_SMALL_CELLS:
        findings.append(ValidationFinding(
            validator=PUBLICATION_DISCLOSURE_VALIDATOR, severity="error",
            message=(
                f"{SUPPRESSION_UNAVAILABLE_REASON}: a data source's licence requires small cells "
                "to be suppressed and no publication product suppresses them yet."
            ),
            detail={**detail, "reason": SUPPRESSION_UNAVAILABLE_REASON},
        ))
    elif listed or unread:
        findings.append(ValidationFinding(
            validator=PUBLICATION_DISCLOSURE_VALIDATOR, severity="warning",
            message=(
                f"{listed} count(s) from 1 to {SMALL_CELL_BELOW - 1} appear in the reader tables, "
                f"cited numbers or exported result tables"
                + (f", and {len(unread)} result table(s) could not be read" if unread else "")
                + f". Review {PUBLICATION_DISCLOSURE_REVIEW_FILENAME} before signing the report off."
            ),
            detail=detail,
        ))
    return PublicationDisclosureReview(payload=payload, findings=tuple(findings))


def persist_publication_disclosure_review(
    *,
    context: Optional[ResearchContext],
    plan: Optional[AnalysisPlan],
    evidence_records: Sequence[EvidenceRecord],
    numeric_binding_map: Mapping[str, Any],
    run_dir: Path,
    evidence: Any,
    findings: list[ValidationFinding],
) -> None:
    """Write the run's review beside the bound manuscript and add its findings."""

    review = review_publication_disclosure(
        context=context, plan=plan, evidence_records=evidence_records,
        run_dir=run_dir, numeric_binding_map=numeric_binding_map,
    )
    if review is None:
        return
    path = run_dir / PUBLICATION_DISCLOSURE_REVIEW_FILENAME
    path.write_text(
        json.dumps(review.payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evidence.register_file(
        kind="log",
        description=(
            "Counts from 1 to 10 that the reader tables, cited numbers and exported "
            "result tables show, listed for the person who signs the report off."
        ),
        source_path=path,
        evidence_id="publication_disclosure_review",
        producer="pipeline",
        generation_mode="system",
        metadata={
            "schema_version": review.payload["schema_version"],
            "profile": review.payload["profile"],
        },
        on_sha_change="new_id",
    )
    findings.extend(review.findings)


__all__ = [
    "PUBLICATION_DISCLOSURE_REVIEW_FILENAME",
    "PUBLICATION_DISCLOSURE_REVIEW_SCHEMA",
    "PUBLICATION_DISCLOSURE_VALIDATOR",
    "PublicationDisclosureReview",
    "persist_publication_disclosure_review",
    "review_publication_disclosure",
    "study_disclosures",
]
