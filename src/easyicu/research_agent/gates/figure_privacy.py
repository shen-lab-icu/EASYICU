"""Host-owned privacy audit deciding whether a figure may leave the machine.

The figure-egress gate refuses to upload an image unless its evidence record
declares ``aggregate_only``. Deriving that declaration from the contract's
``panel.role`` alone is not enough: ``role`` is a *claim* written by the
planner or by generated code, and a panel labelled ``validation`` can still be
a scatter with one marker per stay. Role therefore stays an **input** to this
audit, never the authorization.

What the host can actually prove, deterministically, without trusting the
producer:

* the panel roles are ones whose renderers draw summaries, not per-subject
  glyphs (necessary, not sufficient);
* every source artefact the figure was drawn from is resolvable, readable and
  inspectable — an artefact the host cannot open cannot be cleared;
* every source artefact still hashes to the digest it was registered under, so
  the clearance is about the bytes that were actually inspected;
* no source artefact exposes a subject identifier or an event timestamp, the
  two column families that make a mark per-patient in the first place;
* no source artefact *value* is identifier-shaped, whatever its column is
  called — a ``label, predicted`` table holding ``patient_30042318`` per row
  passes every name-level check ever written;
* no source artefact declares a group/stratum count below the small-cell
  floor;
* no text the contract renders into the image (titles, claims, notes) carries
  an identifier-shaped token.

Findings never quote the offending value: the reasons are written into the
figure's evidence metadata and into a receipt a reviewer reads, so echoing an
identifier there would move it out of the artefact and into the audit trail.

Anything this audit cannot establish is reported as ``aggregate_only=False``
with the reason, and the egress gate then refuses the upload. That is the
intended failure direction: a false refusal costs external visual QA, a false
clearance costs patient data.

What this audit deliberately does *not* claim: it does not count the marks in
the rendered raster and compare them to the cohort size. The receipt records
``mark_count_verified: False`` so a reviewer sees which question was answered
by inspection and which was answered by construction.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from ..authority.evidence_store import sha256_of_file

#: Minimum size for any group/stratum count a source artefact declares.
MIN_DISCLOSED_GROUP_SIZE = 20

#: Column/key names that carry a subject identifier across the six supported
#: databases, plus the generic ``*_id`` shape.
IDENTIFIER_COLUMNS = frozenset(
    {
        "admissionid",
        "caseid",
        "hadm_id",
        "icustay_id",
        "mrn",
        "patient_id",
        "patientid",
        "patienthealthsystemstayid",
        "patientunitstayid",
        "person_id",
        "stay_id",
        "subject_id",
        "uniquepid",
    }
)

_IDENTIFIER_SUFFIX_RE = re.compile(r"(?:^|_)(?:id|ids)$", re.IGNORECASE)

#: Column/key names carrying an event time. A per-event timestamp is what turns
#: an aggregate table back into a per-subject one.
TIME_COLUMNS = frozenset(
    {
        "admittime",
        "charttime",
        "chartdate",
        "datetime",
        "deathtime",
        "dischtime",
        "dob",
        "dod",
        "endtime",
        "entertime",
        "intime",
        "outtime",
        "starttime",
        "storetime",
        "timestamp",
    }
)

#: Keys whose integer value denotes how many subjects a cell contains.
GROUP_SIZE_KEYS = frozenset(
    {
        "cell_count",
        "group_size",
        "n",
        "n_cases",
        "n_control",
        "n_controls",
        "n_events",
        "n_exposed",
        "n_patients",
        "n_stays",
        "n_subjects",
        "stratum_n",
        "subgroup_n",
    }
)

#: Keys/columns whose value is a magnitude, not an identity. A cohort total or
#: a measurement count legitimately runs to six or more digits, so these are
#: exempt from the identifier-token scan (they are still checked for small
#: cells, which is the opposite direction).
_MAGNITUDE_NAME_RE = re.compile(
    r"(?:^|_)(?:count|n|num|number|rows|size|total|totals|sum|denominator)$",
    re.IGNORECASE,
)

#: Suffixes this audit knows how to open. Anything else is uninspectable and
#: therefore uncleared.
INSPECTABLE_SUFFIXES = frozenset({".json", ".csv", ".tsv", ".parquet"})

#: A run of digits long enough to be a record identifier rather than a year, a
#: count or a p-value. The leading ``[\d.]`` exclusion keeps ``0.000001`` and
#: the tail of a long decimal from reading as an identifier.
_IDENTIFIER_TOKEN_RE = re.compile(r"(?<![\d.])\d{6,}(?!\d)")

_MAX_JSON_NODES = 20_000

#: Row cap for the delimited-file scan. A figure source is a summary table, so
#: this is generous; exceeding it is reported rather than silently ignored.
_MAX_SCANNED_ROWS = 50_000

#: Cap on how many distinct value-level findings one artefact reports. The
#: reason list is evidence a human reads; one leaking column would otherwise
#: produce 50,000 identical lines.
_MAX_VALUE_FINDINGS = 5


@dataclass
class FigurePrivacyAudit:
    """Deterministic verdict for one figure, with everything it looked at."""

    figure_id: str
    aggregate_only: bool
    reasons: List[str] = field(default_factory=list)
    inspected_sources: List[Dict[str, Any]] = field(default_factory=list)
    panel_roles: List[str] = field(default_factory=list)
    #: Explicitly recorded so a reviewer is not left assuming the raster
    #: itself was counted. It was not.
    mark_count_verified: bool = False

    def as_metadata(self) -> Dict[str, Any]:
        """The subset merged into the figure's evidence metadata."""

        metadata: Dict[str, Any] = {
            "aggregate_only": self.aggregate_only,
            "aggregate_only_basis": "host_privacy_audit",
            "aggregate_only_audit_version": FIGURE_PRIVACY_AUDIT_VERSION,
            "aggregate_only_roles": sorted(set(self.panel_roles)),
            "aggregate_only_sources_inspected": len(self.inspected_sources),
            "aggregate_only_mark_count_verified": self.mark_count_verified,
        }
        if not self.aggregate_only:
            metadata["aggregate_only_reason"] = "; ".join(self.reasons) or "unproven"
        return metadata

    @property
    def verified_source_sha256(self) -> Dict[str, str]:
        """Digest of each source *as this audit read it*, not as registered.

        The egress gate re-checks this against the store at upload time: a
        clearance is only about the bytes that were inspected.
        """

        return {
            str(entry.get("evidence_id") or ""): str(entry.get("sha256_verified") or "")
            for entry in self.inspected_sources
            if entry.get("sha256_verified")
        }

    def as_receipt(self) -> Dict[str, Any]:
        return {
            "schema": FIGURE_PRIVACY_RECEIPT_SCHEMA,
            "audit_version": FIGURE_PRIVACY_AUDIT_VERSION,
            "figure_id": self.figure_id,
            "aggregate_only": self.aggregate_only,
            "reasons": list(self.reasons),
            "panel_roles": sorted(set(self.panel_roles)),
            "inspected_sources": list(self.inspected_sources),
            "source_sha256": self.verified_source_sha256,
            "mark_count_verified": self.mark_count_verified,
            "min_disclosed_group_size": MIN_DISCLOSED_GROUP_SIZE,
        }


#: Bumped whenever what the audit actually proves changes. 1.1.0 added
#: value-level scanning (Parquet/CSV/JSON cells, not just names) and source
#: re-hashing; the egress gate refuses a clearance produced by an audit version
#: it does not know, so an older receipt cannot authorize under the new rules.
FIGURE_PRIVACY_AUDIT_VERSION = "1.1.0"

FIGURE_PRIVACY_RECEIPT_SCHEMA = "easyicu.figure_privacy_audit/2"

#: Audit versions whose clearance the egress gate will still honour.
TRUSTED_AUDIT_VERSIONS = frozenset({FIGURE_PRIVACY_AUDIT_VERSION})


def _is_identifier_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    if not lowered:
        return False
    if lowered in IDENTIFIER_COLUMNS:
        return True
    if lowered in {"id", "ids"}:
        return True
    return bool(_IDENTIFIER_SUFFIX_RE.search(lowered)) and lowered not in {
        "evidence_id",
        "figure_id",
        "panel_id",
        "run_id",
        "step_id",
        "claim_id",
        "contract_id",
        "model_id",
        "analysis_id",
        "schema_id",
    }


def _is_time_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    if lowered in TIME_COLUMNS:
        return True
    return lowered.endswith("_time") or lowered.endswith("_datetime")


def _column_findings(columns: Sequence[str]) -> List[str]:
    reasons: List[str] = []
    identifiers = sorted({c for c in columns if _is_identifier_name(c)})
    times = sorted({c for c in columns if _is_time_name(c)})
    if identifiers:
        reasons.append("subject identifier column(s): " + ", ".join(identifiers))
    if times:
        reasons.append("event timestamp column(s): " + ", ".join(times))
    return reasons


def _is_magnitude_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    if lowered in GROUP_SIZE_KEYS:
        return True
    return bool(_MAGNITUDE_NAME_RE.search(lowered))


def _mask_identifier_token(token: str) -> str:
    """Describe an identifier-shaped token without reproducing it.

    The reasons this audit returns are written into the figure's evidence
    metadata and into a receipt a reviewer reads. Echoing the offending value
    there would move the identifier out of the artefact and into the audit
    trail — the audit would leak exactly what it exists to catch.
    """

    return f"<{len(str(token))}-digit token>"


class _ValueScanner:
    """Streaming scan of ``(name, value)`` cells, bounded by column count.

    Column names alone clear a file that says ``label,predicted`` and then
    writes ``patient_30042318`` into every row. The name of the column is the
    producer's choice; the digits in the cell are the disclosure. Scanning is
    incremental rather than list-building because a source table may legally
    reach the row cap and materialising every cell would cost more memory than
    the figure it is clearing.
    """

    def __init__(self) -> None:
        self.small_cells: Set[str] = set()
        self.identifier_hits: Dict[str, int] = {}
        self.cells_scanned = 0

    def add(self, name: Any, value: Any) -> None:
        self.cells_scanned += 1
        lowered = str(name).strip().lower()
        if lowered in GROUP_SIZE_KEYS:
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = None
            if count is not None and 0 < count < MIN_DISCLOSED_GROUP_SIZE:
                self.small_cells.add(f"{name}={count}")
            return
        if value is None or isinstance(value, bool):
            return
        if _is_magnitude_name(lowered):
            return
        if isinstance(value, float):
            # A float is a measurement or an estimate. Identifiers are stored
            # as integers or strings; scanning floats would flag long sums and
            # high-precision statistics for no gain.
            return
        if _IDENTIFIER_TOKEN_RE.search(str(value)):
            column = str(name)
            self.identifier_hits[column] = self.identifier_hits.get(column, 0) + 1

    def add_rows(self, header: Sequence[Any], row: Sequence[Any]) -> None:
        for name, value in zip(header, row):
            self.add(name, value)

    def reasons(self) -> List[str]:
        found: List[str] = []
        if self.small_cells:
            found.append(
                f"declared group size(s) below {MIN_DISCLOSED_GROUP_SIZE}: "
                + ", ".join(sorted(self.small_cells))
            )
        if self.identifier_hits:
            named = sorted(self.identifier_hits.items())[:_MAX_VALUE_FINDINGS]
            found.append(
                "identifier-shaped value(s) in "
                + ", ".join(f"{column} ({count} cell(s))" for column, count in named)
            )
        return found


def _inspect_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: Set[str] = set()
    scanner = _ValueScanner()
    # The stack carries the key each node was reached under, so a list inherits
    # its parent's name: ``{"labels": ["patient 30042318"]}`` is a disclosure
    # by ``labels``, and its elements have no name of their own.
    stack: List[tuple] = [("<root>", payload)]
    nodes = 0
    truncated = False
    while stack:
        key, node = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            truncated = True
            break
        if isinstance(node, Mapping):
            for child_key, value in node.items():
                names.add(str(child_key))
                stack.append((child_key, value))
        elif isinstance(node, (list, tuple)):
            stack.extend((key, item) for item in node)
        else:
            scanner.add(key, node)
    reasons = _column_findings(sorted(names)) + scanner.reasons()
    if truncated:
        reasons.append(
            f"payload exceeds {_MAX_JSON_NODES} nodes and was not fully inspected"
        )
    return {
        "keys_scanned": len(names),
        "values_scanned": scanner.cells_scanned,
        "reasons": reasons,
    }


def _inspect_delimited(path: Path, *, delimiter: str) -> Dict[str, Any]:
    scanner = _ValueScanner()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return {"columns": [], "reasons": ["file is empty"]}
        reasons = _column_findings(header)
        rows = 0
        truncated = False
        for row in reader:
            rows += 1
            if rows > _MAX_SCANNED_ROWS:
                truncated = True
                break
            scanner.add_rows(header, row)
    if truncated:
        # Every column is scanned for identifiers and small cells now, so any
        # unscanned row may hide either. Uncleared beats guessing.
        reasons.append(
            f"file exceeds {_MAX_SCANNED_ROWS} rows and was not fully scanned"
        )
    return {
        "columns": list(header),
        "rows": rows,
        "values_scanned": scanner.cells_scanned,
        "reasons": reasons + scanner.reasons(),
    }


def _inspect_parquet(path: Path) -> Dict[str, Any]:
    """Read the values, not only the schema.

    A schema-only check clears a two-column ``label, predicted`` table whose
    ``label`` column holds one stay identifier per row, and clears a
    ``subgroup, n`` table whose ``n`` is 3. Both are per-subject disclosures
    that the column names alone cannot show.
    """

    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    columns = list(handle.schema_arrow.names)
    reasons = _column_findings(columns)
    scanner = _ValueScanner()
    rows = 0
    truncated = False
    for batch in handle.iter_batches(batch_size=4096):
        table = batch.to_pydict()
        height = batch.num_rows
        if rows + height > _MAX_SCANNED_ROWS:
            height = max(0, _MAX_SCANNED_ROWS - rows)
            truncated = True
        for name in columns:
            values = table.get(name) or ()
            for value in list(values)[:height]:
                scanner.add(name, value)
        rows += height
        if truncated:
            break
    if truncated:
        reasons.append(
            f"file exceeds {_MAX_SCANNED_ROWS} rows and was not fully scanned"
        )
    return {
        "columns": columns,
        "rows": rows,
        "values_scanned": scanner.cells_scanned,
        "reasons": reasons + scanner.reasons(),
    }


def _inspect_source(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in INSPECTABLE_SUFFIXES:
        return {
            "reasons": [
                f"source artefact type {suffix or '<none>'} cannot be inspected "
                "for identifier columns"
            ]
        }
    try:
        if suffix == ".json":
            return _inspect_json(path)
        if suffix in {".csv", ".tsv"}:
            return _inspect_delimited(path, delimiter="\t" if suffix == ".tsv" else ",")
        return _inspect_parquet(path)
    except Exception as exc:  # noqa: BLE001 - any read failure means uncleared
        return {"reasons": [f"source artefact could not be read: {exc}"]}


def _rendered_text(contract: Any) -> List[str]:
    texts = [
        str(getattr(contract, "core_claim", "") or ""),
        str(getattr(contract, "statistics_note", "") or ""),
        str(getattr(contract, "image_integrity_note", "") or ""),
    ]
    for panel in getattr(contract, "panels", ()) or ():
        texts.append(str(getattr(panel, "title", "") or ""))
        texts.append(str(getattr(panel, "claim", "") or ""))
    return [text for text in texts if text]


def audit_figure_privacy(
    *,
    contract: Any,
    evidence: Any,
    run_dir: Path,
    source_evidence_ids: Sequence[str],
    allowed_panel_roles: Optional[frozenset] = None,
) -> FigurePrivacyAudit:
    """Decide, from artefacts rather than from declarations, if a figure may leave.

    ``allowed_panel_roles`` is the necessary role condition; passing it does not
    clear a figure on its own.
    """

    panels = list(getattr(contract, "panels", ()) or ())
    roles = [str(getattr(panel, "role", "") or "") for panel in panels]
    audit = FigurePrivacyAudit(
        figure_id=str(getattr(contract, "figure_id", "") or "unknown"),
        aggregate_only=False,
        panel_roles=roles,
    )

    if not panels:
        audit.reasons.append("no panels declared")
        return audit

    if allowed_panel_roles is not None:
        off_list = sorted({role for role in roles if role not in allowed_panel_roles})
        if off_list:
            audit.reasons.append(
                "panel role(s) whose renderers may draw per-subject marks: "
                + ", ".join(off_list)
            )

    ids = [str(item) for item in dict.fromkeys(source_evidence_ids) if str(item)]
    if not ids:
        audit.reasons.append("figure declares no source evidence to inspect")

    root = Path(run_dir)
    for evidence_id in ids:
        record = None
        try:
            record = evidence.get(evidence_id)
        except Exception:  # noqa: BLE001 - an unreadable store clears nothing
            record = None
        entry: Dict[str, Any] = {"evidence_id": evidence_id}
        if record is None:
            entry["reasons"] = ["source evidence id does not resolve to a record"]
            audit.inspected_sources.append(entry)
            audit.reasons.extend(
                f"{evidence_id}: {reason}" for reason in entry["reasons"]
            )
            continue
        registered_sha = str(getattr(record, "sha256", "") or "")
        entry["sha256"] = registered_sha
        path = root / str(getattr(record, "relative_path", "") or "")
        entry["suffix"] = path.suffix.lower()
        if not path.is_file():
            entry["reasons"] = ["source artefact is missing from the run directory"]
        else:
            # Re-hash before reading. Without this the audit clears whatever
            # is on disk *now* while the receipt names the digest registered
            # earlier, so a source rewritten after registration would be
            # inspected under someone else's provenance.
            try:
                actual_sha = sha256_of_file(path)
            except OSError as exc:
                actual_sha = ""
                entry["reasons"] = [f"source artefact could not be hashed: {exc}"]
            else:
                entry["sha256_verified"] = actual_sha
                if not registered_sha:
                    entry["reasons"] = [
                        "source evidence record carries no registered digest"
                    ]
                elif actual_sha != registered_sha:
                    entry["reasons"] = [
                        "source artefact no longer matches its registered digest "
                        f"({actual_sha[:12]}… != {registered_sha[:12]}…)"
                    ]
                else:
                    entry.update(_inspect_source(path))
        audit.inspected_sources.append(entry)
        audit.reasons.extend(f"{evidence_id}: {reason}" for reason in entry["reasons"])

    for text in _rendered_text(contract):
        for token in _IDENTIFIER_TOKEN_RE.findall(text):
            audit.reasons.append(
                "rendered text carries an identifier-shaped token: "
                + _mask_identifier_token(token)
            )

    audit.aggregate_only = not audit.reasons
    return audit


__all__ = [
    "FIGURE_PRIVACY_AUDIT_VERSION",
    "FIGURE_PRIVACY_RECEIPT_SCHEMA",
    "GROUP_SIZE_KEYS",
    "IDENTIFIER_COLUMNS",
    "INSPECTABLE_SUFFIXES",
    "MIN_DISCLOSED_GROUP_SIZE",
    "TIME_COLUMNS",
    "TRUSTED_AUDIT_VERSIONS",
    "FigurePrivacyAudit",
    "audit_figure_privacy",
]
