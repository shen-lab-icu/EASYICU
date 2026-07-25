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
* no source artefact exposes a subject identifier or an event timestamp, the
  two column families that make a mark per-patient in the first place;
* no source artefact declares a group/stratum count below the small-cell
  floor;
* no text the contract renders into the image (titles, claims, notes) carries
  an identifier-shaped token.

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

#: Suffixes this audit knows how to open. Anything else is uninspectable and
#: therefore uncleared.
INSPECTABLE_SUFFIXES = frozenset({".json", ".csv", ".tsv", ".parquet"})

#: A run of digits long enough to be a record identifier rather than a year,
#: a count or a p-value.
_IDENTIFIER_TOKEN_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")

_MAX_JSON_NODES = 20_000


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

    def as_receipt(self) -> Dict[str, Any]:
        return {
            "schema": "easyicu.figure_privacy_audit/1",
            "audit_version": FIGURE_PRIVACY_AUDIT_VERSION,
            "figure_id": self.figure_id,
            "aggregate_only": self.aggregate_only,
            "reasons": list(self.reasons),
            "panel_roles": sorted(set(self.panel_roles)),
            "inspected_sources": list(self.inspected_sources),
            "mark_count_verified": self.mark_count_verified,
            "min_disclosed_group_size": MIN_DISCLOSED_GROUP_SIZE,
        }


FIGURE_PRIVACY_AUDIT_VERSION = "1.0.0"


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


def _small_cell_findings(pairs: Sequence[tuple]) -> List[str]:
    """``pairs`` is (key, value); flag declared group sizes below the floor."""

    small: List[str] = []
    for key, value in pairs:
        if str(key).strip().lower() not in GROUP_SIZE_KEYS:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < count < MIN_DISCLOSED_GROUP_SIZE:
            small.append(f"{key}={count}")
    if small:
        return [
            f"declared group size(s) below {MIN_DISCLOSED_GROUP_SIZE}: "
            + ", ".join(sorted(small))
        ]
    return []


def _inspect_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: Set[str] = set()
    pairs: List[tuple] = []
    stack: List[Any] = [payload]
    nodes = 0
    truncated = False
    while stack:
        node = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            truncated = True
            break
        if isinstance(node, Mapping):
            for key, value in node.items():
                names.add(str(key))
                pairs.append((key, value))
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    reasons = _column_findings(sorted(names)) + _small_cell_findings(pairs)
    if truncated:
        reasons.append(
            f"payload exceeds {_MAX_JSON_NODES} nodes and was not fully inspected"
        )
    return {"keys_scanned": len(names), "reasons": reasons}


def _inspect_delimited(path: Path, *, delimiter: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return {"columns": [], "reasons": ["file is empty"]}
        reasons = _column_findings(header)
        pairs: List[tuple] = []
        rows = 0
        for row in reader:
            rows += 1
            for name, value in zip(header, row):
                if str(name).strip().lower() in GROUP_SIZE_KEYS:
                    pairs.append((name, value))
    return {
        "columns": list(header),
        "rows": rows,
        "reasons": reasons + _small_cell_findings(pairs),
    }


def _inspect_parquet(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    columns = list(schema.names)
    return {"columns": columns, "reasons": _column_findings(columns)}


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
        entry["sha256"] = str(getattr(record, "sha256", "") or "")
        path = root / str(getattr(record, "relative_path", "") or "")
        entry["suffix"] = path.suffix.lower()
        if not path.is_file():
            entry["reasons"] = ["source artefact is missing from the run directory"]
        else:
            entry.update(_inspect_source(path))
        audit.inspected_sources.append(entry)
        audit.reasons.extend(f"{evidence_id}: {reason}" for reason in entry["reasons"])

    for text in _rendered_text(contract):
        for token in _IDENTIFIER_TOKEN_RE.findall(text):
            audit.reasons.append(
                f"rendered text carries an identifier-shaped token: {token}"
            )

    audit.aggregate_only = not audit.reasons
    return audit


__all__ = [
    "FIGURE_PRIVACY_AUDIT_VERSION",
    "GROUP_SIZE_KEYS",
    "IDENTIFIER_COLUMNS",
    "INSPECTABLE_SUFFIXES",
    "MIN_DISCLOSED_GROUP_SIZE",
    "TIME_COLUMNS",
    "FigurePrivacyAudit",
    "audit_figure_privacy",
]
