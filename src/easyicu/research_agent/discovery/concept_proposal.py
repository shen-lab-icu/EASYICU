"""Deterministic gatekeeping for AI-proposed (T2) concept extensions.

When idea-mining flags a candidate as **T2** (not in the EasyICU dictionary but
measured by the source database), a user may want to research with it. The
honest way to expose that — without "dumping it on the LLM to write scripts" —
is:

* the LLM does **selection only**: pick which catalog ``itemid``(s) represent the
  concept and propose declarative metadata (unit, role, bounds). It writes NO
  extraction code; the existing trusted ``DataConverter`` / ``ConceptResolver``
  engine performs extraction from the resulting dictionary entry.
* **we** gatekeep the selection with deterministic rules grounded in the frozen
  source-item catalog metadata (``fluid`` / ``param_type`` / table) and the real
  value distribution. Every gate is rule-based and inspectable; nothing is
  accepted on the model's say-so.

A draft can only ever reach ``needs_human_review`` — never ``accepted``. The
human signs off on the *evidence* (catalog rows + distribution probe + gate
findings), and the approved concept stays **run-quarantined and provenance-
tagged** (see ``ConceptProposalDraft.quarantine``); it is never auto-written to
the shared ``concept-dict.json``.

This module is a leaf: it must not import ``idea_mining``. It depends on the
catalog snapshot (via ``idea_mining_feasibility_tier.SourceItemIndex`` metadata)
and an injected distribution probe, mirroring how feasibility probing is wired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# Source roles we can map deterministically from the catalog. A "measurement"
# concept must be a numeric value; events/durations are a different concept kind.
MEASUREMENT_ROLE = "measurement"
EVENT_ROLE = "event"
DURATION_ROLE = "duration"
VALID_ROLES = (MEASUREMENT_ROLE, EVENT_ROLE, DURATION_ROLE)

# param_type values (icu d_items) that are genuine numeric measurements.
_NUMERIC_PARAM_TYPES = frozenset({"numeric", "numeric with tag"})
# table (linksto) → the concept kind it can legitimately back.
_EVENT_TABLES = frozenset(
    {"icu/procedureevents", "icu/datetimeevents", "icu/outputevents"}
)

# Blood-compatible specimen labels in d_labitems.fluid for a blood analyte.
_BLOOD_FLUIDS = frozenset({"blood", "serum", "plasma", "whole blood"})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class DistributionStat:
    """Real-data distribution for one itemid (from an injected probe)."""

    itemid: int
    n_rows: int
    n_stays: int
    coverage_fraction: float
    p01: Optional[float] = None
    p50: Optional[float] = None
    p99: Optional[float] = None
    units: tuple[str, ...] = ()


# probe(itemids, table) -> {itemid: DistributionStat}. Injected so this module
# stays data-source agnostic and unit-testable, exactly like the feasibility
# probe wiring.
DistributionProbe = Callable[[Sequence[int], str], Dict[int, DistributionStat]]


@dataclass(frozen=True)
class ConceptProposalDraft:
    """An LLM *selection-only* proposal — declarative, no extraction code."""

    concept_name: str
    candidate_itemids: tuple[int, ...]
    role: str = MEASUREMENT_ROLE
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_fluid: Optional[str] = None  # e.g. "Blood" for a blood analyte
    rationale: str = ""
    source_quote: str = ""
    # quarantine is structural, not negotiable: approved drafts run-local only.
    quarantine: bool = True


@dataclass(frozen=True)
class GateFinding:
    gate: str
    severity: str  # "error" (blocks) | "warning" (human-review) | "info"
    message: str


@dataclass(frozen=True)
class ConceptProposalValidation:
    status: str  # "rejected" | "needs_human_review"
    findings: tuple[GateFinding, ...]
    resolved_itemids: tuple[int, ...]
    dropped_itemids: tuple[int, ...]
    distribution: tuple[DistributionStat, ...] = field(default=())

    @property
    def blocked(self) -> bool:
        return self.status == "rejected"


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _catalog_rows(source_index, itemids: Sequence[int]) -> Dict[int, dict]:
    """Pull the frozen catalog row for each itemid (no free-text trust)."""
    by_id = {int(it["itemid"]): it for it in getattr(source_index, "_items", [])}
    return {iid: by_id[iid] for iid in itemids if iid in by_id}


# --- LLM selection-only step -------------------------------------------------
# The model never writes extraction code; it only *selects* itemids from the
# frozen catalog and proposes declarative metadata. complete(system, user) ->
# JSON string. Injected so tests stay hermetic.
LLMComplete = Callable[[str, str], str]

_SELECTION_SYSTEM_PROMPT = (
    "You map a clinical concept to source-database item ids. You are given a "
    "fixed list of catalog items (itemid, label, fluid, category, param_type, "
    "table). SELECT ONLY itemids from that list that represent the concept — "
    "never invent an itemid. Prefer blood/serum/plasma specimens for blood "
    "analytes. Do NOT write extraction code. Return ONLY JSON with keys: "
    "selected_itemids (list of int), role (one of measurement/event/duration), "
    "unit (string or null), min_value (number or null), max_value (number or "
    "null), target_fluid (string or null), rationale (string). For a "
    "measurement role you MUST give a unit and clinically plausible "
    "min_value/max_value bounds."
)


def gather_candidate_rows(
    source_index, concept_name: str, *, limit: int = 15
) -> List[dict]:
    """Catalog rows whose label/abbrev share specific tokens with the concept."""
    hits = source_index.match(concept_name, limit=limit)
    rows = _catalog_rows(source_index, [h.itemid for h in hits])
    # preserve match order
    return [rows[h.itemid] for h in hits if h.itemid in rows]


def build_selection_messages(
    concept_name: str, rows: Sequence[dict]
) -> tuple[str, str]:
    lines = [
        f"{r['itemid']}\t{r.get('label')}\tfluid={r.get('fluid') or '-'}"
        f"\tcategory={r.get('category') or '-'}\tparam_type={r.get('param_type') or '-'}"
        f"\ttable={r.get('table')}"
        for r in rows
    ]
    user = (
        f"Concept to map: {concept_name!r}\n\n"
        "Catalog items (tab-separated):\n" + "\n".join(lines)
    )
    return _SELECTION_SYSTEM_PROMPT, user


def _extract_json(text: str) -> dict:
    raw = str(text or "").strip()
    if "```" in raw:
        # strip the first fenced block
        import re as _re

        m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
        if m:
            raw = m.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in LLM selection response")
    import json as _json

    return _json.loads(raw[start : end + 1])


def _coerce_float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def propose_concept_selection(
    concept_name: str,
    rows: Sequence[dict],
    *,
    complete: LLMComplete,
) -> ConceptProposalDraft:
    """Turn an LLM selection over the candidate rows into a declarative draft.

    Defense in depth: selected itemids are intersected with the supplied
    catalog rows here (the model cannot smuggle in an itemid that was not
    offered), and ``validate_concept_proposal`` re-checks against the catalog.
    """
    system, user = build_selection_messages(concept_name, rows)
    payload = _extract_json(complete(system, user))
    offered = {int(r["itemid"]) for r in rows}
    selected: List[int] = []
    for i in payload.get("selected_itemids") or []:
        # Accept ints, int-strings, AND integer-valued floats/float-strings:
        # JSON models sometimes emit an itemid as ``50954.0`` or ``"50954.0"``,
        # which ``str(i).isdigit()`` rejected, silently dropping a legitimately
        # selected item. The ``in offered`` check remains the correctness gate.
        try:
            itemid = int(float(i))
        except (TypeError, ValueError):
            continue
        if itemid in offered:
            selected.append(itemid)
    role = str(payload.get("role") or MEASUREMENT_ROLE).strip().lower()
    return ConceptProposalDraft(
        concept_name=concept_name,
        candidate_itemids=tuple(dict.fromkeys(selected)),
        role=role if role in VALID_ROLES else MEASUREMENT_ROLE,
        unit=(payload.get("unit") or None),
        min_value=_coerce_float(payload.get("min_value")),
        max_value=_coerce_float(payload.get("max_value")),
        target_fluid=(payload.get("target_fluid") or None),
        rationale=str(payload.get("rationale") or ""),
    )


def validate_concept_proposal(
    draft: ConceptProposalDraft,
    *,
    source_index,
    distribution_probe: Optional[DistributionProbe] = None,
    min_coverage_fraction: float = 0.01,
) -> ConceptProposalValidation:
    """Run the deterministic gate battery over an LLM concept selection.

    Hard ``error`` findings reject the draft. If no hard error fires the draft
    reaches ``needs_human_review`` (never ``accepted``) and carries every gate
    finding plus the real-data distribution for the human to sign off on.
    """
    findings: List[GateFinding] = []
    requested = list(dict.fromkeys(int(i) for i in draft.candidate_itemids))

    # --- Gate 1: catalog grounding (no invented itemids) -----------------
    rows = _catalog_rows(source_index, requested)
    invalid = [iid for iid in requested if iid not in rows]
    if invalid:
        findings.append(
            GateFinding(
                "catalog_grounding",
                "error",
                f"itemids not in the frozen source catalog: {invalid}",
            )
        )
    if not rows:
        return ConceptProposalValidation(
            status="rejected",
            findings=tuple(findings)
            or (GateFinding("catalog_grounding", "error", "no valid itemids"),),
            resolved_itemids=(),
            dropped_itemids=tuple(requested),
        )

    if draft.role not in VALID_ROLES:
        findings.append(GateFinding("role", "error", f"unknown role '{draft.role}'"))

    kept: List[int] = []
    dropped: List[int] = []

    # --- Gate 2: specimen consistency (labs) -----------------------------
    want_fluid = (draft.target_fluid or "").strip().lower()
    for iid, row in rows.items():
        fluid = str(row.get("fluid") or "").strip().lower()
        if want_fluid and fluid and fluid != want_fluid:
            # blood concept tolerates serum/plasma synonyms
            if not (want_fluid in _BLOOD_FLUIDS and fluid in _BLOOD_FLUIDS):
                dropped.append(iid)
                findings.append(
                    GateFinding(
                        "specimen_consistency",
                        "warning",
                        f"itemid {iid} '{row.get('label')}' is fluid="
                        f"{row.get('fluid')!r}, not target {draft.target_fluid!r}"
                        " — dropped",
                    )
                )
                continue
        kept.append(iid)

    # --- Gate 3: role / measurability ------------------------------------
    if draft.role == MEASUREMENT_ROLE:
        still: List[int] = []
        for iid in kept:
            row = rows[iid]
            table = str(row.get("table") or "")
            ptype = str(row.get("param_type") or "").strip().lower()
            is_lab = table == "hosp/labevents"
            is_numeric_chart = (
                table == "icu/chartevents" and ptype in _NUMERIC_PARAM_TYPES
            )
            if is_lab or is_numeric_chart or (table == "icu/chartevents" and not ptype):
                still.append(iid)
            else:
                dropped.append(iid)
                findings.append(
                    GateFinding(
                        "role_measurability",
                        "error",
                        f"itemid {iid} '{row.get('label')}' is "
                        f"{table} param_type={row.get('param_type')!r} — not a "
                        "numeric measurement; declare role=event/duration instead",
                    )
                )
        kept = still
    elif draft.role in (EVENT_ROLE, DURATION_ROLE):
        for iid in list(kept):
            table = str(rows[iid].get("table") or "")
            if table not in _EVENT_TABLES and table != "icu/chartevents":
                findings.append(
                    GateFinding(
                        "role_measurability",
                        "warning",
                        f"itemid {iid} table={table} is unusual for an "
                        f"{draft.role} concept — confirm",
                    )
                )

    if not kept:
        findings.append(
            GateFinding(
                "role_measurability",
                "error",
                "no itemids survived specimen/role gates",
            )
        )
        return ConceptProposalValidation(
            status="rejected",
            findings=tuple(findings),
            resolved_itemids=(),
            dropped_itemids=tuple(dict.fromkeys(dropped + invalid)),
        )

    # --- Gate 3b: declared metadata completeness (measurement) -----------
    # A measurement concept with no declared unit/bounds means the
    # distribution-plausibility gate cannot run — flag it so a clean
    # ``needs_human_review`` does not hide a skipped check.
    if draft.role == MEASUREMENT_ROLE:
        if not (draft.unit or "").strip():
            findings.append(
                GateFinding(
                    "declared_metadata",
                    "warning",
                    "measurement concept has no declared unit — set it before use",
                )
            )
        if draft.min_value is None or draft.max_value is None:
            findings.append(
                GateFinding(
                    "declared_metadata",
                    "warning",
                    "measurement concept has no declared [min, max] — the "
                    "real-distribution plausibility gate could not run",
                )
            )

    # --- Gate 4: unit consistency (catalog hint) -------------------------
    units = {
        str(rows[iid].get("unitname") or "").strip()
        for iid in kept
        if str(rows[iid].get("unitname") or "").strip()
        and str(rows[iid].get("unitname")).strip().lower() != "none"
    }
    if len(units) > 1:
        findings.append(
            GateFinding(
                "unit_consistency",
                "warning",
                f"kept itemids span multiple catalog units {sorted(units)} — "
                "declare a harmonization rule or split the concept",
            )
        )

    # --- Gate 5: real-data distribution & bounds -------------------------
    distribution: List[DistributionStat] = []
    if distribution_probe is not None:
        table = rows[kept[0]].get("table", "")
        stats = distribution_probe(kept, table)
        total_cov = 0.0
        for iid in kept:
            st = stats.get(iid)
            if st is None:
                findings.append(
                    GateFinding(
                        "distribution",
                        "warning",
                        f"itemid {iid} returned no distribution from the probe",
                    )
                )
                continue
            distribution.append(st)
            total_cov += st.coverage_fraction
            if len(st.units) > 1:
                findings.append(
                    GateFinding(
                        "unit_consistency",
                        "error",
                        f"itemid {iid} carries multiple value units {st.units} "
                        "in the real data — extraction would mix scales",
                    )
                )
            if (
                draft.role == MEASUREMENT_ROLE
                and st.p50 is not None
                and draft.min_value is not None
                and draft.max_value is not None
                and not (draft.min_value <= st.p50 <= draft.max_value)
            ):
                findings.append(
                    GateFinding(
                        "distribution_bounds",
                        "error",
                        f"itemid {iid} real median {st.p50} is outside the "
                        f"declared [{draft.min_value}, {draft.max_value}] — "
                        "wrong item, wrong unit, or wrong bounds",
                    )
                )
        if total_cov < min_coverage_fraction:
            findings.append(
                GateFinding(
                    "coverage",
                    "error",
                    f"joint coverage {total_cov:.4f} < {min_coverage_fraction} "
                    "— too sparse to analyze",
                )
            )
    else:
        findings.append(
            GateFinding(
                "distribution",
                "warning",
                "no distribution probe supplied — proposal NOT validated against "
                "real data; cannot be approved until probed",
            )
        )

    status = (
        "rejected"
        if any(f.severity == "error" for f in findings)
        else "needs_human_review"
    )
    return ConceptProposalValidation(
        status=status,
        findings=tuple(findings),
        resolved_itemids=tuple(kept),
        dropped_itemids=tuple(dict.fromkeys(dropped + invalid)),
        distribution=tuple(distribution),
    )
