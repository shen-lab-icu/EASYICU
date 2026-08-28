"""Data-foundation catalog + coverage judgement for the research agent (L2).

This is the deterministic half of the "agent picks its own concepts" flow.
A real EasyICU user hands the agent only **what they happened to extract** —
maybe the full export, maybe a few modules they thought mattered, maybe a
pre-filtered cohort. So before any analysis the agent must answer two
questions, and both are answered here without an LLM:

1. **What is available?** — :func:`build_available_catalog` enumerates the
   concepts physically present in the provided export/cohort and enriches
   them with descriptions/categories from the EasyICU concept dictionary.
2. **Is it enough for this question?** — :func:`assess_coverage` compares the
   concepts the agent *asked for* against what is available and returns a
   verdict. When something is missing it says so and advises re-extraction
   (the web app surfaces this to the user) rather than silently proceeding
   on absent data — a "not extracted" gap is structural no-source, distinct
   from a measured-but-missing value.

The concept *selection* itself (which concepts a question needs) is the
agent's LLM judgement and lives in the data-foundation agent; this module
only provides the catalog it chooses from and checks its choice against
reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from easyicu.concept.metadata_sidecar import ColumnMetadataBinding

from ..intake.export_package import index_export_package, resolve_exported_concept


@dataclass
class CatalogConcept:
    """One concept physically available in the provided data."""

    concept_id: str
    description: str = ""
    category: str = ""
    file_name: str = ""
    n_rows: int = 0
    # Compact advisory methodology tag (derived from concept structure), e.g.
    # "treatment: confounder vs mediator?". Empty for a plain-safe concept.
    methodology: str = ""
    resolved_column: str = ""
    typed_metadata: bool = False
    column_role: str = ""
    selection_mode: str = "ordinary"
    selection_note: str = ""
    canonical_alternative: str = ""


@dataclass
class AvailableCatalog:
    """The concepts a provided export/cohort actually contains."""

    source: str
    concepts: List[CatalogConcept] = field(default_factory=list)

    def ids(self) -> List[str]:
        return [c.concept_id for c in self.concepts]

    def by_category(self) -> Dict[str, List[CatalogConcept]]:
        groups: Dict[str, List[CatalogConcept]] = {}
        for c in self.concepts:
            groups.setdefault(c.category or "uncategorized", []).append(c)
        return groups

    def render_for_prompt(self, *, max_per_category: int = 0) -> str:
        """Human/LLM-readable catalog grouped by category, with descriptions.

        ``max_per_category`` 0 means list all; a positive cap truncates long
        categories (the agent still sees the category exists).
        """
        has_tags = any(
            c.methodology or c.selection_mode != "ordinary" for c in self.concepts
        )
        lines = [
            f"Available concepts in the provided data ({len(self.concepts)} "
            f"total). The agent may ONLY request concepts from this list; "
            f"anything else counts as not-extracted (advise re-extraction):"
        ]
        if has_tags:
            lines.append(
                "  [⚠ tags are methodological cautions for the role you assign a "
                "concept — heed them when choosing exposure / outcome / covariates. "
                "An explicit-only concept may be selected only when the user "
                "positively names that variant.]"
            )
        for category, items in sorted(self.by_category().items()):
            shown = items if max_per_category <= 0 else items[:max_per_category]
            lines.append(f"\n[{category}] ({len(items)})")
            for c in sorted(shown, key=lambda x: x.concept_id):
                desc = f" — {c.description}" if c.description else ""
                warnings = [
                    value for value in (c.methodology, c.selection_note) if value
                ]
                warn = f"  ⚠ {'; '.join(warnings)}" if warnings else ""
                lines.append(f"  - {c.concept_id}{desc}{warn}")
            if max_per_category > 0 and len(items) > max_per_category:
                lines.append(f"  … (+{len(items) - max_per_category} more)")
        return "\n".join(lines)


@dataclass
class CoverageReport:
    """Verdict on whether the available data covers the requested concepts."""

    requested: List[str]
    resolved: Dict[str, str]  # requested -> actual available column
    available: List[str]  # requested concepts that resolved
    missing: List[str]  # requested concepts absent from the provided data
    advice: List[str] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        """Whether every *requested* concept resolved — not scientific adequacy.

        ``missing`` is derived from ``requested``, so this answers "did we get
        what we asked for", never "is this enough to answer the question". A
        request that never named a needed confounder, time anchor or censoring
        variable is reported ``sufficient``, and so is an empty request: both
        produce an empty ``missing`` list, indistinguishable from full
        coverage. A caller that needs the second question answered must state
        its requirements (``required_feature_concepts``) and check those; it
        cannot recover them from this property.
        """
        return not self.missing

    def to_dict(self) -> Dict[str, object]:
        return {
            "sufficient": self.sufficient,
            "requested": list(self.requested),
            "resolved": dict(self.resolved),
            "available": list(self.available),
            "missing": list(self.missing),
            "advice": list(self.advice),
        }


# ---------------------------------------------------------------------------
# Concept-dictionary enrichment (descriptions / categories), best-effort.
# ---------------------------------------------------------------------------


def _concept_dict_meta() -> Dict[str, Dict[str, str]]:
    """Return ``concept_id -> {description, category}`` from the concept dict.

    Best-effort: the concept-dict schema varies, so we defensively pull a few
    likely fields and fall back to empty strings. Never raises — a missing or
    malformed dictionary just yields no enrichment.
    """
    try:
        from easyicu.concept.loader import _load_concept_dict_cached

        raw = _load_concept_dict_cached()
    except Exception:
        raw = {}
    meta: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw, Mapping):
        raw = {}
    from easyicu.concept.export_metadata import concept_declares_event_status
    from easyicu.concept.schema import ConceptDefinition

    for cid, entry in raw.items():
        if not isinstance(entry, Mapping):
            continue
        description = (
            entry.get("description") or entry.get("desc") or entry.get("name") or ""
        )
        category = (
            entry.get("category")
            or entry.get("class")
            or entry.get("group")
            or entry.get("unit")
            or ""
        )
        try:
            definition = ConceptDefinition.from_name_and_payload(str(cid), entry)
        except Exception:
            definition = None
        meta[str(cid)] = {
            "description": str(description),
            "category": str(category),
            "column_role": (
                "event_status"
                if concept_declares_event_status(str(cid), definition)
                else ""
            ),
        }
    # Derived catalog outputs need not exist in the raw extraction dictionary.
    # Their public descriptions are still owner metadata, not a filename or
    # dtype inference, so add them as a truthful fallback for legacy exports.
    from easyicu.concept.catalog import CONCEPT_DESCRIPTIONS, CONCEPT_DICTIONARY

    for cid, (name, _name_zh, _unit) in CONCEPT_DICTIONARY.items():
        row = meta.setdefault(str(cid), {})
        description, _description_zh = CONCEPT_DESCRIPTIONS.get(cid, ("", ""))
        row.setdefault("description", str(description or name or ""))
        row.setdefault("category", "")
        row.setdefault("column_role", "")
    return meta


def _methodology_tag(concept_id: str, category: str) -> str:
    """Compact advisory methodology tag for a concept (best-effort).

    Derived from concept structure by the rules layer; never raises — a failure
    just yields no tag (the catalog still lists the concept).
    """
    try:
        from ..icu_rules import concept_methodology_tag

        return concept_methodology_tag(concept_id, category=category)
    except Exception:
        return ""


def build_available_catalog(export_dir: Union[str, Path]) -> AvailableCatalog:
    """Enumerate the concepts present in an EasyICU export package.

    Reads which concepts are physically present (:func:`index_export_package`)
    and enriches each with a description/category from the concept dictionary
    when known. This is "what the user gave us" — the menu the agent selects
    from.
    """
    index = index_export_package(export_dir)
    concepts: List[CatalogConcept] = []
    typed_index = any(info.get("column_metadata_v2") is True for info in index.values())
    # A typed package already sealed its prompt-facing semantics.  Re-reading
    # the mutable packaged dictionary here would create a second authority and
    # allow descriptions/categories to drift after export.
    meta = {} if typed_index else _concept_dict_meta()
    if typed_index:
        primary_by_source: Dict[str, List[tuple[str, Mapping[str, object]]]] = {}
        for column, info in index.items():
            source_concept = info.get("source_concept")
            if (
                not isinstance(source_concept, str)
                or not source_concept
                or info.get("column_metadata_role") not in {"value", "event_status"}
            ):
                continue
            primary_by_source.setdefault(source_concept, []).append((column, info))
        catalog_rows = [
            (source_concept, owned[0][0], owned[0][1], True)
            for source_concept, owned in sorted(primary_by_source.items())
            if len(owned) == 1
        ]
    else:
        # Preserve legacy manifest insertion order exactly.
        catalog_rows = [(cid, cid, info, False) for cid, info in index.items()]

    for cid, resolved_column, info, typed_metadata in catalog_rows:
        if typed_metadata:
            binding = info.get("column_metadata_binding")
            if not isinstance(binding, ColumnMetadataBinding):
                raise ValueError(
                    f"typed catalog concept {cid!r} lacks sealed column metadata"
                )
            description = binding.metadata.description or ""
            category = binding.metadata.category or ""
            column_role = binding.metadata.role.value
        else:
            m = meta.get(cid, {})
            description = m.get("description", "")
            category = m.get("category", "")
            column_role = m.get("column_role", "")
            if not column_role:
                from easyicu.concept.export_metadata import (
                    concept_declares_event_status,
                )

                column_role = (
                    "event_status" if concept_declares_event_status(cid) else ""
                )
        from easyicu.concept.selection_policy import concept_selection_policy

        selection_policy = concept_selection_policy(cid)
        concepts.append(
            CatalogConcept(
                concept_id=cid,
                description=description,
                category=category,
                file_name=str(info.get("file_name", "")),
                n_rows=int(info.get("rows", 0) or 0),
                resolved_column=resolved_column,
                typed_metadata=typed_metadata,
                column_role=column_role,
                methodology=_methodology_tag(cid, category),
                selection_mode=(
                    selection_policy.selection_mode if selection_policy else "ordinary"
                ),
                selection_note=(selection_policy.rationale if selection_policy else ""),
                canonical_alternative=(
                    selection_policy.canonical_alternative or ""
                    if selection_policy
                    else ""
                ),
            )
        )
    return AvailableCatalog(source=str(export_dir), concepts=concepts)


def build_database_capability_catalog(database: str) -> AvailableCatalog:
    """Build the metadata-only concept menu for one supported database.

    Unlike :func:`build_available_catalog`, this function never inspects an
    export package or patient table. It projects only concepts that EasyICU's
    packaged dictionary and database adapters declare executable. Planner-only
    Web runs use this menu to decide what a later narrow extraction must carry;
    it is not evidence that any patient has a recorded value.
    """

    from easyicu.concept.selection_policy import concept_selection_policy

    from ..concept_availability import (
        explain_concept_availability,
        normalize_database_name,
    )
    from ..concept_catalog import load_concept_catalog

    normalized_database = normalize_database_name(database)
    dictionary_catalog = load_concept_catalog()
    meta = _concept_dict_meta()
    concepts: List[CatalogConcept] = []
    for concept_id in dictionary_catalog.available_concepts:
        availability = explain_concept_availability(
            concept=concept_id,
            database=normalized_database,
        )
        if not availability.available:
            continue
        row = meta.get(concept_id, {})
        aliases = dictionary_catalog.concept_aliases.get(concept_id, [])
        description = str(row.get("description") or "").strip()
        if not description and aliases:
            description = ", ".join(str(value) for value in aliases[:4])
        category = str(
            row.get("category")
            or dictionary_catalog.concept_categories.get(concept_id)
            or ""
        )
        selection_policy = concept_selection_policy(concept_id)
        concepts.append(
            CatalogConcept(
                concept_id=concept_id,
                description=description,
                category=category,
                methodology=_methodology_tag(concept_id, category),
                column_role=str(row.get("column_role") or ""),
                selection_mode=(
                    selection_policy.selection_mode
                    if selection_policy
                    else "ordinary"
                ),
                selection_note=(
                    selection_policy.rationale if selection_policy else ""
                ),
                canonical_alternative=(
                    selection_policy.canonical_alternative or ""
                    if selection_policy
                    else ""
                ),
            )
        )
    return AvailableCatalog(
        source=f"easyicu-database-capability:{normalized_database}",
        concepts=concepts,
    )


def assess_coverage(
    requested_concepts: Sequence[str],
    catalog: AvailableCatalog,
    *,
    extra_available: Optional[Sequence[str]] = None,
) -> CoverageReport:
    """Judge whether ``catalog`` covers the agent's requested concepts.

    Uses the same conservative aliasing the reader uses (exact match wins; a
    unique ``<concept>_<suffix>`` resolves, ambiguous ones do not). For every
    requested concept that cannot be resolved we record it as missing and emit
    a re-extraction advisory — this is the message the web app shows the user
    ("this question needs X; your data does not have it, re-extract it").

    ``extra_available`` lets a caller mark concepts that exist outside the
    export index (e.g. columns already in a provided cohort parquet) as
    present, so a pre-filtered cohort is judged against its own columns.
    """
    index = {
        c.concept_id: (
            {
                "column_metadata_v2": True,
                "source_concept": c.concept_id,
                "column_metadata_role": "value",
            }
            if c.typed_metadata
            else {"columns": [c.concept_id]}
        )
        for c in catalog.concepts
    }
    by_id = {c.concept_id: c for c in catalog.concepts}
    extra = set(extra_available or [])
    resolved: Dict[str, str] = {}
    available: List[str] = []
    missing: List[str] = []
    advice: List[str] = []

    # category lookup for nicer re-extraction advice
    cat_of = {c.concept_id: c.category for c in catalog.concepts}

    for concept in dict.fromkeys(requested_concepts):  # dedupe, keep order
        if concept in extra:
            resolved[concept] = concept
            available.append(concept)
            continue
        hit = resolve_exported_concept(index, concept)
        if hit is not None:
            catalog_entry = by_id[hit]
            resolved[concept] = catalog_entry.resolved_column or hit
            available.append(concept)
        else:
            missing.append(concept)
            cat = cat_of.get(concept, "")
            module_hint = f" (module: {cat})" if cat else ""
            advice.append(
                f"`{concept}` is not in the provided data{module_hint}. "
                f"Re-extract it with EasyICU before this analysis can use it."
            )
    return CoverageReport(
        requested=list(dict.fromkeys(requested_concepts)),
        resolved=resolved,
        available=available,
        missing=missing,
        advice=advice,
    )
