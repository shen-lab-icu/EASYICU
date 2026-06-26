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

from .easyicu_case_builder import index_export_package, resolve_exported_concept


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
        has_tags = any(c.methodology for c in self.concepts)
        lines = [
            f"Available concepts in the provided data ({len(self.concepts)} "
            f"total). The agent may ONLY request concepts from this list; "
            f"anything else counts as not-extracted (advise re-extraction):"
        ]
        if has_tags:
            lines.append(
                "  [⚠ tags are methodological cautions for the role you assign a "
                "concept — heed them when choosing exposure / outcome / covariates.]"
            )
        for category, items in sorted(self.by_category().items()):
            shown = items if max_per_category <= 0 else items[:max_per_category]
            lines.append(f"\n[{category}] ({len(items)})")
            for c in sorted(shown, key=lambda x: x.concept_id):
                desc = f" — {c.description}" if c.description else ""
                warn = f"  ⚠ {c.methodology}" if c.methodology else ""
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
        from ..concept.loader import _load_concept_dict_cached  # heavy import

        raw = _load_concept_dict_cached()
    except Exception:
        return {}
    meta: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw, Mapping):
        return {}
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
        meta[str(cid)] = {
            "description": str(description),
            "category": str(category),
        }
    return meta


def _methodology_tag(concept_id: str, category: str) -> str:
    """Compact advisory methodology tag for a concept (best-effort).

    Derived from concept structure by the rules layer; never raises — a failure
    just yields no tag (the catalog still lists the concept).
    """
    try:
        from .icu_rules import concept_methodology_tag

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
    meta = _concept_dict_meta()
    concepts: List[CatalogConcept] = []
    for cid, info in index.items():
        m = meta.get(cid, {})
        category = m.get("category", "")
        concepts.append(
            CatalogConcept(
                concept_id=cid,
                description=m.get("description", ""),
                category=category,
                file_name=str(info.get("file_name", "")),
                n_rows=int(info.get("rows", 0) or 0),
                methodology=_methodology_tag(cid, category),
            )
        )
    return AvailableCatalog(source=str(export_dir), concepts=concepts)


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
    index = {c.concept_id: {"columns": [c.concept_id]} for c in catalog.concepts}
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
            resolved[concept] = hit
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
