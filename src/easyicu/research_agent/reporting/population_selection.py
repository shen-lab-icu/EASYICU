"""The population a run analyzed, as its manuscript must state it.

A research question names the population a researcher has in mind; it selects
no one.  Rows enter an analysis in two typed ways only:

* the selection the source export applied
  (``research_context.export_selection``): its inclusion and exclusion
  contracts and a concept-derived population;
* the plan's cohort predicates, when it states any.

This module reads those owners and says which population the manuscript
describes.  The Writer receives it as the ANALYZED POPULATION block, so no
section calls the analyzed stays a population only the question names.  An
export whose selection the host did not record may have applied criteria the
context does not state (a preset's age bound, a prepared package's own
cohort), so the block then asserts neither that it applied any nor that the
analysis was unrestricted; only a criterion the host applied itself is stated
as applied.  The host cites a population statement where the
manuscript states its population only when the plan selected its rows by
predicate.  Otherwise the registered owners such a statement could cite
record the export or the question, not a selection.

The execution kernel's authority is to own the same record, with these field
names, for the manuscript's population method fact and its claim check; this
module then calls that builder instead of reading the owners itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

from ..research_context.concept_population import (
    ConceptCohortWindow,
    ConceptCohortWindowError,
)
from ..research_context.export_selection import AppliedContracts, export_applied_selection
from ..schema import AnalysisPlan, ResearchContext


SourceScope = Literal[
    "predicate_selected",
    "all_input_rows_of_contracted_export",
    "all_icu_stays_of_source_export",
    "all_input_rows_of_unrecorded_export",
]

#: Receipt reason for a population statement the host declined to cite.
POPULATION_STATEMENT_NOT_HOST_CITED = "population_statement_not_host_cited"

#: Items listed on one line of the Writer's block; the rest are counted.
_MAX_LISTED = 12

#: The sentence frames a report uses to say who an analysis selected: rows
#: that were included, excluded, eligible or enrolled, its inclusion or
#: exclusion criteria, what a study population or cohort comprised, and a
#: restriction to a group.  A frame counts only where the manuscript states
#: its population (``is_population_place``); elsewhere the same words name a
#: model's terms or another study's population.
_POPULATION_STATEMENT_RE = re.compile(
    r"\b(?:"
    r"(?:were|was)\s+(?:(?:not|only)\s+)?(?:included|excluded|eligible|enrolled|selected)"
    r"|(?:we|this\s+study|the\s+study)\s+(?:included|excluded|enrolled|selected)"
    r"|(?:inclusion|exclusion|eligibility)\s+criteri(?:on|a)"
    r"|eligible\s+(?:if|when|for|stays|patients|admissions|records)"
    r"|(?:study|target|source|analy[sz]ed|analytic)\s+population\s+"
    r"(?:was|were|comprised|consisted|included)"
    r"|cohort\s+(?:comprised|consisted\s+of|included|was\s+(?:restricted|limited|defined))"
    r"|(?:restricted|limited)\s+to\s+(?:patients|adults|stays|admissions|individuals|those|records)"
    r"|included\s+(?:all\s+|every\s+|only\s+)?"
    r"(?:adult\s+|adults\b|patients|stays|icu\s+stays|admissions|individuals|records)"
    r")",
    re.I,
)

#: The Methods subsection that states the population (the Writer contract's
#: first required Methods subsection) and the Abstract's Methods paragraph.
POPULATION_SUBSECTION = "Study design and cohort"
_METHODS_HEADING_RE = re.compile(r"##\s+(?:\d+\.?\s+)?methods\s*", re.I)
_POPULATION_SUBSECTION_RE = re.compile(
    r"###\s+(?:\d+(?:\.\d+)*\.?\s+)?" + re.escape(POPULATION_SUBSECTION) + r"\s*", re.I
)
_ABSTRACT_HEADING_RE = re.compile(r"##\s+abstract\s*", re.I)
_ABSTRACT_METHODS_LABEL_RE = re.compile(r"(?:[-*]\s*)?\*\*methods:?\*\*", re.I)


@dataclass(frozen=True)
class AnalyzedPopulation:
    """Which rows the analysis included, from the plan and the source export."""

    selection_mode: Literal["all_input_rows", "predicate_filtered"]
    inclusion_predicates: tuple[Mapping[str, Any], ...]
    exclusion_predicates: tuple[Mapping[str, Any], ...]
    #: The export's contracts known to be applied
    #: (``ExportAppliedSelection.known_applied``).
    applied_contracts: AppliedContracts
    concept_population: Optional[ConceptCohortWindow]
    source_scope: SourceScope
    #: Whether the host recorded the export's whole selection
    #: (``ExportAppliedSelection.recorded``).  Without it the concept
    #: population is declared, not known to be applied.
    source_selection_recorded: bool = False
    #: The contracts declared but not known to be applied.
    unverified_contracts: AppliedContracts = AppliedContracts()

    def record(self) -> dict[str, Any]:
        """The JSON shape both owners of this record agree on."""

        concept = self.concept_population
        return {
            "selection_mode": self.selection_mode,
            "inclusion_predicates": [dict(item) for item in self.inclusion_predicates],
            "exclusion_predicates": [dict(item) for item in self.exclusion_predicates],
            "applied_contracts": _contracts_record(self.applied_contracts),
            "unverified_contracts": _contracts_record(self.unverified_contracts),
            "concept_population": (
                {
                    "definition": concept.definition,
                    "window_end_hours": concept.window_end_hours,
                }
                if concept is not None
                else None
            ),
            "source_scope": self.source_scope,
            "source_selection_recorded": self.source_selection_recorded,
        }


def analyzed_population(
    *,
    plan: AnalysisPlan | None,
    context: ResearchContext | None,
) -> AnalyzedPopulation | None:
    """The analyzed population, or ``None`` when the typed owners cannot state it.

    A plan without a cohort, or an export whose concept-population record is
    unreadable, gives ``None``: the host then states no population and vouches
    for no population statement.  A predicate-filtered cohort with no
    predicate selects every row, as the trajectory design reads it.  Without
    the host's record of the export's selection, an export with no stated
    criterion is not every ICU stay: what it selected is unknown.
    """

    cohort = plan.cohort if plan is not None else None
    if cohort is None or context is None:
        return None
    try:
        selection = export_applied_selection(context)
    except ConceptCohortWindowError:
        return None
    inclusion = tuple(predicate.to_dict() for predicate in cohort.inclusion)
    exclusion = tuple(predicate.to_dict() for predicate in cohort.exclusion)
    contracts = selection.contracts
    concept = selection.concept_population
    scope: SourceScope
    if cohort.selection_mode != "all_input_rows" and (inclusion or exclusion):
        scope = "predicate_selected"
    elif not selection.recorded:
        scope = "all_input_rows_of_unrecorded_export"
    elif contracts.inclusion or contracts.exclusion or concept is not None:
        scope = "all_input_rows_of_contracted_export"
    else:
        scope = "all_icu_stays_of_source_export"
    return AnalyzedPopulation(
        selection_mode=cohort.selection_mode,
        inclusion_predicates=inclusion,
        exclusion_predicates=exclusion,
        applied_contracts=selection.known_applied,
        concept_population=concept,
        source_scope=scope,
        source_selection_recorded=selection.recorded,
        unverified_contracts=selection.unverified,
    )


def host_may_cite_population_statement(population: AnalyzedPopulation | None) -> bool:
    """Only a plan's predicates give a population statement a selection to cite."""

    return population is not None and population.source_scope == "predicate_selected"


def states_population_selection(sentence: str) -> bool:
    return _POPULATION_STATEMENT_RE.search(sentence) is not None


def is_population_place(*, section: str, subsection: str, line: str) -> bool:
    """Whether a line stands where the manuscript states its population.

    That is the Methods subsection the Writer contract names for it, or the
    Abstract's Methods paragraph.  ``section`` and ``subsection`` are the
    line's ``##`` and ``###`` headings.
    """

    if _METHODS_HEADING_RE.fullmatch(section.strip()):
        return _POPULATION_SUBSECTION_RE.fullmatch(subsection.strip()) is not None
    return (
        _ABSTRACT_HEADING_RE.fullmatch(section.strip()) is not None
        and _ABSTRACT_METHODS_LABEL_RE.match(line.strip()) is not None
    )


#: The Writer reads this text at the head of every section instruction.  It
#: is shared Writer policy outside the section specs, so a change to it
#: bumps MANUSCRIPT_WRITER_CONTRACT_VERSION, as any such policy change does.
_SCOPE_TEXT: dict[str, str] = {
    "predicate_selected": (
        "the source export's rows that meet the plan's cohort predicates below."
    ),
    "all_input_rows_of_contracted_export": (
        "every input row of the source export; the plan applied no inclusion "
        "or exclusion predicate."
    ),
    "all_icu_stays_of_source_export": (
        "every ICU stay in the source export; neither the export nor the plan "
        "applied an inclusion or exclusion criterion."
    ),
    "all_input_rows_of_unrecorded_export": (
        "every input row of the source export, which the plan did not filter."
    ),
}

_DESCRIBE_RULE = (
    "- Describe this study's population only as stated here. Eligibility for one "
    "analysis (a landmark risk set, observed windows, complete data) comes from "
    "the executed method boundary."
)
_QUESTION_RULE = (
    "- The research question, and any criterion or cohort wording in RESEARCH "
    "CONTEXT, select no one: never call this study's rows, stays, patients or "
    "cohort a narrower population they name (a condition, treatment, procedure, "
    "setting or age group) that is not listed here; {consequence} Background "
    "about that condition is allowed."
)
#: Only a recorded selection shows the analysis was not restricted to it.
_RECORDED_CONSEQUENCE = "in Methods, say once that the analysis was not restricted to it."
_UNRECORDED_CONSEQUENCE = (
    "the export's selection is not recorded, so never say whether the analysis "
    "was restricted to it."
)


def writer_population_block(population: AnalyzedPopulation | None) -> str:
    """The ANALYZED POPULATION block every Writer section receives."""

    lines = ["ANALYZED POPULATION (host-stated; the only population this study analyzed):"]
    recorded = population is not None and population.source_selection_recorded
    if population is None:
        lines.append(
            "- Rows analyzed: not stated by the host, because the plan states no "
            "cohort or the export's selection record cannot be read. State no "
            "criterion as applied."
        )
    else:
        lines.append("- Rows analyzed: " + _SCOPE_TEXT[population.source_scope])
        concept = _concept_items(population.concept_population)
        applied = _contract_items(population.applied_contracts)
        if recorded:
            lines.append("- Criteria applied before analysis: " + _listed(applied + concept))
        else:
            # Only the host's own criteria are known; "none" would claim more.
            if applied:
                lines.append("- Criteria applied before analysis: " + _listed(applied))
            lines.append(
                "- Export's selection: not recorded; declared criteria (unverified): "
                + _listed(_contract_items(population.unverified_contracts) + concept)
            )
        if population.source_scope == "predicate_selected":
            lines.append(
                "- Plan inclusion predicates: "
                + _listed([_predicate_text(item) for item in population.inclusion_predicates])
            )
            lines.append(
                "- Plan exclusion predicates: "
                + _listed([_predicate_text(item) for item in population.exclusion_predicates])
            )
    lines.append(_DESCRIBE_RULE)
    lines.append(
        _QUESTION_RULE.format(
            consequence=_RECORDED_CONSEQUENCE if recorded else _UNRECORDED_CONSEQUENCE
        )
    )
    return "\n".join(lines) + "\n\n"


def _contracts_record(contracts: AppliedContracts) -> dict[str, list[str]]:
    return {"inclusion": list(contracts.inclusion), "exclusion": list(contracts.exclusion)}


def _contract_items(contracts: AppliedContracts) -> list[str]:
    items = [f"inclusion: {item}" for item in contracts.inclusion]
    items.extend(f"exclusion: {item}" for item in contracts.exclusion)
    return items


def _concept_items(concept: ConceptCohortWindow | None) -> list[str]:
    if concept is None:
        return []
    return [
        f"concept-derived population {concept.definition}: a stay entered on a "
        f"positive {concept.definition} record at or before "
        f"{concept.window_end_hours:g} h after ICU admission"
    ]


def _listed(items: Sequence[str]) -> str:
    if not items:
        return "none."
    shown = "; ".join(items[:_MAX_LISTED])
    hidden = len(items) - _MAX_LISTED
    return shown + (f"; and {hidden} more." if hidden > 0 else ".")


def _offset_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:g}"
    return str(value)


def _predicate_text(predicate: Mapping[str, Any]) -> str:
    """One plan predicate as a reader checks it: what it tests, over which hours."""

    window = predicate.get("time_window") or {}
    op = str(predicate.get("op") or "")
    test = f"{predicate.get('concept_id')} {op}"
    if op not in {"missing", "not_missing"}:
        test += f" {predicate.get('value')!r}"
    return (
        f"{test} ({predicate.get('aggregation')} over "
        f"{_offset_text(window.get('start_offset_hours'))} to "
        f"{_offset_text(window.get('end_offset_hours'))} h from {window.get('anchor')})"
    )


__all__ = [
    "AnalyzedPopulation",
    "AppliedContracts",
    "POPULATION_STATEMENT_NOT_HOST_CITED",
    "POPULATION_SUBSECTION",
    "SourceScope",
    "analyzed_population",
    "host_may_cite_population_statement",
    "is_population_place",
    "states_population_selection",
    "writer_population_block",
]
