"""The population a run analyzed, as its manuscript must state it.

A research question names the population a researcher has in mind; it selects
no one.  Rows enter an analysis in two typed ways only:

* the selection the source export applied
  (``research_context.export_selection``): its inclusion and exclusion
  contracts and a concept-derived population;
* the plan's cohort predicates, when it states any.

This module reads those owners and says which population the manuscript
describes.  The Writer receives it as the ANALYZED POPULATION block, so no
section calls the analyzed stays a population only the question names.  A
population no plan predicate selected gets a name for the title and the
Abstract, its unit and the source database's display name ("ICU stays in"
the database), so that no title calls a study population the rows of an
export.  An
export whose selection the host did not record may have applied criteria the
context does not state (a preset's age bound, a prepared package's own
cohort), so the block then asserts neither that it applied any nor that the
analysis was unrestricted; only a criterion the host applied itself is stated
as applied.  A prepared package that declares itself the study's cohort is
that cohort by its own declaration, which the host accepted without verifying
it, so the block names the cohort only as the package's declaration.  A
population criterion the plan names but no predicate applies
(``CohortDefinition.unapplied_population_criteria``) selected no one: the
block lists it as not applied, so the manuscript says so and never calls the
rows by it.  The host cites a population statement where the
manuscript states its population only when the plan selected its rows by
predicate.  Otherwise the registered owners such a statement could cite
record the export or the question, not a selection.

A recorded selection also states how many ICU stays each criterion excluded,
from the source database to the analysis input
(``export_selection.export_selection_counts``), and how the export capped
its stays if it did.  The block lists them so that the manuscript can report
its selection as a flow diagram does; the host registers the counts as
``research_context`` claims.  Without counts that chain, the block states
none, and why is kept for audit only.

The execution kernel's authority is to own the same record, with these field
names, for the manuscript's population method fact and its claim check; this
module then calls that builder instead of reading the owners itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

from easyicu.databases.profiles import get_database_profile

from ..research_context.concept_population import (
    ConceptCohortWindow,
    ConceptCohortWindowError,
)
from ..research_context.export_selection import (
    AppliedContracts,
    ExportCap,
    SelectionCounts,
    SelectionStep,
    export_applied_selection,
    export_selection_counts,
)
from ..schema import AnalysisPlan, ResearchContext


SourceScope = Literal[
    "predicate_selected",
    "all_input_rows_of_contracted_export",
    "all_icu_stays_of_source_export",
    "all_input_rows_of_unrecorded_export",
    "all_input_rows_of_declared_package",
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
    #: How the host knows the export's selection (``ExportAppliedSelection.basis``);
    #: ``None`` without a record.
    source_selection_basis: Optional[str] = None
    #: Population criteria the plan names that no predicate applies
    #: (``CohortDefinition.unapplied_population_criteria``); they select no row.
    unapplied_population_criteria: tuple[str, ...] = ()
    #: The ICU stays each selection criterion excluded, from the source
    #: database to the analysis input; ``None`` without counts that chain.
    selection_counts: Optional[SelectionCounts] = None
    #: How the export capped the stays that met its criteria, if it did.
    export_cap: Optional[ExportCap] = None
    #: Why ``selection_counts`` is ``None``: for audit, never in ``record()``
    #: or the Writer's block.
    selection_counts_unavailable: Optional[str] = None
    #: The source database's display name (its ``databases.profiles``
    #: profile), which names the population for readers; ``None`` for a
    #: database without a profile.  A reader label, not part of ``record()``.
    source_database_label: Optional[str] = None

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
            "source_selection_basis": self.source_selection_basis,
            "unapplied_population_criteria": list(self.unapplied_population_criteria),
            "selection_counts": (
                self.selection_counts.record()
                if self.selection_counts is not None
                else None
            ),
            "export_cap": (
                self.export_cap.record() if self.export_cap is not None else None
            ),
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
    criterion is not every ICU stay: what it selected is unknown.  A package
    that declares itself the study's cohort is all of that declared cohort.
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
    elif selection.basis == "package_declaration":
        scope = "all_input_rows_of_declared_package"
    elif not selection.recorded:
        scope = "all_input_rows_of_unrecorded_export"
    elif contracts.inclusion or contracts.exclusion or concept is not None:
        scope = "all_input_rows_of_contracted_export"
    else:
        scope = "all_icu_stays_of_source_export"
    counts = export_selection_counts(context)
    return AnalyzedPopulation(
        selection_mode=cohort.selection_mode,
        inclusion_predicates=inclusion,
        exclusion_predicates=exclusion,
        applied_contracts=selection.known_applied,
        concept_population=concept,
        source_scope=scope,
        source_selection_recorded=selection.recorded,
        unverified_contracts=selection.unverified,
        source_selection_basis=selection.basis,
        unapplied_population_criteria=tuple(
            getattr(cohort, "unapplied_population_criteria", None) or ()
        ),
        selection_counts=counts.counts,
        export_cap=counts.cap,
        selection_counts_unavailable=counts.unavailable,
        source_database_label=_database_label(context.cohort.database),
    )


def _database_label(database: Any) -> Optional[str]:
    try:
        label = get_database_profile(str(database or "")).display_name
    except KeyError:
        return None
    return " ".join(str(label or "").split()) or None


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
    "all_input_rows_of_declared_package": (
        "every input row of a prepared cohort package that declares itself this "
        "study's cohort; the plan did not filter it."
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
#: A criterion the plan names but did not apply selected no one.
_UNAPPLIED_RULE = (
    "- A criterion listed as not applied is not part of this study's population: "
    "say it was not applied, and never call this study's rows, stays, patients or "
    "cohort by it."
)
#: A declared package is the study's cohort only by its own word.
_DECLARED_CONSEQUENCE = (
    "the package declares itself this study's cohort, so name that cohort only as "
    "the package's declaration, never as selected or verified by this study."
)
#: The rows of an unfiltered input are a reader's population only by name.
_NAME_RULE = (
    "- In the title and the Abstract, call this population by its population "
    "name, never by its rows, records or export."
)


def writer_population_block(population: AnalyzedPopulation | None) -> str:
    """The ANALYZED POPULATION block every Writer section receives."""

    lines = ["ANALYZED POPULATION (host-stated; the only population this study analyzed):"]
    recorded = population is not None and population.source_selection_recorded
    declared = (
        population is not None
        and population.source_selection_basis == "package_declaration"
    )
    name = _population_name(population) if population is not None else None
    if population is None:
        lines.append(
            "- Rows analyzed: not stated by the host, because the plan states no "
            "cohort or the export's selection record cannot be read. State no "
            "criterion as applied."
        )
    else:
        lines.append("- Rows analyzed: " + _SCOPE_TEXT[population.source_scope])
        if name is not None:
            lines.append(f"- Population name: {name}.")
        concept = _concept_items(population.concept_population)
        applied = _contract_items(population.applied_contracts)
        if recorded:
            lines.append("- Criteria applied before analysis: " + _listed(applied + concept))
            if population.selection_counts is not None:
                lines.append(_selection_counts_line(population.selection_counts))
            if (
                population.export_cap is not None
                and population.export_cap.cut is not False
            ):
                lines.append(_export_cap_line(population.export_cap))
        elif declared:
            if applied:
                lines.append("- Criteria applied before analysis: " + _listed(applied))
            stated = _contract_items(population.unverified_contracts) + concept
            lines.append(
                "- Package's declaration (accepted, not verified by the host): it holds "
                "this study's cohort, as the cohort wording in RESEARCH CONTEXT states it"
                + ("; declared criteria: " + _listed(stated) if stated else ".")
            )
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
        if population.unapplied_population_criteria:
            lines.append(
                "- Population criteria the plan names but did not apply (no "
                "predicate applies them, so they selected no one): "
                + _listed(list(population.unapplied_population_criteria))
            )
    lines.append(_DESCRIBE_RULE)
    if name is not None:
        lines.append(_NAME_RULE)
    if population is not None and population.unapplied_population_criteria:
        lines.append(_UNAPPLIED_RULE)
    lines.append(
        _QUESTION_RULE.format(
            consequence=(
                _RECORDED_CONSEQUENCE
                if recorded
                else _DECLARED_CONSEQUENCE
                if declared
                else _UNRECORDED_CONSEQUENCE
            )
        )
    )
    return "\n".join(lines) + "\n\n"


def _population_name(population: AnalyzedPopulation) -> Optional[str]:
    """What the title and Abstract call a population no plan predicate selected.

    Plan predicates name the population they select, and a declared package
    is named only as its declaration, so neither gets a name here; nor does
    a database without a profile, whose population the block then states by
    its scope alone.  The criteria known to be applied before analysis
    qualify the name, as the block lists them.
    """

    if population.source_scope in (
        "predicate_selected",
        "all_input_rows_of_declared_package",
    ):
        return None
    label = population.source_database_label
    if label is None:
        return None
    applied = _contract_items(population.applied_contracts)
    if population.source_selection_recorded:
        applied += _concept_items(population.concept_population)
    if applied:
        return f"ICU stays in {label} that meet the criteria applied before analysis"
    return f"ICU stays in {label}"


#: How a cap chose the stays it kept, for the rules that keep the first ones.
_CAP_RULE_TEXT = {
    "identifier_text_order": "the first by identifier compared as text",
    "identifier_order": "the first by identifier",
    "source_file_order": "the first in the order the source's stay table lists them",
}


def _selection_counts_line(counts: SelectionCounts) -> str:
    """The stays from the source database to the analysis input, step by step."""

    parts = [f"the source database held {counts.source_total:,}"]
    parts += [_step_text(step) for step in counts.steps if step.stage == "export"]
    parts.append(f"the export held {counts.exported:,}")
    parts += [_step_text(step) for step in counts.steps if step.stage == "host"]
    parts.append(f"the analysis input held {counts.analysis_input:,}")
    return (
        "- Selection counts, in ICU stays, never patients (cite research_context): "
        + "; ".join(parts)
        + "."
    )


def _step_text(step: SelectionStep) -> str:
    missing = (
        f" ({step.n_excluded_missing:,} of them for want of a value)"
        if step.n_excluded_missing
        else ""
    )
    return (
        f"{_step_label(step)}: {step.n_excluded:,} excluded{missing}, "
        f"{step.n_remaining:,} remain"
    )


def _export_cap_line(cap: ExportCap) -> str:
    kept = f"- Export cap: the export kept at most {cap.max_patients:,} stays, "
    if cap.rule == "seeded_random_sample":
        return kept + "a random sample with a fixed seed."
    if cap.rule == "unrecorded":
        # How it chose them is unknown, so whether they are random is too.
        return kept + (
            "chosen in an order the export did not record, so they cannot be "
            "taken as a random sample."
        )
    return kept + (
        f"{_CAP_RULE_TEXT[cap.rule]}; they are not a random sample, and the "
        "stays kept may cluster by hospital or period."
    )


def _step_label(step: SelectionStep) -> str:
    """What a selection step excluded, in the words a flow diagram uses."""

    parameters = step.parameters
    if step.criterion == "age":
        low, high = parameters.get("age_min"), parameters.get("age_max")
        if low is not None and high is not None:
            return f"age outside {_offset_text(low)} to {_offset_text(high)} years"
        if low is not None:
            return f"age under {_offset_text(low)} years"
        return f"age over {_offset_text(high)} years"
    if step.criterion in {"first_icu_stay", "first_icu_stay_restriction"}:
        label = (
            "the patient's first ICU stay"
            if parameters.get("first_icu_stay") is False
            else "not the patient's first ICU stay"
        )
        if step.stage == "host":
            label += " (restricted by the host)"
        return label
    if step.criterion == "los":
        bounds = []
        if parameters.get("los_min") is not None:
            bounds.append(f"shorter than {_offset_text(parameters['los_min'])} h")
        if parameters.get("los_max") is not None:
            bounds.append(f"longer than {_offset_text(parameters['los_max'])} h")
        return "ICU stay " + " or ".join(bounds)
    if step.criterion == "gender" and parameters.get("gender") is not None:
        return f"sex other than {parameters['gender']}"
    if step.criterion == "survived" and parameters.get("survived") is not None:
        return (
            "not alive at hospital discharge"
            if parameters["survived"]
            else "alive at hospital discharge"
        )
    if step.criterion == "has_sepsis" and parameters.get("has_sepsis") is not None:
        return "without Sepsis-3" if parameters["has_sepsis"] else "with Sepsis-3"
    if step.criterion == "concept_population" and parameters.get("definition"):
        return f"outside the concept-derived population {parameters['definition']}"
    return _STEP_LABELS.get(step.criterion, step.criterion.replace("_", " "))


#: The labels of the steps whose wording takes no parameter.
_STEP_LABELS = {
    "demographics": "the export's demographic criteria, together",
    "concept_population": "outside the concept-derived population",
    "icd": "the diagnosis-code (ICD) criteria",
    "cap": "beyond the export's cap",
}


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
