"""The case-neutral methodology layer of the pre-plan literature pack.

The curated pre-plan references were, until now, entirely *topic* and *data
source*: what Sepsis-3 is, what SOFA is, what MIMIC-IV is, what ricu is.  Those
tell the Planner what the variables mean.  None of them tell it how an
observational ICU study is supposed to be designed, so every design decision the
Planner made -- when follow-up starts, what to do about repeated stays, whether
a continuous covariate enters linearly, how missingness is handled -- was made
without a single methodological source in front of it.

This module supplies that missing layer.  It is deliberately:

* **case-neutral** -- no benchmark task, variable, score, or database appears
  here.  Case-specific requirements belong in the study protocol.
* **method cards, not reading** -- each entry states the design question it
  answers and what a study must therefore report, so it survives being
  summarised into a prompt.
* **freezable** -- :func:`method_literature_digest` hashes the pack so a run can
  record which methodology it planned against, and two runs of the same study
  can be shown to have seen the same guidance.

Identifiers remain unset unless they were verified against the primary source
registry.  A confidently wrong identifier is worse than an absent one: it
looks verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Sequence

__all__ = [
    "METHOD_CARDS",
    "METHOD_LITERATURE_SCHEMA_VERSION",
    "MethodCard",
    "method_binding_support",
    "method_cards_for_layers",
    "method_literature_citations",
    "method_literature_digest",
    "method_literature_pack",
]

METHOD_LITERATURE_SCHEMA_VERSION = "easyicu.method_literature_pack/3"


@dataclass(frozen=True)
class MethodCard:
    """One reusable design requirement, tied to the source that motivates it.

    ``question`` is what a Planner is actually deciding when this card applies;
    ``requirement`` is what the study must then do or report.  Keeping those
    separate is what makes the card usable at plan time rather than only at
    review time.
    """

    id: str
    layer: str
    question: str
    requirement: str
    source_key: str
    source_title: str
    source_year: str
    source_venue: str = ""
    source_pmid: str = ""
    source_doi: str = ""
    source_url: str = ""
    # Exact planner design elements this *card* can govern.  A source may own
    # several cards (STROBE currently owns reporting, repeated-unit dependence,
    # and absolute/relative interpretation), so source-key membership alone is
    # not evidence that every card from that paper was actually applied.
    design_elements: tuple[str, ...] = field(default=())
    # Optional companion sources for the same requirement.
    also_see: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "question": self.question,
            "requirement": self.requirement,
            "source_key": self.source_key,
            "source_title": self.source_title,
            "source_year": self.source_year,
            "source_venue": self.source_venue,
            "source_pmid": self.source_pmid,
            "source_doi": self.source_doi,
            "source_url": self.source_url,
            "design_elements": list(self.design_elements),
            "also_see": list(self.also_see),
        }


METHOD_CARDS: tuple[MethodCard, ...] = (
    MethodCard(
        id="reporting_observational_study",
        layer="reporting_standard",
        question="What must this study report for a reader to appraise it?",
        requirement=(
            "Report the setting, eligibility criteria and their order, the "
            "exact source of each variable, the number of units at every stage "
            "of selection, how missing data were handled, and which analyses "
            "were pre-specified versus exploratory."
        ),
        source_key="strobe_2007",
        source_title=(
            "The Strengthening the Reporting of Observational Studies in "
            "Epidemiology (STROBE) statement: guidelines for reporting "
            "observational studies."
        ),
        source_year="2007",
        source_venue="Annals of Internal Medicine / BMJ / Lancet (co-published)",
        source_pmid="17938396",
        source_doi="10.7326/0003-4819-147-8-200710160-00010",
        source_url="https://pubmed.ncbi.nlm.nih.gov/17938396/",
        design_elements=("reporting",),
    ),
    MethodCard(
        id="reporting_routinely_collected_data",
        layer="reporting_standard",
        question="What extra reporting does routinely collected health data need?",
        requirement=(
            "State the database and its version, the exact codes or concept "
            "definitions used to construct every variable, how the analysis "
            "population was extracted from the source, and any data-cleaning "
            "step that changed the population. Routinely collected data were "
            "recorded for care, not for this study, so the extraction is part "
            "of the method."
        ),
        source_key="record_2015",
        source_title=(
            "The REporting of studies Conducted using Observational "
            "Routinely-collected health Data (RECORD) statement."
        ),
        source_year="2015",
        source_venue="PLoS Medicine",
        source_pmid="26440803",
        source_doi="10.1371/journal.pmed.1001885",
        source_url="https://pubmed.ncbi.nlm.nih.gov/26440803/",
        design_elements=("reporting",),
    ),
    MethodCard(
        id="time_zero_and_immortal_time",
        layer="time_alignment",
        question="When does follow-up start relative to when exposure is decided?",
        requirement=(
            "Define one time zero per unit and start follow-up there. If "
            "exposure status is only determined over a period after time zero, "
            "the interval during which it is being determined must not be "
            "counted as exposed follow-up: units are immortal in that window by "
            "construction. Either begin follow-up at the end of the window, or "
            "treat exposure as time-varying."
        ),
        source_key="suissa_immortal_time_2008",
        source_title="Immortal time bias in pharmacoepidemiology.",
        source_year="2008",
        source_venue="American Journal of Epidemiology",
        source_pmid="18056625",
        source_doi="10.1093/aje/kwm324",
        source_url="https://pubmed.ncbi.nlm.nih.gov/18056625/",
        design_elements=("time_zero", "exposure", "estimand"),
        also_see=("levesque_immortal_time_2010",),
    ),
    MethodCard(
        id="landmark_analysis",
        layer="time_alignment",
        question="How is a post-baseline exposure classification analysed fairly?",
        requirement=(
            "Choose the landmark time in advance, restrict to units still at "
            "risk at that time, classify exposure using only information "
            "available up to it, and report how many units were excluded for "
            "not surviving to it. A landmark analysis changes the population "
            "being described, so say which estimate is primary and why."
        ),
        source_key="anderson_landmark_1983",
        source_title=(
            "Analysis of survival by tumor response and other comparisons of "
            "time-to-event by outcome variables."
        ),
        source_year="1983",
        source_venue="Journal of Clinical Oncology",
        source_pmid="6668489",
        source_doi="10.1200/JCO.1983.1.11.710",
        source_url="https://pubmed.ncbi.nlm.nih.gov/6668489/",
        design_elements=("time_zero", "exposure", "estimand"),
    ),
    MethodCard(
        id="repeated_units_per_patient",
        layer="dependence",
        question="Are the analysis units independent?",
        requirement=(
            "When one patient can contribute several units, either restrict to "
            "one unit per patient or account for the within-patient "
            "correlation (cluster-robust standard errors or a mixed model). "
            "State which was done. If no patient identifier is available, say "
            "so explicitly and treat it as a limitation rather than assuming "
            "independence."
        ),
        source_key="strobe_2007",
        source_title=(
            "The Strengthening the Reporting of Observational Studies in "
            "Epidemiology (STROBE) statement: guidelines for reporting "
            "observational studies."
        ),
        source_year="2007",
        source_venue="Annals of Internal Medicine / BMJ / Lancet (co-published)",
        design_elements=("dependence",),
    ),
    MethodCard(
        id="continuous_covariate_functional_form",
        layer="functional_form",
        question="Should a continuous covariate enter the model linearly?",
        requirement=(
            "Do not assume linearity in the log-odds or log-hazard by default. "
            "With a large sample, model continuous covariates flexibly (for "
            "example restricted cubic splines with a pre-specified number of "
            "knots) and keep the linear form as a stated sensitivity analysis. "
            "Do not categorise a continuous covariate at data-driven cutpoints."
        ),
        source_key="durrleman_splines_1989",
        source_title="Flexible regression models with cubic splines.",
        source_year="1989",
        source_venue="Statistics in Medicine",
        source_pmid="2657958",
        source_doi="10.1002/sim.4780080504",
        source_url="https://pubmed.ncbi.nlm.nih.gov/2657958/",
        design_elements=("adjustment", "robustness"),
        also_see=("harrell_rms",),
    ),
    MethodCard(
        id="missing_data_handling",
        layer="missing_data",
        question="What is being assumed by dropping incomplete units?",
        requirement=(
            "Report the amount and pattern of missingness per variable, and "
            "state the assumption the chosen handling makes. Complete-case "
            "analysis is defensible when missingness is negligible or plausibly "
            "unrelated to the outcome given the covariates; say which applies. "
            "In routinely collected data, whether a measurement exists is "
            "itself informative and should be examined, not only imputed."
        ),
        source_key="sterne_missing_data_2009",
        source_title=(
            "Multiple imputation for missing data in epidemiological and "
            "clinical research: potential and pitfalls."
        ),
        source_year="2009",
        source_venue="BMJ",
        source_pmid="19564179",
        source_doi="10.1136/bmj.b2393",
        source_url="https://pubmed.ncbi.nlm.nih.gov/19564179/",
        design_elements=("missing_data", "robustness"),
        also_see=("little_rubin_missing_data",),
    ),
    MethodCard(
        id="absolute_and_relative_effects",
        layer="interpretation",
        question="Is a ratio measure enough to convey the finding?",
        requirement=(
            "Report an absolute measure alongside any ratio measure (risk "
            "difference, or adjusted absolute risk by exposure group). A large "
            "odds ratio on a rare outcome and a modest one on a common outcome "
            "carry very different clinical weight, and a ratio alone does not "
            "distinguish them."
        ),
        source_key="strobe_2007",
        source_title=(
            "The Strengthening the Reporting of Observational Studies in "
            "Epidemiology (STROBE) statement: guidelines for reporting "
            "observational studies."
        ),
        source_year="2007",
        source_venue="Annals of Internal Medicine / BMJ / Lancet (co-published)",
        design_elements=("outcome", "estimand"),
    ),
)


def method_binding_support(
    citation_key: str,
    design_elements: Sequence[str],
) -> dict[str, Any]:
    """Project one typed source binding onto exact method-card authority.

    The public contract is deliberately small and dependency-neutral.  It does
    not decide whether a topic/direct-comparator paper is relevant; it only
    answers whether a curated *method source* has a card whose declared scope
    overlaps each Planner-owned design element.  Unknown/non-method sources
    remain outside this check and continue through source-excerpt review.
    """

    key = str(citation_key or "").strip()
    declared = tuple(
        dict.fromkeys(
            str(value or "").strip()
            for value in design_elements or ()
            if str(value or "").strip()
        )
    )
    cards = tuple(card for card in METHOD_CARDS if card.source_key == key)
    if not cards:
        return {
            "method_source": False,
            "matched_card_ids": [],
            "matched_layers": [],
            "unsupported_design_elements": [],
        }
    matched = tuple(
        card
        for card in cards
        if set(card.design_elements).intersection(declared)
    )
    supported_elements = {
        element for card in cards for element in card.design_elements
    }
    return {
        "method_source": True,
        "matched_card_ids": sorted(card.id for card in matched),
        "matched_layers": sorted({card.layer for card in matched}),
        "unsupported_design_elements": sorted(
            set(declared) - supported_elements
        ),
    }


def method_cards_for_layers(
    layers: Sequence[str] | None = None,
) -> tuple[MethodCard, ...]:
    """Return the cards in the requested layers, or every card."""

    if not layers:
        return METHOD_CARDS
    wanted = {str(layer).strip().lower() for layer in layers}
    return tuple(card for card in METHOD_CARDS if card.layer in wanted)


def method_literature_citations() -> tuple[dict[str, Any], ...]:
    """Return one deduplicated citation payload per distinct source.

    Shaped for :class:`~easyicu.research_agent.literature.CitationRecord` but
    returned as plain dicts so this module stays importable on its own -- the
    same reason the concept schema is separable from its resolver.
    """

    seen: dict[str, dict[str, Any]] = {}
    for card in METHOD_CARDS:
        if card.source_key in seen:
            # Keep one relevance line per source, naming every card it backs.
            seen[card.source_key]["relevance"] += f"; {card.question}"
            continue
        seen[card.source_key] = {
            "key": card.source_key,
            "title": card.source_title,
            "year": card.source_year,
            "venue": card.source_venue or None,
            "relevance": f"Methodology: {card.question}",
            "pmid": card.source_pmid or None,
            "doi": card.source_doi or None,
            "url": card.source_url or None,
        }
    return tuple(seen.values())


def method_literature_pack(layers: Sequence[str] | None = None) -> dict[str, Any]:
    """Return the frozen, serialisable methodology pack."""

    cards = method_cards_for_layers(layers)
    return {
        "schema_version": METHOD_LITERATURE_SCHEMA_VERSION,
        "layer": "general_methodology",
        "case_neutral": True,
        "cards": [card.as_dict() for card in cards],
    }


def method_literature_digest(layers: Sequence[str] | None = None) -> str:
    """Return a stable sha256 over the pack, for run-to-run comparison."""

    raw = json.dumps(
        method_literature_pack(layers),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
