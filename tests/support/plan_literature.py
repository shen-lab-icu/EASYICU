"""Small synthetic literature fixtures; no network or external datasets."""
from datetime import datetime, timezone
from easyicu.research_agent import literature as owner
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext


def context(kind="lab"):
    exposure, concept, question = (
        ("marker", "creat", "Does creatinine predict mortality in adult ICU patients?")
        if kind == "lab" else
        ("drug", "norepi_rate", "Is norepinephrine associated with mortality in adult ICU patients?")
    )
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="adult ICU", database="miiv", n_stays=12),
        variables=[
            ConceptDescriptor(name=exposure, source_concept=concept, dtype="float64",
                              description="unrelated storage duration"),
            ConceptDescriptor(name="endpoint", source_concept="death", dtype="int64",
                              description="mortality"),
        ],
        primary_exposure=exposure, target_outcome="endpoint",
    )


def record(index=0, kind="lab", **changes):
    term = "creatinine" if kind == "lab" else "norepinephrine"
    values = dict(
        key=f"synthetic_{kind}_{index}", pmid=str(10000 + index),
        title=f"{term} and mortality in adult ICU patients: cohort {index}",
        year=str(datetime.now(timezone.utc).year - 1),
        publication_types=["Journal Article"],
        relevance=(f"Study-design excerpt: We studied adult patients admitted to intensive care. "
                   f"We evaluated {term} as a predictor of mortality using multivariable regression. "
                   "We used the MIMIC-IV database and adjusted for age and comorbidity. "
                   "Mortality was defined as death during hospitalization. "
                   "Sensitivity analyses assessed missing data."),
    )
    values.update(changes)
    return owner.CitationRecord(**values)


class FakePubMed:
    def __init__(self, records):
        self.records = records

    def search_for_context(self, context, *, retmax):
        return self.records[:retmax]
