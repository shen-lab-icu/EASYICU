"""The result fact a host report-fact owner hands the Writer.

Every owner that admits a result sentence from verified output returns this
one type (``descriptive_report_facts``, ``benchmark_report_facts``), so an
owner that composes another's facts never imports that owner back.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptiveReportFact:
    subsection: str
    text: str
    evidence_id: str
    source_sha256: str
    source_fields: tuple[str, ...]
    replaces_claim_ref: str | None = None
    cohort_n: int | None = None
    outcome_label: str | None = None
    group_label: str | None = None
    estimate_pct: float | None = None
    required_result_sections: tuple[str, ...] = ("Abstract", "Results")
    supporting_evidence_ids: tuple[str, ...] = ()
    step_id: str | None = None
    product_id: str | None = None
    outcome: str | None = None
    analysis_unit: str = "records"
    cohort_source_field: str = "cohort_n"
    abstract_text: str | None = None

    @property
    def scaffold(self) -> str:
        citations = " ".join(
            f"{{evidence:{key}}}"
            for key in dict.fromkeys((self.evidence_id, *self.supporting_evidence_ids))
        )
        return f"{self.text} {citations}."
