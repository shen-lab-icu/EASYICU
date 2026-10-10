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

    @property
    def scaffold(self) -> str:
        return f"{self.text} {{evidence:{self.evidence_id}}}."
