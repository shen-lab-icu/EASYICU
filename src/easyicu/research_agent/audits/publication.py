"""Replication and publication-claim auditors."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


from ..replication.metrics import compare_metric_values
from ..schema import (
    PaperProfile,
    PaperResultLedger,
    ReplicationDeviationReport,
    ValidationFinding,
)

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

class ReplicationDesignAuditor:
    """Validate whether a parsed paper is reproducible in EasyICU."""

    name = "replication_design_auditor"

    def audit(
        self,
        *,
        paper_profile: PaperProfile,
        deviation_report: ReplicationDeviationReport,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        if paper_profile.paper_type == "unsupported_or_underspecified":
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        "Paper is unsupported or underspecified for strict replication: "
                        + "; ".join(
                            paper_profile.unsupported_reasons or ["no reason recorded"]
                        )
                    ),
                )
            )
        for item in deviation_report.items:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity=item.severity,
                    message=f"{item.item}: {item.reason}",
                    detail={
                        "original": item.original,
                        "easyicu_proxy": item.easyicu_proxy,
                    },
                )
            )
        return findings


class ReplicationResultComparator:
    """Compare original-paper claims to EasyICU structured metrics."""

    name = "replication_result_comparator"

    _metric_map = {
        "or": "primary_or",
        "hr": "primary_or",
        "rr": "primary_or",
        "auroc": "auroc",
        "auc": "auroc",
        "brier_score": "brier_score",
        "p_value": "primary_pvalue",
        "p": "primary_pvalue",
        "n": "n_stays",
    }

    def compare(
        self,
        *,
        paper_profile: PaperProfile,
        ledger: PaperResultLedger,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for claim in paper_profile.key_claims:
            easyicu_value = ledger.easyicu_metrics.get(
                self._metric_map.get((claim.metric or "").lower(), "")
            )
            alignment, reason = compare_metric_values(
                metric=claim.metric,
                paper_value=claim.numeric_value,
                paper_direction=claim.direction,
                easyicu_value=easyicu_value,
            )
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "paper_claim": claim.sentence,
                    "paper_value": claim.paper_value or "",
                    "easyicu_value": (
                        "" if easyicu_value is None else str(easyicu_value)
                    ),
                    "alignment_status": alignment,
                    "reason_if_mismatch": reason,
                    "metric": claim.metric or "",
                }
            )
        return rows

    def findings_from_rows(
        self, rows: Sequence[Dict[str, Any]]
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        if not rows:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="No result-alignment rows were produced for the parsed paper claims.",
                )
            ]
        for row in rows:
            if row.get("alignment_status") != "not_aligned":
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"Claim {row.get('claim_id')} was not aligned with EasyICU results: "
                        f"{row.get('reason_if_mismatch')}"
                    ),
                    detail=dict(row),
                )
            )
        return findings


class PublicationClaimAuditor:
    """Block showcase manuscripts that misrepresent the replication relationship."""

    name = "publication_claim_auditor"

    def audit(
        self,
        *,
        manuscript_text: str,
        deviation_report: ReplicationDeviationReport,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        text = manuscript_text or ""
        lower = text.lower()
        prohibited = (
            "exactly reproduced",
            "identical to the original paper",
            "fully reproduced the original study",
            "same dataset as the original paper",
        )
        for phrase in prohibited:
            if phrase in lower:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=f"Showcase manuscript over-claims replication fidelity via phrase: {phrase!r}.",
                    )
                )
        if "replication" not in lower:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="Showcase manuscript does not state that it is a replication study.",
                )
            )
        if "easyicu" not in lower:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="Showcase manuscript does not identify EasyICU as the cohort source.",
                )
            )
        if deviation_report.items and not re.search(
            r"\bdeviation|differ|limitation|harmoni[sz]ation\b", lower
        ):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="Showcase manuscript does not explain replication deviations/limitations.",
                )
            )
        if (
            re.search(r"\boriginal paper\b", lower)
            and "original paper reported" not in lower
        ):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message="References to the original paper should be explicitly framed as reported original results.",
                )
            )
        return findings
