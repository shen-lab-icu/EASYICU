"""Three-role simulated reviewer loop (O15).

Design
------

After the bound manuscript is produced and deterministic audits
(``CriticAgent``, causal audit, reporting checklist, multiple-testing)
have run, ``ReviewerAgent`` simulates three journal reviewers:

* **statistician** — statistical rigor, multiple testing, calibration,
  effect-size transparency, missing-data handling.
* **clinician** — ICU domain plausibility, pitfall awareness,
  endpoint choice, confounder coverage.
* **methodologist** — study design, reporting-guideline coverage,
  reproducibility, pre-registration, data / code availability.

Each reviewer emits a :class:`ReviewerCritique` with severity-graded
comments. The aggregated report is persisted as
``reviewer_report.md`` / ``reviewer_report.json`` and registered in
the EvidenceStore. The pipeline can optionally drive a revision loop
that asks the ``WriterAgent`` to regenerate the Results section once
per round, subject to ``max_revision_rounds``.

Constraints
-----------

* **Deterministic-first.** Every reviewer is seeded with a
  deterministic checklist derived from existing pipeline artefacts
  (causal labels, multiple-testing summary, reporting coverage,
  evidence aliases). The LLM is asked only to phrase those
  observations as a reviewer comment; it cannot introduce comments
  about things the pipeline did not observe. Tests do not need a
  real LLM.
* **No additional validators.** ReviewerAgent reads the same
  registered evidence and findings that the critic already
  processes; it does not re-run auditors.
* **Revision loop is bounded and opt-in.** Default
  ``max_revision_rounds=0``. When > 0, the pipeline will ask the
  writer to regenerate the manuscript once per round and re-run
  only the three reviewers (not the deterministic gates).

Nothing in this module rewrites existing artefacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


_SEVERITIES = ("info", "minor", "major", "reject")
_SEVERITY_RANK = {s: i for i, s in enumerate(_SEVERITIES)}


@dataclass
class ReviewerComment:
    """One severity-graded reviewer comment."""

    reviewer: str
    severity: str  # "info" | "minor" | "major" | "reject"
    topic: str
    message: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "severity": self.severity,
            "topic": self.topic,
            "message": self.message,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class ReviewerCritique:
    """Aggregated comments from one reviewer role."""

    reviewer: str
    comments: List[ReviewerComment] = field(default_factory=list)

    def recommendation(self) -> str:
        """Return a single recommendation tag for the role."""
        if any(c.severity == "reject" for c in self.comments):
            return "reject"
        if any(c.severity == "major" for c in self.comments):
            return "major_revision"
        if any(c.severity == "minor" for c in self.comments):
            return "minor_revision"
        return "accept"


@dataclass
class ReviewerReport:
    """The full three-role reviewer bundle for one manuscript draft."""

    round_index: int = 0
    critiques: List[ReviewerCritique] = field(default_factory=list)

    def aggregated_recommendation(self) -> str:
        worst = "accept"
        for critique in self.critiques:
            rec = critique.recommendation()
            for candidate in ("reject", "major_revision", "minor_revision", "accept"):
                if rec == candidate:
                    # Escalate worst to the most severe seen so far.
                    order = ("accept", "minor_revision", "major_revision", "reject")
                    if order.index(candidate) > order.index(worst):
                        worst = candidate
                    break
        return worst

    def summary(self) -> Dict[str, Any]:
        counts = {"info": 0, "minor": 0, "major": 0, "reject": 0}
        for critique in self.critiques:
            for c in critique.comments:
                counts[c.severity] = counts.get(c.severity, 0) + 1
        return {
            "round": self.round_index,
            "reviewers": [c.reviewer for c in self.critiques],
            "counts": counts,
            "aggregated_recommendation": self.aggregated_recommendation(),
            "per_role": {c.reviewer: c.recommendation() for c in self.critiques},
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "round": self.round_index,
            "summary": self.summary(),
            "critiques": [
                {
                    "reviewer": c.reviewer,
                    "recommendation": c.recommendation(),
                    "comments": [cm.to_json() for cm in c.comments],
                }
                for c in self.critiques
            ],
        }

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# Simulated reviewer report (round {self.round_index + 1})",
            "",
            f"**Aggregated recommendation:** `{s['aggregated_recommendation']}`",
            "",
            f"Comments by severity: "
            f"info={s['counts'].get('info',0)}, "
            f"minor={s['counts'].get('minor',0)}, "
            f"major={s['counts'].get('major',0)}, "
            f"reject={s['counts'].get('reject',0)}",
        ]
        for critique in self.critiques:
            lines += [
                "",
                f"## {critique.reviewer.capitalize()} — recommends `{critique.recommendation()}`",
            ]
            if not critique.comments:
                lines.append("No substantive comments.")
                continue
            lines.append("")
            lines.append("| Severity | Topic | Comment | Evidence |")
            lines.append("|---|---|---|---|")
            for c in critique.comments:
                ev = ", ".join(f"`{e}`" for e in c.evidence_ids) or "—"
                lines.append(
                    "| {sev} | {topic} | {msg} | {ev} |".format(
                        sev=c.severity,
                        topic=c.topic.replace("|", "/")[:40],
                        msg=c.message.replace("|", "/").replace("\n", " ")[:240],
                        ev=ev,
                    )
                )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Deterministic checklist per role
# ---------------------------------------------------------------------------


def _available_evidence_ids(evidence_records: Iterable[Any]) -> set[str]:
    """Return searchable tokens from verified records, not only opaque ids."""

    aliases: set[str] = set()
    for record in evidence_records:
        for attr in (
            "evidence_id",
            "produced_by_step",
            "description",
            "relative_path",
            "kind",
        ):
            value = (
                record.get(attr)
                if isinstance(record, dict)
                else getattr(record, attr, None)
            )
            if value:
                aliases.add(str(value))
        metadata = (
            record.get("metadata")
            if isinstance(record, dict)
            else getattr(record, "metadata", None)
        )
        if isinstance(metadata, dict):
            aliases.update(
                str(value)
                for value in metadata.values()
                if isinstance(value, (str, int, float, bool))
            )
    return aliases


def _has_evidence_token(aliases: Iterable[str], *needles: str) -> bool:
    haystack = "\n".join(str(alias).lower() for alias in aliases)
    return any(needle.lower() in haystack for needle in needles)


def _finding_msg(findings: Iterable[Any], validator: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in findings:
        if isinstance(f, dict):
            f_validator = f.get("validator")
            f_severity = f.get("severity")
            f_message = f.get("message") or ""
            f_detail = f.get("detail")
        else:
            f_validator = getattr(f, "validator", None)
            f_severity = getattr(f, "severity", None)
            f_message = getattr(f, "message", None) or ""
            f_detail = getattr(f, "detail", None)
        if f_validator != validator:
            continue
        out.append(
            {"severity": f_severity, "message": f_message, "detail": f_detail}
        )
    return out


def _build_statistician_comments(
    *, evidence_records: Iterable[Any], findings: Iterable[Any]
) -> List[ReviewerComment]:
    comments: List[ReviewerComment] = []
    aliases = _available_evidence_ids(evidence_records)
    mt = _finding_msg(findings, "multiple_testing")
    if not any(
        f["severity"] == "info" and "BH-FDR" in (f["message"] or "")
        for f in mt
    ):
        comments.append(
            ReviewerComment(
                reviewer="statistician",
                severity="minor",
                topic="multiple_testing",
                message=(
                    "Multiple-testing correction does not appear in the "
                    "pipeline findings. Please define the relevant "
                    "hypothesis families and report family-scoped "
                    "BH-adjusted or family-wise corrected p-values."
                ),
                evidence_ids=["multiple_testing_report"] if "multiple_testing_report" in aliases else [],
            )
        )
    if any(f["severity"] == "warning" for f in mt):
        comments.append(
            ReviewerComment(
                reviewer="statistician",
                severity="major",
                topic="multiple_testing",
                message=(
                    "At least one raw-significant result did not survive "
                    "BH-FDR within its declared hypothesis family. The "
                    "primary / secondary endpoint distinction, family "
                    "definition, and corrected p-values must be stated "
                    "explicitly."
                ),
                evidence_ids=["multiple_testing_report"],
            )
        )
    has_primary_effect = (
        _has_evidence_token(
            aliases,
            "primary association",
            "primary_association",
            "primary_effect",
            "association_model",
            "association_estimate",
            "adjusted_odds_ratio",
            "odds_ratio",
            "hazard_ratio",
            "cox_summary",
            "survival_summary",
            "prediction_performance",
            "model_performance",
            "auroc",
            "average_precision",
        )
    )
    if not has_primary_effect:
        comments.append(
            ReviewerComment(
                reviewer="statistician",
                severity="major",
                topic="effect_estimate",
                message=(
                    "No primary effect estimate (OR / HR / AUROC) is "
                    "registered in the evidence store. The Results "
                    "section cannot stand on descriptive statistics alone."
                ),
            )
        )
    has_missingness_profile = "missingness" in aliases or _has_evidence_token(
        aliases,
        "missingness",
        "missingness_audit",
        "missingness profile",
        "missing strategy",
    )
    if not has_missingness_profile:
        comments.append(
            ReviewerComment(
                reviewer="statistician",
                severity="minor",
                topic="missingness",
                message=(
                    "A missingness profile is not registered. Even a one-row "
                    "missingness summary strengthens the Methods section and "
                    "is the minimum STROBE item 12c requires."
                ),
            )
        )
    return comments


def _build_clinician_comments(
    *, evidence_records: Iterable[Any], findings: Iterable[Any]
) -> List[ReviewerComment]:
    comments: List[ReviewerComment] = []
    aliases = _available_evidence_ids(evidence_records)

    causal = _finding_msg(findings, "causal_audit")
    if any(f["severity"] == "error" for f in causal):
        comments.append(
            ReviewerComment(
                reviewer="clinician",
                severity="reject",
                topic="causal_overclaim",
                message=(
                    "The manuscript uses causal language against an effect "
                    "that was not estimated with an identification strategy "
                    "(IPTW / TMLE / g-computation). Soften the language or "
                    "supply the required support artefacts."
                ),
                evidence_ids=["causal_audit_report"],
            )
        )
    if any(f["severity"] == "warning" for f in causal):
        comments.append(
            ReviewerComment(
                reviewer="clinician",
                severity="major",
                topic="causal_language",
                message=(
                    "Causal phrasing is applied to an associational "
                    "estimate. Revise to 'was associated with' / "
                    "'correlated with' for observational ICU cohort work."
                ),
                evidence_ids=["causal_audit_report"],
            )
        )

    cohort = _finding_msg(findings, "cohort_auditor")
    if any(f["severity"] == "error" for f in cohort):
        comments.append(
            ReviewerComment(
                reviewer="clinician",
                severity="major",
                topic="cohort_integrity",
                message=(
                    "A cohort-audit error was raised. ICU cohort integrity "
                    "issues (duplicate stays, missing outcome coding, "
                    "impossible ages) must be resolved before review can "
                    "proceed."
                ),
                evidence_ids=["cohort_audit"],
            )
        )

    return comments


def _build_methodologist_comments(
    *, evidence_records: Iterable[Any], findings: Iterable[Any]
) -> List[ReviewerComment]:
    comments: List[ReviewerComment] = []
    aliases = _available_evidence_ids(evidence_records)

    # Checklist coverage — look for the pipeline's info finding.
    checklist = _finding_msg(findings, "reporting_checklist")
    coverage = None
    for f in checklist:
        detail = f.get("detail") or {}
        if isinstance(detail, dict) and "coverage" in detail:
            coverage = detail.get("coverage")
            break
    if coverage is None:
        comments.append(
            ReviewerComment(
                reviewer="methodologist",
                severity="minor",
                topic="reporting_guideline",
                message=(
                    "No STROBE / TRIPOD+AI checklist is attached. Journal "
                    "submission guidelines for observational and prediction "
                    "studies require one; auto-generate it and include it "
                    "as supplementary."
                ),
            )
        )
    elif coverage < 0.5:
        comments.append(
            ReviewerComment(
                reviewer="methodologist",
                severity="major",
                topic="reporting_guideline",
                message=(
                    f"Reporting-checklist coverage is {coverage:.0%}, below "
                    "the 50% informal floor we consider submittable. The "
                    "Methods sections need to explicitly address the open "
                    "items before the Results story holds together."
                ),
                evidence_ids=["reporting_checklist_strobe"]
                if "reporting_checklist_strobe" in aliases
                else [],
            )
        )
    if "reproducibility_envelope" not in aliases:
        comments.append(
            ReviewerComment(
                reviewer="methodologist",
                severity="minor",
                topic="reproducibility",
                message=(
                    "No reproducibility envelope (LLM prompts/responses, "
                    "seeds, environment snapshot) is attached. For an "
                    "LLM-in-the-loop analysis this is a common reviewer "
                    "demand; please run the pipeline with "
                    "enable_reproducibility_envelope=True."
                ),
            )
        )
    if "literature_bundle" not in aliases:
        comments.append(
            ReviewerComment(
                reviewer="methodologist",
                severity="info",
                topic="literature",
                message=(
                    "The literature bundle is not attached. Even a small "
                    "curated-plus-PubMed bundle is enough to ground the "
                    "Introduction and Discussion."
                ),
            )
        )

    blocking: List[Dict[str, Any]] = []
    for finding in findings:
        if isinstance(finding, dict):
            severity = finding.get("severity")
            validator = str(finding.get("validator") or "unknown")
            message = str(finding.get("message") or "")
            detail = finding.get("detail")
            evidence_ids = list(finding.get("evidence_ids") or [])
        else:
            severity = getattr(finding, "severity", None)
            validator = str(getattr(finding, "validator", None) or "unknown")
            message = str(getattr(finding, "message", None) or "")
            detail = getattr(finding, "detail", None)
            evidence_ids = list(getattr(finding, "evidence_ids", None) or [])
        explicit_block = isinstance(detail, dict) and any(
            detail.get(field) is False
            for field in (
                "paper_authority",
                "paper_authorization_allowed",
                "reportability_allowed",
                "analysis_validated",
            )
        )
        if severity == "error" or explicit_block:
            blocking.append(
                {
                    "validator": validator,
                    "message": message,
                    "evidence_ids": evidence_ids,
                }
            )
    if blocking:
        validators = sorted({item["validator"] for item in blocking})
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for item in blocking
                for evidence_id in item["evidence_ids"]
            )
        )
        comments.append(
            ReviewerComment(
                reviewer="methodologist",
                severity="major",
                topic="scientific_gate",
                message=(
                    "The current run carries unresolved scientific or "
                    "reportability blockers from: "
                    + ", ".join(validators)
                    + ". These must be closed before an accept recommendation."
                ),
                evidence_ids=evidence_ids,
            )
        )
    return comments


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_reviewer_round(
    *,
    evidence_records: Iterable[Any],
    findings: Iterable[Any],
    round_index: int = 0,
) -> ReviewerReport:
    """Run one round of the three-role reviewer critique."""
    # Materialise the iterables so they can be re-used.
    recs = list(evidence_records)
    finds = list(findings)
    critiques = [
        ReviewerCritique(
            reviewer="statistician",
            comments=_build_statistician_comments(
                evidence_records=recs, findings=finds
            ),
        ),
        ReviewerCritique(
            reviewer="clinician",
            comments=_build_clinician_comments(
                evidence_records=recs, findings=finds
            ),
        ),
        ReviewerCritique(
            reviewer="methodologist",
            comments=_build_methodologist_comments(
                evidence_records=recs, findings=finds
            ),
        ),
    ]
    return ReviewerReport(round_index=round_index, critiques=critiques)


__all__ = [
    "ReviewerComment",
    "ReviewerCritique",
    "ReviewerReport",
    "run_reviewer_round",
]
