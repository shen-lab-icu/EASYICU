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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..authority.planned_role import unique_verified_primary_record
from ..authority.runtime_artifacts import (
    current_evidence_records,
    current_step_records,
    verified_run_evidence_path,
)
from ..contracts.product_identity import typed_product


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


_SEVERITIES = ("info", "minor", "major", "reject")
_SEVERITY_RANK = {s: i for i, s in enumerate(_SEVERITIES)}


@dataclass(frozen=True)
class ReviewerPrimaryResultBinding:
    """Host-verified result identity, never an LLM or metadata self-claim.

    The caller projects these from its scientific result owner after runtime
    validation. A declared output alone is insufficient. ``analysis_role`` is
    the product's role (a primary step can also emit sensitivity products), and
    ``claim_ceiling`` is the runtime ceiling, not the catalogue's potential.
    This binding identifies an available result; it grants no paper authority.
    """

    product: str
    evidence_id: str
    sha256: str
    produced_by_step: str
    analysis_role: str
    claim_ceiling: str


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


def _record_field(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name, None)


def _restricted_result(record: Any) -> bool:
    return (
        _record_field(record, "diagnostic_only") not in (None, False)
        or _record_field(record, "analysis_only") is True
        or any(
            _record_field(record, key) is not None
            and _record_field(record, key) != "reportable"
            for key in ("claim_ceiling", "scientific_validation", "reportability")
        )
        or any(
            _record_field(record, key) is False
            for key in ("analysis_validated", "reportability_allowed")
        )
    )


def _verified_primary_result_ids(
    *,
    evidence_records: Sequence[Any],
    per_step_records: Optional[Sequence[Mapping[str, Any]]],
    primary_result_bindings: Iterable[ReviewerPrimaryResultBinding],
    run_dir: Optional[Path],
) -> set[str]:
    """Intersect host result identities with current, digest-bound products."""
    if run_dir is None or per_step_records is None:
        return set()
    primary = unique_verified_primary_record(current_step_records(per_step_records))
    if primary is None or primary.get("status") != "ok" or _restricted_result(primary):
        return set()
    step_id = primary["step_id"]
    declared = primary["analysis_request"]["step"].get("expected_outputs") or []
    records = {
        _record_field(record, "evidence_id"): record
        for record in current_evidence_records(evidence_records, per_step_records)
    }
    summary_record = records.get(primary.get("step_summary_evidence_id"))
    if summary_record is None or _record_field(summary_record, "produced_by_step") != step_id:
        return set()
    summary_path = verified_run_evidence_path(run_dir, summary_record)
    if summary_path is None:
        return set()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if not isinstance(summary, Mapping) or summary.get("status") != "ok" or _restricted_result(summary):
        return set()
    outputs = summary.get("output_files")
    if not isinstance(outputs, Mapping):
        return set()
    accepted: set[str] = set()
    for binding in primary_result_bindings:
        if not isinstance(binding, ReviewerPrimaryResultBinding):
            continue
        product = typed_product(binding.product)
        if (
            binding.analysis_role != "primary"
            or binding.claim_ceiling != "reportable"
            or binding.produced_by_step != step_id
            or binding.product not in declared
            or product is None
            or product[0] not in {"table", "statistic"}
        ):
            continue
        record = records.get(binding.evidence_id)
        if record is None:
            continue
        metadata = _record_field(record, "metadata") or {}
        if not isinstance(metadata, Mapping) or _restricted_result(record) or _restricted_result(metadata):
            continue
        if (
            _record_field(record, "kind") != product[0]
            or _record_field(record, "produced_by_step") != step_id
            or _record_field(record, "sha256") != binding.sha256
        ):
            continue
        path = verified_run_evidence_path(run_dir, record)
        filename = outputs.get(binding.product)
        if (
            path is not None
            and isinstance(filename, str)
            and Path(filename).name == filename
            and path.name == f"{binding.evidence_id}__{filename}"
        ):
            accepted.add(binding.evidence_id)
    return accepted


def derive_reviewer_primary_result_bindings(
    *,
    evidence_store: Any,
    per_step_records: Sequence[Mapping[str, Any]],
    current_case_scientific_runtime_authority: Any = None,
    scientific_runtime_projection_sha256: Optional[str] = None,
) -> List[ReviewerPrimaryResultBinding]:
    """Project executed primary products using their existing scientific owner.

    Consume the same current sidecar/evidence reader as Writer. Envelopes bind
    artifacts but do not assign product-level scientific roles or grant paper
    authority. The landmark owner's sealed contract and matching runtime
    receipt distinguish its curve/contrasts from same-step sensitivity outputs.
    Unknown owners and missing authority remain unverified, without a catalogue
    or generated-plan fallback. This function performs no execution or writes.
    """
    from ..audits.envelope_consumers import (
        RegisteredOutputAuthorityError, RegisteredOutputEnvelopeConsumer,
    )
    from ..authority.current_case_scientific_runtime import (
        LandmarkSplineRuntimeAuthority, load_current_case_scientific_runtime_authority,
    )
    from ..contracts.capability_ids import (
        LANDMARK_SPLINE_ANALYSIS_KIND, LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID,
    )
    from ..contracts.landmark_spline_validation import LandmarkSplineRuntimeReceipt

    primary = unique_verified_primary_record(current_step_records(per_step_records))
    if (
        primary is None or primary.get("status") != "ok" or _restricted_result(primary)
        or primary.get("generation_mode") != "deterministic_standard"
        or primary.get("deterministic_standard_analysis") != LANDMARK_SPLINE_ANALYSIS_KIND
        or current_case_scientific_runtime_authority is None
        or not scientific_runtime_projection_sha256
    ):
        return []
    try:
        raw_authority = current_case_scientific_runtime_authority
        if hasattr(raw_authority, "model_dump"):
            raw_authority = raw_authority.model_dump(mode="json")
        authority = load_current_case_scientific_runtime_authority(raw_authority)
        if not isinstance(authority, LandmarkSplineRuntimeAuthority):
            return []
        step = primary["analysis_request"]["step"]
        if (
            step.get("scientific_capability") != LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID
            or step.get("method") != authority.plan_method
            or tuple(step.get("expected_outputs") or []) != authority.plan_outputs
            or authority.plan_rule_ref not in (step.get("icu_rule_refs") or [])
        ):
            return []
        projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
            [primary], evidence_store=evidence_store,
        )[0]
        summary = projected["step_summary"]
        if _restricted_result(summary):
            return []
        receipt = LandmarkSplineRuntimeReceipt.model_validate(summary.get("scientific_runtime_receipt"))
        if (
            receipt.execution_contract_sha256 != authority.execution_contract_sha256
            or receipt.protocol_content_sha256 != authority.protocol_content_sha256
            or receipt.runtime_projection_sha256 != scientific_runtime_projection_sha256
        ):
            return []
        artifacts = projected["writer_artifact_bindings"]
        records = list(evidence_store.records())
        by_id = {_record_field(record, "evidence_id"): record for record in records}
        receipt_record = by_id[artifacts[authority.receipt_product]["evidence_id"]]
        receipt_path = verified_run_evidence_path(evidence_store.root, receipt_record)
        if receipt_path is None or LandmarkSplineRuntimeReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        ) != receipt:
            return []
        bindings = [
            ReviewerPrimaryResultBinding(
                product=product, evidence_id=artifacts[product]["evidence_id"],
                sha256=artifacts[product]["sha256"], produced_by_step=primary["step_id"],
                analysis_role="primary", claim_ceiling="reportable",
            )
            for product in (authority.curve_product, authority.downstream_parent_product)
        ]
    except (RegisteredOutputAuthorityError, ValueError, TypeError, KeyError, OSError):
        return []
    verified_ids = _verified_primary_result_ids(
        evidence_records=records, per_step_records=per_step_records,
        primary_result_bindings=bindings, run_dir=Path(evidence_store.root),
    )
    return [binding for binding in bindings if binding.evidence_id in verified_ids]


def _has_literature_bundle(evidence_records: Iterable[Any]) -> bool:
    # These are exact host registration identities, not words in descriptions.
    return any(
        _record_field(record, "evidence_id") in {"literature_bundle", "preplan_literature_bundle"}
        and _record_field(record, "kind") == "log"
        and not _record_field(record, "produced_by_step")
        for record in evidence_records
    )


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
    *, evidence_records: Iterable[Any], findings: Iterable[Any],
    primary_result_evidence_ids: set[str],
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
    if not primary_result_evidence_ids:
        comments.append(
            ReviewerComment(
                reviewer="statistician",
                severity="major",
                topic="effect_estimate",
                message=(
                    "A reportable primary result could not be verified from "
                    "the supplied host result bindings and current evidence. "
                    "Check its product identity, planned role and runtime "
                    "claim ceiling; this does not establish that no result exists."
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
    if not _has_literature_bundle(evidence_records):
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
        development_identity_only = (
            validator == "development_runtime_lineage" and severity == "warning"
            and isinstance(detail, dict) and detail.get("paper_authority") is False
            and detail.get("diagnostic_only") is True
            and not any(detail.get(key) is False for key in (
                "analysis_validated", "reportability_allowed", "paper_authorization_allowed",
            ))
        )
        if development_identity_only:
            comments.append(ReviewerComment(
                reviewer="methodologist", severity="info", topic="publication_scope",
                message="Development runtime lineage is diagnostic-only and grants no paper authority.",
                evidence_ids=evidence_ids,
            ))
            continue
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
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
    primary_result_bindings: Iterable[ReviewerPrimaryResultBinding] = (),
    run_dir: Optional[Path] = None,
) -> ReviewerReport:
    """Run a simulated checklist, not independent clinical peer review.

    Callers supply current verified records and host-validated result bindings
    to both pre- and post-writing rounds. Missing authority fails closed. With
    ``run_dir`` supplied, registered bytes are checked again at this boundary.
    """
    # Materialise the iterables so they can be re-used.
    recs = list(evidence_records)
    if run_dir is not None:
        recs = [
            record for record in current_evidence_records(recs, per_step_records)
            if verified_run_evidence_path(run_dir, record) is not None
        ]
    finds = list(findings)
    primary_ids = _verified_primary_result_ids(
        evidence_records=recs, per_step_records=per_step_records,
        primary_result_bindings=primary_result_bindings, run_dir=run_dir,
    )
    critiques = [
        ReviewerCritique(
            reviewer="statistician",
            comments=_build_statistician_comments(
                evidence_records=recs, findings=finds,
                primary_result_evidence_ids=primary_ids,
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
    "derive_reviewer_primary_result_bindings",
    "ReviewerPrimaryResultBinding",
    "ReviewerComment",
    "ReviewerCritique",
    "ReviewerReport",
    "run_reviewer_round",
]
