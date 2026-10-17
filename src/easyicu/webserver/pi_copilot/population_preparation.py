"""Project a paused plan's population into an exact extraction proposal.

No prose parsing, patient reads, grants or job submission happen here. The
review request seals the compiler's complete record and the offered plan.
Only admission axes the extraction contract can express exactly are projected;
other criteria remain visible and prevent applying a partial remedy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.frozen_payload import thaw_payload
from easyicu.research_agent.planning.population_review import sealed_population
from easyicu.research_agent.planning.population_spec import (
    AgeYears,
    FirstIcuStay,
    IcuStayHours,
    PopulationSpec,
    STUDY_WORDING_SOURCES,
)
from easyicu.webserver import primary_cohort, study_contexts

STOP = "population_inclusion_requires_extraction"


def _admission_patch(criterion: Any, cohort: dict, time_zero: Any) -> str | None:
    """Intersect one inclusion with existing admission axes, or name the gap."""
    if criterion.role != "include":
        return "population_preparation_kind_unsupported"
    current = primary_cohort.normalize_execution_cohort(cohort)
    if isinstance(criterion, FirstIcuStay):
        if "exclude_readmissions" in cohort and not current["exclude_readmissions"]:
            return "population_preparation_all_stays_conflict"
        cohort["exclude_readmissions"] = True
        return None
    if isinstance(criterion, AgeYears):
        bounds = (
            ("age_min", criterion.min_years, max),
            ("age_max", criterion.max_years, min),
        )
        for field, value, intersect in bounds:
            if value is not None:
                # Data Extraction treats age_min=0 and age_max>=100 as
                # unbounded, and normalization swaps bounds above 100.
                # None of those spellings can promise this exact bound.
                if not float(value).is_integer() or not (
                    0 < value <= 100 if field == "age_min" else 0 < value < 100
                ):
                    return "population_preparation_bound_unrepresentable"
                cohort[field] = intersect(current[field], int(value))
        if cohort.get("age_min", current["age_min"]) > cohort.get(
            "age_max", current["age_max"]
        ):
            return "population_preparation_empty_age_range"
        return None
    if isinstance(criterion, IcuStayHours):
        value = criterion.min_hours
        if (
            criterion.max_hours is not None
            or value is None
            or not float(value).is_integer()
            or value > 720
        ):
            return "population_preparation_bound_unrepresentable"
        if time_zero is not None and value > time_zero:
            return "population_determined_after_time_zero"
        cohort["min_icu_los_hours"] = max(current["min_icu_los_hours"], int(value))
        return None
    # Diagnosis systems, arbitrary concept columns, windows and thresholds
    # have no faithful projection into these admission fields. Do not turn
    # them into a preset or drop their coding system to make extraction run.
    return "population_preparation_kind_unsupported"


def population_preparation_proposal(
    *,
    study: Mapping[str, Any],
    run_id: str,
    requests: Sequence[Any],
) -> dict[str, Any] | None:
    """Build a read-only proposal from exactly one host-owned pending request.

    Older runs lack a sealed compilation and offer no new mutation. An
    amended plan, malformed record or inconsistent digest likewise offers
    nothing; an intact but unsupported criterion explains its blocked remedy.
    """
    selected = [
        request for request in requests if request.payload.get("reason") == STOP
    ]
    if len(selected) != 1:
        return None
    request = selected[0]
    payload = thaw_payload(request.payload)
    record = sealed_population(request, reason=STOP)
    if record is None:
        return None
    try:
        rows = record["criteria"]
        spec = PopulationSpec.model_validate(
            {"criteria": [row["criterion"] for row in rows]}
        )
        time_zero = record.get("time_zero_hours")
        if time_zero is not None and (
            isinstance(time_zero, bool)
            or not isinstance(time_zero, (int, float))
            or not math.isfinite(time_zero)
        ):
            return None
        if not any(
            row["disposition"] == "requires_extraction" and criterion.role == "include"
            for row, criterion in zip(rows, spec.criteria)
        ):
            return None
        cohort = dict(study.get("cohort") or {})
        previous = primary_cohort.normalize_execution_cohort(cohort)
        notes = []
        blockers = []
        for row, criterion in zip(rows, spec.criteria, strict=True):
            disposition = row["disposition"]
            if disposition not in {
                "applied_by_plan",
                "applied_by_source",
                "requires_extraction",
                "not_applied",
            }:
                return None
            refusal = None
            if disposition == "requires_extraction":
                refusal = _admission_patch(criterion, cohort, time_zero)
            elif disposition == "not_applied" and criterion.role == "include":
                refusal = "population_inclusion_not_applied"
            if refusal:
                blockers.append(refusal)
            notes.append(
                {
                    "id": criterion.id,
                    "quote": criterion.quote,
                    "source": criterion.source,
                    "proposed_by_system": criterion.source not in STUDY_WORDING_SOURCES,
                    "disposition": disposition,
                    "reason": refusal or row.get("reason"),
                }
            )
        normalized = primary_cohort.normalize_execution_cohort(cohort)
        changes = (
            [
                {"field": field, "before": previous[field], "after": normalized[field]}
                for field in primary_cohort.ADMISSION_ELIGIBILITY_FIELDS
                if previous[field] != normalized[field]
            ]
            if not blockers
            else []
        )
        result = {
            "schema_version": "easyicu.population_preparation/1",
            "run_id": run_id,
            "study_id": study["id"],
            "study_revision": study["revision"],
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(
                study
            ),
            "review_authority_sha256": request.authority_sha256,
            "population_compile_sha256": payload["population_compile_sha256"],
            "plan_sha256": payload["plan_review_authority"]["plan_sha256"],
            "available": not blockers and bool(changes),
            "criteria": notes,
            "status": "blocked"
            if blockers
            else "ready"
            if changes
            else "already_configured",
            "changes": changes,
            "blocker_codes": list(dict.fromkeys(blockers)),
            **({"cohort": cohort} if not blockers else {}),
        }
        return {**result, "proposal_sha256": canonical_sha256(result)}
    except (ValueError, KeyError, TypeError):
        return None


def current_population_preparation(
    study: Mapping[str, Any], review: Mapping[str, Any]
) -> dict | None:
    """Display and apply only a proposal for this exact study revision."""
    proposal = review.get("population_preparation")
    if (
        not isinstance(proposal, dict)
        or proposal.get("study_id") != study.get("id")
        or proposal.get("study_revision") != study.get("revision")
        or proposal.get("run_id") != review.get("run_id")
        or proposal.get("scientific_configuration_sha256")
        != study_contexts.scientific_configuration_sha256(study)
        or proposal.get("proposal_sha256")
        != canonical_sha256(
            {k: v for k, v in proposal.items() if k != "proposal_sha256"}
        )
    ):
        return None
    return proposal
