"""Project the plan review's database capability decision for the approval card.

The decision itself is owned by ``easyicu.databases.research_requirements`` and
bound to the reviewed plan by ``easyicu.databases.research_approval``; this
module only bounds what the browser receives. An exact study projects nothing.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

_STATUSES = {"substitution", "refusal"}
_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


def _token(value: Any, limit: int = 120) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _term(value: Any) -> str | dict[str, Any] | None:
    if isinstance(value, str):
        return _token(value, 80) or None
    if not isinstance(value, Mapping):
        return None
    number = value.get("value")
    if (
        _token(value.get("concept"), 40)
        and value.get("operator") in _OPERATORS
        and isinstance(number, (int, float))
        and not isinstance(number, bool)
        and math.isfinite(number)
    ):
        return {"concept": _token(value.get("concept"), 40), "operator": value["operator"], "value": number}
    return None


def _field(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    field = {key: _token(value.get(key)) for key in ("table", "column")}
    return field if any(field.values()) else None


def project_source_adjustments(review: Any) -> dict[str, Any] | None:
    """Bound a ``database_capability_review`` companion; None when nothing to show."""
    if not isinstance(review, Mapping) or not isinstance(review.get("decision"), Mapping):
        return None
    decision = review["decision"]
    status = decision.get("status")
    if status not in _STATUSES:
        return None
    substitutions = []
    for item in list(decision.get("substitutions") or [])[:12]:
        if not isinstance(item, Mapping):
            continue
        original, substitute = _term(item.get("original")), _term(item.get("substitute"))
        if _token(item.get("capability"), 80) and original is not None and substitute is not None:
            substitutions.append({
                "capability": _token(item.get("capability"), 80),
                "original": original,
                "substitute": substitute,
                "reason_code": _token(item.get("reason_code"), 80),
                "source_field": _field(item.get("source_field")),
            })
    missing = [
        {"capability": _token(item.get("capability"), 80), "code": _token(item.get("code"), 80)}
        for item in list(decision.get("missing") or [])[:20]
        if isinstance(item, Mapping) and _token(item.get("code"), 80)
    ]
    unapplied = [
        _token(name, 80) for name in list(review.get("unapplied_substitutions") or [])[:12]
        if _token(name, 80)
    ]
    if status == "substitution" and not substitutions:
        return None
    return {
        "decision": {
            "status": status,
            "database": _token(decision.get("database"), 40),
            "substitutions": substitutions,
            "missing": missing,
        },
        "unapplied_substitutions": unapplied,
    }


def current_source_adjustments(review_authority: Mapping[str, Any]) -> dict[str, Any] | None:
    review = review_authority.get("scientific_plan_review")
    facts = review.get("facts") if isinstance(review, Mapping) else None
    projected = project_source_adjustments(
        facts.get("database_capability_review") if isinstance(facts, Mapping) else None
    )
    if projected is None or not isinstance(facts, Mapping):
        return projected
    candidate = facts.get("database_time_origin_review")
    selection = candidate.get("selection") if isinstance(candidate, Mapping) else None
    receipt = candidate.get("registered_export_receipt") if isinstance(candidate, Mapping) else None
    if not isinstance(selection, Mapping) or not isinstance(receipt, Mapping):
        return projected
    field = selection.get("source_field")
    if (candidate.get("execution_contract_available") is not False
            or candidate.get("disposition") not in {"proposed_not_selected", "proposed_not_approved"}
            or selection.get("requested_origin") != "icu_admission"
            or selection.get("selected_origin") != "first_intensive_or_intermediate_care_transfer"
            or not isinstance(field, Mapping) or field.get("table") != "cases" or field.get("column") != "ICUOffset"
            or receipt.get("selected_origin") != selection.get("selected_origin")
            or receipt.get("source_identity_sha256") != selection.get("source_identity_sha256")
            or any(not isinstance(receipt.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", receipt[key])
                   for key in ("source_identity_sha256", "manifest_sha256", "export_authority_sha256"))):
        return projected
    return {**projected, "time_origin_candidate": {"field": "cases.ICUOffset"}}
