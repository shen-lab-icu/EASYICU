"""Compile attached method identities and enforce their existing plan coordinates.

The host carries these typed records through UserPreferences.data_constraints;
no Planner input/output schema or scientific finding vocabulary is extended.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from easyicu.extensions.contracts import ExtensionRegistryError
from easyicu.extensions.skill_attachments import parse_attachments

from easyicu.research_agent.method_skills import METHOD_SKILLS, METHOD_SKILL_REGISTRY_VERSION
from easyicu.research_agent.planning.capability_registry import get_capability_by_id, resolve_primary_capability
from easyicu.research_agent.planning.analysis_types import canonical_analysis_family, list_analysis_types


def question_family_for_method(family: str) -> str:
    return {"ordinal_dose_response": "association_study", "dynamic_prediction": "prediction_model"}.get(family, family)


def builtin_attachment(skill_id: str) -> dict:
    skill = next((item for item in METHOD_SKILLS if item.skill_id == skill_id), None)
    if skill is None:
        raise ExtensionRegistryError("skill_attachment_unknown", "skill_attachment_unknown")
    digest = hashlib.sha256(json.dumps(
        skill.to_dict(enabled=True), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return {"source": "builtin", "id": skill_id, "version": METHOD_SKILL_REGISTRY_VERSION, "digest": digest}


def method_constraints(attachments: object) -> list[dict]:
    constraints = []
    for ref in parse_attachments(attachments):
        if ref.source != "builtin":
            continue
        identity = builtin_attachment(ref.id)
        if ref.version != identity["version"] or (ref.digest is not None and ref.digest != identity["digest"]):
            raise ExtensionRegistryError("skill_attachment_stale", "skill_attachment_stale")
        spec = next(item for item in METHOD_SKILLS if item.skill_id == ref.id)
        capability = get_capability_by_id(spec.capability_id)
        if capability is None:
            # Data-definition and planning tools have no study design to pin.
            raise ExtensionRegistryError("skill_attachment_disabled", "skill_attachment_disabled")
        subtypes = [item for item in list_analysis_types() if item.capability_id == spec.capability_id]
        subtype = subtypes[0] if len(subtypes) == 1 else None
        family = subtype.key if subtype else canonical_analysis_family(capability.family)
        if family is None:
            family = {"time_to_event": "survival", "causal_emulation": "causal_inference"}[capability.family]
        constraints.append({
            "id": ref.id, "analysis_family": family,
            "capability_id": spec.capability_id, "action_ids": list(spec.action_ids),
        })
    if len({item["analysis_family"] for item in constraints}) > 1:
        raise ExtensionRegistryError("skill_attachment_family_conflict", "skill_attachment_family_conflict")
    return constraints


def validate_attached_methods(*, plan: Any, context: Any) -> None:
    """Refuse a candidate that drops a host pin before it can be persisted."""
    raw = getattr(getattr(context, "user_preferences", None), "data_constraints", None)
    if not raw:
        return
    try:
        constraints = json.loads(raw)
    except (TypeError, ValueError):
        return  # Legacy free-text constraints have no attachment authority.
    if not isinstance(constraints, dict) or not constraints.get("attached_skills"):
        return
    expected = method_constraints(constraints["attached_skills"])
    choice = constraints.get("question_family_check") or {}
    if choice.get("status") == "researcher_chose":
        expected = [item for item in expected if question_family_for_method(item["analysis_family"]) == choice.get("chosen_family")]
    if constraints.get("method_skill_constraints") != expected:
        raise ValueError("skill_attachment_constraint_mismatch")
    for item in expected:
        family = getattr(plan, "analysis_type", None)
        if family != item["analysis_family"]:
            raise ValueError("skill_attachment_plan_family_mismatch")
        verdict = resolve_primary_capability(analysis_type=family, plan=plan)
        capabilities = {getattr(step, "scientific_capability", None) for step in plan.steps}
        spec = next(spec for spec in METHOD_SKILLS if spec.skill_id == item["id"])
        if spec.layer == "research_workflow" or verdict.capability_id == item["capability_id"]:
            capabilities = set()
        if verdict.coherent:
            capabilities.add(verdict.capability_id)
        if item["capability_id"] not in capabilities:
            raise ValueError("skill_attachment_plan_capability_mismatch")
        actions = {getattr(step, "scientific_action_id", None) for step in plan.steps}
        if not set(item["action_ids"]) <= actions:
            raise ValueError("skill_attachment_plan_action_mismatch")
