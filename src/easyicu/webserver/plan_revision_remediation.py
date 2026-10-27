"""Single-use structural authorizations in a digest-bound plan review."""

from __future__ import annotations

import json

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.requirement_coverage import PlanRevisionReceipt, PlanStructuralChange
from easyicu.research_agent.planning.scientific_review import remediation_route_for_finding


def _module(value: str | None) -> str:
    return str(value or "").strip().casefold().split("(", 1)[0].strip()


class StructuralRepairScope:
    """Each exact remediation is consumed once, even if two findings name it."""

    def __init__(self, review, previous):
        self.review = review
        self.source_ids = {step["step_id"] for step in previous["steps"]}
        self.scopes = [
            (finding.code, index, remediation)
            for finding in review.findings
            if not finding.requires_user_authorization
            and remediation_route_for_finding(finding) == "agent_plan_revision"
            for index, remediation in enumerate(finding.structural_remediations)
        ]
        self.used: set[str] = set()
        self.changes: list[PlanStructuralChange] = []

    def authorize(self, step, remaining) -> tuple[bool, bool]:
        """Return (authorized, related scope); related refusals cannot fall back."""

        related = False
        for code, index, repair in self.scopes:
            action_matches = step.scientific_action_id == repair.action
            module_matches = repair.module is None or _module(step.method) == _module(repair.module)
            if not (repair.action is not None and action_matches or repair.module is not None and module_matches):
                continue
            related = True
            identity = canonical_sha256({**repair.model_dump(mode="json"), "inputs": sorted(repair.inputs)})
            if identity in self.used or not action_matches or not module_matches:
                continue
            if step.planned_analysis_role != repair.planned_analysis_role or not set(step.inputs) <= set(repair.inputs):
                continue
            # A model's hidden roster cannot extend a declared input ceiling.
            columns = {value for req in step.model_requirements for value in (
                req.outcome, req.exposure_source, *req.covariates, *(term.name for term in req.model_terms),
            ) if value}
            if not columns <= set(repair.inputs):
                continue
            source = next((i for i, (old, _) in enumerate(remaining) if old["step_id"] == repair.step_id), None)
            if repair.operation == "add_step":
                if step.step_id in self.source_ids:
                    continue
            elif step.step_id != repair.step_id or source is None:
                continue
            if repair.operation == "replace_module":
                remaining.pop(source)
            self.used.add(identity)
            self.changes.append(PlanStructuralChange(
                finding_code=code, remediation_index=index, step_id=step.step_id, remediation=repair,
            ))
            return True, True
        return False, related

    def receipt(self, plan):
        if not self.scopes:
            return None  # Preserve the return and serialization of legacy repairs.
        return PlanRevisionReceipt(
            source_plan_sha256=self.review.plan_sha256,
            revised_plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
            structural_changes=tuple(self.changes),
        )

    def prompt(self) -> str:
        if not self.scopes:
            return ""
        return "\n- Finding-authorized structural changes (each scope once): " + json.dumps([
            {"finding_code": code, **repair.model_dump(mode="json")}
            for code, _, repair in self.scopes
        ], ensure_ascii=False, separators=(",", ":"))
