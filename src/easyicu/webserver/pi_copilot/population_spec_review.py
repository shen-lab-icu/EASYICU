"""Read the typed population from the exact plan offered for human review.

This is a display projection, not a research execution authority.  Missing,
legacy, altered, or differently normalized evidence yields no projection.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.frozen_payload import thaw_payload
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.cohort_contract import cohort_concept_id_scope, sealed_cohort_concept_ids
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.progressive_artifacts import ProgressivePlanningAuthority
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileReceipt, ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.scientific_review import (
    plan_time_zero_hours, trajectory_representation_facts,
)
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.research_agent.schema import AnalysisPlan

_MAX_FILE = 16 * 1024 * 1024
_FILES = {
    "progressive_planning_authority": "progressive_planning_authority.json",
    "progressive_plan_skeleton": "progressive_plan_skeleton.json",
    "progressive_plan_compile_receipt": "progressive_plan_compile_receipt.json",
    "research_context": "research_context.json",
}


def _bytes(path: Path, digest: str) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_FILE:
        raise ValueError("population review evidence unavailable")
    data = path.read_bytes()
    if len(data) > _MAX_FILE or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("population review evidence digest mismatch")
    return data


def reviewed_population_spec(*, run_dir: Path, pending: Any,
                             study_context_id: str | None = None) -> dict[str, Any] | None:
    """Return only source criteria bound to this pending review's exact plan."""
    try:
        authorities = []
        for request in pending.requests:
            payload = request.payload if isinstance(request.payload, Mapping) else {}
            raw = payload.get("plan_review_authority")
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                return None
            authorities.append(PlanReviewAuthority.model_validate(raw))
        if not authorities or any(a != authorities[0] for a in authorities[1:]):
            return None
        review = authorities[0]
        content = {
            key: _bytes(Path(run_dir) / filename, review.evidence_sha256[key])
            for key, filename in _FILES.items()
        }
        authority = ProgressivePlanningAuthority.model_validate_json(content["progressive_planning_authority"])
        skeleton_json = json.loads(content["progressive_plan_skeleton"])
        skeleton = ProgressivePlanSkeleton.model_validate(skeleton_json)
        receipt = ProgressivePlanCompileReceipt.model_validate_json(content["progressive_plan_compile_receipt"])
        if (
            authority.normalized_plan_sha256 != review.plan_sha256
            or authority.skeleton_artifact_sha256 != review.evidence_sha256["progressive_plan_skeleton"]
            or authority.compile_receipt_artifact_sha256 != review.evidence_sha256["progressive_plan_compile_receipt"]
            or canonical_sha256(skeleton_json) != authority.compiled_skeleton_sha256
            or receipt.skeleton_sha256 != authority.compiled_skeleton_sha256
            or receipt.analysis_plan_sha256 != authority.compiled_analysis_plan_sha256
        ):
            return None
        raw_spec = skeleton.cohort.population_spec
        if raw_spec is None:
            return None
        spec = PopulationSpec.model_validate(raw_spec)
        context = parse_research_context_json(content["research_context"].decode("utf-8"))
        with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
            plan = AnalysisPlan.model_validate(thaw_payload(review.plan_payload))
        if plan.cohort is None:
            return None
        if study_context_id is not None:
            from easyicu.webserver.first_observed_run_guard import host_first_observed_review_scope
            scope = host_first_observed_review_scope(study_context_id=study_context_id)
        else:
            scope = nullcontext()
        with scope:
            compiled = compile_population(
                spec, context,
                time_zero_hours=plan_time_zero_hours(context, trajectory_representation_facts(context, plan), None),
            )
        observed = compiled.cohort_definition().plan_dict()
        approved = plan.cohort.plan_dict()
        if {k: v for k, v in observed.items() if k != "name"} != {
            k: v for k, v in approved.items() if k != "name"
        }:
            return None
        return {
            "schema_version": "easyicu.reviewed_population_spec_projection/1",
            "plan_sha256": review.plan_sha256,
            "skeleton_sha256": authority.compiled_skeleton_sha256,
            "criteria": [
                {"criterion": item.criterion.model_dump(mode="json"), "disposition": item.disposition}
                for item in compiled.criteria
            ],
        }
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return None
