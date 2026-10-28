"""Recover an unsealed survival family through an explicit, study-bound choice.

Only metadata and saved coordinates are read. The existing plan decision
compiler writes the declaration; the survival suite validates and executes it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from easyicu.outcome_availability import fixed_horizon_mortality_endpoint
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.research_agent.contracts.rmst import RMSTSpec
from easyicu.webserver import study_contexts
from easyicu.webserver.landmark_survival_runtime_projection import (
    landmark_survival_rmst_spec,
    survival_family_declared,
    survival_inference_supported,
)
from easyicu.webserver.study_scientific_configuration import ScientificConfiguration

from .survival_decision_source import (
    SurvivalDecisionSource, read_survival_decision_source, study_with_run_coordinates,
)

if TYPE_CHECKING:
    from .plan_decisions import CompiledPlanDecision

DECISION_CODE = "SURVIVAL_LANDMARK_OWNER_NOT_SEALED"
OPTION_ID = "fixed_landmark_survival"


class SurvivalLandmarkDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    decision_code: Literal["SURVIVAL_LANDMARK_OWNER_NOT_SEALED"] = DECISION_CODE
    option_id: Literal["fixed_landmark_survival"] = OPTION_ID
    scientific_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_run_id: str | None = Field(default=None, min_length=1, max_length=160)
    research_context_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    exposure: str = Field(min_length=1, max_length=80)
    landmark_hours: float = Field(gt=0, le=8760)
    endpoint: str = Field(min_length=1, max_length=80)
    horizon_days: float = Field(gt=0)
    followup_column: str = Field(min_length=1, max_length=80)
    covariates: tuple[str, ...] = Field(min_length=1, max_length=64)
    rmst_alternative: RMSTSpec | None = None
    card_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _card_digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def propose_survival_landmark(
    study: Mapping[str, Any], *, source: SurvivalDecisionSource | None = None,
) -> SurvivalLandmarkDecision | None:
    """Offer a saved or host-resolved event-window design, without adopting it."""
    from easyicu.outcome_availability import stated_mean_survival_mentions, stated_mortality_horizon_mentions
    from easyicu.webserver.question_mean_survival import asked_mean_survival

    try:
        study_digest = study_contexts.scientific_configuration_sha256(dict(study))
        if source is not None:
            bound = study_with_run_coordinates(study, source)
            if bound is None:
                return None
            study = bound
        configuration = ScientificConfiguration.inspect(study)
        if (not survival_family_declared(study)
                or not survival_inference_supported(study["analysis_design"])
                or configuration.covariate_selection() != "exact"):
            return None
        endpoint = fixed_horizon_mortality_endpoint(configuration.executable_target_outcome() or "")
        exposure = configuration.executable_primary_exposure()
        covariates = configuration.covariates()
        if endpoint is None or not exposure or not covariates:
            return None
        # Never replace an existing timing design, or discard a competing suite.
        specs = configuration.sensitivity_specs()
        if any(spec.axis == "timing" or spec.strategy == "restricted_mean_survival" for spec in specs):
            return None
        _start, hours = configuration.materialization_window(
            window_finding=study_contexts.materialization_window_finding(dict(study)),
        )
        if hours / 24 >= endpoint.horizon_days:
            return None
        database = str((study.get("data_source") or {}).get("database") or "")
        catalog = {row.concept_id: row for row in build_database_capability_catalog(database).concepts}
        if exposure not in catalog or catalog[exposure].column_role != "event_status":
            return None
        if len({exposure, endpoint.event_concept, endpoint.followup_concept, *covariates}) != 3 + len(covariates):
            return None
        question = str(study.get("question") or "")
        if asked_mean_survival(question):
            return None
        mean_survival = stated_mean_survival_mentions(question)
        # Feed recognized RMST mentions to the existing horizon reader using
        # its survival noun; the acronym itself is not an endpoint word there.
        horizon_text = question
        for mention in reversed(mean_survival):
            horizon_text = horizon_text[:mention.start] + "survival" + horizon_text[mention.end:]
        if mean_survival and any(
            not mention.horizon.admits(endpoint.horizon_days)
            for mention in stated_mortality_horizon_mentions(horizon_text)
        ):
            return None
        rmst = landmark_survival_rmst_spec(
            exposure=exposure, landmark_hours=hours, endpoint=endpoint,
        ) if mean_survival else None
        body = {
            "decision_code": DECISION_CODE, "option_id": OPTION_ID,
            "scientific_configuration_sha256": study_digest,
            "source_run_id": source.run_id if source else None,
            "research_context_sha256": source.context_sha256 if source else None,
            "exposure": exposure, "landmark_hours": hours,
            "endpoint": endpoint.event_concept, "horizon_days": float(endpoint.horizon_days),
            "followup_column": endpoint.followup_concept, "covariates": list(covariates),
            "rmst_alternative": rmst.model_dump(mode="json") if rmst else None,
        }
        return SurvivalLandmarkDecision(**body, card_sha256=_card_digest(body))
    except (KeyError, TypeError, ValueError):
        # A study outside the suite's coordinates keeps its original stop.
        return None


def family_stop_decision(
    study: Mapping[str, Any], *, code: str, typed_failure: Mapping[str, Any],
    wrapper_dir: Path | None = None,
) -> dict[str, Any] | None:
    if (code != "research_pipeline_progressive_compile_failed"
            or typed_failure.get("owner") != "easyicu.planning.progressive_compiler_v1"
            or typed_failure.get("reason_code") != "progressive_family_result_contract_unwritable"):
        return None
    proposed = propose_survival_landmark(study, source=read_survival_decision_source(wrapper_dir))
    return proposed.model_dump(mode="json") if proposed else None


def pending_survival_decision(
    study: Mapping[str, Any], payloads: Mapping[str, Any],
    *, wrapper_dir: Path | None = None,
) -> SurvivalLandmarkDecision | None:
    manifest = payloads.get("source_run_manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("status") != "plan_decision_pending":
        return None
    try:
        card = SurvivalLandmarkDecision.model_validate(manifest.get("plan_decision"))
    except ValidationError:
        return None
    source = read_survival_decision_source(wrapper_dir)
    return card if card == propose_survival_landmark(study, source=source) else None


def family_stop_result(wrapper_dir: Path) -> dict[str, Any] | None:
    """Return the persisted decision as a completed plan-only job result."""
    try:
        manifest = json.loads((wrapper_dir / "source_run_manifest.json").read_text())
        if not isinstance(manifest, dict) or manifest.get("status") != "plan_decision_pending":
            return None
        card = SurvivalLandmarkDecision.model_validate(manifest.get("plan_decision"))
        gate = json.loads((wrapper_dir / "quality_gate.json").read_text())["gate"]
    except (OSError, ValueError, KeyError):
        return None
    return {
        "run_id": manifest["run_id"], "project_dir": str(wrapper_dir),
        "status": "plan_decision_pending", "gate": gate,
        "plan_decision": card.model_dump(mode="json"),
        "human_review_pending": False, "analysis_started": False, "uploads": 0,
    }


def compile_survival_landmark_choice(
    *, study: Mapping[str, Any], option_id: str,
    decision_card: Mapping[str, Any] | None, card_sha256: str | None,
    wrapper_dir: Path | None = None,
) -> CompiledPlanDecision:
    from .plan_decisions import (
        CompiledPlanDecision, PlanDecisionError, _survival_design_configuration,
    )

    source = read_survival_decision_source(wrapper_dir)
    proposed = propose_survival_landmark(study, source=source)
    if (option_id != OPTION_ID or proposed is None
            or decision_card != proposed.model_dump(mode="json")
            or not hmac.compare_digest(str(card_sha256 or ""), proposed.card_sha256)):
        raise PlanDecisionError(
            "plan_decision_source_superseded",
            "The landmark decision changed; reload its current card before choosing it.",
        )
    bound = study_with_run_coordinates(study, source) if source else dict(study)
    compiled = _survival_design_configuration(
        bound, codes=(DECISION_CODE,), first_stay_restricted=False,
        saved_study_roster=True,
        review_facts={"landmark_survival_suite": {
            "sealed": False, "executable": True,
            "exposure": proposed.exposure, "event_column": proposed.endpoint,
            "followup_column": proposed.followup_column,
            "landmark_hours": proposed.landmark_hours, "covariates": proposed.covariates,
            "covariate_rationales": bound.get("covariate_rationales") or {},
            "covariate_temporal_roles": bound.get("covariate_temporal_roles") or {},
        }},
    )
    patch = dict(compiled.patch)
    if source and bound.get("covariate_authority") == "researcher_stated":
        # The compiler validates the declaration. Persist only this click's
        # coordinates and timing/roster choice, preserving other study fields.
        patch = {key: patch[key] for key in (
            "execution_concepts", "covariates", "covariate_selection", "covariate_authority",
            "covariate_rationales", "covariate_operationalizations", "sensitivity_specs",
        )}
        patch["time_window"] = bound["time_window"]
        patch["confirmations"] = ScientificConfiguration.inspect(study).merge_confirmations(
            plan_adjustment_set_confirmed=True, plan_timing_landmark_24h=True,
            plan_timing_descriptive_only=False, plan_timing_time_varying=False,
        )
    if proposed.rmst_alternative is not None:
        patch["sensitivity_specs"] = [*patch["sensitivity_specs"], {
            "spec_id": "landmark_survival_rmst", "axis": "estimand",
            "strategy": "restricted_mean_survival",
            "rmst_execution": proposed.rmst_alternative.model_dump(mode="json"),
        }]
    return CompiledPlanDecision(
        patch=patch, display_label_en="Use the fixed-landmark survival design",
        display_label_zh="已采用固定 landmark 生存分析设计", next_action="replan",
    )
