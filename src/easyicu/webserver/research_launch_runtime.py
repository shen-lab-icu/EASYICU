"""Provider credential binding and execution-runtime launch preflight."""

from __future__ import annotations

import re
from typing import Any, Mapping

from easyicu.webserver import provider_adapter
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


def _clean_text(value: Any, limit: int = 1_200) -> str:
    return re.sub(r"\\s+", " ", str(value or "")).strip()[:limit]


def _validated_pipeline_credential_source(
    credential_source: str,
    *,
    provider: Mapping[str, Any],
) -> str:
    """Bind one Web credential source to the matching provider family."""

    selected = str(credential_source or "").strip().lower()
    if selected not in {"pi_verified", "codex_user_auth"}:
        raise ResearchPipelineRunError(
            "research_pipeline_credential_source_invalid",
            "Choose one server-verified Research Agent credential source.",
        )
    provider_name = _clean_text(provider.get("provider"), 64).lower()
    account_provider = provider_adapter.is_user_account_provider(provider_name)
    if selected == "codex_user_auth" and not account_provider:
        raise ResearchPipelineRunError(
            "research_pipeline_codex_user_auth_provider_required",
            "Codex user authentication requires the reviewed Codex account provider.",
        )
    if selected == "pi_verified" and account_provider:
        raise ResearchPipelineRunError(
            "research_pipeline_codex_user_auth_required",
            "The Codex account provider requires this browser user's ChatGPT login.",
        )
    return selected


def _submission_profile_ref(*, budget_mode: str, live_pubmed: bool) -> str:
    """Resolve the one submission profile a budget mode selects.

    The launch-time runtime preflight and the run itself have to agree on which
    profile will be used. Compiling that mapping twice is how a preflight ends
    up guarding a profile the run never selects, so both read it from here.
    """

    from easyicu.research_agent.orchestration.profiles import (
        CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF,
        CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF,
    )

    if str(budget_mode or "").strip().lower() == "full_reviewed":
        return (
            CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF
            if live_pubmed
            else CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF
        )
    return (
        CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF
        if live_pubmed
        else CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF
    )


def _require_execution_runtime(*, budget_mode: str, runner_image: str) -> None:
    """Refuse a launch whose execution backend is already known to be down.

    Whether the mandated container runtime can run is a static fact a bounded
    probe settles in seconds. Left unasked at launch, it is discovered only
    inside the pipeline -- after the provider is authorized, the data
    foundation is built and the cohort is materialized -- so a stopped daemon
    costs a minute and a half of real provider spend before saying so.
    """

    from easyicu.research_agent.execution import runner as runner_module
    from easyicu.research_agent.orchestration.profiles import get_submission_profile

    required: set[str] = set()
    for live_pubmed in (False, True):
        # Which of the two variants a run picks depends on a literature
        # binding resolved inside the run, so guard both.
        profile = get_submission_profile(
            _submission_profile_ref(budget_mode=budget_mode, live_pubmed=live_pubmed)
        )
        # A planner-only profile never launches generated code, and its runs
        # must stay startable on a host with no container runtime at all.
        if profile.planner_only:
            continue
        kind = str(profile.requires_runner or "").strip().lower()
        if kind:
            required.add(kind)
    for kind in sorted(required):
        availability = runner_module.probe_runner_availability(
            kind,
            image=(runner_image or runner_module.DockerRunner.DEFAULT_IMAGE),
        )
        if availability.available:
            continue
        raise ResearchPipelineRunError(
            "research_pipeline_execution_runtime_unavailable",
            "The governed execution runtime is not ready, so this run would "
            "fail after the provider and cohort work it is about to spend. "
            + runner_module.runner_unavailable_remediation(availability.reason_code),
            details={
                "owner": "easyicu.research_agent.execution.runner",
                "reason_code": availability.reason_code,
                "runner_kind": availability.kind,
                "runner_image": availability.image,
            },
        )
