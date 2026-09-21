"""Shared test doubles for Pi Copilot contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from easyicu.webserver import research_run_submission
from easyicu.webserver.pi_copilot.contracts import (
    PiSessionDataSourceAuthorization,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfig
from easyicu.webserver.pi_copilot.service import PiCopilotService


def record_pipeline_submission(
    submitted: list[Any],
    request: research_run_submission.ResearchRunSubmissionRequest,
    *,
    job_id: str,
    authorize: Any = None,
    **_kwargs: Any,
) -> research_run_submission.ResearchRunSubmissionReceipt:
    if authorize is not None:
        authorize()
    submitted.append(request)
    return research_run_submission.ResearchRunSubmissionReceipt(
        job_id=job_id,
        kind="agent-run",
        status="queued",
        study_context_id=request.study_context_id,
        study_context_revision=1,
        budget_mode="full_reviewed",
        planner_start_mode=request.planner_start_mode,
    )


def allow_unrelated_message_test(
    service: PiCopilotService,
    session_id: str,
) -> None:
    """Keep non-consent tests focused on their original owner contract."""

    record = service._get_record(session_id)
    record.data_source_authorization = PiSessionDataSourceAuthorization(
        status="legacy_confirmed",
        confirmation_mode="legacy_session",
    )
    service._save_record(record)


class FakeGateway:
    def __init__(self, session_dir: Path | None = None) -> None:
        self.environ = {
            "EASYICU_PI_PROVIDER": "easyicu-local",
            "EASYICU_PI_API_KEY": "test-only-placeholder",
        }
        self.calls: list[tuple[str, dict[str, Any], Any]] = []
        self.tool_contexts: list[ToolExecutionContext] = []
        self.session_dir = session_dir
        self.applied_config: PiProviderConfig | None = None

    def installation_status(self) -> dict[str, Any]:
        return {
            "node_available": True,
            "node_version": "24.11.0",
            "node_version_supported": True,
            "entrypoint_available": True,
            "dependency_installed": True,
            "lockfile_present": True,
            "runtime_integrity_verified": True,
            "api_key_configured": True,
            "provider_connection_verified": True,
            "base_url_configured": True,
            "provider": "easyicu-local",
            "model": "gpt5.6 luna",
            "api_transport": "openai-completions",
        }

    def request(
        self,
        method: str,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((method, dict(params), kwargs.get("tool_context")))
        if kwargs.get("tool_context") is not None:
            self.tool_contexts.append(kwargs["tool_context"])
        if method == "session.regenerate.inspect":
            return {
                "message": "Original research question",
                "turn_index": 0,
                "study_context_snapshot": {
                    "schema_version": "easyicu.pi-turn-study-snapshot/1",
                    "study_context_id": "study-test",
                    "source_revision": 3,
                    "configuration": {
                        "question": "Is aggregate lactate associated with mortality?",
                        "data_source": {
                            "database": "mimiciv",
                            "path_hash": hashlib.sha256(
                                b"/private/export"
                            ).hexdigest()[:16],
                        },
                        "cohort": {"cohort_size": 140},
                        "modules": ["lactate"],
                        "outcome": "mortality",
                        "covariates": [],
                        "covariate_selection": "planner_selectable",
                        "time_window": {"hours": 24},
                        "confirmations": {"cohort": True},
                    },
                },
            }
        session_id = str(params.get("session_id") or "")
        return {
            "session_id": session_id,
            "pi_session_id": "pi-internal-test",
            "session_file": "/private/test-session.jsonl",
            "model": {"provider": "easyicu-local", "id": "gpt5.6 luna"},
            "thinking_level": params.get("thinking_level") or "medium",
            "message_count": 1 if method == "session.prompt" else 0,
            "streaming": False,
            "enabled_tools": ["easyicu_inspect_context"],
            "transcript": [],
            "aborted": method == "session.abort",
        }

    def close(self) -> None:
        return None

    def apply_provider_config(self, config: PiProviderConfig) -> None:
        self.applied_config = config
        self.environ.update(config.as_environment())
