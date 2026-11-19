"""B1: a Codex model-service fault reads as retryable; a rejected request keeps its closed code in the receipt."""
from __future__ import annotations

from easyicu.research_agent.providers.codex_app_server import CodexAppServerError
from easyicu.webserver import agent_pipeline_runs as runs


def _turn_failure(**diagnostic: object) -> CodexAppServerError:
    error = CodexAppServerError("codex_app_server_turn_failed")
    error.easyicu_safe_diagnostic.update(diagnostic)
    return error


def test_a_dropped_model_stream_is_a_retryable_provider_stop() -> None:
    error = _turn_failure(error_kind="responseStreamDisconnected")

    assert runs._pipeline_failure_code(error) == "research_pipeline_planner_provider_unavailable"


def test_an_account_service_request_failure_is_a_retryable_provider_stop() -> None:
    assert runs._pipeline_failure_code(
        CodexAppServerError("codex_auth_app_server_request_failed")
    ) == "research_pipeline_planner_provider_unavailable"


def test_a_rejected_request_names_its_kind_and_code_in_the_receipt() -> None:
    error = _turn_failure(error_kind="other", provider_error_code="invalid_json_schema", http_status=400, message="free text")

    assert runs._pipeline_failure_code(error) != "research_pipeline_planner_provider_unavailable"
    assert runs._safe_pipeline_typed_failure(error) == {
        "owner": "easyicu.providers.codex_app_server_v1",
        "reason_code": "codex_app_server_turn_failed",
        "error_kind": "other",
        "provider_error_code": "invalid_json_schema",
        "http_status": 400,
    }
