"""Provider gate for native FastAPI agent runs.

This module deliberately does not construct clients and does not read
credentials. Its only job is to classify a requested provider, reuse the
canonical EasyICU opt-in check, and decide whether the downstream credential
adapter is allowed to run.
"""
from __future__ import annotations

from typing import Any, Dict

from easyicu.ai_optin import (
    AIOptInError,
    check_external_llm_opt_in,
    is_offline_llm_choice,
)

CANONICAL_OPT_IN_SOURCE = "easyicu.ai_optin.check_external_llm_opt_in"


class ProviderGateError(ValueError):
    """Raised when an agent run cannot pass the provider gate."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "provider_gate_error"))
        self.detail = detail


def resolve_provider_gate(
    *,
    run_type: str,
    llm_provider: str,
    external_llm_opt_in: bool,
    ai_enabled: bool,
    language: str = "en",
) -> Dict[str, Any]:
    """Resolve provider policy without loading credentials or clients."""
    resolved_run_type = str(run_type or "preflight").strip().lower()
    provider = _base_provider_info(llm_provider, external_llm_opt_in, ai_enabled)
    if resolved_run_type != "full":
        provider["provider_gate"] = "not_used_for_preflight"
        return provider
    if not provider["external"]:
        return provider

    try:
        check_external_llm_opt_in(
            provider["provider"],
            ai_enabled=provider["ai_enabled"],
            language=language,
        )
        provider["canonical_opt_in_passed"] = True
        provider["provider_gate_order"].append("canonical_opt_in_passed")
    except AIOptInError as exc:
        provider["canonical_opt_in_passed"] = False
        provider["provider_gate_order"].append("canonical_opt_in_blocked")
        raise ProviderGateError({
            **provider,
            "error": "external_llm_opt_in_required",
            "run_type": resolved_run_type,
            "blocked_by": "canonical_ai_opt_in",
            "message": str(exc),
        }) from exc

    if not provider["per_run_opt_in"]:
        provider["provider_gate_order"].append("per_run_opt_in_blocked")
        raise ProviderGateError({
            **provider,
            "error": "external_llm_opt_in_required",
            "run_type": resolved_run_type,
            "blocked_by": "per_run_external_llm_opt_in",
        })

    provider["provider_gate_order"].append("per_run_opt_in_passed")
    provider["provider_gate_order"].append("credential_lookup_allowed")
    provider["provider_gate"] = "credential_lookup_allowed"
    return provider


def _base_provider_info(
    provider: str,
    external_llm_opt_in: bool,
    ai_enabled: bool,
) -> Dict[str, Any]:
    text = str(provider or "mock").strip() or "mock"
    external = not is_offline_llm_choice(text)
    return {
        "provider": text,
        "external": external,
        "ai_enabled": bool(ai_enabled),
        "per_run_opt_in": bool(external_llm_opt_in),
        "canonical_opt_in_source": CANONICAL_OPT_IN_SOURCE,
        "canonical_opt_in_passed": False if external else True,
        "client": "MockLLMClient" if not external else None,
        "provider_gate": "offline_mock" if not external else "blocked_before_client_construction",
        "provider_gate_order": ["classify_provider"],
        "credentials_loaded": False,
        "credentials_attempted": False,
        "credential_source": None,
        "client_constructed": False,
        "mock_calls": 0,
        "external_calls": 0,
    }
