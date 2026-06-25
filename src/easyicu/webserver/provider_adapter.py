"""External provider adapter for native FastAPI agent runs.

The adapter is intentionally narrow:

- credentials are read only after provider gate opt-ins pass;
- credentials are never returned in provider metadata or persisted artifacts;
- prompts contain bounded aggregate summaries only, never patient rows;
- model output must be JSON and still goes through STRICT evidence audit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from easyicu.webserver import agent_outputs

_MAX_EXTERNAL_CALLS_PER_RUN = 1
_DEFAULT_MAX_OUTPUT_TOKENS = 1200
_MIN_MAX_OUTPUT_TOKENS = 128
_ABSOLUTE_MAX_OUTPUT_TOKENS = 4000
_DEFAULT_PROVIDER_ENV_FILE = Path.home() / ".easyicu" / "provider.env"


class ProviderAdapterError(ValueError):
    """Raised when a connected provider cannot be used safely."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "provider_adapter_error"))
        self.detail = detail


def require_external_credentials(
    provider_meta: Dict[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return sanitized provider metadata after checking env credentials."""
    if not provider_meta.get("external"):
        return provider_meta
    try:
        credentials = _load_external_credentials(str(provider_meta.get("provider") or ""), environ=environ)
    except ProviderAdapterError as exc:
        raise ProviderAdapterError({**provider_meta, **exc.detail}) from exc
    updated = dict(provider_meta)
    updated.update(_credential_public_metadata(credentials))
    updated["provider_gate"] = "external_provider_credentials_ready"
    updated.setdefault("provider_gate_order", []).append("credentials_loaded")
    return updated


def generate_bound_provider_payload(
    *,
    provider_meta: Dict[str, Any],
    run_id: str,
    study_id: str,
    question: Optional[str],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
    output_artifacts: Optional[Dict[str, Dict[str, Any]]] = None,
    transport: Optional[Callable[[Dict[str, Any], Dict[str, str]], Dict[str, Any]]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible provider and return bounded artifacts."""
    credentials = _load_external_credentials(str(provider_meta.get("provider") or ""), environ=environ)
    max_output_tokens = _max_output_tokens(environ=environ)
    json_format_style = _json_format_style(environ=environ)
    request = _build_chat_request(
        provider=str(provider_meta.get("provider") or ""),
        run_id=run_id,
        study_id=study_id,
        question=question,
        summary=summary,
        cohort=cohort,
        quality=quality,
        output_artifacts=output_artifacts or {},
        model=credentials["model"],
        max_output_tokens=max_output_tokens,
        json_format_style=json_format_style,
    )
    headers = {
        "Authorization": f"Bearer {credentials['api_key']}",
        "Content-Type": "application/json",
    }
    if transport is None:
        response = _post_chat_completion(
            url=credentials["base_url"],
            request=request,
            headers=headers,
            timeout=45,
        )
    else:
        response = transport(request, headers)
    payload = _coerce_provider_payload(response, run_id=run_id, study_id=study_id, question=question)
    provider_update = dict(provider_meta)
    provider_update.update(_credential_public_metadata(credentials))
    provider_update.update({
        "client": "OpenAICompatibleChat",
        "client_constructed": True,
        "external_calls": int(provider_meta.get("external_calls") or 0) + 1,
        "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
        "max_output_tokens": max_output_tokens,
        "json_format_style": json_format_style,
        "provider_gate": "external_provider_ready",
        "provider_gate_order": [
            *list(provider_meta.get("provider_gate_order") or []),
            "client_constructed",
            "external_call_completed",
        ],
        "usage": _public_usage(response),
    })
    return {
        "agent_plan": payload["agent_plan"],
        "manuscript_draft": payload["manuscript_draft"],
        "provider": provider_update,
        "request_policy": request["easyicu_policy"],
    }


def provider_readiness(
    provider: str,
    *,
    ai_enabled: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return sanitized provider readiness without constructing clients."""
    env, env_file = _provider_env(environ=environ)
    provider_text = str(provider or "openai").strip() or "openai"
    external = not _is_offline_provider(provider_text)
    key_names = _api_key_env_names(provider_text)
    base_names = _base_url_env_names(provider_text)
    model_names = _model_env_names(provider_text)
    key_name, api_key = _first_env(env, key_names)
    base_name, base_url = _first_env(env, base_names)
    default_base = _default_base_url(provider_text)
    model_name, model = _first_env(env, model_names)
    has_base_url = bool(base_url or default_base)
    missing: List[str] = []
    if external and not ai_enabled:
        missing.append("ai_enabled")
    if external and not api_key:
        missing.append("credential")
    if external and not has_base_url:
        missing.append("base_url")
    if external and not model:
        missing.append("model")
    if external and env_file.get("status") == "insecure_permissions":
        missing.append("env_file_permissions")
    return {
        "provider": provider_text,
        "external": external,
        "ai_enabled": bool(ai_enabled),
        "credential_env_candidates": key_names,
        "credential_present": bool(api_key),
        "credential_source": key_name if api_key else None,
        "base_url_env_candidates": base_names,
        "base_url_present": has_base_url,
        "base_url_source": base_name if base_url else ("provider_default" if default_base else None),
        "model_env_candidates": model_names,
        "model_present": bool(model),
        "model_source": model_name if model else None,
        "ready": bool(
            (not external)
            or (
                ai_enabled
                and api_key
                and has_base_url
                and model
                and env_file.get("status") != "insecure_permissions"
            )
        ),
        "missing": missing,
        "env_file": env_file,
        "limits": {
            "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
            "max_output_tokens": _max_output_tokens(environ=env),
            "json_format_style": _json_format_style(environ=env),
        },
        "secrets_returned": False,
        "client_constructed": False,
        "network_calls": 0,
    }


def _load_external_credentials(
    provider: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    env, env_file = _provider_env(environ=environ)
    key_names = _api_key_env_names(provider)
    key_name, api_key = _first_env(env, key_names)
    base_name, base_url = _first_env(env, _base_url_env_names(provider))
    if not base_url:
        base_url = _default_base_url(provider)
    else:
        base_url = _chat_completions_url(base_url)
    model_name, model = _first_env(env, _model_env_names(provider))
    attempted = {
        "credentials_attempted": True,
        "credentials_loaded": False,
        "client_constructed": False,
        "credential_env_candidates": key_names,
        "env_file": env_file,
    }
    if env_file.get("status") == "insecure_permissions":
        raise ProviderAdapterError({
            **attempted,
            "error": "external_provider_env_file_permissions",
            "blocked_by": "external_provider_credentials",
        })
    if not api_key:
        raise ProviderAdapterError({
            **attempted,
            "error": "external_provider_credentials_required",
            "blocked_by": "external_provider_credentials",
        })
    if not base_url:
        raise ProviderAdapterError({
            **attempted,
            "error": "external_provider_base_url_required",
            "blocked_by": "external_provider_credentials",
            "credential_source": key_name,
        })
    if not model:
        raise ProviderAdapterError({
            **attempted,
            "error": "external_provider_model_required",
            "blocked_by": "external_provider_credentials",
            "credential_source": key_name,
        })
    return {
        "provider": provider,
        "api_key": api_key,
        "api_key_env": key_name,
        "base_url": base_url,
        "base_url_env": base_name or "provider_default",
        "model": model,
        "model_env": model_name,
    }


def _credential_public_metadata(credentials: Dict[str, str]) -> Dict[str, Any]:
    return {
        "credentials_attempted": True,
        "credentials_loaded": True,
        "credential_source": credentials["api_key_env"],
        "credential_fingerprint": _fingerprint(credentials["api_key"]),
        "base_url_configured": True,
        "base_url_endpoint": "chat_completions",
        "base_url_source": credentials["base_url_env"],
        "model": credentials["model"],
        "model_source": credentials["model_env"],
        "client_constructed": False,
    }


def _api_key_env_names(provider: str) -> List[str]:
    normalized = _normalize_provider(provider)
    if normalized == "openai":
        return ["OPENAI_API_KEY", "EASYICU_LLM_API_KEY"]
    if normalized == "openrouter":
        return ["OPENROUTER_API_KEY", "EASYICU_LLM_API_KEY"]
    if normalized == "anthropic":
        return ["ANTHROPIC_API_KEY", "EASYICU_LLM_API_KEY"]
    if normalized == "custom":
        return ["EASYICU_LLM_API_KEY"]
    return [f"{_env_token(normalized)}_API_KEY", "EASYICU_LLM_API_KEY"]


def _base_url_env_names(provider: str) -> List[str]:
    normalized = _normalize_provider(provider)
    return [f"{_env_token(normalized)}_BASE_URL", "EASYICU_LLM_BASE_URL"]


def _model_env_names(provider: str) -> List[str]:
    normalized = _normalize_provider(provider)
    return [f"{_env_token(normalized)}_MODEL", "EASYICU_LLM_MODEL"]


def _default_base_url(provider: str) -> str:
    normalized = _normalize_provider(provider)
    if normalized == "openai":
        return "https://api.openai.com/v1/chat/completions"
    if normalized == "openrouter":
        return "https://openrouter.ai/api/v1/chat/completions"
    return ""


def _chat_completions_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/chat/completions"):
        return text
    return text + "/chat/completions"


def _first_env(env: Mapping[str, str], names: List[str]) -> tuple[Optional[str], str]:
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            return name, value
    return None, ""


def _build_chat_request(
    *,
    provider: str,
    run_id: str,
    study_id: str,
    question: Optional[str],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
    output_artifacts: Dict[str, Dict[str, Any]],
    model: str,
    max_output_tokens: int,
    json_format_style: str,
) -> Dict[str, Any]:
    valid_evidence = [
        "run_context.json",
        "cohort_summary.json",
        *agent_outputs.OUTPUT_ARTIFACT_NAMES,
        "quality_gate.json",
    ]
    bounded_context = {
        "run_id": run_id,
        "study_id": study_id,
        "question": question,
        "summary": summary,
        "cohort": cohort,
        "quality": quality,
        "output_artifacts": {
            name: output_artifacts.get(name)
            for name in agent_outputs.OUTPUT_ARTIFACT_NAMES
            if name in output_artifacts
        },
        "valid_evidence_ids": valid_evidence,
    }
    system = (
        "You are generating a locked EasyICU analysis-only draft scaffold. "
        "Use only the bounded aggregate context. Do not invent patient rows. "
        "Return exactly one JSON object with this shape and no sectioned "
        "manuscript keys: "
        "{\"agent_plan\":{\"steps\":[{\"id\":\"step_001\",\"title\":\"...\","
        "\"evidence_ids\":[\"run_context.json\"]}]},"
        "\"manuscript_draft\":{\"claims\":[{\"id\":\"claim_001\","
        "\"text\":\"...\",\"evidence_ids\":[\"cohort_summary.json\"]}],"
        "\"sentences\":[{\"id\":\"sentence_001\",\"text\":\"...\","
        "\"evidence_ids\":[\"quality_gate.json\"]}]}}. "
        "agent_plan must be an object, not an array. Do not return title, "
        "abstract, introduction, methods, or results sections. Every claim "
        "and sentence must include evidence_ids drawn only from "
        "valid_evidence_ids."
    )
    user = json.dumps(bounded_context, ensure_ascii=False, sort_keys=True)
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "easyicu_policy": {
            "provider": provider,
            "bounded_aggregate_snapshot_only": True,
            "patient_rows_excluded": True,
            "allowed_evidence_ids": valid_evidence,
            "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
            "max_output_tokens": max_output_tokens,
            "json_format_style": json_format_style,
        },
    }
    schema = _agent_payload_json_schema(valid_evidence)
    if json_format_style == "responses":
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "easyicu_agent_run",
                "schema": schema,
                "strict": True,
            }
        }
    elif json_format_style == "both":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "easyicu_agent_run",
                "schema": schema,
                "strict": True,
            },
        }
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "easyicu_agent_run",
                "schema": schema,
                "strict": True,
            }
        }
    else:
        request["response_format"] = {"type": "json_object"}
    return request


def _agent_payload_json_schema(valid_evidence: List[str]) -> Dict[str, Any]:
    evidence_ids = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string", "enum": valid_evidence},
    }
    evidence_bound_record = {
        "type": "object",
        "additionalProperties": True,
        "required": ["id", "text", "evidence_ids"],
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "evidence_ids": evidence_ids,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["agent_plan", "manuscript_draft"],
        "properties": {
            "agent_plan": {
                "type": "object",
                "additionalProperties": True,
                "required": ["steps"],
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                },
            },
            "manuscript_draft": {
                "type": "object",
                "additionalProperties": True,
                "required": ["claims", "sentences"],
                "properties": {
                    "claims": {"type": "array", "items": evidence_bound_record},
                    "sentences": {"type": "array", "items": evidence_bound_record},
                },
            },
        },
    }


def _post_chat_completion(
    *,
    url: str,
    request: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
) -> Dict[str, Any]:
    import requests

    safe_request = {k: v for k, v in request.items() if k != "easyicu_policy"}
    response = requests.post(url, json=safe_request, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ProviderAdapterError({"error": "external_provider_response_not_object"})
    return data


def _coerce_provider_payload(
    response: Dict[str, Any],
    *,
    run_id: str,
    study_id: str,
    question: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    if "agent_plan" in response and "manuscript_draft" in response:
        payload = response
    else:
        content = _extract_message_content(response)
        payload = _parse_json_object(content)
    if not isinstance(payload, dict):
        raise ProviderAdapterError({"error": "external_provider_payload_not_object"})
    plan = payload.get("agent_plan")
    draft = payload.get("manuscript_draft")
    if isinstance(plan, list):
        plan = {"steps": plan}
    if not isinstance(plan, dict) or not isinstance(draft, dict):
        raise ProviderAdapterError({"error": "external_provider_payload_missing_artifacts"})
    plan.setdefault("run_id", run_id)
    plan.setdefault("study_id", study_id)
    plan.setdefault("execution", "external_provider_scaffold")
    draft.setdefault("run_id", run_id)
    draft.setdefault("study_id", study_id)
    draft.setdefault("question", question)
    draft.setdefault("status", "locked_until_human_signoff")
    draft.setdefault("claims", [])
    draft.setdefault("sentences", [])
    return {"agent_plan": plan, "manuscript_draft": draft}


def _extract_message_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderAdapterError({"error": "external_provider_response_missing_choices"})
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderAdapterError({"error": "external_provider_response_missing_content"})
    return content


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = content.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderAdapterError({
            "error": "external_provider_response_json_invalid",
            "message": str(exc),
        }) from exc
    if not isinstance(parsed, dict):
        raise ProviderAdapterError({"error": "external_provider_response_json_not_object"})
    return parsed


def _public_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def _normalize_provider(provider: str) -> str:
    return str(provider or "").strip().lower() or "custom"


def _is_offline_provider(provider: str) -> bool:
    return _normalize_provider(provider) in {"mock", "offline", "none", "local", "disabled"}


def _env_token(provider: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_") or "EASYICU_LLM"


def _max_output_tokens(*, environ: Optional[Mapping[str, str]] = None) -> int:
    env, _ = _provider_env(environ=environ)
    raw = str(env.get("EASYICU_LLM_MAX_TOKENS") or "").strip()
    if not raw:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    return max(_MIN_MAX_OUTPUT_TOKENS, min(_ABSOLUTE_MAX_OUTPUT_TOKENS, value))


def _json_format_style(*, environ: Optional[Mapping[str, str]] = None) -> str:
    env, _ = _provider_env(environ=environ)
    value = str(env.get("EASYICU_LLM_JSON_FORMAT_STYLE") or "").strip().lower()
    if value in {"responses", "text"}:
        return "responses"
    if value in {"both", "dual"}:
        return "both"
    return "chat"


def _provider_env(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    source = os.environ if environ is None else environ
    base = {str(k): str(v) for k, v in source.items()}
    if _truthy(base.get("EASYICU_DISABLE_PROVIDER_ENV_FILE")):
        return base, {
            "enabled": False,
            "status": "disabled",
            "present": False,
            "loaded_keys": [],
            "secrets_returned": False,
        }
    configured = str(base.get("EASYICU_LLM_ENV_FILE") or "").strip()
    path = Path(configured).expanduser() if configured else _DEFAULT_PROVIDER_ENV_FILE
    meta: Dict[str, Any] = {
        "enabled": True,
        "status": "missing",
        "present": False,
        "configured": "custom" if configured else "default",
        "loaded_keys": [],
        "secrets_returned": False,
    }
    if not path.exists():
        return base, meta
    meta["present"] = True
    if not path.is_file():
        meta["status"] = "not_file"
        return base, meta
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        meta["status"] = "insecure_permissions"
        return base, meta
    parsed = _parse_env_file(path)
    for key, value in parsed.items():
        base.setdefault(key, value)
    meta["status"] = "loaded"
    meta["loaded_keys"] = sorted(parsed)
    return base, meta


def _parse_env_file(path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            continue
        parsed[key] = _unquote_env_value(value.strip())
    return parsed


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
