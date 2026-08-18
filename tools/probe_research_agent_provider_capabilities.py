#!/usr/bin/env python3
"""Probe an external provider with two synthetic, non-clinical requests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any, Callable, Mapping


REPORT_SCHEMA = "easyicu.provider_capability_probe/1"
REPORT_FILENAME = "provider_capability_probe.json"
PROBE_CALL_CAP = 2
_MAX_ENV_FILE_BYTES = 64 * 1024


def _bootstrap() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (repo_root, repo_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


def _parse_env_file(path: Path) -> dict[str, str]:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("env file must be a regular non-symlink file")
    metadata = candidate.stat()
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("env file permissions must be 0600 or stricter")
    if metadata.st_size > _MAX_ENV_FILE_BYTES:
        raise ValueError("env file is too large")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(
        candidate.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"invalid env assignment at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise ValueError(f"invalid env key at line {line_number}")
        if key in values:
            raise ValueError(f"duplicate env key at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _strict_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key {key!r}")
            payload[key] = value
        return payload

    payload = json.loads(
        cleaned,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value!r}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    if payload != {"status": "ready", "value": 7}:
        raise ValueError("response did not preserve the synthetic contract")
    return payload


def _safe_usage(value: object) -> dict[str, int]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = source.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def _safe_actual_model(value: object) -> str | None:
    source = value if isinstance(value, Mapping) else {}
    actual_model = source.get("actual_model")
    if not isinstance(actual_model, str):
        return None
    normalized = actual_model.strip()
    if not normalized or len(normalized) > 256 or not normalized.isprintable():
        return None
    return normalized


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError("unsafe capability report destination")
    # ``absolute`` intentionally does not dereference the final path component.
    # Resolving first would turn an existing symlink into its target and defeat
    # the fail-closed check above.
    target = candidate.absolute()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise ValueError("unsafe capability report destination")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _probe_one(
    client: Any,
    *,
    max_tokens: int,
    structured_output: object = None,
) -> dict[str, Any]:
    from easyicu.research_agent.providers.factory import authorized_complete
    from easyicu.research_agent.providers.llm import safe_provider_finish_reason
    from easyicu.research_agent.providers.protocol import LLMMessage

    messages = (
        LLMMessage(
            role="system",
            content=(
                "Return JSON only. This is a synthetic software compatibility "
                "probe with no clinical or patient data."
            ),
        ),
        LLMMessage(
            role="user",
            content='Return exactly {"status":"ready","value":7}.',
        ),
    )
    started = time.monotonic()
    complete_with_usage = getattr(client, "complete_with_usage", None)
    if callable(complete_with_usage):
        text, usage = complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            structured_output=structured_output,
        )
    else:
        text = authorized_complete(
            client,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            structured_output=structured_output,
        )
        usage = {}
    latency = time.monotonic() - started
    _strict_json_object(text)
    result = {
        "status": "passed",
        "response_chars": len(text),
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "usage": _safe_usage(usage),
        "finish_reason": safe_provider_finish_reason(
            getattr(client, "last_finish_reason", None)
        ),
        "latency_seconds": latency,
        "transport_attempts": max(
            1,
            int(getattr(client, "last_transport_attempts", 0) or 0),
        ),
    }
    actual_model = _safe_actual_model(usage)
    if actual_model:
        result["actual_model"] = actual_model
    return result


def run_capability_probe(
    *,
    provider: str,
    model: str | None,
    environment: Mapping[str, str],
    output_path: Path,
    request_timeout: float = 180.0,
    max_tokens: int = 20_000,
    external_llm_opt_in: bool = False,
    client_factory: Callable[..., Any] | None = None,
    account_client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run at most two calls and persist no prompt, response, or credential."""

    _bootstrap()
    from easyicu import ai_optin
    from easyicu.research_agent.providers import (
        ANTHROPIC_MESSAGES,
        SUPPORTED_CLI_ACCOUNT_NAMES,
        SUPPORTED_PROVIDER_NAMES,
        build_provider_client,
        cli_account_profile,
        provider_profile,
        resolve_provider_base_url,
    )
    from easyicu.research_agent.providers.llm import build_llm_client
    from easyicu.research_agent.providers.protocol import (
        StructuredOutputCapabilityError,
        StructuredOutputRequest,
    )

    normalized_provider = str(provider or "").strip().lower()
    api_profile = normalized_provider in SUPPORTED_PROVIDER_NAMES
    account_profile = (
        cli_account_profile(normalized_provider)
        if normalized_provider in SUPPORTED_CLI_ACCOUNT_NAMES
        else None
    )
    api_definition = provider_profile(normalized_provider) if api_profile else None
    native_anthropic = bool(
        api_definition is not None
        and api_definition.transport == ANTHROPIC_MESSAGES
    )
    if not api_profile and account_profile is None:
        raise ValueError(f"unsupported provider: {provider!r}")
    normalized_model = str(model or "").strip()
    if account_profile is not None and not normalized_model:
        _source, normalized_model = account_profile.model(environment)
        normalized_model = normalized_model or "cli-default"
    if api_profile and not normalized_model:
        raise ValueError("model is required")
    if max_tokens < 128 or max_tokens > 100_000:
        raise ValueError("max_tokens must be between 128 and 100000")
    ai_optin.check_external_llm_opt_in(
        normalized_provider,
        ai_enabled=bool(external_llm_opt_in),
        language="en",
    )
    safe_environment = {str(key): str(value) for key, value in environment.items()}
    safe_environment["EASYICU_ALLOW_EXTERNAL_LLM"] = "1"
    builder = client_factory or build_provider_client
    account_builder = account_client_factory or build_llm_client

    def make_client(*, strict: bool, json_object_mode: bool = False) -> Any:
        if account_profile is not None:
            selection = account_builder(
                prefer=normalized_provider,
                model=None if normalized_model == "cli-default" else normalized_model,
                allow_mock=False,
                ladder=[normalized_provider],
                request_timeout=float(request_timeout),
                environment=safe_environment,
            )
            return getattr(selection, "client", selection)
        return builder(
            provider=normalized_provider,
            model=normalized_model,
            request_timeout=float(request_timeout),
            title="EasyICU provider capability probe",
            environment=safe_environment,
            max_retries=0,
            stream_enabled=False,
            supports_strict_json_schema=strict,
            extra_body=(
                {"response_format": {"type": "json_object"}}
                if json_object_mode and not native_anthropic
                else None
            ),
            allow_environment_overrides=False,
        )

    plain: dict[str, Any]
    try:
        plain = _probe_one(
            make_client(
                strict=False,
                json_object_mode=account_profile is None and not native_anthropic,
            ),
            max_tokens=max_tokens,
        )
        plain["transport_mode"] = (
            "prompted_json"
            if account_profile is not None or native_anthropic
            else "json_object"
        )
    except Exception as exc:
        plain = {
            "status": "failed",
            "transport_mode": (
                "prompted_json"
                if account_profile is not None or native_anthropic
                else "json_object"
            ),
            "error_type": type(exc).__name__,
        }

    strict_request = StructuredOutputRequest.from_schema(
        name="easyicu_provider_probe",
        schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ready"]},
                "value": {"type": "integer", "enum": [7]},
            },
            "required": ["status", "value"],
            "additionalProperties": False,
        },
    )
    strict: dict[str, Any]
    try:
        strict = _probe_one(
            make_client(strict=True),
            max_tokens=max_tokens,
            structured_output=strict_request,
        )
    except StructuredOutputCapabilityError as exc:
        strict = {"status": "unsupported", "error_type": type(exc).__name__}
    except Exception as exc:
        strict = {"status": "failed", "error_type": type(exc).__name__}

    endpoint_identity = (
        account_profile.endpoint_identity
        if account_profile is not None
        else resolve_provider_base_url(
            normalized_provider,
            environment=safe_environment,
        )
    )
    usable = plain.get("status") == "passed"
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "usable" if usable else "unusable",
        "development_only": True,
        "paper_authorized": False,
        "contains_clinical_data": False,
        "contains_patient_data": False,
        "stores_prompt_or_response_text": False,
        "provider": normalized_provider,
        "model": normalized_model,
        "endpoint_sha256": hashlib.sha256(
            endpoint_identity.encode("utf-8")
        ).hexdigest(),
        "transport_family": (
            "account_session"
            if account_profile is not None
            else ("anthropic_messages" if native_anthropic else "openai_compatible_api")
        ),
        "requested_max_tokens": max_tokens,
        "transport_attempt_cap": PROBE_CALL_CAP,
        "host_validated_json": plain,
        "strict_json_schema": strict,
        "capabilities": {
            "host_validated_json": usable,
            "json_object_mode": usable and account_profile is None and not native_anthropic,
            "strict_json_schema": strict.get("status") == "passed",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    _bootstrap()
    from easyicu.research_agent.providers import (
        SUPPORTED_CLI_ACCOUNT_NAMES,
        SUPPORTED_PROVIDER_NAMES,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_CLI_ACCOUNT_NAMES + SUPPORTED_PROVIDER_NAMES),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=20_000)
    parser.add_argument("--external-llm-opt-in", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.external_llm_opt_in:
        raise SystemExit("--external-llm-opt-in is required")
    from easyicu.research_agent.providers import SUPPORTED_CLI_ACCOUNT_NAMES

    is_account_profile = args.provider in SUPPORTED_CLI_ACCOUNT_NAMES
    if not is_account_profile and args.env_file is None:
        raise SystemExit("--env-file is required for API-key providers")
    environment = dict(os.environ) if is_account_profile else {}
    if args.env_file is not None:
        environment.update(_parse_env_file(args.env_file))
    report = run_capability_probe(
        provider=args.provider,
        model=args.model,
        environment=environment,
        output_path=args.out,
        request_timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        external_llm_opt_in=args.external_llm_opt_in,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "provider": report["provider"],
                "model": report["model"],
                "capabilities": report["capabilities"],
                "report": str(args.out.expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "usable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
