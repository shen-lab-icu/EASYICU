"""Digest-bound persistent cache for optional LLM concept review."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from .providers.prompts import PROMPT_PACK_VERSION
from .schema import AnalysisStep, ResearchContext, ValidationFinding


_CACHE_SCHEMA = "easyicu.llm_concept_audit_cache/6"
_AUDIT_POLICY_VERSION = "2026-07-14-step-scoped-v1"
_LOCK = threading.Lock()
_NON_SEMANTIC_CONTEXT_FIELDS = frozenset(
    {"created_at", "generated_at", "updated_at"}
)
_NON_CACHEABLE_ISSUE_CODES = frozenset(
    {
        "llm_concept_audit_provider_failure",
        "llm_concept_audit_response_invalid",
    }
)


def _contains_non_cacheable_failure(
    findings: list[ValidationFinding],
) -> bool:
    """Keep availability/protocol failures out of the semantic audit cache."""

    return any(
        str((finding.detail or {}).get("issue_code") or "")
        in _NON_CACHEABLE_ISSUE_CODES
        for finding in findings
    )


def _semantic_context_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_context_payload(item)
            for key, item in value.items()
            if str(key) not in _NON_SEMANTIC_CONTEXT_FIELDS
        }
    if isinstance(value, list):
        return [_semantic_context_payload(item) for item in value]
    return value


class LLMConceptAuditCache:
    """Cache only LLM findings; deterministic validators always rerun."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / ".cache" / "llm_concept_audit.json"

    @staticmethod
    def key(
        *,
        context: ResearchContext,
        step: AnalysisStep,
        script_text: str,
        audit_prompt: str,
        environment_sha256: str,
        auditor_identity: str,
        authority_bindings: Optional[Mapping[str, Any]] = None,
        validator_implementation_sha256: Optional[str] = None,
    ) -> str:
        normalized_environment = str(environment_sha256).strip()
        normalized_auditor = str(auditor_identity).strip()
        if not normalized_environment or not normalized_auditor:
            raise ValueError(
                "LLM concept-audit cache keys require environment and auditor identity"
            )
        authority_payload = json.dumps(
            authority_bindings or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        payload = {
            "schema": _CACHE_SCHEMA,
            "policy": _AUDIT_POLICY_VERSION,
            "prompt_pack": PROMPT_PACK_VERSION,
            "environment_sha256": normalized_environment,
            "auditor_identity": normalized_auditor,
            "audit_prompt_sha256": hashlib.sha256(
                audit_prompt.encode("utf-8")
            ).hexdigest(),
            "script_sha256": hashlib.sha256(script_text.encode("utf-8")).hexdigest(),
            "authority_bindings_sha256": hashlib.sha256(
                authority_payload
            ).hexdigest(),
            "validator_implementation_sha256": str(
                validator_implementation_sha256 or ""
            ),
            "step": step.model_dump(mode="json"),
            "context": _semantic_context_payload(context.model_dump(mode="json")),
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema": _CACHE_SCHEMA, "entries": {}}
        if payload.get("schema") != _CACHE_SCHEMA or not isinstance(
            payload.get("entries"), dict
        ):
            return {"schema": _CACHE_SCHEMA, "entries": {}}
        return payload

    def get(self, key: str) -> Optional[list[ValidationFinding]]:
        with _LOCK:
            raw = self._read().get("entries", {}).get(key)
        if not isinstance(raw, list):
            return None
        try:
            findings = [ValidationFinding.model_validate(item) for item in raw]
        except Exception:
            return None
        if _contains_non_cacheable_failure(findings):
            return None
        return findings

    def put(self, key: str, findings: list[ValidationFinding]) -> None:
        if _contains_non_cacheable_failure(findings):
            return
        with _LOCK:
            payload = self._read()
            payload["entries"][key] = [
                finding.model_dump(mode="json") for finding in findings
            ]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(
                f".tmp.{os.getpid()}.{threading.get_ident()}"
            )
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)


__all__ = ["LLMConceptAuditCache"]
