"""Digest-bound persistent cache for optional LLM concept review."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

from .prompts import PROMPT_PACK_VERSION
from .schema import AnalysisStep, ResearchContext, ValidationFinding


_CACHE_SCHEMA = "easyicu.llm_concept_audit_cache/1"
_AUDIT_POLICY_VERSION = "2026-07-14-step-scoped-v1"
_LOCK = threading.Lock()


class LLMConceptAuditCache:
    """Cache only LLM findings; deterministic validators always rerun."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / ".cache" / "llm_concept_audit.json"

    @staticmethod
    def key(
        *, context: ResearchContext, step: AnalysisStep, script_text: str
    ) -> str:
        payload = {
            "schema": _CACHE_SCHEMA,
            "policy": _AUDIT_POLICY_VERSION,
            "prompt_pack": PROMPT_PACK_VERSION,
            "script_sha256": hashlib.sha256(script_text.encode("utf-8")).hexdigest(),
            "step": step.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
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
            return [ValidationFinding.model_validate(item) for item in raw]
        except Exception:
            return None

    def put(self, key: str, findings: list[ValidationFinding]) -> None:
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
