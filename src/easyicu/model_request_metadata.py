"""Closed, payload-free projection shared by request receipts and help downloads.

These receipts describe attempts, not confirmed network delivery. Byte counts
exclude provider framing. Missing historical metadata stays explicitly unknown;
neither a small byte count nor a successful request proves redaction occurred.
"""

from collections.abc import Mapping
from typing import Any

PROVIDERS = frozenset({"openai", "openrouter", "anthropic", "openai-codex", "codex",
                       "ollama", "deepseek", "qwen", "gemini", "mock", "replay", "local", "custom", "claude"})
PURPOSES = frozenset({"planner", "coder", "analyzer", "writer", "replanner", "reviewer",
                      "auditor", "concept_auditor", "vlm", "visual_qa", "figure_review", "conversation",
                      "provider_recovery_probe", "literature", "idea_mining", "repair", "legacy_model_roster_migration"})
STATES = frozenset({"pending", "running", "paused", "completed", "failed", "cancelled",
                    "in_progress", "completed_usage_unreported", "budget_exhausted",
                    "batch_canary_blocked", "mixed", "unknown", "failed_usage_unknown", "completed_usage_overflow"})
DATA_CLASSES = frozenset({"aggregate", "schema", "text", "code"})


def closed(value: Any, choices: frozenset[str], default: str = "unknown") -> str:
    return value if isinstance(value, str) and value in choices else default


def provider_name(manifest: Mapping[str, Any]) -> str:
    providers = {closed(row.get("provider"), PROVIDERS) for row in manifest["clients"]}
    return next(iter(providers)) if len(providers) == 1 else "unknown"


def project_request(row: Mapping[str, Any], *, sequence: int) -> dict[str, Any]:
    """Never copy payloads, model names, free-form roles, URLs or aggregate values."""
    size = row.get("byte_size", row.get("prompt_payload_bytes"))
    classes = row.get("data_classes")
    return {
        "sequence": sequence,
        "purpose": closed(row.get("purpose", row.get("role")), PURPOSES),
        "provider": closed(row.get("provider"), PROVIDERS),
        "data_classes": list(dict.fromkeys(
            item for item in classes if isinstance(item, str) and item in DATA_CLASSES
        )) if isinstance(classes, list) else [],
        "data_class_basis": "recorded_container_types",
        "byte_size": size if type(size) is int and 0 <= size < 2**53 else None,
        "byte_size_basis": closed(row.get("byte_size_basis"), frozenset({
            "context_json_utf8", "prompt_content_utf8_and_schema",
        }), "prompt_content_utf8_and_schema"),
        "redaction_status": "not_recorded",
        "payload_status": "withheld",
        "state": closed(row.get("state"), STATES),
        "failed": bool(row.get("error_type")) or row.get("state") in ("failed", "failed_usage_unknown"),
        "delivery_status": "not_confirmed",
    }
