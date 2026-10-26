"""Small metadata-only fixtures for the trust/help API owners."""

import json

from easyicu.research_agent.authority.provider_hard_stop import PROVIDER_HARD_STOP_SCHEMA
from easyicu.research_agent.canonical_json import canonical_sha256


def ledger(path, calls, *, state="failed"):
    payload = {"schema_version": PROVIDER_HARD_STOP_SCHEMA,
               "tasks": [{"task_id": "synthetic", "status": state, "calls": calls}]}
    payload["sha256"] = canonical_sha256(payload, trailing_newline=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def call(**changes):
    return {"role": "planner", "state": "failed", "prompt_payload_bytes": 42,
            "accounted_tokens": 50, "accounted_estimated_cost_usd": 0.001,
            "provider": "openrouter", "data_classes": ["text", "schema"],
            "redaction_status": "not_recorded", **changes}
