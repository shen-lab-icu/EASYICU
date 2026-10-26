"""Publish the workflow demo's run files from its registered source run.

Usage: python tools/build_demo_run_files.py WRAPPER_RUN_DIR

The demo conversation opens the same run files a researcher's own run offers
(run_context.json, result_tables.json, ...), read by the same views. This tool
copies those files from the demo's registered wrapper run, verifies each one
against the run's evidence ledger where the ledger lists it, removes the
Provider credential fingerprint, refuses any file that carries a host path,
and writes them with an index of source and published digests under
``assets/demo/run-files/``. It reads nothing else from the run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/easyicu/webserver/static/assets/demo/run-files"
RUN_FILES = (
    "run_context.json", "cohort_summary.json", "quality_gate.json", "agent_plan.json",
    "literature_evidence.json", "scientific_plan_review.json", "scientific_readiness.json",
    "manuscript_draft.json", "figure_gallery.json", "result_tables.json",
    "source_run_manifest.json", "evidence_ledger.json",
)
#: Fields that identify the operator's Provider credential, not the run.
DROPPED = ("credential_fingerprint",)
HOST_PATH = re.compile(r"(?:/Users/|/Volumes/|/private/|/home/|[A-Za-z]:\\\\|\.easyicu/)")


def _drop(value: Any, dropped: list[str], where: str = "") -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in DROPPED:
                dropped.append(f"{where}{key}")
                continue
            out[key] = _drop(item, dropped, f"{where}{key}.")
        return out
    if isinstance(value, list):
        return [_drop(item, dropped, where) for item in value]
    return value


def publish(run_dir: Path) -> dict[str, dict[str, Any]]:
    ledger = json.loads((run_dir / "evidence_ledger.json").read_text(encoding="utf-8"))
    registered = {row["name"]: row["sha256"] for row in ledger.get("artifacts", []) if row.get("sha256")}
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, Any]] = {}
    for name in RUN_FILES:
        raw = (run_dir / name).read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        if name in registered and registered[name] != source_sha:
            raise SystemExit(f"{name} does not match its evidence-ledger digest")
        if HOST_PATH.search(raw.decode("utf-8")):
            raise SystemExit(f"{name} carries a host path")
        dropped: list[str] = []
        payload = _drop(json.loads(raw), dropped)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        (OUT / name).write_text(text, encoding="utf-8")
        index[name] = {
            "source_sha256": source_sha,
            "ledger_verified": name in registered,
            "published_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "dropped_fields": dropped,
        }
    (OUT / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    print(json.dumps(publish(Path(sys.argv[1])), indent=2))
