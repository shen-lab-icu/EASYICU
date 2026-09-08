"""Read-only reuse of a failed Writer's outputs after an owner repair.

Every output still crosses current quality, claim, numeric and export gates.
An incomplete replay cannot silently start another provider attempt.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from easyicu.research_agent.reporting.writer_only_migration import WriterOnlyMigrationError


@dataclass
class SavedWriterReplay:
    revision_id: str
    rows: list[tuple[Path, str, dict]]
    cursor: int = 0

    def section(self, *, section_name: str, instruction: str) -> str:
        if self.cursor >= len(self.rows):
            raise WriterOnlyMigrationError(code="WRITER_ONLY_REPLAY_EXHAUSTED", detail="Saved outputs do not cover the current repair.")
        path, digest, row = self.rows[self.cursor]
        if (path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            or row.get("section") != section_name or row.get("instruction") != instruction):
            raise WriterOnlyMigrationError(code="WRITER_ONLY_REPLAY_MISMATCH", detail="Saved Writer output does not match the repair request.")
        self.cursor += 1
        return row["text"]


def load_failed_writer_replay(wrapper: Path, prepared) -> SavedWriterReplay | None:
    root = wrapper / "report_revisions"
    if root.is_symlink() or not root.exists():
        return None
    receipts = sorted(root.glob("*/writer_only_migration_receipt.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in receipts[:100]:
        if path.is_symlink() or path.parent.is_symlink():
            continue
        receipt = json.loads(path.read_text())
        if (receipt.get("status") != "failed"
            or receipt.get("reason_code") != "WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED"
            or receipt.get("source_run_modified") is not False
            or receipt.get("source_hashes") != dict(prepared.source_hashes)
            or receipt.get("source_run_dir") != str(prepared.source_run_dir)):
            continue
        preflight_path = path.parent / "preflight.json"
        if preflight_path.is_symlink():
            continue
        preflight = json.loads(preflight_path.read_text())
        if (preflight.get("migration_draft_sha256") != prepared.migration_draft_sha256
            or preflight.get("migration_draft_path") != (str(prepared.migration_draft_path) if prepared.migration_draft_path else None)
            or preflight.get("writer_evidence_digest_sha256") != hashlib.sha256(prepared.evidence_digest.encode()).hexdigest()):
            continue
        runtime = path.parent / "runtime"
        if runtime.is_symlink():
            continue
        rows = []
        for candidate in sorted(runtime.glob("writer_candidate_*.json")):
            if candidate.is_symlink() or candidate.stat().st_size > 512 * 1024:
                return None
            raw = candidate.read_bytes()
            row = json.loads(raw)
            if not all(isinstance(row.get(k), str) and row[k] for k in ("section", "instruction", "text")):
                return None
            rows.append((candidate, hashlib.sha256(raw).hexdigest(), row))
        if rows and len(rows) == receipt.get("provider_summary", {}).get("n_calls") and len(rows) <= 6:
            return SavedWriterReplay(path.parent.name, rows)
    return None
