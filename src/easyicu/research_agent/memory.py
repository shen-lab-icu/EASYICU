"""RunMemory — cross-run lessons (HealthFlow-inspired).

HealthFlow [1] argues that an autonomous research agent should
*learn from its failures*: a successful run, a buggy script, a
spurious finding all become training data for the next run's
planner. We adopt the same idea but make the storage trivial,
inspectable and entirely off the LLM:

* every run writes a small ``run_summary`` to ``<workdir>/.memory/runs/``;
* before planning, the planner is fed a digest of past runs ranked by
  relevance to the current research question;
* the digest is a plain markdown chunk — reviewers can read it and
  compare it to the model's resulting plan.

The memory is *additive only*; it does not delete past lessons. We
prefer a slow accumulation of ICU-specific gotchas over an opaque
self-modifying agent.

References
----------
[1] HealthFlow: A Self-Evolving AI Agent with Meta-Planning for
    Autonomous Healthcare Research.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import ValidationFinding


@dataclass
class RunMemoryRecord:
    run_id: str
    research_question: str
    database: str
    target_outcome: Optional[str]
    findings_count: int
    error_count: int
    warning_count: int
    notable_findings: List[Dict[str, Any]]
    finished_at: str
    workdir: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMemoryRecord":
        return cls(**data)


class RunMemory:
    """Append-only persistent memory of past runs."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / ".memory"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------

    def record(
        self,
        *,
        run_id: str,
        research_question: str,
        database: str,
        target_outcome: Optional[str],
        findings: Iterable[ValidationFinding],
        workdir: Path,
    ) -> RunMemoryRecord:
        findings_list = list(findings)
        notable: List[Dict[str, Any]] = []
        for f in findings_list:
            if f.severity in {"warning", "error"}:
                notable.append({
                    "validator": f.validator,
                    "severity": f.severity,
                    "message": f.message,
                })
        record = RunMemoryRecord(
            run_id=run_id,
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
            findings_count=len(findings_list),
            error_count=sum(1 for f in findings_list if f.severity == "error"),
            warning_count=sum(1 for f in findings_list if f.severity == "warning"),
            notable_findings=notable[:20],
            finished_at=datetime.now(timezone.utc).isoformat(),
            workdir=str(workdir),
        )
        path = self.runs_dir / f"{run_id}.json"
        path.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return record

    # ------------------------------------------------------------------

    def all_records(self) -> List[RunMemoryRecord]:
        out: List[RunMemoryRecord] = []
        for p in sorted(self.runs_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append(RunMemoryRecord.from_dict(data))
            except Exception:
                continue
        return out

    def relevant_to(
        self,
        *,
        research_question: str,
        database: str,
        target_outcome: Optional[str],
        limit: int = 5,
    ) -> List[RunMemoryRecord]:
        """Crude bag-of-words ranking over past run summaries."""
        q_tokens = _tokenise(research_question)

        def score(rec: RunMemoryRecord) -> float:
            s = 0.0
            if rec.database == database:
                s += 1.0
            if target_outcome and rec.target_outcome == target_outcome:
                s += 1.0
            r_tokens = _tokenise(rec.research_question)
            overlap = len(q_tokens & r_tokens)
            s += float(overlap) * 0.5
            # Recency bonus: newer runs rank higher when otherwise tied.
            try:
                ts = datetime.fromisoformat(rec.finished_at)
                age_days = max(
                    0.0,
                    (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0,
                )
                s += 1.0 / (1.0 + age_days * 0.1)
            except Exception:
                pass
            return s

        ranked = sorted(self.all_records(), key=score, reverse=True)
        return ranked[:limit]

    # ------------------------------------------------------------------

    def digest_for_prompt(
        self,
        *,
        research_question: str,
        database: str,
        target_outcome: Optional[str],
        limit: int = 5,
    ) -> str:
        """Render a short, model-readable digest of past lessons."""
        relevant = self.relevant_to(
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
            limit=limit,
        )
        if not relevant:
            return "(no prior runs recorded; this is the first analysis)"
        lines: List[str] = ["Past runs and their notable findings (most relevant first):"]
        for r in relevant:
            lines.append(
                f"- [{r.run_id}] db={r.database} outcome={r.target_outcome or '-'}: "
                f"{r.research_question[:120]}"
            )
            for nf in r.notable_findings[:3]:
                lines.append(f"    • {nf['severity']} [{nf['validator']}] {nf['message']}")
        return "\n".join(lines)

    def rank_skill_keys(
        self,
        *,
        skill_keys: Sequence[str],
        research_question: str,
        database: str,
        target_outcome: Optional[str],
    ) -> List[Tuple[str, float]]:
        """HealthFlow-style meta-planner signal for ClinicalSkill order.

        Scores skills by how often relevant past runs mention the skill
        key or its tokens, lightly penalising runs that ended with
        errors. The ranking is advisory: the pipeline feeds it to the
        planner but never auto-selects a skill without user intent.
        """
        relevant = self.relevant_to(
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
            limit=20,
        )
        q_tokens = _tokenise(research_question)
        out: List[Tuple[str, float]] = []
        for key in skill_keys:
            key_tokens = _tokenise(key.replace("_", " "))
            score = 0.0
            if key_tokens & q_tokens:
                score += 1.5
            for rec in relevant:
                blob = " ".join([
                    rec.research_question,
                    " ".join(f.get("message", "") for f in rec.notable_findings),
                ])
                toks = _tokenise(blob)
                if key_tokens & toks:
                    score += 1.0
                score += max(0.0, rec.warning_count * 0.05 - rec.error_count * 0.2)
            out.append((key, round(score, 3)))
        return sorted(out, key=lambda x: x[1], reverse=True)

    def meta_planner_digest(
        self,
        *,
        skill_keys: Sequence[str],
        research_question: str,
        database: str,
        target_outcome: Optional[str],
    ) -> str:
        ranking = self.rank_skill_keys(
            skill_keys=skill_keys,
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
        )
        if not ranking:
            return "(no ClinicalSkill registry available for meta-planning)"
        bits = [f"{key}={score:.2f}" for key, score in ranking[:8]]
        return (
            "Meta-planner skill ranking from RunMemory (advisory only): "
            + ", ".join(bits)
        )


def _tokenise(text: str) -> set:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text or "") if len(w) > 2}


__all__ = ["RunMemory", "RunMemoryRecord"]
