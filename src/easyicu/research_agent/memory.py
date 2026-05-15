"""RunMemory — cross-run lessons.

An autonomous research agent should learn from its failures: a
successful run, a buggy script, a spurious finding all become
training data for the next run's planner. Here the storage stays
trivial, inspectable and entirely off the LLM:

* every run writes a small ``run_summary`` to ``<workdir>/.memory/runs/``;
* before planning, the planner is fed a digest of past runs ranked by
  relevance to the current research question;
* the digest is a plain markdown chunk — reviewers can read it and
  compare it to the model's resulting plan.
* repeated run outcomes are distilled into small ``StrategyCard``
  files. These are not executable code; they are reusable planning
  skeletons and guardrails that the planner can inspect.

The memory is *additive only*; it does not delete past lessons. We
prefer a slow accumulation of ICU-specific gotchas over an opaque
self-modifying agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .concept_availability import (
    default_public_databases,
    hypothesis_cross_database_feasibility,
    normalize_database_name,
)
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


@dataclass
class StrategyCard:
    """Reusable procedural knowledge distilled from prior runs.

    The counter / lifecycle fields (``confidence``, ``times_retrieved``,
    ``validation_count``, ``last_validated_at``, ``retired``,
    ``retired_reason``) make the memory layer self-correcting:
    callers bump ``validation_count`` via :meth:`RunMemory.validate_card`
    after a run reused the card without regressions, and mark cards
    retired via :meth:`RunMemory.retire_card` when later evidence
    contradicts the recommendation. Retired cards are excluded from
    default retrieval.
    """

    strategy_id: str
    task_family: str
    trigger_tokens: List[str]
    recommended_plan: List[str]
    guardrails: List[str]
    supporting_run_ids: List[str]
    updated_at: str
    applicable_databases: List[str] = field(default_factory=list)
    contraindicated_databases: List[str] = field(default_factory=list)
    concept_dependencies: List[str] = field(default_factory=list)
    confidence: float = 0.5
    times_retrieved: int = 0
    validation_count: int = 0
    last_validated_at: Optional[str] = None
    retired: bool = False
    retired_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyCard":
        # Drop unknown keys so cards persisted by older builds load cleanly
        # under a newer schema (and vice versa).
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)


@dataclass
class MemoryScoreBreakdown:
    """Transparent score record for one StrategyCard retrieval decision."""

    overlap: float = 0.0
    support_bonus: float = 0.0
    dependency_bonus: float = 0.0
    database_bonus: float = 0.0
    outcome_bonus: float = 0.0
    confidence_bonus: float = 0.0
    validation_bonus: float = 0.0
    retired_penalty: float = 0.0
    total: float = 0.0


@dataclass
class MemoryRetrievalAuditEntry:
    """One line of a retrieval-decision audit log."""

    strategy_id: str
    task_family: str
    score: MemoryScoreBreakdown
    disposition: str  # "selected" | "dropped" | "retired" | "blocked"
    rationale: str


class RunMemory:
    """Append-only persistent memory of past runs."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / ".memory"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self.strategies_dir = self.root / "strategies"
        self.strategies_dir.mkdir(exist_ok=True)

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
        for card in self.distill_strategy_cards(record):
            self._upsert_strategy_card(card)
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

    def all_strategy_cards(self) -> List[StrategyCard]:
        out: List[StrategyCard] = []
        for p in sorted(self.strategies_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append(StrategyCard.from_dict(data))
            except Exception:
                continue
        return out

    def distill_strategy_cards(self, record: RunMemoryRecord) -> List[StrategyCard]:
        """Distill one run record into reusable strategy cards.

        This is deliberately deterministic. It is the "strategy layer" above
        raw run logs, but remains inspectable and additive.
        """
        family = _task_family_for_record(record)
        cards: List[StrategyCard] = []
        blob = " ".join(
            [record.research_question]
            + [f.get("message", "") for f in record.notable_findings]
        ).lower()
        now = datetime.now(timezone.utc).isoformat()

        if _mentions_any(blob, {"sofa", "sofa2", "ordinal", "gcs"}) and _mentions_any(
            blob, {"missing", "zero", "component", "stratum", "strata"}
        ):
            cards.append(
                StrategyCard(
                    strategy_id="ordinal_score_missingness_audit",
                    task_family="ordinal_score_outcome_association",
                    trigger_tokens=[
                        "sofa",
                        "sofa2",
                        "gcs",
                        "ordinal",
                        "mortality",
                        "missingness",
                    ],
                    recommended_plan=[
                        "Audit component availability and score-zero strata "
                        "before association modeling.",
                        "Report score-level outcome rates and denominator "
                        "counts before adjusted estimates.",
                        "Use median/IQR or level distributions for ordinal "
                        "scores; avoid mean-based claims.",
                    ],
                    guardrails=[
                        "A zero score can encode missing components, not "
                        "absence of organ dysfunction.",
                        "Do not let manuscript claims use score strata unless "
                        "the stratum audit is registered evidence.",
                    ],
                    supporting_run_ids=[record.run_id],
                    updated_at=now,
                    applicable_databases=[record.database],
                    concept_dependencies=["sofa2"],
                )
            )

        if _mentions_any(blob, {"aki", "kdigo", "creat", "creatinine"}):
            cards.append(
                StrategyCard(
                    strategy_id="aki_kdigo_window_first",
                    task_family="aki_outcome_association",
                    trigger_tokens=[
                        "aki",
                        "kdigo",
                        "creat",
                        "creatinine",
                        "mortality",
                    ],
                    recommended_plan=[
                        "Define KDIGO stage and creatinine baseline/window "
                        "semantics before modeling.",
                        "Audit urine-output availability separately from creatinine-only staging.",
                        "Report AKI stage distribution and outcome rates by stage.",
                    ],
                    guardrails=[
                        "Creatinine-only AKI can under-detect events when "
                        "urine output is unavailable.",
                        "Do not mix admission AKI and incident AKI windows "
                        "without a temporal note.",
                    ],
                    supporting_run_ids=[record.run_id],
                    updated_at=now,
                    applicable_databases=[record.database],
                    concept_dependencies=["kdigo_aki", "crea", "urine"],
                )
            )

        if _mentions_any(blob, {"auc", "calibration", "brier", "prediction", "model"}):
            cards.append(
                StrategyCard(
                    strategy_id="prediction_model_calibration_bundle",
                    task_family="icu_prediction_model",
                    trigger_tokens=[
                        "prediction",
                        "model",
                        "auc",
                        "calibration",
                        "brier",
                    ],
                    recommended_plan=[
                        "Predefine discrimination, calibration, and clinical "
                        "utility outputs together.",
                        "Validate event count per candidate predictor before "
                        "fitting adjusted models.",
                        "Generate calibration evidence before manuscript performance claims.",
                    ],
                    guardrails=[
                        "AUC alone is insufficient for ICU prediction-model reporting.",
                        "Sparse events require simpler models or feasibility framing.",
                    ],
                    supporting_run_ids=[record.run_id],
                    updated_at=now,
                    applicable_databases=[record.database],
                )
            )

        if not cards and record.error_count == 0:
            cards.append(
                StrategyCard(
                    strategy_id=f"{family}_successful_skeleton",
                    task_family=family,
                    trigger_tokens=sorted(_tokenise(record.research_question))[:12],
                    recommended_plan=[
                        "Start with cohort/variable feasibility, then primary "
                        "analysis, then validator-bound claim writing.",
                        "Keep every table, figure, statistic, and manuscript "
                        "claim bound to evidence ids.",
                    ],
                    guardrails=[
                        "Do not broaden a narrow user question into a full "
                        "manuscript checklist unless requested.",
                    ],
                    supporting_run_ids=[record.run_id],
                    updated_at=now,
                    applicable_databases=[record.database],
                )
            )

        return cards

    def _score_strategy_card(
        self,
        card: StrategyCard,
        *,
        q_tokens: set,
        database: str,
        target_outcome: Optional[str],
    ) -> Tuple[MemoryScoreBreakdown, str]:
        """Compute a transparent score for a single card.

        Returns ``(breakdown, dependency_status)``. The total score is
        already populated on ``breakdown``. ``dependency_status`` is
        ``"blocked"`` when the card's required concepts are unavailable
        on this database; callers should drop blocked cards from
        retrieval.
        """
        card_tokens = set(card.trigger_tokens) | _tokenise(card.task_family)
        breakdown = MemoryScoreBreakdown()
        breakdown.overlap = float(len(q_tokens & card_tokens))
        breakdown.support_bonus = min(2.0, len(card.supporting_run_ids) * 0.25)

        dependency_status = _strategy_dependency_status(card, database)
        if dependency_status == "full":
            breakdown.dependency_bonus = 0.5
        elif dependency_status == "degraded":
            breakdown.dependency_bonus = -0.75

        db = database.lower()
        if db in {d.lower() for d in card.applicable_databases}:
            breakdown.database_bonus = 1.0
        if db in {d.lower() for d in card.contraindicated_databases}:
            breakdown.database_bonus -= 2.0

        if target_outcome and target_outcome.lower() in card_tokens:
            breakdown.outcome_bonus = 0.5

        # Confidence + validation give cards earned trust over time. The
        # confidence bonus is measured *relative to* the 0.5 baseline so
        # that an untouched default-confidence card contributes nothing
        # — only cards whose synthesizer or operator raised confidence
        # above baseline earn a retrieval bump.
        breakdown.confidence_bonus = 0.5 * (float(card.confidence) - 0.5)
        breakdown.validation_bonus = min(1.0, 0.25 * card.validation_count)
        if card.retired:
            breakdown.retired_penalty = -10.0

        breakdown.total = (
            breakdown.overlap
            + breakdown.support_bonus
            + breakdown.dependency_bonus
            + breakdown.database_bonus
            + breakdown.outcome_bonus
            + breakdown.confidence_bonus
            + breakdown.validation_bonus
            + breakdown.retired_penalty
        )
        return breakdown, dependency_status

    def scored_strategy_cards(
        self,
        *,
        research_question: str,
        database: str,
        target_outcome: Optional[str],
        limit: int = 4,
        include_retired: bool = False,
        audit_path: Optional[Path] = None,
    ) -> List[Tuple[StrategyCard, MemoryScoreBreakdown]]:
        """Rank strategy cards with a transparent score breakdown.

        Unlike :meth:`relevant_strategy_cards` (which returns bare cards
        for backward compatibility), this method returns
        ``(card, MemoryScoreBreakdown)`` pairs so callers can audit why
        a card was selected. Pass ``audit_path`` to also persist a JSONL
        audit log of the retrieval decision.
        """
        q_tokens = _tokenise(
            " ".join([research_question, database, target_outcome or ""])
        )
        decisions: List[Tuple[StrategyCard, MemoryScoreBreakdown, str, str]] = []
        for card in self.all_strategy_cards():
            if card.retired and not include_retired:
                decisions.append(
                    (
                        card,
                        MemoryScoreBreakdown(retired_penalty=-10.0, total=-10.0),
                        "retired",
                        f"card retired ({card.retired_reason or 'no reason'})",
                    )
                )
                continue
            breakdown, dependency_status = self._score_strategy_card(
                card,
                q_tokens=q_tokens,
                database=database,
                target_outcome=target_outcome,
            )
            if dependency_status == "blocked":
                decisions.append(
                    (
                        card,
                        breakdown,
                        "blocked",
                        f"required concepts unavailable on {database}",
                    )
                )
                continue
            disposition = "candidate" if breakdown.total > 0.0 else "dropped"
            decisions.append((card, breakdown, disposition, ""))

        # Rank only the candidate set.
        candidates = [d for d in decisions if d[2] == "candidate"]
        candidates.sort(key=lambda d: d[1].total, reverse=True)
        selected = candidates[:limit]
        selected_ids = {c.strategy_id for c, _, _, _ in selected}

        # Finalize dispositions for audit.
        if audit_path is not None:
            audit_entries: List[MemoryRetrievalAuditEntry] = []
            for card, breakdown, disposition, rationale in decisions:
                final_disposition = (
                    "selected"
                    if card.strategy_id in selected_ids
                    else (disposition if disposition != "candidate" else "dropped")
                )
                audit_entries.append(
                    MemoryRetrievalAuditEntry(
                        strategy_id=card.strategy_id,
                        task_family=card.task_family,
                        score=breakdown,
                        disposition=final_disposition,
                        rationale=rationale or f"score={breakdown.total:.3f}",
                    )
                )
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "research_question": research_question,
                            "database": database,
                            "target_outcome": target_outcome,
                            "entries": [
                                {
                                    "strategy_id": e.strategy_id,
                                    "task_family": e.task_family,
                                    "disposition": e.disposition,
                                    "rationale": e.rationale,
                                    "score": e.score.__dict__,
                                }
                                for e in audit_entries
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return [(c, b) for c, b, _, _ in selected]

    def relevant_strategy_cards(
        self,
        *,
        research_question: str,
        database: str,
        target_outcome: Optional[str],
        limit: int = 4,
    ) -> List[StrategyCard]:
        scored = self.scored_strategy_cards(
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
            limit=limit,
        )
        return [card for card, _ in scored]

    def validate_card(self, strategy_id: str) -> Optional[StrategyCard]:
        """Mark a card as validated by another successful retrieval.

        Bumps ``validation_count``, sets ``last_validated_at`` to now,
        rewrites the card file. Returns the updated card or ``None`` if
        the card no longer exists on disk.
        """
        path = self.strategies_dir / f"{strategy_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        card = StrategyCard.from_dict(data)
        card.validation_count += 1
        card.last_validated_at = datetime.now(timezone.utc).isoformat()
        path.write_text(
            json.dumps(card.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return card

    def retire_card(
        self,
        strategy_id: str,
        *,
        reason: str,
    ) -> Optional[StrategyCard]:
        """Retire a card so it is excluded from default retrieval.

        Retirement is reversible — set ``retired`` back to ``False`` by
        editing the JSON on disk. The ``reason`` is preserved so future
        readers can see why the lesson was withdrawn.
        """
        path = self.strategies_dir / f"{strategy_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        card = StrategyCard.from_dict(data)
        card.retired = True
        card.retired_reason = reason
        path.write_text(
            json.dumps(card.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return card

    def record_retrieval(self, strategy_ids: Sequence[str]) -> None:
        """Bump ``times_retrieved`` on each card that was just selected.

        Separate from :meth:`validate_card`: retrieval indicates the
        card was *surfaced* to the planner, validation indicates it was
        successfully *applied* without later regressions.
        """
        for sid in strategy_ids:
            path = self.strategies_dir / f"{sid}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            card = StrategyCard.from_dict(data)
            card.times_retrieved += 1
            path.write_text(
                json.dumps(card.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def strategy_digest_for_prompt(
        self,
        *,
        research_question: str,
        database: str,
        target_outcome: Optional[str],
        limit: int = 4,
    ) -> str:
        cards = self.relevant_strategy_cards(
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
            limit=limit,
        )
        if not cards:
            return "(no reusable StrategyCards matched this request)"
        lines = ["Reusable StrategyCards distilled from prior runs:"]
        for card in cards:
            lines.append(
                f"- [{card.strategy_id}] family={card.task_family} "
                f"support={len(card.supporting_run_ids)} run(s)"
            )
            if card.applicable_databases:
                lines.append("    databases: " + ", ".join(card.applicable_databases))
            if card.concept_dependencies:
                lines.append("    concepts: " + ", ".join(card.concept_dependencies))
            for step in card.recommended_plan[:3]:
                lines.append(f"    plan: {step}")
            for guardrail in card.guardrails[:2]:
                lines.append(f"    guardrail: {guardrail}")
        return "\n".join(lines)

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
        strategy_digest = self.strategy_digest_for_prompt(
            research_question=research_question,
            database=database,
            target_outcome=target_outcome,
            limit=3,
        )
        if not strategy_digest.startswith("(no reusable"):
            lines.append("")
            lines.append(strategy_digest)
        return "\n".join(lines)

    def rank_skill_keys(
        self,
        *,
        skill_keys: Sequence[str],
        research_question: str,
        database: str,
        target_outcome: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Advisory meta-planner signal for ClinicalSkill order.

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

    def _upsert_strategy_card(self, card: StrategyCard) -> None:
        path = self.strategies_dir / f"{card.strategy_id}.json"
        if path.exists():
            try:
                existing = StrategyCard.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                run_ids = sorted(
                    set(existing.supporting_run_ids) | set(card.supporting_run_ids)
                )
                merged = StrategyCard(
                    strategy_id=existing.strategy_id,
                    task_family=existing.task_family,
                    trigger_tokens=sorted(
                        set(existing.trigger_tokens) | set(card.trigger_tokens)
                    ),
                    recommended_plan=_merge_unique(
                        existing.recommended_plan,
                        card.recommended_plan,
                    ),
                    guardrails=_merge_unique(existing.guardrails, card.guardrails),
                    supporting_run_ids=run_ids,
                    updated_at=card.updated_at,
                    applicable_databases=_merge_unique(
                        existing.applicable_databases,
                        card.applicable_databases,
                    ),
                    contraindicated_databases=_merge_unique(
                        existing.contraindicated_databases,
                        card.contraindicated_databases,
                    ),
                    concept_dependencies=_merge_unique(
                        existing.concept_dependencies,
                        card.concept_dependencies,
                    ),
                )
                card = merged
            except Exception:
                pass
        path.write_text(
            json.dumps(card.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _tokenise(text: str) -> set:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text or "") if len(w) > 2}


def _mentions_any(text: str, tokens: set) -> bool:
    hay = _tokenise(text)
    return bool(hay & {str(t).lower() for t in tokens})


def _task_family_for_record(record: RunMemoryRecord) -> str:
    blob = " ".join(
        [record.research_question, record.target_outcome or ""]
        + [f.get("message", "") for f in record.notable_findings]
    ).lower()
    if _mentions_any(blob, {"sofa", "sofa2", "gcs", "ordinal"}):
        return "ordinal_score_outcome_association"
    if _mentions_any(blob, {"aki", "kdigo", "creat", "creatinine"}):
        return "aki_outcome_association"
    if _mentions_any(blob, {"prediction", "auc", "calibration", "model"}):
        return "icu_prediction_model"
    if _mentions_any(blob, {"mortality", "death"}):
        return "icu_mortality_association"
    return "generic_icu_analysis"


def _merge_unique(left: Sequence[str], right: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in list(left) + list(right):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _strategy_dependency_status(card: StrategyCard, database: str) -> str:
    """Return full/degraded/blocked for a card's concept dependencies."""

    if not card.concept_dependencies:
        return "none"
    db = normalize_database_name(database)
    if db not in set(default_public_databases()):
        return "unknown"
    try:
        summary = hypothesis_cross_database_feasibility(
            concepts=card.concept_dependencies,
            databases=[db],
        )
        statuses = summary.get("cross_database_feasibility", {})
        status = statuses.get(db)
        if status in {"full", "degraded", "blocked"}:
            return str(status)
    except Exception:
        return "unknown"
    return "unknown"


__all__ = ["RunMemory", "RunMemoryRecord", "StrategyCard"]
