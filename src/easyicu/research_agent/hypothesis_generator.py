"""Front-door hypothesis generation (O17).

The existing ``HypothesisBlueprintAgent`` refines a *given* research
question into an audit-ready blueprint. This module goes one step
earlier: starting from a cohort + a literature bundle, it produces
a *ranked* list of candidate research questions that are (a) feasible
on the cohort, (b) not obviously answered by the literature already,
and (c) within EasyICU's ICU concept dictionary.

Design
------

Deterministic first: the pipeline-side helper pairs composite /
ordinal / lab predictors with outcomes in the context and ranks
each pair using three simple signals:

1. ``variable_coverage`` — fraction of non-null rows in the cohort.
2. ``literature_novelty`` — 1.0 if no curated/PubMed citation
   mentions the predictor + outcome pair, else proportionally less.
3. ``icu_gate`` — penalty when ICU rules explicitly forbid a
   naive analysis (SOFA mean, GCS mean, etc.).

The LLM role is optional. When a real client is available the
``HypothesisGeneratorAgent`` can refine the top-K candidates into
natural-language research questions; the deterministic ranking is
always emitted so the selection is traceable.

Pure stdlib + numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import ConceptDescriptor, ResearchContext, VariableRole


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HypothesisCandidate:
    predictor: str
    outcome: str
    question: str
    variable_coverage: float
    literature_novelty: float
    icu_gate: float
    priority_score: float
    rationale: str
    forbidden_note: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor,
            "outcome": self.outcome,
            "question": self.question,
            "variable_coverage": self.variable_coverage,
            "literature_novelty": self.literature_novelty,
            "icu_gate": self.icu_gate,
            "priority_score": self.priority_score,
            "rationale": self.rationale,
            "forbidden_note": self.forbidden_note,
        }


@dataclass
class HypothesisGeneratorResult:
    candidates: List[HypothesisCandidate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        top = self.candidates[0] if self.candidates else None
        return {
            "n_candidates": len(self.candidates),
            "top": top.to_json() if top else None,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "candidates": [c.to_json() for c in self.candidates],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Hypothesis generator (O17)",
            "",
            f"Candidates ranked: **{len(self.candidates)}**",
            "",
            "| Rank | Predictor | Outcome | Coverage | Novelty | ICU gate | Priority | Question |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for idx, c in enumerate(self.candidates, start=1):
            lines.append(
                "| {r} | {p} | {o} | {cov:.2f} | {nov:.2f} | {gate:.2f} | {pri:.2f} | {q} |".format(
                    r=idx,
                    p=c.predictor,
                    o=c.outcome,
                    cov=c.variable_coverage,
                    nov=c.literature_novelty,
                    gate=c.icu_gate,
                    pri=c.priority_score,
                    q=(c.question[:70] + "…") if len(c.question) > 70 else c.question,
                )
            )
        if self.notes:
            lines += ["", "## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_PREDICTOR_ROLES = (
    VariableRole.COMPOSITE_SCORE,
    VariableRole.ORDINAL_SCORE,
    VariableRole.LAB,
    VariableRole.INTERVENTION,
)


def _variable_coverage(v: ConceptDescriptor) -> float:
    if v.missingness is None:
        return 0.5
    return max(0.0, 1.0 - float(v.missingness.fraction_missing))


def _literature_novelty(predictor: str, outcome: str, citations: Sequence[Any]) -> float:
    """1.0 when no citation mentions the pair; falls linearly as matches grow."""
    predictor_l = predictor.lower()
    outcome_l = outcome.lower()
    hits = 0
    for c in citations:
        title = (getattr(c, "title", "") or "").lower()
        relevance = (getattr(c, "relevance", "") or "").lower()
        if predictor_l in title or predictor_l in relevance:
            if outcome_l in title or outcome_l in relevance:
                hits += 1
    if hits == 0:
        return 1.0
    # saturating novelty: 5 matches -> 0.5, 20 matches -> 0.1
    return max(0.1, 1.0 / (1.0 + 0.2 * hits))


def _icu_gate(predictor: ConceptDescriptor) -> Tuple[float, Optional[str]]:
    if predictor.pitfalls:
        # Hard penalty because SOFA components / GCS / KDIGO stages are
        # famously mishandled; we still rank them but they need a
        # careful hypothesis.
        return 0.5, ", ".join(predictor.pitfalls[:2])
    if predictor.role in {VariableRole.COMPOSITE_SCORE, VariableRole.ORDINAL_SCORE}:
        return 0.75, "ordinal / composite score — confirm aggregation rule"
    return 1.0, None


def _pick_outcome_candidates(context: ResearchContext) -> List[str]:
    declared = context.target_outcome
    outcomes: List[str] = []
    if declared:
        outcomes.append(declared)
    for v in context.variables:
        if v.role == VariableRole.OUTCOME and v.name not in outcomes:
            outcomes.append(v.name)
    return outcomes


def _pick_predictor_candidates(context: ResearchContext) -> List[ConceptDescriptor]:
    return [v for v in context.variables if v.role in _PREDICTOR_ROLES]


def _template_question(predictor: str, outcome: str) -> str:
    return (
        f"Is {predictor} associated with {outcome} among the ICU patients in this cohort?"
    )


def generate_hypotheses(
    *,
    context: ResearchContext,
    citations: Sequence[Any] = (),
    top_k: int = 5,
) -> HypothesisGeneratorResult:
    """Rank candidate (predictor, outcome) pairs.

    The scoring is intentionally simple and explainable so reviewers
    can follow why question X was promoted over question Y. Weights:
    coverage × 0.4 + novelty × 0.4 + gate × 0.2.
    """
    predictors = _pick_predictor_candidates(context)
    outcomes = _pick_outcome_candidates(context)
    candidates: List[HypothesisCandidate] = []
    notes: List[str] = []
    if not outcomes:
        notes.append("No outcome variables in context; cannot propose hypotheses.")
        return HypothesisGeneratorResult(candidates=[], notes=notes)
    if not predictors:
        notes.append(
            "No composite/ordinal/lab/intervention predictors in context."
        )
        return HypothesisGeneratorResult(candidates=[], notes=notes)
    for pred in predictors:
        cov = _variable_coverage(pred)
        gate, forbidden = _icu_gate(pred)
        for outcome in outcomes:
            if outcome == pred.name:
                continue
            nov = _literature_novelty(pred.name, outcome, citations)
            priority = 0.4 * cov + 0.4 * nov + 0.2 * gate
            rationale = (
                f"coverage={cov:.2f}, novelty={nov:.2f}, icu_gate={gate:.2f}."
            )
            candidates.append(
                HypothesisCandidate(
                    predictor=pred.name,
                    outcome=outcome,
                    question=_template_question(pred.name, outcome),
                    variable_coverage=cov,
                    literature_novelty=nov,
                    icu_gate=gate,
                    priority_score=priority,
                    rationale=rationale,
                    forbidden_note=forbidden,
                )
            )
    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return HypothesisGeneratorResult(candidates=candidates[:top_k], notes=notes)


__all__ = [
    "HypothesisCandidate",
    "HypothesisGeneratorResult",
    "generate_hypotheses",
]
