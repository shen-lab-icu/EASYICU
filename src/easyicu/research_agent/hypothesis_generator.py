"""Front-door hypothesis generation (O17).

The existing ``HypothesisBlueprintAgent`` refines a *given* research
question into an audit-ready blueprint. This module goes one step
earlier: starting from a cohort + a literature bundle, it produces
a *ranked* list of candidate research questions that are (a) feasible
on the cohort, (b) not already saturated in the supplied literature,
and (c) within EasyICU's ICU concept dictionary.

Design
------

Deterministic first: the pipeline-side helper pairs composite /
ordinal / lab predictors with outcomes in the context and ranks
each pair using three simple signals:

1. ``variable_coverage`` - fraction of complete rows used for ranking.
   When callers provide precomputed pair-level feasibility, this is the
   predictor-outcome ``joint_fraction_complete`` from the outcome-blind
   feasibility probe; otherwise it falls back to the predictor's single
   variable missingness profile.
2. ``literature_saturation_signal`` - supplied-citation density for the
   predictor + outcome pair. Higher values mean the pair is more saturated
   in the supplied citations. Ranking uses ``1 - saturation`` as a gap
   signal. This is a triage signal only, not a novelty claim.
3. ``icu_gate`` - penalty when ICU rules explicitly forbid a
   naive analysis (SOFA mean, GCS mean, etc.).

The LLM role is optional. When a real client is available the
``HypothesisGeneratorAgent`` can refine the top-K candidates into
natural-language research questions; the deterministic ranking is
always emitted so the selection is traceable.

Pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import ConceptDescriptor, ResearchContext, VariableRole


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


LITERATURE_SATURATION_SIGNAL_STATEMENT = (
    "literature_saturation_signal is a ranking signal only, not a novelty "
    "claim; higher values mean the predictor-outcome pair is more saturated "
    "in the supplied citations."
)

_LOW_JOINT_COMPLETENESS_THRESHOLD = 0.70


@dataclass(frozen=True)
class HypothesisFeasibilitySignal:
    """Precomputed, outcome-blind feasibility signal for one candidate pair.

    The hypothesis generator intentionally does not read parquet files or run
    feasibility probes. Callers may compute these values upstream (for example
    with ``real_data_concept_feasibility``) and pass the pair-level signal into
    ``generate_hypotheses``.
    """

    joint_fraction_complete: float
    n_joint_complete: Optional[int] = None
    denominator_n: Optional[int] = None
    source: str = "precomputed"
    note: Optional[str] = None


@dataclass
class HypothesisCandidate:
    hypothesis_family_id: str
    candidate_id: str
    predictor: str
    outcome: str
    question: str
    variable_coverage: float
    literature_saturation_signal: float
    icu_gate: float
    priority_score: float
    rationale: str
    coverage_source: str = "variable_missingness"
    forbidden_note: Optional[str] = None
    feasibility_note: Optional[str] = None
    n_joint_complete: Optional[int] = None
    denominator_n: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "hypothesis_family_id": self.hypothesis_family_id,
            "candidate_id": self.candidate_id,
            "predictor": self.predictor,
            "outcome": self.outcome,
            "question": self.question,
            "variable_coverage": self.variable_coverage,
            "literature_saturation_signal": self.literature_saturation_signal,
            "icu_gate": self.icu_gate,
            "priority_score": self.priority_score,
            "rationale": self.rationale,
            "coverage_source": self.coverage_source,
            "forbidden_note": self.forbidden_note,
            "feasibility_note": self.feasibility_note,
            "n_joint_complete": self.n_joint_complete,
            "denominator_n": self.denominator_n,
        }


@dataclass
class HypothesisGeneratorResult:
    candidates: List[HypothesisCandidate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    hypothesis_family_id: Optional[str] = None
    signal_statement: str = LITERATURE_SATURATION_SIGNAL_STATEMENT

    def summary(self) -> Dict[str, Any]:
        top = self.candidates[0] if self.candidates else None
        return {
            "n_candidates": len(self.candidates),
            "hypothesis_family_id": self.hypothesis_family_id,
            "top": top.to_json() if top else None,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "hypothesis_family_id": self.hypothesis_family_id,
            "signal_statement": self.signal_statement,
            "candidates": [c.to_json() for c in self.candidates],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Hypothesis generator (O17)",
            "",
            f"Candidates ranked: **{len(self.candidates)}**",
            f"Hypothesis family: `{self.hypothesis_family_id or 'n/a'}`",
            "",
            f"Signal statement: {self.signal_statement}",
            "",
            "| Rank | Candidate ID | Predictor | Outcome | Coverage | Literature saturation | ICU gate | Priority | Question |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for idx, c in enumerate(self.candidates, start=1):
            lines.append(
                "| {r} | `{cid}` | {p} | {o} | {cov:.2f} | {sat:.2f} | {gate:.2f} | {pri:.2f} | {q} |".format(
                    r=idx,
                    cid=c.candidate_id,
                    p=c.predictor,
                    o=c.outcome,
                    cov=c.variable_coverage,
                    sat=c.literature_saturation_signal,
                    gate=c.icu_gate,
                    pri=c.priority_score,
                    q=(c.question[:70] + "...") if len(c.question) > 70 else c.question,
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
    VariableRole.VITAL,
    VariableRole.LAB,
    VariableRole.INTERVENTION,
)


def _variable_coverage(v: ConceptDescriptor) -> float:
    if v.missingness is None:
        return 0.5
    return max(0.0, 1.0 - float(v.missingness.fraction_missing))


def _literature_saturation_signal(
    predictor: str,
    outcome: str,
    citations: Sequence[Any],
) -> float:
    """Citation-density signal for the pair; higher means more saturated.

    This is a crude deterministic ranking signal over the supplied citation
    bundle. It must not be presented as a formal novelty assessment.
    """
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
        return 0.0
    # Saturating density: 5 matches -> 0.5, 20 matches -> 0.8.
    return min(0.9, 1.0 - (1.0 / (1.0 + 0.2 * hits)))


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


def _normalize_pair_key(predictor: str, outcome: str) -> Tuple[str, str]:
    return (str(predictor).strip().lower(), str(outcome).strip().lower())


def _coerce_pair_key(raw_key: Any) -> Tuple[str, str]:
    if isinstance(raw_key, tuple) and len(raw_key) == 2:
        return _normalize_pair_key(str(raw_key[0]), str(raw_key[1]))
    if isinstance(raw_key, str):
        for sep in ("|", "::"):
            if sep in raw_key:
                left, right = raw_key.split(sep, 1)
                return _normalize_pair_key(left, right)
    raise ValueError(
        "feasibility_by_pair keys must be (predictor, outcome) tuples or "
        "'predictor|outcome' strings"
    )


def _bounded_fraction(value: Any) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_fraction_complete must be numeric") from exc
    if not math.isfinite(fraction):
        raise ValueError("joint_fraction_complete must be finite")
    return max(0.0, min(1.0, fraction))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("feasibility counts must be integers when provided") from exc


def _coerce_feasibility_signal(value: Any) -> HypothesisFeasibilitySignal:
    if isinstance(value, HypothesisFeasibilitySignal):
        return HypothesisFeasibilitySignal(
            joint_fraction_complete=_bounded_fraction(value.joint_fraction_complete),
            n_joint_complete=value.n_joint_complete,
            denominator_n=value.denominator_n,
            source=value.source,
            note=value.note,
        )
    if isinstance(value, (int, float)):
        return HypothesisFeasibilitySignal(joint_fraction_complete=_bounded_fraction(value))
    if isinstance(value, Mapping):
        joint = value.get("joint_fraction_complete")
        if joint is None:
            raise ValueError("feasibility mapping values require joint_fraction_complete")
        return HypothesisFeasibilitySignal(
            joint_fraction_complete=_bounded_fraction(joint),
            n_joint_complete=_optional_int(value.get("n_joint_complete")),
            denominator_n=_optional_int(value.get("denominator_n")),
            source=str(value.get("source") or "precomputed"),
            note=str(value["note"]) if value.get("note") is not None else None,
        )
    joint = getattr(value, "joint_fraction_complete", None)
    if joint is None:
        raise ValueError("feasibility objects require joint_fraction_complete")
    return HypothesisFeasibilitySignal(
        joint_fraction_complete=_bounded_fraction(joint),
        n_joint_complete=_optional_int(getattr(value, "n_joint_complete", None)),
        denominator_n=_optional_int(getattr(value, "denominator_n", None)),
        source=value.__class__.__name__,
        note=getattr(value, "note", None),
    )


def _normalize_feasibility_by_pair(
    feasibility_by_pair: Optional[Mapping[Any, Any]],
) -> Dict[Tuple[str, str], HypothesisFeasibilitySignal]:
    if not feasibility_by_pair:
        return {}
    out: Dict[Tuple[str, str], HypothesisFeasibilitySignal] = {}
    for raw_key, raw_value in feasibility_by_pair.items():
        out[_coerce_pair_key(raw_key)] = _coerce_feasibility_signal(raw_value)
    return out


def _coerce_saturation_signal(value: Any) -> float:
    if isinstance(value, Mapping):
        if "literature_saturation_signal" in value:
            return _bounded_fraction(value["literature_saturation_signal"])
        if "saturation" in value:
            return _bounded_fraction(value["saturation"])
        if "prior_art_saturation" in value:
            return _bounded_fraction(value["prior_art_saturation"])
    raw = getattr(value, "literature_saturation_signal", value)
    return _bounded_fraction(raw)


def _normalize_saturation_by_pair(
    saturation_by_pair: Optional[Mapping[Any, Any]],
) -> Dict[Tuple[str, str], float]:
    if not saturation_by_pair:
        return {}
    out: Dict[Tuple[str, str], float] = {}
    for raw_key, raw_value in saturation_by_pair.items():
        out[_coerce_pair_key(raw_key)] = _coerce_saturation_signal(raw_value)
    return out


def _default_hypothesis_family_id(
    context: ResearchContext,
    predictors: Sequence[ConceptDescriptor],
    outcomes: Sequence[str],
) -> str:
    payload = {
        "cohort_name": context.cohort.cohort_name,
        "database": context.cohort.database,
        "predictors": sorted(v.name for v in predictors),
        "outcomes": sorted(outcomes),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"hypothesis_family:{digest}"


def _candidate_id(hypothesis_family_id: str, predictor: str, outcome: str) -> str:
    payload = {
        "hypothesis_family_id": hypothesis_family_id,
        "predictor": str(predictor).strip().lower(),
        "outcome": str(outcome).strip().lower(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"candidate:{digest}"


def _feasibility_note(signal: HypothesisFeasibilitySignal) -> Optional[str]:
    parts: List[str] = []
    if signal.joint_fraction_complete < _LOW_JOINT_COMPLETENESS_THRESHOLD:
        parts.append(
            "joint completeness below "
            f"{_LOW_JOINT_COMPLETENESS_THRESHOLD:.0%}: "
            f"{signal.joint_fraction_complete:.0%}"
        )
    if signal.n_joint_complete is not None and signal.denominator_n is not None:
        parts.append(f"joint complete n={signal.n_joint_complete}/{signal.denominator_n}")
    if signal.note:
        parts.append(signal.note)
    if not parts:
        return None
    return "; ".join(parts)


def generate_hypotheses(
    *,
    context: ResearchContext,
    citations: Sequence[Any] = (),
    top_k: int = 5,
    feasibility_by_pair: Optional[Mapping[Any, Any]] = None,
    saturation_by_pair: Optional[Mapping[Any, Any]] = None,
    hypothesis_family_id: Optional[str] = None,
) -> HypothesisGeneratorResult:
    """Rank candidate (predictor, outcome) pairs.

    The scoring is intentionally simple and explainable so reviewers
    can follow why question X was promoted over question Y. Weights:
    coverage × 0.4 + literature_gap × 0.4 + gate × 0.2, where
    ``literature_gap = 1 - literature_saturation_signal``.

    ``feasibility_by_pair`` is an optional precomputed mapping keyed by
    ``(predictor, outcome)``. When provided, ranking uses pair-level
    ``joint_fraction_complete`` from that mapping as ``variable_coverage``.
    ``saturation_by_pair`` is an optional precomputed prior-art mapping keyed
    the same way. When provided, ranking uses that value as
    ``literature_saturation_signal`` instead of counting substrings in the
    supplied citations.
    This function never reads cohort files or launches analysis runs.
    """
    predictors = _pick_predictor_candidates(context)
    outcomes = _pick_outcome_candidates(context)
    family_id = hypothesis_family_id or _default_hypothesis_family_id(
        context,
        predictors,
        outcomes,
    )
    pair_feasibility = _normalize_feasibility_by_pair(feasibility_by_pair)
    pair_saturation = _normalize_saturation_by_pair(saturation_by_pair)
    candidates: List[HypothesisCandidate] = []
    notes: List[str] = [LITERATURE_SATURATION_SIGNAL_STATEMENT]
    if not outcomes:
        notes.append("No outcome variables in context; cannot propose hypotheses.")
        return HypothesisGeneratorResult(
            candidates=[],
            notes=notes,
            hypothesis_family_id=family_id,
        )
    if not predictors:
        notes.append(
            "No composite/ordinal/vital/lab/intervention predictors in context."
        )
        return HypothesisGeneratorResult(
            candidates=[],
            notes=notes,
            hypothesis_family_id=family_id,
        )
    for pred in predictors:
        fallback_cov = _variable_coverage(pred)
        gate, forbidden = _icu_gate(pred)
        for outcome in outcomes:
            if outcome == pred.name:
                continue
            pair_key = _normalize_pair_key(pred.name, outcome)
            feasibility_signal = pair_feasibility.get(pair_key)
            if feasibility_signal is not None:
                cov = feasibility_signal.joint_fraction_complete
                coverage_source = "pair_joint_feasibility"
                feasibility_note = _feasibility_note(feasibility_signal)
                n_joint_complete = feasibility_signal.n_joint_complete
                denominator_n = feasibility_signal.denominator_n
            else:
                cov = fallback_cov
                coverage_source = "variable_missingness"
                feasibility_note = None
                n_joint_complete = None
                denominator_n = None
            saturation = pair_saturation.get(pair_key)
            if saturation is None:
                saturation = _literature_saturation_signal(pred.name, outcome, citations)
            literature_gap = 1.0 - saturation
            priority = 0.4 * cov + 0.4 * literature_gap + 0.2 * gate
            rationale = (
                f"coverage={cov:.2f} ({coverage_source}), "
                f"literature_gap={literature_gap:.2f}, "
                f"literature_saturation_signal={saturation:.2f}, "
                f"icu_gate={gate:.2f}."
            )
            candidates.append(
                HypothesisCandidate(
                    hypothesis_family_id=family_id,
                    candidate_id=_candidate_id(family_id, pred.name, outcome),
                    predictor=pred.name,
                    outcome=outcome,
                    question=_template_question(pred.name, outcome),
                    variable_coverage=cov,
                    literature_saturation_signal=saturation,
                    icu_gate=gate,
                    priority_score=priority,
                    rationale=rationale,
                    coverage_source=coverage_source,
                    forbidden_note=forbidden,
                    feasibility_note=feasibility_note,
                    n_joint_complete=n_joint_complete,
                    denominator_n=denominator_n,
                )
            )
    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return HypothesisGeneratorResult(
        candidates=candidates[:top_k],
        notes=notes,
        hypothesis_family_id=family_id,
    )


__all__ = [
    "HypothesisCandidate",
    "HypothesisFeasibilitySignal",
    "HypothesisGeneratorResult",
    "LITERATURE_SATURATION_SIGNAL_STATEMENT",
    "generate_hypotheses",
]
