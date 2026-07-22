"""Data-foundation agent (L2): the agent picks its own concepts + cohort.

This closes the gap the user flagged: a real EasyICU user provides only a
database export and a question — nobody hand-writes the per-question
concept list or cohort filter. So the *agent* must:

* read the catalog of what is actually available (see :mod:`.catalog`);
* **select** the concepts the question needs — for both the analysis and the
  cohort definition (纳排) — using clinical judgement (the LLM call here);
* have that selection checked for coverage (sufficient? else advise
  re-extraction) and then materialised by the trusted, deterministic
  extractor (:mod:`..cohort.materializer`) into the wide universe the sandbox
  consumes.

The selection (judgement) is the model's; the extraction (execution) stays
deterministic and reproducible. The actual inclusion/exclusion *thresholds*
and preprocessing are applied by the agent's in-sandbox analysis code — here
we only make sure the concepts those steps need are extracted in the first
place.

This works with any provider (DeepSeek, OpenRouter, …): the selection is a
plain plan/JSON call, so there is no per-model integration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .catalog import (
    AvailableCatalog,
    CoverageReport,
    assess_coverage,
    build_available_catalog,
)
from ..providers.protocol import LLMClient, LLMMessage
from ..intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    MaterializedTrajectoryError,
    load_verified_materialized_trajectory_authority,
)

_SELECTION_SYSTEM = (
    "You are the data-foundation step of an ICU research agent. You are given "
    "a research question and a catalog of the concepts that are actually "
    "available in the user's prepared data. Your job is to SELECT which "
    "concepts the analysis will need — covering BOTH the cohort definition "
    "(inclusion/exclusion, e.g. concepts needed to require adults, a first "
    "ICU stay, a minimum length of stay) AND the statistical analysis "
    "(exposure, outcome, and a defensible set of adjustment covariates). "
    "Select only concepts that appear in the catalog. Be parsimonious: do not "
    "request every concept, but do not omit the outcome, the demographics "
    "needed for the cohort, or any concept your stated inclusion/exclusion "
    "criteria depend on. Reply with a single JSON object and nothing else."
)


def _extract_json(raw: str) -> Optional[dict]:
    """Tolerant JSON extraction (fenced block or first {...} object)."""
    text = (raw or "").strip()
    if "```" in text:
        # strip a ```json ... ``` fence
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class ConceptSelection:
    """The agent's chosen concepts + its intended cohort, plus the verdict."""

    selected_concepts: List[str]
    inclusion_exclusion: List[str] = field(default_factory=list)
    rationale: str = ""
    coverage: Optional[CoverageReport] = None
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_concepts": list(self.selected_concepts),
            "inclusion_exclusion": list(self.inclusion_exclusion),
            "rationale": self.rationale,
            "coverage": self.coverage.to_dict() if self.coverage else None,
        }


class DataFoundationAgent:
    """LLM-driven concept selection over an available-data catalog."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def select_concepts(
        self,
        *,
        question: str,
        catalog: AvailableCatalog,
        target_outcome: Optional[str] = None,
    ) -> ConceptSelection:
        user = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            + (
                f"TARGET OUTCOME concept: {target_outcome}\n\n"
                if target_outcome
                else ""
            )
            + catalog.render_for_prompt()
            + '\n\nReturn JSON: {"selected_concepts": [concept_id, ...], '
            '"inclusion_exclusion": ["plain-text criterion", ...], '
            '"rationale": "why these concepts"}. '
            "selected_concepts MUST be a subset of the catalog above."
        )
        raw = self.llm.complete(
            [
                LLMMessage(role="system", content=_SELECTION_SYSTEM),
                LLMMessage(role="user", content=user),
            ],
            # Generous cap: some models are verbose (large rationale) and a
            # truncated completion yields unparseable JSON -> empty selection.
            max_tokens=4096,
            temperature=0.1,
        )
        data = _extract_json(raw) or {}
        selected = [
            str(c) for c in (data.get("selected_concepts") or []) if str(c).strip()
        ]
        incl = [
            str(x) for x in (data.get("inclusion_exclusion") or []) if str(x).strip()
        ]
        rationale = str(data.get("rationale") or "")
        coverage = assess_coverage(selected, catalog)
        return ConceptSelection(
            selected_concepts=selected,
            inclusion_exclusion=incl,
            rationale=rationale,
            coverage=coverage,
            raw_response=raw if isinstance(raw, str) else "",
        )


@dataclass
class AcquisitionResult:
    """Outcome of acquiring a question's universe from a user's export."""

    universe_path: Optional[Path]
    provenance_path: Optional[Path]
    selection: ConceptSelection
    materialized_concepts: List[str]
    coverage: CoverageReport
    blocked: bool = False
    note: str = ""
    # Token usage + estimated USD of the concept-SELECTION LLM call. The
    # selection runs as a pre-sandbox data-foundation stage (like extraction,
    # it needs data access), so its cost is recorded here rather than in the
    # in-sandbox analysis run's cost_summary.json.
    selection_usage: Optional[Dict[str, int]] = None
    selection_cost_usd: Optional[float] = None
    selection_model: Optional[str] = None
    cohort_authority_path: Optional[Path] = None
    cohort_authority_ref: Optional[MaterializedCohortAuthorityRef] = None
    trajectory_path: Optional[Path] = None
    trajectory_provenance_path: Optional[Path] = None
    trajectory_authority_path: Optional[Path] = None
    trajectory_authority_ref: Optional[MaterializedTrajectoryAuthorityRef] = None

    def __post_init__(self) -> None:
        if (self.cohort_authority_path is None) != (self.cohort_authority_ref is None):
            raise MaterializedMetadataError(
                "cohort authority path and reference must be present together"
            )
        if (self.trajectory_path is None) != (self.trajectory_provenance_path is None):
            raise MaterializedTrajectoryError(
                "trajectory path and provenance must be present together"
            )
        if (self.trajectory_authority_path is None) != (
            self.trajectory_authority_ref is None
        ):
            raise MaterializedTrajectoryError(
                "trajectory authority path and reference must be present together"
            )
        if self.trajectory_authority_ref is not None and self.trajectory_path is None:
            raise MaterializedTrajectoryError(
                "trajectory authority requires the selected trajectory artifact"
            )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "universe_path": str(self.universe_path) if self.universe_path else None,
            "provenance_path": (
                str(self.provenance_path) if self.provenance_path else None
            ),
            "selection": self.selection.to_dict(),
            "materialized_concepts": list(self.materialized_concepts),
            "coverage": self.coverage.to_dict(),
            "blocked": self.blocked,
            "note": self.note,
            "selection_usage": self.selection_usage,
            "selection_cost_usd": self.selection_cost_usd,
            "selection_model": self.selection_model,
        }
        if (self.cohort_authority_path is None) != (self.cohort_authority_ref is None):
            raise MaterializedMetadataError(
                "cohort authority path and reference must be present together"
            )
        if self.cohort_authority_path is not None:
            payload["cohort_authority_path"] = str(self.cohort_authority_path)
            payload["cohort_authority_ref"] = self.cohort_authority_ref.to_dict()
        if self.trajectory_path is not None:
            payload["trajectory_path"] = str(self.trajectory_path)
            payload["trajectory_provenance_path"] = str(self.trajectory_provenance_path)
        if self.trajectory_authority_path is not None:
            payload["trajectory_authority_path"] = str(self.trajectory_authority_path)
            payload["trajectory_authority_ref"] = (
                self.trajectory_authority_ref.to_dict()
            )
        return payload


def _selection_cost(llm: LLMClient) -> tuple:
    """Best-effort (usage, estimated_usd, model) for the last selection call."""
    usage = getattr(llm, "last_usage", None)
    # OpenAIClient stores the model id privately as ``_model``; accept either.
    model = getattr(llm, "model", None) or getattr(llm, "_model", None)
    cost: Optional[float] = None
    if isinstance(usage, dict) and model:
        try:
            from ..providers.cost import CostMeter

            cost = CostMeter().estimate_cost(
                str(model),
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
            )
        except Exception:
            cost = None
    return (
        usage if isinstance(usage, dict) else None,
        cost,
        str(model) if model else None,
    )


def acquire_universe_for_question(
    *,
    export_dir: Union[str, Path],
    question: str,
    llm: LLMClient,
    output_dir: Union[str, Path],
    stem: str = "universe",
    target_outcome: Optional[str],
    outcome_concepts: Sequence[str],
    required_feature_concepts: Sequence[str] = (),
    static_concepts: Sequence[str] = (),
    cohort_window: tuple = (0.0, 24.0),
    database: str = "miiv",
    require_outcome: bool = True,
    emit_trajectory: bool = True,
    trajectory_window: Optional[tuple] = (-24.0, 168.0),
) -> AcquisitionResult:
    """Agent selects concepts, we check coverage, then materialise the universe.

    The materialised table is the WIDE universe (no question-specific 纳排 —
    the agent applies inclusion/exclusion in-sandbox); it carries the
    agent-selected feature concepts plus the explicitly declared outcome and
    static concepts. When the user's data does not cover the selection, the
    coverage report's advice tells them what to re-extract; we still
    materialise the available subset so our own (full-export) experiments
    proceed, and flag ``blocked`` only if the outcome itself is missing.
    """
    from ..cohort.materializer import materialize_to_parquet

    catalog = build_available_catalog(export_dir)
    selection = DataFoundationAgent(llm).select_concepts(
        question=question, catalog=catalog, target_outcome=target_outcome
    )
    sel_usage, sel_cost, sel_model = _selection_cost(llm)
    coverage = selection.coverage

    # Only the available, resolved concepts can be materialised as features;
    # outcome + demographics are passed via the dedicated args below.
    available_selected = [
        c for c in selection.selected_concepts if c in coverage.available
    ]
    feature_concepts = [
        c
        for c in available_selected
        if c not in set(outcome_concepts) | set(static_concepts)
    ]
    required_feature_coverage = assess_coverage(
        list(required_feature_concepts), catalog
    )
    if required_feature_concepts and not required_feature_coverage.sufficient:
        return AcquisitionResult(
            universe_path=None,
            provenance_path=None,
            selection=selection,
            materialized_concepts=[],
            coverage=coverage,
            blocked=True,
            note=(
                "Required analysis concepts are not available in the prepared "
                f"export: {list(required_feature_coverage.missing)}"
            ),
            selection_usage=sel_usage,
            selection_cost_usd=sel_cost,
            selection_model=sel_model,
        )
    feature_concepts = list(
        dict.fromkeys(
            [
                *feature_concepts,
                *[
                    concept
                    for concept in required_feature_concepts
                    if concept not in set(outcome_concepts) | set(static_concepts)
                ],
            ]
        )
    )
    typed_catalog = any(item.typed_metadata for item in catalog.concepts)
    if typed_catalog:
        static_coverage = assess_coverage(list(static_concepts), catalog)
        effective_static_concepts = [
            concept
            for concept in static_concepts
            if concept in static_coverage.available
        ]
    else:
        effective_static_concepts = list(static_concepts)

    # Hard block only when the outcome itself cannot be sourced — every other
    # gap is advisory (re-extract) and we proceed on what is present.
    outcome_ok = assess_coverage(list(outcome_concepts), catalog).sufficient
    if require_outcome and not outcome_ok:
        return AcquisitionResult(
            universe_path=None,
            provenance_path=None,
            selection=selection,
            materialized_concepts=[],
            coverage=coverage,
            blocked=True,
            note=(
                f"Target outcome {list(outcome_concepts)} is not in the provided "
                "data; cannot build the cohort. Re-extract the outcome concept."
            ),
            selection_usage=sel_usage,
            selection_cost_usd=sel_cost,
            selection_model=sel_model,
        )

    paths = materialize_to_parquet(
        output_dir=output_dir,
        stem=stem,
        feature_concepts=feature_concepts,
        database=database,
        data_path=str(export_dir),
        outcome_concepts=list(outcome_concepts),
        static_concepts=effective_static_concepts,
        cohort_window=cohort_window,
        # no cohort_definition => wide universe; agent does 纳排 in-sandbox
        # Also emit the long-format trajectory for the analysis concepts so the
        # agent can build threshold-crossing onsets / incident-after-exposure
        # endpoints / landmark designs the wide summary cannot express. Written
        # as <stem>_trajectory.parquet next to the universe; the runner
        # auto-discovers it and exposes TRAJECTORY_PARQUET.
        emit_trajectory=emit_trajectory,
        trajectory_concepts=[*feature_concepts, *outcome_concepts],
        trajectory_window=trajectory_window,
    )
    verified_authority = load_verified_materialized_cohort_authority(
        Path(paths["parquet"])
    )
    if "cohort_authority" in paths and verified_authority is None:
        raise MaterializedMetadataError(
            "typed materializer declared an authority but acquisition could not "
            "verify it"
        )
    trajectory_path = Path(paths["trajectory"]) if "trajectory" in paths else None
    typed_trajectory_declared = "trajectory_authority" in paths
    verified_trajectory = (
        load_verified_materialized_trajectory_authority(
            trajectory_path,
            expected_universe_authority=(
                verified_authority.reference if verified_authority is not None else None
            ),
        )
        if trajectory_path is not None and typed_trajectory_declared
        else None
    )
    if typed_trajectory_declared and verified_trajectory is None:
        raise MaterializedTrajectoryError(
            "typed materializer declared a trajectory authority but acquisition "
            "could not verify it"
        )
    if (
        verified_authority is not None
        and trajectory_path is not None
        and (verified_trajectory is None)
    ):
        raise MaterializedTrajectoryError(
            "typed acquisition trajectory is missing its sealed authority"
        )
    note = ""
    if not coverage.sufficient:
        note = (
            "Some agent-requested concepts are not in the provided data; "
            "proceeding on the available subset. Advice: " + " ".join(coverage.advice)
        )
    return AcquisitionResult(
        universe_path=Path(paths["parquet"]),
        provenance_path=Path(paths["provenance"]),
        cohort_authority_path=(
            Path(paths["parquet"]).parent / verified_authority.reference.file
            if verified_authority is not None
            else None
        ),
        cohort_authority_ref=(
            verified_authority.reference if verified_authority is not None else None
        ),
        trajectory_path=trajectory_path,
        trajectory_provenance_path=(
            Path(paths["trajectory_provenance"])
            if trajectory_path is not None
            else None
        ),
        trajectory_authority_path=(
            trajectory_path.parent / verified_trajectory.reference.file
            if verified_trajectory is not None and trajectory_path is not None
            else None
        ),
        trajectory_authority_ref=(
            verified_trajectory.reference if verified_trajectory is not None else None
        ),
        selection=selection,
        materialized_concepts=feature_concepts,
        coverage=coverage,
        blocked=False,
        note=note,
        selection_usage=sel_usage,
        selection_cost_usd=sel_cost,
        selection_model=sel_model,
    )
