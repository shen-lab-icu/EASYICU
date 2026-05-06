"""Build a :class:`ResearchContext` from an EasyICU cohort dataframe.

The builder is the bridge between the *data* world (a parquet file
or DataFrame produced by ``easyicu.load_concepts`` / ``filter_patients``)
and the *agent* world (a structured ``ResearchContext`` that drives
prompts and validators).

The builder is intentionally tolerant: it works on a plain DataFrame
even if EasyICU is not installed, falling back to dtype-only
classification. When EasyICU *is* installed, it enriches each
column with description, source databases and category from the
concept dictionary.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .icu_rules import (
    ICU_RULES,
    VariableKind,
    aggregation_rule_for,
    classify_variable,
    default_time_windows,
)
from .schema import (
    AggregationRule,
    CohortDescriptor,
    ConceptDescriptor,
    MissingnessProfile,
    ResearchContext,
    TimeWindow,
    VariableRole,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_get_concept_info(name: str) -> Optional[Dict[str, Any]]:
    """Best-effort fetch of EasyICU concept metadata. Returns None if unavailable."""
    try:
        from easyicu import get_concept_info  # type: ignore
    except Exception:
        return None
    try:
        return get_concept_info(name)
    except Exception:
        return None


def _missingness_kind(fraction: float) -> str:
    """Crude heuristic — over-conservative on purpose.

    The point of this field is *not* to authoritatively classify
    missingness mechanism (that needs domain reasoning) — it is to
    nudge the agent into checking when missingness is high enough
    that ignoring it is risky.
    """
    if fraction == 0.0:
        return "MCAR_likely"
    if fraction < 0.05:
        return "MCAR_likely"
    if fraction < 0.30:
        return "MAR_likely"
    return "MNAR_likely"


def _profile_missingness(series: pd.Series) -> MissingnessProfile:
    n_total = int(len(series))
    n_missing = int(series.isna().sum())
    fraction = (n_missing / n_total) if n_total > 0 else 0.0
    return MissingnessProfile(
        fraction_missing=fraction,
        n_missing=n_missing,
        n_total=n_total,
        missingness_kind=_missingness_kind(fraction),  # type: ignore[arg-type]
    )


def _allowed_aggregations(role: VariableRole, kind: VariableKind) -> List[AggregationRule]:
    return aggregation_rule_for(role, kind)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_research_context(
    *,
    research_question: str,
    cohort: Union[pd.DataFrame, str, Path],
    cohort_name: str,
    database: str,
    inclusion_criteria: Optional[Sequence[str]] = None,
    exclusion_criteria: Optional[Sequence[str]] = None,
    target_outcome: Optional[str] = None,
    cross_database_validation: Optional[Sequence[str]] = None,
    id_columns: Optional[Sequence[str]] = None,
    time_columns: Optional[Sequence[str]] = None,
    outcome_columns: Optional[Sequence[str]] = None,
    concept_descriptions: Optional[Dict[str, str]] = None,
    time_windows: Optional[Sequence[TimeWindow]] = None,
    notes: Optional[str] = None,
) -> ResearchContext:
    """Build a :class:`ResearchContext` from a cohort dataframe.

    Parameters
    ----------
    research_question
        Plain-language research question; becomes the spine of the
        agent prompts and the manuscript scaffold.
    cohort
        DataFrame or path to a parquet file containing one row per
        analysis unit (typically per ICU stay) with all variables
        already aggregated to that unit. The cohort is *not*
        re-aggregated by the pipeline.
    cohort_name, database
        Human-friendly tags. ``database`` should match an EasyICU
        source key (``miiv``, ``eicu``, ``aumc``, ``hirid``, ``sic``)
        when applicable so the cross-database validator can compare.
    inclusion_criteria, exclusion_criteria
        Free-text criteria that already shaped the cohort. The
        ``CohortAuditor`` checks these against generated code's
        further filters.
    target_outcome
        Name of the primary outcome column. Used by validators to
        confirm the analysis actually predicts this column and not a
        proxy.
    cross_database_validation
        Other databases to replicate this analysis on. The pipeline
        does not run those analyses itself in v1 — it surfaces them
        for the human to schedule — but the manifest tracks which
        databases were promised vs. actually run.
    """
    # --- normalise cohort input
    if isinstance(cohort, (str, Path)):
        cohort_path = str(Path(cohort).resolve())
        df = pd.read_parquet(cohort_path)
    else:
        cohort_path = None
        df = cohort

    if not isinstance(df, pd.DataFrame):
        raise TypeError("cohort must be a pandas DataFrame or a parquet path")

    # --- cohort descriptor
    id_columns = list(id_columns) if id_columns else _guess_id_columns(df)
    time_columns = list(time_columns) if time_columns else _guess_time_columns(df)
    outcome_columns = list(outcome_columns) if outcome_columns else _guess_outcome_columns(df)

    n_patients = _count_unique(df, id_columns[:1]) if id_columns else int(len(df))
    n_stays = int(len(df))

    cohort_desc = CohortDescriptor(
        cohort_name=cohort_name,
        database=database,
        n_patients=n_patients,
        n_stays=n_stays,
        inclusion_criteria=list(inclusion_criteria or []),
        exclusion_criteria=list(exclusion_criteria or []),
        id_columns=id_columns,
        time_columns=time_columns,
        outcome_columns=outcome_columns,
    )

    # --- per-column descriptors
    descriptors: List[ConceptDescriptor] = []
    user_descriptions = dict(concept_descriptions or {})
    for col in df.columns:
        descriptors.append(
            _describe_column(
                df=df,
                col=col,
                user_descriptions=user_descriptions,
                id_columns=id_columns,
                time_columns=time_columns,
                outcome_columns=outcome_columns,
            )
        )

    # --- time windows: provided or default
    windows = list(time_windows) if time_windows else default_time_windows()

    return ResearchContext(
        research_question=research_question,
        cohort=cohort_desc,
        variables=descriptors,
        time_windows=windows,
        target_outcome=target_outcome,
        cross_database_validation=list(cross_database_validation or []),
        cohort_parquet=cohort_path,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Column-level reasoning
# ---------------------------------------------------------------------------


def _describe_column(
    *,
    df: pd.DataFrame,
    col: str,
    user_descriptions: Dict[str, str],
    id_columns: Sequence[str],
    time_columns: Sequence[str],
    outcome_columns: Sequence[str],
) -> ConceptDescriptor:
    series = df[col]
    sample = series.dropna().head(50).tolist() if len(series) else []
    hint = classify_variable(col, str(series.dtype), sample)

    # role fix-ups: respect user-declared id/time/outcome
    role = hint.role
    if col in id_columns:
        role = VariableRole.ID
    elif col in time_columns:
        role = VariableRole.TIME
    elif col in outcome_columns:
        role = VariableRole.OUTCOME

    # description: prefer user, fall back to EasyICU concept dict
    description = user_descriptions.get(col)
    source_concept = None
    source_databases: List[str] = []
    if description is None:
        info = _safe_get_concept_info(col)
        if info is not None:
            description = info.get("description") or None
            source_concept = info.get("name") or col
            srcs = info.get("sources") or info.get("source_databases") or []
            if isinstance(srcs, dict):
                source_databases = sorted(map(str, srcs.keys()))
            else:
                source_databases = [str(s) for s in srcs]

    allowed = _allowed_aggregations(role, hint.kind)
    miss = _profile_missingness(series)

    return ConceptDescriptor(
        name=col,
        description=description,
        role=role,
        dtype=str(series.dtype),
        unit=hint.unit,
        valid_range=list(hint.valid_range) if hint.valid_range else None,
        allowed_aggregations=allowed,
        aggregation_default=hint.aggregation_default,
        is_ordinal=hint.is_ordinal,
        ordinal_levels=list(hint.ordinal_levels) if hint.ordinal_levels else None,
        source_concept=source_concept,
        source_databases=source_databases,
        pitfalls=list(hint.pitfalls),
        missingness=miss,
    )


def _guess_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {
            "patient_id", "icustay_id", "hadm_id", "stay_id", "subject_id",
            "patientunitstayid", "uniquepid", "admissionid",
        }
    ]
    return candidates[:3]


def _guess_time_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        s = df[c]
        if "datetime" in str(s.dtype).lower() or "timestamp" in str(s.dtype).lower():
            out.append(c)
        elif c.lower() in {"intime", "outtime", "admittime", "dischtime", "deathtime", "charttime"}:
            out.append(c)
    return out


def _guess_outcome_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if cl in {"death", "death_icu", "death_hosp", "mortality", "los_icu", "los_hosp",
                  "readmission", "readmit_30d"}:
            out.append(c)
        elif cl.startswith("outcome_"):
            out.append(c)
    return out


def _count_unique(df: pd.DataFrame, cols: Sequence[str]) -> int:
    if not cols:
        return int(len(df))
    try:
        return int(df[list(cols)].drop_duplicates().shape[0])
    except Exception:
        return int(len(df))


# ---------------------------------------------------------------------------
# Naive context builder (T1.4 — ablation arm)
# ---------------------------------------------------------------------------


def build_naive_research_context(
    *,
    research_question: str,
    cohort: Union[pd.DataFrame, str, Path],
    cohort_name: str,
    database: str,
    inclusion_criteria: Optional[Sequence[str]] = None,
    exclusion_criteria: Optional[Sequence[str]] = None,
    target_outcome: Optional[str] = None,
    cross_database_validation: Optional[Sequence[str]] = None,
    id_columns: Optional[Sequence[str]] = None,
    time_columns: Optional[Sequence[str]] = None,
    outcome_columns: Optional[Sequence[str]] = None,
    concept_descriptions: Optional[Dict[str, str]] = None,
    time_windows: Optional[Sequence[TimeWindow]] = None,
    notes: Optional[str] = None,
) -> ResearchContext:
    """Hero-ablation "naive" builder.

    Emits the *minimum* viable context: every column gets only its
    name, dtype, and a single allowed aggregation ``ANY``. No
    ICU-specific role inference, no pitfalls, no missingness profile
    hints, no ordinal flags. This is what a generic
    OpenLens-style agent would synthesise from a CSV.

    The returned :class:`ResearchContext` is structurally identical to
    the ICU-aware one — same schema, same fields — so downstream code
    is unchanged. Only the *informational content* of the context is
    stripped, which is the variable T1.4 measures.
    """
    if isinstance(cohort, (str, Path)):
        cohort_path = str(Path(cohort).resolve())
        df = pd.read_parquet(cohort_path)
    else:
        cohort_path = None
        df = cohort
    if not isinstance(df, pd.DataFrame):
        raise TypeError("cohort must be a pandas DataFrame or a parquet path")

    id_cols = list(id_columns) if id_columns else _guess_id_columns(df)
    time_cols = list(time_columns) if time_columns else _guess_time_columns(df)
    out_cols = list(outcome_columns) if outcome_columns else _guess_outcome_columns(df)

    n_patients = _count_unique(df, id_cols[:1]) if id_cols else int(len(df))
    n_stays = int(len(df))
    cohort_desc = CohortDescriptor(
        cohort_name=cohort_name, database=database,
        n_patients=n_patients, n_stays=n_stays,
        inclusion_criteria=list(inclusion_criteria or []),
        exclusion_criteria=list(exclusion_criteria or []),
        id_columns=id_cols, time_columns=time_cols, outcome_columns=out_cols,
    )

    user_descriptions = dict(concept_descriptions or {})
    descriptors: List[ConceptDescriptor] = []
    for col in df.columns:
        # role = OTHER for everything except declared id / time / outcome.
        if col in id_cols:
            role = VariableRole.ID
        elif col in time_cols:
            role = VariableRole.TIME
        elif col in out_cols:
            role = VariableRole.OUTCOME
        else:
            role = VariableRole.OTHER
        descriptors.append(ConceptDescriptor(
            name=col,
            description=user_descriptions.get(col),
            role=role,
            dtype=str(df[col].dtype),
            unit=None,
            valid_range=None,
            allowed_aggregations=[AggregationRule.ANY],
            aggregation_default=AggregationRule.ANY,
            is_ordinal=False,
            ordinal_levels=None,
            source_concept=None,
            source_databases=[],
            pitfalls=[],
            missingness=None,
        ))

    # Default windows are also stripped — a naive agent does not know
    # about "first_24h"; keep an empty list unless the caller passed one.
    windows = list(time_windows) if time_windows else []
    return ResearchContext(
        research_question=research_question,
        cohort=cohort_desc,
        variables=descriptors,
        time_windows=windows,
        target_outcome=target_outcome,
        cross_database_validation=list(cross_database_validation or []),
        cohort_parquet=cohort_path,
        notes=notes,
    )


def retrieve_context_variables(
    context: ResearchContext,
    *,
    query: str,
    top_k: int = 40,
) -> List[ConceptDescriptor]:
    """Return the most relevant concept descriptors for a question.

    O6 — long-context guard. This is deliberately dependency-free: it
    uses lexical overlap across variable name, description, role and
    pitfall text, with small boosts for target outcomes and explicitly
    question-mentioned variables. If a future install has an embedding
    index, it can replace this scorer without changing the pipeline
    contract.
    """
    if top_k <= 0 or top_k >= len(context.variables):
        return list(context.variables)
    q_tokens = _tokens(query or context.research_question)
    scored: List[Tuple[float, int, ConceptDescriptor]] = []
    for i, v in enumerate(context.variables):
        haystack = " ".join([
            v.name,
            v.description or "",
            v.role.value,
            v.dtype,
            " ".join(v.pitfalls),
            v.missingness_semantics or "",
            " ".join(v.forbidden_transformations),
            " ".join(v.cross_database_notes),
        ])
        v_tokens = _tokens(haystack)
        overlap = len(q_tokens & v_tokens)
        score = float(overlap)
        name_norm = re.sub(r"[^a-z0-9]+", "", v.name.lower())
        q_norm = re.sub(r"[^a-z0-9]+", "", (query or context.research_question).lower())
        if name_norm and name_norm in q_norm:
            score += 4.0
        if context.target_outcome and v.name == context.target_outcome:
            score += 3.0
        if v.role in {VariableRole.OUTCOME, VariableRole.COMPOSITE_SCORE, VariableRole.ORDINAL_SCORE}:
            score += 1.0
        if v.pitfalls:
            score += 0.5
        scored.append((score, -i, v))
    ranked = sorted(scored, key=lambda t: (t[0], t[1]), reverse=True)
    selected = [v for score, _, v in ranked[:top_k] if score > 0]
    if not selected:
        selected = [v for _, _, v in ranked[:top_k]]

    # Always preserve declared id/time/outcome columns even if the
    # natural-language query did not mention them.
    required = set(context.cohort.id_columns + context.cohort.time_columns + context.cohort.outcome_columns)
    if context.target_outcome:
        required.add(context.target_outcome)
    by_name = {v.name: v for v in context.variables}
    selected_names = {v.name for v in selected}
    for name in required:
        if name in by_name and name not in selected_names:
            selected.append(by_name[name])
            selected_names.add(name)
    return selected


def build_retrieved_research_context(
    context: ResearchContext,
    *,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
) -> ResearchContext:
    """Return a prompt-sized context with only top-K variables.

    The full :class:`ResearchContext` should still be used by validators
    and manifest writing. This helper is for agent prompts only.
    """
    if top_k is None or top_k <= 0 or top_k >= len(context.variables):
        return context
    selected = retrieve_context_variables(
        context,
        query=query or context.research_question,
        top_k=top_k,
    )
    selected_names = ", ".join(v.name for v in selected)
    retrieval_note = (
        f"Context retrieval active: showing {len(selected)}/"
        f"{len(context.variables)} variables selected for this question. "
        f"Selected variables: {selected_names}."
    )
    notes = f"{context.notes}\n\n{retrieval_note}" if context.notes else retrieval_note
    return context.model_copy(update={"variables": selected, "notes": notes})


def _tokens(text: str) -> set:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_]+", text or "")
        if len(t) >= 2
    }


__all__ = [
    "build_research_context",
    "build_naive_research_context",
    "retrieve_context_variables",
    "build_retrieved_research_context",
]
