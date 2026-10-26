"""Compare a study's requirements with its bound source's capabilities.

The 2026-10-26 six-database scope decision replaces whole-source bans with
evidenced capability checks. Legacy refusal codes remain stable for callers.

The family is the one the study records, else the one the conversation
stated for the question and the host's check settled (``question_family_check``),
else the family its question states (``study_family_reading``; the launch
records its design first), else the one the planner's own family routing
reads in the question -- the inference the plan itself would use -- so the
refusal and the plan cannot disagree.  While the conversation's reading and
the host's differ, nothing is refused and nothing is planned: the gap is
``study_family_reading_disagreement`` on every database, and the researcher
chooses.  A question that states a description or an association is planned
on that design, so a question that only mentions survival words is not
refused.  The workflow asks the same question of the bound source
(:func:`bound_study_family_gap`), so the conversation says it before offering
a plan.  The gap is codes only: the family, the database and the databases v1
offers that family on.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = [
    "STUDY_FAMILY_DATABASE_OUT_OF_SCOPE",
    "STUDY_FAMILY_GAP_FIELD",
    "bound_study_family_gap",
    "scope_database_key",
    "stop_family_out_of_scope",
    "study_family_database_gap",
    "v1_scope_description",
]

STUDY_FAMILY_DATABASE_OUT_OF_SCOPE = "study_family_database_out_of_scope"
STUDY_FAMILY_GAP_FIELD = "study_family_gap"


def v1_scope_description(
    *, reason_code: str, detail: Mapping[str, Any], database: Optional[str] = None,
) -> Dict[str, Any]:
    """Scope-copy coordinates from the same owners that refuse a study.

    The presentation owns the wording behind these stable label keys. Never
    copy a refusal's free text or infer another endpoint for the researcher.
    """

    import re

    from easyicu.webserver.target_trial_setup import TARGET_TRIAL_DATABASE_OUT_OF_SCOPE
    from easyicu.webserver.research_capability_scope import eligible_family_databases

    trial = reason_code == TARGET_TRIAL_DATABASE_OUT_OF_SCOPE
    source = database or detail.get("database")
    family = detail.get("family")
    return {
        "scope_label_key": (
            "v1_scope.target_trial" if trial else "v1_scope.study_family"
        ),
        "database": source if isinstance(source, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", source) else None,
        "family": family if isinstance(family, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", family) else None,
        "supported_databases": tuple(detail["supported_databases"]) if "supported_databases" in detail
        else eligible_family_databases("causal_inference" if trial else str(family or "")),
        **({"missing_capabilities": detail["missing_capabilities"]} if "missing_capabilities" in detail else {}),
    }


def _reading_disagreement(study: Mapping[str, Any], question: str) -> Optional[Dict[str, Any]]:
    from easyicu.webserver.question_family_check import (
        QUESTION_FAMILY_CHECK_FIELD,
        open_disagreement,
    )

    design = study.get("analysis_design")
    if isinstance(design, Mapping) and str(design.get("analysis_family") or "").strip():
        return None
    return open_disagreement(study.get(QUESTION_FAMILY_CHECK_FIELD), question)


def _study_family(study: Mapping[str, Any], question: str, database: str) -> Optional[str]:
    from easyicu.research_agent.planning.analysis_types import (
        canonical_analysis_family,
        infer_analysis_type,
    )
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    design = study.get("analysis_design")
    recorded = (
        canonical_analysis_family(str(design.get("analysis_family") or "") or None)
        if isinstance(design, Mapping)
        else None
    )
    if recorded:
        return recorded
    text = str(question or "").strip()
    if not text:
        return None
    from easyicu.webserver.question_family_check import (
        QUESTION_FAMILY_CHECK_FIELD,
        settled_question_family,
    )

    settled = settled_question_family(study.get(QUESTION_FAMILY_CHECK_FIELD), text)
    if settled is not None:
        return settled
    # The family alone: the design's variance needs the source's patient
    # grouping, which a polled workflow does not read.
    from easyicu.webserver.study_family_reading import study_family_reading

    stated = study_family_reading(text)
    if stated is not None:
        return canonical_analysis_family(stated.family) or stated.family
    context = ResearchContext(
        research_question=text,
        cohort=CohortDescriptor(cohort_name="study", database=database, n_stays=0),
        variables=[],
    )
    return infer_analysis_type(context).key


def study_family_database_gap(
    study: Mapping[str, Any], *, question: str, database: str
) -> Optional[Dict[str, Any]]:
    """Why v1 does not offer the study's family on ``database``, or ``None``.

    While the study's two readings of its question differ, the gap is that
    disagreement, with both families, on any database.
    """

    disagreement = _reading_disagreement(study, question)
    if disagreement is not None:
        return {
            **disagreement,
            "database": database,
            "choices": _choices(disagreement),
        }
    family = _study_family(study, question, database)
    if family is None:
        return None
    from easyicu.webserver.research_capability_scope import capability_scope_gap

    return capability_scope_gap(study, family=family, database=database,
                                code=STUDY_FAMILY_DATABASE_OUT_OF_SCOPE, question=question)


def scope_database_key(database: Optional[str]) -> str:
    """The database whose v1 scope a source has; a demo has its parent's."""

    from easyicu.databases import get_database_profile

    key = str(database or "").strip().lower()
    try:
        profile = get_database_profile(key)
    except KeyError:
        return key
    return profile.parent_key or profile.key


def _choices(disagreement: Mapping[str, Any]) -> list[Dict[str, str]]:
    from easyicu.webserver.question_family_check import family_choice_texts

    zh = family_choice_texts(disagreement, "zh")
    en = family_choice_texts(disagreement, "en")
    return [
        {"family": family, "zh": text_zh, "en": text_en}
        for (family, text_zh), (_family, text_en) in zip(zh, en)
    ]


def bound_study_family_gap(study: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The gap of the study's own question on its bound source, or ``None``."""

    from easyicu.databases import normalize_database_key

    source = study.get("data_source")
    raw = str(source.get("database") or "").strip() if isinstance(source, Mapping) else ""
    if not raw:
        return None
    try:
        database = normalize_database_key(raw)
    except KeyError:
        return None
    return study_family_database_gap(
        study, question=str(study.get("question") or ""), database=database
    )


def stop_family_out_of_scope(
    study: Mapping[str, Any], *, question: str, database: str
) -> None:
    """Stop a launch whose study family v1 does not offer on its database."""

    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    gap = study_family_database_gap(study, question=question, database=database)
    if gap is None:
        return
    if gap["code"] != STUDY_FAMILY_DATABASE_OUT_OF_SCOPE:
        raise ResearchPipelineRunError(
            gap["code"],
            f"The question reads as {gap['model_family']} or as {gap['host_family']}; "
            "the researcher chooses before planning, so no model was called and no "
            "analysis was run.",
            details={STUDY_FAMILY_GAP_FIELD: gap},
        )
    raise ResearchPipelineRunError(
        gap["code"],
        f"This version does not offer {gap['family']} studies on {database}, so "
        "planning stopped before any model was called and no analysis was run.",
        details={STUDY_FAMILY_GAP_FIELD: gap},
    )
