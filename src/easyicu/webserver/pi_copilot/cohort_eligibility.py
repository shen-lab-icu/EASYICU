"""Standard ICU eligibility criteria Copilot can put to the researcher.

Owner: Copilot study setup.
Public contract: given a StudyContext, say whether the study has stated any
eligibility criterion, and offer the canonical options a researcher chooses
between. Every option carries the exact ``cohort`` patch it would apply, so no
consumer re-derives what an option means.

Why this exists
---------------
The Research Agent is deliberately forbidden from inventing eligibility. Only
explicit structured filter fields authorize a predicate-filtered primary
cohort (``agent_pipeline_runs.primary_cohort_selection_mode``); with none set,
every bound input row is the prespecified denominator and the Planner is told
not to invent a completeness, anchor, or proxy filter. That boundary is right:
an agent adding "age >= 18" on its own is making the researcher's study-design
decision.

But nothing then *asked* the researcher. Across the 799 study contexts on one
development host, 771 carried no cohort preset at all and exactly one used the
first-admission adult preset, so studies configured entirely through the
conversation reached the Planner with no eligibility statement and were
written up on "all bound input rows" -- the first thing a reviewer asks about.

This module supplies the question, never the answer. Declining is a first
class option that keeps the whole input universe, and nothing here is applied
without the researcher choosing it.

The options are case-neutral: they name no benchmark case, exposure, outcome,
score, or database, only eligibility axes every ICU cohort study has to settle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


SCHEMA_VERSION = "easyicu.pi-cohort-eligibility/1"

#: Fields whose presence authorizes a predicate-filtered primary cohort. This
#: mirrors the pipeline-side roster deliberately rather than importing it: the
#: two must agree, and the contract test in
#: ``tests/test_pi_copilot_cohort_eligibility.py`` fails if they drift.
ELIGIBILITY_FIELDS: Tuple[str, ...] = (
    "age_min",
    "age_max",
    "min_icu_los_hours",
    "include_diagnoses",
    "exclude_diagnoses",
    "exclude_readmissions",
)


def _option(
    *,
    option_id: str,
    label_en: str,
    label_zh: str,
    detail_en: str,
    detail_zh: str,
    cohort: Mapping[str, Any],
    declines: bool = False,
) -> Dict[str, Any]:
    return {
        "id": option_id,
        "label": {"en": label_en, "zh": label_zh},
        "detail": {"en": detail_en, "zh": detail_zh},
        # The exact patch this option applies. A consumer sends it as-is; it
        # never rebuilds the meaning of an option from its id or label.
        "cohort": dict(cohort),
        "declines_eligibility": bool(declines),
    }


#: The canonical choices, in the order a researcher weighs them. Each carries
#: explicit field values rather than leaning on a preset default: the
#: ``adult_all`` preset normalises ``age_min`` to 0 unless the caller states
#: 18, so an option that sent the preset alone would silently include children
#: under a label that says adults.
ELIGIBILITY_OPTIONS: Tuple[Dict[str, Any], ...] = (
    _option(
        option_id="adults_first_admission",
        label_en="Adults, first ICU admission only",
        label_zh="成人 · 仅首次 ICU 入住",
        detail_en=(
            "Excludes stays under 18 and every readmission after the first. "
            "The usual default when repeated stays would otherwise be counted "
            "as independent observations."
        ),
        detail_zh=(
            "排除 18 岁以下，以及首次之后的每一次再入住。"
            "当重复入住会被当成独立观测时，这通常是默认选择。"
        ),
        cohort={
            "preset": "adult_first",
            "age_min": 18,
            "exclude_readmissions": True,
        },
    ),
    _option(
        option_id="adults_all_admissions",
        label_en="Adults, every ICU admission",
        label_zh="成人 · 全部 ICU 入住",
        detail_en=(
            "Excludes stays under 18 and keeps repeat admissions. Choose this "
            "when the analysis unit is the stay and repeated stays are handled "
            "in the model rather than by exclusion."
        ),
        detail_zh=(
            "排除 18 岁以下，但保留重复入住。当分析单位是每次住院、"
            "且重复入住由模型而非排除来处理时选它。"
        ),
        cohort={
            "preset": "adult_all",
            "age_min": 18,
            "exclude_readmissions": False,
        },
    ),
    _option(
        option_id="adults_first_admission_min_stay",
        label_en="Adults, first admission, at least 24 h in ICU",
        label_zh="成人 · 仅首次入住 · ICU 停留 ≥ 24 小时",
        detail_en=(
            "Adds a minimum stay to the first-admission adult cohort, so very "
            "short stays that cannot carry the planned observation window are "
            "excluded before analysis rather than after."
        ),
        detail_zh=(
            "在成人首次入住的基础上加最短停留时长，"
            "让承载不了计划观察窗口的极短住院在分析前就被排除，而不是事后处理。"
        ),
        cohort={
            "preset": "adult_first",
            "age_min": 18,
            "exclude_readmissions": True,
            "min_icu_los_hours": 24,
        },
    ),
    _option(
        option_id="no_eligibility_filter",
        label_en="No eligibility filter: every bound ICU stay",
        label_zh="不设资格标准 · 全部绑定的 ICU 住院",
        detail_en=(
            "Every row of the bound export is the analysis cohort. A "
            "deliberate choice, not a default: the write-up will state that "
            "no inclusion or exclusion criterion was applied."
        ),
        detail_zh=(
            "绑定数据包的每一行都是分析队列。这是一个明确选择而非默认值："
            "稿件会写明未施加任何纳入或排除标准。"
        ),
        # Only the preset. A zero age floor is not "no filter" to the pipeline:
        # `primary_cohort_selection_mode` treats any present value as declared
        # filtering authority, so sending `age_min: 0` would flip this option
        # to predicate_filtered with no predicate behind it -- neither what the
        # researcher chose nor what the ledger could then draw.
        cohort={"preset": "all_icu"},
        declines=True,
    ),
)


def _cohort(study: Any) -> Mapping[str, Any]:
    raw = study.get("cohort") if isinstance(study, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def stated_eligibility_fields(study: Any) -> List[str]:
    """Which eligibility fields this study actually carries a value for."""

    cohort = _cohort(study)
    stated: List[str] = []
    for field in ELIGIBILITY_FIELDS:
        value = cohort.get(field)
        if value is None or value == "" or value == []:
            continue
        if field == "exclude_readmissions" and value is not True:
            continue
        if field in {"age_min", "age_max", "min_icu_los_hours"}:
            try:
                if int(value) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        stated.append(field)
    return stated


def eligibility_stated(study: Any) -> bool:
    """Has the researcher settled eligibility for this study, either way?

    A recorded preset counts even when it applies no filter: choosing "every
    bound stay" is an answer. What does not count is a cohort slot the
    conversation never wrote, which is the state the question exists for.
    """

    cohort = _cohort(study)
    if str(cohort.get("preset") or "").strip():
        return True
    return bool(stated_eligibility_fields(study))


def eligibility_proposal(study: Any) -> Dict[str, Any]:
    """Compile the eligibility question, or record that it is already settled.

    Always returns the same shape so a consumer branches on ``stated`` rather
    than on whether a key exists.
    """

    from ..agent_pipeline_runs import primary_cohort_selection_mode

    cohort = _cohort(study)
    stated = eligibility_stated(study)
    try:
        selection_mode = primary_cohort_selection_mode({"cohort": dict(cohort)})
    except Exception:  # noqa: BLE001 - a setup hint must not break the snapshot
        selection_mode = "all_input_rows"
    return {
        "schema_version": SCHEMA_VERSION,
        "stated": stated,
        "stated_fields": stated_eligibility_fields(study),
        "preset": str(cohort.get("preset") or ""),
        # What the Planner would be bound to if the study ran as it stands.
        "selection_mode": selection_mode,
        "options": [] if stated else [dict(option) for option in ELIGIBILITY_OPTIONS],
    }


def cohort_patch_for_option(option_id: Any) -> Dict[str, Any]:
    """Return the exact cohort patch one option applies, or ``{}``."""

    wanted = str(option_id or "").strip()
    for option in ELIGIBILITY_OPTIONS:
        if option["id"] == wanted:
            return dict(option["cohort"])
    return {}


def option_ids() -> Tuple[str, ...]:
    return tuple(str(option["id"]) for option in ELIGIBILITY_OPTIONS)


__all__ = [
    "ELIGIBILITY_FIELDS",
    "ELIGIBILITY_OPTIONS",
    "SCHEMA_VERSION",
    "cohort_patch_for_option",
    "eligibility_proposal",
    "eligibility_stated",
    "option_ids",
    "stated_eligibility_fields",
]
