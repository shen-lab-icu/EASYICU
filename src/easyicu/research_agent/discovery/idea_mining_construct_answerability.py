"""Typed construct answerability for Idea Mining.

This owner joins existing EasyICU concept availability with a small set of
reviewed clinical-event recipes.  It does not extract data or materialize a new
phenotype.  Its job is to distinguish a directly observed concept, a registered
derivation, an event that can be reconstructed only after semantic review, a
non-equivalent proxy, and a construct the selected source cannot support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal, Optional, Sequence

from easyicu.concept import catalog as concept_catalog
from easyicu.research_agent.concept_availability import (
    PUBLIC_DATABASES,
    explain_concept_availability,
    normalize_concept_name,
    normalize_database_name,
)

ResolutionKind = Literal[
    "direct_observed",
    "validated_derived",
    "event_reconstructable",
    "proxy_only",
    "unavailable",
]
SourceState = Literal[
    "present_in_export",
    "constructible_from_export",
    "reextractable",
    "database_capability_only",
    "source_not_selected",
    "proxy_only",
    "not_in_database",
]
Verdict = Literal["ready", "needs_review", "blocked"]


@dataclass(frozen=True, slots=True)
class ConstructAnswerability:
    construct_id: str
    label_zh: str
    requested_term: str
    resolution_kind: ResolutionKind
    source_state: SourceState
    verdict: Verdict
    required_primitives: tuple[str, ...] = ()
    available_primitives: tuple[str, ...] = ()
    missing_primitives: tuple[str, ...] = ()
    unresolved_requirements: tuple[str, ...] = ()
    supported_databases: tuple[str, ...] = ()
    materialized: bool = False
    requires_confirmation: bool = False
    recipe_id: Optional[str] = None
    recipe_version: Optional[str] = None
    rationale_zh: str = ""
    semantic_warning: str = ""

    def to_dict(self) -> dict[str, object]:
        label = {
            "ready": "可以直接研究",
            "needs_review": "可以构造，但需要确认定义",
            "blocked": "当前数据无法可靠研究",
        }[self.verdict]
        return {
            "schema_version": "easyicu.idea_construct_answerability/1",
            "construct_id": self.construct_id,
            "label_zh": self.label_zh,
            "requested_term": self.requested_term,
            "resolution_kind": self.resolution_kind,
            "source_state": self.source_state,
            "verdict": self.verdict,
            "required_primitives": list(self.required_primitives),
            "available_primitives": list(self.available_primitives),
            "missing_primitives": list(self.missing_primitives),
            "unresolved_requirements": list(self.unresolved_requirements),
            "supported_databases": list(self.supported_databases),
            "materialized": self.materialized,
            "requires_confirmation": self.requires_confirmation,
            "recipe": (
                {"recipe_id": self.recipe_id, "version": self.recipe_version}
                if self.recipe_id and self.recipe_version
                else None
            ),
            "rationale_zh": self.rationale_zh,
            "semantic_warning": self.semantic_warning,
            "user_facing": {"label": label, "explanation": self.rationale_zh},
        }


@dataclass(frozen=True, slots=True)
class _ConstructRecipe:
    construct_id: str
    label_zh: str
    aliases: tuple[str, ...]
    resolution_kind: Literal["validated_derived", "event_reconstructable"]
    alternatives: tuple[tuple[str, ...], ...]
    materialized_concepts: tuple[str, ...] = ()
    proxy_concepts: tuple[str, ...] = ()
    unresolved_requirements: tuple[str, ...] = ()
    semantic_warning: str = ""
    version: str = "1.0.0"


_RECIPES: tuple[_ConstructRecipe, ...] = (
    _ConstructRecipe(
        construct_id="extubation_failure",
        label_zh="拔管失败或再插管",
        aliases=(
            "extubation failure",
            "reintubation",
            "拔管失败",
            "再插管",
        ),
        resolution_kind="event_reconstructable",
        alternatives=(("mech_vent", "death"), ("vent_start", "vent_end", "death")),
        proxy_concepts=("vent_free_days_28", "vent_ind"),
        unresolved_requirements=(
            "invasive_ventilation_episode_authority",
            "extubation_and_reintubation_window_definition",
            "tracheostomy_transfer_competing_event_handling",
        ),
        semantic_warning=(
            "呼吸机记录中断不等于拔管；还必须排除转科、数据缺失、气切和死亡，"
            "并按预先确认的时间窗重建再插管。"
        ),
    ),
    _ConstructRecipe(
        construct_id="durable_liberation",
        label_zh="持久脱机",
        aliases=("durable liberation", "successful liberation", "持久脱机", "成功撤机"),
        resolution_kind="event_reconstructable",
        alternatives=(("mech_vent", "death"), ("vent_start", "vent_end", "death")),
        proxy_concepts=("vent_free_days_28", "vent_ind"),
        unresolved_requirements=(
            "invasive_ventilation_episode_authority",
            "durable_liberation_gap_definition",
            "competing_event_handling",
        ),
        semantic_warning=(
            "一次通气结束不等于持久脱机；必须声明再次通气间隔、死亡、气切和转出处理。"
        ),
    ),
    _ConstructRecipe(
        construct_id="extubation",
        label_zh="拔管事件",
        aliases=("extubation", "拔管", "撤机", "脱机"),
        resolution_kind="event_reconstructable",
        alternatives=(("mech_vent",), ("vent_start", "vent_end")),
        proxy_concepts=("vent_free_days_28", "vent_ind"),
        unresolved_requirements=(
            "invasive_ventilation_episode_authority",
            "airway_removal_semantics",
        ),
        semantic_warning=(
            "呼吸机记录中断不等于拔管；必须区分无创通气、气切、转科、死亡和数据缺失。"
        ),
    ),
    _ConstructRecipe(
        construct_id="awakening_after_sedation",
        label_zh="停镇静后的清醒恢复",
        aliases=(
            "awakening after sedation",
            "delayed awakening",
            "清醒恢复",
            "延迟清醒",
            "延迟苏醒",
            "不清醒",
            "未清醒",
            "意识不清",
            "持续昏迷",
        ),
        resolution_kind="event_reconstructable",
        alternatives=(
            ("propofol_rate", "rass"),
            ("midazolam_rate", "rass"),
            ("fentanyl_rate", "rass"),
            ("propofol_rate", "gcs"),
            ("midazolam_rate", "gcs"),
        ),
        proxy_concepts=("propofol", "midazolam", "dexmedetomidine", "gcs", "rass"),
        unresolved_requirements=(
            "medication_interval_end_authority",
            "awakening_threshold_definition",
            "post_discontinuation_observation_window",
        ),
        semantic_warning=(
            "最后一条给药记录不等于临床决定停药，单次 GCS/RASS 也不等于已恢复清醒；"
            "必须确认药物区间终点和清醒阈值。"
        ),
    ),
    _ConstructRecipe(
        construct_id="sedation_discontinuation",
        label_zh="镇静药停药或减量时点",
        aliases=(
            "sedation discontinuation",
            "sedative discontinuation",
            "sedation interruption",
            "停镇静药",
            "停用镇静药",
            "停用镇静",
            "镇静药停用",
            "镇静停药",
            "镇静减量",
            "停药",
        ),
        resolution_kind="event_reconstructable",
        alternatives=(("propofol_rate",), ("midazolam_rate",), ("fentanyl_rate",)),
        proxy_concepts=("propofol", "midazolam", "dexmedetomidine", "fentanyl"),
        unresolved_requirements=("medication_interval_end_authority",),
        semantic_warning=(
            "最后一条给药记录不等于临床决定停药；只有带可靠结束时间的用药区间"
            "才能进入停药事件重建。"
        ),
    ),
    _ConstructRecipe(
        construct_id="cumulative_fluid_balance",
        label_zh="累计液体平衡",
        aliases=(
            "cumulative fluid balance",
            "fluid balance",
            "累计液体平衡",
            "液体平衡",
        ),
        resolution_kind="validated_derived",
        alternatives=(("total_input_ml", "urine"),),
        materialized_concepts=("fluid_balance_cumulative",),
        proxy_concepts=("total_input_ml",),
        semantic_warning="总输液量不等于液体平衡；必须同时纳入规定窗口内的可用出量。",
    ),
)

_RECIPE_BY_ID = {recipe.construct_id: recipe for recipe in _RECIPES}


def _normalised_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("_", " "))


def _matches_alias(text: str, alias: str) -> bool:
    candidate = _normalised_text(alias)
    if not candidate:
        return False
    if any("\u4e00" <= char <= "\u9fff" for char in candidate):
        return candidate in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", text))


def _recipe_for_term(term: str) -> Optional[_ConstructRecipe]:
    normalized_id = str(term or "").strip().lower().replace(" ", "_")
    if normalized_id in _RECIPE_BY_ID:
        return _RECIPE_BY_ID[normalized_id]
    text = _normalised_text(term)
    return next(
        (
            recipe
            for recipe in _RECIPES
            if any(_matches_alias(text, alias) for alias in recipe.aliases)
        ),
        None,
    )


@lru_cache(maxsize=None)
def _supported_databases(recipe: _ConstructRecipe) -> tuple[str, ...]:
    supported: list[str] = []
    for database in PUBLIC_DATABASES:
        materialized = any(
            explain_concept_availability(concept=concept, database=database).available
            for concept in recipe.materialized_concepts
        )
        primitive_route = any(
            all(
                explain_concept_availability(concept=concept, database=database).available
                for concept in alternative
            )
            for alternative in recipe.alternatives
        )
        if materialized or primitive_route:
            supported.append(database)
    return tuple(supported)


def _selected_alternative(
    recipe: _ConstructRecipe, available: set[str]
) -> tuple[tuple[str, ...], bool]:
    for alternative in recipe.alternatives:
        if set(alternative).issubset(available):
            return alternative, True
    if not recipe.alternatives:
        return (), True
    best = max(
        recipe.alternatives,
        key=lambda alternative: len(set(alternative) & available),
    )
    return best, False


def _assess_recipe(
    recipe: _ConstructRecipe,
    *,
    requested_term: str,
    database: Optional[str],
    available_concepts: Optional[set[str]],
) -> ConstructAnswerability:
    supported = _supported_databases(recipe)
    if available_concepts is None:
        if database:
            db = normalize_database_name(database)
            capable = db in supported
            return ConstructAnswerability(
                construct_id=recipe.construct_id,
                label_zh=recipe.label_zh,
                requested_term=requested_term,
                resolution_kind=recipe.resolution_kind if capable else "unavailable",
                source_state="database_capability_only" if capable else "not_in_database",
                verdict="needs_review" if capable else "blocked",
                required_primitives=recipe.alternatives[0] if recipe.alternatives else (),
                unresolved_requirements=recipe.unresolved_requirements,
                supported_databases=supported,
                requires_confirmation=True,
                recipe_id=recipe.construct_id,
                recipe_version=recipe.version,
                rationale_zh=(
                    "该数据库的 EasyICU 概念层具备候选基础信号，但尚未选择并检查真实导出。"
                    if capable
                    else "该数据库当前没有完成此构念所需的 EasyICU 基础信号。"
                ),
                semantic_warning=recipe.semantic_warning,
            )
        return ConstructAnswerability(
            construct_id=recipe.construct_id,
            label_zh=recipe.label_zh,
            requested_term=requested_term,
            resolution_kind=recipe.resolution_kind if supported else "unavailable",
            source_state="source_not_selected" if supported else "not_in_database",
            verdict="needs_review" if supported else "blocked",
            required_primitives=recipe.alternatives[0] if recipe.alternatives else (),
            unresolved_requirements=recipe.unresolved_requirements,
            supported_databases=supported,
            requires_confirmation=True,
            recipe_id=recipe.construct_id,
            recipe_version=recipe.version,
            rationale_zh=(
                "EasyICU 在部分支持数据库中具备候选基础信号；绑定真实数据后才能检查覆盖率、时间顺序和事件定义。"
                if supported
                else "EasyICU 当前没有发现能够支持该构念的数据库级基础信号。"
            ),
            semantic_warning=recipe.semantic_warning,
        )

    available = {normalize_concept_name(value) for value in available_concepts}
    materialized = next(
        (concept for concept in recipe.materialized_concepts if concept in available),
        None,
    )
    if materialized:
        return ConstructAnswerability(
            construct_id=recipe.construct_id,
            label_zh=recipe.label_zh,
            requested_term=requested_term,
            resolution_kind="validated_derived",
            source_state="present_in_export",
            verdict="ready",
            required_primitives=recipe.alternatives[0] if recipe.alternatives else (),
            available_primitives=(materialized,),
            supported_databases=supported,
            materialized=True,
            recipe_id=recipe.construct_id,
            recipe_version=recipe.version,
            rationale_zh="该构念已经作为注册的 EasyICU 派生特征存在于当前导出中。",
            semantic_warning=recipe.semantic_warning,
        )

    selected, satisfied = _selected_alternative(recipe, available)
    available_primitives = tuple(value for value in selected if value in available)
    missing_primitives = tuple(value for value in selected if value not in available)
    if satisfied:
        return ConstructAnswerability(
            construct_id=recipe.construct_id,
            label_zh=recipe.label_zh,
            requested_term=requested_term,
            resolution_kind=recipe.resolution_kind,
            source_state="constructible_from_export",
            verdict="needs_review",
            required_primitives=selected,
            available_primitives=available_primitives,
            unresolved_requirements=recipe.unresolved_requirements,
            supported_databases=supported,
            requires_confirmation=True,
            recipe_id=recipe.construct_id,
            recipe_version=recipe.version,
            rationale_zh=(
                "当前导出具备构造所需的基础信号，但派生事件尚未物化；必须先确认临床定义并生成可追溯回执。"
            ),
            semantic_warning=recipe.semantic_warning,
        )

    proxies = tuple(value for value in recipe.proxy_concepts if value in available)
    if proxies:
        return ConstructAnswerability(
            construct_id=recipe.construct_id,
            label_zh=recipe.label_zh,
            requested_term=requested_term,
            resolution_kind="proxy_only",
            source_state="proxy_only",
            verdict="needs_review",
            required_primitives=selected,
            available_primitives=proxies,
            missing_primitives=missing_primitives,
            unresolved_requirements=recipe.unresolved_requirements,
            supported_databases=supported,
            requires_confirmation=True,
            recipe_id=recipe.construct_id,
            recipe_version=recipe.version,
            rationale_zh="当前导出只有相关代理信号，不能把它直接当作目标临床事件。",
            semantic_warning=recipe.semantic_warning,
        )
    return ConstructAnswerability(
        construct_id=recipe.construct_id,
        label_zh=recipe.label_zh,
        requested_term=requested_term,
        resolution_kind="unavailable",
        source_state="not_in_database",
        verdict="blocked",
        required_primitives=selected,
        available_primitives=available_primitives,
        missing_primitives=missing_primitives,
        unresolved_requirements=recipe.unresolved_requirements,
        supported_databases=supported,
        requires_confirmation=True,
        recipe_id=recipe.construct_id,
        recipe_version=recipe.version,
        rationale_zh="当前导出缺少可靠构造该临床事件所需的基础信号。",
        semantic_warning=recipe.semantic_warning,
    )


def _concept_resolution_kind(concept_id: str, database: str) -> ResolutionKind:
    cell = explain_concept_availability(concept=concept_id, database=database)
    if cell.direct_source:
        return "direct_observed"
    if cell.available_dependencies:
        return "validated_derived"
    return "direct_observed"


def _assess_catalog_concept(
    concept_id: str,
    *,
    requested_term: str,
    database: Optional[str],
    available_concepts: Optional[set[str]],
) -> ConstructAnswerability:
    entry = concept_catalog.CONCEPT_DICTIONARY[concept_id]
    label_zh = str(entry[1] if len(entry) > 1 else entry[0])
    supported = tuple(
        db
        for db in PUBLIC_DATABASES
        if explain_concept_availability(concept=concept_id, database=db).available
    )
    if available_concepts is None:
        if database:
            db = normalize_database_name(database)
            cell = explain_concept_availability(concept=concept_id, database=db)
            kind = _concept_resolution_kind(concept_id, db) if cell.available else "unavailable"
            return ConstructAnswerability(
                construct_id=concept_id,
                label_zh=label_zh,
                requested_term=requested_term,
                resolution_kind=kind,
                source_state="database_capability_only" if cell.available else "not_in_database",
                verdict="needs_review" if cell.available else "blocked",
                required_primitives=tuple(cell.available_dependencies),
                missing_primitives=tuple(cell.missing_dependencies),
                supported_databases=supported,
                rationale_zh=(
                    "该数据库的 EasyICU 概念层支持此特征，但尚未检查当前真实导出。"
                    if cell.available
                    else "该数据库的 EasyICU 概念层不支持此特征。"
                ),
            )
        derived = any(
            explain_concept_availability(concept=concept_id, database=db).available_dependencies
            for db in supported
        )
        return ConstructAnswerability(
            construct_id=concept_id,
            label_zh=label_zh,
            requested_term=requested_term,
            resolution_kind="validated_derived" if derived else "direct_observed",
            source_state="source_not_selected" if supported else "not_in_database",
            verdict="needs_review" if supported else "blocked",
            supported_databases=supported,
            rationale_zh=(
                "EasyICU 概念层支持此特征；绑定真实数据后才能确认是否已提取及其覆盖率。"
                if supported
                else "EasyICU 当前没有支持此特征的数据库级定义。"
            ),
        )

    available = {normalize_concept_name(value) for value in available_concepts}
    db = normalize_database_name(database or "")
    cell = (
        explain_concept_availability(concept=concept_id, database=db)
        if db
        else None
    )
    if concept_id in available:
        kind = _concept_resolution_kind(concept_id, db) if db else "direct_observed"
        return ConstructAnswerability(
            construct_id=concept_id,
            label_zh=label_zh,
            requested_term=requested_term,
            resolution_kind=kind,
            source_state="present_in_export",
            verdict="ready",
            required_primitives=tuple(cell.available_dependencies) if cell else (),
            available_primitives=(concept_id,),
            supported_databases=supported,
            materialized=True,
            rationale_zh=(
                "该特征已由注册的 EasyICU 派生规则物化在当前导出中。"
                if kind == "validated_derived"
                else "该特征已直接存在于当前 EasyICU 导出中。"
            ),
        )
    if cell and cell.available:
        return ConstructAnswerability(
            construct_id=concept_id,
            label_zh=label_zh,
            requested_term=requested_term,
            resolution_kind=_concept_resolution_kind(concept_id, db),
            source_state="reextractable",
            verdict="needs_review",
            required_primitives=tuple(cell.available_dependencies),
            missing_primitives=(concept_id,),
            supported_databases=supported,
            rationale_zh="该数据库支持此 EasyICU 特征，但当前导出尚未包含它，需要重新提取或补充模块。",
        )
    return ConstructAnswerability(
        construct_id=concept_id,
        label_zh=label_zh,
        requested_term=requested_term,
        resolution_kind="unavailable",
        source_state="not_in_database",
        verdict="blocked",
        missing_primitives=(concept_id,),
        supported_databases=supported,
        rationale_zh="当前数据库无法通过已注册的 EasyICU 概念定义提供此特征。",
    )


def assess_research_construct(
    term: str,
    *,
    database: Optional[str] = None,
    available_concepts: Optional[Iterable[str]] = None,
) -> ConstructAnswerability:
    """Assess one clinical construct without reading patient rows."""

    requested = str(term or "").strip()
    available = set(available_concepts) if available_concepts is not None else None
    recipe = _recipe_for_term(requested)
    if recipe is not None:
        return _assess_recipe(
            recipe,
            requested_term=requested,
            database=database,
            available_concepts=available,
        )
    concept_id = normalize_concept_name(requested)
    if concept_id in concept_catalog.CONCEPT_DICTIONARY:
        return _assess_catalog_concept(
            concept_id,
            requested_term=requested,
            database=database,
            available_concepts=available,
        )
    return ConstructAnswerability(
        construct_id=concept_id or "unknown_construct",
        label_zh=requested or "未知研究构念",
        requested_term=requested,
        resolution_kind="unavailable",
        source_state="not_in_database",
        verdict="blocked",
        rationale_zh="该研究构念尚未映射到 EasyICU 概念或经审阅的派生规则。",
    )


def _inferred_recipe_ids(text: str) -> tuple[str, ...]:
    normalized = _normalised_text(text)
    matches = [
        recipe.construct_id
        for recipe in _RECIPES
        if any(_matches_alias(normalized, alias) for alias in recipe.aliases)
    ]
    return tuple(dict.fromkeys(matches))


def assess_idea_constructs(
    text: str,
    *,
    mapped_concepts: Sequence[str] = (),
    database: Optional[str] = None,
    available_concepts: Optional[Iterable[str]] = None,
    max_items: int = 8,
) -> list[dict[str, object]]:
    """Return bounded construct verdicts for one Idea Mining candidate."""

    terms: list[str] = list(_inferred_recipe_ids(text))
    terms.extend(
        str(value or "").strip()
        for value in mapped_concepts
        if str(value or "").strip()
    )
    rows = [
        assess_research_construct(
            term,
            database=database,
            available_concepts=available_concepts,
        ).to_dict()
        for term in list(dict.fromkeys(terms))[: max(1, max_items)]
    ]
    return rows


__all__ = [
    "ConstructAnswerability",
    "assess_idea_constructs",
    "assess_research_construct",
]
