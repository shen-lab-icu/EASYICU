"""Governed conversational updates to one StudyContext.

This module owns user-turn authority, bounded patch transformation, source and
concept validation, persistence, and the path-free update receipt.  The Pi tool
registry is only an adapter to this interface.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
)

from easyicu.webserver import agent_pipeline_runs, dataio, sources, study_contexts

from .contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiToolResult,
    ToolExecutionContext,
)
from .projections import (
    bounded_json_projection,
    ensure_safe_projection,
    project_study_context,
    reject_sensitive_message,
)
from .workflow import build_project_workflow_projection


def _result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    owner: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner=owner,
        details=bounded_json_projection(details or {}),
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")
    return ensure_safe_projection(payload)


def _consume_action(
    context: ToolExecutionContext, action: str
) -> Optional[Dict[str, Any]]:
    outcome = context.grant.consume_once(action)
    if outcome == "granted":
        return None
    if outcome == "consumed":
        return _result(
            context,
            status="blocked",
            code="pi_action_grant_consumed",
            summary=f"The one-use {action} grant for this message was already consumed.",
            owner="easyicu.webserver.pi_copilot",
        )
    return _result(
        context,
        status="blocked",
        code="pi_action_authorization_required",
        summary=f"This action requires a one-use {action} grant for the current message.",
        owner="easyicu.webserver.pi_copilot",
    )


def _require_args(
    params: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(params) - allowed_set)
    if unknown:
        raise PiCopilotError(
            "pi_tool_unknown_arguments",
            "The EasyICU tool received unknown arguments.",
            details={"fields": unknown},
        )
    missing = sorted(
        key
        for key in required
        if key not in params or not str(params.get(key) or "").strip()
    )
    if missing:
        raise PiCopilotError(
            "pi_tool_arguments_required",
            "The EasyICU tool is missing required arguments.",
            details={"fields": missing},
        )


def _bound_context(binding: AuthorityBinding) -> Optional[Dict[str, Any]]:
    if binding.study_context_id:
        try:
            return study_contexts.get_context(binding.study_context_id)
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_invalid"),
                "The bound StudyContext could not be loaded.",
                details=exc.detail,
            ) from exc
    return study_contexts.get_active_context()


def _workflow_snapshot(
    context: ToolExecutionContext,
    *,
    study_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    projection = build_project_workflow_projection(
        study_context_id=context.session.binding.study_context_id,
        study_override=study_override,
    )
    return projection.workflow.model_dump(mode="json")


_STUDY_SETUP_FIELDS = frozenset(
    {
        "title",
        "question",
        "purpose",
        "cohort",
        "modules",
        "outcome",
        "primary_exposure",
        "covariates",
        "covariate_selection",
        "covariate_rationales",
        "covariate_temporal_roles",
        "covariate_operationalizations",
        "execution_concepts",
        "analysis_design",
        "sensitivity_specs",
        "time_window",
        "comparator",
        "export_format",
        "analysis_goal",
        "confirmations",
        "bind_active_export",
        "bind_source_id",
    }
)

_NESTED_STUDY_PATCH_FIELDS = frozenset(
    {
        "cohort",
        "time_window",
        "confirmations",
        "execution_concepts",
        "analysis_design",
        "covariate_rationales",
        "covariate_temporal_roles",
        "covariate_operationalizations",
    }
)

_COVARIATE_MODELING_ROLE_TOKENS = frozenset(
    {
        "binary",
        "categorical",
        "continuous",
        "linear",
        "nonlinear",
        "nonlinear_continuous",
        "ordinal",
    }
)


def _message_explicitly_selects_first_stay(message: str) -> bool:
    """Return whether this user turn explicitly chooses one first ICU stay."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"首次\s*(?:icu|重症监护)",
            r"首个\s*(?:icu|重症监护)",
            r"第一(?:次|个)\s*(?:icu|重症监护)",
            r"\bfirst[\s_-]+(?:icu|intensive care)(?:[\s_-]+stay|[\s_-]+admission)?\b",
            r"\bindex[\s_-]+(?:icu|intensive care)(?:[\s_-]+stay|[\s_-]+admission)?\b",
            r"\bone[\s_-]+(?:icu[\s_-]+)?stay[\s_-]+per[\s_-]+patient\b",
            r"排除\s*(?:再次|重复|再入|重返).*?(?:icu|重症监护)",
            r"\bexclude[\s_-]+(?:icu[\s_-]+)?readmissions?\b",
        )
    )


def _message_explicitly_selects_all_stays(message: str) -> bool:
    """Return whether this user turn explicitly retains eligible ICU stays."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"所有符合条件.*?(?:icu|重症监护).*?(?:stay|入住|住院)",
            r"(?:包括|保留|纳入).*?(?:重复|再次|再入).*?(?:icu|重症监护)",
            r"不(?:限制|限于|只纳入).*?(?:首次|首个|第一).*?(?:icu|重症监护)",
            r"\ball[\s_-]+eligible[\s_-]+(?:adult[\s_-]+)?icu[\s_-]+stays?\b",
            r"\b(?:include|retain)[\s_-]+(?:repeated|repeat|readmission)[\s_-]+(?:icu[\s_-]+)?stays?\b",
            r"\bnot[\s_-]+(?:restricted[\s_-]+to[\s_-]+)?(?:the[\s_-]+)?first[\s_-]+icu[\s_-]+stay\b",
            # The canonical wording of the eligibility option that means this.
            # Copilot now puts the choice to the user as a named option
            # (cohort_eligibility.adults_all_admissions); picking it by its own
            # label is an explicit selection, and refusing it here would make
            # the offered answer unusable.
            r"(?:全部|所有).{0,4}(?:icu|重症监护).{0,4}(?:入住|住院)",
            r"\b(?:every|all)[\s_-]+icu[\s_-]+admissions?\b",
        )
    )


def _message_directly_names_label(message: str, proposed: Any) -> bool:
    """Return whether the user's own wording contains one proposed display label.

    These are reader-facing StudyContext labels, not executable concept ids.
    Keeping an exact label stated in the research question gives Planner the
    user's actual intent while the later digest-bound plan review still owns
    executable semantics and approval.
    """

    raw_message = str(message or "").casefold()
    raw_proposed = str(proposed or "").casefold()
    if re.search(r"[\u4e00-\u9fff]", raw_proposed):
        normalized_message = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+", "", raw_message
        )
        normalized_proposed = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+", "", raw_proposed
        )
        return (
            len(normalized_proposed) >= 2
            and normalized_proposed in normalized_message
        )
    normalized_message = re.sub(r"[^a-z0-9]+", " ", raw_message).strip()
    normalized_proposed = re.sub(r"[^a-z0-9]+", " ", raw_proposed).strip()
    return (
        len(normalized_proposed) >= 3
        and f" {normalized_proposed} " in f" {normalized_message} "
    )


def _message_explicitly_selects_primary_outcome(
    message: str, proposed: Any = ""
) -> bool:
    """Return whether this turn chooses a candidate outcome label."""

    if _message_directly_names_label(message, proposed):
        return True

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"主要结局.*?(?:使用|采用|设为|定义为|改为|替换为)",
            r"(?:确认|同意).*?主要结局.*?(?:为|是|采用|使用)",
            r"(?:使用|采用|设为|改为|替换为).*?(?:死亡|mortality).*?(?:主要)?结局",
            r"结局.*?(?:使用|采用|设为|定义为|改为|替换为).*?(?:死亡|mortality)",
            r"不(?:改成|改为|替换为).*?(?:死亡|mortality)",
            # The Copilot renders this compact affirmative choice directly
            # beneath one concrete outcome proposal.  It is an explicit user
            # decision even though the button text avoids repeating the long
            # clinical definition.
            r"^(?:是|确认)[，,\s]*(?:采用|使用|确认)?(?:该|此|上述)?(?:定义|主要结局定义)[。.]?$",
            # A rendered next-step choice is itself an explicit selection even
            # when its button label is a compact noun phrase.  Requiring the
            # user to add a verb after clicking the option makes the same
            # scientific decision repeat indefinitely.
            r"^(?:icu\s*(?:stay\s*)?(?:住院期间|期间|内)?死亡|icu\s*mortality)(?:\s*[（(]推荐[）)])?$",
            r"^(?:hospital|in[\s_-]*hospital|28[\s_-]*day|30[\s_-]*day|90[\s_-]*day)[\s_-]*(?:death|mortality)$",
            r"\bprimary[\s_-]+outcome[\s_-]+(?:is|uses?|will[\s_-]+be)\b",
            r"\buse[\s_-]+.+?[\s_-]+(?:as[\s_-]+the[\s_-]+)?primary[\s_-]+outcome\b",
            r"\bdo[\s_-]+not[\s_-]+change[\s_-]+(?:it[\s_-]+)?to\b",
        )
    )


def _message_explicitly_selects_primary_exposure(
    message: str, proposed: Any = ""
) -> bool:
    """Return whether this turn chooses a candidate exposure label."""

    if _message_directly_names_label(message, proposed):
        return True

    normalized = str(message or "").casefold()
    proposed_label = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+", "", str(proposed or "").casefold()
    )
    modeling_assignment = re.search(
        r"(?:把|将)\s*(?P<label>[a-z0-9\u4e00-\u9fff_-]{2,80}?)\s*"
        r"(?:作为|设为|定义为|按)\s*(?:一?个?)?"
        r"(?:有序|二元|连续|分类).{0,8}变量",
        normalized,
    )
    if modeling_assignment is not None:
        selected_label = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+",
            "",
            modeling_assignment.group("label"),
        )
        if selected_label and selected_label in proposed_label:
            return True
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"主要暴露.*?(?:使用|采用|设为|定义为)",
            r"(?:确认|同意).*?主要暴露.*?(?:为|是|采用|使用)",
            # Rendered decision buttons commonly put the chosen concept before
            # the role: "确认将乳酸水平作为本研究的主要暴露".  That is the
            # same explicit authority as "主要暴露采用乳酸水平" and must not
            # loop back to another confirmation prompt.
            r"(?:确认|同意).*?(?:作为|设为|定义为).*?(?:本研究的)?主要暴露",
            r"(?:使用|采用|设为).*?(?:sepsis|脓毒症).*?(?:定义|作为.*?暴露)",
            r"(?:sepsis|脓毒症).*?(?:定义|版本).*?(?:使用|采用)",
            r"不要使用.*?(?:sofa|sepsis|脓毒症)",
            r"^(?:是|确认)[，,\s]*(?:采用|使用|确认)?(?:该|此|上述)?(?:定义|主要暴露定义)[。.]?$",
            r"\bprimary[\s_-]+exposure[\s_-]+(?:is|uses?|will[\s_-]+be)\b",
            r"\b(?:use|adopt)[\s_-]+.+?[\s_-]+(?:as[\s_-]+the[\s_-]+)?primary[\s_-]+exposure\b",
            r"\bdo[\s_-]+not[\s_-]+use[\s_-]+.+?(?:sofa|sepsis)\b",
        )
    )


def _message_explicitly_selects_clustered_inference(message: str) -> bool:
    """Return whether this turn explicitly chooses patient-clustered inference."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:按|以).*?患者.*?(?:聚类|相关性|相关)",
            r"患者.*?(?:聚类稳健|聚类处理|cluster)",
            r"(?:聚类坐标|cluster.*coordinate).*?(?:患者|patient)",
            r"\bcluster(?:ed|ing)?[\s_-]+(?:robust[\s_-]+)?(?:by[\s_-]+)?patient\b",
            r"\bpatient[\s_-]+cluster(?:ed|ing|[\s_-]+robust)?\b",
        )
    )


def _message_explicitly_changes_variance_estimator(message: str) -> bool:
    """Return whether this turn directly selects an inference variance policy."""

    normalized = str(message or "").casefold()
    return _message_explicitly_selects_clustered_inference(message) or any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:普通|异方差|heteroskedastic).*?稳健标准误",
            r"(?:模型|model)[\s_-]*(?:标准误|based)",
            r"(?:不进行|不做|仅报告).*?(?:关联推断|统计推断|患病率)",
            r"\b(?:model[\s_-]*based|heteroskedastic[\s_-]*robust|counts[\s_-]*only)\b",
        )
    )


def _message_explicitly_selects_analysis_goal(
    message: str, proposed: Any = ""
) -> bool:
    """Return whether this turn directly chooses the scientific analysis goal."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:分析目标|研究目标).*?(?:患病率|关联|因果|预测|描述)",
            r"(?:分析目标|研究目标).*?(?:同步|替换|改为|更新).*?(?:死亡|mortality|患病率|关联)",
            r"(?:计划|输出|结果).{0,24}(?:应|需|需要|必须|要).{0,160}(?:包含|包括|纳入|展示|报告).{0,160}(?:table\s*1|图|figure|趋势|审计)",
            r"(?:估计|报告|计算).*?(?:患病率|比例|分布).*?(?:并|以及|同时).*?(?:描述|比较|展示).*?(?:关系|关联|死亡|结局)",
            r"先报告.*?患病率.*?(?:关联|关系)",
            r"(?:观察性关联|非因果|不要写成因果|不作因果解释|不.*?因果效应)",
            r"^(?:描述|报告).*?患病率.*?(?:未调整|协变量调整|调整后).*?关联(?:分析)?(?:\s*[（(]推荐[）)])?$",
            r"^仅(?:描述|报告).*?患病率.*?(?:不分析|不评估).*?(?:死亡)?关联$",
            r"\b(?:analysis|study)[\s_-]+goal\b",
            r"\b(?:plan|outputs?|results?).{0,48}\b(?:must|should|need(?:s)?\s+to).{0,96}\b(?:include|contain|report|show)\b",
            r"\b(?:observational[\s_-]+association|non[\s_-]*causal)\b",
            r"\bfirst[\s_-]+report[\s_-]+.+?prevalence\b",
        )
    )


def _message_requests_descriptive_plan_default(message: str) -> bool:
    """Return whether a conservative counts-only plan follows from the question.

    This deliberately recognizes only a narrow prevalence-plus-description
    construction and rejects wording that requests an inferential, predictive,
    or causal estimate. The result is still a candidate plan setting: the user
    reviews the final digest-bound plan before any analysis starts.
    """

    normalized = str(message or "").casefold()
    if re.search(
        r"(?:调整后|校正后|回归|风险比|优势比|因果|效应|预测|"
        r"adjusted|regression|hazard|odds|causal|effect|predict)",
        normalized,
    ):
        return False
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:估计|报告|计算).*?(?:患病率|比例|分布).*?(?:并|以及|同时).*?(?:描述|比较|展示).*?(?:关系|死亡|结局|分布)",
            r"\b(?:estimate|report|calculate).+?(?:prevalence|proportion|distribution).+?(?:and|then).+?(?:describe|summari[sz]e|compare).+?(?:relationship|mortality|outcome|distribution)\b",
        )
    )


def _message_explicitly_selects_export_format(message: str) -> bool:
    """Return whether this turn directly chooses a package format."""

    normalized = str(message or "").casefold()
    return bool(
        re.search(
            r"(?:导出|数据包|格式|export).*?(?:parquet|csv|excel|xlsx)|"
            r"(?:parquet|csv|excel|xlsx).*?(?:导出|数据包|格式|export)|"
            r"^(?:parquet|csv|excel|xlsx)(?:\s*[（(].*?[）)])?$",
            normalized,
        )
    )


def _message_explicitly_requests_source_rebind(message: str) -> bool:
    """Return whether this turn explicitly asks to replace the bound source."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:更换|切换|重新选择|改用|换成).{0,24}(?:数据源|数据库|数据目录|文件夹)",
            r"(?:数据源|数据库|数据目录|文件夹).{0,24}(?:更换|切换|重新选择|改用|换成)",
            r"\b(?:switch|change|replace|rebind|use a different).{0,32}(?:data source|database|data folder)\b",
        )
    )


def _immutable_baseline_demographic(value: Any) -> str:
    """Return the owner-known baseline demographic family for one label."""

    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())
    if normalized in {"age", "年龄", "入院年龄"}:
        return "age"
    if normalized in {"sex", "gender", "性别", "生理性别"}:
        return "sex"
    return ""


def _message_explicitly_selects_demographic_adjustment(
    message: str, covariates: Sequence[Any]
) -> bool:
    """Return whether the user directly selected an immutable baseline roster."""

    normalized = str(message or "").casefold()
    families = [_immutable_baseline_demographic(value) for value in covariates]
    if not families or not all(families):
        return False
    if not re.search(r"(?:主要)?调整|协变量|adjust(?:ed|ing)?\s+for|covariates?", normalized):
        return False
    return all(
        (family == "age" and ("年龄" in normalized or re.search(r"\bage\b", normalized)))
        or (
            family == "sex"
            and (
                "性别" in normalized
                or re.search(r"\b(?:sex|gender)\b", normalized)
            )
        )
        for family in families
    )


def _message_explicitly_clears_covariate_adjustment(message: str) -> bool:
    """Return whether the user directly removes the adjustment roster.

    Empty model arguments are not authority to erase a prior scientific
    decision.  This helper recognizes only turns that explicitly choose an
    unadjusted or no-covariate analysis, so StudyContext can clear the roster
    and its owner metadata atomically without weakening the ordinary
    fail-closed merge behavior.
    """

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:清除|移除|删除|取消).{0,24}(?:协变量|调整(?:登记|配置|变量)?)",
            r"(?:协变量|调整(?:登记|配置|变量)?).{0,24}(?:清除|移除|删除|取消)",
            r"(?:无|不使用|不登记|不保留|不进行|不执行|不做).{0,16}(?:调整协变量|协变量调整|协变量|调整)",
            r"(?:仅|只).{0,16}(?:描述|描述性).{0,16}(?:不调整|无调整|不含协变量)",
            r"\b(?:clear|remove|drop|delete).{0,32}(?:adjustment|covariates?)\b",
            r"\b(?:no|without).{0,16}(?:adjustment|adjustment covariates?|covariates?)\b",
            r"\b(?:unadjusted|descriptive-only|descriptive only)\b",
            r"\bdo not adjust\b",
        )
    )


def _merge_nested_study_patch(
    current: Mapping[str, Any], patch: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge bounded conversational object patches without erasing siblings.

    Pi tool arguments naturally contain only the slot the user just changed.
    StudyContext persists complete nested value objects.  Replacing the whole
    object would let ``cohort.exclude_readmissions=true`` silently erase age,
    comparator, and cohort-review authority.  Lists and scalar leaves still
    replace exactly; only mappings merge recursively.
    """

    def merge(existing: Mapping[str, Any], proposed: Mapping[str, Any]) -> Dict[str, Any]:
        combined = dict(existing)
        for key, value in proposed.items():
            prior = combined.get(key)
            if isinstance(prior, Mapping) and isinstance(value, Mapping):
                combined[key] = merge(prior, value)
            else:
                combined[key] = value
        return combined

    merged = dict(patch)
    for field in _NESTED_STUDY_PATCH_FIELDS:
        if field not in merged:
            continue
        existing = current.get(field)
        proposed = merged.get(field)
        if isinstance(existing, Mapping) and isinstance(proposed, Mapping):
            merged[field] = merge(existing, proposed)
    return merged


_GATED_SLOT_OMISSION_CODES = {
    "cohort": "study_cohort_preset_confirmation_required",
    "outcome": "study_primary_outcome_confirmation_required",
    "primary_exposure": "study_primary_exposure_confirmation_required",
    "analysis_goal": "study_analysis_goal_confirmation_required",
}


def _unconfirmed_gated_slots(
    params: Mapping[str, Any],
    current: Mapping[str, Any],
    user_message: str,
) -> FrozenSet[str]:
    """Gated slots this turn proposes but does not explicitly select.

    Each confirmation guard below keeps a genuinely confirmed change in the
    same call and drops only its own unconfirmed slot.  That test must not
    read a *sibling* candidate slot as the user's confirmation: one research
    question mentioning Sepsis-3 and ICU mortality proposes outcome,
    primary_exposure and analysis_goal together, and letting each one vouch
    for the others made all three take the silent-drop branch, so none could
    ever be saved and none could ever reach its typed
    ``*_confirmation_required`` receipt.
    """

    message = str(user_message or "").strip()
    if not message:
        return frozenset()
    current = current or {}
    unconfirmed: set[str] = set()

    proposed_cohort = params.get("cohort")
    if isinstance(proposed_cohort, Mapping) and "preset" in proposed_cohort:
        proposed_preset = str(proposed_cohort.get("preset") or "").strip().lower()
        current_preset = str((current.get("cohort") or {}).get("preset") or "").strip()
        if (
            proposed_preset
            and proposed_preset != current_preset
            and not _message_explicitly_selects_all_stays(message)
            and not _message_explicitly_selects_first_stay(message)
        ):
            unconfirmed.add("cohort")

    for slot, selects_explicitly in (
        ("outcome", _message_explicitly_selects_primary_outcome),
        ("primary_exposure", _message_explicitly_selects_primary_exposure),
        ("analysis_goal", _message_explicitly_selects_analysis_goal),
    ):
        proposed = str(params.get(slot) or "").strip()
        if not proposed or proposed == str(current.get(slot) or "").strip():
            continue
        if not selects_explicitly(message, proposed):
            unconfirmed.add(slot)

    return frozenset(unconfirmed)


def _has_other_confirmed_change(
    params: Mapping[str, Any],
    *,
    slot_machinery: Set[str],
    unconfirmed_gated: FrozenSet[str],
) -> bool:
    """Whether this call also carries a change the user actually confirmed."""

    return any(
        key in params
        for key in _STUDY_SETUP_FIELDS - slot_machinery - unconfirmed_gated
    )


def _restore_unconfirmed_study_slot(
    patch: Dict[str, Any],
    current: Mapping[str, Any],
    *,
    slot: str,
) -> None:
    """Discard one unconfirmed proposal while retaining confirmed changes."""

    prior = current.get(slot)
    if prior:
        patch[slot] = prior
    else:
        patch.pop(slot, None)

    if slot in {"outcome", "primary_exposure"}:
        proposed_execution = patch.get("execution_concepts")
        prior_execution = current.get("execution_concepts")
        if isinstance(proposed_execution, Mapping):
            restored_execution = dict(proposed_execution)
            prior_value = (
                prior_execution.get(slot)
                if isinstance(prior_execution, Mapping)
                else None
            )
            if prior_value:
                restored_execution[slot] = prior_value
            else:
                restored_execution.pop(slot, None)
            if restored_execution:
                patch["execution_concepts"] = restored_execution
            else:
                patch.pop("execution_concepts", None)
        if "modules" in patch:
            prior_modules = list(current.get("modules") or [])
            if prior_modules:
                patch["modules"] = prior_modules
            else:
                patch.pop("modules", None)


def update_study_context(
    context: ToolExecutionContext,
    params: Mapping[str, Any],
    *,
    load_context: Callable[[AuthorityBinding], Optional[Dict[str, Any]]] = _bound_context,
    project_workflow: Callable[..., Dict[str, Any]] = _workflow_snapshot,
) -> Dict[str, Any]:
    """Persist conversational setup through the existing typed owner."""

    # Argument names are the catalog's to declare; execute_tool has already
    # enforced them for this tool. _STUDY_SETUP_FIELDS below stays as the
    # *patch* vocabulary, which is a different question from what the
    # model may send.
    binding = context.session.binding
    current = load_context(binding)
    if binding.study_context_id and current is None:
        return _result(
            context,
            status="not_found",
            code="study_context_not_found",
            summary="The bound StudyContext no longer exists; no replacement context was created.",
            owner="easyicu.webserver.study_contexts",
        )
    if current is None and isinstance(params.get("cohort"), Mapping):
        return _result(
            context,
            status="blocked",
            code="cohort_eligibility_study_context_required",
            summary=(
                "The host must allocate and bind a StudyContext before a cohort "
                "proposal can be saved or confirmed."
            ),
            owner="easyicu.webserver.study_contexts",
        )
    if current and current.get("active_job_id"):
        return _result(
            context,
            status="blocked",
            code="study_context_active_job_conflict",
            summary="Study setup cannot change while its authoritative EasyICU job is active.",
            owner="easyicu.webserver.study_contexts",
        )

    patch = {
        key: params[key]
        for key in _STUDY_SETUP_FIELDS - {"bind_active_export", "bind_source_id"}
        if key in params
    }
    if isinstance(patch.get("sensitivity_specs"), list):
        patch["sensitivity_specs"] = [
            {key: value for key, value in spec.items() if value is not None}
            if isinstance(spec, Mapping)
            else spec
            for spec in patch["sensitivity_specs"]
        ]
    omitted_unconfirmed_fields: list[str] = []
    unconfirmed_omissions: list[Dict[str, str]] = []
    unconfirmed_gated = _unconfirmed_gated_slots(
        params, current or {}, context.user_message
    )

    def _record_unconfirmed_omission(slot: str, field: str) -> None:
        """Make one silently dropped slot visible in the typed receipt."""

        if field in omitted_unconfirmed_fields:
            return
        omitted_unconfirmed_fields.append(field)
        unconfirmed_omissions.append(
            {"field": field, "code": _GATED_SLOT_OMISSION_CODES[slot]}
        )

    if current:
        patch = _merge_nested_study_patch(current, patch)
    if current and current.get("id"):
        patch["id"] = current["id"]
    proposed_modules = {
        str(value or "").strip().lower() for value in (params.get("modules") or [])
    }
    if "sepsis3_sofa2" in proposed_modules:
        from easyicu.concept.selection_policy import (
            concept_selection_confirmation_key,
            evaluate_concept_selection,
        )

        concept_id = "sep3_sofa2"
        confirmation_key = concept_selection_confirmation_key(concept_id)
        previously_confirmed = bool(
            ((current or {}).get("confirmations") or {}).get(confirmation_key)
        )
        decision = evaluate_concept_selection(
            concept_id,
            user_intent=context.user_message,
            owner_confirmed=previously_confirmed,
        )
        if not decision.allowed:
            return _result(
                context,
                status="blocked",
                code=decision.reason_code,
                summary=(
                    "The proposed feature module contains an explicit-only "
                    "clinical variant that the user did not request."
                ),
                owner="easyicu.concept.selection_policy",
                details={
                    **decision.to_dict(),
                    "field": "modules",
                    "proposed_module": "sepsis3_sofa2",
                    "canonical_alternative_module": "sepsis3_sofa1",
                },
            )
        confirmations = dict(((current or {}).get("confirmations") or {}))
        confirmations.update(dict(patch.get("confirmations") or {}))
        confirmations[confirmation_key] = True
        patch["confirmations"] = confirmations
    proposed_cohort = params.get("cohort")
    if isinstance(proposed_cohort, Mapping) and "preset" in proposed_cohort:
        try:
            normalized_preset = dataio.normalize_export_cohort_preset(
                proposed_cohort.get("preset")
            )
        except dataio.ExportCohortError as exc:
            return _result(
                context,
                status="blocked",
                code=exc.error,
                summary=(
                    "Use one canonical Data Extraction cohort preset and keep "
                    "the user's natural-language cohort wording in label or review."
                ),
                owner="easyicu.webserver.dataio",
                details={
                    "field": "cohort.preset",
                    "allowed": list(exc.detail.get("supported") or []),
                },
            )
        normalized_cohort = dict(patch.get("cohort") or {})
        normalized_cohort["preset"] = normalized_preset
        patch["cohort"] = normalized_cohort
        current_preset = str(((current or {}).get("cohort") or {}).get("preset") or "")
        if (
            normalized_preset == "adult_all"
            and current_preset != "adult_all"
            and str(context.user_message or "").strip()
            and not _message_explicitly_selects_all_stays(context.user_message)
        ):
            other_confirmed_change = _has_other_confirmed_change(
                params,
                slot_machinery={
                    "cohort",
                    "execution_concepts",
                    "modules",
                    "confirmations",
                },
                unconfirmed_gated=unconfirmed_gated,
            )
            if other_confirmed_change:
                _restore_unconfirmed_study_slot(patch, current or {}, slot="cohort")
                _record_unconfirmed_omission("cohort", "cohort.preset")
            else:
                return _result(
                    context,
                    status="blocked",
                    code="study_cohort_all_stays_confirmation_required",
                    summary=(
                        "An adult ICU population does not by itself choose between "
                        "all eligible stays and one stay per patient."
                    ),
                    owner="easyicu.webserver.study_contexts",
                    details={
                        "field": "cohort.preset",
                        "proposed": "adult_all",
                        "safe_alternatives": ["adult_all", "adult_first"],
                    },
                )
        if (
            normalized_preset == "adult_first"
            and current_preset != "adult_first"
            and not _message_explicitly_selects_first_stay(context.user_message)
        ):
            other_confirmed_change = _has_other_confirmed_change(
                params,
                slot_machinery={
                    "cohort",
                    "execution_concepts",
                    "modules",
                    "confirmations",
                },
                unconfirmed_gated=unconfirmed_gated,
            )
            if other_confirmed_change:
                _restore_unconfirmed_study_slot(patch, current or {}, slot="cohort")
                _record_unconfirmed_omission("cohort", "cohort.preset")
            else:
                return _result(
                    context,
                    status="blocked",
                    code="study_cohort_first_stay_confirmation_required",
                    summary=(
                        "A first-ICU-stay restriction changes the analysis unit and "
                        "requires the user's explicit selection."
                    ),
                    owner="easyicu.webserver.study_contexts",
                    details={
                        "field": "cohort.preset",
                        "proposed": "adult_first",
                        "safe_alternatives": ["adult_all", "all_icu"],
                    },
                )
    if (
        params.get("outcome")
        and str(context.user_message or "").strip()
        and str(params.get("outcome") or "").strip()
        != str((current or {}).get("outcome") or "").strip()
        and not _message_explicitly_selects_primary_outcome(
            context.user_message, params.get("outcome")
        )
    ):
        other_confirmed_change = _has_other_confirmed_change(
            params,
            slot_machinery={
                "outcome",
                "execution_concepts",
                "modules",
                "confirmations",
            },
            unconfirmed_gated=unconfirmed_gated,
        )
        if other_confirmed_change:
            _restore_unconfirmed_study_slot(patch, current or {}, slot="outcome")
            _record_unconfirmed_omission("outcome", "outcome")
        else:
            return _result(
                context,
                status="blocked",
                code="study_primary_outcome_confirmation_required",
                summary=(
                    "Mentioning an outcome in the research question records candidate "
                    "intent but does not confirm the primary outcome definition."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "outcome"},
            )
    if (
        params.get("primary_exposure")
        and str(context.user_message or "").strip()
        and str(params.get("primary_exposure") or "").strip()
        != str((current or {}).get("primary_exposure") or "").strip()
        and not _message_explicitly_selects_primary_exposure(
            context.user_message, params.get("primary_exposure")
        )
    ):
        other_confirmed_change = _has_other_confirmed_change(
            params,
            slot_machinery={
                "primary_exposure",
                "execution_concepts",
                "modules",
                "confirmations",
            },
            unconfirmed_gated=unconfirmed_gated,
        )
        if other_confirmed_change:
            _restore_unconfirmed_study_slot(
                patch, current or {}, slot="primary_exposure"
            )
            _record_unconfirmed_omission("primary_exposure", "primary_exposure")
        else:
            return _result(
                context,
                status="blocked",
                code="study_primary_exposure_confirmation_required",
                summary=(
                    "Mentioning an exposure in the research question records candidate "
                    "intent but does not confirm its primary executable definition."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "primary_exposure"},
            )
    if (
        params.get("analysis_goal")
        and str(context.user_message or "").strip()
        and str(params.get("analysis_goal") or "").strip()
        != str((current or {}).get("analysis_goal") or "").strip()
        and not _message_explicitly_selects_analysis_goal(context.user_message)
    ):
        other_confirmed_change = _has_other_confirmed_change(
            params,
            slot_machinery={"analysis_goal"},
            unconfirmed_gated=unconfirmed_gated,
        )
        if other_confirmed_change:
            _restore_unconfirmed_study_slot(
                patch, current or {}, slot="analysis_goal"
            )
            _record_unconfirmed_omission("analysis_goal", "analysis_goal")
        else:
            return _result(
                context,
                status="blocked",
                code="study_analysis_goal_confirmation_required",
                summary=(
                    "The research question suggests candidate objectives but does "
                    "not confirm the analysis goal or causal interpretation."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "analysis_goal"},
            )
    current_design = (current or {}).get("analysis_design")
    if (
        "analysis_design" not in params
        and (not isinstance(current_design, Mapping) or not current_design)
        and _message_requests_descriptive_plan_default(context.user_message)
    ):
        # A question that explicitly asks for prevalence plus description has
        # already chosen the conservative scientific ceiling. Persist that
        # owner-interpreted default in the candidate setup so Planner does not
        # manufacture an association study and then ask the user to undo it.
        patch["analysis_design"] = {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        }
        confirmations = dict(((current or {}).get("confirmations") or {}))
        confirmations.update(dict(patch.get("confirmations") or {}))
        confirmations.update(
            {
                "plan_timing_landmark_24h": False,
                "plan_timing_descriptive_only": True,
                "plan_timing_time_varying": False,
            }
        )
        patch["confirmations"] = confirmations
    proposed_covariates = list(params.get("covariates") or [])
    explicitly_clears_covariates = (
        "covariates" in params
        and not proposed_covariates
        and _message_explicitly_clears_covariate_adjustment(context.user_message)
    )
    if explicitly_clears_covariates:
        # Recursive StudyContext merging intentionally preserves omitted
        # nested siblings.  An explicit empty roster is different: the roster,
        # its rationales/temporal roles, and executable bindings form one owner
        # contract and must be cleared together.  The user-message gate above
        # prevents a model-authored empty list from silently erasing a prior
        # scientific decision.
        patch["covariates"] = []
        patch["covariate_selection"] = "exact"
        patch["covariate_authority"] = "user"
        patch["covariate_rationales"] = {}
        patch["covariate_temporal_roles"] = {}
        patch["covariate_operationalizations"] = {}
        prior_execution = (current or {}).get("execution_concepts")
        proposed_execution = patch.get("execution_concepts")
        if isinstance(proposed_execution, Mapping) or isinstance(
            prior_execution, Mapping
        ):
            synchronized_execution = dict(
                proposed_execution
                if isinstance(proposed_execution, Mapping)
                else prior_execution
            )
            synchronized_execution["covariates"] = []
            patch["execution_concepts"] = synchronized_execution
    elif _message_explicitly_selects_demographic_adjustment(
        context.user_message, proposed_covariates
    ):
        # Age and sex are immutable baseline demographics. Once the user
        # explicitly chooses them for adjustment, their temporal role is an
        # EasyICU-owned semantic fact rather than another user decision.
        patch["covariate_selection"] = "exact"
        patch["covariate_authority"] = "user"
        rationales = dict(patch.get("covariate_rationales") or {})
        temporal_roles = dict(patch.get("covariate_temporal_roles") or {})
        for covariate in proposed_covariates:
            key = str(covariate)
            family = _immutable_baseline_demographic(covariate)
            rationales[key] = (
                f"Pre-specified {family} baseline demographic confounder selected by the user."
            )
            temporal_roles[key] = "baseline_static"
        patch["covariate_rationales"] = rationales
        patch["covariate_temporal_roles"] = temporal_roles
        # If this study already has executable concept coordinates, keep the
        # exact user-approved demographic roster synchronized with them. Age
        # and sex are owner-known catalog identifiers, so leaving a stale
        # ``execution_concepts.covariates=[]`` would make the visible study
        # configuration disagree with the ResearchContext and fail only after
        # a background run starts.
        prior_execution = (current or {}).get("execution_concepts")
        proposed_execution = patch.get("execution_concepts")
        if isinstance(proposed_execution, Mapping) or isinstance(
            prior_execution, Mapping
        ):
            synchronized_execution = dict(
                proposed_execution
                if isinstance(proposed_execution, Mapping)
                else prior_execution
            )
            synchronized_execution["covariates"] = list(proposed_covariates)
            patch["execution_concepts"] = synchronized_execution
    if params.get("export_format") and _message_explicitly_selects_export_format(
        context.user_message
    ):
        # An export-format turn authorizes exactly one new confirmation. Do not
        # let a model-authored partial confirmations object erase prior owner
        # receipts such as the already-confirmed feature window.
        confirmations = dict(((current or {}).get("confirmations") or {}))
        confirmations["export_format"] = True
        patch["confirmations"] = confirmations
    for field in (
        "title",
        "question",
        "purpose",
        "outcome",
        "primary_exposure",
        "comparator",
        "analysis_goal",
    ):
        if patch.get(field):
            reject_sensitive_message(str(patch[field]))
    for covariate in patch.get("covariates") or []:
        reject_sensitive_message(str(covariate))
    for rationale in (patch.get("covariate_rationales") or {}).values():
        reject_sensitive_message(str(rationale))
    operationalizations = patch.get("covariate_operationalizations") or {}
    invalid_operationalizations = sorted(
        str(key)
        for key, value in operationalizations.items()
        if str(value or "").strip().casefold() in _COVARIATE_MODELING_ROLE_TOKENS
    )
    if invalid_operationalizations:
        return _result(
            context,
            status="blocked",
            code="study_covariate_operationalization_requires_column",
            summary=(
                "A covariate operationalization must be an exact materialized "
                "analysis column, not a modeling role."
            ),
            owner="easyicu.webserver.study_contexts",
            details={
                "field": "covariate_operationalizations",
                "covariates": invalid_operationalizations,
                "reason": "modeling_role_is_not_a_materialized_column",
            },
        )
    for operational in operationalizations.values():
        reject_sensitive_message(str(operational))
    for spec in patch.get("sensitivity_specs") or []:
        if isinstance(spec, Mapping):
            for variable in spec.get("execution_variables") or []:
                reject_sensitive_message(str(variable))

    bind_source_id = str(params.get("bind_source_id") or "").strip()
    if params.get("bind_active_export") or bind_source_id:
        registry = sources.load_registry()
        if bind_source_id:
            source = next(
                (
                    row
                    for row in (registry.get("sources") or [])
                    if isinstance(row, Mapping)
                    and row.get("ok")
                    and str(row.get("id") or "") == bind_source_id
                ),
                None,
            )
            if source is None:
                return _result(
                    context,
                    status="blocked",
                    code="pi_data_source_not_registered",
                    summary=(
                        "The selected source id is not a validated registered "
                        "EasyICU export. List sources again before binding."
                    ),
                    owner="easyicu.webserver.sources",
                )
            selected_path = str(source.get("path") or "").strip()
        else:
            selected_path = str(registry.get("active_path") or "").strip()
            source = next(
                (
                    row
                    for row in (registry.get("sources") or [])
                    if isinstance(row, Mapping)
                    and row.get("ok")
                    and str(row.get("path") or "") == selected_path
                ),
                None,
            )
        if not selected_path or source is None:
            return _result(
                context,
                status="blocked",
                code="no_active_export",
                summary="No validated active EasyICU export is available to bind.",
                owner="easyicu.webserver.sources",
            )
        database = str(source.get("database") or "").strip()
        if not database:
            return _result(
                context,
                status="blocked",
                code="pi_data_source_database_unavailable",
                summary=(
                    "The selected registered export has no validated database "
                    "identity and cannot be bound for extraction."
                ),
                owner="easyicu.webserver.sources",
            )
        current_source = (current or {}).get("data_source")
        current_source = (
            current_source if isinstance(current_source, Mapping) else {}
        )
        current_path = str(current_source.get("path") or "").strip()
        source_rebind_blocked = bool(
            context.session.data_source_authorization.status == "confirmed"
            and current_path
            and current_path != selected_path
            and not _message_explicitly_requests_source_rebind(context.user_message)
        )
        if source_rebind_blocked:
            has_other_setup_change = any(
                key in params
                for key in _STUDY_SETUP_FIELDS
                - {"bind_active_export", "bind_source_id"}
            )
            if not has_other_setup_change:
                return _result(
                    context,
                    status="blocked",
                    code="study_data_source_rebind_confirmation_required",
                    summary=(
                        "This conversation already has a confirmed data source. "
                        "Changing it requires an explicit user request."
                    ),
                    owner="easyicu.webserver.study_contexts",
                    details={"field": "data_source"},
                )
        else:
            patch["data_source"] = {
                "path": selected_path,
                "label": source.get("label") or "active EasyICU export",
                "database": database,
            }
            confirmations = dict(((current or {}).get("confirmations") or {}))
            confirmations.update(dict(patch.get("confirmations") or {}))
            # A validated registry row is the Data Extraction owner's receipt
            # for an already prepared EasyICU export. Reusing it must not send
            # the user back to the raw-database folder workflow.
            confirmations["extraction_completed"] = True
            patch["confirmations"] = confirmations
    effective_source = patch.get("data_source") or (current or {}).get("data_source")
    effective_source = effective_source if isinstance(effective_source, Mapping) else {}
    effective_modules = patch.get("modules")
    if effective_modules is None:
        effective_modules = (current or {}).get("modules") or []
    execution = patch.get("execution_concepts")
    if execution is not None:
        try:
            normalized_execution = study_contexts.normalize_execution_concepts(
                execution
            )
        except study_contexts.StudyContextError as exc:
            return _result(
                context,
                status="blocked",
                code=str(exc.detail.get("error") or "study_execution_concepts_invalid"),
                summary="The StudyContext owner rejected the executable concept binding.",
                owner="easyicu.webserver.study_contexts",
                details={
                    key: exc.detail.get(key)
                    for key in ("field", "fields", "max_items")
                    if exc.detail.get(key) is not None
                },
            )
        source_path = str(effective_source.get("path") or "").strip()
        if not source_path:
            return _result(
                context,
                status="blocked",
                code="study_execution_source_required",
                summary="Bind a validated data source before saving executable concept identifiers.",
                owner="easyicu.webserver.study_contexts",
            )
        from easyicu.research_agent.acquisition.catalog import build_available_catalog

        catalog = build_available_catalog(Path(source_path).expanduser())
        allowed_modules = {
            str(value).strip().lower()
            for value in effective_modules
            if str(value).strip()
        }
        concept_modules = {
            concept.concept_id: Path(concept.file_name).stem.lower()
            for concept in catalog.concepts
        }
        bound_ids = [
            value
            for value in (
                normalized_execution.get("outcome"),
                normalized_execution.get("primary_exposure"),
                *(normalized_execution.get("covariates") or []),
            )
            if value
        ]
        absent_from_source = sorted(set(bound_ids) - set(concept_modules))
        if absent_from_source:
            return _result(
                context,
                status="blocked",
                code="study_execution_concepts_unavailable",
                summary="One or more executable concepts are absent from the selected source.",
                owner="easyicu.research_agent.acquisition.catalog",
                details={"unavailable_concepts": absent_from_source},
            )
        required_modules = {concept_modules[value] for value in bound_ids}
        if required_modules - allowed_modules:
            # Module selection for already-authorized exact concepts is an
            # EasyICU implementation detail, not another scientific choice.
            # Add only the catalog-proven owning modules; never invent a module
            # name or broaden to unrelated features.
            patch["modules"] = sorted(allowed_modules | required_modules)
            effective_modules = patch["modules"]
            allowed_modules = set(effective_modules)
        primary_concept = normalized_execution.get("primary_exposure")
        if primary_concept:
            from easyicu.concept.selection_policy import (
                concept_selection_confirmation_key,
                evaluate_concept_selection,
            )

            # The model must not self-authorize an experimental concept by
            # putting its name in a generated exposure label or analysis goal.
            # The persisted scientific question is the user-intent authority.
            # The current user message is the strongest authority.  Do not
            # combine it with a model-authored replacement question: phrases
            # such as "do not filter the cohort by Sepsis" can otherwise be
            # misread as negating the separately explicit SOFA-2 selection.
            # For older already-persisted contexts, the prior scientific
            # question remains a compatibility fallback.
            authority_intent = str(
                context.user_message or (current or {}).get("question") or ""
            )
            confirmation_key = concept_selection_confirmation_key(primary_concept)
            confirmations = dict(((current or {}).get("confirmations") or {}))
            confirmations.update(dict(patch.get("confirmations") or {}))
            previously_confirmed = bool(
                ((current or {}).get("confirmations") or {}).get(confirmation_key)
            )
            decision = evaluate_concept_selection(
                primary_concept,
                user_intent=authority_intent,
                owner_confirmed=previously_confirmed,
            )
            if not decision.allowed:
                return _result(
                    context,
                    status="blocked",
                    code=decision.reason_code,
                    summary=(
                        "The selected primary concept is an explicit-only "
                        "variant that the user's research intent did not request."
                    ),
                    owner="easyicu.concept.selection_policy",
                    details=decision.to_dict(),
                )
            # Persist only the host-verified decision.  A model-proposed
            # confirmation cannot authorize itself because it is ignored
            # above; the current user turn or a prior owner receipt must pass.
            confirmations[confirmation_key] = True
            patch["confirmations"] = confirmations
        patch["execution_concepts"] = normalized_execution
    if "analysis_design" in patch:
        proposed_design = patch.get("analysis_design")
        proposed_design = (
            proposed_design if isinstance(proposed_design, Mapping) else {}
        )
        requested_design = params.get("analysis_design")
        requested_design = (
            requested_design if isinstance(requested_design, Mapping) else {}
        )
        current_design = (current or {}).get("analysis_design")
        current_design = (
            current_design if isinstance(current_design, Mapping) else {}
        )
        if (
            requested_design.get("variance_estimator") == "none_counts_only"
            and _message_explicitly_changes_variance_estimator(
                context.user_message
            )
        ):
            # ``cluster_unit`` is meaningful only for clustered inference.
            # The nested merge may retain the previous patient coordinate, so
            # remove it when the user explicitly switches to counts-only
            # description instead of forcing a second technical decision.
            proposed_design = dict(proposed_design)
            proposed_design.pop("cluster_unit", None)
            patch["analysis_design"] = proposed_design
        if (
            current_design
            and str(context.user_message or "").strip()
            and not _message_explicitly_changes_variance_estimator(
                context.user_message
            )
        ):
            proposed_design = dict(proposed_design)
            for field in ("variance_estimator", "cluster_unit"):
                prior_value = current_design.get(field)
                if prior_value:
                    proposed_design[field] = prior_value
                else:
                    proposed_design.pop(field, None)
            patch["analysis_design"] = proposed_design
        proposes_patient_clustering = (
            str(proposed_design.get("variance_estimator") or "").strip()
            == "cluster_robust"
            or str(proposed_design.get("cluster_unit") or "").strip() == "patient"
        )
        current_patient_clustering = (
            str(current_design.get("variance_estimator") or "").strip()
            == "cluster_robust"
            or str(current_design.get("cluster_unit") or "").strip() == "patient"
        )
        if (
            proposes_patient_clustering
            and not current_patient_clustering
            and str(context.user_message or "").strip()
            and not _message_explicitly_selects_clustered_inference(
                context.user_message
            )
        ):
            return _result(
                context,
                status="blocked",
                code="study_patient_clustering_confirmation_required",
                summary=(
                    "Retaining repeated ICU stays does not by itself authorize "
                    "patient-clustered inference."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "analysis_design.variance_estimator"},
            )
        try:
            patch["analysis_design"] = study_contexts.normalize_analysis_design(
                patch.get("analysis_design")
            )
        except study_contexts.StudyContextError as exc:
            return _result(
                context,
                status="blocked",
                code=str(exc.detail.get("error") or "study_analysis_design_invalid"),
                summary="The StudyContext owner rejected the proposed analysis design.",
                owner="easyicu.webserver.study_contexts",
                details={
                    key: exc.detail.get(key)
                    for key in ("field", "fields", "allowed")
                    if exc.detail.get(key) is not None
                },
            )
    if "covariate_selection" in patch:
        selection = str(patch.get("covariate_selection") or "").strip()
        if selection not in {"planner_selectable", "exact"}:
            return _result(
                context,
                status="blocked",
                code="study_covariate_selection_invalid",
                summary="The adjustment-set authority must be planner_selectable or exact.",
                owner="easyicu.webserver.study_contexts",
                details={
                    "field": "covariate_selection",
                    "allowed": ["exact", "planner_selectable"],
                },
            )
        patch["covariate_selection"] = selection
    if not patch:
        raise PiCopilotError(
            "pi_tool_arguments_required",
            "At least one typed study-setup field is required.",
        )

    # Validate the complete proposed StudyContext before spending the one-use
    # Configure grant.  A rejected typed proposal is not a mutation; consuming
    # the grant here used to prevent Pi from correcting a mechanical schema
    # error in the same turn.
    try:
        patch = study_contexts.validate_context_update(
            patch,
            current_context=current,
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        return _result(
            context,
            status="blocked",
            code=str(exc.detail.get("error") or "study_context_update_blocked"),
            summary="The typed StudyContext owner rejected the proposed setup update.",
            owner="easyicu.webserver.study_contexts",
            details={
                key: exc.detail.get(key)
                for key in (
                    "error",
                    "field",
                    "fields",
                    "reason",
                    "allowed",
                    "required_design",
                    "alternative",
                )
                if exc.detail.get(key) is not None
            },
        )

    if "analysis_design" in params:
        proposed = {**dict(current or {}), **patch}
        try:
            agent_pipeline_runs.validate_analysis_design_for_execution(proposed)
        except agent_pipeline_runs.ResearchPipelineRunError as exc:
            return _result(
                context,
                status="blocked",
                code=exc.code,
                summary=str(exc),
                owner="easyicu.webserver.agent_pipeline_runs.analysis_design",
                details=exc.details,
            )

    grant_block = _consume_action(context, "configure")
    if grant_block is not None:
        return grant_block

    try:
        updated = study_contexts.upsert_context(
            patch,
            active=True,
            expected_revision=(int(current.get("revision") or 0) if current else None),
            require_revision=bool(current),
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        return _result(
            context,
            status="blocked",
            code=str(exc.detail.get("error") or "study_context_update_blocked"),
            summary="The typed StudyContext owner rejected the proposed setup update.",
            owner="easyicu.webserver.study_contexts",
            details={
                key: exc.detail.get(key)
                for key in (
                    "error",
                    "field",
                    "fields",
                    "reason",
                    "allowed",
                    "required_design",
                    "alternative",
                    "expected_revision",
                    "current_revision",
                )
                if exc.detail.get(key) is not None
            },
        )
    workflow = project_workflow(context, study_override=updated)
    summary = (
        f"Saved typed StudyContext revision {int(updated.get('revision') or 0)} "
        "and projected the post-update workflow for the next scientific decision."
    )
    if omitted_unconfirmed_fields:
        # Preserve the omission and reason, but let the workflow decide when
        # a choice is needed. Execution requirements must not become an
        # opening questionnaire ahead of the candidate Planner.
        summary += (
            " NOT saved this turn: "
            + ", ".join(omitted_unconfirmed_fields)
            + ". Those slots stay unset because the turn only mentions them as "
            "candidate intent, not approved design. Follow the returned "
            "workflow next action: unresolved design belongs in the candidate "
            "plan for review, not a pre-plan confirmation questionnaire."
        )
    result = _result(
        context,
        status="ok",
        code="study_context_updated",
        summary=summary,
        owner="easyicu.webserver.study_contexts",
        details={
            "study": project_study_context(updated),
            "workflow": workflow,
            "rebind_required": True,
            "host_rebind_after_turn": True,
            "omitted_unconfirmed_fields": omitted_unconfirmed_fields,
            "unconfirmed_omissions": unconfirmed_omissions,
        },
    )
    context.invalidate_authority("study_context_updated")
    return result

__all__ = ["update_study_context"]
