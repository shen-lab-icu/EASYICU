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

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

from easyicu.webserver import study_contexts


SCHEMA_VERSION = "easyicu.pi-cohort-eligibility/2"
AUTHORITY_SCHEMA_VERSION = "easyicu.cohort-eligibility-authority/1"

#: StudyContext owns the fields that alter the primary denominator. Copilot and
#: the pipeline both import this exact tuple; no mirrored roster is permitted.
ELIGIBILITY_FIELDS = study_contexts.COHORT_ELIGIBILITY_FIELDS


class CohortEligibilityOptionError(ValueError):
    """Typed failure for a stale or unknown UI/model option id."""

    code = "cohort_eligibility_option_unknown"

    def __init__(self, option_id: Any) -> None:
        self.option_id = str(option_id or "").strip()
        self.allowed = tuple(str(option["id"]) for option in ELIGIBILITY_OPTIONS)
        super().__init__(self.code)


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


def _option_copy(option: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **dict(option),
        "label": dict(option.get("label") or {}),
        "detail": dict(option.get("detail") or {}),
        "cohort": dict(option.get("cohort") or {}),
    }


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


def cohort_patch_for_option(option_id: Any) -> Dict[str, Any]:
    """Return one canonical patch or raise a typed contract error."""

    wanted = str(option_id or "").strip()
    for option in ELIGIBILITY_OPTIONS:
        if option["id"] == wanted:
            return dict(option["cohort"])
    raise CohortEligibilityOptionError(option_id)


def option_id_for_patch(value: Any) -> str:
    """Resolve an exact canonical eligibility patch without reading its label."""

    if not isinstance(value, Mapping):
        return ""
    fields = ("preset", *ELIGIBILITY_FIELDS)
    proposed = {field: value[field] for field in fields if field in value}
    matches: List[Tuple[int, str]] = []
    for option in ELIGIBILITY_OPTIONS:
        expected = {
            field: option["cohort"][field]
            for field in fields
            if field in option["cohort"]
        }
        if proposed == expected:
            matches.append((len(expected), str(option["id"])))
    return max(matches, default=(0, ""))[1]


def apply_option_to_cohort(current: Any, option_id: Any) -> Dict[str, Any]:
    """Replace only eligibility coordinates, preserving unrelated cohort metadata."""

    existing = dict(current) if isinstance(current, Mapping) else {}
    for field in ("preset", *ELIGIBILITY_FIELDS):
        existing.pop(field, None)
    existing.update(cohort_patch_for_option(option_id))
    return existing


def _authority_digest(authority: Mapping[str, Any]) -> str:
    payload = {
        key: authority[key]
        for key in sorted(authority)
        if key not in {"decision_sha256", "receipt_id"}
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _confirmation_timestamp_is_valid(value: Any) -> bool:
    timestamp = str(value or "").strip()
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def confirmation_authority_for_option(
    option_id: Any,
    *,
    study_context_id: Any,
    study_context_revision: int,
    confirmed_at: str = "",
) -> Dict[str, Any]:
    """Build the server-owned receipt for one explicit researcher selection."""

    if (
        isinstance(study_context_revision, bool)
        or not isinstance(study_context_revision, int)
        or study_context_revision < 1
    ):
        raise ValueError("cohort_eligibility_authority_revision_invalid")
    context_id = str(study_context_id or "").strip()
    if not context_id:
        raise ValueError("cohort_eligibility_authority_study_required")
    patch = cohort_patch_for_option(option_id)
    option = next(
        option for option in ELIGIBILITY_OPTIONS if option["id"] == str(option_id)
    )
    if option.get("declines_eligibility"):
        repeated_policy = "all_bound_icu_stays"
    elif patch.get("exclude_readmissions") is True:
        repeated_policy = "first_icu_admission_only"
    elif patch.get("exclude_readmissions") is False:
        repeated_policy = "all_icu_admissions"
    else:
        repeated_policy = "unspecified"
    timestamp = str(confirmed_at or "").strip() or datetime.now(timezone.utc).isoformat()
    if not _confirmation_timestamp_is_valid(timestamp):
        raise ValueError("cohort_eligibility_authority_timestamp_invalid")
    authority: Dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "cohort_scope_sha256": study_contexts.cohort_eligibility_scope_sha256(
            {"cohort": patch}
        ),
        "study_context_id": context_id,
        "study_context_revision": study_context_revision,
        "selection_state": (
            "declined" if option.get("declines_eligibility") else "confirmed"
        ),
        "option_id": str(option["id"]),
        "selection_mode": study_contexts.primary_cohort_selection_mode(
            {"cohort": patch}
        ),
        "repeated_admission_policy": repeated_policy,
        "origin": "copilot_user_selection",
        "confirmed_by": "researcher",
        "confirmed_at": timestamp,
        "stated_fields": [
            field for field in ELIGIBILITY_FIELDS if field in patch
        ],
    }
    if patch.get("age_min") is not None:
        authority["minimum_age"] = patch["age_min"]
    if patch.get("min_icu_los_hours") is not None:
        authority["minimum_icu_duration_hours"] = patch["min_icu_los_hours"]
    authority["decision_sha256"] = _authority_digest(authority)
    authority["receipt_id"] = f"cohort_eligibility_{authority['decision_sha256'][:24]}"
    return authority


def validated_authority(study: Any) -> Dict[str, Any] | None:
    """Return a valid decision bound to this exact cohort scope, else ``None``."""

    if not isinstance(study, Mapping):
        return None
    raw = study.get("cohort_eligibility_authority")
    authority = dict(raw) if isinstance(raw, Mapping) else {}
    if authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        return None
    try:
        patch = cohort_patch_for_option(authority.get("option_id"))
    except CohortEligibilityOptionError:
        return None
    current_scope = study_contexts.cohort_eligibility_scope_sha256(study)
    if authority.get("cohort_scope_sha256") != current_scope:
        return None
    if authority.get("cohort_scope_sha256") != (
        study_contexts.cohort_eligibility_scope_sha256({"cohort": patch})
    ):
        return None
    expected_digest = _authority_digest(authority)
    if authority.get("decision_sha256") != expected_digest:
        return None
    if authority.get("receipt_id") != f"cohort_eligibility_{expected_digest[:24]}":
        return None
    revision = authority.get("study_context_revision")
    current_revision = study.get("revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or isinstance(current_revision, bool)
        or not isinstance(current_revision, int)
        or current_revision < revision
    ):
        return None
    option = next(
        option for option in ELIGIBILITY_OPTIONS if option["id"] == authority["option_id"]
    )
    expected_state = "declined" if option.get("declines_eligibility") else "confirmed"
    if authority.get("selection_state") != expected_state:
        return None
    expected_mode = study_contexts.primary_cohort_selection_mode(study)
    if authority.get("selection_mode") != expected_mode:
        return None
    expected_repeated_policy = (
        "all_bound_icu_stays"
        if option.get("declines_eligibility")
        else "first_icu_admission_only"
        if patch.get("exclude_readmissions") is True
        else "all_icu_admissions"
        if patch.get("exclude_readmissions") is False
        else "unspecified"
    )
    if authority.get("repeated_admission_policy") != expected_repeated_policy:
        return None
    if authority.get("origin") != "copilot_user_selection":
        return None
    if authority.get("confirmed_by") != "researcher":
        return None
    if not _confirmation_timestamp_is_valid(authority.get("confirmed_at")):
        return None
    expected_fields = [field for field in ELIGIBILITY_FIELDS if field in patch]
    if authority.get("stated_fields") != expected_fields:
        return None
    optional_values = {
        "minimum_age": patch.get("age_min"),
        "minimum_icu_duration_hours": patch.get("min_icu_los_hours"),
    }
    for field, expected in optional_values.items():
        if expected is None:
            if field in authority:
                return None
        elif authority.get(field) != expected:
            return None
    if str(authority.get("study_context_id") or "") != str(study.get("id") or ""):
        return None
    return authority


def eligibility_stated(study: Any) -> bool:
    """Whether a server-issued decision covers the exact current cohort."""

    return validated_authority(study) is not None


def eligibility_proposal(study: Any) -> Dict[str, Any]:
    """Compile the eligibility question, or record that it is already settled.

    Always returns the same shape so a consumer branches on ``stated`` rather
    than on whether a key exists.
    """

    cohort = _cohort(study)
    authority = validated_authority(study)
    stated = authority is not None
    raw_authority = (
        study.get("cohort_eligibility_authority")
        if isinstance(study, Mapping)
        else None
    )
    legacy_value = bool(
        str(cohort.get("preset") or "").strip()
        or stated_eligibility_fields(study)
    )
    selection_state = (
        str(authority.get("selection_state"))
        if authority is not None
        else "legacy_unconfirmed"
        if legacy_value
        else "unresolved"
    )
    selection_mode = study_contexts.primary_cohort_selection_mode(
        {"cohort": dict(cohort)}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stated": stated,
        "selection_state": selection_state,
        "stated_fields": stated_eligibility_fields(study),
        "preset": str(cohort.get("preset") or ""),
        # What the Planner would be bound to if the study ran as it stands.
        "selection_mode": selection_mode,
        "authority_status": (
            "valid"
            if authority is not None
            else "stale"
            if isinstance(raw_authority, Mapping) and raw_authority
            else "absent"
        ),
        "authority": dict(authority) if authority is not None else None,
        "blocker_code": (
            None if stated else "cohort_eligibility_confirmation_required"
        ),
        "options": (
            []
            if stated
            else [_option_copy(option) for option in ELIGIBILITY_OPTIONS]
        ),
    }


def option_ids() -> Tuple[str, ...]:
    return tuple(str(option["id"]) for option in ELIGIBILITY_OPTIONS)


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "CohortEligibilityOptionError",
    "ELIGIBILITY_FIELDS",
    "ELIGIBILITY_OPTIONS",
    "SCHEMA_VERSION",
    "apply_option_to_cohort",
    "cohort_patch_for_option",
    "confirmation_authority_for_option",
    "eligibility_proposal",
    "eligibility_stated",
    "option_id_for_patch",
    "option_ids",
    "stated_eligibility_fields",
    "validated_authority",
]
