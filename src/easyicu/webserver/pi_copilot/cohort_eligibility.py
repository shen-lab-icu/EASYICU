"""Host-confirmed admission eligibility over one canonical primary cohort.

Owner: Copilot eligibility presentation and receipt validation.
Data Extraction owns execution normalization in ``webserver.primary_cohort``;
this module presents that immutable contract and records only a structured
host click. Natural-language interpretation may propose a cohort but cannot
mint researcher authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

from easyicu.webserver import primary_cohort, study_contexts


SCHEMA_VERSION = "easyicu.pi-cohort-eligibility/3"
AUTHORITY_SCHEMA_VERSION = "easyicu.cohort-eligibility-authority/2"
SELECTION_EVENT_SCHEMA_VERSION = "easyicu.cohort-eligibility-selection-event/1"
CUSTOM_OPTION_ID = "confirm_current_cohort"

# Compatibility projection only. Scope, invalidation, and execution all use
# NormalizedPrimaryCohortScope rather than this raw roster.
ELIGIBILITY_FIELDS = study_contexts.COHORT_ELIGIBILITY_FIELDS


class CohortEligibilityOptionError(ValueError):
    """Typed failure for a stale or unknown host option coordinate."""

    code = "cohort_eligibility_option_unknown"

    def __init__(self, option_id: Any) -> None:
        self.option_id = str(option_id or "").strip()
        self.allowed = (*option_ids(), CUSTOM_OPTION_ID)
        super().__init__(self.code)


class CohortEligibilityAuthorityError(ValueError):
    """Typed, owner-attributable invalid receipt."""

    def __init__(self, code: str, *, field: str = "") -> None:
        self.code = str(code)
        self.field = str(field)
        super().__init__(self.code)

    @property
    def detail(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            **({"field": self.field} if self.field else {}),
        }


@dataclass(frozen=True)
class CohortEligibilitySelectionEvent:
    """One-use host event bound to a displayed option and exact revision."""

    option_id: str
    study_context_id: str
    expected_revision: int
    session_id: str
    user_turn_id: str
    event_id: str
    one_use_grant_id: str
    primary_cohort_contract_sha256: str
    selected_at: str
    actor_id_sha256: str
    schema_version: str = SELECTION_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.option_id not in {*option_ids(), CUSTOM_OPTION_ID}:
            raise CohortEligibilityOptionError(self.option_id)
        if not self.study_context_id or not self.session_id or not self.user_turn_id:
            raise ValueError("cohort_eligibility_selection_binding_required")
        if (
            isinstance(self.expected_revision, bool)
            or not isinstance(self.expected_revision, int)
            or self.expected_revision < 1
        ):
            raise ValueError("cohort_eligibility_selection_revision_invalid")
        for value in (
            self.event_id,
            self.one_use_grant_id,
            self.primary_cohort_contract_sha256,
            self.actor_id_sha256,
        ):
            if not re.fullmatch(r"[a-f0-9]{64}", str(value or "")):
                raise ValueError("cohort_eligibility_selection_digest_invalid")
        if not _confirmation_timestamp_is_valid(self.selected_at):
            raise ValueError("cohort_eligibility_selection_timestamp_invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "option_id": self.option_id,
            "study_context_id": self.study_context_id,
            "expected_revision": self.expected_revision,
            "session_id": self.session_id,
            "user_turn_id": self.user_turn_id,
            "event_id": self.event_id,
            "one_use_grant_id": self.one_use_grant_id,
            "primary_cohort_contract_sha256": (
                self.primary_cohort_contract_sha256
            ),
            "selected_at": self.selected_at,
            "actor_id_sha256": self.actor_id_sha256,
        }


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
        # Admission-axis patch only. The current population/phenotype and its
        # diagnosis definition remain intact when the researcher chooses it.
        "cohort": dict(cohort),
        "declines_eligibility": bool(declines),
    }


ELIGIBILITY_OPTIONS: Tuple[Dict[str, Any], ...] = (
    _option(
        option_id="first_admission_only",
        label_en="First ICU admission only",
        label_zh="仅首次 ICU 入住",
        detail_en=(
            "Keep the current population definition without adding an age or "
            "duration restriction, and retain only the first ICU admission."
        ),
        detail_zh="保留当前人群定义，不另加年龄或停留时长限制，并仅保留首次 ICU 入住。",
        cohort={"exclude_readmissions": True},
    ),
    _option(
        option_id="adults_first_admission",
        label_en="Adults, first ICU admission only",
        label_zh="成人 · 仅首次 ICU 入住",
        detail_en="Keep the current population definition, exclude stays under 18, and retain only the first ICU admission.",
        detail_zh="保留当前人群定义，排除 18 岁以下，并仅保留首次 ICU 入住。",
        cohort={"age_min": 18, "exclude_readmissions": True},
    ),
    _option(
        option_id="adults_all_admissions",
        label_en="Adults, every ICU admission",
        label_zh="成人 · 全部 ICU 入住",
        detail_en="Keep the current population definition, exclude stays under 18, and retain repeated ICU admissions.",
        detail_zh="保留当前人群定义，排除 18 岁以下，并保留重复 ICU 入住。",
        cohort={"age_min": 18, "exclude_readmissions": False},
    ),
    _option(
        option_id="adults_first_admission_min_stay",
        label_en="Adults, first admission, at least 24 h in ICU",
        label_zh="成人 · 仅首次入住 · ICU 停留 ≥ 24 小时",
        detail_en="Keep the current population definition and require adult first admissions with at least 24 hours in ICU.",
        detail_zh="保留当前人群定义，并要求成人首次入住且 ICU 停留至少 24 小时。",
        cohort={
            "age_min": 18,
            "exclude_readmissions": True,
            "min_icu_los_hours": 24,
        },
    ),
    _option(
        option_id="no_eligibility_filter",
        label_en="No additional admission eligibility filter",
        label_zh="不另设入住资格标准",
        detail_en="Keep the current population definition and retain every admission within it without age, duration, or first-stay restrictions.",
        detail_zh="保留当前人群定义，不再增加年龄、停留时长或首次入住限制。",
        # Applying this option clears every admission-axis field. Keeping
        # neutral zero/false values in the durable StudyContext would falsely
        # look like a positive choice to downstream typed design gates.
        cohort={},
        declines=True,
    ),
)


def _cohort(study: Any) -> Mapping[str, Any]:
    raw = study.get("cohort") if isinstance(study, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def _scope_for_cohort(cohort: Any) -> primary_cohort.NormalizedPrimaryCohortScope:
    return primary_cohort.normalize_primary_cohort_scope(cohort)


def _scope(study: Any) -> primary_cohort.NormalizedPrimaryCohortScope:
    return _scope_for_cohort(_cohort(study))


def _option_copy(option: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **dict(option),
        "label": dict(option.get("label") or {}),
        "detail": dict(option.get("detail") or {}),
        "cohort": dict(option.get("cohort") or {}),
    }


def cohort_patch_for_option(option_id: Any) -> Dict[str, Any]:
    wanted = str(option_id or "").strip()
    for option in ELIGIBILITY_OPTIONS:
        if option["id"] == wanted:
            return dict(option["cohort"])
    raise CohortEligibilityOptionError(option_id)


def apply_option_to_cohort(current: Any, option_id: Any) -> Dict[str, Any]:
    """Apply one admission choice without replacing population semantics."""

    existing = dict(current) if isinstance(current, Mapping) else {}
    preset = str(existing.get("preset") or "").strip().lower()
    if not preset or preset in {"adult_first", "adult_all"}:
        existing["preset"] = "all_icu"
    for field in primary_cohort.ADMISSION_ELIGIBILITY_FIELDS:
        existing.pop(field, None)
    existing.update(cohort_patch_for_option(option_id))
    return existing


def _admission_option_id(scope: primary_cohort.NormalizedPrimaryCohortScope) -> str:
    admission = scope.admission_eligibility
    values = (
        int(admission.get("minimum_age_years") or 0),
        int(admission.get("maximum_age_years") or 100),
        int(admission.get("minimum_icu_duration_hours") or 0),
        str(admission.get("repeated_admission_policy") or ""),
    )
    matches = {
        (0, 100, 0, "first_icu_admission_only"): "first_admission_only",
        (18, 100, 0, "first_icu_admission_only"): "adults_first_admission",
        (18, 100, 0, "all_icu_admissions"): "adults_all_admissions",
        (18, 100, 24, "first_icu_admission_only"): (
            "adults_first_admission_min_stay"
        ),
        (0, 100, 0, "all_icu_admissions"): "no_eligibility_filter",
    }
    return matches.get(values, "")


def option_id_for_patch(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    try:
        return _admission_option_id(_scope_for_cohort(value))
    except primary_cohort.PrimaryCohortContractError:
        return ""


def selection_cohort_for_option(study: Any, option_id: Any) -> Dict[str, Any]:
    wanted = str(option_id or "").strip()
    current = dict(_cohort(study))
    if wanted == CUSTOM_OPTION_ID:
        if not current:
            raise CohortEligibilityOptionError(wanted)
        return current
    return apply_option_to_cohort(current, wanted)


def selection_options_for_study(study: Any) -> List[Dict[str, Any]]:
    """Return exact full-contract previews for host-rendered choices."""

    options: List[Dict[str, Any]] = []
    for static in ELIGIBILITY_OPTIONS:
        target = selection_cohort_for_option(study, static["id"])
        scope = _scope_for_cohort(target)
        options.append(
            {
                **_option_copy(static),
                "cohort": target,
                "primary_cohort_contract": scope.to_dict(),
                "primary_cohort_contract_sha256": scope.sha256,
            }
        )
    current = dict(_cohort(study))
    if current and not _admission_option_id(_scope_for_cohort(current)):
        scope = _scope_for_cohort(current)
        options.append(
            {
                "id": CUSTOM_OPTION_ID,
                "label": {
                    "en": "Confirm this exact custom primary cohort",
                    "zh": "确认当前这份自定义主队列",
                },
                "detail": {
                    "en": "Confirm the normalized population, eligibility, phenotype window, diagnosis rules, and sampling cap shown here.",
                    "zh": "确认此处展示的标准化人群、入住资格、表型时间窗、诊断规则和抽样上限。",
                },
                "cohort": current,
                "declines_eligibility": False,
                "primary_cohort_contract": scope.to_dict(),
                "primary_cohort_contract_sha256": scope.sha256,
            }
        )
    return options


def stated_eligibility_fields(study: Any) -> List[str]:
    return list(_scope(study).stated_fields)


def build_selection_event(
    *,
    option_id: Any,
    study_context_id: Any,
    expected_revision: int,
    session_id: Any,
    user_turn_id: Any,
    event_id: Any,
    one_use_grant_id: Any,
    primary_cohort_contract_sha256: Any,
    actor_id_sha256: Any,
    selected_at: str = "",
) -> CohortEligibilitySelectionEvent:
    timestamp = str(selected_at or "").strip() or datetime.now(timezone.utc).isoformat()
    return CohortEligibilitySelectionEvent(
        option_id=str(option_id or "").strip(),
        study_context_id=str(study_context_id or "").strip(),
        expected_revision=expected_revision,
        session_id=str(session_id or "").strip(),
        user_turn_id=str(user_turn_id or "").strip(),
        event_id=str(event_id or "").strip(),
        one_use_grant_id=str(one_use_grant_id or "").strip(),
        primary_cohort_contract_sha256=str(
            primary_cohort_contract_sha256 or ""
        ).strip(),
        selected_at=timestamp,
        actor_id_sha256=str(actor_id_sha256 or "").strip(),
    )


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
    current_cohort: Any = None,
    selection_event: CohortEligibilitySelectionEvent | Mapping[str, Any] | None,
    confirmed_at: str = "",
) -> Dict[str, Any]:
    """Build a receipt only from one validated host selection event."""

    if selection_event is None:
        raise ValueError("cohort_eligibility_selection_event_required")
    event = (
        selection_event
        if isinstance(selection_event, CohortEligibilitySelectionEvent)
        else build_selection_event(
            option_id=selection_event.get("option_id"),
            study_context_id=selection_event.get("study_context_id"),
            expected_revision=selection_event.get("expected_revision"),
            session_id=selection_event.get("session_id"),
            user_turn_id=selection_event.get("user_turn_id"),
            event_id=selection_event.get("event_id"),
            one_use_grant_id=selection_event.get("one_use_grant_id"),
            primary_cohort_contract_sha256=selection_event.get(
                "primary_cohort_contract_sha256"
            ),
            actor_id_sha256=selection_event.get("actor_id_sha256"),
            selected_at=selection_event.get("selected_at"),
        )
    )
    context_id = str(study_context_id or "").strip()
    if not context_id:
        raise ValueError("cohort_eligibility_authority_study_required")
    if (
        isinstance(study_context_revision, bool)
        or not isinstance(study_context_revision, int)
        or study_context_revision < 2
    ):
        raise ValueError("cohort_eligibility_authority_revision_invalid")
    if event.study_context_id != context_id:
        raise ValueError("cohort_eligibility_selection_study_mismatch")
    if event.expected_revision + 1 != study_context_revision:
        raise ValueError("cohort_eligibility_selection_revision_mismatch")
    wanted = str(option_id or "").strip()
    if event.option_id != wanted:
        raise ValueError("cohort_eligibility_selection_option_mismatch")
    target = selection_cohort_for_option({"cohort": current_cohort or {}}, wanted)
    scope = _scope_for_cohort(target)
    if event.primary_cohort_contract_sha256 != scope.sha256:
        raise ValueError("cohort_eligibility_selection_scope_mismatch")
    timestamp = str(confirmed_at or "").strip() or event.selected_at
    if not _confirmation_timestamp_is_valid(timestamp):
        raise ValueError("cohort_eligibility_authority_timestamp_invalid")
    declines = wanted == "no_eligibility_filter"
    admission = scope.admission_eligibility
    authority: Dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "cohort_scope_sha256": scope.sha256,
        "primary_cohort_contract_sha256": scope.sha256,
        "study_context_id": context_id,
        "study_context_revision": study_context_revision,
        "selection_state": "declined" if declines else "confirmed",
        "option_id": wanted,
        "selection_mode": scope.selection_mode,
        "repeated_admission_policy": str(
            admission["repeated_admission_policy"]
        ),
        "origin": "host_selection_event",
        "confirmed_by": "local_interactive_user",
        "confirmed_actor_id_sha256": event.actor_id_sha256,
        "session_id": event.session_id,
        "user_turn_id": event.user_turn_id,
        "confirmation_event_schema_version": event.schema_version,
        "selection_event_id": event.event_id,
        "one_use_grant_id": event.one_use_grant_id,
        "confirmed_at": timestamp,
        "stated_fields": list(scope.stated_fields),
    }
    minimum_age = int(admission.get("minimum_age_years") or 0)
    minimum_duration = int(admission.get("minimum_icu_duration_hours") or 0)
    if minimum_age:
        authority["minimum_age"] = minimum_age
    if minimum_duration:
        authority["minimum_icu_duration_hours"] = minimum_duration
    authority["decision_sha256"] = _authority_digest(authority)
    authority["receipt_id"] = f"cohort_eligibility_{authority['decision_sha256'][:24]}"
    return authority


_REQUIRED_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "decision_sha256",
        "cohort_scope_sha256",
        "primary_cohort_contract_sha256",
        "study_context_id",
        "study_context_revision",
        "selection_state",
        "option_id",
        "selection_mode",
        "repeated_admission_policy",
        "origin",
        "confirmed_by",
        "confirmed_actor_id_sha256",
        "session_id",
        "user_turn_id",
        "confirmation_event_schema_version",
        "selection_event_id",
        "one_use_grant_id",
        "confirmed_at",
        "stated_fields",
    }
)


def validate_authority_payload(
    study: Any,
    *,
    expected_revision: int | None = None,
) -> Dict[str, Any]:
    """Validate one persisted receipt against the canonical current contract."""

    if not isinstance(study, Mapping):
        raise CohortEligibilityAuthorityError("cohort_eligibility_study_invalid")
    raw = study.get("cohort_eligibility_authority")
    authority = dict(raw) if isinstance(raw, Mapping) else {}
    missing = sorted(_REQUIRED_AUTHORITY_FIELDS - set(authority))
    if missing:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_incomplete",
            field="cohort_eligibility_authority",
        )
    if authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_schema_invalid",
            field="cohort_eligibility_authority.schema_version",
        )
    option_id = str(authority.get("option_id") or "")
    if option_id not in {*option_ids(), CUSTOM_OPTION_ID}:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_option_unknown",
            field="cohort_eligibility_authority.option_id",
        )
    scope = _scope(study)
    for field in ("cohort_scope_sha256", "primary_cohort_contract_sha256"):
        if authority.get(field) != scope.sha256:
            raise CohortEligibilityAuthorityError(
                "cohort_eligibility_authority_scope_mismatch",
                field=f"cohort_eligibility_authority.{field}",
            )
    if option_id != CUSTOM_OPTION_ID and _admission_option_id(scope) != option_id:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_option_semantics_mismatch",
            field="cohort_eligibility_authority.option_id",
        )
    context_id = str(study.get("id") or "")
    if not context_id or authority.get("study_context_id") != context_id:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_study_mismatch",
            field="cohort_eligibility_authority.study_context_id",
        )
    revision = authority.get("study_context_revision")
    current_revision = study.get("revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 2
        or isinstance(current_revision, bool)
        or not isinstance(current_revision, int)
        or current_revision < revision
        or (expected_revision is not None and revision != expected_revision)
    ):
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_revision_mismatch",
            field="cohort_eligibility_authority.study_context_revision",
        )
    expected_state = "declined" if option_id == "no_eligibility_filter" else "confirmed"
    if authority.get("selection_state") != expected_state:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_state_mismatch",
            field="cohort_eligibility_authority.selection_state",
        )
    if authority.get("selection_mode") != scope.selection_mode:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_mode_mismatch",
            field="cohort_eligibility_authority.selection_mode",
        )
    repeated = str(scope.admission_eligibility["repeated_admission_policy"])
    if authority.get("repeated_admission_policy") != repeated:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_repeat_policy_mismatch",
            field="cohort_eligibility_authority.repeated_admission_policy",
        )
    if authority.get("origin") != "host_selection_event":
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_origin_invalid",
            field="cohort_eligibility_authority.origin",
        )
    if authority.get("confirmed_by") != "local_interactive_user":
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_actor_invalid",
            field="cohort_eligibility_authority.confirmed_by",
        )
    if not str(authority.get("session_id") or "").strip():
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_session_required",
            field="cohort_eligibility_authority.session_id",
        )
    if not str(authority.get("user_turn_id") or "").strip():
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_user_turn_required",
            field="cohort_eligibility_authority.user_turn_id",
        )
    if authority.get("confirmation_event_schema_version") != SELECTION_EVENT_SCHEMA_VERSION:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_event_schema_invalid",
            field="cohort_eligibility_authority.confirmation_event_schema_version",
        )
    for field in (
        "confirmed_actor_id_sha256",
        "selection_event_id",
        "one_use_grant_id",
    ):
        if not re.fullmatch(r"[a-f0-9]{64}", str(authority.get(field) or "")):
            raise CohortEligibilityAuthorityError(
                "cohort_eligibility_authority_event_digest_invalid",
                field=f"cohort_eligibility_authority.{field}",
            )
    if not _confirmation_timestamp_is_valid(authority.get("confirmed_at")):
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_timestamp_invalid",
            field="cohort_eligibility_authority.confirmed_at",
        )
    if authority.get("stated_fields") != list(scope.stated_fields):
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_fields_mismatch",
            field="cohort_eligibility_authority.stated_fields",
        )
    optional_values = {
        "minimum_age": int(scope.admission_eligibility.get("minimum_age_years") or 0),
        "minimum_icu_duration_hours": int(
            scope.admission_eligibility.get("minimum_icu_duration_hours") or 0
        ),
    }
    for field, expected in optional_values.items():
        if expected:
            if authority.get(field) != expected:
                raise CohortEligibilityAuthorityError(
                    "cohort_eligibility_authority_optional_value_mismatch",
                    field=f"cohort_eligibility_authority.{field}",
                )
        elif field in authority:
            raise CohortEligibilityAuthorityError(
                "cohort_eligibility_authority_optional_value_mismatch",
                field=f"cohort_eligibility_authority.{field}",
            )
    digest = _authority_digest(authority)
    if authority.get("decision_sha256") != digest:
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_digest_mismatch",
            field="cohort_eligibility_authority.decision_sha256",
        )
    if authority.get("receipt_id") != f"cohort_eligibility_{digest[:24]}":
        raise CohortEligibilityAuthorityError(
            "cohort_eligibility_authority_receipt_id_mismatch",
            field="cohort_eligibility_authority.receipt_id",
        )
    return authority


def validated_authority(study: Any) -> Dict[str, Any] | None:
    try:
        return validate_authority_payload(study)
    except (CohortEligibilityAuthorityError, primary_cohort.PrimaryCohortContractError):
        return None


def eligibility_stated(study: Any) -> bool:
    return validated_authority(study) is not None


def eligibility_proposal(study: Any) -> Dict[str, Any]:
    cohort = _cohort(study)
    try:
        scope = _scope(study)
    except primary_cohort.PrimaryCohortContractError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "stated": False,
            "selection_state": "legacy_unconfirmed",
            "stated_fields": [],
            "preset": str(cohort.get("preset") or ""),
            "selection_mode": "invalid",
            "primary_cohort_contract": None,
            "primary_cohort_contract_sha256": None,
            "authority_status": "invalid",
            "authority": None,
            "blocker_code": exc.code,
            "contract_error": {"code": exc.code, **dict(exc.detail)},
            "options": [],
        }
    authority = validated_authority(study)
    stated = authority is not None
    raw_authority = (
        study.get("cohort_eligibility_authority")
        if isinstance(study, Mapping)
        else None
    )
    legacy_value = bool(cohort)
    return {
        "schema_version": SCHEMA_VERSION,
        "stated": stated,
        "selection_state": (
            str(authority.get("selection_state"))
            if authority is not None
            else "legacy_unconfirmed"
            if legacy_value
            else "unresolved"
        ),
        "stated_fields": list(scope.stated_fields),
        "preset": str(cohort.get("preset") or ""),
        "selection_mode": scope.selection_mode,
        "primary_cohort_contract": scope.to_dict(),
        "primary_cohort_contract_sha256": scope.sha256,
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
        "options": [] if stated else selection_options_for_study(study),
    }


def option_ids() -> Tuple[str, ...]:
    return tuple(str(option["id"]) for option in ELIGIBILITY_OPTIONS)


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "CUSTOM_OPTION_ID",
    "CohortEligibilityAuthorityError",
    "CohortEligibilityOptionError",
    "CohortEligibilitySelectionEvent",
    "ELIGIBILITY_FIELDS",
    "ELIGIBILITY_OPTIONS",
    "SCHEMA_VERSION",
    "SELECTION_EVENT_SCHEMA_VERSION",
    "apply_option_to_cohort",
    "build_selection_event",
    "cohort_patch_for_option",
    "confirmation_authority_for_option",
    "eligibility_proposal",
    "eligibility_stated",
    "option_id_for_patch",
    "option_ids",
    "selection_cohort_for_option",
    "selection_options_for_study",
    "stated_eligibility_fields",
    "validate_authority_payload",
    "validated_authority",
]
