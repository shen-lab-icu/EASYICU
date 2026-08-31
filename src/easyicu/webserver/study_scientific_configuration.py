"""Deep, dependency-neutral interpretation of StudyContext scientific fields.

Persistence, workflow, plan-choice, and Research Agent modules are adapters to
this authority.  This module does not load contexts, jobs, sources, or files.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


class ScientificConfigurationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class SetupFacts:
    active_export_present: bool
    eligibility_stated: bool
    dependence_finding: Optional[Mapping[str, Any]] = None
    window_finding: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ScientificSetupAssessment:
    missing_fields: tuple[str, ...]
    planning_prerequisites_missing: tuple[str, ...]


_PLANNING_PREREQUISITE_FIELDS = frozenset({"question", "data_source"})
_NOT_APPLICABLE_OUTCOMES = frozenset(
    {
        "none",
        "n/a",
        "na",
        "not applicable",
        "descriptive only",
        "无",
        "不适用",
        "仅描述",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


@dataclass(frozen=True)
class ScientificConfiguration:
    study: Mapping[str, Any]

    @classmethod
    def inspect(cls, study: Mapping[str, Any]) -> "ScientificConfiguration":
        return cls(dict(study))

    def target_outcome(self) -> Optional[str]:
        execution = _mapping(self.study.get("execution_concepts"))
        value = _clean(execution.get("outcome") or self.study.get("outcome"), 160)
        return None if value.casefold() in _NOT_APPLICABLE_OUTCOMES else value or None

    def primary_exposure(self) -> Optional[str]:
        execution = _mapping(self.study.get("execution_concepts"))
        return (
            _clean(
                execution.get("primary_exposure") or self.study.get("primary_exposure"),
                160,
            )
            or None
        )

    def primary_exposure_aggregation(self) -> Optional[str]:
        execution = _mapping(self.study.get("execution_concepts"))
        return _clean(execution.get("primary_exposure_aggregation"), 16).lower() or None

    def covariates(self) -> tuple[str, ...]:
        execution = _mapping(self.study.get("execution_concepts"))
        declared_raw = self.study.get("covariates")
        raw = execution.get("covariates") if "covariates" in execution else declared_raw
        if not isinstance(raw, (list, tuple)):
            return ()
        resolved = tuple(
            dict.fromkeys(
                _clean(value, 160)
                for value in raw
                if isinstance(value, str) and _clean(value, 160)
            )
        )
        if (
            self.covariate_selection() == "exact"
            and isinstance(declared_raw, (list, tuple))
            and "covariates" in execution
        ):
            declared = tuple(
                dict.fromkeys(
                    _clean(value, 160)
                    for value in declared_raw
                    if isinstance(value, str) and _clean(value, 160)
                )
            )
            if set(resolved) != set(declared):
                raise ScientificConfigurationError(
                    "research_pipeline_covariate_execution_binding_mismatch",
                    "The exact adjustment roster does not match its executable concept binding.",
                    details={"field": "execution_concepts.covariates"},
                )
        return resolved

    def covariate_selection(self) -> str:
        selection = str(
            self.study.get("covariate_selection") or "planner_selectable"
        ).strip()
        if selection not in {"planner_selectable", "exact"}:
            raise ScientificConfigurationError(
                "research_pipeline_covariate_selection_invalid",
                "StudyContext covariate_selection must be planner_selectable or exact.",
            )
        return selection

    def sensitivity_specs(self) -> tuple[Any, ...]:
        from easyicu.research_agent.planning.sensitivity_authority import (
            normalize_prespecified_sensitivities,
        )

        try:
            return normalize_prespecified_sensitivities(
                self.study.get("sensitivity_specs")
            )
        except (TypeError, ValueError) as exc:
            raise ScientificConfigurationError(
                "research_pipeline_sensitivity_specs_invalid",
                "The configured prespecified sensitivity contract is invalid.",
                details={"field": "sensitivity_specs", "reason": str(exc)[:500]},
            ) from exc

    def materialization_window(
        self,
        *,
        window_finding: Optional[Mapping[str, Any]] = None,
    ) -> tuple[float, float]:
        window = _mapping(self.study.get("time_window"))
        if not window:
            raise ScientificConfigurationError(
                "research_pipeline_time_window_required",
                "A typed study time window is required before pipeline launch.",
                details={"field": "time_window"},
            )
        value = window.get("hours")
        if value is None:
            value = window.get("observation_hours")
        if value is None:
            raise ScientificConfigurationError(
                "research_pipeline_time_window_hours_required",
                "The time-window label or preset has no executable duration; hours or observation_hours must be explicitly bound.",
                details={"field": "time_window.hours"},
            )
        if not _clean(window.get("anchor"), 160):
            raise ScientificConfigurationError(
                "research_pipeline_time_window_anchor_required",
                "The typed study time window requires an explicit scientific anchor.",
                details={"field": "time_window.anchor"},
            )
        if window_finding is not None:
            raise ScientificConfigurationError(
                "research_pipeline_materialization_window_anchor_unsupported",
                "The configured time-window anchor is not an executable outer materialization coordinate for this pipeline.",
                details={
                    key: value
                    for key, value in window_finding.items()
                    if key != "error"
                },
            )
        if isinstance(value, bool):
            raise ScientificConfigurationError(
                "research_pipeline_time_window_invalid",
                "The configured study time window must be a finite number of hours.",
            )
        try:
            hours = float(value)
        except (TypeError, ValueError) as exc:
            raise ScientificConfigurationError(
                "research_pipeline_time_window_invalid",
                "The configured study time window must be a finite number of hours.",
            ) from exc
        if not math.isfinite(hours) or not 0 < hours <= 24 * 365:
            raise ScientificConfigurationError(
                "research_pipeline_time_window_invalid",
                "The configured study time window is outside the supported range.",
            )
        return (0.0, hours)

    def modules(self) -> tuple[str, ...]:
        raw = self.study.get("modules")
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(
            dict.fromkeys(
                str(module).strip().lower()
                for module in raw
                if isinstance(module, str) and str(module).strip()
            )
        )

    def decision_is_resolved(self, decision_code: str) -> bool:
        confirmations = _mapping(self.study.get("confirmations"))
        code = str(decision_code or "").strip()
        keys = {
            "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED": (
                "plan_timing_landmark_24h",
                "plan_timing_descriptive_only",
                "plan_timing_time_varying",
            ),
            "REPEATED_STAY_IDENTITY_UNAVAILABLE": (
                "plan_repeated_stays_clustered",
                "plan_repeated_stays_first",
            ),
            "ADJUSTMENT_SET_NOT_USER_CONFIRMED": ("plan_adjustment_set_confirmed",),
            "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY": (
                "plan_required_sensitivities_executable",
            ),
        }.get(code, ())
        return any(confirmations.get(key) is True for key in keys)

    def merge_confirmations(self, **values: bool) -> dict[str, bool]:
        merged = dict(_mapping(self.study.get("confirmations")))
        merged.update(values)
        return merged

    def replace_sensitivity(
        self,
        *,
        axis: str,
        replacement: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        current = self.study.get("sensitivity_specs")
        rows = (
            [
                dict(item)
                for item in current
                if isinstance(item, Mapping) and str(item.get("axis") or "") != axis
            ]
            if isinstance(current, Sequence) and not isinstance(current, (str, bytes))
            else []
        )
        if replacement is not None:
            rows.append(dict(replacement))
        return rows

    def assess_setup(self, facts: SetupFacts) -> ScientificSetupAssessment:
        execution = _mapping(self.study.get("execution_concepts"))
        time_window = _mapping(self.study.get("time_window"))
        confirmations = _mapping(self.study.get("confirmations"))
        human_outcome = bool(_clean(self.study.get("outcome"), 500))
        human_exposure = bool(_clean(self.study.get("primary_exposure"), 160))
        exposure_required = self._requires_primary_exposure()
        executable_exposure = bool(_clean(execution.get("primary_exposure"), 160))
        executable_outcome = bool(_clean(execution.get("outcome"), 160))
        analysis_design_present = bool(_mapping(self.study.get("analysis_design")))
        window_hours = time_window.get("hours")
        if window_hours is None:
            window_hours = time_window.get("observation_hours")
        clinical_confirmation = self._missing_clinical_definition_confirmation()
        checks = (
            ("question", bool(_clean(self.study.get("question"), 1200))),
            (
                "data_source",
                facts.active_export_present
                or bool(_mapping(self.study.get("data_source"))),
            ),
            ("cohort", bool(_mapping(self.study.get("cohort")))),
            ("cohort_eligibility", facts.eligibility_stated),
            ("outcome", human_outcome),
            *(((("primary_exposure", human_exposure),)) if exposure_required else ()),
            ("analysis_goal", bool(_clean(self.study.get("analysis_goal"), 500))),
            *(
                ((("time_window", False),))
                if not time_window
                else (
                    ("time_window.hours", window_hours is not None),
                    (
                        "time_window.anchor",
                        bool(_clean(time_window.get("anchor"), 160)),
                    ),
                    *(
                        (("time_window.anchor_supported", False),)
                        if facts.window_finding is not None
                        else ()
                    ),
                    *(
                        (("confirmations.feature_time_window", False),)
                        if (
                            window_hours is not None
                            and bool(_clean(time_window.get("anchor"), 160))
                            and facts.window_finding is None
                            and confirmations.get("feature_time_window") is not True
                        )
                        else ()
                    ),
                )
            ),
            *(
                (
                    (
                        "covariates",
                        str(self.study.get("covariate_selection") or "").strip()
                        in {"exact", "planner_selectable"},
                    ),
                )
                if exposure_required
                else ()
            ),
            (
                "export_format",
                bool(_clean(self.study.get("export_format"), 80))
                and confirmations.get("export_format") is True,
            ),
            ("modules", bool(self.study.get("modules"))),
            *(
                (("execution_concepts.outcome", executable_outcome),)
                if human_outcome
                else ()
            ),
            *(
                (
                    (
                        ("execution_concepts.primary_exposure", executable_exposure),
                        ("analysis_design", analysis_design_present),
                        *(
                            (("analysis_design.dependence", False),)
                            if analysis_design_present
                            and facts.dependence_finding is not None
                            else ()
                        ),
                    )
                )
                if human_exposure or executable_exposure
                else ()
            ),
            *(((clinical_confirmation, False),) if clinical_confirmation else ()),
        )
        missing = tuple(name for name, present in checks if not present)
        planning = tuple(
            field
            for field in missing
            if field.split(".", 1)[0] in _PLANNING_PREREQUISITE_FIELDS
        )
        return ScientificSetupAssessment(missing, planning)

    def _requires_primary_exposure(self) -> bool:
        if _clean(self.study.get("primary_exposure"), 160):
            return True
        intent = " ".join(
            str(self.study.get(field) or "") for field in ("question", "purpose")
        ).casefold()
        return bool(
            re.search(
                r"(?:关系|关联|相关|效应|影响|预测|危险因素|association|associated|relationship|effect|predict|risk factor)",
                intent,
            )
        )

    def _missing_clinical_definition_confirmation(self) -> str:
        cohort = _mapping(self.study.get("cohort"))
        confirmations = _mapping(self.study.get("confirmations"))
        normalized_text = re.sub(
            r"[^a-z0-9]+",
            " ",
            " ".join(
                str(self.study.get(field) or "")
                for field in ("question", "purpose", "primary_exposure")
            ).lower(),
        )
        for field, contract in cohort.items():
            clean_field = str(field or "").strip().lower()
            contract = _mapping(contract)
            if not clean_field.endswith("_definition") or not contract:
                continue
            owner_locked = bool(
                contract.get("definition_locked") is True
                and _clean(contract.get("runtime_profile"), 160)
                and _clean(contract.get("implementation_profile"), 160)
                and _mapping(contract.get("locked_core"))
            )
            if owner_locked:
                continue
            phenotype = clean_field.removesuffix("_definition").strip("_")
            key = f"clinical_definition_{phenotype}"
            if (
                phenotype.replace("_", " ") in normalized_text
                and confirmations.get(key) is not True
            ):
                return f"confirmations.{key}"
        return ""


__all__ = [
    "ScientificConfiguration",
    "ScientificConfigurationError",
    "ScientificSetupAssessment",
    "SetupFacts",
]
