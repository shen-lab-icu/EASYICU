"""Typed scientific authority supplied by an external benchmark row.

This owner validates study inputs before a benchmark can launch a Provider.
It does not infer endpoint semantics, choose covariates, or inspect data values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from ..contracts.endpoint import EndpointSpec
from ..schema import TimeWindow, UserPreferences


class ExternalBenchmarkAuthorityError(ValueError):
    """Owner-attributed fail-closed error for an external study contract."""

    def __init__(
        self,
        *,
        reason_code: str,
        field: str,
        message: str,
    ) -> None:
        self.reason_code = reason_code
        self.field = field
        super().__init__(
            f"external benchmark study authority [{reason_code}] {field}: {message}"
        )


@dataclass(frozen=True)
class ExternalBenchmarkStudyAuthority:
    endpoint: EndpointSpec | None
    user_preferences: dict[str, Any] | None
    concept_descriptions: dict[str, str]
    time_windows: tuple[dict[str, Any], ...]
    id_columns: tuple[str, ...]
    time_columns: tuple[str, ...]
    outcome_columns: tuple[str, ...]


def compile_external_benchmark_study_authority(
    *,
    row: Mapping[str, Any],
    target_outcome: str | None,
    cohort_columns: Sequence[Any],
    id_columns: Sequence[str] = (),
    time_columns: Sequence[str] = (),
    outcome_columns: Sequence[str] = (),
) -> ExternalBenchmarkStudyAuthority:
    """Validate explicit row authority without deriving missing decisions."""

    columns = {str(column) for column in cohort_columns}
    raw_endpoint = row.get("endpoint")
    endpoint = None
    if raw_endpoint is not None:
        if not isinstance(raw_endpoint, Mapping):
            raise ExternalBenchmarkAuthorityError(
                reason_code="ENDPOINT_NOT_OBJECT",
                field="endpoint",
                message="must be an object",
            )
        try:
            endpoint = EndpointSpec.model_validate(dict(raw_endpoint))
        except ValidationError as exc:
            raise ExternalBenchmarkAuthorityError(
                reason_code="ENDPOINT_INVALID",
                field="endpoint",
                message=str(exc.errors(include_url=False)),
            ) from exc
        if target_outcome is None or endpoint.name != str(target_outcome):
            raise ExternalBenchmarkAuthorityError(
                reason_code="ENDPOINT_TARGET_MISMATCH",
                field="endpoint.name",
                message="must match target_outcome",
            )
        if endpoint.name not in columns:
            raise ExternalBenchmarkAuthorityError(
                reason_code="ENDPOINT_COLUMN_MISSING",
                field="endpoint.name",
                message="must name a sealed cohort column",
            )

    raw_preferences = row.get("user_preferences")
    preferences = None
    if raw_preferences is not None:
        if not isinstance(raw_preferences, Mapping):
            raise ExternalBenchmarkAuthorityError(
                reason_code="USER_PREFERENCES_NOT_OBJECT",
                field="user_preferences",
                message="must be an object",
            )
        try:
            validated = UserPreferences.model_validate(dict(raw_preferences))
        except ValidationError as exc:
            raise ExternalBenchmarkAuthorityError(
                reason_code="USER_PREFERENCES_INVALID",
                field="user_preferences",
                message=str(exc.errors(include_url=False)),
            ) from exc
        missing_covariates = sorted(set(validated.covariates) - columns)
        if missing_covariates:
            raise ExternalBenchmarkAuthorityError(
                reason_code="COVARIATE_COLUMN_MISSING",
                field="user_preferences.covariates",
                message=(
                    "exact covariates must be sealed cohort columns: "
                    + ", ".join(missing_covariates)
                ),
            )
        preferences = validated.model_dump(mode="json")

    raw_descriptions = row.get("concept_descriptions")
    if raw_descriptions is None:
        descriptions: dict[str, str] = {}
    elif not isinstance(raw_descriptions, Mapping):
        raise ExternalBenchmarkAuthorityError(
            reason_code="CONCEPT_DESCRIPTIONS_NOT_OBJECT",
            field="concept_descriptions",
            message="must be an object",
        )
    else:
        descriptions = {
            str(name).strip(): " ".join(str(description or "").split())
            for name, description in raw_descriptions.items()
        }
        invalid = sorted(
            name
            for name, description in descriptions.items()
            if not name or name not in columns or not description
        )
        if invalid:
            raise ExternalBenchmarkAuthorityError(
                reason_code="CONCEPT_DESCRIPTION_COLUMN_INVALID",
                field="concept_descriptions",
                message=(
                    "concept descriptions require non-empty sealed cohort columns: "
                    + ", ".join(invalid)
                ),
            )

    raw_windows = row.get("time_windows")
    if raw_windows is None:
        windows: tuple[dict[str, Any], ...] = ()
    elif not isinstance(raw_windows, list):
        raise ExternalBenchmarkAuthorityError(
            reason_code="TIME_WINDOWS_NOT_LIST",
            field="time_windows",
            message="must be a list",
        )
    else:
        try:
            windows = tuple(
                TimeWindow.model_validate(value).model_dump(mode="json")
                for value in raw_windows
            )
        except ValidationError as exc:
            raise ExternalBenchmarkAuthorityError(
                reason_code="TIME_WINDOW_INVALID",
                field="time_windows",
                message=str(exc.errors(include_url=False)),
            ) from exc

    role_columns = {
        "id_columns": tuple(str(value) for value in id_columns),
        "time_columns": tuple(str(value) for value in time_columns),
        "outcome_columns": tuple(str(value) for value in outcome_columns),
    }
    for field, declared in role_columns.items():
        missing = sorted(set(declared) - columns)
        if missing:
            raise ExternalBenchmarkAuthorityError(
                reason_code="ROLE_COLUMN_MISSING",
                field=field,
                message="must contain only sealed cohort columns: " + ", ".join(missing),
            )
    if (
        role_columns["outcome_columns"]
        and target_outcome not in role_columns["outcome_columns"]
    ):
        raise ExternalBenchmarkAuthorityError(
            reason_code="TARGET_OUTCOME_ROLE_MISSING",
            field="outcome_columns",
            message="must include target_outcome when explicitly declared",
        )

    return ExternalBenchmarkStudyAuthority(
        endpoint=endpoint,
        user_preferences=preferences,
        concept_descriptions=descriptions,
        time_windows=windows,
        id_columns=role_columns["id_columns"],
        time_columns=role_columns["time_columns"],
        outcome_columns=role_columns["outcome_columns"],
    )


__all__ = [
    "ExternalBenchmarkAuthorityError",
    "ExternalBenchmarkStudyAuthority",
    "compile_external_benchmark_study_authority",
]
