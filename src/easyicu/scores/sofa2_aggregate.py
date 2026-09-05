"""Observation-level SOFA-2 aggregation and identity contracts."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

from .sofa2_validation import SOFA2InputError, validate_numeric_input


SOFA2_COMPONENT_NAMES = (
    "sofa2_resp",
    "sofa2_coag",
    "sofa2_liver",
    "sofa2_cardio",
    "sofa2_cns",
    "sofa2_renal",
)
SOFA2_TRUSTED_ID_COLUMNS = (
    "stay_id",
    "icustay_id",
    "patientunitstayid",
    "admissionid",
    "patientid",
    "CaseID",
    "hadm_id",
    "subject_id",
)
SOFA2_TRUSTED_TIME_COLUMNS = ("charttime", "time", "hour", "index_var")

# A patient-level missing measurement is imputed to the normal score of zero in
# the published SOFA-2 primary analysis.  That rule must not be stretched to a
# database that lacks an entire organ-domain owner: such a database cannot
# provide a total score at all.  SICDB currently has no canonical CNS owner;
# its five supported component trajectories remain usable independently.
SOFA2_STRUCTURALLY_UNAVAILABLE_COMPONENTS_BY_DATABASE = {
    "sic": frozenset({"sofa2_cns"}),
}


def sofa2_total_structurally_supported(database: object) -> bool:
    """Return whether a database supports all six SOFA-2 organ domains."""

    normalized = str(database or "").strip().lower()
    if normalized == "sicdb":
        normalized = "sic"
    return not SOFA2_STRUCTURALLY_UNAVAILABLE_COMPONENTS_BY_DATABASE.get(normalized)


def _key_names(
    value: Optional[Sequence[str]],
    *,
    field: str,
    allow_empty: bool = False,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field=field,
            reason_code="sofa2_aggregate_key_contract_invalid",
            message="SOFA-2 aggregate keys must be a sequence of column names",
            invalid_count=1,
        )
    names = [str(name) for name in value]
    invalid = (
        (not allow_empty and not names)
        or any(not name for name in names)
        or len(names) != len(set(names))
    )
    if invalid:
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field=field,
            reason_code="sofa2_aggregate_key_contract_invalid",
            message="SOFA-2 aggregate keys must be valid and unique",
            invalid_count=len(names),
        )
    return names


def _resolve_keys(
    components: Dict[str, pd.DataFrame],
    *,
    id_cols: Optional[Sequence[str]],
    time_cols: Optional[Sequence[str]],
) -> tuple[list[str], list[str]]:
    """Resolve join keys from the SOFA-2 components alone.

    ``components`` must already be narrowed to the six required frames. Keys are
    inferred from their common columns, so an unrelated companion frame carried
    along in the caller's mapping cannot shrink that intersection and strand an
    identity key the components all agree on.
    """

    frames = list(components.values())
    if not frames or any(not isinstance(frame, pd.DataFrame) for frame in frames):
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field="data_dict",
            reason_code="sofa2_aggregate_component_frame_invalid",
            message="Every SOFA-2 component must be a pandas DataFrame",
            invalid_count=sum(not isinstance(frame, pd.DataFrame) for frame in frames),
        )
    common_columns = set(frames[0].columns)
    for frame in frames[1:]:
        common_columns.intersection_update(frame.columns)

    resolved_ids = _key_names(id_cols, field="id_cols")
    if resolved_ids is None:
        resolved_ids = next(
            ([name] for name in SOFA2_TRUSTED_ID_COLUMNS if name in common_columns),
            [],
        )
    if not resolved_ids:
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field="id_cols",
            reason_code="sofa2_aggregate_identity_keys_unresolved",
            message="SOFA-2 aggregation requires an explicit or trusted identity key",
        )

    resolved_times = _key_names(
        time_cols,
        field="time_cols",
        allow_empty=True,
    )
    if resolved_times is None:
        resolved_times = next(
            ([name] for name in SOFA2_TRUSTED_TIME_COLUMNS if name in common_columns),
            [],
        )
    overlap = set(resolved_ids).intersection(resolved_times)
    if overlap:
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field="time_cols",
            reason_code="sofa2_aggregate_key_contract_invalid",
            message="Identity and time keys must not overlap",
            invalid_count=len(overlap),
        )

    keys = resolved_ids + resolved_times
    missing_count = sum(
        any(key not in frame.columns for key in keys) for frame in frames
    )
    if missing_count:
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field="keys",
            reason_code="sofa2_aggregate_component_keys_missing",
            message="A SOFA-2 component is missing a declared aggregate key",
            invalid_count=missing_count,
        )
    return resolved_ids, resolved_times


def _project_component(
    frame: pd.DataFrame,
    *,
    component: str,
    key_columns: list[str],
) -> pd.DataFrame:
    if frame.columns.duplicated().any() or component not in frame.columns:
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field=component,
            reason_code="sofa2_aggregate_component_schema_invalid",
            message="SOFA-2 component frame has an invalid value-column schema",
            invalid_count=int(frame.columns.duplicated().sum())
            + int(component not in frame),
        )
    null_keys = frame[key_columns].isna().any(axis=1)
    if null_keys.any():
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field=component,
            reason_code="sofa2_aggregate_component_keys_null",
            message="SOFA-2 component identity/time keys cannot be missing",
            invalid_count=int(null_keys.sum()),
        )
    duplicate_keys = frame.duplicated(key_columns, keep=False)
    if duplicate_keys.any():
        raise SOFA2InputError(
            component="sofa2_aggregate",
            field=component,
            reason_code="sofa2_aggregate_component_keys_nonunique",
            message="SOFA-2 component identity/time keys must be unique",
            invalid_count=int(duplicate_keys.sum()),
        )

    value_columns = [component]
    for receipt in (f"{component}_observed", f"{component}_available"):
        if receipt in frame.columns:
            value_columns.append(receipt)
    projected = frame[key_columns + value_columns].copy()
    projected[component] = validate_numeric_input(
        projected[component],
        component="sofa2_aggregate",
        field=component,
        minimum=0,
        maximum=4,
        integer=True,
    )
    asserted = projected[component].notna().astype("int8")
    for suffix in ("observed", "available"):
        receipt = f"{component}_{suffix}"
        if receipt not in projected:
            projected[receipt] = asserted
            continue
        projected[receipt] = validate_numeric_input(
            projected[receipt],
            component="sofa2_aggregate",
            field=receipt,
            minimum=0,
            maximum=1,
            integer=True,
        )
    return projected


def sofa2_score(
    data_dict: Dict[str, pd.DataFrame],
    *,
    id_cols: Optional[Sequence[str]] = None,
    time_cols: Optional[Sequence[str]] = None,
    keep_components: bool = False,
) -> pd.DataFrame:
    """Aggregate one aligned observation/day-1 record of SOFA-2.

    Omitted keys are inferred only from fixed trusted ICU identity/time names
    common to the six components; arbitrary overlapping metadata is never a join
    key, and entries beyond the six components are ignored rather than allowed
    to constrain key inference. Component keys must be
    unique. Multiple time rows per identity fail closed because longitudinal
    scoring belongs to the production owner-receipt callback.
    """

    required = list(SOFA2_COMPONENT_NAMES)
    missing = [name for name in required if name not in data_dict]
    if missing:
        raise ValueError(f"Missing required component: {missing[0]}")

    components = {name: data_dict[name] for name in required}
    resolved_ids, resolved_times = _resolve_keys(
        components,
        id_cols=id_cols,
        time_cols=time_cols,
    )
    keys = resolved_ids + resolved_times
    result: Optional[pd.DataFrame] = None
    for component in required:
        projected = _project_component(
            data_dict[component],
            component=component,
            key_columns=keys,
        )
        result = (
            projected
            if result is None
            else result.merge(
                projected,
                on=keys,
                how="outer",
                validate="one_to_one",
                sort=False,
            )
        )

    assert result is not None
    if resolved_times and not result.empty:
        repeated = (
            result.groupby(
                resolved_ids,
                dropna=False,
                sort=False,
            ).size()
            > 1
        )
        if repeated.any():
            raise SOFA2InputError(
                component="sofa2_aggregate",
                field="time_cols",
                reason_code="sofa2_aggregate_longitudinal_policy_required",
                message=(
                    "Longitudinal SOFA-2 rows require the owner-issued LOCF "
                    "and availability-receipt pathway"
                ),
                invalid_count=int(repeated.sum()),
            )

    observed = [f"{component}_observed" for component in required]
    available = [f"{component}_available" for component in required]
    available_frame = result[available].fillna(0).eq(1)
    # The published SOFA-2 primary analysis uses normal-value imputation:
    # a domain without patient-level evidence contributes zero.  Availability
    # receipts remain separate so callers can reproduce the complete-case
    # sensitivity analysis instead of changing the primary score definition.
    effective_components = result[required].where(
        available_frame.to_numpy(),
        0,
    )
    result["sofa2"] = (
        effective_components.fillna(0).sum(axis=1)
        .round()
        .astype("Int64")
    )
    result["sofa2_n_observed_components"] = (
        result[observed].fillna(0).sum(axis=1).astype(int)
    )
    result["sofa2_n_available_components"] = (
        result[available].fillna(0).sum(axis=1).astype(int)
    )
    result["sofa2_n_components"] = result["sofa2_n_available_components"]

    if keep_components:
        for component in required:
            result[f"{component}_comp"] = result[component]
    else:
        result = result.drop(columns=required + observed + available)
    return result


__all__ = [
    "SOFA2_COMPONENT_NAMES",
    "SOFA2_STRUCTURALLY_UNAVAILABLE_COMPONENTS_BY_DATABASE",
    "sofa2_score",
    "sofa2_total_structurally_supported",
]
