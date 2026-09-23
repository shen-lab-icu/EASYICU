"""Step 1 -- load, validate and landmark-restrict the analysis cohort.

``load_cohort`` never chooses anything: the specification names every column
and the landmark rule, and the eligibility mask is the host's own
``landmark_eligibility_mask`` so the skill and the pipeline agree row for row on
who is "alive and observed at the landmark".

Rows whose exposure is missing are *kept* in the landmark cohort as the
``unknown`` level.  They are counted in every descriptive denominator and are
excluded from every model; the cohort flow records the split explicitly instead
of hiding it inside a model's complete-case count.

Data provenance is a declared fact, not an inference: the caller names the
source (``provenance=``), or a DataFrame carries it in ``attrs["provenance"]``
(the synthetic example does).  A cohort with neither is recorded as
``undeclared_by_caller`` so the manifest and report show the gap instead of a
blank field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ....contracts.model_terms import level_spelling
from ....execution.runners.landmark_categorical_association_executor import (
    LandmarkCategoricalExecutionError,
    landmark_eligibility_mask,
)
from ..spec import LandmarkCategoricalSpec

LOAD_TOKEN = "✓ Cohort loaded and landmark-restricted successfully!"
#: Recorded when neither the caller nor the frame states where the rows came from.
UNDECLARED_PROVENANCE = "undeclared_by_caller"

CohortSource = Union[str, Path, pd.DataFrame]


class CohortContractError(ValueError):
    """The supplied cohort does not satisfy the declared specification."""


@dataclass(frozen=True)
class LoadedCohort:
    """The landmark cohort together with the ledger that produced it."""

    spec: LandmarkCategoricalSpec
    frame: pd.DataFrame
    flow: pd.DataFrame
    n_source: int
    n_landmark: int
    n_exposure_known: int
    n_exposure_unknown: int
    source_path: Optional[str]
    source_sha256: Optional[str]
    provenance: str = UNDECLARED_PROVENANCE
    level_column: str = "__exposure_level__"
    known_mask_column: str = "__exposure_known__"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def known(self) -> pd.DataFrame:
        """Landmark rows whose exposure is a declared level."""

        return self.frame.loc[self.frame[self.known_mask_column]]

    def summary(self) -> dict[str, Any]:
        return {
            "n_source": self.n_source,
            "n_landmark": self.n_landmark,
            "n_exposure_known": self.n_exposure_known,
            "n_exposure_unknown": self.n_exposure_unknown,
            "landmark_hours": float(self.spec.landmark_hours),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "provenance": self.provenance,
        }


def resolve_provenance(source: CohortSource, declared: Optional[str]) -> str:
    """Return the declared provenance, the frame's own, or the explicit default."""

    if declared is not None and str(declared).strip():
        return str(declared).strip()
    if isinstance(source, pd.DataFrame):
        carried = source.attrs.get("provenance") if hasattr(source, "attrs") else None
        if carried is not None and str(carried).strip():
            return str(carried).strip()
    return UNDECLARED_PROVENANCE


def _read_source(source: CohortSource) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    if isinstance(source, pd.DataFrame):
        return source.copy(), None, None
    path = Path(source)
    if not path.is_file():
        raise CohortContractError(f"cohort file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise CohortContractError(
            f"unsupported cohort format {suffix!r}; use .parquet, .csv or .tsv"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame, str(path), digest


def spell_levels(series: pd.Series) -> pd.Series:
    """Map observed exposure values onto the declared level spelling."""

    values = series.astype("object").where(series.notna(), None)
    return values.map(level_spelling)


def _validate_exposure_column(
    frame: pd.DataFrame, column: str, spec: LandmarkCategoricalSpec
) -> pd.Series:
    if column not in frame.columns:
        raise CohortContractError(f"exposure column {column!r} is absent")
    spelled = spell_levels(frame[column])
    observed = set(spelled[spelled.ne("")].unique().tolist())
    undeclared = sorted(observed - set(spec.exposure_levels))
    if undeclared:
        raise CohortContractError(
            f"exposure column {column!r} carries values outside the declared "
            f"levels {spec.exposure_levels}: {undeclared}"
        )
    return spelled


def _validate_covariates(frame: pd.DataFrame, spec: LandmarkCategoricalSpec) -> list[str]:
    notes: list[str] = []
    for covariate in spec.covariates:
        series = frame[covariate.name]
        if covariate.coding == "continuous":
            numeric = pd.to_numeric(series, errors="coerce")
            if bool((series.notna() & numeric.isna()).any()):
                raise CohortContractError(
                    f"continuous covariate {covariate.name!r} has non-numeric values"
                )
            continue
        spelled = spell_levels(series)
        observed = set(spelled[spelled.ne("")].unique().tolist())
        declared = {level_spelling(item) for item in (covariate.levels or [])}
        undeclared = sorted(observed - declared)
        if undeclared:
            raise CohortContractError(
                f"covariate {covariate.name!r} carries undeclared levels {undeclared}"
            )
        absent = sorted(declared - observed)
        if absent:
            notes.append(
                f"covariate {covariate.name!r}: declared level(s) {absent} not observed"
            )
    return notes


def load_cohort(
    source: CohortSource,
    spec: LandmarkCategoricalSpec | Mapping[str, Any],
    *,
    provenance: Optional[str] = None,
    verbose: bool = True,
) -> LoadedCohort:
    """Validate the cohort against ``spec`` and keep the landmark-eligible rows."""

    contract = (
        spec
        if isinstance(spec, LandmarkCategoricalSpec)
        else LandmarkCategoricalSpec.model_validate(spec)
    )
    data_provenance = resolve_provenance(source, provenance)
    frame, source_path, source_sha256 = _read_source(source)
    if frame.empty:
        raise CohortContractError("the cohort has no rows")
    missing = [column for column in contract.required_columns() if column not in frame.columns]
    if missing:
        raise CohortContractError(
            "cohort lacks required column(s): " + ", ".join(sorted(missing))
        )
    if bool(frame[contract.identity_column].isna().any()):
        raise CohortContractError("identity column contains missing values")
    if bool(frame[contract.identity_column].duplicated().any()):
        raise CohortContractError("identity column must be unique per row")

    notes = _validate_covariates(frame, contract)
    for column in [contract.exposure, *contract.alternate_exposures]:
        _validate_exposure_column(frame, column, contract)

    try:
        nonnegative, alive, observed = landmark_eligibility_mask(
            frame,
            outcome_column=contract.outcome,
            event_time_column=contract.event_time_column,
            observation_duration_column=contract.observation_duration_column,
            observation_duration_unit=contract.observation_duration_unit,
            landmark_hours=contract.landmark_hours,
        )
    except LandmarkCategoricalExecutionError as exc:
        raise CohortContractError(str(exc)) from exc

    stages = [
        ("source_cohort", pd.Series(True, index=frame.index)),
        ("nonnegative_event_time", nonnegative),
        ("alive_at_landmark", alive),
        ("observed_at_landmark", observed),
    ]
    mask = pd.Series(True, index=frame.index)
    flow_rows: list[dict[str, Any]] = []
    for order, (predicate, condition) in enumerate(stages):
        before = int(mask.sum())
        if order:
            mask &= condition
        remaining = int(mask.sum())
        flow_rows.append(
            {
                "step_order": order,
                "predicate_kind": predicate,
                "action": "universe" if order == 0 else "exclude",
                "n_before": before,
                "n_excluded": before - remaining,
                "n_remaining": remaining,
            }
        )
    if not bool(mask.any()):
        raise CohortContractError("no row is alive and observed at the landmark")

    landmark = frame.loc[mask].copy()
    spelled = spell_levels(landmark[contract.exposure])
    known = spelled.ne("")
    landmark[LoadedCohort.level_column] = spelled.where(
        known, contract.unknown_level_label
    )
    landmark[LoadedCohort.known_mask_column] = known.to_numpy(dtype=bool)
    n_known = int(known.sum())
    n_unknown = int((~known).sum())
    flow_rows.append(
        {
            "step_order": len(flow_rows),
            "predicate_kind": "exposure_known",
            "action": "split_not_exclude",
            "n_before": int(len(landmark)),
            "n_excluded": 0,
            "n_remaining": int(len(landmark)),
            "n_exposure_known": n_known,
            "n_exposure_unknown": n_unknown,
        }
    )
    flow = pd.DataFrame(flow_rows)
    for column in ("n_exposure_known", "n_exposure_unknown"):
        if column in flow.columns:
            flow[column] = flow[column].astype("Int64")
    if n_known == 0:
        raise CohortContractError("every landmark row has an unknown exposure")

    loaded = LoadedCohort(
        spec=contract,
        frame=landmark,
        flow=flow,
        n_source=int(len(frame)),
        n_landmark=int(len(landmark)),
        n_exposure_known=n_known,
        n_exposure_unknown=n_unknown,
        source_path=source_path,
        source_sha256=source_sha256,
        provenance=data_provenance,
        notes=tuple(notes),
    )
    if verbose:
        print("=== Loading landmark cohort ===")
        print(f"   Source rows: {loaded.n_source}")
        print(f"   Data provenance: {loaded.provenance}")
        for row in flow_rows[1:4]:
            print(
                f"   {row['predicate_kind']}: excluded {row['n_excluded']}, "
                f"remaining {row['n_remaining']}"
            )
        print(
            f"   Exposure known / unknown at landmark: {n_known} / {n_unknown} "
            f"(unknown kept as '{contract.unknown_level_label}', never recoded to "
            f"'{contract.reference_level}')"
        )
        for note in notes:
            print(f"   NOTE: {note}")
        print(LOAD_TOKEN)
    return loaded


def frame_sha256(frame: pd.DataFrame) -> str:
    """Stable digest of a frame's values for manifests and idempotence checks."""

    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(np.ascontiguousarray(hashed).tobytes()).hexdigest()


__all__ = [
    "LOAD_TOKEN",
    "UNDECLARED_PROVENANCE",
    "CohortContractError",
    "LoadedCohort",
    "frame_sha256",
    "load_cohort",
    "resolve_provenance",
    "spell_levels",
]
