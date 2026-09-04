"""Typed local adapter for cluster-robust counting-process Cox regression.

The generic Python Cox implementations available to EasyICU do not provide a
reliable clustered covariance estimate for repeated start/stop intervals.  R's
well-established ``survival::coxph`` does.  This adapter owns only that narrow
computational bridge: it accepts an already-closed numeric counting-process
panel, invokes the local R runtime without network access, and returns a
path-free aggregate result.  It does not choose an exposure definition,
imputation policy, covariates, or scientific estimand.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


_RSCRIPT = r"""
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly=TRUE)
input <- args[[1]]
output <- args[[2]]
data <- read.csv(input, check.names=FALSE, stringsAsFactors=FALSE)
terms <- grep("^easyicu_cov_", names(data), value=TRUE)
if (length(terms) == 0L) stop("no model covariates")
data$easyicu_cluster <- as.factor(data$easyicu_cluster)
formula <- as.formula(paste(
  "Surv(easyicu_start, easyicu_stop, easyicu_event) ~",
  paste(terms, collapse=" + "),
  "+ cluster(easyicu_cluster)"
))
fit <- withCallingHandlers(coxph(
  formula,
  data=data,
  robust=TRUE,
  ties="efron",
  singular.ok=FALSE,
  x=TRUE,
  model=TRUE
), warning=function(w) {
  stop(paste("EASYICU_COX_FIT_WARNING:", conditionMessage(w)), call.=FALSE)
})
coefficients <- summary(fit)$coefficients
if (!"robust se" %in% colnames(coefficients)) stop("robust covariance unavailable")
result <- data.frame(
  term=rownames(coefficients),
  coefficient=coefficients[, "coef"],
  standard_error=coefficients[, "robust se"],
  z_value=coefficients[, "z"],
  p_value=coefficients[, "Pr(>|z|)"],
  r_version=as.character(getRversion()),
  survival_version=as.character(packageVersion("survival")),
  row.names=NULL,
  check.names=FALSE
)
write.csv(result, output, row.names=FALSE)
"""


class TimeVaryingExposureCoxError(ValueError):
    """The sealed counting-process model cannot be estimated safely."""

    def __init__(self, message: str, *, code: str = "time_varying_cox_input_invalid"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TimeVaryingExposureCoxFit:
    """Aggregate fitted coefficients and local execution receipt."""

    estimates: pd.DataFrame
    receipt: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TimeVaryingExposureCoxError(
            "counting-process input lacks columns: " + ", ".join(missing)
        )


def _numeric(values: pd.Series, *, label: str) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(converted).all():
        raise TimeVaryingExposureCoxError(f"{label} must be finite and complete")
    return converted


def _validate_panel(
    frame: pd.DataFrame,
    *,
    id_col: str,
    start_col: str,
    stop_col: str,
    event_col: str,
    group_col: str,
    covariates: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...], int, int]:
    if not isinstance(frame, pd.DataFrame):
        raise TimeVaryingExposureCoxError("counting-process input must be a dataframe")
    columns = tuple(str(value) for value in covariates)
    if not columns or len(columns) != len(set(columns)):
        raise TimeVaryingExposureCoxError("model covariates must be unique and nonempty")
    _require_columns(
        frame,
        {id_col, start_col, stop_col, event_col, group_col, *columns},
    )
    selected = frame.loc[:, [id_col, start_col, stop_col, event_col, group_col, *columns]].copy()
    if selected[id_col].isna().any() or selected[group_col].isna().any():
        raise TimeVaryingExposureCoxError("interval and cluster identities are required")
    start = _numeric(selected[start_col], label="interval starts")
    stop = _numeric(selected[stop_col], label="interval stops")
    if (start < 0).any() or (stop <= start).any():
        raise TimeVaryingExposureCoxError(
            "counting-process intervals require 0 <= start < stop"
        )
    event = _numeric(selected[event_col], label="event indicator")
    if not np.isin(event, [0.0, 1.0]).all():
        raise TimeVaryingExposureCoxError("event indicator must be binary")
    for column in columns:
        _numeric(selected[column], label=f"covariate {column!r}")

    ordered = selected.assign(__start=start, __stop=stop, __event=event).sort_values(
        [id_col, "__start", "__stop"], kind="mergesort"
    )
    for _, intervals in ordered.groupby(id_col, sort=False):
        if intervals[group_col].nunique() != 1:
            raise TimeVaryingExposureCoxError("each stay must belong to one patient cluster")
        starts = intervals["__start"].to_numpy(dtype=float)
        stops = intervals["__stop"].to_numpy(dtype=float)
        events = intervals["__event"].to_numpy(dtype=int)
        if starts[0] != 0.0 or not np.allclose(
            starts[1:], stops[:-1], rtol=0.0, atol=1e-10
        ):
            raise TimeVaryingExposureCoxError(
                "each stay must provide contiguous intervals from time zero"
            )
        if events.sum() > 1 or (events.sum() and events[-1] != 1):
            raise TimeVaryingExposureCoxError(
                "a stay can record at most one final-interval event"
            )
    event_count = int(event.sum())
    if event_count <= len(columns):
        raise TimeVaryingExposureCoxError(
            "too few events for the declared counting-process model"
        )
    cluster_count = int(selected[group_col].nunique(dropna=True))
    if cluster_count < 2:
        raise TimeVaryingExposureCoxError(
            "cluster-robust covariance requires at least two clusters"
        )
    return ordered, columns, event_count, cluster_count


def fit_cluster_robust_time_varying_cox(
    frame: pd.DataFrame,
    *,
    id_col: str,
    start_col: str,
    stop_col: str,
    event_col: str,
    group_col: str,
    covariates: Sequence[str],
) -> TimeVaryingExposureCoxFit:
    """Fit an Efron counting-process Cox model with patient-clustered SEs.

    ``frame`` is local-only input.  Every covariate must already be finite;
    this adapter deliberately refuses to decide how an ``unmeasured`` exposure
    state should become a model term.
    """

    ordered, columns, event_count, cluster_count = _validate_panel(
        frame,
        id_col=id_col,
        start_col=start_col,
        stop_col=stop_col,
        event_col=event_col,
        group_col=group_col,
        covariates=covariates,
    )
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise TimeVaryingExposureCoxError(
            "local Rscript runtime is unavailable",
            code="time_varying_cox_runtime_unavailable",
        )
    r_columns = {column: f"easyicu_cov_{index}" for index, column in enumerate(columns)}
    local = pd.DataFrame(
        {
            "easyicu_start": ordered["__start"].to_numpy(dtype=float),
            "easyicu_stop": ordered["__stop"].to_numpy(dtype=float),
            "easyicu_event": ordered["__event"].to_numpy(dtype=int),
            "easyicu_cluster": ordered[group_col].astype("string").to_numpy(),
            **{
                r_name: pd.to_numeric(ordered[column], errors="raise").to_numpy(
                    dtype=float
                )
                for column, r_name in r_columns.items()
            },
        }
    )
    with tempfile.TemporaryDirectory(prefix="easyicu-time-varying-cox-") as tempdir:
        root = Path(tempdir)
        input_path = root / "counting_process.csv"
        output_path = root / "coefficients.csv"
        local.to_csv(input_path, index=False)
        try:
            completed = subprocess.run(
                [rscript, "--vanilla", "-e", _RSCRIPT, str(input_path), str(output_path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeVaryingExposureCoxError(
                "cluster-robust time-varying Cox fit timed out",
                code="time_varying_cox_fit_timeout",
            ) from exc
        if "EASYICU_COX_FIT_WARNING:" in completed.stderr:
            # R can exit successfully with finite coefficients despite an
            # infinite estimate/non-convergence. The fit owner promotes every
            # fit warning to an error; no estimates or success receipt escape.
            raise TimeVaryingExposureCoxError(
                "cluster-robust time-varying Cox emitted a fit warning",
                code="time_varying_cox_fit_warning",
            )
        if completed.returncode != 0 or not output_path.is_file():
            raise TimeVaryingExposureCoxError(
                "cluster-robust time-varying Cox fit failed",
                code="time_varying_cox_fit_failed",
            )
        try:
            raw = pd.read_csv(output_path)
        except (OSError, ValueError) as exc:
            raise TimeVaryingExposureCoxError(
                "cluster-robust time-varying Cox output is unreadable"
            ) from exc
    expected_terms = list(r_columns.values())
    if list(raw.get("term", ())) != expected_terms:
        raise TimeVaryingExposureCoxError(
            "cluster-robust time-varying Cox output terms changed unexpectedly"
        )
    numeric = raw.loc[:, ["coefficient", "standard_error", "z_value", "p_value"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or bool(
        (numeric["standard_error"] <= 0).any()
    ):
        raise TimeVaryingExposureCoxError(
            "cluster-robust time-varying Cox produced invalid covariance"
        )
    estimates = raw.copy()
    estimates["term"] = columns
    estimates["hazard_ratio"] = np.exp(estimates["coefficient"])
    estimates["ci_low"] = np.exp(
        estimates["coefficient"] - 1.96 * estimates["standard_error"]
    )
    estimates["ci_high"] = np.exp(
        estimates["coefficient"] + 1.96 * estimates["standard_error"]
    )
    if not np.isfinite(
        estimates[["hazard_ratio", "ci_low", "ci_high"]].to_numpy(dtype=float)
    ).all():
        raise TimeVaryingExposureCoxError(
            "cluster-robust time-varying Cox produced non-finite contrasts"
        )
    return TimeVaryingExposureCoxFit(
        estimates=estimates.loc[
            :, [
                "term",
                "coefficient",
                "standard_error",
                "hazard_ratio",
                "ci_low",
                "ci_high",
                "z_value",
                "p_value",
            ]
        ],
        receipt={
            "schema_version": "easyicu.clustered_time_varying_cox/1",
            "method": "coxph_counting_process",
            "engine": "R_survival",
            "engine_versions": {
                "R": str(raw["r_version"].iloc[0]),
                "survival": str(raw["survival_version"].iloc[0]),
            },
            "diagnostics": {"converged": True, "warnings": []},
            "ties": "efron",
            "variance_estimator": "cluster_robust",
            "interval_rows": int(len(ordered)),
            "stay_count": int(ordered[id_col].nunique()),
            "event_count": event_count,
            "cluster_count": cluster_count,
            "covariates": list(columns),
            "privacy": {
                "patient_rows_returned": False,
                "identifier_values_returned": False,
                "source_paths_returned": False,
            },
        },
    )


__all__ = [
    "TimeVaryingExposureCoxError",
    "TimeVaryingExposureCoxFit",
    "fit_cluster_robust_time_varying_cox",
]
