"""Integration contracts for the local clustered counting-process Cox adapter."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.methods.time_varying_exposure_cox import (
    TimeVaryingExposureCoxError,
    fit_cluster_robust_time_varying_cox,
)


def _require_r_survival() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available")
    probe = subprocess.run(
        [rscript, "--vanilla", "-e", "quit(status=!requireNamespace('survival'))"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        pytest.skip("R survival package is not available")


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260903)
    rows: list[dict[str, object]] = []
    for patient in range(90):
        for stay in range(2):
            stay_id = patient * 2 + stay
            age = float(rng.normal(65.0, 11.0))
            initial_lactate = float(rng.normal(2.3, 0.7))
            event_probability = 1.0 / (
                1.0 + np.exp(-(-2.2 + 0.55 * initial_lactate + 0.01 * (age - 65)))
            )
            event = int(rng.random() < event_probability)
            stop = float(rng.uniform(36.0, 180.0))
            change = float(stop * rng.uniform(0.2, 0.6))
            rows.extend(
                [
                    {
                        "stay_id": stay_id,
                        "patient": f"patient-{patient}",
                        "start": 0.0,
                        "stop": change,
                        "death": 0,
                        "lactate": initial_lactate,
                        "age": age,
                    },
                    {
                        "stay_id": stay_id,
                        "patient": f"patient-{patient}",
                        "start": change,
                        "stop": stop,
                        "death": event,
                        "lactate": initial_lactate + float(rng.normal(0.1, 0.25)),
                        "age": age,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_adapter_fits_cluster_robust_counting_process_model() -> None:
    _require_r_survival()
    panel = _panel()

    result = fit_cluster_robust_time_varying_cox(
        panel,
        id_col="stay_id",
        start_col="start",
        stop_col="stop",
        event_col="death",
        group_col="patient",
        covariates=("lactate", "age"),
    )

    assert result.estimates["term"].tolist() == ["lactate", "age"]
    assert np.isfinite(
        result.estimates[["coefficient", "standard_error", "p_value"]].to_numpy()
    ).all()
    assert (result.estimates["standard_error"] > 0).all()
    assert result.receipt["variance_estimator"] == "cluster_robust"
    assert result.receipt["cluster_count"] == 90
    assert result.receipt["event_count"] > 2
    assert result.receipt["diagnostics"] == {"converged": True, "warnings": []}
    assert result.receipt["engine_versions"]["survival"]
    assert result.receipt["engine_versions"]["R"]


def test_adapter_refuses_to_choose_missingness_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    panel = _panel()
    panel.loc[0, "lactate"] = np.nan

    with pytest.raises(TimeVaryingExposureCoxError, match="must be finite"):
        fit_cluster_robust_time_varying_cox(
            panel,
            id_col="stay_id",
            start_col="start",
            stop_col="stop",
            event_col="death",
            group_col="patient",
            covariates=("lactate", "age"),
        )


def test_adapter_rejects_finite_output_from_a_separated_model() -> None:
    _require_r_survival()
    panel = pd.DataFrame(
        {
            "stay_id": np.arange(100),
            "patient": np.arange(100) // 2,
            "start": np.zeros(100),
            "stop": np.r_[np.arange(1, 51), np.full(50, 60)],
            "death": np.r_[np.ones(50), np.zeros(50)],
            "exposure": np.r_[np.ones(50), np.zeros(50)],
        }
    )

    with pytest.raises(TimeVaryingExposureCoxError) as raised:
        fit_cluster_robust_time_varying_cox(
            panel,
            id_col="stay_id", start_col="start", stop_col="stop",
            event_col="death", group_col="patient", covariates=("exposure",),
        )

    assert raised.value.code == "time_varying_cox_fit_warning"


def test_adapter_rejects_changing_patient_group_within_one_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    panel = _panel()
    panel.loc[1, "patient"] = "different-patient"
    with pytest.raises(TimeVaryingExposureCoxError, match="one patient cluster"):
        fit_cluster_robust_time_varying_cox(
            panel,
            id_col="stay_id", start_col="start", stop_col="stop",
            event_col="death", group_col="patient", covariates=("lactate", "age"),
        )


def test_adapter_reports_missing_runtime_after_validating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(TimeVaryingExposureCoxError) as raised:
        fit_cluster_robust_time_varying_cox(
            _panel(),
            id_col="stay_id",
            start_col="start",
            stop_col="stop",
            event_col="death",
            group_col="patient",
            covariates=("lactate", "age"),
        )

    assert raised.value.code == "time_varying_cox_runtime_unavailable"
