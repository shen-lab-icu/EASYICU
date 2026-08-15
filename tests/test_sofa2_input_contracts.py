from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu import SOFA2InputError as PublicSOFA2InputError
from easyicu.scores.sofa2 import (
    SOFA2_COMPONENT_NAMES,
    SOFA2InputError,
    sofa2_cardio,
    sofa2_cns,
    sofa2_coag,
    sofa2_liver,
    sofa2_renal,
    sofa2_resp,
    sofa2_score,
)


def _assert_reason(
    exc_info: pytest.ExceptionInfo[SOFA2InputError], reason: str
) -> None:
    assert exc_info.value.reason_code == reason


def _component_frames(*, rows: int = 1) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for index, component in enumerate(SOFA2_COMPONENT_NAMES):
        frames[component] = pd.DataFrame(
            {
                "stay_id": [1] * rows,
                "hour": list(range(rows)),
                "source_note": ["b" if index == 0 else "a"] * rows,
                component: [1] * rows,
            }
        )
    return frames


def test_sofa2_input_error_is_public() -> None:
    assert PublicSOFA2InputError is SOFA2InputError


@pytest.mark.clinical_conformance
def test_cardio_contradictory_vaso_unavailable_fails_closed() -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_cardio(
            pd.Series([80.0]),
            norepi=pd.Series([0.5]),
            vasopressors_unavailable=pd.Series([True]),
        )

    _assert_reason(exc_info, "sofa2_cardio_vasopressor_state_conflict")


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("fio2", [-1.0, 0.2, 2.0, 20.0, 101.0])
def test_resp_rejects_invalid_fio2_domain(fio2: float) -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_resp(
            pd.Series([np.nan]),
            spo2=pd.Series([90.0]),
            fio2=pd.Series([fio2]),
        )

    _assert_reason(exc_info, "sofa2_resp_fio2_domain_invalid")


@pytest.mark.clinical_conformance
def test_resp_rejects_mixed_fio2_units() -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_resp(
            pd.Series([np.nan, np.nan]),
            spo2=pd.Series([90.0, 90.0]),
            fio2=pd.Series([0.5, 50.0]),
        )

    _assert_reason(exc_info, "sofa2_resp_fio2_units_mixed")


@pytest.mark.clinical_conformance
def test_resp_accepts_each_explicit_fio2_unit_domain() -> None:
    fractional = sofa2_resp(
        pd.Series([np.nan]),
        spo2=pd.Series([90.0]),
        fio2=pd.Series([0.5]),
    )
    percentage = sofa2_resp(
        pd.Series([np.nan]),
        spo2=pd.Series([90.0]),
        fio2=pd.Series([50.0]),
    )

    assert fractional.tolist() == percentage.tolist() == [2]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("gcs", [-1.0, 0.0, 2.0, 3.5, 16.0, 99.0])
def test_cns_rejects_invalid_gcs_domain(gcs: float) -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_cns(pd.Series([gcs]))

    expected = (
        "sofa2_cns_gcs_integer_required"
        if gcs == 3.5
        else "sofa2_cns_gcs_domain_invalid"
    )
    _assert_reason(exc_info, expected)


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("motor", [0.0, 2.5, 7.0])
def test_cns_rejects_invalid_motor_response_domain(motor: float) -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_cns(pd.Series([np.nan]), motor_response=pd.Series([motor]))

    expected = (
        "sofa2_cns_motor_response_integer_required"
        if motor == 2.5
        else "sofa2_cns_motor_response_domain_invalid"
    )
    _assert_reason(exc_info, expected)


@pytest.mark.clinical_conformance
@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (lambda: sofa2_resp(pd.Series([-1.0])), "sofa2_resp_pafi_domain_invalid"),
        (
            lambda: sofa2_resp(
                pd.Series([np.nan]),
                spo2=pd.Series([101.0]),
                fio2=pd.Series([0.5]),
            ),
            "sofa2_resp_spo2_domain_invalid",
        ),
        (lambda: sofa2_coag(pd.Series([-1.0])), "sofa2_coag_platelets_domain_invalid"),
        (
            lambda: sofa2_liver(pd.Series([-1.0])),
            "sofa2_liver_bilirubin_domain_invalid",
        ),
        (lambda: sofa2_cardio(pd.Series([-1.0])), "sofa2_cardio_map_domain_invalid"),
        (
            lambda: sofa2_cardio(pd.Series([70.0]), norepi=pd.Series([-0.01])),
            "sofa2_cardio_norepi_domain_invalid",
        ),
        (
            lambda: sofa2_renal(pd.Series([-1.0])),
            "sofa2_renal_creatinine_domain_invalid",
        ),
        (
            lambda: sofa2_renal(
                pd.Series([1.0]),
                urine_mlkgph=pd.Series([-0.1]),
                urine_duration_h=pd.Series([6.0]),
            ),
            "sofa2_renal_urine_mlkgph_domain_invalid",
        ),
    ],
)
def test_sofa2_rejects_out_of_domain_physiology(call, reason: str) -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        call()

    _assert_reason(exc_info, reason)


@pytest.mark.clinical_conformance
def test_cns_rejects_invalid_delirium_evidence_state() -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_cns(
            pd.Series([15.0]),
            delirium_tx_evidence=pd.Series(["confirmd"]),
        )

    _assert_reason(exc_info, "sofa2_cns_delirium_tx_evidence_invalid")


@pytest.mark.parametrize(
    "key_kwargs",
    [{}, {"id_cols": ["stay_id"], "time_cols": ["hour"]}],
)
def test_sofa2_aggregate_uses_only_trusted_or_explicit_identity_columns(
    key_kwargs: dict[str, list[str]],
) -> None:
    result = sofa2_score(_component_frames(), **key_kwargs)

    assert result["sofa2"].tolist() == [6]
    assert result["sofa2_n_components"].tolist() == [6]
    assert "source_note" not in result.columns


def test_sofa2_aggregate_rejects_nonunique_component_keys() -> None:
    frames = _component_frames()
    frames["sofa2_resp"] = pd.concat(
        [frames["sofa2_resp"], frames["sofa2_resp"]],
        ignore_index=True,
    )

    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_score(frames, id_cols=["stay_id"], time_cols=["hour"])

    _assert_reason(exc_info, "sofa2_aggregate_component_keys_nonunique")


def test_sofa2_observation_aggregate_rejects_longitudinal_rows() -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_score(
            _component_frames(rows=2),
            id_cols=["stay_id"],
            time_cols=["hour"],
        )

    _assert_reason(exc_info, "sofa2_aggregate_longitudinal_policy_required")
