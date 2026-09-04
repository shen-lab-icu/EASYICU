from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import easyicu
from easyicu import SOFA2InputError as PublicSOFA2InputError
from easyicu.io.ts_utils import change_interval
from easyicu.resources import load_dictionary
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
from easyicu.scores.sofa2_aggregate import sofa2_total_structurally_supported
from easyicu.scores.sofa2_validation import validate_numeric_input
from easyicu.table import ICUTable


def test_sofa2_total_structural_support_distinguishes_sicdb_alias() -> None:
    assert sofa2_total_structurally_supported("aumc")
    assert sofa2_total_structurally_supported("eicu")
    assert not sofa2_total_structurally_supported("sic")
    assert not sofa2_total_structurally_supported("sicdb")


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


def test_motor_response_hourly_aggregation_preserves_ordinal_domain() -> None:
    definition = load_dictionary(include_sofa2=True).get("motor_response")
    assert definition is not None
    assert definition.aggregate == "min"
    table = ICUTable(
        data=pd.DataFrame(
            {
                "stay_id": [1, 1],
                "charttime": [0.1, 0.8],
                "motor_response": [5.0, 4.0],
            }
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column="motor_response",
    )

    result = change_interval(
        table,
        interval=pd.Timedelta(hours=1),
        aggregation=definition.aggregate,
        time_unit="hours",
    ).data

    assert result["motor_response"].tolist() == [4.0]


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


def test_sofa2_aggregate_normal_imputes_score_disclaimed_by_receipt() -> None:
    frames = _component_frames()
    frames["sofa2_resp"]["sofa2_resp_available"] = 0

    result = sofa2_score(frames)

    assert result.loc[0, "sofa2"] == 5
    assert result.loc[0, "sofa2_n_available_components"] == 5


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


# --- 2026-08-15 review remediation: each test below fails on the prior implementation ---


def test_sofa2_cardio_allows_inotrope_only_ceiling_of_care() -> None:
    """Dobutamine is an inotrope, so it is not evidence a vasopressor was given.

    The first release of the conflict gate counted dobutamine as vasopressor
    exposure and aborted the whole component for a real ceiling-of-care
    combination (vasopressors precluded while an inotrope runs).
    """

    score = sofa2_cardio(
        map=pd.Series([70.0, 45.0]),
        dobu60=pd.Series([5.0, 0.0]),
        vasopressors_unavailable=pd.Series([True, True]),
    )

    # The inotrope row keeps its adjunct score instead of being overwritten by
    # the MAP cutoff; the drug-free row still takes the footnote (m) fallback.
    assert score.tolist() == [2, 3]


def test_sofa2_cardio_still_rejects_a_true_vasopressor_conflict() -> None:
    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_cardio(
            map=pd.Series([70.0]),
            norepi60=pd.Series([0.3]),
            vasopressors_unavailable=pd.Series([True]),
        )

    _assert_reason(exc_info, "sofa2_cardio_vasopressor_state_conflict")


def test_sofa2_numeric_inputs_reject_temporal_dtypes() -> None:
    """``pd.to_numeric`` turns datetimes into epoch integers, not into NaN."""

    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_coag(plt=pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"])))

    _assert_reason(exc_info, "sofa2_coag_platelets_numeric_dtype_invalid")

    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_coag(plt=pd.Series(pd.to_timedelta([1, 2], unit="h")))

    _assert_reason(exc_info, "sofa2_coag_platelets_numeric_dtype_invalid")


def test_sofa2_numeric_inputs_still_accept_categorical_encodings() -> None:
    frames = {
        component: pd.DataFrame(
            {"stay_id": [1, 2], component: pd.Categorical([1, 2])}
        )
        for component in SOFA2_COMPONENT_NAMES
    }

    assert sofa2_score(frames)["sofa2"].tolist() == [6, 12]


def test_sofa2_resp_rejects_a_pafi_unit_error() -> None:
    """FiO2 passed as 21-100 instead of 0.21-1.0 makes P/F 100x too small.

    Unbounded, that lands in the single digits and silently scores a maximal 4.
    """

    with pytest.raises(SOFA2InputError) as exc_info:
        sofa2_resp(pafi=pd.Series([4.76]), adv_resp=pd.Series([True]))

    _assert_reason(exc_info, "sofa2_resp_pafi_domain_invalid")

    # No ceiling on purpose: a high P/F scores 0, so an implausibly high value
    # is benign, and a ceiling would couple this module to dictionary bounds.
    assert sofa2_resp(pafi=pd.Series([100_000.0])).tolist() == [0]


def test_sofa2_resp_keeps_the_derivable_pafi_range() -> None:
    """Valid under the shipped dictionary and under the pending widened bounds.

    fix/concept-clinical-bounds widens po2 to [20, 700], which moves the
    derivable P/F range to [20, 3333]. The floor must accept both extremes.
    """

    score = sofa2_resp(
        pafi=pd.Series([20.0, 40.0, 120.0, 350.0, 2857.0, 3333.4]),
        adv_resp=pd.Series([True] * 6),
    )

    assert score.tolist() == [4, 4, 3, 0, 0, 0]


def test_sofa2_aggregate_ignores_entries_beyond_the_six_components() -> None:
    """Key inference must read the components, not every entry in the mapping."""

    frames = _component_frames()
    frames["companion_notes"] = pd.DataFrame({"note_id": [9], "text": ["x"]})

    result = sofa2_score(frames)

    assert result["sofa2"].tolist() == [6]


@pytest.mark.parametrize(
    "dtype",
    ["bool[pyarrow]", "int64[pyarrow]", "float64[pyarrow]", "boolean", "Int64"],
)
def test_sofa2_numeric_inputs_accept_arrow_and_nullable_backings(dtype: str) -> None:
    """The ordinary extraction path is Arrow-backed, so these must not crash.

    pd.to_numeric preserves the input's extension backing, and Arrow arrays
    raise NotImplementedError on ``%`` and on the float cast the finiteness
    check used, so an Arrow-backed receipt column killed the aggregate with an
    untyped error. bool[pyarrow] additionally failed the dtype allow-list
    because it is not is_numeric_dtype, unlike numpy bool and nullable boolean.
    """

    value = pd.Series([1, None], dtype=dtype)

    out = validate_numeric_input(
        value,
        component="sofa2_aggregate",
        field="sofa2_resp_available",
        minimum=0,
        maximum=1,
        integer=True,
    )

    assert out.notna().tolist() == [True, False]
    assert float(out.dropna().iloc[0]) == 1.0


def test_sofa2_arrow_backed_inputs_still_fail_closed() -> None:
    """Normalising for the arithmetic must not weaken any contract."""

    with pytest.raises(SOFA2InputError) as exc_info:
        validate_numeric_input(
            pd.Series([7.0, None], dtype="float64[pyarrow]"),
            component="sofa2_aggregate",
            field="sofa2_resp",
            minimum=0,
            maximum=4,
            integer=True,
        )
    _assert_reason(exc_info, "sofa2_aggregate_sofa2_resp_domain_invalid")

    with pytest.raises(SOFA2InputError) as exc_info:
        validate_numeric_input(
            pd.Series([2.5, None], dtype="float64[pyarrow]"),
            component="sofa2_aggregate",
            field="sofa2_resp",
            minimum=0,
            maximum=4,
            integer=True,
        )
    _assert_reason(exc_info, "sofa2_aggregate_sofa2_resp_integer_required")


def _shipped_concept_payloads() -> dict:
    """Merge the two shipped dictionaries as raw JSON.

    ``load_dictionary`` drops ``unit``/``min``/``max``, so the ordinal
    domain is only visible in the payload itself.
    """
    data_dir = Path(easyicu.__file__).resolve().parent / "data"
    merged: dict = {}
    for name in ("concept-dict.json", "sofa2-dict.json"):
        raw = json.loads((data_dir / name).read_text(encoding="utf-8"))
        merged.update(raw.get("concepts", raw))
    return merged


def _sofa2_closure(payloads: dict) -> set[str]:
    seen: set[str] = set()
    stack = [name for name in payloads if name.startswith("sofa2")]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(payloads.get(name, {}).get("concepts") or [])
    return seen


def test_every_ordinal_sofa2_input_declares_its_aggregate() -> None:
    """Ordinal inputs must not fall through to the numeric median default.

    ``_load_single_concept`` picks ``median`` for any numeric leaf that does
    not declare an aggregate.  Median is meaningless on an ordinal scale: two
    readings in the same hour yield a half-step that is not a member of the
    domain, and it also drops the worst value the severity score is asking
    for.  This is a class guard — pinning one concept is what let
    ``sedated_gcs`` sit unnoticed next to ``motor_response``.
    """
    payloads = _shipped_concept_payloads()
    missing = sorted(
        name
        for name in _sofa2_closure(payloads)
        if isinstance(payloads.get(name), dict)
        and not payloads[name].get("concepts")
        and str(payloads[name].get("unit", "")).lower() in {"score", "points"}
        and payloads[name].get("aggregate") is None
    )

    assert missing == [], (
        "ordinal SOFA-2 inputs without an explicit aggregate fall back to "
        f"median, which is off-domain for a score: {missing}"
    )
