from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.concept.callbacks import (
    CALLBACK_REGISTRY,
    ConceptCallbackContext,
    _callback_septic_shock_sepsis3_2016,
)
from easyicu.scores.septic_shock import (
    EXPECTED_VASOPRESSOR_CONCEPTS,
    septic_shock_sepsis3_2016,
)
from easyicu.table import ICUTable


ROOT = Path(__file__).resolve().parents[1]


def _frame(column: str, rows: list[tuple[int, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["stay_id", "time", column])


def _empty_vasopressor_tables() -> dict[str, pd.DataFrame]:
    return {
        name: _frame(name, [])
        for name in EXPECTED_VASOPRESSOR_CONCEPTS
    }


def test_septic_shock_requires_sepsis_vasopressor_and_lactate_above_two() -> None:
    sepsis = _frame("sep3", [(1, 10.0, True), (2, 10.0, True)])
    lactate = _frame("lact", [(1, 12.0, 2.1), (2, 12.0, 2.0)])
    vasopressors = _empty_vasopressor_tables()
    vasopressors["norepi_rate"] = _frame(
        "norepi_rate",
        [(1, 11.0, 0.05), (2, 11.0, 0.05)],
    )

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
    )

    assert result["septic_shock_sepsis3_2016"].tolist() == [True, False]
    assert result["reason_code"].tolist() == [
        "criteria_met_fluid_adequacy_unobserved",
        "lactate_not_elevated",
    ]
    assert result.loc[0, "clinical_definition_complete"] == False  # noqa: E712
    assert result.loc[0, "fluid_resuscitation_ascertainment"] == "not_observed"
    assert result.loc[0, "vasopressor_indication_ascertainment"] == "not_observed"


def test_septic_shock_requires_temporal_pairing_with_vasopressor() -> None:
    sepsis = _frame("sep3", [(1, 0.0, True)])
    lactate = _frame("lact", [(1, 2.0, 4.0)])
    vasopressors = _empty_vasopressor_tables()
    vasopressors["norepi_rate"] = _frame("norepi_rate", [(1, 10.0, 0.1)])

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
        lactate_vasopressor_tolerance=pd.Timedelta(hours=6),
    )

    assert result.loc[0, "septic_shock_sepsis3_2016"] == False  # noqa: E712
    assert result.loc[0, "reason_code"] == "criteria_not_temporally_aligned"


def test_septic_shock_positive_evidence_survives_incomplete_drug_coverage() -> None:
    sepsis = _frame("sep3", [(1, 0.0, True)])
    lactate = _frame("lact", [(1, 2.0, 3.0)])
    vasopressors = {
        "norepi_rate": _frame("norepi_rate", [(1, 1.0, 0.03)]),
    }

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
    )

    assert result.loc[0, "septic_shock_sepsis3_2016"] == True  # noqa: E712
    assert result.loc[0, "vasopressor_ascertainment"] == "positive_direct_evidence"


def test_septic_shock_negative_fails_closed_when_drug_coverage_is_incomplete() -> None:
    sepsis = _frame("sep3", [(1, 0.0, True)])
    lactate = _frame("lact", [(1, 2.0, 3.0)])
    vasopressors = {"norepi_rate": _frame("norepi_rate", [])}

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
    )

    assert pd.isna(result.loc[0, "septic_shock_sepsis3_2016"])
    assert result.loc[0, "reason_code"] == "vasopressor_not_ascertained"
    assert result.loc[0, "vasopressor_ascertainment"].startswith("incomplete:")


def test_septic_shock_complete_absence_of_vasopressor_is_a_negative() -> None:
    sepsis = _frame("sep3", [(1, 0.0, True)])
    lactate = _frame("lact", [(1, 2.0, 3.0)])

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        _empty_vasopressor_tables(),
        id_cols=["stay_id"],
        index_col="time",
    )

    assert result.loc[0, "septic_shock_sepsis3_2016"] == False  # noqa: E712
    assert result.loc[0, "reason_code"] == "vasopressor_not_required"
    assert result.loc[0, "vasopressor_ascertainment"] == "complete_no_positive_event"


def test_septic_shock_missing_lactate_is_unknown_when_vasopressor_is_present() -> None:
    sepsis = _frame("sep3", [(1, pd.Timestamp("2026-01-01"), True)])
    lactate = _frame("lact", [])
    vasopressors = _empty_vasopressor_tables()
    vasopressors["epi_rate"] = _frame(
        "epi_rate",
        [(1, pd.Timestamp("2026-01-01 01:00"), 0.02)],
    )

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
    )

    assert pd.isna(result.loc[0, "septic_shock_sepsis3_2016"])
    assert result.loc[0, "reason_code"] == "lactate_not_observed"


def test_non_vasopressor_inotrope_cannot_create_a_positive() -> None:
    sepsis = _frame("sep3", [(1, 0.0, True)])
    lactate = _frame("lact", [(1, 1.0, 4.0)])
    vasopressors = _empty_vasopressor_tables()
    vasopressors["dobu_rate"] = _frame("dobu_rate", [(1, 1.0, 5.0)])

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
    )

    assert result.loc[0, "septic_shock_sepsis3_2016"] == False  # noqa: E712
    assert result.loc[0, "reason_code"] == "vasopressor_not_required"


def test_sepsis_false_and_unknown_propagate_without_reclassifying_shock() -> None:
    sepsis = _frame("sep3", [(1, 0.0, False), (2, 0.0, pd.NA)])
    lactate = _frame("lact", [(1, 1.0, 4.0), (2, 1.0, 4.0)])
    vasopressors = _empty_vasopressor_tables()
    vasopressors["norepi_rate"] = _frame(
        "norepi_rate",
        [(1, 1.0, 0.1), (2, 1.0, 0.1)],
    )

    result = septic_shock_sepsis3_2016(
        sepsis,
        lactate,
        vasopressors,
        id_cols=["stay_id"],
        index_col="time",
    )

    assert result.loc[0, "septic_shock_sepsis3_2016"] == False  # noqa: E712
    assert result.loc[0, "reason_code"] == "sepsis_not_present"
    assert pd.isna(result.loc[1, "septic_shock_sepsis3_2016"])
    assert result.loc[1, "reason_code"] == "sepsis_not_ascertained"
    assert str(result["septic_shock_sepsis3_2016"].dtype) == "boolean"


def test_septic_shock_concept_callback_preserves_receipts() -> None:
    tables = {
        "sep3": ICUTable(
            _frame("sep3", [(1, 0.0, True)]),
            id_columns=["stay_id"],
            index_column="time",
            value_column="sep3",
        ),
        "lact": ICUTable(
            _frame("lact", [(1, 2.0, 3.0)]),
            id_columns=["stay_id"],
            index_column="time",
            value_column="lact",
        ),
    }
    for concept, frame in _empty_vasopressor_tables().items():
        tables[concept] = ICUTable(
            frame,
            id_columns=["stay_id"],
            index_column="time",
            value_column=concept,
        )
    tables["norepi_rate"] = ICUTable(
        _frame("norepi_rate", [(1, 1.0, 0.05)]),
        id_columns=["stay_id"],
        index_column="time",
        value_column="norepi_rate",
    )
    ctx = ConceptCallbackContext(
        concept_name="septic_shock_sepsis3_2016",
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
    )

    result = _callback_septic_shock_sepsis3_2016(tables, ctx)

    assert result.value_column == "septic_shock_sepsis3_2016"
    assert result.data.loc[0, "septic_shock_sepsis3_2016"] == True  # noqa: E712
    assert result.data.loc[0, "reason_code"] == "criteria_met_fluid_adequacy_unobserved"
    assert CALLBACK_REGISTRY["septic_shock_sepsis3_2016"] is _callback_septic_shock_sepsis3_2016


def test_committed_septic_shock_golden_vector_executes_owner() -> None:
    fixture = json.loads(
        (
            ROOT
            / "tests/clinical_specs/septic_shock_sepsis3_2016_operational_v1.json"
        ).read_text(encoding="utf-8")
    )
    inputs = fixture["inputs"]

    def fixture_frame(name: str) -> pd.DataFrame:
        return _frame(
            name,
            [
                (row["stay_id"], row["time"], row["value"])
                for row in inputs[name]
            ],
        )

    result = septic_shock_sepsis3_2016(
        fixture_frame("sep3"),
        fixture_frame("lact"),
        {
            name: fixture_frame(name)
            for name in EXPECTED_VASOPRESSOR_CONCEPTS
        },
        id_cols=["stay_id"],
        index_col="time",
    )

    for key, expected in fixture["expected"].items():
        assert result.loc[0, key] == expected
