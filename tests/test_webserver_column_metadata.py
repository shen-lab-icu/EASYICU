"""Producer-owned contracts for native export column metadata."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.config import load_src_cfg
from easyicu.resources import load_dictionary
from easyicu.webserver import dataio


def _build(
    frame: pd.DataFrame,
    *,
    concepts: list[str],
    database: str = "miiv",
):
    cfg = load_src_cfg(database)
    return dataio._build_export_file_metadata_binding(
        relative_path="module.csv",
        module="test",
        frame=frame,
        concept_ids=concepts,
        database=database,
        database_class_prefixes=tuple(cfg.class_prefix),
        dictionary=load_dictionary(include_sofa2=True),
    )


def test_arbitrary_single_output_never_becomes_a_concept_alias() -> None:
    binding = _build(pd.DataFrame({"stay_id": [1], "height": [180.0]}), concepts=["hr"])

    assert dict(binding.columns) == {}


def test_exact_logical_or_boolean_outputs_are_event_status() -> None:
    death = _build(
        pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}), concepts=["death"]
    )
    ventilation = _build(
        pd.DataFrame({"stay_id": [1, 2], "mech_vent": [False, True]}),
        concepts=["mech_vent"],
    )

    assert death.columns["death"].metadata.role is ConceptColumnRole.EVENT_STATUS
    assert (
        ventilation.columns["mech_vent"].metadata.role is ConceptColumnRole.EVENT_STATUS
    )


def test_catalog_only_boolean_output_is_event_status() -> None:
    binding = _build(
        pd.DataFrame({"stay_id": [1, 2], "mort_28d": [False, True]}),
        concepts=["mort_28d"],
    )

    assert binding.columns["mort_28d"].metadata.role is ConceptColumnRole.EVENT_STATUS


def test_logical_companions_use_concept_semantics_not_storage_dtype() -> None:
    binding = _build(
        pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "death_mean": [0.0, 0.5, 1.0],
                "death_max": [0, 1, 1],
            }
        ),
        concepts=["death"],
    )

    assert (
        binding.columns["death_mean"].metadata.role is ConceptColumnRole.EVENT_FRACTION
    )
    assert binding.columns["death_max"].metadata.role is ConceptColumnRole.EVENT_STATUS


@pytest.mark.parametrize(
    ("frame", "concept"),
    [
        (pd.DataFrame({"stay_id": [1, 2], "death": [0, 2]}), "death"),
        (
            pd.DataFrame({"stay_id": [1, 2], "death_mean": [0.0, 1.5]}),
            "death",
        ),
        (
            pd.DataFrame({"stay_id": [1, 2], "lact_measured": [0, 2]}),
            "lact",
        ),
        (
            pd.DataFrame({"stay_id": [1, 2], "lact_n": ["many", "few"]}),
            "lact",
        ),
        (
            pd.DataFrame({"stay_id": [1, 2], "lact_n": [False, True]}),
            "lact",
        ),
        (
            pd.DataFrame(
                {"stay_id": [1, 2], "charttime": [False, True], "age": [65, 70]}
            ),
            "age",
        ),
        (
            pd.DataFrame({"stay_id": [1, 2], "lact": [False, True]}),
            "lact",
        ),
        (
            pd.DataFrame(
                {
                    "stay_id": [1, 2],
                    "lact": [1.2, 2.4],
                    "lact_mean": [False, True],
                }
            ),
            "lact",
        ),
        (
            pd.DataFrame(
                {
                    "stay_id": [1, 2],
                    "lact": [1.2, 2.4],
                    "lact_max": [False, True],
                }
            ),
            "lact",
        ),
        (
            pd.DataFrame(
                {
                    "stay_id": [1, 2],
                    "charttime": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                    "age": [65, 70],
                }
            ),
            "age",
        ),
    ],
)
def test_typed_roles_fail_closed_when_physical_values_violate_the_contract(
    frame: pd.DataFrame, concept: str
) -> None:
    with pytest.raises(dataio.ExportCohortError) as exc_info:
        _build(frame, concepts=[concept])
    assert exc_info.value.detail["error"] == "column_metadata_value_domain_invalid"


@pytest.mark.parametrize(
    ("database", "identity"),
    [
        ("miiv", "stay_id"),
        ("mimic", "icustay_id"),
        ("mimic_demo", "icustay_id"),
        ("eicu", "patientunitstayid"),
        ("eicu_demo", "patientunitstayid"),
        ("aumc", "admissionid"),
        ("hirid", "patientid"),
        ("sic", "CaseID"),
    ],
)
def test_database_registry_selects_the_primary_icu_identity(
    database: str, identity: str
) -> None:
    frame = pd.DataFrame({identity: [1], "age": [65.0]})
    binding = _build(frame, concepts=["age"], database=database)

    assert binding.identity_column == identity


def test_auxiliary_ids_do_not_make_primary_identity_ambiguous() -> None:
    binding = _build(
        pd.DataFrame(
            {"stay_id": [1], "subject_id": [2], "hadm_id": [3], "age": [65.0]}
        ),
        concepts=["age"],
    )

    assert binding.identity_column == "stay_id"
    assert set(binding.columns) == {"age"}


@pytest.mark.parametrize(
    ("database", "raw_time"),
    [("eicu", "observationoffset"), ("aumc", "measuredat"), ("sic", "Offset")],
)
def test_unprojected_database_native_time_is_not_given_false_coordinates(
    database: str, raw_time: str
) -> None:
    cfg = load_src_cfg(database)
    identity = cfg.id_configs["icustay"].id
    frame = pd.DataFrame({identity: [1], raw_time: [60], "age": [65.0]})

    with pytest.raises(dataio.ExportCohortError) as exc_info:
        _build(frame, concepts=["age"], database=database)
    assert exc_info.value.detail["error"] == (
        "column_metadata_time_coordinate_unprojected"
    )


def test_duplicate_or_non_string_physical_columns_fail_before_write() -> None:
    duplicate = pd.DataFrame([[1, 80, 90]], columns=["stay_id", "hr", "hr"])
    with pytest.raises(dataio.ExportCohortError) as duplicate_exc:
        _build(duplicate, concepts=["hr"])
    assert duplicate_exc.value.detail["reason"] == "duplicate"

    non_string = pd.DataFrame([[1, 80]], columns=["stay_id", 123])
    with pytest.raises(dataio.ExportCohortError) as non_string_exc:
        _build(non_string, concepts=["hr"])
    assert non_string_exc.value.detail["reason"] == "non_string_or_empty"


def test_unsupported_boolean_median_companion_remains_unauthorized() -> None:
    binding = _build(
        pd.DataFrame({"stay_id": [1], "death_median": [True]}),
        concepts=["death"],
    )

    assert dict(binding.columns) == {}
