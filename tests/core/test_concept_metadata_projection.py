from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from easyicu.concept.metadata_projection import (
    METADATA_SCHEMA_VERSION,
    ColumnProjectionSpec,
    ConceptColumnRole,
    MetadataProjectionError,
    NumericBounds,
    canonical_metadata_bytes,
    derive_concept_column_metadata,
    is_range_preserving_projection,
    metadata_payload_sha256,
    metadata_sha256,
    project_concept_column_metadata,
)
from easyicu.concept.schema import ConceptDefinition

ROOT = Path(__file__).resolve().parents[2]
DICT_PATH = ROOT / "src" / "easyicu" / "data" / "concept-dict.json"
MODULE_PATH = ROOT / "src" / "easyicu" / "concept" / "metadata_projection.py"


def _definition(name: str) -> ConceptDefinition:
    payload = json.loads(DICT_PATH.read_text(encoding="utf-8"))[name]
    return ConceptDefinition.from_name_and_payload(name, payload)


def _spec(
    column_name: str,
    role: ConceptColumnRole,
    *,
    source_concept: str = "lact",
    aggregation: str | None = None,
    time_origin: str | None = None,
    time_unit: str | None = None,
) -> ColumnProjectionSpec:
    return ColumnProjectionSpec(
        column_name=column_name,
        source_concept=source_concept,
        role=role,
        aggregation=aggregation,
        time_origin=time_origin,
        time_unit=time_unit,
    )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("first", True),
        ("last", True),
        ("max", True),
        ("mean", True),
        ("median", True),
        ("min", True),
        ("sum", False),
    ],
)
def test_range_preserving_projection_contract_is_public_and_case_neutral(
    aggregation: str,
    expected: bool,
) -> None:
    assert (
        is_range_preserving_projection(
            ConceptColumnRole.NUMERIC_AGGREGATE,
            aggregation,
        )
        is expected
    )


def test_value_projection_always_preserves_range() -> None:
    assert is_range_preserving_projection(ConceptColumnRole.VALUE, None) is True


def test_projects_lact_value_with_separate_extraction_analysis_and_run_authorities():
    metadata = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec("lact_mean", ConceptColumnRole.VALUE),
        source_database="MIIV",
        analysis_plausibility_range=NumericBounds(0, 30),
    )

    assert metadata.schema_version == METADATA_SCHEMA_VERSION
    assert metadata.canonical_unit == "mmol/L"
    assert metadata.accepted_units == ("mmol/L",)
    assert metadata.extraction_bounds == NumericBounds(0, 50)
    assert metadata.analysis_plausibility_range == NumericBounds(0, 30)
    assert metadata.source_database == "miiv"
    assert metadata.dictionary_source_database == "miiv"
    assert metadata.source_resolution_chain == ("miiv",)
    assert metadata.source_declared_for_database is True
    assert metadata.availability_basis == "direct_source"
    assert set(metadata.available_databases) == {
        "aumc",
        "eicu",
        "eicu_demo",
        "hirid",
        "miiv",
        "mimic",
        "mimic_demo",
        "sic",
    }
    assert len(metadata.source_lineage) == 1
    lineage = metadata.source_lineage[0]
    assert lineage.database == "miiv"
    assert lineage.table == "labevents"
    assert lineage.selector_variable == "itemid"
    assert lineage.to_dict()["item_ids"] == [50813, 52442, 53154]


@pytest.mark.parametrize(
    ("role", "column_name", "allowed_values", "time_origin", "time_unit"),
    [
        (ConceptColumnRole.COUNT, "lact_n", None, None, None),
        (ConceptColumnRole.MEASUREMENT_STATUS, "lact_measured", (0, 1), None, None),
        (
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            "lact_first_time",
            None,
            "icu_admission",
            "h",
        ),
        (
            ConceptColumnRole.LAST_OBSERVATION_TIME,
            "lact_last_time",
            None,
            "icu_admission",
            "h",
        ),
    ],
)
def test_structural_companions_do_not_inherit_physiological_units_or_ranges(
    role: ConceptColumnRole,
    column_name: str,
    allowed_values: tuple[int, ...] | None,
    time_origin: str | None,
    time_unit: str | None,
):
    metadata = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec(
            column_name,
            role,
            time_origin=time_origin,
            time_unit=time_unit,
        ),
        source_database="miiv",
    )

    assert metadata.canonical_unit is None
    assert metadata.accepted_units == ()
    assert metadata.extraction_bounds is None
    assert metadata.analysis_plausibility_range is None
    assert metadata.allowed_values == allowed_values
    assert metadata.time_origin == time_origin
    assert metadata.time_unit == time_unit
    assert metadata.source_concept == "lact"
    assert metadata.source_database == "miiv"


def test_range_preserving_aggregate_inherits_ranges_but_sum_does_not():
    definition = _definition("lact")
    maximum = project_concept_column_metadata(
        definition,
        spec=_spec(
            "lact_max",
            ConceptColumnRole.NUMERIC_AGGREGATE,
            aggregation="MAX",
        ),
        source_database="miiv",
        analysis_plausibility_range=NumericBounds(0, 30),
    )
    total = project_concept_column_metadata(
        definition,
        spec=_spec(
            "lact_sum",
            ConceptColumnRole.NUMERIC_AGGREGATE,
            aggregation="sum",
        ),
        source_database="miiv",
    )

    assert maximum.aggregation == "max"
    assert maximum.extraction_bounds == NumericBounds(0, 50)
    assert maximum.analysis_plausibility_range == NumericBounds(0, 30)
    assert total.canonical_unit == "mmol/L"
    assert total.extraction_bounds is None
    assert total.analysis_plausibility_range is None


def test_event_status_and_fraction_are_typed_without_physiological_metadata():
    definition = _definition("lact")
    status = project_concept_column_metadata(
        definition,
        spec=_spec(
            "lact_max",
            ConceptColumnRole.EVENT_STATUS,
            aggregation="max",
        ),
        source_database="miiv",
    )
    fraction = project_concept_column_metadata(
        definition,
        spec=_spec(
            "lact_mean",
            ConceptColumnRole.EVENT_FRACTION,
            aggregation="mean",
        ),
        source_database="miiv",
    )

    assert status.allowed_values == (0, 1)
    assert status.canonical_unit is None
    assert status.extraction_bounds is None
    assert fraction.allowed_values is None
    assert fraction.canonical_unit is None
    assert fraction.extraction_bounds is None


def test_derived_materialized_metadata_preserves_source_authority_without_lookup():
    source = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec("lact", ConceptColumnRole.VALUE),
        source_database="miiv",
        analysis_plausibility_range=NumericBounds(0, 30),
    )

    derived = derive_concept_column_metadata(
        source,
        spec=_spec(
            "lact_max",
            ConceptColumnRole.NUMERIC_AGGREGATE,
            aggregation="max",
        ),
    )

    assert derived.column_name == "lact_max"
    assert derived.role is ConceptColumnRole.NUMERIC_AGGREGATE
    assert derived.aggregation == "max"
    assert derived.extraction_bounds == NumericBounds(0, 50)
    assert derived.analysis_plausibility_range == NumericBounds(0, 30)
    for field in (
        "source_concept",
        "source_database",
        "dictionary_source_database",
        "source_resolution_chain",
        "available_databases",
        "source_declared_for_database",
        "availability_basis",
        "source_lineage",
        "derived_from_concepts",
    ):
        assert getattr(derived, field) == getattr(source, field)


@pytest.mark.parametrize(
    ("role", "column_name", "aggregation", "time_origin", "time_unit"),
    [
        (ConceptColumnRole.COUNT, "lact_n", None, None, None),
        (
            ConceptColumnRole.MEASUREMENT_STATUS,
            "lact_measured",
            None,
            None,
            None,
        ),
        (
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            "lact_first_time",
            None,
            "icu_admission",
            "h",
        ),
    ],
)
def test_derived_structural_metadata_strips_source_physiology(
    role, column_name, aggregation, time_origin, time_unit
):
    source = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec("lact", ConceptColumnRole.VALUE),
        source_database="miiv",
        analysis_plausibility_range=NumericBounds(0, 30),
    )

    derived = derive_concept_column_metadata(
        source,
        spec=_spec(
            column_name,
            role,
            aggregation=aggregation,
            time_origin=time_origin,
            time_unit=time_unit,
        ),
    )

    assert derived.canonical_unit is None
    assert derived.accepted_units == ()
    assert derived.extraction_bounds is None
    assert derived.analysis_plausibility_range is None
    assert derived.source_lineage == source.source_lineage


def test_derived_metadata_rejects_source_concept_rebinding():
    source = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec("lact", ConceptColumnRole.VALUE),
        source_database="miiv",
    )

    with pytest.raises(MetadataProjectionError):
        derive_concept_column_metadata(
            source,
            spec=_spec(
                "decoy_max",
                ConceptColumnRole.NUMERIC_AGGREGATE,
                source_concept="decoy",
                aggregation="max",
            ),
        )


@pytest.mark.parametrize(
    ("source_role", "derived_role", "aggregation", "time_origin", "time_unit"),
    [
        (
            ConceptColumnRole.VALUE,
            ConceptColumnRole.EVENT_STATUS,
            "any",
            None,
            None,
        ),
        (
            ConceptColumnRole.VALUE,
            ConceptColumnRole.EVENT_FRACTION,
            "mean",
            None,
            None,
        ),
        (
            ConceptColumnRole.EVENT_STATUS,
            ConceptColumnRole.NUMERIC_AGGREGATE,
            "max",
            None,
            None,
        ),
        (
            ConceptColumnRole.COUNT,
            ConceptColumnRole.VALUE,
            None,
            None,
            None,
        ),
        (
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.VALUE,
            None,
            None,
            None,
        ),
    ],
)
def test_derived_metadata_rejects_role_escalation(
    source_role,
    derived_role,
    aggregation,
    time_origin,
    time_unit,
):
    source = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec(
            "lact_source",
            source_role,
            aggregation=(
                "max" if source_role is ConceptColumnRole.NUMERIC_AGGREGATE else None
            ),
            time_origin=(
                "icu_admission"
                if source_role is ConceptColumnRole.FIRST_OBSERVATION_TIME
                else None
            ),
            time_unit=(
                "h" if source_role is ConceptColumnRole.FIRST_OBSERVATION_TIME else None
            ),
        ),
        source_database="miiv",
    )

    with pytest.raises(MetadataProjectionError, match="not authorized"):
        derive_concept_column_metadata(
            source,
            spec=_spec(
                "derived",
                derived_role,
                aggregation=aggregation,
                time_origin=time_origin,
                time_unit=time_unit,
            ),
        )


@pytest.mark.parametrize(
    ("role", "aggregation"),
    [
        (ConceptColumnRole.EVENT_FRACTION, None),
        (ConceptColumnRole.EVENT_FRACTION, "max"),
        (ConceptColumnRole.EVENT_STATUS, "mean"),
    ],
)
def test_event_projection_rejects_inconsistent_aggregations(role, aggregation):
    with pytest.raises(MetadataProjectionError):
        _spec("event_projection", role, aggregation=aggregation)


def test_strict_metadata_parser_round_trips_projector_output():
    original = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec(
            "lact_first_time",
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            time_origin="icu_admission",
            time_unit="h",
        ),
        source_database="miiv",
    )

    parsed = type(original).from_dict(original.to_dict())

    assert parsed == original
    assert canonical_metadata_bytes(parsed) == canonical_metadata_bytes(original)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "easyicu.concept_column_metadata/999"),
        ("source_database", "eicu"),
        ("accepted_units", ["mmol/L", "mmol/L"]),
        ("allowed_values", [1, 0]),
    ],
)
def test_strict_metadata_parser_rejects_noncanonical_or_inconsistent_payloads(
    field, replacement
):
    metadata = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec("lact_measured", ConceptColumnRole.MEASUREMENT_STATUS),
        source_database="miiv",
    )
    payload = metadata.to_dict()
    payload[field] = replacement

    with pytest.raises(MetadataProjectionError):
        type(metadata).from_dict(payload)


@pytest.mark.parametrize(
    "bounds",
    [
        NumericBounds(None, 1),
        NumericBounds(0, None),
        NumericBounds(-1.5, 2.5),
    ],
)
def test_numeric_bounds_support_finite_one_sided_or_two_sided_ranges(bounds):
    assert bounds.to_dict() == {
        "minimum": None if bounds.minimum is None else float(bounds.minimum),
        "maximum": None if bounds.maximum is None else float(bounds.maximum),
    }


@pytest.mark.parametrize(
    "args",
    [
        (float("nan"), 1),
        (0, float("inf")),
        (2, 1),
        (True, 1),
        ("not-a-number", 1),
    ],
)
def test_numeric_bounds_reject_invalid_ranges(args):
    with pytest.raises(MetadataProjectionError):
        NumericBounds(*args)


def test_projection_rejects_ambiguous_or_inconsistent_specs():
    definition = _definition("lact")
    with pytest.raises(MetadataProjectionError, match="require an explicit"):
        _spec("lact_mean", ConceptColumnRole.NUMERIC_AGGREGATE)
    with pytest.raises(MetadataProjectionError, match="unsupported"):
        _spec(
            "lact_slope",
            ConceptColumnRole.NUMERIC_AGGREGATE,
            aggregation="slope",
        )
    with pytest.raises(MetadataProjectionError, match="only valid"):
        _spec("lact_n", ConceptColumnRole.COUNT, aggregation="sum")
    with pytest.raises(MetadataProjectionError, match="ConceptColumnRole"):
        ColumnProjectionSpec(
            column_name="lact",
            source_concept="lact",
            role="value",  # type: ignore[arg-type]
        )
    with pytest.raises(MetadataProjectionError, match="must be strings"):
        ColumnProjectionSpec(
            column_name=1,  # type: ignore[arg-type]
            source_concept="lact",
            role=ConceptColumnRole.VALUE,
        )
    with pytest.raises(MetadataProjectionError, match="explicit time_origin"):
        _spec("lact_first_time", ConceptColumnRole.FIRST_OBSERVATION_TIME)
    with pytest.raises(MetadataProjectionError, match="only valid for time"):
        _spec(
            "lact",
            ConceptColumnRole.VALUE,
            time_origin="icu_admission",
            time_unit="h",
        )
    with pytest.raises(MetadataProjectionError, match="time_origin must be a string"):
        ColumnProjectionSpec(
            column_name="charttime",
            source_concept="lact",
            role=ConceptColumnRole.EVENT_TIME,
            time_origin={"kind": "absolute"},  # type: ignore[arg-type]
            time_unit="timestamp",
        )
    with pytest.raises(MetadataProjectionError, match="source_concept"):
        project_concept_column_metadata(
            definition,
            spec=_spec(
                "crea_mean",
                ConceptColumnRole.VALUE,
                source_concept="crea",
            ),
            source_database="miiv",
        )
    with pytest.raises(MetadataProjectionError, match="value-like"):
        project_concept_column_metadata(
            definition,
            spec=_spec("lact_n", ConceptColumnRole.COUNT),
            source_database="miiv",
            analysis_plausibility_range=NumericBounds(0, 30),
        )
    with pytest.raises(MetadataProjectionError, match="range-preserving"):
        project_concept_column_metadata(
            definition,
            spec=_spec(
                "lact_sum",
                ConceptColumnRole.NUMERIC_AGGREGATE,
                aggregation="sum",
            ),
            source_database="miiv",
            analysis_plausibility_range=NumericBounds(0, 30),
        )
    with pytest.raises(MetadataProjectionError, match="must be NumericBounds"):
        project_concept_column_metadata(
            definition,
            spec=_spec("lact", ConceptColumnRole.VALUE),
            source_database="miiv",
            analysis_plausibility_range={"minimum": 0},  # type: ignore[arg-type]
        )
    with pytest.raises(MetadataProjectionError, match="source_database must be"):
        project_concept_column_metadata(
            definition,
            spec=_spec("lact", ConceptColumnRole.VALUE),
            source_database=False,  # type: ignore[arg-type]
        )
    with pytest.raises(MetadataProjectionError, match="must be non-empty"):
        project_concept_column_metadata(
            definition,
            spec=_spec("lact", ConceptColumnRole.VALUE),
            source_database="  ",
        )


def test_actual_source_and_cross_database_availability_are_not_conflated():
    definition = _definition("lact")
    unavailable = project_concept_column_metadata(
        definition,
        spec=_spec("lact", ConceptColumnRole.VALUE),
        source_database="unknown_db",
    )
    unspecified = project_concept_column_metadata(
        definition,
        spec=_spec("lact", ConceptColumnRole.VALUE),
        source_database=None,
    )

    assert "miiv" in unavailable.available_databases
    assert unavailable.source_database == "unknown_db"
    assert unavailable.dictionary_source_database is None
    assert unavailable.source_resolution_chain == ("unknown_db",)
    assert unavailable.source_declared_for_database is False
    assert unavailable.availability_basis == "source_not_declared"
    assert unavailable.source_lineage == ()
    assert unspecified.source_database is None
    assert unspecified.dictionary_source_database is None
    assert unspecified.source_resolution_chain == ()
    assert unspecified.source_declared_for_database is None
    assert unspecified.availability_basis == "source_database_not_supplied"


@pytest.mark.parametrize("derived_field", ["depends_on", "concepts", "callback"])
def test_derived_concepts_without_direct_sources_do_not_fabricate_availability(
    derived_field: str,
):
    payload: dict[str, object] = {
        "description": "derived test",
        "sources": {},
    }
    payload[derived_field] = (
        "compute_derived"
        if derived_field == "callback"
        else ["component_a", "component_b"]
    )
    definition = ConceptDefinition.from_name_and_payload("derived_test", payload)
    metadata = project_concept_column_metadata(
        definition,
        spec=_spec(
            "derived_test",
            ConceptColumnRole.VALUE,
            source_concept="derived_test",
        ),
        source_database="miiv",
    )

    assert metadata.source_declared_for_database is False
    assert metadata.availability_basis == "derived_dependencies_not_resolved"
    assert metadata.source_lineage == ()


def test_lineage_and_payload_digests_are_independent_of_mapping_insertion_order():
    first_payload = {
        "unit": ["mmol/L", "mmol/l"],
        "min": 0,
        "max": 10,
        "sources": {
            "miiv": [
                {
                    "table": "z_table",
                    "ids": [2, 1, "1"],
                    "sub_var": "itemid",
                    "unit_var": "unit_z",
                },
                {
                    "table": "a_table",
                    "ids": [4, 3],
                    "sub_var": "code",
                    "unit_var": "unit_a",
                },
            ],
            "eicu": [{"table": "lab", "ids": "lactate"}],
        },
    }
    second_payload = {
        **first_payload,
        "sources": {
            "eicu": first_payload["sources"]["eicu"],
            "miiv": list(reversed(first_payload["sources"]["miiv"])),
        },
    }
    first = project_concept_column_metadata(
        ConceptDefinition.from_name_and_payload("stable", first_payload),
        spec=_spec("stable", ConceptColumnRole.VALUE, source_concept="stable"),
        source_database="miiv",
    )
    second = project_concept_column_metadata(
        ConceptDefinition.from_name_and_payload("stable", second_payload),
        spec=_spec("stable", ConceptColumnRole.VALUE, source_concept="stable"),
        source_database="miiv",
    )
    companion = project_concept_column_metadata(
        ConceptDefinition.from_name_and_payload("stable", first_payload),
        spec=_spec("stable_n", ConceptColumnRole.COUNT, source_concept="stable"),
        source_database="miiv",
    )

    assert first.to_dict() == second.to_dict()
    assert first.source_lineage[1].to_dict()["item_ids"] == ["1", 1, 2]
    assert canonical_metadata_bytes(first) == canonical_metadata_bytes(second)
    assert metadata_sha256(first) == metadata_sha256(second)
    assert metadata_payload_sha256({"stable": first, "stable_n": companion}) == (
        metadata_payload_sha256({"stable_n": companion, "stable": second})
    )
    with pytest.raises(MetadataProjectionError, match="payload key"):
        metadata_payload_sha256({"wrong_name": first})
    with pytest.raises(MetadataProjectionError, match="keys must be strings"):
        metadata_payload_sha256({1: first, "stable": first})  # type: ignore[dict-item]


def test_explicit_event_time_coordinates_are_preserved_without_guessing():
    metadata = project_concept_column_metadata(
        _definition("lact"),
        spec=_spec(
            "charttime",
            ConceptColumnRole.EVENT_TIME,
            time_origin="database_native_absolute",
            time_unit="timestamp",
        ),
        source_database="miiv",
    )

    assert metadata.time_origin == "database_native_absolute"
    assert metadata.time_unit == "timestamp"
    assert metadata.canonical_unit is None
    assert metadata.extraction_bounds is None


def test_empty_declared_source_for_derived_concept_is_not_called_undeclared():
    definition = _definition("rrt_criteria")
    assert "miiv" in definition.sources
    assert definition.sources["miiv"] == []
    metadata = project_concept_column_metadata(
        definition,
        spec=_spec(
            "rrt_criteria",
            ConceptColumnRole.VALUE,
            source_concept="rrt_criteria",
        ),
        source_database="miiv",
    )

    assert "miiv" in metadata.available_databases
    assert metadata.source_declared_for_database is True
    assert metadata.availability_basis == "declared_derived_or_unresolved"
    assert metadata.source_lineage == ()


def test_database_class_inheritance_preserves_actual_and_dictionary_source_identity():
    metadata = project_concept_column_metadata(
        _definition("rrt"),
        spec=_spec("rrt", ConceptColumnRole.VALUE, source_concept="rrt"),
        source_database="eicu_demo",
        source_database_class_prefixes=("eicu",),
    )

    assert metadata.source_database == "eicu_demo"
    assert metadata.dictionary_source_database == "eicu"
    assert metadata.source_resolution_chain == ("eicu_demo", "eicu")
    assert metadata.source_declared_for_database is True
    assert metadata.availability_basis == "inherited_direct_source"
    assert len(metadata.source_lineage) == 3
    assert {entry.database for entry in metadata.source_lineage} == {"eicu"}


def test_most_specific_empty_source_does_not_fall_through_to_parent():
    definition = ConceptDefinition.from_name_and_payload(
        "derived_demo",
        {
            "depends_on": ["component"],
            "sources": {
                "eicu_demo": [],
                "eicu": [{"table": "events", "ids": [1]}],
            },
        },
    )
    metadata = project_concept_column_metadata(
        definition,
        spec=_spec(
            "derived_demo",
            ConceptColumnRole.VALUE,
            source_concept="derived_demo",
        ),
        source_database="eicu_demo",
        source_database_class_prefixes=("eicu",),
    )

    assert metadata.dictionary_source_database == "eicu_demo"
    assert metadata.availability_basis == "declared_derived_or_unresolved"
    assert metadata.source_lineage == ()


def test_empty_comment_only_source_entry_is_not_direct_lineage():
    definition = ConceptDefinition.from_name_and_payload(
        "comment_only",
        {
            "sources": {
                "miiv": [{"_comment": "not an extraction binding"}],
            }
        },
    )
    metadata = project_concept_column_metadata(
        definition,
        spec=_spec(
            "comment_only",
            ConceptColumnRole.VALUE,
            source_concept="comment_only",
        ),
        source_database="miiv",
    )

    assert metadata.source_declared_for_database is True
    assert metadata.availability_basis == "declared_without_direct_source"
    assert metadata.source_lineage == ()


@pytest.mark.parametrize(
    "attachment",
    [
        {"unit": "valueuom"},
        {"interval": "6h"},
        {"grp_var": "stay_id"},
        {"ids": [1], "sub_var": "itemid"},
    ],
)
def test_source_attachments_without_executable_anchor_are_not_direct_lineage(
    attachment: dict[str, object],
):
    definition = ConceptDefinition.from_name_and_payload(
        "unanchored",
        {"sources": {"miiv": [attachment]}},
    )
    metadata = project_concept_column_metadata(
        definition,
        spec=_spec(
            "unanchored",
            ConceptColumnRole.VALUE,
            source_concept="unanchored",
        ),
        source_database="miiv",
    )

    assert metadata.source_declared_for_database is True
    assert metadata.availability_basis == "declared_without_direct_source"
    assert metadata.source_lineage == ()


def test_source_resolution_prefixes_are_typed_and_require_an_actual_database():
    definition = _definition("rrt")
    with pytest.raises(MetadataProjectionError, match="must be a sequence"):
        project_concept_column_metadata(
            definition,
            spec=_spec("rrt", ConceptColumnRole.VALUE, source_concept="rrt"),
            source_database="eicu_demo",
            source_database_class_prefixes="eicu",  # type: ignore[arg-type]
        )
    with pytest.raises(MetadataProjectionError, match="require source_database"):
        project_concept_column_metadata(
            definition,
            spec=_spec("rrt", ConceptColumnRole.VALUE, source_concept="rrt"),
            source_database=None,
            source_database_class_prefixes=("eicu",),
        )


def test_source_lineage_includes_semantic_interval_and_params_but_not_comments():
    definition = ConceptDefinition.from_name_and_payload(
        "parameterized",
        {
            "unit": "mL",
            "sources": {
                "miiv": [
                    {
                        "table": "events",
                        "sub_var": "itemid",
                        "ids": [1],
                        "interval": "6h",
                        "stop_var": "endtime",
                        "grp_var": "stay_id",
                        "unit_val": {"mL": 1, "L": 1000},
                        "_comment": "non-semantic prose must not affect authority",
                    }
                ]
            },
        },
    )
    metadata = project_concept_column_metadata(
        definition,
        spec=_spec(
            "parameterized",
            ConceptColumnRole.VALUE,
            source_concept="parameterized",
        ),
        source_database="miiv",
    )
    lineage = metadata.source_lineage[0].to_dict()

    assert lineage["interval_iso8601"] == "P0DT6H0M0S"
    assert lineage["semantic_parameters"] == {
        "grp_var": "stay_id",
        "stop_var": "endtime",
        "unit_val": {"L": 1000, "mL": 1},
    }
    assert "_comment" not in lineage["semantic_parameters"]


def test_negative_zero_is_canonicalized_in_bounds_ids_and_parameters():
    negative = ConceptDefinition.from_name_and_payload(
        "signed_zero",
        {
            "min": -0.0,
            "max": 1,
            "sources": {"miiv": [{"table": "events", "ids": [-0.0], "scale": -0.0}]},
        },
    )
    positive = ConceptDefinition.from_name_and_payload(
        "signed_zero",
        {
            "min": 0.0,
            "max": 1,
            "sources": {"miiv": [{"table": "events", "ids": [0.0], "scale": 0.0}]},
        },
    )
    spec = _spec(
        "signed_zero",
        ConceptColumnRole.VALUE,
        source_concept="signed_zero",
    )

    assert metadata_sha256(
        project_concept_column_metadata(
            negative,
            spec=spec,
            source_database="miiv",
        )
    ) == metadata_sha256(
        project_concept_column_metadata(
            positive,
            spec=spec,
            source_database="miiv",
        )
    )


def test_leaf_projector_has_no_web_or_research_agent_dependency():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert all("research_agent" not in name for name in imported)
    assert all("webserver" not in name for name in imported)
    assert all("webapp" not in name for name in imported)
