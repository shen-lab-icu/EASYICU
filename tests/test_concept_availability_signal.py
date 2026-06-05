from __future__ import annotations

import logging

import pandas as pd

import easyicu.concept as concept_module
from easyicu.config import DataSourceConfig
from easyicu.concept import ConceptResolver
from easyicu.concept_availability_signal import ConceptAvailabilityRecord
from easyicu.concept_schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.table import ICUTable


class AvailabilityDataSource:
    base_path = None

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.config = DataSourceConfig(
            name="unit",
            tables={
                "events": {
                    "defaults": {
                        "id_var": "stay_id",
                        "index_var": "charttime",
                        "val_var": "value",
                    }
                },
                "missing_events": {
                    "defaults": {
                        "id_var": "stay_id",
                        "index_var": "charttime",
                        "val_var": "value",
                    }
                }
            },
        )

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        del filters, verbose
        if self.mode == "unavailable" or (
            self.mode == "partial" and table_name == "missing_events"
        ):
            raise FileNotFoundError("events parquet is missing")
        if self.mode == "empty":
            frame = pd.DataFrame(columns=["stay_id", "charttime", "value"])
        else:
            frame = pd.DataFrame(
                {
                    "stay_id": [1],
                    "charttime": [0.0],
                    "value": [42.0],
                }
            )
        if columns:
            keep_cols = ["stay_id", "charttime"] + [
                col for col in columns if col in frame.columns
            ]
            frame = frame[list(dict.fromkeys(keep_cols))]
        return ICUTable(
            data=frame,
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="value",
        )


def _dictionary(*concept_names: str) -> ConceptDictionary:
    return ConceptDictionary(
        {
            concept_name: ConceptDefinition(
                name=concept_name,
                sources={
                    "unit": [
                        ConceptSource(
                            table="events",
                            value_var="value",
                        )
                    ]
                },
            )
            for concept_name in concept_names
        }
    )


def test_availability_sink_marks_unmapped_without_warning(caplog):
    dictionary = ConceptDictionary(
        {
            "not_in_unit": ConceptDefinition(
                name="not_in_unit",
                sources={
                    "other": [
                        ConceptSource(table="events", value_var="value"),
                    ]
                },
            )
        }
    )
    sink: dict[str, ConceptAvailabilityRecord] = {}
    caplog.set_level(logging.WARNING, logger="easyicu.concept")

    loaded = ConceptResolver(dictionary).load_concepts(
        ["not_in_unit"],
        AvailabilityDataSource(mode="present"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink=sink,
    )

    assert loaded["not_in_unit"].data.empty
    assert sink["not_in_unit"].reason == "unmapped"
    assert sink["not_in_unit"].status == "blocked"
    assert sink["not_in_unit"].sources_defined == ()
    assert not caplog.records


def test_availability_sink_marks_source_unavailable_and_dedupes_warning(caplog):
    concept_module._MISSING_SOURCE_WARNED.clear()
    sink: dict[str, ConceptAvailabilityRecord] = {}
    caplog.set_level(logging.WARNING, logger="easyicu.concept")

    ConceptResolver(_dictionary("first", "second")).load_concepts(
        ["first", "second"],
        AvailabilityDataSource(mode="unavailable"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink=sink,
    )

    assert sink["first"].reason == "source_unavailable"
    assert sink["first"].missing_tables == ("events",)
    assert sink["second"].reason == "source_unavailable"
    warnings = [
        record
        for record in caplog.records
        if "marked source_unavailable" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "events" in warnings[0].getMessage()
    assert "first" in warnings[0].getMessage()
    assert "second" in warnings[0].getMessage()

    ConceptResolver(_dictionary("first", "second")).load_concepts(
        ["first", "second"],
        AvailabilityDataSource(mode="unavailable"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink={},
    )
    repeated_warnings = [
        record
        for record in caplog.records
        if "marked source_unavailable" in record.getMessage()
    ]
    assert len(repeated_warnings) == 1


def test_availability_sink_marks_empty_loaded_table_as_data_missing():
    sink: dict[str, ConceptAvailabilityRecord] = {}

    ConceptResolver(_dictionary("empty_value")).load_concepts(
        ["empty_value"],
        AvailabilityDataSource(mode="empty"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink=sink,
    )

    record = sink["empty_value"]
    assert record.reason == "data_missing"
    assert record.status == "degraded"
    assert record.n_rows == 0
    assert record.missing_tables == ()


def test_availability_sink_marks_present_rows():
    sink: dict[str, ConceptAvailabilityRecord] = {}

    ConceptResolver(_dictionary("value_concept")).load_concepts(
        ["value_concept"],
        AvailabilityDataSource(mode="present"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink=sink,
    )

    record = sink["value_concept"]
    assert record.reason == "mapped_present"
    assert record.status == "full"
    assert record.n_rows == 1
    assert record.sources_defined == ("events",)


def test_partial_missing_source_keeps_present_concept_with_audit_trail():
    dictionary = ConceptDictionary(
        {
            "multi_source": ConceptDefinition(
                name="multi_source",
                sources={
                    "unit": [
                        ConceptSource(table="missing_events", value_var="value"),
                        ConceptSource(table="events", value_var="value"),
                    ]
                },
            )
        }
    )
    sink: dict[str, ConceptAvailabilityRecord] = {}

    ConceptResolver(dictionary).load_concepts(
        ["multi_source"],
        AvailabilityDataSource(mode="partial"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink=sink,
    )

    record = sink["multi_source"]
    assert record.reason == "mapped_present"
    assert record.status == "full"
    assert record.sources_defined == ("missing_events", "events")
    assert record.missing_tables == ("missing_events",)


def test_load_concepts_return_value_is_unchanged_without_availability_sink():
    resolver = ConceptResolver(_dictionary("value_concept"))
    data_source = AvailabilityDataSource(mode="present")

    without_sink = resolver.load_concepts(
        ["value_concept"],
        data_source,
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )
    sink: dict[str, ConceptAvailabilityRecord] = {}
    with_sink = ConceptResolver(_dictionary("value_concept")).load_concepts(
        ["value_concept"],
        AvailabilityDataSource(mode="present"),
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
        availability_sink=sink,
    )

    assert without_sink["value_concept"].data.equals(
        with_sink["value_concept"].data
    )
