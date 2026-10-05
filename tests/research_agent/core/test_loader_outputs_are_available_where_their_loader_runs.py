"""Outputs of code loaders are available where their loader runs.

Some concepts come from code loaders rather than from the concept dictionary:
comorbidity indices, culture results, circulatory failure, Sepsis-3 (SOFA-1),
creatinine baselines and urine-output rates.  The cross-database availability
owner read only the dictionary, so it reported every one of them as not found
on every database, and the planner's concept menu, hypothesis feasibility and
idea mining never offered them.  Each loader now declares once what it needs;
the loader and the availability owner both read that declaration.
Packaged metadata and synthetic declarations only; no data source is opened.
"""

from __future__ import annotations

import pandas as pd
import pytest

import easyicu.concept_output_sources as output_sources
from easyicu.concept_output_sources import (
    COMPOSITE_CONCEPT_OUTPUT_SOURCES,
    COMPOSITE_LOADER_SUPPORT,
    CONCEPT_OUTPUT_LOAD_SOURCES,
    CompositeLoaderSupport,
)
from easyicu.research_agent import concept_availability
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.research_agent.concept_availability import (
    PUBLIC_DATABASES,
    explain_concept_availability,
)
from easyicu.resources import load_dictionary
from easyicu.scores import circ_failure, comorbidity, microbiology

_VIEW = (
    "status",
    "available",
    "structural_unavailable",
    "available_dependencies",
    "degraded_dependencies",
    "missing_dependencies",
)


def _cell(concept: str, database: str):
    return explain_concept_availability(concept=concept, database=database)


def _view(cell) -> tuple:
    return tuple(getattr(cell, field) for field in _VIEW)


def _outputs_of(loader: str) -> list[str]:
    return [
        output
        for output, source in COMPOSITE_CONCEPT_OUTPUT_SOURCES.items()
        if source == loader
    ]


@pytest.fixture
def fresh_availability():
    concept_availability._explain_concept_availability_cached.cache_clear()
    yield
    concept_availability._explain_concept_availability_cached.cache_clear()


def test_no_loader_output_is_reported_missing_from_the_dictionary() -> None:
    for output in COMPOSITE_CONCEPT_OUTPUT_SOURCES:
        for database in PUBLIC_DATABASES:
            assert _cell(output, database).reason != "concept_not_found", (
                output,
                database,
            )


def test_an_output_of_a_dictionary_concept_is_available_where_its_source_is() -> None:
    dictionary = load_dictionary(include_sofa2=True)
    pairs = {
        output: source
        for output, source in {
            **COMPOSITE_CONCEPT_OUTPUT_SOURCES,
            **CONCEPT_OUTPUT_LOAD_SOURCES,
        }.items()
        if dictionary.get(output) is None and dictionary.get(source) is not None
    }
    assert {"sep3_sofa1", "creat_low_past_48hr", "uo_rt_6hr"} <= set(pairs)

    for output, source in pairs.items():
        for database in PUBLIC_DATABASES:
            cell = _cell(output, database)
            assert (cell.concept, cell.requested_concept) == (output, output)
            assert _view(cell) == _view(_cell(source, database)), (output, database)


def test_a_database_without_the_loaders_source_blocks_its_outputs() -> None:
    checked: set[str] = set()
    for loader in ("comorbidity_loader", "microbiology_loader"):
        support = COMPOSITE_LOADER_SUPPORT[loader]
        for output in _outputs_of(loader):
            for database in PUBLIC_DATABASES:
                cell = _cell(output, database)
                if database in support.no_source_databases:
                    assert (cell.status, cell.structural_unavailable, cell.reason) == (
                        "blocked",
                        True,
                        support.no_source_reason,
                    ), (output, database)
                else:
                    assert (cell.status, cell.available, cell.reason) == (
                        "full",
                        True,
                        "loader_source_available",
                    ), (output, database)
            checked.add(output)

    assert {"charlson", "elixhauser", "culture_positive", "bld_culture_positive"} <= checked
    # A demo copy carries its database's tables.
    assert _cell("charlson", "aumc-demo").status == "blocked"


@pytest.mark.parametrize(
    ("required", "optional", "expected"),
    [
        (
            ("lact", "no_such_input"),
            (),
            ("blocked", "required_dependency_blocked", ["no_such_input"], True),
        ),
        (
            ("lact",),
            ("no_such_input",),
            ("degraded", "partial_dependency_availability", ["no_such_input"], False),
        ),
        (("lact",), (), ("full", "all_dependencies_available", [], False)),
        ((), (), ("full", "loader_source_available", [], False)),
    ],
)
def test_a_loaders_inputs_decide_its_outputs(
    monkeypatch, fresh_availability, required, optional, expected
) -> None:
    monkeypatch.setattr(
        output_sources,
        "COMPOSITE_CONCEPT_OUTPUT_SOURCES",
        {**COMPOSITE_CONCEPT_OUTPUT_SOURCES, "synthetic_output": "synthetic_loader"},
    )
    monkeypatch.setattr(
        output_sources,
        "COMPOSITE_LOADER_SUPPORT",
        {
            "synthetic_loader": CompositeLoaderSupport(
                required_concepts=required, optional_concepts=optional
            )
        },
    )

    cell = _cell("synthetic_output", "miiv")

    assert (
        cell.status,
        cell.reason,
        cell.missing_dependencies,
        cell.structural_unavailable,
    ) == expected
    assert cell.available is (expected[0] != "blocked")


def test_circulatory_failure_is_available_where_lactate_and_map_are() -> None:
    support = COMPOSITE_LOADER_SUPPORT["circ_failure_loader"]
    inputs = [*support.required_concepts, *support.optional_concepts]

    for database in PUBLIC_DATABASES:
        cell = _cell("circ_failure", database)
        assert cell.available is all(
            _cell(concept, database).available for concept in support.required_concepts
        )
        assert cell.missing_dependencies == [
            concept for concept in inputs if _cell(concept, database).status == "blocked"
        ]
        assert _view(_cell("circ_event", database)) == _view(cell)


def test_the_planner_menu_offers_loader_outputs_where_they_are_available() -> None:
    offered = {item.concept_id for item in build_database_capability_catalog("miiv").concepts}
    assert {
        "circ_failure",
        "circ_event",
        "sep3_sofa1",
        "culture_positive",
        "bld_culture_positive",
        "creat_low_past_48hr",
        "uo_rt_6hr",
    } <= offered

    without_sources = {
        item.concept_id for item in build_database_capability_catalog("hirid").concepts
    }
    assert "circ_failure" in without_sources
    assert not {"culture_positive", "bld_culture_positive"} & without_sources


def test_the_comorbidity_and_culture_loaders_read_the_declaration(monkeypatch) -> None:
    def no_tables(*args, **kwargs):
        pytest.fail("a database declared without the source must not open a table")

    monkeypatch.setattr(comorbidity, "_build_datasource", no_tables)
    monkeypatch.setattr(microbiology, "build_datasource", no_tables)
    for database in COMPOSITE_LOADER_SUPPORT["comorbidity_loader"].no_source_databases:
        assert comorbidity.load_comorbidity(database).empty
    for database in COMPOSITE_LOADER_SUPPORT["microbiology_loader"].no_source_databases:
        assert microbiology.load_microbiology(database).empty

    # Whatever the declaration names, the loader follows it.
    declared = {
        loader: CompositeLoaderSupport(
            no_source_databases=frozenset({"miiv"}), no_source_reason="synthetic"
        )
        for loader in ("comorbidity_loader", "microbiology_loader")
    }
    monkeypatch.setattr(comorbidity, "COMPOSITE_LOADER_SUPPORT", declared)
    monkeypatch.setattr(microbiology, "COMPOSITE_LOADER_SUPPORT", declared)
    assert comorbidity.load_comorbidity("miiv").empty
    assert microbiology.load_microbiology("miiv").empty


def _stream(name: str) -> pd.DataFrame:
    return pd.DataFrame({"stay_id": [1, 1], "charttime": [0, 5], name: [1.0, 1.0]})


def test_the_circulatory_failure_loader_loads_its_declared_inputs(monkeypatch) -> None:
    calls: list[list[str]] = []
    rates = ("norepi_rate", "epi_rate", "adh_rate")

    def load_concepts(**kwargs):
        calls.append(list(kwargs["concepts"]))
        return {name: _stream(name) for name in kwargs["concepts"] if name in rates}

    monkeypatch.setattr("easyicu.api.load_concepts", load_concepts)
    preloaded = {"lact": _stream("lact"), "map": _stream("map")}

    monkeypatch.setattr(
        circ_failure,
        "COMPOSITE_LOADER_SUPPORT",
        {
            "circ_failure_loader": CompositeLoaderSupport(
                required_concepts=("lact", "map"), optional_concepts=rates
            )
        },
    )
    result = circ_failure.load_circ_failure("miiv", preloaded_data=preloaded, verbose=False)

    assert calls == [list(rates)]
    assert "circ_event" in result.columns

    # A declared required input that cannot be loaded stops the loader.
    monkeypatch.setattr(
        circ_failure,
        "COMPOSITE_LOADER_SUPPORT",
        {
            "circ_failure_loader": CompositeLoaderSupport(
                required_concepts=("lact", "map", "synthetic_core")
            )
        },
    )
    with pytest.raises(ValueError, match="synthetic_core"):
        circ_failure.load_circ_failure("miiv", preloaded_data=preloaded, verbose=False)
