from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.clinical_contracts import (
    load_clinical_contracts,
    render_clinical_conformance_matrix_markdown,
    validate_clinical_contracts,
)
from easyicu.config import DataSourceConfig
from easyicu.concept import ConceptResolver, ConceptSource
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sofa2_score,
    _callback_sofa_component,
)
from easyicu.resources import load_dictionary
from easyicu.scores.kdigo_aki import _calc_aki_stage_creat
from easyicu.scores.sepsis import sep3
from easyicu.scores.sepsis_sofa2 import sep3_sofa2
from easyicu.scores.sofa2 import (
    sofa2_cardio,
    sofa2_cns,
    sofa2_cns_ascertainment,
    sofa2_cns_proxy_sensitivity,
    sofa2_coag,
    sofa2_liver,
    sofa2_renal,
    sofa2_resp,
    sofa2_score,
)
from easyicu.table import ICUTable


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.clinical_conformance
def test_clinical_contract_registry_has_complete_definition_and_vector_coverage() -> None:
    dictionary = load_dictionary(include_sofa2=True)

    assert validate_clinical_contracts(dictionary, repo_root=ROOT) == []


@pytest.mark.clinical_conformance
def test_registry_separates_spec_and_runtime_golden_vectors() -> None:
    registry = json.loads(
        (ROOT / "src/easyicu/data/clinical-contracts.json").read_text(
            encoding="utf-8"
        )
    )

    for contract in registry.values():
        assert "golden_vector" not in contract
        assert contract["spec_golden_vectors"]
        assert contract["runtime_golden_vectors"]

    cns = registry["sofa2_cns_2025"]
    assert cns["spec_golden_vectors"] != cns["runtime_golden_vectors"]


@pytest.mark.clinical_conformance
def test_sofa2_aggregate_cannot_outrank_or_omit_a_component_contract(tmp_path: Path) -> None:
    registry_path = ROOT / "src/easyicu/data/clinical-contracts.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    aggregate = registry["sofa2_aggregate_2025"]
    aggregate["status"] = "validated_definition"
    aggregate["depends_on_contracts"].remove("sofa2_renal_2025")
    mutated = tmp_path / "clinical-contracts.json"
    mutated.write_text(json.dumps(registry), encoding="utf-8")

    findings = validate_clinical_contracts(
        load_dictionary(include_sofa2=True),
        repo_root=ROOT,
        contracts_path=mutated,
    )

    assert "sofa2_aggregate_2025:status_exceeds_weakest_dependency" in findings
    assert "sofa2_aggregate_2025:component_dependencies_incomplete" in findings
    assert "sofa2_aggregate_2025:dictionary_status_mismatch:sofa2" in findings


@pytest.mark.clinical_conformance
def test_sofa2_contract_rejects_fixture_input_without_runtime_owner(tmp_path: Path) -> None:
    registry_path = ROOT / "src/easyicu/data/clinical-contracts.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    del registry["sofa2_resp_2025"]["runtime_inputs"]["pafi"]
    mutated = tmp_path / "clinical-contracts.json"
    mutated.write_text(json.dumps(registry), encoding="utf-8")

    findings = validate_clinical_contracts(
        load_dictionary(include_sofa2=True),
        repo_root=ROOT,
        contracts_path=mutated,
    )

    assert "sofa2_resp_2025:runtime_inputs_mismatch:sofa2_resp" in findings
    assert "sofa2_resp_2025:runtime_golden_input_unowned:pafi" in findings


@pytest.mark.clinical_conformance
def test_sofa2_aggregate_rejects_component_without_runtime_owner(tmp_path: Path) -> None:
    registry_path = ROOT / "src/easyicu/data/clinical-contracts.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    del registry["sofa2_aggregate_2025"]["runtime_inputs"]["sofa2_resp"]
    mutated = tmp_path / "clinical-contracts.json"
    mutated.write_text(json.dumps(registry), encoding="utf-8")

    findings = validate_clinical_contracts(
        load_dictionary(include_sofa2=True),
        repo_root=ROOT,
        contracts_path=mutated,
    )

    assert "sofa2_aggregate_2025:runtime_inputs_mismatch:sofa2" in findings
    assert (
        "sofa2_aggregate_2025:runtime_golden_input_unowned:sofa2_resp"
        in findings
    )


@pytest.mark.clinical_conformance
def test_kdigo_creatinine_golden_vectors_are_executed_from_independent_fixture() -> None:
    fixture = json.loads(
        (ROOT / "tests/clinical_specs/kdigo_aki_2012.json").read_text(encoding="utf-8")
    )["creatinine_vectors"]
    result = _calc_aki_stage_creat(
        pd.Series([row["current"] for row in fixture]),
        pd.Series([row["low_48h"] for row in fixture]),
        pd.Series([row["low_7d"] for row in fixture]),
    )

    assert result.tolist() == [row["expected_stage"] for row in fixture]


_SOFA2_COMPONENT_EXECUTORS = [
    ("sofa2_resp_2025.json", "sofa2_resp_2025_spec.json", "sofa2_resp", sofa2_resp),
    ("sofa2_coag_2025.json", "sofa2_coag_2025_spec.json", "sofa2_coag", sofa2_coag),
    ("sofa2_liver_2025.json", "sofa2_liver_2025_spec.json", "sofa2_liver", sofa2_liver),
    ("sofa2_cardio_2025.json", "sofa2_cardio_2025_spec.json", "sofa2_cardio", sofa2_cardio),
    ("sofa2_cns_2025.json", "sofa2_cns_2025_spec.json", "sofa2_cns", sofa2_cns),
    ("sofa2_renal_2025.json", "sofa2_renal_2025_spec.json", "sofa2_renal", sofa2_renal),
]


def _clinical_context(name: str, **kwargs) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name=name,
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
        kwargs=kwargs,
    )


def _clinical_table(name: str, values) -> ICUTable:
    return ICUTable(
        pd.DataFrame(
            {
                "stay_id": [1] * len(values),
                "charttime": list(range(len(values))),
                name: values,
            }
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column=name,
    )


@pytest.mark.clinical_conformance
@pytest.mark.parametrize(
    ("fixture_name", "spec_fixture_name", "concept_name", "executor"),
    _SOFA2_COMPONENT_EXECUTORS,
)
def test_sofa2_component_golden_vectors_execute_direct_and_production_paths(
    fixture_name: str,
    spec_fixture_name: str,
    concept_name: str,
    executor,
) -> None:
    del spec_fixture_name
    fixture = json.loads(
        (ROOT / "tests/clinical_specs" / fixture_name).read_text(encoding="utf-8")
    )
    inputs = {name: pd.Series(values) for name, values in fixture["inputs"].items()}

    direct = executor(**inputs)
    production = _callback_sofa_component(executor)(
        {name: _clinical_table(name, values) for name, values in fixture["inputs"].items()},
        _clinical_context(concept_name),
    )

    assert direct.tolist() == fixture["expected"]
    assert production.data[concept_name].tolist() == fixture["expected"]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize(
    ("runtime_fixture_name", "fixture_name", "concept_name", "executor"),
    _SOFA2_COMPONENT_EXECUTORS,
)
def test_sofa2_spec_golden_vectors_are_not_limited_to_runtime_inputs(
    runtime_fixture_name: str,
    fixture_name: str,
    concept_name: str,
    executor,
) -> None:
    del runtime_fixture_name
    fixture = json.loads(
        (ROOT / "tests/clinical_specs" / fixture_name).read_text(encoding="utf-8")
    )
    inputs = {name: pd.Series(values) for name, values in fixture["inputs"].items()}

    assert executor(**inputs).tolist() == fixture["expected"]
    if concept_name == "sofa2_cns":
        assert sofa2_cns_proxy_sensitivity(**inputs).tolist() == fixture[
            "expected_proxy_sensitivity"
        ]
        assert sofa2_cns_ascertainment(**inputs).tolist() == fixture[
            "expected_ascertainment"
        ]


class _FixtureDataSource:
    base_path = None
    config = DataSourceConfig(
        name="fixture",
        tables={
            "events": {
                "defaults": {
                    "id_var": "stay_id",
                    "index_var": "charttime",
                    "val_var": "value",
                }
            }
        },
    )

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        del table_name, filters, verbose
        frame = self.frame.copy()
        if columns:
            keep = list(
                dict.fromkeys(
                    [
                        "stay_id",
                        "charttime",
                        *(column for column in columns if column in frame.columns),
                    ]
                )
            )
            frame = frame[keep]
        values = [c for c in frame if c not in {"stay_id", "charttime"}]
        return ICUTable(
            frame,
            id_columns=["stay_id"],
            index_column="charttime",
            value_column=values[0] if len(values) == 1 else None,
        )


@pytest.mark.clinical_conformance
def test_delirium_proxy_evidence_and_legacy_alias_resolve_from_dictionary() -> None:
    dictionary = copy.deepcopy(load_dictionary(include_sofa2=True))
    dictionary["delirium_tx_proxy"].sources["fixture"] = [
        ConceptSource(
            table="events",
            value_var="delirium_tx_proxy",
            index_var="charttime",
        )
    ]
    events = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 1],
            "delirium_tx_proxy": [True, False],
        }
    )

    loaded = ConceptResolver(dictionary).load_concepts(
        ["delirium_tx_proxy", "delirium_tx_evidence", "delirium_tx"],
        _FixtureDataSource(events),
        merge=False,
        interval=pd.Timedelta(hours=1),
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )

    assert loaded["delirium_tx_proxy"].data["delirium_tx_proxy"].tolist() == [
        True,
        False,
    ]
    assert loaded["delirium_tx_evidence"].data[
        "delirium_tx_evidence"
    ].tolist() == ["proxy_only", "unavailable"]
    assert loaded["delirium_tx"].data["delirium_tx"].tolist() == [True, False]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize(
    ("fixture_name", "spec_fixture_name", "concept_name", "executor"),
    _SOFA2_COMPONENT_EXECUTORS,
)
def test_shipped_sofa2_component_resolver_graphs_execute_runtime_vectors(
    fixture_name: str,
    spec_fixture_name: str,
    concept_name: str,
    executor,
) -> None:
    del spec_fixture_name, executor
    fixture = json.loads(
        (ROOT / "tests/clinical_specs" / fixture_name).read_text(encoding="utf-8")
    )
    dictionary = copy.deepcopy(load_dictionary(include_sofa2=True))
    for input_name in fixture["inputs"]:
        dictionary[input_name].sources["fixture"] = [
            ConceptSource(
                table="events",
                value_var=input_name,
                index_var="charttime",
            )
        ]
    row_count = len(next(iter(fixture["inputs"].values())))
    events = pd.DataFrame(
        {
            "stay_id": [1] * row_count,
            "charttime": list(range(row_count)),
            **fixture["inputs"],
        }
    )

    loaded = ConceptResolver(dictionary).load_concepts(
        [concept_name],
        _FixtureDataSource(events),
        merge=False,
        interval=pd.Timedelta(hours=1),
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )

    assert loaded[concept_name].data[concept_name].tolist() == fixture["expected"]


@pytest.mark.clinical_conformance
def test_sofa2_aggregate_golden_vector_preserves_completeness() -> None:
    fixture = json.loads(
        (ROOT / "tests/clinical_specs/sofa2_aggregate_2025.json").read_text(encoding="utf-8")
    )
    frames = {
        name: pd.DataFrame({"stay_id": [1], name: [score]})
        for name, score in fixture["components"].items()
    }
    result = sofa2_score(frames)

    production_frames = {
        name: _clinical_table(name, [score])
        for name, score in fixture["components"].items()
    }
    production = _callback_sofa2_score(
        production_frames,
        _clinical_context("sofa2"),
    ).data

    assert result["sofa2"].tolist() == [fixture["expected_total"]]
    assert result["sofa2_n_observed_components"].tolist() == [
        fixture["expected_observed_components"]
    ]
    assert result["sofa2_n_available_components"].tolist() == [
        fixture["expected_available_components"]
    ]
    assert result["sofa2_n_components"].tolist() == [
        fixture["expected_available_components"]
    ]
    assert production["sofa2"].tolist() == [fixture["expected_total"]]
    assert production["sofa2_n_observed_components"].tolist() == [
        fixture["expected_observed_components"]
    ]
    assert production["sofa2_n_available_components"].tolist() == [
        fixture["expected_available_components"]
    ]
    assert production["sofa2_n_components"].tolist() == [
        fixture["expected_available_components"]
    ]


@pytest.mark.clinical_conformance
def test_sofa2_aggregate_spec_vector_executes_independently() -> None:
    fixture = json.loads(
        (ROOT / "tests/clinical_specs/sofa2_aggregate_2025_spec.json").read_text(
            encoding="utf-8"
        )
    )
    frames = {
        name: pd.DataFrame({"stay_id": [1], name: [score]})
        for name, score in fixture["components"].items()
    }

    result = sofa2_score(frames)

    assert result["sofa2"].tolist() == [fixture["expected_total"]]
    assert result["sofa2_n_observed_components"].tolist() == [
        fixture["expected_observed_components"]
    ]
    assert result["sofa2_n_available_components"].tolist() == [
        fixture["expected_available_components"]
    ]


@pytest.mark.clinical_conformance
def test_shipped_sofa2_aggregate_resolver_graph_executes_runtime_vector() -> None:
    fixture = json.loads(
        (ROOT / "tests/clinical_specs/sofa2_aggregate_2025.json").read_text(
            encoding="utf-8"
        )
    )
    dictionary = copy.deepcopy(load_dictionary(include_sofa2=True))
    for component_name in fixture["components"]:
        dictionary[component_name].sources["fixture"] = [
            ConceptSource(
                table="events",
                value_var=component_name,
                index_var="charttime",
            )
        ]
    events = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0],
            **{name: [value] for name, value in fixture["components"].items()},
        }
    )

    loaded = ConceptResolver(dictionary).load_concepts(
        ["sofa2"],
        _FixtureDataSource(events),
        merge=False,
        interval=pd.Timedelta(hours=1),
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )["sofa2"].data

    assert loaded["sofa2"].tolist() == [fixture["expected_total"]]
    assert loaded["sofa2_n_observed_components"].tolist() == [
        fixture["expected_observed_components"]
    ]
    assert loaded["sofa2_n_available_components"].tolist() == [
        fixture["expected_available_components"]
    ]
    assert loaded["sofa2_n_components"].tolist() == [
        fixture["expected_available_components"]
    ]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize(
    ("fixture_name", "score_column", "event_column", "executor"),
    [
        ("sepsis3_2016.json", "sofa", "sep3", sep3),
        (
            "sepsis3_sofa2_sensitivity_2025.json",
            "sofa2",
            "sep3_sofa2",
            sep3_sofa2,
        ),
    ],
)
def test_sepsis_golden_vectors_execute_the_bound_phenotype(
    fixture_name: str,
    score_column: str,
    event_column: str,
    executor,
) -> None:
    fixture = json.loads(
        (ROOT / "tests/clinical_specs" / fixture_name).read_text(encoding="utf-8")
    )
    score = pd.DataFrame(
        {
            "stay_id": [1] * len(fixture["sofa_times"]),
            "time": fixture["sofa_times"],
            score_column: fixture["sofa_values"],
        }
    )
    suspected_infection = pd.DataFrame(
        {
            "stay_id": [1],
            "time": [fixture["suspected_infection_time"]],
            "susp_inf": [True],
        }
    )
    kwargs = {
        "id_cols": ["stay_id"],
        "index_col": "time",
    }
    result = executor(score, suspected_infection, **kwargs)

    assert result[event_column].tolist() == [True]
    assert result["time"].tolist() == [fixture["expected_event_time"]]


def test_committed_clinical_conformance_matrix_is_generated_from_registry() -> None:
    committed = (ROOT / "docs/clinical_conformance_matrix.md").read_text(encoding="utf-8")
    assert committed == render_clinical_conformance_matrix_markdown()
    assert set(load_clinical_contracts()) >= {
        "kdigo_aki_2012",
        "sofa2_resp_2025",
        "sofa2_coag_2025",
        "sofa2_liver_2025",
        "sofa2_cardio_2025",
        "sofa2_cns_2025",
        "sofa2_renal_2025",
        "sepsis3_sofa2_sensitivity_2025",
    }
