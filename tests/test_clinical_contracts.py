from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.clinical_contracts import (
    load_clinical_contracts,
    render_clinical_conformance_matrix_markdown,
    validate_clinical_contracts,
)
from easyicu.resources import load_dictionary
from easyicu.scores.kdigo_aki import _calc_aki_stage_creat
from easyicu.scores.sepsis import sep3
from easyicu.scores.sepsis_sofa2 import sep3_sofa2
from easyicu.scores.sofa2 import sofa2_cns, sofa2_score


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.clinical_conformance
def test_clinical_contract_registry_has_complete_definition_and_vector_coverage() -> None:
    dictionary = load_dictionary(include_sofa2=True)

    assert validate_clinical_contracts(dictionary, repo_root=ROOT) == []


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


@pytest.mark.clinical_conformance
def test_sofa2_cns_golden_vectors_are_executed_from_independent_fixture() -> None:
    vectors = json.loads(
        (ROOT / "tests/clinical_specs/sofa2_cns_2025.json").read_text(encoding="utf-8")
    )["vectors"]
    result = sofa2_cns(
        pd.Series([row["gcs"] for row in vectors]),
        delirium_tx=pd.Series([row["delirium_treatment"] for row in vectors]),
        delirium_positive=pd.Series([row["cam_positive"] for row in vectors]),
    )

    assert result.tolist() == [row["expected"] for row in vectors]


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

    assert result["sofa2"].tolist() == [fixture["expected_total"]]
    assert result["sofa2_n_components"].tolist() == [fixture["expected_components_observed"]]


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
        "sofa2_cns_2025",
        "sepsis3_sofa2_sensitivity_2025",
    }
