from __future__ import annotations

import json
import re
from pathlib import Path


DICTIONARY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "data"
    / "concept-dict.json"
)


def _dictionary() -> dict:
    return json.loads(DICTIONARY.read_text(encoding="utf-8"))


def test_kdigo_uses_explicit_phenotype_input_contracts() -> None:
    dictionary = _dictionary()

    assert dictionary["kdigo_aki"]["concepts"] == [
        "kdigo_creatinine_input",
        "kdigo_urine_input",
        "weight",
        "acute_rrt_input",
    ]
    assert dictionary["kdigo_creat"]["concepts"] == [
        "kdigo_creatinine_input"
    ]
    assert dictionary["kdigo_uo"]["concepts"] == [
        "kdigo_urine_input",
        "weight",
    ]
    assert (
        dictionary["kdigo_creatinine_input"][
            "pre_admission_lookback_hours"
        ]
        == 168
    )


def test_official_mimic_gu_irrigation_ids_and_transform_are_declared() -> None:
    sources = _dictionary()["urine"]["sources"]
    for database in ("miiv", "mimic", "mimic_demo"):
        source = sources[database][0]
        assert {227488, 227489}.issubset(set(source["ids"]))
        assert source["callback"] == "mimic_urine_output"


def test_eicu_urine_mapping_is_scoped_to_official_output_cellpaths() -> None:
    source = _dictionary()["urine"]["sources"]["eicu"][0]
    pattern = re.compile(source["regex"])
    prefix = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"

    accepted = [
        prefix + "Urine",
        prefix + "Urine Output-left nephrostomy",
        prefix + "Straight Catheter Output",
        prefix + "foley catheter",
    ]
    rejected = [
        "flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|Urine",
        prefix + "Urine/Stool mixed output",
        prefix + "foley PACU",
        prefix + "Chest tube output",
    ]

    assert all(pattern.search(value) for value in accepted)
    assert not any(pattern.search(value) for value in rejected)


def test_aumc_urine_contract_uses_official_repair_callback() -> None:
    source = _dictionary()["urine"]["sources"]["aumc"][0]

    assert source["ids"] == [8794, 8796, 8798, 8800, 8803, 10743, 10745, 19921, 19922]
    assert source["callback"] == "aumc_urine_output"
    assert _dictionary()["urine"]["max"] == 5000
