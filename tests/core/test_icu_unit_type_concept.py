from __future__ import annotations

import json
from pathlib import Path

from easyicu.api.extraction import EXTRACT_MODULES
from easyicu.concept.catalog import CONCEPT_DICTIONARY


def test_icu_unit_type_is_a_demographics_export_concept():
    assert CONCEPT_DICTIONARY["icu_unit_type"] == (
        "ICU Unit Type",
        "ICU单元类型",
        "category",
    )
    assert "icu_unit_type" in EXTRACT_MODULES["demographics"]


def test_icu_unit_type_reads_the_eicu_patient_unit_type_field():
    path = Path("src/easyicu/data/concept-dict.json")
    definition = json.loads(path.read_text(encoding="utf-8"))["icu_unit_type"]

    assert definition["target"] == "id_tbl"
    assert definition["class_name"] == "fct_cncpt"
    for database in ("eicu", "eicu_demo"):
        assert definition["sources"][database] == [
            {
                "table": "patient",
                "val_var": "unittype",
                "class_name": "col_itm",
            }
        ]
