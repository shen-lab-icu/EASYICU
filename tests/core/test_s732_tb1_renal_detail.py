"""Reviewed kidney-detail dictionary and synthetic source contracts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, RENAL_DETAIL_METADATA
from easyicu.resources import load_dictionary

EXPECTED = json.loads(r'''
{
  "abdominal_circumference": {
    "unit": "cm",
    "min": 0,
    "max": 300,
    "description": "Abdominal circumference; flagged proxy, not pressure",
    "category": "ventilator",
    "sources": {
      "aumc": [
        {
          "ids": [
            12087
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*cm\\s*$"
          }
        }
      ]
    }
  },
  "beta2_microglobulin_urine": {
    "unit": "ug/L",
    "min": 0,
    "description": "Urinary beta-2 microglobulin concentration",
    "category": "renal",
    "sources": {
      "aumc": [
        {
          "ids": [
            10164
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*\u00b5g/l\\s*$"
          },
          "callback": "convert_unit(binary_op(`*`, 1), 'ug/L')"
        },
        {
          "ids": [
            10165
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*\u00b5g/l\\s*$"
          },
          "callback": "convert_unit(binary_op(`*`, 1), 'ug/L')"
        }
      ]
    }
  },
  "crea_clearance_measured": {
    "unit": "mL/min",
    "min": 0,
    "max": 500,
    "description": "Measured urinary creatinine clearance",
    "category": "renal",
    "sources": {
      "aumc": [
        {
          "ids": [
            7108
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*ml/min\\s*$"
          }
        }
      ],
      "miiv": [
        {
          "ids": [
            51080
          ],
          "table": "labevents",
          "sub_var": "itemid",
          "unit_var": "valueuom",
          "require_regex": {
            "valueuom": "(?i)^\\s*mL/min\\s*$"
          }
        }
      ]
    }
  },
  "intra_abdominal_pres": {
    "unit": "mmHg",
    "min": 0,
    "max": 100,
    "description": "Intra-abdominal pressure measured through bladder",
    "category": "cardiovascular",
    "sources": {
      "aumc": [
        {
          "ids": [
            8463
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mmHg\\s*$"
          }
        }
      ]
    }
  },
  "renal_resistive_index": {
    "unit": "ratio",
    "min": 0,
    "max": 1,
    "description": "Renal Doppler resistive index",
    "category": "renal",
    "sources": {
      "aumc": [
        {
          "ids": [
            20631
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*Geen\\s*$"
          },
          "callback": "convert_unit(binary_op(`*`, 1), 'ratio')"
        }
      ]
    }
  },
  "urine_cl": {
    "unit": "mmol/L",
    "min": 0,
    "max": 400,
    "description": "Urine chloride concentration",
    "category": "renal",
    "sources": {
      "aumc": [
        {
          "ids": [
            10273
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mmol/l\\s*$"
          }
        },
        {
          "ids": [
            10274
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mmol/l\\s*$"
          }
        },
        {
          "ids": [
            8917
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mol/l\\s*$"
          },
          "callback": "convert_unit(binary_op(`*`, 1000), 'mmol/L')"
        }
      ],
      "miiv": [
        {
          "ids": [
            51078
          ],
          "table": "labevents",
          "sub_var": "itemid",
          "unit_var": "valueuom",
          "require_regex": {
            "valueuom": "(?i)^\\s*mEq/L\\s*$"
          },
          "callback": "convert_unit(binary_op(`*`, 1), 'mmol/L')"
        }
      ]
    }
  },
  "urine_k": {
    "unit": "mmol/L",
    "min": 0,
    "max": 400,
    "description": "Urine potassium concentration",
    "category": "renal",
    "sources": {
      "aumc": [
        {
          "ids": [
            10271
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mmol/l\\s*$"
          }
        },
        {
          "ids": [
            10272
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mmol/l\\s*$"
          }
        },
        {
          "ids": [
            7101
          ],
          "table": "numericitems",
          "sub_var": "itemid",
          "unit_var": "unit",
          "require_regex": {
            "unit": "(?i)^\\s*mmol/l\\s*$"
          }
        }
      ],
      "miiv": [
        {
          "ids": [
            51097
          ],
          "table": "labevents",
          "sub_var": "itemid",
          "unit_var": "valueuom",
          "require_regex": {
            "valueuom": "(?i)^\\s*mEq/L\\s*$"
          },
          "callback": "convert_unit(binary_op(`*`, 1), 'mmol/L')"
        }
      ],
      "sic": [
        {
          "ids": [
            464
          ],
          "table": "laboratory",
          "sub_var": "LaboratoryID"
        }
      ]
    }
  }
}
''')
DICTIONARY = json.loads(
    (Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text()
)


@pytest.mark.parametrize("name", EXPECTED)
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert DICTIONARY[name].get("class_name", "num_cncpt") == "num_cncpt"
    assert name in EXTRACT_MODULES["renal_detail"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in RENAL_DETAIL_METADATA
    assert "renal_detail" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


def test_pressure_and_proxy_are_distinct():
    pressure = EXPECTED["intra_abdominal_pres"]["sources"]["aumc"][0]["ids"]
    circumference = EXPECTED["abdominal_circumference"]["sources"]["aumc"][0]["ids"]
    assert pressure != circumference


@pytest.mark.parametrize("name,database,index,valid,factor", [
    ("beta2_microglobulin_urine", "aumc", 0, "µg/l", 1.),
    ("beta2_microglobulin_urine", "aumc", 1, "µg/l", 1.),
    ("renal_resistive_index", "aumc", 0, "Geen", 1.),
    ("urine_cl", "aumc", 2, "mol/l", 1000.),
    ("urine_cl", "miiv", 0, "mEq/L", 1.),
    ("urine_k", "miiv", 0, "mEq/L", 1.),
])
def test_conversions_and_wrong_units(name, database, index, valid, factor):
    source = load_dictionary()[name].sources[database][index]
    unit_col = source.unit_var
    frame = pd.DataFrame({name: [0., .1, .1], unit_col: [valid, valid, "wrong"]})
    filtered = _drop_declared_row_exclusions(frame, source, name)
    assert len(filtered) == 2
    result = _apply_callback(filtered, source, name, unit_column=unit_col)
    assert result[name].tolist() == [0., .1 * factor]


@pytest.mark.parametrize("database,name", [
    (db, name) for name, definition in EXPECTED.items()
    for db in definition["sources"] if db in {"aumc", "sic"}
])
def test_generated_synthetic_source(tmp_path, database, name):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / database
    generate_source(root, database, stays=12)
    try:
        result = load_concepts([name], database=database, data_path=root, merge=False,
                               verbose=False, concept_workers=1, parallel_workers=1)
        frame = result if isinstance(result, pd.DataFrame) else result[name]
        assert not frame.empty and frame[name].notna().any()
    finally:
        clear_global_loader()


@pytest.mark.parametrize("name", ["crea_clearance_measured", "urine_cl", "urine_k"])
def test_miiv_synthetic_source_and_unit_filter(tmp_path, name):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    entry = EXPECTED[name]["sources"]["miiv"][0]
    unit = "mL/min" if name == "crea_clearance_measured" else "mEq/L"
    now = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11, 11], "hadm_id": [21, 21],
                          "itemid": entry["ids"] * 2, "charttime": [now, now + pd.Timedelta(hours=1)],
                          "storetime": [now, now + pd.Timedelta(hours=1)],
                          "valuenum": [2., 2.], "value": ["2", "2"],
                          "valueuom": [unit, "wrong"]})
    source = mimic_synthetic_source(tmp_path, "miiv", labevents=frame)
    actual = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert actual.tolist() == [2.]
