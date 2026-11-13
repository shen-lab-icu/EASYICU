"""Accepted skipped mappings, with invented source rows and strict unknowns."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _parse_mapping
from easyicu.resources import load_dictionary
from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS

EXPECTED = json.loads(r'''
{
  "cam_icu": {
    "unit": "boolean",
    "description": "Explicit CAM-ICU result; unassessable stays missing",
    "category": "neurological",
    "class_name": "lgl_cncpt",
    "aggregate": "max",
    "sources": {
      "aumc": [
        {
          "ids": [
            20351
          ],
          "table": "listitems",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`POSITIEF` = 1, `NEGATIEF` = 0), strict = TRUE)"
        }
      ]
    }
  },
  "ramsay": {
    "unit": "score",
    "min": 1,
    "max": 6,
    "description": "Ramsay sedation score",
    "category": "neurological",
    "sources": {
      "aumc": [
        {
          "ids": [
            9232
          ],
          "table": "listitems",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`6 in slaap, helemaal geen reactie` = 6, `4 in slaap, maar snelle respons op prikkel` = 4, `3 wakker, alleen respons op opdracht` = 3, `2 wakker, cooperatief, georiënteerd en rustig` = 2, `1 wakker, angstig, geagiteerd of rusteloos` = 1, `5 inslaap, trage, slome reaktie op prikkel` = 5), strict = TRUE)"
        }
      ]
    }
  },
  "sas": {
    "unit": "score",
    "min": 1,
    "max": 7,
    "description": "Riker Sedation-Agitation Scale",
    "category": "neurological",
    "sources": {
      "miiv": [
        {
          "ids": [
            223753
          ],
          "table": "chartevents",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`Unarousable` = 1, `Very Sedated` = 2, `Sedated` = 3, `Calm/Cooperative` = 4, `Agitated` = 5, `Very Agitated` = 6, `Danger Agitation` = 7), strict = TRUE)"
        }
      ]
    }
  },
  "rrt_modality": {
    "unit": "category",
    "description": "Charted CRRT modality",
    "category": "renal",
    "class_name": "fct_cncpt",
    "target": "ts_tbl",
    "levels": [
      "cvvh",
      "cvvhd",
      "cvvhdf",
      "scuf"
    ],
    "sources": {
      "miiv": [
        {
          "ids": [
            227290
          ],
          "table": "chartevents",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`CVVH` = 'cvvh', `CVVHD` = 'cvvhd', `CVVHDF` = 'cvvhdf', `SCUF` = 'scuf'), strict = TRUE)"
        }
      ]
    }
  },
  "crrt_dilution_mode_set": {
    "unit": "category",
    "description": "Prescribed CRRT dilution mode",
    "category": "renal",
    "class_name": "fct_cncpt",
    "target": "ts_tbl",
    "levels": [
      "pre",
      "pre_post"
    ],
    "sources": {
      "aumc": [
        {
          "ids": [
            20542
          ],
          "table": "listitems",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`Predilutie` = 'pre', `Pre+Postdilutie` = 'pre_post'), strict = TRUE)"
        }
      ]
    }
  },
  "ecmo_type": {
    "unit": "category",
    "description": "ECMO circuit configuration",
    "category": "ventilator",
    "class_name": "fct_cncpt",
    "target": "ts_tbl",
    "levels": [
      "va",
      "vv",
      "vav",
      "other",
      "unknown",
      "conflicting"
    ],
    "sources": {
      "aumc": [
        {
          "ids": [
            20662
          ],
          "table": "listitems",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`VA ECMO` = 'va'), strict = TRUE)"
        }
      ],
      "miiv": [
        {
          "ids": [
            229268
          ],
          "table": "chartevents",
          "sub_var": "itemid",
          "val_var": "value",
          "callback": "apply_map(c(`VA` = 'va', `VV` = 'vv', `VAV` = 'vav', `---` = 'unknown'), strict = TRUE)"
        }
      ]
    }
  },
  "remifentanil_rate": {
    "unit": "mcg/kg/min",
    "min": 0,
    "max": 10,
    "description": "Continuous intravenous remifentanil infusion rate",
    "category": "neurological",
    "sources": {
      "aumc": [
        {
          "ids": [
            12402
          ],
          "table": "drugitems",
          "sub_var": "itemid",
          "rel_weight": "doserateperkg",
          "rate_uom": "doserateunit",
          "stop_var": "stop",
          "callback": "aumc_rate_kg",
          "extra_vars": [
            "action",
            "doserateperkg",
            "doserateunit",
            "doseunit",
            "iscontinuous",
            "ordercategory"
          ],
          "require_regex": {
            "iscontinuous": "^(1|1\\.0|True|true)$",
            "doseunit": "(?i)^(?:mg|µg|ug|mcg|g|ng)$",
            "doserateunit": "(?i)^(?:min|uur|hour|hr|h|sec|s|dag)$"
          },
          "exclude_regex": {
            "action": "(?i)^\\s*flush\\s*$",
            "ordercategory": "(?i)^\\s*niet iv"
          }
        }
      ],
      "hirid": [
        {
          "ids": [
            1000705
          ],
          "table": "pharma",
          "sub_var": "pharmaid",
          "grp_var": "infusionid",
          "callback": "hirid_rate_kg",
          "extra_vars": [
            "route"
          ],
          "require_regex": {
            "route": "(?i)^\\s*[ci]v-inf\\s*$"
          }
        }
      ]
    }
  },
  "enoximone_rate": {
    "unit": "mcg/kg/min",
    "min": 0,
    "max": 50,
    "description": "Continuous intravenous enoximone infusion rate",
    "category": "cardiovascular",
    "sources": {
      "aumc": [
        {
          "ids": [
            7196
          ],
          "table": "drugitems",
          "sub_var": "itemid",
          "rel_weight": "doserateperkg",
          "rate_uom": "doserateunit",
          "stop_var": "stop",
          "callback": "aumc_rate_kg",
          "extra_vars": [
            "action",
            "doserateperkg",
            "doserateunit",
            "doseunit",
            "iscontinuous",
            "ordercategory"
          ],
          "require_regex": {
            "iscontinuous": "^(1|1\\.0|True|true)$",
            "doseunit": "(?i)^(?:mg|µg|ug|mcg|g|ng)$",
            "doserateunit": "(?i)^(?:min|uur|hour|hr|h|sec|s|dag)$"
          },
          "exclude_regex": {
            "action": "(?i)^\\s*flush\\s*$",
            "ordercategory": "(?i)^\\s*niet iv"
          }
        }
      ]
    }
  },
  "milrinone_rate": {
    "unit": [
      "mcg/kg/min",
      "mcgkgmin"
    ],
    "min": 0,
    "max": 3,
    "description": "milrinone rate",
    "category": "medications",
    "sources": {
      "miiv": [
        {
          "ids": 221986,
          "table": "inputevents",
          "sub_var": "itemid",
          "stop_var": "endtime",
          "callback": "mimic_rate_mv",
          "extra_vars": [
            "ordercategorydescription",
            "rateuom",
            "statusdescription"
          ],
          "status_var": "statusdescription",
          "excluded_statuses": [
            "Rewritten",
            "Cancelled",
            "Canceled",
            "Bolus"
          ],
          "require_regex": {
            "rateuom": "(?i)^\\s*mcg/kg/min\\s*$",
            "ordercategorydescription": "^Continuous Med$"
          }
        }
      ],
      "hirid": [
        {
          "ids": 1000441,
          "table": "pharma",
          "sub_var": "pharmaid",
          "grp_var": "infusionid",
          "callback": "hirid_rate_kg",
          "extra_vars": [
            "route"
          ],
          "require_regex": {
            "route": "(?i)^\\s*[ci]v-inf\\s*$"
          }
        }
      ],
      "eicu": [
        {
          "ids": [
            "Milrinone (mcg/kg/hr)"
          ],
          "table": "infusiondrug",
          "sub_var": "drugname",
          "weight_var": "patientweight",
          "extra_vars": [
            "drugname"
          ],
          "callback": "eicu_rate_kg(ml_to_mcg = NA)"
        },
        {
          "ids": [
            "Milrinone (mcg/kg/min)"
          ],
          "table": "infusiondrug",
          "sub_var": "drugname",
          "weight_var": "patientweight",
          "extra_vars": [
            "drugname"
          ],
          "callback": "eicu_rate_kg(ml_to_mcg = NA)"
        },
        {
          "ids": [
            "primacore (mcg/kg/min)"
          ],
          "table": "infusiondrug",
          "sub_var": "drugname",
          "weight_var": "patientweight",
          "extra_vars": [
            "drugname"
          ],
          "callback": "eicu_rate_kg(ml_to_mcg = NA)"
        }
      ]
    }
  }
}
''')
CODED = ("cam_icu", "ramsay", "sas", "rrt_modality", "crrt_dilution_mode_set", "ecmo_type")


@pytest.mark.parametrize("name", EXPECTED)
def test_dictionary_and_opt_in_contract(name):
    actual = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
    assert actual[name] == EXPECTED[name]
    assert name in HIDDEN_DICTIONARY_CONCEPTS
    assert any(name in values for module, values in EXTRACT_MODULES.items() if module not in EXTRACT_MODULE_ORDER)
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name,db", [(n, db) for n in CODED for db in EXPECTED[n]["sources"]])
def test_all_declared_codes_and_unknowns_through_loader(tmp_path, name, db):
    from easyicu import load_concepts
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    source = load_dictionary()[name].sources[db][0]
    mapping = _parse_mapping(source.callback.split("c(", 1)[1].rsplit(")", 2)[0])
    labels = list(mapping) + ["unmapped", "1", "4", "", None]
    if db == "miiv":
        times = pd.date_range("2150-01-01 00:15", periods=len(labels), freq="h")
        frame = pd.DataFrame({"subject_id": 11, "hadm_id": 21, "stay_id": 1,
            "itemid": source.ids[0], "charttime": times, "storetime": times,
            "value": labels, "valuenum": None, "valueuom": ""})
        data = mimic_synthetic_source(tmp_path, "miiv", chartevents=frame)
        result = mimic_synthetic_load(data, [name])[name].data
    else:
        pd.DataFrame({"admissionid": [1], "patientid": [1], "admittedat": [0],
            "dischargedat": [72 * 3600000]}).to_parquet(tmp_path / "admissions.parquet", index=False)
        frame = pd.DataFrame({"admissionid": 1, "itemid": source.ids[0],
            "measuredat": pd.Series(range(len(labels))) * 3600000,
            "value": labels})
        frame.to_parquet(tmp_path / "listitems.parquet", index=False)
        actual = load_concepts([name], database="aumc", data_path=tmp_path,
            merge=False, verbose=False, concept_workers=1, parallel_workers=1)
        result = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert result[name].dropna().tolist() == list(mapping.values())


@pytest.mark.parametrize("name,db", [("remifentanil_rate", "aumc"), ("enoximone_rate", "aumc"),
                                    ("remifentanil_rate", "hirid"), ("milrinone_rate", "hirid")])
def test_rate_synthetic_extraction(tmp_path, name, db):
    from easyicu import load_concepts
    from easyicu.synthetic_sources import generate_source

    generate_source(tmp_path / "source", db, stays=12)
    actual = load_concepts([name], database=db, data_path=tmp_path / "source", merge=False,
        verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].notna().any()


@pytest.mark.parametrize("name", ["remifentanil_rate", "enoximone_rate"])
def test_aumc_rate_exact_units_and_continuous_gate(name):
    source = load_dictionary()[name].sources["aumc"][0]
    frame = pd.DataFrame({"admissionid": [1] * 5, "start": [0.] * 5, "stop": [60.] * 5,
        name: [60., .06, 60., 60., 60.], "doseunit": ["µg", "mg", "µg", "µg", "wrong"],
        "doserateunit": ["uur"] * 5, "doserateperkg": [True] * 5,
        "action": ["Start", "Start", "Flush", "Start", "Start"],
        "iscontinuous": [True, True, True, False, True], "ordercategory": "Infuus"})
    out = _apply_callback(frame, source, name, unit_column="doseunit")
    assert out[name].tolist() == pytest.approx([1., 1.])


def test_milrinone_eicu_all_native_labels_through_loader(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    entries = EXPECTED["milrinone_rate"]["sources"]["eicu"]
    frame = pd.DataFrame({"patientunitstayid": 1, "infusionoffset": [0, 60, 120],
        "drugname": [s["ids"][0] for s in entries], "drugrate": [60., 1., 1.], "patientweight": 80.})
    frame.to_parquet(tmp_path / "infusiondrug.parquet", index=False)
    pd.DataFrame({"patientunitstayid": [1], "admissionweight": [80.],
        "hospitaldischargeoffset": [4320], "unitdischargeoffset": [4320]}).to_parquet(tmp_path / "patient.parquet", index=False)
    data = ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path, enable_cache=False)
    out = ConceptResolver(load_dictionary()).load_concepts(["milrinone_rate"], data,
        merge=False, r_compatible=False, patient_ids={"patientunitstayid": [1]},
        interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1)["milrinone_rate"].data
    assert out.milrinone_rate.tolist() == pytest.approx([1., 1., 1.])
