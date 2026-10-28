"""Small metadata fixtures and a synthetic native-export API."""

from contextlib import contextmanager

import pandas as pd

from easyicu.concept.schema import ConceptDictionary


def synthetic_dictionary() -> ConceptDictionary:
    return ConceptDictionary.from_payload({
        "lab_signal": {
            "description": "Synthetic laboratory measurement",
            "unit": "mmol/L",
            "sources": {"miiv": [{
                "table": "synthetic_labs", "sub_var": "itemid", "ids": [71, 72],
                "val_var": "value", "unit_var": "unit",
                "callback": "convert_unit(binary_op(`/`, 10), 'mmol/L', 'mg/dL')",
            }]},
        },
        "coded_condition": {
            "description": "Synthetic condition defined by the declared code list",
            "unit": "boolean", "definition_source": "synthetic_protocol",
            "definition_version": "v1",
            "sources": {"miiv": [{
                "table": "synthetic_diagnoses", "sub_var": "icd_code",
                "ids": ["Z01", "Z02"], "val_var": "present",
                "code_descriptions": {"Z01": "Synthetic criterion one"},
            }]},
        },
    })



class ExportJob:
    def emit(self, payload):
        pass


def patch_demographic_export(monkeypatch):
    import easyicu.api as api

    @contextmanager
    def keep_cache(**kwargs):
        yield None

    def load(concepts, **kwargs):
        from types import SimpleNamespace
        from easyicu.concept.provenance import record_resolver_inputs
        from easyicu.resources import load_dictionary
        from easyicu.config import load_src_cfg
        record_resolver_inputs(kwargs, SimpleNamespace(
            concept_resolver=SimpleNamespace(dictionary=load_dictionary()),
            datasource=SimpleNamespace(config=load_src_cfg(kwargs["database"])),
        ))
        selected = kwargs.get("patient_ids") or []
        ids = selected.get("stay_id", []) if isinstance(selected, dict) else selected
        return pd.DataFrame({"stay_id": ids, "age": [65.0] * len(ids)})

    monkeypatch.setattr(api, "keep_cache", keep_cache)
    monkeypatch.setattr(api, "load_concepts", load)
