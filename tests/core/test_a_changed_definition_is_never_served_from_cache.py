"""A changed concept definition is never served from a cache, and overlays keep every field.

11-03 external review, reproduced on synthetic data: with ``peep_set`` values
8 and 10 in one hour, a dictionary maximum of 40 gives 9; lowering the
maximum to 9 at the same path must give 8.  Both result caches returned 9:
the public API cache keyed the dictionary by its path, and the resolver's
disk-cache signature covered only callbacks, aggregates, sub-concepts and
sources.  A dictionary overlay that only edited a description also dropped
``pre_admission_lookback_hours`` (168 h for the KDIGO creatinine input).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu import api
from easyicu.api import concepts as concept_api
from easyicu.concept import ConceptResolver
from easyicu.concept.definition_identity import (
    definition_fingerprint,
    dictionary_files_fingerprint,
)
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary

BASE = {
    "peep_set": {
        "sources": {"miiv": [{"table": "chartevents", "ids": [220339]}]},
        "min": 0, "max": 40, "unit": "cmH2O", "interval": "1h",
        "pre_admission_lookback_hours": 168,
    },
}


def _dictionary(**changes) -> ConceptDictionary:
    definition = {**BASE["peep_set"], **changes}
    return ConceptDictionary.from_payload({"peep_set": definition})


@pytest.mark.parametrize("field,value", [
    ("max", 9), ("min", 1), ("unit", "mbar"), ("interval", "2h"),
    ("pre_admission_lookback_hours", 24),
])
def test_every_value_bearing_field_changes_the_resolver_signature(tmp_path, field, value):
    before = ConceptResolver(_dictionary(), cache_dir=tmp_path).dictionary_signature
    assert ConceptResolver(_dictionary(), cache_dir=tmp_path).dictionary_signature == before
    assert ConceptResolver(_dictionary(**{field: value}), cache_dir=tmp_path).dictionary_signature != before


def test_the_api_cache_recomputes_after_the_dictionary_changes_at_the_same_path(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    dictionary = tmp_path / "dict.json"
    dictionary.write_text(json.dumps(BASE), encoding="utf-8")
    computed = []

    def load(**kwargs):
        computed.append(json.loads(dictionary.read_text())["peep_set"]["max"])
        return pd.DataFrame({"stay_id": [1], "peep_set": [8.0 if computed[-1] == 9 else 9.0]})

    monkeypatch.setattr(api, "load_concepts", load)

    def cached():
        return api.load_concept_cached(
            "peep_set", "miiv", data_path, cache_dir=tmp_path / "cache",
            verbose=False, dict_path=str(dictionary),
        )["peep_set"].tolist()

    assert cached() == [9.0] and cached() == [9.0]
    assert computed == [40]
    dictionary.write_text(json.dumps({"peep_set": {**BASE["peep_set"], "max": 9}}), encoding="utf-8")
    assert cached() == [8.0]
    assert computed == [40, 9]


def test_the_in_memory_loader_is_not_reused_for_an_edited_dictionary(monkeypatch, tmp_path):
    dictionary = tmp_path / "dict.json"
    dictionary.write_text(json.dumps(BASE), encoding="utf-8")
    built = []

    class Loader:
        def __init__(self, **kwargs):
            built.append(json.loads(dictionary.read_text())["peep_set"]["max"])

    monkeypatch.setattr(concept_api, "BaseICULoader", Loader)
    monkeypatch.setattr(concept_api, "_loader_cache", type(concept_api._loader_cache)())
    concept_api._get_global_loader(database="miiv", dict_path=str(dictionary))
    concept_api._get_global_loader(database="miiv", dict_path=str(dictionary))
    dictionary.write_text(json.dumps({"peep_set": {**BASE["peep_set"], "max": 9}}), encoding="utf-8")
    concept_api._get_global_loader(database="miiv", dict_path=str(dictionary))
    assert built == [40, 9]


def test_file_identity_follows_content_not_path(tmp_path):
    dictionary = tmp_path / "dict.json"
    dictionary.write_text("{}", encoding="utf-8")
    before = dictionary_files_fingerprint(dictionary)
    assert dictionary_files_fingerprint(str(dictionary)) == before
    dictionary.write_text('{"x": {}}', encoding="utf-8")
    assert dictionary_files_fingerprint(dictionary) != before
    assert dictionary_files_fingerprint(None).startswith("packaged:")


def test_an_overlay_that_edits_one_field_keeps_every_other_field():
    base = _dictionary(levels=[0, 1], depends_on=["fio2"])
    before = base["peep_set"]
    base.update(ConceptDictionary.from_payload({"peep_set": {"description": "PEEP set"}}))
    after = base["peep_set"]
    assert after.description == "PEEP set"
    assert after.pre_admission_lookback_hours == 168
    unchanged = {
        name: getattr(after, name) for name in ConceptDefinition.__dataclass_fields__
        if name != "description"
    }
    assert unchanged == {name: getattr(before, name) for name in unchanged}
    # The merged result signs differently only through the edited field.
    assert definition_fingerprint(base) != definition_fingerprint(_dictionary(levels=[0, 1], depends_on=["fio2"]))


def test_an_overlay_value_still_replaces_the_base_value():
    base = _dictionary()
    base.update(ConceptDictionary.from_payload({"peep_set": {"pre_admission_lookback_hours": 24, "max": 30}}))
    assert base["peep_set"].pre_admission_lookback_hours == 24
    assert base["peep_set"].maximum == 30
