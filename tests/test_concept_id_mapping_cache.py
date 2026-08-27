from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDictionary


class _MappingDataSource:
    def __init__(self) -> None:
        self.config = SimpleNamespace(name="miiv")
        self.calls: list[list[int]] = []
        self.mapping = pd.DataFrame(
            {
                "stay_id": [1, 2],
                "subject_id": [101, 102],
            }
        )

    def load_table(self, _table, *, columns, filters, verbose):
        del columns, verbose
        requested = list(filters[0].value)
        self.calls.append(requested)
        return SimpleNamespace(
            data=self.mapping[self.mapping["stay_id"].isin(requested)]
        )


def test_patient_id_mapping_cache_loads_missing_disjoint_cohorts() -> None:
    resolver = ConceptResolver(ConceptDictionary(concepts={}))
    source = _MappingDataSource()

    first = resolver._expand_patient_ids(
        {"stay_id": [1]}, "subject_id", source
    )
    second = resolver._expand_patient_ids(
        {"stay_id": [2]}, "subject_id", source
    )

    assert first == {"stay_id": [1], "subject_id": [101]}
    assert second == {"stay_id": [2], "subject_id": [102]}
    assert source.calls == [[1], [2]]
    assert set(resolver._id_mapping_cache["stay_id"]) == {1, 2}


def test_patient_id_mapping_materializes_an_empty_target_filter() -> None:
    resolver = ConceptResolver(ConceptDictionary(concepts={}))
    source = _MappingDataSource()

    expanded = resolver._expand_patient_ids(
        {"stay_id": [999]}, "subject_id", source
    )

    assert expanded == {"stay_id": [999], "subject_id": []}
    assert source.calls == [[999]]
