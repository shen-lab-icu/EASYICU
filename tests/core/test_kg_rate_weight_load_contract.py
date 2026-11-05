"""Per-kg rate weight loading must not turn a load failure into NA rates.

The CareVue view's MetaVision weight source (``inputevents_mv`` without
``icustay_id``) returns an empty table that lacks the concept column.  Its
``convert_unit`` callback raised ``KeyError: 'weight'``; ``mimic_kg_rate``
caught it and silently set every per-kg rate of the batch to NA.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback

MIMIC = SimpleNamespace(config=SimpleNamespace(name="mimic"))


def test_convert_unit_passes_an_empty_source_without_the_value_column():
    empty = pd.DataFrame({"subject_id": pd.Series([], dtype="int64")})
    source = ConceptSource(
        callback="convert_unit(binary_op(`*`, 1), 'kg')",
        table="inputevents_mv",
        value_var="patientweight",
        unit_var="rateuom",
    )

    result = _apply_callback(empty, source, concept_name="weight")

    assert result.empty


def test_convert_unit_rejects_rows_without_the_value_column():
    rows = pd.DataFrame({"subject_id": [1], "rateuom": ["kg"]})
    source = ConceptSource(callback="convert_unit(binary_op(`*`, 1), 'kg')")

    with pytest.raises(KeyError, match="weight"):
        _apply_callback(rows, source, concept_name="weight")


def test_mimic_kg_rate_raises_when_the_weight_concept_cannot_load():
    class _FailingResolver:
        def _load_single_concept(self, *_args, **_kwargs):
            raise KeyError("weight")

    frame = pd.DataFrame(
        {"subject_id": [7], "icustay_id": [11], "value": [7.0], "unit": ["mcgmin"]}
    )

    with pytest.raises(RuntimeError, match="mimic_kg_rate_weight_load_failed"):
        _apply_callback(
            frame,
            ConceptSource(callback="mimic_kg_rate", value_var="value", unit_var="unit"),
            concept_name="norepi_rate",
            resolver=_FailingResolver(),
            data_source=MIMIC,
        )


def test_mimic_kg_rate_without_any_weight_rows_keeps_the_na_contract():
    class _EmptyResolver:
        def _load_single_concept(self, *_args, **_kwargs):
            return SimpleNamespace(data=pd.DataFrame())

    frame = pd.DataFrame(
        {"subject_id": [7], "icustay_id": [11], "value": [7.0], "unit": ["mcgmin"]}
    )

    result = _apply_callback(
        frame,
        ConceptSource(callback="mimic_kg_rate", value_var="value", unit_var="unit"),
        concept_name="norepi_rate",
        resolver=_EmptyResolver(),
        data_source=MIMIC,
    )

    assert result["value"].isna().all()
