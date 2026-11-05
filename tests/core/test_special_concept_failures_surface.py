"""A failing special-concept producer fails the request instead of vanishing.

``load_concepts`` routes AKI, circulatory failure, comorbidity, outcome and
microbiology outputs to their own producers.  A producer exception used to be
logged and the requested output silently left out; native-v2 then sealed the
missing column as structurally unavailable.  Each failure now raises with a
reason code and the producer's cause.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

CASES = [
    ("aki_stage_reference", "easyicu.scores.aki_profiles", "load_renal_aki_bundle", "renal_aki_bundle_load_failed"),
    ("circ_failure", "easyicu.scores.circ_failure", "load_circ_failure", "circ_failure_load_failed"),
    ("charlson", "easyicu.scores.comorbidity", "load_comorbidity", "comorbidity_load_failed"),
    ("mort_28d", "easyicu.scores.outcomes", "load_outcomes", "outcomes_load_failed"),
    ("culture_positive", "easyicu.scores.microbiology", "load_microbiology", "microbiology_load_failed"),
]


def _stub_loader(monkeypatch, tmp_path):
    from easyicu.api import concepts as api

    base = pd.DataFrame({"stay_id": [11, 12], "charttime": [0.0, 0.0], "urine": [70.0, 80.0]})
    loader = SimpleNamespace(
        database="miiv", data_path=tmp_path, load_concepts=lambda *a, **k: base.copy()
    )
    monkeypatch.setattr(api, "_get_global_loader", lambda **kwargs: loader)
    return api


@pytest.mark.parametrize("concept, module, attr, reason", CASES)
def test_special_producer_failure_raises_with_reason(monkeypatch, tmp_path, concept, module, attr, reason):
    api = _stub_loader(monkeypatch, tmp_path)

    def broken(*_args, **_kwargs):
        raise OSError("disk read failed")

    monkeypatch.setattr(importlib.import_module(module), attr, broken)

    with pytest.raises(RuntimeError, match=reason) as info:
        api.load_concepts(
            ["urine", concept], database="miiv", data_path=tmp_path, patient_ids=[11, 12],
            merge=True, concept_workers=1, parallel_workers=1, chunk_size=100,
        )
    assert isinstance(info.value.__cause__, OSError)


def test_special_merge_failure_raises(monkeypatch, tmp_path):
    api = _stub_loader(monkeypatch, tmp_path)
    from easyicu.scores import circ_failure

    monkeypatch.setattr(
        circ_failure, "load_circ_failure",
        lambda **kwargs: pd.DataFrame({"stay_id": [11], "charttime": [0.0], "circ_failure": [True]}),
    )
    original_merge = pd.DataFrame.merge

    def failing(frame, right, *args, **kwargs):
        if "circ_failure" in right.columns:
            raise ValueError("You are trying to merge on float64 and datetime64[ns] columns")
        return original_merge(frame, right, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", failing)
    with pytest.raises(RuntimeError, match="special_concept_merge_failed"):
        api.load_concepts(
            ["urine", "circ_failure"], database="miiv", data_path=tmp_path, patient_ids=[11, 12],
            merge=True, concept_workers=1, parallel_workers=1, chunk_size=100,
        )
