"""Portable exports must not inherit Python's salted set iteration order."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from easyicu.api.extraction import EXTRACT_MODULES


@pytest.mark.parametrize(
    ("module", "special_order"),
    [
        ("sepsis_shared", ["bld_culture_positive", "culture_positive"]),
        ("circulatory", ["circ_event", "circ_failure"]),
        ("other_scores", ["elixhauser", "charlson"]),
        (
            "outcome",
            [
                "followup_days_28d", "followup_days_365d", "followup_days_90d",
                "icu_free_days_28", "icu_readmission",
                "mort_28d", "mort_365d", "mort_90d", "vent_free_days_28",
            ],
        ),
    ],
)
@pytest.mark.parametrize("merge", [False, True])
def test_special_output_order_is_stable_across_processes(
    tmp_path, module, special_order, merge
):
    script = textwrap.dedent("""
        import json
        import sys
        from types import SimpleNamespace
        import pandas as pd
        from easyicu import api
        from easyicu.api import concepts
        from easyicu.api.extraction import EXTRACT_MODULES, _module_parquet_columns
        from easyicu.scores import circ_failure, comorbidity, microbiology, outcomes

        merge = sys.argv[1] == "True"
        module = sys.argv[3]
        requested = EXTRACT_MODULES[module]

        def load_standard(concepts, **kwargs):
            keys = {"stay_id": [1], "charttime": [0.0]}
            if merge:
                return pd.DataFrame({**keys, **{name: [1.0] for name in concepts}})
            return {name: pd.DataFrame({**keys, name: [1.0]}) for name in concepts}

        loader = SimpleNamespace(
            database="miiv", data_path=sys.argv[2],
            load_concepts=load_standard,
        )
        concepts._get_global_loader = lambda **kwargs: loader
        circ_failure.load_circ_failure = lambda *args, **kwargs: pd.DataFrame({
            "stay_id": [1], "circ_failure": [1.0], "circ_event": [0.0],
        })
        comorbidity.load_comorbidity = lambda *args, **kwargs: pd.DataFrame({
            "stay_id": [1], "charlson_index": [2.0], "elixhauser_vw": [3.0],
        })
        outcomes.load_outcomes = lambda *args, **kwargs: pd.DataFrame({
            "stay_id": [1], "mort_28d": [False], "mort_90d": [False],
            "mort_365d": [False], "followup_days_28d": [28.0],
            "followup_days_90d": [90.0], "followup_days_365d": [365.0],
            "icu_free_days_28": [27.0], "icu_readmission": [False],
            "vent_free_days_28": [28.0],
        })
        microbiology.load_microbiology = lambda *args, **kwargs: pd.DataFrame({
            "stay_id": [1], "culture_positive": [True], "bld_culture_positive": [False],
        })
        result = api.load_concepts(
            requested,
            patient_ids={"stay_id": [1]}, database="miiv", data_path=sys.argv[2],
            merge=merge, verbose=False, concept_workers=1, parallel_workers=1,
        )
        columns = list(result) if merge else ["stay_id", "charttime", *result]
        print(json.dumps({
            "loaded": list(result),
            "exported": _module_parquet_columns(columns, requested),
        }))
    """)
    orders = []
    for seed in ("0", "42"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(merge), str(tmp_path), module],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True, capture_output=True, text=True, timeout=30,
        )
        orders.append(json.loads(result.stdout))

    assert orders[0] == orders[1]
    requested = EXTRACT_MODULES[module]
    expected = [name for name in requested if name not in special_order] + special_order
    if merge:
        expected = ["stay_id", "charttime", *expected]
    assert orders[0]["loaded"] == expected
    assert orders[0]["exported"] == ["stay_id", "charttime", *requested]
