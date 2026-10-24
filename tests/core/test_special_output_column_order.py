"""Portable exports must not inherit Python's salted set iteration order."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("seed", ["0", "1", "7"])
@pytest.mark.parametrize("merge", [False, True])
def test_special_output_order_is_stable_across_processes(tmp_path, seed, merge):
    script = textwrap.dedent("""
        import json
        import sys
        from types import SimpleNamespace
        import pandas as pd
        from easyicu import api
        from easyicu.api import concepts
        from easyicu.scores import comorbidity, microbiology

        merge = sys.argv[1] == "True"
        standard = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "hr": [70.0]})
        loader = SimpleNamespace(
            database="miiv", data_path=sys.argv[2],
            load_concepts=lambda **kwargs: standard if merge else {"hr": standard},
        )
        concepts._get_global_loader = lambda **kwargs: loader
        comorbidity.load_comorbidity = lambda *args, **kwargs: pd.DataFrame({
            "stay_id": [1], "charlson_index": [2.0], "elixhauser_vw": [3.0],
        })
        microbiology.load_microbiology = lambda *args, **kwargs: pd.DataFrame({
            "stay_id": [1], "culture_positive": [True], "bld_culture_positive": [False],
        })
        result = api.load_concepts(
            ["hr", "charlson", "elixhauser", "culture_positive", "bld_culture_positive"],
            patient_ids={"stay_id": [1]}, database="miiv", data_path=sys.argv[2],
            merge=merge, verbose=False, concept_workers=1, parallel_workers=1,
        )
        print(json.dumps(list(result)))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(merge), str(tmp_path)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        check=True, capture_output=True, text=True, timeout=30,
    )
    expected = ["hr", "elixhauser", "charlson", "bld_culture_positive", "culture_positive"]
    if merge:
        expected = ["stay_id", "charttime", *expected]
    assert json.loads(result.stdout) == expected
