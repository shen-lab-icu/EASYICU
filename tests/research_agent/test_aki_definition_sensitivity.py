from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "discovery_aki_definition_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "discovery_aki_definition_sensitivity", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aki_definition_summary_keeps_missing_stage_separate_from_negative() -> None:
    script = _load_script()
    all_stays = pd.Index([1, 2, 3, 4])
    renal = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [1.0, 1.0, 1.0],
            "urine24": [1200.0, 400.0, None],
            "aki_stage_creat": [1.0, 0.0, None],
            "aki_stage_uo": [0.0, 2.0, None],
            "aki_stage_rrt": [0.0, 0.0, None],
            "aki_stage": [1.0, 2.0, None],
        }
    )
    chemistry = pd.DataFrame(
        {
            "stay_id": [1, 3],
            "charttime": [1.0, 1.0],
            "crea": [1.3, 0.8],
        }
    )

    summary, overlap = script._summarize_window(
        window_name="fixture",
        renal=renal,
        chemistry=chemistry,
        all_stays=all_stays,
    )

    assert summary["n_stays"] == 4
    assert summary["crea_measured_n"] == 2
    assert summary["urine24_measured_n"] == 2
    assert summary["creatinine_stage_evaluable_n"] == 2
    assert summary["urine_output_stage_evaluable_n"] == 2
    assert summary["creatinine_positive_n"] == 1
    assert summary["urine_output_positive_n"] == 1
    assert summary["combined_kdigo_positive_n"] == 2
    assert summary["component_discordant_positive_n"] == 2
    by_metric = dict(zip(overlap["metric"], overlap["n"], strict=True))
    assert by_metric["creatinine/urine-output discordant positive"] == 2
