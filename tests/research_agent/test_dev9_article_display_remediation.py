from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9 import (
    render_dev9_article_display_remediation as renderer,
)


def _write_h2_source(run_dir: Path, *, effect_estimate: float | None = None) -> Path:
    output_dir = run_dir / renderer.H2_FEASIBILITY_RELATIVE.parent
    output_dir.mkdir(parents=True)
    path = output_dir / renderer.H2_FEASIBILITY_RELATIVE.name
    pd.DataFrame(
        [
            {
                "source": "typed_vasopressor_source",
                "window_start_hours": 0,
                "window_end_hours": 24,
                "verified_non_use_available": False,
                "binary_control_arm_authorized": False,
                "causal_contrast_authorized": False,
                "decision": "fail_closed",
                "reason_code": "H2_VERIFIED_NON_USE_UNAVAILABLE",
                "effect_estimate": effect_estimate,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_copy_source_preserves_nested_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    frame = pd.DataFrame(
        [{"source_row_index": 7, "source_file": "upstream.csv", "value": 3.0}]
    )
    frame.to_csv(source, index=False)

    output = tmp_path / "copied.csv"
    renderer._copy_source(frame, source, output)
    copied = pd.read_csv(output)

    assert copied.loc[0, "source_row_index"] == 0
    assert copied.loc[0, "upstream_source_row_index"] == 7
    assert copied.loc[0, "upstream_source_file"] == "upstream.csv"
    assert copied.loc[0, "source_sha256"] == renderer._sha256(source)


def test_h2_renderer_emits_only_supplementary_fail_closed_figure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_h2_source(run_dir)

    summary = renderer._render_h2(run_dir, tmp_path / "out")

    assert summary["main_figure_count"] == 0
    assert summary["supplementary_figure_count"] == 1
    assert summary["scientific_status"] == "failed_closed"
    contract_path = (
        tmp_path
        / "out"
        / "h2_supplementary_figure_s1_fail_closed_feasibility.figure_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["panels"][0]["metadata"]["placement"] == "supplementary"
    assert "No effect estimate" in contract["statistics_note"] or contract["panels"][0][
        "claim"
    ].endswith("no effect estimate exists.")


def test_h2_renderer_rejects_an_effect_estimate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_h2_source(run_dir, effect_estimate=0.8)

    with pytest.raises(ValueError, match="unauthorized causal result"):
        renderer._render_h2(run_dir, tmp_path / "out")
