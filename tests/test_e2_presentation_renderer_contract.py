from pathlib import Path

import pytest

from tools.render_e2_presentation_figures import resolve_output_run_dir


def test_e2_renderer_does_not_invent_unbound_study_labels() -> None:
    renderer = (
        Path(__file__).resolve().parents[1] / "tools" / "render_e2_presentation_figures.py"
    ).read_text(encoding="utf-8")

    for unbound_claim in (
        "Peak lactate during 0–24 h",
        "Adjusted for age, sex and Charlson index",
        "registered 24-hour landmark analysis",
        "lactate-eligible",
    ):
        assert unbound_claim not in renderer


def test_e2_renderer_keeps_verified_sources_and_output_in_one_run(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()

    assert resolve_output_run_dir(run_dir, None) == run_dir
    assert resolve_output_run_dir(run_dir, run_dir / ".") == run_dir


def test_e2_renderer_rejects_cross_run_output(tmp_path: Path) -> None:
    run_dir = (tmp_path / "source-run").resolve()
    output_run_dir = (tmp_path / "web-run").resolve()
    run_dir.mkdir()
    output_run_dir.mkdir()

    with pytest.raises(ValueError, match="verifies every source binding"):
        resolve_output_run_dir(run_dir, output_run_dir)
