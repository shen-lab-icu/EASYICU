from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.reporting.publication_bundles import (
    _promote_prior_publication_bundle,
    _promote_sibling_figure_exports,
    _seal_corrupt_step_summary,
)


def _sibling_output(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "figure" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir.parent / "outputs.png").write_bytes(b"figure")
    return run_dir, out_dir


def test_sibling_promotion_does_not_follow_symlink_destination(tmp_path: Path) -> None:
    run_dir, out_dir = _sibling_output(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    (out_dir / "publication_figure.png").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        _promote_sibling_figure_exports(out_dir=out_dir, run_dir=run_dir)

    assert outside.read_bytes() == b"untouched"


def test_sibling_promotion_does_not_follow_symlink_summary(tmp_path: Path) -> None:
    run_dir, out_dir = _sibling_output(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    (out_dir / "step_summary.json").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        _promote_sibling_figure_exports(out_dir=out_dir, run_dir=run_dir)

    assert outside.read_bytes() == b"untouched"
    assert not (out_dir / "publication_figure.png").exists()


def test_sibling_promotion_rejects_symlink_source(tmp_path: Path) -> None:
    run_dir, out_dir = _sibling_output(tmp_path)
    source = out_dir.parent / "outputs.png"
    source.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"private")
    source.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        _promote_sibling_figure_exports(out_dir=out_dir, run_dir=run_dir)

    assert not (out_dir / "publication_figure.png").exists()


def test_sibling_promotion_rejects_symlink_output_directory(tmp_path: Path) -> None:
    run_dir, out_dir = _sibling_output(tmp_path)
    out_dir.rmdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    out_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _promote_sibling_figure_exports(out_dir=out_dir, run_dir=run_dir)

    assert list(outside_dir.iterdir()) == []


def test_corrupt_summary_seal_refuses_conflicting_existing_file(tmp_path: Path) -> None:
    raw = b"{not json"
    (tmp_path / "step_summary.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()[:8]
    seal = tmp_path / f"step_summary.corrupt.{digest}.json"
    seal.write_bytes(b"wrong evidence")

    with pytest.raises(ValueError, match="seal"):
        _seal_corrupt_step_summary(tmp_path)

    assert (tmp_path / "step_summary.json").read_bytes() == raw
    assert seal.read_bytes() == b"wrong evidence"


def test_corrupt_summary_seal_does_not_follow_symlink(tmp_path: Path) -> None:
    raw = b"{not json"
    (tmp_path / "step_summary.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()[:8]
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_bytes(b"untouched")
    (tmp_path / f"step_summary.corrupt.{digest}.json").symlink_to(outside)

    with pytest.raises(ValueError, match="seal"):
        _seal_corrupt_step_summary(tmp_path)

    assert (tmp_path / "step_summary.json").read_bytes() == raw
    assert outside.read_bytes() == b"untouched"


def test_sibling_promotion_preserves_corrupt_summary_bytes(tmp_path: Path) -> None:
    run_dir, out_dir = _sibling_output(tmp_path)
    corrupt = b"{not json"
    (out_dir / "step_summary.json").write_bytes(corrupt)

    result = _promote_sibling_figure_exports(out_dir=out_dir, run_dir=run_dir)

    assert result == "sibling_figure_exports_promote_v1"
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    seal_name = summary["publication_figure_rescue"]["corrupt_summary_sealed_as"]
    assert (out_dir / seal_name).read_bytes() == corrupt


def test_prior_promotion_does_not_follow_symlink_destination(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source_dir = run_dir / "steps" / "analysis" / "outputs"
    out_dir = run_dir / "steps" / "figure" / "outputs"
    source_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (source_dir / "source.png").write_bytes(b"figure")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    (out_dir / "publication_figure.png").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        _promote_prior_publication_bundle(
            run_dir=run_dir, current_step_id="figure", out_dir=out_dir
        )

    assert outside.read_bytes() == b"untouched"
