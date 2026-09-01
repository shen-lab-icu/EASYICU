from __future__ import annotations

import hashlib
import json
from pathlib import Path

from easyicu.webserver import agent_runs
from easyicu.webserver.agent_pipeline_runs import _figure_projection
from easyicu.webserver.figure_presentation import verified_presentation_gallery


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_digest_bound_presentation_gallery_is_used_without_overwriting_original(
    tmp_path: Path,
) -> None:
    original_png = tmp_path / "original.png"
    original_png.write_bytes(b"original-figure")
    canonical_gallery = {
        "status": "ok",
        "primary_count": 1,
        "supporting_count": 0,
        "figures": [
            {
                "figure_id": "original",
                "relative_path": "original.png",
                "status": "registered",
            }
        ],
    }
    _write_json(tmp_path / "figure_gallery.json", canonical_gallery)
    source_table = tmp_path / "source.csv"
    source_table.write_text("x,y\n1,2\n", encoding="utf-8")
    presentation_png = tmp_path / "presentation_figures" / "polished.png"
    presentation_png.parent.mkdir(parents=True)
    presentation_png.write_bytes(b"polished-figure")
    _write_json(
        presentation_png.parent / "presentation_figure_gallery.json",
        {
            "schema_version": "easyicu.presentation-figure-gallery/1",
            "status": "presentation_only",
            "authority_ceiling": "analysis_only",
            "derived_from": {
                "artifact": "figure_gallery.json",
                "sha256": hashlib.sha256(
                    (tmp_path / "figure_gallery.json").read_bytes()
                ).hexdigest(),
            },
            "source_bindings": [
                {
                    "relative_path": "source.csv",
                    "sha256": hashlib.sha256(source_table.read_bytes()).hexdigest(),
                }
            ],
            "primary_count": 1,
            "supporting_count": 0,
            "figures": [
                {
                    "figure_id": "polished",
                    "label": "Polished result",
                    "relative_path": "presentation_figures/polished.png",
                    "sha256": hashlib.sha256(presentation_png.read_bytes()).hexdigest(),
                    "status": "presentation_only",
                }
            ],
        },
    )

    payload = _figure_projection(tmp_path)

    assert payload["presentation_variant"] is True
    assert payload["authority_ceiling"] == "analysis_only"
    assert payload["original_run_figures_preserved"] is True
    assert [row["relative_path"] for row in payload["figures"]] == [
        "presentation_figures/polished.png"
    ]
    assert original_png.read_bytes() == b"original-figure"

    browser_payload = agent_runs.read_run_artifact(
        str(tmp_path), "figure_gallery.json"
    )["payload"]
    assert browser_payload["presentation_variant"] is True
    assert browser_payload["embedded_count"] == 1
    assert browser_payload["figures"][0]["data_url"].startswith(
        "data:image/png;base64,"
    )


def test_tampered_presentation_source_falls_back_to_registered_gallery(
    tmp_path: Path,
) -> None:
    original_png = tmp_path / "original.png"
    original_png.write_bytes(b"original-figure")
    _write_json(
        tmp_path / "figure_gallery.json",
        {
            "status": "ok",
            "primary_count": 1,
            "supporting_count": 0,
            "figures": [
                {
                    "figure_id": "original",
                    "relative_path": "original.png",
                    "status": "registered",
                }
            ],
        },
    )
    source_table = tmp_path / "source.csv"
    source_table.write_text("x,y\n1,2\n", encoding="utf-8")
    presentation_dir = tmp_path / "presentation_figures"
    presentation_dir.mkdir()
    (presentation_dir / "polished.png").write_bytes(b"polished-figure")
    polished_sha256 = hashlib.sha256(
        (presentation_dir / "polished.png").read_bytes()
    ).hexdigest()
    _write_json(
        presentation_dir / "presentation_figure_gallery.json",
        {
            "schema_version": "easyicu.presentation-figure-gallery/1",
            "status": "presentation_only",
            "authority_ceiling": "analysis_only",
            "derived_from": {
                "artifact": "figure_gallery.json",
                "sha256": hashlib.sha256(
                    (tmp_path / "figure_gallery.json").read_bytes()
                ).hexdigest(),
            },
            "source_bindings": [{"relative_path": "source.csv", "sha256": "0" * 64}],
            "figures": [
                {
                    "relative_path": "presentation_figures/polished.png",
                    "sha256": polished_sha256,
                    "status": "presentation_only",
                }
            ],
        },
    )

    payload = _figure_projection(tmp_path)

    assert payload["presentation_variant"] is False
    assert [row["relative_path"] for row in payload["figures"]] == ["original.png"]


def test_tampered_presentation_png_falls_back_to_registered_gallery(
    tmp_path: Path,
) -> None:
    original_png = tmp_path / "original.png"
    original_png.write_bytes(b"original-figure")
    _write_json(
        tmp_path / "figure_gallery.json",
        {
            "status": "ok",
            "primary_count": 1,
            "supporting_count": 0,
            "figures": [{"relative_path": "original.png", "status": "registered"}],
        },
    )
    source_table = tmp_path / "source.csv"
    source_table.write_text("x,y\n1,2\n", encoding="utf-8")
    presentation_dir = tmp_path / "presentation_figures"
    presentation_dir.mkdir()
    presentation_png = presentation_dir / "polished.png"
    presentation_png.write_bytes(b"registered-presentation")
    registered_sha256 = hashlib.sha256(presentation_png.read_bytes()).hexdigest()
    _write_json(
        presentation_dir / "presentation_figure_gallery.json",
        {
            "schema_version": "easyicu.presentation-figure-gallery/1",
            "status": "presentation_only",
            "authority_ceiling": "analysis_only",
            "derived_from": {
                "artifact": "figure_gallery.json",
                "sha256": hashlib.sha256(
                    (tmp_path / "figure_gallery.json").read_bytes()
                ).hexdigest(),
            },
            "source_bindings": [
                {
                    "relative_path": "source.csv",
                    "sha256": hashlib.sha256(source_table.read_bytes()).hexdigest(),
                }
            ],
            "figures": [
                {
                    "relative_path": "presentation_figures/polished.png",
                    "sha256": registered_sha256,
                    "status": "presentation_only",
                }
            ],
        },
    )
    presentation_png.write_bytes(b"tampered-presentation")

    payload = _figure_projection(tmp_path)

    assert payload["presentation_variant"] is False
    assert [row["relative_path"] for row in payload["figures"]] == ["original.png"]


def test_oversized_presentation_manifest_is_rejected_before_json_load(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "figure_gallery.json"
    _write_json(canonical, {"status": "ok", "figures": []})
    presentation_dir = tmp_path / "presentation_figures"
    presentation_dir.mkdir()
    source_table = tmp_path / "source.csv"
    source_table.write_text("x,y\n1,2\n", encoding="utf-8")
    png = presentation_dir / "same.png"
    png.write_bytes(b"png")
    _write_json(
        presentation_dir / "presentation_figure_gallery.json",
        {
            "schema_version": "easyicu.presentation-figure-gallery/1",
            "status": "presentation_only",
            "authority_ceiling": "analysis_only",
            "derived_from": {
                "artifact": "figure_gallery.json",
                "sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
            },
            "source_bindings": [
                {
                    "relative_path": "source.csv",
                    "sha256": hashlib.sha256(source_table.read_bytes()).hexdigest(),
                }
            ],
            "figures": [
                {
                    "relative_path": "presentation_figures/same.png",
                    "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                }
            ],
            "padding": "x" * 300_000,
        },
    )

    assert verified_presentation_gallery(tmp_path, {}) is None


def test_presentation_gallery_rejects_unbounded_figure_count(tmp_path: Path) -> None:
    canonical = tmp_path / "figure_gallery.json"
    _write_json(canonical, {"status": "ok", "figures": []})
    source_table = tmp_path / "source.csv"
    source_table.write_text("x,y\n1,2\n", encoding="utf-8")
    presentation_dir = tmp_path / "presentation_figures"
    presentation_dir.mkdir()
    png = presentation_dir / "same.png"
    png.write_bytes(b"png")
    _write_json(
        presentation_dir / "presentation_figure_gallery.json",
        {
            "schema_version": "easyicu.presentation-figure-gallery/1",
            "status": "presentation_only",
            "authority_ceiling": "analysis_only",
            "derived_from": {
                "artifact": "figure_gallery.json",
                "sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
            },
            "source_bindings": [
                {
                    "relative_path": "source.csv",
                    "sha256": hashlib.sha256(source_table.read_bytes()).hexdigest(),
                }
            ],
            "figures": [
                {
                    "relative_path": "presentation_figures/same.png",
                    "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                }
                for _ in range(41)
            ],
        },
    )

    assert verified_presentation_gallery(tmp_path, {}) is None
