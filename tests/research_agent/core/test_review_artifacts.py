from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.reporting.review_artifacts import build_review_artifact_payloads


def _write_contract(
    root: Path,
    relative_dir: str,
    stem: str,
    *,
    figure_id: str,
    roles: list[str],
) -> str:
    out = root / relative_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.png").write_bytes(b"png")
    (out / f"{stem}.svg").write_text("<svg></svg>", encoding="utf-8")
    (out / f"{stem}.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": figure_id,
                "panels": [
                    {
                        "panel_id": chr(65 + index),
                        "title": role.replace("_", " ").title(),
                        "role": role,
                        "chart_type": "dot_interval",
                        "claim": f"{role} is shown.",
                    }
                    for index, role in enumerate(roles)
                ],
            }
        ),
        encoding="utf-8",
    )
    return f"{relative_dir}/{stem}.figure_contract.json"


def test_review_artifact_payloads_prioritize_primary_and_archive_stale_support(
    tmp_path: Path,
):
    primary = _write_contract(
        tmp_path,
        "publication_figures",
        "easyicu_publication_figure",
        figure_id="easyicu_publication_figure",
        roles=["descriptive_result", "relationship", "data_quality"],
    )
    covered = _write_contract(
        tmp_path,
        "steps/03_old_primary/outputs",
        "publication_figure",
        figure_id="publication_figure",
        roles=["descriptive_result", "relationship"],
    )
    missingness = _write_contract(
        tmp_path,
        "steps/04_missingness/outputs",
        "missingness_panel",
        figure_id="missingness_panel",
        roles=["audit"],
    )
    duplicate_missingness = _write_contract(
        tmp_path,
        "steps/05_missingness_repeat/outputs",
        "missingness_panel",
        figure_id="missingness_panel",
        roles=["audit"],
    )
    robustness = _write_contract(
        tmp_path,
        "steps/06_robustness/outputs",
        "robustness_panel",
        figure_id="robustness_panel",
        roles=["robustness"],
    )

    review, gallery, canonical = build_review_artifact_payloads(
        run_dir=tmp_path,
        gates={
            "display_primary_publication_contract_paths": [primary],
            "display_supporting_figure_contract_paths": [
                covered,
                missingness,
                duplicate_missingness,
                robustness,
            ],
        },
    )

    assert canonical["primary_publication_figure"] == (
        "publication_figures/easyicu_publication_figure.png"
    )
    assert canonical["primary_publication_figure_contract"] == primary
    assert canonical["primary_publication_figure_png"] == (
        "publication_figures/easyicu_publication_figure.png"
    )
    assert canonical["primary_publication_figure_svg"] == (
        "publication_figures/easyicu_publication_figure.svg"
    )
    assert review["primary_publication_figures"][0]["data_url"].startswith(
        "data:image/png;base64,"
    )
    assert all("data_url" not in row for row in review["supporting_figures"])
    assert [row["relative_path"] for row in gallery["figures"]] == [
        "publication_figures/easyicu_publication_figure.png",
        "steps/04_missingness/outputs/missingness_panel.png",
        "steps/06_robustness/outputs/robustness_panel.png",
    ]
    assert gallery["primary_count"] == 1
    assert gallery["supporting_count"] == 2
    assert gallery["archived_supporting_count"] == 2
    assert {
        row["archive_reason"] for row in review["archived_supporting_figures"]
    } == {
        "covered_by_primary_publication_figure",
        "duplicate_supporting_figure_id",
    }
    assert review["policy"][
        "supporting_step_figures_are_not_canonical_main_figures"
    ] is True


def test_review_artifact_payloads_fail_closed_without_primary_figure(tmp_path: Path):
    supporting = _write_contract(
        tmp_path,
        "steps/04_missingness/outputs",
        "missingness_panel",
        figure_id="missingness_panel",
        roles=["audit"],
    )

    review, gallery, canonical = build_review_artifact_payloads(
        run_dir=tmp_path,
        gates={
            "display_primary_publication_contract_paths": [],
            "display_supporting_figure_contract_paths": [supporting],
        },
    )

    assert canonical == {}
    assert review["primary_publication_figures"] == []
    assert [row["relative_path"] for row in review["supporting_figures"]] == [
        "steps/04_missingness/outputs/missingness_panel.png"
    ]
    assert gallery["status"] == "no_primary_publication_figure"
    assert gallery["primary_count"] == 0
    assert gallery["supporting_count"] == 1


def test_review_artifact_payloads_infer_chart_types_without_explicit_metadata(
    tmp_path: Path,
):
    primary = _write_contract(
        tmp_path,
        "publication_figures",
        "easyicu_publication_figure",
        figure_id="easyicu_publication_figure",
        roles=["descriptive_result"],
    )
    support_dir = tmp_path / "steps" / "04_supporting" / "outputs"
    support_dir.mkdir(parents=True, exist_ok=True)
    (support_dir / "supporting_quality.png").write_bytes(b"png")
    (support_dir / "supporting_quality.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "supporting_quality",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Missingness availability denominator count",
                        "role": "audit",
                        "claim": "Missingness and denominator counts are shown.",
                    },
                    {
                        "panel_id": "B",
                        "title": "Adjusted odds ratio forest estimate",
                        "role": "relationship",
                        "claim": "Adjusted odds ratios are shown on a forest scale.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    _, gallery, _ = build_review_artifact_payloads(
        run_dir=tmp_path,
        gates={
            "display_primary_publication_contract_paths": [primary],
            "display_supporting_figure_contract_paths": [
                "steps/04_supporting/outputs/supporting_quality.figure_contract.json"
            ],
        },
    )

    supporting = next(row for row in gallery["figures"] if row["tier"] == "supporting_step")
    assert supporting["chart_types"] == ["bar", "forest"]
