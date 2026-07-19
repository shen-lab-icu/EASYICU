"""Deterministic publication rendering for a simple cohort-flow contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.contracts.declared_product import (
    bind_declared_figure_products,
)
from easyicu.research_agent.pipeline import (
    _render_authorized_sealed_publication_bundle,
    _render_cohort_flow_publication_bundle_from_prior_outputs as cohort_flow_rescue,
    _render_publication_bundle_from_prior_outputs_for_step as routed_rescue,
)
from easyicu.research_agent.schema import AnalysisStep


def _write_cohort_flow_parent(run_dir: Path) -> Path:
    parent = run_dir / "steps" / "01_primary_cohort_flow" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stage": [
                "universe",
                "identifier_present",
                "eligible_age",
                "analysis_cohort",
            ],
            "n": [1000, 990, 900, 840],
            "percent_of_universe": [100.0, 99.0, 90.0, 84.0],
            "n_removed_from_prior_stage": [0, 10, 90, 60],
            "criterion": [
                "All supplied records",
                "Identifier is present",
                "Age meets the prespecified eligibility rule",
                "All registered eligibility criteria are met",
            ],
        }
    ).to_csv(parent / "cohort_flow.csv", index=False)
    pd.DataFrame(
        {
            "attrition_category": [
                "universe",
                "missing_identifier",
                "ineligible_age",
                "other_exclusion",
                "analysis_cohort",
            ],
            "n": [1000, 10, 90, 60, 840],
            "percent_of_universe": [100.0, 1.0, 9.0, 6.0, 84.0],
            "status": [
                "denominator",
                "excluded",
                "excluded",
                "excluded",
                "retained",
            ],
            "reason": [
                "All supplied records",
                "Identifier is missing",
                "Age criterion is not met",
                "Another registered criterion is not met",
                "Eligible analysis cohort",
            ],
            "partition_role": [
                "denominator_only",
                "partition_category",
                "partition_category",
                "partition_category",
                "partition_category",
            ],
        }
    ).to_csv(parent / "attrition.csv", index=False)
    (parent / "step_summary.json").write_text(
        json.dumps({"analysis_family": "cohort_definition_sensitivity"}),
        encoding="utf-8",
    )
    return parent


def test_cohort_flow_rescue_writes_publication_bundle_and_traceable_sources(
    tmp_path: Path,
):
    parent = _write_cohort_flow_parent(tmp_path)
    out = tmp_path / "steps" / "01_primary_cohort_flow_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(
        json.dumps(
            {
                "rendering_only": True,
                "deterministic_publication_figure_rescue": "no_parent_outputs",
                "figure_files": [],
                "warning": "No compatible parent outputs were available.",
            }
        ),
        encoding="utf-8",
    )

    repair_id = cohort_flow_rescue(
        run_dir=tmp_path,
        current_step_id="01_primary_cohort_flow_figure",
        out_dir=out,
    )

    assert repair_id == "cohort_flow_publication_bundle_from_parent_outputs_v1"
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out / f"publication_figure.{suffix}").exists()
    flow_source = out / "publication_figure_source_data.csv"
    attrition_source = out / "publication_figure_attrition_source_data.csv"
    assert flow_source.exists()
    assert attrition_source.exists()
    assert (
        FigureSourceDataValidator._compare_source_to_upstream(
            source_df=pd.read_csv(flow_source),
            source_path=flow_source,
            upstream_path=parent / "cohort_flow.csv",
        )["ok"]
        is True
    )
    assert (
        FigureSourceDataValidator._compare_source_to_upstream(
            source_df=pd.read_csv(attrition_source),
            source_path=attrition_source,
            upstream_path=parent / "attrition.csv",
        )["ok"]
        is True
    )

    contract_path = out / "publication_figure.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B"]
    assert (
        FigureContractQualityValidator().audit_contract_file(
            contract_path,
            manuscript_facing=True,
        )
        == []
    )

    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["source_step_id"] == "01_primary_cohort_flow"
    assert summary["deterministic_publication_figure_rescue"] == (
        "cohort_flow_publication_bundle_from_parent_outputs_v1"
    )
    assert "warning" not in summary
    assert set(summary["figure_files"]) == {
        "publication_figure.png",
        "publication_figure.svg",
        "publication_figure.pdf",
        "publication_figure.tiff",
    }
    assert summary["figure_path"] == "publication_figure.png"
    assert summary["figure_contract"] == "publication_figure.figure_contract.json"
    assert summary["n_flow_stages"] == 4
    assert summary["n_exclusion_categories"] == 3

    assert (
        FigureSourceDataValidator().audit(
            step=AnalysisStep(
                step_id="01_primary_cohort_flow_figure",
                intent="Render the cohort flow declared by step '01_primary_cohort_flow'.",
            ),
            out_dir=out,
            run_dir=tmp_path,
            step_summary=summary,
        )
        == []
    )
    assert bind_declared_figure_products(
        out_dir=out,
        declared_products=["figure:cohort_flow"],
        authorized_product_slots={"figure:cohort_flow": "cohort_flow"},
        renderer_repair_id=("cohort_flow_publication_bundle_from_parent_outputs_v1"),
        renderer_implementation_sha256="c" * 64,
        renderer_parent_digests={
            "step_summary.json": "1" * 64,
            "cohort_flow.csv": "2" * 64,
            "attrition.csv": "3" * 64,
        },
    )


def test_authorized_cohort_flow_renderer_rejects_parent_mutation(tmp_path: Path):
    parent = _write_cohort_flow_parent(tmp_path)
    sealed_paths = [
        parent / "step_summary.json",
        parent / "cohort_flow.csv",
        parent / "attrition.csv",
    ]
    seal = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sealed_paths
    }
    repair_id = "cohort_flow_publication_bundle_from_parent_outputs_v1"
    figure_step_id = "01_primary_cohort_flow_figure"

    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=repair_id,
            run_dir=tmp_path,
            current_step_id=figure_step_id,
            out_dir=tmp_path / "steps" / figure_step_id / "sealed_outputs",
            parent_artifact_digests=seal,
        )
        == repair_id
    )

    (parent / "attrition.csv").write_text("status,n\nexcluded,999\n", encoding="utf-8")
    assert (
        _render_authorized_sealed_publication_bundle(
            repair_id=repair_id,
            run_dir=tmp_path,
            current_step_id=figure_step_id,
            out_dir=tmp_path / "steps" / figure_step_id / "mutated_outputs",
            parent_artifact_digests=seal,
        )
        is None
    )


def test_cohort_route_falls_back_to_flow_but_preserves_overlap_priority(
    tmp_path: Path,
    monkeypatch,
):
    _write_cohort_flow_parent(tmp_path)
    out = tmp_path / "steps" / "01_primary_cohort_flow_figure" / "outputs"

    repair_id = routed_rescue(
        run_dir=tmp_path,
        current_step_id="01_primary_cohort_flow_figure",
        out_dir=out,
    )
    assert repair_id == "cohort_flow_publication_bundle_from_parent_outputs_v1"

    import easyicu.research_agent.pipeline as pipeline

    calls: list[str] = []

    def overlap_renderer(**_kwargs):
        calls.append("overlap")
        return "overlap_renderer"

    def flow_renderer(**_kwargs):
        calls.append("flow")
        return "flow_renderer"

    monkeypatch.setattr(
        pipeline,
        "_render_cohort_overlap_publication_bundle_from_prior_outputs",
        overlap_renderer,
    )
    monkeypatch.setattr(
        pipeline,
        "_render_cohort_flow_publication_bundle_from_prior_outputs",
        flow_renderer,
    )

    assert (
        routed_rescue(
            run_dir=tmp_path,
            current_step_id="01_primary_cohort_flow_figure",
            out_dir=out,
        )
        == "overlap_renderer"
    )
    assert calls == ["overlap"]
