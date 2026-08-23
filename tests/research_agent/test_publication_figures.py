from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from easyicu.research_agent.figures.publication import (
    PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
    PanelSpec,
    apply_publication_style,
    audit_figure_contract,
    audit_publication_exports,
    make_figure_contract,
    save_publication_figure,
)


def _prepare_robustness_authority(ra, run_dir: Path, evidence, rows) -> None:
    """Give panel fixtures the same lock + digest-bound row authority as runs."""
    from easyicu.research_agent.cohort.schema import CohortDefinition
    from easyicu.research_agent.robustness.panel import (
        RobustnessSpec,
        default_robustness_specs,
        write_locked_robustness_specs,
    )

    specs = list(default_robustness_specs())
    known = {spec.spec_id for spec in specs}
    for row in rows:
        if row.spec_id == "primary" or row.spec_id in known:
            continue
        kwargs = {}
        if row.axis == "cohort":
            kwargs["cohort_override"] = CohortDefinition(name=row.spec_id)
        elif row.axis == "missing":
            kwargs["missing_override"] = {"strategy": f"test_{row.spec_id}"}
        elif row.axis == "outcome":
            kwargs["outcome_override"] = {"target": row.spec_id}
        specs.append(
            RobustnessSpec(
                spec_id=row.spec_id,
                axis=row.axis,
                description="Test-owned robustness specification.",
                **kwargs,
            )
        )
        known.add(row.spec_id)
    plan = ra.AnalysisPlan(research_question="test", steps=[], robustness_specs=specs)
    write_locked_robustness_specs(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="test",
    )
    for row in rows:
        if not row.converged or row.point_estimate is None:
            continue
        payload = (
            {
                "primary_or": row.point_estimate,
                "primary_ci_low": row.ci_low,
                "primary_ci_high": row.ci_high,
                "sample_size": row.n,
            }
            if row.spec_id == "primary"
            else {"robustness_rows": [row.to_dict()]}
        )
        evidence.register_json(
            kind="statistic",
            description="Publication-figure fixture row authority.",
            payload=payload,
            filename=f"{row.evidence_id}.json",
            evidence_id=row.evidence_id,
        )


@pytest.mark.parametrize("variant_count", [0, 3])
def test_robustness_panel_publication_figure_has_no_header_title_overlap(
    ra, tmp_path: Path, variant_count: int
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.figures.skill import PublicationFigureSkill
    from easyicu.research_agent.robustness.panel import (
        RobustnessPanel,
        RobustnessPanelRow,
    )

    rows = [
        RobustnessPanelRow(
            spec_id="primary",
            axis="primary",
            n=100,
            point_estimate=1.14,
            ci_low=1.08,
            ci_high=1.21,
            se=0.03,
            evidence_id="primary_row",
            converged=True,
        )
    ]
    for idx in range(variant_count):
        rows.append(
            RobustnessPanelRow(
                spec_id=f"alt_variant_{idx}",
                axis="cohort",
                n=90 - idx,
                point_estimate=1.05 + idx * 0.03,
                ci_low=0.98 + idx * 0.02,
                ci_high=1.18 + idx * 0.02,
                se=0.04,
                evidence_id=f"alt_row_{idx}",
                converged=True,
            )
        )
    panel = RobustnessPanel.from_rows(rows)
    (tmp_path / "robustness_panel.json").write_text(
        json.dumps(panel.to_dict()), encoding="utf-8"
    )
    evidence = EvidenceStore(tmp_path)
    source_record = evidence.register_json(
        kind="statistic",
        description="Robustness panel.",
        payload=panel.to_dict(),
        filename="robustness_panel.json",
        evidence_id="robustness_panel",
    )
    context = ra.ResearchContext(
        research_question="Does early severity predict mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    result = PublicationFigureSkill()._render_robustness_panel(
        context=context,
        plan=ra.AnalysisPlan(
            research_question=context.research_question,
            steps=[],
            display_labels={"death": "In-hospital mortality"},
        ),
        evidence=evidence,
        run_dir=tmp_path,
        source_record=source_record,
        panel=panel,
        prompt_pack_version="test",
    )

    assert result.generated is True
    assert not any(
        finding.severity == "error" and "overlapping text" in finding.message
        for finding in result.findings
    )
    assert not any(
        finding.severity == "error" and finding.validator == "figure_contract_quality"
        for finding in result.findings
    )
    contract_path = (
        tmp_path
        / "publication_figures"
        / "easyicu_publication_figure.figure_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert (
        tmp_path
        / "publication_figures"
        / "publication_figure_source_robustness_axis_summary.csv"
    ).exists()


def test_publication_figure_skill_rebuilds_stale_single_panel_bundle(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.figures.skill import PublicationFigureSkill
    from easyicu.research_agent.robustness.panel import (
        RobustnessPanel,
        RobustnessPanelRow,
    )

    out = tmp_path / "publication_figures"
    out.mkdir(parents=True, exist_ok=True)
    stale_png = out / "easyicu_publication_figure.png"
    stale_png.write_text("old figure", encoding="utf-8")
    (out / "easyicu_publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Old robustness result.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary effect and robustness variants",
                        "role": "robustness",
                        "claim": "Old single-panel result.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                spec_id="primary",
                axis="primary",
                n=100,
                point_estimate=1.14,
                ci_low=1.08,
                ci_high=1.21,
                se=0.03,
                evidence_id="primary_row",
                converged=True,
            ),
            RobustnessPanelRow(
                spec_id="alt_cohort",
                axis="cohort",
                n=92,
                point_estimate=1.08,
                ci_low=1.01,
                ci_high=1.18,
                se=0.04,
                evidence_id="alt_row",
                converged=True,
            ),
        ]
    )
    (tmp_path / "robustness_panel.json").write_text(
        json.dumps(panel.to_dict()),
        encoding="utf-8",
    )
    evidence = EvidenceStore(tmp_path)
    evidence.register_file(
        kind="figure",
        description="Old publication figure.",
        source_path=stale_png,
        evidence_id="publication_figure_png",
        aliases=["publication_figure"],
        producer=PublicationFigureSkill.name,
        generation_mode="deterministic_figure_skill",
    )
    evidence.register_json(
        kind="statistic",
        description="Robustness panel.",
        payload=panel.to_dict(),
        filename="robustness_panel.json",
        evidence_id="robustness_panel",
    )
    context = ra.ResearchContext(
        research_question="Does early severity predict mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="05_sensitivity_comparison_figure",
                intent="Render a manuscript-facing sensitivity figure.",
                expected_outputs=["figure:publication"],
            )
        ],
    )

    result = PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=tmp_path,
        prompt_pack_version="test",
    )

    assert result.generated is True
    contract = json.loads(
        (out / "easyicu_publication_figure.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert any(eid.endswith("_v2") for eid in result.figure_evidence_ids)


def test_curated_publication_bundle_requires_current_policy_version(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.figures.skill import (
        _has_curated_publication_figure_bundle,
        _source_fingerprint_metadata,
    )

    evidence = EvidenceStore(tmp_path)
    source_path = tmp_path / "publication_figure_source_data.csv"
    source_path.write_text("term,estimate\nsepsis3,1.05\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Publication source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_data",
    )
    current_metadata = _source_fingerprint_metadata(
        evidence, [source_record.evidence_id]
    )
    assert (
        current_metadata["figure_skill_policy_version"]
        == PUBLICATION_FIGURE_SKILL_POLICY_VERSION
    )
    stale_metadata = {
        key: value
        for key, value in current_metadata.items()
        if key != "figure_skill_policy_version"
    }
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Current figure contract.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary estimate",
                        "role": "primary_estimand",
                        "claim": "Primary estimate is shown.",
                        "evidence_ids": [source_record.evidence_id],
                    },
                    {
                        "panel_id": "B",
                        "title": "Sensitivity context",
                        "role": "sensitivity",
                        "claim": "Sensitivity context is shown.",
                        "evidence_ids": [source_record.evidence_id],
                    },
                ],
                "source_data": [source_record.evidence_id],
            }
        ),
        encoding="utf-8",
    )
    contract_record = evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=stale_metadata,
    )
    for suffix in ("svg", "png"):
        path = tmp_path / f"easyicu_publication_figure.{suffix}"
        path.write_text("<svg></svg>" if suffix == "svg" else "png", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={
                "figure_role": "publication_figure",
                "contract_evidence_id": contract_record.evidence_id,
                **stale_metadata,
            },
        )

    assert _has_curated_publication_figure_bundle(evidence, run_dir=tmp_path) is False


def test_curated_publication_bundle_rejects_digest_stale_figure_bytes(
    ra, tmp_path: Path
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.figures.skill import (
        _has_curated_publication_figure_bundle,
        _source_fingerprint_metadata,
    )

    evidence = EvidenceStore(tmp_path)
    source_path = tmp_path / "publication_figure_source_data.csv"
    source_path.write_text("term,estimate\nsepsis3,1.05\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Publication source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_data",
    )
    metadata = _source_fingerprint_metadata(evidence, [source_record.evidence_id])
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Current figure contract.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary estimate",
                        "role": "primary_estimand",
                        "claim": "Primary estimate is shown.",
                        "evidence_ids": [source_record.evidence_id],
                    },
                    {
                        "panel_id": "B",
                        "title": "Sensitivity context",
                        "role": "sensitivity",
                        "claim": "Sensitivity context is shown.",
                        "evidence_ids": [source_record.evidence_id],
                    },
                ],
                "source_data": [source_record.evidence_id],
            }
        ),
        encoding="utf-8",
    )
    contract_record = evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=metadata,
    )
    figure_path = tmp_path / "easyicu_publication_figure.svg"
    figure_path.write_text("<svg></svg>", encoding="utf-8")
    figure_record = evidence.register_file(
        kind="figure",
        description="Publication figure export.",
        source_path=figure_path,
        evidence_id="publication_figure_svg",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata={
            "figure_role": "publication_figure",
            "figure_id": "easyicu_publication_figure",
            "contract_evidence_id": contract_record.evidence_id,
            "figure_privacy_audit_evidence_id": "privacy-a",
            **metadata,
        },
    )
    assert _has_curated_publication_figure_bundle(evidence, run_dir=tmp_path) is False
    png_path = tmp_path / "easyicu_publication_figure.png"
    png_path.write_bytes(b"png")
    evidence.register_file(
        kind="figure",
        description="Publication figure export.",
        source_path=png_path,
        evidence_id="publication_figure_png",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata={
            "figure_role": "publication_figure",
            "figure_id": "easyicu_publication_figure",
            "contract_evidence_id": contract_record.evidence_id,
            "figure_privacy_audit_evidence_id": "privacy-b",
            **metadata,
        },
    )
    assert _has_curated_publication_figure_bundle(evidence, run_dir=tmp_path) is False
    evidence.register_file(
        kind="figure",
        description="Publication figure export.",
        source_path=png_path,
        evidence_id="publication_figure_png_current",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata={
            "figure_role": "publication_figure",
            "figure_id": "easyicu_publication_figure",
            "contract_evidence_id": contract_record.evidence_id,
            "figure_privacy_audit_evidence_id": "privacy-a",
            **metadata,
        },
    )
    assert _has_curated_publication_figure_bundle(evidence, run_dir=tmp_path) is True
    (tmp_path / figure_record.relative_path).write_text(
        "<svg>tampered</svg>", encoding="utf-8"
    )

    assert _has_curated_publication_figure_bundle(evidence, run_dir=tmp_path) is False


def test_publication_figure_skill_promotes_deterministic_step_before_family_renderer(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    from PIL import Image

    from easyicu.research_agent.robustness.panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    outputs = (
        run_dir
        / "steps"
        / "05_sensitivity_comparison_across_definitions_figure"
        / "outputs"
    )
    outputs.mkdir(parents=True, exist_ok=True)
    svg = outputs / "sensitivity_forest.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="140">'
        '<rect width="240" height="140" fill="white"/>'
        '<text x="16" y="32">Sensitivity forest</text>'
        '<text x="16" y="64">No lactate covariate</text>'
        "</svg>",
        encoding="utf-8",
    )
    png = outputs / "sensitivity_forest.png"
    Image.new("RGB", (240, 140), "white").save(png)
    source = outputs / "sensitivity_forest_source_data.csv"
    source.write_text(
        "spec_id,effect_scale,point_estimate,ci_low,ci_high\n"
        "primary,OR,1.05,0.99,1.11\n"
        "no_lactate,OR,1.24,1.18,1.31\n",
        encoding="utf-8",
    )
    contract = make_figure_contract(
        figure_id="sensitivity_forest",
        core_claim="Registered sensitivity estimates show which design choices drive the association.",
        panels=[
            {
                "panel_id": "A",
                "title": "Ratio-scale sensitivity",
                "role": "robustness",
                "claim": "The ratio-scale estimates are drawn from the sensitivity table.",
            },
            {
                "panel_id": "B",
                "title": "Denominator audit",
                "role": "audit",
                "claim": "Analytic sample sizes are shown next to the sensitivity estimates.",
            },
        ],
        source_data=["sensitivity_forest_source_data.csv"],
    )
    contract_path = outputs / "sensitivity_forest.figure_contract.json"
    contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")

    metadata = {
        "figure_role": "publication_figure",
        "step_id": "05_sensitivity_comparison_across_definitions_figure",
        "generation_mode": "deterministic_standard",
    }
    for path, kind, evidence_id in (
        (svg, "figure", "figure_sensitivity_forest_svg"),
        (png, "figure", "figure_sensitivity_forest_png"),
        (contract_path, "log", "log_sensitivity_forest_contract"),
        (source, "table", "table_sensitivity_forest_source_data"),
    ):
        evidence.register_file(
            kind=kind,
            description="Registered sensitivity forest bundle.",
            source_path=path,
            evidence_id=evidence_id,
            producer="runner",
            generation_mode="deterministic_standard",
            metadata=metadata if kind != "table" else {"step_id": metadata["step_id"]},
        )
    active_step_evidence_ids = [
        "figure_sensitivity_forest_svg",
        "figure_sensitivity_forest_png",
        "log_sensitivity_forest_contract",
        "table_sensitivity_forest_source_data",
    ]

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary", "primary", 100, 9.99, 9.0, 11.0, 0.1, "e1", True
            ),
            RobustnessPanelRow("alt", "cohort", 90, 8.88, 8.0, 10.0, 0.2, "e2", True),
        ],
        locked_at="2026-07-02T00:00:00Z",
    )
    _prepare_robustness_authority(ra, run_dir, evidence, panel.rows)
    write_robustness_panel(
        run_dir=run_dir,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )
    context = ra.ResearchContext(
        research_question="Is Sepsis-3 associated with mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="05_sensitivity_comparison_across_definitions_figure",
                intent="Render a sensitivity comparison figure.",
                expected_outputs=["figure:sensitivity_forest"],
            )
        ],
    )

    def family_renderer_must_not_run(*args, **kwargs):
        raise AssertionError("deterministic step bundle must outrank family synthesis")

    monkeypatch.setattr(
        "easyicu.research_agent.figures.skill.render_family_figure",
        family_renderer_must_not_run,
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["generation_mode"] == "promoted_step_publication_figure"
    assert summary["promoted_from_stem"] == "sensitivity_forest"
    promoted_contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert "sensitivity estimates" in promoted_contract["core_claim"]
    assert "robustness panel" not in promoted_contract["statistics_note"].lower()
    assert promoted_contract["source_data"] == ["sensitivity_forest_source_data.csv"]
    assert (
        run_dir / "publication_figures" / "sensitivity_forest_source_data.csv"
    ).is_file()
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready

    readiness = _publication_figure_bundle_ready(
        evidence=evidence,
        run_dir=run_dir,
        per_step_records=[
            {
                "step_id": "05_sensitivity_comparison_across_definitions_figure",
                "status": "ok",
                "evidence_ids": active_step_evidence_ids,
            }
        ],
    )
    assert readiness["publication_figure_contract_ready"] is True
    assert readiness["publication_figure_source_data_ready"] is True
    assert readiness["publication_figure_bundle_ready"] is True


def test_publication_figure_skill_prefers_primary_bundle_over_sensitivity(
    ra,
    tmp_path: Path,
):
    from PIL import Image

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)

    def register_bundle(
        *,
        step_id: str,
        stem: str,
        core_claim: str,
        panel_roles: list[str],
    ) -> None:
        outputs = run_dir / "steps" / step_id / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        svg = outputs / f"{stem}.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="140">'
            '<rect width="240" height="140" fill="white"/>'
            f'<text x="16" y="32">{stem}</text>'
            "</svg>",
            encoding="utf-8",
        )
        png = outputs / f"{stem}.png"
        Image.new("RGB", (240, 140), "white").save(png)
        source = outputs / f"{stem}_source_data.csv"
        source.write_text("term,value\nprimary,1.05\n", encoding="utf-8")
        contract = make_figure_contract(
            figure_id=stem,
            core_claim=core_claim,
            panels=[
                {
                    "panel_id": chr(ord("A") + idx),
                    "title": f"Panel {idx + 1}",
                    "role": role,
                    "chart_type": (
                        "dot_interval_absolute_risk"
                        if role == "descriptive_result"
                        else "forest"
                    ),
                    "claim": f"{role} evidence is displayed.",
                }
                for idx, role in enumerate(panel_roles)
            ],
            source_data=[source.name],
        )
        contract_path = outputs / f"{stem}.figure_contract.json"
        contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")

        metadata = {
            "figure_role": "publication_figure",
            "step_id": step_id,
            "generation_mode": "fallback",
        }
        for path, kind, evidence_id in (
            (svg, "figure", f"figure_{stem}_svg"),
            (png, "figure", f"figure_{stem}_png"),
            (contract_path, "log", f"log_{stem}_contract"),
            (source, "table", f"table_{stem}_source_data"),
        ):
            evidence.register_file(
                kind=kind,
                description=f"Registered {stem} bundle.",
                source_path=path,
                evidence_id=evidence_id,
                producer="runner",
                generation_mode="fallback",
                metadata=metadata if kind != "table" else {"step_id": step_id},
            )

    register_bundle(
        step_id="03_primary_results_publication_figure_repair",
        stem="primary_results_figure",
        core_claim="The primary results show prevalence, absolute risk, and adjusted effect.",
        panel_roles=["descriptive_result", "primary_estimand"],
    )
    register_bundle(
        step_id="05_sensitivity_comparison_across_definitions_figure",
        stem="sensitivity_forest",
        core_claim="Sensitivity estimates show which design choices drive the association.",
        panel_roles=["robustness", "audit"],
    )
    context = ra.ResearchContext(
        research_question="Is Sepsis-3 associated with mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="03_primary_results_publication_figure_repair",
                intent="Render the primary manuscript figure.",
                expected_outputs=["figure:publication_figure"],
            ),
            ra.AnalysisStep(
                step_id="05_sensitivity_comparison_across_definitions_figure",
                intent="Render a sensitivity comparison figure.",
                expected_outputs=["figure:sensitivity_forest"],
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        summary["promoted_from_step_id"]
        == "03_primary_results_publication_figure_repair"
    )
    assert summary["promoted_from_stem"] == "primary_results_figure"
    promoted_contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert "primary results" in promoted_contract["core_claim"]
    assert promoted_contract["panels"][0]["metadata"]["chart_type"] == (
        "dot_interval_absolute_risk"
    )
    assert promoted_contract["panels"][1]["metadata"]["chart_type"] == "forest"


def test_publication_figure_skill_resolves_contract_source_from_parent_step(
    ra,
    tmp_path: Path,
):
    from PIL import Image

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    parent_outputs = run_dir / "steps" / "05_primary_model" / "outputs"
    parent_outputs.mkdir(parents=True, exist_ok=True)
    shared_source = parent_outputs / "shared_source.csv"
    shared_source.write_text(
        "term,odds_ratio,ci_low,ci_high\nexposure,1.8,1.6,2.0\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="Parent analysis source table.",
        source_path=shared_source,
        evidence_id="table_shared_source",
        produced_by_step="05_primary_model",
        producer="runner",
        generation_mode="fallback",
    )

    child_step = "05_primary_model_figure"
    outputs = run_dir / "steps" / child_step / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    svg = outputs / "publication_figure.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="140">'
        '<rect width="240" height="140" fill="white"/>'
        '<text x="16" y="32">Primary article figure</text>'
        "</svg>",
        encoding="utf-8",
    )
    png = outputs / "publication_figure.png"
    Image.new("RGB", (240, 140), "white").save(png)
    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim="Absolute risk, adjusted association, robustness, and missingness are shown.",
        panels=[
            {
                "panel_id": "A",
                "title": "Absolute risk",
                "role": "descriptive_result",
                "chart_type": "event_rate_panel",
                "claim": "Absolute risk is shown.",
                "evidence_ids": [shared_source.name],
            },
            {
                "panel_id": "B",
                "title": "Adjusted association",
                "role": "primary_estimand",
                "chart_type": "forest",
                "claim": "The primary adjusted estimate is shown.",
                "evidence_ids": [shared_source.name],
            },
            {
                "panel_id": "C",
                "title": "Robustness",
                "role": "robustness",
                "chart_type": "dot_interval",
                "claim": "A sensitivity estimate is shown.",
                "evidence_ids": [shared_source.name],
            },
            {
                "panel_id": "D",
                "title": "Missingness",
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Measurement availability is shown.",
                "evidence_ids": [shared_source.name],
            },
        ],
        source_data=[shared_source.name],
    )
    contract_path = outputs / "publication_figure.figure_contract.json"
    contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")
    metadata = {"figure_role": "publication_figure", "step_id": child_step}
    for path, kind, evidence_id in (
        (svg, "figure", "figure_child_svg"),
        (png, "figure", "figure_child_png"),
        (contract_path, "log", "log_child_contract"),
    ):
        evidence.register_file(
            kind=kind,
            description="Child publication bundle.",
            source_path=path,
            evidence_id=evidence_id,
            produced_by_step=child_step,
            producer="runner",
            generation_mode="fallback",
            metadata=metadata,
        )

    context = ra.ResearchContext(
        research_question="Is the exposure associated with mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id=child_step,
                intent="Render the primary manuscript figure.",
                expected_outputs=["figure:publication_figure"],
            )
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["generation_mode"] == "promoted_step_publication_figure"
    assert summary["promoted_from_step_id"] == child_step
    assert "table_shared_source" in summary["source_evidence_ids"]


def test_publication_figure_skill_rebuilds_sparse_primary_bundle_when_source_tables_exist(
    ra,
    tmp_path: Path,
):
    from PIL import Image

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    outputs = run_dir / "steps" / "03_primary_results_figure" / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    svg = outputs / "primary_results_figure.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="140">'
        '<rect width="240" height="140" fill="white"/>'
        '<text x="16" y="32">Sparse primary figure</text>'
        "</svg>",
        encoding="utf-8",
    )
    png = outputs / "primary_results_figure.png"
    Image.new("RGB", (240, 140), "white").save(png)
    sparse_source = outputs / "primary_results_figure_source_data.csv"
    sparse_source.write_text("term,value\nprimary,1.05\n", encoding="utf-8")
    sparse_contract = make_figure_contract(
        figure_id="primary_results_figure",
        core_claim="Prevalence and the primary adjusted estimate are shown.",
        panels=[
            {
                "panel_id": "A",
                "title": "Prevalence and absolute outcome risk",
                "role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Exposure prevalence and absolute outcome risk are shown.",
            },
            {
                "panel_id": "B",
                "title": "Primary adjusted association",
                "role": "primary_estimand",
                "chart_type": "dot_interval",
                "claim": "The adjusted odds ratio and interval are shown.",
            },
        ],
        source_data=[sparse_source.name],
    )
    sparse_contract_path = outputs / "primary_results_figure.figure_contract.json"
    sparse_contract_path.write_text(sparse_contract.to_json(indent=2), encoding="utf-8")
    metadata = {
        "figure_role": "publication_figure",
        "step_id": "03_primary_results_figure",
        "generation_mode": "fallback",
    }
    for path, kind, evidence_id in (
        (svg, "figure", "figure_sparse_primary_svg"),
        (png, "figure", "figure_sparse_primary_png"),
        (sparse_contract_path, "log", "log_sparse_primary_contract"),
        (sparse_source, "table", "table_sparse_primary_source"),
    ):
        evidence.register_file(
            kind=kind,
            description="Registered sparse primary bundle.",
            source_path=path,
            evidence_id=evidence_id,
            producer="runner",
            generation_mode="fallback",
            metadata=metadata if kind != "table" else {"step_id": metadata["step_id"]},
        )

    primary_table = tmp_path / "adjusted_association.csv"
    primary_table.write_text(
        "term,odds_ratio,ci_low,ci_high\nsepsis3,1.05,0.99,1.10\n",
        encoding="utf-8",
    )
    outcome_table = tmp_path / "outcome_by_exposure.csv"
    outcome_table.write_text(
        "exposure_label,n,outcome_rate\nUnexposed,680,0.085\nExposed,320,0.122\n",
        encoding="utf-8",
    )
    missingness_table = tmp_path / "missingness.csv"
    missingness_table.write_text(
        "variable,missing_fraction\nlactate,0.43\ncreatinine,0.02\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="Primary adjusted association table.",
        source_path=primary_table,
        evidence_id="primary_association_table",
        aliases=["primary_association"],
        produced_by_step="02_primary_association",
    )
    evidence.register_file(
        kind="table",
        description="Observed outcome by exposure group.",
        source_path=outcome_table,
        evidence_id="outcome_by_exposure",
        produced_by_step="02_primary_association",
    )
    evidence.register_file(
        kind="table",
        description="Feature missingness table.",
        source_path=missingness_table,
        evidence_id="missingness",
    )
    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="02_primary_association",
                intent="Estimate the primary adjusted association.",
                expected_outputs=[
                    "table:primary_association",
                    "table:outcome_by_exposure",
                ],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_primary_results_figure",
                intent="Render the primary manuscript figure.",
                inputs=["table:primary_association"],
                expected_outputs=["figure:publication_figure"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["generation_mode"] == "primary_association_publication_figure"
    contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
        "data_quality",
    ]
    from PIL import Image

    image = Image.open(
        run_dir / "publication_figures" / "easyicu_publication_figure.png"
    )
    assert image.height / image.width < 1.2


def test_publication_figure_skill_uses_auxiliary_outcome_incidence_for_main_figure(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    primary_table = tmp_path / "adjusted_association_estimates.csv"
    primary_table.write_text(
        "term,odds_ratio,ci_low,ci_high\nexposure,1.61,1.54,1.68\n",
        encoding="utf-8",
    )
    outcome_table = tmp_path / "outcome_incidence.csv"
    outcome_table.write_text(
        "stratum,numerator,denominator,incidence_proportion,incidence_pct\n"
        "overall,95,1000,0.095,9.5\n"
        "exposure_status_0,50,700,0.0714286,7.14286\n"
        "exposure_status_1,45,300,0.15,15.0\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="Primary adjusted association table.",
        source_path=primary_table,
        evidence_id="adjusted_association_estimates",
        aliases=["primary_association"],
        produced_by_step="07_adjusted_association_model",
    )
    evidence.register_file(
        kind="table",
        description="Observed outcome incidence by exposure group.",
        source_path=outcome_table,
        evidence_id="outcome_incidence",
        produced_by_step="04_outcome_incidence",
    )
    context = ra.ResearchContext(
        research_question="Estimate an adjusted exposure-outcome association.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=1000,
            n_stays=1000,
        ),
        variables=[],
        target_outcome="outcome",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="04_outcome_incidence",
                intent="Report observed outcome incidence by exposure.",
                expected_outputs=["table:outcome_incidence"],
                planned_analysis_role="auxiliary",
            ),
            ra.AnalysisStep(
                step_id="07_adjusted_association_model",
                intent="Estimate the primary adjusted association.",
                expected_outputs=["table:adjusted_association_estimates"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="08_publication_figure",
                intent="Render the primary manuscript figure.",
                inputs=["table:adjusted_association_estimates"],
                expected_outputs=["figure:publication_figure"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
    ]
    descriptive = contract["panels"][1]
    assert descriptive["metadata"]["chart_type"] == "event_rate_panel"
    assert descriptive["evidence_ids"] == ["outcome_incidence"]


def test_publication_figure_skill_uses_distribution_and_labels_unadjusted_model(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    primary_table = tmp_path / "adjusted_association_estimates.csv"
    primary_table.write_text(
        "exposure,outcome,covariates,estimate,ci_low,ci_high,effect_scale\n"
        "exposure,death,,1.70,1.50,1.92,odds_ratio\n",
        encoding="utf-8",
    )
    distribution = tmp_path / "exposure_outcome_distribution.csv"
    distribution.write_text(
        "row_role,exposure_level,n_rows,outcome_rate_pct\n"
        "exposure_level,0,700,8.0\n"
        "exposure_level,1,300,14.0\n"
        "overall,,1000,9.8\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="Primary unadjusted association table.",
        source_path=primary_table,
        evidence_id="adjusted_association_estimates",
        produced_by_step="primary",
    )
    evidence.register_file(
        kind="table",
        description="Observed outcome distribution by exposure.",
        source_path=distribution,
        evidence_id="exposure_outcome_distribution",
        produced_by_step="distribution",
    )
    context = ra.ResearchContext(
        research_question="Estimate an unadjusted association.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=1000,
            n_stays=1000,
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="distribution",
                intent="Report absolute outcome risk by exposure.",
                expected_outputs=["table:exposure_outcome_distribution"],
                planned_analysis_role="auxiliary",
            ),
            ra.AnalysisStep(
                step_id="primary",
                intent="Estimate the primary unadjusted association.",
                expected_outputs=["table:adjusted_association_estimates"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="figure",
                intent="Render the primary figure.",
                inputs=["table:adjusted_association_estimates"],
                expected_outputs=["figure:publication_figure"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["title"] == "Unadjusted association"
    assert contract["panels"][1]["role"] == "descriptive_result"
    assert contract["panels"][1]["evidence_ids"] == [
        "exposure_outcome_distribution"
    ]


def test_association_figure_semantics_come_from_executed_result_fields() -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.figures.skill import (
        _association_adjustment_state,
        _sole_source_landmark_hours,
        _sole_source_semantic,
    )

    adjusted = pd.DataFrame(
        {
            "is_reference": [True, False],
            "exposure": ["sofa", "sofa"],
            "outcome": ["death", "death"],
            "adjustment_covariates": ["age;sex", ""],
            "landmark_hours": [24.0, 48.0],
        }
    )
    assert _association_adjustment_state(
        adjusted,
        primary_requirements=[SimpleNamespace(covariates=["age", "sex"])],
    ) is True
    assert _sole_source_semantic(adjusted, "outcome") == "death"
    assert _sole_source_landmark_hours(adjusted) == 24.0

    with pytest.raises(ValueError, match="conflicts with the typed primary Plan"):
        _association_adjustment_state(
            adjusted,
            primary_requirements=[SimpleNamespace(covariates=[])],
        )

    undeclared = pd.DataFrame({"estimate": [1.4]})
    assert _association_adjustment_state(
        undeclared,
        primary_requirements=[],
    ) is None


def test_distribution_overall_audit_row_is_not_drawn_as_a_third_exposure_group():
    from easyicu.research_agent.figures.skill import _normalise_strata_frame

    frame = pd.DataFrame(
        {
            "row_role": ["exposure_level", "exposure_level", "overall"],
            "exposure_level": [0, 1, None],
            "n_rows": [700, 300, 1000],
            "outcome_rate_pct": [8.0, 14.0, 9.8],
        }
    )

    normalized = _normalise_strata_frame(frame)

    assert normalized["score"].tolist() == ["Unexposed", "Exposed"]
    assert normalized["n"].tolist() == [700, 300]


def _e1_typed_distribution_frame(*, include_risk_difference: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_role": ["exposure_level", "exposure_level", "overall"],
            "exposure_level_index": [0, 1, None],
            "exposure_level": [0, 1, None],
            "n_rows": [62_900, 31_600, 94_500],
            # Deliberately differ from n_rows: Panel B must use the analysed
            # outcome denominator, not whichever structural count comes first.
            "outcome_denominator": [62_862, 31_596, 94_458],
            "outcome_rate_pct": [8.210047, 13.625142, 10.021385],
            "ci_low_pct": [7.995860, 13.245177, 9.830581],
            "ci_high_pct": [8.424235, 14.005108, 10.212189],
            "exposure_column": ["sep3_sofa1_max"] * 3,
        }
    )
    if include_risk_difference:
        frame = frame.assign(
            risk_difference_pct=5.415095,
            risk_difference_ci_low_pct=4.977273,
            risk_difference_ci_high_pct=5.852917,
            risk_difference_reference_index=0,
            risk_difference_comparison_index=1,
            risk_difference_effect_measure="risk_difference",
        )
    return frame


def test_e1_typed_distribution_uses_authorized_risk_difference_not_level_codes(ra):
    from easyicu.research_agent.figures.skill import (
        _association_axis_metadata,
        _normalise_association_frame,
    )

    normalized = _normalise_association_frame(
        _e1_typed_distribution_frame(include_risk_difference=True),
        primary_exposure="sep3_sofa1_max",
        display_labels={"sep3_sofa1_max": "Sepsis-3"},
    )

    assert normalized["estimate"].tolist() == pytest.approx([5.415095])
    assert normalized["lower"].tolist() == pytest.approx([4.977273])
    assert normalized["upper"].tolist() == pytest.approx([5.852917])
    assert normalized["label"].tolist() == ["Sepsis-3: Exposed vs Unexposed"]
    assert not set(normalized["estimate"]) & {0.0, 1.0}
    axis = _association_axis_metadata(normalized)
    assert axis["null_value"] == 0.0
    assert axis["ratio_scale"] is False
    assert axis["xlabel"] == "Risk difference (percentage points)"


def test_binary_level_display_labels_reach_adjusted_and_distribution_panels(ra):
    from easyicu.research_agent.figures.skill import (
        _normalise_association_frame,
        _normalise_strata_frame,
        _strata_score_label,
    )

    labels = {
        "marker_a=0": "Marker A absent",
        "marker_a=1": "Marker A present",
    }
    adjusted = _normalise_association_frame(
        pd.DataFrame(
            {
                "exposure": ["marker_a"],
                "estimate": [1.4],
                "ci_low": [1.2],
                "ci_high": [1.6],
            }
        ),
        primary_exposure="marker_a",
        display_labels=labels,
    )
    distribution = _e1_typed_distribution_frame(
        include_risk_difference=True
    ).assign(exposure_column="marker_a")
    strata = _normalise_strata_frame(distribution, display_labels=labels)

    assert adjusted["label"].tolist() == [
        "Marker A present vs Marker A absent"
    ]
    assert strata["score"].tolist() == ["Marker A absent", "Marker A present"]
    assert _strata_score_label(strata) == "Marker A status"


def test_e1_typed_distribution_without_authorized_contrast_fails_closed(ra):
    from easyicu.research_agent.figures.skill import _normalise_association_frame

    incomplete_typed = _normalise_association_frame(
        _e1_typed_distribution_frame(include_risk_difference=False),
        primary_exposure="sep3_sofa1_max",
    )
    arbitrary_numeric = _normalise_association_frame(
        pd.DataFrame(
            {
                "exposure_level": [0, 1],
                "n_rows": [700, 300],
                "outcome_rate_pct": [8.0, 14.0],
            }
        )
    )

    assert incomplete_typed.empty
    assert arbitrary_numeric.empty


def test_association_frame_rejects_context_rates_and_labels_continuous_contrasts(
    ra,
):
    from easyicu.research_agent.figures.skill import _normalise_association_frame

    context_rates = _normalise_association_frame(
        pd.DataFrame(
            {
                "exposure": ["marker", "marker"],
                "estimate_type": ["prevalence", "outcome_risk"],
                "estimate": [0.54, 0.14],
                "ci_low": [0.53, 0.13],
                "ci_high": [0.55, 0.15],
            }
        )
    )
    contrasts = _normalise_association_frame(
        pd.DataFrame(
            {
                "biomarker_mmol_l": [1.0, 5.0],
                "reference_biomarker_mmol_l": [2.1, 2.1],
                "adjusted_odds_ratio": [0.76, 1.96],
                "ci_low": [0.72, 1.89],
                "ci_high": [0.81, 2.03],
            }
        )
    )

    assert context_rates.empty
    assert contrasts["label"].tolist() == [
        "Biomarker Mmol L: 1 vs 2.1",
        "Biomarker Mmol L: 5 vs 2.1",
    ]
    assert contrasts.attrs["xlabel"] == "Odds ratio"


def test_counts_only_descriptive_bundle_is_promoted_without_invented_contrast(
    ra,
    tmp_path: Path,
):
    from PIL import Image

    from easyicu.research_agent.contracts.figure_plan import (
        EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS,
    )
    from easyicu.research_agent.planning.study_design import (
        infer_study_design_family,
    )

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    primary_step_id = "primary_exposure_outcome_distribution"
    figure_step_id = "05_primary_result_figure"

    distribution = tmp_path / "exposure_outcome_distribution.csv"
    counts_only = _e1_typed_distribution_frame(
        include_risk_difference=False
    ).assign(
        ci_low_pct=pd.NA,
        ci_high_pct=pd.NA,
    )
    counts_only.to_csv(distribution, index=False)
    evidence.register_file(
        kind="table",
        description="Counts-only E1 exposure/outcome distribution.",
        source_path=distribution,
        evidence_id="table_step_artifact_distribution",
        produced_by_step=primary_step_id,
        producer="runner",
        generation_mode="deterministic_standard",
    )

    outputs = run_dir / "steps" / figure_step_id / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    svg = outputs / "primary_result.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="340">'
        '<rect width="720" height="340" fill="white"/>'
        '<text x="20" y="30">Counts-only descriptive result</text>'
        "</svg>",
        encoding="utf-8",
    )
    png = outputs / "primary_result.png"
    Image.new("RGB", (720, 340), "white").save(png)
    source = outputs / "primary_result_input_source_data.csv"
    counts_only.to_csv(source, index=False)
    templates = EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS
    contract = make_figure_contract(
        figure_id="figure:primary_result",
        core_claim=(
            "Exposure prevalence and observed outcome rates are reproduced "
            "from one registered counts-only distribution table."
        ),
        panels=[
            {
                "panel_id": template.panel_id,
                "title": (
                    "Exposure distribution"
                    if index == 0
                    else "Outcome rate by exposure"
                ),
                "role": template.article_role,
                "chart_type": template.chart_type,
                "claim": "Observed counts and denominators are displayed.",
                "evidence_ids": [source.name],
            }
            for index, template in enumerate(templates)
        ],
        source_data=[source.name],
        statistics_note=(
            "Counts and observed percentages only; no uncertainty or contrast "
            "is computed."
        ),
    )
    contract_path = outputs / "primary_result.figure_contract.json"
    contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")
    metadata = {
        "figure_role": "publication_figure",
        "step_id": figure_step_id,
    }
    for path, kind, evidence_id in (
        (svg, "figure", "figure_primary_result_svg"),
        (png, "figure", "figure_primary_result_png"),
        (contract_path, "log", "log_primary_result_contract"),
        (source, "table", "table_primary_result_source"),
    ):
        evidence.register_file(
            kind=kind,
            description="Registered counts-only E1 result bundle.",
            source_path=path,
            evidence_id=evidence_id,
            produced_by_step=figure_step_id,
            producer="runner",
            generation_mode="deterministic_standard",
            metadata=metadata if kind != "table" else {"step_id": figure_step_id},
        )

    context = ra.ResearchContext(
        research_question=(
            "Describe exposure prevalence and observed outcome rates across "
            "the declared exposure levels."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="MIMIC-IV ICU stays",
            database="mimiciv",
            n_patients=65_366,
            n_stays=94_458,
        ),
        variables=[],
        primary_exposure="sep3_sofa2_max",
        target_outcome="death",
    )
    assert infer_study_design_family(context) == "descriptive"
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id=primary_step_id,
                intent="Report the declared counts-only distribution.",
                expected_outputs=["table:exposure_outcome_distribution"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id=figure_step_id,
                intent="Render the primary descriptive result.",
                inputs=["table:exposure_outcome_distribution"],
                expected_outputs=["figure:primary_result"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    from easyicu.research_agent.figures.skill import (
        _PrimaryLineageEvidenceView,
        _bundle_primary_strategy_ready,
        _declared_primary_lineage_step_ids,
        _select_existing_step_publication_figure_bundle,
    )

    lineage = _PrimaryLineageEvidenceView(
        evidence,
        _declared_primary_lineage_step_ids(plan),
    )
    bundle = _select_existing_step_publication_figure_bundle(lineage)
    assert bundle is not None
    assert _bundle_primary_strategy_ready(context, bundle), bundle[
        "contract_payload"
    ]

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    assert not (
        run_dir
        / "publication_figures"
        / "publication_figure_source_primary_association.csv"
    ).exists()
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["generation_mode"] == "promoted_step_publication_figure"
    promoted = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in promoted["panels"]] == [
        "distribution",
        "descriptive_result",
    ]
    assert "no uncertainty or contrast is computed" in promoted["statistics_note"]


def test_e1_typed_distribution_panel_b_uses_rate_ci_and_outcome_denominator(ra):
    from easyicu.research_agent.figures.skill import _draw_strata_panel
    from easyicu.research_agent.figures.strata import normalise_strata_frame

    normalized = normalise_strata_frame(
        _e1_typed_distribution_frame(include_risk_difference=True)
    )

    assert normalized["score"].tolist() == ["Unexposed", "Exposed"]
    assert normalized["rate"].tolist() == pytest.approx([0.08210047, 0.13625142])
    assert normalized["lower"].tolist() == pytest.approx([0.07995860, 0.13245177])
    assert normalized["upper"].tolist() == pytest.approx([0.08424235, 0.14005108])
    assert normalized["n"].tolist() == [62_862, 31_596]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_strata_panel(
        ax,
        normalized,
        palette={"blue": "#0F4D92", "neutral_light": "#D8D8D8"},
        outcome_label="In-hospital mortality",
    )
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["Unexposed (n=62,862)", "Exposed (n=31,596)"]
    # One collection comes from the lollipop stems and another from the
    # horizontal confidence intervals.  A point-only panel has only the former.
    assert len(ax.collections) >= 2
    plt.close(fig)


def test_publication_figure_skill_e1_typed_distribution_projects_both_panels(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    distribution = tmp_path / "exposure_outcome_distribution.csv"
    _e1_typed_distribution_frame(include_risk_difference=True).to_csv(
        distribution,
        index=False,
    )
    evidence.register_file(
        kind="table",
        description="E1 typed exposure/outcome distribution.",
        source_path=distribution,
        evidence_id="exposure_outcome_distribution",
        produced_by_step="03_distribution",
    )
    missingness = tmp_path / "measurement_missingness.csv"
    missingness.write_text(
        "variable,missing_fraction\nage,0.0\ndeath,0.0\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="E1 measurement missingness.",
        source_path=missingness,
        evidence_id="measurement_missingness",
        produced_by_step="02_table_one",
    )
    context = ra.ResearchContext(
        research_question="Describe Sepsis-3 and in-hospital mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="MIMIC-IV ICU stays",
            database="mimiciv",
            n_patients=65_366,
            n_stays=94_458,
        ),
        variables=[],
        primary_exposure="sep3_sofa1_max",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        display_labels={
            "sep3_sofa1_max=0": "Sepsis-3 absent",
            "sep3_sofa1_max=1": "Sepsis-3 present",
            "age": "ICU入科时年龄（岁）",
            "death": "院内死亡",
        },
        steps=[
            ra.AnalysisStep(
                step_id="03_distribution",
                intent="Estimate the prespecified descriptive contrast.",
                expected_outputs=["table:exposure_outcome_distribution"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="04_figure",
                intent="Render the publication figure.",
                inputs=["table:exposure_outcome_distribution"],
                expected_outputs=["figure:publication_figure"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    assert result.findings == []
    primary_source = pd.read_csv(
        run_dir
        / "publication_figures"
        / "publication_figure_source_primary_association.csv"
    )
    strata_source = pd.read_csv(
        run_dir
        / "publication_figures"
        / "publication_figure_source_stratified_outcome.csv"
    )
    assert primary_source["estimate"].tolist() == pytest.approx([5.415095])
    assert primary_source["label"].tolist() == [
        "Sepsis-3 present vs Sepsis-3 absent"
    ]
    assert not set(primary_source["estimate"]) & {0.0, 1.0}
    assert strata_source["rate"].tolist() == pytest.approx([0.08210047, 0.13625142])
    assert strata_source["n"].tolist() == [62_862, 31_596]
    assert strata_source["score"].tolist() == [
        "Sepsis-3 absent",
        "Sepsis-3 present",
    ]
    contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["title"] == "Unadjusted risk difference"
    assert "percentage points" in contract["panels"][0]["claim"]
    assert "analysed denominators" in contract["panels"][1]["claim"]
    assert "relative estimate" not in json.dumps(contract)
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
        "data_quality",
    ]


def test_figure_contract_enforces_unique_panel_ids():
    with pytest.raises(ValueError):
        make_figure_contract(
            figure_id="Figure2",
            core_claim="SOFA2 zero is an audit target.",
            panels=[
                {"panel_id": "a", "title": "one", "role": "overview", "claim": "x"},
                {"panel_id": "a", "title": "two", "role": "audit", "claim": "y"},
            ],
        )


def test_contract_audit_flags_missing_evidence_and_duplicate_roles():
    contract = make_figure_contract(
        figure_id="Figure2",
        core_claim="SOFA2 zero is an audit target.",
        panels=[
            PanelSpec(panel_id="a", title="Overview", role="overview", claim="x"),
            PanelSpec(
                panel_id="b", title="Another overview", role="overview", claim="y"
            ),
        ],
    )

    findings = audit_figure_contract(contract)
    messages = " ".join(f.message for f in findings)
    assert "repeats panel role" in messages
    assert "without evidence ids" in messages


def test_make_figure_contract_accepts_agent_style_aliases():
    contract = make_figure_contract(
        {
            "figure_id": "STEP07_claim_first_multipanel",
            "title": "Claim-first EasyICU publication figure",
            "claim": "Mortality rises across SOFA-2 while missingness must remain explicit.",
            "panels": [
                {
                    "panel": "a",
                    "title": "Outcome",
                    "claim": "Mortality is quantified with confidence intervals.",
                    "source_evidence": "figure_source_cohort_summary.csv",
                },
                {
                    "panel": "b",
                    "title": "Missingness audit",
                    "claim": "Missingness is reported instead of silently imputed.",
                    "source_evidence": "figure_source_missingness.csv",
                },
            ],
            "source_evidence": {
                "cohort_summary": "figure_source_cohort_summary.csv",
                "missingness": "figure_source_missingness.csv",
            },
            "statistical_notes": [
                "Ordinal scores are not averaged.",
                "Missingness is explicit.",
            ],
            "target_outcome": "death",
            "cohort": "miiv_crossdb_cohort",
        }
    )

    assert contract.figure_id == "STEP07_claim_first_multipanel"
    assert contract.core_claim.startswith("Mortality rises")
    assert [p.panel_id for p in contract.panels] == ["a", "b"]
    assert contract.panels[0].evidence_ids == ["figure_source_cohort_summary.csv"]
    assert contract.panels[1].role == "audit"
    assert contract.source_data == [
        "figure_source_cohort_summary.csv",
        "figure_source_missingness.csv",
    ]
    assert "Ordinal scores are not averaged." in (contract.statistics_note or "")


def test_make_figure_contract_accepts_deferred_agent_panel_append(tmp_path: Path):
    contract = make_figure_contract(
        figure_id="fig_trajectory_clustering",
        core_claim="ICU shock physiology clusters have distinct outcomes.",
    )
    contract["panels"].append(
        {
            "panel_id": "A",
            "title": "Cluster Profiles",
            "role": "profile_plot",
            "claim": "Profiles differ across shock physiology clusters.",
            "evidence_ids": ["cluster_profile_data"],
        }
    )

    findings = audit_figure_contract(contract)
    assert not any(f.severity == "error" for f in findings)
    saved = save_publication_figure(contract, tmp_path)

    assert saved["contract"].exists()
    assert contract.panels[0].panel_id == "A"
    assert contract.panels[0].role == "overview"


def test_make_figure_contract_accepts_agent_role_aliases_and_source_dicts():
    contract = make_figure_contract(
        figure_id="FigureAgent",
        core_claim="Severity and physiology both matter.",
        panels=[
            {
                "panel_id": "A",
                "title": "Anchor",
                "role": "cohort_anchor",
                "claim": "Outcome is defined.",
                "evidence_ids": ["src_incidence"],
            },
            {
                "panel_id": "B",
                "title": "Association forest",
                "role": "association_forest",
                "claim": "Associations are complete-case only.",
                "evidence_ids": ["src_assoc"],
            },
        ],
        source_data=[
            {"evidence_id": "src_incidence", "path": "incidence.csv"},
            {"evidence_id": "src_assoc", "path": "assoc.csv"},
        ],
    )

    assert [p.role for p in contract.panels] == ["overview", "robustness"]
    assert contract.source_data == ["incidence.csv", "assoc.csv"]


def test_make_figure_contract_preserves_article_level_panel_roles():
    contract = make_figure_contract(
        figure_id="FigureArticleRoles",
        core_claim="Prevalence, absolute risk, adjusted effect, and data quality are visible.",
        panels=[
            {
                "panel_id": "A",
                "title": "Absolute outcome risk",
                "role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Exposure prevalence and absolute outcome risk are shown.",
                "evidence_ids": ["absolute_risk_source"],
            },
            {
                "panel_id": "B",
                "title": "Adjusted effect",
                "role": "primary_estimand",
                "chart_type": "forest",
                "claim": "The adjusted effect estimate is shown with uncertainty.",
                "evidence_ids": ["primary_model_source"],
            },
            {
                "panel_id": "C",
                "title": "Measurement availability",
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Measurement availability is shown for the analytic denominator.",
                "evidence_ids": ["missingness_source"],
            },
        ],
    )

    assert [panel.role for panel in contract.panels] == [
        "descriptive_result",
        "primary_estimand",
        "data_quality",
    ]
    assert contract.panels[0].metadata["chart_type"] == "dot_interval_absolute_risk"


def test_make_figure_contract_accepts_legacy_positional_signature():
    contract = make_figure_contract(
        "missingness_summary",
        "Missingness Summary",
        [{"variable": "vaso", "missing_pct": 77.5}],
        "Variable",
        "Missingness (%)",
    )

    assert contract.figure_id == "missingness_summary"
    assert contract.core_claim == "Missingness Summary"
    assert len(contract.panels) == 1
    assert contract.panels[0].title == "Missingness Summary"


def test_make_figure_contract_wraps_string_source_data():
    contract = make_figure_contract(
        figure_id="FigureStringSource",
        core_claim="Single-string source data should stay a single entry.",
        panels=[
            {
                "panel_id": "a",
                "title": "Overview",
                "role": "main",
                "claim": "Overview exists.",
            }
        ],
        source_data="Cohort Data",
    )

    assert contract.source_data == ["Cohort Data"]


def test_figure_contract_defaults_include_tiff():
    contract = make_figure_contract(
        figure_id="FigureDefault",
        core_claim="Default export bundle should include TIFF.",
        panels=[
            {
                "panel_id": "a",
                "title": "Overview",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["src"],
            }
        ],
    )

    assert contract.export_formats == ["svg", "pdf", "png", "tiff"]


def test_publication_export_keeps_svg_text_editable(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("SOFA-2")
    ax.set_ylabel("Mortality")
    ax.set_title("Editable text")

    contract = make_figure_contract(
        figure_id="FigureTest",
        core_claim="A simple line can be exported with editable SVG text.",
        panels=[
            {
                "panel_id": "a",
                "title": "Line",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["statistic_step_summary"],
            }
        ],
        export_formats=["svg", "pdf", "png"],
    )
    paths = save_publication_figure(
        fig, tmp_path / "figure_test", contract=contract, dpi=150
    )
    plt.close(fig)

    assert {"svg", "pdf", "png", "contract"} <= set(paths)
    svg = paths["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    assert paths["contract"].exists()
    assert audit_publication_exports(paths) == []


def test_apply_publication_style_accepts_legacy_fig_argument():
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, _ax = plt.subplots(figsize=(2.5, 1.8))
    palette = apply_publication_style(fig)
    plt.close(fig)
    assert "blue" in palette


def test_publication_export_caps_and_compresses_tiff(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.plot([0, 1, 2, 3], [0.1, 0.4, 0.2, 0.7], linewidth=1.5)
    ax.set_xlabel("SOFA-2")
    ax.set_ylabel("Death risk")
    ax.set_title("Compressed TIFF export")

    paths = save_publication_figure(
        fig, tmp_path / "figure_tiff", formats=["tiff"], dpi=600
    )
    plt.close(fig)

    assert paths["tiff"].exists()
    assert paths["tiff"].stat().st_size < 8_000_000


def test_svg_audit_flags_pathified_text(tmp_path: Path):
    svg = tmp_path / "bad.svg"
    svg.write_text("<svg><path d='M0 0L1 1'/></svg>", encoding="utf-8")
    findings = audit_publication_exports([svg], min_bytes=1)
    assert any("editable <text>" in f.message for f in findings)


def test_publication_export_audit_flags_svg_text_overlap(tmp_path: Path):
    svg = tmp_path / "overlap.svg"
    svg.write_text(
        """
        <svg width="220pt" height="160pt" viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg">
          <rect width="220" height="160" fill="white"/>
          <g id="title_a">
            <text x="80" y="50" style="font-size: 15px; text-anchor: middle">Primary association</text>
          </g>
          <g id="title_b">
            <text x="84" y="52" style="font-size: 15px; text-anchor: middle">Ascertainment audit</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    findings = audit_publication_exports([svg], min_bytes=1)

    assert any("overlapping text" in f.message for f in findings)


def test_publication_export_writes_pptx(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    paths = save_publication_figure(fig, tmp_path / "figure_pptx", formats=["pptx"])
    plt.close(fig)

    pptx = paths["pptx"]
    assert pptx.exists()
    with zipfile.ZipFile(pptx) as z:
        names = set(z.namelist())
    assert "ppt/slides/slide1.xml" in names
    assert "ppt/media/image1.png" in names


def test_publication_export_audit_accepts_output_dir_and_stem(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Audit path")
    paths = save_publication_figure(
        fig, tmp_path / "figure_audit", formats=["svg", "png"]
    )
    plt.close(fig)

    findings = audit_publication_exports(tmp_path, min_bytes=1)
    assert findings == []
    assert paths["svg"].exists()


def test_save_publication_figure_accepts_legacy_contract_and_output_dir_call(
    tmp_path: Path,
):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [1, 0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contract = make_figure_contract(
        figure_id="FigureLegacy",
        core_claim="Legacy call path still exports correctly.",
        panels=[
            {
                "panel_id": "a",
                "title": "Panel",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["src"],
            }
        ],
    )

    paths = save_publication_figure(
        fig,
        contract,
        tmp_path,
        "legacy_figure",
        formats=["svg", "png"],
        dpi=150,
    )
    plt.close(fig)

    assert {"svg", "png", "contract"} <= set(paths)
    assert paths["svg"].name == "legacy_figure.svg"


def test_save_publication_figure_accepts_agent_output_dir_name_kwargs(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [1, 0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contract = make_figure_contract(
        figure_id="sofa_mortality_by_stratum",
        core_claim="Legacy agent call should write into STEP_OUT_DIR.",
        panels=[
            {
                "panel_id": "a",
                "title": "Panel",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["src"],
            }
        ],
    )

    paths = save_publication_figure(
        fig,
        out_dir,
        contract,
        png_name="sofa_mortality_by_stratum.png",
        svg_name="sofa_mortality_by_stratum.svg",
        formats=["svg", "png"],
        dpi=150,
    )
    plt.close(fig)

    assert paths["svg"] == out_dir / "sofa_mortality_by_stratum.svg"
    assert paths["png"] == out_dir / "sofa_mortality_by_stratum.png"
    assert (
        paths["contract"] == out_dir / "sofa_mortality_by_stratum.figure_contract.json"
    )
    assert audit_publication_exports(out_dir, min_bytes=1) == []


def test_save_publication_figure_accepts_contract_only_output_dir_call(tmp_path: Path):
    contract = make_figure_contract(
        figure_id="FigureContractOnly",
        core_claim="Contract-only save should still persist contract JSON.",
        panels=[
            {
                "panel_id": "a",
                "title": "Overview",
                "role": "overview",
                "claim": "Line exists.",
            }
        ],
    )

    paths = save_publication_figure(contract, tmp_path)

    assert "contract" in paths
    assert paths["contract"].exists()


def test_runner_synthesizes_contract_for_step_figure_exports(tmp_path: Path):
    from easyicu.research_agent.audits.validators import FigureContractQualityValidator
    from easyicu.research_agent.execution.phase import _ensure_step_figure_contract
    from easyicu.research_agent.schema import AnalysisStep

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    png = out_dir / "missingness_measurement_panel.png"
    svg = out_dir / "missingness_measurement_panel.svg"
    source = out_dir / "missingness_measurement_panel_source_data.csv"
    png.write_bytes(b"not-a-real-png-but-present")
    svg.write_text("<svg><text>ok</text></svg>", encoding="utf-8")
    source.write_text("variable,missing_pct\nlactate,40.7\n", encoding="utf-8")
    step = AnalysisStep(
        step_id="02_baseline_characteristics_and_data_quality_figure",
        intent="Render missingness and measurement quality for manuscript review.",
        expected_outputs=["figure:missingness_measurement_panel"],
        method="matplotlib",
    )
    summary = {
        "figure_files": [str(png), str(svg)],
        "source_data_files": [str(source)],
    }

    contract_path = _ensure_step_figure_contract(
        step=step,
        out_dir=out_dir,
        step_summary=summary,
        evidence_ids=["table_missingness_source"],
    )

    assert (
        contract_path == out_dir / "missingness_measurement_panel.figure_contract.json"
    )
    findings = FigureContractQualityValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=tmp_path,
        step_summary=summary,
    )
    assert [finding for finding in findings if finding.severity == "error"] == []


def test_audit_publication_exports_returns_plain_findings_list(tmp_path: Path):
    svg = tmp_path / "ok.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="10">ok</text></svg>',
        encoding="utf-8",
    )
    findings = audit_publication_exports([svg], min_bytes=1)
    assert isinstance(findings, list)
    with pytest.raises(TypeError):
        findings["figure_contract"] = {"figure_id": "Figure1"}


def test_publication_figure_skill_renders_from_registered_association_table(
    ra, tmp_path: Path
):
    from easyicu.research_agent.discovery.discovery_package import _figure_inventory

    run_dir = tmp_path / "run"
    source = tmp_path / "primary_association.csv"
    pd.DataFrame(
        {
            "variable": ["lactate", "MAP"],
            "odds_ratio": [1.35, 0.82],
            "or_lower": [1.10, 0.70],
            "or_upper": [1.66, 0.96],
        }
    ).to_csv(source, index=False)

    evidence = ra.EvidenceStore(run_dir)
    evidence.register_file(
        kind="table",
        description="Primary association table.",
        source_path=source,
        evidence_id="primary_association",
        aliases=["primary_association_table"],
        produced_by_step="04_primary_association",
    )
    context = ra.ResearchContext(
        research_question="Are hemodynamic variables associated with mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=2,
            n_stays=2,
        ),
        variables=[
            ra.ConceptDescriptor(name="lactate", role="lab", dtype="float64"),
            ra.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="04_primary_association",
                intent="Estimate primary associations.",
                inputs=["lactate", "death"],
                expected_outputs=[
                    "table:primary_association",
                    "figure:primary_association_curve",
                ],
                planned_analysis_role="primary",
            )
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
    )

    assert result.generated is True
    assert evidence.get("publication_figure") is not None
    assert evidence.get("publication_figure_contract") is not None
    assert evidence.get("publication_figure_skill_summary") is not None
    assert (run_dir / "publication_figures" / "easyicu_publication_figure.svg").exists()
    inventory = _figure_inventory(run_dir)
    assert len(inventory) == 1
    assert inventory[0].contract_registered is True
    assert inventory[0].provenance_valid is True


def test_publication_figure_refuses_digest_stale_association_table(
    ra, tmp_path: Path
):
    run_dir = tmp_path / "run"
    source = tmp_path / "primary_association.csv"
    pd.DataFrame(
        {
            "variable": ["lactate"],
            "odds_ratio": [1.35],
            "or_lower": [1.10],
            "or_upper": [1.66],
        }
    ).to_csv(source, index=False)
    evidence = ra.EvidenceStore(run_dir)
    record = evidence.register_file(
        kind="table",
        description="Primary association table.",
        source_path=source,
        evidence_id="primary_association",
        produced_by_step="04_primary_association",
    )
    context = ra.ResearchContext(
        research_question="Is lactate associated with mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=2,
            n_stays=2,
        ),
        variables=[
            ra.ConceptDescriptor(name="lactate", role="lab", dtype="float64"),
            ra.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
        primary_exposure="lactate",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="04_primary_association",
                intent="Estimate the primary association.",
                inputs=["lactate", "death"],
                expected_outputs=[
                    "table:primary_association",
                    "figure:primary_association_curve",
                ],
                planned_analysis_role="primary",
            )
        ],
    )
    pd.DataFrame(
        {
            "variable": ["lactate"],
            "odds_ratio": [9.99],
            "or_lower": [8.0],
            "or_upper": [12.0],
        }
    ).to_csv(run_dir / record.relative_path, index=False)

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
    )

    assert result.generated is False
    assert not any(record.kind == "figure" for record in evidence.records())


def test_association_forest_axis_metadata_tracks_effect_measure(ra):
    from easyicu.research_agent.figures.skill import _normalise_association_frame

    hr_frame = _normalise_association_frame(
        pd.DataFrame(
            {
                "term": ["age"],
                "hazard_ratio": [1.22],
                "hr_lower": [1.05],
                "hr_upper": [1.41],
            }
        )
    )
    assert hr_frame.attrs["xlabel"] == "Hazard ratio"
    assert hr_frame.attrs["header"] == "HR (95% CI)"
    assert hr_frame.attrs["null_value"] == 1.0
    assert hr_frame.attrs["ratio_scale"] is True

    ate_frame = _normalise_association_frame(
        pd.DataFrame(
            {
                "term": ["treatment"],
                "average_treatment_effect": [-0.8],
                "ci_low": [-1.2],
                "ci_high": [-0.3],
            }
        )
    )
    assert ate_frame.attrs["xlabel"] == "Average treatment effect"
    assert ate_frame.attrs["header"] == "ATE (95% CI)"
    assert ate_frame.attrs["null_value"] == 0.0
    assert ate_frame.attrs["ratio_scale"] is False


def test_association_frame_filters_to_primary_exposure_and_point_estimate(ra):
    from easyicu.research_agent.figures.skill import _normalise_association_frame

    frame = _normalise_association_frame(
        pd.DataFrame(
            {
                "term": ["const", "sepsis3", "age_per_10y"],
                "point_estimate": [0.15, 1.05, 1.26],
                "ci_low": [0.15, 0.99, 1.23],
                "ci_high": [0.15, 1.10, 1.28],
            }
        ),
        primary_exposure="sepsis3",
        display_labels={"sepsis3": "Sepsis-3"},
    )

    assert frame["label"].tolist() == ["Sepsis-3"]
    assert frame["estimate"].tolist() == [1.05]


def test_publication_figure_skill_e1_like_layout_has_no_svg_overlap_errors(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    primary = tmp_path / "adjusted_association_death.csv"
    pd.DataFrame(
        {
            "exposure": ["sepsis3"],
            "point_estimate": [1.05],
            "ci_low": [0.99],
            "ci_high": [1.10],
            "effect_scale": ["adjusted odds ratio"],
        }
    ).to_csv(primary, index=False)
    strata = tmp_path / "outcome_by_exposure.csv"
    pd.DataFrame(
        {
            "sepsis3": [0, 1],
            "death_pct": [8.5, 12.2],
            "n": [46600, 28229],
        }
    ).to_csv(strata, index=False)
    missingness = tmp_path / "cohort_missingness_audit.csv"
    pd.DataFrame(
        {
            "variable": [
                "lact_mean",
                "lact_min",
                "lact_max",
                "lact_first",
                "temp_mean",
                "temp_min",
                "temp_max",
                "temp_first",
                "resp_mean",
                "resp_min",
                "resp_max",
                "resp_first",
                "sep3_sofa2_n",
            ],
            "missing_fraction": [
                0.407,
                0.407,
                0.407,
                0.407,
                0.027,
                0.027,
                0.027,
                0.027,
                0.003,
                0.003,
                0.003,
                0.003,
                0.0,
            ],
        }
    ).to_csv(missingness, index=False)
    evidence.register_file(
        kind="table",
        description="Primary adjusted association.",
        source_path=primary,
        evidence_id="table_adjusted_association_death",
        aliases=["adjusted_association_death"],
        produced_by_step="03_primary_results",
    )
    evidence.register_file(
        kind="table",
        description="Outcome by primary exposure.",
        source_path=strata,
        evidence_id="table_outcome_by_exposure",
        aliases=["outcome_by_exposure"],
        produced_by_step="03_primary_results",
    )
    evidence.register_file(
        kind="table",
        description="Cohort missingness audit.",
        source_path=missingness,
        evidence_id="table_cohort_missingness_audit",
        aliases=["cohort_missingness_audit"],
    )
    context = ra.ResearchContext(
        research_question="Is Sepsis-3 associated with ICU mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=74829,
            n_stays=74829,
        ),
        variables=[],
        target_outcome="icu_mortality",
        primary_exposure="sepsis3",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        display_labels={
            "icu_mortality": "ICU mortality",
            "sepsis3": "Sepsis-3",
            "lact": "Lactate",
            "temp": "Temperature",
            "resp": "Resp. rate",
        },
        steps=[
            ra.AnalysisStep(
                step_id="03_primary_results",
                intent="Render primary association and audit context.",
                expected_outputs=[
                    "table:adjusted_association_death",
                    "table:outcome_by_exposure",
                    "figure:publication",
                ],
                planned_analysis_role="primary",
            )
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    assert not any(
        finding.severity == "error" and "overlapping text" in finding.message
        for finding in result.findings
    )
    assert not any(
        "outside the canvas" in finding.message for finding in result.findings
    )
    source = pd.read_csv(
        run_dir / "publication_figures" / "publication_figure_source_missingness.csv"
    )
    assert source["variable"].tolist() == ["Lactate", "Temperature", "Resp. rate"]


def test_primary_association_selector_prefers_single_primary_estimand(
    ra, tmp_path: Path
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.figures.skill import _select_primary_association_record

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    full = run_dir / "adjusted_association_death_full_coefficients.csv"
    full.write_text(
        "term,estimate,ci_low,ci_high,effect_scale\n"
        "const,0.15,0.15,0.15,adjusted odds ratio\n"
        "sepsis3,1.01,0.96,1.07,adjusted odds ratio\n"
        "age,1.26,1.23,1.28,adjusted odds ratio\n",
        encoding="utf-8",
    )
    single = run_dir / "adjusted_association_death.csv"
    single.write_text(
        "exposure,point_estimate,ci_low,ci_high,effect_scale\n"
        "sepsis3,1.05,0.99,1.10,adjusted odds ratio\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="Full coefficient table.",
        source_path=full,
        evidence_id="full_coefficients",
    )
    evidence.register_file(
        kind="table",
        description="Primary adjusted association.",
        source_path=single,
        evidence_id="single_primary",
    )
    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="sepsis3",
    )

    selected = _select_primary_association_record(
        evidence,
        run_dir=run_dir,
        context=context,
        names=["adjusted_association_death"],
    )

    assert selected is not None
    assert selected.evidence_id == "single_primary"


def test_publication_figure_ignores_misleading_sensitivity_alias_outside_primary_lineage(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    primary = tmp_path / "adjusted_primary.csv"
    primary.write_text(
        "exposure,point_estimate,ci_low,ci_high,effect_scale\n"
        "vasopressor,1.10,1.01,1.20,adjusted odds ratio\n",
        encoding="utf-8",
    )
    sensitivity = tmp_path / "primary_association.csv"
    sensitivity.write_text(
        "exposure,point_estimate,ci_low,ci_high,effect_scale\n"
        "vasopressor,9.90,8.00,12.00,adjusted odds ratio\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="Planner-owned primary association.",
        source_path=primary,
        evidence_id="true_primary_association",
        produced_by_step="02_primary_model",
    )
    evidence.register_file(
        kind="table",
        description="Sensitivity estimate with a misleading primary alias.",
        source_path=sensitivity,
        evidence_id="sensitivity_decoy",
        aliases=["primary_association"],
        produced_by_step="03_sensitivity_model",
    )
    context = ra.ResearchContext(
        research_question="Is vasopressor exposure associated with mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="vasopressor",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="02_primary_model",
                intent="Estimate the primary adjusted association.",
                expected_outputs=[
                    "table:adjusted_primary",
                    "figure:primary_association_curve",
                ],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity_model",
                intent="Estimate a sensitivity association.",
                expected_outputs=["table:sensitivity_association"],
                planned_analysis_role="sensitivity",
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert "true_primary_association" in summary["source_evidence_ids"]
    assert "sensitivity_decoy" not in summary["source_evidence_ids"]


def test_missingness_frame_deduplicates_variable_labels(ra):
    from easyicu.research_agent.figures.skill import _normalise_missingness_frame

    frame = _normalise_missingness_frame(
        pd.DataFrame(
            {
                "variable": ["lact_max", "lact_max", "bun_max"],
                "missing_fraction": [0.41, 0.39, 0.02],
            }
        ),
        display_labels={"lact": "Lactate", "temp": "Temperature"},
    )

    assert frame["variable"].tolist() == ["Lact Max", "Bun Max"]


def test_missingness_frame_groups_measurement_summary_features(ra):
    from easyicu.research_agent.figures.skill import _normalise_missingness_frame

    frame = _normalise_missingness_frame(
        pd.DataFrame(
            {
                "variable": [
                    "lact_mean",
                    "lact_min",
                    "lact_max",
                    "lact_first",
                    "temp_mean",
                    "temp_min",
                ],
                "missing_fraction": [0.407, 0.407, 0.407, 0.407, 0.027, 0.027],
            }
        ),
        display_labels={"lact": "Lactate", "temp": "Temperature"},
    )

    assert frame["variable"].tolist() == ["Lactate", "Temperature"]
    assert frame["feature_count"].tolist() == [4, 2]
    assert frame["missing_fraction"].round(3).tolist() == [0.407, 0.027]


def test_strata_axis_label_comes_from_score_column(ra):
    from easyicu.research_agent.figures.skill import (
        _normalise_strata_frame,
        _strata_score_label,
    )

    kdigo_frame = _normalise_strata_frame(
        pd.DataFrame({"kdigo_stage": [1, 2, 3], "outcome_rate": [0.1, 0.2, 0.3]}),
        display_labels={"kdigo_stage": "KDIGO stage"},
    )
    assert _strata_score_label(kdigo_frame) == "KDIGO stage"

    generic_frame = _normalise_strata_frame(
        pd.DataFrame({"score": [1, 2, 3], "outcome_rate": [0.1, 0.2, 0.3]})
    )
    assert _strata_score_label(generic_frame) == "Score"

    exposure_frame = _normalise_strata_frame(
        pd.DataFrame(
            {
                "sepsis3": [0, 1],
                "death_pct": [8.5, 12.2],
                "n": [46600, 28229],
            }
        ),
        display_labels={"sepsis3": "Sepsis-3"},
    )
    assert _strata_score_label(exposure_frame) == "Sepsis-3 status"
    assert exposure_frame["rate"].round(3).tolist() == [0.085, 0.122]
    assert exposure_frame["score"].tolist() == [
        "Sepsis-3 negative",
        "Sepsis-3 positive",
    ]
    assert exposure_frame.attrs["score_is_numeric"] is False


def test_display_labels_are_planner_owned_and_fallback_is_nonsemantic(ra):
    from easyicu.research_agent.figures.skill import _display_label

    assert _display_label("death") == "Death"
    assert _display_label("death", {"death": "In-hospital mortality"}) == (
        "In-hospital mortality"
    )
    assert _display_label("DEATH-HOSP", {"death_hosp": "Hospital mortality"}) == (
        "Hospital mortality"
    )


def test_strata_frame_matches_predictor_named_group_column(ra):
    """An exposure/severity stratum is often named after the predictor
    (``lactate_group``, ``sofa2_stratum``). The exact-name candidate list can
    never enumerate these, so a general grouping-suffix fallback must still
    resolve the score column instead of returning an empty frame."""
    from easyicu.research_agent.figures.skill import _normalise_strata_frame

    frame = _normalise_strata_frame(
        pd.DataFrame(
            {
                "lactate_group": [
                    "Unmeasured",
                    "<2 mmol/L",
                    "2 to <4 mmol/L",
                    ">=4 mmol/L",
                ],
                "group_order": [1, 2, 3, 4],
                "mortality_risk": [0.056, 0.085, 0.109, 0.349],
            }
        )
    )
    assert not frame.empty
    assert frame["rate"].round(3).tolist() == [0.056, 0.085, 0.109, 0.349]
    assert "Unmeasured" in frame["score"].tolist()


def test_association_frame_uses_model_label_for_row_labels(ra):
    """When the model table carries ``model_label`` (e.g. primary vs
    complete-case comparator), the forest rows must read as those labels,
    not as the raw odds-ratio floats."""
    from easyicu.research_agent.figures.skill import _normalise_association_frame

    frame = _normalise_association_frame(
        pd.DataFrame(
            {
                "model_label": [
                    "primary_imputed_analytic_cohort",
                    "complete_case_comparator",
                ],
                "point_estimate": [1.0774, 1.0244],
                "ci_low": [1.0220, 0.9640],
                "ci_high": [1.1358, 1.0885],
            }
        )
    )
    assert frame["label"].tolist() == [
        "Primary Imputed Analytic Cohort",
        "Complete Case Comparator",
    ]


def test_publication_figure_skill_renders_from_robustness_panel_without_table(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.robustness.panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary", "primary", 100, 1.33, 1.2, 1.47, 0.1, "e1", True
            ),
            RobustnessPanelRow(
                "alt_cohort", "cohort", 90, 1.10, 0.8, 1.52, 0.2, "e2", True
            ),
        ],
        locked_at="2026-05-27T00:00:00Z",
    )
    _prepare_robustness_authority(ra, run_dir, evidence, panel.rows)
    write_robustness_panel(
        run_dir=run_dir,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )
    context = ra.ResearchContext(
        research_question="Is severity associated with ICU mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[
            ra.ConceptDescriptor(name="sofa", role="ordinal_score", dtype="float64"),
            ra.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="04_report",
                intent="Summarize the registered robustness panel.",
                expected_outputs=["table:summary"],
            )
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
    )

    assert result.generated is True
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["generation_mode"] == "robustness_panel_publication_figure"
    for suffix in ("svg", "png", "pdf", "tiff"):
        assert evidence.get(f"publication_figure_{suffix}") is not None
        assert (
            run_dir / "publication_figures" / f"easyicu_publication_figure.{suffix}"
        ).exists()
    assert evidence.get("publication_figure_contract") is not None


def test_publication_figure_skill_promotes_prediction_validation_bundle(
    ra, tmp_path: Path
):
    from PIL import Image

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    svg = tmp_path / "discrimination_calibration.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
        '<rect width="120" height="80" fill="white"/>'
        '<text x="12" y="28">AUROC 0.78</text>'
        '<text x="12" y="52">Calibration</text>'
        "</svg>",
        encoding="utf-8",
    )
    png = tmp_path / "discrimination_calibration.png"
    Image.new("RGB", (120, 80), "white").save(png)
    summary = tmp_path / "step_summary.json"
    summary.write_text(
        json.dumps(
            {
                "auroc": 0.78,
                "brier_score": 0.18,
                "baseline_prevalence": 0.10,
            }
        ),
        encoding="utf-8",
    )

    evidence = ra.EvidenceStore(run_dir)
    evidence.register_file(
        kind="figure",
        description="Discrimination and calibration figure from model training.",
        source_path=svg,
        evidence_id="figure_discrimination_calibration_svg",
        aliases=["discrimination_calibration"],
    )
    evidence.register_file(
        kind="figure",
        description="Discrimination and calibration figure from model training.",
        source_path=png,
        evidence_id="figure_discrimination_calibration_png",
        aliases=["discrimination_calibration_png"],
    )
    evidence.register_file(
        kind="statistic",
        description="Prediction model summary.",
        source_path=summary,
        evidence_id="statistic_step_summary_model",
        aliases=["01_model_training"],
    )

    context = ra.ResearchContext(
        research_question="Build an ICU mortality prediction model.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[
            ra.ConceptDescriptor(name="age", role="demographic", dtype="float64"),
            ra.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_model_training_figure",
                intent="Render the publication figure(s) declared by step '01_model_training'.",
                expected_outputs=["figure:discrimination_calibration"],
            )
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
    )

    assert result.generated is True
    for suffix in ("svg", "png", "pdf", "tiff"):
        assert evidence.get(f"publication_figure_{suffix}") is not None
        assert (
            run_dir / "publication_figures" / f"easyicu_publication_figure.{suffix}"
        ).exists()
    assert evidence.get("publication_figure_contract") is not None


def test_make_figure_contract_backfills_blank_panel_titles_from_role():
    """Regression (M3 clustering figure): panels with rich roles/claims but
    ``title=""`` must not trip the ``figure_contract_quality`` gate — the blank
    title is backfilled from the panel's own declared role (title-cased)."""
    contract = make_figure_contract(
        {
            "figure_id": "clustering_visualization",
            "core_claim": "Candidate sepsis subphenotypes differ in outcome.",
            "source_data": ["clustering_assignments"],
            "panels": [
                {
                    "panel_id": "A",
                    "title": "",
                    "role": "data_quality",
                    "claim": "Feature availability differs",
                    "evidence_ids": ["a"],
                },
                {
                    "panel_id": "B",
                    "title": "",
                    "role": "phenotype_structure",
                    "claim": "PCA geometry",
                    "evidence_ids": ["b"],
                },
            ],
        }
    )
    titles = {p.panel_id: p.title for p in contract.panels}
    assert titles["A"] == "Data quality"
    assert titles["B"] == "Phenotype structure"
    assert all(p.title.strip() for p in contract.panels)


def test_make_figure_contract_preserves_explicit_panel_title():
    contract = make_figure_contract(
        {
            "figure_id": "f",
            "core_claim": "cc",
            "source_data": ["s"],
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Held-out AUROC",
                    "role": "validation",
                    "claim": "x",
                    "evidence_ids": ["e"],
                },
            ],
        }
    )
    assert contract.panels[0].title == "Held-out AUROC"
