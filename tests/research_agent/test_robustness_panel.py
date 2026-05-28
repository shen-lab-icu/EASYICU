"""Pre-specified robustness panel contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_planner_emits_minimum_axes() -> None:
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    with pytest.raises(ValueError, match="at least 3 cohort"):
        AnalysisPlan(
            research_question="Does severity predict mortality?",
            steps=[
                AnalysisStep(
                    step_id="01_model",
                    intent="Fit the primary model.",
                    expected_outputs=["statistic:primary_or"],
                )
            ],
            robustness_specs=[
                {
                    "spec_id": "alt_missing_complete_case",
                    "axis": "missing",
                    "description": "Complete cases only.",
                    "cohort_override": None,
                    "missing_override": {"strategy": "complete_case"},
                    "outcome_override": None,
                }
            ],
        )


def test_panel_freezes_after_plan(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.robustness_panel import (
        assert_robustness_specs_locked,
        default_robustness_specs,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Does severity predict mortality?",
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the primary model.",
                expected_outputs=["statistic:primary_or"],
            )
        ],
        robustness_specs=default_robustness_specs(),
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    changed = plan.model_copy(
        update={"robustness_specs": list(plan.robustness_specs[1:])}
    )
    with pytest.raises(Exception, match="changed after plan lock"):
        assert_robustness_specs_locked(run_dir=tmp_path, plan=changed)


def test_locked_robustness_specs_restore_after_replan_drop(ra, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        default_robustness_specs,
        robustness_specs_for_execution,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = default_robustness_specs()
    plan = AnalysisPlan(
        research_question="Does severity predict mortality?",
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the primary model.",
                expected_outputs=["statistic:primary_or"],
            )
        ],
        robustness_specs=specs,
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    replanned_without_specs = SimpleNamespace(robustness_specs=[])
    restored = robustness_specs_for_execution(
        run_dir=tmp_path,
        plan=replanned_without_specs,
    )

    assert [spec.spec_id for spec in restored] == [spec.spec_id for spec in specs]


def test_each_spec_produces_panel_row() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
    )

    specs = default_robustness_specs()
    records = [
        {
            "step_id": "01_model",
            "step_summary_evidence_id": "stat_model",
            "step_summary": {
                "primary_or": 1.4,
                "primary_ci_low": 1.1,
                "primary_ci_high": 1.8,
                "n_total": 100,
                "robustness_rows": [
                    {
                        "spec_id": spec.spec_id,
                        "axis": spec.axis,
                        "n": 90,
                        "point_estimate": 1.2 + idx / 10,
                        "ci_low": 0.9 + idx / 10,
                        "ci_high": 1.5 + idx / 10,
                        "se": 0.1,
                        "converged": True,
                        "notes": "ok",
                    }
                    for idx, spec in enumerate(specs)
                ],
            },
        }
    ]

    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=records,
        locked_at="2026-05-27T00:00:00Z",
    )

    assert len(panel.rows) == 8
    assert panel.n_variants == 7
    assert panel.rows[0].spec_id == "primary"


def test_non_convergence_does_not_abort() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
    )

    specs = default_robustness_specs()
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=[
            {
                "step_id": "01_model",
                "step_summary": {
                    "primary_or": 1.4,
                    "primary_ci_low": 1.1,
                    "primary_ci_high": 1.8,
                    "robustness_rows": [
                        {
                            "spec_id": specs[0].spec_id,
                            "axis": specs[0].axis,
                            "n": 0,
                            "point_estimate": None,
                            "ci_low": None,
                            "ci_high": None,
                            "se": None,
                            "converged": False,
                            "notes": "n=0 after applying override",
                        }
                    ],
                },
            }
        ],
    )

    row = next(r for r in panel.rows if r.spec_id == specs[0].spec_id)
    assert row.converged is False
    assert row.n == 0


def test_panel_range_correctness() -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow("a", "cohort", 90, 1.5, 0.8, 2.0, 0.2, "e2", True),
            RobustnessPanelRow("b", "missing", 0, None, None, None, None, "e3", False),
        ],
        locked_at="2026-05-27T00:00:00Z",
    )

    assert panel.range_low == 0.8
    assert panel.range_high == 2.0


def test_writer_digest_contains_panel_block(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest_v2,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow("cohort_worst", "cohort", 90, 1.1, 0.7, 1.8, 0.2, "e2", True),
            RobustnessPanelRow("cohort_hidden", "cohort", 90, 1.9, 1.5, 2.3, 0.2, "e3", True),
        ],
        locked_at="2026-05-27T00:00:00Z",
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    digest = _render_writer_evidence_digest_v2(
        [{"step_id": "01_model", "status": "ok", "step_summary": {"primary_or": 1.2}}],
        run_dir=tmp_path,
        evidence=evidence,
    )

    assert "## robustness panel" in digest
    assert "n_variants=2" in digest
    assert "cohort_hidden" not in digest
    assert "OR=1.9" not in digest


def test_writer_forbidden_terms_blocked_in_strict(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import enforce_writer_claim_language

    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        enforce_writer_claim_language(
            "Surprisingly, the model was stable.",
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
        )
    assert "surprisingly" in exc_info.value.detail["forbidden_terms"]


def test_writer_forbidden_terms_annotated_in_soft(ra) -> None:
    from easyicu.research_agent.manuscript_post import enforce_writer_claim_language

    annotated, detail = enforce_writer_claim_language(
        "Surprisingly, the model was stable.",
        enforcement_mode=ra.EvidenceEnforcementMode.SOFT,
    )

    assert "<!-- LEXICON:surprisingly -->" in annotated
    assert detail["forbidden_terms"] == ["surprisingly"]


def test_manifest_records_panel_artifact(tmp_path: Path) -> None:
    from easyicu.research_agent.schema import AnalysisManifest

    manifest = AnalysisManifest(
        run_id="run",
        research_question="Question?",
        started_at="2026-05-27T00:00:00Z",
        context_path="context.json",
        robustness_panel_path="robustness_panel.json",
        robustness_panel_sha="abc",
        robustness_n_variants=7,
        robustness_range_low=0.8,
        robustness_range_high=2.1,
    )

    assert manifest.robustness_panel_sha == "abc"
    assert manifest.robustness_n_variants == 7


def test_panel_numerics_registered_in_evidence_store(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow("a", "cohort", 90, 1.5, 0.8, 2.0, 0.2, "e2", True),
        ]
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    fields = {claim.source_field for claim in evidence.numeric_claims()}
    assert {"n_variants", "range_low", "range_high"} <= fields
    assert "primary_point_estimate" in fields
    assert "row_primary_point_estimate" not in fields


def test_primary_only_panel_does_not_register_duplicate_range_claims(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
        ]
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    fields = {claim.source_field for claim in evidence.numeric_claims()}
    assert "primary_point_estimate" in fields
    assert "row_primary_point_estimate" not in fields
    assert "range_low" not in fields
    assert "range_high" not in fields
