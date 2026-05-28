"""Writer digest coverage for the pre-specified robustness panel block."""

from __future__ import annotations

from pathlib import Path


def _write_panel(ra, tmp_path: Path, rows):
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        write_robustness_panel,
    )

    evidence = ra.EvidenceStore(tmp_path)
    panel = RobustnessPanel.from_rows(
        rows,
        locked_at="2026-05-27T00:00:00Z",
    )
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )
    return evidence


def test_digest_contains_panel_block_when_panel_populated(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest,
    )
    from easyicu.research_agent.robustness_panel import RobustnessPanelRow

    _write_panel(
        ra,
        tmp_path,
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow("cohort_worst", "cohort", 90, 1.1, 0.7, 1.8, 0.2, "e2", True),
        ],
    )

    digest = _render_writer_evidence_digest(
        [{"step_id": "01_model", "status": "ok", "step_summary": {"primary_or": 1.2}}],
        run_dir=tmp_path,
    )

    assert "## robustness panel" in digest
    assert "CANONICAL PRIMARY EFFECT SOURCE" in digest
    assert "primary: spec_id=primary, point=1.2, CI=[1, 1.4], n=100" in digest
    assert "n_variants=1" in digest


def test_digest_suppresses_generated_primary_effect_when_panel_is_canonical(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest,
    )
    from easyicu.research_agent.robustness_panel import RobustnessPanelRow

    _write_panel(
        ra,
        tmp_path,
        [
            RobustnessPanelRow("primary", "primary", 100, 1.33, 1.2, 1.47, 0.1, "e1", True),
            RobustnessPanelRow("cohort_worst", "cohort", 90, 1.1, 0.7, 1.8, 0.2, "e2", True),
        ],
    )

    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": "03_association_model",
                "status": "ok",
                "step_summary": {"odds_ratio": 0.75, "p_value": 0.73, "n": 225},
            }
        ],
        run_dir=tmp_path,
    )

    assert "CANONICAL PRIMARY EFFECT SOURCE" in digest
    assert "point=1.33" in digest
    assert '"odds_ratio": 0.75' not in digest
    assert '"p_value": 0.73' not in digest


def test_digest_panel_block_shows_range_not_rows(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest,
    )
    from easyicu.research_agent.robustness_panel import RobustnessPanelRow

    _write_panel(
        ra,
        tmp_path,
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow("cohort_worst", "cohort", 90, 1.1, 0.7, 1.8, 0.2, "e2", True),
            RobustnessPanelRow("cohort_hidden", "cohort", 90, 1.777, 1.6, 1.8, 0.2, "e3", True),
        ],
    )

    digest = _render_writer_evidence_digest([], run_dir=tmp_path)

    assert "range across variants point" in digest
    assert "cohort_worst" in digest
    assert "cohort_hidden" not in digest
    assert "point=1.777" not in digest


def test_digest_panel_block_handles_zero_converged(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest,
    )
    from easyicu.research_agent.robustness_panel import RobustnessPanelRow

    _write_panel(
        ra,
        tmp_path,
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow(
                "alt_missing",
                "missing",
                0,
                None,
                None,
                None,
                None,
                "e2",
                False,
                "not implemented",
            ),
        ],
    )

    digest = _render_writer_evidence_digest([], run_dir=tmp_path)

    assert "## robustness panel" in digest
    assert "no robustness variants converged" in digest


def test_digest_panel_block_lists_worst_per_axis(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest,
    )
    from easyicu.research_agent.robustness_panel import RobustnessPanelRow

    _write_panel(
        ra,
        tmp_path,
        [
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
            RobustnessPanelRow("cohort_worst", "cohort", 90, 0.9, 0.6, 1.2, 0.2, "e2", True),
            RobustnessPanelRow("missing_worst", "missing", 95, 1.0, 0.8, 1.3, 0.2, "e3", True),
            RobustnessPanelRow("outcome_worst", "outcome", 92, 1.1, 0.9, 1.5, 0.2, "e4", True),
        ],
    )

    digest = _render_writer_evidence_digest([], run_dir=tmp_path)

    assert "worst on cohort axis: spec_id=cohort_worst, point=0.9" in digest
    assert "worst on missing axis: spec_id=missing_worst, point=1" in digest
    assert "worst on outcome axis: spec_id=outcome_worst, point=1.1" in digest
