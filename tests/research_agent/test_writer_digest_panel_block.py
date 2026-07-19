"""Writer digest coverage for the pre-specified robustness panel block."""

from __future__ import annotations

import json
from pathlib import Path


def _write_panel(ra, tmp_path: Path, rows):
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessSpec,
        default_robustness_specs,
        write_locked_robustness_specs,
        write_robustness_panel,
    )
    from easyicu.research_agent.schema import AnalysisPlan
    from easyicu.research_agent.cohort.schema import CohortDefinition

    evidence = ra.EvidenceStore(tmp_path)
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
    plan = AnalysisPlan(research_question="test", steps=[], robustness_specs=specs)
    write_locked_robustness_specs(
        run_dir=tmp_path,
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
            description="Digest fixture row authority.",
            payload=payload,
            filename=f"{row.evidence_id}.json",
            evidence_id=row.evidence_id,
        )
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


def test_digest_contains_blocked_outcome_gate_guard(tmp_path: Path) -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest,
    )

    step_out = tmp_path / "steps" / "04_outcome_gate" / "outputs"
    step_out.mkdir(parents=True)
    (step_out / "step_summary.json").write_text(
        json.dumps(
            {
                "step_id": "04_outcome_gate",
                "primary_analysis_authorized": False,
                "grouped_death_analysis_executed": False,
                "target_outcome": "death",
                "named_blocking_policy": ["no_silent_imputation"],
            }
        ),
        encoding="utf-8",
    )
    (step_out / "outcome_feasibility_gate.csv").write_text(
        "status,blocking_decision,future_rerun_condition,target_outcome\n"
        "blocked,Outcome linkage is blocked.,Supply certified status columns.,death\n",
        encoding="utf-8",
    )

    digest = _render_writer_evidence_digest(
        [{"step_id": "04_outcome_gate", "status": "ok", "step_summary": {}}],
        run_dir=tmp_path,
    )

    assert "## blocked outcome gate" in digest
    assert "do not report outcome associations" in digest
    assert "blocked_steps=04_outcome_gate" in digest
    assert "Outcome linkage is blocked" in digest


def test_digest_suppresses_robustness_effect_when_outcome_gate_blocked(
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
            RobustnessPanelRow("primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True),
        ],
    )
    step_out = tmp_path / "steps" / "04_outcome_gate" / "outputs"
    step_out.mkdir(parents=True)
    (step_out / "step_summary.json").write_text(
        json.dumps(
            {
                "step_id": "04_outcome_gate",
                "primary_analysis_authorized": False,
                "grouped_death_analysis_executed": False,
                "target_outcome": "death",
            }
        ),
        encoding="utf-8",
    )

    digest = _render_writer_evidence_digest([], run_dir=tmp_path)

    assert "## blocked outcome gate" in digest
    assert "not manuscript-facing" in digest
    assert "CANONICAL PRIMARY EFFECT SOURCE" not in digest
    assert "primary: spec_id=primary" not in digest


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
