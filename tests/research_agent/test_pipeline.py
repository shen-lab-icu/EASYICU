"""End-to-end pipeline test with the synthetic SOFA cohort.

This is the integration test the ROADMAP's "mock pipeline must always
pass" rule rests on. If this regresses, the demo (and any reviewer
clicking "run") gets a broken artefact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_pipeline_end_to_end_synthetic_cohort(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="synthetic_test_cohort",
        database="synthetic",
        target_outcome="death",
        cross_database_validation=["miiv", "eicu"],
    )
    # 1) The result paths exist and are populated.
    paths = result.as_paths()
    for k, p in paths.items():
        assert Path(p).exists(), f"missing {k}: {p}"

    run_dir = Path(result.workdir)

    # 2) Manifest and evidence were written.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence"], "manifest has no registered evidence"
    kinds = {e["kind"] for e in manifest["evidence"]}
    assert {"code", "log", "table", "figure", "statistic"} <= kinds, (
        f"evidence kinds incomplete: {kinds}"
    )
    # at least 6 artefacts as required by the roadmap
    assert len(manifest["evidence"]) >= 6, manifest["evidence"]

    # 3) The bound manuscript should have ZERO ``[evidence missing: …]``
    #    lines (T1.2 acceptance criterion).
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    assert "[evidence missing:" not in bound, (
        "bound manuscript contains unresolved evidence placeholders:\n" + bound
    )

    # 4) The SOFA-zero anomaly should appear in at least one step_summary.json.
    summaries = list(run_dir.rglob("step_summary.json"))
    assert summaries, "no step_summary.json was produced"
    flagged = False
    for ssj in summaries:
        try:
            data = json.loads(ssj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("sofa_zero_anomaly"):
            flagged = True
            break
    assert flagged, "synthetic cohort SOFA2==0 anomaly was not detected"

    # 5) The manifest's findings should mention the anomaly.
    finding_msgs = " ".join(f.get("message", "") for f in manifest["findings"])
    assert "non-monotonic" in finding_msgs.lower() or "exceeds" in finding_msgs.lower(), (
        f"validator did not surface the SOFA-zero anomaly:\n{finding_msgs}"
    )


def test_pipeline_with_clinical_skill(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        cohort=synthetic_cohort,
        cohort_name="synthetic_skill_cohort",
        database="synthetic",
        skill="sofa_mortality",
    )
    assert result.evidence_count > 0
    # The skill plan must include a sofa_zero audit step.
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    step_ids = [s["step_id"] for s in plan["steps"]]
    assert any("sofa_zero" in sid for sid in step_ids)


def test_pipeline_can_pause_after_analysis_phase(ra, synthetic_cohort, tmp_path: Path):
    events = []
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="synthetic_analysis_only",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
        progress_callback=events.append,
    )

    run_dir = Path(result.workdir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "paused_after_analysis" in manifest["notes"]
    assert not (run_dir / "manuscript_scaffold.tex").exists()
    assert not (run_dir / "literature_bundle.json").exists()
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    assert "stopped after the analysis phase" in bound
    report = (run_dir / "results_report.md").read_text(encoding="utf-8")
    assert "PAUSED AFTER ANALYSIS" in report
    assert any(e.get("stage") == "step" and e.get("status") == "complete" for e in events)
    assert any(e.get("stage") == "pause" and e.get("status") == "paused" for e in events)


def test_mock_planner_honours_sofa2_when_sofa_is_also_present(ra, synthetic_cohort, tmp_path: Path):
    """A SOFA-2 question must not silently fall back to a legacy ``sofa`` column."""
    cohort = synthetic_cohort.copy()
    cohort.insert(3, "sofa", cohort["sofa2"])

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is early SOFA-2 associated with ICU mortality?",
        cohort=cohort,
        cohort_name="synthetic_sofa_and_sofa2",
        database="synthetic",
        target_outcome="death",
    )

    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    by_id = {step["step_id"]: step for step in plan["steps"]}
    assert by_id["04_primary_association"]["inputs"][:2] == ["sofa2", "death"]
    assert by_id["05_sofa_zero_audit"]["inputs"][:2] == ["sofa2", "death"]


def test_mock_planner_maps_clinical_phrases_to_expected_predictors(ra, tmp_path: Path):
    """Clinical wording such as KDIGO stage / vasopressor should not fall back to age."""
    cases = [
        (
            "Is peak first-24h KDIGO AKI stage associated with ICU mortality?",
            "kdigo_stage",
            pd.DataFrame({
                "stay_id": range(1, 81),
                "age": [60 + (i % 20) for i in range(80)],
                "kdigo_stage": [i % 4 for i in range(80)],
                "creat": [0.8 + 0.2 * (i % 4) for i in range(80)],
                "death": [1 if i % 5 == 0 else 0 for i in range(80)],
            }),
        ),
        (
            "Is any-vasopressor exposure within the first 24h associated with ICU mortality?",
            "vaso",
            pd.DataFrame({
                "stay_id": range(1, 81),
                "age": [60 + (i % 20) for i in range(80)],
                "vaso": [i % 2 for i in range(80)],
                "death": [1 if i % 5 == 0 else 0 for i in range(80)],
            }),
        ),
    ]

    for question, predictor, cohort in cases:
        pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / predictor, llm=ra.MockLLMClient())
        result = pipeline.run(
            question=question,
            cohort=cohort,
            cohort_name=f"{predictor}_phrase_test",
            database="synthetic",
            target_outcome="death",
        )
        plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
        by_id = {step["step_id"]: step for step in plan["steps"]}
        assert by_id["04_primary_association"]["inputs"][:2] == [predictor, "death"]
