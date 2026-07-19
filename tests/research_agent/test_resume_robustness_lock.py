"""Probe/replan and resume must preserve the immutable robustness roster."""

from __future__ import annotations

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.pipeline import (
    _restore_resume_plan_robustness_lock,
)
from easyicu.research_agent.execution.phase import (
    _preserve_locked_robustness_specs_after_replan,
)
from easyicu.research_agent.robustness.panel import (
    default_robustness_specs,
    robustness_specs_sha,
    write_locked_robustness_specs,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(*, revision: int = 1) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Test the immutable robustness roster.",
        revision=revision,
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the agent-selected primary model.",
                method="adjusted_association_models",
                expected_outputs=["table:adjusted_association_estimates"],
            )
        ],
        robustness_specs=default_robustness_specs(),
    )


def _write_lock(tmp_path, plan: AnalysisPlan) -> EvidenceStore:
    evidence = EvidenceStore(tmp_path)
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    return evidence


def test_probe_replan_cannot_mutate_locked_robustness_specs(tmp_path) -> None:
    current = _plan()
    _write_lock(tmp_path, current)
    revised = current.model_copy(
        update={
            "revision": 2,
            "robustness_specs": list(reversed(current.robustness_specs)),
        }
    )

    preserved, finding = _preserve_locked_robustness_specs_after_replan(
        current_plan=current,
        revised_plan=revised,
        run_dir=tmp_path,
    )

    assert finding is not None
    assert finding.validator == "replanner"
    assert robustness_specs_sha(preserved.robustness_specs) == robustness_specs_sha(
        current.robustness_specs
    )
    assert preserved.revision == 2


def test_resume_migrates_drifted_plan_to_verified_lock(tmp_path) -> None:
    locked_plan = _plan()
    evidence = _write_lock(tmp_path, locked_plan)
    drifted = locked_plan.model_copy(
        update={
            "revision": 2,
            "robustness_specs": list(reversed(locked_plan.robustness_specs)),
        }
    )

    restored, revision_path = _restore_resume_plan_robustness_lock(
        plan=drifted,
        run_dir=tmp_path,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    assert revision_path == tmp_path / "analysis_plan_revision_3.json"
    assert revision_path.is_file()
    assert restored.revision == 3
    assert robustness_specs_sha(restored.robustness_specs) == robustness_specs_sha(
        locked_plan.robustness_specs
    )
    assert evidence.get("analysis_plan_revision_3") is not None
    # The normal resume lock writer now reuses the existing lock rather than
    # failing on the older replanner's drift.
    assert (
        write_locked_robustness_specs(
            run_dir=tmp_path,
            plan=restored,
            evidence=evidence,
            prompt_pack_version="test",
            llm_signature="mock",
        )
        == tmp_path / "robustness_specs_locked.json"
    )
