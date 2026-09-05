"""Progressive planner checkpoint, resume, and evidence-artifact authority."""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import pytest
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
)
from easyicu.research_agent.planning.progressive_artifacts import (
    ProgressiveCompileFailureReplay,
    ProgressivePlannerCheckpointRecorder,
    ProgressivePlanningArtifactError,
    load_progressive_compile_failure_replay,
    load_progressive_planner_checkpoint_chain,
    persist_progressive_planner_checkpoint,
    persist_progressive_planning_artifacts,
    persist_progressive_planning_authority,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlannerCheckpoint,
)
from easyicu.research_agent.planning.progressive_resume import (
    ProgressivePrefixState,
    compile_progressive_prefix,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    ProgressiveDesignCanaryDraft,
    run_progressive_planner,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.authority.plan_lifecycle import (
    build_normalized_plan_lineage,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _context,
    _foundation_payload as _foundation_payload,
    _materialization_payloads as _materialization_payloads,
    _outline_payload as _outline_payload,
)


class _RecordingEvidence:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def get(self, evidence_id_or_alias: str) -> object | None:
        return self.records.get(evidence_id_or_alias)

    def register_file(self, **kwargs: object) -> object:
        evidence_id = str(kwargs["evidence_id"])
        source_path = Path(str(kwargs["source_path"]))
        self.records[evidence_id] = {
            **dict(kwargs),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        return self.records[evidence_id]


def test_progressive_compile_failure_persists_for_zero_provider_replay(
    tmp_path: Path,
) -> None:
    materializations = _materialization_payloads()
    invalid_distribution = json.loads(json.dumps(materializations[2]))
    invalid_distribution["step"]["comparison_exposure_level_index"] = 0
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in materializations[:2]],
            json.dumps(invalid_distribution),
            json.dumps(invalid_distribution),
            json.dumps(invalid_distribution),
        ]
    )
    llm.supports_strict_json_schema = True
    evidence = _RecordingEvidence()
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic replay cohort")

    with pytest.raises(ProgressivePlanCompileError):
        run_progressive_planner(
            planner=ProgressivePlannerAgent(llm),
            context=_context(),
            run_dir=tmp_path,
            evidence=evidence,
            prompt_pack_version="test-v1",
            resume_checkpoint_path=None,
            resume_checkpoint_sha256=None,
            cohort_path=cohort_path,
            llm_signature="mock:test",
            planner_kwargs={},
            know_how_binding=PlannerKnowHowBinding(),
            planning_contract_context="",
            finding_sink=lambda _finding: None,
        )

    replay_path = tmp_path / "progressive_compile_failure_replay.json"
    replay = load_progressive_compile_failure_replay(
        replay_path=replay_path,
        expected_artifact_sha256=str(
            evidence.records["progressive_compile_failure_replay"]["sha256"]
        ),
    )
    assert isinstance(replay, ProgressiveCompileFailureReplay)
    assert replay.prefix_checkpoint_sequence == 3
    assert len(replay.attempts) == 3
    assert evidence.records["progressive_compile_failure_replay"]["inputs"] == [
        "research_context",
        "progressive_planner_checkpoint_003",
    ]

    checkpoint = ProgressivePlannerCheckpoint.model_validate_json(
        (tmp_path / "progressive_planner_checkpoint_003.json").read_bytes()
    )
    assert checkpoint.foundation is not None
    state = ProgressivePrefixState()
    for materialization in checkpoint.materializations:
        state = compile_progressive_prefix(
            state,
            materialization,
            outline=checkpoint.outline,
            foundation=checkpoint.foundation.foundation,
            context=_context(),
            allowed_literature_citation_keys=(),
            allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )
    with pytest.raises(ProgressivePlanCompileError) as replayed:
        compile_progressive_prefix(
            state,
            replay.attempts[0].materialization,
            outline=checkpoint.outline,
            foundation=checkpoint.foundation.foundation,
            context=_context(),
            allowed_literature_citation_keys=(),
            allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )
    assert replayed.value.reason_code == (
        replay.attempts[0].compiler_finding.reason_code
    )

    replay_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProgressivePlanningArtifactError) as tampered:
        load_progressive_compile_failure_replay(
            replay_path=replay_path,
            expected_artifact_sha256=str(
                evidence.records["progressive_compile_failure_replay"]["sha256"]
            ),
        )
    assert tampered.value.reason_code == ("progressive_compile_replay_digest_mismatch")


def test_progressive_checkpoints_persist_as_a_digest_verified_chain(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    evidence = _RecordingEvidence()
    paths = []

    def checkpoint_callback(checkpoint) -> None:
        paths.append(
            persist_progressive_planner_checkpoint(
                run_dir=tmp_path,
                evidence=evidence,
                checkpoint=checkpoint,
                prompt_pack_version="test",
            )
        )

    agent.run(_context(), checkpoint_callback=checkpoint_callback)

    assert [path.name for path in paths] == [
        f"progressive_planner_checkpoint_{index:03d}.json" for index in range(9)
    ]
    assert set(evidence.records) == {
        f"progressive_planner_checkpoint_{index:03d}" for index in range(9)
    }
    assert evidence.records["progressive_planner_checkpoint_008"]["inputs"] == [
        "research_context",
        "progressive_planner_checkpoint_007",
    ]

    loaded = load_progressive_planner_checkpoint_chain(
        last_checkpoint_path=paths[-1],
        expected_artifact_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
    )
    assert [item.sequence for item in loaded] == list(range(9))
    assert (
        loaded[-1].checkpoint_sha256
        == json.loads(paths[-1].read_text(encoding="utf-8"))["checkpoint_sha256"]
    )


def test_progressive_design_canary_stops_after_one_validated_outline(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient([json.dumps(_outline_payload())])
    llm.supports_strict_json_schema = True
    evidence = _RecordingEvidence()
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"design canary cohort")

    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(llm),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        prompt_pack_version="test-v1",
        resume_checkpoint_path=None,
        resume_checkpoint_sha256=None,
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=lambda _finding: None,
        stop_after_outline=True,
    )

    assert isinstance(result, ProgressiveDesignCanaryDraft)
    assert result.checkpoint.stage == "outline"
    assert result.outline.design_selection is not None
    assert len(llm.calls) == 1
    assert not (tmp_path / "progressive_plan_foundation.json").exists()


def test_progressive_resume_loader_rejects_incomplete_source_chain(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    agent.run(_context(), checkpoint_callback=checkpoints.append)
    terminal = tmp_path / "progressive_planner_checkpoint_004.json"
    terminal.write_text(checkpoints[4].model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        load_progressive_planner_checkpoint_chain(
            last_checkpoint_path=terminal,
            expected_artifact_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest(),
        )

    assert caught.value.reason_code == "progressive_resume_checkpoint_missing"


def test_resume_checkpoint_recorder_imports_only_after_validation(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    agent.run(_context(), checkpoint_callback=checkpoints.append)
    evidence = _RecordingEvidence()
    recorder = ProgressivePlannerCheckpointRecorder(
        run_dir=tmp_path,
        evidence=evidence,
        prompt_pack_version="test",
        source_chain=tuple(checkpoints[:5]),
    )

    recorder.record(checkpoints[5])

    assert evidence.records == {}
    assert list(tmp_path.glob("progressive_planner_checkpoint_*.json")) == []

    receipt = recorder.persist_validated_resume()

    assert receipt.source_sequence == 4
    assert receipt.reused_materialization_count == 3
    assert receipt.new_checkpoint_count == 1
    assert set(evidence.records) == {
        f"progressive_planner_checkpoint_{index:03d}" for index in range(6)
    }
    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        recorder.record(checkpoints[6])
    assert caught.value.reason_code == ("progressive_resume_checkpoint_recorder_closed")


def test_progressive_orchestrator_resumes_and_imports_validated_chain(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"development cohort authority")
    dependency_context = {
        "cohort_file_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "llm_signature": "mock:test",
        "prompt_version": "test-v1",
    }
    materializations = _materialization_payloads()
    source_agent = ProgressivePlannerAgent(
        ScriptedMockLLMClient(
            [
                json.dumps(_outline_payload()),
                json.dumps(_foundation_payload()),
                *[json.dumps(item) for item in materializations],
            ]
        )
    )
    source_agent.llm.supports_strict_json_schema = True
    source_checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=source_checkpoints.append,
        resume_dependency_context=dependency_context,
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_evidence = _RecordingEvidence()
    source_paths = [
        persist_progressive_planner_checkpoint(
            run_dir=source_dir,
            evidence=source_evidence,
            checkpoint=checkpoint,
            prompt_pack_version="test-v1",
        )
        for checkpoint in source_checkpoints[:5]
    ]
    resumed_llm = ScriptedMockLLMClient(
        [json.dumps(item) for item in materializations[3:]]
    )
    resumed_llm.supports_strict_json_schema = True
    findings = []
    current_evidence = _RecordingEvidence()
    current_dir = tmp_path / "current"
    current_dir.mkdir()

    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(resumed_llm),
        context=_context(),
        run_dir=current_dir,
        evidence=current_evidence,
        prompt_pack_version="test-v1",
        resume_checkpoint_path=source_paths[-1],
        resume_checkpoint_sha256=hashlib.sha256(
            source_paths[-1].read_bytes()
        ).hexdigest(),
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=findings.append,
    )

    assert result.generation_mode == "llm_progressive_v2_dev_resume"
    assert len(result.plan.steps) == 7
    assert result.facts.resume_validated is True
    assert result.facts.complete_for_persistence is True
    assert len(result.facts.materializations) == 7
    assert len(resumed_llm.calls) == 4
    assert findings[0].detail["reason_code"] == (
        "progressive_development_checkpoint_resumed"
    )
    assert {
        key
        for key in current_evidence.records
        if key.startswith("progressive_planner_checkpoint_")
    } == {f"progressive_planner_checkpoint_{index:03d}" for index in range(9)}


def test_progressive_orchestrator_persists_validated_resume_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"development cohort authority")
    dependencies = {
        "cohort_file_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "llm_signature": "mock:test",
        "prompt_version": "test-v1",
    }
    source_agent = ProgressivePlannerAgent(
        ScriptedMockLLMClient(
            [
                json.dumps(_outline_payload()),
                json.dumps(_foundation_payload()),
                *[json.dumps(item) for item in _materialization_payloads()],
            ]
        )
    )
    checkpoints = []
    source_agent.run(
        _context(),
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=dependencies,
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_evidence = _RecordingEvidence()
    paths = [
        persist_progressive_planner_checkpoint(
            run_dir=source_dir,
            evidence=source_evidence,
            checkpoint=checkpoint,
            prompt_pack_version="test-v1",
        )
        for checkpoint in checkpoints[:2]
    ]
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    evidence = _RecordingEvidence()
    planner = ProgressivePlannerAgent(ScriptedMockLLMClient([]))

    def interrupted(*_args, **_kwargs):
        planner._attempt.resume_validated = True
        raise KeyboardInterrupt("operator stop")

    monkeypatch.setattr(planner, "_run_output", interrupted)
    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        run_progressive_planner(
            planner=planner,
            context=_context(),
            run_dir=current_dir,
            evidence=evidence,
            prompt_pack_version="test-v1",
            resume_checkpoint_path=paths[-1],
            resume_checkpoint_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
            cohort_path=cohort_path,
            llm_signature="mock:test",
            planner_kwargs={},
            know_how_binding=PlannerKnowHowBinding(),
            planning_contract_context="",
            finding_sink=lambda _finding: None,
        )

    assert (current_dir / "progressive_planner_checkpoint_001.json").exists()


def test_progressive_checkpoint_rejects_mutated_predecessor(tmp_path: Path) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    agent.run(_context(), checkpoint_callback=checkpoints.append)
    evidence = _RecordingEvidence()
    first_path = persist_progressive_planner_checkpoint(
        run_dir=tmp_path,
        evidence=evidence,
        checkpoint=checkpoints[0],
        prompt_pack_version="test",
    )
    first_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        persist_progressive_planner_checkpoint(
            run_dir=tmp_path,
            evidence=evidence,
            checkpoint=checkpoints[1],
            prompt_pack_version="test",
        )

    assert caught.value.reason_code == ("progressive_source_artifact_digest_mismatch")


def test_progressive_artifacts_bind_each_schema_authority(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    plan = agent.run(_context())
    assert agent.last_result.facts.outline is not None
    assert agent.last_result.facts.foundation is not None
    assert agent.last_result.facts.skeleton is not None
    assert agent.last_result.facts.compile_receipt is not None
    evidence = _RecordingEvidence()

    paths = persist_progressive_planning_artifacts(
        run_dir=tmp_path,
        evidence=evidence,
        outline=agent.last_result.facts.outline,
        foundation=agent.last_result.facts.foundation,
        materializations=agent.last_result.facts.materializations,
        skeleton=agent.last_result.facts.skeleton,
        compile_receipt=agent.last_result.facts.compile_receipt,
        prompt_metrics=agent.last_result.facts.prompt_metrics,
        prompt_pack_version="test",
    )

    ledger = json.loads(paths.materializations.read_text(encoding="utf-8"))
    requests = [call[1]["structured_output"] for call in llm.calls]
    assert ledger["outline_structured_output_authority_sha256"] == (
        requests[0].authority_sha256
    )
    assert ledger["foundation_structured_output_authority_sha256"] == (
        requests[1].authority_sha256
    )
    assert [
        item["structured_output_authority_sha256"]
        for item in ledger["materializations"]
    ] == [request.authority_sha256 for request in requests[2:]]
    assert [item["step_id"] for item in ledger["materializations"]] == [
        item.step.step_id for item in agent.last_result.facts.materializations
    ]
    assert set(evidence.records) == {
        "progressive_plan_outline",
        "progressive_plan_foundation",
        "progressive_step_materializations",
        "progressive_plan_skeleton",
        "progressive_plan_compile_receipt",
    }
    assert evidence.records["progressive_plan_skeleton"]["inputs"] == [
        "progressive_plan_outline",
        "progressive_plan_foundation",
        "progressive_step_materializations",
        "research_context",
    ]

    metrics_path = tmp_path / "planner_prompt_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.planner_prompt_metrics/1",
                **agent.last_result.facts.prompt_metrics,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.register_file(
        evidence_id="planner_prompt_metrics",
        source_path=metrics_path,
    )
    plan_path = tmp_path / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(evidence_id="analysis_plan", source_path=plan_path)
    normalized = build_normalized_plan_lineage(
        proposed_plan=plan,
        proposed_source="llm_progressive_v2",
        pre_normalization_plan=plan,
        normalized_plan=plan,
        resume_scientific_semantics_changed=False,
        host_scientific_semantics_changed=False,
    )
    lifecycle_path = tmp_path / "plan_lifecycle_revision_0.json"
    lifecycle_path.write_text(
        normalized.model_dump_json(indent=2),
        encoding="utf-8",
    )
    evidence.register_file(
        evidence_id="plan_lifecycle_revision_0",
        source_path=lifecycle_path,
    )

    authority = persist_progressive_planning_authority(
        run_dir=tmp_path,
        evidence=evidence,
        proposed_plan_sha256=normalized.proposed.plan_sha256,
        normalized_plan_sha256=normalized.plan_sha256,
        normalized_plan_authority_sha256=normalized.authority_sha256,
        normalized_plan_evidence_id="plan_lifecycle_revision_0",
        normalized_plan_filename="plan_lifecycle_revision_0.json",
        prompt_pack_version="test",
    )

    assert authority.strict_transport_bound is True
    assert authority.compiled_analysis_plan_sha256 == normalized.proposed.plan_sha256
    assert authority.normalized_plan_authority_sha256 == normalized.authority_sha256
    assert [item.step_id for item in authority.ordered_steps] == [
        item.step_id for item in agent.last_result.facts.outline.steps
    ]
    assert evidence.records["progressive_planning_authority"]["inputs"][-1] == (
        "plan_lifecycle_revision_0"
    )


def test_progressive_artifacts_fail_closed_on_schema_authority_drift(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    agent.run(_context())
    assert agent.last_result.facts.outline is not None
    assert agent.last_result.facts.foundation is not None
    assert agent.last_result.facts.skeleton is not None
    assert agent.last_result.facts.compile_receipt is not None
    drifted_metrics = dict(agent.last_result.facts.prompt_metrics)
    drifted_metrics["step_materialization_schema_sha256"] = ["0" * 64]

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        persist_progressive_planning_artifacts(
            run_dir=tmp_path,
            evidence=_RecordingEvidence(),
            outline=agent.last_result.facts.outline,
            foundation=agent.last_result.facts.foundation,
            materializations=agent.last_result.facts.materializations,
            skeleton=agent.last_result.facts.skeleton,
            compile_receipt=agent.last_result.facts.compile_receipt,
            prompt_metrics=drifted_metrics,
            prompt_pack_version="test",
        )

    assert caught.value.reason_code == (
        "progressive_step_schema_authority_count_mismatch"
    )


def test_progressive_artifacts_do_not_overwrite_existing_evidence_identity(
    tmp_path: Path,
) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    agent.run(_context())
    assert agent.last_result.facts.outline is not None
    assert agent.last_result.facts.foundation is not None
    assert agent.last_result.facts.skeleton is not None
    assert agent.last_result.facts.compile_receipt is not None
    evidence = _RecordingEvidence()
    paths = persist_progressive_planning_artifacts(
        run_dir=tmp_path,
        evidence=evidence,
        outline=agent.last_result.facts.outline,
        foundation=agent.last_result.facts.foundation,
        materializations=agent.last_result.facts.materializations,
        skeleton=agent.last_result.facts.skeleton,
        compile_receipt=agent.last_result.facts.compile_receipt,
        prompt_metrics=agent.last_result.facts.prompt_metrics,
        prompt_pack_version="test",
    )
    original_ledger = paths.materializations.read_bytes()
    changed_step = agent.last_result.facts.materializations[0].step.model_copy(
        update={"objective": "A different unreviewed objective."}
    )
    changed_materializations = [
        agent.last_result.facts.materializations[0].model_copy(
            update={"step": changed_step}
        ),
        *agent.last_result.facts.materializations[1:],
    ]

    with pytest.raises(ProgressivePlanningArtifactError) as caught:
        persist_progressive_planning_artifacts(
            run_dir=tmp_path,
            evidence=evidence,
            outline=agent.last_result.facts.outline,
            foundation=agent.last_result.facts.foundation,
            materializations=changed_materializations,
            skeleton=agent.last_result.facts.skeleton,
            compile_receipt=agent.last_result.facts.compile_receipt,
            prompt_metrics=agent.last_result.facts.prompt_metrics,
            prompt_pack_version="test",
        )

    assert caught.value.reason_code == (
        "progressive_existing_evidence_identity_mismatch"
    )
    assert paths.materializations.read_bytes() == original_ledger
