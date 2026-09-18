"""Final acceptance must feed bounded suffix repair without changing bindings."""
import json
import hashlib
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner as owner
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.progressive_contract import ProgressivePlannerCheckpoint
from easyicu.research_agent.planning.progressive_artifacts import (
    ProgressivePlanningAuthority,
    ProgressivePlanningArtifactError,
    load_progressive_planner_checkpoint_chain,
    persist_progressive_planner_checkpoint,
    persist_progressive_planning_artifacts,
    persist_progressive_planning_authority,
)
from easyicu.research_agent.authority.plan_lifecycle import build_normalized_plan_lineage
from easyicu.research_agent.planning.progressive_resume import (
    article_role_repair_owners, final_acceptance_repair_start,
    restore_progressive_resume_prefix,
)
from easyicu.research_agent.planning.progressive_contract import canonical_sha256
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from .test_progressive_planner_checkpoints import _RecordingEvidence
from .progressive_planner_fixtures import (
    _context, _foundation_payload, _materialization_payloads, _outline_payload,
)


def _run(monkeypatch, *, failures, code="progressive_primary_result_invalid", index=3):
    payloads = _materialization_payloads()
    accepted = []
    real_accept = owner._accept_compiled_plan

    def accept(**kwargs):
        plan = kwargs["plan"]
        accepted.append(len(plan.steps))
        if len(accepted) <= failures:
            raise ProgressivePlanCompileError(
                code, "Recheck the declared model product for this same question.",
                step_id=payloads[index]["step"]["step_id"], step_index=index,
                path="primary_result",
            )
        return real_accept(**kwargs)

    monkeypatch.setattr(owner, "_accept_compiled_plan", accept)
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        _outline_payload(), _foundation_payload(), *payloads,
        *payloads[index:], *payloads[index:],
    ]])
    agent = owner.ProgressivePlannerAgent(llm)
    context = _context()
    before = context.model_dump_json()
    return agent, llm, context, before, accepted


def test_final_finding_reaches_suffix_and_preserves_prefix(monkeypatch):
    agent, llm, context, before, accepted = _run(monkeypatch, failures=1)
    checkpoints = []
    plan = agent.run(context, checkpoint_callback=checkpoints.append)
    assert len(plan.steps) == 7
    assert accepted == [7, 7]
    assert len(llm.calls) == 13  # outline + foundation + 7 initial + 4 suffix
    assert "progressive_primary_result_invalid" in llm.calls[9][0][-1].content
    assert context.model_dump_json() == before
    complete = next(c for c in checkpoints if len(c.materializations) == 7)
    final = checkpoints[-1]
    assert complete.materializations[:3] == final.materializations[:3]
    assert all(c.outline == complete.outline for c in checkpoints)
    assert all(c.foundation == complete.foundation for c in checkpoints if c.foundation)
    assert agent.last_result.facts.prompt_metrics["suffix_revision_count"] == 1
    assert agent.last_result.facts.prompt_metrics["full_revision_count"] == 0


def test_final_repair_has_a_hard_bound(monkeypatch):
    agent, llm, context, before, accepted = _run(monkeypatch, failures=99)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(context)
    assert caught.value.reason_code == "progressive_primary_result_invalid"
    assert accepted == [7, 7, 7]
    assert len(llm.calls) == 17
    assert context.model_dump_json() == before


@pytest.mark.parametrize("code", [
    "progressive_design_input_structurally_unavailable",
    "progressive_step_outline_digest_mismatch",
    "progressive_unknown_owner_failure",
])
def test_non_planner_or_binding_failure_does_not_spend_on_suffix(monkeypatch, code):
    agent, llm, context, before, accepted = _run(monkeypatch, failures=1, code=code)
    with pytest.raises(ProgressivePlanCompileError):
        agent.run(context)
    assert accepted == [7]
    assert len(llm.calls) == 9
    assert context.model_dump_json() == before


@pytest.mark.parametrize("failures", [1, 2])
def test_revised_checkpoint_persists_and_loads_without_replacing_v1(monkeypatch, tmp_path, failures):
    agent, llm, context, before, accepted = _run(monkeypatch, failures=failures)
    evidence, paths, original = _RecordingEvidence(), [], {}

    def persist(cp):
        path = persist_progressive_planner_checkpoint(
            run_dir=tmp_path, evidence=evidence, checkpoint=cp, prompt_pack_version="test",
        )
        paths.append(path)
        if cp.sequence <= 8:
            original[path] = path.read_bytes()
            legacy = cp.model_dump(mode="json")
            assert "revision_offset" not in legacy and "repair_start_index" not in legacy
            assert canonical_sha256({k: v for k, v in legacy.items() if k != "checkpoint_sha256"}) == cp.checkpoint_sha256

    agent.run(context, checkpoint_callback=persist)
    loaded = load_progressive_planner_checkpoint_chain(
        last_checkpoint_path=paths[-1],
        expected_artifact_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
    )
    assert [cp.sequence for cp in loaded] == list(range(9 + 4 * failures))
    assert all(path.read_bytes() == content for path, content in original.items())
    assert all(cp.schema_version.endswith("/1") for cp in loaded[:9])
    assert all(cp.schema_version.endswith("/2") for cp in loaded[9:])
    assert all(cp.materializations[:3] == loaded[8].materializations[:3] for cp in loaded[9:])
    assert len(loaded[-1].prompt_metrics["step_materialization_schema_sha256"]) == 7 + 4 * failures
    assert len(loaded[-1].prompt_metrics["active_step_materialization_schema_sha256"]) == 7

    # Old request history must not bind a revised active prefix on resume.
    replay = loaded[-1].model_dump(mode="json")
    replay["prompt_metrics"]["step_materialization_schema_sha256"] = ["a" * 64] * (7 + 4 * failures)
    replay["checkpoint_sha256"] = canonical_sha256({k: v for k, v in replay.items() if k != "checkpoint_sha256"})
    revised = ProgressivePlannerCheckpoint.model_validate(replay)
    restored = restore_progressive_resume_prefix(
        checkpoint=revised, outline=revised.outline, foundation=revised.foundation.foundation,
        context=context, step_schema_authority=lambda *_: None,
        allowed_literature_citation_keys=(), allowed_know_how_decisions=None,
        reporting_method_source_keys=(),
    )
    assert restored.materializations == revised.materializations
    replay["prompt_metrics"]["active_step_materialization_schema_sha256"][3] = "b" * 64
    replay["checkpoint_sha256"] = canonical_sha256({k: v for k, v in replay.items() if k != "checkpoint_sha256"})
    with pytest.raises(ProgressivePlanCompileError) as rejected:
        restore_progressive_resume_prefix(
            checkpoint=ProgressivePlannerCheckpoint.model_validate(replay),
            outline=revised.outline, foundation=revised.foundation.foundation,
            context=context, step_schema_authority=lambda *_: None,
            allowed_literature_citation_keys=(), allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )
    assert rejected.value.reason_code == "progressive_resume_step_schema_authority_mismatch"

    # Even with a newly valid body digest and self-consistent retained-prefix
    # digest, changing the old prefix cannot cross the persisted parent chain.
    forged = loaded[9].model_dump(mode="json")
    forged["materializations"][0]["step"]["objective"] += " changed"
    forged["prompt_metrics"]["final_acceptance_repairs"][-1]["retained_materializations_sha256"] = canonical_sha256(forged["materializations"][:3])
    forged["checkpoint_sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "checkpoint_sha256"})
    forged_cp = ProgressivePlannerCheckpoint.model_validate(forged)
    with pytest.raises(ProgressivePlanningArtifactError, match="retained prefix"):
        persist_progressive_planner_checkpoint(
            run_dir=tmp_path, evidence=evidence, checkpoint=forged_cp, prompt_pack_version="test",
        )
    paths[9].write_text(json.dumps(forged))
    with pytest.raises(ProgressivePlanningArtifactError, match="retained prefix"):
        load_progressive_planner_checkpoint_chain(
            last_checkpoint_path=paths[9],
            expected_artifact_sha256=hashlib.sha256(paths[9].read_bytes()).hexdigest(),
        )


def test_missing_role_start_uses_declared_owners_not_primary_or_step_names():
    outline = SimpleNamespace(steps=[SimpleNamespace(step_id=f"s{i}") for i in range(7)])
    contract = SimpleNamespace(requirements=[
        SimpleNamespace(role="causal_protocol", module_id="target_trial_protocol"),
        SimpleNamespace(role="balance_positivity", module_id="baseline_balance"),
    ])
    plan = SimpleNamespace(analysis_type="causal_emulation", steps=[
        SimpleNamespace(step_id="misleading_causal_protocol", method="custom", scientific_action_id=None, expected_outputs=[]),
        SimpleNamespace(step_id="opaque_one", method="custom", scientific_action_id=None, expected_outputs=["artifact:causal_protocol"]),
        SimpleNamespace(step_id="opaque_two", method="custom", scientific_action_id=None, expected_outputs=["table:balance_positivity_diagnostics"]),
    ])
    roles = ["causal_protocol", "balance_positivity"]
    owners = article_role_repair_owners(plan, contract, roles)
    assert owners == {"causal_protocol": [1], "balance_positivity": [2]}
    error = ProgressivePlanCompileError(
        "progressive_article_required_roles_missing", "s6 mentions a later primary",
        step_id="s6", step_index=6,
        findings=[{"missing_roles": roles, "role_owner_indices": owners}],
    )
    start = final_acceptance_repair_start(error, outline)
    assert start == 1 and all(start <= min(indices) for indices in owners.values())


def test_unlocated_article_role_requires_full_fixed_outline_materialization():
    outline = SimpleNamespace(steps=[SimpleNamespace(step_id=f"s{i}") for i in range(7)])
    error = ProgressivePlanCompileError(
        "progressive_article_required_roles_missing", "s6",
        step_id="s6", step_index=6,
        findings=[{"missing_roles": ["causal_protocol", "balance_positivity"],
                   "role_owner_indices": {"causal_protocol": [1], "balance_positivity": []}}],
    )
    assert final_acceptance_repair_start(error, outline) == 0


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_complete_suffix_persists_final_artifacts_and_reloads_authority(monkeypatch, tmp_path, failures):
    agent, llm, context, _, _ = _run(monkeypatch, failures=failures)
    llm.supports_strict_json_schema = True
    evidence = _RecordingEvidence()
    plan = agent.run(context, checkpoint_callback=lambda cp: persist_progressive_planner_checkpoint(
        run_dir=tmp_path, evidence=evidence, checkpoint=cp, prompt_pack_version="test",
    ))
    facts = agent.last_result.facts
    metrics = dict(facts.prompt_metrics)
    assert len(metrics["step_materialization_schema_sha256"]) == 7 + 4 * failures
    if failures:
        assert len(metrics["active_step_materialization_schema_sha256"]) == 7
    else:
        assert "active_step_materialization_schema_sha256" not in metrics

    artifact_kwargs = dict(
        run_dir=tmp_path, evidence=evidence, outline=facts.outline,
        foundation=facts.foundation, materializations=facts.materializations,
        skeleton=facts.skeleton, compile_receipt=facts.compile_receipt,
        prompt_pack_version="test",
    )
    paths = persist_progressive_planning_artifacts(**artifact_kwargs, prompt_metrics=metrics)
    requests = [call[1]["structured_output"] for call in llm.calls]
    expected = (
        [r.authority_sha256 for r in requests[2:5]]
        + [r.authority_sha256 for r in requests[-4:]]
        if failures else [r.authority_sha256 for r in requests[2:]]
    )
    ledger = json.loads(paths.materializations.read_bytes())
    assert [e["structured_output_authority_sha256"] for e in ledger["materializations"]] == expected

    normalized = build_normalized_plan_lineage(
        proposed_plan=plan, proposed_source="llm_progressive_v2", pre_normalization_plan=plan,
        normalized_plan=plan, resume_scientific_semantics_changed=False,
        host_scientific_semantics_changed=False,
    )

    def register(name, content):
        path = tmp_path / f"{name}.json"
        path.write_text(content)
        evidence.register_file(evidence_id=name, source_path=path)
        return path

    register("planner_prompt_metrics", json.dumps(metrics))
    register("analysis_plan", plan.model_dump_json())
    register("plan_lifecycle_revision_0", normalized.model_dump_json())
    authority_kwargs = dict(
        run_dir=tmp_path, evidence=evidence,
        proposed_plan_sha256=normalized.proposed.plan_sha256,
        normalized_plan_sha256=normalized.plan_sha256,
        normalized_plan_authority_sha256=normalized.authority_sha256,
        normalized_plan_evidence_id="plan_lifecycle_revision_0",
        normalized_plan_filename="plan_lifecycle_revision_0.json", prompt_pack_version="test",
    )
    authority = persist_progressive_planning_authority(**authority_kwargs)
    assert authority.strict_transport_bound
    assert [s.structured_output_authority_sha256 for s in authority.ordered_steps] == expected
    authority_path = tmp_path / "progressive_planning_authority.json"
    sealed_bytes = authority_path.read_bytes()
    assert ProgressivePlanningAuthority.model_validate_json(sealed_bytes) == authority
    assert persist_progressive_planning_authority(**authority_kwargs) == authority
    assert authority_path.read_bytes() == sealed_bytes

    if failures:
        missing_active = {k: v for k, v in metrics.items() if k != "active_step_materialization_schema_sha256"}
        with pytest.raises(ProgressivePlanningArtifactError) as missing:
            persist_progressive_planning_artifacts(**artifact_kwargs, prompt_metrics=missing_active)
        assert missing.value.reason_code == "progressive_step_schema_authority_count_mismatch"
        register("planner_prompt_metrics", json.dumps(missing_active))
        with pytest.raises(ProgressivePlanningArtifactError) as missing:
            persist_progressive_planning_authority(**authority_kwargs)
        assert missing.value.reason_code == "progressive_step_schema_authority_count_mismatch"
        # A valid-looking digest that disagrees with the ledger still fails on
        # final authority readback; history must not silently replace it.
        drifted = {**metrics, "active_step_materialization_schema_sha256": [*expected[:3], "a" * 64, *expected[4:]]}
        register("planner_prompt_metrics", json.dumps(drifted))
        with pytest.raises(ProgressivePlanningArtifactError) as drift:
            persist_progressive_planning_authority(**authority_kwargs)
        assert drift.value.reason_code == "progressive_schema_authority_mismatch"
