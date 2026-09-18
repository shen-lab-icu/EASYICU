"""A historical native label cannot authorize Coder-repaired result reuse."""

from __future__ import annotations

import copy
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.run_input import (
    invalidate_unverified_successful_steps,
)


def _step(
    root, store, step_id, *, mode="deterministic_standard", metadata=None, inputs=()
):
    script_path = root / f"{step_id}.py"
    script_path.write_text("print('sealed result')\n")
    script = store.register_file(
        kind="code",
        description="Executed script",
        source_path=script_path,
        evidence_id=f"{step_id}_code",
        produced_by_step=step_id,
        generation_mode=mode,
        metadata=metadata or {},
    )
    summary_path = root / f"{step_id}.json"
    summary_path.write_text('{"estimate": 1}')
    summary = store.register_file(
        kind="statistic",
        description="Executed summary",
        source_path=summary_path,
        evidence_id=f"{step_id}_summary",
        produced_by_step=step_id,
        script_evidence_id=script.evidence_id,
        inputs=list(inputs),
    )
    return {
        "step_id": step_id,
        "status": "ok",
        "generation_mode": mode,
        "evidence_ids": [script.evidence_id, summary.evidence_id],
        "script_evidence_id": script.evidence_id,
        "step_summary_evidence_id": summary.evidence_id,
    }


def _revalidate(root, store, checkpoints):
    state = {"per_step_records": checkpoints, "findings": []}
    original = copy.deepcopy(state)
    files = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    records = {r.evidence_id: r.model_dump(mode="json") for r in store.records()}
    result = invalidate_unverified_successful_steps(
        run_dir=root,
        resume_state=state,
        records=records,
    )
    assert state == original
    assert files == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    return result


@pytest.mark.parametrize("flag_location", ["checkpoint", "script", "both"])
def test_native_repair_invalidates_only_affected_dependency_chain(
    tmp_path, flag_location
):
    store = EvidenceStore(tmp_path)
    prefix = _step(tmp_path, store, "valid_prefix")
    repaired = _step(
        tmp_path,
        store,
        "repaired_native",
        metadata={"llm_repair_used": True} if flag_location != "checkpoint" else {},
    )
    if flag_location != "script":
        repaired["llm_repair_used"] = True
    downstream = _step(tmp_path, store, "plot", inputs=("repaired_native_summary",))
    final = _step(tmp_path, store, "report", inputs=("plot_summary",))
    updated, invalidated = _revalidate(
        tmp_path, store, [prefix, repaired, downstream, final]
    )
    assert set(invalidated) == {"repaired_native", "plot", "report"}
    assert "Coder repair" in invalidated["repaired_native"]
    assert updated["per_step_records"][:4] == [prefix, repaired, downstream, final]


@pytest.mark.parametrize(
    "mode, metadata",
    [
        ("llm", {"llm_repair_used": True}),
        ("repaired", {"llm_repair_used": True}),
        ("deterministic_standard", {"llm_repair_used": False, "repair_attempts": 1}),
        ("deterministic_standard", {}),
    ],
)
def test_generated_or_host_only_repair_is_not_reclassified(tmp_path, mode, metadata):
    store = EvidenceStore(tmp_path)
    checkpoint = _step(tmp_path, store, "model", mode=mode, metadata=metadata)
    _, invalidated = _revalidate(tmp_path, store, [checkpoint])
    assert invalidated == {}


def test_new_coder_revision_does_not_claim_its_native_drafting_source(tmp_path):
    store = EvidenceStore(tmp_path)
    checkpoint = _step(tmp_path, store, "model")
    (tmp_path / "model.py").write_text("print('new audited agent revision')\n")
    script = store.register_file(
        kind="code",
        description="Repaired script",
        source_path=tmp_path / "model.py",
        evidence_id="new_agent_revision",
        produced_by_step="model",
        generation_mode="repaired",
        metadata={
            "llm_repair_used": True,
            "resumed_code_evidence_id": "model_code",
            "resumed_from_generation_mode": "deterministic_standard",
        },
    )
    summary = store.register_file(
        kind="statistic",
        description="New execution summary",
        source_path=tmp_path / "model.json",
        evidence_id="new_agent_summary",
        produced_by_step="model",
        script_evidence_id=script.evidence_id,
    )
    checkpoint.update(
        generation_mode="repaired",
        script_evidence_id=script.evidence_id,
        step_summary_evidence_id=summary.evidence_id,
        evidence_ids=[script.evidence_id, summary.evidence_id],
    )
    _, invalidated = _revalidate(tmp_path, store, [checkpoint])
    assert invalidated == {}


def test_repeated_resume_retains_native_repair_provenance(tmp_path):
    store = EvidenceStore(tmp_path)
    checkpoint = _step(tmp_path, store, "model", metadata={"llm_repair_used": True})
    old_id = checkpoint["script_evidence_id"]
    for index in range(2):
        script = store.register_file(
            kind="code",
            description="Reused script",
            source_path=tmp_path / "model.py",
            evidence_id=f"reuse_{index}",
            produced_by_step="model",
            generation_mode="resumed_code_reuse",
            metadata={"resumed_code_evidence_id": old_id, "llm_repair_used": False},
        )
        old_id = script.evidence_id
    checkpoint.update(generation_mode="resumed_code_reuse", script_evidence_id=old_id)
    checkpoint["evidence_ids"].append(old_id)
    _, invalidated = _revalidate(tmp_path, store, [checkpoint])
    assert "Coder repair" in invalidated["model"]


@pytest.mark.parametrize("flag", ["false", "true", 0, 1])
def test_native_repair_flag_must_be_boolean(tmp_path, flag):
    store = EvidenceStore(tmp_path)
    checkpoint = _step(tmp_path, store, "model", metadata={"llm_repair_used": flag})
    _, invalidated = _revalidate(tmp_path, store, [checkpoint])
    assert "invalid repair provenance" in invalidated["model"]


@pytest.mark.parametrize("source", ["missing", "wrong_step", "different_code", "cycle"])
def test_unverifiable_reuse_cannot_hide_native_origin(tmp_path, source):
    store = EvidenceStore(tmp_path)
    checkpoint = _step(tmp_path, store, "model")
    source_id = "model_code"
    if source == "missing":
        source_id = "absent"
    elif source == "wrong_step":
        source_id = _step(tmp_path, store, "other")["script_evidence_id"]
    elif source == "different_code":
        (tmp_path / "model.py").write_text("print('changed')\n")
    elif source == "cycle":
        source_id = "reused"
    script = store.register_file(
        kind="code",
        description="Reused script",
        source_path=tmp_path / "model.py",
        evidence_id="reused",
        produced_by_step="model",
        generation_mode="resumed_code_reuse",
        metadata={"resumed_code_evidence_id": source_id},
    )
    checkpoint.update(
        generation_mode="resumed_code_reuse", script_evidence_id=script.evidence_id
    )
    checkpoint["evidence_ids"].append(script.evidence_id)
    _, invalidated = _revalidate(tmp_path, store, [checkpoint])
    assert "script reuse provenance" in invalidated["model"]
