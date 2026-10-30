"""Wave receipt propagation through the real step executor and host runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runner import (
    CodeRunner,
    WaveArtifactAuthority,
    capture_wave_artifact_authority,
)
from easyicu.research_agent.execution.step_execution import (
    LockedStepExecutionRequest,
    StepExecutor,
)


_PROBE = """
import json, os
from pathlib import Path
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _run_robustness_preflight_from_env,
)
snapshot = os.environ.get('EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT')
result = {
    'receipt': json.loads(Path(snapshot).read_text()) if snapshot else None,
    'error': os.environ.get('EASYICU_RUN_ARTIFACT_AUTHORITY_ERROR'),
}
Path(os.environ['STEP_OUT_DIR'], 'probe.json').write_text(json.dumps(result))
"""


def _write_checkpoint(root: Path, *, sequence: int, records: list[dict]) -> None:
    (root / "manifest_partial.json").write_text(json.dumps({
        "run_id": "wave-run", "checkpoint_sequence": sequence,
        "per_step_records": records, "evidence": [],
    }), encoding="utf-8")


def test_workers_use_same_predispatch_receipt_after_live_checkpoint_changes(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort, index=False)
    prior = [{"step_id": "prior", "status": "ok", "evidence_ids": []}]
    _write_checkpoint(root, sequence=7, records=prior)
    wave = capture_wave_artifact_authority(
        workdir=root, run_id="wave-run", step_records=prior,
        evidence_records=[], resumed=False,
    )
    assert not wave.error
    _write_checkpoint(root, sequence=8, records=prior + [{"step_id": "peer", "status": "ok"}])
    runner = CodeRunner(workdir=root, cohort_parquet=cohort, timeout_seconds=15,
                        allow_unsafe_host_fallback=True)
    executor = StepExecutor(clear_output_dir=lambda path: None)
    receipts = []
    for step_id in ("first", "second", "first"):
        result = executor.execute(runner=runner, request=LockedStepExecutionRequest(
            step_id=step_id, code=_PROBE, resolved_inputs_path=None,
            output_dir=root / "steps" / step_id / "outputs", wave_authority=wave,
        ))
        assert result.succeeded
        payload = json.loads((result.out_dir / "probe.json").read_text())
        assert payload["error"] is None
        receipts.append(payload["receipt"])
    assert receipts[0] == receipts[1] == receipts[2]
    assert receipts[0]["checkpoint_sequence"] == 7
    assert receipts[0]["authority"]["per_step_records"] == prior


@pytest.mark.parametrize("condition", ["tampered", "missing", "invalid_payload"])
def test_invalid_wave_receipt_never_starts_or_reads_cohort(
    tmp_path: Path, condition: str, monkeypatch: pytest.MonkeyPatch,
):
    from easyicu.research_agent.execution import runner as runner_module

    root = tmp_path / "run"
    root.mkdir()
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"not read")
    _write_checkpoint(root, sequence=7, records=[])
    runner = CodeRunner(workdir=root, cohort_parquet=cohort, timeout_seconds=15,
                        allow_unsafe_host_fallback=True)
    if condition == "tampered":
        wave = WaveArtifactAuthority(payload=b"tampered", sha256="0" * 64)
    elif condition == "invalid_payload":
        wave = WaveArtifactAuthority(payload=b"{}", sha256=hashlib.sha256(b"{}").hexdigest())
    else:
        wave = WaveArtifactAuthority(error="receipt unavailable")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid authority must not read cohort or launch a process")

    monkeypatch.setattr(runner_module, "_parquet_row_count", forbidden)
    monkeypatch.setattr(runner_module.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="Wave authority receipt"):
        runner.run(step_id="worker", code=_PROBE, wave_authority=wave)
    deleted = []
    executor = StepExecutor(clear_output_dir=lambda path: deleted.append(path))
    with pytest.raises(ValueError, match="Wave authority receipt"):
        executor.execute(runner=runner, request=LockedStepExecutionRequest(
            step_id="worker", code=_PROBE, resolved_inputs_path=None,
            output_dir=root / "steps" / "worker" / "outputs", wave_authority=wave,
        ))
    assert deleted == []
    assert not (root / "steps").exists()


@pytest.mark.parametrize("change", ["run_id", "steps", "evidence", "resume"])
def test_wave_capture_refuses_unmatched_or_unrecoverable_authority(tmp_path: Path, change: str):
    _write_checkpoint(tmp_path, sequence=1, records=[])
    kwargs = dict(workdir=tmp_path, run_id="wave-run", step_records=[], evidence_records=[], resumed=False)
    if change == "run_id":
        kwargs["run_id"] = "other"
    elif change == "steps":
        kwargs["step_records"] = [{"step_id": "other", "status": "ok"}]
    elif change == "evidence":
        from easyicu.research_agent.authority.evidence_store import EvidenceStore
        store = EvidenceStore(tmp_path)
        store.register_text(kind="log", description="late", text="x", filename="late.txt")
        kwargs["evidence_records"] = store.records()
    else:
        kwargs["resumed"] = True
    wave = capture_wave_artifact_authority(**kwargs)
    assert wave.error
    assert not wave.payload


def test_unsupported_runner_is_refused_before_output_cleanup(tmp_path: Path):
    deleted = []
    executor = StepExecutor(clear_output_dir=lambda path: deleted.append(path))
    with pytest.raises(ValueError, match="cannot receive"):
        executor.execute(runner=object(), request=LockedStepExecutionRequest(
            step_id="worker", code=_PROBE, resolved_inputs_path=None, output_dir=tmp_path,
            wave_authority=WaveArtifactAuthority(error="missing"),
        ))
    assert deleted == []
