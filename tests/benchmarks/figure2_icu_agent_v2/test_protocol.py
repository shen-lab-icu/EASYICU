from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.figure2_icu_agent_v2.protocol import (
    ACTION_SPACE_PATH,
    DEV9_TASK_IDS,
    EXPERIMENT_PROTOCOL_PATH,
    HELDOUT27_TASK_IDS,
    HELDOUT_TASKBANK_PATH,
    QUALIFICATION12_TASK_IDS,
    SCORING_DIMENSIONS,
    BenchmarkContractError,
    load_action_space,
    load_experiment_protocol,
    load_heldout_taskbank,
    validate_experiment_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebind_taskbank_digest(protocol_path: Path, taskbank_path: Path) -> None:
    payload = json.loads(EXPERIMENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["splits"][2]["taskbank_sha256"] = _sha256(taskbank_path)
    protocol_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_versioned_bundle_has_exact_split_and_coverage_contract() -> None:
    receipt = validate_experiment_bundle()
    protocol = load_experiment_protocol()
    action_space = load_action_space()
    taskbank = load_heldout_taskbank()

    assert receipt.dev_task_count == 9
    assert receipt.qualification_task_count == 12
    assert receipt.heldout_task_count == 27
    assert receipt.action_space_sha256 == _sha256(ACTION_SPACE_PATH)
    assert receipt.heldout_taskbank_sha256 == _sha256(HELDOUT_TASKBANK_PATH)
    assert tuple(protocol.splits[0].task_ids) == DEV9_TASK_IDS
    assert tuple(protocol.splits[1].task_ids) == QUALIFICATION12_TASK_IDS
    assert tuple(protocol.splits[2].task_ids) == HELDOUT27_TASK_IDS
    assert tuple(protocol.scoring_dimensions) == SCORING_DIMENSIONS
    assert tuple(protocol.formal_run_policy.arms) == ("aware",)
    assert protocol.formal_run_policy.primary_runs_per_task == 1
    assert protocol.formal_run_policy.failures_remain_in_denominator is True
    assert len(action_space.stages) == 11
    assert Counter(task.difficulty for task in taskbank.tasks) == {
        "basic": 9,
        "intermediate": 9,
        "advanced": 9,
    }
    assert Counter(task.database for task in taskbank.tasks) == {
        "miiv": 5,
        "mimic": 4,
        "eicu": 6,
        "aumc": 5,
        "hirid": 3,
        "sic": 4,
    }


def test_heldout_items_are_answer_free_and_item_scoped() -> None:
    raw_rows = [
        json.loads(line)
        for line in HELDOUT_TASKBANK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    forbidden_fields = {
        "gold_answer",
        "expected_or_direction",
        "expected_effect_direction",
        "numeric_targets",
        "reference_result",
    }

    assert len(raw_rows) == 27
    assert all(forbidden_fields.isdisjoint(row) for row in raw_rows)
    assert all(row["agent_visibility"] == "item_only_at_run" for row in raw_rows)
    assert all(row["paper_authority_before_freeze"] is False for row in raw_rows)
    assert len({row["question"] for row in raw_rows}) == 27


def test_every_action_stage_names_a_real_owner_and_stable_failure() -> None:
    repo_root = ACTION_SPACE_PATH.parents[2]
    action_space = load_action_space()

    for stage in action_space.stages:
        assert (repo_root / stage.owner).exists(), stage.owner
        assert stage.failure_reason_codes
        assert len(stage.failure_reason_codes) == len(set(stage.failure_reason_codes))


def test_taskbank_byte_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "heldout27.jsonl"
    changed.write_bytes(HELDOUT_TASKBANK_PATH.read_bytes() + b"\n")

    with pytest.raises(BenchmarkContractError) as exc_info:
        validate_experiment_bundle(taskbank_path=changed)

    assert exc_info.value.reason_code == "HELDOUT_TASKBANK_DIGEST_MISMATCH"


def test_action_space_omission_fails_even_when_taskbank_digest_is_rebound(
    tmp_path: Path,
) -> None:
    rows = [
        json.loads(line)
        for line in HELDOUT_TASKBANK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["required_stages"] = rows[0]["required_stages"][:-1]
    taskbank_path = tmp_path / "heldout27.jsonl"
    taskbank_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"
    _rebind_taskbank_digest(protocol_path, taskbank_path)

    with pytest.raises(BenchmarkContractError) as exc_info:
        validate_experiment_bundle(
            protocol_path=protocol_path,
            taskbank_path=taskbank_path,
        )

    assert exc_info.value.reason_code == "TASK_ACTION_SPACE_INCOMPLETE"


def test_protocol_rejects_more_than_the_aware_primary_arm(tmp_path: Path) -> None:
    payload = json.loads(EXPERIMENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["formal_run_policy"]["arms"] = ["aware", "aware"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(BenchmarkContractError) as exc_info:
        load_experiment_protocol(path)

    assert exc_info.value.reason_code == "EXPERIMENT_PROTOCOL_INVALID"


def test_duplicate_taskbank_json_key_fails_closed(tmp_path: Path) -> None:
    first, *remaining = HELDOUT_TASKBANK_PATH.read_text(encoding="utf-8").splitlines()
    duplicate = first.replace(
        '"task_id":"icu27_t01"',
        '"task_id":"icu27_t01","task_id":"icu27_t01"',
        1,
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join([duplicate, *remaining]) + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkContractError) as exc_info:
        load_heldout_taskbank(path)

    assert exc_info.value.reason_code == "HELDOUT_TASKBANK_INVALID"
    assert "duplicate JSON key" in exc_info.value.detail


def test_symlinked_authority_file_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "taskbank.jsonl"
    link.symlink_to(HELDOUT_TASKBANK_PATH)

    with pytest.raises(BenchmarkContractError) as exc_info:
        load_heldout_taskbank(link)

    assert exc_info.value.reason_code == "BENCHMARK_AUTHORITY_PATH_INVALID"
