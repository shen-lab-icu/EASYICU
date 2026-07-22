from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.development_repair_framework import (
    DevelopmentRepairProtocolError,
    evaluate_development_repair_readiness,
    load_development_repair_protocol,
    render_development_repair_report,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS


def test_current_repair_report_is_exact_nine_and_never_authorizes_real_run() -> None:
    rows, protocol_sha, binding_sha = evaluate_development_repair_readiness()
    report = render_development_repair_report(
        rows,
        repair_protocol_sha256=protocol_sha,
        input_binding_sha256=binding_sha,
    )
    assert tuple(row.task_id for row in rows) == tuple(FIGURE2_TASK_IDS)
    assert report["real_run_authorized"] is False
    assert all(not row.launch_ready for row in rows)
    assert all(row.state == "blocked" for row in rows)


def test_framework_keeps_owner_and_scientific_blocks_out_of_auto_repair() -> None:
    rows, _, _ = evaluate_development_repair_readiness()
    by_id = {row.task_id: row for row in rows}
    e2 = {
        requirement.code: requirement
        for requirement in by_id["e2_lactate_mortality"].requirements
    }
    h2 = {
        requirement.code: requirement
        for requirement in by_id["h2_vasopressor_causal"].requirements
    }
    h3 = {
        requirement.code: requirement
        for requirement in by_id["h3_trajectory_clustering"].requirements
    }
    assert e2["LACTATE_PROTOCOL_CARD_REQUIRED"].work_kind == "case_protocol"
    assert h2["EXPOSURE_DATA_CONTRACT_REQUIRED"].auto_action is False
    assert h2["SCIENTIFIC_FEASIBILITY_REQUIRED"].work_kind == "scientific_redesign"
    assert h3["STABILITY_REDESIGN_REQUIRED"].auto_action is False


def test_protocol_rejects_auto_action_for_human_owned_work(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[3]
        / "benchmarks"
        / "figure2_canonical9"
        / "development_repair_protocol_v1.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["tasks"][0]["requirements"][0]["auto_action"] = True
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DevelopmentRepairProtocolError, match="must not auto-act"):
        load_development_repair_protocol(path)


def test_protocol_rejects_changed_task_order(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[3]
        / "benchmarks"
        / "figure2_canonical9"
        / "development_repair_protocol_v1.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["tasks"][0], payload["tasks"][1] = payload["tasks"][1], payload["tasks"][0]
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DevelopmentRepairProtocolError, match="exact Canonical9 task order"
    ):
        load_development_repair_protocol(path)
