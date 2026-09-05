"""Current-contract reading preserves scope, content identity, and failures."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path

import pytest

from easyicu.research_agent.figures.contracts import FigureContractInventory


def _write(root, directory="publication_figures", name="current", payload=None):
    path = root / directory / f"{name}.figure_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload
            or {
                "figure_id": name,
                "panels": [
                    {
                        "panel_id": "A",
                        "role": "data_quality",
                        "metadata": {"source": ["sealed.csv"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _record(step_id, *, files=None, status="ok"):
    record = {"step_id": step_id, "status": status}
    if files is not None:
        record["step_summary"] = {"contract_files": files}
    return record


def test_snapshot_is_deeply_immutable_and_bound_to_the_bytes_observed(tmp_path):
    path = _write(tmp_path)
    content = path.read_bytes()
    inventory = FigureContractInventory.load(tmp_path, per_step_records=[])
    snapshot = inventory.snapshots[0]
    assert snapshot.sha256 == sha256(content).hexdigest()
    with pytest.raises(FrozenInstanceError):
        snapshot.sha256 = "forged"
    with pytest.raises(TypeError):
        snapshot._payload["panels"][0]["metadata"]["source"] = ()
    projection = snapshot.to_payload()
    projection["panels"][0]["metadata"]["source"].append("another.csv")
    assert snapshot.to_payload()["panels"][0]["metadata"]["source"] == ["sealed.csv"]
    path.write_text('{"figure_id":"later","panels":[]}', encoding="utf-8")
    assert inventory.snapshots[0].to_payload()["figure_id"] == "current"
    later = FigureContractInventory.load(tmp_path, per_step_records=[])
    assert later.snapshots[0].sha256 != snapshot.sha256
    assert later.snapshots[0].to_payload()["figure_id"] == "later"


def test_same_pass_reuse_does_not_read_files_again(tmp_path, monkeypatch):
    _write(tmp_path)
    inventory = FigureContractInventory.load(tmp_path, per_step_records=[])
    def forbidden(_path):
        raise AssertionError("a reporting-pass inventory must not re-read the file")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert FigureContractInventory.load(tmp_path, per_step_records=[], current=inventory) is inventory


def test_current_selection_and_primary_lineage_projection_share_one_inventory(tmp_path):
    _write(tmp_path, name="publication")
    current = _write(tmp_path, "steps/current/outputs")
    _write(tmp_path, "steps/current/outputs", "superseded")
    _write(tmp_path, "steps/failed/outputs")
    _write(tmp_path, "steps/absent/outputs")
    _write(tmp_path, "steps/explicit_empty/outputs")
    _write(tmp_path, "steps/legacy/outputs", "legacy")
    records = [
        _record("current", files=[current.name]),
        _record("failed", status="failed"),
        _record("explicit_empty", files=[]),
        _record("legacy"),
    ]
    inventory = FigureContractInventory.load(tmp_path, per_step_records=records)
    assert {item.to_payload()["figure_id"] for item in inventory.snapshots} == {
        "publication",
        "current",
        "legacy",
    }
    assert inventory.texts(allowed_step_ids={"current"}) == (
        "current\nA\ndata_quality",
    )
    assert inventory.texts(allowed_step_ids=set()) == ()
    assert (
        FigureContractInventory.load(
            tmp_path, per_step_records=list(reversed(records)), current=inventory
        )
        is inventory
    )
    with pytest.raises(ValueError, match="figure_contract_inventory_scope_mismatch"):
        FigureContractInventory.load(tmp_path, per_step_records=[], current=inventory)
    with pytest.raises(ValueError, match="figure_contract_inventory_scope_mismatch"):
        FigureContractInventory.load(
            tmp_path / "different_run", per_step_records=records, current=inventory
        )


@pytest.mark.parametrize(
    "content,reason",
    [
        (b"{broken", "figure_contract_invalid_json"),
        (b"\xff", "figure_contract_invalid_json"),
        (b"[]", "figure_contract_invalid_shape"),
        (b'{"panels":{}}', "figure_contract_invalid_shape"),
        (b'{"panels":[null]}', "figure_contract_invalid_shape"),
        (b'{"value":NaN}', "figure_contract_invalid_shape"),
    ],
)
def test_read_failure_is_distinct_from_an_absent_contract(tmp_path, content, reason):
    assert FigureContractInventory.load(tmp_path).errors == ()
    path = _write(tmp_path)
    path.write_bytes(content)
    inventory = FigureContractInventory.load(tmp_path)
    assert inventory.paths == (path,)
    assert inventory.snapshots == ()
    assert len(inventory.errors) == 1
    assert inventory.errors[0].reason_code == reason
    assert path.name in inventory.error_messages()[0]


def test_unreadable_current_contract_keeps_the_lower_layer_cause(monkeypatch, tmp_path):
    path = _write(tmp_path)

    def denied(_path):
        raise PermissionError("contract is not readable")

    monkeypatch.setattr(Path, "read_bytes", denied)
    inventory = FigureContractInventory.load(tmp_path)
    assert inventory.errors[0].path == path
    assert inventory.errors[0].reason_code == "figure_contract_unreadable"
    assert "contract is not readable" in inventory.errors[0].detail


def test_contract_symlink_cannot_import_another_runs_content(tmp_path):
    path = _write(tmp_path / "other_run")
    run = tmp_path / "current_run"
    directory = run / "publication_figures"
    directory.mkdir(parents=True)
    (directory / path.name).symlink_to(path)
    inventory = FigureContractInventory.load(run)
    assert inventory.snapshots == ()
    assert inventory.errors[0].reason_code == "figure_contract_outside_run"
