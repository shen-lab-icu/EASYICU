"""Reviewed figure-source canonicalization and atomic installation contracts."""

from __future__ import annotations

import json

import pytest


def test_consistent_local_figure_source_descriptor_is_canonicalized_for_consumers(
    tmp_path,
):
    from easyicu.research_agent.discovery.discovery_package import _string_list
    from easyicu.research_agent.figures.skill import (
        _contract_payload_source_references,
    )
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "result_source_data.csv").write_text("x,y\na,1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [
                    {
                        "file": "result_source_data.csv",
                        "filename": "result_source_data.csv",
                        "path": "result_source_data.csv",
                        "relative_path": "result_source_data.csv",
                        "kind": "table",
                        "evidence_ids": [],
                    }
                ],
                "panels": [],
            }
        ),
        encoding="utf-8",
    )

    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    _before, canonical_text, names = candidate
    assert names == ["result_source_data.csv"]
    _install_figure_contract_source_data_canonicalization(
        contract_path=contract_path,
        expected_before=_before,
        canonical_text=canonical_text,
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload["source_data"] == ["result_source_data.csv"]
    assert _contract_payload_source_references(payload) == ["result_source_data.csv"]
    assert _string_list(payload["source_data"]) == ["result_source_data.csv"]


def test_figure_contract_canonicalization_does_not_follow_predictable_temp_symlink(
    tmp_path,
):
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [{"file": "source.csv", "path": "source.csv"}],
            }
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.json"
    outside.write_text("do-not-touch", encoding="utf-8")
    predictable = out_dir / ".result.figure_contract.json.schema.tmp"
    try:
        predictable.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    before, after, _names = candidate
    _install_figure_contract_source_data_canonicalization(
        contract_path=contract_path,
        expected_before=before,
        canonical_text=after,
    )

    assert outside.read_text(encoding="utf-8") == "do-not-touch"
    assert json.loads(contract_path.read_text(encoding="utf-8"))["source_data"] == [
        "source.csv"
    ]


def test_figure_contract_canonicalization_rejects_changed_reviewed_contract(
    tmp_path,
):
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [{"file": "source.csv", "path": "source.csv"}],
            }
        ),
        encoding="utf-8",
    )
    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    before, after, _names = candidate
    contract_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after canonicalization review"):
        _install_figure_contract_source_data_canonicalization(
            contract_path=contract_path,
            expected_before=before,
            canonical_text=after,
        )


@pytest.mark.parametrize(
    "source_data",
    [
        [{"file": "source.csv", "path": "other.csv"}],
        [{"file": 7}],
        [{"file": "/tmp/source.csv"}],
        [{"file": "nested/source.csv"}],
        [{"evidence_id": "table_source"}],
        [["source.csv"]],
    ],
)
def test_figure_source_descriptor_canonicalization_fails_closed(
    tmp_path,
    source_data,
):
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    (out_dir / "other.csv").write_text("x\n2\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps({"figure_id": "result", "source_data": source_data}),
        encoding="utf-8",
    )

    assert (
        _figure_contract_source_data_canonicalization_candidate(
            contract_path=contract_path,
            out_dir=out_dir,
        )
        is None
    )
