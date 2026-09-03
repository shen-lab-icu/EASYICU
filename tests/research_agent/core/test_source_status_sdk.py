"""Host-owned source-status materialization authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.source_status import SourceStatusContract
from easyicu.research_agent.authority.source_status_sdk import (
    SourceStatusMaterializationError,
    materialize_verified_absence,
)
from easyicu.research_agent.authority.typed_input_receipt import (
    typed_input_row_identity_sha256,
)
from easyicu.research_agent.authority.typed_input_sdk import load_typed_input

INPUT_KEY = "artifact:source_status"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path):
    run_root = tmp_path / "run"
    evidence = run_root / "evidence"
    resolved = run_root / "resolved_inputs"
    evidence.mkdir(parents=True)
    resolved.mkdir()
    frame = pd.DataFrame(
        {
            "stay_id": [11, 12, 13, 14],
            "vaso_ind_max": [1.0, None, None, None],
            "vaso_ind_max_n": [2, 0, 0, 0],
            "vaso_ind_source_status": [
                "observed",
                "verified_absent",
                "unmeasured",
                "source_missing",
            ],
        }
    )
    artifact = evidence / "status.parquet"
    frame.to_parquet(artifact, index=False)
    artifact_sha = _sha(artifact)
    identity_sha = typed_input_row_identity_sha256(frame["stay_id"])
    identity = {
        "input_key": INPUT_KEY,
        "declared_kind": "artifact",
        "product": "source_status",
        "evidence_id": "ev_status",
        "sha256": artifact_sha,
        "produced_by_step": "host_intake",
    }
    binding = {
        "evidence_id": "ev_status",
        "declared_kind": "artifact",
        "product": "source_status",
        "evidence_kind": "table",
        "relative_path": artifact.relative_to(run_root).as_posix(),
        "absolute_path": str(artifact),
        "sha256": artifact_sha,
        "produced_by_step": "host_intake",
        "identity_row": identity,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v2",
            "identity_row": identity,
            "tabular_format": "parquet",
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "row_identity_column": "stay_id",
            "row_count": len(frame),
            "row_identity_sha256": identity_sha,
        },
    }
    manifest_payload = {
        "schema_version": "2.1",
        "step_id": "source_materialization",
        "planner_declared_inputs": [INPUT_KEY],
        "inputs": {INPUT_KEY: binding},
    }
    manifest = resolved / "source_materialization.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_typed_input(
        resolved_inputs_path=manifest,
        expected_resolved_inputs_sha256=_sha(manifest),
        run_root=run_root,
        input_key=INPUT_KEY,
        consumer_step_id="source_materialization",
        consumer_code_sha256="c" * 64,
    )
    contract = SourceStatusContract.model_validate(
        {
            "schema_version": "easyicu.source_status_contract/1",
            "variable": "vaso_ind_max",
            "n_total": 4,
            "counts": {
                "observed": 1,
                "verified_absent": 1,
                "unmeasured": 1,
                "source_missing": 1,
                "contradictory": 0,
            },
            "source_coverage": "partial",
            "verified_absent_value": 0,
            "authority_kind": "event_reconciliation",
            "authority_evidence_sha256": "a" * 64,
            "source_columns": ["vaso_ind_max", "vaso_ind_max_n"],
            "row_status_artifact_sha256": artifact_sha,
            "row_status_column": "vaso_ind_source_status",
            "row_identity_sha256": identity_sha,
        }
    )
    return loaded, contract


def test_materializes_only_verified_absence_and_preserves_unknowns(tmp_path: Path):
    loaded, contract = _case(tmp_path)

    result = materialize_verified_absence(
        source_input=loaded,
        contract=contract,
        value_column="vaso_ind_max",
    )

    values = result.to_pandas()["vaso_ind_max"].tolist()
    assert values[:2] == [1.0, 0.0]
    assert pd.isna(values[2]) and pd.isna(values[3])
    assert result.receipt.counts == contract.counts
    assert result.receipt.source_input_receipt_sha256 == loaded.receipt.receipt_sha256


def test_wrong_artifact_binding_fails_closed(tmp_path: Path):
    loaded, contract = _case(tmp_path)
    wrong = contract.model_copy(update={"row_status_artifact_sha256": "d" * 64})

    with pytest.raises(SourceStatusMaterializationError, match="artifact"):
        materialize_verified_absence(
            source_input=loaded,
            contract=wrong,
            value_column="vaso_ind_max",
        )


def test_wrong_row_identity_fails_closed(tmp_path: Path):
    loaded, contract = _case(tmp_path)
    wrong = contract.model_copy(update={"row_identity_sha256": "d" * 64})

    with pytest.raises(SourceStatusMaterializationError, match="row identity"):
        materialize_verified_absence(
            source_input=loaded,
            contract=wrong,
            value_column="vaso_ind_max",
        )


def test_cannot_materialize_a_different_value_column(tmp_path: Path):
    loaded, contract = _case(tmp_path)

    with pytest.raises(SourceStatusMaterializationError, match="contract-bound"):
        materialize_verified_absence(
            source_input=loaded,
            contract=contract,
            value_column="vaso_ind_max_n",
        )


def test_row_level_counts_must_match_contract(tmp_path: Path):
    loaded, contract = _case(tmp_path)
    wrong = contract.model_copy(
        update={
            "counts": contract.counts.model_copy(
                update={"verified_absent": 0, "unmeasured": 2}
            )
        }
    )

    with pytest.raises(SourceStatusMaterializationError, match="contract counts"):
        materialize_verified_absence(
            source_input=loaded,
            contract=wrong,
            value_column="vaso_ind_max",
        )
