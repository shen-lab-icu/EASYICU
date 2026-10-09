"""A static prediction run answers with its primary model's registered performance.

The host binds the one ``table:model_performance`` row of the plan's primary
prediction step to its evidence, as the kernel's typed binding binds a table
product, and projects it with the host's own authority.  Tables of other steps
that also carry an ``auroc`` column are never read, and a run whose row cannot
be bound gets no block.

Synthetic run records; no benchmark item.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.prediction_performance_projection import (
    prediction_performance_projection,
)
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)

_ROW = {
    "model": "logistic_regression_l2",
    "authority_scope": "analysis_only",
    "paper_authorization_allowed": "False",
    "development_n": "800",
    "validation_n": "200",
    "validation_subject_n": "190",
    "validation_event_n": "30",
    "validation_event_rate": "0.15",
    "auroc": "0.8123",
    "auroc_se": "0.0195",
    "auroc_ci_low": "0.7712",
    "auroc_ci_high": "0.8478",
    "auroc_ci_method": "delong_logit_normal_95pct",
    "average_precision": "0.41",
    "brier_score": "0.1034",
    "calibration_status": "estimated",
    "calibration_intercept": "0.02",
    "calibration_slope": "0.957",
    "repeated_split_n": "5",
    "repeated_split_auroc_mean": "0.8051",
    "repeated_split_auroc_sd": "0.0102",
}
_PASSED = {
    "execution_complete": True,
    "analysis_validated": True,
    "evidence_complete": True,
    "numeric_verified": True,
}


def _plan(**primary: Any) -> Dict[str, Any]:
    return {
        "analysis_type": "prediction_model",
        "steps": [
            {
                "step_id": "robustness_auroc",
                "planned_analysis_role": "sensitivity",
                "expected_outputs": ["table:robustness_matrix"],
            },
            {
                "step_id": "primary_performance",
                "planned_analysis_role": "primary",
                "scientific_action_id": "prediction.discrimination_calibration",
                "expected_outputs": [
                    "table:prediction_scores",
                    "table:model_performance",
                ],
                **primary,
            },
            {
                "step_id": "internal_validation",
                "planned_analysis_role": "secondary",
                "scientific_action_id": "prediction.internal_validation",
                "expected_outputs": ["table:validation"],
            },
        ],
    }


def _record(**overrides: Any) -> Dict[str, Any]:
    return {
        "step_id": "primary_performance",
        "status": "ok",
        "evidence_ids": ["code_analysis_primary", "table_perf", "table_scores"],
        "step_summary": {
            "output_files": {
                "table:model_performance": "prediction_performance.csv",
                "table:prediction_scores": "prediction_scores.csv",
            }
        },
        **overrides,
    }


def _csv(rows: Sequence[Mapping[str, str]]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(_ROW))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(
    run_dir: Path,
    *,
    rows: Sequence[Mapping[str, str]] = (_ROW,),
    plan: Optional[Mapping[str, Any]] = None,
    records: Optional[List[Mapping[str, Any]]] = None,
    gates: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    evidence = run_dir / "evidence"
    evidence.mkdir(parents=True)
    index: List[Dict[str, str]] = []

    def register(evidence_id: str, filename: str, text: str, step: str) -> None:
        path = evidence / f"{evidence_id}__{filename}"
        path.write_text(text, encoding="utf-8")
        index.append(
            {
                "evidence_id": evidence_id,
                "kind": "table",
                "description": f"Table {Path(filename).stem} from step {step}.",
                "relative_path": f"evidence/{path.name}",
                "sha256": _sha(path),
                "produced_by_step": step,
            }
        )

    # Listed first: another step's table that also has an auroc column.
    register(
        "table_robustness",
        "robustness_auroc.csv",
        "specification_id,auroc,brier_score\nprimary,0.6111,0.2222\n",
        "robustness_auroc",
    )
    register(
        "table_perf", "prediction_performance.csv", _csv(rows), "primary_performance"
    )
    register(
        "table_scores",
        "prediction_scores.csv",
        "unit_id,subject_id,split,outcome,probability\n1,1,validation,0,0.1\n",
        "primary_performance",
    )
    register(
        "table_validation",
        "internal_validation.csv",
        "evaluation_n,auroc,brier_score\n200,0.7999,0.1111\n",
        "internal_validation",
    )
    (evidence / "evidence_index.json").write_text(json.dumps(index), encoding="utf-8")
    payload = dict(plan or _plan())
    plan_path = run_dir / "approved_plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "current_plan_authority": {
            "relative_path": "approved_plan.json",
            "sha256": _sha(plan_path),
        },
        "per_step_records": records if records is not None else [_record()],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": dict(gates or _PASSED)}), encoding="utf-8"
    )
    return payload


def _block(run_dir: Path, plan: Mapping[str, Any], **authority: Any):
    return prediction_performance_projection(
        run_dir,
        plan=plan,
        gate_status=authority.get("gate_status", "analysis_only"),
        paper_authorized=authority.get("paper_authorized", False),
    )


def test_the_block_is_the_primary_row_bound_to_its_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    plan = _write_run(run_dir)

    block = _block(run_dir, plan)

    assert block is not None
    assert block["schema_version"] == "easyicu.web-prediction-performance/1"
    assert block["product"] == "table:model_performance"
    assert (block["step_id"], block["evidence_id"]) == (
        "primary_performance",
        "table_perf",
    )
    assert (block["auroc"], block["auroc_ci_low"], block["auroc_ci_high"]) == (
        0.8123,
        0.7712,
        0.8478,
    )
    assert block["auroc_ci_method"] == "delong_logit_normal_95pct"
    assert (block["brier_score"], block["average_precision"]) == (0.1034, 0.41)
    assert (block["calibration_intercept"], block["calibration_slope"]) == (0.02, 0.957)
    assert (block["development_n"], block["validation_n"]) == (800, 200)
    assert (block["validation_subject_n"], block["validation_event_n"]) == (190, 30)
    assert block["repeated_split_n"] == 5
    # Nothing from the other tables with an auroc column.
    assert not {0.6111, 0.7999, 0.2222, 0.1111} & set(block.values())


def test_the_authority_is_the_host_s_not_the_executor_s(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    plan = _write_run(run_dir)

    block = _block(run_dir, plan, gate_status="blocked", paper_authorized=True)

    assert block["authority_scope"] == "blocked"
    assert block["paper_authorization_allowed"] is True


def _rows(**changes: str) -> List[Dict[str, str]]:
    return [{**_ROW, **changes}]


@pytest.mark.parametrize(
    "case",
    [
        {"rows": []},
        {"rows": [_ROW, _ROW]},
        {"plan": _plan(planned_analysis_role="secondary")},
        {"plan": _plan(scientific_action_id="prediction.internal_validation")},
        {
            "plan": {
                **_plan(),
                "steps": _plan()["steps"]
                + [
                    {"step_id": "copy", "expected_outputs": ["table:model_performance"]}
                ],
            }
        },
        # A later failed attempt supersedes the step's earlier success.
        {"records": [_record(), _record(status="failed")]},
        {"records": [_record(evidence_ids=["table_scores"])]},
        {"records": [_record(step_summary={"output_files": {}})]},
        {"rows": _rows(auroc="0.9001")},
        {"rows": _rows(auroc="")},
        {"rows": _rows(brier_score="nan")},
        {"rows": _rows(validation_n="200.5")},
        {"rows": _rows(validation_event_n="201")},
        {"rows": _rows(validation_subject_n="201")},
        {"rows": _rows(validation_subject_n="0")},
        # The kernel's CalibrationStatus is a closed vocabulary.
        {"rows": _rows(calibration_status="not_estimable")},
    ],
)
def test_a_row_that_cannot_be_bound_gives_no_block(
    tmp_path: Path, case: Mapping[str, Any]
) -> None:
    run_dir = tmp_path / "run"
    plan = _write_run(run_dir, **case)

    assert _block(run_dir, plan) is None


def test_a_changed_evidence_file_gives_no_block(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    plan = _write_run(run_dir)
    path = run_dir / "evidence" / "table_perf__prediction_performance.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("0.8123", "0.8124"), encoding="utf-8"
    )

    assert _block(run_dir, plan) is None


def test_a_calibration_that_was_not_estimated_is_not_reported(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    plan = _write_run(
        run_dir,
        rows=_rows(
            calibration_status="not_estimable_constant_probability",
            calibration_intercept="",
            calibration_slope="",
        ),
    )

    block = _block(run_dir, plan)

    assert block["calibration_status"] == "not_estimable_constant_probability"
    assert (block["calibration_intercept"], block["calibration_slope"]) == (None, None)


def _project(tmp_path: Path, run_dir: Path) -> Dict[str, Any]:
    wrapper = tmp_path / "wrapper"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
    )
    return json.loads((wrapper / "result_tables.json").read_text(encoding="utf-8"))


def test_the_run_projection_carries_the_block_with_its_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, gates={**_PASSED, "numeric_verified": False})

    result_tables = _project(tmp_path, run_dir)

    block = result_tables["prediction_performance"]
    assert block["evidence_id"] == "table_perf"
    assert block["authority_scope"] == "blocked"
    assert block["paper_authorization_allowed"] is False
    assert "table_perf__prediction_performance.csv" in {
        table["name"] for table in result_tables["tables"]
    }


def test_a_study_that_is_not_a_prediction_study_carries_no_block(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    plan = _plan(scientific_action_id="association.adjusted_logistic")
    _write_run(run_dir, plan={**plan, "analysis_type": "association"})

    assert "prediction_performance" not in _project(tmp_path, run_dir)
