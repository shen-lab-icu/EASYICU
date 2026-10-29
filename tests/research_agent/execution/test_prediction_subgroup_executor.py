"""Synthetic artifact bindings exercise the normal deterministic input seam."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.contracts.prediction_subgroup import (
    PREDICTION_SUBGROUP_ACTION, PREDICTION_SUBGROUP_OUTPUTS, PredictionSubgroupError,
    prediction_subgroup_method,
)
from easyicu.research_agent.contracts.prediction_subgroup_output import prediction_subgroup_output_findings
from easyicu.research_agent.execution.runners.prediction_subgroup_executor import (
    bind_subgroup_membership, prediction_subgroup_executor_code, run_prediction_subgroup_step,
)
from easyicu.research_agent.execution.runners.typed_input_binding import TypedInputBindingError
from tests.research_agent.core.test_prediction_subgroup import _frame, _spec


def _binding(key, path, frame, root):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kind, _, product = key.partition(":")
    identity = dict(input_key=key, declared_kind=kind, product=product, sha256=digest,
                    evidence_id=product, produced_by_step="synthetic_intake")
    return {**identity, "identity_row": identity, "evidence_kind": "table",
            "relative_path": str(path.relative_to(root)),
            "product_contract": {"columns": list(frame), "row_count": len(frame)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest}}


def _fixture(root):
    frame = _frame()
    scores = frame.drop(columns="group")
    members = frame[["unit_id", "group"]].rename(columns={"unit_id": "row_id"}).iloc[::-1]
    inputs = {}
    for key, name, data in (("table:prediction_scores", "scores.csv", scores),
                            ("artifact:analysis_cohort", "members.csv", members)):
        path = root / name
        data.to_csv(path, index=False)
        inputs[key] = _binding(key, path, data, root)
    return {"step_id": "subgroups", "inputs": inputs}


def test_digest_bound_runner_joins_by_key_and_exports_only_aggregates(tmp_path):
    manifest = _fixture(tmp_path)
    result = run_prediction_subgroup_step(
        spec=_spec(), run_dir=tmp_path, resolved_inputs=manifest,
        step_id="subgroups", out_dir=tmp_path / "out",
    )
    table = pd.read_csv(tmp_path / "out/subgroup_performance.csv")
    assert len(table) == 4
    assert table.loc[1, "brier_score"] == pytest.approx(0.1875)
    assert table.loc[2, "calibration_slope"] == pytest.approx(0.5)
    assert table.loc[3, "evaluation_n"] == 0 and pd.isna(table.loc[3, "auroc"])
    assert not {"unit_id", "subject_id", "row_id"} & set(table)
    provenance = json.loads((tmp_path / "out/subgroup_performance_provenance.json").read_text())
    assert provenance["prediction_artifact_sha256"] == manifest["inputs"]["table:prediction_scores"]["sha256"]
    assert provenance["model_refitted"] is False
    assert result["authority_scope"] == "analysis_only" and result["paper_authorization_allowed"] is False
    assert "no_fairness_judgment" in result["subgroup_performance"]["interpretation"]


@pytest.mark.parametrize("defect,reason", [
    ("digest", "digest_mismatch"), ("consumption", "consumption_contract_mismatch"),
    ("step", "input_binding_mismatch"), ("extra_input", "input_binding_mismatch"),
])
def test_runner_refuses_changed_unapproved_or_widened_inputs_before_outputs(tmp_path, defect, reason):
    manifest = _fixture(tmp_path)
    if defect == "digest":
        (tmp_path / "scores.csv").write_text("tampered", "utf-8")
    elif defect == "consumption":
        manifest["inputs"]["table:prediction_scores"]["consumption_contract"]["artifact_sha256"] = "0" * 64
    elif defect == "step":
        manifest["step_id"] = "different"
    else:
        manifest["inputs"]["table:unexpected"] = {}
    with pytest.raises((TypedInputBindingError, PredictionSubgroupError), match=reason):
        run_prediction_subgroup_step(spec=_spec(), run_dir=tmp_path, resolved_inputs=manifest,
                                     step_id="subgroups", out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("defect,reason", [
    ("duplicate", "duplicate_unit"), ("missing", "identity_missing"), ("unmatched", "unmatched"),
])
def test_membership_does_not_infer_row_order_or_collapse_duplicate_keys(defect, reason):
    frame = _frame()
    members = frame[["unit_id", "group"]].rename(columns={"unit_id": "row_id"})
    if defect == "duplicate":
        members.loc[1, "row_id"] = members.loc[0, "row_id"]
    elif defect == "missing":
        members.loc[1, "row_id"] = None
    else:
        members.loc[1, "row_id"] = "no_match"
    with pytest.raises(PredictionSubgroupError, match=reason):
        bind_subgroup_membership(frame.drop(columns="group"), members, _spec())


def test_existing_prediction_scalar_encoding_requires_an_explicit_unique_key():
    frame = _frame()
    scores = frame.drop(columns="group")
    scores["unit_id"] = [f"int:{i}" for i in range(len(frame))]
    members = frame[["subject_id", "group"]].rename(columns={"subject_id": "row_id"})
    with pytest.raises(PredictionSubgroupError, match="unmatched"):
        bind_subgroup_membership(scores, members, _spec())
    joined = bind_subgroup_membership(scores, members.iloc[::-1], _spec(subgroup_key_encoding="prediction_typed_scalar"))
    assert joined["group"].tolist() == frame["group"].tolist()
    members.loc[1, "row_id"] = 0
    scores.loc[0, "unit_id"] += ":row0"
    with pytest.raises(PredictionSubgroupError, match="duplicate_unit"):
        bind_subgroup_membership(scores, members, _spec(subgroup_key_encoding="prediction_typed_scalar"))


def test_rendered_scaffold_executes_the_bound_deterministic_runner(tmp_path, monkeypatch, capsys):
    spec = _spec()
    step = SimpleNamespace(
        step_id="subgroups", scientific_action_id=PREDICTION_SUBGROUP_ACTION,
        method=prediction_subgroup_method(spec), inputs=("table:prediction_scores", spec.subgroup_input_key),
        expected_outputs=PREDICTION_SUBGROUP_OUTPUTS, planned_analysis_role="secondary",
    )
    manifest = _fixture(tmp_path)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(manifest), "utf-8")
    for name, value in {"EASYICU_RUN_DIR": tmp_path, "EASYICU_RESOLVED_INPUTS_JSON": path,
                        "STEP_OUT_DIR": tmp_path / "out"}.items():
        monkeypatch.setenv(name, str(value))
    exec(compile(prediction_subgroup_executor_code(step), "subgroup_scaffold", "exec"), {})
    summary = json.loads(capsys.readouterr().out)
    assert summary["performance_rows"] == 4
    assert summary["subgroup_performance"]["rows"][1]["calibration_slope"] == pytest.approx(1)
    assert json.loads((tmp_path / "out/step_summary.json").read_text()) == summary


def _step(spec=None):
    spec = spec or _spec()
    return SimpleNamespace(
        step_id="subgroups", scientific_action_id=PREDICTION_SUBGROUP_ACTION,
        method=prediction_subgroup_method(spec), inputs=("table:prediction_scores", spec.subgroup_input_key),
        expected_outputs=PREDICTION_SUBGROUP_OUTPUTS, planned_analysis_role="secondary",
    )


def _gate_fixture(root):
    manifest = _fixture(root)
    output = root / "steps/subgroups/outputs"
    summary = run_prediction_subgroup_step(spec=_spec(), run_dir=root, resolved_inputs=manifest,
                                           step_id="subgroups", out_dir=output)
    return manifest, output, summary


def test_output_gate_recomputes_from_trusted_bindings_and_normal_pipeline_summary(tmp_path):
    manifest, output, summary = _gate_fixture(tmp_path)
    assert prediction_subgroup_output_findings(
        step=_step(), step_summary=summary, resolved_input_bindings=manifest["inputs"], out_dir=output,
    ) == []


@pytest.mark.parametrize("defect", ["csv", "provenance", "saved_summary", "authority", "source", "missing_bindings", "symlink"])
def test_output_gate_refuses_tampering_even_if_self_reported_digests_are_replaced(tmp_path, defect):
    manifest, output, summary = _gate_fixture(tmp_path)
    bindings = manifest["inputs"]
    if defect == "csv":
        table_path = output / "subgroup_performance.csv"
        table = pd.read_csv(table_path)
        table.loc[1, "brier_score"] = 0.1
        table.to_csv(table_path, index=False)
        summary["output_sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
        (output / "step_summary.json").write_text(json.dumps(summary), "utf-8")
    elif defect == "provenance":
        summary["provenance"]["model_refitted"] = True
        path = output / "subgroup_performance_provenance.json"
        path.write_text(json.dumps(summary["provenance"]), "utf-8")
        summary["provenance_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "step_summary.json").write_text(json.dumps(summary), "utf-8")
    elif defect == "saved_summary":
        path = output / "step_summary.json"
        saved = json.loads(path.read_text())
        saved["subgroup_performance"]["rows"][1]["brier_score"] = 0.1
        path.write_text(json.dumps(saved), "utf-8")
    elif defect == "authority":
        summary["authority_scope"] = "paper_authorized"
    elif defect == "source":
        (tmp_path / "scores.csv").write_text("tampered", "utf-8")
    elif defect == "missing_bindings":
        bindings = None
    else:
        path = output / "subgroup_performance.csv"
        target = tmp_path / "relocated.csv"
        path.replace(target)
        path.symlink_to(target)
    findings = prediction_subgroup_output_findings(
        step=_step(), step_summary=summary, resolved_input_bindings=bindings, out_dir=output,
    )
    assert len(findings) == 1 and findings[0].severity == "error"


def test_output_gate_refuses_a_different_schema_valid_result_with_its_valid_hashes(tmp_path):
    manifest, output, _ = _gate_fixture(tmp_path)
    alternative = tmp_path / "alternative"
    alternative.mkdir()
    alternate_manifest = _fixture(alternative)
    scores_path = alternative / "scores.csv"
    scores = pd.read_csv(scores_path)
    scores["probability"] = 0.5
    scores.to_csv(scores_path, index=False)
    alternate_manifest["inputs"]["table:prediction_scores"] = _binding("table:prediction_scores", scores_path, scores, alternative)
    # Replacing all public outputs and all their hashes with a consistent,
    # legitimately computed result still cannot replace the trusted inputs.
    alternate_summary = run_prediction_subgroup_step(spec=_spec(), run_dir=alternative,
                                                      resolved_inputs=alternate_manifest,
                                                      step_id="subgroups", out_dir=output)
    findings = prediction_subgroup_output_findings(
        step=_step(), step_summary=alternate_summary, resolved_input_bindings=manifest["inputs"], out_dir=output,
    )
    assert findings and findings[0].detail["reason"] == "prediction_subgroup_summary_recomputation_mismatch"


def test_renderer_checks_plausibility_step_identity_and_uses_host_receipt():
    from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope, PlausibilityScopeError

    scope = FlagOnlyPlausibilityScope(step_id="subgroups", expected_columns=("group",),
                                      source_contracts_sha256="a" * 64, authority_kind="synthetic")
    code = prediction_subgroup_executor_code(_step(), plausibility_scope=scope)
    assert "plausibility_audit" in code and "step_summary.json" in code
    with pytest.raises(PlausibilityScopeError, match="different analysis step"):
        prediction_subgroup_executor_code(SimpleNamespace(**{**vars(_step()), "step_id": "different"}), plausibility_scope=scope)
