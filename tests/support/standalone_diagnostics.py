"""Synthetic source-bound balance and cluster-size diagnostic fixtures."""
import hashlib
import json
import pandas as pd
from easyicu.research_agent.contracts.diagnostic_specs import BALANCE_OUTPUTS, CLUSTER_SIZE_OUTPUTS, ClusterSizeSpec, CovariateBalanceSpec
from easyicu.research_agent.schema import AnalysisStep


def _balance_spec(**changes):
    return CovariateBalanceSpec.model_validate(
        {
            "unit_id_column": "unit",
            "analysis_unit": "encounter",
            "treatment_column": "treated",
            "weight_column": "weight",
            "covariates": [{"column": "x", "kind": "continuous"}],
            **changes,
        }
    )

def _balance_frame():
    return pd.DataFrame(
        {
            "unit": list("abcdef"),
            "treated": [1, 1, 1, 0, 0, 0],
            "weight": [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 0.0, 1.0, 2.0],
        }
    )

def _size_spec(**changes):
    return ClusterSizeSpec.model_validate(
        {
            "unit_id_column": "unit",
            "analysis_unit": "encounter",
            "assignment_source": "table:phenotype_assignments",
            "assignment_unit_id_column": "unit_id",
            "label_column": "cluster",
            "expected_clusters": ["A", "B", "C"],
            "missing_label_policy": "exclude",
            **changes,
        }
    )

def _sizes_frames():
    return pd.DataFrame({"unit": list("abcde")}), pd.DataFrame(
        {"unit_id": ["d", "a", "c", "b"], "cluster": ["B", "A", "A", None]}
    )

def _step(kind: str):
    if kind == "balance":
        return AnalysisStep(
            step_id="balance",
            intent="Describe measured covariate balance.",
            method="covariate_balance",
            planned_analysis_role="secondary",
            scientific_action_id="causal_emulation.covariate_balance",
            inputs=["artifact:analysis_cohort", "unit", "treated", "weight", "x"],
            expected_outputs=list(BALANCE_OUTPUTS),
            covariate_balance_spec=_balance_spec(),
        )
    return AnalysisStep(
        step_id="sizes",
        intent="Count existing frozen cluster memberships.",
        method="cluster_sizes",
        planned_analysis_role="secondary",
        scientific_action_id="phenotyping.cluster_sizes",
        inputs=["artifact:analysis_cohort", "table:phenotype_assignments", "unit"],
        expected_outputs=list(CLUSTER_SIZE_OUTPUTS),
        cluster_size_spec=_size_spec(),
    )

def _binding(run_dir, key, frame, filename, step_id):
    path = run_dir / filename
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kind, _, product = key.partition(":")
    binding = {
        "relative_path": filename,
        "sha256": digest,
        "declared_kind": kind,
        "evidence_kind": "table",
        "product": product,
        "evidence_id": "ev_" + product,
        "produced_by_step": "producer_" + product,
        "step_id": step_id,
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
    }
    binding["identity_row"] = {
        "input_key": key,
        **{
            name: binding[name]
            for name in ("sha256", "declared_kind", "product", "evidence_id")
        },
    }
    return binding

def _bound_case(tmp_path, kind):
    step = _step(kind)
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    out_dir.mkdir(parents=True)
    frame = _balance_frame() if kind == "balance" else _sizes_frames()[0]
    bindings = {
        "artifact:analysis_cohort": _binding(
            run_dir, "artifact:analysis_cohort", frame, "cohort.parquet", step.step_id
        )
    }
    if kind == "sizes":
        assignments = _sizes_frames()[1].assign(
            source_cohort_sha256=bindings["artifact:analysis_cohort"]["sha256"],
            source_identity_column="unit",
        )
        bindings["table:phenotype_assignments"] = _binding(
            run_dir,
            "table:phenotype_assignments",
            assignments,
            "assignments.parquet",
            step.step_id,
        )
    resolved = run_dir / "resolved.json"
    resolved.write_text(
        json.dumps({"step_id": step.step_id, "inputs": bindings}), encoding="utf-8"
    )
    return step, out_dir, resolved, bindings
