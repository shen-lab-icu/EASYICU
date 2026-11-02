"""Synthetic frozen-prediction subgroup and artifact binding fixtures."""
import hashlib
import pandas as pd
from easyicu.research_agent.contracts.prediction_subgroup import PredictionSubgroupSpec
from easyicu.research_agent.execution.runners.prediction_subgroup_executor import run_prediction_subgroup_step


def _spec(**updates):
    payload = dict(
        analysis_unit="subject", subgroup_input_key="artifact:analysis_cohort",
        subgroup_unit_id_column="row_id", axes=({"column": "group", "levels": ("A", "B", "C")},),
        minimum_group_n=2, minimum_class_n=1,
    )
    return PredictionSubgroupSpec.model_validate({**payload, **updates})

def _frame():
    # At each of two probability values, the observed risks are exactly 1/4
    # and 3/4. A is perfectly calibrated; B's logit is twice the true logit.
    return pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(16)], "subject_id": range(16),
        "split": "validation", "outcome": [0, 0, 0, 1, 0, 1, 1, 1] * 2,
        "probability": [0.25] * 4 + [0.75] * 4 + [0.1] * 4 + [0.9] * 4,
        "group": ["A"] * 8 + ["B"] * 8,
    })

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

def _gate_fixture(root):
    manifest = _fixture(root)
    output = root / "steps/subgroups/outputs"
    summary = run_prediction_subgroup_step(spec=_spec(), run_dir=root, resolved_inputs=manifest,
                                           step_id="subgroups", out_dir=output)
    return manifest, output, summary
