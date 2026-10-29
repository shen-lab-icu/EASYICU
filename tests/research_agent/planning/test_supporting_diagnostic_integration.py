"""Closed diagnostic plans resolve to the kernels exposed by the skill hub."""

import copy
import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.prediction_subgroup import prediction_subgroup_method
from easyicu.research_agent.agents.progressive_prompt_contracts import step_materialization_shape_contract
from easyicu.research_agent.agents.progressive_payload import progressive_structured_output_request
from easyicu.research_agent.methods.cluster_sizes import compute_cluster_sizes
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.gates.step_contract import _step_contract_findings
from easyicu.research_agent.planning.progressive_compiler import _compile_one_step, compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton, ProgressiveSkeletonStep
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.research_agent.core.test_phenotyping_feature_roles import _skeleton as _cluster_skeleton
from tests.research_agent.core.test_cross_sectional_phenotyping_executor import _context as _cluster_context
from tests.research_agent.core.test_standalone_balance_and_cluster_sizes import _bound_case
from tests.research_agent.core.test_standalone_balance_and_cluster_sizes import _sizes_frames, _size_spec
from tests.research_agent.execution.test_prediction_subgroup_executor import _gate_fixture, _spec
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload
from tests.support.schema_names import authority_schema_json


def _declaration(kind):
    field, action, family, raw, products, spec, outputs = {
        "balance": (
            "covariate_balance_spec", "causal_emulation.covariate_balance", "causal_inference",
            ["exposure_flag", "age_years"], ["artifact:analysis_cohort"],
            dict(unit_id_column="stay_id", analysis_unit="encounter", treatment_column="exposure_flag",
                 covariates=[dict(column="age_years", kind="continuous")]),
            ["table:covariate_balance", "log:covariate_balance_receipt"],
        ),
        "cluster": (
            "cluster_size_spec", "phenotyping.cluster_sizes", "trajectory_clustering", [],
            ["artifact:analysis_cohort", "table:phenotype_assignments"],
            dict(unit_id_column="stay_id", analysis_unit="encounter", assignment_source="table:phenotype_assignments",
                 assignment_unit_id_column="unit_id", label_column="cluster", expected_clusters=["A", "B"], missing_label_policy="exclude"),
            ["table:cluster_sizes", "log:cluster_sizes_receipt"],
        ),
        "prediction": (
            "prediction_subgroup_spec", "prediction.subgroup_fairness", "prediction_model", [],
            ["table:prediction_scores", "artifact:analysis_cohort"], _spec().model_dump(mode="json"),
            ["table:subgroup_performance"],
        ),
    }[kind]
    step = dict(step_id="diagnostic", planned_analysis_role="secondary", module_id="custom_analysis",
                objective="Describe the declared frozen analysis products.", depends_on=["01_cohort", "primary"],
                raw_inputs=raw, product_inputs=[dict(product_id=p, producer_step_id="01_cohort" if p == "artifact:analysis_cohort" else "primary") for p in products],
                outputs=[dict(product_id=p, semantic_role="custom") for p in outputs],
                scientific_action_id=action, custom_method="declared_supporting_diagnostic", **{field: json.dumps(spec)})
    return field, family, step


@pytest.mark.parametrize("kind", ["balance", "cluster", "prediction"])
def test_planner_wire_compiles_to_one_real_host_owner(kind):
    field, family, payload = _declaration(kind)
    step = ProgressiveSkeletonStep.model_validate(payload)
    source = ProgressivePlanSkeleton.model_validate(_payload())
    skeleton = source.model_copy(update={"analysis_type": family, "steps": [source.steps[0], source.steps[5], step]})
    context = _context()
    compiled = _compile_one_step(
        context=context, skeleton=skeleton, step=step, step_index=2,
        variables={item.name: item for item in context.variables}, allowed_citations=frozenset(),
        producers={"artifact:analysis_cohort": "01_cohort", "table:prediction_scores": "primary", "table:phenotype_assignments": "primary"},
        outputs_by_step={},
    )
    sealed = compiled.step
    selected = select_standard_executor(sealed, plan=AnalysisPlan(research_question="Synthetic diagnostics.", steps=[sealed]))
    assert selected is not None
    assert selected.analysis_kind == {"balance": "covariate_balance", "cluster": "cluster_sizes", "prediction": "prediction_subgroup_performance"}[kind]
    assert selected.consumed_input_keys == tuple(p["product_id"] for p in payload["product_inputs"])
    assert "easyicu.research_agent.execution.runners." in selected.code
    assert selected.host_sealed_renderer
    prompt = step_materialization_shape_contract(outline_step=step, outline_step_sha256="a" * 64)
    assert field in prompt and "JSON" in prompt
    if kind == "prediction":
        assert sealed.method.startswith("prediction_subgroup_performance(")
        assert field not in type(sealed).model_fields
    else:
        assert getattr(sealed, field) == getattr(step, field)
        assert "stay_id" in sealed.inputs


@pytest.mark.parametrize("kind", ["balance", "cluster", "prediction"])
def test_planner_refuses_missing_contract_wrong_role_and_wrong_action(kind):
    field, _, payload = _declaration(kind)
    for changes in ({field: None}, {"planned_analysis_role": "primary"}, {"scientific_action_id": "association.evalue"}):
        invalid = copy.deepcopy(payload)
        invalid.update(changes)
        with pytest.raises(ValidationError):
            ProgressiveSkeletonStep.model_validate(invalid)


@pytest.mark.parametrize("kind", ["balance", "cluster", "prediction"])
def test_runtime_transport_offers_the_diagnostic_spec_only_on_custom_actions(kind):
    field, family, payload = _declaration(kind)
    request = progressive_structured_output_request(
        analysis_types=[family], variable_names=["exposure_flag", "age_years"],
        cohort_concept_ids=["stay_id"], scientific_action_ids=[payload["scientific_action_id"]],
        allowed_literature_citation_keys=[],
    )
    schema = authority_schema_json(request.schema_json)
    branches = schema["$defs"]["ProgressiveSkeletonStep"]["anyOf"]
    custom = next(b for b in branches if b["properties"]["module_id"].get("const") == "custom_analysis")
    assert field in custom["properties"]
    assert {"type": "string"} in custom["properties"][field]["anyOf"]
    for branch in branches:
        if branch is not custom:
            assert field not in branch["properties"]


def test_cluster_diagnostic_compiles_through_the_full_progressive_plan():
    skeleton = _cluster_skeleton(["marker_a", "marker_b"])
    _, _, payload = _declaration("cluster")
    payload["depends_on"] = ["01_cohort", "cluster"]
    payload["product_inputs"][1]["producer_step_id"] = "cluster"
    skeleton = skeleton.model_copy(update={"steps": [*skeleton.steps, ProgressiveSkeletonStep.model_validate(payload)]})
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=_cluster_context(0))
    selected = select_standard_executor(plan.steps[-1], plan=plan)
    assert selected is not None and selected.analysis_kind == "cluster_sizes"


def test_cluster_absent_assignment_is_distinct_from_a_matched_missing_label():
    result = compute_cluster_sizes(*_sizes_frames(), _size_spec())
    assert result.unmatched_unit_n == 1
    assert result.matched_missing_label_n == 1
    assert result.missing_label_n == 2


@pytest.mark.parametrize("kind", ["balance", "sizes", "prediction"])
def test_normal_step_gate_replays_diagnostics_and_rejects_numeric_tampering(tmp_path, monkeypatch, kind):
    if kind == "prediction":
        manifest, output, summary = _gate_fixture(tmp_path)
        step = AnalysisStep(step_id="subgroups", intent="Describe frozen predictions by declared subgroup.",
                            planned_analysis_role="secondary", scientific_action_id="prediction.subgroup_fairness",
                            method=prediction_subgroup_method(_spec()),
                            inputs=["table:prediction_scores", "artifact:analysis_cohort"], expected_outputs=["table:subgroup_performance"])
        bindings = manifest["inputs"]
    else:
        step, output, resolved, bindings = _bound_case(tmp_path, kind)
        for key, value in {"EASYICU_RUN_DIR": resolved.parent, "EASYICU_RESOLVED_INPUTS_JSON": resolved, "STEP_OUT_DIR": output}.items():
            monkeypatch.setenv(key, str(value))
        selected = select_standard_executor(step, plan=AnalysisPlan(research_question="Synthetic diagnostics.", steps=[step]))
        exec(compile(selected.code, "synthetic diagnostic", "exec"), {})
        summary = json.loads((output / "step_summary.json").read_text())
    assert not [f for f in _step_contract_findings(step=step, step_summary=summary, resolved_input_bindings=bindings, out_dir=output) if f.severity == "error"]
    table = next(output.glob("*.csv"))
    table.write_text(table.read_text() + "corrupt-row\n")
    assert any(f.severity == "error" for f in _step_contract_findings(step=step, step_summary=summary, resolved_input_bindings=bindings, out_dir=output))
