"""A frozen clustering is described on exactly its source cohort, without refits."""

import copy
import json

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.phenotype_comparison_executor import (
    phenotype_comparison_executor_code,
    run_phenotype_comparison,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.execution.runners.typed_input_binding import sha256_file
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    PhenotypeComparisonSpec,
    TableOneVariableSpec,
    VariableRole,
)

from .test_cross_sectional_phenotyping_executor import _binding, _context, _primary_step
from .test_phenotyping_feature_roles import _skeleton


def _comparison_spec():
    return PhenotypeComparisonSpec(
        identity_column="stay_id",
        outcome_columns=["death"],
        variables=[
            TableOneVariableSpec(
                name="age",
                variable_kind="continuous",
                summary="median_iqr",
                test="none_descriptive_smd_only",
            ),
            TableOneVariableSpec(
                name="death",
                variable_kind="categorical",
                summary="count_percent",
                test="none_descriptive_smd_only",
                levels=[0, 1],
            ),
        ],
    )


def _comparison_step():
    return AnalysisStep(
        step_id="clinical_comparison",
        planned_analysis_role="secondary",
        intent="Describe clinical characteristics and observed mortality for the frozen clusters.",
        method="descriptive_profile_by_frozen_cluster",
        scientific_action_id="phenotyping.outcome_by_cluster",
        inputs=[
            "stay_id",
            "age",
            "death",
            "artifact:analysis_cohort",
            "table:phenotype_assignments",
        ],
        expected_outputs=["table:outcome_by_cluster"],
        phenotype_comparison_spec=_comparison_spec(),
    )


def _comparison_context(n=0):
    context = _context(n)
    context.variables.append(
        ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64")
    )
    context.variable("death").observed_domain = {"is_binary": True, "levels": [0, 1]}
    return context


def _comparison_skeleton(rows=None):
    payload = _skeleton(["marker_a", "marker_b"]).model_dump(mode="json")
    step = copy.deepcopy(payload["steps"][1])
    step.update(
        step_id="clinical_comparison",
        planned_analysis_role="secondary",
        depends_on=["01_cohort", "cluster"],
        raw_inputs=["stay_id", "age", "death"],
        product_inputs=[
            {"producer_step_id": "01_cohort", "product_id": "artifact:analysis_cohort"},
            {
                "producer_step_id": "cluster",
                "product_id": "table:phenotype_assignments",
            },
        ],
        outputs=[{"product_id": "table:outcome_by_cluster", "semantic_role": "custom"}],
        scientific_action_id="phenotyping.outcome_by_cluster",
        custom_method="descriptive_profile_by_frozen_cluster",
        phenotyping_feature_columns=None,
        phenotyping_comparison_variables=rows,
    )
    payload["steps"].append(step)
    return ProgressivePlanSkeleton.model_validate(payload)


def test_compiler_and_selector_bind_the_independent_comparison():
    skeleton = _comparison_skeleton(
        [
            {"name": "age", "summary": "median_iqr"},
            {"name": "death", "summary": "count_percent"},
        ]
    )
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=_comparison_context())
    step = plan.steps[2]
    assert step.phenotype_comparison_spec.outcome_columns == ["death"]
    assert [v.name for v in step.phenotype_comparison_spec.variables] == [
        "age",
        "death",
    ]
    assert "run_phenotype_comparison" in phenotype_comparison_executor_code(step)
    selected = select_standard_executor(step=step, plan=plan)
    assert selected.analysis_kind == "phenotype_comparison"
    assert set(selected.consumed_input_keys) == {
        "artifact:analysis_cohort",
        "table:phenotype_assignments",
    }


@pytest.mark.parametrize(
    "rows,reason",
    [
        (None, "progressive_phenotype_comparison_roster_missing"),
        (
            [{"name": "age", "summary": "median_iqr"}],
            "progressive_phenotype_comparison_outcome_missing",
        ),
        (
            [{"name": "death", "summary": "mean_sd"}],
            "progressive_phenotype_comparison_summary_incompatible",
        ),
    ],
)
def test_comparison_must_declare_executable_outcome_summaries(rows, reason):
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=_comparison_skeleton(rows), context=_comparison_context()
        )
    assert caught.value.reason_code == reason


def test_review_does_not_credit_readable_death_or_fitting_profiles_as_comparison():
    plan = AnalysisPlan(
        research_question="Describe clinical groups and death.",
        analysis_type="trajectory_clustering",
        steps=[_primary_step()],
    )
    review = build_plan_scientific_review(context=_comparison_context(), plan=plan)
    finding = next(
        f
        for f in review.findings
        if f.code == "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE"
    )
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    assert not review.approval_allowed
    plan.steps.append(_comparison_step())
    review = build_plan_scientific_review(context=_comparison_context(), plan=plan)
    assert not any(
        f.code == "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" for f in review.findings
    )


def test_review_rejects_a_binary_outcome_disguised_as_a_numeric_profile():
    step = _comparison_step()
    payload = step.phenotype_comparison_spec.model_dump(mode="python")
    payload["variables"][1] = dict(
        name="death",
        variable_kind="continuous",
        summary="mean_sd",
        test="none_descriptive_smd_only",
    )
    step = step.model_copy(
        update={
            "phenotype_comparison_spec": PhenotypeComparisonSpec.model_validate(payload)
        }
    )
    plan = AnalysisPlan(
        research_question="Compare death.",
        analysis_type="trajectory_clustering",
        steps=[_primary_step(), step],
    )
    review = build_plan_scientific_review(context=_comparison_context(), plan=plan)
    codes = {finding.code for finding in review.findings}
    assert {
        "PHENOTYPING_COMPARISON_CONTRACT_INVALID",
        "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE",
    } <= codes


def test_compiler_binds_the_unique_cohort_identity_without_asking_for_an_extra_field():
    skeleton = _comparison_skeleton([{"name": "death", "summary": "count_percent"}])
    payload = skeleton.model_dump(mode="python")
    payload["steps"][2]["raw_inputs"] = ["death"]
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=_comparison_context(),
    )
    assert plan.steps[2].phenotype_comparison_spec.identity_column == "stay_id"
    assert "stay_id" in plan.steps[2].inputs


def _inputs(tmp_path, mutate=None):
    cohort = pd.DataFrame(
        {
            "stay_id": [f"{i:03}" for i in range(12)],
            "age": [
                20.0,
                30.0,
                40.0,
                None,
                50.0,
                60.0,
                70.0,
                80.0,
                21.0,
                31.0,
                41.0,
                51.0,
            ],
            "death": [0.0, 1.0, None, 1.0, 0.0, 0.0, 1.0, 1.0, None, None, None, None],
        }
    )
    tmp_path.mkdir(exist_ok=True)
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    cohort_path = source_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    source_digest = sha256_file(cohort_path)
    assignments = (
        pd.DataFrame(
            {
                "unit_id": cohort.stay_id,
                "cluster": [0] * 4 + [1] * 4 + [2] * 4,
                "source_cohort_sha256": source_digest,
                "source_identity_column": "stay_id",
            }
        )
        .sample(frac=1, random_state=13)
        .reset_index(drop=True)
    )
    if mutate:
        mutate(cohort, assignments)
        cohort.to_parquet(cohort_path, index=False)
    assignment_path = source_dir / "assignments.csv"
    assignments.to_csv(assignment_path, index=False)
    cohort_binding = _binding(
        "artifact:analysis_cohort", cohort, cohort_path, "clinical_comparison"
    )
    cohort_binding["declared_kind"] = "artifact"
    cohort_binding["identity_row"]["declared_kind"] = "artifact"
    manifest = {
        "step_id": "clinical_comparison",
        "inputs": {
            "artifact:analysis_cohort": cohort_binding,
            "table:phenotype_assignments": _binding(
                "table:phenotype_assignments",
                assignments,
                assignment_path,
                "clinical_comparison",
            ),
        },
    }
    (tmp_path / "research_context.json").write_text(
        _comparison_context(len(cohort)).model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "analysis_plan.json").write_text(
        AnalysisPlan(
            research_question="Compare clusters.",
            analysis_type="trajectory_clustering",
            steps=[_primary_step(), _comparison_step()],
        ).model_dump_json(),
        encoding="utf-8",
    )
    return cohort, assignments, manifest


def _run(tmp_path, manifest):
    return run_phenotype_comparison(
        spec=_comparison_spec(),
        typed_cohort_input="artifact:analysis_cohort",
        step_id="clinical_comparison",
        run_dir=tmp_path,
        out_dir=tmp_path / "comparison",
        resolved_inputs=manifest,
    )


def test_exact_identity_join_missing_denominators_and_no_refit(tmp_path):
    cohort, assignments, manifest = _inputs(tmp_path)
    original = {
        name: sha256_file(tmp_path / "inputs" / name)
        for name in ("cohort.parquet", "assignments.csv")
    }
    summary = _run(tmp_path, manifest)
    table = pd.read_csv(
        tmp_path / "comparison/outcome_by_cluster.csv",
        dtype={"category": "string", "group": "string"},
    )
    death = table[(table.variable == "death") & (table.category == "1")].set_index(
        "group"
    )
    assert death.denominator_n.to_dict() == {"Overall": 12, "0": 4, "1": 4, "2": 4}
    assert death.nonmissing_n.to_dict() == {"Overall": 7, "0": 3, "1": 4, "2": 0}
    assert death.missing_n.to_dict() == {"Overall": 5, "0": 1, "1": 0, "2": 4}
    assert death.loc["0", "percentage"] == pytest.approx(200 / 3)
    assert pd.isna(death.loc["2", "percentage"])
    assert death.loc["2", "count"] == 0
    age = table[table.variable == "age"].set_index("group")
    assert age.loc["0", "median"] == 30.0
    assert age.loc["0", "q25"] == 25.0
    assert age.loc["0", "q75"] == 35.0
    assert set(table.test_name) == {"not_reported_data_derived_groups"}
    assert table.p_value.isna().all()
    assert table.standardized_mean_difference.isna().all()
    assert summary["authority_scope"] == "analysis_only"
    assert summary["refit_performed"] is False
    assert summary["output_files"] == {
        "table:outcome_by_cluster": "outcome_by_cluster.csv"
    }
    for name, digest in original.items():
        assert sha256_file(tmp_path / "inputs" / name) == digest
    assert (
        assignments.unit_id.str.len().eq(3).all()
    )  # CSV parsing must preserve zero-prefixed keys.


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda c, a: a.__setitem__("source_cohort_sha256", "0" * 64),
            "phenotype_comparison_source_mismatch",
        ),
        (
            lambda c, a: c.__setitem__("age", c.age + 1),
            "phenotype_comparison_source_mismatch",
        ),
        (
            lambda c, a: a.loc.__setitem__((0, "unit_id"), a.loc[1, "unit_id"]),
            "phenotype_comparison_identity_invalid",
        ),
        (
            lambda c, a: a.loc.__setitem__((0, "unit_id"), "foreign"),
            "phenotype_comparison_membership_mismatch",
        ),
        (
            lambda c, a: a.drop(index=0, inplace=True),
            "phenotype_comparison_membership_mismatch",
        ),
        (
            lambda c, a: a.loc.__setitem__((0, "cluster"), float("nan")),
            "phenotype_comparison_cluster_invalid",
        ),
        (
            lambda c, a: a.__setitem__("source_identity_column", "different_id"),
            "phenotype_comparison_source_mismatch",
        ),
    ],
)
def test_wrong_source_or_membership_fails_without_publishing(
    tmp_path, mutation, reason
):
    _, _, manifest = _inputs(tmp_path, mutate=mutation)
    with pytest.raises((RuntimeError, ValueError), match=reason):
        _run(tmp_path, manifest)
    assert not (tmp_path / "comparison/outcome_by_cluster.csv").exists()


def test_result_gate_rejects_missing_or_tampered_source_table(tmp_path):
    from easyicu.research_agent.contracts.phenotype_comparison import (
        phenotype_comparison_output_findings,
    )

    _, _, manifest = _inputs(tmp_path)
    summary = _run(tmp_path, manifest)
    kwargs = dict(
        step=_comparison_step(),
        step_summary=summary,
        context=_comparison_context(),
        resolved_input_bindings=manifest["inputs"],
        out_dir=tmp_path / "comparison",
    )
    assert phenotype_comparison_output_findings(**kwargs) == []
    path = tmp_path / "comparison/outcome_by_cluster.csv"
    table = pd.read_csv(path)
    table.loc[0, "nonmissing_n"] = 100
    table.to_csv(path, index=False)
    findings = phenotype_comparison_output_findings(**kwargs)
    assert findings and all(f.severity == "error" for f in findings)
    path.unlink()
    assert phenotype_comparison_output_findings(**kwargs)


def test_generated_runtime_code_and_real_step_contract_gate(tmp_path, monkeypatch):
    from easyicu.research_agent.gates.step_contract import _step_contract_findings

    _, _, manifest = _inputs(tmp_path)
    manifest_path = tmp_path / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "comparison"))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    exec(phenotype_comparison_executor_code(_comparison_step()), {})
    summary = json.loads((tmp_path / "comparison/step_summary.json").read_text())
    findings = _step_contract_findings(
        step=_comparison_step(),
        step_summary=summary,
        context=_comparison_context(12),
        resolved_input_bindings=manifest["inputs"],
        out_dir=tmp_path / "comparison",
    )
    assert findings == []


def test_data_derived_table_mode_cannot_bypass_the_comparison_owner():
    from easyicu.research_agent.authority.table_one_binding import (
        bind_table_one_execution_spec,
    )
    from easyicu.research_agent.contracts.phenotype_comparison import (
        comparison_table_spec,
    )
    from easyicu.research_agent.execution.runners.table_one_executor import (
        table_one_executor_owns_step,
    )

    table_spec = comparison_table_spec(
        _comparison_spec(), _comparison_context(), [0, 1]
    )
    step = AnalysisStep(
        step_id="fake_table",
        intent="Describe a normal cohort.",
        method="table_one",
        inputs=[table_spec.group_by, "age", "death", "artifact:analysis_cohort"],
        expected_outputs=["table:table_one"],
        table_one_spec=table_spec,
    )
    assert not table_one_executor_owns_step(step)
    with pytest.raises(
        ValueError, match="table_one_derived_groups_require_phenotype_comparison_owner"
    ):
        bind_table_one_execution_spec(step, _comparison_context())


@pytest.mark.parametrize("token", ["NA", "1e6", "001.0", "9999999999999999999"])
def test_delimited_identity_tokens_are_not_numeric_measurements(tmp_path, token):
    _, _, manifest = _inputs(tmp_path)
    cohort_path = tmp_path / "inputs/cohort.parquet"
    cohort = pd.read_parquet(cohort_path)
    cohort.loc[0, "stay_id"] = token
    cohort.to_parquet(cohort_path, index=False)
    assignment_path = tmp_path / "inputs/assignments.csv"
    assignments = pd.read_csv(assignment_path, converters={"unit_id": str})
    assignments.loc[assignments.unit_id.eq("000"), "unit_id"] = token
    assignments["source_cohort_sha256"] = sha256_file(cohort_path)
    assignments.to_csv(assignment_path, index=False)
    for key, path in (
        ("artifact:analysis_cohort", cohort_path),
        ("table:phenotype_assignments", assignment_path),
    ):
        binding = manifest["inputs"][key]
        digest = sha256_file(path)
        binding["sha256"] = binding["identity_row"]["sha256"] = binding[
            "consumption_contract"
        ]["artifact_sha256"] = digest
    assert _run(tmp_path, manifest)["n_rows"] == 12


@pytest.mark.parametrize(
    "mutation",
    ["digest", "wrong_step", "no_cohort", "partial_consumption", "wrong_evidence_kind"],
)
def test_manifest_scope_is_enforced_at_the_real_loader(tmp_path, mutation):
    _, _, manifest = _inputs(tmp_path)
    if mutation == "digest":
        with (tmp_path / "inputs/assignments.csv").open("a") as handle:
            handle.write("\n")
    elif mutation == "wrong_step":
        manifest["step_id"] = "another_step"
    elif mutation == "no_cohort":
        del manifest["inputs"]["artifact:analysis_cohort"]
    elif mutation == "wrong_evidence_kind":
        manifest["inputs"]["artifact:analysis_cohort"]["evidence_kind"] = "artifact"
    else:
        manifest["inputs"]["table:phenotype_assignments"]["consumption_contract"][
            "mode"
        ] = "single_row"
    with pytest.raises(RuntimeError):
        _run(tmp_path, manifest)
    assert not (tmp_path / "comparison/outcome_by_cluster.csv").exists()


def test_comparison_transport_remains_within_the_existing_schema_budget():
    from easyicu.research_agent.agents.progressive_payload import (
        progressive_structured_output_request,
    )

    request = progressive_structured_output_request(
        analysis_types=["trajectory_clustering"],
        variable_names=["age", "marker_a", "marker_b", "death"],
        cohort_concept_ids=["marker_a"],
        scientific_action_ids=[
            "phenotyping.cluster_solution",
            "phenotyping.k_selection",
            "phenotyping.cluster_stability",
            "phenotyping.outcome_by_cluster",
        ],
        allowed_literature_citation_keys=["strobe_2007"],
    )
    assert len(request.canonical_payload_json.encode()) < 12_000
    branches = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"][
        "anyOf"
    ]
    custom = next(
        branch
        for branch in branches
        if branch["properties"]["module_id"].get("const") == "custom_analysis"
    )
    assert "phenotyping_comparison_variables" in custom["properties"]


def _comparison_outline(include_comparison=True, outcome=True, dependency=True):
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanOutline,
    )

    skeleton = _comparison_skeleton([{"name": "death", "summary": "count_percent"}])
    steps = [
        dict(
            step_id=step.step_id,
            planned_analysis_role=step.planned_analysis_role,
            module_id=step.module_id,
            objective=step.objective,
            depends_on=step.depends_on,
            variable_names=step.raw_inputs or ["stay_id"],
            scientific_action_id=step.scientific_action_id,
            literature_citation_keys=["strobe_2007"]
            if step.planned_analysis_role != "auxiliary"
            else [],
        )
        for step in skeleton.steps
    ]
    if not include_comparison:
        steps.pop()
    else:
        if not outcome:
            steps[-1]["variable_names"] = ["age"]
        if not dependency:
            steps[-1]["depends_on"] = ["01_cohort"]
    return ProgressivePlanOutline(
        analysis_type="trajectory_clustering",
        cohort_objective="Use the declared source cohort.",
        rationale="Plan the full question before materializing steps.",
        steps=steps,
    )


@pytest.mark.parametrize(
    "kwargs",
    [dict(include_comparison=False), dict(outcome=False), dict(dependency=False)],
)
def test_outline_fails_early_when_the_requested_comparison_is_absent(kwargs):
    from easyicu.research_agent.agents.progressive_planner import (
        ProgressivePlannerAgent,
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            _comparison_outline(**kwargs),
            analysis_types=["trajectory_clustering"],
            variable_names=[v.name for v in _comparison_context().variables],
            allowed_literature_citation_keys=["strobe_2007"],
            target_outcome="death",
        )
    assert (
        caught.value.reason_code
        == "progressive_outline_phenotype_comparison_incomplete"
    )


def test_outline_accepts_bound_comparison_but_does_not_invent_an_unrequested_outcome():
    from easyicu.research_agent.agents.progressive_planner import (
        ProgressivePlannerAgent,
    )

    options = dict(
        analysis_types=["trajectory_clustering"],
        variable_names=[v.name for v in _comparison_context().variables],
        allowed_literature_citation_keys=["strobe_2007"],
    )
    ProgressivePlannerAgent._validate_outline_authority(
        _comparison_outline(), target_outcome="death", **options
    )
    ProgressivePlannerAgent._validate_outline_authority(
        _comparison_outline(include_comparison=False), target_outcome=None, **options
    )
