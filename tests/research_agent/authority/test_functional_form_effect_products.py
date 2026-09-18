"""Synthetic end-to-end contracts; no real study or external model is executed."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import LandmarkSplineRuntimeAuthority
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.execution.runners.functional_form_effect_products import consume_functional_form_effects
from easyicu.research_agent.execution.runners.landmark_spline_executor import run_landmark_spline_association
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.execution.runners.typed_input_binding import load_typed_input
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.robustness.panel import write_locked_robustness_specs
from easyicu.research_agent.schema import AnalysisStep

from .test_functional_form_target import _bound_sensitivity, _form, _synthetic_frame
from ..planning.progressive_planner_fixtures import _context


def _register_input(store, root, key, path, step_id):
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    record = store.register_file(
        kind="table", description=f"Synthetic fixture {key}", source_path=path,
        evidence_id="fixture_" + key.partition(":")[2], produced_by_step=step_id,
        generation_mode="deterministic_standard",
    )
    return {
        "relative_path": record.relative_path, "sha256": record.sha256,
        "declared_kind": key.partition(":")[0], "evidence_kind": "table",
        "product": key.partition(":")[2], "evidence_id": record.evidence_id,
        "product_contract": {"columns": list(data.columns), "row_count": len(data)},
        "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": record.sha256},
    }


def _robust_step():
    products = {
        "primary_or": "primary_effect", "complete_case_n": "complete_case_n",
        "robustness_summary": "robustness_summary", "missingness_strategy_notes": "missingness_strategy_notes",
        "robustness_matrix": "robustness_matrix",
    }
    return AnalysisStep.model_validate({
        "step_id": "robustness", "planned_analysis_role": "sensitivity",
        "intent": "Compare the actual effects of every planned adjustment-form refit.",
        "method": "robustness_sensitivity", "inputs": [],
        "sensitivity_spec_ids": ["complete_cases"],
        "expected_outputs": ["statistic:primary_or", "statistic:complete_case_n",
                             "table:robustness_summary", "log:missingness_strategy_notes", "table:robustness_matrix"],
        "robustness_replay_spec": {"products": [{"product_id": k, "output": v} for k, v in products.items()]},
    })


def _execute_scaffold(step, plan, authority, projection, root, manifest):
    selected = select_standard_executor(
        step, plan=plan, current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert selected is not None
    manifest_path = root / f"{step.step_id}_inputs.json"
    manifest_path.write_text(json.dumps(manifest))
    out = root / step.step_id
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EASYICU_RUN_DIR", str(root))
        patch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
        patch.setenv("STEP_OUT_DIR", str(out))
        exec(compile(selected.code, "<native fixture scaffold>", "exec"), {})
    return json.loads((out / "step_summary.json").read_text())


def _prepare(root, *, clustered=False, targets=("age",)):
    projection, original, draft = _bound_sensitivity(targets[0])
    body = original.model_dump(mode="json", exclude={"execution_contract_sha256"})
    body["observation_duration_column"] = "los_hospital"
    if clustered:
        body.update(schema_version="easyicu.landmark_spline_runtime_authority/4", dependence={
            "schema_version": "easyicu.planned_dependence/1", "variance_estimator": "cluster_robust",
            "cluster_unit": "patient", "group_source": "patient_key", "group_derivation": "identity", "delimiter": None,
        })
    authority = LandmarkSplineRuntimeAuthority.model_validate_json(json.dumps({**body, "execution_contract_sha256": canonical_sha256(body)}))
    children = [draft.steps[1].model_copy(update={
        "step_id": f"form_{target}", "functional_form_spec": _form(target),
        "sensitivity_spec_ids": [f"shape_{target}"], "expected_outputs": [f"table:form_{target}"],
    }) for target in targets]
    spec = RobustnessSpec(
        spec_id="complete_cases", axis="missing", description="Document the actual primary complete-case set.",
        missing_override={"strategy": "complete_case", "variables": list(authority.model_complete_case_columns)},
    )
    plan = authority.bind_plan(draft.model_copy(update={
        "steps": [draft.steps[0], *children, _robust_step()], "robustness_specs": [spec],
    }))
    authority.validate_plan(plan)
    frame = _synthetic_frame()
    frame["los_hospital"] = frame.los_icu
    # Hospital, not ICU, is the declared clock. This stay must remain included.
    frame.loc[3, "los_icu"] = .25
    frame.to_parquet(root / "cohort.parquet", index=False)
    before = hashlib.sha256((root / "cohort.parquet").read_bytes()).hexdigest()
    run_landmark_spline_association(
        frame=frame, authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=root / "primary",
    )
    store = EvidenceStore(root, enforcement_mode="strict")
    write_locked_robustness_specs(run_dir=root, plan=plan, evidence=store, prompt_pack_version=None, llm_signature="offline-fixture")
    context = _context()
    variable = context.variables[0].model_copy(update={"name": authority.exposure_column, "description": "Synthetic exposure", "unit": "units"})
    context = context.model_copy(update={"variables": [*context.variables, variable]})
    shared = {}
    for key, payload in (("context", context.model_dump(mode="json")), ("plan", plan.model_dump(mode="json"))):
        path = root / f"{key}.json"
        path.write_text(json.dumps(payload))
        shared[key] = {"relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    inputs = {"dataset:analysis_cohort": _register_input(store, root, "dataset:analysis_cohort", root / "cohort.parquet", "cohort")}
    for key in (authority.downstream_parent_product, authority.linear_sensitivity_product):
        inputs[key] = _register_input(store, root, key, root / "primary" / f"{key.partition(':')[2]}.csv", plan.steps[0].step_id)
    records, child_outputs = [], {}
    for step in plan.steps[1:]:
        manifest = {"step_id": step.step_id, **shared, "inputs": {key: inputs[key] for key in step.inputs if ":" in key}}
        summary = _execute_scaffold(step, plan, authority, projection, root, manifest)
        if step.functional_form_spec is not None:
            child_outputs[step.step_id] = summary
            for key, filename in summary["output_files"].items():
                inputs[key] = _register_input(store, root, key, root / step.step_id / filename, step.step_id)
        record = store.register_json(
            kind="statistic", description=f"Synthetic native summary {step.step_id}", payload=summary,
            filename=f"{step.step_id}_summary.json", evidence_id=f"summary_{step.step_id}",
            produced_by_step=step.step_id, generation_mode="deterministic_standard",
        )
        store.register_step_summary_numerics(step_id=step.step_id, evidence_id=record.evidence_id, summary=summary)
        records.append({"step_id": step.step_id, "status": "ok", "generation_mode": "deterministic_standard",
                        "step_summary": summary, "step_summary_evidence_id": record.evidence_id, "evidence_ids": [record.evidence_id]})
    assert hashlib.sha256((root / "cohort.parquet").read_bytes()).hexdigest() == before
    return SimpleNamespace(root=root, authority=authority, projection=projection, plan=plan,
                           frame=frame, inputs=inputs, children=child_outputs, summary=summary,
                           store=store, records=records)


def test_binding_orders_covariate_effect_producer_before_robustness_consumer():
    _, authority, draft = _bound_sensitivity("age")
    form_step = draft.steps[1]
    robustness = _robust_step()
    spec = RobustnessSpec(
        spec_id="complete_cases",
        axis="missing",
        description="Document the actual primary complete-case set.",
        missing_override={
            "strategy": "complete_case",
            "variables": list(authority.model_complete_case_columns),
        },
    )

    bound = authority.bind_plan(
        draft.model_copy(
            update={
                "steps": [draft.steps[0], robustness, form_step],
                "robustness_specs": [spec],
            }
        )
    )

    step_ids = [step.step_id for step in bound.steps]
    assert step_ids.index(form_step.step_id) < step_ids.index(robustness.step_id)
    rebound_robustness = next(step for step in bound.steps if step.step_id == robustness.step_id)
    assert set(form_step.expected_outputs[1:]) <= set(rebound_robustness.inputs)
    authority.validate_plan(bound)


@pytest.fixture(scope="module")
def effects(tmp_path_factory):
    return _prepare(tmp_path_factory.mktemp("functional-effects"), targets=("age", "charlson_first"), clustered=True)


def _consume_arguments(fixture, child_index=1):
    step = fixture.plan.steps[child_index]
    authority = fixture.authority
    def read(key):
        return pd.read_csv(fixture.root / fixture.inputs[key]["relative_path"])
    primary = {}
    for key in (authority.downstream_parent_product, authority.linear_sensitivity_product):
        primary[key] = load_typed_input(
            input_key=key, run_dir=fixture.root,
            resolved_inputs={"step_id": "check", "inputs": fixture.inputs}, step_id="check",
            require_consumption_contract=True,
        )
    return dict(
        step=step, authority=authority, runtime_projection_sha256=fixture.projection.runtime_projection_sha256,
        curve=read(step.expected_outputs[1]), points=read(step.expected_outputs[2]),
        contrasts=read(authority.downstream_parent_product), linear_sensitivity=read(authority.linear_sensitivity_product),
        primary_input_bindings=primary,
    )


def test_full_plan_to_native_effects_robustness_claims_and_strict_numbers(effects):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
    from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest_v2

    summary = effects.summary
    assert summary["complete_case_n"] == 597
    assert summary["n_converged_variants"] == 3  # linear exposure + two refits, not four point rows
    matrix = pd.DataFrame(summary["robustness_rows"])
    assert matrix.loc[matrix.axis == "missing", "independent_variant"].tolist() == [False]
    primary_coordinates = set(matrix.loc[matrix.axis == "primary", "contrast_id"])
    for target in ("age", "charlson_first"):
        rows = matrix[matrix.spec_id == f"shape_{target}"]
        assert len(rows) == 2 and set(rows.contrast_id) == primary_coordinates
        contract = consume_functional_form_effects(**_consume_arguments(effects, 1 if target == "age" else 2))
        assert contract.n == 597 and contract.cluster_count == 299
        assert contract.observation_duration_column == "los_hospital"
        child = effects.children[f"form_{target}"]
        assert "parameter_values" not in json.dumps(child)
        contract_json = _consume_arguments(effects, 1 if target == "age" else 2)["curve"].functional_form_effects_json.iloc[0]
        assert child["functional_form_effect_products"]["contract_sha256"] == hashlib.sha256(contract_json.encode()).hexdigest()
        assert len(child["output_files"]) == 3
        assert contract.primary_exposure_nonlinearity_p_value == pytest.approx(
            _consume_arguments(effects)["linear_sensitivity"].iloc[0].nonlinearity_p_value,
        )
    claims = derive_scientific_claim_drafts(summary)
    assert len(claims) == 7
    assert [c.analysis_role for c in claims] == ["primary", "primary", "sensitivity", "sensitivity", "sensitivity", "sensitivity", "sensitivity"]
    assert all("shape_" not in claim.estimand for claim in claims[3:])
    for target in ("age", "charlson_first"):
        reader_target = target.replace("_", " ")
        assert any(
            f"prespecified sensitivity: {reader_target} modeled with its reviewed restricted cubic spline"
            in claim.estimand for claim in claims[3:]
        )
    scaffold = "## Results\n\n" + "\n\n".join(claim.placeholder for claim in effects.store.scientific_claims())
    bound = effects.store.bind_manuscript(scaffold, per_step_records=effects.records)
    _, bindings, untraced = bind_numeric_values(bound, evidence=effects.store, per_step_records=effects.records)
    assert not untraced
    for claim in claims[3:]:
        for value in (claim.point_estimate, claim.interval_lower, claim.interval_upper):
            assert any(float(item.canonical) == pytest.approx(value) for item in bindings.values())
    digest = _render_writer_evidence_digest_v2(effects.records, run_dir=effects.root, evidence=effects.store)
    for forbidden in ("covariance_matrix", "parameter_values", "contrast_vector", "functional_form_effects_json"):
        assert forbidden not in digest
        assert all(forbidden not in claim.source_field for claim in effects.store.numeric_claims())
    assert "shape_age" not in digest and "shape_charlson_first" not in digest
    assert "prespecified sensitivity: age modeled" in digest
    assert "prespecified sensitivity: charlson first modeled" in digest
    assert len(digest.encode()) < 64 * 1024
    (effects.root / "writer-digest.txt").write_text(digest)
    (effects.root / "projection-receipt.json").write_text(json.dumps({
        "synthetic_only": True, "provider_calls": 0,
        "continuous_adjustment_targets": ["age", "charlson_first"],
        "n": summary["complete_case_n"], "clusters": contract.cluster_count,
        "independent_variants": summary["n_converged_variants"],
        "native_scientific_claims": len(claims),
        "functional_form_sensitivity_claims": len(claims[3:]),
        "strict_untraced": len(untraced), "strict_bound_unique_values": len(bindings),
        "writer_digest_bytes": len(digest.encode()),
        "parameter_covariance_numeric_leaves": 0,
        "raw_contract_json_in_writer": False,
    }, indent=2) + "\n")


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("target", ["age", "charlson_first"])
def test_full_curve_matches_independent_truncated_power_basis(tmp_path, clustered, target):
    import statsmodels.api as sm

    fixture = _prepare(tmp_path, clustered=clustered, targets=(target,))
    arguments = _consume_arguments(fixture)
    contract = consume_functional_form_effects(**arguments)
    source = fixture.frame
    rows = source.loc[((source.death == 0) | (source.death_time > 24)) & (source.los_hospital >= 1)].dropna(
        subset=["lact_max", "death", "age", "sex", "charlson_first"],
    )
    assert 3 in rows.index and rows.loc[3, "los_icu"] < 1
    def nonlinear(values, knots):
        lo, mid, hi = knots
        return (np.maximum(values - lo, 0)**3 - np.maximum(values - mid, 0)**3 * (hi-lo)/(hi-mid)
                + np.maximum(values-hi, 0)**3 * (mid-lo)/(hi-mid)) / (hi-lo)**2
    exposure_knots = np.quantile(rows.lact_max, [.1, .5, .9])
    design = pd.DataFrame({
        "const": 1., "exposure": rows.lact_max, "exposure_nonlinear": nonlinear(rows.lact_max, exposure_knots),
        "age": rows.age, "sex_M": (rows.sex == "M").astype(float), "charlson": rows.charlson_first,
        "target_nonlinear": nonlinear(rows[target], np.quantile(rows[target], [.1, .5, .9])),
    })
    kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": rows.patient_key}} if clustered else {}
    fit = sm.GLM(rows.death, design, family=sm.families.Binomial()).fit(maxiter=200, **kwargs)
    curve = arguments["curve"]
    grid = curve.exposure_value.to_numpy()
    vectors = np.zeros((len(grid), design.shape[1]))
    vectors[:, 1] = grid - contract.reference
    vectors[:, 2] = nonlinear(grid, exposure_knots) - nonlinear(contract.reference, exposure_knots)
    eta = vectors @ np.asarray(fit.params)
    se = np.sqrt(np.einsum("ij,jk,ik->i", vectors, np.asarray(fit.cov_params()), vectors))
    expected = np.exp(np.column_stack([eta, eta - 1.96*se, eta + 1.96*se]))
    np.testing.assert_allclose(curve[["adjusted_odds_ratio", "ci_low", "ci_high"]], expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("analysis_role", "analysis_only"), ("analysis_role", "primary"),
    ("independent_refit", False), ("step_id", "foreign"), ("spec_id", "foreign"),
    ("exposure", "foreign"), ("outcome_time_column", "foreign"),
    ("observation_duration_column", "los_icu"), ("landmark_hours", 12.),
    ("runtime_projection_sha256", "d"*64), ("execution_contract_sha256", "d"*64),
    ("protocol_content_sha256", "d"*64), ("n", 598), ("events", 599),
    ("cluster_count", 298), ("reference", 42.), ("primary_exposure_nonlinearity_p_value", .999),
])
def test_wrong_scientific_identity_never_becomes_robustness(effects, field, value):
    arguments = _consume_arguments(effects)
    for key in ("curve", "points"):
        contract = json.loads(arguments[key].functional_form_effects_json.iloc[0])
        contract[field] = value
        arguments[key]["functional_form_effects_json"] = json.dumps(contract)
    with pytest.raises(ValueError):
        consume_functional_form_effects(**arguments)


@pytest.mark.parametrize("mutation", ["fit_only", "missing_point", "partial_curve", "ci", "coordinate", "vector", "different_parent", "mixed_receipt"])
def test_incomplete_or_forged_effect_products_fail_closed(effects, mutation):
    arguments = _consume_arguments(effects)
    if mutation == "fit_only":
        arguments["curve"] = pd.DataFrame([{"spline_aic": 123., "nonlinearity_p_value": .01}])
    elif mutation == "missing_point":
        arguments["points"] = arguments["points"].iloc[:1]
    elif mutation == "partial_curve":
        arguments["curve"] = arguments["curve"].iloc[:-1]
    elif mutation == "ci":
        arguments["curve"].loc[4, "ci_high"] *= 1.02
    elif mutation == "coordinate":
        arguments["curve"].loc[4, "exposure_value"] += .01
    elif mutation == "vector":
        values = json.loads(arguments["curve"].loc[4, "contrast_vector"])
        values[0] = .1
        arguments["curve"].loc[4, "contrast_vector"] = json.dumps(values)
    else:
        for key in (("curve", "points") if mutation == "different_parent" else ("curve",)):
            contract = json.loads(arguments[key].functional_form_effects_json.iloc[0])
            contract["input_bindings"][1]["sha256"] = "d"*64
            arguments[key]["functional_form_effects_json"] = json.dumps(contract)
    with pytest.raises(ValueError):
        consume_functional_form_effects(**arguments)


@pytest.mark.parametrize("mutation", ["old_single_output", "missing_curve_edge", "producer_after_consumer"])
def test_incomplete_approved_plan_requires_revision(effects, mutation):
    plan = effects.plan.model_copy(deep=True)
    if mutation == "old_single_output":
        plan.steps[1].expected_outputs = plan.steps[1].expected_outputs[:1]
    elif mutation == "missing_curve_edge":
        plan.steps[-1].inputs.remove(plan.steps[1].expected_outputs[1])
    else:
        plan.steps = [plan.steps[0], plan.steps[-1], *plan.steps[1:-1]]
    with pytest.raises(ValueError):
        effects.authority.validate_plan(plan)


def test_alternate_claims_cannot_lose_their_curve_or_become_primary(effects):
    summary = copy.deepcopy(effects.summary)
    summary["functional_form_effect_sources"][0]["curve_evidence_id"] = "unconsumed"
    with pytest.raises(ValueError, match="both consumed"):
        derive_scientific_claim_drafts(summary)
    summary = copy.deepcopy(effects.summary)
    summary["reportable_model_contrasts"]["contrasts"][3]["kind"] = "spline_point"
    with pytest.raises(ValueError):
        derive_scientific_claim_drafts(summary)


@pytest.mark.parametrize("mutation", ["wrong_parent_effect", "missing_lineage", "nonconverged"])
def test_invalid_refit_never_writes_success_products(effects, tmp_path, monkeypatch, mutation):
    from easyicu.research_agent.execution.runners.landmark_spline_functional_form_executor import run_landmark_spline_functional_form
    from easyicu.research_agent.execution.runners import landmark_spline_fit

    arguments = _consume_arguments(effects)
    receipts = effects.children["form_age"]["input_bindings"]
    if mutation == "wrong_parent_effect":
        arguments["contrasts"].loc[0, "adjusted_odds_ratio"] *= 1.01
    elif mutation == "missing_lineage":
        receipts = receipts[:2]
    else:
        original = landmark_spline_fit.fit_binomial_model
        def failed_fit(**kwargs):
            fit, clusters = original(**kwargs)
            fit.converged = False
            return fit, clusters
        monkeypatch.setattr(landmark_spline_fit, "fit_binomial_model", failed_fit)
    with pytest.raises(ValueError):
        run_landmark_spline_functional_form(
            step=arguments["step"], authority=effects.authority,
            runtime_projection_sha256=effects.projection.runtime_projection_sha256,
            linear_sensitivity=arguments["linear_sensitivity"],
            linear_evidence_id=effects.inputs[effects.authority.linear_sensitivity_product]["evidence_id"],
            cohort_frame=effects.frame, primary_contrasts=arguments["contrasts"], input_bindings=receipts,
            out_dir=tmp_path / "failed",
        )
    assert not (tmp_path / "failed").exists()
