"""A signed association displays its registered observed risks, without refitting."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.typed_binding import (
    _attach_verified_consumption_contract,
    _resolved_typed_input_binding,
)
from easyicu.research_agent.audits.figures import FigureContractQualityValidator
from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step
from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import run_association_overview_figure
from easyicu.research_agent.execution.runners.selection import resolve_standard_executor
from easyicu.research_agent.figures.skill import PublicationFigureSkill
from easyicu.research_agent.reporting.display_suite import summarize_display_suite_status
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime


RISK = "table:absolute_risk_context"
EFFECT = "table:adjusted_association_estimates"


def _case(tmp_path, *, database="eicu", categorical=False, risk_change=None):
    exposure = "admission_type" if categorical else "sex"
    outcome, identity = ("death", "patientunitstayid") if database == "eicu" else ("hospital_expire_flag", "stay_id")
    context = baseline_context(exposure, outcome=outcome, covariates=("chronic_flag",))
    context.cohort.database = database
    context.cohort.id_columns = [identity]
    context.variables[0] = context.variables[0].model_copy(update={"name": identity})
    plan = baseline_plan_attempt(context)[0].output
    runtime = baseline_runtime(context, plan).seal_for_plan(plan)
    plan, _ = runtime.bind_plan(plan)
    levels = runtime.current_case.exposure_levels
    # Small deterministic strata, with events and non-events in every cell.
    frame = pd.DataFrame([
        {identity: index * 100 + flag * 40 + row, exposure: level, outcome: row < 3 + index * 3 + flag,
         "chronic_flag": flag}
        for index, level in enumerate(levels)
        for flag in (0, 1)
        for row in range(20)
    ])
    if not categorical:
        frame[exposure] = frame[exposure].astype(int)
    evidence = EvidenceStore(tmp_path)
    summaries = {}
    for step_id in ("adjusted_association", "absolute_risk_context"):
        summaries[step_id] = run_baseline_step(
            step_id=step_id, plan=plan, authority=runtime.current_case, runtime_projection_sha256="a" * 64,
            frame=frame, output_dir=tmp_path / "steps" / step_id / "outputs",
            primary_table=pd.read_csv(tmp_path / "steps/adjusted_association/outputs/adjusted_association_estimates.csv")
            if step_id == "absolute_risk_context" else None,
        )
    step = next(item for item in plan.steps if item.step_id == "visualization")
    bindings = {}
    for product, step_id in ((EFFECT, "adjusted_association"), (RISK, "absolute_risk_context")):
        path = tmp_path / "steps" / step_id / "outputs" / summaries[step_id]["output_files"][product]
        if product == RISK and risk_change is not None:
            risk_change(pd.read_csv(path)).to_csv(path, index=False)
        ref = evidence.register_file(
            kind="table", source_path=path, evidence_id=step_id, produced_by_step=step_id,
            description="Synthetic signed association product", producer="runner", generation_mode="deterministic_standard",
        )
        binding = _resolved_typed_input_binding(
            input_name=product, evidence_ref=ref, evidence_records=evidence.records(), run_dir=tmp_path,
        )
        assert binding is not None
        bindings[product] = _attach_verified_consumption_contract(step=step, input_name=product, binding=binding)
    return context, plan, runtime, step, evidence, bindings


def _render(tmp_path, monkeypatch, plan, runtime, step, bindings):
    decision = resolve_standard_executor(
        step, plan=plan, current_case_scientific_runtime_authority=runtime.current_case,
        scientific_runtime_projection_sha256="a" * 64, resolved_bindings=bindings,
    )
    assert decision.claimed_by == "association_overview_figure"
    selection = decision.render_selection()
    assert set(selection.consumed_input_keys) == {RISK, EFFECT}
    out = tmp_path / "steps/visualization/outputs"
    out.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "resolved_inputs.json"
    manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": bindings}))
    monkeypatch.setenv("STEP_OUT_DIR", str(out))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest))
    exec(compile(selection.code, "<synthetic-render>", "exec"), {})
    return out, json.loads((out / "step_summary.json").read_text())


@pytest.mark.parametrize("database", ["eicu", "mimiciv"])
@pytest.mark.parametrize("categorical", [False, True], ids=["binary", "categorical"])
def test_registered_risks_reach_the_primary_figure(tmp_path, monkeypatch, database, categorical):
    context, plan, runtime, step, evidence, bindings = _case(tmp_path, database=database, categorical=categorical)
    assert set(step.inputs) == {RISK, EFFECT}
    assert {item.input_key for item in step.input_consumption_contracts if item.mode == "all_rows"} == {RISK, EFFECT}
    out, summary = _render(tmp_path, monkeypatch, plan, runtime, step, bindings)
    assert set(summary["source_inputs"]) == {RISK, EFFECT}
    assert summary["source_evidence_ids"][RISK] == "absolute_risk_context"
    source = pd.read_csv(out / "visualization_distribution_source_data.csv")
    original = pd.read_csv(tmp_path / bindings[RISK]["relative_path"])
    pd.testing.assert_frame_equal(source, original)
    groups = source.loc[source.row_role.eq("exposure_level")]
    assert len(groups) == (3 if categorical else 2)
    assert groups.outcome_denominator.tolist() == [40] * len(groups)
    assert groups.outcome_rate_pct.tolist() == pytest.approx([17.5, 32.5, 47.5][:len(groups)])
    svg = (out / "visualization.svg").read_text()
    for count in groups.outcome_events:
        assert f"{int(count)}/40" in svg
    for path in out.iterdir():
        if path.suffix not in {".svg", ".png", ".json", ".csv"} or path.name == "step_summary.json":
            continue
        evidence.register_file(
            kind="table" if path.suffix == ".csv" else "log" if path.suffix == ".json" else "figure",
            source_path=path, evidence_id="render_" + path.name.replace(".", "_"),
            produced_by_step=step.step_id, producer="runner", generation_mode="deterministic_standard",
            description="Synthetic primary figure", metadata={"figure_role": "publication_figure", "step_id": step.step_id},
        )
    result = PublicationFigureSkill().run(context=context, plan=plan, evidence=evidence, run_dir=tmp_path, prompt_pack_version="test")
    assert result.generated
    path = tmp_path / "publication_figures/easyicu_publication_figure.figure_contract.json"
    contract = json.loads(path.read_text())
    assert len(contract["panels"]) == 2
    assert contract["panels"][0]["metadata"]["source_products"] == [RISK]
    findings = FigureContractQualityValidator().audit_contract_file(contract_path=path, manuscript_facing=True)
    assert not [finding for finding in findings if finding.severity == "error"]
    suite = summarize_display_suite_status(context=context, plan=plan, evidence=evidence, run_dir=tmp_path, publication={})
    assert suite["display_primary_publication_absolute_risk_visual_present"]
    assert not any("absolute-risk panel" in error for error in suite["display_suite_errors"])
    # A registered risk table alone still cannot satisfy the visual requirement.
    contract["panels"] = contract["panels"][1:]
    path.write_text(json.dumps(contract))
    findings = FigureContractQualityValidator().audit_contract_file(contract_path=path, manuscript_facing=True)
    assert any(finding.severity == "error" and "only 1 panel" in finding.message for finding in findings)
    suite = summarize_display_suite_status(context=context, plan=plan, evidence=evidence, run_dir=tmp_path, publication={})
    assert not suite["display_suite_complete"]
    assert not suite["display_primary_publication_absolute_risk_visual_present"]
    assert any("absolute-risk panel" in error for error in suite["display_suite_errors"])


@pytest.mark.parametrize("defect", ["absent", "wrong_schema", "tampered"])
def test_missing_or_unverified_risk_cannot_be_rendered(tmp_path, monkeypatch, defect):
    _, plan, runtime, step, _, bindings = _case(tmp_path)
    assert RISK in step.inputs
    if defect == "absent":
        del bindings[RISK]
    elif defect == "wrong_schema":
        bindings[RISK]["product_contract"]["columns"] = ["risk_difference"]
    else:
        (tmp_path / bindings[RISK]["relative_path"]).write_text("unverified bytes\n")
    if defect == "tampered":
        with pytest.raises(RuntimeError, match="digest"):
            _render(tmp_path, monkeypatch, plan, runtime, step, bindings)
    else:
        decision = resolve_standard_executor(step, plan=plan, resolved_bindings=bindings)
        assert decision.claimed_by != "association_overview_figure"
        assert decision.claimed_by != "adjusted_association_figure"


@pytest.mark.parametrize("column,value", [
    ("exposure_column", "unrelated_exposure"), ("outcome_column", "unrelated_outcome"),
    ("row_role", "missing_overall"), ("outcome_denominator", 99), ("outcome_events", 31),
])
def test_risks_must_describe_the_primary_population(tmp_path, monkeypatch, column, value):
    def alter(frame):
        if column in {"exposure_column", "outcome_column"}:
            frame[column] = value
        else:
            frame.loc[frame.row_role.eq("overall"), column] = value
        return frame

    _, plan, runtime, step, _, bindings = _case(tmp_path, risk_change=alter)
    with pytest.raises(ValueError, match="absolute-risk context disagrees"):
        _render(tmp_path, monkeypatch, plan, runtime, step, bindings)


def test_runtime_rechecks_the_risk_schema(tmp_path):
    _, _, _, step, _, bindings = _case(tmp_path, risk_change=lambda frame: frame.drop(columns="exposure_pct"))
    with pytest.raises(ValueError, match="distribution owner's schema"):
        run_association_overview_figure(
            run_dir=tmp_path, out_dir=tmp_path / "out", step_id=step.step_id, figure_product="primary",
            resolved_inputs={"step_id": step.step_id, "inputs": bindings}, distribution_input=RISK,
        )


@pytest.mark.parametrize("defect", ["extra_input", "two_risk_products", "missing_consumption"])
def test_overview_consumes_only_the_declared_pair(tmp_path, defect):
    _, plan, _, step, _, bindings = _case(tmp_path)
    if defect == "missing_consumption":
        step.input_consumption_contracts = [item for item in step.input_consumption_contracts if item.input_key != RISK]
    else:
        key = "table:unrelated" if defect == "extra_input" else "table:exposure_outcome_distribution"
        step.inputs.append(key)
        bindings[key] = bindings[RISK]
    assert resolve_standard_executor(step, plan=plan, resolved_bindings=bindings).claimed_by != "association_overview_figure"


def test_runtime_cannot_relabel_an_unregistered_risk_product(tmp_path):
    with pytest.raises(ValueError, match="unsupported observed-risk input"):
        run_association_overview_figure(
            run_dir=tmp_path, out_dir=tmp_path / "out", step_id="figure", figure_product="primary",
            resolved_inputs={}, distribution_input="table:unregistered_risk",
        )


@pytest.mark.parametrize("kind,method,scale,estimate", [
    ("logistic", "statsmodels_logit_mle", "odds_ratio", 3.0),
    ("glm_poisson", "statsmodels_glm_poisson_log_robust", "risk_ratio", 2.0),
])
def test_host_binary_estimators_keep_their_scale_in_the_overview(tmp_path, kind, method, scale, estimate):
    from easyicu.research_agent.execution.runners.adjusted_association_executor import run_adjusted_association_from_env
    from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import _distribution_rows
    from easyicu.research_agent.schema import AnalysisStep, ExposureOutcomeDistributionSpec

    data = pd.DataFrame([
        {"exposure": exposure, "outcome": int(row < events)}
        for exposure, events in ((0, 10), (1, 20)) for row in range(40)
    ])
    run_adjusted_association_from_env(
        frame=data, output_dir=tmp_path, requirement_id="primary", exposure="exposure", outcome="outcome",
        covariates=[], model_terms=[dict(name="exposure", role="exposure", coding="binary",
                                       levels=["0", "1"], reference_level="0", transform="treatment_contrast")],
        estimator_kind=kind, method_family=method, analysis_set="complete_case", analysis_role="primary",
        emit_step_summary=False,
    )
    risk_path = tmp_path / "absolute_risk_context.csv"
    pd.DataFrame(_distribution_rows(data, spec=ExposureOutcomeDistributionSpec(
        exposure="exposure", exposure_levels=[0, 1], outcome="outcome", outcome_levels=[0, 1],
        outcome_positive_value=1, level_match_policy="exact_typed", denominator_policy="all_declared_rows",
        missing_outcome_policy="structural_absence_is_non_event", confidence_level=0.95,
    ))).to_csv(risk_path, index=False)
    step = AnalysisStep(
        step_id="figure", intent="Show observed risks and the declared adjusted effect", method="visualization",
        inputs=[RISK, EFFECT], expected_outputs=["figure:overview"],
        input_consumption_contracts=[dict(input_key=key, mode="all_rows") for key in (RISK, EFFECT)],
    )
    evidence = EvidenceStore(tmp_path)
    bindings = {}
    for product, path in ((EFFECT, tmp_path / "adjusted_association_estimates.csv"), (RISK, risk_path)):
        ref = evidence.register_file(
            kind="table", source_path=path, evidence_id=path.stem, produced_by_step=path.stem,
            description="Synthetic binary association", producer="runner", generation_mode="deterministic_standard",
        )
        binding = _resolved_typed_input_binding(
            input_name=product, evidence_ref=ref, evidence_records=evidence.records(), run_dir=tmp_path,
        )
        bindings[product] = _attach_verified_consumption_contract(step=step, input_name=product, binding=binding)
    assert resolve_standard_executor(step, plan=None, resolved_bindings=bindings).claimed_by == "association_overview_figure"
    manifest = {"step_id": step.step_id, "inputs": bindings}
    run_association_overview_figure(
        run_dir=tmp_path, out_dir=tmp_path / "figure", step_id=step.step_id,
        figure_product="overview", resolved_inputs=manifest, distribution_input=RISK,
    )
    plotted = pd.read_csv(tmp_path / "figure/overview_association_source_data.csv")
    assert plotted.effect_scale.tolist() == [scale]
    assert plotted.estimate.tolist() == pytest.approx([estimate])
    pd.testing.assert_frame_equal(plotted, pd.read_csv(tmp_path / "adjusted_association_estimates.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "figure/overview_distribution_source_data.csv"),
                                  pd.read_csv(risk_path))
    assert ("risk ratio" if kind == "glm_poisson" else "odds ratio") in (tmp_path / "figure/overview.svg").read_text()
    # The integration must preserve the complete host schema check at both boundaries.
    bindings[EFFECT]["product_contract"]["columns"].remove("outcome_proportion")
    assert resolve_standard_executor(step, plan=None, resolved_bindings=bindings).claimed_by != "association_overview_figure"
    with pytest.raises(ValueError, match="not written by the host model owner"):
        run_association_overview_figure(
            run_dir=tmp_path, out_dir=tmp_path / "refused", step_id=step.step_id,
            figure_product="overview", resolved_inputs=manifest, distribution_input=RISK,
        )
