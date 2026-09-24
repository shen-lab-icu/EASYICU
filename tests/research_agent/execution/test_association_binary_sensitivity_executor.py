"""Host-owned binary sensitivity variants: first-stay restriction and covariate RCS.

These steps carry the agent-coded ``association_freeform_v1`` capability with a
closed result contract. The executor under test claims only the two strategies
it can derive from typed coordinates, refits the parent model on the bound
cohort, and writes the same ``analysis_rows`` result the contract validates.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts
from easyicu.research_agent.audits import StepSummaryIntegrityValidator
from easyicu.research_agent.authority.typed_binding import (
    _write_host_input_binding_receipts,
)
from easyicu.research_agent.contracts.association_execution import (
    association_binary_sensitivity_result_issues,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    run_adjusted_association_from_env,
)
from easyicu.research_agent.execution.runners.association_binary_sensitivity_executor import (
    ASSOCIATION_BINARY_SENSITIVITY_ANALYSIS_KIND,
    AssociationBinarySensitivityError,
    association_binary_sensitivity_executor_code,
    association_binary_sensitivity_executor_owns_step,
    resolve_binary_sensitivity_variant,
    run_association_binary_sensitivity,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.execution.runners.typed_input_binding import (
    TypedInputBindingError,
)
from easyicu.research_agent.schema import AnalysisPlan


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260923)
    patients = 300
    n = patients * 2
    stage = rng.integers(0, 4, size=n)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (
        1.0 + np.exp(-(-3.0 + 0.4 * stage + 0.015 * (age - 64.0) + 0.0004 * (age - 64.0) ** 2))
    )
    death = rng.binomial(1, probability, size=n)
    return pd.DataFrame(
        {
            "patient_stay_id": [f"p{p}:s{s}" for p in range(patients) for s in (1, 2)],
            "first_icu_stay": np.tile([1.0, 0.0], patients),
            "aki_stage_strict": stage,
            "death": death,
            "age": age,
            "sex": sex,
        }
    )


def _plan(*, sensitivity_inputs: list[str] | None = None, method: str = "first_stay_association", functional_form: dict | None = None) -> AnalysisPlan:
    sensitivity = {
        "step_id": "sensitivity_first_icu_stay_only",
        "planned_analysis_role": "sensitivity",
        "intent": "Refit on first ICU stays only.",
        "inputs": sensitivity_inputs
        or [
            "aki_stage_strict",
            "death",
            "age",
            "sex",
            "first_icu_stay",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ],
        "expected_outputs": ["table:sensitivity_first_icu_stay_only"],
        "method": method,
        "scientific_capability": "association_freeform_v1",
        "sensitivity_spec_ids": ["first_icu_stay_only"],
    }
    if "table:adjusted_association_estimates" in sensitivity["inputs"]:
        sensitivity["input_consumption_contracts"] = [
            {"input_key": "table:adjusted_association_estimates", "mode": "all_rows"}
        ]
    if functional_form is not None:
        sensitivity = {
            **sensitivity,
            "step_id": "age_functional_form",
            "intent": "Refit with age as a restricted cubic spline.",
            "expected_outputs": ["table:age_functional_form_sensitivity"],
            "sensitivity_spec_ids": ["age_restricted_cubic_spline"],
            "functional_form_spec": functional_form,
        }
    return AnalysisPlan.model_validate(
        {
            "research_question": "Compare KDIGO stages with in-hospital mortality.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "define_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the analysis cohort.",
                    "inputs": ["death"],
                    "expected_outputs": ["artifact:analysis_cohort"],
                    "method": "cohort_definition_and_attrition",
                },
                {
                    "step_id": "primary_adjusted_association",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate adjusted stage contrasts.",
                    "inputs": [
                        "artifact:analysis_cohort",
                        "aki_stage_strict",
                        "death",
                        "age",
                        "sex",
                        "patient_stay_id",
                    ],
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "method": "adjusted_association_models",
                    "model_requirements": [
                        {
                            "requirement_id": "primary_stage_model",
                            "outcome": "death",
                            "outcome_type": "binary",
                            "method_family": "statsmodels_logit_mle",
                            "exposure_source": "aki_stage_strict",
                            "analysis_role": "primary",
                            "analysis_set": "source_aware",
                            "covariates": ["age", "sex"],
                            "model_terms": [
                                {
                                    "name": "aki_stage_strict",
                                    "role": "exposure",
                                    "coding": "categorical",
                                    "levels": ["0", "1", "2", "3"],
                                    "reference_level": "0",
                                    "transform": "treatment_contrast",
                                },
                                {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
                                {
                                    "name": "sex",
                                    "role": "covariate",
                                    "coding": "binary",
                                    "levels": ["Female", "Male"],
                                    "reference_level": "Female",
                                    "transform": "treatment_contrast",
                                },
                            ],
                            "exposure_levels": ["0", "1", "2", "3"],
                            "exposure_reference_level": "0",
                            "primary_contrast_level": "3",
                            "dependence": {
                                "group_source": "patient_stay_id",
                                "group_derivation": "prefix_before_delimiter",
                                "delimiter": ":s",
                            },
                        }
                    ],
                },
                sensitivity,
            ],
        }
    )


FUNCTIONAL_FORM = {
    "target_column": "age",
    "comparison": "restricted_cubic_spline_vs_linear",
    "knot_quantiles": [0.1, 0.5, 0.9],
}


def test_owns_only_derivable_first_stay_and_functional_form_variants() -> None:
    first_stay = _plan()
    step = first_stay.steps[2]
    variant = resolve_binary_sensitivity_variant(step, plan=first_stay)
    assert variant is not None and variant.strategy == "first_stay"
    assert variant.restriction_column == "first_icu_stay"
    assert variant.parent_step_id == "primary_adjusted_association"
    selection = select_standard_executor(step, plan=first_stay)
    assert selection is not None
    assert selection.analysis_kind == ASSOCIATION_BINARY_SENSITIVITY_ANALYSIS_KIND

    form = _plan(
        method="restricted_cubic_spline_sensitivity",
        functional_form=FUNCTIONAL_FORM,
        sensitivity_inputs=[
            "age",
            "aki_stage_strict",
            "death",
            "sex",
            "patient_stay_id",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ],
    )
    variant = resolve_binary_sensitivity_variant(form.steps[2], plan=form)
    assert variant is not None and variant.strategy == "functional_form"
    assert variant.target_column == "age" and variant.knot_quantiles == (0.1, 0.5, 0.9)

    # Two candidate restriction columns: the host would have to choose, so it declines.
    ambiguous = _plan(
        sensitivity_inputs=[
            "aki_stage_strict",
            "death",
            "age",
            "sex",
            "first_icu_stay",
            "another_flag",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ]
    )
    assert not association_binary_sensitivity_executor_owns_step(ambiguous.steps[2], plan=ambiguous)
    assert select_standard_executor(ambiguous.steps[2], plan=ambiguous) is None
    # A spline on a categorical covariate is not the declared comparison.
    categorical_target = _plan(
        method="restricted_cubic_spline_sensitivity",
        functional_form={**FUNCTIONAL_FORM, "target_column": "sex"},
    )
    assert not association_binary_sensitivity_executor_owns_step(
        categorical_target.steps[2], plan=categorical_target
    )
    # A method the host does not implement stays agent-coded.
    other = _plan(method="multiple_imputation_sensitivity")
    assert not association_binary_sensitivity_executor_owns_step(other.steps[2], plan=other)
    # Without the parent product beside the cohort the schema itself refuses
    # the capability, so there is no orphan shape for the owner to consider.
    with pytest.raises(ValueError, match="association_binary_sensitivity_shape_invalid"):
        _plan(
            sensitivity_inputs=[
                "aki_stage_strict", "death", "age", "sex", "first_icu_stay", "artifact:analysis_cohort",
            ]
        )


def test_the_host_seals_both_typed_inputs_of_an_owned_refit(tmp_path) -> None:
    """The bound cohort and the parent table each get a host receipt.

    A refit declares two typed inputs, so the generic sole-cohort rule answers
    "none" for it; sealing the parent table alone left the integrity gate
    refusing every owned refit bound to a typed cohort.
    """

    form = _plan(
        method="restricted_cubic_spline_sensitivity",
        functional_form=FUNCTIONAL_FORM,
        sensitivity_inputs=[
            "age",
            "aki_stage_strict",
            "death",
            "sex",
            "patient_stay_id",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ],
    )
    both = ("artifact:analysis_cohort", "table:adjusted_association_estimates")
    for plan in (_plan(), form):
        selection = select_standard_executor(plan.steps[2], plan=plan)
        assert selection is not None
        assert selection.consumed_input_keys == both

    cohort_path = tmp_path / "analysis_cohort.parquet"
    _frame().to_parquet(cohort_path, index=False)
    parent_path = tmp_path / "adjusted_association_estimates.csv"
    pd.DataFrame({"requirement_id": ["primary"], "estimate": [1.2]}).to_csv(
        parent_path, index=False
    )
    bindings = {
        "artifact:analysis_cohort": {
            "absolute_path": str(cohort_path),
            "evidence_id": "cohort",
            "sha256": "a" * 64,
        },
        "table:adjusted_association_estimates": {
            "absolute_path": str(parent_path),
            "evidence_id": "parent",
            "sha256": "b" * 64,
        },
    }

    def coverage_findings(consumed: tuple[str, ...]) -> list:
        out_dir = tmp_path / ("out_" + str(len(consumed)))
        out_dir.mkdir()
        summary = _write_host_input_binding_receipts(
            out_dir=out_dir,
            step_summary={"status": "ok"},
            resolved_input_bindings=bindings,
            consumed_input_keys=consumed,
        )
        return [
            finding
            for finding in StepSummaryIntegrityValidator().audit(
                step=form.steps[2],
                step_summary=summary,
                resolved_input_bindings=bindings,
                cohort_path=cohort_path,
            )
            if finding.detail.get("issue") == "input_binding_coverage_incomplete"
        ]

    assert coverage_findings(both) == []
    # The parent table alone is what the host sealed before.
    (missing,) = coverage_findings(("table:adjusted_association_estimates",))
    assert missing.detail["missing_input_keys"] == ["artifact:analysis_cohort"]


def _bind_parent(tmp_path, monkeypatch, plan: AnalysisPlan):
    frame = _frame()
    cohort_path = tmp_path / "analysis_cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    requirement = plan.steps[1].model_requirements[0]
    primary_dir = tmp_path / "primary"
    parent_summary = run_adjusted_association_from_env(
        requirement_id=requirement.requirement_id,
        exposure=requirement.exposure_source,
        outcome=requirement.outcome,
        covariates=list(requirement.covariates),
        model_terms=[term.model_dump(mode="json") for term in requirement.model_terms],
        estimator_kind="logistic",
        analysis_set=requirement.analysis_set,
        analysis_role="primary",
        method_family=requirement.method_family,
        primary_contrast_level=requirement.primary_contrast_level,
        dependence=requirement.dependence,
        typed_cohort_input=None,
        frame=frame,
        cohort_path=cohort_path,
        emit_step_summary=False,
        output_dir=primary_dir,
    )
    table_path = primary_dir / "adjusted_association_estimates.csv"
    manifest = {"step_id": plan.steps[2].step_id, "inputs": {}}
    for key, path in [
        ("artifact:analysis_cohort", cohort_path),
        ("table:adjusted_association_estimates", table_path),
    ]:
        bound = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["inputs"][key] = {
            "relative_path": str(path.relative_to(tmp_path)),
            "sha256": digest,
            "declared_kind": key.partition(":")[0],
            "evidence_kind": "table",
            "product": key.partition(":")[2],
            "evidence_id": key,
            "product_contract": {"columns": list(bound.columns), "row_count": len(bound)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
        }
    (tmp_path / "resolved_inputs.json").write_text(json.dumps(manifest), encoding="utf-8")
    out_dir = tmp_path / "sensitivity_out"
    for key, value in {
        "EASYICU_RUN_DIR": tmp_path,
        "EASYICU_STEP_ID": plan.steps[2].step_id,
        "EASYICU_RESOLVED_INPUTS_JSON": tmp_path / "resolved_inputs.json",
        "COHORT_PARQUET": cohort_path,
        "STEP_OUT_DIR": out_dir,
    }.items():
        monkeypatch.setenv(key, str(value))
    return frame, cohort_path, table_path, parent_summary, out_dir


def test_first_stay_variant_refits_the_parent_model_on_first_stays(tmp_path, monkeypatch) -> None:
    plan = _plan()
    frame, cohort_path, _, parent_summary, out_dir = _bind_parent(tmp_path, monkeypatch, plan)
    summary = run_association_binary_sensitivity(
        frame=frame,
        cohort_path=cohort_path,
        step=plan.steps[2],
        parent_step=plan.steps[1],
        run_dir=tmp_path,
        resolved_inputs=tmp_path / "resolved_inputs.json",
        out_dir=out_dir,
    )
    assert summary["deterministic_standard_analysis"] == ASSOCIATION_BINARY_SENSITIVITY_ANALYSIS_KIND
    assert association_binary_sensitivity_result_issues(plan.steps[2], summary) == ()
    [row] = summary["analysis_rows"]
    assert row["analysis_id"] == "first_icu_stay_only"
    assert row["n_restricted_rows"] == int((frame["first_icu_stay"] == 1).sum())
    assert row["n_stays"] == row["n_restricted_rows"]
    assert row["n_stays"] < parent_summary["n_total"]
    assert 0 < row["ci_low"] <= row["odds_ratio"] <= row["ci_high"]
    assert row["exposure_level"] == "3" and row["reference_level"] == "0"
    # One stay per patient leaves nothing to cluster: the variance drops to one row per group.
    assert row["cluster_count"] == row["n_stays"]
    assert summary["primary_reference"]["odds_ratio"] == pytest.approx(parent_summary["primary_estimate"])
    reporting = summary["reportable_sensitivity_results"]
    assert reporting["strategy"] == "first_stay" and reporting["covariate"] is None
    assert (reporting["estimate"], reporting["lower"], reporting["upper"]) == (
        row["odds_ratio"], row["ci_low"], row["ci_high"]
    )
    assert (reporting["n"], reporting["events"]) == (row["n_stays"], row["n_deaths"])
    [claim] = derive_scientific_claim_drafts(summary)
    assert claim.analysis_role == "sensitivity"
    assert claim.population.endswith("restricted to the first ICU stay of each patient")
    # A four-level exposure: the refit's claim names the contrast it reports.
    assert (reporting["exposure_level"], reporting["reference_level"]) == ("3", "0")
    assert claim.exposure == "aki_stage_strict=3 versus aki_stage_strict=0"
    table = pd.read_csv(out_dir / summary["output_files"]["table:sensitivity_first_icu_stay_only"])
    assert table.loc[0, "odds_ratio"] == pytest.approx(row["odds_ratio"])
    code = association_binary_sensitivity_executor_code(plan.steps[2], plan=plan)
    compile(code, "sensitivity_step", "exec")
    assert "run_association_binary_sensitivity(" in code
    assert "typed_cohort_input='artifact:analysis_cohort'" in code


def test_functional_form_variant_reports_spline_effect_and_nonlinearity(tmp_path, monkeypatch) -> None:
    plan = _plan(
        method="restricted_cubic_spline_sensitivity",
        functional_form=FUNCTIONAL_FORM,
        sensitivity_inputs=[
            "age",
            "aki_stage_strict",
            "death",
            "sex",
            "patient_stay_id",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ],
    )
    frame, cohort_path, _, parent_summary, out_dir = _bind_parent(tmp_path, monkeypatch, plan)
    summary = run_association_binary_sensitivity(
        frame=frame,
        cohort_path=cohort_path,
        step=plan.steps[2],
        parent_step=plan.steps[1],
        run_dir=tmp_path,
        resolved_inputs=tmp_path / "resolved_inputs.json",
        out_dir=out_dir,
    )
    assert association_binary_sensitivity_result_issues(plan.steps[2], summary) == ()
    [row] = summary["analysis_rows"]
    assert row["analysis_id"] == "age_restricted_cubic_spline"
    assert row["covariate"] == "age" and row["basis"] == "restricted_cubic_spline"
    assert row["n_stays"] == parent_summary["n_total"]
    assert row["n_deaths"] == parent_summary["n_events"]
    assert row["primary_or_linear_covariate"] == pytest.approx(parent_summary["primary_estimate"])
    assert row["nonlinearity_df"] == 1 and 0.0 <= row["nonlinearity_p_value"] <= 1.0
    assert row["log_or_delta_vs_linear"] == pytest.approx(
        np.log(row["odds_ratio"]) - np.log(row["primary_or_linear_covariate"])
    )
    assert len(row["knots"].split("|")) == 3
    receipt = summary["sensitivity_runtime_receipt"]
    assert receipt["strategy"] == "functional_form"
    reporting = summary["reportable_sensitivity_results"]
    assert reporting["strategy"] == "functional_form" and reporting["covariate"] == "age"
    [claim] = derive_scientific_claim_drafts(summary)
    assert claim.claim_id == "sensitivity_age_restricted_cubic_spline"
    assert (reporting["exposure_level"], reporting["reference_level"]) == ("3", "0")
    assert claim.exposure == "aki_stage_strict=3 versus aki_stage_strict=0"
    assert (claim.point_estimate, claim.interval_lower, claim.interval_upper) == (
        row["odds_ratio"], row["ci_low"], row["ci_high"]
    )
    assert any(column.startswith("age__rcs_") for column in receipt["design_columns"])
    assert (out_dir / "age_functional_form_sensitivity.csv").exists()


def test_variant_fails_closed_on_bad_flags_and_tampered_parent(tmp_path, monkeypatch) -> None:
    plan = _plan()
    frame, cohort_path, table_path, _, out_dir = _bind_parent(tmp_path, monkeypatch, plan)
    unknown = frame.copy()
    unknown.loc[unknown.index[:3], "first_icu_stay"] = 2.0
    with pytest.raises(AssociationBinarySensitivityError, match="not a 0/1 indicator"):
        run_association_binary_sensitivity(
            frame=unknown, cohort_path=cohort_path, step=plan.steps[2], parent_step=plan.steps[1],
            run_dir=tmp_path, resolved_inputs=tmp_path / "resolved_inputs.json", out_dir=out_dir,
        )
    none_first = frame.copy()
    none_first["first_icu_stay"] = 0.0
    with pytest.raises(AssociationBinarySensitivityError, match="no rows satisfy"):
        run_association_binary_sensitivity(
            frame=none_first, cohort_path=cohort_path, step=plan.steps[2], parent_step=plan.steps[1],
            run_dir=tmp_path, resolved_inputs=tmp_path / "resolved_inputs.json", out_dir=out_dir,
        )
    # The parent estimates table is digest-bound: editing it after binding fails closed.
    table = pd.read_csv(table_path)
    table.loc[0, "n"] += 1
    table.to_csv(table_path, index=False)
    with pytest.raises(TypedInputBindingError, match="digest_mismatch"):
        run_association_binary_sensitivity(
            frame=frame, cohort_path=cohort_path, step=plan.steps[2], parent_step=plan.steps[1],
            run_dir=tmp_path, resolved_inputs=tmp_path / "resolved_inputs.json", out_dir=out_dir,
        )
    # A step/parent pair the resolver does not own is refused rather than guessed.
    with pytest.raises(AssociationBinarySensitivityError, match="not one host-executable"):
        run_association_binary_sensitivity(
            frame=frame, cohort_path=cohort_path, step=plan.steps[2], parent_step=plan.steps[0],
            run_dir=tmp_path, resolved_inputs=tmp_path / "resolved_inputs.json", out_dir=out_dir,
        )
