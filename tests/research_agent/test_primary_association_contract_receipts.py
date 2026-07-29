"""The owner of the paper's primary result must satisfy the host's own gate.

fresh25b (``run_20260729T151307_a6245e``) reached six of seven steps, five of
them host-owned and four with no provider call at all.  Step 06 --
``adjusted_association_estimates``, the study's primary estimate -- executed
correctly: return code 0 in 1.5 s, an executed digest equal to the
concept-approved one, and a complete summary carrying OR 1.566 (1.025, 2.395)
over 1000 stays and 102 deaths.

It was then failed by ``PrimaryModelContractValidator`` with four issues:
``missing_model_contracts``, ``required_model_missing``,
``exactly_one_primary_model_required`` and
``missing_term_level_coefficient_table``.  The host had built a deterministic
owner for the primary association that could not pass the host's own contract,
so every such step was a guaranteed dead step -- and the repair budget was then
spent asking a model to rewrite a script that is entirely host property.

Replaying ``_step_deterministic_contract_findings`` on that run's real plan,
context, bindings and regenerated outputs reproduced exactly that one error, and
reproduces zero after this fix.

The other half of the fix is a message: the validator's
``missing_term_level_coefficient_table`` named a column
(``estimate_or_odds_ratio``) that its own reader does not accept -- it takes
``estimate``, ``odds_ratio`` or ``or``.  A table written to the spelling the
error asks for is skipped, and the step is told again that its table is
missing; the repair loop reading that message has no other source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels.api")

from easyicu.research_agent.audits.validators import (  # noqa: E402
    PrimaryModelContractValidator,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (  # noqa: E402
    ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS,
    MODEL_CONTRACT_FIELDS,
    AdjustedAssociationError,
    _coefficient_rows,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.robustness.estimators import (  # noqa: E402
    EstimatorTerm,
    fit_estimator,
)
from easyicu.research_agent.schema import (  # noqa: E402
    AnalysisStep,
    CohortDescriptor,
    PlannedModelRequirement,
    ResearchContext,
)

_EXPOSURE = "sep3_sofa2_max"
_OUTCOME = "death"
#: ``sex`` holds words on purpose: the real cohort does, and a treatment-coded
#: contrast is the one term whose name is not its source column.
_COVARIATES = ["age", "sex", "charlson_first"]
_REQUIREMENT_ID = "primary_full_cohort_logistic"
_N = 3000


def _cohort() -> "pd.DataFrame":
    rng = np.random.default_rng(20260729)
    exposure = rng.integers(0, 2, _N).astype(float)
    age = rng.normal(65.0, 12.0, _N)
    charlson = rng.integers(0, 8, _N).astype(float)
    logit = -3.0 + 1.4 * exposure + 0.03 * (age - 65.0) + 0.15 * charlson
    return pd.DataFrame(
        {
            _EXPOSURE: exposure,
            "age": age,
            "sex": rng.choice(["Male", "Female"], _N),
            "charlson_first": charlson,
            _OUTCOME: (rng.random(_N) < 1.0 / (1.0 + np.exp(-logit))).astype(float),
        }
    )


def _requirement() -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id=_REQUIREMENT_ID,
        outcome=_OUTCOME,
        outcome_type="binary",
        method_family="binary_logistic_regression",
        exposure_source=_EXPOSURE,
        analysis_role="primary",
        analysis_set="source_aware",
        required_for_step_success=True,
        covariates=list(_COVARIATES),
    )


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="06_primary_adjusted_association",
        intent="Estimate the adjusted association between the exposure and death.",
        method="adjusted_association_models",
        inputs=["artifact:analysis_cohort", _EXPOSURE, _OUTCOME, *_COVARIATES],
        expected_outputs=["table:adjusted_association_estimates"],
        planned_analysis_role="primary",
        model_requirements=[_requirement()],
    )


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is the exposure associated with in-hospital death?",
        cohort=CohortDescriptor(
            cohort_name="test",
            database="mock",
            n_patients=_N,
            n_stays=_N,
        ),
        variables=[],
        primary_exposure=_EXPOSURE,
        target_outcome=_OUTCOME,
    )


def _run(out_dir: Path, frame=None, **overrides):
    payload = {
        "requirement_id": _REQUIREMENT_ID,
        "exposure": _EXPOSURE,
        "outcome": _OUTCOME,
        "covariates": list(_COVARIATES),
        "estimator_kind": "logistic",
        "analysis_set": "source_aware",
        "analysis_role": "primary",
        "method_family": "binary_logistic_regression",
    }
    payload.update(overrides)
    previous = os.environ.get("STEP_OUT_DIR")
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    try:
        return run_adjusted_association_from_env(
            frame=_cohort() if frame is None else frame,
            cohort_path=Path("cohort.parquet"),
            **payload,
        )
    finally:
        if previous is None:
            os.environ.pop("STEP_OUT_DIR", None)
        else:
            os.environ["STEP_OUT_DIR"] = previous


def _audit(out_dir: Path, summary, cohort: "pd.DataFrame", tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path)
    return PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary=summary,
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )


def test_the_owner_now_satisfies_the_gate_that_killed_the_real_step(
    tmp_path: Path,
) -> None:
    """The load-bearing one: this exact shape produced four issues before."""

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cohort = _cohort()
    summary = _run(out_dir, frame=cohort)

    findings = _audit(out_dir, summary, cohort, tmp_path)

    assert [f.message for f in findings if f.severity == "error"] == []


def test_the_contract_carries_every_field_the_validator_fixes(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    summary = _run(out_dir)

    contracts = summary["model_contracts"]
    assert len(contracts) == 1
    missing = [field for field in MODEL_CONTRACT_FIELDS if field not in contracts[0]]
    assert missing == []
    # The roster is keyed by requirement_id; a contract without one is a model
    # nobody asked for.
    assert contracts[0]["requirement_id"] == _REQUIREMENT_ID


def test_the_effect_column_is_one_the_validators_reader_actually_accepts(
    tmp_path: Path,
) -> None:
    """Guards the trap directly: the error's spelling is not the reader's."""

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    _run(out_dir)

    table = pd.read_csv(out_dir / "adjusted_association_coefficients.csv")
    accepted = {"estimate", "odds_ratio", "or"}
    assert accepted.intersection(table.columns)
    assert list(table.columns) == list(ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS)
    # The reader that decides whether a table exists at all must find this one.
    assert PrimaryModelContractValidator._coefficient_rows(out_dir) is not None


def test_the_message_names_only_columns_its_own_reader_takes() -> None:
    """A fail-closed message whose implied fix does not satisfy the check."""

    out_dir_without_table = Path(__file__).parent
    assert PrimaryModelContractValidator._coefficient_rows(out_dir_without_table) is (
        None
    )
    # Read the emitted detail rather than the source, so a future edit that
    # reintroduces the description-as-a-name is caught where a reader sees it.
    validator = PrimaryModelContractValidator()
    findings = validator.audit(
        step=_step(),
        step_summary={"model_contracts": []},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir_without_table,
        cohort_path=Path("nonexistent.parquet"),
    )
    issues = [
        issue
        for finding in findings
        for issue in (finding.detail or {}).get("issues", [])
        if issue.get("issue") == "missing_term_level_coefficient_table"
    ]
    assert issues, "the missing-table issue must still be raised"
    named = set(issues[0]["required_columns"])
    assert "estimate_or_odds_ratio" not in named
    assert set(issues[0]["required_effect_column_one_of"]) == {
        "estimate",
        "odds_ratio",
        "or",
    }


def test_a_treatment_coded_contrast_names_the_cohort_column_it_came_from(
    tmp_path: Path,
) -> None:
    """Lineage the validator checks: every non-intercept row names one column."""

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cohort = _cohort()
    _run(out_dir, frame=cohort)

    table = pd.read_csv(out_dir / "adjusted_association_coefficients.csv")
    contrast = table[table["term"] == "sex=Male"]
    assert len(contrast) == 1
    assert contrast.iloc[0]["source_variable"] == "sex"
    assert contrast.iloc[0]["term_role"] == "adjustment"

    non_intercept = table[table["term_role"] != "intercept"]
    for source in non_intercept["source_variable"]:
        assert list(cohort.columns).count(source) == 1

    assert set(table["term_role"]) <= {
        "intercept",
        "exposure",
        "availability",
        "adjustment",
    }
    assert list(table["term_role"]).count("exposure") == 1


def test_the_term_table_reports_the_same_number_as_the_headline(
    tmp_path: Path,
) -> None:
    """Two tables that disagree are worse than one, so they must not."""

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    summary = _run(out_dir)

    table = pd.read_csv(out_dir / "adjusted_association_coefficients.csv")
    exposure_row = table[table["term_role"] == "exposure"].iloc[0]
    assert exposure_row["estimate"] == pytest.approx(summary["primary_estimate"])
    assert exposure_row["ci_low"] == pytest.approx(
        summary["primary_estimate_interval"][0]
    )
    assert exposure_row["ci_high"] == pytest.approx(
        summary["primary_estimate_interval"][1]
    )
    assert exposure_row["effect_scale"] == summary["effect_scale"] == "odds_ratio"


def test_the_fit_reports_every_design_column_including_the_intercept() -> None:
    """Only the fit has them; a second fit would be a second estimate."""

    cohort = _cohort()
    result = fit_estimator(
        cohort=None,
        X=cohort[[_EXPOSURE, *_COVARIATES]],
        y=cohort[_OUTCOME],
        kind="logistic",
        term=_EXPOSURE,
    )

    assert result.converged is True
    terms = {term.term: term for term in result.terms}
    # exposure + age + charlson_first + sex=Male + const
    assert set(terms) == {"const", _EXPOSURE, "age", "charlson_first", "sex=Male"}
    assert terms[_EXPOSURE].estimate == pytest.approx(result.point_estimate)
    assert terms["sex=Male"].source_variable == "sex"
    assert terms["age"].source_variable == "age"
    # The logistic result reports odds ratios, so its terms do too -- an odds
    # ratio is positive, a log-odds coefficient need not be.
    assert all(term.estimate > 0 for term in result.terms)


def test_a_term_the_declaration_does_not_name_is_refused_not_relabelled() -> None:
    """A role has no honest default, so the classifier has no else branch.

    Tested at the helper, not through ``run_adjusted_association_from_env``:
    that path builds the design from ``[exposure, *covariates]``, so it cannot
    itself produce an undeclared term.  The refusal exists because the only
    alternative for a second caller is to stamp ``adjustment`` on a column
    nobody declared -- a label the plan cannot be checked against.
    """

    stray = EstimatorTerm(
        term="ventilation_hours",
        source_variable="ventilation_hours",
        estimate=1.1,
        ci_low=0.9,
        ci_high=1.4,
        se=0.1,
    )

    with pytest.raises(AdjustedAssociationError) as excinfo:
        _coefficient_rows(
            [stray],
            model_id=_REQUIREMENT_ID,
            exposure=_EXPOSURE,
            adjustment=_COVARIATES,
            effect_scale="odds_ratio",
        )

    assert "ventilation_hours" in str(excinfo.value)


def test_a_linear_model_reports_coefficients_not_odds_ratios(
    tmp_path: Path,
) -> None:
    """The scale travels with the family, in both tables."""

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cohort = _cohort()
    cohort["los_days"] = cohort["age"] * 0.05 + cohort[_EXPOSURE] * 1.5
    summary = _run(
        out_dir,
        frame=cohort,
        outcome="los_days",
        estimator_kind="linear",
        method_family="linear_regression",
    )

    assert summary["model_contracts"][0]["outcome_type"] == "continuous"
    assert summary["model_contracts"][0]["fit_method"] == "statsmodels_ols"
    table = pd.read_csv(out_dir / "adjusted_association_coefficients.csv")
    assert set(table["effect_scale"]) == {"coefficient"}
