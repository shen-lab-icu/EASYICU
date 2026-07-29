"""The study's primary estimate stops being written by the stochastic coder.

``table:adjusted_association_estimates`` is the most frequently declared product
no deterministic owner could emit -- 233 of 1812 recorded real steps -- and it
is the paper's primary result.  175 of those 233 declare exactly one model
requirement, which is the shape this owner claims.

``test_the_real_fresh19_primary_step_is_claimed_once_covariates_are_declared``
is the load-bearing one: it takes the real recorded step that died in fresh19,
adds only the adjustment set, and shows the host now owns it.

``test_a_model_that_cannot_be_fitted_raises_instead_of_writing_a_null_row`` is
the other: a null primary effect is not a weaker result, it is an absent one,
and the accumulated repair guidance in plan_utils exists because scripts kept
satisfying this contract with nulls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels.api")

from easyicu.research_agent.execution.runners.adjusted_association_executor import (  # noqa: E402
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
    AdjustedAssociationError,
    adjusted_association_executor_code,
    adjusted_association_executor_owns_step,
    adjusted_association_executor_scaffold,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.execution.runners.selection import (  # noqa: E402
    select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep  # noqa: E402

_FIXTURE = Path(__file__).parent / "fixtures" / "real_plan_steps_fresh17_fresh19.json"
_REAL_STEP_ID = "07_primary_adjusted_association"
_COVARIATES = ["age", "sex", "charlson_max"]


def _real_step_payload() -> dict:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plan = next(e for e in document["plans"] if e["label"] == "fresh19")["plan"]
    return next(s for s in plan["steps"] if s["step_id"] == _REAL_STEP_ID)


def _step(*, covariates=_COVARIATES, **overrides) -> AnalysisStep:
    payload = json.loads(json.dumps(_real_step_payload()))
    if covariates is not None:
        payload["model_requirements"][0]["covariates"] = list(covariates)
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def _cohort(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(20260729)
    exposure = rng.integers(0, 2, n).astype(float)
    age = rng.normal(65.0, 12.0, n)
    sex = rng.integers(0, 2, n).astype(float)
    charlson = rng.integers(0, 8, n).astype(float)
    logit = -3.0 + 1.4 * exposure + 0.03 * (age - 65.0) + 0.15 * charlson
    outcome = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return pd.DataFrame(
        {
            "sep3_sofa2_max": exposure,
            "age": age,
            "sex": sex,
            "charlson_max": charlson,
            "death": outcome,
        }
    )


def _run(tmp_path, frame=None, **overrides):
    import os

    payload = {
        "requirement_id": "primary_full_cohort_logistic",
        "exposure": "sep3_sofa2_max",
        "outcome": "death",
        "covariates": _COVARIATES,
        "estimator_kind": "logistic",
        "analysis_set": "source_aware",
    }
    payload.update(overrides)
    previous = os.environ.get("STEP_OUT_DIR")
    os.environ["STEP_OUT_DIR"] = str(tmp_path)
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


# --------------------------------------------------------------------------
# ownership


def test_the_real_fresh19_primary_step_is_claimed_once_covariates_are_declared() -> (
    None
):
    """The step that died in fresh19, with only the adjustment set added."""

    without = _step(covariates=None)
    with_set = _step()

    assert adjusted_association_executor_owns_step(without) is False
    assert adjusted_association_executor_owns_step(with_set) is True

    plan = AnalysisPlan.model_validate(
        {
            "research_question": "q",
            "analysis_type": "association_study",
            "steps": [json.loads(with_set.model_dump_json())],
        }
    )
    selection = select_standard_executor(with_set, plan=plan)

    assert selection is not None
    assert selection.analysis_kind == "adjusted_association_estimates"


def test_an_undeclared_adjustment_set_is_not_claimed() -> None:
    """No declaration means the coder path, exactly as before this owner."""

    assert adjusted_association_executor_owns_step(_step(covariates=None)) is False


def test_a_deliberately_unadjusted_model_is_claimed() -> None:
    """`[]` is a declaration; it is not the same as saying nothing."""

    assert adjusted_association_executor_owns_step(_step(covariates=[])) is True


def test_two_model_requirements_are_not_claimed() -> None:
    """bind_primary_output binds a one-row table, so two models is not this."""

    payload = json.loads(json.dumps(_real_step_payload()))
    first = payload["model_requirements"][0]
    first["covariates"] = _COVARIATES
    second = dict(first)
    second["requirement_id"] = "secondary_landmark_logistic"
    second["analysis_role"] = "secondary"
    payload["model_requirements"] = [first, second]

    assert (
        adjusted_association_executor_owns_step(AnalysisStep.model_validate(payload))
        is False
    )


def test_an_unimplemented_method_family_is_not_claimed() -> None:
    """Fitting OLS for a declared quantile regression answers another question."""

    payload = json.loads(json.dumps(_real_step_payload()))
    payload["model_requirements"][0].update(
        covariates=_COVARIATES,
        outcome_type="continuous",
        method_family="quantile_regression",
    )

    assert (
        adjusted_association_executor_owns_step(AnalysisStep.model_validate(payload))
        is False
    )


def test_an_extra_declared_product_is_not_claimed() -> None:
    step = _step(
        expected_outputs=[
            "table:adjusted_association_estimates",
            "figure:adjusted_association_forest",
        ]
    )

    assert adjusted_association_executor_owns_step(step) is False


# --------------------------------------------------------------------------
# the fit


def test_the_declared_model_is_fitted_and_the_row_matches_the_reader(tmp_path) -> None:
    """bind_primary_output requires one row, fit_status=fitted, finite bounds."""

    summary = _run(tmp_path)
    table = pd.read_csv(tmp_path / "adjusted_association_estimates.csv")

    assert list(table.columns) == list(ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["fit_status"] == "fitted"
    assert row["effect_scale"] == "odds_ratio"
    assert row["exposure"] == "sep3_sofa2_max"
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["covariates"] == "age;sex;charlson_max"
    assert summary["output_files"] == {
        "table:adjusted_association_estimates": "adjusted_association_estimates.csv"
    }
    assert summary["primary_or"] == pytest.approx(row["estimate"])


def test_the_reported_effect_is_the_exposure_not_a_covariate(tmp_path) -> None:
    """The named-term guarantee, checked through the executor.

    The cohort is built with a strong positive exposure effect and covariates
    that do not move the outcome that way, so a fit reporting the wrong column
    would land somewhere else entirely.
    """

    adjusted = _run(tmp_path)
    crude = _run(tmp_path, covariates=[])

    assert adjusted["adjusted_effect"] == pytest.approx(np.exp(1.4), rel=0.25)
    assert adjusted["adjusted_effect"] != pytest.approx(crude["adjusted_effect"])


def test_a_model_that_cannot_be_fitted_raises_instead_of_writing_a_null_row(
    tmp_path,
) -> None:
    """A null primary effect is not a weaker result -- it is an absent one."""

    frame = _cohort()
    frame["death"] = 0.0  # one outcome class: nothing to estimate

    with pytest.raises(AdjustedAssociationError, match="could not be fitted"):
        _run(tmp_path, frame=frame)

    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


def test_a_missing_declared_column_names_itself(tmp_path) -> None:
    frame = _cohort().drop(columns=["charlson_max"])

    with pytest.raises(AdjustedAssociationError, match="charlson_max"):
        _run(tmp_path, frame=frame)


# --------------------------------------------------------------------------
# the host/agent boundary


def test_the_step_has_no_agent_writable_region_at_all() -> None:
    """Sealing the values while leaving the call editable protects nothing.

    An earlier draft of this executor put the call in the agent body and kept
    the declared model in the prologue.  That reads as safe and is not: a
    contract repair rewriting the body could call the same host function with a
    different exposure or a shorter adjustment set, and the sealed declaration
    above it would sit there unread.  Everything this step does is fixed by the
    plan, so the body is empty and any edit at all is detected.
    """

    scaffold = adjusted_association_executor_scaffold(_step())

    assert scaffold.body == ""
    assert "sep3_sofa2_max" in scaffold.prologue
    assert "'age', 'sex', 'charlson_max'" in scaffold.prologue.replace('"', "'")
    assert "run_adjusted_association_from_env(" in scaffold.prologue

    for edit in (
        ("'charlson_max'", "'charlson_max_v2'"),  # a shorter adjustment set
        ("'sep3_sofa2_max'", "'age'"),  # a different exposure
        ("**declared_model,", "exposure='age', outcome='death',"),  # bypass the seal
    ):
        rewritten = scaffold.assembled().replace(*edit)
        assert rewritten != scaffold.assembled(), edit
        assert scaffold.host_regions_intact(rewritten) is False, edit

    assert scaffold.host_regions_intact(scaffold.assembled()) is True


def test_the_generated_script_is_the_assembled_scaffold() -> None:
    step = _step()

    assert (
        adjusted_association_executor_code(step)
        == adjusted_association_executor_scaffold(step).assembled()
    )


def test_the_generated_script_compiles() -> None:
    compile(adjusted_association_executor_code(_step()), "<generated>", "exec")


def test_an_unowned_step_cannot_be_rendered() -> None:
    with pytest.raises(ValueError, match="not owned"):
        adjusted_association_executor_code(_step(covariates=None))
