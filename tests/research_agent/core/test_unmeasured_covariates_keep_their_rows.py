"""A declared covariate that was never measured keeps its rows as a state.

Under the default policy a model fits only rows where every declared
covariate was measured.  On routinely collected data that can discard most of
a cohort, and when measurement itself tracks the outcome it biases what is
left.  A model requirement may instead list covariates whose unmeasured rows
stay in the fit as an explicit unmeasured state.  These tests pin the one
owner of those rows and what a fit under it publishes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels.api")

from easyicu.research_agent.agents.plan_payload import (  # noqa: E402
    PlannerHostOwnedFieldError,
    _normalise_plan_payload,
    _planner_transport_schema,
    decode_planner_transport_payload,
    parse_runtime_plan_suffix,
)
from easyicu.research_agent.audits.validators import (  # noqa: E402
    PrimaryModelContractValidator,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (  # noqa: E402
    AssociationModelGridNonlinearTerm,
    AssociationModelGridVariant,
    CurrentCaseScientificAuthorityError,
    _refuse_missing_category_policy,
)
from easyicu.research_agent.contracts.model_terms import (  # noqa: E402
    ModelTermSpec,
    PlannedModelRequirement,
)
from easyicu.research_agent.execution.model_matrix import (  # noqa: E402
    ModelTermCompilationError,
    primary_model_rows,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (  # noqa: E402
    run_adjusted_association_from_env,
)
from easyicu.research_agent.execution.runners.association_binary_sensitivity_executor import (  # noqa: E402
    BinarySensitivityVariant,
    _refit_functional_form,
)
from easyicu.research_agent.execution.runners.association_model_grid_executor import (  # noqa: E402
    _variant_model,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (  # noqa: E402
    _matching_primary_contract,
    _verified_complete_case_equivalence,
    relabel_complete_case_replay,
)
from easyicu.research_agent.robustness.estimators import fit_estimator  # noqa: E402
from easyicu.research_agent.robustness.panel import RobustnessSpec  # noqa: E402
from easyicu.research_agent.schema import AnalysisStep  # noqa: E402

TERMS = [
    ModelTermSpec(name="exposure", role="exposure", coding="continuous", transform="identity"),
    ModelTermSpec(name="age", role="covariate", coding="continuous", transform="identity"),
    ModelTermSpec(name="lab", role="covariate", coding="continuous", transform="identity"),
    ModelTermSpec(
        name="group",
        role="covariate",
        coding="categorical",
        transform="treatment_contrast",
        levels=["a", "b", "c"],
        reference_level="a",
    ),
]


def _frame(n: int = 2000, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.normal(size=n)
    age = rng.normal(60.0, 10.0, n)
    lab = rng.normal(2.0, 1.0, n)
    group = rng.choice(["a", "b", "c"], n)
    # Unmeasured labs are the sicker half: missingness tracks the outcome.
    logit = -1.0 + 0.5 * exposure + 0.02 * (age - 60.0) + 0.4 * lab
    outcome = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    frame = pd.DataFrame(
        {"outcome": outcome, "exposure": exposure, "age": age, "lab": lab, "group": group}
    )
    lab_unmeasured = rng.random(n) < np.where(outcome == 1.0, 0.2, 0.6)
    frame.loc[lab_unmeasured, "lab"] = np.nan
    frame.loc[rng.random(n) < 0.1, "group"] = None
    return frame


def test_without_a_declared_policy_the_rows_are_the_complete_cases() -> None:
    frame = _frame()

    rows = primary_model_rows(frame, terms=TERMS, exposure="exposure", outcome="outcome")

    complete = frame.notna().all(axis=1)
    assert list(rows.design.index) == list(frame.index[complete])
    assert rows.missing_category_terms == ()
    assert not rows.design.isna().any().any()


def test_a_listed_covariate_keeps_its_unmeasured_rows_as_their_own_state() -> None:
    frame = _frame()

    rows = primary_model_rows(
        frame,
        terms=TERMS,
        exposure="exposure",
        outcome="outcome",
        missing_category_covariates=["lab", "group"],
    )

    assert len(rows.design) == len(frame)
    assert not rows.design.isna().any().any()
    by_covariate = {item.covariate: item for item in rows.missing_category_terms}
    lab, group = by_covariate["lab"], by_covariate["group"]
    assert lab.indicator == "lab__unmeasured"
    assert lab.fill_value == pytest.approx(float(frame["lab"].median()))
    assert lab.n_missing == int(frame["lab"].isna().sum())
    unmeasured = frame["lab"].isna()
    assert rows.design.loc[unmeasured, "lab"].eq(lab.fill_value).all()
    assert rows.design["lab__unmeasured"].eq(unmeasured.astype(float)).all()
    # A categorical covariate gains a level: its contrasts are zero there.
    assert group.indicator == "group__is_unmeasured"
    assert group.fill_value is None
    missing_group = frame["group"].isna()
    assert rows.design.loc[missing_group, ["group__is_b", "group__is_c"]].eq(0.0).all().all()
    assert rows.availability_columns == ("lab__unmeasured", "group__is_unmeasured")


def test_a_listed_covariate_that_is_always_measured_adds_nothing() -> None:
    frame = _frame().dropna(subset=["lab"])

    rows = primary_model_rows(
        frame,
        terms=TERMS,
        exposure="exposure",
        outcome="outcome",
        missing_category_covariates=["lab"],
    )

    assert rows.missing_category_terms == ()
    assert "lab__unmeasured" not in rows.design.columns


def test_the_exposure_estimate_does_not_depend_on_the_fill_value() -> None:
    # With a linear term and its indicator, any constant fill is absorbed by
    # the indicator: the median is a presentation choice, not a result.
    frame = _frame()
    rows = primary_model_rows(
        frame,
        terms=TERMS[:3],
        exposure="exposure",
        outcome="outcome",
        missing_category_covariates=["lab"],
    )
    zero_filled = rows.design.copy()
    zero_filled.loc[zero_filled["lab__unmeasured"].eq(1.0), "lab"] = 0.0
    outcome = frame.loc[rows.design.index, "outcome"]

    fits = [
        fit_estimator(
            cohort=None,
            X=design,
            y=outcome,
            kind="logistic",
            term="exposure",
            source_by_design_column=rows.source_by_design_column,
        )
        for design in (rows.design, zero_filled)
    ]

    assert fits[0].point_estimate == pytest.approx(fits[1].point_estimate, rel=1e-6)


@pytest.mark.parametrize(
    "listed,terms,code",
    [
        (["exposure"], TERMS, "missing_category_exposure_refused"),
        (["heart_rate"], TERMS, "missing_category_covariate_undeclared"),
        (
            ["lab", "lab_copy"],
            [
                *TERMS,
                ModelTermSpec(
                    name="lab_copy", role="covariate", coding="continuous", transform="identity"
                ),
            ],
            "missing_category_indicator_collinear",
        ),
    ],
)
def test_a_policy_the_fit_cannot_honour_is_refused(listed, terms, code) -> None:
    frame = _frame()
    frame["lab_copy"] = frame["lab"] * 2.0

    with pytest.raises(ModelTermCompilationError) as raised:
        primary_model_rows(
            frame,
            terms=terms,
            exposure="exposure",
            outcome="outcome",
            missing_category_covariates=listed,
        )

    assert raised.value.reason_code == code


def test_a_covariate_never_measured_on_a_fitting_row_is_refused() -> None:
    frame = _frame()
    frame["lab"] = np.nan

    with pytest.raises(ModelTermCompilationError) as raised:
        primary_model_rows(
            frame,
            terms=TERMS,
            exposure="exposure",
            outcome="outcome",
            missing_category_covariates=["lab"],
        )

    assert raised.value.reason_code == "missing_category_covariate_unobserved"


def _fit(tmp_path: Path, **overrides):
    payload = {
        "requirement_id": "primary",
        "exposure": "exposure",
        "outcome": "outcome",
        "covariates": ["age", "lab", "group"],
        "model_terms": [term.model_dump(mode="json") for term in TERMS],
        "estimator_kind": "logistic",
        "analysis_set": "source_aware",
        "analysis_role": "primary",
        "method_family": "binary_logistic_regression",
        "frame": _frame(),
        "cohort_path": Path("cohort.parquet"),
        "output_dir": tmp_path,
    }
    payload.update(overrides)
    return run_adjusted_association_from_env(**payload)


def test_the_fit_publishes_its_policy_and_names_unmeasured_terms(tmp_path: Path) -> None:
    summary = _fit(tmp_path, missing_category_covariates=["lab", "group"])

    contract = summary["model_contracts"][0]
    assert contract["baseline_missing_policy"] == "explicit_missing_category"
    assert contract["missing_category_covariates"] == ["lab", "group"]
    assert [item["term"] for item in contract["missing_category_terms"]] == [
        "lab__unmeasured",
        "group__is_unmeasured",
    ]
    assert contract["n"] == 2000
    assert summary["missing_category_terms"] == contract["missing_category_terms"]
    coefficients = pd.read_csv(tmp_path / summary["coefficient_table"])
    availability = coefficients.loc[coefficients["term_role"].eq("availability")]
    assert dict(zip(availability["term"], availability["source_variable"])) == {
        "lab__unmeasured": "lab",
        "group__is_unmeasured": "group",
    }


def test_a_fit_without_the_policy_publishes_what_it_always_did(tmp_path: Path) -> None:
    summary = _fit(tmp_path)

    contract = summary["model_contracts"][0]
    assert contract["baseline_missing_policy"] == "drop_missing_baseline"
    assert "missing_category_covariates" not in contract
    assert "missing_category_terms" not in summary
    assert contract["n"] == int(_frame().notna().all(axis=1).sum())
    coefficients = pd.read_csv(tmp_path / summary["coefficient_table"])
    assert not coefficients["term_role"].eq("availability").any()


def _requirement(**overrides) -> dict:
    payload = {
        "requirement_id": "primary",
        "outcome": "outcome",
        "outcome_type": "binary",
        "method_family": "binary_logistic_regression",
        "exposure_source": "exposure",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "covariates": ["age", "lab"],
    }
    payload.update(overrides)
    return payload


def test_a_requirement_without_the_policy_serialises_exactly_as_before() -> None:
    requirement = PlannedModelRequirement.model_validate(_requirement())

    assert "baseline_missing_handling" not in requirement.model_dump(mode="json")
    assert requirement.missing_category_covariates() == ()
    declared = PlannedModelRequirement.model_validate(
        _requirement(baseline_missing_handling={"covariates": ["lab"]})
    )
    assert declared.missing_category_covariates() == ("lab",)
    assert declared.model_dump(mode="json")["baseline_missing_handling"] == {
        "policy": "explicit_missing_category",
        "covariates": ["lab"],
        "continuous_fill": "observed_median",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"analysis_set": "complete_case", "baseline_missing_handling": {"covariates": ["lab"]}},
        {"baseline_missing_handling": {"covariates": ["heart_rate"]}},
        {"baseline_missing_handling": {"covariates": []}},
        {"baseline_missing_handling": {"covariates": ["lab", "lab"]}},
    ],
)
def test_a_policy_that_names_no_declared_covariate_is_refused(overrides) -> None:
    with pytest.raises(ValueError):
        PlannedModelRequirement.model_validate(_requirement(**overrides))


def test_the_planner_can_neither_see_nor_write_the_policy() -> None:
    assert PlannedModelRequirement.HOST_OWNED_FIELDS == {"baseline_missing_handling"}
    transport = json.dumps(_planner_transport_schema())
    assert "baseline_missing_handling" not in transport

    step = {
        "step_id": "primary",
        "model_requirements": [
            _requirement(baseline_missing_handling={"covariates": ["lab"]})
        ],
    }
    with pytest.raises(PlannerHostOwnedFieldError) as raised:
        decode_planner_transport_payload({"research_question": "q", "steps": [step]})
    assert raised.value.fields == ("baseline_missing_handling",)
    assert raised.value.path == "steps[0].model_requirements[0]"
    # A runtime suffix step is Planner output too and decodes at the same boundary.
    suffix = {"replace_from_step_id": "primary", "replacement_step": step, "rationale": "x" * 8}
    with pytest.raises(PlannerHostOwnedFieldError):
        parse_runtime_plan_suffix(json.dumps(suffix))

    # An explicit null is the default, not a decision.
    unset = {"step_id": "primary", "model_requirements": [_requirement(baseline_missing_handling=None)]}
    decoded = decode_planner_transport_payload({"steps": [unset]})
    assert "baseline_missing_handling" not in decoded["steps"][0]["model_requirements"][0]
    # The projection itself keeps every declared field; only the Planner is refused.
    normalized, _dropped = _normalise_plan_payload({"research_question": "q", "steps": [step]})
    assert normalized["steps"][0]["model_requirements"][0]["baseline_missing_handling"] == {
        "covariates": ["lab"]
    }


def _declared_requirement(listed: list[str]) -> PlannedModelRequirement:
    return PlannedModelRequirement.model_validate(
        _requirement(
            covariates=["age", "lab", "group"],
            model_terms=[term.model_dump(mode="json") for term in TERMS],
            baseline_missing_handling={"covariates": listed},
        )
    )


# The audit reads the fit's rows ------------------------------------------


@pytest.mark.parametrize("listed", [[], ["lab", "group"]])
def test_the_audit_counts_exactly_the_rows_the_fit_used(tmp_path: Path, listed) -> None:
    contract = _fit(tmp_path, missing_category_covariates=listed)["model_contracts"][0]

    expected = PrimaryModelContractValidator._expected_denominator(
        frame=_frame(),
        outcome="outcome",
        outcome_type="binary",
        covariates=["age", "lab", "group"],
        contract=contract,
        raw_exposure_source="exposure",
    )

    assert expected == (contract["n"], contract["event_n"])


def test_a_fit_that_dropped_the_declared_rows_does_not_satisfy_the_plan(
    tmp_path: Path,
) -> None:
    step = AnalysisStep(
        step_id="primary",
        intent="Fit one adjusted association model.",
        method="adjusted_association_models",
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[_declared_requirement(["lab", "group"])],
    )
    check = PrimaryModelContractValidator._planned_model_requirement_issues
    kept_dir, dropped_dir = tmp_path / "kept", tmp_path / "dropped"
    kept_dir.mkdir()
    dropped_dir.mkdir()
    kept = _fit(kept_dir, missing_category_covariates=["lab", "group"])
    dropped = _fit(dropped_dir)

    assert check(step=step, contracts=kept["model_contracts"])[0] == []
    (issue,) = check(step=step, contracts=dropped["model_contracts"])[0]
    assert set(issue["mismatches"]) == {
        "baseline_missing_policy",
        "missing_category_covariates",
    }


def test_only_a_listed_covariate_is_audited_with_an_unmeasured_level() -> None:
    frame = pd.DataFrame(
        {
            "outcome": [1, 0] * 20,
            "exposure": np.linspace(-1.0, 1.0, 40),
            "group": ["a", "a", "b", "b"] * 10,
            "site": ["x", "y", "y", "x"] * 10,
        }
    )
    frame.loc[8:31, "group"] = None  # 24 unmeasured rows, both outcomes
    # Three survivors with an unmeasured site carry the only rare level.
    frame.loc[[1, 3, 5], "group"] = "z"
    frame.loc[[1, 3, 5], "site"] = None
    listed = {
        "baseline_missing_policy": "explicit_missing_category",
        "missing_category_covariates": ["group"],
        "exposure_source": "exposure",
        "analysis_set": "source_aware",
    }
    legacy = {key: value for key, value in listed.items() if key != "missing_category_covariates"}

    def cells(contract):
        return PrimaryModelContractValidator._categorical_zero_event_cells(
            frame=frame, outcome="outcome", covariates=["group", "site"], contract=contract
        )

    # The unlisted site's unmeasured rows are not fitted, so neither are they.
    assert cells(listed) == []
    # A contract without the list keeps its old reading: every state kept.
    assert cells(legacy) == [
        {"variable": "group", "level": "z", "n": 3, "event_n": 0},
        {"variable": "site", "level": "<missing>", "n": 3, "event_n": 0},
    ]


def _with_a_small_unmeasured_group() -> pd.DataFrame:
    frame = _frame()
    frame["group"] = frame["group"].fillna("a")
    frame.loc[frame.index[:12], "group"] = None
    return frame


def test_a_state_too_small_to_estimate_leaves_the_fit_and_says_why(tmp_path: Path) -> None:
    frame = _with_a_small_unmeasured_group()

    rows = primary_model_rows(
        frame,
        terms=TERMS,
        exposure="exposure",
        outcome="outcome",
        missing_category_covariates=["lab", "group"],
    )

    assert [item.covariate for item in rows.missing_category_terms] == ["lab"]
    (dropped,) = rows.unmeasured_rows_dropped
    assert dropped.public() == {
        "covariate": "group",
        "n_missing": 12,
        "reason_code": "unmeasured_rows_below_minimum",
    }
    assert len(rows.design) == len(frame) - 12
    contract = _fit(
        tmp_path, frame=frame, missing_category_covariates=["lab", "group"]
    )["model_contracts"][0]
    assert contract["unmeasured_rows_dropped"] == [dropped.public()]
    assert PrimaryModelContractValidator._expected_denominator(
        frame=frame,
        outcome="outcome",
        outcome_type="binary",
        covariates=["age", "lab", "group"],
        contract=contract,
        raw_exposure_source="exposure",
    ) == (contract["n"], contract["event_n"]) == (len(frame) - 12, contract["event_n"])


def test_an_unmeasured_state_with_one_outcome_is_not_estimated() -> None:
    frame = _frame()
    survivors = frame.index[frame["outcome"].eq(0.0) & frame["lab"].notna()][:30]
    frame["group"] = frame["group"].fillna("a")
    frame.loc[survivors, "group"] = None

    rows = primary_model_rows(
        frame,
        terms=TERMS,
        exposure="exposure",
        outcome="outcome",
        missing_category_covariates=["group"],
    )

    assert rows.missing_category_terms == ()
    assert [item.reason_code for item in rows.unmeasured_rows_dropped] == [
        "unmeasured_rows_share_one_outcome"
    ]


# A complete-case replay is named for what it fitted ----------------------

DEFINITION = {"exposure": "exposure", "outcome": "outcome", "covariates": ["age", "lab"]}
LOCKED = ["exposure", "outcome", "age", "lab"]


def _replay(*, error=None, coefficient_role="adjustment", **contract) -> dict:
    return {
        "row": None,
        "index": {},
        "error": error,
        "contracts": [
            {
                "model_id": "primary",
                "analysis_set": "source_aware",
                "analysis_role": "primary",
                "baseline_missing_policy": "explicit_missing_category",
                "missing_category_covariates": ["lab"],
                "missing_category_terms": [],
                **contract,
            }
        ],
        "coefficient_rows": [
            {"term": "exposure", "term_role": "exposure"},
            {"term": "lab", "term_role": coefficient_role},
        ],
    }


def test_a_proved_complete_case_replay_is_named_complete_case() -> None:
    replay = relabel_complete_case_replay(
        _replay(), locked_variables=LOCKED, definition=DEFINITION
    )

    (contract,) = replay["contracts"]
    assert contract["analysis_set"] == "complete_case"
    assert contract["baseline_missing_policy"] == "drop_missing_baseline"
    assert contract["analysis_role"] == "sensitivity"
    assert contract["source_analysis_set"] == "source_aware"
    assert contract["source_baseline_missing_policy"] == "explicit_missing_category"
    assert contract["source_missing_category_covariates"] == ["lab"]
    assert "missing_category_covariates" not in contract
    assert {row["analysis_set"] for row in replay["coefficient_rows"]} == {"complete_case"}


@pytest.mark.parametrize(
    "replay,locked",
    [
        (_replay(missing_category_terms=[{"covariate": "lab"}]), LOCKED),
        (_replay(coefficient_role="availability"), LOCKED),
        (_replay(), ["exposure", "outcome", "age"]),
        (_replay(error="replay failed"), LOCKED),
    ],
)
def test_a_replay_not_proved_complete_case_keeps_its_labels(replay, locked) -> None:
    assert (
        relabel_complete_case_replay(replay, locked_variables=locked, definition=DEFINITION)
        == replay
    )


def test_a_primary_that_kept_unmeasured_rows_is_replayed_not_reused() -> None:
    spec = RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Use complete cases for every primary model input.",
        missing_override={"strategy": "complete_case", "variables": LOCKED},
    )
    source = {
        "summary": {"analysis_definition": DEFINITION},
        "primary_contract": {
            "baseline_missing_policy": "explicit_missing_category",
            "missing_category_terms": [{"covariate": "lab", "term": "lab__unmeasured"}],
        },
    }

    _row, _coefficients, contract, error = _verified_complete_case_equivalence(
        spec=spec, source=source, primary_data=_frame()
    )

    # The caller replays the sealed primary on the locked rows on exactly this.
    assert contract is None
    assert "membership is not identical" in str(error)


def test_a_fit_that_kept_unmeasured_rows_is_never_the_complete_case_model() -> None:
    kept = {
        "exposure_source": "exposure",
        "exposure_role": "primary",
        "analysis_set": "complete_case",
        "baseline_missing_policy": "explicit_missing_category",
    }
    source = {"primary_contract": {"exposure_source": "exposure"}, "summary": {"model_contracts": [kept]}}

    assert _matching_primary_contract(source, analysis_set="complete_case") is None
    dropped = dict(kept, baseline_missing_policy="drop_missing_baseline")
    source["summary"]["model_contracts"].append(dropped)
    assert _matching_primary_contract(source, analysis_set="complete_case") == dropped


# Consumers of the primary model ------------------------------------------


def _functional_form(listed: list[str]):
    requirement = (
        _declared_requirement(listed)
        if listed
        else PlannedModelRequirement.model_validate(
            _requirement(
                covariates=["age", "lab", "group"],
                model_terms=[term.model_dump(mode="json") for term in TERMS],
            )
        )
    )
    variant = BinarySensitivityVariant(
        strategy="functional_form",
        spec_id="lab_spline",
        parent_step_id="primary",
        output_product="table:lab_spline",
        target_column="lab",
        knot_quantiles=(0.1, 0.5, 0.9),
    )
    return _refit_functional_form(
        frame=_frame(), requirement=requirement, variant=variant, linear_reference={}
    )


def test_the_functional_form_refit_uses_the_primary_rows_and_measured_knots() -> None:
    frame = _frame()

    row, receipt = _functional_form(["lab", "group"])

    assert receipt["n_complete_rows"] == row["n_stays"] == len(frame)
    assert "lab__unmeasured" in receipt["design_columns"]
    # A filled median is not an observation, so the knots ignore it.
    assert receipt["knots"] == pytest.approx(
        np.quantile(frame["lab"].dropna().to_numpy(), [0.1, 0.5, 0.9])
    )
    _row, default = _functional_form([])
    assert default["n_complete_rows"] == int(frame.notna().all(axis=1).sum())


def test_a_model_grid_spline_keeps_the_parents_unmeasured_rows(tmp_path: Path) -> None:
    frame = _frame()
    variant = AssociationModelGridVariant(
        analysis_id="lab_spline",
        nonlinear_terms=(
            AssociationModelGridNonlinearTerm(
                source_column="lab",
                basis="natural_cubic_spline",
                degrees_of_freedom=4,
                center_before_basis=True,
            ),
        ),
        metadata={},
    )

    model_frame, exposure, covariates, terms, receipts, groups = _variant_model(
        frame, requirement=_declared_requirement(["lab"]), variant=variant
    )

    # The basis bends on measured values; its columns are one kept state.
    assert sorted(groups) == receipts[0]["generated_columns"]
    assert set(groups.values()) == {"lab"}
    assert receipts[0]["center"] == pytest.approx(float(frame["lab"].dropna().mean()))
    (tmp_path / "parent").mkdir()
    (tmp_path / "variant").mkdir()
    parent = _fit(tmp_path / "parent", missing_category_covariates=["lab"])
    spline = run_adjusted_association_from_env(
        requirement_id="primary__lab_spline",
        exposure=exposure,
        outcome="outcome",
        covariates=covariates,
        model_terms=terms,
        estimator_kind="logistic",
        analysis_set="lab_spline",
        analysis_role="sensitivity",
        method_family="binary_logistic_regression",
        missing_category_covariates=["lab"],
        missing_category_term_groups=groups,
        frame=model_frame,
        cohort_path=Path("cohort.parquet"),
        output_dir=tmp_path / "variant",
    )
    assert (spline["n_total"], spline["n_events"]) == (parent["n_total"], parent["n_events"])
    (kept,) = spline["model_contracts"][0]["missing_category_terms"]
    assert kept == {
        "covariate": "lab",
        "term": "lab__unmeasured",
        "coding": "continuous_basis",
        "fill_value": 0.0,
        "n_missing": parent["missing_category_terms"][0]["n_missing"],
    }


def test_a_family_whose_estimator_fits_complete_rows_refuses_the_policy() -> None:
    declared = SimpleNamespace(step_id="primary", model_requirements=[_declared_requirement(["lab"])])
    default = SimpleNamespace(
        step_id="primary",
        model_requirements=[PlannedModelRequirement.model_validate(_requirement())],
    )

    _refuse_missing_category_policy([default])
    with pytest.raises(
        CurrentCaseScientificAuthorityError, match="missing_category_family_unsupported"
    ):
        _refuse_missing_category_policy([default, declared])


def test_the_sealed_script_carries_the_declared_policy_into_the_fit() -> None:
    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        adjusted_association_executor_code,
    )

    def script(requirement: PlannedModelRequirement) -> str:
        step = AnalysisStep(
            step_id="primary",
            planned_analysis_role="primary",
            intent="Fit one adjusted association model.",
            method="adjusted_association_models",
            inputs=["artifact:analysis_cohort", "exposure", "outcome", "age", "lab", "group"],
            expected_outputs=["table:adjusted_association_estimates"],
            model_requirements=[requirement],
        )
        return str(adjusted_association_executor_code(step))

    declared = script(_declared_requirement(["lab", "group"]))
    legacy = script(
        PlannedModelRequirement.model_validate(
            _requirement(
                covariates=["age", "lab", "group"],
                model_terms=[term.model_dump(mode="json") for term in TERMS],
            )
        )
    )

    compile(declared, "<sealed>", "exec")
    assert "declared_model[\"missing_category_covariates\"] = ['lab', 'group']" in declared
    assert "**declared_model" in declared
    assert "missing_category" not in legacy


def test_a_declared_fit_that_kept_no_unmeasured_row_is_the_complete_case_fit(
    tmp_path: Path,
) -> None:
    # Every unmeasured state was too small to estimate, so the declared fit
    # is exactly the complete-case fit and is reused under that name.
    frame = _frame()
    frame["lab"] = frame["lab"].fillna(frame["lab"].median())
    frame.loc[frame.index[:5], "lab"] = np.nan
    frame["group"] = frame["group"].fillna("a")
    summary = _fit(tmp_path, frame=frame, missing_category_covariates=["lab"])
    contract = summary["model_contracts"][0]
    assert contract["missing_category_terms"] == []
    assert [item["covariate"] for item in contract["unmeasured_rows_dropped"]] == ["lab"]
    spec = RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Use complete cases for every primary model input.",
        missing_override={
            "strategy": "complete_case",
            "variables": ["exposure", "outcome", "age", "lab", "group"],
        },
    )
    source = {
        "summary": {
            "analysis_definition": {
                "exposure": "exposure",
                "outcome": "outcome",
                "covariates": ["age", "lab", "group"],
            }
        },
        "primary_contract": contract,
        "outputs_dir": tmp_path,
        "coefficient_path": tmp_path / summary["coefficient_table"],
        "coefficient_evidence_id": "coefficients",
    }

    _row, _coefficients, replayed, error = _verified_complete_case_equivalence(
        spec=spec, source=source, primary_data=frame
    )

    assert error is None
    assert replayed["analysis_set"] == "complete_case"
    assert replayed["baseline_missing_policy"] == "drop_missing_baseline"
    assert replayed["source_baseline_missing_policy"] == "explicit_missing_category"
    assert replayed["source_missing_category_covariates"] == ["lab"]
    assert "unmeasured_rows_dropped" not in replayed
