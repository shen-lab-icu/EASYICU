"""A continuous survival estimate is reported per readable step and follows its spline check.

The continuous-exposure survival suite reported its hazard ratio per one unit
of the exposure's recorded scale, and its spline check of the linear term
changed nothing.  For an exposure recorded in large numbers (an enzyme in U/L)
one unit is a negligible change: the estimate and both interval bounds print
as the same number.  And an association the check found not to be linear was
still summarized by one per-unit hazard ratio.

Now the step is read from the modelled exposure alone, before any model is
fitted: the largest one, two or five times a power of ten within its
interquartile range, or, for a heaped exposure, within its 10th-90th
percentile range, then its range.  When the spline check rejects a linear
term while the PH test holds, the spline's hazard ratios at the 10th and 90th
percentiles against the median are the result and the per-step one is
withheld; when the PH test rejected, the interval estimates stay the result
and are read as an average log-linear trend.  The spline terms are also tested
together against no exposure term, and a model is called adjusted only when it
adjusted for a covariate.

Synthetic, seeded rows only: an enzyme-like value in U/L with a linear
association, a sodium-like value whose risk rises at both ends, a bounded
score heaped at its ceiling, 28-day mortality, every stay alive at the
24-hour landmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from easyicu.research_agent.authority import prespecified_rule_outcomes
from easyicu.research_agent.authority.continuous_survival_scientific_claims import (
    ContinuousSurvivalReporting,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.contracts import executed_method_design, executor_stop
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    validate_executed_method_design,
)
from easyicu.research_agent.execution.runners import (
    landmark_continuous_survival_executor,
)
from easyicu.research_agent.execution.runners.landmark_continuous_survival_executor import (
    run_landmark_continuous_survival_suite,
)
from easyicu.research_agent.execution.runners.landmark_continuous_survival_figure import (
    run_landmark_continuous_survival_figure,
)
from easyicu.research_agent.figures.publication import (
    FigureContract,
    audit_publication_exports,
)
from easyicu.research_agent.methods.rcs_dose_response import rcs_basis
from easyicu.research_agent.methods.time_varying_cox import (
    fit_piecewise_time_varying_cox,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)
from tests.support.continuous_survival import continuous_authority_body

pytest.importorskip("lifelines")

STEP = "01_primary"
EVIDENCE = "statistic_step_summary_continuous_survival_suite"
SUITE = "signed_landmark_continuous_survival_suite"
ONE_VALUE = "continuous_survival_exposure_has_one_value"
UNADJUSTED = {
    "adjustment_columns": [],
    "categorical_adjustment_columns": [],
    "table_one_columns": [],
}
#: Words that would name a shape the rule never decided.
SHAPE_WORDS = (
    "u-shaped",
    "j-shaped",
    "threshold",
    "plateau",
    "curvilinear",
    "non-linear relationship",
    "nonlinear relationship",
)
CONTRAST_CLAIMS = [
    ("spline_hazard_ratio_at_percentile_10", "primary"),
    ("spline_hazard_ratio_at_percentile_90", "primary"),
    ("interval_1_adjusted_hazard_ratio_per_unit", "secondary"),
    ("interval_2_adjusted_hazard_ratio_per_unit", "secondary"),
    ("interval_3_adjusted_hazard_ratio_per_unit", "secondary"),
    ("proportional_hazards_rule", "primary"),
    ("functional_form_rule", "primary"),
]


def _authority(**overrides):
    return build_current_case_scientific_runtime_authority(
        continuous_authority_body(**overrides)
    )


def _rows(
    *,
    seed: int,
    exposure: Callable[[Any, int], Any],
    log_hazard: Callable[[Any], Any],
    crossing: Callable[[Any], Any] | None = None,
    n: int = 1500,
) -> pd.DataFrame:
    """Stays alive at the 24-hour landmark; deaths by day 28 from ICU admission.

    ``log_hazard`` is the exposure's log hazard, constant over follow-up;
    ``crossing`` adds one that reverses five days after the landmark.
    """

    rng = np.random.default_rng(seed)
    age = rng.normal(65.0, 12.0, n)
    sex = rng.choice(["F", "M"], n)
    value = exposure(rng, n)
    base = 0.03 * np.exp(log_hazard(value) + 0.02 * (age - 65.0))
    if crossing is None:
        after = rng.exponential(1.0 / base)
    else:
        early = rng.exponential(1.0 / (base * np.exp(crossing(value))))
        late = 5.0 + rng.exponential(1.0 / (base * np.exp(-crossing(value))))
        after = np.where(early < 5.0, early, late)
    death = after <= 27.0
    return pd.DataFrame(
        {
            "lab_max": value,
            "mort_28d": death.astype(int),
            "followup_days_28d": np.where(death, 1.0 + after, 28.0),
            "age": age,
            "sex": sex,
        }
    )


def _enzyme(rng, n):
    """An enzyme-like value in U/L: a median of 1,000, skewed to the right."""

    return np.exp(rng.normal(np.log(1000.0), 0.5, n))


def _enzyme_effect(value):
    return 0.0008 * (value - 1000.0)


def _sodium(rng, n):
    return rng.normal(139.0, 4.5, n)


def _both_ends(value):
    """A risk that rises at both ends of the range, constant over follow-up."""

    return 0.5 * (((value - 139.0) / 4.5) ** 2 - 1.0)


def _early_only(value):
    return 0.9 * (value - 139.0) / 4.5


def _none(value):
    return 0.0 * value


def _ceiling(rng, n):
    """A bounded score from 3 to 15 with 85% of the stays at 15."""

    return np.where(rng.random(n) < 0.85, 15.0, rng.integers(3, 15, n).astype(float))


def _score_effect(value):
    return 0.1 * (value - 13.0)


def _censored_from(frame: pd.DataFrame, day: float) -> pd.DataFrame:
    """The same stays, every death from ``day`` after the landmark censored at day 28."""

    late = frame["mort_28d"].eq(1) & (frame["followup_days_28d"] - 1.0).ge(day)
    return frame.assign(
        mort_28d=frame["mort_28d"].where(~late, 0),
        followup_days_28d=frame["followup_days_28d"].where(~late, 28.0),
    )


def _run(directory: Path, frame: pd.DataFrame, authority) -> dict:
    summary = run_landmark_continuous_survival_suite(
        frame=frame,
        authority=authority.model_dump(mode="json"),
        runtime_projection_sha256="b" * 64,
        out_dir=directory,
        input_product="table:analysis_cohort",
        input_evidence_id="cohort_evidence",
        input_sha256="c" * 64,
    )
    return json.loads(json.dumps(summary, allow_nan=False))


@dataclass(frozen=True)
class _Suite:
    authority: Any
    frame: pd.DataFrame
    summary: dict
    out_dir: Path

    @property
    def envelope(self) -> dict:
        return self.summary["reportable_survival_results"]

    @property
    def receipt(self) -> dict:
        return self.summary["scientific_runtime_receipt"]

    def table(self, product: str) -> pd.DataFrame:
        return pd.read_csv(self.out_dir / self.summary["output_files"][product])


def _suite(factory, name: str, frame: pd.DataFrame, **overrides) -> _Suite:
    authority = _authority(**overrides)
    directory = factory.mktemp(name)
    return _Suite(authority, frame, _run(directory, frame, authority), directory)


@pytest.fixture(scope="module")
def enzyme(tmp_path_factory) -> _Suite:
    """A linear association on a large scale: one step is 500 U/L."""

    frame = _rows(seed=1, exposure=_enzyme, log_hazard=_enzyme_effect)
    return _suite(tmp_path_factory, "enzyme", frame, exposure_unit="U/L")


@pytest.fixture(scope="module")
def both_ends(tmp_path_factory) -> _Suite:
    """A risk rising at both ends while the PH test holds: the spline's result."""

    frame = _rows(seed=1, exposure=_sodium, log_hazard=_both_ends)
    return _suite(tmp_path_factory, "both_ends", frame)


@pytest.fixture(scope="module")
def both_ends_unadjusted(tmp_path_factory) -> _Suite:
    frame = _rows(seed=1, exposure=_sodium, log_hazard=_both_ends)
    return _suite(tmp_path_factory, "both_ends_unadjusted", frame, **UNADJUSTED)


@pytest.fixture(scope="module")
def crossing(tmp_path_factory) -> _Suite:
    """Both ends raise the risk and the slope reverses: the PH test rejects."""

    frame = _rows(seed=1, exposure=_sodium, log_hazard=_both_ends, crossing=_early_only)
    return _suite(tmp_path_factory, "crossing", frame)


@pytest.fixture(scope="module")
def no_association(tmp_path_factory) -> _Suite:
    frame = _rows(seed=1, exposure=_sodium, log_hazard=_none)
    return _suite(tmp_path_factory, "no_association", frame)


@pytest.fixture(scope="module")
def ceiling(tmp_path_factory) -> _Suite:
    """A score heaped at its ceiling: no interquartile range and tied knots."""

    frame = _rows(seed=1, exposure=_ceiling, log_hazard=_score_effect)
    return _suite(tmp_path_factory, "ceiling", frame)


def _model_columns(frame: pd.DataFrame, *, adjusted: bool = True) -> pd.DataFrame:
    """The suite's complete-case model frame, rebuilt from the synthetic rows."""

    data = pd.DataFrame(
        {
            "time": frame["followup_days_28d"] - 1.0,
            "event": frame["mort_28d"].astype(float),
            "lab_max": frame["lab_max"].astype(float),
        }
    )
    if adjusted:
        data["age"] = frame["age"].astype(float)
        data["sex_M"] = (frame["sex"] == "M").astype(float)
    return data


def _with_spline(data: pd.DataFrame, knots) -> pd.DataFrame:
    basis = rcs_basis(data["lab_max"].to_numpy(dtype=float), knots=knots)
    return data.assign(spline_term=[row[1] for row in basis.matrix])


def _fit(data: pd.DataFrame):
    from lifelines import CoxPHFitter

    return CoxPHFitter().fit(data, duration_col="time", event_col="event")


def _numeric_leaves(value) -> int:
    if isinstance(value, dict):
        return sum(_numeric_leaves(child) for child in value.values())
    if isinstance(value, list):
        return sum(_numeric_leaves(child) for child in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def _strict_store(directory: Path, summary: dict, *, max_leaves=None):
    """The summary registered as the host runner registers a signed owner's."""

    source = directory / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(directory, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic",
        description="Signed continuous survival suite summary",
        source_path=source,
        evidence_id=EVIDENCE,
        produced_by_step=STEP,
        producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id=EVIDENCE, summary=summary, max_leaves=max_leaves
    )
    return store, [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]


def _methods_fact(directory: Path, summary: dict) -> str:
    """The executed design as its Methods fact, bound without an untraced number."""

    store, ledger = _strict_store(directory, summary)
    (fact,) = [
        fact
        for fact in store.manuscript_method_facts(ledger)
        if fact.source_field.endswith(EXECUTED_METHOD_DESIGN_KEY)
    ]
    scaffold = f"## Methods\n\n### Variables\n\n{fact.scaffold}\n"
    safe, removed = store.enforce_evidence_bound_scaffold(
        scaffold, per_step_records=ledger
    )
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced
    return fact.text


def _digest_row(directory: Path, summary: dict) -> dict:
    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": STEP,
                "status": "ok",
                "generation_mode": "deterministic_standard",
                "step_summary": summary,
            }
        ],
        run_dir=directory,
        evidence=None,
    )
    lines = digest.splitlines()
    head = next(
        index for index, line in enumerate(lines) if line.startswith(f"- {STEP} [")
    )
    return json.loads(lines[head + 1])


def _figure(directory: Path, suite: _Suite, *, spline_table=None):
    authority = suite.authority
    paths = {
        product: suite.out_dir / name
        for product, name in suite.summary["output_files"].items()
    }
    result = run_landmark_continuous_survival_figure(
        km_table=pd.read_csv(paths[authority.km_product]),
        spline_table=(
            pd.read_csv(paths[authority.spline_product])
            if spline_table is None
            else spline_table
        ),
        time_varying_table=pd.read_csv(paths[authority.time_varying_cox_product]),
        risk_flow=pd.read_csv(paths[authority.risk_set_product]),
        ph_table=pd.read_csv(paths[authority.ph_product]),
        source_paths={
            product: paths[product] for product in authority.figure_input_products
        },
        authority=authority.model_dump(mode="json"),
        out_dir=directory,
    )
    assert audit_publication_exports(directory) == []
    assets = result["figure_assets"]
    contract = FigureContract.model_validate_json(
        (directory / assets["contract"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (directory / assets["runtime_receipt"]).read_text(encoding="utf-8")
    )
    svg = (directory / result["output_files"][authority.figure_product]).read_text(
        encoding="utf-8"
    )
    return contract, receipt, svg


def test_a_large_scale_exposure_is_reported_per_a_readable_step(enzyme) -> None:
    receipt = enzyme.receipt
    assert (
        receipt["exposure_increment_text"],
        receipt["exposure_increment_mantissa"],
        receipt["exposure_increment_exponent"],
        receipt["exposure_increment_spread"],
    ) == ("500", 5, 2, "interquartile_range")
    low, high = enzyme.frame["lab_max"].quantile([0.25, 0.75])
    assert receipt["exposure_increment_spread_width"] == pytest.approx(high - low)
    assert 500.0 <= high - low < 1000.0
    design = validate_executed_method_design(enzyme.summary[EXECUTED_METHOD_DESIGN_KEY])
    assert (design.exposure_increment, design.exposure_increment_spread) == (
        500.0,
        "interquartile_range",
    )
    envelope = enzyme.envelope
    assert envelope["exposure_increment"] == 500.0
    assert envelope["functional_form"]["disposition"] == "linearity_not_rejected"
    assert envelope["primary_estimate"] == "per_step_hazard_ratio"
    ratio = envelope["adjusted_hazard_ratio_per_unit"]
    # The suite's estimate is the per-unit model's, read over 500 U/L.
    per_unit = float(_fit(_model_columns(enzyme.frame)).params_["lab_max"])
    assert ratio["hazard_ratio"] == pytest.approx(math.exp(500.0 * per_unit), rel=1e-5)
    bounds = (ratio["hazard_ratio"], ratio["ci_low"], ratio["ci_high"])
    # Per unit, the estimate and both bounds print as one number.
    assert len({f"{value ** (1 / 500):.3f}" for value in bounds}) == 1
    assert len({f"{value:.3f}" for value in bounds}) == 3
    (claim, first_interval, *_others) = derive_scientific_claim_drafts(enzyme.summary)
    assert claim.claim_id == "adjusted_hazard_ratio_per_unit"
    assert claim.estimand == (
        "per-500 U/L adjusted hazard ratio over the post-landmark follow-up"
    )
    assert claim.point_estimate == ratio["hazard_ratio"]
    # A linear term the check kept needs no qualification within an interval.
    assert first_interval.estimand == (
        "per-500 U/L adjusted hazard ratio for days 0 to 7 after the landmark"
    )
    # The curves stay on the exposure's own scale: the linear one per U/L.
    curve = enzyme.table(enzyme.authority.spline_product)
    first = curve.iloc[0]
    assert first["linear_hazard_ratio"] == pytest.approx(
        math.exp(per_unit * (first["exposure_value"] - first["reference_value"])),
        rel=1e-5,
    )
    cox = enzyme.table(enzyme.authority.cox_product)
    exposure_row = cox["term"].eq("lab_max")
    assert cox.loc[exposure_row, "exposure_increment"].tolist() == [500.0]
    assert cox.loc[~exposure_row, "exposure_increment"].isna().all()
    assert cox.loc[exposure_row, "hazard_ratio"].iloc[0] == pytest.approx(
        ratio["hazard_ratio"]
    )
    rule = ContinuousSurvivalReporting.model_validate(
        envelope
    ).functional_form_outcome()
    assert rule.result_sentence() == (
        "The restricted cubic spline check did not reject a linear association with "
        "the log hazard at the prespecified alpha of 0.050, so the per-step "
        "estimates stand."
    )


def test_the_step_reads_the_exposure_alone_and_rescales_nothing_else(
    tmp_path, enzyme
) -> None:
    shuffled = enzyme.frame.copy()
    order = np.random.default_rng(5).permutation(len(shuffled))
    outcome = ["mort_28d", "followup_days_28d"]
    shuffled[outcome] = shuffled[outcome].to_numpy()[order]
    blind = _run(tmp_path / "blind", shuffled, enzyme.authority)
    scaled = _run(
        tmp_path / "scaled",
        enzyme.frame.assign(lab_max=enzyme.frame["lab_max"] * 10.0),
        enzyme.authority,
    )

    def step(summary):
        receipt = summary["scientific_runtime_receipt"]
        return (
            receipt["exposure_increment_mantissa"],
            receipt["exposure_increment_exponent"],
            receipt["exposure_increment_spread"],
        )

    # The outcome never enters the step.
    assert step(blind) == step(enzyme.summary) == (5, 2, "interquartile_range")
    # Ten times the scale is ten times the step, and the same result.
    assert step(scaled) == (5, 3, "interquartile_range")
    original, rescaled = enzyme.envelope, scaled["reportable_survival_results"]
    for name in ("hazard_ratio", "ci_low", "ci_high"):
        assert rescaled["adjusted_hazard_ratio_per_unit"][name] == pytest.approx(
            original["adjusted_hazard_ratio_per_unit"][name], rel=1e-9
        )
    for name in ("global_p_value", "exposure_p_value"):
        assert rescaled["proportional_hazards_test"][name] == pytest.approx(
            original["proportional_hazards_test"][name], rel=1e-6, abs=1e-12
        )
    for name in ("likelihood_ratio_statistic", "overall_likelihood_ratio_statistic"):
        assert rescaled["functional_form"][name] == pytest.approx(
            original["functional_form"][name], rel=1e-5, abs=1e-6
        )


def _next_round_step(mantissa: int, exponent: int) -> float:
    following = {1: (2, exponent), 2: (5, exponent), 5: (1, exponent + 1)}
    larger, power = following[mantissa]
    return float(Decimal(larger).scaleb(power))


@pytest.mark.parametrize(
    ("width", "mantissa", "exponent", "text"),
    [
        (1450.0, 1, 3, "1000"),
        (683.0, 5, 2, "500"),
        (2.0, 2, 0, "2"),
        (10.0, 1, 1, "10"),
        (0.3, 2, -1, "0.2"),
        (0.07, 5, -2, "0.05"),
        # A width just below a power of ten, whose logarithm rounds up to it.
        (math.nextafter(1000.0, 0.0), 5, 2, "500"),
        (math.nextafter(0.1, 0.0), 5, -2, "0.05"),
    ],
)
def test_the_step_is_the_largest_round_step_within_the_spread(
    width, mantissa, exponent, text
) -> None:
    step = landmark_continuous_survival_executor._exposure_step(
        pd.Series([0.0, 0.0, width, width])
    )

    assert (step.mantissa, step.exponent, step.spread) == (
        mantissa,
        exponent,
        "interquartile_range",
    )
    assert step.spread_width == width
    assert (step.text, step.value) == (text, float(text))
    assert step.value <= width < _next_round_step(mantissa, exponent)
    assert executed_method_design.is_round_exposure_step(step.value)
    assert executed_method_design.exposure_step_text(step.value) == text


def test_a_heaped_exposure_reads_its_step_from_a_wider_spread() -> None:
    step = landmark_continuous_survival_executor._exposure_step

    central = step(pd.Series([3.0] * 15 + [15.0] * 85))
    whole = step(pd.Series([3.0] * 5 + [15.0] * 95))

    assert (central.spread, central.spread_width, central.text) == (
        "central_eighty_percent_range",
        12.0,
        "10",
    )
    assert (whole.spread, whole.spread_width, whole.text) == ("range", 12.0, "10")
    assert step(pd.Series([7.0] * 20)) is None


def test_an_exposure_with_one_value_stops_before_any_model(tmp_path) -> None:
    frame = _rows(seed=1, exposure=lambda rng, n: np.full(n, 7.0), log_hazard=_none)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()  # the runner creates the step's output directory first

    with pytest.raises(executor_stop.ExecutorStop) as caught:
        _run(out_dir, frame, _authority())

    stop = caught.value
    assert (stop.owner, stop.reason_code, stop.cause_code) == (SUITE, ONE_VALUE, None)
    assert str(stop).startswith(f"{ONE_VALUE}: ")
    # No model ran and no table was written: the stop's record is the only file.
    record = out_dir / executor_stop.EXECUTOR_STOP_RECORD_NAME
    assert [path.name for path in out_dir.iterdir()] == [record.name]
    recorded = executor_stop.parse_executor_stop_record(
        record.read_bytes(), expected_owner=SUITE
    )
    assert (recorded.reason_code, recorded.cause_code) == (ONE_VALUE, None)
    # The host reads it as this suite's stop, which names no cause.
    step_record = {
        "deterministic_standard_analysis": SUITE,
        "executor_stop_reason_code": ONE_VALUE,
        "executor_stop_cause_code": None,
    }
    assert executor_stop.executor_stop_codes(step_record) == {"reason_code": ONE_VALUE}
    assert executor_stop.registered_executor_stop(ONE_VALUE, "did_not_converge") is None
    assert (
        executor_stop.registered_executor_stop(ONE_VALUE, None, owner="another_suite")
        is None
    )


def test_the_authority_seals_the_step_rule_and_the_spline_rule() -> None:
    authority = _authority()
    assert authority.exposure_increment_rule == (
        "largest_round_step_within_interquartile_range"
    )
    assert authority.effect_measure == "hazard_ratio_per_exposure_step"
    assert (authority.functional_form_alpha, authority.functional_form_policy) == (
        0.05,
        "spline_contrasts_replace_linear_estimate",
    )
    body = continuous_authority_body()

    def without(*names):
        return {key: value for key, value in body.items() if key not in names}

    for broken in (
        {
            **body,
            "schema_version": "easyicu.landmark_continuous_survival_runtime_authority/1",
        },
        {**without("exposure_increment_rule"), "exposure_increment": 1.0},
        {**body, "exposure_increment_rule": "one_unit"},
        {**body, "effect_measure": "hazard_ratio_per_unit"},
        without("functional_form_alpha"),
        {**body, "functional_form_alpha": 0.0},
        {**body, "functional_form_policy": "report_only"},
    ):
        with pytest.raises(ValueError):
            build_current_case_scientific_runtime_authority(broken)


def test_a_step_that_is_not_round_is_refused(enzyme) -> None:
    design = enzyme.summary[EXECUTED_METHOD_DESIGN_KEY]

    for step in (3.0, 0.3, 250.0, 1500.0):
        with pytest.raises(ValueError):
            validate_executed_method_design({**design, "exposure_increment": step})
        with pytest.raises(ValueError, match="one, two or five times a power of ten"):
            ContinuousSurvivalReporting.model_validate(
                {**enzyme.envelope, "exposure_increment": step}
            )
    for step in (0.00002, 0.05, 1000000.0):
        validate_executed_method_design({**design, "exposure_increment": step})
        ContinuousSurvivalReporting.model_validate(
            {**enzyme.envelope, "exposure_increment": step}
        )


def test_a_rejected_linear_term_leaves_the_spline_contrasts_as_the_result(
    tmp_path, both_ends
) -> None:
    envelope = both_ends.envelope
    form = envelope["functional_form"]
    assert envelope["proportional_hazards_test"]["disposition"] == (
        "assumption_not_rejected"
    )
    assert (form["disposition"], envelope["primary_estimate"]) == (
        "linearity_rejected",
        "spline_percentile_contrasts",
    )
    assert form["p_value"] < form["alpha"] == 0.05
    # The per-step estimate is withheld: no leaf a manuscript could bind.
    assert "adjusted_hazard_ratio_per_unit" not in envelope
    assert envelope["exposure_increment"] == 5.0
    curve = both_ends.table(both_ends.authority.spline_product)
    assert set(curve["functional_form_disposition"]) == {"linearity_rejected"}
    coordinates = both_ends.receipt["spline_contrast_coordinates"]
    assert both_ends.receipt["primary_estimate"] == "spline_percentile_contrasts"
    lower, upper = form["contrasts"]
    assert (lower["percentile"], upper["percentile"]) == (10.0, 90.0)
    for contrast, row, value in (
        (lower, curve.iloc[0], coordinates["lower_percentile_value"]),
        (upper, curve.iloc[-1], coordinates["upper_percentile_value"]),
    ):
        assert contrast["exposure_value"] == value
        assert contrast["exposure_value"] == pytest.approx(row["exposure_value"])
        assert contrast["reference_value"] == form["reference_value"]
        assert contrast["reference_value"] == coordinates["median"]
        assert (contrast["hazard_ratio"], contrast["ci_low"], contrast["ci_high"]) == (
            pytest.approx(
                (
                    row["spline_hazard_ratio"],
                    row["spline_ci_low"],
                    row["spline_ci_high"],
                )
            )
        )
        # Risk rises at both ends: both contrasts exceed one.
        assert contrast["ci_low"] > 1.0
    assert (
        lower["exposure_value"],
        coordinates["median"],
        upper["exposure_value"],
    ) == pytest.approx(tuple(both_ends.frame["lab_max"].quantile([0.1, 0.5, 0.9])))

    drafts = derive_scientific_claim_drafts(both_ends.summary)
    assert [(draft.claim_id, draft.analysis_role) for draft in drafts] == (
        CONTRAST_CLAIMS
    )
    low_text, median_text, high_text = (
        f"{value:.4g}"
        for value in (
            lower["exposure_value"],
            coordinates["median"],
            upper["exposure_value"],
        )
    )
    assert drafts[0].estimand == (
        f"10th-percentile-versus-median ({low_text} versus {median_text} mmol/L) "
        "adjusted hazard ratio from the restricted cubic spline model, over the "
        "post-landmark follow-up"
    )
    assert drafts[1].estimand.startswith(
        f"90th-percentile-versus-median ({high_text} versus {median_text} mmol/L) "
        "adjusted hazard ratio"
    )
    assert (drafts[0].point_estimate, drafts[0].direction) == (
        lower["hazard_ratio"],
        "positive",
    )
    assert drafts[2].estimand == (
        "per-5 mmol/L adjusted hazard ratio for days 0 to 7 after the landmark, an "
        "average log-linear trend within the interval"
    )
    rule = drafts[-1].rule_outcome
    assert rule.result_sentence() == (
        "The restricted cubic spline check rejected a linear association with the log "
        "hazard at the prespecified alpha of 0.050, so the spline's hazard ratios at "
        "two prespecified percentiles of the exposure relative to its median are the "
        "primary estimates instead of one per-step hazard ratio."
    )
    abstract = [
        claim["scientific_claim_id"]
        for claim in envelope["manuscript_projection"]["claims"]
        if "scientific_claim_id" in claim
    ]
    assert abstract == [claim_id for claim_id, _role in CONTRAST_CLAIMS[:2]]
    (check,) = [
        claim
        for claim in envelope["manuscript_projection"]["claims"]
        if claim["claim_id"] == "restricted_cubic_spline_check"
    ]
    assert [
        fragment["numeric_path"]
        for fragment in check["fragments"]
        if "numeric_path" in fragment
    ] == [
        "functional_form.likelihood_ratio_statistic",
        "functional_form.degrees_of_freedom",
        "functional_form.overall_likelihood_ratio_statistic",
        "functional_form.overall_degrees_of_freedom",
    ]
    # The Writer reads the contrasts as the result, and no per-step estimate.
    reported = _digest_row(tmp_path, both_ends.summary)["reportable_survival_results"]
    assert reported["functional_form"]["contrasts"] == form["contrasts"]
    assert "adjusted_hazard_ratio_per_unit" not in reported


@pytest.mark.parametrize(
    ("name", "printed"),
    [
        ("enzyme", "(per-500 U/L adjusted hazard ratio over the post-landmark"),
        (
            "both_ends",
            "(10th-percentile-versus-median ({low:.4g} versus {median:.4g} mmol/L)",
        ),
    ],
)
def test_every_number_a_continuous_result_prints_binds_to_its_run(
    tmp_path, request, name, printed
) -> None:
    suite = request.getfixturevalue(name)
    # The whole summary fits the per-step numeric cap the pipeline applies.
    assert _numeric_leaves(suite.summary) <= 100
    store, ledger = _strict_store(tmp_path / "run", suite.summary, max_leaves=100)
    claims = store.authoritative_scientific_claims(ledger)
    results = "\n\n".join(claim.placeholder for claim in claims)

    bound = store.bind_manuscript(
        f"## Results\n\n### Survival results\n\n{results}\n", per_step_records=ledger
    )
    _, binding, untraced = bind_numeric_values(
        bound, evidence=store, per_step_records=ledger
    )

    # The step and the exposure values print before the ratio's name, so each
    # binds to the field it comes from.
    assert untraced == []
    assert {claim.step_id for claim in binding.values()} == {STEP}
    contrasts = suite.envelope["functional_form"].get("contrasts") or [{}]
    assert (
        printed.format(
            low=contrasts[0].get("exposure_value", 0.0),
            median=contrasts[0].get("reference_value", 0.0),
        )
        in bound
    )
    text = " ".join(
        sentence
        for claim in claims
        for sentence in (
            claim.render_reader_text(),
            claim.render_reader_text(include_estimate=False),
        )
    ).casefold()
    assert not any(word in text for word in SHAPE_WORDS)


def test_the_writer_reads_the_spline_result_when_the_intervals_have_none(
    tmp_path,
) -> None:
    # No death after day 13: the last interval has no event, and the interval
    # model, here a secondary model, has no estimate.
    frame = _censored_from(_rows(seed=2, exposure=_sodium, log_hazard=_both_ends), 13.0)
    summary = _run(tmp_path / "suite", frame, _authority())

    envelope = summary["reportable_survival_results"]
    assert envelope["proportional_hazards_test"]["disposition"] == (
        "assumption_not_rejected"
    )
    assert envelope["primary_estimate"] == "spline_percentile_contrasts"
    assert envelope["time_varying_adjusted_association"]["status"] == "not_estimable"
    assert [draft.claim_id for draft in derive_scientific_claim_drafts(summary)] == [
        "spline_hazard_ratio_at_percentile_10",
        "spline_hazard_ratio_at_percentile_90",
        "proportional_hazards_rule",
        "functional_form_rule",
    ]
    # The contrasts alone authorize the envelope for the Writer.
    reported = _digest_row(tmp_path, summary)["reportable_survival_results"]
    assert (
        reported["functional_form"]["contrasts"]
        == (envelope["functional_form"]["contrasts"])
    )
    assert "intervals" not in reported["time_varying_adjusted_association"]
    assert "adjusted_hazard_ratio_per_unit" not in reported


def test_a_rejected_ph_test_keeps_the_intervals_as_an_average_trend(
    tmp_path, crossing
) -> None:
    envelope = crossing.envelope
    assert envelope["proportional_hazards_test"]["disposition"] == "assumption_rejected"
    assert envelope["functional_form"]["disposition"] == "linearity_rejected"
    assert envelope["primary_estimate"] == "interval_per_step_hazard_ratios"
    assert "contrasts" not in envelope["functional_form"]
    assert "adjusted_hazard_ratio_per_unit" not in envelope

    # Each interval estimate is the per-unit interval model's, read over 5 mmol/L.
    native = fit_piecewise_time_varying_cox(
        _model_columns(crossing.frame),
        duration_col="time",
        event_col="event",
        covariates=["lab_max", "age", "sex_M"],
        interval_cutpoints=[7.0, 14.0],
        exposure_col="lab_max",
    )
    per_unit = native.loc[native["is_exposure"].astype(bool), "hazard_ratio"]
    reported = envelope["time_varying_adjusted_association"]["intervals"]
    assert [item["hazard_ratio"] for item in reported] == pytest.approx(
        [float(value) ** 5 for value in per_unit], rel=1e-5
    )
    drafts = derive_scientific_claim_drafts(crossing.summary)
    intervals = [draft for draft in drafts if draft.claim_id.startswith("interval_")]
    assert [draft.analysis_role for draft in intervals] == ["primary"] * 3
    assert all(
        draft.estimand.endswith(", an average log-linear trend within the interval")
        for draft in intervals
    )
    assert drafts[-1].rule_outcome.result_sentence() == (
        "The restricted cubic spline check rejected a linear association with the log "
        "hazard at the prespecified alpha of 0.050; the interval-specific per-step "
        "hazard ratios remain the primary estimates and describe an average "
        "log-linear trend within each interval."
    )
    contract, receipt, svg = _figure(tmp_path / "figure", crossing)
    assert receipt["promoted_effect_measure"] == (
        "interval_specific_hazard_ratio_per_exposure_step"
    )
    assert (
        "The spline check rejected a linear term, so each describes an average "
        "log-linear trend within its interval."
    ) in contract.reader_caption
    assert "Adjusted HR per 5 mmol/L (95% CI)" in svg


def test_the_figure_states_the_result_the_rules_chose(tmp_path, both_ends) -> None:
    contract, receipt, svg = _figure(tmp_path / "figure", both_ends)

    assert receipt["promoted_effect_measure"] == "spline_percentile_contrasts"
    assert contract.core_claim.endswith(
        "replaced by the spline's percentile contrasts because the spline check "
        "rejected a linear term."
    )
    assert (
        "The spline check rejected a linear term, so the spline's hazard ratios at "
        "those two percentiles are the primary estimates."
    ) in contract.reader_caption
    (panel_b,) = [panel for panel in contract.panels if panel.panel_id == "b"]
    assert (panel_b.title, panel_b.metadata["chart_type"]) == (
        "Adjusted hazard ratio curve",
        "hazard_ratio_curve",
    )
    assert "Adjusted HR vs median (95% CI)" in svg
    # A spline table that states no disposition, or one its status forbids.
    curve = both_ends.table(both_ends.authority.spline_product)
    for index, broken in enumerate(
        (
            curve.drop(columns=["functional_form_disposition"]),
            curve.assign(functional_form_disposition="not_assessable"),
            curve.assign(spline_status="tied_knots"),
        )
    ):
        with pytest.raises(ValueError, match="functional-form disposition"):
            _figure(tmp_path / f"broken_{index}", both_ends, spline_table=broken)


def test_the_spline_terms_are_tested_against_no_exposure_term(
    no_association, both_ends, both_ends_unadjusted
) -> None:
    for suite in (no_association, both_ends, both_ends_unadjusted):
        form = suite.envelope["functional_form"]
        assert form["overall_degrees_of_freedom"] == 2
        assert form["overall_p_value"] == pytest.approx(
            chi2.sf(form["overall_likelihood_ratio_statistic"], 2)
        )
    assert no_association.envelope["functional_form"]["overall_p_value"] > 0.05
    assert both_ends.envelope["functional_form"]["overall_p_value"] < 1e-6
    # The reference keeps the covariates without the exposure, or has no term.
    form = both_ends.envelope["functional_form"]
    data = _model_columns(both_ends.frame)
    expected = 2.0 * (
        _fit(_with_spline(data, form["knots"])).log_likelihood_
        - _fit(data.drop(columns=["lab_max"])).log_likelihood_
    )
    assert form["overall_likelihood_ratio_statistic"] == pytest.approx(
        expected, rel=1e-6
    )
    form = both_ends_unadjusted.envelope["functional_form"]
    alone = _with_spline(
        _model_columns(both_ends_unadjusted.frame, adjusted=False), form["knots"]
    )
    assert form["overall_likelihood_ratio_statistic"] == pytest.approx(
        _fit(alone).log_likelihood_ratio_test().test_statistic, rel=1e-6
    )


def test_the_overall_test_is_left_out_when_its_reference_has_no_result(
    tmp_path, monkeypatch, both_ends
) -> None:
    fit = landmark_continuous_survival_executor._cox_fit

    def reference_fails(frame, **arguments):
        fitter, nonconvergence = fit(frame, **arguments)
        if "lab_max" not in frame.columns:
            return fitter, "the reference model did not converge"
        return fitter, nonconvergence

    monkeypatch.setattr(
        landmark_continuous_survival_executor, "_cox_fit", reference_fails
    )

    summary = _run(tmp_path / "suite", both_ends.frame, both_ends.authority)

    envelope = summary["reportable_survival_results"]
    form = envelope["functional_form"]
    assert form["disposition"] == "linearity_rejected"
    assert not {
        "overall_likelihood_ratio_statistic",
        "overall_degrees_of_freedom",
        "overall_p_value",
    } & set(form)
    ContinuousSurvivalReporting.model_validate(envelope)
    (check,) = [
        claim
        for claim in envelope["manuscript_projection"]["claims"]
        if claim["claim_id"] == "restricted_cubic_spline_check"
    ]
    assert [
        fragment["numeric_path"]
        for fragment in check["fragments"]
        if "numeric_path" in fragment
    ] == [
        "functional_form.likelihood_ratio_statistic",
        "functional_form.degrees_of_freedom",
    ]
    assert check["fragments"][-1] == {"text": " degree of freedom."}


def test_the_envelope_refuses_a_result_its_rules_do_not_choose(
    both_ends, enzyme
) -> None:
    contrasts, per_step = both_ends.envelope, enzyme.envelope
    form, linear = contrasts["functional_form"], per_step["functional_form"]
    ContinuousSurvivalReporting.model_validate(contrasts)
    ContinuousSurvivalReporting.model_validate(per_step)

    def contrasts_with(**fields):
        return {**contrasts, "functional_form": {**form, **fields}}

    for broken, message in (
        ({**contrasts, "primary_estimate": "per_step_hazard_ratio"}, "follows from"),
        (
            {
                **contrasts,
                "adjusted_hazard_ratio_per_unit": per_step[
                    "adjusted_hazard_ratio_per_unit"
                ],
            },
            "per-step hazard ratio is reported exactly",
        ),
        (
            {
                **contrasts,
                "functional_form": {
                    key: value for key, value in form.items() if key != "contrasts"
                },
            },
            "spline contrasts are reported exactly",
        ),
        (
            contrasts_with(
                contrasts=[
                    {**form["contrasts"][0], "percentile": 25.0},
                    form["contrasts"][1],
                ]
            ),
            "boundary knot percentiles",
        ),
        (
            contrasts_with(
                contrasts=[
                    {**item, "reference_value": item["reference_value"] + 1.0}
                    for item in form["contrasts"]
                ]
            ),
            "median reference",
        ),
        (
            contrasts_with(disposition="linearity_not_rejected"),
            "contradicts its p value",
        ),
        (
            {
                **contrasts,
                "functional_form": {
                    key: value
                    for key, value in form.items()
                    if key != "overall_p_value"
                },
            },
            "all of its fields or none",
        ),
        (
            {**per_step, "primary_estimate": "spline_percentile_contrasts"},
            "follows from",
        ),
        (
            {
                **per_step,
                "functional_form": {
                    **linear,
                    "contrasts": [
                        {**item, "reference_value": linear["reference_value"]}
                        for item in form["contrasts"]
                    ],
                },
            },
            "spline contrasts are reported exactly",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            ContinuousSurvivalReporting.model_validate(broken)


def test_the_functional_form_outcome_refuses_what_its_test_contradicts() -> None:
    outcome = prespecified_rule_outcomes.FunctionalFormTestOutcome
    base = {
        "schema_version": prespecified_rule_outcomes.RULE_OUTCOME_SCHEMA_VERSION,
        "rule": "functional_form_test",
        "diagnostic": "restricted_cubic_spline_likelihood_ratio_test",
        "alpha": 0.05,
    }
    for fields in (
        {"nonlinearity_p_value": 0.01, "disposition": "linearity_not_rejected"},
        {"nonlinearity_p_value": 0.2, "disposition": "linearity_rejected"},
        {"disposition": "not_assessable"},
        {
            "nonlinearity_p_value": 0.2,
            "disposition": "not_assessable",
            "not_assessable_reason": "tied_knots",
        },
        {
            "nonlinearity_p_value": 0.01,
            "disposition": "linearity_rejected",
            "not_assessable_reason": "tied_knots",
            "primary_estimate": "spline_percentile_contrasts",
        },
        # The contrasts are the result exactly when linearity is rejected and
        # the PH test left a constant estimate.
        {
            "nonlinearity_p_value": 0.01,
            "disposition": "linearity_rejected",
            "primary_estimate": "per_step_hazard_ratio",
        },
        {
            "nonlinearity_p_value": 0.2,
            "disposition": "linearity_not_rejected",
            "primary_estimate": "spline_percentile_contrasts",
        },
    ):
        with pytest.raises(ValueError):
            outcome(**{**base, "primary_estimate": "per_step_hazard_ratio", **fields})
    assert outcome(
        **base,
        nonlinearity_p_value=0.01,
        disposition="linearity_rejected",
        primary_estimate="interval_per_step_hazard_ratios",
    ).model_dump(mode="json") == {
        **base,
        "nonlinearity_p_value": 0.01,
        "disposition": "linearity_rejected",
        "primary_estimate": "interval_per_step_hazard_ratios",
    }


def test_a_check_without_a_result_says_why(tmp_path, ceiling) -> None:
    envelope = ceiling.envelope
    form = envelope["functional_form"]
    assert (form["status"], form["reason"], form["disposition"]) == (
        "not_estimable",
        "tied_knots",
        "not_assessable",
    )
    assert envelope["primary_estimate"] != "spline_percentile_contrasts"
    assert set(
        ceiling.table(ceiling.authority.spline_product)["functional_form_disposition"]
    ) == {"not_assessable"}
    assert (
        ceiling.receipt["functional_form_status"],
        ceiling.receipt["functional_form_disposition"],
        ceiling.receipt["exposure_increment_spread"],
    ) == ("tied_knots", "not_assessable", "central_eighty_percent_range")
    low, high = ceiling.frame["lab_max"].quantile([0.1, 0.9])
    assert ceiling.receipt["exposure_increment_spread_width"] == pytest.approx(
        high - low
    )
    rule = ContinuousSurvivalReporting.model_validate(
        envelope
    ).functional_form_outcome()
    assert rule.result_sentence() == (
        "The restricted cubic spline check of the linear exposure term had no result, "
        "because the exposure had too few distinct values for its three spline knots, "
        "so the per-step estimates stand unchecked."
    )
    assert derive_scientific_claim_drafts(ceiling.summary)[-1].claim_id == (
        "functional_form_rule"
    )
    assert (
        "within the exposure's range between its tenth and ninetieth percentiles in "
        "the modelled records"
    ) in _methods_fact(tmp_path / "methods", ceiling.summary)


def test_a_model_without_covariates_is_not_called_adjusted(
    tmp_path, both_ends_unadjusted, both_ends
) -> None:
    store, ledger = _strict_store(tmp_path / "unadjusted", both_ends_unadjusted.summary)
    claims = store.authoritative_scientific_claims(ledger)
    assert [(claim.claim_id, claim.analysis_role) for claim in claims] == (
        CONTRAST_CLAIMS
    )
    assert all(claim.adjusted_for == [] for claim in claims)
    assert claims[0].estimand.startswith("10th-percentile-versus-median (")
    assert " mmol/L) hazard ratio from the restricted cubic spline model" in (
        claims[0].estimand
    )
    assert claims[2].estimand.startswith(
        "per-5 mmol/L hazard ratio for days 0 to 7 after the landmark"
    )
    sentences = [
        sentence
        for claim in claims
        for sentence in (
            claim.render_reader_text(),
            claim.render_reader_text(include_estimate=False),
        )
    ]
    assert not any("adjust" in sentence.casefold() for sentence in sentences)
    assert claims[-2].rule_outcome.result_sentence() == (
        "The Schoenfeld residual test did not reject the proportional-hazards "
        "assumption at the prespecified alpha of 0.050, so the primary estimates are "
        "hazard ratios constant over follow-up."
    )
    contract, _receipt, svg = _figure(tmp_path / "figure", both_ends_unadjusted)
    (panel_b,) = [panel for panel in contract.panels if panel.panel_id == "b"]
    assert panel_b.title == "Hazard ratio curve"
    assert "(b) Hazard ratio across" in contract.reader_caption
    assert "Adjusted hazard ratio" not in contract.reader_caption
    assert "HR vs median (95% CI)" in svg and "Adjusted HR" not in svg
    assert "adjusted for" not in _methods_fact(
        tmp_path / "methods", both_ends_unadjusted.summary
    )
    # The adjusted models say so (control).
    adjusted = derive_scientific_claim_drafts(both_ends.summary)
    assert " mmol/L) adjusted hazard ratio from the restricted cubic spline model" in (
        adjusted[0].estimand
    )
    assert adjusted[0].adjusted_for == ["age", "sex"]


def test_the_methods_state_the_step_and_the_rule_that_chose_the_result(
    tmp_path, enzyme
) -> None:
    text = _methods_fact(tmp_path / "methods", enzyme.summary)

    assert (
        "per 500 units of the exposure's recorded scale (“U/L”), the largest step of "
        "one, two or five times a power of ten within the exposure's interquartile "
        "range in the modelled records"
    ) in text
    assert (
        "the linear exposure term was compared with a restricted cubic spline with "
        "knots at the 10th, 50th and 90th percentiles of the exposure by a "
        "likelihood-ratio test at a prespecified alpha of 0.05, and when it rejected "
        "linearity while proportional hazards held, the spline's contrasts at the "
        "10th and 90th percentiles against the median replaced the per-step "
        "association; the spline terms were also tested together against no exposure "
        "term; the proportional-hazards test used the linear term, so a misspecified "
        "linear form can appear as non-proportional hazards"
    ) in text
