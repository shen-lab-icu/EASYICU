"""Executable reference workflow packages ("skill packages") over the research-agent kernels.

Not to be confused with ``research_agent.skills`` (the ClinicalSkill registry):
this package holds runnable, tested analysis workflows.

A skill is one method family packaged the way a reviewer can run it: a
``SKILL.md`` that fixes the standard workflow, a ``scripts/`` package with a
few typed entry functions that print verification tokens, ``references/`` for
method notes, and a synthetic example cohort so the whole package can be
smoke-tested in under a minute.

Skills compose existing owners -- the landmark eligibility mask, the adjusted
association kernel, the ordered-trend and spline primitives -- instead of
re-deriving them.  They do not select the exposure, outcome, cohort or estimand:
those arrive in a typed specification that the caller (a reviewed plan, or a
user running the script directly) supplies.  Every number a skill writes into
``report.md`` is copied from the tables it exported, and ``export_all`` refuses
to print its completion token until the export consistency gate has passed.

This package must stay free of webserver and orchestration imports; the host
pipeline may call a skill, a skill never calls the host.

The task entries below also provide scalar-only sample-size planning and a
descriptive treatment composition. Numerical dependencies are lazy so browsing
the Skill Hub does not import statistical kernels. Planning is not a study run.
"""

from __future__ import annotations

import math
from numbers import Real
from statistics import NormalDist
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from ..schema import TableOneSpec


class PlanningInputError(ValueError):
    def __init__(self, reason_code: str, field: str):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {field}")


def _number(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise PlanningInputError("planning.invalid_number", field)
    return float(value)


def _probability(value: float, field: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 < value < 1):
        raise PlanningInputError("planning.invalid_probability", field)
    return float(value)


def _positive(value: float, field: str) -> float:
    number = _number(value, field)
    if number <= 0:
        raise PlanningInputError("planning.positive_required", field)
    return number


def _design(alpha: float, power: float) -> tuple[float, float]:
    alpha = _probability(alpha, "alpha")
    power = _probability(power, "power")
    if power <= alpha:
        raise PlanningInputError("planning.power_not_above_alpha", "power")
    return alpha, power


def _result(method: str, assumptions: dict[str, Any], *, basis: str,
            aggregate_receipt: str | None, **values: Any) -> dict[str, Any]:
    if basis not in {"user_assumptions", "governed_aggregate"}:
        raise PlanningInputError("planning.assumption_source_required", "basis")
    if basis == "governed_aggregate" and (
        not isinstance(aggregate_receipt, str) or not aggregate_receipt.strip()
    ):
        raise PlanningInputError("planning.aggregate_receipt_required", "aggregate_receipt")
    if basis == "user_assumptions" and aggregate_receipt is not None:
        raise PlanningInputError("planning.assumption_source_mismatch", "aggregate_receipt")
    return {
        "schema_version": "easyicu.sample_size_planning/1",
        "method": method, "assumptions": assumptions,
        "basis": basis, "aggregate_receipt": aggregate_receipt,
        "claim_ceiling": "analysis_only", "manuscript_findings": False,
        "patient_data_read": False, **values,
    }


def _two_arm(solver: Any, effect: float, alpha: float, power: float,
             ratio: float) -> dict[str, Any]:
    if effect == 0:
        raise PlanningInputError("planning.null_effect", "effect")
    ratio = _positive(ratio, "allocation_ratio")
    try:
        n = float(solver.solve_power(effect_size=abs(effect), alpha=alpha,
                                    power=power, ratio=ratio, alternative="two-sided"))
        if not math.isfinite(n) or n <= 0:
            raise ValueError("solver did not converge")
        n1 = max(2, math.ceil(n))
        n2 = max(2, math.ceil(n1 * ratio))
        achieved = float(solver.power(effect_size=abs(effect), nobs1=n1,
                                     alpha=alpha, ratio=n2 / n1, alternative="two-sided"))
        if not math.isfinite(achieved) or achieved < power - 1e-8:
            raise ValueError("rounded allocation did not attain power")
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise PlanningInputError("planning.numerical_failure", "sample_size") from exc
    return {"group_sizes": [n1, n2], "total_n": n1 + n2,
            "achieved_power": achieved, "standardized_effect": abs(effect)}


def two_proportions(*, p1: float, p2: float, alpha: float = .05, power: float = .8,
                    allocation_ratio: float = 1, basis: str = "user_assumptions",
                    aggregate_receipt: str | None = None) -> dict[str, Any]:
    """Independent binomial groups; Cohen's arcsine h normal approximation."""
    from statsmodels import __version__
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    p1, p2 = _probability(p1, "p1"), _probability(p2, "p2")
    alpha, power = _design(alpha, power)
    values = _two_arm(NormalIndPower(), float(proportion_effectsize(p1, p2)),
                      alpha, power, allocation_ratio)
    return _result("two_proportions_cohen_h", dict(p1=p1, p2=p2, alpha=alpha,
                   power=power, allocation_ratio=allocation_ratio), basis=basis,
                   aggregate_receipt=aggregate_receipt,
                   implementation=f"statsmodels/{__version__}", **values)


def two_means(*, difference: float, standard_deviation: float, alpha: float = .05,
              power: float = .8, allocation_ratio: float = 1,
              basis: str = "user_assumptions", aggregate_receipt: str | None = None
              ) -> dict[str, Any]:
    """Independent normal groups with common SD; noncentral two-sample t."""
    from statsmodels import __version__
    from statsmodels.stats.power import TTestIndPower

    difference = _number(difference, "difference")
    sd = _positive(standard_deviation, "standard_deviation")
    alpha, power = _design(alpha, power)
    values = _two_arm(TTestIndPower(), difference / sd, alpha, power, allocation_ratio)
    return _result("two_means_common_sd", dict(difference=difference,
                   standard_deviation=sd, alpha=alpha, power=power,
                   allocation_ratio=allocation_ratio), basis=basis,
                   aggregate_receipt=aggregate_receipt,
                   implementation=f"statsmodels/{__version__}", **values)


def survival_events(*, hazard_ratio: float, allocation_fraction: float = .5,
                    alpha: float = .05, power: float = .8,
                    event_fraction: float | None = None, basis: str = "user_assumptions",
                    aggregate_receipt: str | None = None) -> dict[str, Any]:
    """Schoenfeld required events; total N requires a supplied event fraction.

    Event fraction must already reflect follow-up, censoring and attrition.
    This is not a competing-risk, non-proportional or cluster design formula.
    """
    hr = _positive(hazard_ratio, "hazard_ratio")
    if hr == 1:
        raise PlanningInputError("planning.null_effect", "hazard_ratio")
    fraction = _probability(allocation_fraction, "allocation_fraction")
    alpha, power = _design(alpha, power)
    z = NormalDist()
    events = math.ceil((z.inv_cdf(1 - alpha / 2) + z.inv_cdf(power)) ** 2
                       / (fraction * (1 - fraction) * math.log(hr) ** 2))
    rate = None if event_fraction is None else _probability(event_fraction, "event_fraction")
    return _result("schoenfeld_events", dict(hazard_ratio=hr, allocation_fraction=fraction,
                   alpha=alpha, power=power, event_fraction=rate), basis=basis,
                   aggregate_receipt=aggregate_receipt, required_events=events,
                   total_n=None if rate is None else math.ceil(events / rate),
                   implementation="Schoenfeld-1983-equation-port/1")


def prediction_binary(*, parameters: int, cox_snell_r2: float, prevalence: float,
                      shrinkage: float = .9, r2_optimism: float = .05,
                      prevalence_margin: float = .05, confidence: float = .95,
                      basis: str = "user_assumptions", aggregate_receipt: str | None = None
                      ) -> dict[str, Any]:
    """Riley binary logistic development minimum N, maximum of three criteria.

    Count candidate predictor parameters (dummy terms, splines and interactions),
    excluding the intercept. R² is anticipated optimism-adjusted Cox-Snell R²,
    never apparent fitted R² or Nagelkerke R². Precision is a Wald approximation.
    """
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters < 1:
        raise PlanningInputError("planning.invalid_parameter_count", "parameters")
    r2 = _probability(cox_snell_r2, "cox_snell_r2")
    prev = _probability(prevalence, "prevalence")
    shrinkage = _probability(shrinkage, "shrinkage")
    optimism = _probability(r2_optimism, "r2_optimism")
    margin = _probability(prevalence_margin, "prevalence_margin")
    confidence = _probability(confidence, "confidence")
    maximum_r2 = -math.expm1(2 * (prev * math.log(prev) + (1 - prev) * math.log1p(-prev)))
    optimism_shrinkage = r2 / (r2 + optimism * maximum_r2)
    if r2 >= min(maximum_r2, shrinkage, optimism_shrinkage):
        raise PlanningInputError("planning.incompatible_r2", "cox_snell_r2")

    def criterion(target: float) -> int:
        return math.ceil(parameters / ((target - 1) * math.log1p(-r2 / target)))

    criteria = {
        "shrinkage": criterion(shrinkage),
        "r2_optimism": criterion(optimism_shrinkage),
        "intercept_precision": math.ceil(NormalDist().inv_cdf((1 + confidence) / 2) ** 2
                                         * prev * (1 - prev) / margin ** 2),
    }
    n = max(criteria.values())
    return _result("riley_binary_prediction", dict(parameters=parameters, cox_snell_r2=r2,
                   prevalence=prev, shrinkage=shrinkage, r2_optimism=optimism,
                   prevalence_margin=margin, confidence=confidence), basis=basis,
                   aggregate_receipt=aggregate_receipt, minimum_n=n, criteria_n=criteria,
                   expected_events=n * prev, events_per_parameter=n * prev / parameters,
                   maximum_cox_snell_r2=maximum_r2,
                   implementation="Riley-2019-binary-equation-port/1")


class TreatmentPatternError(ValueError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def summarize_treatment_patterns(
    frame: pd.DataFrame, *, table_spec: TableOneSpec, treatment_class: str,
    captured_agents: Sequence[str], capture_confirmed: bool, source_receipt: str,
) -> dict[str, Any]:
    """Return proportions, exposed-only timing and first-use CIF by frozen group.

    Required columns: stay_id, exposed (0/1), first_exposure_hours (null only for
    unexposed), followup_hours (ICU exit), death and discharged_alive (0/1),
    and the reviewed grouping column. Complete ICU stays are required: exactly
    one exit flag must be present. All hours are measured from ICU admission.
    First use is event 1, death before use event 2, alive discharge before use
    event 3. Both exits preclude later in-ICU use. Unresolved use/exit ties stop.
    """
    import numpy as np
    import pandas as pd

    from ..methods.competing_risks import estimate_cif
    from ..methods.table_one import TableOneContractError, build_grouped_table_one
    from ..planning.treatment_capture import packaged_treatment_capture_registry

    loaded = packaged_treatment_capture_registry()
    agents = loaded.registry.class_agents(treatment_class)
    if agents is None:
        raise TreatmentPatternError("treatment.class_unknown")
    if isinstance(captured_agents, (str, bytes)) or set(captured_agents) != agents:
        raise TreatmentPatternError("treatment.class_capture_incomplete")
    if capture_confirmed is not True or not isinstance(source_receipt, str) or not source_receipt.strip():
        raise TreatmentPatternError("treatment.capture_review_required")
    specs = {variable.name: variable for variable in table_spec.variables}
    if (set(specs) != {"exposed", "first_exposure_hours"}
            or specs["exposed"].summary != "count_percent"
            or specs["exposed"].levels != [0, 1]
            or specs["first_exposure_hours"].summary != "median_iqr"
            or table_spec.missing_group_policy != "fail_closed"):
        raise TreatmentPatternError("treatment.summary_contract_invalid")
    required = {"stay_id", "exposed", "first_exposure_hours", "followup_hours", "death", "discharged_alive", table_spec.group_by}
    if not required <= set(frame.columns) or frame.empty:
        raise TreatmentPatternError("treatment.columns_required")
    if frame["stay_id"].isna().any() or frame["stay_id"].duplicated().any():
        raise TreatmentPatternError("treatment.one_row_per_stay_required")
    if (not frame["exposed"].isin([0, 1]).all() or not frame["death"].isin([0, 1]).all()
            or not frame["discharged_alive"].isin([0, 1]).all()
            or frame["death"].eq(frame["discharged_alive"]).any()
            or frame[table_spec.group_by].isna().any()):
        raise TreatmentPatternError("treatment.ascertainment_required")
    exposed = frame["exposed"].eq(1)
    try:
        first = pd.to_numeric(frame["first_exposure_hours"], errors="raise").astype(float)
        end = pd.to_numeric(frame["followup_hours"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TreatmentPatternError("treatment.time_invalid") from exc
    if (not np.isfinite(end).all() or (end < 0).any()
            or not np.isfinite(first[exposed]).all() or (first[exposed] < 0).any()
            or (first[exposed] > end[exposed]).any() or first[~exposed].notna().any()
            or pd.api.types.is_bool_dtype(frame["first_exposure_hours"])
            or pd.api.types.is_bool_dtype(frame["followup_hours"])):
        raise TreatmentPatternError("treatment.time_invalid")
    if (exposed & first.eq(end)).any():
        raise TreatmentPatternError("treatment.event_order_unresolved")

    # Preserve the reviewed Table 1 declaration. Its inferential columns are
    # deliberately not part of this descriptive product's returned contract.
    execution_spec = table_spec
    if not exposed.any():
        execution_spec = table_spec.model_copy(update={"variables": [specs["exposed"]]})
    try:
        table = build_grouped_table_one(frame, execution_spec)
    except TableOneContractError as exc:
        raise TreatmentPatternError("treatment.summary_contract_invalid") from exc
    exposure_columns = ["group", "group_order", "category", "count", "denominator_n", "percentage"]
    exposure_rows = table.loc[table.variable.eq("exposed"), exposure_columns]
    timing = []
    for row in table.loc[table.variable.eq("first_exposure_hours")].to_dict("records"):
        timing.append({"group": row["group"], "group_order": row["group_order"],
                       "exposed_n": row["nonmissing_n"],
                       **{key: None if pd.isna(row[key]) else row[key] for key in ("median", "q25", "q75")},
                       "unit": "hours", "denominator": "exposed_stays"})
    cif = {}
    for group in table_spec.group_levels:
        mask = frame[table_spec.group_by].eq(group)
        events = np.where(exposed[mask], 1, np.where(frame.loc[mask, "death"].eq(1), 2, 3))
        durations = np.where(exposed[mask], first[mask], end[mask])
        if not (events == 1).any():
            # The CIF kernel requires an observed event of interest. Do not
            # fabricate a zero-valued fitted curve when that precondition fails.
            cif[str(group)] = {"status": "not_estimable", "reason_code": "treatment.no_exposures",
                               "n": int(mask.sum())}
        else:
            cif[str(group)] = estimate_cif(durations, events, 1).to_json()
    return {"schema_version": "easyicu.treatment_patterns/1", "claim_ceiling": "analysis_only",
            "treatment_class": treatment_class, "source_receipt": source_receipt,
            "capture_registry_sha256": loaded.sha256,
            "exposure": exposure_rows.to_dict("records"), "timing": timing,
            "timing_status": "available" if exposed.any() else "treatment.no_exposures",
            "cumulative_incidence": cif,
            "time_origin": "icu_admission", "time_interpretation": "first_recorded_in_icu_use",
            "censoring": "none; complete ICU stays required",
            "competing_events": {"2": "death_before_first_exposure",
                                 "3": "alive_discharge_before_first_exposure"}}


def public_demo() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eight invented stays, distributable without patient data or file IO."""
    import pandas as pd
    from ..planning.treatment_capture import packaged_treatment_capture_registry
    from ..schema import TableOneSpec

    frame = pd.DataFrame({
        "stay_id": list(range(8)), "group": ["A"] * 4 + ["B"] * 4,
        "exposed": [1, 1, 0, 0, 1, 1, 0, 0],
        "first_exposure_hours": [2, 4, None, None, 1, 7, None, None],
        "followup_hours": [12, 14, 6, 18, 10, 20, 5, 24],
        "death": [0, 0, 1, 0, 0, 0, 1, 0],
        "discharged_alive": [1, 1, 0, 1, 1, 1, 0, 1],
    })
    spec = TableOneSpec(group_by="group", group_levels=["A", "B"], variables=[
        {"name": "exposed", "variable_kind": "categorical", "summary": "count_percent",
         "test": "chi_square_with_fisher_exact_for_sparse_2x2", "levels": [0, 1]},
        {"name": "first_exposure_hours", "variable_kind": "continuous", "summary": "median_iqr",
         "test": "mann_whitney_or_kruskal"},
    ])
    return frame, {"table_spec": spec, "treatment_class": "vasopressor",
                   "captured_agents": sorted(packaged_treatment_capture_registry().registry.class_agents("vasopressor")),
                   "capture_confirmed": True, "source_receipt": "public-synthetic-demo/1"}


__all__ = ["PlanningInputError", "TreatmentPatternError", "two_proportions", "two_means",
           "survival_events", "prediction_binary", "summarize_treatment_patterns", "public_demo"]
