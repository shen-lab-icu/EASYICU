"""One declared binary model, with contrasts and stratum estimates from its covariance.

The model is the logistic odds-ratio default, or the modified-Poisson
risk-ratio model when the requirement declares that estimator; the owner reads
the estimator from the requirement and never chooses it. The effect-modification
action adds main effects and exposure interactions for the declared modifiers.
It never refits strata or chooses terms from results. The baseline family also
uses this owner with an empty modifier roster.
"""

from __future__ import annotations

import math

from ...contracts.association_execution import association_estimator_support
from ...contracts.model_terms import level_spelling
from ...robustness.estimators import fit_estimator
from ..model_matrix import primary_model_rows
from .adjusted_association_executor import _cluster_groups
from .landmark_spline_fit import restricted_cubic_spline_basis


def baseline_model_population(frame, requirement, authority):
    terms = list(requirement.model_terms)
    existing = {term.name for term in terms}
    terms.extend(term for term in authority.subgroup_terms if term.name not in existing)
    needed = list(dict.fromkeys([authority.outcome_column, *(term.name for term in terms),
                                *((authority.dependence.group_source,) if authority.dependence else ())]))
    if set(needed) - set(frame.columns):
        raise ValueError("baseline primary input lacks declared columns")
    compiled = primary_model_rows(frame[needed], terms=terms, exposure=authority.exposure_column,
                                  outcome=authority.outcome_column)
    return frame.loc[compiled.design.index], compiled


#: The effect scale each binary estimator kind reports.
BINARY_EFFECT_SCALES = {"logistic": "odds_ratio", "glm_poisson": "risk_ratio"}


def baseline_estimator_kind(requirement) -> str:
    """The binary estimator the requirement declares; anything else is refused."""

    support = association_estimator_support(requirement)
    if not support.supported or support.runtime_kind not in BINARY_EFFECT_SCALES:
        raise ValueError(
            "baseline association requires a declared binary estimator this owner "
            "implements: " + support.reason
        )
    return support.runtime_kind


def fit_baseline_association(frame, *, requirement, authority):
    import numpy as np
    import pandas as pd
    import patsy
    from scipy.stats import chi2, norm
    from scipy.special import expit

    kind = baseline_estimator_kind(requirement)
    effect_scale = BINARY_EFFECT_SCALES[kind]
    population, compiled = baseline_model_population(frame, requirement, authority)
    design = compiled.design.copy()
    exposure = authority.exposure_column
    exposure_columns = list(compiled.exposure_columns)
    mapping = dict(compiled.source_by_design_column)
    contrasts = []
    if authority.exposure_encoding == "restricted_cubic_spline":
        knots = tuple(float(v) for v in population[exposure].quantile(authority.spline_knot_quantiles))
        basis = restricted_cubic_spline_basis(population[exposure], knots=knots, prefix=exposure)
        design = pd.concat([basis, design.drop(columns=exposure_columns)], axis=1)
        exposure_columns = list(basis.columns)
        coordinates = np.linspace(knots[0], knots[-1], 21)
        reference = np.asarray(patsy.build_design_matrices([basis.design_info], {"x": [knots[1]]})[0])[0]
        for value in coordinates:
            vector = np.asarray(patsy.build_design_matrices([basis.design_info], {"x": [value]})[0])[0] - reference
            contrasts.append((str(float(value)), str(knots[1]), dict(zip(exposure_columns, vector)), value == knots[-1]))
        mapping.update({column: exposure for column in exposure_columns})
    else:
        term = next(term for term in requirement.model_terms if term.role == "exposure")
        for level, column in zip(term.contrast_levels, exposure_columns):
            contrasts.append((level, authority.exposure_reference_level, {column: 1.0}, level == authority.primary_contrast_level))
    interactions = {}
    for modifier in authority.subgroup_terms:
        columns = [column for column, source in mapping.items() if source == modifier.name and column in design]
        if not columns:
            raise ValueError("effect modification lacks declared modifier main effects")
        interactions[modifier.name] = []
        for modifier_column in columns:
            for exposure_column in exposure_columns:
                name = exposure_column + ":" + modifier_column
                design[name] = design[exposure_column] * design[modifier_column]
                mapping[name] = exposure
                interactions[modifier.name].append((name, exposure_column, modifier_column))
    groups, variance = _cluster_groups(frame=population, dependence=authority.dependence)
    result = fit_estimator(cohort=None, X=design, y=population[authority.outcome_column], kind=kind,
                           term=exposure_columns[0], source_by_design_column=mapping,
                           variance_estimator=variance, cluster_groups=groups)
    if not result.converged or result.separation_detected or not result.coefficient_covariance:
        raise ValueError(f"baseline {kind} model is not estimable: " + result.notes)
    names = [term.term for term in result.terms]
    beta = np.array([math.log(term.estimate) for term in result.terms])
    covariance = np.asarray(result.coefficient_covariance)
    if not np.isfinite(beta).all() or not np.isfinite(covariance).all():
        raise ValueError("baseline model covariance is not finite")
    tests = {}
    for modifier, entries in interactions.items():
        indices = [names.index(name) for name, _, _ in entries]
        block = covariance[np.ix_(indices, indices)]
        if np.linalg.matrix_rank(block) != len(indices):
            raise ValueError("interaction covariance is singular")
        statistic = float(beta[indices] @ np.linalg.solve(block, beta[indices]))
        tests[modifier] = float(chi2.sf(statistic, len(indices)))
    strata = [(None, None, population.index)]
    for modifier in authority.subgroup_terms:
        for level in modifier.levels:
            rows = population.index[population[modifier.name].map(level_spelling).eq(level)]
            if len(rows) == 0:
                raise ValueError("a declared subgroup stratum is empty")
            strata.append((modifier.name, level, rows))
    output = []
    for modifier, level, indices in strata:
        for exposed, reference, delta, primary in contrasts:
            vector = pd.Series(0.0, index=names)
            for column, value in delta.items():
                vector[column] = value
            for entries in interactions.values():
                for name, exposure_column, modifier_column in entries:
                    vector[name] = delta.get(exposure_column, 0.0) * float(design.loc[indices, modifier_column].mean())
            log_ratio = float(vector.to_numpy() @ beta)
            se = math.sqrt(max(0.0, float(vector.to_numpy() @ covariance @ vector.to_numpy())))
            z = float(norm.ppf(0.975))
            output.append(dict(
                fit_status="fitted", estimate=math.exp(log_ratio), ci_low=math.exp(log_ratio - z * se),
                ci_high=math.exp(log_ratio + z * se), standard_error=se, effect_scale=effect_scale,
                exposure=exposure, outcome=authority.outcome_column, model_id=requirement.requirement_id,
                requirement_id=requirement.requirement_id, covariates=";".join(requirement.covariates),
                analysis_role="primary", analysis_set=requirement.analysis_set, estimator_kind=kind,
                n=result.n, n_events=result.n_events, n_parameters=len(design.columns), variance_estimator=result.variance_estimator,
                outcome_proportion=result.n_events / result.n,
                cluster_count=result.cluster_count, exposure_level=exposed, reference_level=reference,
                contrast=f"{exposed} vs {reference}" + (f"; {modifier}={level}" if modifier else ""),
                is_primary_contrast=primary and modifier is None,
                subgroup=modifier or "", stratum=level or "", stratum_n=len(indices),
                interaction_p_value=tests.get(modifier), notes=result.notes,
            ))
            if authority.exposure_encoding == "restricted_cubic_spline":
                gradients, risks = [], []
                for value in (float(reference), float(exposed)):
                    prediction = design.loc[indices].copy()
                    encoded = np.asarray(patsy.build_design_matrices([basis.design_info], {"x": [value]})[0])[0]
                    prediction[exposure_columns] = np.tile(encoded, (len(prediction), 1))
                    for entries in interactions.values():
                        for name, exposure_column, modifier_column in entries:
                            prediction[name] = prediction[exposure_column] * prediction[modifier_column]
                    prediction.insert(0, "const", 1.0)
                    matrix = prediction[names].to_numpy()
                    # Standardized risk under the fitted link: d expit(x'b)/db =
                    # p(1-p)x for the logit; d exp(x'b)/db = p x for the log link.
                    if kind == "glm_poisson":
                        probability = np.exp(matrix @ beta)
                        slope = probability
                    else:
                        probability = expit(matrix @ beta)
                        slope = probability * (1 - probability)
                    risks.append(float(probability.mean()))
                    gradients.append((matrix * slope[:, None]).mean(axis=0))
                if not all(0.0 <= risk <= 1.0 for risk in risks):
                    # A log-link model can predict above one; a mean risk
                    # outside [0, 1] is not a risk and is not reported as one.
                    raise ValueError("baseline standardized risk falls outside [0, 1] under the declared model")
                gradient = gradients[1] - gradients[0]
                risk_se = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
                difference = risks[1] - risks[0]
                output[-1].update(absolute_risk=risks[1], reference_risk=risks[0], risk_difference=difference,
                                  risk_difference_ci_low=difference - z * risk_se,
                                  risk_difference_ci_high=difference + z * risk_se,
                                  risk_estimand="model_standardized_difference")
    return population, design, result, output, tests
