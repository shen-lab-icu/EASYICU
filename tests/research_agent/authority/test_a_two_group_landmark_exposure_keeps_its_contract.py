"""A two-group exposure keeps its signed landmark categorical contract.

UX leg 3 (2026-11-19): first-day lactate grouped as <=2 / >2 mmol/L was
signed as an ordinal two-level landmark categorical contract. The host's own
family template codes a two-level exposure term ``binary``, and the contract
accepted only ``categorical``, so every re-plan was refused on
``exposure_coding``. Exactly two levels are coded ``binary`` whatever the
exposure's kind (S1432); more levels stay ``categorical``. Synthetic,
case-neutral data.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    CurrentCaseScientificAuthorityError,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.planning.family_spec.landmark_categorical_template import (
    __file__ as _template_file,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection
from tests.research_agent.authority.test_landmark_categorical_plan_bound_roster import (
    _dependence,
    _draft_plan,
    _frame,
    _landmark_spec,
)


def _exact_authority(tmp_path):
    universe = tmp_path / "universe.parquet"
    _frame().to_parquet(universe, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"}, sensitivity_specs=(_landmark_spec(),),
        primary_exposure="aki_stage_strict", primary_exposure_source="aki_stage",
        target_outcome="death", declared_covariates=("age", "sex"), covariate_operationalizations={},
        target_is_event_status=True, universe_path=universe,
        scientific_configuration_sha256="b" * 64, dependence=_dependence(),
    )
    return load_current_case_scientific_runtime_authority(projection.authority)


def _two_groups(authority):
    """The same signed contract over an ordered two-group exposure (levels 1 < 2)."""
    return authority.model_copy(update={
        "exposure_kind": "ordinal", "exposure_levels": ("1", "2"),
        "exposure_reference_level": "1", "primary_contrast_level": "2",
    })


def _primary(levels: list[str], coding: str):
    plan = _draft_plan().model_dump(mode="json")
    requirement = plan["steps"][1]["model_requirements"][0]
    exposure = requirement["model_terms"][0]
    exposure.update({"coding": coding, "levels": levels, "reference_level": levels[0]})
    requirement.update({
        "exposure_levels": levels, "exposure_reference_level": levels[0],
        "primary_contrast_level": levels[-1],
    })
    return AnalysisPlan.model_validate(plan).steps[1]


@pytest.mark.parametrize("kind", ["ordinal", "categorical", "binary"])
def test_a_two_level_exposure_is_valid_only_as_binary_whatever_its_kind(tmp_path, kind):
    authority = _two_groups(_exact_authority(tmp_path)).model_copy(update={"exposure_kind": kind})
    assert authority.exposure_term_codings == ("binary",)
    assert authority._validated_requirement(_primary(["1", "2"], "binary"))
    with pytest.raises(CurrentCaseScientificAuthorityError, match="exposure_coding"):
        authority._validated_requirement(_primary(["1", "2"], "categorical"))


def test_the_published_contract_names_the_coding_the_host_template_writes(tmp_path):
    authority = _two_groups(_exact_authority(tmp_path))
    assert '"coding": "binary"' in authority.planning_contract_context()
    source = open(_template_file, encoding="utf-8").read()
    # The family template's own rule for a two-level exposure term.
    assert '"categorical" if len(request.exposure_levels) > 2 else "binary"' in source


def test_more_than_two_levels_stay_categorical_only(tmp_path):
    authority = _exact_authority(tmp_path)
    assert len(authority.exposure_levels) > 2
    levels = list(authority.exposure_levels)
    assert authority._validated_requirement(_primary(levels, "categorical"))
    assert authority.exposure_term_codings == ("categorical",)  # the schema forbids binary here
    assert '"coding": "categorical"' in authority.planning_contract_context()


def _fit(frame, coding: str):
    import numpy as np
    import statsmodels.api as sm

    from easyicu.research_agent.contracts.model_terms import ModelTermSpec
    from easyicu.research_agent.execution.model_matrix import compile_model_terms

    terms = [
        ModelTermSpec(name="group", role="exposure", coding=coding, levels=["1", "2"],
                      reference_level="1", transform="treatment_contrast"),
        ModelTermSpec(name="age", role="covariate", coding="continuous", transform="identity"),
    ]
    compiled = compile_model_terms(frame, terms=terms, exposure="group")
    design = sm.add_constant(compiled.design, has_constant="add")
    fit = sm.Logit(frame["death"].astype(float), design).fit(disp=0, tol=1e-14, maxiter=200)
    column, = compiled.exposure_columns
    return column, np.array([
        fit.params[column], *fit.conf_int().loc[column].to_numpy(), fit.pvalues[column],
    ])


def test_binary_and_categorical_treatment_coding_of_two_levels_fit_identically():
    """S1432 (2): same design column, estimate, CI and p; the signed direction is kept."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20261119)
    n = 600
    group = rng.choice(["1", "2"], size=n, p=[0.6, 0.4])
    age = rng.normal(65, 12, size=n)
    risk = 1 / (1 + np.exp(-(-2.0 + 0.9 * (group == "2") + 0.02 * (age - 65))))
    frame = pd.DataFrame({"group": group, "age": age, "death": rng.binomial(1, risk)})
    binary_column, binary = _fit(frame, "binary")
    categorical_column, categorical = _fit(frame, "categorical")
    assert binary_column == categorical_column  # the contrast column for level "2" vs reference "1"
    np.testing.assert_allclose(binary, categorical, rtol=0, atol=1e-10)
    assert binary[0] > 0  # level 2 vs reference 1: the simulated harm, not its inverse
