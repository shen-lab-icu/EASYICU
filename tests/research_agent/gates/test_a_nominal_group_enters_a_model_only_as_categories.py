"""A nominal exposure grouping's codes enter a model only as categories.

A study's nominal grouping is staged as integer level codes
(``contracts.exposure_group_rules``): ``1`` to ``k`` name its groups and
``k + 1`` its unmeasured stays.  The codes carry no order or spacing, so the
deterministic code gate (``audits.patterns`` through ``audits.nominal_groups``)
refuses a script that enters them in a linear model as one numeric term, or
that reads them as the ordered levels of an ordered-stratified analysis.
Strata, clusters and weights are named, not modelled, and an ordinal
grouping's codes lie along its scale: both pass.  So does a nominal grouping
a model predicts -- a generalized propensity score's outcome -- and a table
that drops the codes before it is modelled.  Synthetic contexts and scripts
only.
"""

from __future__ import annotations

import textwrap
from typing import Optional

import pytest

from easyicu.research_agent.audits.patterns import AnalysisPatternAuditor
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
    EXPOSURE_GROUP_TRANSFORM_ID,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

_LINEAR = "nominal_group_used_as_linear_term"
_ORDERED = "nominal_group_read_as_ordered"


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Compare hospital death across lactate groups.",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="miiv", n_stays=800, n_patients=800
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"
            ),
            ConceptDescriptor(name="sex", role=VariableRole.DEMOGRAPHIC, dtype="str"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(name="los_icu", role=VariableRole.TIME, dtype="float64"),
            ConceptDescriptor(
                name="lact_group_x1",
                role=VariableRole.OTHER,
                dtype="int64",
                unit_normalization=EXPOSURE_GROUP_TRANSFORM_ID,
            ),
            ConceptDescriptor(
                name="lact_group_x2",
                role=VariableRole.OTHER,
                dtype="int64",
                unit_normalization=EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
                is_ordinal=True,
            ),
        ],
        target_outcome="death",
    )


def _refusals(script: str, step: Optional[AnalysisStep] = None) -> list[str]:
    findings = AnalysisPatternAuditor().audit(
        context=_context(), script_text=textwrap.dedent(script), step=step
    )
    return [
        str(finding.detail.get("kind"))
        for finding in findings
        if finding.severity == "error"
        and str(finding.detail.get("kind", "")).startswith("nominal_group")
    ]


_HEAD = """\
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.read_parquet("cohort.parquet")
"""


@pytest.mark.parametrize(
    ("formula", "refused"),
    [
        pytest.param(
            "death ~ C(lact_group_x1, Treatment(reference=1)) + age",
            [],
            id="categories",
        ),
        pytest.param("death ~ lact_group_x1 + age", [_LINEAR], id="one-number"),
        pytest.param("death ~ age * lact_group_x1", [_LINEAR], id="interaction"),
        pytest.param(
            "death ~ C(lact_group_x1) + lact_group_x1", [_LINEAR], id="also-a-number"
        ),
        pytest.param('death ~ Q("lact_group_x1") + age', [_LINEAR], id="quoted"),
        pytest.param("death ~ lact_group_x2 + age", [], id="ordinal-scale"),
    ],
)
def test_a_formula_takes_a_nominal_group_only_as_categories(
    formula: str, refused: list[str]
) -> None:
    script = _HEAD + f"    model = smf.logit({formula!r}, data=df).fit()\n"

    assert _refusals(script) == refused


@pytest.mark.parametrize(
    ("assembled", "refused"),
    [
        pytest.param(
            '    formula = "death ~ lact_group_x1 + age"\n', [_LINEAR], id="named"
        ),
        pytest.param(
            '    terms = ["age", "lact_group_x1"]\n'
            '    formula = "death ~ " + " + ".join(terms)\n',
            [_LINEAR],
            id="joined",
        ),
        pytest.param(
            '    exposure = "lact_group_x1"\n'
            '    formula = f"death ~ {exposure} + age"\n',
            [_LINEAR],
            id="f-string",
        ),
        pytest.param(
            '    terms = ["age", "C(lact_group_x1)"]\n'
            '    formula = "death ~ " + " + ".join(terms)\n',
            [],
            id="joined-categories",
        ),
        pytest.param(
            '    df["lact_group_x1"] = df["lact_group_x1"].astype("category")\n'
            '    formula = "death ~ lact_group_x1 + age"\n',
            [],
            id="cast-to-categories",
        ),
        pytest.param(
            '    exposure = "lact_group_x1"\n'
            '    formula = f"death ~ C({exposure}, Treatment(reference=1)) + age"\n',
            [],
            id="f-string-categories",
        ),
        pytest.param(
            '    formula = "death ~ C({}) + age".format("lact_group_x1")\n',
            [],
            id="format-categories",
        ),
        pytest.param(
            '    groups = ["lact_group_x1"]\n'
            '    formula = "death ~ age + " + " + ".join(\n'
            '        f"C({name}, Treatment(reference=1))" for name in groups\n'
            "    )\n",
            [],
            id="joined-comprehension-categories",
        ),
        pytest.param(
            '    formula = "death ~ {group} + age".format(group="lact_group_x1")\n',
            [_LINEAR],
            id="format-number",
        ),
        pytest.param(
            '    terms = ["age"]\n'
            '    terms.append("lact_group_x1")\n'
            '    formula = "death ~ " + " + ".join(terms)\n',
            [_LINEAR],
            id="a-list-grown-after-it-was-bound",
        ),
    ],
)
def test_a_formula_assembled_from_pieces_is_read_whole(
    assembled: str, refused: list[str]
) -> None:
    script = _HEAD + assembled + "    model = smf.logit(formula, data=df).fit()\n"

    assert _refusals(script) == refused


_SURVIVAL = """\
    import pandas as pd
    from lifelines import CoxPHFitter
    df = pd.read_parquet("cohort.parquet")
    table = df[["los_icu", "death", "age", "lact_group_x1"]]
    cph = CoxPHFitter()
    cph.fit(table, duration_col="los_icu", event_col="death"{extra})
"""


@pytest.mark.parametrize(
    ("extra", "refused"),
    [
        pytest.param("", [_LINEAR], id="a-covariate"),
        pytest.param(', strata=["lact_group_x1"]', [], id="strata"),
        pytest.param(', formula="lact_group_x1 + age"', [_LINEAR], id="formula"),
        pytest.param(', formula="C(lact_group_x1) + age"', [], id="formula-categories"),
    ],
)
def test_a_survival_model_names_a_nominal_group_as_strata_or_categories(
    extra: str, refused: list[str]
) -> None:
    assert _refusals(_SURVIVAL.format(extra=extra)) == refused


def test_a_survival_formula_held_in_a_name_is_read() -> None:
    script = """\
        import pandas as pd
        from lifelines import CoxPHFitter
        df = pd.read_parquet("cohort.parquet")
        terms = "lact_group_x1 + age"
        CoxPHFitter().fit(df, duration_col="los_icu", event_col="death", formula=terms)
    """

    assert _refusals(script) == [_LINEAR]


_MATRIX = """\
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet("cohort.parquet")
    {coding}
    X_train, X_test, y_train, y_test = train_test_split(X, df["death"])
    model = {model}
    model.fit(X_train, y_train)
"""
_LOGISTIC = "LogisticRegression(max_iter=500)"


@pytest.mark.parametrize(
    ("coding", "model", "refused"),
    [
        pytest.param(
            'X = df[["lact_group_x1", "age"]]', _LOGISTIC, [_LINEAR], id="codes"
        ),
        pytest.param(
            'X = pd.get_dummies(df[["lact_group_x1", "age"]], '
            'columns=["lact_group_x1"], drop_first=True)',
            _LOGISTIC,
            [],
            id="indicators",
        ),
        pytest.param(
            'X = pd.get_dummies(df[["lact_group_x1", "age"]], drop_first=True)',
            _LOGISTIC,
            [_LINEAR],
            id="table-keeps-integer-codes",
        ),
        pytest.param(
            'sex = pd.get_dummies(df["sex"], drop_first=True)\n'
            '    X = df[["lact_group_x1", "age"]].join(sex)',
            _LOGISTIC,
            [_LINEAR],
            id="another-column-coded",
        ),
        pytest.param(
            'X = df[["lact_group_x1", "age"]]',
            "RandomForestClassifier(random_state=1)",
            [],
            id="a-tree",
        ),
    ],
)
def test_a_linear_model_reads_a_nominal_group_only_as_indicators(
    coding: str, model: str, refused: list[str]
) -> None:
    assert _refusals(_MATRIX.format(coding=coding, model=model)) == refused


_PIPELINE = """\
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    df = pd.read_parquet("cohort.parquet")
    X = df[["lact_group_x1", "age"]]
    model = make_pipeline({first}, LogisticRegression(max_iter=500))
    model.fit(X, df["death"])
"""


@pytest.mark.parametrize(
    ("first", "refused"),
    [
        pytest.param("StandardScaler()", [_LINEAR], id="scaled-codes"),
        pytest.param(
            'ColumnTransformer([("groups", OneHotEncoder(drop="first"), '
            '["lact_group_x1"])], remainder="passthrough")',
            [],
            id="one-hot-coded",
        ),
    ],
)
def test_a_pipeline_that_ends_in_a_linear_model_codes_the_groups_first(
    first: str, refused: list[str]
) -> None:
    assert _refusals(_PIPELINE.format(first=first)) == refused


def test_a_cross_validated_linear_model_reads_its_design() -> None:
    script = """\
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact_group_x1", "age"]]
        scores = cross_val_score(LogisticRegression(max_iter=500), X, df["death"])
    """

    assert _refusals(script) == [_LINEAR]


def test_a_design_passed_through_a_call_is_read() -> None:
    script = """\
        import pandas as pd
        import statsmodels.api as sm
        df = pd.read_parquet("cohort.parquet")
        fit = sm.Logit(df["death"], sm.add_constant(df[["lact_group_x1", "age"]])).fit()
    """

    assert _refusals(script) == [_LINEAR]


@pytest.mark.parametrize(
    ("column", "refused"),
    [
        pytest.param("lact_group_x1", [_ORDERED], id="nominal"),
        pytest.param("lact_group_x2", [], id="ordinal"),
    ],
)
def test_an_ordered_stratified_analysis_reads_no_nominal_group(
    column: str, refused: list[str]
) -> None:
    step = AnalysisStep(
        step_id="04_outcomes_across_lactate_groups",
        intent="Summarize two outcomes across ordered exposure levels.",
        inputs=[column, "death", "los_icu"],
        expected_outputs=["table:outcomes_by_lactate_group", "test:ordinal_trend"],
        method="ordinal_stratified_descriptive_analysis",
    )

    assert _refusals(_HEAD, step=step) == refused


_PROPENSITY = """\
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet("cohort.parquet")
    X = df[["age", "los_icu"]]
    {fit}
"""


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(
            'LogisticRegression(max_iter=500).fit(X, df["lact_group_x1"])',
            id="fit-y",
        ),
        pytest.param(
            'LogisticRegression(max_iter=500).fit(X, y=df["lact_group_x1"])',
            id="fit-y-named",
        ),
        pytest.param(
            'sm.MNLogit(df["lact_group_x1"], sm.add_constant(X)).fit()', id="endog"
        ),
        pytest.param(
            'sm.MNLogit(endog=df["lact_group_x1"], exog=X).fit()', id="endog-named"
        ),
        pytest.param('smf.mnlogit("lact_group_x1 ~ age", df).fit()', id="formula-left"),
        pytest.param(
            'X_train, X_test, g_train, g_test = train_test_split(X, df["lact_group_x1"])\n'
            "    LogisticRegression(max_iter=500).fit(X_train, g_train)",
            id="split",
        ),
    ],
)
def test_a_model_that_predicts_the_groups_does_not_model_their_codes(
    fit: str,
) -> None:
    assert _refusals(_PROPENSITY.format(fit=fit)) == []


@pytest.mark.parametrize(
    ("design", "refused"),
    [
        pytest.param(
            'df.drop(columns=["lact_group_x1", "death"])', [], id="dropped-in-place"
        ),
        pytest.param("X", [], id="dropped-before"),
        pytest.param(
            'df[["lact_group_x1", "age"]].drop(columns=["age"])',
            [_LINEAR],
            id="another-column-dropped",
        ),
    ],
)
def test_a_table_that_drops_the_codes_does_not_model_them(
    design: str, refused: list[str]
) -> None:
    script = """\
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact_group_x1", "age"]].drop(columns=["lact_group_x1"])
        LogisticRegression(max_iter=500).fit({design}, df["death"])
    """.format(design=design)

    assert _refusals(script) == refused
