"""Single-imputation preflight: ad-hoc fills must declare a strategy."""

from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.schema import AnalysisStep

_STEP = AnalysisStep(
    step_id="primary_model",
    intent="Fit the primary adjusted model.",
    inputs=["exposure", "death", "age"],
    expected_outputs=["table:adjusted_association_estimates"],
    method="adjusted_association_models",
)


def _reasons(script: str) -> list[str]:
    return [
        str((finding.detail or {}).get("reason") or "")
        for finding in audit_mechanical_code_contracts(script, _STEP)
        if finding.validator == "mechanical_code_preflight"
    ]


def test_fillna_method_is_rejected() -> None:
    reasons = _reasons("def f(df):\n    return df.fillna(method='ffill')\n")
    assert "single_imputation_fillna_method" in reasons


def test_ffill_bfill_interpolate_are_rejected() -> None:
    reasons = _reasons(
        "def f(df):\n"
        "    a = df['x'].ffill()\n"
        "    b = df['y'].bfill()\n"
        "    c = df['z'].interpolate()\n"
        "    return a, b, c\n"
    )
    assert "single_imputation_forward_backward_fill" in reasons
    assert "single_imputation_interpolation" in reasons


def test_aggregate_fill_is_rejected() -> None:
    reasons = _reasons(
        "def f(df):\n"
        "    df['x'] = df['x'].fillna(df['x'].median())\n"
        "    df['y'] = df['y'].fillna(df['y'].mean())\n"
        "    return df\n"
    )
    assert reasons.count("single_imputation_aggregate_fill") == 2


def test_bare_simple_imputer_is_rejected() -> None:
    reasons = _reasons(
        "def f(X):\n"
        "    from sklearn.impute import SimpleImputer\n"
        "    return SimpleImputer(strategy='median').fit_transform(X)\n"
    )
    assert "single_imputation_bare_estimator" in reasons


def test_pipeline_wrapped_imputer_stays_allowed() -> None:
    reasons = _reasons(
        "def f(X):\n"
        "    from sklearn.pipeline import Pipeline\n"
        "    from sklearn.impute import SimpleImputer\n"
        "    from sklearn.preprocessing import StandardScaler\n"
        "    pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])\n"
        "    return pipe.fit_transform(X)\n"
    )
    assert "single_imputation_bare_estimator" not in reasons


def test_pipeline_bound_imputer_alias_stays_allowed() -> None:
    reasons = _reasons(
        "def f(X):\n"
        "    from sklearn.pipeline import Pipeline\n"
        "    from sklearn.impute import SimpleImputer\n"
        "    imputer = SimpleImputer(strategy='median')\n"
        "    pipe = Pipeline([('imputer', imputer)])\n"
        "    return pipe.fit_transform(X)\n"
    )
    assert "single_imputation_bare_estimator" not in reasons


def test_reassigned_imputer_alias_is_not_treated_as_pipeline_wrapped() -> None:
    reasons = _reasons(
        "def f(X):\n"
        "    from sklearn.pipeline import Pipeline\n"
        "    from sklearn.impute import SimpleImputer\n"
        "    imputer = SimpleImputer(strategy='median')\n"
        "    imputer = object()\n"
        "    pipe = Pipeline([('imputer', imputer)])\n"
        "    return pipe.fit_transform(X)\n"
    )
    assert "single_imputation_bare_estimator" in reasons


def test_pipeline_alias_does_not_hide_bare_imputation_call() -> None:
    reasons = _reasons(
        "def f(X):\n"
        "    from sklearn.pipeline import Pipeline\n"
        "    from sklearn.impute import SimpleImputer\n"
        "    imputer = SimpleImputer(strategy='median')\n"
        "    filled = imputer.fit_transform(X)\n"
        "    pipe = Pipeline([('imputer', imputer)])\n"
        "    return filled, pipe\n"
    )
    assert "single_imputation_bare_estimator" in reasons


def test_scalar_fill_and_bare_aggregates_stay_allowed() -> None:
    reasons = _reasons(
        "def f(df):\n"
        "    df['flag'] = df['flag'].fillna(0)\n"
        "    med = df['x'].median()\n"
        "    return df, med\n"
    )
    assert not [reason for reason in reasons if reason.startswith("single_imputation")]
