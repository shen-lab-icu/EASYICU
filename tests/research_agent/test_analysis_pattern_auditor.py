"""Tests for the analysis-pattern auditor (generic ICU footguns)."""

from __future__ import annotations

import textwrap

import pytest


def _ctx(ra, roles=None):
    """Build a minimal ResearchContext with typed variables."""
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    default_roles = {
        "stay_id": VariableRole.ID,
        "sofa2": VariableRole.COMPOSITE_SCORE,
        "gcs": VariableRole.ORDINAL_SCORE,
        "lact": VariableRole.LAB,
        "age": VariableRole.DEMOGRAPHIC,
        "sex_M": VariableRole.DEMOGRAPHIC,
        "death": VariableRole.OUTCOME,
        "los_icu": VariableRole.TIME,
    }
    if roles:
        default_roles.update(roles)
    return ResearchContext(
        research_question="test",
        cohort=CohortDescriptor(
            cohort_name="test", database="miiv", n_stays=800, n_patients=800,
        ),
        variables=[
            ConceptDescriptor(name=name, role=role, dtype="float64")
            for name, role in default_roles.items()
        ],
        target_outcome="death",
    )


# ---------------------------------------------------------------------------
# Clustering / distance
# ---------------------------------------------------------------------------


def test_kmeans_on_ordinal_is_error(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        import pandas as pd
        from sklearn.cluster import KMeans
        df = pd.read_parquet("cohort.parquet")
        X = df[["sofa2", "lact", "age"]]
        km = KMeans(n_clusters=3)
        df["cluster"] = km.fit_predict(X)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    errors = [f for f in findings if f.severity == "error"]
    assert any("ordinal" in f.message.lower() or "distance" in f.message.lower() for f in errors)


def test_kmeans_without_scaler_warns(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.cluster import KMeans
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact", "age"]]
        km = KMeans(n_clusters=3)
        km.fit(X)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    warnings = [f for f in findings if f.severity == "warning"]
    assert any("scaler" in f.message.lower() or "scale" in f.message.lower() for f in warnings)


def test_kmeans_with_scaler_no_scale_warning(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact", "age"]]
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(X_scaled)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    scale_warnings = [f for f in findings if "scaler" in f.message.lower() or "scale" in f.message.lower()]
    assert not scale_warnings


# ---------------------------------------------------------------------------
# Outcome leakage
# ---------------------------------------------------------------------------


def test_outcome_in_feature_matrix_is_error(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["sofa2", "lact", "death"]]
        y = df["death"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        LogisticRegression(random_state=0).fit(X_train, y_train)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    errors = [f for f in findings if f.severity == "error"]
    assert any("outcome" in f.message.lower() for f in errors)


# ---------------------------------------------------------------------------
# ID / time leakage
# ---------------------------------------------------------------------------


def test_id_column_in_features_warns(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["stay_id", "sofa2", "lact"]]
        y = df["death"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        RandomForestClassifier(random_state=0).fit(X_train, y_train)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    warnings = [f for f in findings if f.severity == "warning"]
    assert any("id" in f.message.lower() or "identity" in f.message.lower() for f in warnings)


# ---------------------------------------------------------------------------
# No train/test split
# ---------------------------------------------------------------------------


def test_supervised_without_split_warns(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.linear_model import LogisticRegression
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact", "age"]]
        y = df["death"]
        LogisticRegression(random_state=0).fit(X, y)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    warnings = [f for f in findings if f.severity == "warning"]
    assert any("split" in f.message.lower() or "in-sample" in f.message.lower() for f in warnings)


# ---------------------------------------------------------------------------
# PCA without scaler
# ---------------------------------------------------------------------------


def test_pca_without_scaler_warns(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.decomposition import PCA
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact", "age", "sofa2"]]
        pca = PCA(n_components=2)
        pca.fit(X)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    warnings = [f for f in findings if f.severity == "warning"]
    assert any("pca" in f.message.lower() or "scaler" in f.message.lower() for f in warnings)


# ---------------------------------------------------------------------------
# Random state nudge
# ---------------------------------------------------------------------------


def test_missing_random_state_is_info(ra):
    auditor = ra.AnalysisPatternAuditor()
    code = textwrap.dedent("""\
        from sklearn.cluster import KMeans
        import pandas as pd
        df = pd.read_parquet("cohort.parquet")
        X = df[["lact", "age"]]
        km = KMeans(n_clusters=3)
        km.fit(X)
    """)
    findings = auditor.audit(context=_ctx(ra), script_text=code)
    infos = [f for f in findings if f.severity == "info"]
    assert any("random_state" in f.message for f in infos)


# ---------------------------------------------------------------------------
# Pipeline integration: verify auditor runs in the step loop
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_pattern_auditor_fires_on_clustering_skill(
    ra, synthetic_cohort, tmp_path
):
    """The mock pipeline's association-analysis skill generates code that
    references sofa2 in a feature context. Verify the pattern auditor
    is wired and doesn't crash the pipeline."""
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    # Pipeline should complete without error.
    assert result.evidence_count > 0
    assert result.findings_count > 0
