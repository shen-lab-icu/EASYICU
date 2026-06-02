#!/usr/bin/env python
"""Demonstrate the analysis-pattern auditor on free-form clustering code.

This is a closed-form demo: we hand-craft three coder outputs that a
real LLM might emit for a clustering / prediction / survival question,
and show what the AnalysisPatternAuditor returns. The auditor decides
purely from the variable role table — no skill, no question parsing.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from easyicu.research_agent import AnalysisPatternAuditor
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    ctx = ResearchContext(
        research_question="Cluster ICU patients by SOFA-2 + lactate + MAP + vasopressor.",
        cohort=CohortDescriptor(
            cohort_name="demo", database="miiv", n_stays=600, n_patients=600,
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(name="sex_M", role=VariableRole.DEMOGRAPHIC, dtype="int8"),
            ConceptDescriptor(name="sofa2", role=VariableRole.COMPOSITE_SCORE, dtype="int64"),
            ConceptDescriptor(name="gcs", role=VariableRole.ORDINAL_SCORE, dtype="int64"),
            ConceptDescriptor(name="lact", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(name="creat", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(name="map", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(name="vaso", role=VariableRole.INTERVENTION, dtype="int8"),
            ConceptDescriptor(name="los_icu", role=VariableRole.TIME, dtype="float64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int8"),
        ],
        target_outcome="death",
    )

    auditor = AnalysisPatternAuditor()

    cases = [
        (
            "Case 1 — KMeans on ordinal SOFA-2 (the trap)",
            textwrap.dedent("""\
                import pandas as pd
                from sklearn.cluster import KMeans

                df = pd.read_parquet('cohort.parquet')
                X = df[['sofa2', 'lact', 'creat', 'map', 'vaso']]
                km = KMeans(n_clusters=3)
                df['cluster'] = km.fit_predict(X)
            """),
        ),
        (
            "Case 2 — Same KMeans but only continuous labs (still missing scaler)",
            textwrap.dedent("""\
                import pandas as pd
                from sklearn.cluster import KMeans

                df = pd.read_parquet('cohort.parquet')
                X = df[['lact', 'creat', 'map']]
                km = KMeans(n_clusters=3)
                df['cluster'] = km.fit_predict(X)
            """),
        ),
        (
            "Case 3 — KMeans + StandardScaler on labs (the right way)",
            textwrap.dedent("""\
                import pandas as pd
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                df = pd.read_parquet('cohort.parquet')
                X = df[['lact', 'creat', 'map']]
                X_scaled = StandardScaler().fit_transform(X)
                km = KMeans(n_clusters=3, random_state=2026)
                df['cluster'] = km.fit_predict(X_scaled)
            """),
        ),
        (
            "Case 4 — Logistic regression with outcome leaked into X",
            textwrap.dedent("""\
                import pandas as pd
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import train_test_split

                df = pd.read_parquet('cohort.parquet')
                X = df[['sofa2', 'lact', 'death']]   # leak!
                y = df['death']
                Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
                LogisticRegression(random_state=0).fit(Xtr, ytr)
            """),
        ),
        (
            "Case 5 — Cox model with sensible duration / event columns",
            textwrap.dedent("""\
                import pandas as pd
                from lifelines import CoxPHFitter

                df = pd.read_parquet('cohort.parquet')
                cph = CoxPHFitter()
                cph.fit(df, duration_col='los_icu', event_col='death')
            """),
        ),
    ]

    for title, code in cases:
        findings = auditor.audit(context=ctx, script_text=code)
        print(f"\n=== {title} ===")
        if not findings:
            print("  (no findings)")
            continue
        for f in findings:
            print(f"  [{f.severity}] {f.message[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
