"""Cohort-hygiene flags are impartial and advisory.

They name a cohort property a reader should know about, stays without a
patient identifier beside an outcome or a share of short stays, and never
filter or block: they are warnings that reach callers through
``CohortAuditor.audit``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _hygiene_ctx(ra, df):
    return ra.build_research_context(
        research_question="Does sepsis predict ICU mortality?",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )


def test_cohort_hygiene_flags_missing_patient_id_when_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "los_icu": [2.0, 3.0, 5.0],
            "death": [0, 1, 0],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    pid = [
        f
        for f in findings
        if f.detail.get("subkind") == "patient_independence_unassessable"
    ]
    assert len(pid) == 1
    assert pid[0].severity == "warning"
    assert pid[0].detail["structural_no_source"] is True
    # Advice, not a mandate: it must not assert independence or demand a filter.
    assert "re-extract" in pid[0].message.lower()


def test_cohort_hygiene_no_patient_flag_with_patient_id(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "subject_id": [10, 10, 11],
            "stay_id": [1, 2, 3],
            "los_icu": [2.0, 3.0, 5.0],
            "death": [0, 1, 0],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable" for f in findings
    )


def test_cohort_hygiene_no_patient_flag_without_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({"stay_id": [1, 2, 3], "los_icu": [2.0, 3.0, 5.0]})
    ctx = ra.build_research_context(
        research_question="Describe LoS.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome=None,
    )
    findings = cohort_hygiene_findings(df, ctx)
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable" for f in findings
    )


def test_cohort_hygiene_short_stay_reported_not_enforced(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "los_icu": [0.2, 0.5, 3.0, 5.0],  # half are <1 day
            "death": [0, 1, 0, 1],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    short = [f for f in findings if f.detail.get("subkind") == "short_stay_exposure"]
    assert len(short) == 1
    assert short[0].severity == "warning"
    assert short[0].detail["fraction_los_under_1_day"] == 0.5
    assert "no minimum-los filter is imposed" in short[0].message.lower()


def test_cohort_hygiene_findings_never_block(ra):
    """Impartiality: hygiene flags are advisory and must never fail-close."""
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "los_icu": [0.1, 0.2, 5.0],
            "death": [0, 1, 0],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert findings  # both flags fire
    assert all(f.severity == "warning" for f in findings)
    assert all(f.detail.get("impartial") is True for f in findings)


def test_cohort_auditor_surfaces_hygiene_flags(ra, tmp_path: Path):
    """The hygiene flags reach callers through CohortAuditor.audit."""
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "los_icu": [0.2, 3.0, 5.0],
            "age": [60.0, 70.0, 80.0],
            "death": [0, 1, 0],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = _hygiene_ctx(ra, df)
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(f.detail.get("kind") == "cohort_hygiene" for f in findings)
