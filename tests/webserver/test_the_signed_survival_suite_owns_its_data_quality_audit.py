"""The signed survival suite publishes the availability audit of its own columns.

On a prepared extract a covariate with medium missingness asks the article for
data-quality evidence.  The signed landmark survival suite already counted the
missing values of every sealed column for its receipt, but published no
data-quality product and its outline had no owner for that role, so the
sealed replan stopped after the Planner call.  The suite now declares the
audit as its own product, the outline names it, and the bound suite owns the
role.  A suite signed before the product keeps its digest and claims nothing.
Synthetic study and synthetic rows only (renal replacement therapy and 90-day
mortality).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    run_landmark_survival_suite,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    roles_covered_by_plan,
)
from tests.support.survival_sealed import bound_survival_plan, sealed_draft, sealed_survival

AUDIT = "table:landmark_measurement_audit"


def _unsigned_audit(authority):
    """The same suite as signed before the audit product existed."""

    body = authority.model_dump(mode="json", exclude={"execution_contract_sha256", "measurement_audit_product"})
    body["plan_outputs"] = [value for value in body["plan_outputs"] if value != AUDIT]
    return build_current_case_scientific_runtime_authority(body)


def _frame(n: int = 600, *, sex_missing: int = 37) -> pd.DataFrame:
    rng = np.random.default_rng(20261001)
    treated = rng.binomial(1, 0.4, size=n)
    onset = np.where(
        treated == 1,
        rng.choice([-2.0, 3.0, 9.0, 18.0, 30.0], size=n, p=[0.05, 0.4, 0.3, 0.2, 0.05]),
        np.nan,
    )
    died = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-1.2 + 0.7 * treated))), size=n)
    sex = rng.binomial(1, 0.5, size=n).astype(float)
    sex[rng.choice(n, size=sex_missing, replace=False)] = np.nan
    return pd.DataFrame({
        "rrt": treated.astype("int64"),
        "rrt_first_time": onset,
        "mort_90d": died.astype("int64"),
        "followup_days_90d": np.where(died == 1, rng.uniform(1.5, 89.0, size=n), 90.0),
        "age": rng.normal(64.0, 13.0, size=n),
        "sex": sex,
    })


def _run(authority, frame, out_dir):
    return run_landmark_survival_suite(
        frame=frame, authority=authority, runtime_projection_sha256="e" * 64, out_dir=out_dir,
        input_product="table:analysis_cohort", input_evidence_id="cohort", input_sha256="b" * 64,
    )


def test_the_suite_publishes_the_audit_of_the_columns_it_executes(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    assert authority.measurement_audit_product == AUDIT
    assert authority.plan_outputs.index(AUDIT) == authority.plan_outputs.index(authority.receipt_product) - 1

    frame = _frame()
    summary = _run(authority, frame, tmp_path / "out")

    assert set(summary["output_files"]) == set(authority.analysis_plan_outputs)
    audit = pd.read_csv(tmp_path / "out" / summary["output_files"][AUDIT])
    assert list(audit["column"]) == list(authority.required_columns)
    assert dict(zip(audit["column"], audit["column_role"])) == {
        "rrt": "exposure_status", "rrt_first_time": "exposure_onset", "mort_90d": "event",
        "followup_days_90d": "followup_time", "age": "adjustment", "sex": "adjustment",
    }
    by_column = audit.set_index("column")
    assert by_column.loc["sex", "source_missing_n"] == 37
    assert (by_column["source_n"] == len(frame)).all()
    assert (by_column["landmark_missing_n"] <= by_column["source_missing_n"]).all()
    receipt = json.loads((tmp_path / "out" / summary["output_files"][authority.receipt_product]).read_text())
    counted = receipt["missingness_measurement_audit"]
    assert counted["source_missing_n_by_column"] == dict(by_column["source_missing_n"].astype(int))
    assert set(by_column["landmark_population_n"]) == {counted["landmark_population_n"]}


def test_a_suite_signed_before_the_audit_keeps_its_digest_and_its_outputs(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    legacy = _unsigned_audit(authority)

    assert legacy.measurement_audit_product is None
    assert "measurement_audit_product" not in legacy.model_dump(mode="json")
    # A saved authority signed without the field still verifies on load.
    assert load_current_case_scientific_runtime_authority(legacy.model_dump(mode="json")) == legacy
    summary = _run(legacy, _frame(), tmp_path / "out")
    assert set(summary["output_files"]) == set(legacy.analysis_plan_outputs)
    assert not (tmp_path / "out" / "landmark_measurement_audit.csv").exists()


def test_an_unevenly_measured_covariate_is_answered_by_the_suites_own_audit(tmp_path):
    context, authority = sealed_survival(tmp_path, unevenly_measured="sex")
    assert "data_quality" in build_article_analysis_contract(context, analysis_type="survival").required_roles
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)

    draft = sealed_draft(context, authorities)
    assert any(step.measurement_audit_spec is not None for step in draft.steps)
    bound = bound_survival_plan(context, authorities)
    assert not any(step.measurement_audit_spec is not None for step in bound.steps)
    suite = next(step for step in bound.steps if step.method == "signed_landmark_survival_suite")
    assert AUDIT in suite.expected_outputs

    review = build_plan_scientific_review(
        context=context, plan=bound, require_reportable_capability=True, runtime_authority=authority,
    )
    assert "ARTICLE_CONTENT_ROLES_INCOMPLETE" not in {finding.code for finding in review.findings}
    assert review.approval_allowed is True


def test_a_suite_signed_without_the_audit_cannot_answer_for_data_quality(tmp_path):
    context, authority = sealed_survival(tmp_path, unevenly_measured="sex")
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=_unsigned_audit(authority))

    with pytest.raises(ProgressivePlanCompileError) as caught:
        sealed_draft(context, authorities)
    assert caught.value.reason_code == "progressive_outline_article_result_owner_missing"
    assert "data_quality" in str(caught.value)

    # Bound where nothing asked for it, that suite's step owns no audit role.
    contract = build_article_analysis_contract(context, analysis_type="survival")
    even, _authority = sealed_survival(tmp_path)
    legacy_plan = bound_survival_plan(even, authorities)
    signed_plan = bound_survival_plan(even, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    assert "data_quality" not in roles_covered_by_plan(legacy_plan, contract)
    assert "data_quality" in roles_covered_by_plan(signed_plan, contract)
