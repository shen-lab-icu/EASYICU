"""Scope and straight-line dominance regressions for coercion-loss guards."""

from __future__ import annotations

from easyicu.research_agent.code_preflight import audit_mechanical_code_contracts


def _step(ra):
    return ra.AnalysisStep(
        step_id="numeric_exposure_qc",
        intent="Validate an already selected numeric exposure.",
        inputs=["declared_exposure"],
        expected_outputs=["table:exposure_qc"],
        method="ordered_exposure_quality_control",
    )


def _lossy_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "lossy_numeric_coercion"
    ]


def test_immediate_scalar_raise_and_assert_remain_valid(ra):
    raised = """
import pandas as pd
raw = cohort["declared_exposure"]
coerced = pd.to_numeric(raw, errors="coerce")
loss_n = int((raw.notna() & coerced.isna()).sum())
if loss_n > 0:
    raise ValueError("numeric coercion lost observed values")
"""
    asserted = raised.replace(
        'if loss_n > 0:\n    raise ValueError("numeric coercion lost observed values")',
        'assert loss_n == 0, "numeric coercion lost observed values"',
    )

    assert _lossy_findings(raised, ra) == []
    assert _lossy_findings(asserted, ra) == []


def test_direct_tuple_receipt_guard_remains_valid(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        )
    }
    return coerced, record

coerced, audit = audit_numeric(cohort["declared_exposure"])
if audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("numeric coercion lost observed values")
"""

    assert _lossy_findings(script, ra) == []


def test_guard_after_same_line_return_does_not_dominate(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {"newly_invalid_or_coerced_n": int((raw.notna() & coerced.isna()).sum())}; return coerced, record
    if int(record["newly_invalid_or_coerced_n"]) > 0:
        raise ValueError("unreachable guard")

coerced, audit = audit_numeric(cohort["declared_exposure"])
"""

    assert _lossy_findings(script, ra)


def test_guard_in_one_function_does_not_cover_same_names_in_another(ra):
    script = """
import pandas as pd

def guarded(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    loss_n = int((raw.notna() & coerced.isna()).sum())
    if loss_n > 0:
        raise ValueError("numeric coercion lost observed values")
    return coerced

def unguarded(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    loss_n = int((raw.notna() & coerced.isna()).sum())
    return coerced
"""

    assert _lossy_findings(script, ra)


def test_every_direct_receipt_call_must_be_immediately_guarded(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        )
    }
    return coerced, record

first, first_audit = audit_numeric(cohort["first"])
if first_audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("first coercion lost observed values")
second, second_audit = audit_numeric(cohort["second"])
"""

    assert _lossy_findings(script, ra)


def test_guard_swallowed_by_try_handler_does_not_fail_closed(ra):
    script = """
import pandas as pd

try:
    raw = cohort["declared_exposure"]
    coerced = pd.to_numeric(raw, errors="coerce")
    loss_n = int((raw.notna() & coerced.isna()).sum())
    if loss_n > 0:
        raise ValueError("numeric coercion lost observed values")
except ValueError:
    pass
"""

    assert _lossy_findings(script, ra)


def test_integer_cast_of_fractional_loss_rate_is_not_a_guard(ra):
    script = """
import pandas as pd
raw = cohort["declared_exposure"]
coerced = pd.to_numeric(raw, errors="coerce")
loss_rate = (raw.notna() & coerced.isna()).sum() / len(raw)
if int(loss_rate) > 0:
    raise ValueError("one dirty row in a larger cohort is truncated to zero")
"""

    assert _lossy_findings(script, ra)


def test_overwritable_dict_count_is_not_a_guard(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        ),
        **{"newly_invalid_or_coerced_n": 0},
    }
    return coerced, record

coerced, audit = audit_numeric(cohort["declared_exposure"])
if audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("numeric coercion lost observed values")
"""

    assert _lossy_findings(script, ra)


def test_duplicate_dict_count_is_not_a_guard(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        ),
        "newly_invalid_or_coerced_n": 0,
    }
    return coerced, record

coerced, audit = audit_numeric(cohort["declared_exposure"])
if audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("numeric coercion lost observed values")
"""

    assert _lossy_findings(script, ra)


def test_transformed_loss_value_is_not_an_exact_count_guard(ra):
    script = """
import pandas as pd
raw = cohort["declared_exposure"]
coerced = pd.to_numeric(raw, errors="coerce")
loss_n = (raw.notna() & coerced.isna()).sum() / len(raw)
if loss_n > 0:
    raise ValueError("fraction is not the host-standard integer count")
"""

    assert _lossy_findings(script, ra)


def test_branch_local_guard_does_not_dominate_outer_coercion(ra):
    script = """
import pandas as pd
raw = cohort["declared_exposure"]
coerced = pd.to_numeric(raw, errors="coerce")
if strict:
    loss_n = int((raw.notna() & coerced.isna()).sum())
    if loss_n > 0:
        raise ValueError("guard is skipped when strict is false")
"""

    assert _lossy_findings(script, ra)


def test_rebound_receipt_helper_does_not_prove_original_guard(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        )
    }
    return coerced, record

audit_numeric = lambda raw: (
    raw,
    {"newly_invalid_or_coerced_n": 0},
)
coerced, audit = audit_numeric(cohort["declared_exposure"])
if audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("numeric coercion lost observed values")
"""

    assert _lossy_findings(script, ra)


def test_decorated_receipt_helper_is_not_assumed_to_keep_identity(ra):
    script = """
import pandas as pd

@replace_with_fake_auditor
def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        )
    }
    return coerced, record

coerced, audit = audit_numeric(cohort["declared_exposure"])
if audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("numeric coercion lost observed values")
"""

    assert _lossy_findings(script, ra)


def test_wrapper_receipt_guard_cannot_hide_outer_exception_swallow(ra):
    script = """
import pandas as pd

def audit_numeric(raw):
    coerced = pd.to_numeric(raw, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (raw.notna() & coerced.isna()).sum()
        )
    }
    return coerced, record

def wrapper(raw):
    coerced, audit = audit_numeric(raw)
    if audit["newly_invalid_or_coerced_n"] > 0:
        raise ValueError("numeric coercion lost observed values")
    return coerced

try:
    wrapper(cohort["declared_exposure"])
except ValueError:
    pass
"""

    assert _lossy_findings(script, ra)
