"""A1-1/A1-2/A1-3 — lossy numeric coercion is an AST-detected, typed failure.

Test matrix rows T1-T4 (proposal rev.3 §4):

T1  a script computes a coercion-loss count but never fails closed on it
    -> the mechanical preflight itself emits ``reason="lossy_numeric_coercion"``
    (no LLM prose, no ``issue_code=other`` guessing) and the typed classifier
    maps it to ``RepairReason.LOSSY_NUMERIC_COERCION``.
T2  the historical E3 Step-02 quarantined shape (loss count recorded into an
    audit dict + notna-gated domain check) is re-identified by the AST.
T3  a domain check that only covers post-coercion non-null values is flagged
    as a gap (dirty values silently leak into missingness).
T4  a script that fails closed (``if newly_invalid > 0: raise`` or the host
    helper ``strict_numeric_input``) passes the preflight.

Classification invariants: llm_concept_auditor findings keep their existing
routing (golden behavior preserved); the new precision comes from the AST
running BEFORE any LLM audit.
"""

from __future__ import annotations

from easyicu.research_agent.code_preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repair_reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.schema import ValidationFinding


def _step(ra):
    return ra.AnalysisStep(
        step_id="02_exposure_derivation_and_qc",
        intent="Derive the ordered exposure with strict numeric QC.",
        inputs=["aki_stage_max"],
        expected_outputs=["table:exposure_qc"],
        method="ordered_category_exposure_qc",
    )


def _lossy_findings(script: str, ra) -> list[ValidationFinding]:
    findings = audit_mechanical_code_contracts(script, _step(ra))
    return [
        finding
        for finding in findings
        if (finding.detail or {}).get("reason") == "lossy_numeric_coercion"
    ]


# ---------------------------------------------------------------------------
# T1 — unchecked coercion-loss count
# ---------------------------------------------------------------------------

_T1_UNCHECKED_LOSS_COUNT = """
import pandas as pd

def numeric_coercion_audit(frame, column):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {
        "n_rows": int(len(frame)),
        "newly_invalid_or_coerced_n": int(
            (original.notna() & coerced.isna()).sum()
        ),
    }
    return coerced, record

coerced, audit = numeric_coercion_audit(cohort, "aki_stage_max")
"""


def test_t1_unchecked_loss_count_is_flagged_by_ast(ra):
    lossy = _lossy_findings(_T1_UNCHECKED_LOSS_COUNT, ra)
    assert lossy, "AST preflight must flag the unchecked coercion-loss count"
    finding = lossy[0]
    assert finding.validator == "mechanical_code_preflight"
    assert finding.severity == "error"
    gaps = {
        str(issue.get("gap")) for issue in (finding.detail or {}).get("issues", [])
    } or {str((finding.detail or {}).get("gap"))}
    assert "unchecked_coercion_loss_count" in gaps


def test_t1_ast_finding_classifies_as_lossy_numeric_coercion(ra):
    finding = _lossy_findings(_T1_UNCHECKED_LOSS_COUNT, ra)[0]
    assert repair_reason_for_finding(finding) is RepairReason.LOSSY_NUMERIC_COERCION


# ---------------------------------------------------------------------------
# T2 — the historical E3 Step-02 quarantined shape is re-identified
# (structural excerpt of the real quarantined analysis.py: loss count stored
# in an audit record + notna-gated domain validation, no loss fail-close)
# ---------------------------------------------------------------------------

_T2_E3_QUARANTINED_SHAPE = """
import numpy as np
import pandas as pd

def numeric_coercion_audit(frame, column, numeric_kind):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {
        "variable": column,
        "numeric_kind": numeric_kind,
        "original_missing_n": int(original.isna().sum()),
        "post_coercion_missing_n": int(coerced.isna().sum()),
        "newly_invalid_or_coerced_n": int(
            (original.notna() & coerced.isna()).sum()
        ),
    }
    return coerced, record

stage_numeric, coercion_record = numeric_coercion_audit(
    cohort, "aki_stage_max", "ordinal"
)
stage_domain_valid = (
    stage_numeric.notna()
    & (np.floor(stage_numeric) == stage_numeric)
    & stage_numeric.isin([0, 1, 2, 3])
)
stage_invalid_n = int((stage_numeric.notna() & ~stage_domain_valid).sum())
if stage_invalid_n > 0:
    raise ValueError("aki_stage_max contains out-of-domain non-missing values.")
"""


def test_t2_quarantined_shape_reidentified_as_lossy_numeric_coercion(ra):
    lossy = _lossy_findings(_T2_E3_QUARANTINED_SHAPE, ra)
    assert lossy, (
        "the historical quarantined shape must be re-identified by the AST, "
        "not left to issue_code=other prose guessing"
    )
    assert repair_reason_for_finding(lossy[0]) is RepairReason.LOSSY_NUMERIC_COERCION


# ---------------------------------------------------------------------------
# T3 — domain validation gated on notna() only
# ---------------------------------------------------------------------------

_T3_NOTNA_GATED_DOMAIN_ONLY = """
import pandas as pd

stage_numeric = pd.to_numeric(cohort["aki_stage_max"], errors="coerce")
stage_domain_valid = stage_numeric.notna() & stage_numeric.isin([0, 1, 2, 3])
stage_invalid_n = int((stage_numeric.notna() & ~stage_domain_valid).sum())
if stage_invalid_n > 0:
    raise ValueError("out-of-domain stage values")
"""


def test_t3_notna_gated_domain_check_is_flagged(ra):
    lossy = _lossy_findings(_T3_NOTNA_GATED_DOMAIN_ONLY, ra)
    assert lossy, (
        "a domain check that only sees post-coercion non-null values lets "
        "coerced dirty values leak into missingness and must be flagged"
    )
    gaps = set()
    for finding in lossy:
        detail = finding.detail or {}
        for issue in detail.get("issues", []):
            gaps.add(str(issue.get("gap")))
        if detail.get("gap"):
            gaps.add(str(detail.get("gap")))
    assert "domain_check_gated_on_notna" in gaps


# ---------------------------------------------------------------------------
# T4 — fail-closed scripts pass
# ---------------------------------------------------------------------------

_T4_GUARDED_BY_RAISE = """
import pandas as pd

original = cohort["aki_stage_max"]
coerced = pd.to_numeric(original, errors="coerce")
newly_invalid = int((original.notna() & coerced.isna()).sum())
if newly_invalid > 0:
    raise ValueError("aki_stage_max lost values during numeric coercion")
stage_domain_valid = coerced.notna() & coerced.isin([0, 1, 2, 3])
stage_invalid_n = int((coerced.notna() & ~stage_domain_valid).sum())
if stage_invalid_n > 0:
    raise ValueError("out-of-domain stage values")
"""

_T4_GUARDED_VIA_DICT_KEY = """
import pandas as pd

def numeric_coercion_audit(frame, column):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (original.notna() & coerced.isna()).sum()
        ),
    }
    return coerced, record

coerced, audit = numeric_coercion_audit(cohort, "aki_stage_max")
if audit["newly_invalid_or_coerced_n"] > 0:
    raise ValueError("numeric coercion dropped observed values")
"""

_T4_GUARDED_BY_HOST_HELPER = """
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input

coerced = strict_numeric_input(cohort["aki_stage_max"]).values
stage_domain_valid = coerced.notna() & coerced.isin([0, 1, 2, 3])
"""


def test_t4_raise_guard_passes(ra):
    assert _lossy_findings(_T4_GUARDED_BY_RAISE, ra) == []


def test_t4_dict_key_guard_passes(ra):
    assert _lossy_findings(_T4_GUARDED_VIA_DICT_KEY, ra) == []


def test_t4_host_helper_passes(ra):
    assert _lossy_findings(_T4_GUARDED_BY_HOST_HELPER, ra) == []


def test_t4_assert_guard_passes(ra):
    script = """
import pandas as pd

original = cohort["aki_stage_max"]
coerced = pd.to_numeric(original, errors="coerce")
newly_invalid = int((original.notna() & coerced.isna()).sum())
assert newly_invalid == 0, "coercion dropped observed values"
"""
    assert _lossy_findings(script, ra) == []


# ---------------------------------------------------------------------------
# repair symmetry (T9 companion): removing the guard re-flags the script,
# so a "repair" that deletes the guard cannot pass the gate
# ---------------------------------------------------------------------------


def test_removing_guard_reflags_script(ra):
    assert _lossy_findings(_T4_GUARDED_BY_RAISE, ra) == []
    unguarded = _T4_GUARDED_BY_RAISE.replace(
        "if newly_invalid > 0:\n    raise ValueError("
        '"aki_stage_max lost values during numeric coercion")\n',
        "",
    )
    assert _lossy_findings(unguarded, ra), (
        "deleting the fail-close guard must re-flag the script; repair means "
        "adding the guard, never relaxing the gate"
    )


# ---------------------------------------------------------------------------
# false-positive guards: scripts without a coercion source stay silent
# ---------------------------------------------------------------------------


def test_no_coercion_source_no_finding(ra):
    script = """
import pandas as pd

values = cohort["aki_stage_max"]
newly_invalid = int((values.notna() & values.shift().isna()).sum())
domain_valid = values.notna() & values.isin([0, 1, 2, 3])
"""
    assert _lossy_findings(script, ra) == []


def test_strict_to_numeric_without_coerce_no_finding(ra):
    script = """
import pandas as pd

values = pd.to_numeric(cohort["aki_stage_max"], errors="raise")
domain_valid = values.notna() & values.isin([0, 1, 2, 3])
"""
    assert _lossy_findings(script, ra) == []


def test_unrelated_strict_helper_does_not_hide_unguarded_coercion(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input
import pandas as pd

height = strict_numeric_input(cohort["height"]).values
original = cohort["aki_stage_max"]
coerced = pd.to_numeric(original, errors="coerce")
newly_invalid = int((original.notna() & coerced.isna()).sum())
"""
    assert _lossy_findings(script, ra)


def test_wrong_direction_guard_does_not_pass(ra):
    script = _T4_GUARDED_BY_RAISE.replace(
        "if newly_invalid > 0:", "if newly_invalid == 0:"
    )
    assert _lossy_findings(script, ra)


def test_conditional_nested_raise_does_not_pass(ra):
    script = _T4_GUARDED_BY_RAISE.replace(
        '    raise ValueError("aki_stage_max lost values during numeric coercion")',
        '    if debug:\n        raise ValueError("debug-only raise")',
    )
    assert _lossy_findings(script, ra)


def test_guard_for_one_coercion_does_not_hide_another(ra):
    script = _T4_GUARDED_BY_RAISE + """
raw_second = cohort["lactate"]
coerced_second = pd.to_numeric(raw_second, errors="coerce")
second_loss = int((raw_second.notna() & coerced_second.isna()).sum())
"""
    assert _lossy_findings(script, ra)


def test_unrelated_domain_check_is_not_claimed_by_other_coercion(ra):
    script = """
import pandas as pd

height = pd.to_numeric(cohort["height"], errors="coerce")
stage = cohort["aki_stage_max"]
stage_valid = stage.notna() & stage.isin([0, 1, 2, 3])
"""
    assert _lossy_findings(script, ra) == []


# ---------------------------------------------------------------------------
# classification invariants (A1-2): typed routing, no behavior drift
# ---------------------------------------------------------------------------


def test_detail_reason_maps_to_lossy_numeric_coercion():
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="coercion loss is computed but never fails closed",
        detail={"reason": "lossy_numeric_coercion"},
    )
    assert repair_reason_for_finding(finding) is RepairReason.LOSSY_NUMERIC_COERCION


def test_llm_auditor_known_issue_codes_keep_existing_routing():
    for issue_code in (
        "audit_only_companion_row_gating_required",
        "finalized_exposure_missing_reconciliation",
        "finalized_exposure_overridden",
        "finalized_exposure_forced_raw_reconciliation",
        "plausibility_range_exclusion_required",
        "other",
    ):
        finding = ValidationFinding(
            validator="llm_concept_auditor",
            severity="error",
            message="semantic finding",
            detail={"issue_code": issue_code},
        )
        assert (
            repair_reason_for_finding(finding)
            is RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
        )


def test_llm_auditor_structured_ast_reason_takes_precedence():
    # If the auditor echoes the AST's structured reason, the precise typed
    # route wins over the validator-level fallback.
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="prose about coercion",
        detail={"issue_code": "other", "reason": "lossy_numeric_coercion"},
    )
    assert repair_reason_for_finding(finding) is RepairReason.LOSSY_NUMERIC_COERCION
