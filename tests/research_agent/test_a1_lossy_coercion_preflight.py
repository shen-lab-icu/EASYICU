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

import pandas as pd
import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import (
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


def test_literal_result_getattr_does_not_hide_fail_closed_loss_guard(ra):
    script = """
import pandas as pd

original = cohort["aki_stage_max"]
coerced = pd.to_numeric(original, errors="coerce")
newly_invalid = int((original.notna() & coerced.isna()).sum())
if newly_invalid > 0:
    raise ValueError("numeric coercion invalidated observed values")
iterations = getattr(model_result, "iterations", None)
"""

    assert _lossy_findings(script, ra) == []


@pytest.mark.parametrize(
    "lookup",
    [
        "value = getattr(model_result, attribute_name)",
        'replacement = getattr(holder, "int")',
    ],
    ids=["dynamic-name", "protected-int-name"],
)
def test_ambiguous_getattr_keeps_loss_guard_proof_fail_closed(ra, lookup):
    script = f"""
import pandas as pd

original = cohort["aki_stage_max"]
coerced = pd.to_numeric(original, errors="coerce")
newly_invalid = int((original.notna() & coerced.isna()).sum())
if newly_invalid > 0:
    raise ValueError("numeric coercion invalidated observed values")
{lookup}
"""

    lossy = _lossy_findings(script, ra)
    assert lossy


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


def test_t4_raise_guard_survives_unrelated_exception_type_logging(ra):
    script = _T4_GUARDED_BY_RAISE + """
try:
    publish_results()
except Exception as exc:
    diagnostic = {"error_type": type(exc).__name__}
"""

    assert _lossy_findings(script, ra) == []


def test_t4_dict_key_guard_passes(ra):
    assert _lossy_findings(_T4_GUARDED_VIA_DICT_KEY, ra) == []


def test_t4_host_helper_passes(ra):
    assert _lossy_findings(_T4_GUARDED_BY_HOST_HELPER, ra) == []


def test_lossy_coercion_guard_is_a_typed_deterministic_minimal_repair(ra):
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    repaired, names = deterministic_concept_audit_repair(
        _T2_E3_QUARANTINED_SHAPE,
        ["human-facing wording is intentionally irrelevant"],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(_T2_E3_QUARANTINED_SHAPE, ra),
    )

    assert names == ["lossy_numeric_coercion_guard_v1"]
    assert repaired.count("_easyicu_lossy_numeric_coercion_guard_v1") == 1
    assert repaired.count("record['newly_invalid_or_coerced_n']") == 1
    assert "return coerced, record" in repaired
    assert _lossy_findings(repaired, ra) == []


@pytest.mark.parametrize(
    ("key_source", "runtime_key"),
    [
        ("coercion_loss_n", "coercion_loss_n"),
        (r"coercion\nloss_n", "coercion\nloss_n"),
    ],
    ids=["renamed-key", "escaped-newline-key"],
)
def test_lossy_guard_uses_the_unique_structural_count_key(
    ra,
    key_source: str,
    runtime_key: str,
) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    renamed = _T2_E3_QUARANTINED_SHAPE.replace(
        "newly_invalid_or_coerced_n",
        key_source,
    )
    repaired, names = deterministic_concept_audit_repair(
        renamed,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(renamed, ra),
    )

    assert names == ["lossy_numeric_coercion_guard_v1"]
    assert f"record[{runtime_key!r}] > 0" in repaired
    assert _lossy_findings(repaired, ra) == []
    namespace: dict[str, object] = {"cohort": pd.DataFrame({"aki_stage_max": [0.0]})}
    exec(repaired, namespace)
    with pytest.raises(ValueError, match="numeric coercion invalidated"):
        namespace["numeric_coercion_audit"](
            pd.DataFrame({"stage": [0, "dirty"]}),
            "stage",
            "ordinal",
        )


def test_lossy_guard_refuses_two_structural_keys_in_one_record(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    ambiguous = """
import pandas as pd

def numeric_coercion_audit(frame, column):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {
        "first_loss_n": int((original.notna() & coerced.isna()).sum()),
        "second_loss_n": int((original.notna() & coerced.isna()).sum()),
    }
    return coerced, record
"""
    repaired, names = deterministic_concept_audit_repair(
        ambiguous,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(ambiguous, ra),
    )

    assert repaired == ambiguous
    assert names == []


def test_lossy_guard_refuses_renamed_count_key_overwrite(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    renamed = _T2_E3_QUARANTINED_SHAPE.replace(
        "newly_invalid_or_coerced_n",
        "coercion_loss_n",
    )
    overwritten = renamed.replace(
        "        ),\n    }",
        '        ),\n        "coercion_loss_n": 0,\n    }',
        1,
    )
    repaired, names = deterministic_concept_audit_repair(
        overwritten,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(overwritten, ra),
    )

    assert repaired == overwritten
    assert names == []


def test_lossy_guard_refuses_dynamic_count_key_overwrite(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    renamed = _T2_E3_QUARANTINED_SHAPE.replace(
        "newly_invalid_or_coerced_n",
        "coercion_loss_n",
    )
    overwritten = renamed.replace(
        "import pandas as pd",
        'import pandas as pd\n\ndynamic_key = "coercion_loss_n"',
        1,
    ).replace(
        "        ),\n    }",
        "        ),\n        dynamic_key: 0,\n    }",
        1,
    )
    repaired, names = deterministic_concept_audit_repair(
        overwritten,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(overwritten, ra),
    )

    assert repaired == overwritten
    assert names == []


def test_dynamic_key_overwrite_keeps_handwritten_guard_blocked(ra) -> None:
    unsafe_guard = """
import pandas as pd

dynamic_key = "coercion_loss_n"

def numeric_coercion_audit(frame, column):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {
        "coercion_loss_n": int((original.notna() & coerced.isna()).sum()),
        dynamic_key: 0,
    }
    if record["coercion_loss_n"] > 0:
        raise ValueError("numeric coercion invalidated observed values")
    return coerced, record
"""

    assert _lossy_findings(unsafe_guard, ra), (
        "a computed dict key can overwrite the literal loss count at runtime; "
        "the apparent guard must not make preflight pass"
    )


def test_lossy_guard_does_not_route_on_human_message_text() -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    repaired, names = deterministic_concept_audit_repair(
        _T2_E3_QUARANTINED_SHAPE,
        ["lossy_numeric_coercion"],
        repair_reasons=[],
    )
    assert repaired == _T2_E3_QUARANTINED_SHAPE
    assert names == []


def test_lossy_guard_refuses_ambiguous_multiple_audit_sites(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    ambiguous = (
        _T1_UNCHECKED_LOSS_COUNT
        + "\n"
        + _T1_UNCHECKED_LOSS_COUNT.replace(
            "numeric_coercion_audit", "second_numeric_coercion_audit"
        )
    )
    repaired, names = deterministic_concept_audit_repair(
        ambiguous,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(ambiguous, ra),
    )
    assert repaired == ambiguous
    assert names == []


@pytest.mark.parametrize(
    "replacement",
    [
        "(original.notna() & coerced.isna()).sum() / len(original)",
        "int((original.notna() & coerced.isna()).sum()) / len(original)",
    ],
    ids=["fraction", "integer-then-normalized"],
)
def test_lossy_guard_refuses_normalized_rates(ra, replacement: str) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    normalized = _T2_E3_QUARANTINED_SHAPE.replace(
        "int(\n            (original.notna() & coerced.isna()).sum()\n        )",
        replacement,
    )
    repaired, names = deterministic_concept_audit_repair(
        normalized,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(normalized, ra),
    )
    assert repaired == normalized
    assert names == []


@pytest.mark.parametrize(
    "dict_tail",
    [
        ', "newly_invalid_or_coerced_n": 0',
        ", **{'newly_invalid_or_coerced_n': 0}",
    ],
    ids=["duplicate-key", "mapping-overwrite"],
)
def test_lossy_guard_refuses_count_overwrite(ra, dict_tail: str) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    overwritten = _T2_E3_QUARANTINED_SHAPE.replace(
        "        ),\n    }",
        f"        ){dict_tail},\n    }}",
    )
    repaired, names = deterministic_concept_audit_repair(
        overwritten,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(overwritten, ra),
    )
    assert repaired == overwritten
    assert names == []


def test_lossy_guard_refuses_same_line_return(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    same_line = """
import pandas as pd

def numeric_coercion_audit(frame, column):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {"newly_invalid_or_coerced_n": int((original.notna() & coerced.isna()).sum())}; return coerced, record
"""
    repaired, names = deterministic_concept_audit_repair(
        same_line,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(same_line, ra),
    )
    assert repaired == same_line
    assert names == []


def test_lossy_guard_requires_exact_structured_finding(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    ordinal_finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="irrelevant",
        detail={"reason": "lossy_ordinal_rounding", "lines": [1]},
    )
    repaired, names = deterministic_concept_audit_repair(
        _T2_E3_QUARANTINED_SHAPE,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=[ordinal_finding],
    )
    assert repaired == _T2_E3_QUARANTINED_SHAPE
    assert names == []


def test_lossy_guard_raises_on_dirty_observed_value(ra) -> None:
    from easyicu.research_agent.repairs.source import (
        deterministic_concept_audit_repair,
    )

    repaired, names = deterministic_concept_audit_repair(
        _T2_E3_QUARANTINED_SHAPE,
        [],
        repair_reasons=[RepairReason.LOSSY_NUMERIC_COERCION],
        repair_findings=_lossy_findings(_T2_E3_QUARANTINED_SHAPE, ra),
    )
    namespace: dict[str, object] = {"cohort": pd.DataFrame({"aki_stage_max": [0.0]})}
    exec(repaired, namespace)
    with pytest.raises(ValueError, match="numeric coercion invalidated"):
        namespace["numeric_coercion_audit"](
            pd.DataFrame({"stage": [0, "dirty"]}),
            "stage",
            "ordinal",
        )
    assert names == ["lossy_numeric_coercion_guard_v1"]


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


def test_compound_or_guard_covers_consecutive_loss_bindings(ra):
    script = """
import pandas as pd

raw_measured = cohort["measured"]
raw_count = cohort["count"]
measured_num = pd.to_numeric(raw_measured, errors="coerce")
measured_loss = int((raw_measured.notna() & measured_num.isna()).sum())
count_num = pd.to_numeric(raw_count, errors="coerce")
count_loss = int((raw_count.notna() & count_num.isna()).sum())
if measured_loss > 0 or count_loss > 0:
    raise ValueError("provenance values were lost during numeric coercion")
"""

    assert _lossy_findings(script, ra) == []


def test_compound_guard_cannot_skip_scientific_work(ra):
    script = """
import pandas as pd

raw_measured = cohort["measured"]
raw_count = cohort["count"]
measured_num = pd.to_numeric(raw_measured, errors="coerce")
measured_loss = int((raw_measured.notna() & measured_num.isna()).sum())
model_result = fit_model(measured_num)
count_num = pd.to_numeric(raw_count, errors="coerce")
count_loss = int((raw_count.notna() & count_num.isna()).sum())
if measured_loss > 0 or count_loss > 0:
    raise ValueError("provenance values were lost during numeric coercion")
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
