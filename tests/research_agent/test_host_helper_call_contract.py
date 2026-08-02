"""Host-owned helper signatures are checked and repaired before sandbox launch."""

from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import repair_reason_for_finding
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="measurement_qc",
        intent="Audit declared measurement provenance.",
        inputs=["value_measured", "value_n"],
        expected_outputs=["table:measurement_qc"],
        method="measurement_quality_control",
    )


def _table_step(ra):
    return ra.AnalysisStep(
        step_id="table_one",
        intent="Produce the grouped baseline table.",
        inputs=["death", "age", "sex"],
        expected_outputs=["table:table_one"],
        method="table_one",
        table_one_spec={
            "group_by": "death",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                },
                {
                    "name": "sex",
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                    "levels": ["Female", "Male"],
                },
            ],
        },
    )


def _figure_step(ra):
    return ra.AnalysisStep(
        step_id="measurement_qc_figure",
        intent="Render the registered measurement audit products.",
        inputs=[
            "table:missingness_measurement_audit",
            "table:measurement_process_audit",
        ],
        expected_outputs=["figure:measurement_qc"],
        method="visualization",
    )


def _signature_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "host_helper_call_signature_invalid"
    ]


def _provenance_scope_findings(script: str, step):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, step)
        if (finding.detail or {}).get("reason")
        == "measurement_provenance_pair_undeclared"
    ]


def _unpack_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "local_helper_unpack_arity_mismatch"
    ]


def _level_index_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "closed_counts_table_index_used_as_levels"
    ]


def _count_domain_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "count_companion_closed_domain_invalid"
    ]


def test_declared_count_companion_cannot_use_closed_binary_domain(ra):
    script = """
def allowed_values_for(name):
    return raw_contracts[name].get("allowed_values")

count_levels = allowed_values_for("value_n")
require_binary(count, "value_n", count_levels)
counts = closed_categorical_counts(count, declared_levels=count_levels)
"""

    findings = _count_domain_findings(script, ra)

    assert [
        (
            finding.detail["helper_name"],
            finding.detail["line"],
            finding.detail["column"],
        )
        for finding in findings
    ] == [
        ("allowed_values_for", 5, "value_n"),
        ("require_binary", 6, "value_n"),
        ("closed_categorical_counts", 7, "value_n"),
    ]
    assert all(
        repair_reason_for_finding(finding).value
        == "SCIENTIFIC_SEMANTICS_VIOLATION"
        for finding in findings
    )


def test_positional_keyword_only_host_arguments_fail_before_execution(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(frame, measured_column, count_column)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail == {
        "reason": "host_helper_call_signature_invalid",
        "helper_name": "measurement_provenance_receipt",
        "line": 3,
        "max_positional": 1,
        "required_keywords": ["measured_column", "count_column"],
        "violations": [
            "keyword_only_parameters_passed_positionally",
            "required_keyword_only_argument_missing",
        ],
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_alias_import_is_bound_to_the_same_host_contract(ra):
    script = """
import easyicu.research_agent.methods.descriptive_inputs as host_inputs
receipt = host_inputs.measurement_provenance_receipt(frame, measured, count)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["helper_name"] == "measurement_provenance_receipt"


def test_function_local_host_import_is_bound_to_the_same_contract(ra):
    script = """
def audit(frame, measured, count):
    from easyicu.research_agent.methods.descriptive_inputs import (
        measurement_provenance_receipt,
    )
    return measurement_provenance_receipt(frame, measured, count)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["line"] == 6


def test_exact_keyword_host_call_passes(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column=measured_column,
    count_column=count_column,
)
"""

    assert _signature_findings(script, ra) == []


def test_measurement_receipt_requires_exact_planner_declared_pair(ra):
    step = ra.AnalysisStep(
        step_id="cohort",
        intent="Apply host-bound cohort predicates.",
        inputs=[],
        expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
        method="cohort_definition_and_attrition",
    )
    script = """
from easyicu.research_agent.methods import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column="signal_measured",
    count_column="signal_n",
)
"""

    findings = _provenance_scope_findings(script, step)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "measurement_provenance_pair_undeclared",
        "helper_name": "measurement_provenance_receipt",
        "line": 3,
        "declared_pairs": [],
        "observed_pair": ["signal_measured", "signal_n"],
    }


def test_measurement_receipt_rejects_different_literal_pair(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column="other_measured",
    count_column="other_n",
)
"""

    findings = _provenance_scope_findings(script, _step(ra))

    assert len(findings) == 1
    assert findings[0].detail["declared_pairs"] == [
        ["value_measured", "value_n"]
    ]
    assert findings[0].detail["observed_pair"] == [
        "other_measured",
        "other_n",
    ]


def test_measurement_receipt_dynamic_call_passes_with_declared_pair(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column=measured_column,
    count_column=count_column,
)
"""

    assert _provenance_scope_findings(script, _step(ra)) == []


def test_measurement_receipt_runtime_introspection_is_blocked(ra):
    script = """
import inspect
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
signature = inspect.signature(measurement_provenance_receipt)
receipt = measurement_provenance_receipt(
    frame,
    measured_column="value_measured",
    count_column="value_n",
)
"""

    findings = [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "host_helper_runtime_introspection"
    ]

    assert len(findings) == 1
    assert findings[0].detail["helper_name"] == "measurement_provenance_receipt"


def test_literal_measurement_receipt_columns_must_match_declared_companion_roles(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column="value_max",
    count_column="value_n",
)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["violations"] == ["measured_column_role_invalid"]
    assert findings[0].detail["observed_measured_column"] == "value_max"
    assert findings[0].detail["expected_measured_column"] == "value_measured"

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["measurement_receipt_stable_binding_v1"]
    assert "measured_column='value_measured'" in repaired
    assert _signature_findings(repaired, ra) == []


def test_literal_measurement_receipt_accepts_exact_declared_companion_pair(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column="value_measured",
    count_column="value_n",
)
"""

    assert _signature_findings(script, ra) == []


def test_measurement_receipt_unknown_keyword_and_frame_adapter_are_normalized(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame=frame,
    planned_flag=measured_column,
    measured_column=measured_column,
    count_column=count_column,
)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert findings[0].detail["violations"] == ["unknown_keyword_argument"]
    assert names == ["measurement_receipt_stable_binding_v1"]
    assert "planned_flag" not in repaired
    assert (
        "measurement_provenance_receipt(frame, "
        "measured_column=measured_column, count_column=count_column)"
    ) in repaired
    assert _signature_findings(repaired, ra) == []


def test_render_only_figure_rejects_raw_row_provenance_helper(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)
receipt_frame = aggregate[["measured_n", "cohort_n"]].copy()
receipt_frame.columns = ["measured", "count"]
receipt = measurement_provenance_receipt(
    receipt_frame,
    measured_column="measured",
    count_column="measured_n",
)
"""

    findings = [
        finding
        for finding in audit_mechanical_code_contracts(script, _figure_step(ra))
        if (finding.detail or {}).get("reason") == "render_only_raw_provenance_helper"
    ]

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "render_only_raw_provenance_helper",
        "helper_name": "measurement_provenance_receipt",
        "line": 7,
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")
    assert _signature_findings(script, ra) == []


def test_local_same_name_without_host_import_is_not_claimed(ra):
    script = """
def measurement_provenance_receipt(frame, measured, count):
    return {}

receipt = measurement_provenance_receipt(frame, measured, count)
"""

    assert _signature_findings(script, ra) == []


def test_outer_host_import_shadowed_by_local_binding_is_not_claimed(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

def audit(measurement_provenance_receipt, frame, measured, count):
    return measurement_provenance_receipt(frame, measured, count)
"""

    assert _signature_findings(script, ra) == []


def test_deterministic_repair_moves_only_existing_names_to_keyword_slots(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(frame, measured_name, count_name)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["host_helper_keyword_only_call_v1"]
    assert (
        "measurement_provenance_receipt(frame, "
        "measured_column=measured_name, count_column=count_name)"
    ) in repaired
    assert _signature_findings(repaired, ra) == []


def test_deterministic_repair_removes_exact_legacy_helper_adapter(ra):
    script = """
import inspect
from easyicu.research_agent.methods.descriptive_inputs import (
    closed_categorical_counts,
    measurement_provenance_receipt,
)

def call_helper_adaptively(helper, *args, **kwargs):
    signature = inspect.signature(helper)
    return helper(*args, **kwargs)

def audit(df, measured_column, count_column, value_column, levels):
    try:
        measurement_provenance_receipt(
            df[measured_column],
            df[count_column],
            variable_name=value_column,
        )
    except TypeError:
        call_helper_adaptively(
            measurement_provenance_receipt,
            df[measured_column],
            value_column,
            count_series=df[count_column],
        )
    first = call_helper_adaptively(
        closed_categorical_counts,
        df[value_column],
        value_column,
        levels=levels,
    )
    second = call_helper_adaptively(
        closed_categorical_counts,
        df[count_column],
        count_column,
        levels=levels,
    )
    return first, second
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["host_helper_keyword_only_call_v1"]
    assert "call_helper_adaptively" not in repaired
    assert "import inspect" not in repaired
    assert (
        "measurement_provenance_receipt(df, "
        "measured_column=measured_column, count_column=count_column)"
    ) in repaired
    assert (
        "closed_categorical_counts(df[value_column], declared_levels=levels)"
        in repaired
    )
    assert (
        "closed_categorical_counts(df[count_column], declared_levels=levels)"
        in repaired
    )
    assert _signature_findings(repaired, ra) == []


def test_deterministic_repair_refuses_ambiguous_same_line_calls(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
first = measurement_provenance_receipt(frame, measured_a, count_a); second = measurement_provenance_receipt(frame, measured_b, count_b)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert repaired == script
    assert names == []


def test_deterministic_repair_binds_multiple_independent_literal_role_calls(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
first = measurement_provenance_receipt(frame, "first_n", "first_measured")
second = measurement_provenance_receipt(frame, "second_measured", "second_n")
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert names == ["host_helper_keyword_only_call_v1"]
    assert (
        'measurement_provenance_receipt(frame, measured_column="first_measured", '
        'count_column="first_n")'
    ) in repaired
    assert (
        'measurement_provenance_receipt(frame, measured_column="second_measured", '
        'count_column="second_n")'
    ) in repaired
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_requires_explicit_declared_levels_before_execution(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, levels):
    return closed_categorical_counts(series)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "host_helper_call_signature_invalid",
        "helper_name": "closed_categorical_counts",
        "line": 5,
        "max_positional": 1,
        "required_keywords": ["declared_levels"],
        "violations": ["required_keyword_only_argument_missing"],
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_closed_counts_explicit_declared_levels_passes(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, levels):
    return closed_categorical_counts(series, declared_levels=levels)
"""

    assert _signature_findings(script, ra) == []


def test_closed_counts_table_index_is_not_a_category_level_contract(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts
counts = closed_categorical_counts(sex, declared_levels=["Female", "Male"])
count_table = counts.table
levels = list(count_table.index)
if set(levels) != {"Female", "Male"}:
    raise RuntimeError("unexpected levels")
"""

    findings = _level_index_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail == {
        "reason": "closed_counts_table_index_used_as_levels",
        "helper_name": "closed_categorical_counts",
        "line": 5,
        "result_name": "counts",
        "table_name": "count_table",
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_closed_counts_direct_table_index_is_repaired_to_level_column(ra):
    script = """
import easyicu.research_agent.methods.descriptive_inputs as host_inputs
counts = host_inputs.closed_categorical_counts(
    sex, declared_levels=["Female", "Male"]
)
levels = list(counts.table.index)
if set(levels) != {"Female", "Male"}:
    raise RuntimeError("unexpected levels")
"""
    findings = _level_index_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["closed_counts_level_column_v1"]
    assert 'levels = list(counts.table["level"])' in repaired
    assert _level_index_findings(repaired, ra) == []


def test_closed_counts_named_table_index_is_repaired_to_level_column(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts
counts = closed_categorical_counts(sex, declared_levels=["Female", "Male"])
count_table = counts.table
levels = list(count_table.index)
if set(levels) != {"Female", "Male"}:
    raise RuntimeError("unexpected levels")
"""
    findings = _level_index_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["closed_counts_level_column_v1"]
    assert 'levels = list(count_table["level"])' in repaired
    assert _level_index_findings(repaired, ra) == []


def test_closed_counts_explicit_level_column_passes(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts
counts = closed_categorical_counts(sex, declared_levels=["Female", "Male"])
levels = list(counts.table["level"])
if set(levels) != {"Female", "Male"}:
    raise RuntimeError("unexpected levels")
"""

    assert _level_index_findings(script, ra) == []


def test_generic_dataframe_index_is_outside_closed_counts_contract(ra):
    script = """
import pandas as pd
table = pd.DataFrame({"level": ["Female", "Male"]})
levels = list(table.index)
if set(levels) != {"Female", "Male"}:
    raise RuntimeError("unexpected levels")
"""

    assert _level_index_findings(script, ra) == []


def test_unrelated_helper_table_index_is_not_claimed(ra):
    script = """
def custom_counts(series):
    return build_result(series)

counts = custom_counts(sex)
count_table = counts.table
levels = list(count_table.index)
if set(levels) != {"Female", "Male"}:
    raise RuntimeError("unexpected levels")
"""

    assert _level_index_findings(script, ra) == []


def test_shadowed_closed_counts_import_is_not_claimed(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def summarize(closed_categorical_counts, sex):
    counts = closed_categorical_counts(sex)
    levels = list(counts.table.index)
    if set(levels) != {"Female", "Male"}:
        raise RuntimeError("unexpected levels")
    return levels
"""

    assert _level_index_findings(script, ra) == []


def test_table_one_sdk_repairs_local_schema_to_exact_planner_spec(ra):
    # The prologue binding `frame` is part of the fixture, not decoration: the
    # gate runs on the assembled script, and a module-level read of a name
    # nothing binds is itself a blocking finding.
    script = """
import pandas as pd
from easyicu.research_agent.methods.table_one import build_grouped_table_one
frame = pd.DataFrame({"age": [61.0], "death": [0]})
table_one_spec = {
    "group_by": "death",
    "group_levels": [0, 1],
    "group_labels": {0: "No", 1: "Yes"},
    "variables": [{"name": "age", "type": "continuous"}],
}
result = build_grouped_table_one(frame, table_one_spec)
"""
    step = _table_step(ra)
    findings = [
        finding
        for finding in audit_mechanical_code_contracts(script, step)
        if (finding.detail or {}).get("reason") == "table_one_spec_not_planner_owned"
    ]

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert names == ["table_one_planner_spec_binding_v1"]
    assert audit_mechanical_code_contracts(repaired, step) == []
    assert "group_labels" not in repaired
    assert "variable_kind" in repaired


def test_table_one_sdk_accepts_exact_planner_spec(ra):
    step = _table_step(ra)
    spec = step.table_one_spec.model_dump(mode="python")
    script = (
        "import pandas as pd\n"
        "from easyicu.research_agent.methods.table_one import "
        "build_grouped_table_one\n"
        'frame = pd.DataFrame({"age": [61.0], "death": [0]})\n'
        f"table_one_spec = {spec!r}\n"
        "result = build_grouped_table_one(frame, table_one_spec)\n"
    )

    assert audit_mechanical_code_contracts(script, step) == []


def _spec_not_passed_findings(script: str, step):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, step)
        if (finding.detail or {}).get("reason") == "table_one_spec_not_passed"
    ]


def test_table_one_sdk_call_that_passes_no_spec_is_refused_before_launch(ra):
    """The gate constrained WHICH spec is passed and never that one is.

    m1's ``02_baseline_table_one`` died in the sandbox with
    ``build_grouped_table_one() missing 1 required positional argument:
    'spec'`` -- while the correct spec sat four lines above, host-rendered,
    and was used later in the same script for the receipt.  The gate that
    exists to constrain this exact call could not see it: its entry condition
    required two arguments before it would look at anything, so the one shape
    that supplies no spec skipped the check entirely.

    Measured across the recorded corpus: 121 calls pass the spec positionally
    and 3 pass nothing, in three different runs under three different step
    names.  This is not a one-off typo.
    """

    step = _table_step(ra)
    spec = step.table_one_spec.model_dump(mode="python")
    script = (
        "import pandas as pd\n"
        "from easyicu.research_agent.methods.table_one import "
        "build_grouped_table_one\n"
        'frame = pd.DataFrame({"age": [61.0], "death": [0]})\n'
        f"table_one_spec = {spec!r}\n"
        "result = build_grouped_table_one(frame)\n"
    )

    findings = _spec_not_passed_findings(script, step)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["line"] == 5


def test_table_one_sdk_accepts_the_spec_by_keyword(ra):
    """The signature names the parameter, so naming it is a legal call."""

    step = _table_step(ra)
    spec = step.table_one_spec.model_dump(mode="python")
    script = (
        "import pandas as pd\n"
        "from easyicu.research_agent.methods.table_one import "
        "build_grouped_table_one\n"
        'frame = pd.DataFrame({"age": [61.0], "death": [0]})\n'
        f"table_one_spec = {spec!r}\n"
        "result = build_grouped_table_one(frame, spec=table_one_spec)\n"
    )

    assert _spec_not_passed_findings(script, step) == []


def test_a_spec_forwarded_through_a_mapping_splat_is_not_refused(ra):
    """Refusing what cannot be resolved here would block legal code.

    The gate reads one call site; a ``**mapping`` may carry the spec and
    nothing at this layer can prove otherwise.  Staying silent keeps the
    refusal aimed at the shape that is unambiguously wrong.
    """

    step = _table_step(ra)
    spec = step.table_one_spec.model_dump(mode="python")
    script = (
        "import pandas as pd\n"
        "from easyicu.research_agent.methods.table_one import "
        "build_grouped_table_one\n"
        'frame = pd.DataFrame({"age": [61.0], "death": [0]})\n'
        f"table_one_spec = {spec!r}\n"
        'kwargs = {"spec": table_one_spec}\n'
        "result = build_grouped_table_one(frame, **kwargs)\n"
    )

    assert _spec_not_passed_findings(script, step) == []


def test_a_step_with_no_table_one_spec_is_not_policed_here(ra):
    """No Planner declaration, nothing for this gate to enforce."""

    script = (
        "import pandas as pd\n"
        "from easyicu.research_agent.methods.table_one import "
        "build_grouped_table_one\n"
        'frame = pd.DataFrame({"age": [61.0]})\n'
        "result = build_grouped_table_one(frame)\n"
    )

    assert _spec_not_passed_findings(script, _step(ra)) == []


def test_closed_counts_missing_levels_is_repaired_without_inventing_categories(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, levels):
    return closed_categorical_counts(series)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["closed_counts_declared_levels_binding_v1"]
    assert "closed_categorical_counts(series, declared_levels=levels)" in repaired
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_repair_refuses_ambiguous_local_category_parameter(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, categories):
    return closed_categorical_counts(series)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert repaired == script
    assert names == []


def test_closed_counts_unknown_diagnostic_keyword_is_repaired_without_science_change(
    ra,
):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, variable, levels):
    return closed_categorical_counts(
        series,
        variable=variable,
        declared_levels=levels,
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert findings[0].detail["violations"] == ["unknown_keyword_argument"]
    assert names == ["closed_counts_stable_keywords_v1"]
    assert "closed_categorical_counts(series, declared_levels=levels)" in repaired
    assert "variable=variable" not in repaired
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_allowed_values_adapter_is_rebound_without_science_change(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, allowed_values, name):
    return closed_categorical_counts(
        series,
        allowed_values=allowed_values,
        variable_name=name,
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert findings[0].detail["violations"] == [
        "required_keyword_only_argument_missing",
        "unknown_keyword_argument",
    ]
    assert names == ["closed_counts_stable_keywords_v1"]
    assert (
        "closed_categorical_counts(series, declared_levels=allowed_values)"
        in repaired
    )
    assert "variable_name=name" not in repaired
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_unknown_keywords_are_repaired_atomically(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def first(series, variable, first_levels):
    return closed_categorical_counts(
        series, variable=variable, levels=first_levels
    )

def second(series, variable, second_levels):
    return closed_categorical_counts(
        series, variable=variable, declared_levels=second_levels
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert names == ["closed_counts_stable_keywords_v1"]
    assert repaired.count("variable=variable") == 0
    assert repaired.count("declared_levels=") == 2
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_stable_keyword_repair_refuses_ambiguous_or_unknown_inputs(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, variable, levels, declared_levels):
    return closed_categorical_counts(
        series,
        variable=variable,
        levels=levels,
        declared_levels=declared_levels,
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []


def test_closed_counts_stable_keyword_repair_does_not_bind_reassigned_levels(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, variable, levels):
    levels = infer_levels(series)
    return closed_categorical_counts(series, variable=variable)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []


def test_fixed_local_return_arity_must_match_direct_unpack(ra):
    script = """
def collect(frame):
    left = frame["left"]
    right = frame["right"]
    return left, right

def main(frame):
    receipt, left, right = collect(frame)
"""

    findings = _unpack_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "local_helper_unpack_arity_mismatch",
        "function_name": "collect",
        "call_line": 8,
        "return_lines": [5],
        "return_arity": 2,
        "target_arity": 3,
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_dynamic_or_matching_local_returns_are_not_claimed(ra):
    matching = """
def collect(frame):
    return frame["left"], frame["right"]

def main(frame):
    left, right = collect(frame)
"""
    dynamic = """
def collect(frame):
    return make_result(frame)

def main(frame):
    left, right, extra = collect(frame)
"""

    assert _unpack_findings(matching, ra) == []
    assert _unpack_findings(dynamic, ra) == []


def test_deterministic_repair_threads_discarded_host_receipt(ra):
    script = """
def collect(frame, measured_column, count_column):
    from easyicu.research_agent.methods.descriptive_inputs import (
        measurement_provenance_receipt,
    )
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    measured = frame[measured_column]
    count = frame[count_column]
    return measured, count

def main(frame, measured_column, count_column):
    receipt, measured, count = collect(frame, measured_column, count_column)
"""
    findings = _unpack_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["local_helper_unpack_receipt_v1"]
    assert "receipt = measurement_provenance_receipt(" in repaired
    assert "return receipt, measured, count" in repaired
    assert _unpack_findings(repaired, ra) == []


def test_discarded_receipt_repair_refuses_unaligned_unpack_tail(ra):
    script = """
def collect(frame, measured_column, count_column):
    from easyicu.research_agent.methods.descriptive_inputs import (
        measurement_provenance_receipt,
    )
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    measured = frame[measured_column]
    count = frame[count_column]
    return measured, count

def main(frame, measured_column, count_column):
    receipt, count, measured = collect(frame, measured_column, count_column)
"""
    findings = _unpack_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert repaired == script
    assert names == []


def test_publication_export_audit_exact_fresh_api_passes(ra):
    script = """
from easyicu.research_agent.figures.publication import audit_publication_exports
qa = audit_publication_exports(
    paths=out_dir,
    min_bytes=2048,
    require_svg_text=True,
)
"""

    assert _signature_findings(script, ra) == []


def test_publication_export_audit_retired_keywords_repair_before_execution(ra):
    script = """
from easyicu.research_agent.figures.publication import audit_publication_exports
qa = audit_publication_exports(out_dir=out_dir, stem=stem)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "host_helper_call_signature_invalid",
        "helper_name": "audit_publication_exports",
        "line": 3,
        "max_positional": 1,
        "required_keywords": [],
        "violations": ["paths_argument_missing", "unknown_keyword_argument"],
    }
    assert names == ["publication_export_audit_paths_v1"]
    assert "audit_publication_exports(paths=out_dir)" in repaired
    assert "stem=" not in repaired
    assert _signature_findings(repaired, ra) == []


def test_archived_step03_publication_audit_shape_is_repaired(ra):
    script = """
from easyicu.research_agent.figures.publication import audit_publication_exports

def render(out_dir, stem):
    qa = audit_publication_exports(
        out_dir=out_dir,
        stem=stem,
    )
    return qa
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["publication_export_audit_paths_v1"]
    assert "audit_publication_exports(paths=out_dir)" in repaired
    assert _signature_findings(repaired, ra) == []


def test_archived_step06_two_positional_fallback_remains_finding_only(ra):
    script = """
from easyicu.research_agent.figures.publication import audit_publication_exports

def render(step_out_dir):
    try:
        publication_qa = audit_publication_exports(
            out_dir=step_out_dir,
            stem="icu_los_adjusted_effect",
        )
    except TypeError:
        publication_qa = audit_publication_exports(
            step_out_dir, "icu_los_adjusted_effect"
        )
    return publication_qa
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert names == ["publication_export_audit_paths_v1"]
    assert "audit_publication_exports(paths=step_out_dir)" in repaired
    remaining = _signature_findings(repaired, ra)
    assert len(remaining) == 1
    assert remaining[0].detail["violations"] == [
        "keyword_only_parameters_passed_positionally"
    ]


def test_publication_export_audit_alias_import_is_finding_only(ra):
    script = """
from easyicu.research_agent.figures.publication import (
    audit_publication_exports as audit_exports,
)
qa = audit_exports(out_dir=out_dir, stem=stem)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []


def test_publication_export_audit_dynamic_arguments_are_finding_only(ra):
    script = """
from easyicu.research_agent.figures.publication import audit_publication_exports
qa = audit_publication_exports(*args, **kwargs)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []


def test_publication_export_audit_unknown_extra_keyword_is_not_repaired(ra):
    script = """
from easyicu.research_agent.figures.publication import audit_publication_exports
qa = audit_publication_exports(out_dir=out_dir, stem=stem, strict=True)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []
