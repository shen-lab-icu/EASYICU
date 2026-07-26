from __future__ import annotations

from easyicu.research_agent.repairs.figure_distribution import (
    patch_categorical_distribution_clinical_bin_role,
    patch_text_distribution_denominator_from_counts,
)
from easyicu.research_agent.repairs.import_repair import patch_known_host_helper_import
from easyicu.research_agent.repairs.host_helper_failure import (
    patch_host_validation_helper_reraise,
)
from easyicu.research_agent.schema import ValidationFinding
from easyicu.research_agent.repairs.source import _deterministic_summary_repair

SUMMARY = {
    "status": "failed",
    "diagnostics": [
        {
            "stage": "rendering",
            "error": (
                "No supported biomarker categorical distribution rows were found. "
                "Expected non-null category rows with an explicitly supported "
                "distribution statistic."
            ),
        }
    ],
}


CODE = """
valid_distribution_statistics = {
    "distribution",
    "category_distribution",
    "level_distribution",
    "frequency",
    "count",
    "percentage",
}
category_mask = (
    work["category"].notna()
    & work["_statistic_norm"].isin(valid_distribution_statistics)
)
distribution = work.loc[category_mask].copy()
"""


def test_adds_closed_clinical_bin_role_to_failed_categorical_renderer() -> None:
    repaired = patch_categorical_distribution_clinical_bin_role(CODE, SUMMARY)
    assert repaired is not None
    assert '"clinical_bin",' in repaired
    assert repaired.replace('    "clinical_bin",\n', "") == CODE


def test_summary_repair_surfaces_registered_repair_id() -> None:
    repaired = _deterministic_summary_repair(code=CODE, step_summary=SUMMARY)
    assert repaired is not None
    repair_id, repaired_code = repaired
    assert repair_id == "categorical_distribution_clinical_bin_role_v1"
    assert '"clinical_bin",' in repaired_code


def test_does_not_patch_without_exact_runtime_diagnostic() -> None:
    assert (
        patch_categorical_distribution_clinical_bin_role(CODE, {"status": "failed"})
        is None
    )


def test_does_not_patch_without_category_nonnull_guard() -> None:
    assert (
        patch_categorical_distribution_clinical_bin_role(
            CODE.replace('work["category"].notna()', "True"), SUMMARY
        )
        is None
    )


def test_does_not_patch_dynamic_or_ambiguous_role_sets() -> None:
    dynamic = CODE.replace('    "percentage",\n', "    *extra_roles,\n")
    assert patch_categorical_distribution_clinical_bin_role(dynamic, SUMMARY) is None
    ambiguous = CODE + CODE.replace("valid_distribution_statistics", "other_roles")
    assert patch_categorical_distribution_clinical_bin_role(ambiguous, SUMMARY) is None


def test_is_idempotent() -> None:
    once = patch_categorical_distribution_clinical_bin_role(CODE, SUMMARY)
    assert once is not None
    assert patch_categorical_distribution_clinical_bin_role(once, SUMMARY) is None


def test_relocates_exact_known_host_helper_import_after_module_error() -> None:
    code = (
        "from easyicu.research_agent.methods.validation "
        "import strict_numeric_input\n"
    )
    repaired = patch_known_host_helper_import(
        code,
        "ModuleNotFoundError: No module named "
        "'easyicu.research_agent.methods.validation'",
    )
    assert repaired == (
        "from easyicu.research_agent.methods.descriptive_inputs "
        "import strict_numeric_input\n"
    )


def test_recovers_exact_fail_closed_stub_to_known_host_helper() -> None:
    code = """# auto-stubs for stripped fake imports
def strict_numeric_input(*args, **kwargs): raise NotImplementedError("strict_numeric_input from easyicu.research_agent.methods.validation is not available; reimplement inline using numpy/scipy/statsmodels.")
# stripped: import from non-existent easyicu.research_agent.methods.validation
value = strict_numeric_input(series).values
"""
    repaired = patch_known_host_helper_import(
        code,
        {
            "error": (
                "strict_numeric_input from easyicu.research_agent.methods.validation "
                "is not available; reimplement inline using numpy/scipy/statsmodels."
            )
        },
    )
    assert repaired is not None
    assert "methods.descriptive_inputs import strict_numeric_input" in repaired
    assert "auto-stubs" not in repaired
    assert "stripped: import" not in repaired
    assert "value = strict_numeric_input(series).values" in repaired


def test_relocates_measurement_provenance_helper_import() -> None:
    code = (
        "from easyicu.research_agent.methods.measurement_provenance_receipt "
        "import measurement_provenance_receipt\n"
    )
    repaired = patch_known_host_helper_import(
        code,
        "ModuleNotFoundError: No module named "
        "'easyicu.research_agent.methods.measurement_provenance_receipt'",
    )

    assert repaired == (
        "from easyicu.research_agent.methods.descriptive_inputs "
        "import measurement_provenance_receipt\n"
    )


def test_relocates_measurement_provenance_helper_from_package_root() -> None:
    code = (
        "from easyicu.research_agent.methods "
        "import measurement_provenance_receipt\n"
    )
    repaired = patch_known_host_helper_import(
        code,
        "ImportError: cannot import name 'measurement_provenance_receipt' "
        "from 'easyicu.research_agent.methods'",
    )

    assert repaired == (
        "from easyicu.research_agent.methods.descriptive_inputs "
        "import measurement_provenance_receipt\n"
    )


def test_known_host_helper_relocation_rejects_unclosed_helpers() -> None:
    code = "from easyicu.research_agent.methods.validation import fit_model\n"
    assert (
        patch_known_host_helper_import(
            code,
            "ModuleNotFoundError: No module named "
            "'easyicu.research_agent.methods.validation'",
        )
        is None
    )


def test_reraises_exact_preflight_named_host_helper_handler() -> None:
    code = """from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input
try:
    values = strict_numeric_input(series).values
    summary["status"] = "completed"
except Exception as exc:
    summary["diagnostics"].append({"error": str(exc)})
finally:
    write_summary(summary)
"""
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="caught host validation failure",
        detail={
            "reason": "host_validation_helper_error_swallowed",
            "helper_names": ["strict_numeric_input"],
            "line": 5,
        },
    )
    repaired = patch_host_validation_helper_reraise(code, findings=[finding])
    assert "# _easyicu_host_validation_helper_reraise_v1\n    raise\n" in repaired
    assert 'summary["status"] = "completed"' in repaired
    assert 'summary["diagnostics"].append' in repaired


def test_host_helper_reraise_requires_exact_handler_and_helper() -> None:
    code = """try:
    values = other_helper(series)
except Exception:
    pass
"""
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="caught host validation failure",
        detail={
            "reason": "host_validation_helper_error_swallowed",
            "helper_names": ["strict_numeric_input"],
            "line": 3,
        },
    )
    assert patch_host_validation_helper_reraise(code, findings=[finding]) == code


DENOMINATOR_CODE = """import numpy as np
import pandas as pd

distribution = pd.DataFrame(
    {
        "count": [2, 3],
        "denominator": ["valid observed", "valid observed"],
    }
)
original_count = distribution["count"].copy()
original_denominator = distribution["denominator"].copy()
distribution["count"] = pd.to_numeric(original_count, errors="coerce")
distribution["denominator_numeric"] = pd.to_numeric(
    original_denominator, errors="coerce"
)
"""


def test_text_denominator_uses_complete_same_role_category_counts() -> None:
    repaired = patch_text_distribution_denominator_from_counts(
        DENOMINATOR_CODE,
        "ValueError: Distribution denominator must be numeric for figure reconciliation",
    )
    assert repaired is not None
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    distribution = namespace["distribution"]
    assert distribution["denominator_numeric"].tolist() == [5.0, 5.0]


def test_text_denominator_rejects_multiple_semantic_roles() -> None:
    code = DENOMINATOR_CODE.replace(
        '["valid observed", "valid observed"]',
        '["valid observed", "locked cohort"]',
    )
    repaired = patch_text_distribution_denominator_from_counts(
        code,
        "ValueError: Distribution denominator must be numeric for figure reconciliation",
    )
    assert repaired is not None
    try:
        exec(repaired, {})
    except ValueError as exc:
        assert "not one complete semantic role" in str(exc)
    else:
        raise AssertionError("ambiguous semantic denominator must fail closed")


def test_text_denominator_repair_requires_exact_failure_and_numeric_counts() -> None:
    assert (
        patch_text_distribution_denominator_from_counts(
            DENOMINATOR_CODE,
            "ValueError: another failure",
        )
        is None
    )
    assert (
        patch_text_distribution_denominator_from_counts(
            DENOMINATOR_CODE.replace(
                'distribution["count"] = pd.to_numeric(original_count, errors="coerce")',
                'distribution["count"] = original_count',
            ),
            "ValueError: Distribution denominator must be numeric for figure reconciliation",
        )
        is None
    )
