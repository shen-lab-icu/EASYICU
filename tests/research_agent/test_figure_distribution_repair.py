from __future__ import annotations

from easyicu.research_agent.repairs.figure_distribution import (
    patch_categorical_distribution_clinical_bin_role,
)
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
