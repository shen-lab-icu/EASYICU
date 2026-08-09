from __future__ import annotations

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.profile_roles import (
    patch_all_rows_profile_roles_display,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    ArtifactConsumptionContract,
    ValidationFinding,
)


_CODE = '''
INPUT_KEY = "table:phenotype_profiles"


def main():
    continuous = profiles.loc[
        profiles["summary_type"].eq("continuous_median_iqr")
    ].copy()

    if continuous.empty:
        raise RuntimeError("No continuous median/IQR rows available for profile figure")

    continuous["variable"] = continuous["variable"].astype(str)

    numeric_columns_used = [
        "cluster_label",
        "n",
    ]

    if len(continuous) != expected_pairs:
        raise RuntimeError(
            "Continuous profile rows do not form a complete variable-cluster grid"
        )

    median_matrix = continuous.pivot(
        index="variable",
        columns="cluster_label",
        values="median",
    )

    contract = make_figure_contract(
        figure_id=figure_id,
        core_claim=(
            "Candidate cluster profiles are displayed as standardized "
            "cluster-level median patterns, with measurement availability "
            "shown separately."
        ),
        panels=[
            {
                "panel_id": "A",
                "role": "descriptive_result",
                "title": "Standardized median profiles",
                "claim": (
                    "Standardized median patterns across candidate clusters; "
                    "medians are retained as robust summaries for skewed "
                    "variables."
                ),
                "evidence_ids": [binding["evidence_id"]],
                "chart_type": "profile_heatmap",
            },
            {
                "panel_id": "B",
                "role": "data_quality",
                "title": "Measurement availability",
                "claim": "Availability stays separate.",
                "evidence_ids": [binding["evidence_id"]],
                "chart_type": "availability_heatmap",
            },
        ],
        source_data=source_files,
    )

    n_rows = max(5.5, 0.28 * len(variables) + 1.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, n_rows),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28},
        constrained_layout=True,
    )

    axes = np.asarray(axes).reshape(-1)
    ax_profile, ax_availability = axes[:2]

    ax_profile.set_title("Standardized median profiles", loc="left", fontsize=11)
    profile_colorbar.set_label("Feature-wise z-score of cluster median")
    availability_colorbar.set_label("Nonmissing fraction")

    add_panel_label(ax_profile, "A")
    add_panel_label(ax_availability, "B")

    summary = {
        "profile_row_count": int(len(continuous)),
        "raw_nonfinite_nonmissing_counts": raw_nonfinite,
        "standardization": {
            "type": "within-feature z-score across cluster medians",
            "constant_features_set_to_zero": int(constant_feature_count),
        },
    }
'''


def _step(*, mode: str = "all_rows") -> AnalysisStep:
    return AnalysisStep(
        step_id="profile_figure",
        intent="Render every typed phenotype-profile row by role.",
        method="visualization",
        inputs=["table:phenotype_profiles"],
        expected_outputs=["figure:cluster_profile_visualisation"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key="table:phenotype_profiles",
                mode=mode,
            )
        ],
    )


def _finding() -> ValidationFinding:
    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message=(
            "The script filters to summary_type='continuous_median_iqr' and "
            "omits other rows from the all_rows phenotype_profiles input, "
            "including typed feature roles."
        ),
        detail={
            "issue_code": "other",
            "step_id": "profile_figure",
            "variables": ["summary_type", "continuous_median_iqr"],
        },
    )


def test_exact_all_rows_profile_figure_adds_distinct_role_panel() -> None:
    code = _CODE

    repaired, repair_name = patch_all_rows_profile_roles_display(
        code,
        step=_step(),
        audit_messages=[_finding().message],
        repair_findings=[_finding()],
    )

    assert repair_name == "all_rows_profile_roles_display_v1"
    assert repaired != code
    assert 'other_profile_roles = profiles.loc[' in repaired
    assert 'ax_roles.set_title("Other typed profile roles"' in repaired
    assert '"all_profile_row_count": int(len(profiles))' in repaired
    assert '"scope": "display_only_not_reused_analytically"' in repaired


def test_profile_role_repair_is_idempotent() -> None:
    code = _CODE
    once, first_name = patch_all_rows_profile_roles_display(
        code,
        step=_step(),
        audit_messages=[_finding().message],
        repair_findings=[_finding()],
    )
    twice, second_name = patch_all_rows_profile_roles_display(
        once,
        step=_step(),
        audit_messages=[_finding().message],
        repair_findings=[_finding()],
    )

    assert first_name == "all_rows_profile_roles_display_v1"
    assert second_name is None
    assert twice == once


def test_profile_role_repair_refuses_missing_host_authority_or_finding() -> None:
    code = _CODE

    assert patch_all_rows_profile_roles_display(
        code,
        step=_step(mode="single_row"),
        audit_messages=[_finding().message],
        repair_findings=[_finding()],
    ) == (code, None)
    assert patch_all_rows_profile_roles_display(
        code,
        step=_step(),
        audit_messages=["Unrelated concept warning."],
        repair_findings=[],
    ) == (code, None)


def test_profile_role_repair_is_registered_structural_and_automatic() -> None:
    metadata = repair_metadata_for("all_rows_profile_roles_display_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert automatic_repair_allowed("all_rows_profile_roles_display_v1")
