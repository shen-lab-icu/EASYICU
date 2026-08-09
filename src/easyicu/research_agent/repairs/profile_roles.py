"""Preserve every typed phenotype-profile role in an all-rows figure.

The Planner can bind ``table:phenotype_profiles`` with ``mode="all_rows"``.
Generated profile figures sometimes render only ``continuous_median_iqr``
rows even though the same typed parent also contains categorical measurement
status and cluster-size rows.  This module owns one narrow representation
repair: add a separate percentage panel for those already-computed rows.

The transform does not select clusters, recompute profiles, impute values, or
change the upstream table.  It requires the exact host-owned all-rows contract,
an explicit concept finding, and a closed generated-script shape.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any


_REPAIR_ID = "all_rows_profile_roles_display_v1"
_INPUT_KEY = "table:phenotype_profiles"


def _finding_payload(finding: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(finding, Mapping):
        message = str(finding.get("message") or "")
        detail = finding.get("detail")
    else:
        message = str(getattr(finding, "message", "") or "")
        detail = getattr(finding, "detail", None)
    return message, detail if isinstance(detail, Mapping) else {}


def _finding_requires_all_profile_roles(
    *,
    audit_messages: Sequence[str],
    repair_findings: Sequence[Any],
) -> bool:
    texts = [str(message or "") for message in audit_messages]
    variables: set[str] = set()
    for finding in repair_findings:
        message, detail = _finding_payload(finding)
        texts.append(message)
        raw_variables = detail.get("variables")
        if isinstance(raw_variables, Sequence) and not isinstance(
            raw_variables, (str, bytes)
        ):
            variables.update(str(value).strip().lower() for value in raw_variables)
    normalized = " ".join(texts).lower()
    return (
        {"summary_type", "continuous_median_iqr"} <= variables
        and "all_rows" in normalized
        and any(token in normalized for token in ("filter", "omit", "preserv"))
        and "role" in normalized
    )


def _step_authorizes_all_profile_rows(step: Any) -> bool:
    if step is None or str(getattr(step, "method", "") or "") != "visualization":
        return False
    inputs = tuple(str(value) for value in (getattr(step, "inputs", ()) or ()))
    outputs = tuple(
        str(value) for value in (getattr(step, "expected_outputs", ()) or ())
    )
    if _INPUT_KEY not in inputs or not any(
        value.startswith("figure:") and "cluster_profile" in value
        for value in outputs
    ):
        return False
    matching = []
    for contract in getattr(step, "input_consumption_contracts", ()) or ():
        input_key = str(getattr(contract, "input_key", "") or "")
        mode = str(getattr(contract, "mode", "") or "")
        if input_key == _INPUT_KEY:
            matching.append(mode)
    return matching == ["all_rows"]


def _replace_once(source: str, old: str, new: str) -> str | None:
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def patch_all_rows_profile_roles_display(
    code: str,
    *,
    step: Any,
    audit_messages: Sequence[str],
    repair_findings: Sequence[Any] = (),
) -> tuple[str, str | None]:
    """Add a distinct panel consuming every non-continuous profile row."""

    if not _step_authorizes_all_profile_rows(step) or not (
        _finding_requires_all_profile_roles(
            audit_messages=audit_messages,
            repair_findings=repair_findings,
        )
    ):
        return code, None
    try:
        ast.parse(code)
    except SyntaxError:
        return code, None
    if code.count('INPUT_KEY = "table:phenotype_profiles"') != 1:
        return code, None

    replacements = (
        (
            '''    continuous = profiles.loc[
        profiles["summary_type"].eq("continuous_median_iqr")
    ].copy()

    if continuous.empty:
''',
            '''    continuous = profiles.loc[
        profiles["summary_type"].eq("continuous_median_iqr")
    ].copy()
    other_profile_roles = profiles.loc[
        ~profiles["summary_type"].eq("continuous_median_iqr")
    ].copy()

    if len(continuous) + len(other_profile_roles) != len(profiles):
        raise RuntimeError("Typed profile-role partition lost input rows")
    if continuous.empty:
''',
        ),
        (
            '''    continuous["variable"] = continuous["variable"].astype(str)

    numeric_columns_used = [
''',
            '''    continuous["variable"] = continuous["variable"].astype(str)

    if other_profile_roles.empty:
        raise RuntimeError("All-rows profile input has no distinct role rows")
    other_profile_roles["cluster_label"] = coerce_numeric_fail_closed(
        other_profile_roles, "cluster_label"
    )
    other_profile_roles["percentage"] = coerce_numeric_fail_closed(
        other_profile_roles, "percentage"
    )
    if other_profile_roles["percentage"].isna().any():
        raise RuntimeError("Non-continuous profile roles lack percentages")
    if bool(
        (
            (other_profile_roles["percentage"] < 0.0)
            | (other_profile_roles["percentage"] > 100.0)
        ).any()
    ):
        raise RuntimeError("Non-continuous profile percentages exceed [0, 100]")
    other_profile_roles["variable"] = other_profile_roles["variable"].astype(str)
    other_profile_roles["category"] = (
        other_profile_roles["category"].where(
            other_profile_roles["category"].notna(), "all"
        ).astype(str)
    )
    other_profile_roles["display_role"] = (
        other_profile_roles["summary_type"].astype(str)
        + ": "
        + other_profile_roles["variable"]
        + " = "
        + other_profile_roles["category"]
    )

    numeric_columns_used = [
''',
        ),
        (
            '''    if len(continuous) != expected_pairs:
        raise RuntimeError(
            "Continuous profile rows do not form a complete variable-cluster grid"
        )

    median_matrix = continuous.pivot(
''',
            '''    if len(continuous) != expected_pairs:
        raise RuntimeError(
            "Continuous profile rows do not form a complete variable-cluster grid"
        )

    role_clusters = sorted(
        other_profile_roles["cluster_label"].unique().tolist()
    )
    if role_clusters != clusters:
        raise RuntimeError("Profile-role rows do not cover the same clusters")
    duplicate_role_keys = other_profile_roles.duplicated(
        subset=["display_role", "cluster_label"],
        keep=False,
    )
    if bool(duplicate_role_keys.any()):
        raise RuntimeError("Profile-role rows are not unique within cluster")
    role_labels = sorted(other_profile_roles["display_role"].unique().tolist())
    role_matrix = other_profile_roles.pivot(
        index="display_role",
        columns="cluster_label",
        values="percentage",
    ).reindex(index=role_labels, columns=clusters)
    if role_matrix.isna().any().any():
        raise RuntimeError("Profile-role percentages are incomplete")

    median_matrix = continuous.pivot(
''',
        ),
        (
            '''            "Candidate cluster profiles are displayed as standardized "
            "cluster-level median patterns, with measurement availability "
            "shown separately."
''',
            '''            "Candidate cluster profiles preserve every typed parent row: "
            "continuous medians use display-only normalization, measurement "
            "availability is separate, and remaining profile roles retain "
            "their upstream percentages in a distinct panel."
''',
        ),
        (
            '''                "title": "Standardized median profiles",
                "claim": (
                    "Standardized median patterns across candidate clusters; "
                    "medians are retained as robust summaries for skewed "
                    "variables."
                ),
''',
            '''                "title": "Display-normalized median profiles",
                "claim": (
                    "Display-only normalization is applied to completed cluster "
                    "medians; it is not reused for clustering or inference."
                ),
''',
        ),
        (
            '''                "chart_type": "availability_heatmap",
            },
        ],
''',
            '''                "chart_type": "availability_heatmap",
            },
            {
                "panel_id": "C",
                "role": "descriptive_result",
                "title": "Other typed profile roles",
                "claim": (
                    "Categorical measurement-status and cluster-size rows are "
                    "shown separately at their upstream percentages."
                ),
                "evidence_ids": [binding["evidence_id"]],
                "chart_type": "role_percentage_heatmap",
            },
        ],
''',
        ),
        (
            '''    n_rows = max(5.5, 0.28 * len(variables) + 1.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, n_rows),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28},
        constrained_layout=True,
    )

    axes = np.asarray(axes).reshape(-1)
    ax_profile, ax_availability = axes[:2]
''',
            '''    n_rows = max(5.5, 0.28 * max(len(variables), len(role_labels)) + 1.8)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, n_rows),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.35], "wspace": 0.34},
        constrained_layout=True,
    )

    axes = np.asarray(axes).reshape(-1)
    ax_profile, ax_availability, ax_roles = axes[:3]
''',
        ),
        (
            '''    ax_profile.set_title("Standardized median profiles", loc="left", fontsize=11)
''',
            '''    ax_profile.set_title(
        "Display-normalized median profiles", loc="left", fontsize=11
    )
''',
        ),
        (
            '''    profile_colorbar.set_label("Feature-wise z-score of cluster median")
''',
            '''    profile_colorbar.set_label(
        "Display-only feature-wise z-score of completed cluster median"
    )
''',
        ),
        (
            '''    availability_colorbar.set_label("Nonmissing fraction")

    add_panel_label(ax_profile, "A")
    add_panel_label(ax_availability, "B")
''',
            '''    availability_colorbar.set_label("Nonmissing fraction")

    roles_image = ax_roles.imshow(
        role_matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap="Purples",
        norm=Normalize(vmin=0.0, vmax=100.0),
        interpolation="nearest",
    )
    ax_roles.set_xticks(range(len(clusters)))
    ax_roles.set_xticklabels(
        [f"Cluster {int(x)}" for x in clusters],
        rotation=35,
        ha="right",
    )
    ax_roles.set_yticks(range(len(role_labels)))
    ax_roles.set_yticklabels(role_labels, fontsize=7)
    ax_roles.set_xlabel("Candidate cluster")
    ax_roles.set_title("Other typed profile roles", loc="left", fontsize=11)
    roles_colorbar = fig.colorbar(
        roles_image,
        ax=ax_roles,
        fraction=0.046,
        pad=0.03,
    )
    roles_colorbar.set_label("Upstream percentage")

    add_panel_label(ax_profile, "A")
    add_panel_label(ax_availability, "B")
    add_panel_label(ax_roles, "C")
''',
        ),
        (
            '''        "profile_row_count": int(len(continuous)),
        "raw_nonfinite_nonmissing_counts": raw_nonfinite,
''',
            '''        "profile_row_count": int(len(continuous)),
        "other_profile_role_row_count": int(len(other_profile_roles)),
        "all_profile_row_count": int(len(profiles)),
        "summary_types_preserved": sorted(
            profiles["summary_type"].astype(str).unique().tolist()
        ),
        "raw_nonfinite_nonmissing_counts": raw_nonfinite,
''',
        ),
        (
            '''        "standardization": {
            "type": "within-feature z-score across cluster medians",
            "constant_features_set_to_zero": int(constant_feature_count),
        },
''',
            '''        "standardization": {
            "type": "within-feature z-score across completed cluster medians",
            "scope": "display_only_not_reused_analytically",
            "constant_features_set_to_zero": int(constant_feature_count),
        },
''',
        ),
    )

    repaired = code
    for old, new in replacements:
        updated = _replace_once(repaired, old, new)
        if updated is None:
            return code, None
        repaired = updated
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code, None
    return repaired, _REPAIR_ID


__all__ = ["patch_all_rows_profile_roles_display"]
