from __future__ import annotations

import csv
import json

from easyicu.research_agent.execution.output_files import bind_primary_output


def test_binds_registered_adjusted_association_row(tmp_path) -> None:
    path = tmp_path / "adjusted_association_estimates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exposure",
                "effect_scale",
                "estimate",
                "ci_low",
                "ci_high",
                "fit_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "exposure": "exposure_col",
                "effect_scale": "odds_ratio",
                "estimate": "1.6",
                "ci_low": "1.5",
                "ci_high": "1.7",
                "fit_status": "fitted",
            }
        )

    bound = bind_primary_output(
        {
            "output_files": {
                "table:adjusted_association_estimates": path.name,
            }
        },
        tmp_path,
    )

    assert bound["primary_or"] == 1.6
    assert bound["primary_or_ci"] == [1.5, 1.7]
    assert bound["primary_association_term"] == "exposure_col"


def test_refuses_unregistered_or_multirow_estimates(tmp_path) -> None:
    path = tmp_path / "adjusted_association_estimates.csv"
    path.write_text(
        "exposure,effect_scale,estimate,ci_low,ci_high,fit_status\n"
        "a,odds_ratio,1.1,1.0,1.2,fitted\n"
        "b,odds_ratio,1.2,1.1,1.3,fitted\n",
        encoding="utf-8",
    )

    assert bind_primary_output({}, tmp_path).get("primary_or") is None
    assert (
        bind_primary_output(
            {
                "output_files": {
                    "table:adjusted_association_estimates": path.name,
                }
            },
            tmp_path,
        ).get("primary_or")
        is None
    )


def test_binds_registered_primary_or_statistic(tmp_path) -> None:
    path = tmp_path / "primary_or.json"
    path.write_text(
        json.dumps(
            {
                "name": "primary_or",
                "estimate": 1.6,
                "ci_low": 1.5,
                "ci_high": 1.7,
            }
        ),
        encoding="utf-8",
    )

    bound = bind_primary_output(
        {"output_files": {"statistic:primary_or": path.name}},
        tmp_path,
    )

    assert bound["primary_or"] == 1.6
    assert bound["primary_or_ci"] == [1.5, 1.7]
    assert bound["primary_estimate_label"] == "odds_ratio"


def test_refuses_unregistered_mismatched_or_unsafe_primary_or_statistic(
    tmp_path,
) -> None:
    path = tmp_path / "primary_or.json"
    path.write_text(
        json.dumps(
            {
                "name": "different_statistic",
                "estimate": 1.6,
                "ci_low": 1.5,
                "ci_high": 1.7,
            }
        ),
        encoding="utf-8",
    )

    assert (
        bind_primary_output(
            {"output_files": {"statistic:primary_or": path.name}},
            tmp_path,
        ).get("primary_or")
        is None
    )

    path.write_text(
        json.dumps(
            {
                "name": "primary_or",
                "estimate": 1.6,
                "value": 1.7,
                "ci_low": 1.5,
                "ci_high": 1.8,
            }
        ),
        encoding="utf-8",
    )
    assert (
        bind_primary_output(
            {"output_files": {"statistic:primary_or": path.name}},
            tmp_path,
        ).get("primary_or")
        is None
    )

    outside = tmp_path.parent / "outside-primary-or.json"
    outside.write_text(
        json.dumps({"name": "primary_or", "estimate": 1.6}),
        encoding="utf-8",
    )
    assert (
        bind_primary_output(
            {"output_files": {"statistic:primary_or": f"../{outside.name}"}},
            tmp_path,
        ).get("primary_or")
        is None
    )
    assert bind_primary_output({}, tmp_path).get("primary_or") is None

    path.write_text(
        json.dumps({"name": "primary_or", "estimate": True}),
        encoding="utf-8",
    )
    assert (
        bind_primary_output(
            {"output_files": {"statistic:primary_or": path.name}},
            tmp_path,
        ).get("primary_or")
        is None
    )
