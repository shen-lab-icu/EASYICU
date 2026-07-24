from __future__ import annotations

import csv
import json

from easyicu.research_agent.execution.output_files import (
    bind_primary_output,
    normalize_typed_statistic_sidecars,
)


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


def test_normalizes_typed_statistic_identity_without_changing_values(
    tmp_path,
) -> None:
    primary = tmp_path / "primary_or.json"
    primary.write_text(
        json.dumps({"estimate": 1.6, "ci_low": 1.5, "ci_high": 1.7}),
        encoding="utf-8",
    )
    grouped = tmp_path / "robustness_summary.json"
    grouped.write_text(
        json.dumps({"primary_or": 1.6, "complete_case_or": 1.55}),
        encoding="utf-8",
    )

    receipts = normalize_typed_statistic_sidecars(
        {
            "output_files": {
                "statistic:primary_or": primary.name,
                "statistic:robustness_summary": grouped.name,
            }
        },
        tmp_path,
    )

    assert [receipt["product"] for receipt in receipts] == [
        "statistic:primary_or",
        "statistic:robustness_summary",
    ]
    assert all(
        receipt["before_sha256"] != receipt["after_sha256"] for receipt in receipts
    )
    assert json.loads(primary.read_text()) == {
        "name": "primary_or",
        "estimate": 1.6,
        "ci_low": 1.5,
        "ci_high": 1.7,
    }
    assert json.loads(grouped.read_text()) == {
        "name": "robustness_summary",
        "primary_or": 1.6,
        "complete_case_or": 1.55,
    }


def test_output_normalizer_refuses_conflicting_or_nonnumeric_payloads(
    tmp_path,
) -> None:
    conflicting = tmp_path / "metric.json"
    conflicting.write_text(
        json.dumps({"name": "different_metric", "value": 1.0}),
        encoding="utf-8",
    )
    nonnumeric = tmp_path / "status.json"
    nonnumeric.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    assert (
        normalize_typed_statistic_sidecars(
            {
                "output_files": {
                    "statistic:metric": conflicting.name,
                    "statistic:status": nonnumeric.name,
                }
            },
            tmp_path,
        )
        == []
    )
    assert json.loads(conflicting.read_text())["name"] == "different_metric"
    assert "name" not in json.loads(nonnumeric.read_text())


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
