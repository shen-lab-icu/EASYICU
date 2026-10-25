from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.webserver.agent_pipeline_runs import _table_projection


def _table_record(index: int, name: str, *, step: str = "support") -> dict[str, str]:
    return {
        "kind": "table",
        "evidence_id": f"table_{index}",
        "description": f"Table {name} from step {step}.",
        "relative_path": f"evidence/{name}.csv",
        "produced_by_step": step,
        "producer": "runner",
        "generation_mode": "deterministic_standard",
    }


def test_result_table_projection_keeps_primary_population_and_effect_tables(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    records: list[dict[str, str]] = []
    for index in range(12):
        name = f"support_audit_{index}"
        (evidence_dir / f"{name}.csv").write_text(
            "variable,n_total\nvar,100\n", encoding="utf-8"
        )
        records.append(_table_record(index, name))

    primary_tables = {
        "landmark_population_flow": "stage,n,excluded_from_previous,population_rule\nsource_cohort,100,0,source\ncomplete_case_model_population,60,40,complete\n",
        "time_varying_cox_estimates": "term,hazard_ratio,ci_low,ci_high\nlactate,1.2,1.1,1.3\n",
        "landmark_rcs_contrasts": "contrast_id,estimate,ci_low,ci_high,effect_scale\nq3_vs_q1,1.3,1.1,1.6,odds_ratio\n",
        "robustness_matrix": "specification_id,estimate,ci_low,ci_high,effect_scale\nprimary,1.3,1.1,1.6,odds_ratio\n",
    }
    for offset, (name, csv_text) in enumerate(primary_tables.items(), start=12):
        (evidence_dir / f"{name}.csv").write_text(csv_text, encoding="utf-8")
        records.append(_table_record(offset, name, step="primary_model"))
    (evidence_dir / "evidence_index.json").write_text(
        json.dumps(records), encoding="utf-8"
    )

    projection = _table_projection(run_dir)

    names = {table["name"] for table in projection["tables"]}
    assert len(names) == 12
    assert "landmark_population_flow.csv" in names
    assert "time_varying_cox_estimates.csv" in names
    assert "landmark_rcs_contrasts.csv" in names
    assert "robustness_matrix.csv" in names
    assert next(table for table in projection["tables"] if table["name"] == "landmark_rcs_contrasts.csv")["produced_by_step"] == "primary_model"


def test_a_prediction_run_projects_its_model_tables_before_supporting_audits(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    records: list[dict[str, str]] = []
    for index in range(12):
        name = f"support_audit_{index}"
        (evidence_dir / f"{name}.csv").write_text(
            "variable,n_total\nvar,100\n", encoding="utf-8"
        )
        records.append(_table_record(index, name))
    model_tables = {
        "prediction_performance": "primary_performance",
        "internal_validation": "internal_validation",
    }
    for offset, (name, step) in enumerate(model_tables.items(), start=12):
        (evidence_dir / f"{name}.csv").write_text(
            "auroc,brier_score\n0.8,0.1\n", encoding="utf-8"
        )
        records.append(_table_record(offset, name, step=step))
    (evidence_dir / "evidence_index.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    plan = {
        "analysis_type": "prediction_model",
        "steps": [
            {
                "step_id": "primary_performance",
                "scientific_action_id": "prediction.discrimination_calibration",
            },
            {
                "step_id": "internal_validation",
                "scientific_action_id": "prediction.internal_validation",
            },
        ],
    }

    names = [table["name"] for table in _table_projection(run_dir, plan=plan)["tables"]]

    assert len(names) == 12
    assert names[:2] == ["prediction_performance.csv", "internal_validation.csv"]
    # Read for a study that is not a prediction study, the order is unchanged.
    unchanged = _table_projection(run_dir)
    assert (
        _table_projection(run_dir, plan={**plan, "analysis_type": "association"})
        == unchanged
    )
    assert "prediction_performance.csv" not in {
        table["name"] for table in unchanged["tables"]
    }


def _registered_tables(tmp_path: Path, contents: dict[str, str]) -> list[dict]:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    records = []
    for index, (name, text) in enumerate(contents.items()):
        path = evidence_dir / f"{name}.csv"
        path.write_text(text, encoding="utf-8")
        records.append({**_table_record(index, name),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_index(tmp_path, records)
    return records


def _write_index(run_dir: Path, records: list[dict]) -> None:
    (run_dir / "evidence/evidence_index.json").write_text(json.dumps(records), encoding="utf-8")


def test_source_copies_do_not_displace_the_original_or_cohort_audits(tmp_path: Path) -> None:
    contents = {f"adjusted_association_estimates_source_data_{i}": "n,estimate\n24,1.25\n"
                for i in range(14)}
    originals = ["adjusted_association_estimates", "absolute_risk_context", "cohort_flow",
                 "table_one", "analytic_denominators", "measurement_process_audit"]
    contents.update({name: "n,estimate\n24,1.25\n" for name in originals})
    records = _registered_tables(tmp_path, contents)
    # Publication copies also have an explicit description, without a source_data name.
    records[0]["description"] = "Source data copied beside a promoted publication figure."
    records[0]["relative_path"] = "evidence/promoted.csv"
    (tmp_path / "evidence/promoted.csv").write_text(contents[next(iter(contents))], encoding="utf-8")
    _write_index(tmp_path, records)

    result = _table_projection(tmp_path)

    assert len(result["tables"]) == 12
    assert {table["name"] for table in result["tables"][:6]} == {f"{name}.csv" for name in originals}
    assert all(not table["is_source_copy"] for table in result["tables"][:6])
    assert all(table["is_source_copy"] for table in result["tables"][6:])
    assert result["unpreviewed_table_count"] == 8


def test_wide_risk_table_keeps_events_denominators_and_intervals(tmp_path: Path) -> None:
    headers = ["exposure", "group_type", "group_value", "label", "n_denominator",
               "source_measured_column", "source_count_column", "median", "q25", "q75",
               "minimum", "maximum", "outcome", "outcome_n", "outcome_missing_n",
               "outcome_event_n", "outcome_risk_pct", "outcome_risk_ci_low_pct",
               "outcome_risk_ci_high_pct"]
    _registered_tables(tmp_path, {"absolute_risk_context": ",".join(headers) + "\n" +
                                ",".join(["synthetic"] * len(headers)) + "\n"})

    table = _table_projection(tmp_path)["tables"][0]

    assert len(table["headers"]) == 12
    assert set(headers[12:] + ["exposure", "group_value", "n_denominator"]) <= set(table["headers"])
    assert table["preview_columns_truncated"] is True


@pytest.mark.parametrize("count,truncated", [(29, False), (30, False), (31, True)])
def test_row_omission_is_declared_only_when_an_extra_row_exists(tmp_path: Path, count: int, truncated: bool) -> None:
    _registered_tables(tmp_path, {"audit": "group,n\n" + "synthetic,24\n" * count})
    table = _table_projection(tmp_path)["tables"][0]
    assert len(table["rows"]) == min(count, 30)
    assert table["preview_truncated"] is truncated


def test_source_is_bound_to_the_registered_file_digest_and_existing_preview(tmp_path: Path) -> None:
    from easyicu.webserver.research_evidence_preview import build_evidence_preview

    records = _registered_tables(tmp_path, {"adjusted_association_estimates": "estimate,n\n1.23456789,24\n"})
    records[0]["script_evidence_id"] = "synthetic_code"
    _write_index(tmp_path, records)
    table = _table_projection(tmp_path)["tables"][0]
    assert table["sha256"] == records[0]["sha256"]
    assert table["source_status"] == "current"
    preview = build_evidence_preview(tmp_path, table["evidence_id"], table["sha256"])
    assert preview["renderer"] == "table"
    assert preview["rows"] == [["1.23456789", "24"]]
    assert "source_json_pointer" not in table


@pytest.mark.parametrize("digest", ["f" * 64, "not-a-digest"])
def test_stale_or_invalid_registered_digest_cannot_supply_result_values(tmp_path: Path, digest: str) -> None:
    records = _registered_tables(tmp_path, {"estimate": "estimate,n\n1.25,24\n"})
    records[0]["sha256"] = digest
    _write_index(tmp_path, records)
    assert _table_projection(tmp_path)["tables"] == []


@pytest.mark.parametrize("mode", ["missing_digest", "duplicate_id", "invalid_id"])
def test_unbound_source_never_acquires_a_clickable_digest(tmp_path: Path, mode: str) -> None:
    records = _registered_tables(tmp_path, {"estimate": "estimate,n\n1.25,24\n"})
    if mode == "missing_digest":
        records[0].pop("sha256")
    elif mode == "duplicate_id":
        records.append(dict(records[0]))
    else:
        records[0]["evidence_id"] = "../invalid"
    _write_index(tmp_path, records)
    result = _table_projection(tmp_path)
    assert result["tables"]
    assert all(not table.get("sha256") for table in result["tables"])


def test_late_identifier_column_and_missing_files_stay_unavailable(tmp_path: Path) -> None:
    headers = [f"aggregate_{i}" for i in range(13)] + ["patient_id"]
    records = _registered_tables(tmp_path, {"private": ",".join(headers) + "\n"})
    records.append(_table_record(1, "missing"))
    _write_index(tmp_path, records)
    result = _table_projection(tmp_path)
    assert result["tables"] == []
    assert result["skipped_identifier_tables"] == 1
    assert result["unpreviewed_table_count"] == 2


def test_wide_estimates_keep_model_identity_without_displacing_contrast_columns(tmp_path: Path) -> None:
    columns = ["fit_status", "estimate", "ci_low", "ci_high", "effect_scale", "is_primary_contrast",
               "exposure", "exposure_level", "reference_level", "contrast", "analysis_role", "n", "model_id"]
    row = ["fitted", "1.25", "1.01", "1.55", "odds_ratio", "True", "marker", "B", "A", "B vs A", "primary", "24"]
    _registered_tables(tmp_path, {"adjusted_association_estimates": ",".join(columns) + "\n" +
                                "\n".join(",".join([*row, model]) for model in ("model_a", "model_b")) + "\n"})
    table = _table_projection(tmp_path)["tables"][0]
    assert table["headers"] == columns[:12]
    assert table["row_model_ids"] == ["model_a", "model_b"]
