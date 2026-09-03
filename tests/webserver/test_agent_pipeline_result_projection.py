from __future__ import annotations

import json
from pathlib import Path

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
