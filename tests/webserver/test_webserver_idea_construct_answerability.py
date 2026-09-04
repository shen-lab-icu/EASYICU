import json
from pathlib import Path

import pytest

from easyicu.webserver.ideas import mining as idea_mining
from easyicu.webserver.pi_copilot import tools as tool_module


def _metadata_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    topic: str,
) -> dict:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    return idea_mining.mine_ideas(
        {
            "source_type": "manual",
            "metadata_only": True,
            "topic": topic,
            "title": "One-line Idea Mining seed",
            "excerpt": topic,
        }
    )


def _export(
    tmp_path: Path,
    files: dict[str, str],
    columns: dict[str, list[str]],
) -> tuple[dict, dict]:
    root = tmp_path / "clinical_export"
    root.mkdir()
    file_rows = []
    total_rows = 0
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
        rows = max(len(content.strip().splitlines()) - 1, 0)
        total_rows += rows
        file_rows.append(
            {
                "file": name,
                "module": name.removesuffix(".csv"),
                "columns": columns[name],
                "rows": rows,
            }
        )
    source = {
        "id": "real-clinical-export",
        "label": "Clinical export",
        "database": "miiv",
        "path": str(root),
    }
    desc = {
        "ok": True,
        "path": str(root),
        "database": "miiv",
        "files": file_rows,
        "summary": {"stays": 2, "modules": len(files), "total_rows": total_rows},
    }
    return source, desc


def test_metadata_only_copilot_projection_explains_constructs_in_chinese(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _metadata_run(
        tmp_path,
        monkeypatch,
        topic="ICU患者停镇静药后仍迟迟不清醒，这里面有没有值得研究的方向？",
    )

    projection = tool_module._idea_projection(run)
    rows = projection["idea"]["construct_answerability"]
    by_id = {row["construct_id"]: row for row in rows}

    assert by_id["sedation_discontinuation"]["verdict"] == "needs_review"
    assert by_id["awakening_after_sedation"]["source_state"] == "source_not_selected"
    assert by_id["sedation_discontinuation"]["user_facing"]["label"] == (
        "可以构造，但需要确认定义"
    )
    assert "最后一条给药记录" in by_id["sedation_discontinuation"][
        "semantic_warning"
    ]


def test_source_feasibility_does_not_treat_last_sedative_record_as_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _metadata_run(
        tmp_path,
        monkeypatch,
        topic="研究ICU患者停镇静药后多久清醒",
    )
    export = _export(
        tmp_path,
        files={
            "sedation.csv": (
                "stay_id,charttime,propofol_rate\n1,0,10\n2,0,12\n"
            ),
            "neurology.csv": "stay_id,charttime,rass\n1,4,0\n2,6,-2\n",
        },
        columns={
            "sedation.csv": ["stay_id", "charttime", "propofol_rate"],
            "neurology.csv": ["stay_id", "charttime", "rass"],
        },
    )

    feasibility = idea_mining.bounded_sample_feasibility(
        {
            "run_id": run["run_id"],
            "idea_id": run["selected_idea_id"],
            "concept_bindings": {
                "primary_exposure": "propofol_rate",
                "outcome": "rass",
                "time_zero": "propofol_rate",
                "covariates": [],
            },
        },
        export=export,
    )

    assert feasibility["schema_version"] == (
        "easyicu.web_idea_bounded_sample_feasibility/2"
    )
    assert feasibility["status"] == "needs_review"
    assert "research_construct_requires_definition_or_materialization" in (
        feasibility["blockers"]
    )
    by_id = {
        row["construct_id"]: row for row in feasibility["construct_answerability"]
    }
    assert by_id["sedation_discontinuation"]["resolution_kind"] == (
        "event_reconstructable"
    )
    assert by_id["sedation_discontinuation"]["materialized"] is False
    dumped = json.dumps(feasibility, ensure_ascii=False)
    assert str(export[0]["path"]) not in dumped
    assert "stay_id" not in dumped


def test_materialized_fluid_balance_can_reach_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _metadata_run(
        tmp_path,
        monkeypatch,
        topic="研究ICU入科后24小时累计液体平衡与住院死亡",
    )
    export = _export(
        tmp_path,
        files={
            "fluid.csv": (
                "stay_id,charttime,fluid_balance_cumulative\n1,24,1200\n2,24,-300\n"
            ),
            "outcome.csv": "stay_id,charttime,death\n1,72,0\n2,96,1\n",
        },
        columns={
            "fluid.csv": ["stay_id", "charttime", "fluid_balance_cumulative"],
            "outcome.csv": ["stay_id", "charttime", "death"],
        },
    )

    feasibility = idea_mining.bounded_sample_feasibility(
        {
            "run_id": run["run_id"],
            "idea_id": run["selected_idea_id"],
            "concept_bindings": {
                "primary_exposure": "fluid_balance_cumulative",
                "outcome": "death",
                "time_zero": "fluid_balance_cumulative",
                "covariates": [],
            },
        },
        export=export,
    )

    assert feasibility["status"] == "ready"
    by_id = {
        row["construct_id"]: row for row in feasibility["construct_answerability"]
    }
    assert by_id["cumulative_fluid_balance"]["resolution_kind"] == (
        "validated_derived"
    )
    assert by_id["cumulative_fluid_balance"]["materialized"] is True
    assert by_id["death"]["verdict"] == "ready"


def test_legacy_feasibility_receipt_cannot_bypass_construct_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    adjudication = {
        "prior_art_decision": "differentiated",
        "prior_art_adjudication_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        idea_mining,
        "prior_art_adjudication_binding",
        lambda run_id, idea_id: adjudication,
    )
    run_id = "idea_legacy_receipt"
    idea_id = "idea_sedation"
    run_dir = idea_mining._run_dir(run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "bounded_sample_feasibility.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.web_idea_bounded_sample_feasibility/1",
                "run_id": run_id,
                "idea_id": idea_id,
                "status": "ready",
                "prior_art_adjudication_binding": adjudication,
                "construct_answerability": [
                    {"construct_id": "sedation_discontinuation", "verdict": "ready"}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(idea_mining.IdeaMiningWebError) as exc:
        idea_mining.idea_execution_readiness_binding(run_id, idea_id)

    assert exc.value.detail["error"] == "idea_source_feasibility_schema_outdated"
