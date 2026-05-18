from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.webapp import research_agent as ra_page


def test_module_export_folder_builds_filtered_stay_level_cohort(tmp_path: Path) -> None:
    folder = tmp_path / "mimiciv_export"
    (folder / "sepsis3_sofa2").mkdir(parents=True)
    (folder / "outcome").mkdir()
    (folder / "vitals").mkdir()

    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sep3_sofa2": [1, 0, 1],
    }).to_parquet(folder / "sepsis3_sofa2" / "sep3_sofa2.parquet", index=False)
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "death": [1, 0, 0],
    }).to_parquet(folder / "outcome" / "death.parquet", index=False)
    pd.DataFrame({
        "stay_id": [1, 1, 2, 3],
        "charttime": pd.to_datetime([
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 00:00",
            "2024-01-01 00:00",
        ]),
        "hr": [70, 80, 90, 100],
    }).to_parquet(folder / "vitals" / "hr.parquet", index=False)

    selected = [
        folder / "sepsis3_sofa2" / "sep3_sofa2.parquet",
        folder / "outcome" / "death.parquet",
        folder / "vitals" / "hr.parquet",
    ]
    cohort = ra_page._build_stay_level_from_module_folder(
        folder=folder,
        selected_files=selected,
        id_col="stay_id",
        filter_spec=(folder / "sepsis3_sofa2" / "sep3_sofa2.parquet", "sep3_sofa2", "nonzero / true", ""),
    )

    assert set(cohort["stay_id"]) == {1, 3}
    assert cohort.loc[cohort["stay_id"] == 1, "hr"].iloc[0] == 80
    assert set(["sep3_sofa2", "death", "hr"]) <= set(cohort.columns)


def test_infers_sepsis_filter_defaults_from_question(tmp_path: Path) -> None:
    path = tmp_path / "sep3_sofa2.parquet"
    pd.DataFrame({"stay_id": [1], "sep3_sofa2": [1]}).to_parquet(path, index=False)
    summary = ra_page._parquet_file_summary(path)

    filter_path, filter_col = ra_page._infer_filter_defaults(
        [summary],
        question="Do sepsis patients have higher hospital mortality?",
    )

    assert filter_path == path
    assert filter_col == "sep3_sofa2"


def test_scans_research_agent_history_from_final_and_partial_manifests(tmp_path: Path) -> None:
    workdir = tmp_path / "webapp"
    final_dir = workdir / "run_20260101T000000_final"
    partial_dir = workdir / "run_20260102T000000_partial"
    final_dir.mkdir(parents=True)
    partial_dir.mkdir(parents=True)

    (final_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run_final",
            "research_question": "Does lactate predict mortality?",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "per_step_records": [
                {"step_id": "00_probe", "status": "ok"},
                {"step_id": "01_model", "status": "ok"},
            ],
            "evidence": [
                {"evidence_id": "fig1", "kind": "figure"},
                {"evidence_id": "tbl1", "kind": "table"},
            ],
            "findings": [{"severity": "warning"}],
        }),
        encoding="utf-8",
    )
    (partial_dir / "manifest_partial.json").write_text(
        json.dumps({
            "run_id": "run_partial",
            "research_question": "Build a cohort audit.",
            "started_at": "2026-01-02T00:00:00+00:00",
            "per_step_records": [
                {"step_id": "00_probe", "status": "ok"},
                {"step_id": "01_table", "status": "execution_failed"},
            ],
            "evidence": [],
            "findings": [{"severity": "error"}],
        }),
        encoding="utf-8",
    )

    rows = ra_page._scan_research_agent_runs(workdir)

    assert {row["run_id"] for row in rows} == {"run_final", "run_partial"}
    partial = next(row for row in rows if row["run_id"] == "run_partial")
    assert partial["manifest_partial"] is True
    assert partial["step_ok"] == 1
    assert partial["step_failed"] == 1
    final = next(row for row in rows if row["run_id"] == "run_final")
    assert final["figure_count"] == 1
    assert final["table_count"] == 1


def test_run_summary_counts_failed_steps_and_missing_outputs(tmp_path: Path) -> None:
    manifest = {
        "run_id": "run_failed",
        "per_step_records": [
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "01_clustering", "status": "execution_failed"},
            {"step_id": "02_plot", "status": "blocked_by_concept_audit"},
        ],
        "evidence": [{"evidence_id": "log1", "kind": "log"}],
        "findings": [{"severity": "error"}, {"severity": "warning"}],
    }

    summary = ra_page._run_summary_from_manifest(tmp_path / "run_failed", manifest, partial=False)

    assert summary["step_ok"] == 1
    assert summary["step_failed"] == 2
    assert summary["figure_count"] == 0
    assert summary["table_count"] == 0
    assert summary["finding_errors"] == 1
    assert summary["finding_warnings"] == 1


def test_step_evidence_links_explicit_ids_and_produced_by_step(tmp_path: Path) -> None:
    manifest = {
        "evidence": [
            {
                "evidence_id": "script_1",
                "kind": "code",
                "produced_by_step": "01_model",
            },
            {
                "evidence_id": "table_1",
                "kind": "table",
                "produced_by_step": "01_model",
            },
            {
                "evidence_id": "figure_2",
                "kind": "figure",
                "produced_by_step": "02_plot",
            },
        ],
    }
    record = {"step_id": "01_model", "evidence_ids": ["script_1"]}

    linked = ra_page._evidence_for_step(record, manifest)

    assert [rec["evidence_id"] for rec in linked] == ["script_1", "table_1"]


def test_step_view_hides_raw_logs_json_and_code_from_result_artifacts() -> None:
    records = [
        {"kind": "table", "relative_path": "evidence/table_one.csv"},
        {"kind": "figure", "relative_path": "evidence/plot.png"},
        {"kind": "log", "relative_path": "evidence/log_run__run.log"},
        {"kind": "code", "relative_path": "evidence/code_analysis.py"},
        {"kind": "statistic", "relative_path": "evidence/step_summary.json"},
    ]

    visible = [rec for rec in records if ra_page._is_user_facing_step_artifact(rec)]
    debug = [rec for rec in records if ra_page._is_debug_artifact(rec)]

    assert [rec["kind"] for rec in visible] == ["table", "figure"]
    assert [rec["kind"] for rec in debug] == ["log", "code", "statistic"]


def test_research_grounding_json_payloads_load_from_registered_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    payload_path = evidence_dir / "analysis_plan__analysis_plan.json"
    payload_path.write_text(
        json.dumps({"steps": [{"step_id": "01_table", "intent": "Summarize cohort."}]}),
        encoding="utf-8",
    )
    manifest = {
        "evidence": [
            {
                "evidence_id": "analysis_plan",
                "kind": "log",
                "relative_path": "evidence/analysis_plan__analysis_plan.json",
            }
        ]
    }

    payload = ra_page._json_payload_for_evidence(run_dir, manifest, "analysis_plan")

    assert payload["steps"][0]["step_id"] == "01_table"


def test_table_preview_reads_only_preview_rows(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)

    preview = ra_page._read_table_preview(path, n=10)

    assert preview.shape == (10, 2)
    assert preview["a"].tolist() == list(range(10))


def test_resolve_llm_disables_qwen3_thinking_for_openai_compatible_endpoint() -> None:
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    client = ra_page._resolve_llm(
        {"MockLLMClient": object, "OpenAIClient": FakeOpenAIClient},
        "Custom OpenAI-compatible",
        api_key="vllm",
        model="qwen3-8b",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert client.kwargs["extra_body"]["enable_thinking"] is False
    assert client.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_run_pipeline_enables_deterministic_planner_fallback(tmp_path: Path) -> None:
    class FakePipeline:
        init_kwargs = {}

        def __init__(self, **kwargs):
            FakePipeline.init_kwargs = kwargs

        def run(self, **kwargs):
            return {"ok": True, "run_kwargs": kwargs}

    result = ra_page._run_pipeline(
        handles={"ResearchAgentPipeline": FakePipeline},
        cohort=pd.DataFrame({"stay_id": [1]}),
        skill_key=None,
        question="Does lactate predict mortality?",
        target_outcome=None,
        workdir=tmp_path,
        llm=object(),
        disable_icu_context=False,
    )

    assert FakePipeline.init_kwargs["enable_deterministic_planner_fallback"] is True
    assert FakePipeline.init_kwargs["enable_deterministic_code_fallback"] is True
    assert result["ok"] is True


def test_loaded_concepts_handoff_respects_current_patient_ids() -> None:
    loaded = {
        "age": pd.DataFrame({"stay_id": [1, 2, 3], "age": [60, 70, 80]}),
        "hr": pd.DataFrame({
            "stay_id": [1, 1, 2, 3],
            "time": [0, 1, 0, 0],
            "hr": [70, 75, 80, 90],
        }),
    }

    cohort = ra_page._stay_level_from_loaded_concepts(
        loaded,
        id_col="stay_id",
        patient_ids=[1, 3],
    )

    assert cohort is not None
    assert set(cohort["stay_id"]) == {1, 3}
    assert cohort.loc[cohort["stay_id"] == 1, "hr"].iloc[0] == 75
