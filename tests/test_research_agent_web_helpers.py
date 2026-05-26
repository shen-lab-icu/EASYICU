from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webapp import agent_workbench as wb_page
from easyicu.webapp import research_agent as ra_page
from easyicu.webapp.agent_workbench import (
    _demo_state,
    _result_cards_from_evidence,
    _resolve_workbench_state,
    _step_button_label,
    _step_flow_html,
    _step_legend_html,
    build_workbench_state_from_manifest,
)


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


def test_review_decision_roundtrip_updates_run_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_reviewed"
    run_dir.mkdir()
    manifest = {"run_id": "run_reviewed", "per_step_records": [], "evidence": [], "findings": []}

    path = ra_page._write_review_decision(
        run_dir,
        decision="approved",
        note="Checked evidence links.",
        manifest=manifest,
    )
    loaded = ra_page._load_review_decision(run_dir)
    summary = ra_page._run_summary_from_manifest(run_dir, manifest, partial=False)

    assert path.name == "review_decision.json"
    assert loaded["decision"] == "approved"
    assert loaded["note"] == "Checked evidence links."
    assert summary["review_decision"] == "approved"


def test_execution_preflight_contract_and_signature_change_with_question(tmp_path: Path) -> None:
    cohort = pd.DataFrame({"stay_id": [1, 2], "lactate": [2.0, 3.5]})

    contract = ra_page._build_execution_preflight_contract(
        free_question="Build a mortality model.",
        target_outcome="death",
        cohort=cohort,
        cohort_label="synthetic",
        llm_choice="OpenAI",
        model="gpt-test",
        workdir_text=str(tmp_path),
        stop_after_analysis=True,
        force_manuscript=False,
        template_key="prediction",
        language="en",
    )
    changed = dict(contract)
    changed["question"] = "Run a data quality audit."

    assert contract["external_llm"] is True
    assert contract["cohort_rows"] == 2
    assert contract["template_contract"]["label"] == "Prediction model"
    assert any("manifest.json" in target for target in contract["write_targets"])
    assert ra_page._preflight_signature(contract) != ra_page._preflight_signature(changed)


def test_workbench_state_builds_from_real_run_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260103T000000_abcd12"
    run_dir.mkdir()
    (run_dir / "analysis.py").write_text("print('ok')\n", encoding="utf-8")
    (run_dir / "review_decision.json").write_text(
        json.dumps({"decision": "repair_requested", "note": "Fix model step."}),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({
            "status": "diagnostic_only",
            "gates": {
                "execution_complete": False,
                "evidence_complete": True,
                "numeric_verified": False,
            },
        }),
        encoding="utf-8",
    )
    manifest = {
        "run_id": "run_manifest_bound",
        "research_question": "Does lactate predict mortality?",
        "context_path": "context.json",
        "plan_path": "analysis_plan.json",
        "per_step_records": [
            {"step_id": "01_table_one", "status": "ok", "generation_mode": "system"},
            {
                "step_id": "02_model_training",
                "status": "execution_failed",
                "returncode": 1,
                "evidence_ids": ["script_1"],
            },
        ],
        "evidence": [
            {
                "evidence_id": "script_1",
                "kind": "code",
                "relative_path": "analysis.py",
                "produced_by_step": "02_model_training",
            },
            {
                "evidence_id": "fig_1",
                "kind": "figure",
                "relative_path": "figures/roc.svg",
                "produced_by_step": "02_model_training",
            },
        ],
        "findings": [
            {"severity": "error", "validator": "runner", "message": "model failed"},
            {"severity": "warning", "validator": "cohort_auditor", "message": "missingness high"},
        ],
    }

    state = build_workbench_state_from_manifest(run_dir, manifest, partial=False)

    assert state["run_id"] == "run_manifest_bound"
    assert state["status"] == "blocked"
    assert [step["status"] for step in state["steps"]] == ["ok", "fail"]
    assert "print('ok')" in state["code"]
    assert state["steps"][1]["step_id"] == "02_model_training"
    assert state["step_details"][1]["code_path"] == "analysis.py"
    assert state["step_details"][1]["results"][0]["kind"] == "figure"
    assert state["source_label"] == "Real manifest"
    assert state["audit"]["counts"] == {"errors": 1, "warnings": 1, "info": 0}
    assert any(gate["label"] == "numeric verified" and gate["ok"] is False for gate in state["audit"]["gates"])
    assert state["results"][0]["kind"] == "figure"
    assert state["evidence"][0]["tag"] == "code"
    assert state["summary_outputs"][0]["kind"] == "figure"
    assert state["execution_contract"]["workdir"] == str(run_dir)
    assert state["review_gate_actions"][0]["state"] == "blocked"
    assert state["step_details"][1]["step_contract"]["method"]["label"] == "Statistical association model"
    assert state["step_details"][1]["step_contract"]["outputs"][0]["path"] == "analysis.py"
    assert any(item["ok"] is False for item in state["step_details"][1]["step_contract"]["checkpoints"])
    assert state["review_decisions"][0]["label"] == "Saved: repair_requested"
    assert state["audit_tasks"][0]["tone"] == "danger"


def test_demo_workbench_summary_slots_do_not_expose_fake_metrics() -> None:
    state = _demo_state("en")
    payload = json.dumps(state, ensure_ascii=False)

    assert state["is_demo"] is True
    assert "does not fabricate cohort metrics" in state["summary_outputs"][1]["sub"]
    assert state["execution_contract"]["provider"] == "No LLM call"
    assert state["review_gate_actions"][0]["state"] == "ready"
    assert state["review_decisions"][0]["label"] == "Preview only"
    assert state["audit_tasks"][0]["title"] == "Open a real manifest"
    assert state["step_details"][0]["step_contract"]["method"]["label"] == "Demo method slot"
    assert "AUC" not in payload
    assert "Brier" not in payload
    assert "2,481" not in payload
    assert all(not result.get("svg") for result in state["results"])
    assert all("No generated output" in result.get("preview_html", "") for result in state["results"])


def test_workbench_does_not_auto_bind_demo_preview(monkeypatch) -> None:
    class _StreamlitStub:
        session_state = {
            "entry_mode": "demo",
            "_agent_workbench": {
                "is_demo": True,
                "steps": [{"label": "Fake", "status": "running"}],
            },
        }

    monkeypatch.setattr(wb_page, "st", _StreamlitStub)

    assert _resolve_workbench_state("en") == {}


def test_empty_workbench_removes_ambiguous_open_latest_run_action() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")

    assert "Open latest run" not in source
    assert "打开最近 run" not in source
    assert "_eu_wb_empty_latest" not in source
    assert "_latest_real_workbench_state" not in source
    assert "Open local saved runs" in source
    assert "查看本机历史运行" in source
    assert "will not upload run history" in source
    assert "不会上传 run 历史" in source
    assert "Open manifest history" not in source
    assert "打开 manifest 历史" not in source
    assert "_eu_wb_empty_history" in source


def test_research_agent_history_copy_is_local_only() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    i18n_source = Path(ra_page.__file__).with_name("i18n.py").read_text(encoding="utf-8")

    assert "Local run history is loaded on demand from this workdir only" in source
    assert "not uploaded to GitHub" in source
    assert "本机运行历史只会按需从当前工作目录读取" in source
    assert "Load local recent runs" in source
    assert "加载本机最近 run" in source
    assert "'ra_history_title': 'Local analysis records'" in i18n_source
    assert "'ra_history_title': '本机分析记录'" in i18n_source


def test_empty_workbench_actions_use_responsive_nowrap_buttons() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")

    assert 'st.container(key=f"eu_wb_empty_actions_{summary}")' in source
    assert "st.columns(2)" in source
    assert "st.columns([1.4, 1.75, 6.15])" not in source
    assert "st-key-eu_wb_empty_actions" in css
    assert 'class*="st-key-_eu_wb_empty_"' in css
    assert "max-width: 560px" in css
    assert "height: 38px" in css
    assert "white-space: nowrap" in css
    assert "text-overflow: ellipsis" in css
    assert "@media (max-width: 560px)" in css


def test_real_figure_result_card_uses_bound_artifact_not_placeholder_chart(tmp_path: Path) -> None:
    fig = tmp_path / "figures" / "roc.svg"
    fig.parent.mkdir()
    fig.write_text(
        '<svg viewBox="0 0 10 10"><path d="M1 9 L5 5 L9 1" /></svg>',
        encoding="utf-8",
    )
    evidence = [{
        "evidence_id": "fig_1",
        "kind": "figure",
        "relative_path": "figures/roc.svg",
        "produced_by_step": "02_plot",
    }]

    cards = _result_cards_from_evidence(evidence, run_dir=tmp_path, lang="en")

    assert cards[0]["metric"] == "rendered"
    assert '<svg viewBox="0 0 10 10"' in cards[0]["preview_html"]
    assert "render_tile" not in cards[0]["preview_html"]


def test_live_workbench_state_builds_from_progress_events(tmp_path: Path) -> None:
    progress_events = [
        {
            "stage": "run",
            "message": "Starting research-agent run.",
            "status": "running",
            "timestamp": "2026-05-22T00:00:00+00:00",
        },
        {
            "stage": "step",
            "step_id": "01_table_one",
            "message": "Step 1/3 started: 01_table_one.",
            "status": "running",
            "current_step": 1,
            "total_steps": 3,
            "timestamp": "2026-05-22T00:00:01+00:00",
        },
    ]
    state = build_workbench_state_from_manifest(
        tmp_path / "run_pending_webapp",
        {
            "run_id": "run_pending_webapp",
            "research_question": "Short live run",
            "per_step_records": [],
            "evidence": [],
            "findings": [],
        },
        partial=True,
        progress_events=progress_events,
    )
    dag_html = _step_flow_html(state, "en")

    assert state["status"] == "running"
    assert state["steps"][-1]["step_id"] == "01_table_one"
    assert state["steps"][-1]["status"] == "running"
    assert "eu-agent-flow-node running" in dag_html
    assert "Evidence review" in dag_html


def test_workbench_step_labels_explain_status_semantics() -> None:
    assert "NEEDS FIX" in _step_button_label({"label": "Model", "status": "fail", "sub": "ValueError"}, 3, "en")
    assert "QUEUED" in _step_button_label({"label": "Findings", "status": "pending"}, 8, "en")

    legend_html = _step_legend_html("en")
    assert "Done" in legend_html
    assert "Running" in legend_html
    assert "Queued" in legend_html
    assert "Needs fix" in legend_html
    assert "Retrying" in legend_html


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


def test_resolve_llm_disables_qwen3_thinking_for_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(ra_page.socket, "create_connection", lambda *_args, **_kwargs: _Socket())

    client = ra_page._resolve_llm(
        {"MockLLMClient": object, "OpenAIClient": FakeOpenAIClient},
        "Custom OpenAI-compatible",
        api_key="vllm",
        model="qwen3-8b",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert client.kwargs["extra_body"]["enable_thinking"] is False
    assert client.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_local_llm_endpoint_health_check_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refused(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(ra_page.socket, "create_connection", _refused)

    with pytest.raises(RuntimeError, match="Local LLM endpoint is unreachable"):
        ra_page._assert_local_llm_endpoint_reachable("http://127.0.0.1:8787/v1", timeout=0.01)


def test_local_llm_endpoint_health_check_ignores_remote_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected(*_args, **_kwargs):  # pragma: no cover - fails if called
        raise AssertionError("remote endpoints should not be socket-probed")

    monkeypatch.setattr(ra_page.socket, "create_connection", _unexpected)

    ra_page._assert_local_llm_endpoint_reachable("https://api.openai.com/v1", timeout=0.01)


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
