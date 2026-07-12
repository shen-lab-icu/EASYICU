from __future__ import annotations

import json
from pathlib import Path

from tools.run_research_agent_bench import _artifact_substring_hits, _primary_or


def _write_summary(run_dir: Path, payload: dict) -> None:
    step_dir = run_dir / "steps" / "01_primary" / "outputs"
    step_dir.mkdir(parents=True)
    (step_dir / "step_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_panel(run_dir: Path, value: float) -> None:
    (run_dir / "robustness_panel.json").write_text(
        json.dumps(
            {
                "primary_spec_id": "primary",
                "rows": [
                    {
                        "spec_id": "primary",
                        "axis": "primary",
                        "point_estimate": value,
                        "converged": True,
                    }
                ],
                "primary_point_estimate": value,
            }
        ),
        encoding="utf-8",
    )


def test_artifact_substring_hits_scan_evidence_record_fields() -> None:
    manifest = {
        "evidence": [
            {
                "evidence_id": "table_one__summary",
                "description": "Table one baseline characteristics.",
                "relative_path": "evidence/table_one__summary.csv",
                "kind": "table",
            }
        ]
    }

    assert _artifact_substring_hits(manifest, ["table_one"]) == {"table_one": True}


def test_primary_or_accepts_nested_logistic_model_type(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        {
            "method": None,
            "primary_model": {"model_type": "logistic_regression"},
            "primary_or": 1.1366224560031324,
        },
    )

    assert _primary_or(tmp_path, expected_predictor="sofa2") == 1.1366224560031324


def test_primary_or_leaves_non_logistic_models_unscored(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        {
            "method": None,
            "primary_model": {"model_type": "linear_regression"},
            "primary_or": 1.1366224560031324,
        },
    )

    assert _primary_or(tmp_path, expected_predictor="sofa2") is None


def test_primary_or_prefers_manuscript_facing_panel_primary(
    tmp_path: Path,
) -> None:
    """Q1-style run: panel headline wins over trend and dummy contrast."""
    _write_panel(tmp_path, 1.0184219783832567)
    _write_summary(
        tmp_path,
        {
            "primary_association_estimate": 0.09303264484823856,
            "primary_association_term": "sofa2==1.0",
            "primary_analysis": {"model_type": "logistic_regression"},
            "core_complete_case_model": {
                "fit_method": "Logit(lbfgs)",
                "primary_or": 0.09303264484823856,
            },
            "sofa2_numeric_trend_model": {
                "fit_method": "Logit(lbfgs)",
                "sofa2_or_per_point": 1.2538866781554125,
            },
        },
    )

    assert (
        _primary_or(
            tmp_path,
            expected_predictor="sofa2",
            item_key="analysis_sofa_multisignal_mortality__miiv",
        )
        == 1.0184219783832567
    )


def test_primary_or_skips_continuous_predictor_dummy_level_contrast(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        {
            "primary_association_estimate": 0.09303264484823856,
            "primary_association_term": "sofa2==1.0",
            "primary_analysis": {"model_type": "logistic_regression"},
            "core_complete_case_model": {
                "fit_method": "Logit(lbfgs)",
                "primary_or": 0.09303264484823856,
            },
        },
    )

    assert _primary_or(tmp_path, expected_predictor="sofa2") is None


def test_primary_or_allows_binary_predictor_level_contrast(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        {
            "primary_association_estimate": 1.74,
            "primary_association_term": "vaso==1",
            "primary_analysis": {"model_type": "logistic_regression"},
        },
    )

    assert _primary_or(tmp_path, expected_predictor="vaso") == 1.74


def test_primary_or_leaves_non_or_benchmark_unscored(tmp_path: Path) -> None:
    _write_panel(tmp_path, 1.0184219783832567)
    _write_summary(
        tmp_path,
        {
            "method": "logistic_regression",
            "primary_or": 1.0184219783832567,
        },
    )

    assert (
        _primary_or(
            tmp_path,
            expected_predictor="sofa2",
            item_key="analysis_sofa2_time_to_mortality_cox__miiv",
        )
        is None
    )


def test_bench_item_to_task_surfaces_finding_substrings_as_hazard_key():
    from types import SimpleNamespace

    from tools.run_research_agent_bench import _bench_item_to_task

    item = SimpleNamespace(
        key="sofa2_mortality",
        name="SOFA-2 -> mortality",
        research_question="Is SOFA-2 associated with mortality?",
        expected_finding_substrings=["sofa2", "completeness"],
        expected_artifact_substrings=["table_one", "forest figure"],
    )
    task = _bench_item_to_task(item)
    assert task.task_id == "sofa2_mortality"
    assert task.gold_answer is not None
    assert task.gold_answer.required_warnings == ["sofa2", "completeness"]
    assert task.gold_answer_status == "frozen"


def test_five_dim_scorecard_is_additive_and_robust(tmp_path):
    from types import SimpleNamespace

    from tools.run_research_agent_bench import _five_dim_scorecard

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gates = {
        "required_step_count": 3,
        "completed_step_count": 3,
        "failed_steps": [],
        "execution_complete": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "missing_evidence_count": 0,
        "manuscript_ready": True,
    }
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": gates}), encoding="utf-8"
    )
    (run_dir / "analysis_plan.json").write_text(
        json.dumps(
            {"steps": [{"intent": "table one + forest figure + completeness audit"}]}
        ),
        encoding="utf-8",
    )
    (run_dir / "evidence_audit.json").write_text(
        json.dumps(
            {
                "evidence_complete": True,
                "missing_evidence_count": 0,
                "kinds": {
                    "table": 1,
                    "figure": 1,
                    "metric": 1,
                    "cohort": 1,
                    "model": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "numeric_audit.json").write_text(
        json.dumps({"numeric_verified": True}), encoding="utf-8"
    )
    (run_dir / "claim_ledger.csv").write_text(
        "claim_id,status\nc1,bound\n", encoding="utf-8"
    )

    item = SimpleNamespace(
        key="sofa2_mortality",
        name="SOFA-2 -> mortality",
        research_question="Is SOFA-2 associated with mortality?",
        expected_finding_substrings=["completeness"],
        expected_artifact_substrings=["table_one"],
    )
    card = _five_dim_scorecard(
        run_dir=run_dir, item=item, or_value=0.8, manifest={"findings": []}
    )
    assert card["task_id"] == "sofa2_mortality"
    assert card["tristate"] == "gate_reportable"
    assert card["code"]["level"] == "Full"
    # result-validity stays unscored (no locked reference for legacy items)
    assert card["result_validity"]["level"] is None


def test_five_dim_scorecard_never_raises_on_empty_run(tmp_path):
    from types import SimpleNamespace

    from tools.run_research_agent_bench import _five_dim_scorecard

    item = SimpleNamespace(key="x", expected_finding_substrings=[])
    card = _five_dim_scorecard(
        run_dir=tmp_path, item=item, or_value=None, manifest={"findings": []}
    )
    # empty run dir -> diagnostic_only, but never an exception
    assert card.get("tristate") == "diagnostic_only" or "error" in card


def test_score_arm_reports_active_errors_separately_from_historical_errors(tmp_path):
    from types import SimpleNamespace

    from tools.run_research_agent_bench import _score_arm

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_demo",
                "findings": [
                    {
                        "validator": "old_gate",
                        "severity": "error",
                        "message": "superseded error",
                    }
                ],
                "evidence": [],
                "readiness": {
                    "numeric_error_count": 0,
                    "evidence_error_count": 0,
                    "analysis_error_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        key="demo",
        research_question="demo",
        primary_predictor="",
        expected_or_direction=0,
        expected_finding_substrings=[],
    )

    score = _score_arm(run_dir=run_dir, item=item, label="aware")

    assert score["n_errors"] == 0
    assert score["n_historical_errors"] == 1


def test_score_arm_uses_only_latest_successful_step_records_and_active_evidence(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tools.run_research_agent_bench import _score_arm

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_panel(run_dir, 9.9)
    stale_dir = run_dir / "steps" / "99_stale_downstream" / "outputs"
    stale_dir.mkdir(parents=True)
    (stale_dir / "step_summary.json").write_text(
        json.dumps(
            {
                "method": "filesystem stale workflow",
                "primary_or": 8.8,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_active_view",
                "findings": [],
                "readiness": {
                    "numeric_error_count": 0,
                    "evidence_error_count": 0,
                    "analysis_error_count": 0,
                },
                "per_step_records": [
                    {
                        "step_id": "01_superseded",
                        "status": "ok",
                        "step_summary": {
                            "method": "superseded workflow",
                            "primary_or": 7.7,
                        },
                        "evidence_ids": [
                            "stale_figure",
                            "stale_log",
                            "stale_statistic",
                        ],
                    },
                    {
                        "step_id": "01_superseded",
                        "status": "contract_failed",
                        "step_summary": {
                            "status": "ok",
                            "method": "nested status must not reactivate",
                            "primary_or": 6.6,
                        },
                        "evidence_ids": ["stale_table"],
                    },
                    {
                        "step_id": "02_active",
                        "status": "ok",
                        "step_summary": {
                            "method": "logistic_regression",
                            "title": "active workflow",
                            "primary_or": 1.25,
                        },
                        "evidence_ids": ["active_code", "active_table"],
                    },
                ],
                "evidence": [
                    {
                        "evidence_id": "stale_figure",
                        "kind": "figure",
                        "description": "stale artifact",
                    },
                    {
                        "evidence_id": "stale_log",
                        "kind": "log",
                        "description": "stale artifact",
                    },
                    {
                        "evidence_id": "stale_statistic",
                        "kind": "statistic",
                        "description": "stale artifact",
                    },
                    {
                        "evidence_id": "stale_table",
                        "kind": "table",
                        "description": "stale artifact",
                    },
                    {
                        "evidence_id": "active_code",
                        "kind": "code",
                        "description": "active script",
                    },
                    {
                        "evidence_id": "active_table",
                        "kind": "table",
                        "description": "active artifact",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        key="active_demo",
        research_question="Is sofa2 associated with mortality?",
        primary_predictor="sofa2",
        target_outcome="death",
        expected_or_direction=1,
        expected_finding_substrings=[],
        expected_step_substrings=[
            "active workflow",
            "superseded workflow",
            "filesystem stale workflow",
            "nested status must not reactivate",
        ],
        expected_artifact_substrings=["active artifact", "stale artifact"],
    )

    score = _score_arm(run_dir=run_dir, item=item, label="aware")

    assert score["primary_or"] == 1.25
    assert score["workflow_hits"] == {
        "active workflow": True,
        "superseded workflow": False,
        "filesystem stale workflow": False,
        "nested status must not reactivate": False,
    }
    assert score["artifact_hits"] == {
        "active artifact": True,
        "stale artifact": False,
    }
    assert score["evidence_count"] == 2
    assert score["evidence_kinds"] == {
        "kinds_seen": ["code", "table"],
        "kinds_missing": ["figure", "log", "statistic"],
        "complete": False,
    }


def test_score_arm_legacy_manifest_keeps_filesystem_and_manifest_fallback(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tools.run_research_agent_bench import _score_arm

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_panel(run_dir, 1.4)
    _write_summary(run_dir, {"title": "legacy workflow"})
    evidence = [
        {
            "evidence_id": f"legacy_{kind}",
            "kind": kind,
            "description": "legacy artifact",
        }
        for kind in ("code", "log", "table", "figure", "statistic")
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_legacy",
                "findings": [],
                "readiness": {},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        key="legacy_demo",
        research_question="Is sofa2 associated with mortality?",
        primary_predictor="sofa2",
        target_outcome="death",
        expected_or_direction=1,
        expected_finding_substrings=[],
        expected_step_substrings=["legacy workflow"],
        expected_artifact_substrings=["legacy artifact"],
    )

    score = _score_arm(run_dir=run_dir, item=item, label="aware")

    assert score["primary_or"] == 1.4
    assert score["workflow_hits"] == {"legacy workflow": True}
    assert score["artifact_hits"] == {"legacy artifact": True}
    assert score["evidence_count"] == 5
    assert score["evidence_kinds"]["complete"] is True


def _write_run_status(run_dir: Path, status: str) -> None:
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": status, "gates": {}}), encoding="utf-8"
    )


def _write_audit_log(run_dir: Path, writer_passes: int) -> None:
    lines = []
    for i in range(writer_passes):
        lines.append(
            json.dumps(
                {
                    "timestamp": f"2026-07-03T1{i}:00:00Z",
                    "phase": "writer",
                    "event": "Drafting manuscript scaffold.",
                    "status": "running",
                }
            )
        )
    (run_dir / "audit_log.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_gate_ladder_prefers_run_status_status(tmp_path: Path) -> None:
    from tools.run_research_agent_bench import _gate_ladder

    _write_run_status(tmp_path, "publication_ready")
    assert _gate_ladder(tmp_path, {}) == "publication_ready"


def test_gate_ladder_falls_back_to_readiness_booleans(tmp_path: Path) -> None:
    from tools.run_research_agent_bench import _gate_ladder

    # No run_status.json on disk -> derive from readiness booleans.
    assert (
        _gate_ladder(tmp_path, {"manuscript_ready": True, "publication_ready": False})
        == "manuscript_ready"
    )
    assert (
        _gate_ladder(tmp_path, {"execution_complete": True, "manuscript_ready": False})
        == "analysis_only"
    )


def test_writer_attempts_counts_audit_log_drafts(tmp_path: Path) -> None:
    from tools.run_research_agent_bench import _writer_attempts

    _write_audit_log(tmp_path, writer_passes=4)
    # Manifest gate takes precedence when present...
    assert _writer_attempts(tmp_path, {"writer_attempt_count": 2}) == 2
    # ...otherwise fall back to counting audit-log draft events.
    assert _writer_attempts(tmp_path, {}) == 4


def test_write_stability_report_aggregates_or_spread(tmp_path: Path) -> None:
    from tools.run_research_agent_bench import _write_stability_report

    def _mk_repeat(root: Path, or_value: float) -> Path:
        root.mkdir()
        (root / "ehrflowbench_results.json").write_text(
            json.dumps(
                {
                    "scores": [
                        {
                            "item_key": "E_demo",
                            "aware": {
                                "primary_or": or_value,
                                "gate_status": "publication_ready",
                                "writer_attempts": 1,
                                "workdir": str(root),
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return root

    roots = [
        _mk_repeat(tmp_path / "repeat_01", 1.10),
        _mk_repeat(tmp_path / "repeat_02", 1.30),
        _mk_repeat(tmp_path / "repeat_03", 1.20),
    ]
    _write_stability_report(tmp_path, roots, arms=["aware"])

    report = json.loads((tmp_path / "stability_report.json").read_text())
    item = report["items"]["E_demo"]
    assert item["n_runs"] == 3
    assert item["or_min"] == 1.10
    assert item["or_max"] == 1.30
    assert item["or_median"] == 1.20
    assert abs(item["or_spread"] - 0.20) < 1e-9
    assert item["gate_distribution"] == {"publication_ready": 3}
    assert (tmp_path / "stability_report.md").exists()
