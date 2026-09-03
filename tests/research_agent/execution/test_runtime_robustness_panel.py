from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.robustness import runtime_panel


def _panel() -> SimpleNamespace:
    return SimpleNamespace(
        primary_spec_id="primary",
        n_variants=2,
        range_low=0.8,
        range_high=1.4,
    )


def test_finalizer_returns_findings_and_manifest_projection(
    monkeypatch, tmp_path: Path
) -> None:
    locked_spec = SimpleNamespace(spec_id="alt_locked")
    plan = SimpleNamespace(robustness_specs=[], cohort=object())
    panel = _panel()
    captured = {}

    monkeypatch.setattr(
        runtime_panel,
        "robustness_specs_for_execution",
        lambda **_kwargs: [locked_spec],
    )

    def _fit(**kwargs):
        captured["fit"] = kwargs
        return [SimpleNamespace(spec_id="alt_locked")], ["adapter warning"]

    monkeypatch.setattr(runtime_panel, "fit_robustness_rows_from_records", _fit)
    monkeypatch.setattr(
        runtime_panel,
        "build_robustness_panel_from_records",
        lambda **kwargs: captured.setdefault("build", kwargs) and panel,
    )
    monkeypatch.setattr(
        runtime_panel,
        "write_robustness_panel",
        lambda **kwargs: captured.setdefault("write", kwargs),
    )
    monkeypatch.setattr(runtime_panel, "unexecuted_locked_spec_ids", lambda _panel: [])

    result = runtime_panel.finalize_run_robustness_panel(
        run_dir=tmp_path,
        plan=plan,
        per_step_records=[{"step_id": "01", "status": "ok"}],
        cohort_path=tmp_path / "cohort.parquet",
        context=SimpleNamespace(),
        evidence=SimpleNamespace(),
        prompt_pack_version="prompt-v1",
    )

    assert [(item.validator, item.severity) for item in result.findings] == [
        ("robustness_panel", "warning"),
        ("robustness_estimator", "warning"),
    ]
    assert captured["fit"]["allow_implicit_cohort_refit"] is False
    assert captured["fit"]["primary_cohort"] is plan.cohort
    assert captured["build"]["specs"] == [locked_spec]
    assert captured["write"]["panel"] is panel
    assert result.manifest_update() == {
        "robustness_panel_path": "robustness_panel.json",
        "robustness_n_variants": 2,
        "robustness_range_low": 0.8,
        "robustness_range_high": 1.4,
    }


def test_finalizer_fails_closed_when_locked_spec_was_not_estimated(
    monkeypatch, tmp_path: Path
) -> None:
    locked_spec = SimpleNamespace(spec_id="alt_locked")
    plan = SimpleNamespace(robustness_specs=[locked_spec], cohort=None)
    panel = _panel()

    monkeypatch.setattr(
        runtime_panel,
        "robustness_specs_for_execution",
        lambda **_kwargs: [locked_spec],
    )
    monkeypatch.setattr(
        runtime_panel,
        "fit_robustness_rows_from_records",
        lambda **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        runtime_panel,
        "build_robustness_panel_from_records",
        lambda **_kwargs: panel,
    )
    monkeypatch.setattr(runtime_panel, "write_robustness_panel", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_panel,
        "unexecuted_locked_spec_ids",
        lambda _panel: ["alt_locked"],
    )

    result = runtime_panel.finalize_run_robustness_panel(
        run_dir=tmp_path,
        plan=plan,
        per_step_records=[],
        cohort_path=None,
        context=None,
        evidence=SimpleNamespace(),
        prompt_pack_version=None,
    )

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.validator == "robustness_panel"
    assert finding.severity == "error"
    assert finding.detail["unexecuted_spec_ids"] == ["alt_locked"]
    assert finding.detail["locked_spec_count"] == 1


def test_finalizer_isolates_panel_construction_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runtime_panel,
        "robustness_specs_for_execution",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad lock")),
    )

    result = runtime_panel.finalize_run_robustness_panel(
        run_dir=tmp_path,
        plan=SimpleNamespace(robustness_specs=[]),
        per_step_records=[],
        cohort_path=None,
        context=None,
        evidence=SimpleNamespace(),
        prompt_pack_version=None,
    )

    assert result.manifest_update() == {}
    assert len(result.findings) == 1
    assert result.findings[0].validator == "robustness_panel"
    assert result.findings[0].severity == "warning"
    assert "bad lock" in result.findings[0].message


def test_execute_phase_delegates_run_robustness_transaction_once() -> None:
    from easyicu.research_agent.execution import phase

    tree = ast.parse(inspect.getsource(phase))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "finalize_run_robustness_panel"
    ]
    assert len(calls) == 1

    forbidden_direct_calls = {
        "fit_robustness_rows_from_records",
        "build_robustness_panel_from_records",
        "write_robustness_panel",
        "unexecuted_locked_spec_ids",
    }
    assert not {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forbidden_direct_calls
    }
