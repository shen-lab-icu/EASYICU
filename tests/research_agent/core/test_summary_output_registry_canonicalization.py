"""Fail-closed normalization of exact legacy step-output registries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.declared_product import (
    declared_product_contract_findings,
)
from easyicu.research_agent.contracts.runtime import RunResult
from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.summary import salvage_step_summary
from easyicu.research_agent.schema import AnalysisStep


def _step(*outputs: str) -> AnalysisStep:
    return AnalysisStep(
        step_id="01_prepare_cohort",
        intent="Prepare the locked analysis cohort and its attrition tables.",
        expected_outputs=list(outputs),
    )


def _run_result(out_dir: Path) -> RunResult:
    return RunResult(
        step_id="01_prepare_cohort",
        script_path=out_dir.parent / "analysis.py",
        cwd=out_dir.parent,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )


def _write_summary(out_dir: Path, outputs: dict[str, object], **extra: object) -> None:
    payload = {"status": "ok", "outputs": outputs, **extra}
    (out_dir / "step_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def test_exact_legacy_outputs_are_canonicalized_before_product_gate(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "analysis_cohort.parquet").write_bytes(b"PAR1")
    (out_dir / "cohort_flow.csv").write_text("stage,n\nlocked,1000\n")
    (out_dir / "cohort_attrition.csv").write_text("stage,n\nlocked,1000\n")
    _write_summary(
        out_dir,
        {
            "analysis_cohort": (
                "/easyicu-run/steps/01_prepare_cohort/outputs/"
                "analysis_cohort.parquet"
            ),
            "cohort_flow": (
                "/easyicu-run/steps/01_prepare_cohort/outputs/cohort_flow.csv"
            ),
            "cohort_attrition": (
                "/easyicu-run/steps/01_prepare_cohort/outputs/" "cohort_attrition.csv"
            ),
        },
        cohort_n=1000,
    )
    step = _step(
        "artifact:analysis_cohort",
        "table:cohort_flow",
        "table:cohort_attrition",
    )

    outcome = salvage_step_summary(_run_result(out_dir), step=step)

    assert outcome is not None
    assert outcome.repair_id == "summary_output_registry_canonicalization_v1"
    assert repair_metadata_for(outcome.repair_id).repair_class is RepairClass.STRUCTURAL
    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["output_files"] == {
        "artifact:analysis_cohort": "analysis_cohort.parquet",
        "table:cohort_flow": "cohort_flow.csv",
        "table:cohort_attrition": "cohort_attrition.csv",
    }
    assert summary["cohort_n"] == 1000
    findings = declared_product_contract_findings(
        step=step,
        step_summary=summary,
        effect_method_authorized=False,
        out_dir=out_dir,
    )
    assert not [
        finding
        for finding in findings
        if (finding.detail or {}).get("kind") == "declared_product_missing"
    ]


@pytest.mark.parametrize(
    ("declared", "outputs"),
    [
        (
            ["table:cohort_flow", "table:cohort_attrition"],
            {"cohort_flow": "cohort_flow.csv"},
        ),
        (
            ["table:cohort_flow"],
            {
                "cohort_flow": "cohort_flow.csv",
                "unplanned": "cohort_attrition.csv",
            },
        ),
        (
            ["table:shared", "artifact:shared"],
            {"shared": "cohort_flow.csv"},
        ),
        (
            ["figure:cohort_flow"],
            {"cohort_flow": "cohort_flow.csv"},
        ),
        (
            ["table:cohort_flow"],
            {"cohort_flow": "../cohort_flow.csv"},
        ),
    ],
)
def test_non_bijective_or_incompatible_legacy_outputs_fail_closed(
    tmp_path: Path,
    declared: list[str],
    outputs: dict[str, object],
) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "cohort_flow.csv").write_text("stage,n\nlocked,1000\n")
    (out_dir / "cohort_attrition.csv").write_text("stage,n\nlocked,1000\n")
    _write_summary(out_dir, outputs)
    before = (out_dir / "step_summary.json").read_bytes()

    outcome = salvage_step_summary(_run_result(out_dir), step=_step(*declared))

    assert outcome is None
    assert (out_dir / "step_summary.json").read_bytes() == before


def test_existing_typed_registry_is_never_overwritten(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "cohort_flow.csv").write_text("stage,n\nlocked,1000\n")
    _write_summary(
        out_dir,
        {"cohort_flow": "cohort_flow.csv"},
        output_files={"table:other": "cohort_flow.csv"},
    )
    before = (out_dir / "step_summary.json").read_bytes()

    outcome = salvage_step_summary(
        _run_result(out_dir), step=_step("table:cohort_flow")
    )

    assert outcome is None
    assert (out_dir / "step_summary.json").read_bytes() == before


def test_symlinked_output_is_not_granted_typed_authority(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    target = tmp_path / "outside.csv"
    target.write_text("stage,n\nlocked,1000\n")
    try:
        (out_dir / "cohort_flow.csv").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    _write_summary(out_dir, {"cohort_flow": "cohort_flow.csv"})

    outcome = salvage_step_summary(
        _run_result(out_dir), step=_step("table:cohort_flow")
    )

    assert outcome is None
    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert "output_files" not in summary


def test_scoped_coder_guide_requires_full_typed_output_keys_and_basenames() -> None:
    from easyicu.research_agent.providers.prompts import load_prompt_pack
    from easyicu.research_agent.research_context.prompt_scope import (
        coder_guide_for_step,
    )

    prompt = coder_guide_for_step(
        load_prompt_pack()["coder"],
        _step("artifact:analysis_cohort", "table:cohort_flow"),
    )

    assert 'step_summary["output_files"]' in prompt
    assert "complete typed token as the key" in prompt
    assert "output-local basename as the value" in prompt
    assert "A bare-name `outputs` map" in prompt
