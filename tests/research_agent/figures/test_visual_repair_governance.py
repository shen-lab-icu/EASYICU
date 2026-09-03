from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.providers.mocks import (
    ExternalCaptureMockLLMClient,
    PatternScriptedMockLLMClient,
)
from easyicu.research_agent.schema import ValidationFinding

from tests.research_agent.gates.test_gate_evaluator_contract import gate_call_order


def test_mixed_overlap_and_clipping_visual_error_is_not_cosmetic() -> None:
    from easyicu.research_agent.execution.phase import _is_cosmetic_visual_finding
    from easyicu.research_agent.reporting.readiness import _is_cosmetic_visual_error

    finding = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message=(
            "Overlapping text elements and clipped/missing axis labels; adjust spacing."
        ),
        detail={"reason": "svg_text_overlap_spacing"},
    )

    assert not _is_cosmetic_visual_finding(finding)
    assert not _is_cosmetic_visual_error(finding)


def test_closed_overlap_spacing_reason_remains_cosmetic() -> None:
    from easyicu.research_agent.execution.phase import _is_cosmetic_visual_finding
    from easyicu.research_agent.reporting.readiness import _is_cosmetic_visual_error

    finding = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message="Overlapping text elements detected; adjust spacing.",
        detail={"reason": "svg_text_overlap_spacing"},
    )

    assert _is_cosmetic_visual_finding(finding)
    assert _is_cosmetic_visual_error(finding)


def _svg(*, overlap: bool) -> str:
    legend_x = 102 if overlap else 280
    padding = "x" * 1400
    return f"""
<svg width="400pt" height="240pt" viewBox="0 0 400 240"
     xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="240" fill="white"/>
  <circle cx="80" cy="170" r="18" fill="#1f5a85"/>
  <g id="annotation"><text x="100" y="40" style="font-size: 16px">37,433</text></g>
  <g id="legend_1"><text x="{legend_x}" y="41" style="font-size: 16px">Deaths</text></g>
  <!-- {padding} -->
</svg>
""".strip()


def _script(*, svg: str, include_table: bool) -> str:
    table_write = (
        "(out / 'summary.csv').write_text('metric,value\\nn,3\\n', encoding='utf-8')\n"
        if include_table
        else ""
    )
    summary = {"status": "ok", "contract_ok": include_table}
    if include_table:
        summary["output_files"] = {"table:summary": "summary.csv"}
    return (
        "import json, os\n"
        "from pathlib import Path\n"
        "out = Path(os.environ['STEP_OUT_DIR'])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        f"(out / 'layout.svg').write_text({svg!r}, encoding='utf-8')\n"
        + table_write
        + "(out / 'step_summary.json').write_text("
        f"json.dumps({summary!r}), "
        "encoding='utf-8')\n"
    )


def _visual_governance_llm(
    *,
    initial_code: str,
    contract_code: str | None = None,
    contract_error: Exception | None = None,
    visual_code: str | None = None,
    visual_error: Exception | None = None,
) -> PatternScriptedMockLLMClient:
    plan = json.dumps(
        {
            "research_question": "Summarize the cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "primary",
                    "intent": "Write a summary table and an auxiliary figure.",
                    "inputs": ["stay_id"],
                    "expected_outputs": ["table:summary"],
                    "method": "descriptive_summary",
                    "scientific_action_id": "descriptive.descriptive_summary",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "visual-repair governance fixture",
        }
    )
    contract_response: str | BaseException
    if contract_error is not None:
        contract_response = contract_error
    elif contract_code is not None:
        contract_response = _exact_code_patch(
            [
                ('x="280"', 'x="102"'),
                (
                    initial_code.splitlines()[-1],
                    "\n".join(contract_code.splitlines()[-2:]),
                ),
            ]
        )
    else:
        contract_response = AssertionError("unexpected contract repair")

    visual_response: str | BaseException
    if visual_error is not None:
        visual_response = visual_error
    elif visual_code is not None and contract_code is not None:
        visual_response = _exact_code_patch([('x="102"', 'x="280"')])
    elif visual_code is not None:
        visual_response = visual_code
    else:
        visual_response = AssertionError("unexpected visual repair")

    return PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [plan]),
            ("WRITE THE PYTHON CODE", [initial_code]),
            ("REPAIR THE PYTHON CODE", [contract_response]),
            ("VISUAL QA REJECTED", [visual_response]),
            (
                "INTERPRET THE RESULTS",
                ["The summary is available {evidence:summary}."],
            ),
            (
                "MANUSCRIPT SCAFFOLD",
                ["# Title\n\n## Results\n\nSummary {evidence:summary}."],
            ),
            (
                "EVERY FINDING MUST INCLUDE",
                [json.dumps({"findings": []})],
            ),
        ]
    )


def _matching_calls(client, marker: str):
    folded = marker.casefold()
    return [
        (messages, kwargs)
        for messages, kwargs in client.calls
        if folded
        in "\n".join(str(message.content or "") for message in messages).casefold()
    ]


def _repair_counts(client) -> tuple[int, int]:
    repair_calls = _matching_calls(client, "REPAIR THE PYTHON CODE")
    visual_calls = _matching_calls(client, "VISUAL QA REJECTED")
    return len(repair_calls) - len(visual_calls), len(visual_calls)


def _exact_code_patch(edits: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [
                {"old": old, "new": new, "expected_count": 1} for old, new in edits
            ],
        }
    )


def _record(result) -> dict:
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    return next(
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    )


def _run(
    ra,
    tmp_path: Path,
    llm: PatternScriptedMockLLMClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_deterministic_code_fallback: bool = False,
):
    from easyicu.research_agent.agents.core import PlannerAgent

    original_planner_run = PlannerAgent.run

    def run_without_unrelated_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_planner_run(self, context, **kwargs)

    monkeypatch.setattr(
        PlannerAgent,
        "run",
        run_without_unrelated_article_suite,
    )
    concept_auditor = ExternalCaptureMockLLMClient([json.dumps({"findings": []})] * 2)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        llm_concept_auditor_client=concept_auditor,
        enable_literature=False,
        enable_latex=False,
        enable_vlm_visual_qa=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=enable_deterministic_code_fallback,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=1,
        runner_kind="subprocess",
    )
    return pipeline.run(
        question="Summarize the cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3]}),
        cohort_name="visual_governance_test",
        database="synthetic",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )


def _ignore_figure_provenance_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.research_agent.audits.validators import (
        FigureContractQualityValidator,
        FigureSourceDataValidator,
    )

    monkeypatch.setattr(
        FigureContractQualityValidator,
        "audit",
        lambda self, **kwargs: [],
    )
    monkeypatch.setattr(
        FigureSourceDataValidator,
        "audit",
        lambda self, **kwargs: [],
    )


def test_visual_repair_log_keeps_structured_collision_detail() -> None:
    from easyicu.research_agent.execution.phase import _visual_repair_request_log

    finding = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message="SVG text overlap.",
        detail={
            "path": "/tmp/absolute_risk_by_stage.svg",
            "examples": [
                {
                    "text_a": "37,433",
                    "text_b": "Deaths",
                    "overlap_fraction": 0.388,
                    "bbox_a": [448.8, 28.1, 473.5, 37.3],
                    "bbox_b": [451.2, 33.4, 478.0, 42.6],
                }
            ],
        },
    )

    log = _visual_repair_request_log([finding])
    marker = (
        "STRUCTURED VISUAL FINDINGS " "(diagnostic mirror; not routing authority):\n"
    )

    assert log.startswith(marker)
    assert json.loads(log.removeprefix(marker)) == [
        {
            "validator": finding.validator,
            "severity": finding.severity,
            "message": finding.message,
            "detail": finding.detail,
        }
    ]


def test_visual_qa_stays_before_contract_gate() -> None:
    # Converged onto the AST GateEvaluator contract (test_gate_evaluator_contract):
    # the visual gate runs before the shared deterministic contract gate. This
    # used to be a brittle ``source.index("literal") < source.index("literal")``
    # pair that broke every time a gate implementation moved; it is now a
    # first-call-lineno ordering over the parsed AST. The former second assertion
    # (cosmetic demotion precedes the terminal ``visual_qa_repair_failed`` in the
    # repair-exception handler) is retired on purpose: demotion is now PRECOMPUTED
    # in collect_visual_gate_result (VisualGateResult.demoted_findings), so it no
    # longer depends on statement order — that guarantee is locked by
    # test_visual_gate_component + the gate-purity contract.
    from easyicu.research_agent.execution import phase as pipeline_execute

    stages = (
        pipeline_execute._candidate_success_prepare_transition,
        pipeline_execute._candidate_contract_setup_transition,
    )
    names = {"collect_visual_gate_result", "_step_deterministic_contract_findings"}
    order = {
        name: (stage_index, line)
        for stage_index, stage in enumerate(stages)
        for name, line in gate_call_order(stage, names).items()
    }
    assert (
        order["collect_visual_gate_result"]
        < order["_step_deterministic_contract_findings"]
    )


def test_figure_canonicalization_repair_stays_between_gate_and_figure_audits() -> None:
    # Batch 1a-0 ordering guard (the boundary the dedup must never reorder): the
    # early figure-contract canonicalization REPAIR runs after the shared
    # deterministic contract gate and BEFORE the figure audits, so those
    # validators audit the already-canonicalized contracts. Converged onto the AST
    # contract (was three brittle source.index anchors).
    from easyicu.research_agent.execution import phase as pipeline_execute

    stages = (
        pipeline_execute._candidate_success_prepare_transition,
        pipeline_execute._candidate_contract_setup_transition,
    )
    names = {
        "_step_deterministic_contract_findings",
        "_install_figure_contract_source_data_canonicalization",
        "_post_canonicalization_figure_findings",
    }
    order = {
        name: (stage_index, line)
        for stage_index, stage in enumerate(stages)
        for name, line in gate_call_order(stage, names).items()
    }
    assert (
        order["_step_deterministic_contract_findings"]
        < order["_install_figure_contract_source_data_canonicalization"]
        < order["_post_canonicalization_figure_findings"]
    )

    # Inside that helper the figure-contract audit still precedes the
    # figure-source audit (both see the canonicalized contracts).
    helper_source = inspect.getsource(
        pipeline_execute._post_canonicalization_figure_findings
    )
    assert helper_source.index("figure_contract_validator.audit") < helper_source.index(
        "figure_source_validator.audit"
    )


def test_contract_budget_does_not_consume_visual_layout_budget(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ignore_figure_provenance_gates(monkeypatch)
    from easyicu.research_agent.gates import contract as contract_gate

    def controlled_contract(*, step_summary, **kwargs):
        del kwargs
        if step_summary.get("contract_ok") is True:
            return []
        return [
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message="Controlled contract repair is required.",
            )
        ]

    monkeypatch.setattr(
        contract_gate,
        "_step_contract_findings",
        controlled_contract,
    )
    llm = _visual_governance_llm(
        initial_code=_script(svg=_svg(overlap=False), include_table=False),
        contract_code=_script(svg=_svg(overlap=True), include_table=True),
        visual_code=_script(svg=_svg(overlap=False), include_table=True),
    )

    record = _record(_run(ra, tmp_path, llm, monkeypatch))

    assert _repair_counts(llm) == (1, 1)
    assert record["status"] == "ok"
    assert record["contract_repair_attempts"] == 1
    assert record["visual_repair_attempts"] == 1
    assert record["code_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == ["contract", "visual"]
    assert record["step_provider_call_categories"] == [
        "initial_generation",
        "contract_repair_patch",
        "visual_repair_patch",
        "concept_audit",
        "analyzer",
    ]
    authority_prefix = "HOST-OWNED REPAIR AUTHORITY (typed; verbatim):\n"
    visual_message_batch = _matching_calls(llm, "VISUAL QA REJECTED")[0][0]
    visual_authority_messages = [
        message.content.removeprefix(authority_prefix)
        for message in visual_message_batch
        if message.role == "system" and message.content.startswith(authority_prefix)
    ]
    assert len(visual_authority_messages) == 1
    visual_authority = json.loads(visual_authority_messages[0])
    assert visual_authority["host_guidance"] == {
        "layout_only": True,
        "preserve": [
            "source_data_values_and_rows",
            "step_summary_numeric_and_statistical_values",
            "figure_contract_claims_evidence_and_panel_roles",
        ],
        "forbid": [
            "source_resolution_changes",
            "cohort_or_data_transformations",
            "estimate_or_scientific_label_changes",
        ],
    }


def test_contract_repair_provider_failure_preserves_contract_observability(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider outage must not hide the contract that triggered repair."""

    _ignore_figure_provenance_gates(monkeypatch)
    from easyicu.research_agent.gates import contract as contract_gate

    finding = ValidationFinding(
        validator="step_contract",
        severity="error",
        message="Controlled contract repair is required.",
        detail={"missing": ["table:summary"]},
    )
    monkeypatch.setattr(
        contract_gate,
        "_step_contract_findings",
        lambda **kwargs: [finding],
    )
    llm = _visual_governance_llm(
        initial_code=_script(svg=_svg(overlap=False), include_table=False),
        contract_error=RuntimeError("provider returned HTTP 502"),
    )

    result = _run(ra, tmp_path, llm, monkeypatch)
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _record(result)

    assert _repair_counts(llm) == (1, 0)
    assert record["status"] == "repair_failed"
    assert record["step_summary"] == {"status": "ok", "contract_ok": False}
    assert record["contract_findings"] == [finding.model_dump()]
    assert any(
        item["validator"] == "coder"
        and item["severity"] == "error"
        and "HTTP 502" in item["message"]
        for item in partial["findings"]
    )


def test_cosmetic_visual_repair_provider_failure_keeps_outputs(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ignore_figure_provenance_gates(monkeypatch)
    llm = _visual_governance_llm(
        initial_code=_script(svg=_svg(overlap=True), include_table=True),
        visual_error=RuntimeError("simulated provider outage"),
    )

    result = _run(
        ra,
        tmp_path,
        llm,
        monkeypatch,
        enable_deterministic_code_fallback=True,
    )
    record = _record(result)

    assert _repair_counts(llm) == (0, 1)
    assert record["status"] == "ok"
    assert record["visual_qa_demoted"] is True
    assert record["visual_repair_provider_failed"] is True
    assert record["visual_repair_attempts"] == 1
    assert all(finding["severity"] != "error" for finding in record["visual_findings"])
    assert "deterministic_code_fallback" not in record
    svg_path = Path(result.workdir) / "steps" / "01_summary" / "outputs" / "layout.svg"
    assert svg_path.is_file()
    retained_svg = svg_path.read_text(encoding="utf-8")
    assert 'x="102"' in retained_svg
    assert "37,433" in retained_svg


def test_noncosmetic_visual_repair_provider_failure_remains_fail_closed(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ignore_figure_provenance_gates(monkeypatch)
    malformed_svg = "<svg>" + ("x" * 1400)
    llm = _visual_governance_llm(
        initial_code=_script(svg=malformed_svg, include_table=True),
        visual_error=RuntimeError("simulated provider outage"),
    )

    record = _record(_run(ra, tmp_path, llm, monkeypatch))

    assert _repair_counts(llm) == (0, 1)
    assert record["status"] == "repair_failed"
    assert record.get("visual_qa_demoted") is not True
    assert any(finding["severity"] == "error" for finding in record["visual_findings"])
