from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.schema import ValidationFinding


def test_mixed_overlap_and_clipping_visual_error_is_not_cosmetic() -> None:
    from easyicu.research_agent.pipeline_execute import _is_cosmetic_visual_finding
    from easyicu.research_agent.pipeline_report import _is_cosmetic_visual_error

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
    from easyicu.research_agent.pipeline_execute import _is_cosmetic_visual_finding
    from easyicu.research_agent.pipeline_report import _is_cosmetic_visual_error

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


class _VisualGovernanceLLM:
    name = "visual-governance-llm"

    def __init__(
        self,
        *,
        initial_code: str,
        contract_code: str | None = None,
        contract_error: Exception | None = None,
        visual_code: str | None = None,
        visual_error: Exception | None = None,
    ) -> None:
        self.initial_code = initial_code
        self.contract_code = contract_code
        self.contract_error = contract_error
        self.visual_code = visual_code
        self.visual_error = visual_error
        self.contract_repairs = 0
        self.visual_repairs = 0
        self.visual_prompts: list[str] = []
        self.visual_message_batches: list[list] = []

    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        del max_tokens, temperature
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        upper = user.upper()
        if "ICU-AWARE RESEARCH PLAN" in upper:
            return json.dumps(
                {
                    "research_question": "Summarize the cohort.",
                    "steps": [
                        {
                            "step_id": "01_summary",
                            "intent": "Write a summary table and an auxiliary figure.",
                            "inputs": ["stay_id"],
                            "expected_outputs": ["table:summary"],
                            "method": "descriptive_summary",
                            "icu_rule_refs": [],
                        }
                    ],
                    "rationale": "visual-repair governance fixture",
                }
            )
        if "WRITE THE PYTHON CODE" in upper:
            return self.initial_code
        if "REPAIR THE PYTHON CODE" in upper:
            if "VISUAL QA REJECTED" in upper:
                self.visual_repairs += 1
                self.visual_prompts.append(user)
                self.visual_message_batches.append(list(messages))
                if self.visual_error is not None:
                    raise self.visual_error
                assert self.visual_code is not None
                if "MINIMAL PATCH MODE" in upper:
                    assert self.contract_code is not None
                    return _exact_code_patch([('x="102"', 'x="280"')])
                return self.visual_code
            self.contract_repairs += 1
            if self.contract_error is not None:
                raise self.contract_error
            assert self.contract_code is not None
            if "MINIMAL PATCH MODE" in upper:
                return _exact_code_patch(
                    [
                        ('x="280"', 'x="102"'),
                        (
                            self.initial_code.splitlines()[-1],
                            "\n".join(self.contract_code.splitlines()[-2:]),
                        ),
                    ]
                )
            return self.contract_code
        if "INTERPRET THE RESULTS" in upper:
            return "The summary is available {evidence:summary}."
        if "MANUSCRIPT SCAFFOLD" in upper:
            return "# Title\n\n## Results\n\nSummary {evidence:summary}."
        if "EVERY FINDING MUST INCLUDE" in upper and "RETURN JSON ONLY" in upper:
            return json.dumps({"findings": []})
        return "{}"


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
    llm: _VisualGovernanceLLM,
    *,
    enable_deterministic_code_fallback: bool = False,
):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_latex=False,
        enable_vlm_visual_qa=False,
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
    from easyicu.research_agent.pipeline_execute import _visual_repair_request_log

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
    # Batch 1a-2: the VisualQA audit + classification is now invoked through the
    # typed ``collect_visual_gate_result(...)`` gate (the raw
    # ``VisualQAAuditor().audit_with_expected`` call moved into that helper), the
    # branch selection + LLM-repair recommendation come from
    # ``decide_visual_repair(...)`` (its ``repair_log`` replaces the inline
    # ``qa_log`` assembly), and the exception path reads the cosmetic-demotion
    # projection off the result (``visual_gate.demoted_findings``) instead of
    # re-calling the demoter inline. The GUARANTEE is unchanged: the visual gate
    # runs before the contract gate, and cosmetic demotion precedes the terminal
    # ``visual_qa_repair_failed`` fallback in the repair-exception handler.
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)

    assert source.index("collect_visual_gate_result(") < source.index(
        "early_contract_findings = _step_deterministic_contract_findings"
    )
    # Disambiguate the visual-repair ``except`` block via the decision's repair
    # log, which sits in the LLM_REPAIR branch just above that handler.
    visual_except = source[
        source.index(
            "except Exception as exc:",
            source.index("visual_repair_decision.repair_log"),
        ) :
    ]
    assert visual_except.index("visual_gate.demoted_findings") < visual_except.index(
        "visual_qa_repair_failed"
    )


def test_figure_canonicalization_repair_stays_between_gate_and_figure_audits() -> None:
    # Batch 1a-0 ordering guard (the boundary the dedup must never reorder): the
    # early figure-contract canonicalization REPAIR runs after the shared
    # deterministic contract gate and BEFORE the figure-contract / figure-source
    # audits, so those validators audit the already-canonicalized contracts.
    # Extracting the shared 14-validator contract sequence left this hard
    # sequence intact; lock it so a later gate extraction cannot silently move
    # the repair past the audits (which would fail-open on stale contracts).
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    shared_gate = source.index(
        "early_contract_findings = _step_deterministic_contract_findings"
    )
    canonicalization_repair = source.index(
        "_install_figure_contract_source_data_canonicalization("
    )
    # The figure audits now live in _post_canonicalization_figure_findings, called
    # after the canonicalization repair — so the repair still precedes the audits.
    post_canon_figure_audits = source.index(
        "early_contract_findings += _post_canonicalization_figure_findings"
    )
    assert shared_gate < canonicalization_repair < post_canon_figure_audits

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
    from easyicu.research_agent import pipeline_execute

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
        pipeline_execute,
        "_step_contract_findings",
        controlled_contract,
    )
    llm = _VisualGovernanceLLM(
        initial_code=_script(svg=_svg(overlap=False), include_table=False),
        contract_code=_script(svg=_svg(overlap=True), include_table=True),
        visual_code=_script(svg=_svg(overlap=False), include_table=True),
    )

    record = _record(_run(ra, tmp_path, llm))

    assert llm.contract_repairs == 1
    assert llm.visual_repairs == 1
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
    visual_authority_messages = [
        message.content.removeprefix(authority_prefix)
        for message in llm.visual_message_batches[0]
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
    from easyicu.research_agent import pipeline_execute

    finding = ValidationFinding(
        validator="step_contract",
        severity="error",
        message="Controlled contract repair is required.",
        detail={"missing": ["table:summary"]},
    )
    monkeypatch.setattr(
        pipeline_execute,
        "_step_contract_findings",
        lambda **kwargs: [finding],
    )
    llm = _VisualGovernanceLLM(
        initial_code=_script(svg=_svg(overlap=False), include_table=False),
        contract_error=RuntimeError("provider returned HTTP 502"),
    )

    result = _run(ra, tmp_path, llm)
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _record(result)

    assert llm.contract_repairs == 1
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
    llm = _VisualGovernanceLLM(
        initial_code=_script(svg=_svg(overlap=True), include_table=True),
        visual_error=RuntimeError("simulated provider outage"),
    )

    result = _run(
        ra,
        tmp_path,
        llm,
        enable_deterministic_code_fallback=True,
    )
    record = _record(result)

    assert llm.visual_repairs == 1
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
    llm = _VisualGovernanceLLM(
        initial_code=_script(svg=malformed_svg, include_table=True),
        visual_error=RuntimeError("simulated provider outage"),
    )

    record = _record(_run(ra, tmp_path, llm))

    assert llm.visual_repairs == 1
    assert record["status"] == "repair_failed"
    assert record.get("visual_qa_demoted") is not True
    assert any(finding["severity"] == "error" for finding in record["visual_findings"])
