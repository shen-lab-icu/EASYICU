"""A planning stop for a capability gap says what the question needs, in either language.

The stop's typed cause is the gap's requirement.  It travels from the
compiler's safe diagnostic through the failed run's gate detail to the run
row, and the conversation picks one sentence per requirement.  The host may
not have been able to check the claim, so every sentence says planning found
it.  The copy table and the contract's requirements are the same set.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import get_args

import pytest

from easyicu.research_agent.planning.progressive_contract import (
    CapabilityGapRequirement,
    ProgressivePlanCompileError,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot.projections import gate_detail_projection
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

ERROR_TEXT = (
    Path(agent_routes.__file__).resolve().parents[1]
    / "static"
    / "js"
    / "screens-guided-pi-error-text.js"
)
REQUIREMENTS = get_args(CapabilityGapRequirement)


def _stop(cause: str | None) -> ProgressivePlanCompileError:
    return ProgressivePlanCompileError(
        "progressive_capability_gap",
        "the question needs an exposure element no offered family can express",
        path="capability_gap",
        cause_code=cause,
    )


def _copy(cause: str, lang: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
let errorText = null;
const lang = process.argv[2];
global.window = {
  EU_LANG: lang,
  EU_HTML: { esc: value => String(value) },
  EasyICU: { guidedPi: { declare: (name, api) => { if (name === 'errorText') errorText = api; } } },
};
require(process.argv[1]);
const owner = errorText.create({ tr: (en, zh) => (lang === 'zh' ? zh : en), staticPreview: () => false });
process.stdout.write(owner.runFailureText('research_pipeline_progressive_compile_failed', {
  code: 'progressive_capability_gap', cause: process.argv[3], missing: [],
}));
"""
    result = run_node(node, script, str(ERROR_TEXT), lang, cause, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


@pytest.mark.parametrize(
    ("cause", "en", "zh"),
    [
        (
            "levels_from_thresholds_unavailable",
            "groups a variable by thresholds",
            "按阈值分组",
        ),
        (
            "longitudinal_representation_unavailable",
            "no such time series",
            "没有这样的时间序列",
        ),
        (
            "multiple_sources_required",
            "a separate study in each database",
            "在每个数据库各开一项研究",
        ),
        (
            "estimand_unsupported",
            "for example an association, a prediction",
            "例如关联、预测或描述性汇总",
        ),
        (
            "design_element_unsupported",
            "a design element no executable EasyICU method",
            "设计要素",
        ),
    ],
)
def test_each_requirement_reads_as_its_own_sentence(cause, en, zh) -> None:
    english, chinese = _copy(cause, "en"), _copy(cause, "zh")

    assert english.startswith("Planning found that the question needs something")
    assert en in english
    assert chinese.startswith("规划发现这个问题需要")
    assert zh in chinese


def test_a_cause_without_copy_reads_as_the_stop_itself() -> None:
    english = _copy("unlisted_cause", "en")

    assert english.startswith("Planning found that the question needs something")
    assert english.endswith("No analysis was run.")


def test_the_copy_names_every_requirement_and_nothing_else() -> None:
    source = ERROR_TEXT.read_text(encoding="utf-8")
    table = source.split("function capabilityGapCopy(cause) {", 1)[1].split("};", 1)[0]

    assert set(re.findall(r"^\s+([a-z_]+): tr\(", table, re.M)) == set(REQUIREMENTS)


def test_the_cause_travels_from_the_stop_to_the_run_row() -> None:
    typed = agent_pipeline_runs._safe_pipeline_typed_failure(
        _stop("estimand_unsupported")
    )
    detail = agent_pipeline_runs._failure_gate_detail(
        typed["reason_code"], typed.get("cause_code")
    )

    assert detail == {
        "reason_code": "progressive_capability_gap",
        "cause_code": "estimand_unsupported",
    }
    assert (
        gate_detail_projection(detail)["gate_detail_cause_code"]
        == "estimand_unsupported"
    )
    assert agent_pipeline_runs._failure_gate_detail(
        "progressive_capability_gap", None
    ) == {"reason_code": "progressive_capability_gap"}


def test_the_failed_run_records_the_gap_by_its_codes(tmp_path) -> None:
    stop = ProgressivePlanCompileError(
        "progressive_capability_gap",
        "the question needs an exposure element no offered family can express "
        "(levels_from_thresholds_unavailable, verified): lactate above 4 mmol/L",
        path="capability_gap",
        cause_code="levels_from_thresholds_unavailable",
    )
    code = agent_pipeline_runs._pipeline_failure_code(stop)
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "study_synthetic"},
        provider={},
        exc=stop,
        code=code,
        execution_retry_id=None,
    )

    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["reason"] == "research_pipeline_progressive_compile_failed"
    assert gate["detail"] == {
        "reason_code": "progressive_capability_gap",
        "cause_code": "levels_from_thresholds_unavailable",
    }
    diagnostic = (
        tmp_path / "diagnostics" / "research_pipeline_failure.json"
    ).read_text()
    # The Planner's own words about the study stay out of the record.
    assert "4 mmol/L" not in json.dumps(gate) + diagnostic


def test_a_cause_that_is_not_a_code_never_leaves_the_stop() -> None:
    stop = _stop("Not A Code!")

    assert "cause_code" not in stop.easyicu_safe_diagnostic
    assert "cause_code" not in agent_pipeline_runs._safe_pipeline_typed_failure(stop)


def test_the_run_record_says_planning_found_the_gap() -> None:
    message = agent_pipeline_runs._progressive_compile_failure_message(
        _stop("multiple_sources_required")
    )

    assert message.startswith("Planning found that the question needs something")
    assert message.endswith("No analysis was run.")
