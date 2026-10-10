"""The prediction tables' coded values and the template stop read by name.

The prediction executor and its comparator owner write closed value sets into
``prediction_performance.csv`` and ``benchmark_comparison.csv``; the reader
vocabulary names each value, and a value outside its set is shown as written.
The sets are read from their owners, so a code added there without a reader
name fails here instead of reaching a Chinese reader raw.  The planner's
family-template stop is pinned the same way.
"""

from __future__ import annotations

import json
import shutil
from typing import get_args

import pytest

from easyicu.research_agent.contracts.prediction_validation import CalibrationStatus
from easyicu.research_agent.methods.auc_interval import (
    CLUSTER_BOOTSTRAP_METHOD,
    DELONG_METHOD,
    DELONG_PAIRED_METHOD,
)
from easyicu.research_agent.planning.benchmark_comparator import (
    CALIBRATION_REASONS,
    INFORMATION_WINDOW_RELATIONS,
    ComparatorKind,
)
from easyicu.research_agent.planning.outline_action_rules import (
    STATIC_PREDICTION_TEMPLATE_REQUIRED,
)
from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import STATIC, _read

#: Each column the reader names, with the values its owners write.
OWNER_VALUES = {
    "auroc_ci_method": (DELONG_METHOD, CLUSTER_BOOTSTRAP_METHOD),
    "interval_method": (DELONG_PAIRED_METHOD, CLUSTER_BOOTSTRAP_METHOD),
    "comparator_kind": get_args(ComparatorKind),
    "calibration_reason": CALIBRATION_REASONS,
    "information_window_relation": INFORMATION_WINDOW_RELATIONS,
    "calibration_status": get_args(CalibrationStatus),
}


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    return node


def _names(lang: str) -> dict[str, dict[str, str]]:
    script = f"""
      global.window = {{ t: (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en }};
      eval({json.dumps(_read("js/screens-agent-reader-vocab.js"))});
      const values = {json.dumps({key: list(codes) for key, codes in OWNER_VALUES.items()})};
      const names = {{}};
      for (const [key, codes] of Object.entries(values)) {{
        names[key] = Object.fromEntries(codes.map(code => [code, window.AGENT_READER_VOCAB.value(key, code)]));
      }}
      process.stdout.write(JSON.stringify(names));
    """
    return json.loads(run_node(_node(), script, check=True).stdout)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_value_the_prediction_owners_write_has_a_reader_name(lang: str) -> None:
    names = _names(lang)

    # An English name may be the code's own word ("probability"); a Chinese
    # reader never reads the code.
    unnamed = [
        (key, code)
        for key, codes in OWNER_VALUES.items()
        for code in codes
        if not names[key][code] or (lang == "zh" and names[key][code] == code)
    ]
    assert unnamed == []


def _stop_line(lang: str) -> str:
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
process.stdout.write(owner.runFailureText('research_pipeline_progressive_compile_failed', { code: process.argv[3] }));
"""
    owner = STATIC / "js" / "screens-guided-pi-error-text.js"
    result = run_node(
        _node(),
        script,
        str(owner),
        lang,
        STATIC_PREDICTION_TEMPLATE_REQUIRED,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_the_family_template_stop_names_the_template() -> None:
    assert "planned from the prediction template" in _stop_line("en")
    assert "这份计划没有用模板，规划已停止，没有运行分析" in _stop_line("zh")
    assert "make it the study's primary question" in _stop_line("en")
