"""No question the home page offers is stopped by a question reader.

The home examples (``screens-guided-pi-starters.js``) fill the composer with
a question a clinician can run as it stands.  Two launch stops read the
question itself: a mean survival time no plan answers
(``question_mean_survival``) and a treatment's use the data record only as a
dose (``question_treatment_use``).  An example either stop refuses would offer
a question the platform then turns down, so every example prompt, in English
and in Chinese, is read through both, from the rendered page rather than a
copied list, as the study it states (its family, as the home contract test
records it): the causal example asks about starting norepinephrine, a
treatment strategy its trial reads from the treatment capture registry.
"""

from __future__ import annotations

import html
import json
import re
import shutil

import pytest

from easyicu.webserver.question_mean_survival import (
    MEAN_SURVIVAL_UNPLANNED,
    asked_mean_survival,
    stop_unplanned_mean_survival,
)
from easyicu.webserver.question_treatment_use import stop_unrecorded_treatment_uses
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.support.home_examples import (
    EXAMPLE_REQUIREMENTS as _EXAMPLE_REQUIREMENTS,
    REPLICATION_EXAMPLES as _REPLICATION_EXAMPLES,
)
from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)

# The SICdb example asked the restricted mean survival time difference as its
# main result until it was reworded (fedbc2293) to ask 28-day survival, with
# that difference as a sensitivity analysis.  A checkout that still carries
# the old wording stops it, for this reason and no other.
_ASKED_RMST_AS_THE_MAIN_RESULT = frozenset({
    "In SICdb, with death within 28 days as the endpoint, how much does the restricted mean "
    "survival time of patients ventilated on the first ICU day differ from that of patients who "
    "are not, after adjusting for age and sex?",
    "在 SICdb 中，以 28 天死亡为终点，入 ICU 第一天接受机械通气的患者与未通气的患者相比，"
    "调整年龄和性别后，限制性平均生存时间相差多少天？",
})
_REWORDED_RMST_EXAMPLE = (
    "在 SICdb 中，入 ICU 第一天接受机械通气的患者与未通气的患者相比，调整年龄和性别后 28 天生存是否不同？"
    "并以 28 天限制性平均生存时间差作为敏感性分析。",
    "In SICdb, does 28-day survival differ between patients ventilated on the first ICU day and "
    "those who are not, after adjusting for age and sex? Report the 28-day restricted mean "
    "survival time difference as a sensitivity analysis.",
)


def _home_prompts(lang: str) -> list[tuple[str, str]]:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = f"""
      global.window = {{EU_LANG: {json.dumps(lang)}}};
      eval({json.dumps(_read('js/html-escape.js'))});
      eval({json.dumps(_read('js/screens-guided-pi-starters.js'))});
      const tr = (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en;
      process.stdout.write(window.EU_GUIDED_PI_STARTERS.render({{tr, composer: ''}}));
    """
    page = run_node(node, script, check=True).stdout
    cards = []
    for button in page.split("<button")[1:]:
        compose = re.search(r'data-gpi-starter-compose="([^"]*)"', button)
        if compose is None:
            continue
        example = re.search(r'data-gpi-starter-example="([^"]*)"', button)
        cards.append((example.group(1) if example else "", html.unescape(compose.group(1))))
    return cards


def _family(example: str) -> str:
    if example in _EXAMPLE_REQUIREMENTS:
        return _EXAMPLE_REQUIREMENTS[example][1]
    if example in _REPLICATION_EXAMPLES:
        return _REPLICATION_EXAMPLES[example][0][1]
    return ""


def _stop(prompt: str, family: str = "") -> str | None:
    try:
        stop_unplanned_mean_survival(prompt)
        stop_unrecorded_treatment_uses(prompt, study={"analysis_design": {"analysis_family": family}})
    except ResearchPipelineRunError as error:
        return error.code
    return None


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_no_home_example_is_stopped_by_a_question_reader(lang: str) -> None:
    cards = _home_prompts(lang)
    # Every example card and the demo card.
    assert len(cards) > 10
    assert {example for example, _prompt in cards if example} == set(_EXAMPLE_REQUIREMENTS) | set(
        _REPLICATION_EXAMPLES
    )
    stopped = {
        prompt: code for example, prompt in cards if (code := _stop(prompt, _family(example))) is not None
    }
    assert {
        prompt: code for prompt, code in stopped.items() if prompt not in _ASKED_RMST_AS_THE_MAIN_RESULT
    } == {}
    assert set(stopped.values()) <= {MEAN_SURVIVAL_UNPLANNED}


@pytest.mark.parametrize("prompt", _REWORDED_RMST_EXAMPLE)
def test_the_reworded_rmst_example_asks_the_mean_survival_only_as_a_sensitivity(prompt: str) -> None:
    assert asked_mean_survival(prompt) == ()
    assert _stop(prompt) is None
