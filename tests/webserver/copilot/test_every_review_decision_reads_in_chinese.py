"""Every decision a plan review hands the researcher reads in Chinese.

A review finding that needs the researcher's authorization becomes the plan
review's current decision: a title, and the question to answer in the
conversation.  The reader vocabulary owns their Chinese wording; English
readers keep the host's own text.  The decisions are read from the review
owners' source, so a new decision without Chinese wording fails here.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from tests.support.node import run_node
from tests.support.review_decisions import assert_review_decision_sentinels, decision_codes
from tests.webserver.copilot.pi_copilot_static_fixtures import _read

CJK = re.compile(r"[一-鿿]")

DECISIONS = sorted(decision_codes())


def render(payload, *, lang):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = f"""
      global.window = {{
        EU_HTML: {{
          esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
          escAttr: value => String(value ?? ''),
        }},
        t: (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en,
        icon: () => '',
      }};
      eval({json.dumps(_read('js/screens-agent-reader-vocab.js'))});
      eval({json.dumps(_read('js/screens-agent-render.js'))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView('scientific_plan_review.json', {json.dumps(payload)}));
    """
    return run_node(node, script, check=True).stdout


def _review(code: str) -> dict:
    return {
        "approval_allowed": False,
        "status": "changes_required",
        "findings": [
            {
                "code": code,
                "severity": "blocker",
                "remediation_route": "study_authority_change",
                "requires_user_authorization": True,
                "message": f"Host message for {code}.",
                "remediation": f"Host remediation for {code}.",
                "authorization_question": f"Host question for {code}?",
            }
        ],
    }


def _current_decision(html: str) -> tuple[str, str]:
    match = re.search(
        r'class="ag-science-review-section is-current">.*?<strong>(.*?)</strong>'
        r'.*?class="ag-science-current-question"><p>(.*?)</p>',
        html,
        re.S,
    )
    assert match, html[:400]
    return match.group(1), match.group(2)


def test_the_review_owners_decisions_are_found() -> None:
    assert_review_decision_sentinels(set(DECISIONS))


@pytest.mark.parametrize("code", DECISIONS)
def test_a_decision_reads_in_chinese(code: str) -> None:
    title, question = _current_decision(render(_review(code), lang="zh"))

    assert CJK.search(title) and "Host message" not in title
    assert CJK.search(question) and "Host question" not in question


@pytest.mark.parametrize("code", [
    "REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE", "SURVIVAL_LANDMARK_OWNER_NOT_SEALED",
])
def test_english_readers_keep_the_hosts_question(code: str) -> None:
    title, question = _current_decision(render(_review(code), lang="en"))

    assert title == f"Host message for {code}."
    assert question == f"Host question for {code}?"
