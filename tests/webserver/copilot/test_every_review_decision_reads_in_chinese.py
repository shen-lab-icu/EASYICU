"""Every decision a plan review hands the researcher reads in Chinese.

A review finding that needs the researcher's authorization becomes the plan
review's current decision: a title, and the question to answer in the
conversation.  The reader vocabulary owns their Chinese wording; English
readers keep the host's own text.  The decisions are read from the review
owners' source, so a new decision without Chinese wording fails here.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import easyicu.research_agent as research_agent
from tests.webserver.copilot.pi_copilot_static_fixtures import _read

FINDING_TYPES = {"PlanScientificFinding", "ScientificMaturityFinding"}
CJK = re.compile(r"[一-鿿]")


def _strings(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _decision_codes() -> set[str]:
    """Codes of every review finding built with a researcher authorization.

    A code chosen by the same condition as its authorization (``"A" if
    declared else "B"`` with ``requires_user_authorization=declared``) is a
    decision only on its first branch.
    """

    codes: set[str] = set()
    for path in Path(research_agent.__file__).parent.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if not any(f"{name}(" in text for name in FINDING_TYPES):
            continue
        for call in ast.walk(ast.parse(text)):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", getattr(call.func, "attr", None))
            if name not in FINDING_TYPES:
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            authorization = keywords.get("requires_user_authorization")
            if authorization is None or (
                isinstance(authorization, ast.Constant) and authorization.value is False
            ):
                continue
            code = keywords["code"]
            if isinstance(code, ast.IfExp) and ast.dump(code.test) == ast.dump(authorization):
                code = code.body
            codes.update(_strings(code))
    return codes


DECISIONS = sorted(_decision_codes())


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
    return subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout


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
    # Both owners, a code chosen by a condition, and one whose authorization
    # follows that condition.
    assert {
        "REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE",
        "REPEATED_STAY_METHOD_NOT_DECLARED",
        "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH",
        "PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED",
        "POPULATION_SCOPE_AMENDMENT_DECLARED",
    } <= set(DECISIONS)
    assert "PLAN_POPULATION_REQUIREMENT_DRIFT" not in DECISIONS


@pytest.mark.parametrize("code", DECISIONS)
def test_a_decision_reads_in_chinese(code: str) -> None:
    title, question = _current_decision(render(_review(code), lang="zh"))

    assert CJK.search(title) and "Host message" not in title
    assert CJK.search(question) and "Host question" not in question


def test_english_readers_keep_the_hosts_question() -> None:
    code = "REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE"

    title, question = _current_decision(render(_review(code), lang="en"))

    assert title == f"Host message for {code}."
    assert question == f"Host question for {code}?"
