"""Every plan-review finding reads in Chinese.

The plan review lists each finding by a title and a detail.  The reader
vocabulary, or the renderer's own copy, owns their Chinese wording; without it
a Chinese reader sees the host's English message.  The findings are read from
the review owner's source, so one added later without Chinese wording fails
here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from tests.support.review_decisions import finding_codes
from tests.webserver.copilot.pi_copilot_static_fixtures import _read

CJK = re.compile(r"[一-鿿]")

FINDINGS = sorted(finding_codes({"PlanScientificFinding"}))


def _render_zh(findings: list[dict]) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    payload = {"approval_allowed": False, "status": "changes_required", "findings": findings}
    script = f"""
      global.window = {{
        EU_HTML: {{
          esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
          escAttr: value => String(value ?? ''),
        }},
        t: (en, zh) => zh,
        icon: () => '',
      }};
      eval({json.dumps(_read('js/screens-agent-reader-vocab.js'))});
      eval({json.dumps(_read('js/screens-agent-render.js'))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView('scientific_plan_review.json', {json.dumps(payload)}));
    """
    return subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout


def test_the_review_owner_findings_are_found() -> None:
    # Both branches of a code chosen by a condition.
    assert {
        "PRIMARY_MODEL_RETENTION_INSUFFICIENT",
        "PRIMARY_MODEL_RETENTION_REDUCED",
        "OUTCOME_DEFINITION_UNRESOLVED",
        "STUDY_TIME_ZERO_MISMATCH",
    } <= set(FINDINGS)


def test_every_plan_review_finding_has_a_chinese_title_and_detail() -> None:
    html = _render_zh(
        [
            {
                "code": code,
                "severity": "major",
                "remediation_route": "agent_plan_revision",
                "requires_user_authorization": False,
                "message": f"Host message for {code}.",
                "remediation": f"Host remediation for {code}.",
            }
            for code in FINDINGS
        ]
    )

    rows = re.findall(r"<li><strong>(.*?)</strong><span>(.*?)</span></li>", html, re.S)
    assert len(rows) == len(FINDINGS)
    english = [
        title
        for title, detail in rows
        if not CJK.search(title) or not CJK.search(detail)
    ]
    assert english == []
