"""Research Copilot — pure model-output sanitisers.

Extracted from `llm_chat.py` (incremental Phase-6 split, see
`easyicu美化/copilot_接线施工计划.md` §6.1). These helpers parse / clean raw
OpenAI-compatible model output and have zero dependency on Streamlit session
state, so they unit-test standalone.

The provider/client layer (`_get_client`, `_is_configured`,
`_external_llm_ready`, ...) deliberately stays in `llm_chat.py`: the test-suite
swaps the whole module's Streamlit handle via
`monkeypatch.setattr(llm_chat, "st", ...)`, so those functions must keep
resolving `st` from `llm_chat`'s namespace. Moving them here would silently
bypass that fake and read the real session state.
"""
from __future__ import annotations

import re


def _strip_llm_reasoning(text: str) -> str:
    """Remove model-private reasoning blocks from OpenAI-compatible outputs."""
    if not text:
        return ""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.I | re.S)
    return cleaned.strip()


def _parse_verification_report(text: str) -> dict[str, object]:
    """Parse verifier output into a structured result."""
    text = _strip_llm_reasoning(text)
    result = {
        "status": "uncertain",
        "issues": [],
        "corrected_answer": "",
        "raw": text.strip(),
    }
    if not text:
        return result

    status_match = re.search(r"STATUS:\s*(pass|corrected|uncertain)", text, re.I)
    if status_match:
        result["status"] = status_match.group(1).lower()

    issues_match = re.search(r"ISSUES:\s*(.*?)\nCORRECTED_ANSWER:", text, re.S | re.I)
    if issues_match:
        issue_block = issues_match.group(1).strip()
        result["issues"] = [
            line.lstrip("-* ").strip()
            for line in issue_block.splitlines()
            if line.strip()
        ]

    corrected_match = re.search(r"CORRECTED_ANSWER:\s*(.*)$", text, re.S | re.I)
    if corrected_match:
        result["corrected_answer"] = corrected_match.group(1).strip()

    return result
