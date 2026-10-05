"""The review findings that hand a decision to the researcher.

They are read from the review owners' source, so a test over every decision
also covers one added later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import easyicu.research_agent as research_agent

FINDING_TYPES = {"PlanScientificFinding", "ScientificMaturityFinding"}


def _strings(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def decision_codes() -> set[str]:
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
