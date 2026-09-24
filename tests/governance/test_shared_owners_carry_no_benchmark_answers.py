"""Shared model-facing owners carry no answer to a development question.

The development questions (E1-E3, M1-M3, H1-H3) exist to expose general
defects.  A prompt, method package, family template or decision card that
names one of them, or ships a worked example written for it, hands the model
that question's design; a later pass then measures the example, not the
system.  Case-specific requirements belong in the item, its rubric, or the
reviewed run plan.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "easyicu"

#: Owners whose text reaches the Planner, the Coder, Copilot, or a decision card.
MODEL_FACING = (
    "research_agent/providers/prompts/**/*",
    "research_agent/skill_packages/**/*.md",
    "research_agent/skill_packages/**/*.py",
    "research_agent/planning/family_spec/*.py",
    "research_agent/method_skills.py",
    "data/research_know_how/*.json",
    "webserver/pi_copilot/plan_decisions.py",
    "webserver/static/js/screens-guided-pi-confirmation.js",
)

#: A development-question id is written in capitals (``E3``, ``M1``, ``H2``).
QUESTION_ID = re.compile(r"\b[EMH][1-3]\b")
BENCHMARK_SET = re.compile(
    r"canonical[-_ ]?9|\bdev9\b|nine[- ]question|九题|九问", re.IGNORECASE
)


def _model_facing_files() -> list[Path]:
    files = {
        path
        for pattern in MODEL_FACING
        for path in SOURCE.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return sorted(files)


def test_the_scan_covers_every_model_facing_owner() -> None:
    """A moved or renamed owner must not silently leave the scan."""

    for pattern in MODEL_FACING:
        assert any(path.is_file() for path in SOURCE.glob(pattern)), pattern


def test_no_model_facing_owner_names_a_development_question() -> None:
    offenders = [
        f"{path.relative_to(SOURCE).as_posix()}:{number}: {line.strip()}"
        for path in _model_facing_files()
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if QUESTION_ID.search(line) or BENCHMARK_SET.search(line)
    ]

    assert offenders == []
