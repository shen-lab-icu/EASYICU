"""Shared static asset reader and Node preload fixture for Pi UI contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


STATIC = (
    Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "static"
)


NODE_APP = STATIC.parent / "pi_copilot" / "node_app"


NODE_MODULE_HARNESS = Path(__file__).resolve().parents[2] / "js" / (
    "guided_pi_module_harness.cjs"
)


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


# The screen modules destructure `esc` from window.EU_HTML at the top of their
# IIFE, so these Node harnesses have to install the shared escaping owner into
# the stub window before evaluating a module — the same order index.html uses.
_ESCAPE_OWNER = _read("js/html-escape.js")


@pytest.fixture(autouse=True)
def _load_guided_pi_module_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = os.environ.get("NODE_OPTIONS", "").strip()
    preload = f"--require={NODE_MODULE_HARNESS}"
    monkeypatch.setenv("NODE_OPTIONS", f"{preload} {existing}".strip())
