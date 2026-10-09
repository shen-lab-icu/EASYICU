"""A run file the browser may open shows what it holds.

``agent_runs._RUN_ARTIFACT_NAMES`` lists what a browser may open of a run, and
``agent_runs._public_review_payloads`` says what it is shown of each: the whole
record, or the fields its owner selects.  A file listed without a projection
opened as an empty view, so each listed file has one.  Synthetic payloads only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import agent_runs


@pytest.mark.parametrize("name", agent_runs._RUN_ARTIFACT_NAMES)
def test_each_run_file_the_browser_may_open_has_a_projection(name: str) -> None:
    projected = agent_runs._public_review_payloads({name: {"synthetic": True}})

    assert name in projected


def test_the_receipt_of_the_rendered_draft_opens_with_every_field(
    tmp_path: Path,
) -> None:
    receipt = {
        "schema_version": "easyicu.manuscript_pdf_receipt.v1",
        "status": "rendered",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "engine": "synthetic-engine",
        "security": {
            "network_allowed": False,
            "shell_escape_allowed": False,
            "untrusted_input_mode": True,
            "working_directory_restricted": True,
        },
        "draft_watermark": True,
        "source": {"name": "manuscript_scaffold.tex", "sha256": "a" * 64},
        "bibliography": None,
        "pdf": {"name": "manuscript_scaffold.pdf", "sha256": "b" * 64, "bytes": 1024},
    }
    (tmp_path / "manuscript_pdf_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    opened = agent_runs.read_run_artifact(str(tmp_path), "manuscript_pdf_receipt.json")

    assert opened["ok"] is True
    assert opened["payload"] == receipt
