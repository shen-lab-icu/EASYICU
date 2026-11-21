"""The model-request capacity stop shows how to continue (0.1.3.1).

An orphaned slot is reclaimed on its own once its worker is verifiably gone;
until then, or when that cannot be shown, the card says what to do and that the
earlier request may still be running, instead of ending at "do not retry".
"""

from __future__ import annotations

from pathlib import Path

import easyicu.webserver.agent_pipeline_runs as runs

ROOT = Path(runs.__file__).parent
SLOT_FILE = "~/.easyicu/metadata-provider-slots.sqlite3"


def test_the_card_copy_names_the_file_to_move_and_the_earlier_request_risk():
    text = (ROOT / "static/js/screens-guided-pi-error-text.js").read_text(encoding="utf-8")
    start = text.index("research_pipeline_metadata_provider_capacity_unavailable: tr(")
    block = text[start:text.index("research_pipeline_metadata_shared_scope_unavailable", start)]

    assert SLOT_FILE in block and "keep the file, do not delete it" in block
    assert "may still be running" in block
    assert SLOT_FILE in block and "保留文件，不要删除" in block and "可能仍在运行" in block


def test_the_run_message_carries_the_same_recovery():
    source = (ROOT / "agent_pipeline_runs.py").read_text(encoding="utf-8")
    start = source.index('elif code == "research_pipeline_metadata_provider_capacity_unavailable":')
    block = source[start:source.index('elif code == "research_pipeline_metadata_shared_scope_unavailable"', start)]

    assert "metadata-provider-slots.sqlite3" in block and "do not delete it" in block
    assert "may still be running" in block
