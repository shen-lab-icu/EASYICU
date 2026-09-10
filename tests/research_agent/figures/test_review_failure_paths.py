"""Malformed registered inputs cannot become complete publication figures."""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.figures import base
from easyicu.research_agent.gates.figure_privacy import _rendered_text
from easyicu.research_agent.discovery.discovery_story_figure import (
    _has_blocked_outcome_gate,
)


def test_registered_unreadable_panel_is_not_reported_as_absent(monkeypatch, tmp_path):
    record = SimpleNamespace(evidence_id="missingness")
    monkeypatch.setattr(base, "find_table_records", lambda *a: [record])
    monkeypatch.setattr(base, "verified_record_path", lambda *a: tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="missingness"):
        base.first_normalisable_record(
            None, ["missingness"], run_dir=tmp_path, normalise=lambda f: f
        )


def test_reader_caption_is_part_of_privacy_inspection():
    assert "Patient 12345678" in _rendered_text(
        SimpleNamespace(reader_caption="Patient 12345678")
    )


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("step_summary.json", b"broken json"),
        ("outcome_feasibility_gate.csv", b"\xff\xfe"),
    ],
)
def test_unreadable_outcome_gate_cannot_be_rendered_as_clear(
    tmp_path, filename, contents
):
    folder = tmp_path / "steps" / "one" / "outputs"
    folder.mkdir(parents=True)
    (folder / filename).write_bytes(contents)
    with pytest.raises(ValueError, match="Cannot assess"):
        _has_blocked_outcome_gate(tmp_path)


def test_source_csv_failure_propagates_before_contract_registration(
    monkeypatch, tmp_path
):
    from easyicu.research_agent.figures.skill import PublicationFigureSkill
    from easyicu.research_agent.figures import skill

    monkeypatch.setattr(skill, "_source_fingerprint_metadata", lambda *a: {})

    def fail(*a, **kw):
        raise OSError("disk write failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    rendered = SimpleNamespace(
        source_frames={"required": pd.DataFrame({"v": [1]})}, source_evidence_ids=[]
    )
    with pytest.raises(ValueError, match="Required figure source CSV"):
        PublicationFigureSkill()._finalise_family_figure(
            context=None,
            evidence=None,
            run_dir=tmp_path,
            rendered=rendered,
            prompt_pack_version=None,
        )
