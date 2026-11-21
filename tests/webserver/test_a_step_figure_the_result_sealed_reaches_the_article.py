"""A step's figure the result sealed reaches the article; a different figure never does.

UX rc_077 age leg: 扩展成完整论文 returned 409 "The earlier figure is not
sealed evidence." The gallery showed the cohort-flow figure at its step output
path while the sealed result held the same bytes under its evidence copy.
"""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.webserver.research_append_history import adopt_result_figures

STEP_PATH = "steps/family_cohort_flow_figure/outputs/cohort_flow.png"


def _source(tmp_path, *, shown: bytes):
    run_dir = tmp_path / "source" / "pipeline" / "run_source"
    out = run_dir / "steps" / "family_cohort_flow_figure" / "outputs"
    out.mkdir(parents=True)
    produced = out / "cohort_flow.png"
    produced.write_bytes(b"\x89PNG cohort flow")
    record = EvidenceStore(run_dir).register_file(
        kind="figure", description="Figure cohort_flow from step family_cohort_flow_figure.",
        source_path=produced, produced_by_step="family_cohort_flow_figure",
    )
    assert record.relative_path != STEP_PATH
    produced.write_bytes(shown)
    (run_dir / "figure_gallery.json").write_text(json.dumps({"figures": [
        {"figure_id": "figure:cohort_flow", "relative_path": STEP_PATH, "status": "supporting"},
    ]}))
    seal = {"run_id": "run_source", "products": {"figure:cohort_flow": record.model_dump(mode="json")}}
    return SimpleNamespace(run_dir=run_dir, seal=seal), record


def test_the_step_output_with_the_sealed_bytes_is_adopted(tmp_path) -> None:
    source, record = _source(tmp_path, shown=b"\x89PNG cohort flow")
    child = EvidenceStore(tmp_path / "child")

    figure, = adopt_result_figures(source, None, child)

    adopted = child.get(figure["figure_id"])
    assert adopted.sha256 == record.sha256 and figure["status"] == "available"
    assert adopted.metadata["origin"]["evidence_id"] == record.evidence_id


def test_a_gallery_figure_whose_bytes_were_never_sealed_is_refused(tmp_path) -> None:
    source, _ = _source(tmp_path, shown=b"\x89PNG a different figure")

    with pytest.raises(ValueError, match="not sealed evidence"):
        adopt_result_figures(source, None, EvidenceStore(tmp_path / "child"))
