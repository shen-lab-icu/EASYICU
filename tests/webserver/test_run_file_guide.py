"""A run download reads like a folder: the report first, reader copies, a README.

The review and the download share one rule for the run's report, and the
bundle's figure and table files are decoded from the reviewed payloads.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path

from easyicu.webserver import agent_runs
from easyicu.webserver.run_file_guide import bundle_files, run_file_guide

_PNG = b"\x89PNG\r\n\x1a\n" + bytes(16)


def _gate(*checks: tuple[str, object]) -> bytes:
    rows = [{"id": check_id, "passed": passed} for check_id, passed in checks]
    return json.dumps({"gate": {"status": "analysis_only", "checks": rows}}).encode()


def _placed(names, checks):
    return {
        entry.name: (entry.folder, entry.title_en)
        for entry in run_file_guide(names, gate_checks=checks)
    }


def test_the_first_report_is_history_a_manuscript_or_a_draft() -> None:
    names = ["manuscript_scaffold.pdf", "manuscript_scaffold.tex"]
    passed = [{"id": "manuscript_ready", "passed": True}]

    revised = _placed([*names, "manuscript_revision.pdf"], passed)
    assert revised["manuscript_scaffold.pdf"] == ("provenance", "Original run report (PDF)")
    assert revised["manuscript_scaffold.tex"][0] == "provenance"
    assert revised["manuscript_revision.pdf"][0] == "manuscript"
    assert _placed(names, passed)["manuscript_scaffold.pdf"] == ("manuscript", "Manuscript PDF")
    # Only an explicit pass names it a manuscript, not a gate without the check.
    for checks in (
        [{"id": "manuscript_ready", "passed": False}],
        [{"id": "manuscript_ready", "passed": "true"}],
        [],
    ):
        assert _placed(names, checks)["manuscript_scaffold.pdf"] == (
            "manuscript", "Report draft (PDF)",
        )


def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _contents() -> dict[str, bytes]:
    table = {
        "name": "table_step_artifact_ab__absolute_risk.csv",
        "label": "Table absolute_risk from step risk.",
        "headers": ["group", "risk"],
        "rows": [["A, first", 0.25], ["B", None]],
        "preview_truncated": False,
        "preview_columns_truncated": True,
    }
    return {
        "quality_gate.json": _gate(("manuscript_ready", False), ("evidence_complete", False)),
        "run_context.json": json.dumps({
            "run_id": "run_fixture",
            "question": "Is the first lactate associated with ICU readmission?",
            "source": {"label": "Synthetic ICU source"},
        }).encode(),
        "manuscript_scaffold.pdf": b"%PDF-1.7 fixture",
        "agent_plan.json": b"{}",
        "figure_gallery.json": json.dumps({"figures": [
            {"name": "cohort_flow.png", "label": "figure:cohort flow",
             "caption": "Cohort accounting.", "data_url": _data_url(_PNG)},
            {"name": "forged.png", "label": "Forged", "data_url": _data_url(b"not an image")},
            {"name": "../escape.png", "label": "Escape", "data_url": _data_url(_PNG)},
        ]}).encode(),
        "result_tables.json": json.dumps({"tables": [
            table,
            {**table, "name": "copy__absolute_risk.csv", "label": "Source data beside a figure."},
        ]}).encode(),
    }


def test_the_bundle_folders_each_file_and_adds_reader_copies() -> None:
    contents = _contents()

    files = dict(bundle_files(contents))

    assert next(iter(files)) == "README.md"
    # Each reviewed file keeps its bytes, in its folder.
    assert files["manuscript/manuscript_scaffold.pdf"] == contents["manuscript_scaffold.pdf"]
    assert files["checks/quality_gate.json"] == contents["quality_gate.json"]
    assert files["plan/agent_plan.json"] == b"{}"
    # A figure is the gallery's own image; a name cannot leave its folder and
    # bytes that are not the declared image are not copied.
    assert files["results/figures/cohort_flow.png"] == _PNG
    assert files["results/figures/escape.png"] == _PNG
    assert not any("forged" in path for path in files)
    # One CSV per aggregate, even when a copy sits beside a figure.
    assert files["results/tables/absolute_risk.csv"].decode() == 'group,risk\n"A, first",0.25\nB,\n'
    assert sum(path.startswith("results/tables/") for path in files) == 1

    readme = files["README.md"].decode()
    assert "- `manuscript_scaffold.pdf` — Report draft (PDF) · 报告草稿（PDF）" in readme
    assert "- `figures/cohort_flow.png` — Cohort flow · 图件" in readme
    assert "- `tables/absolute_risk.csv` — Absolute risk · 结果表 CSV" in readme
    assert "Some columns omitted" in readme
    assert "Question · 研究问题: Is the first lactate associated with ICU readmission?" in readme
    assert "manuscript ready · 稿件就绪; evidence complete · 证据完整" in readme
    assert "grant no publication authority" in readme
    assert readme.index("## manuscript/") < readme.index("## results/") < readme.index("## plan/")


def test_the_download_and_the_review_read_one_report_rule(tmp_path) -> None:
    (tmp_path / "quality_gate.json").write_bytes(_gate(("manuscript_ready", True)))
    (tmp_path / "evidence_ledger.json").write_text(json.dumps({"run_id": "run_fixture"}))
    (tmp_path / "manuscript_scaffold.pdf").write_bytes(b"%PDF-1.7 fixture")

    bundle = agent_runs.build_run_bundle(str(tmp_path))
    review = agent_runs.read_run_review(str(tmp_path))

    with zipfile.ZipFile(io.BytesIO(bundle["content"])) as archive:
        assert archive.read("manuscript/manuscript_scaffold.pdf") == b"%PDF-1.7 fixture"
        assert "Manuscript PDF · 稿件 PDF" in archive.read("README.md").decode()
    [report] = [row for row in review["file_guide"] if row["name"] == "manuscript_scaffold.pdf"]
    assert report["title"] == {"en": "Manuscript PDF", "zh": "稿件 PDF"}
    assert report["bundle_path"] == "manuscript/manuscript_scaffold.pdf"


_JS_ENTRY = re.compile(r"'([^']+)': t\('((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)'\)")


def _js_labels(source: str, start: str, end: str) -> dict[str, tuple[str, str]]:
    body = source[source.index(start):source.index(end)]
    return {name: (en.replace("\\'", "'"), zh) for name, en, zh in _JS_ENTRY.findall(body)}


def test_the_guide_reads_like_the_run_review_vocabulary() -> None:
    """The review list shows the guide's words; the shared vocabulary may not drift."""
    source = (
        Path(agent_runs.__file__).parent / "static" / "js" / "screens-agent-render.js"
    ).read_text(encoding="utf-8")
    titles = _js_labels(source, "function artifactTitle", "function artifactCategory")
    purposes = _js_labels(source, "function artifactSummary", "function artifactRank")
    # With a revision present the first report reads as the history it is.
    guide = {
        entry.name: entry
        for entry in run_file_guide(
            [*titles, *purposes, "manuscript_revision.pdf"], gate_checks=[],
        )
    }
    # The reader file says what it holds; a README cannot be clicked.
    exceptions = {"manuscript_provenance.json"}

    assert titles and purposes
    for name, (en, zh) in titles.items():
        assert (guide[name].title_en, guide[name].title_zh) == (en, zh), name
    for name, (en, zh) in purposes.items():
        if name not in exceptions:
            assert (guide[name].purpose_en, guide[name].purpose_zh) == (en, zh), name
