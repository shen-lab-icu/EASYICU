"""The workflow demo's trace record is the registered run's, and only aggregates.

The demo walkthrough's traceability cards read
``static/assets/demo/sofa2-run-trace.json``, built from the source run by
``tools/build_demo_walkthrough_trace.py``. These checks hold the record to the
numbers the demo states, to the run's registered figure, to its own recorded
times, and to the aggregate-only boundary, without needing the run directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "src/easyicu/webserver/static"
TRACE = STATIC / "assets/demo/sofa2-run-trace.json"
TOOL = ROOT / "tools/build_demo_walkthrough_trace.py"


def _trace() -> dict:
    return json.loads(TRACE.read_text(encoding="utf-8"))


def _tool():
    spec = importlib.util.spec_from_file_location("build_demo_walkthrough_trace", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _seconds(span: list[str]) -> float:
    start, end = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in span)
    return (end - start).total_seconds()


def test_the_record_belongs_to_the_demo_run_and_its_registered_figure() -> None:
    trace = _trace()
    demo = (STATIC / "js/screens-guided-pi-demo.js").read_text(encoding="utf-8")
    assert trace["schema_version"] == "easyicu.demo-walkthrough-trace/1"
    assert f"SOURCE_RUN_ID = '{trace['source_run_id']}'" in demo
    figure = trace["evidence"]["records"]["figure_step_artifact_1c86f17d397e23b3"]
    image = (STATIC / "assets/demo/sofa2-phenotype-mortality.png").read_bytes()
    assert hashlib.sha256(image).hexdigest() == figure["sha256"]
    # The chain runs from the figure back to the result table it was drawn from.
    records = trace["evidence"]["records"]
    table_id = trace["evidence"]["chain"][0]
    assert records[table_id]["script"] == trace["evidence"]["chain"][1]
    assert figure["inputs"] == [table_id]


def test_the_numbers_the_demo_states_are_the_result_table() -> None:
    trace = _trace()
    demo = (STATIC / "js/screens-guided-pi-demo.js").read_text(encoding="utf-8")
    rows = {row["level"]: row for row in trace["evidence"]["result_rows"]}
    assert len(trace["evidence"]["result_rows"]) == 3
    for row in rows.values():
        assert row["interval_method"] == "none_counts_only"
        assert row["stays"] / row["denominator"] * 100 == pytest.approx(row["share_pct"], abs=1e-5)
        assert row["deaths"] / row["outcome_denominator"] * 100 == pytest.approx(row["rate_pct"], abs=1e-5)
    with_phenotype, without, overall = rows[1], rows[0], rows[None]
    stated = (
        f"有表型组 {with_phenotype['rate_pct']:.2f}%（{with_phenotype['deaths']:,}/{with_phenotype['stays']:,}），"
        f"无表型组 {without['rate_pct']:.2f}%（{without['deaths']:,}/{without['stays']:,}）；"
        f"总体 {overall['rate_pct']:.2f}%（{overall['deaths']:,}/{overall['stays']:,}）"
    )
    assert stated in demo
    assert f"{with_phenotype['stays']:,} 个（{with_phenotype['share_pct']:.2f}%）" in demo
    assert trace["cohort"]["n_stays"] == overall["stays"]


def test_recorded_times_are_ordered_and_match_the_run() -> None:
    spans = _trace()["spans"]
    order = ["p-scope", "p-cohort", "p-literature", "p-draft", "p-review", "review-wait",
             "e-2", "e-3", "e-4", "e-5", "e-6", "e-figures", "e-writer", "e-register"]
    assert set(order) <= set(spans)
    for earlier, later in zip(order, order[1:]):
        assert spans[earlier][1] <= spans[later][0] or earlier == "p-draft", (earlier, later)
    for name in order:
        assert _seconds(spans[name]) >= 0
    # The plan's one draft took most of the planning time; the researcher's
    # review of it took minutes. Neither is a hand-typed figure.
    assert 60 < _seconds(spans["p-draft"]) < 120
    assert 300 < _seconds(spans["review-wait"]) < 360


def test_the_dictionary_entries_are_the_versions_the_run_fingerprinted() -> None:
    trace = _trace()
    concepts = trace["dictionary"]["concepts"]
    assert [term["concept"] for term in trace["dictionary"]["terms"]] == ["sep3_sofa2", "death", "age"]
    assert concepts["sep3_sofa2"]["children"] == ["sofa2", "susp_inf"]
    assert concepts["death"]["miiv"][0]["table"] == "admissions"
    assert all(not entry.get("missing") for entry in concepts.values())
    for name, path in (("concept_dict", "src/easyicu/data/concept-dict.json"), ("sofa2_dict", "src/easyicu/data/sofa2-dict.json")):
        source = trace["sources"][name]
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{source['commit']}:{path}"], capture_output=True)
        if blob.returncode != 0:
            pytest.skip("repository history is not available in this checkout")
        assert hashlib.sha256(blob.stdout).hexdigest() == source["sha256"]


def test_the_record_carries_aggregates_only() -> None:
    text = TRACE.read_text(encoding="utf-8")
    trace = json.loads(text)
    # No local path, account or patient identity leaves the run.
    assert not re.search(r"/(Users|Volumes|home|private)/", text)
    for identity in ("stay_id", "subject_id", "hadm_id", "patientunitstayid"):
        assert f'"{identity}"' not in text
    assert set(trace) == {
        "schema_version", "source_run_id", "sources", "started_at", "finished_at", "spans",
        "dictionary", "cohort", "plan", "literature", "evidence", "writer",
    }
    assert len(trace["evidence"]["code_excerpt"]) <= 12


def test_the_builder_reads_spans_and_concepts_without_guessing() -> None:
    tool = _tool()
    audit = [
        {"timestamp": "2026-01-01T00:00:00Z", "event": "Execution runtime validated before planning."},
        {"timestamp": "2026-01-01T00:00:02Z", "event": "Research context built."},
    ]
    original = tool.SPANS
    try:
        tool.SPANS = (("p-scope", r"^Execution runtime validated", r"^Research context built"),)
        assert tool.spans(audit) == {"p-scope": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:02Z"]}
        tool.SPANS = (("p-scope", r"^Research context built", r"^Execution runtime validated"),)
        with pytest.raises(SystemExit, match="ends before it starts"):
            tool.spans(audit)
    finally:
        tool.SPANS = original
    closure = tool.concept_closure(
        ["a"], {"a": {"concepts": ["b", "c"], "description": "x"}},
        {"b": {"description": "y", "sources": {"miiv": [{"table": "t", "ids": [1], "sub_var": "itemid", "secret": 1}]}}},
    )
    assert closure["a"]["children"] == ["b", "c"]
    assert closure["b"]["miiv"] == [{"table": "t", "ids": [1], "sub_var": "itemid"}]
    assert closure["c"] == {"missing": True}
    with pytest.raises(SystemExit, match="no translation"):
        tool.zh("An untranslated record sentence.")
