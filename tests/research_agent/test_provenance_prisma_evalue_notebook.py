"""End-to-end coverage for O21 / O23 / O26 / O27 artefacts.

All four land in the write / finalise path of the pipeline, so it is
easier to pin them with a single integration test plus narrow unit
tests for the helpers than to mock the pipeline out piecewise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Unit: hash_sources + build_provenance_bundle
# ---------------------------------------------------------------------------


def test_hash_sources_records_every_file(ra, tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello\n")
    b = tmp_path / "sub" / "b.txt"
    b.parent.mkdir(parents=True)
    b.write_text("world\n")
    bundle = ra.hash_sources(
        [
            {"path": str(a), "role": "raw_csv", "database": "miiv"},
            {"path": str(tmp_path / "sub"), "role": "raw_dir", "database": "miiv"},
        ]
    )
    paths = {rec.relative_path for rec in bundle.records}
    assert "a.txt" in paths
    assert any("b.txt" in p for p in paths)
    for rec in bundle.records:
        if rec.sha256:
            assert len(rec.sha256) == 64
    summary = bundle.summary()
    assert summary["n_sources"] >= 2
    assert summary["n_hashed"] >= 2


def test_hash_sources_respects_size_cap(ra, tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 2048)
    bundle = ra.hash_sources(
        [{"path": str(big), "role": "raw"}],
        max_bytes_per_file=1024,
    )
    assert len(bundle.records) == 1
    assert bundle.records[0].sha256 is None
    assert bundle.records[0].skipped_reason and "exceeds_cap" in bundle.records[0].skipped_reason


# ---------------------------------------------------------------------------
# Unit: E-value
# ---------------------------------------------------------------------------


def test_e_value_for_or_above_one(ra):
    result = ra.compute_e_value(
        estimate=2.0, ci=(1.5, 2.5), estimate_type="or", baseline_prevalence=0.1,
    )
    assert result.e_value > 1.0
    # Lower CI bound gives a smaller E-value than the point estimate.
    assert result.e_value_lower_bound is not None
    assert result.e_value_lower_bound <= result.e_value


def test_e_value_for_protective_or(ra):
    result = ra.compute_e_value(
        estimate=0.5, ci=(0.3, 0.8), estimate_type="or", baseline_prevalence=0.2,
    )
    # Protective effect: E-value is still > 1
    assert result.e_value > 1.0


def test_e_value_raises_on_unsupported_type(ra):
    with pytest.raises(ValueError):
        ra.compute_e_value(estimate=2.0, estimate_type="xyz")


# ---------------------------------------------------------------------------
# Unit: requirements lockfile + notebook
# ---------------------------------------------------------------------------


def test_requirements_lockfile_has_header_and_entries(ra):
    text = ra.build_requirements_lockfile()
    assert "python_version=" in text
    assert "generated_by=easyicu.research_agent.repro_artifacts" in text
    # At least one entry that looks like package==version
    entry_lines = [l for l in text.splitlines() if "==" in l and not l.startswith("#")]
    assert entry_lines


def test_notebook_is_valid_nbformat(ra):
    nb = ra.build_notebook(
        research_question="Q",
        cohort_relative_path="cohort.parquet",
        steps=[
            ra.NotebookStep(step_id="01", intent="table_one", code="print('hi')\n"),
        ],
    )
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    # Expect at least: intro markdown + env-var code + per-step markdown + code
    assert len(nb["cells"]) >= 4
    assert any(c["cell_type"] == "markdown" for c in nb["cells"])
    assert any(c["cell_type"] == "code" for c in nb["cells"])


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_writes_provenance_prisma_evalue_notebook_lockfile(
    ra, synthetic_cohort, tmp_path
):
    # Provide a small source file to hash through --source_files.
    raw = tmp_path / "raw_extract.csv"
    raw.write_text("stay_id,dummy\n1,2\n")

    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="sofa_mortality",
        cohort=cohort_path,
        database="miiv",
        source_files=[{"path": str(raw), "role": "raw_ingest", "database": "miiv"}],
    )
    run_dir = Path(result.manifest_path).parent

    # O27 — provenance
    assert (run_dir / "provenance_sources.json").exists()
    payload = json.loads((run_dir / "provenance_sources.json").read_text())
    assert payload["summary"]["n_sources"] >= 2  # cohort + raw
    # Raw-ingest file must be hashed
    assert any(
        r.get("role") == "raw_ingest" and r.get("sha256")
        for r in payload["records"]
    )

    # O21 — PRISMA
    assert (run_dir / "literature_prisma.json").exists()
    assert (run_dir / "literature_prisma.md").exists()
    prisma_payload = json.loads((run_dir / "literature_prisma.json").read_text())
    assert "prisma" in prisma_payload
    assert "identified" in prisma_payload["prisma"]

    # O23 — E-values
    # Only required when primary_association produced a CSV; the
    # sofa_mortality skill with the mock cohort does. If so, the
    # e_values.csv must list at least one row.
    ev_csv = run_dir / "e_values.csv"
    if ev_csv.exists():
        with ev_csv.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        assert all(r.get("e_value") for r in rows)

    # O26 — notebook + lockfile
    assert (run_dir / "requirements.lock.txt").exists()
    lock_text = (run_dir / "requirements.lock.txt").read_text()
    assert "python_version" in lock_text
    # Notebook is best-effort: only asserted if per-step scripts
    # registered at least one 'code' evidence.
    nb_path = run_dir / "run.ipynb"
    if nb_path.exists():
        nb_payload = json.loads(nb_path.read_text())
        assert nb_payload["nbformat"] == 4

    # Evidence ids wired
    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = {r["evidence_id"] for r in manifest["evidence"]}
    assert "provenance_sources" in ev_ids
    assert "literature_prisma" in ev_ids
    assert "requirements_lockfile" in ev_ids
