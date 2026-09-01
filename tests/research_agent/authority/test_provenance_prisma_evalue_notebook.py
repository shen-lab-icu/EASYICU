"""End-to-end coverage for O21 / O23 / O26 / O27 artefacts.

All four land in the write / finalise path of the pipeline, so it is
easier to pin them with a single integration test plus narrow unit
tests for the helpers than to mock the pipeline out piecewise.
"""

from __future__ import annotations

import csv
import hashlib
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
    assert (
        bundle.records[0].skipped_reason
        and "exceeds_cap" in bundle.records[0].skipped_reason
    )


# ---------------------------------------------------------------------------
# Unit: E-value
# ---------------------------------------------------------------------------


def test_e_value_for_or_above_one(ra):
    result = ra.compute_e_value(
        estimate=2.0,
        ci=(1.5, 2.5),
        estimate_type="or",
        baseline_prevalence=0.1,
    )
    assert result.e_value > 1.0
    # Lower CI bound gives a smaller E-value than the point estimate.
    assert result.e_value_lower_bound is not None
    assert result.e_value_lower_bound <= result.e_value


def test_e_value_for_protective_or(ra):
    result = ra.compute_e_value(
        estimate=0.5,
        ci=(0.3, 0.8),
        estimate_type="or",
        baseline_prevalence=0.2,
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
    assert "generated_by=easyicu.research_agent.replication.notebook" in text
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
        skill="association_analysis",
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
        r.get("role") == "raw_ingest" and r.get("sha256") for r in payload["records"]
    )

    # O21 — PRISMA
    assert (run_dir / "literature_prisma.json").exists()
    assert (run_dir / "literature_prisma.md").exists()
    prisma_payload = json.loads((run_dir / "literature_prisma.json").read_text())
    assert "prisma" in prisma_payload
    # The artifact is always published, but a PRISMA flow is only reported when
    # a retrieval source actually ran.  This run enables none, so it must say so
    # rather than presenting the preset reference list as a completed search.
    provenance = prisma_payload["search_provenance"]
    assert provenance["search_conducted"] is False
    assert provenance["sources_enabled"] == []
    assert prisma_payload["prisma"] is None
    assert "No PRISMA flow is reported" in (
        run_dir / "literature_prisma.md"
    ).read_text(encoding="utf-8")

    # O23 — E-values
    # Only required when primary_association produced a CSV; the
    # association-analysis skill with the mock cohort does. If so, the
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


def test_requirements_lockfile_uses_captured_docker_runtime(ra, tmp_path):
    captured = tmp_path / "runner_requirements.lock.txt"
    captured.write_text(
        "# runtime=docker\n" "# docker_image_id=sha256:abc\n" "numpy==2.0.0\n",
        encoding="utf-8",
    )

    text = ra.build_requirements_lockfile(captured)

    assert "# runtime=docker" in text
    assert "docker_image_id=sha256:abc" in text
    assert "numpy==2.0.0" in text


def _write_runner_snapshot(
    run_dir: Path, step_id: str, *, lock_text: str, image_id: str
) -> None:
    out_dir = run_dir / "steps" / step_id / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "runner_requirements.lock.txt").write_text(lock_text, encoding="utf-8")
    provenance = {
        "runtime": "docker",
        "image_reference": "easyicu:latest",
        "image_id": image_id,
        "repo_digests": [],
        "network": "none",
        "requirements_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
        "method_capabilities": ["pandas", "numpy"],
    }
    (out_dir / "runner_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def test_pipeline_write_accepts_identical_step_runtime_snapshots(tmp_path):
    from easyicu.research_agent.reporting.write_phase import _validated_runtime_lock

    run_dir = tmp_path / "run"
    lock_text = "# runtime=docker\nnumpy==2.0.0\n"
    image_id = "sha256:" + "a" * 64
    _write_runner_snapshot(run_dir, "one", lock_text=lock_text, image_id=image_id)
    _write_runner_snapshot(run_dir, "two", lock_text=lock_text, image_id=image_id)

    selected = _validated_runtime_lock(run_dir)

    assert (
        selected
        == run_dir / "steps" / "one" / "outputs" / "runner_requirements.lock.txt"
    )


def test_pipeline_write_rejects_inconsistent_step_runtime_snapshots(tmp_path):
    from easyicu.research_agent.reporting.write_phase import (
        RuntimeProvenanceMismatchError,
        _validated_runtime_lock,
    )

    run_dir = tmp_path / "run"
    lock_text = "# runtime=docker\nnumpy==2.0.0\n"
    _write_runner_snapshot(
        run_dir, "one", lock_text=lock_text, image_id="sha256:" + "a" * 64
    )
    _write_runner_snapshot(
        run_dir, "two", lock_text=lock_text, image_id="sha256:" + "b" * 64
    )

    with pytest.raises(RuntimeProvenanceMismatchError, match="inconsistent"):
        _validated_runtime_lock(run_dir)


def test_development_profile_allows_audited_multi_image_lineage() -> None:
    from easyicu.research_agent.reporting.write_phase import (
        _development_runtime_lineage_allowed,
    )

    class Pipeline:
        _development_diagnostic = False
        _submission_profile_name = "npj_dm_e1_demo_dev"

    class PaperPipeline:
        _development_diagnostic = False
        _submission_profile_name = "npj_dm"

    assert _development_runtime_lineage_allowed(Pipeline()) is True
    assert _development_runtime_lineage_allowed(PaperPipeline()) is False


def test_development_write_accepts_audited_multi_image_lineage(tmp_path):
    from easyicu.research_agent.reporting.write_phase import (
        _validated_runtime_lock,
        _write_development_runtime_lineage,
    )

    run_dir = tmp_path / "run"
    _write_runner_snapshot(
        run_dir,
        "one",
        lock_text=(
            "# runtime=docker\n"
            "# docker_image_id=sha256:old\n"
            "# execution_kernel_source_sha256=old\n"
            "numpy==2.0.0\n"
        ),
        image_id="sha256:" + "a" * 64,
    )
    _write_runner_snapshot(
        run_dir,
        "two",
        lock_text=(
            "# runtime=docker\n"
            "# docker_image_id=sha256:new\n"
            "# execution_kernel_source_sha256=new\n"
            "numpy==2.0.0\n"
        ),
        image_id="sha256:" + "b" * 64,
    )
    for step_id, suffix, file_count in (("one", "old", 10), ("two", "new", 11)):
        provenance_path = (
            run_dir / "steps" / step_id / "outputs" / "runner_provenance.json"
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.update(
            {
                "execution_kernel_identity_sha256": f"identity-{suffix}",
                "execution_kernel_source_sha256": f"source-{suffix}",
                "execution_kernel_files_sha256": f"files-{suffix}",
                "execution_kernel_file_count": file_count,
            }
        )
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    selected = _validated_runtime_lock(
        run_dir,
        allow_development_lineage=True,
    )
    lineage_path = _write_development_runtime_lineage(run_dir)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))

    assert (
        selected
        == run_dir / "steps" / "two" / "outputs" / "runner_requirements.lock.txt"
    )
    assert lineage["paper_authority"] is False
    assert lineage["diagnostic_only"] is True
    assert lineage["mixed_runtime_snapshots"] is True
    assert [row["step_id"] for row in lineage["steps"]] == ["one", "two"]
    assert all(len(row["provenance_sha256"]) == 64 for row in lineage["steps"])


def test_development_write_rejects_changed_dependency_pins(tmp_path):
    from easyicu.research_agent.reporting.write_phase import (
        RuntimeProvenanceMismatchError,
        _validated_runtime_lock,
    )

    run_dir = tmp_path / "run"
    _write_runner_snapshot(
        run_dir,
        "one",
        lock_text="# runtime=docker\nnumpy==2.0.0\n",
        image_id="sha256:" + "a" * 64,
    )
    _write_runner_snapshot(
        run_dir,
        "two",
        lock_text="# runtime=docker\nnumpy==2.1.0\n",
        image_id="sha256:" + "b" * 64,
    )

    with pytest.raises(RuntimeProvenanceMismatchError, match="dependency pins"):
        _validated_runtime_lock(run_dir, allow_development_lineage=True)


def test_development_write_rejects_changed_network_policy(tmp_path):
    from easyicu.research_agent.reporting.write_phase import (
        RuntimeProvenanceMismatchError,
        _validated_runtime_lock,
    )

    run_dir = tmp_path / "run"
    lock_text = "# runtime=docker\nnumpy==2.0.0\n"
    _write_runner_snapshot(
        run_dir,
        "one",
        lock_text=lock_text,
        image_id="sha256:" + "a" * 64,
    )
    _write_runner_snapshot(
        run_dir,
        "two",
        lock_text=lock_text,
        image_id="sha256:" + "b" * 64,
    )
    provenance_path = run_dir / "steps" / "two" / "outputs" / "runner_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["network"] = "bridge"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(RuntimeProvenanceMismatchError, match="Unsafe"):
        _validated_runtime_lock(run_dir, allow_development_lineage=True)


def test_pipeline_write_rejects_stale_registered_lock_on_resume(ra, tmp_path):
    from easyicu.research_agent.reporting.write_phase import (
        RuntimeProvenanceMismatchError,
        _assert_registered_runtime_lock_matches,
    )

    store = ra.EvidenceStore(tmp_path / "run")
    store.register_text(
        kind="log",
        description="old runtime lock",
        text="numpy==1.0.0\n",
        filename="requirements.lock.txt",
        evidence_id="requirements_lockfile",
    )
    current_lock = tmp_path / "run" / "requirements.lock.txt"
    current_lock.write_text("numpy==2.0.0\n", encoding="utf-8")

    with pytest.raises(RuntimeProvenanceMismatchError, match="already registered"):
        _assert_registered_runtime_lock_matches(store, current_lock)
