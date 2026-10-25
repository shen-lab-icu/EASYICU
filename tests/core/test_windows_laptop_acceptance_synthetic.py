"""Real c–d owners on invented tables, with no server, sockets or downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.windows_laptop_demo import isolated_demo_paths, synthetic_demo_download
from tools import windows_laptop_acceptance as acceptance

pytestmark = pytest.mark.slow


def test_eicu_synthetic_all_module_export(tmp_path: Path, monkeypatch) -> None:
    from easyicu.webserver import sources
    from easyicu.webserver.demo_source_contracts import SOURCES

    work = tmp_path / "EasyICU 数据"
    work.mkdir()
    monkeypatch.setattr(sources, "_CONFIG_PATH", work / "home" / "sources.json")
    source = SOURCES[1]
    phases = {}
    with acceptance.MemoryPressure(0, work):
        acceptance.prepare_dataset(source, isolated_demo_paths(work, source), phases, fetch=synthetic_demo_download)
    assert phases["convert"]["status"] == "passed"
    assert phases["export"]["status"] == "passed"


def test_synthetic_conversion_export_and_read_owners(tmp_path: Path, monkeypatch) -> None:
    from easyicu.webserver import sources
    from easyicu.webserver.demo_source_contracts import SOURCES

    work = tmp_path / "EasyICU 数据"
    work.mkdir()
    monkeypatch.setattr(sources, "_CONFIG_PATH", work / "home" / "sources.json")
    monkeypatch.setenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", "1")
    phases = []
    exports = []
    with acceptance.MemoryPressure(0, work):
        for source in SOURCES:
            measurements = {}
            phases.append(measurements)
            exports.append(acceptance.prepare_dataset(
                source, isolated_demo_paths(work, source), measurements, fetch=synthetic_demo_download,
            ))
        from easyicu.webserver import catalog, cohort_review, crossdb_review, patient_drilldown

        def request(method, path, body):
            if path == "/api/catalog":
                return 200, catalog.build_catalog()
            owners = {
                "/api/cohort-review/summary": cohort_review.cohort_review_summary,
                "/api/patient-review/drilldown": patient_drilldown.patient_review_drilldown,
                "/api/crossdb-review/summary": crossdb_review.crossdb_review_summary,
            }
            return 200, owners[path](body)

        records = []
        acceptance.probe_read_paths(request, exports, records)
        with acceptance.laptop_environment(work, work / "cache", desktop_first_result=True):
            research = acceptance.probe_desktop_first_result(exports)
    assert len(records) == 7
    assert all(record["passed"] for record in records)
    assert research["source_validated"] is True
    assert research["runner_kind"] == "host"
    assert research["docker_reason_code"] == "docker_executable_missing"
    assert research["first_result"]["status"] == "blocked"
    assert research["provider_calls"] == 0
    for measurements in phases:
        assert measurements["convert"]["file_count"] > 0
        assert measurements["export"]["rows"] > 0
        assert measurements["export"]["modules"]
        assert all(phase["status"] == "passed" for phase in measurements.values())
    receipt = json.dumps({"phases": phases, "reads": records})
    assert "subject_id" not in receipt
    assert "synthetic-1" not in receipt
    assert str(work) not in receipt


def test_sparse_synthetic_run_registers_complete_exports_and_keeps_receipts_private(tmp_path: Path, monkeypatch) -> None:
    from easyicu.webserver import sources

    work = tmp_path / "EasyICU 数据"
    monkeypatch.setattr(sources, "_CONFIG_PATH", work / "home" / "sources.json")
    def complete_exports(_work, exports, metrics):
        assert len(exports) == 2
        assert all((path / "_manifest.json").is_file() for path in exports)
        metrics["export_count"] = len(exports)

    monkeypatch.setattr(acceptance, "probe_web", complete_exports)
    monkeypatch.setattr(acceptance, "probe_research", lambda: {
        "reason_code": "docker_executable_missing", "deterministic_analysis_without_docker": "no",
        "local_preflight_without_docker": "yes",
    })
    report = acceptance.run_acceptance(work, work / "cache", tmp_path / "receipts", memory_pressure_gb=0, fetch=synthetic_demo_download)
    assert report["status"] == "passed"
    assert report["failures"] == []
    assert report["web"]["export_count"] == 2
    assert report["memory"]["requested_bytes"] == 0
    assert report["memory"]["released"]["available_bytes"] > 0
    for dataset in report["datasets"]:
        assert dataset["phases"]["convert"]["status"] == "passed"
        assert dataset["phases"]["export"]["status"] == "passed"
        assert dataset["phases"]["export"]["file_count"] > 0
    receipt = json.dumps(report)
    assert str(work) not in receipt
    assert "synthetic-1" not in receipt
    assert "subject_id" not in receipt
    assert sources._CONFIG_PATH.exists()
