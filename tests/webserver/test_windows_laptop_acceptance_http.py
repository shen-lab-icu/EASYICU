"""Full synthetic c–d acceptance; run on a host allowing loopback sockets.

Server-lane workers must leave this file for the lead's Mac: it starts the real
webserver wrapper and uvicorn child, never ASGI TestClient or a browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.windows_laptop_demo import synthetic_demo_download
from tools import windows_laptop_acceptance as acceptance


@pytest.mark.slow
def test_webserver_wrapper_and_uvicorn_child_are_reaped(tmp_path: Path) -> None:
    work = tmp_path / "EasyICU 数据"
    work.mkdir()
    with acceptance.laptop_environment(work, work / "cache"), acceptance.LocalWebServer(work) as server:
        assert server.request("GET", "/api/health", None)[0] == 200
        assert server.request("GET", "/api/catalog", None)[0] == 200
        process = acceptance.psutil.Process(server.process.pid)
        children = process.children(recursive=True)
        assert children
    assert server.process.poll() is not None
    assert all(not child.is_running() for child in children)
    acceptance.check_server_log(work / "webserver.log")


@pytest.mark.slow
@pytest.mark.parametrize("desktop_first_result", [False, True])
def test_synthetic_laptop_acceptance_over_http(tmp_path: Path, monkeypatch, desktop_first_result: bool) -> None:
    from easyicu.webserver import sources

    work = tmp_path / "EasyICU 数据"
    monkeypatch.setattr(sources, "_CONFIG_PATH", work / "home" / ".easyicu" / "webserver_sources.json")
    report = acceptance.run_acceptance(
        work, work / "demo-cache", tmp_path / "receipts", memory_pressure_gb=0,
        fetch=synthetic_demo_download,
        desktop_first_result=desktop_first_result,
    )
    assert report["failures"] == []
    assert report["status"] == ("blocked" if desktop_first_result else "passed")
    if desktop_first_result:
        assert report["research"]["source_validated"] is True
        assert report["research"]["runner_kind"] == "host"
        assert report["research"]["first_result"]["evidence_bound_result"] is False
        assert report["research"]["first_result"]["elapsed_to_boundary_seconds"] > 0
    from easyicu.webserver.demo_source_contracts import SOURCES

    # catalog, dictionary and the cross-database read, plus a cohort and a patient read per demo source
    assert len(report["web"]["reads"]) == 3 + 2 * len(SOURCES)
    assert all(read["status_code"] == 200 for read in report["web"]["reads"])
    assert report["web"]["traceback_found"] is False
    artifact = json.loads((tmp_path / "receipts" / "acceptance.json").read_text("utf-8"))
    assert artifact == report
