from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from easyicu.resources import load_dictionary
from easyicu.webserver import dataio


class _ExportJob:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, payload: dict[str, object]) -> None:
        self.events.append(payload)


def _patch_export_api(
    monkeypatch: pytest.MonkeyPatch, loaded: list[dict[str, object]]
) -> None:
    import easyicu.api as api_module

    from easyicu.concept.export_metadata import concept_declares_event_status

    dictionary = load_dictionary(include_sofa2=True)

    @contextmanager
    def fake_keep_cache(**_: object):
        yield None

    def fake_load_concepts(concepts, **kwargs):
        loaded.append({"concepts": concepts, "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        payload: dict[str, object] = {"stay_id": ids}
        for concept in concepts:
            definition = dictionary.get(concept)
            # The export owner's own event rule, which also covers loader outputs.
            if concept_declares_event_status(concept, definition):
                payload[concept] = [index % 2 for index in range(len(ids))]
            else:
                payload[concept] = [65.0] * len(ids)
        return pd.DataFrame(payload)

    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)


def test_extraction_ui_discloses_resource_fallback() -> None:
    extraction_js = (
        Path("src/easyicu/webserver/static/js/screens-extraction.js")
        .read_text(encoding="utf-8")
    )

    assert "exportResourcePlan.mode === 'patient_batches'" in extraction_js
    assert "exportResourcePlan.advisory_zh" in extraction_js


def test_export_runner_applies_and_persists_memory_fallback_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module
    from easyicu.api.extraction import ExtractionResourcePlan

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module,
        "get_all_patient_ids",
        lambda *_, **__: ([1, 2], "stay_id"),
    )
    fallback = ExtractionResourcePlan(
        mode="patient_batches",
        reason_code="measured_profile_insufficient_memory",
        batch_size=1,
        available_memory_mb=1_700,
        required_available_memory_mb=1_824.1,
        measured_peak_rss_mb=1_658.2,
        modules=("demographics",),
        advisory="Free memory for the fastest mode.",
        advisory_zh="清理内存后可恢复最快模式。",
    )
    monkeypatch.setattr(
        api_module,
        "plan_extraction_resources",
        lambda *_args, **_kwargs: fallback,
    )

    job = _ExportJob()
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        include_feature_definitions=False,
    )

    result = runner(job)
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )
    start = next(event for event in job.events if event["type"] == "start")

    assert loaded[0]["kwargs"]["batch_size"] == 1
    assert start["resource_plan"]["reason_code"] == fallback.reason_code
    assert result["resource_plan"]["advisory_zh"] == fallback.advisory_zh
    assert manifest["resource_plan"] == result["resource_plan"]


def test_export_runner_locks_fast_plan_without_memory_advisory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module
    from easyicu.api.extraction import ExtractionResourcePlan

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module,
        "get_all_patient_ids",
        lambda *_, **__: ([1, 2], "stay_id"),
    )
    fast = ExtractionResourcePlan(
        mode="one_shot",
        reason_code="measured_profile_fast_path",
        batch_size=2,
        available_memory_mb=2_048,
        required_available_memory_mb=1_824.1,
        measured_peak_rss_mb=1_658.2,
        modules=("demographics",),
    )
    monkeypatch.setattr(
        api_module,
        "plan_extraction_resources",
        lambda *_args, **_kwargs: fast,
    )

    job = _ExportJob()
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        include_feature_definitions=False,
    )

    result = runner(job)

    assert loaded[0]["kwargs"]["batch_size"] == 2
    assert result["resource_plan"]["mode"] == "one_shot"
    assert result["resource_plan"]["advisory"] is None
    assert not any(event.get("phase") == "resource" for event in job.events)


def test_minute_module_listing_is_database_scoped_and_opt_in(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from easyicu.webserver.routes.extraction import router

    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        def options(database):
            reply = client.get("/api/extraction/module-options", params={"database": database})
            assert reply.status_code == 200
            return reply.json()["minute_resolution"]

        hirid = options("hirid")
        sic = options("sic")
        unsupported = options("unknown_database")
        monkeypatch.setenv("EASYICU_HOSTED", "1")
        hosted = options("hirid")

    assert [row["id"] for row in hirid["modules"]] == [
        "subhour_signals", "vent_native", "haemo_native",
    ]
    assert all(row["default_selected"] is False and row["interval_minutes"] == 1
               for row in hirid["modules"])
    assert "peep_total" in hirid["modules"][1]["signals"]
    assert "peep_total" not in sic["modules"][1]["signals"]
    assert all(row["unavailable_reason"] == "no_whitelisted_signals" for row in unsupported["modules"])
    assert hosted["unavailable_reason"] == "subhour_unavailable_when_hosted"


def test_minute_route_refuses_an_unmapped_or_hosted_request(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from easyicu.webserver.routes.jobs import submission_router

    app = FastAPI()
    app.include_router(submission_router)
    request = {
        "path": str(tmp_path), "database": "sic", "modules": ["vent_native"],
        "concepts": {"vent_native": ["peep_total"]},
    }
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    with TestClient(app) as client:
        refusal = client.post("/api/jobs/extract", json=request)
        assert refusal.status_code == 400
        assert refusal.json()["detail"]["error"] == "subhour_signal_not_whitelisted"
        monkeypatch.setenv("EASYICU_HOSTED", "1")
        monkeypatch.setenv("EASYICU_DATA_ROOTS", str(tmp_path))
        refusal = client.post("/api/jobs/extract", json={**request, "concepts": None})
        assert refusal.status_code == 403
        assert refusal.json()["detail"]["error"] == "subhour_unavailable_when_hosted"


def test_default_web_extraction_dispatches_the_same_hourly_modules(tmp_path, monkeypatch):
    import easyicu.api as api
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.concept.minute_resolution import MINUTE_RESOLUTION_KWARG

    loaded = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(api, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id"))

    class FirstModuleLoaded(Exception):
        pass

    load = api.load_concepts

    def record_first_module(concepts, **kwargs):
        frame = load(concepts, **kwargs)
        if concepts != ["los_icu"]:
            raise FirstModuleLoaded
        return frame

    monkeypatch.setattr(api, "load_concepts", record_first_module)
    job = _ExportJob()
    runner = dataio.make_export_runner(
        data_path=str(tmp_path), database="miiv", out_dir=str(tmp_path / "out"),
    )
    with pytest.raises(FirstModuleLoaded):
        runner(job)
    start = next(event for event in job.events if event["type"] == "start")
    assert start["modules"] == list(CONCEPT_GROUPS_INTERNAL)
    assert len(start["modules"]) == 20
    assert loaded[0]["concepts"] == ["los_icu"]
    assert len(loaded) == 2
    assert all(MINUTE_RESOLUTION_KWARG not in call["kwargs"] and "interval" not in call["kwargs"] for call in loaded)


@pytest.mark.parametrize("export_format", ["parquet", "csv"])
def test_minute_web_job_publishes_the_resolution_in_its_result(tmp_path, export_format):
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "hirid"
    out = tmp_path / "out"
    generate_source(source, "hirid", stays=3)
    observations = source / "observations" / "1.parquet"
    table = pd.read_parquet(observations)
    anchor = table.loc[table.variableid.eq(200)].sort_values("datetime").iloc[[0]]
    extra = anchor.assign(datetime=anchor.datetime + pd.Timedelta(minutes=2), value=88.0)
    pd.concat([table, extra], ignore_index=True).to_parquet(observations, index=False)
    runner = dataio.make_export_runner(
        data_path=str(source), database="hirid", modules=["subhour_signals"],
        concepts={"subhour_signals": ["hr"]}, export_format=export_format, out_dir=str(out),
    )
    result = runner(_ExportJob())
    manifest = json.loads((out / "_manifest.json").read_text())
    assert result["files"] == manifest["files"]
    receipt = result["files"][0]["time_resolution"]
    assert receipt["interval_minutes"] == 1
    assert list(receipt["signals"]) == ["hr"]
    assert result["files"][0]["concept_ids"] == ["hr"]
    assert manifest["feature_definitions"]["included"]
    path = out / result["files"][0]["file"]
    frame = pd.read_parquet(path) if export_format == "parquet" else pd.read_csv(path)
    assert (frame.loc[frame.hr.eq(88.0), "charttime"] * 60).round(5).tolist() == [2.0]


def test_minute_ui_wires_chooser_selection_and_manifest_result():
    static = Path("src/easyicu/webserver/static")
    page = (static / "js/screens-extraction.js").read_text()
    owner = (static / "js/screens-extraction-minute.js").read_text()
    html = (static / "index.html").read_text()
    assert html.index("js/screens-extraction-minute.js") < html.index("js/screens-extraction.js")
    assert "data-ex-minute-module" in owner
    assert "time_resolution" in owner
    assert "minuteModules.render(" in page
    assert "minuteModules.bind(" in page
    assert "minuteModules.resultDetail(" in page
