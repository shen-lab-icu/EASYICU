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
            if definition is not None and definition.class_name == "lgl_cncpt":
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
