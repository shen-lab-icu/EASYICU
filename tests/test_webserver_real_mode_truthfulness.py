"""Truthfulness guards for native WebApp real-data workflows."""

from pathlib import Path


STATIC_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
    / "js"
)


def _js(name: str) -> str:
    return (STATIC_JS / name).read_text(encoding="utf-8")


def test_real_extraction_never_falls_through_to_demo_completion() -> None:
    source = _js("screens-extraction.js")

    assert "} else if (!real) {" in source
    assert "Demo mode intentionally uses a seeded, in-browser completion." in source
    assert "Real extraction could not start." in source
    assert "Demo / offline fallback" not in source


def test_real_conversion_and_scan_fail_closed_without_local_runtime() -> None:
    source = _js("screens-extraction.js")

    assert "Real conversion could not start because the local job API" in source
    assert "convResult = { converted: CONV_STEPS.length" not in source
    assert "setInterval(() => {\n        convDone++" not in source
    assert "exScanError = 'scan_api_unavailable'" in source
    assert "this screen will not guess a real data layout" in source
    assert "setTimeout(() => { exReal = 'scanresult'" not in source
