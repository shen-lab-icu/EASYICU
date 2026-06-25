#!/usr/bin/env python3
"""Browser QA for the native FastAPI extraction connect -> job flow.

This is intentionally a local QA tool, not a CI test. It drives the real
browser UI against a real local data folder and verifies the bounded job
progress/cancel surface without forcing a long or destructive extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from playwright.sync_api import Page, sync_playwright


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO / "output" / "playwright"
DEFAULT_OBSERVATION_WINDOW_HOURS = 24 * 30


@dataclass
class ServerHandle:
    base_url: str
    process: subprocess.Popen[str] | None
    temp_home: tempfile.TemporaryDirectory[str] | None


def _wait_for_health(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - surfaced below with context.
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy at {base_url}: {last_error}")


def _fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_job_terminal(base_url: str, job_id: str, timeout: float = 60.0) -> dict[str, object]:
    deadline = time.time() + timeout
    latest: dict[str, object] = {}
    while time.time() < deadline:
        latest = _fetch_json(f"{base_url}/api/jobs/{job_id}")
        if latest.get("status") in {"done", "failed", "cancelled"}:
            return latest
        time.sleep(0.5)
    return latest


@contextmanager
def server_context(base_url: str | None, port: int) -> Iterator[ServerHandle]:
    if base_url:
        _wait_for_health(base_url, timeout=5)
        yield ServerHandle(base_url=base_url.rstrip("/"), process=None, temp_home=None)
        return

    temp_home = tempfile.TemporaryDirectory(prefix="easyicu-extraction-qa-home-")
    env = os.environ.copy()
    env.update(
        {
            "HOME": temp_home.name,
            "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
            "PYTHONPATH": str(REPO / "src"),
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "easyicu.webserver.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handle = ServerHandle(
        base_url=f"http://127.0.0.1:{port}",
        process=process,
        temp_home=temp_home,
    )
    try:
        _wait_for_health(handle.base_url)
        yield handle
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        temp_home.cleanup()


def _visible_text(page: Page) -> str:
    return page.locator("body").inner_text(timeout=5000)


def _body_has(page: Page, needle: str) -> bool:
    try:
        return needle in _visible_text(page)
    except Exception:  # noqa: BLE001 - QA observation only.
        return False


def _click_if_visible(page: Page, selector: str, timeout: int = 1000) -> bool:
    locator = page.locator(selector).first
    try:
        locator.wait_for(state="visible", timeout=timeout)
        locator.click()
        return True
    except Exception:  # noqa: BLE001 - visibility is part of the QA result.
        return False


def _wait_for_any_body_text(page: Page, needles: list[str], timeout_ms: int) -> str | None:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        text = _visible_text(page)
        for needle in needles:
            if needle in text:
                return needle
        time.sleep(0.25)
    return None


def _select_module_indices(page: Page, module_indices: list[int]) -> None:
    clear = page.locator("[data-ex-clearmods]").first
    clear.wait_for(state="visible", timeout=10000)
    clear.click()
    for index in module_indices:
        module = page.locator(f'[data-ex-mod="{index}"]').first
        module.wait_for(state="visible", timeout=5000)
        module.click()


def _count_selected_modules(page: Page) -> int:
    return page.locator(".modcard.on").count()


def _wait_selected_modules(page: Page, count: int, timeout: int = 5000) -> None:
    page.wait_for_function(
        "(expected) => document.querySelectorAll('.modcard.on').length === expected",
        arg=count,
        timeout=timeout,
    )


def _switch_hash(page: Page, route: str) -> None:
    page.evaluate("(r) => { window.location.hash = '#' + r; }", route)
    page.wait_for_function("(r) => window.location.hash === '#' + r", arg=route, timeout=5000)
    page.wait_for_timeout(250)


def _run_browser_flow(
    *,
    base_url: str,
    data_path: Path,
    out_dir: Path,
    module_indices: list[int],
    run_mode: str,
    click_cancel: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto(f"{base_url}/?_v=stage46#extraction", wait_until="domcontentloaded")
        page.evaluate("window.setDataMode && window.setDataMode('real', { force: true })")
        page.wait_for_timeout(500)

        path_input = page.locator("#exPathInput").first
        path_input.wait_for(state="visible", timeout=10000)
        path_input.fill(str(data_path))
        page.locator("[data-ex-analyze]").first.click()

        recognized = _wait_for_any_body_text(
            page,
            ["Folder recognized", "Continue with prepared data", "Use this data"],
            timeout_ms=20000,
        )
        if not recognized:
            raise AssertionError("data folder was not recognized by the extraction UI")

        page.locator("[data-ex-usedata]").first.wait_for(state="visible", timeout=10000)
        page.locator("[data-ex-usedata]").first.click()

        page.locator("[data-ex-run='custom']").first.wait_for(state="visible", timeout=15000)
        if not page.locator("[data-ex-clearmods]").first.is_visible():
            _click_if_visible(page, "[data-ex-custom]", timeout=1000)
            page.locator("[data-ex-clearmods]").first.wait_for(state="visible", timeout=10000)
        page_text_before_run = _visible_text(page)

        page.locator(".sumcard").first.wait_for(state="visible", timeout=5000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(250)
        sticky_probe = page.locator(".sumcard").first.evaluate(
            """
            (card) => {
              const rect = card.getBoundingClientRect();
              const btn = card.querySelector('[data-ex-run="custom"]');
              const btnRect = btn ? btn.getBoundingClientRect() : null;
              return {
                top: Math.round(rect.top),
                bottom: Math.round(rect.bottom),
                viewportHeight: window.innerHeight,
                buttonVisible: !!btnRect && btnRect.top >= 0 && btnRect.bottom <= window.innerHeight,
                cardVisible: rect.top >= 48 && rect.top <= 120 && rect.bottom <= window.innerHeight,
              };
            }
            """
        )
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(100)

        preflight: dict[str, object] = {
            "default_format_parquet": page.locator("[data-ex-fmt] button.active[data-val='parquet']").count() == 1,
            "default_selected_modules": _count_selected_modules(page),
            "summary_sticky_after_scroll": sticky_probe,
        }
        page.locator("[data-ex-clearmods]").first.click()
        _wait_selected_modules(page, 0)
        preflight["clear_all_zero_modules"] = _count_selected_modules(page) == 0
        preflight["extract_disabled_when_empty"] = page.locator("[data-ex-run='custom']").first.is_disabled()
        page.locator("[data-ex-selectall]").first.click()
        _wait_selected_modules(page, 19)
        preflight["select_all_modules"] = _count_selected_modules(page)
        page.locator("[data-ex-core]").first.click()
        _wait_selected_modules(page, 6)
        preflight["core_modules"] = _count_selected_modules(page)
        page.locator("[data-ex-fmt] button[data-val='csv']").first.click()
        page.wait_for_function(
            "() => !!document.querySelector('[data-ex-fmt] button.active[data-val=\"csv\"]')",
            timeout=5000,
        )
        preflight["csv_format_selected"] = page.locator("[data-ex-fmt] button.active[data-val='csv']").count() == 1
        page.locator("[data-ex-fmt] button[data-val='parquet']").first.click()
        page.wait_for_function(
            "() => !!document.querySelector('[data-ex-fmt] button.active[data-val=\"parquet\"]')",
            timeout=5000,
        )
        preflight["parquet_format_restored"] = (
            page.locator("[data-ex-fmt] button.active[data-val='parquet']").count() == 1
        )

        if run_mode == "custom":
            _select_module_indices(page, module_indices)

        run_selector = "[data-ex-run='recommended']" if run_mode == "recommended" else "[data-ex-run='custom']"
        run_button = page.locator(run_selector).first
        run_button.wait_for(state="visible", timeout=5000)
        cancel_button_visible = False
        cancel_clicked = False
        cancel_copy_seen = False

        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.rstrip("/").endswith("/api/jobs/extract"),
            timeout=30000,
        ) as extract_response_info:
            run_button.click()
        extract_response = extract_response_info.value
        extract_payload = extract_response.json()
        try:
            extract_request_payload = json.loads(extract_response.request.post_data or "{}")
        except json.JSONDecodeError:
            extract_request_payload = {}
        job_id = str(extract_payload.get("job_id") or "")

        if click_cancel:
            try:
                cancel_button = page.locator("[data-ex-cancel]").first
                cancel_button.wait_for(state="visible", timeout=3000)
                cancel_button_visible = True
                page.wait_for_function(
                    """
                    () => {
                      const btn = document.querySelector('[data-ex-cancel]');
                      return !btn || !btn.disabled || document.body.innerText.includes('Extraction complete');
                    }
                    """,
                    timeout=5000,
                )
                if cancel_button.count() and cancel_button.is_enabled():
                    cancel_button.click()
                    cancel_clicked = True
            except Exception:  # noqa: BLE001 - job may have completed before the control appears.
                cancel_button_visible = page.locator("[data-ex-cancel]").first.count() > 0

        running_marker = _wait_for_any_body_text(
            page,
            [
                "Cancel requested",
                "Preparing extraction",
                "Resolving cohort",
                "Reading module",
                "Writing export",
                "Extraction complete",
                "Extraction cancelled",
            ],
            timeout_ms=20000,
        )
        if not running_marker:
            raise AssertionError("extraction job did not enter a visible progress or completion state")

        if click_cancel:
            if cancel_clicked:
                cancel_copy_seen = bool(
                    _wait_for_any_body_text(
                        page,
                        ["Cancel requested", "Extraction cancelled", "Extraction complete"],
                        timeout_ms=20000,
                    )
                )
        else:
            cancel_button_visible = page.locator("[data-ex-cancel]").first.count() > 0

        final_text = _visible_text(page)
        final_job_snapshot = _wait_for_job_terminal(base_url, job_id, timeout=60) if job_id else {}
        final_job_result = final_job_snapshot.get("result") if isinstance(final_job_snapshot.get("result"), dict) else {}
        export_out_dir = Path(str(final_job_result.get("out_dir") or "")) if final_job_result.get("out_dir") else None
        export_readme_ok = False
        export_manifest_ok = False
        export_dir_tagged = False
        if export_out_dir and export_out_dir.exists():
            export_dir_tagged = bool(
                re.match(r"^easyicu_export_\d{8}_\d{6}_[a-z0-9._-]+_[a-z0-9._-]+", export_out_dir.name)
            )
            readme = export_out_dir / "README.md"
            manifest = export_out_dir / "_manifest.json"
            readme_text = readme.read_text(encoding="utf-8") if readme.exists() else ""
            export_readme_ok = (
                readme.exists()
                and "EasyICU Export" in readme_text
                and "Observation window: `720 hours`" in readme_text
            )
            export_manifest_ok = manifest.exists()
        nav_patient_ok = False
        nav_agent_ok = False
        reset_ok = False
        if not click_cancel and "Extraction complete" in final_text:
            try:
                page.locator(".state-hero [data-nav='patient']").first.click(timeout=5000)
                page.wait_for_function("() => window.location.hash === '#patient'", timeout=5000)
                nav_patient_ok = True
                _switch_hash(page, "extraction")
                page.locator(".state-hero [data-nav='agent']").first.click(timeout=5000)
                page.wait_for_function("() => window.location.hash === '#agent'", timeout=5000)
                nav_agent_ok = True
                _switch_hash(page, "extraction")
                page.locator(".state-hero [data-ex-reset]").first.click(timeout=5000)
                page.wait_for_function("() => document.body.innerText.includes('Recommended extraction')", timeout=5000)
                reset_ok = "Extraction complete" not in _visible_text(page)
            except Exception:  # noqa: BLE001 - captured as explicit QA booleans.
                pass
        page.screenshot(path=str(out_dir / "extraction_job_flow.png"), full_page=True)
        overflow_x = page.evaluate(
            "() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth)"
        )
        result = {
            "base_url": base_url,
            "data_path": str(data_path),
            "run_mode": run_mode,
            "module_indices": module_indices,
            "preflight": preflight,
            "placeholder_path_absent": "~/easyicu/exports/demo" not in page_text_before_run,
            "extract_request": extract_request_payload,
            "job_id": job_id,
            "recognized_marker": recognized,
            "running_marker": running_marker,
            "cancel_button_visible": cancel_button_visible,
            "cancel_clicked": cancel_clicked,
            "cancel_copy_seen": cancel_copy_seen,
            "extraction_complete": "Extraction complete" in final_text,
            "extraction_cancelled": "Extraction cancelled" in final_text,
            "final_job_status": final_job_snapshot.get("status"),
            "final_job_result": final_job_result,
            "final_job_cancel_requested": final_job_snapshot.get("cancel_requested"),
            "final_job_cancel_reason": final_job_snapshot.get("cancel_reason"),
            "export_dir_tagged": export_dir_tagged,
            "export_manifest_ok": export_manifest_ok,
            "export_readme_ok": export_readme_ok,
            "nav_patient_ok": nav_patient_ok,
            "nav_agent_ok": nav_agent_ok,
            "reset_ok": reset_ok,
            "overflowX": overflow_x,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "screenshot": str(out_dir / "extraction_job_flow.png"),
        }
        browser.close()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Existing FastAPI server, for example http://127.0.0.1:8782")
    parser.add_argument("--port", type=int, default=8786, help="Port when starting an isolated server")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(os.environ["EASYICU_QA_DATA_PATH"]) if os.environ.get("EASYICU_QA_DATA_PATH") else None,
        help="Explicit local ICU data folder. Or set EASYICU_QA_DATA_PATH.",
    )
    parser.add_argument(
        "--module-index",
        action="append",
        type=int,
        dest="module_indices",
        help="Feature module index to select after Clear all. Repeatable. Defaults to demographics (0).",
    )
    parser.add_argument("--run-mode", choices=["custom", "recommended"], default="custom")
    parser.add_argument("--no-cancel", action="store_true", help="Do not click the browser cancel control")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    if args.data_path is None:
        parser.error("--data-path is required unless EASYICU_QA_DATA_PATH is set")

    if not args.data_path.exists():
        raise SystemExit(f"data path does not exist: {args.data_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"native_fastapi_extraction_job_flow_{stamp}"
    module_indices = args.module_indices or [0]

    with server_context(args.base_url, args.port) as server:
        result = _run_browser_flow(
            base_url=server.base_url,
            data_path=args.data_path,
            out_dir=out_dir,
            module_indices=module_indices,
            run_mode=args.run_mode,
            click_cancel=not args.no_cancel,
        )

    failures: list[str] = []
    if result["console_errors"]:
        failures.append("console_errors")
    if result["page_errors"]:
        failures.append("page_errors")
    if result["overflowX"] != 0:
        failures.append("horizontal_overflow")
    if not result["recognized_marker"]:
        failures.append("folder_not_recognized")
    if not result["running_marker"]:
        failures.append("job_not_started")
    if not result["job_id"]:
        failures.append("missing_job_id")
    if result.get("final_job_status") not in {"done", "failed", "cancelled"}:
        failures.append("job_not_terminal")
    preflight = result.get("preflight") if isinstance(result.get("preflight"), dict) else {}
    if preflight.get("default_format_parquet") is not True:
        failures.append("default_format_not_parquet")
    if preflight.get("default_selected_modules") != 19:
        failures.append("default_modules_not_all_selected")
    if preflight.get("clear_all_zero_modules") is not True:
        failures.append("clear_all_failed")
    if preflight.get("extract_disabled_when_empty") is not True:
        failures.append("empty_extract_not_disabled")
    if preflight.get("select_all_modules") != 19:
        failures.append("select_all_failed")
    if preflight.get("core_modules") != 6:
        failures.append("core_reset_failed")
    if preflight.get("parquet_format_restored") is not True:
        failures.append("parquet_not_restored")
    if result.get("placeholder_path_absent") is not True:
        failures.append("hardcoded_demo_export_path_visible")
    sticky_probe = preflight.get("summary_sticky_after_scroll")
    if not isinstance(sticky_probe, dict) or sticky_probe.get("cardVisible") is not True:
        failures.append("summary_card_not_sticky_after_scroll")
    if not isinstance(sticky_probe, dict) or sticky_probe.get("buttonVisible") is not True:
        failures.append("summary_extract_button_not_visible_after_scroll")
    request = result.get("extract_request") if isinstance(result.get("extract_request"), dict) else {}
    if request.get("format") != "parquet":
        failures.append("extract_request_not_parquet")
    cohort = request.get("cohort") if isinstance(request.get("cohort"), dict) else {}
    if cohort.get("observation_window_hours") != DEFAULT_OBSERVATION_WINDOW_HOURS:
        failures.append("default_observation_window_not_full_available")
    if result.get("run_mode") == "recommended":
        if request.get("modules") != [
            "demographics",
            "vitals",
            "chemistry",
            "sofa2_score",
            "sepsis3_sofa2",
            "outcome",
        ]:
            failures.append("recommended_modules_not_core_six")
        if cohort.get("preset") != "adult_first":
            failures.append("recommended_cohort_not_adult_first")
    if result.get("run_mode") == "custom" and not result.get("cancel_clicked"):
        if result.get("export_dir_tagged") is not True:
            failures.append("export_dir_not_timestamp_tagged")
        if result.get("export_manifest_ok") is not True:
            failures.append("export_manifest_missing")
        if result.get("export_readme_ok") is not True:
            failures.append("export_readme_missing_or_incomplete")
        if result.get("nav_patient_ok") is not True:
            failures.append("done_patient_nav_failed")
        if result.get("nav_agent_ok") is not True:
            failures.append("done_agent_nav_failed")
        if result.get("reset_ok") is not True:
            failures.append("done_reset_failed")

    result["failures"] = failures
    report_path = out_dir / "extraction_job_flow_qa.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
