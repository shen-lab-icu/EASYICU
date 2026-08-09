#!/usr/bin/env python3
"""Browser regression for Pi workspace preview origin isolation.

The check writes a hostile HTML artifact into an isolated temporary Pi
workspace, starts the native FastAPI app with the same temporary HOME, and
proves that EasyICU-origin localStorage is unavailable inside the nested model
document while Host provenance remains visible in product and direct views.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qa_native_fastapi_patient_drilldown import (  # noqa: E402
    port_free,
    post_json,
    start_server,
    wait_ready,
)

from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace  # noqa: E402

CHROME_EXECUTABLE = Path(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)

HOSTILE_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Hostile preview regression</title>
<output id="storage-result">PENDING</output>
<script>
  const output = document.querySelector('#storage-result');
  try {
    output.textContent = 'LEAK:' + localStorage.getItem('easyicu-preview-secret');
  } catch (error) {
    output.textContent = 'BLOCKED:' + error.name;
  }
</script>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--out-dir", default="output/playwright/pi-preview-security")
    return parser.parse_args()


def _headers(url: str) -> dict[str, str]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return {key.lower(): value for key, value in response.headers.items()}


def run() -> dict[str, object]:
    args = parse_args()
    if not port_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_id = "pi-preview-security"
    relative_file = "security/hostile.html"
    base_url = f"http://127.0.0.1:{args.port}/"
    preview_url = (
        f"{base_url}api/copilot/pi/projects/{project_id}/workspace/preview"
        f"?file=security%2Fhostile.html"
    )

    with tempfile.TemporaryDirectory(prefix="easyicu-pi-preview-qa-") as temp_home:
        home = Path(temp_home)
        workspace = ProjectWorkspace(
            home / ".easyicu" / "pi-agent" / "workspace"
        )
        workspace.write_file(project_id, relative_file, HOSTILE_HTML)
        server = start_server(args.port, home)
        try:
            wait_ready(base_url, server)
            post_json(
                base_url,
                "api/copilot/pi/projects/initialize",
                {
                    "project_id": project_id,
                    "title": "Preview security regression",
                    "confirm_initialization": True,
                },
            )
            response_headers = _headers(preview_url)
            with sync_playwright() as playwright:
                launch_options = (
                    {"executable_path": str(CHROME_EXECUTABLE)}
                    if CHROME_EXECUTABLE.is_file()
                    else {}
                )
                browser = playwright.chromium.launch(
                    headless=True,
                    **launch_options,
                )
                context = browser.new_context()
                page = context.new_page()
                page.goto(base_url, wait_until="domcontentloaded")
                page.evaluate(
                    "localStorage.setItem('easyicu-preview-secret', 'must-not-leak')"
                )

                page.evaluate(
                    """url => {
                      const iframe = document.createElement('iframe');
                      iframe.id = 'hostile-preview-frame';
                      iframe.sandbox = 'allow-scripts';
                      iframe.referrerPolicy = 'no-referrer';
                      iframe.src = url;
                      document.body.appendChild(iframe);
                    }""",
                    preview_url,
                )
                product_wrapper = page.frame_locator("#hostile-preview-frame")
                product_banner = product_wrapper.locator(
                    "[data-easyicu-workspace-provenance]"
                ).inner_text(timeout=10_000)
                product_content = product_wrapper.frame_locator(
                    "#easyicu-workspace-preview-content"
                )
                iframe_result = product_content.locator("#storage-result").inner_text(
                    timeout=10_000
                )

                direct = context.new_page()
                direct.goto(preview_url, wait_until="domcontentloaded")
                direct_banner = direct.locator(
                    "[data-easyicu-workspace-provenance]"
                ).inner_text(timeout=10_000)
                direct_result = direct.frame_locator(
                    "#easyicu-workspace-preview-content"
                ).locator("#storage-result").inner_text(
                    timeout=10_000
                )
                direct_layout = direct.evaluate(
                    """() => {
                      const root = document.documentElement;
                      const banner = document.querySelector('[data-easyicu-workspace-provenance]');
                      const frame = document.querySelector('#easyicu-workspace-preview-content');
                      const bannerRect = banner.getBoundingClientRect();
                      const frameRect = frame.getBoundingClientRect();
                      return {
                        viewport_width: window.innerWidth,
                        scroll_width: root.scrollWidth,
                        banner_height: bannerRect.height,
                        frame_width: frameRect.width,
                        frame_height: frameRect.height,
                      };
                    }"""
                )
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except TimeoutError:
                server.kill()
                server.wait(timeout=5)

    policy = response_headers.get("content-security-policy", "")
    result = {
        "base_url": base_url,
        "preview_url": preview_url,
        "product_provenance": product_banner,
        "direct_provenance": direct_banner,
        "iframe_storage_result": iframe_result,
        "direct_storage_result": direct_result,
        "direct_layout": direct_layout,
        "content_security_policy": policy,
        "referrer_policy": response_headers.get("referrer-policy"),
        "passed": (
            iframe_result.startswith("BLOCKED:")
            and direct_result.startswith("BLOCKED:")
            and "must-not-leak" not in iframe_result
            and "must-not-leak" not in direct_result
            and "Workspace artifact · Unvalidated" in product_banner
            and "Workspace artifact · Unvalidated" in direct_banner
            and direct_layout["scroll_width"] <= direct_layout["viewport_width"]
            and direct_layout["banner_height"] > 0
            and direct_layout["frame_width"] > 0
            and direct_layout["frame_height"] > 0
            and policy.startswith("sandbox allow-scripts;")
            and "frame-src 'self'" in policy
            and "frame-ancestors 'self'" in policy
        ),
    }
    report = out_dir / "report.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise AssertionError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
