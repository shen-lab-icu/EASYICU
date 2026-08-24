#!/usr/bin/env python3
"""Run the executable JS contract tests in tests/js/.

Each harness takes the owner file(s) it exercises as positional arguments
(``process.argv[2]`` onward) and stubs ``window`` before loading them. That
contract was never written down anywhere: CI wires up exactly one of them, and
the other twenty could only be run by whoever remembered the argument list. A
test that cannot be invoked is not a test, so the mapping lives here.

    python tools/run_js_contracts.py              # all of them
    python tools/run_js_contracts.py patient      # substring filter

Order matters where a harness requires its files in stages (see
study_context_lifecycle), so the lists below are positional, not sets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests" / "js"
JS = ROOT / "src" / "easyicu" / "webserver" / "static" / "js"

CONTRACTS: dict[str, list[str]] = {
    "agent_render_security.test.js": ["screens-agent-render.js"],
    "composer_keyboard.test.js": ["composer-keyboard.js"],
    "crossdb_job_continuity.test.js": ["screens-viz-crossdb-job-continuity.js"],
    "crossdb_progress_owner.test.js": ["screens-viz-crossdb-progress.js"],
    "crossdb_raw_scope.test.js": ["screens-viz-crossdb-raw.js"],
    "crossdb_results_owner.test.js": ["screens-viz-crossdb-results.js"],
    "crossdb_setup_owner.test.js": ["screens-viz-crossdb-setup.js"],
    "crossdb_source_choice.test.js": ["screens-viz-crossdb-source.js"],
    "extraction_embedded_scroll.test.js": ["screens-extraction-embedded.js"],
    "extraction_icd_source_binding.test.js": ["screens-icd.js"],
    "extraction_job_continuity.test.js": ["screens-extraction-job-continuity.js"],
    "guided_gate_state.test.js": ["screens-guided.js"],
    # Loads both dedicated Copilot data-view owners itself; takes no arguments.
    "guided_pi_data_workbench.test.js": [],
    "viz_embedded_workbench.test.js": [],
    "guided_idea_flow.test.js": ["screens-guided-idea.js"],
    "guided_project_handoff.test.js": [
        "product-labels.js",
        "screens-guided-projects.js",
    ],
    # Reads the whole js/ directory itself; takes no arguments.
    "job_continuity_404.test.js": [],
    "patient_browse_owners.test.js": [
        "screens-viz-patient-navigation.js",
        "screens-viz-patient-tables.js",
    ],
    "patient_demo_fidelity.test.js": [
        "data-catalog.js",
        "screens-viz-demo.js",
        "screens-viz-demo-drilldown.js",
    ],
    "patient_demo_sources_owner.test.js": ["screens-viz-patient-demo-sources.js"],
    "patient_echarts_owner.test.js": ["screens-viz-patient-charts.js"],
    "patient_feature_loader_owner.test.js": ["screens-viz-patient-feature-loader.js"],
    "patient_scope_truth.test.js": ["screens-viz-patient-overview.js"],
    "patient_series_owner.test.js": [
        "data-catalog.js",
        "screens-viz-demo.js",
        "screens-viz-patient-features.js",
        "screens-viz-patient-series.js",
    ],
    "product_labels.test.js": ["product-labels.js"],
    "project_title_projection.test.js": [
        "product-labels.js",
        "screens-agent-study-context.js",
    ],
    "review_echarts_owners.test.js": [
        "screens-viz-echarts.js",
        "screens-viz-crossdb-charts.js",
        "screens-viz-cohort-charts.js",
    ],
    "run_context_race.test.js": ["screens-agent-study-context.js"],
    "study_context_lifecycle.test.js": [
        "study-context.js",
        "screens-viz-study-context.js",
        "screens-guided-study-context.js",
        "screens-agent-study-context.js",
        "product-labels.js",
    ],
}


def main(argv: list[str]) -> int:
    needle = argv[1] if len(argv) > 1 else ""

    on_disk = {path.name for path in TESTS.glob("*.test.js")}
    missing = on_disk - set(CONTRACTS)
    assert not missing, f"contract tests with no argument list here: {sorted(missing)}"
    stale = set(CONTRACTS) - on_disk
    assert not stale, f"argument lists for tests that no longer exist: {sorted(stale)}"

    failed: list[str] = []
    for name in sorted(CONTRACTS):
        if needle and needle not in name:
            continue
        owners = [JS / owner for owner in CONTRACTS[name]]
        for owner in owners:
            assert owner.is_file(), f"{name}: {owner.name} does not exist"
        result = subprocess.run(
            ["node", str(TESTS / name), *[str(owner) for owner in owners]],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode == 0:
            print(f"  ok   {name}")
            continue
        failed.append(name)
        print(f"  FAIL {name}")
        print("       " + (result.stderr or result.stdout).strip().replace("\n", "\n       "))

    print(f"\n{len(failed)} failing of {len(CONTRACTS) if not needle else '(filtered)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
