#!/usr/bin/env python3
"""Inventory legacy Streamlit CSS before sliced cleanup.

Run from ``EASYICU/``:

    python tools/inventory_legacy_streamlit_css.py

The script is intentionally read-only for ``src/easyicu/webapp``. It writes a
bounded JSON/Markdown report plus copied CSS snapshot under ``output/`` so the
legacy fallback can be compared before any cleanup slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


WEBAPP_DIR = Path("src/easyicu/webapp")
DEFAULT_OUT_ROOT = Path("output/stage21_legacy_css_inventory")
SHELL_STYLES_PATH = WEBAPP_DIR / "shell_styles.py"
OWNERSHIP_TEST_PATH = Path("tests/test_app_rendering.py")
RESEARCH_AGENT_HELPER_TEST_PATH = Path("tests/test_research_agent_web_helpers.py")
LEGACY_STREAMLIT_CSS_ENV = "EASYICU_ENABLE_LEGACY_STREAMLIT_CSS"

CSS_OWNER_BY_FILE = {
    "tokens.css": "global design tokens",
    "shell_overrides.css": "global Streamlit reset/native widgets/shared primitives",
    "alignment.css": "legacy compatibility marker",
    "shell_navigation_overrides.css": "shared shell navigation/topbar/sidebar",
    "visualization_shell_overrides.css": "shared Patient/Cohort/Cross-DB shell",
    "entry_overrides.css": "entry route",
    "extract_overrides.css": "extraction route",
    "patient_overrides.css": "patient route",
    "cohort_overrides.css": "cohort route",
    "crossdb_overrides.css": "crossdb route",
    "agent_overrides.css": "agent route",
    "settings_overrides.css": "settings route",
    "dictionary_overrides.css": "dictionary route",
    "states_overrides.css": "states route",
    "guided_overrides.css": "guided route",
    "tutorial_overrides.css": "tutorial/help route",
}

EXPECTED_RENDER_ORDER = [
    "tokens.css",
    "shell_overrides.css",
    "alignment.css",
    "shell_navigation_overrides.css",
    "entry_overrides.css",
    "tutorial_overrides.css",
    "dictionary_overrides.css",
    "states_overrides.css",
    "settings_overrides.css",
    "extract_overrides.css",
    "visualization_shell_overrides.css",
    "patient_overrides.css",
    "cohort_overrides.css",
    "crossdb_overrides.css",
    "agent_overrides.css",
    "guided_overrides.css",
]
DEFAULT_LOADED_CSS = ["tokens.css", "shell_overrides.css"]
LEGACY_SPLIT_CSS = [
    name for name in EXPECTED_RENDER_ORDER if name not in set(DEFAULT_LOADED_CSS)
]

ROUTE_MARKERS = {
    "entry": [r"eu-entry", r"\bhome-", r"\bentry-"],
    "extract": [r"eu-step[1-4]", r"step[1-4]_", r"extraction", r"extract"],
    "patient": [r"eu-qv", r"qv-", r"patient", r"quick-viz"],
    "cohort": [r"eu-cohort", r"cohort", r"sofa", r"reclassification"],
    "crossdb": [r"crossdb", r"cross-db", r"multidb", r"eu-xdb"],
    "agent": [r"eu-agent", r"agent", r"research-agent", r"eu-summary"],
    "settings": [r"settings", r"setup-row", r"\bprefs?\b"],
    "dictionary": [r"dictionary", r"\bdict-", r"eu-dict"],
    "states": [r"workspace-states", r"\bstates\b", r"state-"],
    "guided": [r"\bgd-", r"guided", r"copilot", r"assistant"],
    "tutorial": [r"tutorial", r"get-started", r"reference"],
}

OWNER_ROUTE_BY_FILE = {
    "entry_overrides.css": "entry",
    "extract_overrides.css": "extract",
    "patient_overrides.css": "patient",
    "cohort_overrides.css": "cohort",
    "crossdb_overrides.css": "crossdb",
    "agent_overrides.css": "agent",
    "settings_overrides.css": "settings",
    "dictionary_overrides.css": "dictionary",
    "states_overrides.css": "states",
    "guided_overrides.css": "guided",
    "tutorial_overrides.css": "tutorial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Report root directory")
    parser.add_argument("--no-copy", action="store_true", help="Skip copying CSS files into the snapshot")
    parser.add_argument(
        "--check-agent-owner-guards",
        action="store_true",
        help="Run the Stage22A Agent CSS owner-guard regression and exit.",
    )
    parser.add_argument(
        "--check-guided-owner-guards",
        action="store_true",
        help="Run the Stage22D Guided/Copilot CSS owner-guard regression and exit.",
    )
    parser.add_argument(
        "--check-extract-owner-guards",
        action="store_true",
        help="Run the Stage22E Data Extraction CSS owner-guard regression and exit.",
    )
    parser.add_argument(
        "--check-patient-owner-guards",
        action="store_true",
        help="Run the Stage22E Patient Review CSS owner-guard regression and exit.",
    )
    parser.add_argument(
        "--check-cohort-owner-guards",
        action="store_true",
        help="Run the Stage22E Cohort Review CSS owner-guard regression and exit.",
    )
    parser.add_argument(
        "--check-crossdb-owner-guards",
        action="store_true",
        help="Run the Stage22E Cross-DB CSS owner-guard regression and exit.",
    )
    parser.add_argument(
        "--check-states-owner-guards",
        action="store_true",
        help="Run the Stage22E Workspace States CSS owner-guard regression and exit.",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


def git_status_entries() -> list[dict[str, str]]:
    output = run_git(["status", "--porcelain=v1", "--untracked-files=all"])
    entries = []
    for line in output.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[2:].strip()
        entries.append({"status": status, "path": path})
    return entries


def git_status(path: Path) -> str:
    output = run_git(["status", "--porcelain=v1", "--untracked-files=all", "--", str(path)])
    if not output:
        return "clean_tracked"
    code = output[:2].strip() or output[:2]
    if output.startswith("??"):
        return "untracked"
    return code.replace(" ", "_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def css_files() -> list[Path]:
    return sorted(WEBAPP_DIR.glob("*.css"), key=lambda p: p.name)


def referenced_css_from_shell_styles() -> list[str]:
    path = WEBAPP_DIR / "shell_styles.py"
    text = path.read_text(encoding="utf-8")
    found = re.findall(r'with_name\("([^"]+\.css)"\)', text)
    return [name for name in EXPECTED_RENDER_ORDER if name in set(found)] + [
        name for name in found if name not in EXPECTED_RENDER_ORDER
    ]


def _present_css_names(names: list[str]) -> list[str]:
    return [name for name in names if (WEBAPP_DIR / name).exists()]


def legacy_css_env_enabled() -> bool:
    return os.environ.get(LEGACY_STREAMLIT_CSS_ENV) == "1"


def default_loaded_css_from_shell_styles() -> list[str]:
    referenced = set(referenced_css_from_shell_styles())
    return _present_css_names([name for name in DEFAULT_LOADED_CSS if name in referenced])


def legacy_enabled_loaded_css_from_shell_styles() -> list[str]:
    referenced = set(referenced_css_from_shell_styles())
    return _present_css_names([name for name in EXPECTED_RENDER_ORDER if name in referenced])


def loaded_css_from_shell_styles() -> list[str]:
    if legacy_css_env_enabled():
        return legacy_enabled_loaded_css_from_shell_styles()
    return default_loaded_css_from_shell_styles()


def marker_hits(text: str, owner_file: str) -> list[dict[str, Any]]:
    owner_route = OWNER_ROUTE_BY_FILE.get(owner_file)
    hits: list[dict[str, Any]] = []
    lines = text.splitlines()
    for route, patterns in ROUTE_MARKERS.items():
        if route == owner_route:
            continue
        if owner_file == "visualization_shell_overrides.css" and route in {"patient", "cohort", "crossdb"}:
            continue
        if owner_file in {
            "shell_overrides.css",
            "shell_navigation_overrides.css",
            "tokens.css",
            "alignment.css",
        }:
            # Shared layers may contain route vocabulary during migration; keep
            # samples, but classify them as shared-layer review items.
            sample_kind = "shared_layer_marker"
        else:
            sample_kind = "foreign_route_marker"
        samples: list[dict[str, Any]] = []
        for line_no, line in enumerate(lines, start=1):
            if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
                samples.append({"line": line_no, "text": line.strip()[:180]})
            if len(samples) >= 5:
                break
        if samples:
            hits.append({"route": route, "kind": sample_kind, "samples": samples})
    return hits


AGENT_OWNER_GUARD_RE = re.compile(
    r"("
    r"eu-agent-page-marker|"
    r"st-key-(?:_?eu_)?(?:agent|ra|wb)|"
    r"eu-agent|"
    r"ra-|"
    r"eu-wb|"
    r"agent_workbench|"
    r"research[_-]agent|"
    r"eu_topbar_controls_research_agent|"
    r"data-eu-agent|"
    r"\bag-"
    r")",
    flags=re.IGNORECASE,
)

AGENT_OWNED_COMPONENT_RE = re.compile(
    r"("
    r"eu-summary-|"
    r"eu-state-|"
    r"eu-step-|"
    r"eu-handoff-note|"
    r"ra-history-|"
    r"ra-idea-|"
    r"ra-step-|"
    r"ra-grounding-|"
    r"ra-repro-"
    r")",
    flags=re.IGNORECASE,
)

STALE_AGENT_TAB_MARKERS = [
    "Agent main view switcher: match the polish (2) prototype's lightweight",
    '.stApp [class*="st-key-_eu_ra_view_"] button {',
]


def _agent_line_context(lines: list[str], index: int) -> str:
    start = max(0, index - 8)
    end = min(len(lines), index + 1)
    return " ".join(line.strip() for line in lines[start:end] if line.strip())


def _agent_hit_classification(route: str, line: str, context: str) -> str:
    text = f"{context} {line}".strip()
    if any(marker in line for marker in STALE_AGENT_TAB_MARKERS):
        return "confirmed_stale_selector_or_comment"
    if AGENT_OWNER_GUARD_RE.search(text) or AGENT_OWNED_COMPONENT_RE.search(text):
        return "valid_agent_owned"
    if "font-feature-settings" in line:
        return "false_positive_css_property"
    if route == "tutorial" and "reference" in line.lower():
        return "false_positive_agent_reference_copy"
    if route == "guided" and "guided_overrides.css" in line:
        return "false_positive_move_provenance_comment"
    return "unclassified_foreign_marker"


def check_agent_owner_guards() -> dict[str, Any]:
    """Validate Stage22A Agent CSS ownership without touching dirty pytest files."""
    path = WEBAPP_DIR / "agent_overrides.css"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        for route, patterns in ROUTE_MARKERS.items():
            if route == "agent":
                continue
            matched = next(
                (
                    pattern
                    for pattern in patterns
                    if re.search(pattern, line, flags=re.IGNORECASE)
                ),
                None,
            )
            if matched is None:
                continue
            context = _agent_line_context(lines, index)
            classification = _agent_hit_classification(route, line, context)
            hits.append(
                {
                    "line": index + 1,
                    "route": route,
                    "pattern": matched,
                    "classification": classification,
                    "text": line.strip()[:180],
                }
            )
            break
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit["classification"]] = counts.get(hit["classification"], 0) + 1
    issues = [
        hit
        for hit in hits
        if hit["classification"]
        in {"confirmed_stale_selector_or_comment", "unclassified_foreign_marker"}
    ]
    if ".eu-agent-page-marker" not in text:
        issues.insert(
            0,
            {
                "line": None,
                "route": "agent",
                "pattern": ".eu-agent-page-marker",
                "classification": "missing_agent_page_guard",
                "text": "Agent CSS must retain the page marker owner guard.",
            },
        )
    return {
        "path": str(path),
        "line_count": len(lines),
        "has_agent_page_marker": ".eu-agent-page-marker" in text,
        "valid_agent_owned": counts.get("valid_agent_owned", 0),
        "false_positive_marker": counts.get("false_positive_css_property", 0)
        + counts.get("false_positive_agent_reference_copy", 0)
        + counts.get("false_positive_move_provenance_comment", 0),
        "confirmed_stale_selector_or_comment": counts.get(
            "confirmed_stale_selector_or_comment", 0
        ),
        "unclassified_foreign_marker": counts.get("unclassified_foreign_marker", 0),
        "classification_counts": counts,
        "issues": issues[:25],
    }


GUIDED_OWNER_GUARD_RE = re.compile(
    r"("
    r"eu-guided-fullscreen-marker|"
    r"eu-copilot-page-marker|"
    r"st-key-ai_assistant_page_panel|"
    r"st-key-inline_ai_assistant_panel|"
    r"st-key-(?:_?llm_ai_page_workspace|eu_copilot|_copilot)|"
    r"eu-copilot|"
    r"inline-ai|"
    r"floating-ai|"
    r"guided[_-]|"
    r"study_depth_|"
    r"route_fallback"
    r")",
    flags=re.IGNORECASE,
)

GUIDED_COMPONENT_RE = re.compile(
    r"("
    r"eu-study-step|"
    r"eu-copilot-(?:conversation|launch|active-study|agent-contract|state|stage|step|rail|evidence)|"
    r"copilot|"
    r"assistant|"
    r"workflow|"
    r"study"
    r")",
    flags=re.IGNORECASE,
)

GUIDED_CONFIRMED_STALE_RE = re.compile(
    r"("
    r"eu-codex-(?:welcome|title|subtitle)|"
    r"eu-copilot-session-(?:empty|item)|"
    r"session-dot|"
    r"eu-copilot-gate-row|"
    r"eu-copilot-study-workspace|"
    r"\.eu-study-step(?:[\s.{:#]|$)|"
    r"flow-(?:head|question|steps|step|facts|api|gate)"
    r")",
    flags=re.IGNORECASE,
)

GUIDED_FALSE_POSITIVE_RE = re.compile(
    r"("
    r"font-feature-settings|"
    r"guided_overrides\.css|"
    r"css/guided\.css|"
    r"EasyICU/js/icons\.js|"
    r"reference(?: glyph| breakpoints| parity| look| values|)"
    r")",
    flags=re.IGNORECASE,
)


def _guided_line_context(lines: list[str], index: int) -> str:
    start = max(0, index - 8)
    end = min(len(lines), index + 1)
    return " ".join(line.strip() for line in lines[start:end] if line.strip())


def _guided_hit_classification(route: str, line: str, context: str) -> str:
    text = f"{context} {line}".strip()
    if GUIDED_CONFIRMED_STALE_RE.search(line):
        return "confirmed_stale_selector_or_comment"
    if GUIDED_FALSE_POSITIVE_RE.search(line):
        return "false_positive_marker"
    if "eu-guided-fullscreen-marker" in text or re.search(r"\bgd-", text, flags=re.IGNORECASE):
        return "valid_guided_owned"
    if GUIDED_OWNER_GUARD_RE.search(text) or GUIDED_COMPONENT_RE.search(text):
        return "valid_copilot_assistant_owned"
    if route in {"agent", "cohort", "crossdb", "extract", "patient", "states", "tutorial"}:
        if re.search(r"(copilot|assistant|guided|study|workflow|handoff|route_fallback)", text, re.I):
            return "valid_copilot_assistant_owned"
    return "unclassified_marker"


def check_guided_owner_guards() -> dict[str, Any]:
    """Validate Guided/Copilot CSS ownership without touching dirty pytest files."""
    path = WEBAPP_DIR / "guided_overrides.css"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        stale_match = GUIDED_CONFIRMED_STALE_RE.search(line)
        matched_route = None
        matched_pattern = None
        if stale_match is not None:
            matched_route = "guided"
            matched_pattern = stale_match.group(0)
        else:
            for route, patterns in ROUTE_MARKERS.items():
                if route == "guided":
                    continue
                matched_pattern = next(
                    (
                        pattern
                        for pattern in patterns
                        if re.search(pattern, line, flags=re.IGNORECASE)
                    ),
                    None,
                )
                if matched_pattern is not None:
                    matched_route = route
                    break
        if matched_pattern is None or matched_route is None:
            continue
        context = _guided_line_context(lines, index)
        classification = _guided_hit_classification(matched_route, line, context)
        hits.append(
            {
                "line": index + 1,
                "route": matched_route,
                "pattern": matched_pattern,
                "classification": classification,
                "text": line.strip()[:180],
            }
        )
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit["classification"]] = counts.get(hit["classification"], 0) + 1
    issues = [
        hit
        for hit in hits
        if hit["classification"]
        in {"confirmed_stale_selector_or_comment", "unclassified_marker"}
    ]
    for marker in (".eu-guided-fullscreen-marker", ".eu-copilot-page-marker"):
        if marker not in text:
            issues.insert(
                0,
                {
                    "line": None,
                    "route": "guided",
                    "pattern": marker,
                    "classification": "missing_guided_owner_guard",
                    "text": "Guided CSS must retain the fullscreen and Copilot page marker owner guards.",
                },
            )
    return {
        "path": str(path),
        "line_count": len(lines),
        "has_guided_fullscreen_marker": ".eu-guided-fullscreen-marker" in text,
        "has_copilot_page_marker": ".eu-copilot-page-marker" in text,
        "valid_guided_owned": counts.get("valid_guided_owned", 0),
        "valid_copilot_assistant_owned": counts.get("valid_copilot_assistant_owned", 0),
        "false_positive_marker": counts.get("false_positive_marker", 0),
        "confirmed_stale_selector_or_comment": counts.get(
            "confirmed_stale_selector_or_comment", 0
        ),
        "unclassified_marker": counts.get("unclassified_marker", 0),
        "classification_counts": counts,
        "issues": issues[:50],
    }


EXTRACT_REQUIRED_GUARDS = (
    ".eu-source-header.page-head",
    ".eu-step2-design-marker",
    ".eu-step3-design-marker",
    ".eu-step4-design-marker",
    ".eu-export-progress-shell",
    ".eu-convert-dialog-marker",
)

EXTRACT_OWNER_GUARD_RE = re.compile(
    r"("
    r"eu-source-header|"
    r"eu_extract_breadcrumb|"
    r"eu_pipeline_step|"
    r"eu-step[1-4]|"
    r"step[1-4]_|"
    r"eu-source|"
    r"eu-real-source|"
    r"eu_demo|"
    r"eu_express|"
    r"eu-cohort|"
    r"eu_cohort|"
    r"cohort_builder|"
    r"cohort_disease_card|"
    r"concept_|"
    r"eu-concept|"
    r"eu-export|"
    r"eu_export|"
    r"post_export|"
    r"eu-extract|"
    r"eu-convert|"
    r"convert_|"
    r"validate_path|"
    r"sidebar_data_path|"
    r"final_export|"
    r"ex2-|"
    r"express|"
    r"icd-"
    r")",
    flags=re.IGNORECASE,
)

EXTRACT_FALSE_POSITIVE_RE = re.compile(
    r"("
    r"Patient/Cohort/Cross-DB routes|"
    r"already-checked tutorial, agent, patient, cohort, or settings pages|"
    r"states\.css|"
    r"running/conflict states|"
    r"doneState\(\)|"
    r"Cohort builder page|"
    r"\bReference:|"
    r"\breference header|"
    r"page-cohort-builder\.jsx|"
    r"font-feature-settings|"
    r"data:image/svg\\+xml|"
    r"w3\\.org"
    r")",
    flags=re.IGNORECASE,
)

EXTRACT_STALE_SOURCE_CLASSES = {
    "eu-export-complete-stats",
    "eu-export-content-row",
    "eu-export-contents-card",
    "eu-express-chips",
    "eu-express-left",
    "eu-express-note",
    "eu-eyebrow",
    "eu-source-metric",
    "eu-source-real-hero",
    "eu-source-real-summary",
    "eu-source-table-card",
    "summary-state",
}


def _webapp_python_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in WEBAPP_DIR.rglob("*.py")
    )


def _extract_line_context(lines: list[str], index: int) -> str:
    start = max(0, index - 8)
    end = min(len(lines), index + 1)
    return " ".join(line.strip() for line in lines[start:end] if line.strip())


def _extract_hit_classification(route: str, line: str, context: str) -> str:
    text = f"{context} {line}".strip()
    if EXTRACT_FALSE_POSITIVE_RE.search(text):
        return "false_positive_marker"
    if EXTRACT_OWNER_GUARD_RE.search(text):
        return "valid_extract_owned"
    if route in {"cohort", "dictionary"} and re.search(
        r"(eu-cohort|cohort_builder|dict-link|eu-source-keyterms)", text, re.I
    ):
        return "valid_extract_owned"
    return "unclassified_marker"


def check_extract_owner_guards() -> dict[str, Any]:
    """Validate Data Extraction CSS ownership without touching dirty pytest files."""
    path = WEBAPP_DIR / "extract_overrides.css"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    source = _webapp_python_text()

    source_missing_hits: list[dict[str, Any]] = []
    for class_name in sorted(EXTRACT_STALE_SOURCE_CLASSES):
        if class_name in source:
            continue
        for index, line in enumerate(lines):
            if class_name in line:
                source_missing_hits.append(
                    {
                        "line": index + 1,
                        "route": "extract",
                        "pattern": class_name,
                        "classification": "confirmed_stale_source_missing_class",
                        "text": line.strip()[:180],
                    }
                )
                break

    marker_hits_for_report: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        for route, patterns in ROUTE_MARKERS.items():
            if route == "extract":
                continue
            matched = next(
                (
                    pattern
                    for pattern in patterns
                    if re.search(pattern, line, flags=re.IGNORECASE)
                ),
                None,
            )
            if matched is None:
                continue
            context = _extract_line_context(lines, index)
            classification = _extract_hit_classification(route, line, context)
            marker_hits_for_report.append(
                {
                    "line": index + 1,
                    "route": route,
                    "pattern": matched,
                    "classification": classification,
                    "text": line.strip()[:180],
                }
            )
            break

    counts: dict[str, int] = {}
    for hit in [*source_missing_hits, *marker_hits_for_report]:
        counts[hit["classification"]] = counts.get(hit["classification"], 0) + 1

    issues = [
        *source_missing_hits,
        *[
            hit
            for hit in marker_hits_for_report
            if hit["classification"] == "unclassified_marker"
        ],
    ]
    for marker in EXTRACT_REQUIRED_GUARDS:
        if marker not in text:
            issues.insert(
                0,
                {
                    "line": None,
                    "route": "extract",
                    "pattern": marker,
                    "classification": "missing_extract_owner_guard",
                    "text": "Data Extraction CSS must retain required route owner guards.",
                },
            )
    return {
        "path": str(path),
        "line_count": len(lines),
        "required_guards_present": {
            marker: marker in text for marker in EXTRACT_REQUIRED_GUARDS
        },
        "valid_extract_owned": counts.get("valid_extract_owned", 0),
        "false_positive_marker": counts.get("false_positive_marker", 0),
        "confirmed_stale_source_missing_class": counts.get(
            "confirmed_stale_source_missing_class", 0
        ),
        "unclassified_marker": counts.get("unclassified_marker", 0),
        "classification_counts": counts,
        "issues": issues[:50],
    }


PATIENT_REQUIRED_GUARDS = (
    ".eu-qv-loaded-root",
    ".eu-qv-design-root",
    ".eu-qv-reference-table",
    ".eu-patient-empty-state",
)

PATIENT_STALE_SOURCE_CLASSES = {
    "eu-crossdb-empty-copy",
    "eu-qv-loaded-bar",
    "eu-qv-loaded-pill",
    "eu-qv-native-tabs-marker",
    "eu-qv-reference-note",
    "mt-4",
}

PATIENT_ALLOWED_RUNTIME_CLASSES = {
    # Plotly inserts this class at runtime; it is not expected in Python source.
    "modebar-container",
}

COHORT_REQUIRED_GUARDS = (
    ".eu-cohort-page-marker",
    ".eu-cohort-summary-grid",
    ".eu-cohort-ref-preflight",
)

COHORT_STALE_SOURCE_CLASSES = {
    "eu-cohort-header-gap",
    "eu-cohort-metric-card",
    "eu-cohort-readiness-node",
    "eu-cohort-readiness-row",
    "eu-readiness-strip",
}

CROSSDB_REQUIRED_GUARDS = (
    ".eu-crossdb-page-marker",
    ".eu-crossdb-summary-table",
    ".eu-crossdb-loaded-copy",
)

CROSSDB_STALE_SOURCE_CLASSES = {
    "eu-cohort-readiness-node",
    "eu-cohort-readiness-row",
    "eu-crossdb-availability-board",
    "eu-crossdb-availability-cell",
    "eu-crossdb-availability-head",
    "eu-crossdb-availability-legend",
    "eu-crossdb-empty-copy",
    "eu-crossdb-empty-steps",
    "eu-crossdb-gate-pill-wrap",
    "eu-crossdb-provenance-pill",
    "eu-crossdb-summary-head",
    "eu-crossdb-summary-ledger",
    "eu-readiness-strip",
}

STATES_REQUIRED_GUARDS = (
    ".eu-states-reference-head",
    ".eu-states-controls-reference",
    ".eu-states-stage-host",
    ".eu-state-preview-card",
)

STATES_STALE_SOURCE_CLASSES = {
    "eu-state-action-slot",
    "eu-state-body-pad",
    "eu-state-callout",
    "eu-state-loading-row",
    "eu-state-pills",
    "eu-states-control-divider",
    "eu-states-control-label",
    "eu-states-control-row",
    "eu-states-controls",
    "eu-states-overview-head",
    "eu-states-preview-actions-success",
}


def _source_has_class(source: str, class_name: str) -> bool:
    return (
        re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(class_name)}(?![A-Za-z0-9_-])",
            source,
        )
        is not None
    )


def _text_has_class_token(text: str, class_name: str) -> bool:
    return (
        re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(class_name)}(?![A-Za-z0-9_-])",
            text,
        )
        is not None
    )


def check_source_missing_owner_guards(
    *,
    path: Path,
    route: str,
    required_guards: tuple[str, ...],
    stale_source_classes: set[str],
) -> dict[str, Any]:
    """Validate route CSS required guards and source-missing stale classes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    source = _webapp_python_text()

    source_missing_hits: list[dict[str, Any]] = []
    for class_name in sorted(stale_source_classes):
        if _source_has_class(source, class_name):
            continue
        for index, line in enumerate(lines):
            if _text_has_class_token(line, class_name):
                source_missing_hits.append(
                    {
                        "line": index + 1,
                        "route": route,
                        "pattern": class_name,
                        "classification": "confirmed_stale_source_missing_class",
                        "text": line.strip()[:180],
                    }
                )
                break

    issues: list[dict[str, Any]] = [*source_missing_hits]
    for marker in required_guards:
        if marker not in text:
            issues.insert(
                0,
                {
                    "line": None,
                    "route": route,
                    "pattern": marker,
                    "classification": f"missing_{route}_owner_guard",
                    "text": f"{route} CSS must retain required route owner guards.",
                },
            )
    return {
        "path": str(path),
        "line_count": len(lines),
        "required_guards_present": {
            marker: marker in text for marker in required_guards
        },
        "confirmed_stale_source_missing_class": len(source_missing_hits),
        "unclassified_marker": 0,
        "issues": issues[:50],
    }


def check_patient_owner_guards() -> dict[str, Any]:
    """Validate Patient Review CSS ownership without touching dirty pytest files."""
    path = WEBAPP_DIR / "patient_overrides.css"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    source = _webapp_python_text()

    source_missing_hits: list[dict[str, Any]] = []
    for class_name in sorted(PATIENT_STALE_SOURCE_CLASSES):
        if class_name in source or class_name in PATIENT_ALLOWED_RUNTIME_CLASSES:
            continue
        for index, line in enumerate(lines):
            if class_name in line:
                source_missing_hits.append(
                    {
                        "line": index + 1,
                        "route": "patient",
                        "pattern": class_name,
                        "classification": "confirmed_stale_source_missing_class",
                        "text": line.strip()[:180],
                    }
                )
                break

    issues: list[dict[str, Any]] = [*source_missing_hits]
    for marker in PATIENT_REQUIRED_GUARDS:
        if marker not in text:
            issues.insert(
                0,
                {
                    "line": None,
                    "route": "patient",
                    "pattern": marker,
                    "classification": "missing_patient_owner_guard",
                    "text": "Patient Review CSS must retain required route owner guards.",
                },
            )
    return {
        "path": str(path),
        "line_count": len(lines),
        "required_guards_present": {
            marker: marker in text for marker in PATIENT_REQUIRED_GUARDS
        },
        "confirmed_stale_source_missing_class": len(source_missing_hits),
        "runtime_allowed_source_missing_class": sorted(PATIENT_ALLOWED_RUNTIME_CLASSES),
        "issues": issues[:50],
    }


def classify_cleanup(
    file_name: str,
    active_loaded: bool,
    loaded_by_default: bool,
    loaded_when_legacy_env_enabled: bool,
    status: str,
    marker_count: int,
) -> str:
    if not loaded_by_default and loaded_when_legacy_env_enabled:
        return "legacy_css_inactive_by_default_keep_until_stage24b"
    if not active_loaded:
        return "present_not_loaded"
    if status == "untracked":
        return "snapshot_first_untracked_imported_do_not_delete"
    if marker_count:
        return "imported_review_marker_samples_before_cleanup"
    return "imported_fallback_owner_do_not_delete"


def inventory() -> dict[str, Any]:
    head = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    present_css_names = [path.name for path in css_files()]
    referenced_css = referenced_css_from_shell_styles()
    default_loaded_css = default_loaded_css_from_shell_styles()
    legacy_enabled_loaded_css = legacy_enabled_loaded_css_from_shell_styles()
    shell_loaded = loaded_css_from_shell_styles()
    loaded_set = set(shell_loaded)
    default_loaded_set = set(default_loaded_css)
    legacy_enabled_loaded_set = set(legacy_enabled_loaded_css)
    missing_required_default_css = [
        name for name in DEFAULT_LOADED_CSS if name not in set(present_css_names)
    ]
    removed_legacy_split_css = [
        name for name in LEGACY_SPLIT_CSS if name not in set(present_css_names)
    ]
    files = []
    totals = {
        "files": 0,
        "lines": 0,
        "bytes": 0,
        "imported_files": 0,
        "active_loaded_files": 0,
        "active_loaded_lines": 0,
        "default_loaded_files": 0,
        "default_loaded_lines": 0,
        "legacy_enabled_loaded_files": 0,
        "legacy_enabled_loaded_lines": 0,
        "inactive_by_default_legacy_files": 0,
        "inactive_by_default_legacy_lines": 0,
        "untracked_files": 0,
    }
    for path in css_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        status = git_status(path)
        active_loaded = path.name in loaded_set
        loaded_by_default = path.name in default_loaded_set
        loaded_when_legacy_env_enabled = path.name in legacy_enabled_loaded_set
        inactive_by_default_legacy = (
            loaded_when_legacy_env_enabled
            and not loaded_by_default
            and path.name in LEGACY_SPLIT_CSS
        )
        hits = marker_hits(text, path.name)
        record = {
            "file": str(path),
            "name": path.name,
            "owner": CSS_OWNER_BY_FILE.get(path.name, "unknown"),
            "owner_route": OWNER_ROUTE_BY_FILE.get(path.name),
            "git_status": status,
            "present_in_webapp": True,
            "imported_by_shell_styles": active_loaded,
            "active_loaded_by_shell_styles": active_loaded,
            "loaded_by_default": loaded_by_default,
            "loaded_when_legacy_env_enabled": loaded_when_legacy_env_enabled,
            "inactive_by_default_legacy_css": inactive_by_default_legacy,
            "render_order": shell_loaded.index(path.name) + 1 if active_loaded else None,
            "default_render_order": default_loaded_css.index(path.name) + 1
            if loaded_by_default
            else None,
            "legacy_enabled_render_order": legacy_enabled_loaded_css.index(path.name) + 1
            if loaded_when_legacy_env_enabled
            else None,
            "lines": len(lines),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "important_count": text.count("!important"),
            "has_selector_count": text.count(":has("),
            "media_rule_count": len(re.findall(r"@media\b", text)),
            "marker_hits": hits,
            "cleanup_recommendation": classify_cleanup(
                path.name,
                active_loaded,
                loaded_by_default,
                loaded_when_legacy_env_enabled,
                status,
                len(hits),
            ),
        }
        files.append(record)
        totals["files"] += 1
        totals["lines"] += record["lines"]
        totals["bytes"] += record["bytes"]
        totals["imported_files"] += int(active_loaded)
        totals["active_loaded_files"] += int(active_loaded)
        totals["active_loaded_lines"] += record["lines"] if active_loaded else 0
        totals["default_loaded_files"] += int(loaded_by_default)
        totals["default_loaded_lines"] += record["lines"] if loaded_by_default else 0
        totals["legacy_enabled_loaded_files"] += int(loaded_when_legacy_env_enabled)
        totals["legacy_enabled_loaded_lines"] += (
            record["lines"] if loaded_when_legacy_env_enabled else 0
        )
        totals["inactive_by_default_legacy_files"] += int(inactive_by_default_legacy)
        totals["inactive_by_default_legacy_lines"] += (
            record["lines"] if inactive_by_default_legacy else 0
        )
        totals["untracked_files"] += int(status == "untracked")
    unimported = [item["name"] for item in files if not item["active_loaded_by_shell_styles"]]
    inactive_by_default = [
        item["name"] for item in files if item["inactive_by_default_legacy_css"]
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"branch": branch, "head": head},
        "legacy_streamlit_css_env": {
            "name": LEGACY_STREAMLIT_CSS_ENV,
            "enabled": legacy_css_env_enabled(),
            "required_value": "1",
        },
        "shell_styles": {
            "path": str(WEBAPP_DIR / "shell_styles.py"),
            "referenced_css": referenced_css,
            "present_css_files": present_css_names,
            "default_loaded_css": default_loaded_css,
            "legacy_enabled_loaded_css": legacy_enabled_loaded_css,
            "loaded_css": shell_loaded,
            "missing_expected_css": missing_required_default_css,
            "missing_required_default_css": missing_required_default_css,
            "stage24b_removed_legacy_split_css": removed_legacy_split_css,
            "inactive_by_default_legacy_css": inactive_by_default,
        },
        "totals": totals,
        "files": files,
        "archive_delete_candidates": unimported,
        "first_slice_decision": {
            "delete_now": [],
            "archive_only": inactive_by_default,
            "deleted_in_stage24b": removed_legacy_split_css,
            "reason": (
                "Stage24B removes legacy Streamlit split CSS after Stage24A made it inactive "
                "by default. Missing split CSS is intentional; only missing default CSS is a "
                "runtime inventory issue."
            ),
        },
    }
    report["baseline_staging_plan"] = baseline_staging_plan(report)
    return report


def baseline_staging_plan(report: dict[str, Any]) -> dict[str, Any]:
    """Return the current reversible legacy CSS decommission boundary."""
    status_entries = git_status_entries()
    default_loaded_files = [item for item in report["files"] if item["loaded_by_default"]]
    default_loaded_css_paths = [item["file"] for item in default_loaded_files]
    legacy_enabled_files = [
        item for item in report["files"] if item["loaded_when_legacy_env_enabled"]
    ]
    inactive_legacy_files = [
        item["file"]
        for item in legacy_enabled_files
        if item["inactive_by_default_legacy_css"]
    ]
    removed_legacy_files = [
        str(WEBAPP_DIR / name)
        for name in report["shell_styles"]["stage24b_removed_legacy_split_css"]
    ]
    dirty_webapp_python = sorted(
        entry["path"]
        for entry in status_entries
        if entry["path"].startswith("src/easyicu/webapp/")
        and entry["path"].endswith(".py")
        and entry["path"] != str(SHELL_STYLES_PATH)
    )
    dirty_tests = sorted(
        entry["path"]
        for entry in status_entries
        if entry["path"].startswith("tests/")
    )
    runnable_required = [
        str(SHELL_STYLES_PATH),
        *default_loaded_css_paths,
    ]
    return {
        "purpose": "stage24b_remove_inactive_legacy_streamlit_split_css",
        "split_css_removed_in_stage24b": True,
        "must_stage_for_default_runtime_boundary": runnable_required,
        "default_loaded_css_snapshot": default_loaded_css_paths,
        "inactive_by_default_legacy_css_remaining": inactive_legacy_files,
        "removed_legacy_split_css": removed_legacy_files,
        "stage24b_delete_candidates_after_validation": inactive_legacy_files,
        "already_tracked_css_no_new_stage_needed_if_unchanged": default_loaded_css_paths,
        "recommended_auxiliary_baseline_files": [
            "tools/inventory_legacy_streamlit_css.py",
            "src/easyicu/webapp/LEGACY.md",
        ],
        "optional_css_ownership_test_baseline": [
            str(OWNERSHIP_TEST_PATH),
        ]
        if any(entry["path"] == str(OWNERSHIP_TEST_PATH) for entry in status_entries)
        else [],
        "defer_dirty_webapp_python": dirty_webapp_python,
        "defer_dirty_tests_not_required_for_css_import_baseline": [
            path for path in dirty_tests if path != str(OWNERSHIP_TEST_PATH)
        ],
        "defer_non_webapp_untracked": sorted(
            entry["path"]
            for entry in status_entries
            if entry["status"] == "??"
            and not entry["path"].startswith("src/easyicu/webapp/")
            and not entry["path"].startswith("tools/inventory_legacy_streamlit_css.py")
        ),
        "why_shell_styles_is_required": (
            "shell_styles.py is the default-runtime cutover point: it loads only "
            "tokens.css and shell_overrides.css unless EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1."
        ),
        "why_no_css_file_delete": (
            "Stage24B has deleted the split CSS files. Restore them from git history "
            "at 63bba1c or earlier only for archive forensics."
        ),
    }


def write_snapshot_files(report: dict[str, Any], out_dir: Path, copy_css: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "css_inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sums = []
    if copy_css:
        css_dir = out_dir / "css_files"
        css_dir.mkdir(parents=True, exist_ok=True)
        for item in report["files"]:
            src = Path(item["file"])
            dst = css_dir / src.name
            shutil.copy2(src, dst)
            sums.append(f"{item['sha256']}  css_files/{src.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + ("\n" if sums else ""), encoding="utf-8")
    (out_dir / "css_inventory.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage21 Legacy Streamlit CSS Inventory",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Git: `{report['git']['branch']}` @ `{report['git']['head']}`",
        f"- CSS files: `{report['totals']['files']}`",
        f"- Total lines: `{report['totals']['lines']}`",
        f"- Active-loaded by `shell_styles.py`: `{report['totals']['active_loaded_files']}`",
        f"- Active-loaded lines: `{report['totals']['active_loaded_lines']}`",
        f"- Default-loaded CSS files: `{report['totals']['default_loaded_files']}`",
        f"- Default-loaded CSS lines: `{report['totals']['default_loaded_lines']}`",
        f"- Legacy-env loaded CSS files: `{report['totals']['legacy_enabled_loaded_files']}`",
        f"- Legacy-env loaded CSS lines: `{report['totals']['legacy_enabled_loaded_lines']}`",
        f"- Legacy CSS inactive by default: `{report['totals']['inactive_by_default_legacy_files']}`",
        f"- Legacy inactive lines: `{report['totals']['inactive_by_default_legacy_lines']}`",
        f"- Stage24B removed legacy split CSS files: `{len(report['shell_styles']['stage24b_removed_legacy_split_css'])}`",
        f"- Missing required default CSS files: `{len(report['shell_styles']['missing_required_default_css'])}`",
        f"- Legacy env: `{report['legacy_streamlit_css_env']['name']}={report['legacy_streamlit_css_env']['required_value']}`",
        f"- Legacy env currently enabled: `{report['legacy_streamlit_css_env']['enabled']}`",
        f"- Untracked imported CSS files: `{report['totals']['untracked_files']}`",
        "",
        "## Active Import Order",
        "",
    ]
    for idx, name in enumerate(report["shell_styles"]["loaded_css"], start=1):
        lines.append(f"{idx}. `{name}`")
    lines.extend(
        [
            "",
            "## File Matrix",
            "",
            "| File | Owner | Git | Imported | Lines | !important | :has | Recommendation |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in sorted(report["files"], key=lambda x: (x["render_order"] or 999, x["name"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['name']}`",
                    item["owner"],
                    f"`{item['git_status']}`",
                    "yes" if item["active_loaded_by_shell_styles"] else "no",
                    str(item["lines"]),
                    str(item["important_count"]),
                    str(item["has_selector_count"]),
                    item["cleanup_recommendation"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Marker Samples",
            "",
            "Marker hits are heuristic review prompts, not deletion proof.",
            "",
        ]
    )
    for item in report["files"]:
        if not item["marker_hits"]:
            continue
        lines.append(f"### `{item['name']}`")
        for hit in item["marker_hits"][:8]:
            sample = hit["samples"][0]
            lines.append(
                f"- {hit['kind']} `{hit['route']}` sample at line {sample['line']}: "
                f"`{sample['text']}`"
            )
        lines.append("")
    decision = report["first_slice_decision"]
    lines.extend(
        [
            "## First Slice Decision",
            "",
            f"- Delete now: `{len(decision['delete_now'])}` files",
            f"- Kept for legacy env opt-in: `{len(decision['archive_only'])}` files",
            f"- Deleted in Stage24B: `{len(decision['deleted_in_stage24b'])}` files",
            f"- Reason: {decision['reason']}",
            "",
        ]
    )
    plan = report["baseline_staging_plan"]
    lines.extend(
        [
            "## Baseline Staging Plan",
            "",
            "Stage these for the reversible default-runtime cutover:",
            "",
        ]
    )
    for path in plan["must_stage_for_default_runtime_boundary"]:
        lines.append(f"- `{path}`")
    if plan["optional_css_ownership_test_baseline"]:
        lines.extend(
            [
                "",
                "Optional test baseline:",
                "",
            ]
        )
        for path in plan["optional_css_ownership_test_baseline"]:
            lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "Do not stage these dirty WebApp Python files for the CSS import baseline:",
            "",
        ]
    )
    for path in plan["defer_dirty_webapp_python"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            f"Reason shell loader is required: {plan['why_shell_styles_is_required']}",
            f"Legacy split CSS recovery path: {plan['why_no_css_file_delete']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.check_agent_owner_guards:
        report = check_agent_owner_guards()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    if args.check_guided_owner_guards:
        report = check_guided_owner_guards()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    if args.check_extract_owner_guards:
        report = check_extract_owner_guards()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    if args.check_patient_owner_guards:
        report = check_patient_owner_guards()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    if args.check_cohort_owner_guards:
        report = check_source_missing_owner_guards(
            path=WEBAPP_DIR / "cohort_overrides.css",
            route="cohort",
            required_guards=COHORT_REQUIRED_GUARDS,
            stale_source_classes=COHORT_STALE_SOURCE_CLASSES,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    if args.check_crossdb_owner_guards:
        report = check_source_missing_owner_guards(
            path=WEBAPP_DIR / "crossdb_overrides.css",
            route="crossdb",
            required_guards=CROSSDB_REQUIRED_GUARDS,
            stale_source_classes=CROSSDB_STALE_SOURCE_CLASSES,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    if args.check_states_owner_guards:
        report = check_source_missing_owner_guards(
            path=WEBAPP_DIR / "states_overrides.css",
            route="states",
            required_guards=STATES_REQUIRED_GUARDS,
            stale_source_classes=STATES_STALE_SOURCE_CLASSES,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["issues"] else 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"inventory_{stamp}"
    report = inventory()
    write_snapshot_files(report, out_dir, copy_css=not args.no_copy)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "files": report["totals"]["files"],
                "lines": report["totals"]["lines"],
                "imported": report["totals"]["imported_files"],
                "active_loaded_lines": report["totals"]["active_loaded_lines"],
                "default_loaded": report["shell_styles"]["default_loaded_css"],
                "default_loaded_lines": report["totals"]["default_loaded_lines"],
                "legacy_env_enabled": report["legacy_streamlit_css_env"]["enabled"],
                "legacy_env_loaded_files": report["totals"]["legacy_enabled_loaded_files"],
                "legacy_env_loaded_lines": report["totals"]["legacy_enabled_loaded_lines"],
                "stage24b_removed_legacy_split_files": len(
                    report["shell_styles"]["stage24b_removed_legacy_split_css"]
                ),
                "missing_required_default_css": report["shell_styles"][
                    "missing_required_default_css"
                ],
                "inactive_by_default_legacy_files": report["totals"][
                    "inactive_by_default_legacy_files"
                ],
                "inactive_by_default_legacy_lines": report["totals"][
                    "inactive_by_default_legacy_lines"
                ],
                "untracked": report["totals"]["untracked_files"],
                "delete_now": report["first_slice_decision"]["delete_now"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
