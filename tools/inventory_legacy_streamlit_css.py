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


def loaded_css_from_shell_styles() -> list[str]:
    path = WEBAPP_DIR / "shell_styles.py"
    text = path.read_text(encoding="utf-8")
    found = re.findall(r'with_name\("([^"]+\.css)"\)', text)
    return [name for name in EXPECTED_RENDER_ORDER if name in set(found)] + [
        name for name in found if name not in EXPECTED_RENDER_ORDER
    ]


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


def classify_cleanup(file_name: str, imported: bool, status: str, marker_count: int) -> str:
    if not imported:
        return "archive_candidate_not_imported"
    if status == "untracked":
        return "snapshot_first_untracked_imported_do_not_delete"
    if marker_count:
        return "imported_review_marker_samples_before_cleanup"
    return "imported_fallback_owner_do_not_delete"


def inventory() -> dict[str, Any]:
    head = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    shell_loaded = loaded_css_from_shell_styles()
    loaded_set = set(shell_loaded)
    files = []
    totals = {"files": 0, "lines": 0, "bytes": 0, "imported_files": 0, "untracked_files": 0}
    for path in css_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        status = git_status(path)
        imported = path.name in loaded_set
        hits = marker_hits(text, path.name)
        record = {
            "file": str(path),
            "name": path.name,
            "owner": CSS_OWNER_BY_FILE.get(path.name, "unknown"),
            "owner_route": OWNER_ROUTE_BY_FILE.get(path.name),
            "git_status": status,
            "imported_by_shell_styles": imported,
            "render_order": shell_loaded.index(path.name) + 1 if imported else None,
            "lines": len(lines),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "important_count": text.count("!important"),
            "has_selector_count": text.count(":has("),
            "media_rule_count": len(re.findall(r"@media\b", text)),
            "marker_hits": hits,
            "cleanup_recommendation": classify_cleanup(path.name, imported, status, len(hits)),
        }
        files.append(record)
        totals["files"] += 1
        totals["lines"] += record["lines"]
        totals["bytes"] += record["bytes"]
        totals["imported_files"] += int(imported)
        totals["untracked_files"] += int(status == "untracked")
    unimported = [item["name"] for item in files if not item["imported_by_shell_styles"]]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"branch": branch, "head": head},
        "shell_styles": {
            "path": str(WEBAPP_DIR / "shell_styles.py"),
            "loaded_css": shell_loaded,
            "missing_expected_css": [name for name in EXPECTED_RENDER_ORDER if name not in loaded_set],
        },
        "totals": totals,
        "files": files,
        "archive_delete_candidates": unimported,
        "first_slice_decision": {
            "delete_now": [],
            "archive_only": [item["name"] for item in files if item["imported_by_shell_styles"]],
            "reason": (
                "All CSS files present in src/easyicu/webapp are loaded by shell_styles.py. "
                "The 14 split route CSS files are untracked but imported, so deleting or editing "
                "them before an archive snapshot would mix cleanup with the dirty fallback line."
            ),
        },
    }
    report["baseline_staging_plan"] = baseline_staging_plan(report)
    return report


def baseline_staging_plan(report: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit no-cleanup staging boundary for legacy fallback CSS."""
    status_entries = git_status_entries()
    imported_files = [item for item in report["files"] if item["imported_by_shell_styles"]]
    imported_css_paths = [item["file"] for item in imported_files]
    untracked_imported_css = [
        item["file"] for item in imported_files if item["git_status"] == "untracked"
    ]
    modified_imported_css = [
        item["file"]
        for item in imported_files
        if item["git_status"] not in {"clean_tracked", "untracked"}
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
        *modified_imported_css,
        *untracked_imported_css,
    ]
    return {
        "purpose": "baseline_legacy_streamlit_fallback_before_selector_cleanup",
        "do_not_cleanup_in_baseline": True,
        "must_stage_for_runnable_fallback_css_baseline": runnable_required,
        "all_imported_css_snapshot": imported_css_paths,
        "untracked_imported_css_should_be_added": untracked_imported_css,
        "modified_imported_css_should_be_added": modified_imported_css,
        "already_tracked_css_no_new_stage_needed_if_unchanged": [
            item["file"] for item in imported_files if item["git_status"] == "clean_tracked"
        ],
        "recommended_auxiliary_baseline_files": [
            "tools/inventory_legacy_streamlit_css.py",
            "tools/legacy_streamlit_fallback_baseline_stage21b.json",
            "docs/legacy_streamlit_fallback_baseline_stage21b.md",
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
            "HEAD shell_styles.py only loads tokens.css and shell_overrides.css; "
            "the current fallback split CSS baseline is runnable only with the "
            "working-tree shell_styles.py loader changes."
        ),
        "why_no_css_file_delete": (
            "All 16 CSS files under src/easyicu/webapp are imported by shell_styles.py, "
            "so file-level deletion has no current evidence base."
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
        f"- Imported by `shell_styles.py`: `{report['totals']['imported_files']}`",
        f"- Untracked imported CSS files: `{report['totals']['untracked_files']}`",
        "",
        "## Import Order",
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
                    "yes" if item["imported_by_shell_styles"] else "no",
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
            f"- Archive-only imported files: `{len(decision['archive_only'])}` files",
            f"- Reason: {decision['reason']}",
            "",
        ]
    )
    plan = report["baseline_staging_plan"]
    lines.extend(
        [
            "## Baseline Staging Plan",
            "",
            "Stage these for a runnable legacy fallback CSS baseline:",
            "",
        ]
    )
    for path in plan["must_stage_for_runnable_fallback_css_baseline"]:
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
            f"Reason no CSS file is deleted: {plan['why_no_css_file_delete']}",
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
