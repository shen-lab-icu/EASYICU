#!/usr/bin/env python3
"""Browser/computed-style guard for legacy Streamlit CSS cleanup.

Run from ``EASYICU/`` with the Streamlit fallback already listening:

    python tools/qa_legacy_streamlit_css_guard.py \
      --base-url http://127.0.0.1:8513/ --label before

The guard records desktop and mobile browser evidence for key legacy fallback
routes. It is intentionally read-only: CSS cleanup patches can run it before
and after a deletion to prove critical computed styles did not move.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


VIEWPORTS = [
    ("desktop", 1440, 900),
    ("mobile", 393, 852),
]

EXTRACT_SELECTORS = [
    "[data-testid='stMain']",
    "[data-testid='stSidebar']",
    ".eu-topbar",
    ".eu-topbar-ref-controls",
    ".eu-source-header.page-head",
    ".eu-extract-session-console",
    ".eu-extract-console-cards",
    ".eu-extract-console-steps",
    ".eu-step2-design-marker",
    ".eu-cohort-header",
    "[class*='st-key-eu_cohort_demographics_card']",
    ".eu-step3-design-marker",
    ".eu-step4-design-marker",
    ".eu-export-progress-shell",
    ".eu-demo-ex2-left",
    ".eu-step3-modules-cfg",
    "[class*='st-key-eu_export_settings_card']",
    "[class*='st-key-eu_step4_summary_card']",
    ".eu-step4-run-link",
]

PATIENT_SELECTORS = [
    "[data-testid='stMain']",
    "[data-testid='stSidebar']",
    ".eu-topbar",
    ".eu-topbar-ref-controls",
    ".eu-qv-design-root",
    ".eu-qv-loaded-root",
    ".eu-qv-idle-root",
    ".eu-qv-panel-root",
    ".eu-qv-reference-table",
    ".eu-qv-reference-stats",
    ".eu-qv-series-grid",
    ".eu-ts-lane-head",
    ".eu-ts-notice",
    ".eu-ts-static-value",
    ".eu-qv-patient-chip-row",
    ".eu-qv-patient-split",
    ".eu-qv-quality-card",
    ".eu-qv-quality-note",
    ".quality-summary-grid",
    ".quality-summary-card",
    ".quality-issue-panel",
    ".quality-issue-card",
    ".eu-quality-notice",
    ".eu-qv-rail",
    ".eu-qv-rail-head",
    ".eu-qv-rail-sep",
    ".eu-qv-rail-edit",
    ".eu-qv-rail .setup-row",
    "[class*='st-key-eu_qv_loaded_bar']",
    ".eu-qv-loaded-copy-line",
    ".loaded-bar",
    ".eu-qv-loaded-export-visual",
    "[data-testid='stPlotlyChart']",
    "[data-testid='stDataFrame']",
    ".eu-qv-nextbar-root",
]

GUIDED_SELECTORS = [
    "[data-testid='stMain']",
    ".eu-guided-fullscreen-marker",
    ".eu-copilot-page-marker",
    "div.st-key-ai_assistant_page_panel",
    "div.st-key-eu_copilot_guided_top",
    ".eu-copilot-topbrand",
    "div.st-key-eu_copilot_guided_shell",
    "div.st-key-eu_copilot_left_rail",
    ".eu-copilot-rail-body",
    "div.st-key-eu_copilot_session_active_0",
    "div.st-key-eu_copilot_conversation_shell",
    "div.st-key-_llm_ai_page_workspace_history",
    ".eu-copilot-welcome-thread",
    ".eu-copilot-dynamic-thread",
    ".eu-copilot-msg",
    ".eu-copilot-gd-conv-marker",
    "div.st-key-_llm_ai_page_workspace_guided_intents",
    "div.st-key-_llm_ai_page_workspace_guided_hints",
    "div.st-key-_llm_ai_page_workspace_composer_wrap",
    ".eu-copilot-composer-foot",
    "div.st-key-eu_copilot_right_rail",
    "div.st-key-eu_copilot_study_rail",
    ".eu-copilot-study-rail-head",
    "div.st-key-eu_study_step_list",
    "div.st-key-eu_study_step_row_0_pending_question",
    ".eu-copilot-evidence-note",
]

SHELL_SELECTORS = [
    "[data-testid='stMain']",
    "[data-testid='stSidebar']",
    "[data-testid='stSidebarUserContent']",
    ".eu-topbar",
    ".eu-topbar-ref-controls",
    ".eu-sidebar-footer-rule",
    "div.st-key-eu_sidebar_nav_area",
    "div.st-key-eu_sidebar_dock",
    "div.st-key-eu_sidebar_footer",
    ".wsnav",
    ".wsgroup-head",
    ".wsg-children",
    ".wsitem",
    "[class*='st-key-eunavrow_classic_workspace']",
    "[class*='st-key-euonav_extract']",
    "[class*='st-key-euonav_quick_viz']",
    "[class*='st-key-euonav_cohort']",
    "[class*='st-key-euonav_cross_db']",
    "div.st-key-main_nav_bar",
    "[data-testid='stRadio']",
    "div.st-key-floating_ai_launcher",
    "div.st-key-floating_ai_panel",
    ".eu-settings-page-marker",
]

ROUTES: dict[str, dict[str, Any]] = {
    "extract": {
        "path": "?page=extract&mode=demo",
        "required_any": [
            [".eu-source-header.page-head"],
        ],
        "selectors": EXTRACT_SELECTORS,
    },
    "extract_idle": {
        "path": "?page=extract&mode=demo",
        "required_any": [
            [".eu-source-header.page-head"],
        ],
        "state_required_any": [
            [".eu-extract-session-console", ".eu-demo-ex2-left", ".eu-source-header.page-head"],
        ],
        "selectors": EXTRACT_SELECTORS,
    },
    "extract_step2": {
        "path": "?page=extract&mode=demo&ex_action=open_adv_cohort&ex_step=2&ex_custom=1&ex_adv_cohort=1",
        "wait_for_guard_ms": 6_000,
        "required_any": [
            ["[data-testid='stMain']"],
        ],
        "state_required_any": [
            [".eu-step2-design-marker", ".eu-cohort-header", "[class*='st-key-eu_cohort_demographics_card']"],
        ],
        "selectors": EXTRACT_SELECTORS,
    },
    "extract_step3": {
        "path": "?page=extract&mode=demo&ex_action=show_core_modules&ex_step=3&ex_show_all=1",
        "wait_for_guard_ms": 6_000,
        "required_any": [
            ["[data-testid='stMain']"],
        ],
        "state_required_any": [
            [".eu-step3-design-marker", ".eu-step3-modules-cfg"],
        ],
        "selectors": EXTRACT_SELECTORS,
    },
    "extract_step4": {
        "path": "?page=extract&mode=demo&ex_action=open_adv_export&ex_step=4&ex_adv_export=1&ex_format=csv&ex_merge=separate",
        "wait_for_guard_ms": 6_000,
        "required_any": [
            ["[data-testid='stMain']"],
        ],
        "state_required_any": [
            [".eu-step4-design-marker", ".eu-step4-run-link"],
        ],
        "selectors": EXTRACT_SELECTORS,
    },
    "extract_export_preview": {
        "path": "?page=extract&mode=demo&ex_action=open_adv_export&ex_step=4&ex_adv_export=1&ex_format=csv&ex_merge=separate",
        "wait_for_guard_ms": 6_000,
        "required_any": [
            ["[data-testid='stMain']"],
        ],
        "state_required_any": [
            [".eu-step4-design-marker", ".eu-step4-run-link"],
            ["[class*='st-key-eu_export_settings_card']", "[class*='st-key-eu_step4_summary_card']"],
        ],
        "selectors": EXTRACT_SELECTORS,
    },
    "patient": {
        "path": "?page=patient&mode=demo",
        "required_any": [
            [".eu-qv-design-root"],
            [".eu-qv-loaded-root", ".eu-qv-idle-root", ".eu-qv-panel-root"],
        ],
        "selectors": PATIENT_SELECTORS,
    },
    "patient_idle": {
        "path": "?page=patient&mode=demo",
        "required_any": [
            [".eu-qv-design-root"],
        ],
        "state_required_any": [
            [".eu-qv-idle-root", ".eu-qv-loaded-root"],
        ],
        "selectors": PATIENT_SELECTORS,
    },
    "patient_loaded_tables": {
        "path": "?page=patient&mode=demo&qv_action=panel&qv_source=demo&qv_panel=data_tables&qv_patients=10&qv_hours=24",
        "wait_for_guard_ms": 8_000,
        "required_any": [
            [".eu-qv-design-root"],
        ],
        "state_required_any": [
            [".eu-qv-loaded-root"],
            [".eu-qv-reference-table", ".eu-qv-reference-stats"],
        ],
        "selectors": PATIENT_SELECTORS,
    },
    "patient_loaded_time_series": {
        "path": "?page=patient&mode=demo&qv_action=panel&qv_source=demo&qv_panel=time_series&qv_patients=10&qv_hours=24",
        "wait_for_guard_ms": 10_000,
        "required_any": [
            [".eu-qv-design-root"],
        ],
        "state_required_any": [
            [".eu-qv-loaded-root"],
            [".eu-qv-series-grid", ".eu-ts-lane-head", ".eu-ts-notice", "[data-testid='stPlotlyChart']"],
        ],
        "selectors": PATIENT_SELECTORS,
    },
    "patient_loaded_overview": {
        "path": "?page=patient&mode=demo&qv_action=panel&qv_source=demo&qv_panel=patient_overview&qv_patients=10&qv_hours=24",
        "wait_for_guard_ms": 8_000,
        "required_any": [
            [".eu-qv-design-root"],
        ],
        "state_required_any": [
            [".eu-qv-loaded-root"],
            [".eu-qv-patient-chip-row", ".eu-qv-patient-split"],
        ],
        "selectors": PATIENT_SELECTORS,
    },
    "patient_loaded_quality": {
        "path": "?page=patient&mode=demo&qv_action=panel&qv_source=demo&qv_panel=data_quality&qv_patients=10&qv_hours=24",
        "wait_for_guard_ms": 10_000,
        "required_any": [
            [".eu-qv-design-root"],
        ],
        "state_required_any": [
            [".eu-qv-loaded-root"],
            [
                ".eu-qv-quality-card",
                ".eu-qv-quality-note",
                ".quality-summary-grid",
                ".quality-issue-panel",
                ".eu-quality-notice",
                "[data-testid='stPlotlyChart']",
            ],
        ],
        "selectors": PATIENT_SELECTORS,
    },
    "guided": {
        "path": "?page=guided",
        "required_any": [
            ["[data-testid='stMain']"],
            ["div.st-key-ai_assistant_page_panel"],
        ],
        "selectors": GUIDED_SELECTORS,
    },
    "guided_welcome": {
        "path": "?page=guided",
        "wait_for_guard_ms": 8_000,
        "required_any": [
            ["[data-testid='stMain']"],
            ["div.st-key-ai_assistant_page_panel"],
            ["div.st-key-eu_copilot_guided_shell"],
        ],
        "state_required_any": [
            [".eu-copilot-welcome-thread", "div.st-key-_llm_ai_page_workspace_guided_intents"],
            ["div.st-key-_llm_ai_page_workspace_composer_wrap"],
        ],
        "state_required_any_by_viewport": {
            "desktop": [
                ["div.st-key-eu_copilot_left_rail"],
                ["div.st-key-eu_copilot_right_rail"],
            ],
            "mobile": [
                ["div.st-key-eu_copilot_right_rail", "div.st-key-eu_copilot_study_rail"],
            ],
        },
        "expected_hidden_any_by_viewport": {
            "mobile": [
                ["div.st-key-eu_copilot_left_rail"],
            ],
        },
        "selectors": GUIDED_SELECTORS,
    },
    "guided_study_workspace": {
        "path": "?page=guided",
        "wait_for_guard_ms": 8_000,
        "required_any": [
            ["[data-testid='stMain']"],
            ["div.st-key-ai_assistant_page_panel"],
            ["div.st-key-eu_copilot_guided_shell"],
        ],
        "state_required_any": [
            [".eu-copilot-dynamic-thread", ".eu-copilot-msg"],
            [".eu-copilot-study-rail-head", "div.st-key-eu_study_step_list"],
            ["div.st-key-_llm_ai_page_workspace_composer_wrap"],
        ],
        "interactions": [
            {
                "type": "click_role",
                "role": "button",
                "name_any": [
                    "All the way to a gated draft",
                    "Data, then a visual review",
                    "Just a cohort & data",
                ],
                "optional": True,
                "wait_ms": 1200,
            }
        ],
        "selectors": GUIDED_SELECTORS,
    },
    "agent": {
        "path": "?page=agent&mode=demo",
        "required_any": [
            [".eu-agent-page-marker"],
            [".eu-agent-reference-head"],
            ["div.st-key-eu_agent_projects_shell"],
        ],
        "selectors": [
            "[data-testid='stMain']",
            "[data-testid='stSidebar']",
            ".eu-topbar-ref-controls",
            ".eu-agent-page-marker",
            ".eu-agent-reference-head",
            "div.st-key-eu_agent_projects_shell",
            ".eu-agent-project-panel",
            ".eu-agent-project-run-list",
            ".eu-agent-project-run-row",
            ".eu-agent-reference-shell.eu-agent-project-main",
            ".eu-agent-project-detail-head",
            ".eu-agent-project-pipeline",
            "div.st-key-eu_agent_project_tabs",
            ".eu-agent-reference-shell.eu-agent-project-body-shell",
            ".eu-agent-reference-gate-card",
            ".eu-agent-reference-plan",
            ".eu-agent-reference-context",
            ".eu-agent-linked-cohort",
            ".eu-agent-context-grid",
            ".handoff",
        ],
    },
    "agent_history": {
        "path": "?page=agent&mode=demo&eu_agent_project_action=view_history&eu_agent_project_study=sepsis&eu_agent_project_mode=analysis",
        "required_any": [
            [".eu-agent-page-marker"],
            [".ag-runs-card"],
            [".runrow"],
        ],
        "selectors": [
            "[data-testid='stMain']",
            "[data-testid='stSidebar']",
            ".eu-agent-reference-head",
            "div.st-key-eu_agent_projects_shell",
            ".eu-agent-project-panel",
            ".eu-agent-reference-shell.eu-agent-project-main",
            "div.st-key-eu_agent_project_tabs",
            ".eu-agent-reference-shell.eu-agent-project-body-shell",
            ".ag-runs-card",
            ".runrow",
            ".rn-node",
        ],
    },
    "agent_outputs": {
        "path": "?page=agent&mode=demo&eu_agent_project_action=view_workbench&eu_agent_project_study=sepsis&eu_agent_project_mode=analysis",
        "required_any": [
            [".eu-agent-page-marker"],
            [".outgrid"],
            [".outcard"],
        ],
        "selectors": [
            "[data-testid='stMain']",
            "[data-testid='stSidebar']",
            ".eu-agent-reference-head",
            "div.st-key-eu_agent_projects_shell",
            ".eu-agent-project-panel",
            ".eu-agent-reference-shell.eu-agent-project-main",
            "div.st-key-eu_agent_project_tabs",
            ".eu-agent-reference-shell.eu-agent-project-body-shell",
            ".outgrid",
            ".outcard",
            ".outthumb",
            ".note.demo",
        ],
    },
    "agent_summary": {
        "path": "?page=agent&mode=demo&eu_agent_project_action=view_summary&eu_agent_project_study=sepsis&eu_agent_project_mode=analysis",
        "required_any": [
            [".eu-agent-page-marker"],
            [".checks2"],
            [".chk"],
        ],
        "selectors": [
            "[data-testid='stMain']",
            "[data-testid='stSidebar']",
            ".eu-agent-reference-head",
            "div.st-key-eu_agent_projects_shell",
            ".eu-agent-project-panel",
            ".eu-agent-reference-shell.eu-agent-project-main",
            "div.st-key-eu_agent_project_tabs",
            ".eu-agent-reference-shell.eu-agent-project-body-shell",
            ".checks2",
            ".chk",
            ".nextbar",
            "[data-ag-signoff]",
        ],
    },
    "shell": {
        "path": "?page=settings&mode=demo",
        "required_any": [
            ["[data-testid='stMain']"],
            ["[data-testid='stSidebar']"],
            [".eu-topbar", ".eu-topbar-ref-controls"],
        ],
        "selectors": SHELL_SELECTORS,
    },
    "shell_navigation": {
        "path": "?page=settings&mode=demo",
        "wait_for_guard_ms": 8_000,
        "required_any": [
            ["[data-testid='stMain']"],
        ],
        "state_required_any_by_viewport": {
            "desktop": [
                ["div.st-key-eu_sidebar_nav_area", ".wsnav", ".wsitem"],
                [".eu-topbar-ref-controls", ".eu-topbar"],
            ],
            "mobile": [
                ["div.st-key-main_nav_bar"],
            ],
        },
        "expected_hidden_any_by_viewport": {
            "mobile": [
                ["div.st-key-eu_sidebar_nav_area", ".wsnav", ".wsitem"],
                [".eu-topbar-ref-controls", ".eu-topbar"],
            ],
        },
        "selectors": SHELL_SELECTORS,
    },
}

STYLE_KEYS = [
    "display",
    "position",
    "boxSizing",
    "gridTemplateColumns",
    "gridTemplateRows",
    "flexDirection",
    "alignItems",
    "justifyContent",
    "width",
    "height",
    "minWidth",
    "maxWidth",
    "paddingTop",
    "paddingRight",
    "paddingBottom",
    "paddingLeft",
    "gap",
    "rowGap",
    "columnGap",
    "marginTop",
    "marginBottom",
    "overflowX",
    "overflowY",
]

STABLE_COMPARE_STYLE_KEYS = [
    key
    for key in STYLE_KEYS
    if key
    not in {
        "width",
        "height",
        "minWidth",
        "maxWidth",
        "gridTemplateColumns",
        "gridTemplateRows",
        "overflowY",
    }
]

QA_JS = """
(payload) => {
  const routeConfig = payload.routeConfig;
  const styleKeys = payload.styleKeys;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const doc = document.documentElement;
  const body = document.body;
  const visible = (el) => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity) !== 0
      && rect.width >= 1 && rect.height >= 1;
  };
  const label = (el) => {
    const cls = typeof el.className === 'string' ? el.className : '';
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      cls: cls.replace(/\\s+/g, '.').slice(0, 120),
    };
  };
  const styleFor = (selector) => {
    const nodes = Array.from(document.querySelectorAll(selector));
    const node = nodes.find(visible) || nodes[0] || null;
    if (!node) return {exists: false, visible: false, count: nodes.length};
    const cs = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    const styles = {};
    for (const key of styleKeys) styles[key] = cs[key] || '';
    return {
      exists: true,
      visible: visible(node),
      count: nodes.length,
      rect: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      },
      styles,
      textLength: (node.textContent || '').trim().length,
    };
  };
  const walker = document.createTreeWalker(body, NodeFilter.SHOW_ELEMENT);
  const offscreen = [];
  const clipped = [];
  while (walker.nextNode()) {
    const el = walker.currentNode;
    if (el.closest('template, script, style, [aria-hidden="true"]')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    if (cs.position === 'fixed' && (rect.left >= vw || rect.right <= 0)) continue;
    if (rect.right > vw + 1 || rect.left < -1) {
      offscreen.push({...label(el), left: Math.round(rect.left), right: Math.round(rect.right), width: Math.round(rect.width)});
    }
    const clipsX = el.scrollWidth > el.clientWidth + 2;
    const clipsY = el.scrollHeight > el.clientHeight + 2;
    if ((clipsX || clipsY) && /(hidden|clip)/.test(`${cs.overflow} ${cs.overflowX} ${cs.overflowY}`)) {
      clipped.push({
        ...label(el),
        scrollWidth: el.scrollWidth,
        clientWidth: el.clientWidth,
        scrollHeight: el.scrollHeight,
        clientHeight: el.clientHeight,
      });
    }
  }
  const required = routeConfig.required_any.map((group) => ({
    anyOf: group,
    present: group.some((selector) => document.querySelectorAll(selector).length > 0),
    visible: group.some((selector) => Array.from(document.querySelectorAll(selector)).some(visible)),
  }));
  const stateRequired = (routeConfig.state_required_any || []).map((group) => ({
    anyOf: group,
    present: group.some((selector) => document.querySelectorAll(selector).length > 0),
    visible: group.some((selector) => Array.from(document.querySelectorAll(selector)).some(visible)),
  }));
  const expectedHidden = (routeConfig.expected_hidden_any || []).map((group) => ({
    anyOf: group,
    present: group.some((selector) => document.querySelectorAll(selector).length > 0),
    visible: group.some((selector) => Array.from(document.querySelectorAll(selector)).some(visible)),
  }));
  const computed = {};
  for (const selector of routeConfig.selectors) computed[selector] = styleFor(selector);
    const keyCounts = {
    buttons: Array.from(document.querySelectorAll('button')).filter(visible).length,
    tables: Array.from(document.querySelectorAll('table, [role="table"], [data-testid="stDataFrame"]')).filter(visible).length,
    rails: Array.from(document.querySelectorAll('.eu-dict-rail, .eu-copilot-stage-rail, .eu-agent-project-panel, .wsitem, [data-testid="stSidebar"]')).filter(visible).length,
    topbar: Array.from(document.querySelectorAll('.eu-topbar, .eu-topbar-ref-controls')).filter(visible).length,
  };
  const h1 = document.querySelector('h1');
  return {
    title: h1 ? h1.textContent.trim() : '',
    url: location.href,
    root: {
      viewport: {width: vw, height: vh},
      documentWidth: Math.max(doc.scrollWidth, body.scrollWidth),
      documentHeight: Math.max(doc.scrollHeight, body.scrollHeight),
      overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - vw,
      overflowY: Math.max(doc.scrollHeight, body.scrollHeight) - vh,
    },
    required,
    stateRequired,
    expectedHidden,
    guardBlockers: [
      ...stateRequired
        .filter((group) => !group.visible)
        .map((group) => `state not visible: ${group.anyOf.join(' OR ')}`),
      ...expectedHidden
        .filter((group) => group.visible)
        .map((group) => `expected hidden but visible: ${group.anyOf.join(' OR ')}`),
    ],
    computed,
    keyCounts,
    offscreenCount: offscreen.length,
    offscreenSample: offscreen.slice(0, 12),
    clippedCount: clipped.length,
    clippedSample: clipped.slice(0, 12),
  };
}
"""


def _route_config_for_viewport(route: str, viewport_name: str) -> dict[str, Any]:
    config = deepcopy(ROUTES[route])
    for key in ("state_required_any", "expected_hidden_any"):
        merged = list(config.get(key) or [])
        by_viewport = config.pop(f"{key}_by_viewport", {}) or {}
        merged.extend(by_viewport.get(viewport_name, []) or [])
        if merged:
            config[key] = merged
        else:
            config.pop(key, None)
    return config


def _apply_interactions(page: Any, route_config: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for idx, action in enumerate(route_config.get("interactions", []), start=1):
        action_type = str(action.get("type") or "")
        optional = bool(action.get("optional", False))
        wait_ms = int(action.get("wait_ms") or 700)
        try:
            if action_type == "click_role":
                role = str(action.get("role") or "button")
                names = action.get("name_any") or [action.get("name")]
                clicked = False
                for raw_name in names:
                    name = str(raw_name or "").strip()
                    if not name:
                        continue
                    try:
                        page.get_by_role(role, name=re.compile(re.escape(name), re.I)).first.click(
                            timeout=2500
                        )
                        clicked = True
                        break
                    except PlaywrightTimeoutError:
                        continue
                if not clicked:
                    raise PlaywrightTimeoutError(f"no matching {role} for {names}")
            else:
                raise ValueError(f"unsupported interaction type: {action_type}")
            try:
                page.wait_for_load_state("networkidle", timeout=8_000)
            except PlaywrightTimeoutError:
                pass
            page.wait_for_timeout(wait_ms)
        except Exception as exc:
            message = f"interaction {idx} {action_type} failed: {exc}"
            if optional:
                blockers.append(message)
                continue
            raise
    return blockers


def _evaluate_guard(page: Any, route_config: dict[str, Any]) -> dict[str, Any]:
    return page.evaluate(QA_JS, {"routeConfig": route_config, "styleKeys": STYLE_KEYS})


def _wait_for_streamlit_idle(page: Any, timeout_ms: int = 12_000) -> None:
    try:
        page.wait_for_function(
            """
            () => {
              const app = document.querySelector('[data-testid="stApp"]');
              if (!app) return false;
              return app.getAttribute('data-test-script-state') !== 'running';
            }
            """,
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        pass


def _wait_for_guard_contract(page: Any, route_config: dict[str, Any]) -> dict[str, Any]:
    """Wait for Streamlit reruns to settle into the expected route state."""
    _wait_for_streamlit_idle(page)
    result = _evaluate_guard(page, route_config)
    deadline = time.monotonic() + (int(route_config.get("wait_for_guard_ms") or 0) / 1000)
    while result.get("guardBlockers") and time.monotonic() < deadline:
        page.wait_for_timeout(500)
        _wait_for_streamlit_idle(page, timeout_ms=5_000)
        result = _evaluate_guard(page, route_config)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Streamlit fallback base URL")
    parser.add_argument("--out-dir", default="output/playwright", help="Guard output root")
    parser.add_argument("--label", default="guard", help="Run label, for example before or after")
    parser.add_argument("--routes", nargs="*", choices=sorted(ROUTES), default=sorted(ROUTES))
    parser.add_argument("--compare-before", help="Previous guard JSON to compare against")
    parser.add_argument("--no-screenshots", action="store_true", help="Skip screenshot capture")
    parser.add_argument("--strict-offscreen", action="store_true", help="Fail on offscreen/clipped samples")
    return parser.parse_args()


def normalize_base(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def assert_streamlit_ready(base_url: str) -> None:
    try:
        with urllib.request.urlopen(base_url + "_stcore/health", timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Streamlit fallback is not ready at {base_url}: {exc}") from exc
    if "ok" not in body.lower():
        raise SystemExit(f"Unexpected Streamlit health response at {base_url}: {body[:120]}")


def collect_one(
    page: Any,
    *,
    base_url: str,
    route: str,
    viewport_name: str,
    width: int,
    height: int,
    out_dir: Path,
    screenshots: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    route_config = _route_config_for_viewport(route, viewport_name)
    target = base_url + route_config["path"]
    page.goto(target, wait_until="domcontentloaded", timeout=45_000)
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(900)
    _wait_for_streamlit_idle(page)
    interaction_blockers = _apply_interactions(page, route_config)
    _wait_for_streamlit_idle(page)
    result = _wait_for_guard_contract(page, route_config)
    if interaction_blockers:
        result["guardBlockers"] = [*result.get("guardBlockers", []), *interaction_blockers]
    result.update(
        {
            "route": route,
            "viewport": viewport_name,
            "width": width,
            "height": height,
            "consoleErrors": errors,
        }
    )
    if screenshots:
        shot = out_dir / f"{viewport_name}_{slug(route)}.png"
        page.screenshot(path=str(shot), full_page=True)
        result["screenshot"] = str(shot)
    return result


def _signature(item: dict[str, Any]) -> dict[str, Any]:
    selectors: dict[str, Any] = {}
    for selector, data in sorted(item.get("computed", {}).items()):
        if not data.get("exists") or not data.get("visible"):
            selectors[selector] = {"visible": False}
            continue
        styles = data.get("styles", {})
        selectors[selector] = {
            "exists": True,
            "visible": bool(data.get("visible")),
            "count": data.get("count", 0),
            "styles": {key: styles.get(key, "") for key in STABLE_COMPARE_STYLE_KEYS},
        }
    key_counts = item.get("keyCounts", {})
    return {
        "rootOverflowX": item.get("root", {}).get("overflowX"),
        "required": item.get("required", []),
        "stateRequired": item.get("stateRequired", []),
        "expectedHidden": item.get("expectedHidden", []),
        "keyCounts": {
            key: key_counts.get(key)
            for key in ("tables", "rails", "topbar")
            if key in key_counts
        },
        "selectors": selectors,
    }


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_items = {
        (item["route"], item["viewport"]): _signature(item)
        for item in before.get("results", [])
    }
    failures: list[str] = []
    for item in after.get("results", []):
        key = (item["route"], item["viewport"])
        previous = before_items.get(key)
        if previous is None:
            failures.append(f"{key}: missing from before report")
            continue
        current = _signature(item)
        if current != previous:
            failures.append(f"{key}: computed-style signature changed")
    return failures


def validate(report: dict[str, Any], strict_offscreen: bool) -> list[str]:
    failures: list[str] = []
    for item in report["results"]:
        label = f"{item['viewport']} {item['route']}"
        if item.get("consoleErrors"):
            failures.append(f"{label}: console errors: {item['consoleErrors']}")
        if item.get("root", {}).get("overflowX", 0) > 1:
            failures.append(f"{label}: horizontal overflow {item['root']['overflowX']}px")
        missing = [group for group in item.get("required", []) if not group.get("visible")]
        if missing:
            failures.append(f"{label}: missing required visible selectors: {missing}")
        if item.get("guardBlockers"):
            failures.append(f"{label}: guard blockers: {item['guardBlockers']}")
        if strict_offscreen and (item.get("offscreenCount", 0) or item.get("clippedCount", 0)):
            failures.append(
                f"{label}: offscreen={item.get('offscreenCount')} clipped={item.get('clippedCount')}"
            )
    return failures


def main() -> int:
    args = parse_args()
    base_url = normalize_base(args.base_url)
    assert_streamlit_ready(base_url)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"stage23_css_guard_{stamp}_{slug(args.label)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            for viewport_name, width, height in VIEWPORTS:
                for route in args.routes:
                    context = browser.new_context(viewport={"width": width, "height": height})
                    page = context.new_page()
                    try:
                        results.append(
                            collect_one(
                                page,
                                base_url=base_url,
                                route=route,
                                viewport_name=viewport_name,
                                width=width,
                                height=height,
                                out_dir=out_dir,
                                screenshots=not args.no_screenshots,
                            )
                        )
                    finally:
                        context.close()
        finally:
            browser.close()

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": base_url,
        "label": args.label,
        "routes": args.routes,
        "viewports": VIEWPORTS,
        "results": results,
    }

    failures = validate(report, strict_offscreen=args.strict_offscreen)
    if args.compare_before:
        before = json.loads(Path(args.compare_before).read_text(encoding="utf-8"))
        failures.extend(compare_reports(before, report))
        report["compare_before"] = args.compare_before

    report["failures"] = failures
    report_path = out_dir / "computed_style_guard.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "failures": failures}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
