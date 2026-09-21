"""Preview, messaging, and frontend security contracts for Guided Pi Copilot."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    NODE_APP as NODE_APP,
    STATIC as STATIC,
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


def test_workspace_preview_never_requests_an_empty_checked_digest() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-preview.js")
    resources = _read("js/screens-guided-pi-resources.js")
    digest = "c" * 64
    script = f"""
      const calls = [];
      global.window = {{
        EU_LANG: 'en',
        EU_API: {{
          piCopilotWorkspacePreviewUrl(projectId, file, checkedSha256) {{
            calls.push([projectId, file, checkedSha256]);
            return '/preview?checked_sha256=' + checkedSha256;
          }},
        }},
      }};
      global.document = {{ getElementById() {{ return null; }} }};
      eval({_ESCAPE_OWNER!r});
      eval({resources!r});
      eval({source!r});
      const host = {{
        hidden: false,
        innerHTML: '',
        addEventListener() {{}},
        replaceChildren() {{ this.innerHTML = ''; }},
      }};
      window.EU_GUIDED_PI_PREVIEW.mount(host);
      window.EU_GUIDED_PI_PREVIEW.open({{
        kind: 'file', file: 'prototype/index.html', media_type: 'text/html',
      }}, 'project-demo');
      console.log(String(host.innerHTML.includes('data-gpi-preview-mode="web"')));
      console.log(String(calls.length));
      window.EU_GUIDED_PI_PREVIEW.open({{
        kind: 'webpage', file: 'prototype/index.html', media_type: 'text/html',
      }}, 'project-demo');
      console.log(String(calls.length));
      console.log(String(host.innerHTML.includes('<iframe')));
      console.log(String(host.innerHTML.includes('checked file digest is missing')));
      window.EU_GUIDED_PI_PREVIEW.open({{
        kind: 'webpage', file: 'prototype/index.html', media_type: 'text/html',
        checked_sha256: '{digest}',
      }}, 'project-demo');
      console.log(String(calls.length));
      console.log(String(host.innerHTML.includes('{digest}')));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "false",
        "0",
        "0",
        "false",
        "true",
        "1",
        "true",
    ]


def test_preview_keeps_bounded_project_scoped_recent_resources() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    preview = _read("js/screens-guided-pi-preview.js")
    resources = _read("js/screens-guided-pi-resources.js")
    digest_a = "a" * 64
    digest_b = "b" * 64
    script = f"""
      global.window = {{ EU_LANG: 'en', EU_API: {{}} }};
      global.document = {{ getElementById() {{ return null; }} }};
      eval({_ESCAPE_OWNER!r});
      eval({resources!r});
      eval({preview!r});
      const host = {{
        hidden: false,
        innerHTML: '',
        addEventListener() {{}},
        replaceChildren() {{ this.innerHTML = ''; }},
      }};
      window.EU_GUIDED_PI_PREVIEW.mount(host);
      const first = {{
        kind: 'webpage', file: 'reports/cohort.html', label: 'Cohort review',
        media_type: 'text/html', checked_sha256: '{digest_a}',
      }};
      const second = {{
        kind: 'webpage', file: 'reports/timeline.html', label: 'Timeline review',
        media_type: 'text/html', checked_sha256: '{digest_b}',
      }};
      window.EU_GUIDED_PI_PREVIEW.open(first, 'project-a');
      console.log(String(host.innerHTML.includes('gpi-preview-recent')));
      window.EU_GUIDED_PI_PREVIEW.open(second, 'project-a');
      console.log(String(host.innerHTML.includes('Cohort review')));
      console.log(String(host.innerHTML.includes('Timeline review')));
      window.EU_GUIDED_PI_PREVIEW.close();
      window.EU_GUIDED_PI_PREVIEW.open(first, 'project-a');
      console.log(String(host.innerHTML.includes('Timeline review')));
      window.EU_GUIDED_PI_PREVIEW.open(first, 'project-b');
      console.log(String(host.innerHTML.includes('gpi-preview-recent')));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["false", "true", "true", "true", "false"]


def test_analysis_only_plan_does_not_claim_missing_unexecuted_robustness_blocks_approval() -> None:
    renderer = _read("js/screens-agent-render.js")

    assert "const analysisOnlyPlan" in renderer
    assert "!endpoint && !analysisOnlyPlan" in renderer
    assert "!robustness.length && !analysisOnlyPlan" in renderer
    assert "String(designSelection.claim_ceiling || '') === 'analysis_only'" in renderer
    assert "/time_varying_cox_estimates/" in renderer
    assert "Time-updated Cox estimates" in renderer
    assert "/exposure_outcome_distribution/" in renderer
    assert "Exposure prevalence and observed outcome" in renderer


def test_continuing_a_conversation_does_not_close_the_open_preview() -> None:
    owner = _read("js/screens-guided-pi.js")
    send_text = owner.split("async function sendText", 1)[1].split(
        "async function sendMessage", 1
    )[0]

    assert "state.currentTurnResources = []" in send_text
    assert "render();" in send_text
    assert "EU_GUIDED_PI_PREVIEW.close" not in send_text
    assert "EU_GUIDED_PI_PREVIEW.clearProject" not in send_text


def test_workspace_sidecar_requires_digest_for_edit_and_teaches_safe_egress() -> None:
    sidecar = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")
    skill = (NODE_APP / "src" / "skills" / "web-prototype" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert sidecar.count("expected_sha256") >= 1
    assert "Create a new bounded artifact. Existing files must be changed" in sidecar
    assert "To change an existing file, read it first" in skill
    assert "expected_sha256" in skill
    assert "may be sent to the\nconfigured Pi model service" in skill
    assert "PHI" in skill
    assert "llm_provider:" not in sidecar


def test_nonconvergent_plan_revision_still_blocks_approval_and_automatic_restart() -> None:
    confirmation = _read("js/screens-guided-pi-confirmation.js")
    actions = _read("js/screens-guided-pi-plan-actions.js")

    assert "code === 'agent_plan_revision_nonconvergent'" in confirmation
    assert "hideEdit: true" in confirmation
    assert "nonApprovable: true" in confirmation
    assert "actionCode === 'agent_plan_revision_nonconvergent'" not in actions


def test_product_label_calls_are_defensive_and_share_one_default() -> None:
    """D-P2-1: no bare EU_PRODUCT_LABELS call; one shared row fallback."""

    index = _read("index.html")
    assert "js/product-labels.js?v=20260917-product-label-defensive1" in index

    consumers = [
        "js/screens-guided-projects.js",
        "js/screens-agent-study-context.js",
        "js/screens-guided.js",
        "js/screens-guided-pi-preview.js",
        "js/screens-guided-pi.js",
        "js/screens-guided-pi-run-files.js",
    ]
    for name in consumers:
        source = _read(name)
        assert "EU_PRODUCT_LABELS" in source
        bare = [
            line
            for line in source.splitlines()
            if "EU_PRODUCT_LABELS.projectTitle(" in line
            or "EU_PRODUCT_LABELS.copilotTitle(" in line
        ]
        assert bare == [], f"{name} calls the label owner without ?. defense: {bare}"
        assert "?.projectTitle?." in source or "?.copilotTitle?." in source
        assert "slice(0, 200)" in source

    projects = _read("js/screens-guided-projects.js")
    assert "GUIDED_ROW_FALLBACK" in projects
    assert "t('Guided project', '研究项目')" not in projects
    assert projects.count("t(...GUIDED_ROW_FALLBACK)") == 3


def test_error_text_return_value_must_pass_through_esc() -> None:
    """D-P2-2: errorText() is raw copy; innerHTML insertions must esc() it."""

    owner = _read("js/screens-guided-pi-error-text.js")
    assert "返回值须经esc后插入innerHTML" in owner or "MUST pass the result through" in owner

    offenders = []
    for path in sorted((STATIC / "js").glob("*.js")):
        if path.name == "screens-guided-pi-error-text.js":
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "innerHTML" in line and "errorText" in line and "esc" not in line:
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], (
        "errorText() without esc() flows into innerHTML: " + ", ".join(offenders)
    )


def test_tweaks_edit_mode_uses_exact_origin_both_ways() -> None:
    """D-P2-3: postMessage targets location.origin; listener checks both."""

    tweaks = _read("js/tweaks.js")
    assert "postMessage({ type: '__edit_mode_set_keys'" in tweaks
    assert ", '*'" not in tweaks
    assert "window.location.origin" in tweaks
    assert "e.source !== window.parent" in tweaks
    assert "e.origin !== window.location.origin" in tweaks


def test_provider_preset_matches_hostnames_exactly_or_by_suffix() -> None:
    """D-P2-5: no substring preset matching; unparseable URLs are custom."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-error-text.js")
    assert ".includes('api.openai.com')" not in owner
    assert "hostnameMatches(host" in owner
    assert "new URL(" in owner
    script = r"""
global.window = { EU_HTML: { esc: (v) => String(v) }, EasyICU: { guidedPi: {} } };
global.window.EasyICU.guidedPi.declare = (n, api) => { globalThis.declared = api; };
require(process.argv[1]);
const { providerPreset } = globalThis.declared.create({ tr: (en) => en, staticPreview: () => false });
const cases = [
  [{ base_url: 'https://api.openai.com/v1' }, {}, 'openai'],
  [{ base_url: 'https://api.openai.com.evil.example/v1' }, {}, 'custom-openai'],
  [{ base_url: 'https://evil.example/?x=api.openai.com' }, {}, 'custom-openai'],
  [{ base_url: 'https://sub.openrouter.ai/api/v1' }, {}, 'openrouter'],
  [{ base_url: 'https://openrouter.ai.evil.example/' }, {}, 'custom-openai'],
  [{ base_url: 'https://api.deepseek.com/v1' }, {}, 'deepseek'],
  [{ base_url: 'http://127.0.0.1:8317/v1' }, {}, 'cliproxyapi'],
  [{ base_url: 'http://localhost:8317/v1' }, {}, 'cliproxyapi'],
  [{ base_url: 'https://127.0.0.1.evil.example:8317/' }, {}, 'custom-openai'],
  [{ base_url: 'http://127.0.0.1:9999/v1' }, {}, 'custom-openai'],
  [{ base_url: 'not a url' }, {}, 'custom-openai'],
  [{ base_url: '' }, {}, 'custom-openai'],
  [{}, { api_transport: 'anthropic-messages' }, 'anthropic'],
  [{}, { api_transport: 'google-generative-ai' }, 'google'],
];
process.stdout.write(JSON.stringify(cases.map(([c, r]) => providerPreset(c, r))));
"""
    completed = subprocess.run(
        [node, "-e", script, str(STATIC / "js" / "screens-guided-pi-error-text.js")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [
        "openai", "custom-openai", "custom-openai", "openrouter", "custom-openai",
        "deepseek", "cliproxyapi", "cliproxyapi", "custom-openai", "custom-openai",
        "custom-openai", "custom-openai", "anthropic", "google",
    ]
