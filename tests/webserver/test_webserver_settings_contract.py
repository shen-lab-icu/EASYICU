"""Contract tests for the native WebApp settings surface and app shell.

Every property below was violated before these tests existed:

1. Every key the API accepts has a reader. Ten keys were coerced, persisted
   and returned by ``/api/settings`` while nothing in the product ever read
   the stored value — the write returned 200 and changed nothing.
2. A patch the store cannot apply fails loudly. Unknown, retired and invalid
   keys were skipped and still answered 200, so a client could not tell a
   stored value from a discarded one.
3. Concurrent processes do not erase each other's writes. The store held a
   ``threading.RLock``, which orders writers inside one uvicorn and not
   between two of them.
4. ``body[data-density]`` has exactly one writer. ``i18n.js`` wrote it from
   ``/api/settings`` and ``tweaks.js`` wrote it from localStorage, so the
   winner was decided by whichever finished last.
5. The screen renders backend state rather than asserting it. The Privacy
   panel printed "enforced" as a literal, and ``dual()`` concatenated both
   scripts on a page that offers a language setting.
6. The shell reports study progress without reading the study-context store
   itself, and only from evidence the context actually carries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app

SRC_ROOT = Path(settings_store.__file__).resolve().parents[1]
STATIC_JS = SRC_ROOT / "webserver" / "static" / "js"
SETTINGS_MODULE = Path(settings_store.__file__).resolve()


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        settings_store, "_CONFIG_PATH", tmp_path / "cfg" / "settings.json"
    )


def _python_sources() -> list[Path]:
    return [
        path
        for path in SRC_ROOT.rglob("*.py")
        if path.resolve() != SETTINGS_MODULE and "__pycache__" not in path.parts
    ]


def test_every_accepted_setting_has_a_reader_outside_the_store() -> None:
    """A setting nothing reads is an API that lies about its own writes."""
    sources = {path: path.read_text(encoding="utf-8") for path in _python_sources()}
    js_sources = {
        path: path.read_text(encoding="utf-8") for path in STATIC_JS.glob("*.js")
    }

    unread: dict[str, None] = {}
    for key in settings_store.DEFAULTS:
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        has_python_reader = any(pattern.search(text) for text in sources.values())
        has_js_reader = any(pattern.search(text) for text in js_sources.values())
        if not (has_python_reader or has_js_reader):
            unread[key] = None

    assert not unread, (
        "these settings are accepted, coerced and persisted but never read: "
        f"{sorted(unread)}. Give each one a consumer or move it to RETIRED_KEYS."
    )


def test_retired_keys_are_gone_from_defaults_and_coercion() -> None:
    overlap = set(settings_store.RETIRED_KEYS) & set(settings_store.DEFAULTS)
    assert not overlap, f"retired keys still accepted: {sorted(overlap)}"
    assert not set(settings_store.RETIRED_KEYS) & set(settings_store._COERCE)


def test_retired_key_is_rejected_with_its_reason(isolated_settings: None) -> None:
    client = TestClient(app)
    response = client.post("/api/settings", json={"token_budget": 42000})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "settings_patch_rejected"
    rejected = detail["rejected"]
    assert [item["key"] for item in rejected] == ["token_budget"]
    assert rejected[0]["code"] == "retired_setting"
    # The reason must name what replaced it, not just say "no".
    assert "provider adapter" in rejected[0]["reason"]


def test_unknown_key_is_rejected_rather_than_silently_dropped(
    isolated_settings: None,
) -> None:
    client = TestClient(app)
    response = client.post("/api/settings", json={"ai_enabledd": True})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["rejected"][0]["code"] == "unknown_setting"


def test_rejected_patch_writes_nothing(isolated_settings: None) -> None:
    """A partial apply would leave the client's view and the file disagreeing."""
    client = TestClient(app)
    client.post("/api/settings", json={"density": "compact"})

    response = client.post(
        "/api/settings",
        json={"reduce_motion": True, "density": "microscopic"},
    )

    assert response.status_code == 400
    after = client.get("/api/settings").json()
    assert after["density"] == "compact"
    assert after["reduce_motion"] is False


def test_ai_optin_gate_still_fails_closed_on_a_non_boolean(
    isolated_settings: None,
) -> None:
    client = TestClient(app)
    response = client.post("/api/settings", json={"ai_enabled": "maybe"})

    assert response.status_code == 400
    assert client.get("/api/settings").json()["ai_enabled"] is False


def test_about_reports_live_host_policy_not_a_literal(
    isolated_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Privacy panel must not be able to show a guarantee that is off."""
    client = TestClient(app)
    strict = client.get("/api/settings").json()["about"]["local_access"]
    assert strict["enforced"] is True
    assert strict["proxy_headers_rejected"] is True

    monkeypatch.setenv("EASYICU_WEB_TRUST_PROXY", "1")
    widened = client.get("/api/settings").json()["about"]["local_access"]
    assert widened["enforced"] is False
    assert widened["proxy_headers_trusted"] is True


def test_stale_settings_file_with_retired_keys_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older build's file must not strand the user on startup."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "settings.json"
    config_path.write_text(
        '{"ai_enabled": true, "token_budget": 999, "density": "compact"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", config_path)

    loaded = settings_store.load_settings()

    assert loaded["ai_enabled"] is True
    assert loaded["density"] == "compact"
    assert "token_budget" not in loaded


def _concurrent_patch(config_dir: str, key: str, value: object) -> None:
    """Run in a *separate process*; a threading.RLock cannot order these."""
    from pathlib import Path as _Path

    from easyicu.webserver import settings as store

    store._CONFIG_DIR = _Path(config_dir)
    store._CONFIG_PATH = _Path(config_dir) / "settings.json"
    for _ in range(12):
        store.update_settings({key: value})


def test_concurrent_processes_do_not_erase_each_others_writes(
    tmp_path: Path,
) -> None:
    """Two servers on one machine must not clobber each other's settings.

    ``easyicu-webapp --background`` plus a foreground run is enough to have two
    processes read the same file, apply different patches, and have the second
    write erase the first. A threading lock does not order them.
    """
    import multiprocessing as mp

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text("{}", encoding="utf-8")

    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(target=_concurrent_patch, args=(str(config_dir), key, value))
        for key, value in (("ai_enabled", True), ("reduce_motion", True))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)

    assert all(worker.exitcode == 0 for worker in workers)
    final = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    # Without a cross-process lock, the last writer's read-modify-write drops
    # whichever key the other process had just added.
    assert final["ai_enabled"] is True
    assert final["reduce_motion"] is True
    assert not list(config_dir.glob("*.lock")), "lock file was not released"


def test_density_attribute_has_exactly_one_writer() -> None:
    """Two owners writing one attribute resolve by network latency, not intent."""
    writers = []
    for path in STATIC_JS.glob("*.js"):
        text = path.read_text(encoding="utf-8")
        if re.search(
            r"""dataset\.density\s*=|setAttribute\(\s*['"]data-density['"]""", text
        ):
            writers.append(path.name)

    assert writers == ["i18n.js"], (
        "body[data-density] must be written only by i18n.js::applyDisplayDom, "
        f"which reads the persisted setting. Found writers: {sorted(writers)}"
    )


def test_tweaks_panel_does_not_own_a_persisted_setting() -> None:
    """Density lived in tweaks.js in three places; none may come back.

    Asserted as code forms rather than the bare word, so the comment that
    explains why density is absent does not itself fail the test.
    """
    tweaks = (STATIC_JS / "tweaks.js").read_text(encoding="utf-8")
    assert "density:" not in tweaks  # DEFAULTS entry
    assert "values.density" not in tweaks  # apply() write
    assert "seg('density'" not in tweaks  # panel control
    # The appearance tokens it does own stay owned by it.
    assert "accent:" in tweaks and "radius:" in tweaks


def test_settings_screen_draws_no_control_for_a_retired_key() -> None:
    settings_js = (STATIC_JS / "screens-settings.js").read_text(encoding="utf-8")
    for key in settings_store.RETIRED_KEYS:
        assert f'data-setting="{key}"' not in settings_js
        assert f'data-setting-input="{key}"' not in settings_js
        assert f"pathCtl('{key}'" not in settings_js
        assert f"segBound('{key}'" not in settings_js


def test_capability_block_honours_the_language_setting() -> None:
    """dual() concatenated both scripts, ignoring the language on its own page."""
    settings_js = (STATIC_JS / "screens-settings.js").read_text(encoding="utf-8")
    assert "function dual(en, zh) {\n    return T(en, zh);\n  }" in settings_js
    assert "return `${en} / ${zh}`;" not in settings_js


def test_bilingual_helper_is_never_handed_two_different_values() -> None:
    """dual(en, zh) takes a translation pair, not (status, reason).

    Two call sites passed backend ``status`` and ``reason`` as if they were a
    translation pair. That was invisible while dual() concatenated both, and
    surfaced the raw code ``remote_compute_enabled_false`` on the Chinese
    screen the moment it started picking one.
    """
    settings_js = (STATIC_JS / "screens-settings.js").read_text(encoding="utf-8")
    assert "dual(String(" not in settings_js
    assert "dual(Number(" not in settings_js


def test_no_raw_backend_reason_code_is_rendered() -> None:
    settings_js = (STATIC_JS / "screens-settings.js").read_text(encoding="utf-8")
    # Reason codes may be mapped to sentences; they may not be interpolated raw.
    assert "remote.reason" not in settings_js


def test_shell_does_not_read_the_study_context() -> None:
    """app.js stays a router: the stage status it once drew in the old sidebar
    is retired with that sidebar, and the shell still never reads the store."""
    shell = (STATIC_JS / "app.js").read_text(encoding="utf-8")
    index = (STATIC_JS.parent / "index.html").read_text(encoding="utf-8")

    assert "EU_STUDY_CONTEXT" not in shell
    assert "EU_STUDY_PROGRESS" not in shell
    assert "js/study-progress.js" not in index
    assert not (STATIC_JS / "study-progress.js").exists()


def test_settings_offer_no_data_mode_switch() -> None:
    """There is one source-backed path; a public demo is a labelled source,
    not a global mode a setting switches."""
    shell = (STATIC_JS / "app.js").read_text(encoding="utf-8")
    settings = (STATIC_JS / "screens-settings.js").read_text(encoding="utf-8")
    shell_css = (STATIC_JS.parent / "css" / "shell.css").read_text(encoding="utf-8")
    assert "mode-seg" not in shell
    assert ".mode-seg" not in shell_css
    assert "set-data-mode" not in settings
    assert "data_mode" not in settings


def test_execution_gate_ships_codes_beside_the_sentences() -> None:
    from easyicu.webserver.ideas.mining import _execution_gate

    blocked = _execution_gate({"go_no_go": "hold"}, {"status": "blocked"}, None)
    assert blocked["blocker_codes"] == [
        "export_not_real",
        "prior_art_not_reviewed",
        "idea_not_recommended",
        "prior_art_not_differentiated",
        "source_feasibility_not_ready",
        "study_definition_not_confirmed",
    ]
    # The sentences stay for seeds already on disk, and must agree with codes.
    assert len(blocked["blockers"]) == len(blocked["blocker_codes"])
    assert blocked["agent_run_ready_after_human_confirmation"] is False

    clear = _execution_gate(
        {"go_no_go": "recommend"},
        {"status": "ready"},
        {"prior_art": {"search_performed": True, "status": "ok"}},
        {
            "prior_art_decision": "differentiated",
            "source_feasibility_status": "ready",
            "idea_definition_sha256": "a" * 64,
        },
    )
    assert clear["blocker_codes"] == []
    assert clear["agent_run_ready_after_human_confirmation"] is True


def test_legacy_seed_sentences_recover_their_code_by_exact_match() -> None:
    """Old seeds carry prose only; recover by lookup, never by pattern."""
    from easyicu.webserver.routes.agent import _blocker_codes

    assert _blocker_codes({}, ["run prior-art review before Agent execution"]) == [
        "prior_art_not_reviewed"
    ]
    # A sentence nobody recognises stays unrecognised.
    assert _blocker_codes({}, ["something a future build invented"]) == []
    # The gate's own codes win over the sentence lookup.
    assert _blocker_codes({"blocker_codes": ["export_not_real"]}, ["anything"]) == [
        "export_not_real"
    ]


def test_crossdb_states_its_scope_limit_before_the_agent_refuses() -> None:
    crossdb = (STATIC_JS / "screens-viz-crossdb-results.js").read_text(encoding="utf-8")
    assert "A cross-DB plan stops at the plan" in crossdb
    assert "跨库计划止于计划本身" in crossdb




def test_missing_job_branch_is_reachable_executable_contract() -> None:
    """Run the Node contract that reproduces api.js's real error shape.

    Asserting on source text alone would not have caught this: the old code
    *looked* like it detected a 404. Only building the error the way api.js
    builds it shows the branch was dead.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable job-continuity contract")

    root = SRC_ROOT.parents[1]
    test_file = root / "tests" / "js" / "job_continuity_404.test.js"
    result = subprocess.run(
        [node, str(test_file)], check=True, capture_output=True, text=True
    )
    assert "keys the missing-job branch on error.status" in result.stdout


def test_transitional_css_buckets_hold_no_route_rules() -> None:
    """A catch-all may only shrink. redesign.css held 30 agent/idea rules.

    Rules moved into agent.css were *prepended*: redesign.css loaded before it,
    and both files define .ag-list / .ag-pipe / .ag-tab / .ag-body, so
    appending would have flipped which declaration wins.
    """
    import re as _re

    css_dir = STATIC_JS.parent / "css"
    route_rule = _re.compile(
        r"^\s*\.(ideas|agent|crossdb|xdb|patient|cohort|guided|extraction|ex-|pt-|ag-|idea-)",
        _re.M,
    )
    for bucket in ("redesign.css", "screens.css", "app.css", "pages.css", "tweaks.css"):
        text = (css_dir / bucket).read_text(encoding="utf-8")
        offenders = route_rule.findall(text)
        assert not offenders, f"{bucket} regained route-specific rules: {offenders[:5]}"

    # The moved rules that still have a renderer live at their owner.
    agent_css = (css_dir / "agent.css").read_text(encoding="utf-8")
    for selector in (".ag-pipe", ".ag-tabs"):
        assert selector in agent_css, f"{selector} lost in the migration"
    # The retired sidebar's entry cards left the shell owner with it.
    shell_css = (css_dir / "shell.css").read_text(encoding="utf-8")
    assert ".agent-entry" not in shell_css and ".ideas-entry" not in shell_css


def test_picker_css_has_one_owner_and_is_not_injected_from_js() -> None:
    """Three screens each carried their own copy of the picker's CSS.

    ensurePickerStyles / ensureSourcePickerStyles / ensureSettingPickerStyles
    injected ~2 KB of identical rules from JavaScript. Two copies were still
    byte-identical; the third had already gained a rule the others never got.
    """
    js_files = list(STATIC_JS.glob("*.js"))
    css_dir = STATIC_JS.parent / "css"

    for path in js_files:
        text = path.read_text(encoding="utf-8")
        assert ".eu-pick-back{position:fixed" not in text, (
            f"{path.name} injects picker CSS from JavaScript again"
        )
        for gone in (
            "ensurePickerStyles",
            "ensureSourcePickerStyles",
            "ensureSettingPickerStyles",
        ):
            assert gone not in text, f"{path.name} still defines/calls {gone}"

    picker_css = (css_dir / "picker.css").read_text(encoding="utf-8")
    for selector in (
        ".eu-pick-back",
        ".eu-pick{",
        ".eu-pick-h",
        ".eu-pick-cur",
        ".eu-pick-sc",
        ".eu-pick-list",
        ".eu-pick-row",
        ".eu-pick-f",
        ".eu-pick-empty",
    ):
        assert selector in picker_css, f"{selector} lost when the copies were merged"

    # The rule that was never a picker rule went back to its own route owner.
    assert ".setting-input" not in picker_css
    assert ".setting-input" in (css_dir / "settings.css").read_text(encoding="utf-8")

    # A JS-injected <style> lands after every <link>; keep that precedence.
    index = (STATIC_JS.parent / "index.html").read_text(encoding="utf-8")
    links = re.findall(r'<link rel="stylesheet" href="(css/[^"?]+)', index)
    assert links[-1] == "css/picker.css", (
        f"picker.css must be the last stylesheet, found {links[-1]}"
    )
