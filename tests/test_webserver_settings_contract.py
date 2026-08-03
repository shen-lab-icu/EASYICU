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


def test_shell_renders_stage_status_without_reaching_into_study_context() -> None:
    """The stage indicator must not break the shell/owner boundary.

    app.js is forbidden from reading EU_STUDY_CONTEXT (locked by
    test_route_handoffs_have_sources_and_viz_mapping_has_its_own_owner), so the
    derivation lives in study-progress.js and the shell renders its snapshot.
    """
    shell = (STATIC_JS / "app.js").read_text(encoding="utf-8")
    owner = (STATIC_JS / "study-progress.js").read_text(encoding="utf-8")
    index = (STATIC_JS.parent / "index.html").read_text(encoding="utf-8")

    assert "EU_STUDY_CONTEXT" not in shell
    assert "window.EU_STUDY_PROGRESS.snapshot()" in shell
    assert "stage-chip" in shell
    assert "window.EU_STUDY_CONTEXT" in owner
    assert "window.EU_STUDY_PROGRESS = { snapshot }" in owner
    # Load order: the owner reads the context store, so it must come after it.
    assert index.index("js/study-context.js") < index.index("js/study-progress.js")
    assert index.index("js/study-progress.js") < index.index("js/app.js")


def test_stage_status_is_evidence_based_not_optimistic() -> None:
    """A stage may only read 'done' when the context carries its artifact."""
    owner = (STATIC_JS / "study-progress.js").read_text(encoding="utf-8")
    # Cross-DB is plan-only scope; it must not light the data stage as done.
    assert "crossdb_review_completed" in owner
    assert "function planOnly(context)" in owner
    # With no context at all, nothing is 'active' — a first-time user must not
    # be told they are mid-way through a study they never started.
    assert "const started = !!context" in owner


def test_data_mode_switch_is_weighted_when_downstream_work_exists() -> None:
    """Demo <-> Real swaps the data source; it is not a display preference."""
    shell = (STATIC_JS / "app.js").read_text(encoding="utf-8")
    shell_css = (STATIC_JS.parent / "css" / "shell.css").read_text(encoding="utf-8")
    assert "const consequential = !!window.EU_HASWORK;" in shell
    assert "${consequential ? 'consequential' : ''}" in shell
    assert ".mode-seg.consequential" in shell_css
    # Reduce-motion users still get the colour, just not the glow.
    assert 'body[data-reduce-motion="true"] .mode-seg.consequential' in shell_css


def test_shell_states_the_two_entry_paths_are_one_pipeline() -> None:
    shell = (STATIC_JS / "app.js").read_text(encoding="utf-8")
    assert "All three feed one pipeline" in shell
    assert "三者进入同一条流水线" in shell
