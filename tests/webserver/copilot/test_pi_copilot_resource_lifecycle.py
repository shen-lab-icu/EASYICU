from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.resource_lifecycle import (
    WebMemoryAdmission,
    WebMemoryPolicy,
)
from easyicu.webserver.pi_copilot.session_storage import SessionStorageMaintenance


REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_LIFECYCLE = (
    REPO_ROOT
    / "src/easyicu/webserver/pi_copilot/node_app/src/session-lifecycle.mjs"
)


def test_web_memory_policy_is_host_scaled_and_emergency_is_fail_closed() -> None:
    policy = WebMemoryPolicy.from_environment({}, total_memory_mb=16 * 1024)
    assert policy.soft_rss_mb == 2457
    assert policy.emergency_rss_mb == 4096

    admission = WebMemoryAdmission(policy)
    assert admission.status(rss_mb=2457)["pressure"] == "soft"
    with pytest.raises(PiCopilotError, match="memory pressure") as raised:
        admission.require_capacity(rss_mb=4096)
    assert raised.value.code == "pi_web_memory_pressure"
    assert raised.value.status_code == 429


def test_web_memory_policy_accepts_only_bounded_overrides() -> None:
    policy = WebMemoryPolicy.from_environment(
        {
            "EASYICU_WEB_SOFT_RSS_MB": "900",
            "EASYICU_WEB_EMERGENCY_RSS_MB": "901",
        },
        total_memory_mb=16 * 1024,
    )
    assert policy.soft_rss_mb == 900
    assert policy.emergency_rss_mb == 1028

    default_emergency = WebMemoryPolicy.from_environment(
        {"EASYICU_WEB_SOFT_RSS_MB": "5000"},
        total_memory_mb=16 * 1024,
    )
    assert default_emergency.emergency_rss_mb == 5128


def test_node_hot_session_lifecycle_unloads_idle_lru_and_rejects_emergency() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = f"""
      import {{ HotSessionLifecycle, sessionLifecycleConfig }} from {json.dumps(NODE_LIFECYCLE.as_uri())};
      const disposed = [];
      const sessions = new Map();
      const record = (id, last) => ({{
        externalId: id,
        lastAccessedAt: last,
        session: {{ isStreaming: false, dispose: () => disposed.push(id) }},
        unsubscribe: () => {{}},
      }});
      sessions.set('old', record('old', 0));
      sessions.set('middle', record('middle', 500));
      sessions.set('new', record('new', 1500));
      const active = new Map([['middle', 'request-1']]);
      let rss = 800;
      const lifecycle = new HotSessionLifecycle({{
        sessions,
        activeRequests: active,
        config: {{ maxOpenSessions: 2, idleMs: 1000, softRssBytes: 700, emergencyRssBytes: 1000 }},
        now: () => 2000,
        rssBytes: () => rss,
      }});
      lifecycle.suspendIdle({{ excludeSessionId: 'new' }});
      rss = 600;
      lifecycle.enforceHotLimit({{ excludeSessionId: 'new' }});
      console.log(JSON.stringify({{ keys: [...sessions.keys()], disposed, status: lifecycle.status() }}));

      const blockedSessions = new Map([['busy', {{
        externalId: 'busy', lastAccessedAt: 0,
        session: {{ isStreaming: true, dispose: () => {{}} }},
      }}]]);
      const blocked = new HotSessionLifecycle({{
        sessions: blockedSessions,
        activeRequests: new Map([['busy', 'request-2']]),
        config: {{ maxOpenSessions: 8, idleMs: 1000, softRssBytes: 700, emergencyRssBytes: 900 }},
        now: () => 2000,
        rssBytes: () => 950,
      }});
      try {{ blocked.admit({{ excludeSessionId: 'busy' }}); }}
      catch (error) {{ console.log(error.code); }}
      console.log(JSON.stringify({{ activeDisposed: blocked.dispose('busy') }}));

      const full = new HotSessionLifecycle({{
        sessions: blockedSessions,
        activeRequests: new Map([['busy', 'request-3']]),
        config: {{ maxOpenSessions: 1, idleMs: 1000, softRssBytes: 2000, emergencyRssBytes: 3000 }},
        now: () => 2000,
        rssBytes: () => 100,
      }});
      try {{ full.admit({{ incoming: 1 }}); }}
      catch (error) {{ console.log(error.code); }}

      console.log(JSON.stringify(sessionLifecycleConfig({{}}, {{ totalMemoryBytes: 16 * 1024 * 1024 * 1024 }})));
      console.log(JSON.stringify(sessionLifecycleConfig(
        {{ EASYICU_PI_SOFT_RSS_MB: '2500' }},
        {{ totalMemoryBytes: 16 * 1024 * 1024 * 1024 }},
      )));
    """
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    first, error_code, protected, capacity, config, overridden = (
        completed.stdout.splitlines()
    )
    payload = json.loads(first)
    assert payload["keys"] == ["middle", "new"]
    assert payload["disposed"] == ["old"]
    assert payload["status"]["hot_sessions"] == 2
    assert error_code == "pi_shell_memory_pressure"
    assert json.loads(protected) == {"activeDisposed": False}
    assert capacity == "pi_shell_session_capacity"
    assert json.loads(config) == {
        "maxOpenSessions": 8,
        "idleMs": 1_800_000,
        "softRssBytes": 1_717_567_488,
        "emergencyRssBytes": 2_576_351_232,
    }
    assert json.loads(overridden)["emergencyRssBytes"] == 2_688_548_864


def test_transcript_quarantine_is_recoverable_and_preserves_references(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    referenced = session_dir / "referenced.jsonl"
    old_orphan = session_dir / "old.jsonl"
    young_orphan = session_dir / "young.jsonl"
    referenced.write_text("referenced\n", encoding="utf-8")
    old_orphan.write_text("old\n", encoding="utf-8")
    young_orphan.write_text("young\n", encoding="utf-8")
    os.utime(referenced, (100, 100))
    os.utime(old_orphan, (100, 100))
    os.utime(young_orphan, (950, 950))
    maintenance = SessionStorageMaintenance(session_dir, grace_seconds=100)

    inventory = maintenance.inventory([referenced], now=1000)
    assert inventory.public_projection() == {
        "session_files": 3,
        "referenced_files": 1,
        "unreferenced_files": 2,
        "eligible_files": 1,
        "total_bytes": 21,
        "unreferenced_bytes": 10,
        "eligible_bytes": 4,
    }
    with pytest.raises(PiCopilotError, match="confirmation"):
        maintenance.quarantine([referenced], confirm=False, now=1000)

    moved = maintenance.quarantine([referenced], confirm=True, now=1000)
    assert moved["moved_files"] == 1
    assert referenced.exists()
    assert young_orphan.exists()
    assert not old_orphan.exists()
    quarantine_id = str(moved["quarantine_id"])
    manifest = json.loads(
        (
            session_dir / "quarantine" / quarantine_id / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["files"][0]["file"] == "old.jsonl"

    restored = maintenance.restore(quarantine_id, confirm=True)
    assert restored["restored_files"] == 1
    assert old_orphan.read_text(encoding="utf-8") == "old\n"


def test_transcript_inventory_ignores_symlinks(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("private\n", encoding="utf-8")
    (session_dir / "linked.jsonl").symlink_to(outside)

    inventory = SessionStorageMaintenance(session_dir).inventory([], now=10_000_000)

    assert inventory.session_files == 0
    assert outside.exists()


def test_transcript_quarantine_rolls_back_if_manifest_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    orphan = session_dir / "orphan.jsonl"
    orphan.write_text("recoverable\n", encoding="utf-8")
    os.utime(orphan, (1, 1))
    maintenance = SessionStorageMaintenance(session_dir, grace_seconds=60)

    def fail_manifest(*_args: object, **_kwargs: object) -> None:
        raise OSError("manifest unavailable")

    monkeypatch.setattr(maintenance, "_write_manifest", fail_manifest)

    with pytest.raises(PiCopilotError) as raised:
        maintenance.quarantine([], confirm=True, now=1000)

    assert raised.value.code == "pi_session_quarantine_io_error"
    assert orphan.read_text(encoding="utf-8") == "recoverable\n"
