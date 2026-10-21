"""Administrator tooling uses synthetic keys and mocked processes/network."""

import importlib
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from easyicu.databases.profiles import get_database_profile
from easyicu.webserver import capabilities, dataio, settings, sources
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore


@pytest.fixture
def tooling(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ])
    return importlib.import_module("tools.hosted_instances")


def create(tooling, tmp_path):
    users = tmp_path / "users"
    data = tmp_path / "shared-data"
    data.mkdir()
    assert tooling.main(["--users-root", str(users), "create", "doctor", "--display-name", "王医生",
                         "--data-root", str(data), "--runner-image", "easyicu-runner:synthetic",
                         "--support-contact", "Hospital help desk", "--public-origin", "https://clinic.example"]) == 0
    return users, users / "doctor"


def test_create_is_private_and_provisions_settings_via_owner(tooling, tmp_path):
    users, instance = create(tooling, tmp_path)
    for directory in (users, instance, instance / "home", instance / "run"):
        assert directory.stat().st_mode & 0o777 == 0o700
    env_file = instance / "instance.env"
    assert env_file.stat().st_mode & 0o777 == 0o600
    env = tooling.instance_environment(instance)
    assert env["EASYICU_HOSTED"] == "1"
    assert env["EASYICU_USER_DISPLAY_NAME"] == "王医生"
    assert env["EASYICU_DATA_ROOTS"] == str(tmp_path / "shared-data")
    assert env["EASYICU_WEB_UNIX_SOCKET"] == str(instance / "run" / "web.sock")
    assert env["EASYICU_WEB_TRUST_PROXY"] == "1"
    assert env["EASYICU_RUNNER_IMAGE"] == "easyicu-runner:synthetic"
    saved = json.loads((instance / "home" / ".easyicu" / "webserver_settings.json").read_text())
    assert saved["support_contact"] == "Hospital help desk"


@pytest.mark.parametrize("name", ["../other", "a/b", ".", "..", "bad name", "-option"])
def test_create_refuses_unsafe_names(tooling, tmp_path, name):
    try:
        code = tooling.main(["--users-root", str(tmp_path / "users"), "create", name, "--display-name", "Synthetic",
                             "--data-root", str(tmp_path), "--runner-image", "synthetic"])
    except SystemExit as exc:
        code = exc.code
    assert code != 0


def test_create_never_overwrites_an_instance(tooling, tmp_path):
    users, instance = create(tooling, tmp_path)
    before = (instance / "instance.env").read_bytes()
    assert tooling.main(["--users-root", str(users), "create", "doctor", "--display-name", "Other",
                         "--data-root", str(tmp_path), "--runner-image", "other"]) != 0
    assert (instance / "instance.env").read_bytes() == before


def test_create_refuses_a_filesystem_root_grant(tooling, tmp_path):
    assert tooling.main(["--users-root", str(tmp_path / "users"), "create", "doctor", "--display-name", "Synthetic",
                         "--data-root", "/", "--runner-image", "synthetic"]) != 0


def fake_verify(method, url, headers, body, timeout):
    assert headers["Authorization"] == "Bearer synthetic-test-key"
    if method == "GET":
        return 200, {"data": [{"id": "deepseek-chat"}]}
    assert body["max_tokens"] == 8
    return 200, {"choices": [{"message": {"content": "READY"}}]}


def test_provision_uses_existing_verifier_and_changes_capability(tooling, tmp_path, monkeypatch, capsys):
    from easyicu.webserver import sources

    _, instance = create(tooling, tmp_path)
    key = tmp_path / "key"
    key.write_text("synthetic-test-key\n")
    key.chmod(0o600)
    store = PiProviderConfigStore(config_path=instance / "home/.easyicu/pi-provider.env",
                                  receipt_path=instance / "home/.easyicu/pi-provider-verification.json")
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": []})
    original = PiProviderConfigStore.public_status
    monkeypatch.setattr(PiProviderConfigStore, "public_status", lambda self, **kw: original(store, environ={}))
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    assert TestClient(app).get("/api/capabilities").json()["deployment"]["provider_provisioned"] is False
    tooling.provision_model(instance, key, verifier=fake_verify)
    status = store.public_status()
    assert status["connection_verified"] and status["inference_verified"]
    deployment = TestClient(app).get("/api/capabilities").json()["deployment"]
    assert deployment == {"hosted": True, "provider_provisioned": True}
    output = capsys.readouterr()
    assert "synthetic-test-key" not in output.out + output.err
    saved = json.loads((instance / "home/.easyicu/webserver_settings.json").read_text())
    assert saved["ai_enabled"] is True
    assert saved["support_contact"] == "Hospital help desk"


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o666])
def test_provision_refuses_readable_key_before_read_or_network(tooling, tmp_path, mode):
    _, instance = create(tooling, tmp_path)
    key = tmp_path / "key"
    key.write_text("synthetic-test-key")
    key.chmod(mode)
    calls = []
    with pytest.raises(ValueError, match="private"):
        tooling.provision_model(instance, key, verifier=lambda *a: calls.append(a))
    assert calls == []
    assert not (instance / "home/.easyicu/pi-provider.env").exists()


def test_failed_verification_does_not_persist_or_enable_ai(tooling, tmp_path):
    _, instance = create(tooling, tmp_path)
    key = tmp_path / "key"
    key.write_text("synthetic-test-key")
    key.chmod(0o600)
    with pytest.raises(Exception, match="credential"):
        tooling.provision_model(instance, key, verifier=lambda *a: (401, {}))
    assert not (instance / "home/.easyicu/pi-provider.env").exists()
    settings = json.loads((instance / "home/.easyicu/webserver_settings.json").read_text())
    assert settings["ai_enabled"] is False


def test_start_uses_shared_checkout_image_and_socket_and_records_identity(tooling, tmp_path, monkeypatch):
    users, instance = create(tooling, tmp_path)
    calls = []
    monkeypatch.setattr(tooling.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(pid=12345, poll=lambda: None))
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 42.0))
    assert tooling.main(["--users-root", str(users), "start", "doctor"]) == 0
    cmd, kwargs = calls[0]
    assert "--uds" in cmd and "--host" not in cmd and "--port" not in cmd
    assert cmd[cmd.index("--uds") + 1] == str(instance / "run/web.sock")
    assert "--proxy-headers" in cmd
    assert kwargs["env"]["EASYICU_RUNNER_IMAGE"] == "easyicu-runner:synthetic"
    assert Path(kwargs["cwd"]) == Path(__file__).resolve().parents[2]
    assert "EASYICU_PI_API_KEY" not in kwargs["env"]
    record = json.loads((instance / "run/instance.pid.json").read_text())
    assert record["pid"] == 12345 and record["create_time"] == 42.0


def test_started_process_creates_private_files_and_directories(tooling, tmp_path, monkeypatch):
    users, instance = create(tooling, tmp_path)
    original_popen = tooling.subprocess.Popen
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 42.0))

    def start_writer(command, **kwargs):
        script = (
            "import os; from pathlib import Path; home = Path(os.environ['EASYICU_HOME']); "
            "(home / 'project').mkdir(); "
            "(home / '.easyicu/webserver_guided_drafts.json').write_text('{}'); "
            "(home / '.easyicu/webserver_sources.json').write_text('{}')"
        )
        process = original_popen([sys.executable, "-c", script], **kwargs)
        assert process.wait(timeout=10) == 0
        return SimpleNamespace(pid=12345, poll=lambda: None)

    monkeypatch.setattr(tooling.subprocess, "Popen", start_writer)
    previous = os.umask(0o002)
    try:
        assert tooling.main(["--users-root", str(users), "start", "doctor"]) == 0
        assert os.umask(0o002) == 0o002
    finally:
        os.umask(previous)
    assert (instance / "home/project").stat().st_mode & 0o777 == 0o700
    for name in ("webserver_guided_drafts.json", "webserver_sources.json"):
        assert (instance / "home/.easyicu" / name).stat().st_mode & 0o777 == 0o600


def test_start_repairs_legacy_home_without_following_symlinks_or_touching_data(tooling, tmp_path, monkeypatch):
    users, instance = create(tooling, tmp_path)
    home = instance / "home"
    project = home / "projects/legacy"
    project.mkdir(parents=True)
    files = [home / ".easyicu/webserver_guided_drafts.json", home / ".easyicu/webserver_sources.json",
             project / "notes.txt"]
    for path in files:
        path.write_text("{}")
        path.chmod(0o664)
    directories = [home, home / ".easyicu", home / "projects", project]
    for path in directories:
        path.chmod(0o775)
    data = tmp_path / "shared-data"
    data.chmod(0o775)
    outside = data / "outside.txt"
    outside.write_text("synthetic")
    outside.chmod(0o664)
    (home / "file-link").symlink_to(outside)
    (home / "directory-link").symlink_to(data, target_is_directory=True)
    (home / "broken-link").symlink_to(data / "missing")

    def start(command, **kwargs):
        assert all(path.stat().st_mode & 0o777 == 0o700 for path in directories)
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)
        return SimpleNamespace(pid=12345, poll=lambda: None)

    monkeypatch.setattr(tooling.subprocess, "Popen", start)
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 42.0))
    assert tooling.main(["--users-root", str(users), "start", "doctor"]) == 0
    assert data.stat().st_mode & 0o777 == 0o775
    assert outside.stat().st_mode & 0o777 == 0o664
    assert all((home / name).is_symlink() for name in ("file-link", "directory-link", "broken-link"))


@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("newline", [b"\r\n", b"\n", b"\r", b""])
def test_set_public_origin_preserves_other_bytes_and_private_atomic_write(
    tooling, tmp_path, monkeypatch, capsys, clear, newline,
):
    from easyicu.hosted_gateway.storage import instance_socket

    users, instance = create(tooling, tmp_path)
    path = instance / "instance.env"
    old = b'EASYICU_PUBLIC_ORIGIN="https://clinic.example"'
    before = b"# administrator comment\r\n\r\n" + path.read_bytes().replace(old + b"\n", b"")
    before += b'EASYICU_UNRELATED="preserve unicode \\u738b and spaces  "\r\n' + old + newline
    path.write_bytes(before)
    original_replace = os.replace
    replacements = []

    def replace(source, destination):
        assert Path(destination) == path
        assert path.read_bytes() == before
        assert Path(source).parent == instance
        assert Path(source).stat().st_mode & 0o777 == 0o600
        replacements.append(source)
        return original_replace(source, destination)

    monkeypatch.setattr(tooling.os, "replace", replace)
    origin = "https://gateway.example:8443"
    arguments = ["--clear"] if clear else [origin]
    assert source_cli(tooling, users, "set-public-origin", *arguments) == 0
    new = b'EASYICU_PUBLIC_ORIGIN=' + json.dumps("" if clear else origin).encode()
    assert path.read_bytes() == before.replace(old + newline, new + newline)
    assert len(replacements) == 1
    assert not Path(replacements[0]).exists()
    assert path.stat().st_mode & 0o777 == 0o600
    assert "start doctor" in capsys.readouterr().out
    if clear:
        with pytest.raises(ValueError):
            instance_socket(users, "doctor", origin)
    else:
        assert instance_socket(users, "doctor", origin) == (instance / "run/web.sock").resolve()


@pytest.mark.parametrize("origin", ["http://gateway.example", "https://gateway.example/path",
                                   "https://gateway.example?query=1", "https://gateway.example#fragment",
                                   "https://user:password@gateway.example", "https://gateway.example:invalid"])
def test_create_and_set_public_origin_refuse_invalid_origin(tooling, tmp_path, origin):
    users, instance = create(tooling, tmp_path)
    path = instance / "instance.env"
    before = path.read_bytes()
    assert source_cli(tooling, users, "set-public-origin", origin) != 0
    assert path.read_bytes() == before
    assert tooling.main(["--users-root", str(users), "create", "other", "--display-name", "Synthetic",
                         "--data-root", str(tmp_path / "shared-data"), "--runner-image", "synthetic",
                         "--public-origin", origin]) != 0
    assert not (users / "other").exists()


@pytest.mark.parametrize("clear", [True, False])
def test_set_public_origin_refuses_a_running_instance(tooling, tmp_path, monkeypatch, capsys, clear):
    users, instance = create(tooling, tmp_path)
    path = instance / "instance.env"
    before = path.read_bytes()
    (instance / "run/instance.pid.json").write_text(json.dumps({"pid": 12345, "create_time": 42.0}))
    command = ["python", "-m", "uvicorn", "easyicu.webserver.app:app", "--uds", str(instance / "run/web.sock")]
    process = SimpleNamespace(create_time=lambda: 42.0, cmdline=lambda: command)
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: process)
    assert source_cli(tooling, users, "status") == 0
    assert "running" in capsys.readouterr().out
    arguments = ["--clear"] if clear else ["https://other.example"]
    assert source_cli(tooling, users, "set-public-origin", *arguments) != 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("arguments", [[], ["--clear", "https://gateway.example"]])
def test_set_public_origin_requires_an_origin_or_clear(tooling, tmp_path, arguments):
    users, instance = create(tooling, tmp_path)
    before = (instance / "instance.env").read_bytes()
    # Python 3.11's argparse rejects an origin after --clear itself (exit 2);
    # later versions parse both and the command refuses (exit 1).
    try:
        code = source_cli(tooling, users, "set-public-origin", *arguments)
    except SystemExit as exc:
        code = exc.code
    assert code != 0
    assert (instance / "instance.env").read_bytes() == before


def test_failed_public_origin_replacement_preserves_env_and_removes_temporary(tooling, tmp_path, monkeypatch):
    users, instance = create(tooling, tmp_path)
    path = instance / "instance.env"
    before = path.read_bytes()

    def fail(source, destination):
        raise OSError("synthetic interrupted replacement")

    monkeypatch.setattr(tooling.os, "replace", fail)
    assert source_cli(tooling, users, "set-public-origin", "https://gateway.example") == 1
    assert path.read_bytes() == before
    assert list(instance.glob(".instance-env-*")) == []


@pytest.mark.parametrize("newline", [b"\n", b""])
def test_set_public_origin_adds_a_missing_legacy_key(tooling, tmp_path, newline):
    users, instance = create(tooling, tmp_path)
    path = instance / "instance.env"
    before = b"".join(line for line in path.read_bytes().splitlines(keepends=True)
                      if not line.startswith(b"EASYICU_PUBLIC_ORIGIN="))
    before += b"# preserved final comment" + newline
    path.write_bytes(before)
    assert source_cli(tooling, users, "set-public-origin", "https://gateway.example") == 0
    assert path.read_bytes().startswith(before)
    assert tooling.instance_environment(instance)["EASYICU_PUBLIC_ORIGIN"] == "https://gateway.example"


@pytest.mark.parametrize("command", [[], ["set-public-origin"]])
def test_public_origin_command_is_documented_in_help(tooling, capsys, command):
    with pytest.raises(SystemExit) as exc:
        tooling.main([*command, "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "set-public-origin" in output
    if command:
        assert "--clear" in output and "origin" in output
    else:
        assert "python tools/hosted_instances.py set-public-origin" in tooling.__doc__


@pytest.mark.parametrize("mismatch", [None, "time", "application", "socket"])
def test_stop_only_signals_matching_recorded_process(tooling, tmp_path, monkeypatch, mismatch):
    users, instance = create(tooling, tmp_path)
    socket = str(instance / "run/web.sock")
    (instance / "run/instance.pid.json").write_text(json.dumps({"pid": 12345, "create_time": 42.0}))
    signals = []
    command = ["python", "-m", "uvicorn", "easyicu.webserver.app:app", "--uds", socket]
    if mismatch == "application":
        command[3] = "another.application:app"
    if mismatch == "socket":
        command[-1] = str(instance / "run/another.sock")
    process = SimpleNamespace(create_time=lambda: 99.0 if mismatch == "time" else 42.0,
                              cmdline=lambda: command,
                              terminate=lambda: signals.append(12345), wait=lambda timeout: None)
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(tooling.psutil, "process_iter", lambda *a, **kw: pytest.fail("No process sweep"))
    assert tooling.main(["--users-root", str(users), "stop", "doctor"]) == 0
    assert signals == ([] if mismatch else [12345])


def test_list_status_do_not_read_provider_keys(tooling, tmp_path, monkeypatch, capsys):
    users, instance = create(tooling, tmp_path)
    original = Path.read_text

    def read(path, *args, **kw):
        assert path.name != "pi-provider.env"
        return original(path, *args, **kw)

    monkeypatch.setattr(Path, "read_text", read)
    assert tooling.main(["--users-root", str(users), "list"]) == 0
    assert tooling.main(["--users-root", str(users), "status", "doctor"]) == 0
    assert "doctor" in capsys.readouterr().out
    assert not (instance / "run/instance.pid.json").exists()
    assert os.environ.get("EASYICU_HOME") != str(instance / "home")


def test_instance_creation_never_writes_agentic_coder_backend(tooling, tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_AGENTIC_CODER_BACKEND", "codex")
    _, instance = create(tooling, tmp_path)
    assert "EASYICU_AGENTIC_CODER_BACKEND" not in tooling.instance_environment(instance)
    assert "EASYICU_AGENTIC_CODER_BACKEND" not in (instance / "instance.env").read_text()


def synthetic_export(root, database="miiv"):
    root.mkdir()
    pd.DataFrame({"stay_id": [101, 101, 202], "synthetic_value": [7, 8, 9]}).to_parquet(
        root / "vitals.parquet", index=False,
    )
    (root / "easyicu_export_manifest.json").write_text(json.dumps({
        "database": database, "patient_count": 2,
        "modules": [{"file": "vitals.parquet", "module": "vitals", "rows": 3}],
    }))
    return root


def source_cli(tooling, users, command, *args):
    return tooling.main(["--users-root", str(users), command, "doctor", *map(str, args)])


@pytest.mark.parametrize("kind", ["outside", "home", "symlink", "prefix", "dotdot"])
def test_add_source_refuses_outside_before_any_probe(tooling, tmp_path, monkeypatch, capsys, kind):
    users, instance = create(tooling, tmp_path)
    root = tmp_path / "shared-data"
    outside = tmp_path / "shared-data-other"
    outside.mkdir()
    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    candidate = {"outside": outside, "home": instance / "home", "symlink": link,
                 "prefix": outside, "dotdot": root / ".." / outside.name}[kind]
    assert source_cli(tooling, users, "add-source", candidate) == 1
    assert "path_outside_data_roots" in capsys.readouterr().err
    for key, value in tooling.instance_environment(instance).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dataio, "describe_export_source", lambda *a: pytest.fail("probed outside roots"))
    assert tooling._source_metadata("add-source", str(candidate))["error"] == "path_outside_data_roots"
    assert not (instance / "home/.easyicu/webserver_sources.json").exists()


@pytest.mark.parametrize("kind,error", [("empty", "no_export_files"),
                                        ("missing", "not_a_directory"), ("file", "not_a_directory")])
def test_add_source_preserves_owner_error_codes(tooling, tmp_path, capsys, kind, error):
    users, instance = create(tooling, tmp_path)
    candidate = tmp_path / "shared-data" / "invalid"
    if kind == "empty":
        candidate.mkdir()
    elif kind == "file":
        candidate.write_text("synthetic non-export")
    assert source_cli(tooling, users, "add-source", candidate) == 1
    assert error in capsys.readouterr().err
    assert not (instance / "home/.easyicu/webserver_sources.json").exists()


@pytest.mark.parametrize("valid", [True, False])
def test_source_commands_leave_administrator_environment_and_settings_unchanged(
    tooling, tmp_path, monkeypatch, capsys, valid,
):
    users, instance = create(tooling, tmp_path)
    admin_home = tmp_path / "admin-home"
    admin_home.mkdir()
    admin_state = admin_home / ".easyicu"
    monkeypatch.setenv("HOME", str(admin_home))
    monkeypatch.setenv("EASYICU_HOME", str(admin_home))
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(admin_home))
    monkeypatch.setattr(settings, "_CONFIG_PATH", admin_state / "webserver_settings.json")
    monkeypatch.setattr(sources, "_CONFIG_PATH", admin_state / "webserver_sources.json")
    settings.update_settings({"support_contact": "Administrator only"})
    saved = settings._CONFIG_PATH.read_bytes()
    before = dict(os.environ)
    paths = settings._CONFIG_PATH, sources._CONFIG_PATH
    candidate = tmp_path / "shared-data" / "prepared"
    if valid:
        synthetic_export(candidate)
    else:
        candidate.mkdir()
    assert source_cli(tooling, users, "add-source", candidate, "--label", "Pilot export") == (0 if valid else 1)
    assert source_cli(tooling, users, "list-sources") == 0
    assert dict(os.environ) == before
    assert Path.home() == admin_home
    assert (settings._CONFIG_PATH, sources._CONFIG_PATH) == paths
    assert settings._CONFIG_PATH.read_bytes() == saved
    assert settings.load_settings()["support_contact"] == "Administrator only"
    assert not sources._CONFIG_PATH.exists()
    assert (instance / "home/.easyicu/webserver_sources.json").exists() is valid
    assert "synthetic_value" not in capsys.readouterr().out


def test_list_sources_reports_only_metadata_and_preserves_inactive_selection(tooling, tmp_path, capsys):
    users, instance = create(tooling, tmp_path)
    first = synthetic_export(tmp_path / "shared-data" / "first")
    second = synthetic_export(tmp_path / "shared-data" / "second", database="eicu")
    assert source_cli(tooling, users, "add-source", first, "--label", "Pilot A") == 0
    added = json.loads(capsys.readouterr().out)
    assert added["sources"] == [{"label": "Pilot A", "database": "miiv", "stays": 2,
                                  "total_rows": 3, "active": True}]
    assert added["registered_source_counts"]["by_database"][get_database_profile("miiv").key] == 1
    assert source_cli(tooling, users, "add-source", second, "--label", "Pilot B", "--inactive") == 0
    inactive = json.loads(capsys.readouterr().out)
    assert inactive["sources"] == [{"label": "Pilot B", "database": "eicu", "stays": 2,
                                     "total_rows": 3, "active": False}]
    assert source_cli(tooling, users, "list-sources") == 0
    output = capsys.readouterr()
    listed = json.loads(output.out)
    assert sorted(listed["sources"], key=lambda row: row["label"]) == added["sources"] + inactive["sources"]
    assert sum(listed["registered_source_counts"]["by_database"].values()) == 2
    assert str(tmp_path) not in output.out + output.err
    assert "path" not in output.out and "synthetic_value" not in output.out
    assert (instance / "home/.easyicu/webserver_sources.json").stat().st_mode & 0o777 == 0o600


def test_running_instance_capabilities_sees_added_source_without_restart(tooling, tmp_path, monkeypatch, capsys):
    users, instance = create(tooling, tmp_path)
    export = synthetic_export(tmp_path / "shared-data" / "prepared")
    for key, value in tooling.instance_environment(instance).items():
        monkeypatch.setenv(key, value)
    state = instance / "home/.easyicu"
    monkeypatch.setattr(settings, "_CONFIG_PATH", state / "webserver_settings.json")
    monkeypatch.setattr(sources, "_CONFIG_PATH", state / "webserver_sources.json")
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", state / "capability_tool_audit.jsonl")
    monkeypatch.setattr(PiProviderConfigStore, "public_status", lambda *a, **kw: {"connection_verified": False})
    client = TestClient(app)
    before = client.get("/api/capabilities")
    assert before.status_code == 200
    assert sum(before.json()["registered_sources"]["by_database"].values()) == 0
    assert source_cli(tooling, users, "add-source", export) == 0
    after = client.get("/api/capabilities")
    assert after.status_code == 200
    assert after.json()["deployment"]["hosted"] is True
    assert after.json()["registered_sources"] == json.loads(capsys.readouterr().out)["registered_source_counts"]
    assert after.json()["registered_sources"]["by_database"][get_database_profile("miiv").key] == 1


@pytest.mark.parametrize("provision", [False, True])
def test_all_instance_state_written_is_private_even_with_permissive_umask(
    tooling, tmp_path, monkeypatch, provision,
):
    previous = os.umask(0o002)
    try:
        _, instance = create(tooling, tmp_path)
        state = instance / "home/.easyicu"
        if provision:
            # Also repair pre-existing launch-b1 instances during provisioning.
            state.chmod(0o775)
            (state / "webserver_settings.json").chmod(0o664)
            stale = state / "webserver_settings.json.tmp"
            stale.write_text("synthetic interrupted settings write")
            stale.chmod(0o664)
            key = tmp_path / "key"
            key.write_text("synthetic-test-key")
            key.chmod(0o600)

            def verify(*args):
                assert state.stat().st_mode & 0o777 == 0o700
                return fake_verify(*args)

            tooling.provision_model(instance, key, verifier=verify)
        assert state.stat().st_mode & 0o777 == 0o700
        assert {file.name for file in state.iterdir()} == (
            {"webserver_settings.json", "pi-provider.env", "pi-provider-verification.json"}
            if provision else {"webserver_settings.json"}
        )
        assert all(file.stat().st_mode & 0o777 == 0o600 for file in state.iterdir())
        assert os.umask(0o002) == 0o002
    finally:
        os.umask(previous)


def test_instance_settings_writes_private_temporary_files_and_restores_umask_on_failure(
    tooling, tmp_path, monkeypatch,
):
    previous = os.umask(0o002)
    original = Path.replace
    writes = []

    def replace(path, target):
        if path.name == "webserver_settings.json.tmp":
            writes.append(path)
            assert path.stat().st_mode & 0o777 == 0o600
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    try:
        _, instance = create(tooling, tmp_path)
        assert len(writes) == 1

        def failed_write(*args, **kwargs):
            raise RuntimeError("synthetic write failure")

        monkeypatch.setattr(settings, "_write_settings", failed_write)
        with pytest.raises(RuntimeError, match="synthetic write failure"):
            tooling._update_instance_settings(instance / "home/.easyicu", {"ai_enabled": True})
        assert os.umask(0o002) == 0o002
    finally:
        os.umask(previous)


def test_source_subprocess_failure_never_prints_private_diagnostics(tooling, tmp_path, monkeypatch, capsys):
    users, _ = create(tooling, tmp_path)

    def failure(*args, **kwargs):
        raise tooling.subprocess.CalledProcessError(
            1, "synthetic", output="synthetic patient rows", stderr="synthetic provider key and upstream text",
        )

    monkeypatch.setattr(tooling.subprocess, "run", failure)
    assert source_cli(tooling, users, "list-sources") == 1
    output = capsys.readouterr()
    assert "CalledProcessError" in output.err
    assert "synthetic" not in output.out + output.err


@pytest.mark.parametrize("command", [[], ["add-source"], ["list-sources"]])
def test_source_commands_are_documented_in_help(tooling, capsys, command):
    with pytest.raises(SystemExit) as exc:
        tooling.main([*command, "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    if not command:
        assert "add-source" in output and "list-sources" in output
        assert "python tools/hosted_instances.py add-source" in tooling.__doc__
        assert "python tools/hosted_instances.py list-sources" in tooling.__doc__
    elif command == ["add-source"]:
        assert "path" in output and "--label" in output and "--inactive" in output
