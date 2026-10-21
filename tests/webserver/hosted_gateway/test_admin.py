import json
import stat

import pytest


def test_admin_lifecycle_and_private_files(run_admin, admin_config, capsys):
    from easyicu.hosted_gateway.auth import verify_password

    config, root = admin_config
    assert run_admin("add-user", "alice", "--instance", "a", "--users-root", str(root)) == 0
    users = json.loads(config.with_name("users.json").read_text())
    user = users["alice"]
    assert user["socket"] == str(root / "a/run/web.sock")
    assert verify_password("synthetic-pass", user["password"])
    assert not verify_password("wrong", user["password"])
    for name in ("users.json", "secret"):
        assert stat.S_IMODE(config.with_name(name).stat().st_mode) == 0o600
    assert len(config.with_name("secret").read_bytes()) >= 32
    assert run_admin("list-users") == 0
    output = capsys.readouterr().out
    assert "alice" in output and "synthetic-pass" not in output
    assert user["password"]["hash"] not in output
    assert run_admin("reset-password", "alice") == 0
    updated = json.loads(config.with_name("users.json").read_text())["alice"]
    assert updated["version"] != user["version"]
    assert updated["password"]["salt"] != user["password"]["salt"]
    assert run_admin("remove-user", "alice") == 0
    assert json.loads(config.with_name("users.json").read_text()) == {}


def test_two_users_cannot_share_instance_or_overwrite_user(run_admin, admin_config):
    _, root = admin_config
    assert run_admin("add-user", "alice", "--instance", "a", "--users-root", str(root)) == 0
    assert run_admin("add-user", "bob", "--instance", "a", "--users-root", str(root)) == 1
    assert run_admin("add-user", "alice", "--instance", "b", "--users-root", str(root)) == 1


@pytest.mark.parametrize("kind", ["missing", "origin", "traversal", "symlink", "proxy"])
def test_add_user_requires_real_matching_private_instance(run_admin, admin_config, kind):
    _, root = admin_config
    name = "a"
    if kind == "missing":
        name = "missing"
    elif kind == "origin":
        env = root / "a/instance.env"
        env.write_text(env.read_text().replace("https://icu.example:8443", "https://elsewhere.example"))
    elif kind == "proxy":
        env = root / "a/instance.env"
        env.write_text(env.read_text().replace('EASYICU_WEB_TRUST_PROXY="1"', 'EASYICU_WEB_TRUST_PROXY="0"'))
    elif kind == "traversal":
        name = "../a"
    else:
        (root / "alias").symlink_to(root / "a", target_is_directory=True)
        name = "alias"
    assert run_admin("add-user", "alice", "--instance", name, "--users-root", str(root)) == 1


def test_password_requires_matching_prompts_no_argv(run_admin, admin_config, monkeypatch):
    from easyicu.hosted_gateway.cli import main

    _, root = admin_config
    answers = iter(["one", "two"])
    monkeypatch.setattr("easyicu.hosted_gateway.cli.getpass.getpass", lambda _: next(answers))
    assert run_admin("add-user", "alice", "--instance", "a", "--users-root", str(root)) == 1
    with pytest.raises(SystemExit):
        main(["reset-password", "alice", "--password", "do-not-accept"])


@pytest.mark.parametrize("target", ["users_file", "secret_file"])
@pytest.mark.parametrize("kind", ["public", "symlink"])
def test_runtime_refuses_unsafe_secret_files(gateway, target, kind):
    from easyicu.hosted_gateway.app import create_app

    config = gateway[1]
    path = getattr(config, target)
    if kind == "public":
        path.chmod(0o644)
    else:
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.symlink_to(saved)
    with pytest.raises((ValueError, OSError)):
        create_app(config)


def test_config_defaults_relative_paths_and_https(admin_config):
    from easyicu.hosted_gateway.config import load_config

    path, _ = admin_config
    config = load_config(path)
    assert config.host == "127.0.0.1" and config.port == 8443
    assert config.users_file == path.with_name("users.json")
    payload = json.loads(path.read_text())
    payload["public_origin"] = "http://icu.example"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_config(path)


def test_serve_enforces_tls_and_disables_unredacted_logging(run_admin, admin_config, monkeypatch):
    import ssl
    import uvicorn

    assert run_admin("make-self-signed-cert", "--host", "127.0.0.1") == 0
    assert run_admin("make-self-signed-cert", "--host", "127.0.0.1") == 1
    config, root = admin_config
    certificate = ssl._ssl._test_decode_cert(str(config.with_name("cert.pem")))
    assert certificate["subjectAltName"] == (("IP Address", "127.0.0.1"),)
    assert stat.S_IMODE(config.with_name("key.pem").stat().st_mode) == 0o600
    assert run_admin("add-user", "alice", "--instance", "a", "--users-root", str(root)) == 0
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    assert run_admin("serve") == 0
    assert calls[0]["host"] == "127.0.0.1" and calls[0]["port"] == 8443
    assert calls[0]["ssl_certfile"] == str(config.with_name("cert.pem"))
    assert calls[0]["ssl_keyfile"] == str(config.with_name("key.pem"))
    assert calls[0]["access_log"] is False and calls[0]["proxy_headers"] is False
    assert calls[0]["workers"] == 1


def test_packaging_declares_entrypoint_template_and_httpx(repo_root):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    config = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["easyicu-hosted-gateway"] == "easyicu.hosted_gateway.cli:main"
    assert "templates/*.html" in config["tool"]["setuptools"]["package-data"]["easyicu.hosted_gateway"]
    assert "httpx>=0.27" in config["project"]["optional-dependencies"]["webapp"]


def test_runtime_refuses_short_secret_or_duplicate_sockets(gateway):
    from easyicu.hosted_gateway.app import create_app

    _, config, _, users = gateway
    original = config.secret_file.read_bytes()
    config.secret_file.write_bytes(b"short")
    with pytest.raises(ValueError):
        create_app(config)
    config.secret_file.write_bytes(original)
    users["bob"]["socket"] = users["alice"]["socket"]
    config.users_file.write_text(json.dumps(users))
    with pytest.raises(ValueError):
        create_app(config)


def test_case_alias_cannot_assign_two_users_to_one_instance(run_admin, admin_config):
    from easyicu.hosted_gateway.app import create_app
    from easyicu.hosted_gateway.config import load_config

    _, root = admin_config
    if not (root / "A").exists():
        pytest.skip("Filesystem is case-sensitive; case aliases do not exist")
    assert run_admin("add-user", "alice", "--instance", "a", "--users-root", str(root)) == 0
    assert run_admin("add-user", "bob", "--instance", "A", "--users-root", str(root)) == 1
    config = load_config(admin_config[0])
    users = json.loads(config.users_file.read_text())
    users["bob"] = {**users["alice"], "socket": str(root / "A/run/web.sock")}
    config.users_file.write_text(json.dumps(users))
    with pytest.raises(ValueError):
        create_app(config)
