"""Synthetic accounts only; no existing instance or credential is opened."""

import json
import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def gateway(tmp_path):
    from easyicu.hosted_gateway.app import create_app
    from easyicu.hosted_gateway.auth import hash_password
    from easyicu.hosted_gateway.config import GatewayConfig

    users = {
        name: {"socket": str(tmp_path / name / "web.sock"),
               "password": hash_password("synthetic-pass"), "version": name + "-v1"}
        for name in ("alice", "bob")
    }
    users_file = tmp_path / "users.json"
    users_file.write_text(json.dumps(users))
    users_file.chmod(0o600)
    secret_file = tmp_path / "secret"
    secret_file.write_bytes(b"synthetic-secret-for-tests-only-32")
    secret_file.chmod(0o600)
    config = GatewayConfig(public_origin="https://icu.example:8443", users_file=users_file,
                           secret_file=secret_file, cert_file=tmp_path / "cert.pem",
                           key_file=tmp_path / "key.pem")
    clock = [1_800_000_000.0]
    app = create_app(config, clock=lambda: clock[0])
    return app, config, clock, users


@pytest.fixture
def client(gateway):
    with TestClient(gateway[0], base_url="https://icu.example:8443",
                    follow_redirects=False) as client:
        yield client


@pytest.fixture
def login():
    def perform(client, username="alice", password="synthetic-pass", **kwargs):
        page = client.get("/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        return client.post("/login", data={"username": username, "password": password,
                                          "csrf": csrf},
                           headers={"origin": "https://icu.example:8443"}, **kwargs)
    return perform


@pytest.fixture
def admin_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"public_origin": "https://icu.example:8443",
                                 "users_file": "users.json", "secret_file": "secret",
                                 "cert_file": "cert.pem", "key_file": "key.pem"}))
    users_root = tmp_path / "instances"
    for name in ("a", "b"):
        instance = users_root / name
        (instance / "run").mkdir(parents=True, mode=0o700)
        instance.chmod(0o700)
        env = instance / "instance.env"
        env.write_text('EASYICU_PUBLIC_ORIGIN="https://icu.example:8443"\n'
                       'EASYICU_HOSTED="1"\nEASYICU_WEB_TRUST_PROXY="1"\n')
        env.chmod(0o600)
    return config, users_root


@pytest.fixture
def run_admin(admin_config, monkeypatch):
    from easyicu.hosted_gateway.cli import main

    monkeypatch.setattr("easyicu.hosted_gateway.cli.getpass.getpass",
                        lambda _: "synthetic-pass")

    def run(*args):
        return main([*args, "--config", str(admin_config[0])])
    return run


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[3]
