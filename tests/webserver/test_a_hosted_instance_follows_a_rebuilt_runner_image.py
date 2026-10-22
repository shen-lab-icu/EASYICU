"""A stopped hosted instance can follow a rebuilt runner image.

A kernel fix lands as a new image; each clinician's instance names its
image in its private env file.  ``set-runner-image`` changes that one line
of a stopped instance, atomically and privately, and refuses a running
instance or anything that is not one image reference.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def tooling():
    return importlib.import_module("tools.hosted_instances")


def _create(tooling, tmp_path):
    users = tmp_path / "users"
    data = tmp_path / "shared-data"
    data.mkdir()
    assert tooling.main([
        "--users-root", str(users), "create", "doctor", "--display-name", "王医生",
        "--data-root", str(data), "--runner-image", "easyicu-runner:old",
    ]) == 0
    return users, users / "doctor"


def _set(tooling, users, *args):
    return tooling.main(["--users-root", str(users), "set-runner-image", "doctor", *args])


@pytest.mark.parametrize(
    "image",
    [
        "easyicu-research-agent:dev9-21e91ee0e-20261022s",
        "registry.example/easyicu/runner",
        f"easyicu-runner:v1@sha256:{'a' * 64}",
    ],
)
def test_only_the_image_line_changes(tooling, tmp_path, capsys, image):
    users, instance = _create(tooling, tmp_path)
    path = instance / "instance.env"
    before = path.read_bytes()

    assert _set(tooling, users, image) == 0

    old = b'EASYICU_RUNNER_IMAGE="easyicu-runner:old"'
    assert path.read_bytes() == before.replace(
        old, b"EASYICU_RUNNER_IMAGE=" + json.dumps(image).encode()
    )
    assert tooling.instance_environment(instance)["EASYICU_RUNNER_IMAGE"] == image
    assert path.stat().st_mode & 0o777 == 0o600
    assert "start doctor" in capsys.readouterr().out


@pytest.mark.parametrize(
    "image", ["", "two words", "Upper:case", "image\nEASYICU_HOSTED=0", "image:tag:extra", "x" * 300],
)
def test_what_is_not_one_image_reference_is_refused(tooling, tmp_path, image):
    users, instance = _create(tooling, tmp_path)
    before = (instance / "instance.env").read_bytes()

    assert _set(tooling, users, image) != 0
    assert (instance / "instance.env").read_bytes() == before


def test_a_running_instance_keeps_its_image(tooling, tmp_path, monkeypatch):
    users, instance = _create(tooling, tmp_path)
    before = (instance / "instance.env").read_bytes()
    (instance / "run/instance.pid.json").write_text(json.dumps({"pid": 12345, "create_time": 42.0}))
    command = ["python", "-m", "uvicorn", "easyicu.webserver.app:app", "--uds", str(instance / "run/web.sock")]
    process = SimpleNamespace(create_time=lambda: 42.0, cmdline=lambda: command)
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: process)

    assert _set(tooling, users, "easyicu-runner:new") != 0
    assert (instance / "instance.env").read_bytes() == before
