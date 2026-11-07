"""Private launch-root regressions; unsafe templates are intercepted before mktemp can write."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.seal_buckets import make_world, make_writable, seal_args
from tools.seal_buckets import seal_rebuild as sr


@pytest.fixture
def launch_world(tmp_path):
    world = make_world(tmp_path / "w")
    world["work"].mkdir()
    shim = tmp_path / "bin"
    shim.mkdir()
    log = tmp_path / "mktemp_calls"
    # Baseline uses /tmp when unset: record and refuse that call without any forbidden write.
    (shim / "mktemp").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >>"$LAUNCH_TEST_LOG"\n'
        'case "${@: -1}" in "$LAUNCH_TEST_ALLOWED"/seal_launch.*) exec /usr/bin/mktemp "$@" ;; esac\n'
        "exit 73\n"
    )
    (shim / "mktemp").chmod(0o700)
    world["launch_env"] = {
        **{
            k: v
            for k, v in os.environ.items()
            if k not in ("TMPDIR", "SEAL_LAUNCH_ROOT", "PYTHONPATH")
        },
        "SEAL_MIRROR": str(world["mirror"]),
        "SEAL_PYTHON": sys.executable,
        "PATH": str(shim) + os.pathsep + os.environ["PATH"],
        "LAUNCH_TEST_LOG": str(log),
        "LAUNCH_TEST_ALLOWED": str(world["work"]),
    }
    world["launch_log"] = log
    yield world
    make_writable(world["root"])


def _launch(world, **environment):
    return subprocess.run(
        [
            "bash",
            str(Path(sr.__file__).with_suffix(".sh")),
            *seal_args(world, "build1", "--only", "sic", "--dry-run"),
        ],
        env={**world["launch_env"], **environment},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "root_form", ["missing", "tmp", "shm", "var_tmp", "tmp_symlink", "unwritable"]
)
def test_launcher_refuses_invalid_root_without_fallback(
    launch_world, tmp_path, root_form
):
    world = launch_world
    root = tmp_path / "missing"
    reason = "launch_root_missing"
    if root_form in ("tmp", "shm", "var_tmp"):
        root = Path(
            {"tmp": "/tmp", "shm": "/dev/shm", "var_tmp": "/var/tmp"}[root_form]
        )
        reason = "launch_root_unsafe"
    elif root_form == "tmp_symlink":
        root.symlink_to("/tmp", target_is_directory=True)
        reason = "launch_root_unsafe"
    elif root_form == "unwritable":
        root.mkdir(mode=0o500)
        reason = "launch_root_unwritable"
    done = _launch(world, SEAL_LAUNCH_ROOT=str(root), TMPDIR=str(world["work"]))
    assert done.returncode == 1 and reason in done.stdout, done.stdout + done.stderr
    assert not world["launch_log"].exists()
    if root_form == "unwritable":
        root.chmod(0o700)


def test_launcher_refuses_missing_derived_work_root(launch_world):
    world = launch_world
    world["work"].rmdir()
    done = _launch(world)
    assert done.returncode == 1 and "launch_root_missing" in done.stdout
    assert not world["launch_log"].exists()


def test_launcher_derives_private_root_without_environment_fallback(launch_world):
    world = launch_world
    done = _launch(world)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        world["launch_log"].read_text().strip()
        == f"-d {world['work']}/seal_launch.XXXXXX"
    )
    assert not list(world["work"].glob("seal_launch.*"))


def test_launcher_ignores_tmpdir_and_refuses_unsafe_derived_work_root(launch_world):
    world = launch_world
    cfg = json.loads(world["config"].read_text())
    cfg["work_root"] = "/tmp"
    sr.write_json(world["config"], cfg)
    done = _launch(
        world,
        TMPDIR=str(world["work"]),
        # The base must stop at the shim; never let it run SR with work_root=/tmp.
        LAUNCH_TEST_ALLOWED=str(world["work"] / "never"),
    )
    assert done.returncode == 1 and "launch_root_unsafe" in done.stdout
    assert not world["launch_log"].exists()
