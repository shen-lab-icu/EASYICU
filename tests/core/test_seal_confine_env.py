"""Synthetic regressions for systemd bus passthrough and launch/verdict separation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.seal_buckets import make_world, make_writable
from tools.seal_buckets import seal_rebuild as sr

BUS = {
    "XDG_RUNTIME_DIR": "/run/user/1002",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1002/bus",
}
TIME_BLOCK = "\tMaximum resident set size (kbytes): 1234\n"
BUS_ERROR = "Failed to connect to bus: $DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not defined\n"


@pytest.fixture
def world(tmp_path):
    w = make_world(tmp_path / "w")
    yield w
    make_writable(w["root"])


def _seal(world, mode, confine):
    cfg = json.loads(world["config"].read_text())
    cfg["confine"] = confine
    seal = sr.Seal(cfg, world["sha"], mode, False, ["sic"], {"sic": "112-119"})
    (seal.work / mode).mkdir(parents=True, exist_ok=True)
    return seal


def _compare_ready(world, monkeypatch, confine):
    """Stub acceptance only; retain compare's environment, launch and verdict handling."""
    seal = _seal(world, "compare", confine)
    for mode in ("build1", "build2"):
        bucket = seal.target("sic", mode) / "medication_bucket"
        bucket.mkdir(parents=True, exist_ok=True)
        (bucket / "_COMPLETE").touch()
        (bucket / "_BUCKET_MANIFEST.json").write_text("{}")
        receipt = seal.work / mode / "receipt_sic.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}")

    def accepted(root, *args, **kwargs):
        return {"tables": {"medication": {
            "manifest_sha256": sr.sha256_file(root / "medication_bucket/_BUCKET_MANIFEST.json"),
            "float_fidelity": {},
        }}}

    monkeypatch.setattr(sr, "accept_root", accepted)
    monkeypatch.setattr(seal, "validator", lambda: None)
    monkeypatch.setattr(seal, "tool", lambda: {})
    monkeypatch.setattr(seal, "helper", lambda rel: Path(rel))
    seal.cfg["dbs"]["sic"]["tables"] = [["medication", ""]]
    return seal


def _popen(monkeypatch, *, timed=False, missing_launcher=False):
    def launch(argv, *, stdout, **kwargs):
        if missing_launcher:
            raise FileNotFoundError("systemd-run")
        stdout.write(TIME_BLOCK if timed else BUS_ERROR)
        return SimpleNamespace(wait=lambda timeout: 1)

    monkeypatch.setattr(sr.subprocess, "Popen", launch)


def test_build_and_compare_receive_only_systemd_bus_passthrough(world, monkeypatch):
    for key, value in BUS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SEAL_UNRELATED_PARENT_VARIABLE", "must-not-be-inherited")
    seen = []

    def capture(seal, db, unit, argv, env, *args):
        seen.append((seal.mode, dict(env)))
        raise sr.SealError("test_stop", unit)

    monkeypatch.setattr(sr, "confined_run", capture)
    plans = []
    for confine, mode in (("none", "build1"), ("systemd", "build2")):
        # Independent build scratch/staging for the two cases.
        seal = _seal(world, mode, confine)
        plans.append((seal, sr.plan_db(seal, "sic")))
    for seal, plan in plans:
        mode, confine = seal.mode, seal.cfg["confine"]
        sr.run_db(seal, "sic", plan, seal.code_dir(mode))
        assert "test_stop" in seal.status["sic"][1]
        compare = _compare_ready(world, monkeypatch, confine)
        sr.compare_db(compare, "sic")
        assert "test_stop" in compare.status["sic"][1]
        assert [mode for mode, _ in seen[-2:]] == [mode, "compare"]
        for _, env in seen[-2:]:
            assert {k: env[k] for k in BUS if k in env} == (
                BUS if confine == "systemd" else {}
            )
            assert "SEAL_UNRELATED_PARENT_VARIABLE" not in env
        assert set(seen[-1][1]) == {"PATH", "LANG", "PYTHONDONTWRITEBYTECODE"} | (
            set(BUS) if confine == "systemd" else set()
        )


@pytest.mark.parametrize("workload", ["medication", "footer", "compare"])
@pytest.mark.parametrize("missing_launcher", [False, True])
def test_confined_launch_failure_is_typed_for_every_workload(
    world, monkeypatch, workload, missing_launcher
):
    seal = _seal(world, "build1", "systemd")
    unit = f"seal-{seal.s9}-build1-sic-{workload}"
    _popen(monkeypatch, missing_launcher=missing_launcher)
    with pytest.raises(sr.SealError) as caught:
        sr.confined_run(
            seal, "sic", unit, ["unused"], {}, seal.work / "launch.log", seal.work, 4, 10
        )
    assert caught.value.reason == "confine_launch_failed"
    assert caught.value.detail == unit


def test_compare_distinguishes_a_nonzero_verdict_from_a_launch_failure(world, monkeypatch):
    seal = _compare_ready(world, monkeypatch, "systemd")
    # The time block proves that the comparator ran. Its nonzero verdict remains a real diff,
    # including if it failed before it could write --out; a bus failure must never become a diff.
    for timed, reason in ((True, "build1_vs_build2_differs"), (False, "confine_launch_failed")):
        _popen(monkeypatch, timed=timed)
        sr.compare_db(seal, "sic")
        assert seal.status["sic"][0] == "FAILED"
        assert reason in seal.status["sic"][1]
        if not timed:
            assert f"seal-{seal.s9}-compare-sic-medication" in seal.status["sic"][1]


def test_compare_bus_passthrough_still_checks_isolation(world, monkeypatch):
    seal = _compare_ready(world, monkeypatch, "systemd")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(seal.scratch / "build2" / "sic"))

    def forbidden(*args, **kwargs):
        pytest.fail("cross-build bus environment reached the launcher")

    monkeypatch.setattr(sr, "confined_run", forbidden)
    sr.compare_db(seal, "sic")
    assert "build_env_crosses_builds XDG_RUNTIME_DIR" in seal.status["sic"][1]
