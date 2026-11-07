"""Seal-tip bucket rebuild tool (tools/seal_buckets): the order, refusal and isolation contract.

The tool decides the formal bucket roots at the seal, so every failure must be typed and exit non-zero,
nothing half-built may appear under a final name, the two builds may share nothing, and the CareVue farm
may only link inside the allowed roots. These tests run the real tool, unconfined, against a synthetic
world (tests/support/seal_buckets.py): a git mirror with a fake builder and two current formal roots.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.seal_buckets import (
    make_world,
    make_writable,
    rewrite_config,
    seal_args,
)
from tools.seal_buckets import seal_rebuild as sr

TOOL = Path(sr.__file__)
WRAPPER = TOOL.with_name("seal_rebuild.sh")


def _run(args, **env):
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=600,
    )


@pytest.fixture
def world(tmp_path):
    w = make_world(tmp_path / "w")
    yield w
    make_writable(w["root"])


def _status(result) -> dict[str, str]:
    return {
        line.split()[0]: line
        for line in result.stdout.splitlines()
        if " FAILED " in line
        or " PASS " in line
        or " DRY_RUN_OK" in line
        or " IDENTICAL" in line
    }


def _target(world, name, mode="build1"):
    return (
        world["b6"]
        / f"{name}_s{world['sha'][:9]}{'_build2' if mode == 'build2' else ''}"
    )


def _ro(path: Path) -> bool:
    return not stat.S_IMODE(os.lstat(path).st_mode) & 0o222


# ---- dry run and refusals before anything is built


def test_dry_run_plans_every_db_and_writes_nothing_under_b6(world):
    before = sorted(p.name for p in world["b6"].iterdir())
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 0, result.stdout + result.stderr
    status = _status(result)
    assert "DRY_RUN_OK" in status["sic"] and "DRY_RUN_OK" in status["cv"]
    assert sorted(p.name for p in world["b6"].iterdir()) == before


def test_dry_run_probe_builds_with_the_seal_builder_and_applies_strict_qc(world):
    result = _run(seal_args(world, "build1", "--dry-run", "--probe"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "probe hash: PASS" in result.stdout
        and "probe variable_stay_range: PASS" in result.stdout
    )


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("no_builder_code", "builder_code_missing"),
        ("dict_encoding", "float_dictionary_encoding_not_disabled"),
    ],
)
def test_dry_run_probe_reports_a_builder_that_would_fail_qc(tmp_path, mode, reason):
    w = make_world(tmp_path / "w", mode=mode)
    try:
        result = _run(seal_args(w, "build1", "--dry-run", "--probe"))
        assert result.returncode == 1
        assert f"probe hash: FAILED {reason}" in result.stdout
    finally:
        make_writable(w["root"])


def test_a_new_bucket_violating_the_storage_contract_is_never_published(tmp_path):
    """delta29t (b): an empty extra bucket directory passes QC and the footer scan, but the reader
    would refuse the layout; the publish path runs the full contract first."""
    w = make_world(tmp_path / "w", mode="extra_empty_dir")
    try:
        result = _run(seal_args(w, "build1", "--only", "sic"))
        assert result.returncode == 1
        assert "sic FAILED bucket_contract_invalid" in result.stdout
        assert "bucket_modulus_mismatch" in result.stdout
        assert not _target(w, "sic_cur").exists()
        assert not sr.pending_marker(_target(w, "sic_cur")).exists()
    finally:
        make_writable(w["root"])


def test_unknown_sha_is_refused(world):
    result = _run(
        [*seal_args(world, "build1")[1:2], "--config", str(world["config"])][:0]
        + [
            "0" * 40,
            "build1",
            "--config",
            str(world["config"]),
            "--cores",
            "sic=188-191",
            "--cores",
            "cv=192-195",
        ]
    )
    assert result.returncode == 1
    assert "all FAILED seal_sha_unknown" in result.stdout


def test_a_builder_without_the_float_check_is_refused(tmp_path):
    w = make_world(tmp_path / "w", mark=False)
    try:
        result = _run(seal_args(w, "build1", "--dry-run"))
        assert (
            result.returncode == 1
            and "all FAILED builder_lacks_float_fidelity" in result.stdout
        )
    finally:
        make_writable(w["root"])


def test_forbidden_or_missing_cores_are_refused(world):
    args = seal_args(world, "build1", "--dry-run")
    bad = [a if a != "sic=188-191" else "sic=100-103" for a in args]
    assert "all FAILED forbidden_cores" in _run(bad).stdout
    assert (
        "all FAILED forbidden_cores"
        in _run([a if a != "sic=188-191" else "sic=224-227" for a in args]).stdout
    )
    missing = [
        world["sha"],
        "build1",
        "--config",
        str(world["config"]),
        "--cores",
        "sic=188-191",
        "--dry-run",
    ]
    result = _run(missing)
    assert result.returncode == 1 and "all FAILED no_cores cv" in result.stdout


def test_unconfined_runs_are_refused_on_the_production_root(world):
    rewrite_config(world, b6_root=sr.PRODUCTION["b6_root"])
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 1 and "unconfined_on_production_root" in result.stdout


def test_an_existing_target_or_staging_root_is_refused(world):
    staging = world["b6"] / f".{_target(world, 'sic_cur').name}.seal_staging"
    staging.mkdir()
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 1 and "sic FAILED target_exists" in result.stdout
    staging.rmdir()
    farm = _target(world, "mimic_cv_dir")
    farm.mkdir()
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 1 and "cv FAILED farm_target_exists" in result.stdout


def test_a_farm_link_outside_the_allowed_roots_is_refused(world, tmp_path):
    outside = tmp_path / "elsewhere.parquet"
    outside.write_bytes(b"PAR1")
    farm = Path(json.loads(world["config"].read_text())["dbs"]["cv"]["current_root"])
    os.symlink(outside, farm / "stray.parquet")
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert (
        result.returncode == 1
        and "cv FAILED farm_link_outside_allowed_roots stray.parquet" in result.stdout
    )


def test_the_farm_must_match_its_reference_link_set(world):
    config = json.loads(world["config"].read_text())
    reference = world["b6"] / "mimic_cv_dir_reference"
    current = Path(config["dbs"]["cv"]["current_root"])
    reference.mkdir()
    for entry in os.scandir(current):
        if entry.is_symlink():
            os.symlink(os.readlink(entry.path), reference / entry.name)
        elif entry.is_dir():
            (reference / entry.name).mkdir()
            for f in os.scandir(entry.path):
                os.symlink(os.readlink(f.path), reference / entry.name / f.name)
        else:
            (reference / entry.name).write_text("{}")
    config["dbs"]["cv"]["farm"]["reference"] = str(reference)
    rewrite_config(world, dbs=config["dbs"])
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 0, result.stdout
    assert "differences': 0" in result.stdout
    os.unlink(reference / "admissions.parquet")
    os.symlink(
        world["view"] / "inputevents_cv.parquet", reference / "admissions.parquet"
    )
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert (
        result.returncode == 1
        and "cv FAILED farm_reference_mismatch admissions.parquet" in result.stdout
    )


# ---- full synthetic builds


def test_build1_installs_read_only_roots_with_receipts_in_order_and_assembles_the_farm(
    world,
):
    result = _run(seal_args(world, "build1"))
    assert result.returncode == 0, result.stdout + result.stderr
    s9 = world["sha"][:9]
    sic, bucket, farm = (
        _target(world, "sic_cur"),
        _target(world, "mimic_cv_bucket"),
        _target(world, "mimic_cv_dir"),
    )
    for root in (sic, bucket, farm):
        assert root.is_dir() and _ro(root)
        assert sr.writable_entries(root) == []
    assert not any(p.name.endswith(".seal_staging") for p in world["b6"].iterdir())
    work = world["work"] / f"seal_{s9}" / "build1"
    receipt = json.loads((work / "receipt_sic.json").read_text())
    assert (
        receipt["verdict"] == "PASS"
        and receipt["install"]["read_only_verified"] is True
    )
    assert receipt["install"]["in_root_receipt_sha256"] == sr.sha256_file(
        sic / sr.IN_ROOT_RECEIPT
    )
    assert receipt["footer_scan"]["ok"] is True
    for table in ("medication", "laboratory"):
        assert receipt["tables"][table]["builder_code"]["commit"] == world["sha"]
        assert (
            receipt["footer_scan"]["buckets"][f"{table}_bucket"]["rows_footer"]
            == receipt["tables"][table]["rows"]
        )
    # the mirrored root: links verbatim, the conversion manifest re-pointed, logs and receipts not carried
    assert os.readlink(sic / "medication" / "1.parquet") == str(
        world["databases"] / "sicx/medication/1.parquet"
    )
    assert json.loads((sic / "conversion_manifest.json").read_text())[
        "data_path"
    ] == str(sic)
    assert not (sic / "_lead_r13_build_medication.log").exists()
    # the CareVue farm: the current farm's entries and targets, the new bucket, data_path re-pointed
    current = Path(json.loads(world["config"].read_text())["dbs"]["cv"]["current_root"])
    skip = {"chartevents_bucket", sr.IN_ROOT_RECEIPT}
    assert sr.link_set(farm, skip) == sr.link_set(current, skip)
    assert os.readlink(farm / "chartevents_bucket") == str(
        bucket / "chartevents_bucket"
    )
    manifest = json.loads((farm / "conversion_manifest.json").read_text())
    original = json.loads((current / "conversion_manifest.json").read_text())
    assert manifest["data_path"] == str(farm)
    assert {k: v for k, v in manifest.items() if k != "data_path"} == {
        k: v for k, v in original.items() if k != "data_path"
    }
    farm_receipt = json.loads((work / "receipt_cv_farm.json").read_text())
    assert (
        farm_receipt["verdict"] == "PASS"
        and farm_receipt["install"]["read_only_verified"]
    )
    # the farm has its own bound in-root receipt (a bucket _COMPLETE never stands for the farm)
    assert farm_receipt["install"]["in_root_receipt_sha256"] == sr.sha256_file(
        farm / sr.IN_ROOT_RECEIPT
    )
    assert not (
        world["b6"] / "_scratch" / f"seal_{s9}" / "build1" / "sic"
    ).exists()  # spill removed


def test_build2_is_isolated_and_compare_requires_identical_twins(world):
    assert _run(seal_args(world, "build1")).returncode == 0
    second = _run(seal_args(world, "build2"))
    assert second.returncode == 0, second.stdout + second.stderr
    s9 = world["sha"][:9]
    work = world["work"] / f"seal_{s9}"
    assert (work / "build1" / "code" / ".git").is_dir() and (
        work / "build2" / "code" / ".git"
    ).is_dir()
    assert _target(world, "sic_cur", "build2").is_dir()
    assert not _target(
        world, "mimic_cv_dir", "build2"
    ).exists()  # the farm is assembled once, from build1
    receipt = json.loads((work / "build2" / "receipt_sic.json").read_text())
    assert "HOME" in receipt["build_env_keys"]
    compare = _run(seal_args(world, "compare"))
    assert compare.returncode == 0, compare.stdout + compare.stderr
    assert "sic IDENTICAL" in compare.stdout and "cv IDENTICAL" in compare.stdout


def test_compare_refuses_until_both_builds_passed(world):
    result = _run(seal_args(world, "compare"))
    assert (
        result.returncode == 1
        and "sic FAILED compare_missing_build build1:sic" in result.stdout
    )


def test_twin_builds_that_differ_fail_the_compare(tmp_path):
    w = make_world(tmp_path / "w", mode="nondeterministic")
    try:
        assert _run(seal_args(w, "build1", "--only", "sic")).returncode == 0
        assert _run(seal_args(w, "build2", "--only", "sic")).returncode == 0
        result = _run(seal_args(w, "compare", "--only", "sic"))
        assert (
            result.returncode == 1
            and "sic FAILED build1_vs_build2_differs" in result.stdout
        )
    finally:
        make_writable(w["root"])


def test_build_env_never_names_the_other_build():
    class Fake:
        mode = "build2"
        work = Path("/w/seal_abc")
        scratch = Path("/b6/_scratch/seal_abc")
        cfg = {"dbs": {"sic": {"current_root": "/b6/sic_108"}}}

        def target(self, db, mode=None):
            return Path(f"/b6/sic_108_sabc{'_build2' if mode == 'build2' else ''}")

    foreign = sr.foreign_paths(Fake())
    assert "/w/seal_abc/build1" in foreign and "/b6/sic_108_sabc" in foreign
    sr.check_isolation({"HOME": "/b6/_scratch/seal_abc/build2/sic/home"}, foreign)
    with pytest.raises(sr.SealError) as caught:
        sr.check_isolation({"TMPDIR": "/b6/_scratch/seal_abc/build1/sic/tmp"}, foreign)
    assert caught.value.reason == "build_env_crosses_builds"


# ---- typed failures during a build: non-zero exit, nothing under a final name


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("fail", "build_failed"),
        ("no_builder_code", "builder_code_missing"),
        ("dict_encoding", "float_dictionary_encoding_not_disabled"),
        ("bad_rows", "float_fidelity_rows_vs_receipt"),
    ],
)
def test_each_typed_build_failure_exits_non_zero_and_installs_nothing(
    tmp_path, mode, reason
):
    w = make_world(tmp_path / "w", mode=mode)
    try:
        result = _run(seal_args(w, "build1", "--only", "sic"))
        assert result.returncode == 1
        assert f"sic FAILED {reason}" in result.stdout, result.stdout
        assert not _target(w, "sic_cur").exists()
        staging = w["b6"] / f".{_target(w, 'sic_cur').name}.seal_staging"
        assert staging.is_dir() and not (staging / sr.IN_ROOT_RECEIPT).exists()
        receipt = json.loads(
            (
                w["work"] / f"seal_{w['sha'][:9]}" / "build1" / "receipt_sic.json"
            ).read_text()
        )
        assert receipt["verdict"] == "FAILED" and receipt["failure"]["reason"] == reason
        rerun = _run(seal_args(w, "build1", "--only", "sic"))
        assert rerun.returncode == 1 and "sic FAILED target_exists" in rerun.stdout
    finally:
        make_writable(w["root"])


def test_a_dirty_or_moved_checkout_is_refused(world):
    code = world["work"] / f"seal_{world['sha'][:9]}" / "build1" / "code"
    subprocess.run(["git", "clone", "-q", str(world["mirror"]), str(code)], check=True)
    (code / "stray.txt").write_text("not from the seal SHA\n")
    result = _run(seal_args(world, "build1"))
    # an existing checkout is never reused, clean or not (D26-ISOLATION)
    assert result.returncode == 1 and "all FAILED checkout_exists" in result.stdout
    assert not _target(world, "sic_cur").exists()
    (code / "stray.txt").unlink()
    git_dirty = sr.verify_checkout
    with pytest.raises(sr.SealError, match="checkout_not_clean"):
        (code / "stray.txt").write_text("x\n")
        git_dirty(
            code,
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=code, capture_output=True, text=True
            ).stdout.strip(),
        )
    (code / "stray.txt").unlink()
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "moved",
        ],
        cwd=code,
        check=True,
    )
    with pytest.raises(sr.SealError, match="checkout_not_at_seal"):
        sr.verify_checkout(code, world["sha"])


def test_install_writes_the_in_root_receipt_last_then_read_only_then_renames(
    tmp_path, monkeypatch
):
    staging, target = tmp_path / ".r.seal_staging", tmp_path / "r"
    (staging / "bucket").mkdir(parents=True)
    (staging / "bucket" / "part.parquet").write_bytes(b"PAR1")
    os.symlink(tmp_path, staging / "link")
    seen = {}
    rename = sr.rename_noreplace

    def spy(src, dst):
        seen["receipt"] = (Path(src) / sr.IN_ROOT_RECEIPT).is_file()
        seen["writable"] = sr.writable_entries(Path(src))  # the root included
        seen["target_absent"] = not Path(dst).exists()
        seen["pending"] = sr.pending_marker(Path(dst)).is_file()
        return rename(src, dst)

    monkeypatch.setattr(sr, "rename_noreplace", spy)
    record = sr.install(staging, target, {"verdict": "PASS"})
    assert seen == {
        "receipt": True,
        "writable": [],
        "target_absent": True,
        "pending": True,
    }
    assert _ro(target) and sr.writable_entries(target) == []
    assert record["in_root_receipt_sha256"] == sr.sha256_file(
        target / sr.IN_ROOT_RECEIPT
    )
    assert record["read_only_verified"] is True
    make_writable(target)


def test_the_shell_entry_point_is_strict_and_passes_the_exit_code(world):
    text = WRAPPER.read_text()
    assert "set -euo pipefail" in text and "|| true" not in text
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "0" * 40,
            "build1",
            "--config",
            str(world["config"]),
            "--cores",
            "sic=188-191",
            "--cores",
            "cv=192-195",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SEAL_PYTHON": sys.executable},
    )
    assert result.returncode == 1 and "seal_sha_unknown" in result.stdout
    usage = subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        env={**os.environ, "SEAL_PYTHON": sys.executable},
    )
    assert usage.returncode == 2


def test_no_check_in_the_tools_is_silenced():
    for path in TOOL.parent.glob("*"):
        if path.suffix in (".py", ".sh"):
            text = path.read_text()
            assert "|| true" not in text and "except Exception: pass" not in text, (
                path.name
            )
