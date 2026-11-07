"""Seal-tip bucket tool, review round 4 (delta34s + delta37s): one regression per B finding.

Ported reviewer probes: publish_root_replacement_after_rebind, publish_with_replaced_validator,
operator_ignores_live_run_locks, operator_commits_invalid_pending_root (boot_cwd_stdlib_shadow and
site_packages_environment_shadow live in test_seal_bucket_round3.py). Synthetic worlds only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.seal_buckets import make_world, make_writable, seal_args
from tools.seal_buckets import seal_rebuild as sr

TOOL = Path(sr.__file__)
TOOLS = TOOL.parent


def _run(args, tool: Path = TOOL):
    return subprocess.run(
        [sys.executable, "-I", str(tool), *args],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture
def world(tmp_path):
    w = make_world(tmp_path / "w")
    yield w
    make_writable(w["root"])


def _target(world, name="sic_cur"):
    return world["b6"] / f"{name}_s{world['sha'][:9]}"


def _swap_in_modified_copy(staging: Path, table: str, scratch: Path) -> None:
    """Replace a whole bucket directory in the staging root by a contract-valid copy whose manifest
    differs from the one QC bound (a rename in the root; no chmod of the bucket itself)."""
    bucket = staging / f"{table}_bucket"
    copy = scratch / f"{table}_copy"
    shutil.copytree(bucket, copy, symlinks=True)
    make_writable(copy)
    manifest = json.loads((copy / "_BUCKET_MANIFEST.json").read_text())
    manifest["note"] = "swapped after QC"
    (copy / "_BUCKET_MANIFEST.json").write_text(json.dumps(manifest))
    bucket.rename(staging / f".{table}_bucket_old")
    copy.rename(bucket)


# ---- D34-PUBLISH-SWAP: the exact final bucket set is bound and re-verified on the frozen tree


def test_publish_root_replacement_after_rebind(world, monkeypatch, tmp_path, capsys):
    """Reviewer probe: a bucket dir renamed into the (still writable) staging root after the QC
    re-bind, before the footer scan, must never be published."""
    real = sr.confined_run

    def swapping(seal, db, unit, argv, *rest):
        if unit.endswith("-footer"):
            _swap_in_modified_copy(Path(argv[3]), "laboratory", tmp_path)
        return real(seal, db, unit, argv, *rest)

    monkeypatch.setattr(sr, "confined_run", swapping)
    assert sr.main(seal_args(world, "build1", "--only", "sic")) == 1
    out = capsys.readouterr().out
    assert "sic FAILED bucket_manifest_changed_after_qc laboratory" in out
    assert not _target(world).exists()


def test_a_swap_while_the_root_is_frozen_is_caught_before_the_rename(
    world, monkeypatch, tmp_path, capsys
):
    real = sr.make_read_only

    def swap_then_freeze(path, include_root):
        if include_root and str(path).endswith(".seal_staging"):
            _swap_in_modified_copy(Path(path), "laboratory", tmp_path)
        return real(path, include_root)

    monkeypatch.setattr(sr, "make_read_only", swap_then_freeze)
    assert sr.main(seal_args(world, "build1", "--only", "sic")) == 1
    assert "sic FAILED bucket_set_changed laboratory" in capsys.readouterr().out
    assert (
        not _target(world).exists() and not sr.pending_marker(_target(world)).exists()
    )


def test_acceptance_re_verifies_the_exact_final_set(world, capsys):
    assert sr.main(seal_args(world, "build1", "--only", "sic")) == 0
    capsys.readouterr()
    target = _target(world)
    external = (
        world["work"] / f"seal_{world['sha'][:9]}" / "build1" / "receipt_sic.json"
    )
    inner = sr.accept_root(target, external, world["sha"], "build1", "sic")
    assert inner["validator"]["source"] == f"git:{world['sha'][:9]}"
    assert set(inner["bucket_set"]) == {"medication", "laboratory"}
    # an operator-level swap on the final name (chmod, swap, re-freeze) is refused at acceptance
    make_writable(target)
    _swap_in_modified_copy(target, "laboratory", target.parent / "_swap")
    sr.make_read_only(target, include_root=True)
    with pytest.raises(sr.SealError, match="bucket_set_changed"):
        sr.accept_root(target, external, world["sha"], "build1", "sic")


# ---- B37-VALIDATOR-ORIGIN: the contract runs from the sealed, pinned bytes, never a sibling import


def test_publish_with_replaced_validator(tmp_path):
    """Reviewer probe: a copy of the tool beside a permissive bucket_compare.py still refuses a layout
    the real contract refuses (the validator comes from the seal SHA's git object)."""
    w = make_world(tmp_path / "w", mode="extra_empty_dir")
    try:
        tools = tmp_path / "copied_tools"
        tools.mkdir()
        for name in ("seal_rebuild.py", "footer_scan.py"):
            shutil.copyfile(TOOLS / name, tools / name)
        (tools / "bucket_compare.py").write_text(
            "class StorageIntegrityError(Exception):\n    pass\n"
            "def validate_contract(root):\n    return None\n"
        )
        result = _run(
            seal_args(w, "build1", "--only", "sic"), tools / "seal_rebuild.py"
        )
        assert result.returncode == 1
        assert "sic FAILED bucket_contract_invalid" in result.stdout
        assert not _target(w).exists()
    finally:
        make_writable(w["root"])


def test_the_validator_pin_matches_the_file_and_links_or_other_bytes_are_refused(
    tmp_path,
):
    data = (TOOLS / "bucket_compare.py").read_bytes()
    assert sr.VALIDATOR_SHA256 == hashlib.sha256(data).hexdigest()
    with pytest.raises(sr.SealError, match="validator_unpinned"):
        sr.Validator(data + b"\n# replaced\n", "test")
    repo = tmp_path / "repo"
    (repo / "tools" / "seal_buckets").mkdir(parents=True)
    (repo / "elsewhere.py").write_bytes(data)
    os.symlink(
        "../../elsewhere.py", repo / "tools" / "seal_buckets" / "bucket_compare.py"
    )
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": os.environ["PATH"],
    }
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "c"]):
        subprocess.run(
            ["git", *args], cwd=repo, check=True, env=env, capture_output=True
        )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(sr.SealError, match="validator_not_a_file"):
        sr.load_validator_from_git(str(repo / ".git"), sha)


# ---- B37-OPERATOR-LOCK / B37-OPERATOR-REVALIDATE: the operator commit is locked and re-verifies


def _pending_root(world, capsys) -> Path:
    assert sr.main(seal_args(world, "build1", "--only", "sic")) == 0
    capsys.readouterr()
    target = _target(world)
    sr.pending_marker(target).write_text(
        '{"verdict": "PENDING"}'
    )  # as left by a crash before commit
    return target


def _operator(world, *extra):
    return _run(
        [
            world["sha"],
            "build1",
            "--operator-commit",
            "sic",
            "--config",
            str(world["config"]),
            *extra,
        ]
    )


def test_the_operator_runbook_is_in_help_and_commits_a_clean_pending_root(
    world, capsys
):
    help_text = _run(["--help"]).stdout
    for needle in (
        "Operator runbook",
        ".seal_PENDING",
        ".seal_REFUSED",
        "--operator-commit",
        "REMOVE",
    ):
        assert needle in help_text, needle
    target = _pending_root(world, capsys)
    done = _operator(world)
    assert done.returncode == 0 and "sic COMMITTED PASS" in done.stdout, done.stdout
    assert not sr.pending_marker(target).exists()
    again = _operator(world)
    assert again.returncode == 1 and "sic FAILED root_not_pending" in again.stdout


def test_operator_ignores_live_run_locks(world, capsys):
    """Reviewer probe: while a seal run of this SHA and mode holds its locks, the operator is refused."""
    target = _pending_root(world, capsys)
    seal = sr.Seal(
        sr.load_config(world["config"]), world["sha"], "build1", False, ["sic"], {}
    )
    locks = sr.acquire_run_lock(seal)
    try:
        done = _operator(world)
    finally:
        locks.close()
    assert done.returncode == 1 and "sic FAILED seal_run_locked" in done.stdout
    assert sr.pending_marker(target).is_file()
    assert not sr.refusal_marker(
        target
    ).exists()  # a busy run is not a defect of the root


def test_operator_commits_invalid_pending_root(world, capsys):
    """Reviewer probe: a pending root whose derived_from source vanished is never committed; the
    refusal marker records why."""
    target = _pending_root(world, capsys)
    (world["databases"] / "sicx" / "medication" / "1.parquet").unlink()
    done = _operator(world)
    assert done.returncode == 1
    assert (
        "sic FAILED bucket_contract_invalid medication: bucket_derived_from_source_missing"
        in (done.stdout)
    )
    assert sr.pending_marker(target).is_file()
    refusal = json.loads(sr.refusal_marker(target).read_text())
    assert (
        refusal["reason"] == "bucket_contract_invalid"
        and refusal["by"] == "operator_commit"
    )
