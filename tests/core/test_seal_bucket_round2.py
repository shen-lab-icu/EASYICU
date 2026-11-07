"""Seal-tip bucket tool, review round 2 (delta26t): one regression per B finding.

Each test provokes the counterexample the review demonstrated against eafd369eb and asserts the typed
refusal. They run the real tools against synthetic worlds (tests/support/seal_buckets.py): no clinical data.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.seal_buckets import (
    make_bucket,
    make_world,
    make_writable,
    seal_args,
    synthetic_seal_argv,
)
from tools.seal_buckets import footer_scan, negzero_table
from tools.seal_buckets import seal_rebuild as sr

TOOL = Path(sr.__file__)
TOOLS = TOOL.parent


def _run(args):
    return subprocess.run(
        synthetic_seal_argv(TOOL, args),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _tool(name: str, *args: str):
    return subprocess.run(
        [sys.executable, "-I", str(TOOLS / name), *args], capture_output=True, text=True
    )


@pytest.fixture
def world(tmp_path):
    w = make_world(tmp_path / "w")
    yield w
    make_writable(w["root"])


def _target(world, name, mode="build1"):
    return (
        world["b6"]
        / f"{name}_s{world['sha'][:9]}{'_build2' if mode == 'build2' else ''}"
    )


def _work(world, mode):
    return world["work"] / f"seal_{world['sha'][:9]}" / mode


# ---- D26-EXIT: success is a structured verdict, never a word found in failure text


def test_failure_text_containing_a_success_word_still_exits_non_zero(world):
    args = seal_args(world, "build1", "--dry-run")
    args[0] = "PASS"  # an unknown SHA whose detail text is a success word
    result = _run(args)
    assert "all FAILED seal_sha_unknown PASS" in result.stdout
    assert result.returncode == 1


# ---- D26-COMPARE-SET: each side is validated against its own inventory before the twins are compared


def test_a_part_missing_on_both_sides_is_an_integrity_failure_not_identical(tmp_path):
    a, b = make_bucket(tmp_path / "a"), make_bucket(tmp_path / "b")
    for root in (a, b):
        (root / "bucket_id=1" / "part.parquet").unlink()
    result = _tool(
        "bucket_compare.py", str(a), str(b), "--out", str(tmp_path / "c.json")
    )
    out = json.loads((tmp_path / "c.json").read_text())
    assert result.returncode == 1
    assert out["verdict"] == "INVALID"
    assert out["integrity"]["reason"] == "bucket_inventory_mismatch"
    assert out["integrity"]["detail"] == "bucket_id=1/part.parquet"


def test_compare_accepts_roots_only_through_bound_in_root_and_farm_receipts(world):
    assert _run(seal_args(world, "build1")).returncode == 0
    assert _run(seal_args(world, "build2")).returncode == 0
    assert _run(seal_args(world, "compare")).returncode == 0
    # the CareVue farm receipt is part of formal acceptance
    farm_receipt = _work(world, "build1") / "receipt_cv_farm.json"
    saved = farm_receipt.read_text()
    farm_receipt.write_text(json.dumps(dict(json.loads(saved), verdict="FAILED")))
    result = _run(seal_args(world, "compare"))
    assert result.returncode == 1 and "cv FAILED receipt_not_pass" in result.stdout
    farm_receipt.write_text(saved)
    # an in-root receipt that no longer matches the bound sha256 is refused
    root = _target(world, "sic_cur", "build2")
    make_writable(root)
    inner = root / sr.IN_ROOT_RECEIPT
    inner.write_text(json.dumps(dict(json.loads(inner.read_text()), note="edited")))
    sr.make_read_only(root, include_root=True)
    result = _run(seal_args(world, "compare"))
    assert (
        result.returncode == 1 and "sic FAILED in_root_receipt_changed" in result.stdout
    )


# ---- D26-INSTALL: a failure after the rename fails the receipt and leaves a persistent refusal


def test_a_verification_failure_after_the_rename_is_failed_and_refused(
    world, monkeypatch, capsys
):
    target = _target(world, "sic_cur")
    real = sr.writable_entries

    def after_rename(root):
        return ["."] if Path(root) == target else real(root)

    monkeypatch.setattr(sr, "writable_entries", after_rename)
    code = sr.main(seal_args(world, "build1", "--only", "sic"))
    capsys.readouterr()
    receipt = json.loads((_work(world, "build1") / "receipt_sic.json").read_text())
    assert code == 1
    assert receipt["verdict"] == "FAILED"
    assert (target.parent / f".{target.name}{sr.REFUSAL_SUFFIX}").is_file()
    with pytest.raises(sr.SealError):
        sr.accept_root(
            target,
            _work(world, "build1") / "receipt_sic.json",
            world["sha"],
            "build1",
            "sic",
        )


# ---- D26-ISOLATION: a checkout is never reused, and links cannot hide the other build's paths


def test_build2_cannot_reuse_build1_checkout_through_a_link(world):
    assert _run(seal_args(world, "build1", "--only", "sic")).returncode == 0
    code2 = _work(world, "build2") / "code"
    code2.parent.mkdir(parents=True)
    os.symlink(_work(world, "build1") / "code", code2)
    result = _run(seal_args(world, "build2", "--only", "sic"))
    assert result.returncode == 1 and "all FAILED checkout_exists" in result.stdout
    assert not _target(world, "sic_cur", "build2").exists()


def test_isolation_resolves_links_before_comparing_paths(tmp_path):
    other = tmp_path / "build1_scratch"
    other.mkdir()
    alias = tmp_path / "innocent_name"
    os.symlink(other, alias)
    with pytest.raises(sr.SealError, match="build_env_crosses_builds"):
        sr.check_isolation({"TMPDIR": str(alias / "tmp")}, [str(other)])


# ---- D26-CONCURRENCY: one run per SHA and mode; scratch is claimed, and only one's own is removed


def test_a_second_entry_for_the_same_sha_and_mode_is_refused(world):
    world["work"].mkdir(parents=True, exist_ok=True)
    with open(world["work"] / f".seal_{world['sha'][:9]}_build1.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 1 and "all FAILED seal_run_locked" in result.stdout


def test_another_runs_scratch_is_refused_and_left_alone(world):
    scratch = world["b6"] / "_scratch" / f"seal_{world['sha'][:9]}" / "build1" / "sic"
    scratch.mkdir(parents=True)
    (scratch / "active.tmp").write_text("another run\n")
    result = _run(seal_args(world, "build1", "--only", "sic"))
    assert result.returncode == 1 and "sic FAILED scratch_exists" in result.stdout
    assert (scratch / "active.tmp").is_file()
    assert not _target(world, "sic_cur").exists()


# ---- D26-CONVERSION: the reserved manifest name is dispatched first, link or not, and re-verified


def test_a_linked_conversion_manifest_is_still_verified_and_copied(world):
    sic = Path(json.loads(world["config"].read_text())["dbs"]["sic"]["current_root"])
    manifest = json.loads((sic / sr.CONVERSION_MANIFEST).read_text())
    manifest["derived_from"]["sha256"] = "0" * 64
    linked = world["databases"] / "linked_manifest.json"
    linked.write_text(json.dumps(manifest))
    (sic / sr.CONVERSION_MANIFEST).unlink()
    os.symlink(linked, sic / sr.CONVERSION_MANIFEST)
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert result.returncode == 1
    assert "sic FAILED conversion_manifest_derived_from_changed" in result.stdout


def test_a_linked_raw_manifest_is_pinned_and_copied_as_an_owned_file(world, tmp_path):
    """A production root may link its raw database's manifest (no derived_from; data_path names the
    raw directory). It plans, is pinned by sha256, is copied as an owned file that keeps every field
    and declares the pinned source as its view (R13-SEAL-VIEW), and a target that changes after
    planning is refused."""
    sic = Path(json.loads(world["config"].read_text())["dbs"]["sic"]["current_root"])
    raw = world["databases"] / "raw_sic"
    raw.mkdir()
    (raw / sr.CONVERSION_MANIFEST).write_text(
        json.dumps({"db": "sic", "data_path": str(raw)})
    )
    (sic / sr.CONVERSION_MANIFEST).unlink()
    os.symlink(raw / sr.CONVERSION_MANIFEST, sic / sr.CONVERSION_MANIFEST)
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert "sic FAILED" not in result.stdout, result.stdout

    pinned = sr.manifest_source(sic / sr.CONVERSION_MANIFEST)
    assert pinned["linked"] and pinned["source"] == os.path.realpath(
        raw / sr.CONVERSION_MANIFEST
    )
    dest = tmp_path / "copy.json"
    sr.write_conversion_manifest(
        sic / sr.CONVERSION_MANIFEST, dest, sic, tmp_path / "new", pinned
    )
    assert not dest.is_symlink()
    source = json.loads((raw / sr.CONVERSION_MANIFEST).read_text())
    assert json.loads(dest.read_text()) == dict(
        source, derived_from={"path": pinned["source"], "sha256": pinned["sha256"]}
    )

    (raw / sr.CONVERSION_MANIFEST).write_text(
        json.dumps({"db": "sic", "data_path": "/elsewhere"})
    )
    with pytest.raises(sr.SealError, match="conversion_manifest_source_changed"):
        sr.write_conversion_manifest(
            sic / sr.CONVERSION_MANIFEST, dest, sic, tmp_path / "new", pinned
        )
    with pytest.raises(sr.SealError, match="conversion_manifest_data_path_unexpected"):
        sr.rewritten_conversion_manifest(
            sic / sr.CONVERSION_MANIFEST, sic, tmp_path / "new"
        )


# ---- D26-LINK-CLOSURE: everything reachable through an allowed directory link stays inside


def test_a_directory_link_whose_descendant_escapes_is_refused(world, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    pq_bytes = (world["databases"] / "sicx" / "medication" / "1.parquet").read_bytes()
    (outside / "x.parquet").write_bytes(pq_bytes)
    inner = world["databases"] / "sub"
    inner.mkdir()
    os.symlink(outside, inner / "escape")
    sic = Path(json.loads(world["config"].read_text())["dbs"]["sic"]["current_root"])
    os.symlink(inner, sic / "extra")
    result = _run(seal_args(world, "build1", "--dry-run"))
    assert (
        result.returncode == 1
        and "sic FAILED link_outside_allowed_roots extra" in result.stdout
    )
    # footer_scan shares the containment contract
    shutil.rmtree(tmp_path / "scanroot", ignore_errors=True)
    root = tmp_path / "scanroot"
    root.mkdir()
    os.symlink(inner, root / "extra")
    scan = _tool(
        "footer_scan.py",
        str(root),
        "--out",
        str(tmp_path / "s.json"),
        "--allowed-root",
        str(world["databases"]),
    )
    out = json.loads((tmp_path / "s.json").read_text())
    assert scan.returncode == 1
    assert {"path": "extra/escape", "reason": "path_escapes_allowed_roots"} in out[
        "failures"
    ]
    assert footer_scan.scan  # the module is importable (no side effects)


# ---- D26-NEGZERO: identity, verdict and set equality before any profile comparison


def _nz_receipt(
    path: Path, db="aumc", build="build1", verdict="PASS", seal="f" * 40, tables=("t",)
):
    profile = {
        "v": {
            "values": 10,
            "nulls": 0,
            "neg_zero": 3,
            "pos_zero": 2,
            "nan": 0,
            "sha256": "a" * 64,
        }
    }
    path.write_text(
        json.dumps(
            {
                "db": db,
                "build": build,
                "verdict": verdict,
                "seal_sha": seal,
                "tables": {t: {"float_fidelity": profile} for t in tables},
            }
        )
    )
    return str(path)


def test_negzero_checks_build2_identity_verdict_and_sets(tmp_path, monkeypatch, capsys):
    def run(*args):
        monkeypatch.setattr(sys, "argv", ["negzero_table.py", *args])
        return negzero_table.main()

    b1 = _nz_receipt(tmp_path / "b1.json")
    ok = _nz_receipt(tmp_path / "ok.json", build="build2")
    assert run(b1, "--build2", ok, "--allow-pending") == 0
    failed = _nz_receipt(tmp_path / "f.json", build="build2", verdict="FAILED")
    assert run(b1, "--build2", failed, "--allow-pending") == 1
    other_sha = _nz_receipt(tmp_path / "s.json", build="build2", seal="e" * 40)
    assert run(b1, "--build2", other_sha, "--allow-pending") == 1
    extra_table = _nz_receipt(tmp_path / "x.json", build="build2", tables=("t", "u"))
    assert run(b1, "--build2", extra_table, "--allow-pending") == 1
    wrong_build = _nz_receipt(tmp_path / "w.json", build="build1")
    assert run(b1, "--build2", wrong_build, "--allow-pending") == 1
    assert run(b1, b1, "--build2", ok, "--allow-pending") == 1  # duplicate DB
    capsys.readouterr()


# ---- import origin: the builder runs the seal clone's easyicu, asserted and recorded


def test_the_builder_import_origin_is_asserted_and_recorded(
    world, monkeypatch, capsys, tmp_path
):
    assert sr.main(seal_args(world, "build1", "--only", "sic")) == 0
    capsys.readouterr()
    receipt = json.loads((_work(world, "build1") / "receipt_sic.json").read_text())
    code = _work(world, "build1") / "code"
    for table in ("medication", "laboratory"):
        origin = receipt["import_origin"][table]
        assert origin["head"] == world["sha"]
        assert origin["path"] == os.path.realpath(
            code / "src" / "easyicu" / "__init__.py"
        )
    # a decoy easyicu ahead of the clone's src cannot win (the boot sets the path explicitly), and a
    # foreign easyicu submodule loaded by any other means is refused, typed (D29-BOOT-HOOK)
    decoy = tmp_path / "decoy" / "src"
    (decoy / "easyicu").mkdir(parents=True)
    (decoy / "easyicu" / "__init__.py").write_text("")
    (decoy / "easyicu" / "storage_layout.py").write_text("FLAG = 'foreign'\n")
    real_env = sr.build_env

    def decoyed(seal, db, code):
        env = real_env(seal, db, code)
        env["PYTHONPATH"] = f"{decoy}{os.pathsep}{env['PYTHONPATH']}"
        env["FAKE_PRELOAD"] = str(decoy / "easyicu" / "storage_layout.py")
        return env

    monkeypatch.setattr(sr, "build_env", decoyed)
    assert sr.main(seal_args(world, "build2", "--only", "sic")) == 1
    out = capsys.readouterr().out
    assert "sic FAILED builder_import_origin_mismatch" in out
    failed = json.loads((_work(world, "build2") / "receipt_sic.json").read_text())
    assert failed["verdict"] == "FAILED"
    origin = failed["import_origin"]["medication"]
    assert origin["path"] == os.path.realpath(
        _work(world, "build2") / "code" / "src" / "easyicu" / "__init__.py"
    )
    assert origin["mismatched"] == ["easyicu.storage_layout"]
    assert not _target(world, "sic_cur", "build2").exists()


def test_the_scan_compare_and_negzero_tools_never_import_easyicu():
    import ast

    for name in ("footer_scan.py", "bucket_compare.py", "negzero_table.py"):
        tree = ast.parse((TOOLS / name).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "easyicu" not in imported, name
