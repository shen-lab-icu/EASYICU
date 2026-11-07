"""delta44s: external tool trust, farm ownership and logical source identity. Synthetic only."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.seal_buckets import (
    make_world,
    make_writable,
    mock_flat_manifest_pin,
    seal_args,
)
from tools.seal_buckets import seal_rebuild as sr

TOOLS = Path(sr.__file__).parent


@pytest.fixture
def world(tmp_path, monkeypatch, request):
    mock_flat_manifest_pin(monkeypatch)
    w = make_world(tmp_path / "w", mode=getattr(request, "param", "ok"))
    monkeypatch.setenv("SEAL_MIRROR", str(w["mirror"]))
    yield w
    make_writable(w["root"])


def _external(w, farm=False, db="cv"):
    return (
        w["work"]
        / f"seal_{w['sha'][:9]}"
        / "build1"
        / f"receipt_{db}{'_farm' if farm else ''}.json"
    )


def _seal(w):
    return sr.Seal(sr.load_config(w["config"]), w["sha"], "build1", False, ["cv"], {})


def _build(w, capsys):
    assert sr.main(seal_args(w, "build1", "--only", "cv")) == 0
    capsys.readouterr()
    return _seal(w).farm_target("cv")


def _accept(w, farm):
    return sr.accept_root(farm, _external(w, True), w["sha"], "build1", "cv")


def _entry(w, *args, launcher=None):
    return subprocess.run(
        ["bash", str(launcher or TOOLS / "seal_rebuild.sh"), w["sha"], *args],
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "SEAL_PYTHON": sys.executable,
            "SEAL_MIRROR": str(w["mirror"]),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_standalone_acceptance_refuses_self_selected_validator_pin(
    world, tmp_path, capsys
):
    farm = _build(world, capsys)
    target = _seal(world).target("cv")
    part = next((target / "chartevents_bucket").rglob("*.parquet"))
    make_writable(target)
    data = part.read_bytes()
    part.write_bytes(b"NOPE" + data[4:-4] + b"NOPE")
    sr.make_read_only(target, include_root=True)
    with pytest.raises(sr.SealError, match="bucket_contract_invalid"):
        sr.accept_root(target, _external(world), world["sha"], "build1", "cv")
    copied = tmp_path / "copied"
    copied.mkdir()
    permissive = b"class StorageIntegrityError(Exception):\n    pass\ndef validate_contract(root):\n    return {}\n"
    (copied / "bucket_compare.py").write_bytes(permissive)
    shutil.copyfile(TOOLS / "footer_scan.py", copied / "footer_scan.py")
    (copied / "seal_rebuild.py").write_text(
        (TOOLS / "seal_rebuild.py")
        .read_text()
        .replace(sr.VALIDATOR_SHA256, hashlib.sha256(permissive).hexdigest())
    )
    spec = importlib.util.spec_from_file_location(
        "self_selected_sr", copied / "seal_rebuild.py"
    )
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    with pytest.raises(
        other.SealError, match="tool_not_sealed: tools/seal_buckets/seal_rebuild.py"
    ):
        other.accept_root(target, _external(world), world["sha"], "build1", "cv")
    assert farm.is_dir()


def test_standalone_acceptance_never_runs_a_caller_selected_validator(world, capsys):
    _build(world, capsys)
    target = _seal(world).target("cv")
    part = next((target / "chartevents_bucket").rglob("*.parquet"))
    make_writable(target)
    data = part.read_bytes()
    part.write_bytes(b"NOPE" + data[4:-4] + b"NOPE")
    sr.make_read_only(target, include_root=True)

    class CallerValidator:
        def check(self, bucket, table):
            return None

    with pytest.raises(sr.SealError, match="bucket_contract_invalid"):
        sr.accept_root(
            target,
            _external(world),
            world["sha"],
            "build1",
            "cv",
            validator=CallerValidator(),
        )


def test_standalone_acceptance_checks_the_receipt_tool_without_an_optional_pin(
    world, capsys
):
    farm = _build(world, capsys)
    assert _accept(world, farm)["verdict"] == "PASS"
    make_writable(farm)
    inner = json.loads((farm / sr.IN_ROOT_RECEIPT).read_text())
    inner["tool"] = {"wrong_version": "0" * 64}
    sr.write_json(farm / sr.IN_ROOT_RECEIPT, inner)
    sr.make_read_only(farm, include_root=True)
    ext = json.loads(_external(world, True).read_text())
    ext["install"]["in_root_receipt_sha256"] = sr.sha256_file(farm / sr.IN_ROOT_RECEIPT)
    sr.write_json(_external(world, True), ext)
    with pytest.raises(sr.SealError, match="receipt_tool_not_sealed"):
        _accept(world, farm)


def test_standalone_shell_entry_uses_sealed_sr_and_validator(world, tmp_path, capsys):
    # The shell runs the production CareVue pin; use SIC to test sealed code/validator trust
    # without substituting the synthetic CareVue pin in the extracted process.
    assert sr.main(seal_args(world, "build1", "--only", "sic")) == 0
    local = tmp_path / "local"
    local.mkdir()
    shutil.copyfile(TOOLS / "seal_rebuild.sh", local / "seal_rebuild.sh")
    sentinel = local / "executed"
    (local / "seal_rebuild.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\nraise SystemExit(99)\n"
    )
    (local / "bucket_compare.py").write_text("raise SystemExit(98)\n")
    args = [
        "accept",
        "--config",
        str(world["config"]),
        "--accept-root",
        str(_seal(world).target("sic")),
        "--external-receipt",
        str(_external(world, db="sic")),
        "--accept-build",
        "build1",
        "--accept-db",
        "sic",
    ]
    result = _entry(world, *args, launcher=local / "seal_rebuild.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sic ACCEPTED PASS" in result.stdout and not sentinel.exists()
    target = _seal(world).target("sic")
    part = next((target / "medication_bucket").rglob("*.parquet"))
    make_writable(target)
    data = part.read_bytes()
    part.write_bytes(b"NOPE" + data[4:-4] + b"NOPE")
    sr.make_read_only(target, include_root=True)
    result = _entry(world, *args, launcher=local / "seal_rebuild.sh")
    assert result.returncode == 1 and "bucket_contract_invalid" in result.stdout
    assert not sentinel.exists()


@pytest.mark.parametrize("mode", ["build1", "build2", "compare", "operator", "probe"])
def test_every_workload_entry_checks_all_executable_siblings(world, tmp_path, mode):
    copied = tmp_path / "copied"
    copied.mkdir()
    for name in ("seal_rebuild.py", "footer_scan.py"):
        shutil.copyfile(TOOLS / name, copied / name)
    (copied / "bucket_compare.py").write_text("raise SystemExit(99)\n")
    args = seal_args(
        world,
        mode if mode in ("build1", "build2", "compare") else "build1",
        "--only",
        "cv",
    )
    if mode == "operator":
        args += ["--operator-commit", "cv"]
    if mode == "probe":
        args += ["--dry-run", "--probe"]
    result = subprocess.run(
        [sys.executable, str(copied / "seal_rebuild.py"), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "tool_not_sealed tools/seal_buckets/bucket_compare.py" in result.stdout
    assert not world["work"].exists()


def test_no_sha_help_never_executes_a_local_sr(tmp_path):
    shutil.copyfile(TOOLS / "seal_rebuild.sh", tmp_path / "seal_rebuild.sh")
    sentinel = tmp_path / "executed"
    (tmp_path / "seal_rebuild.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
    )
    result = subprocess.run(
        ["bash", str(tmp_path / "seal_rebuild.sh"), "--help"],
        env={**os.environ, "SEAL_PYTHON": sys.executable},
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0 and "usage:" in result.stdout and not sentinel.exists()
    )


def test_operator_cross_mode_farm_is_refused_without_touching_build1(world, capsys):
    farm = _build(world, capsys)
    sr.pending_marker(farm).write_text('{"verdict":"PENDING"}')
    seal = _seal(world)
    locks = sr.acquire_run_lock(seal)
    before = {
        p: (p.lstat().st_mtime_ns, p.lstat().st_size) for p in world["b6"].iterdir()
    }
    pending = sr.pending_marker(farm).read_bytes()
    try:
        with pytest.raises(sr.SealError, match="operator_farm_mode_mismatch"):
            sr.operator_commit(
                sr.load_config(world["config"]), world["sha"], "build2", "cv", farm=True
            )
        assert {
            p: (p.lstat().st_mtime_ns, p.lstat().st_size) for p in world["b6"].iterdir()
        } == before
        assert sr.pending_marker(farm).read_bytes() == pending
        assert not sr.refusal_marker(farm).exists()
    finally:
        locks.close()


def test_operator_farm_locks_the_actual_target_and_checks_its_owner_scope(
    world, capsys, monkeypatch
):
    farm = _build(world, capsys)
    sr.pending_marker(farm).write_text('{"verdict":"PENDING"}')
    key = hashlib.sha256(os.fsencode(farm.resolve())).hexdigest()
    with open(farm.parent / f".seal_target_{key}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(sr.SealError, match="seal_run_locked"):
            sr.operator_commit(
                sr.load_config(world["config"]), world["sha"], "build1", "cv", farm=True
            )
    owners = []
    monkeypatch.setattr(
        sr, "scope_alive", lambda seal: owners.append(seal.mode) or True
    )
    with pytest.raises(sr.SealError, match="seal_scope_alive"):
        sr.operator_commit(
            sr.load_config(world["config"]), world["sha"], "build1", "cv", farm=True
        )
    assert owners == ["build1"] and sr.pending_marker(farm).is_file()
    assert not sr.refusal_marker(farm).exists()


def test_farm_receipt_records_every_logical_alias_and_cheap_target_identity(
    world, capsys
):
    current = Path(sr.load_config(world["config"])["dbs"]["cv"]["current_root"])
    (world["view"] / "large.bin").write_bytes(b"x" * 70000)
    os.symlink(world["view"], current / "alias")
    farm = _build(world, capsys)
    links = _accept(world, farm)["link_set"]
    for name in (
        "admissions.parquet",
        "inputevents_cv",
        "inputevents_cv/1.parquet",
        "alias",
        "alias/admissions.parquet",
        "alias/large.bin",
        "chartevents_bucket/bucket_id=0",
    ):
        record = links[name]
        assert {
            "type",
            "target_type",
            "raw_target",
            "resolved_target",
            "identity",
        } <= record.keys()
        assert {"size", "mtime_ns", "inode", "device"} <= record["identity"].keys()
    assert all(
        {"size", "mtime_ns", "inode", "device"} <= record["identity"].keys()
        for record in links.values()
    )
    assert links["admissions.parquet"]["raw_target"] == os.readlink(
        farm / "admissions.parquet"
    )
    assert links["admissions.parquet"]["resolved_target"] == str(
        (farm / "admissions.parquet").resolve()
    )
    assert "sha256" in links["alias/admissions.parquet"]["identity"]
    assert "sha256" not in links["alias/large.bin"]["identity"]
    assert "chartevents_bucket/bucket_id=0/part.parquet" in links


@pytest.mark.parametrize(
    "logical",
    [
        "admissions.parquet",
        "inputevents_cv/1.parquet",
        "alias/admissions.parquet",
        "raw_spelling",
    ],
)
def test_farm_nonbucket_link_retargeted_is_refused(world, capsys, logical):
    current = Path(sr.load_config(world["config"])["dbs"]["cv"]["current_root"])
    alias = world["view"] / "alias"
    alias.mkdir()
    os.symlink(world["view"] / "admissions.parquet", alias / "admissions.parquet")
    os.symlink(alias, current / "alias")
    farm = _build(world, capsys)
    _accept(world, farm)
    make_writable(farm)
    path = farm / ("admissions.parquet" if logical == "raw_spelling" else logical)
    raw = os.readlink(path)
    path.unlink()
    os.symlink(
        raw + "/../admissions.parquet"
        if logical == "raw_spelling"
        else world["view"]
        / (
            "admissions.parquet"
            if logical == "inputevents_cv/1.parquet"
            else "inputevents_cv.parquet"
        ),
        path,
    )
    # Use a different valid raw spelling resolving to the same file.
    if logical == "raw_spelling":
        path.unlink()
        os.symlink(str(world["view"]) + "/./admissions.parquet", path)
    sr.make_read_only(farm, include_root=True)
    with pytest.raises(sr.SealError, match="farm_links_changed"):
        _accept(world, farm)


@pytest.mark.parametrize("change", ["replaced", "moved", "modified"])
def test_farm_raw_source_identity_change_is_refused(world, capsys, change):
    farm = _build(world, capsys)
    _accept(world, farm)
    source = world["view"] / "admissions.parquet"
    old = source.stat()
    data = source.read_bytes()
    if change == "replaced":
        replacement = source.with_suffix(".new")
        replacement.write_bytes(data)
        os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
        replacement.replace(source)
    elif change == "moved":
        source.rename(source.with_suffix(".moved"))
    else:
        # Equal size and restored mtime: the small-file digest still binds the bytes.
        source.write_bytes(data[:-4] + b"NOPE")
        os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(sr.SealError, match="farm_links_changed"):
        _accept(world, farm)


def test_farm_link_swap_before_freeze_is_refused(world, capsys, monkeypatch):
    original = sr.install

    def swap(staging, target, inner, verify=None):
        if inner.get("schema", "").startswith("lead_r13.seal_farm_receipt"):
            (staging / "admissions.parquet").unlink()
            os.symlink(
                world["view"] / "inputevents_cv.parquet", staging / "admissions.parquet"
            )
        return original(staging, target, inner, verify)

    monkeypatch.setattr(sr, "install", swap)
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 1
    assert "farm_links_changed" in capsys.readouterr().out
    farm = _seal(world).farm_target("cv")
    assert not farm.exists() and not sr.pending_marker(farm).exists()


def _copied_tools(tmp_path, monkeypatch):
    copied = tmp_path / "copied_tools"
    copied.mkdir()
    for name in ("seal_rebuild.py", "footer_scan.py", "bucket_compare.py"):
        shutil.copyfile(TOOLS / name, copied / name)
    monkeypatch.setattr(sr, "__file__", str(copied / "seal_rebuild.py"))
    return copied


@pytest.mark.parametrize(
    "name,world,expected_rc",
    [
        ("footer_scan.py", "ok", 0),
        ("bucket_compare.py", "ok", 0),
        ("bucket_compare.py", "nondeterministic", 1),
    ],
    indirect=["world"],
)
def test_helpers_execute_the_verified_private_copy_after_a_sibling_swap(
    world, tmp_path, monkeypatch, capsys, name, expected_rc
):
    if name == "bucket_compare.py":
        for mode in ("build1", "build2"):
            assert sr.main(seal_args(world, mode, "--only", "cv")) == 0
        capsys.readouterr()
    copied = _copied_tools(tmp_path, monkeypatch)
    sentinel = copied / "executed"
    original = sr.confined_run
    executed = []

    def swap(seal, db, unit, argv, *args):
        if len(argv) > 2 and Path(argv[2]).name == name:
            # The entry's origin check already passed. A stale sibling arriving now must never run.
            (copied / name).write_text(
                f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
                "raise SystemExit(0)\n"
            )
            executed.append(Path(argv[2]))
        return original(seal, db, unit, argv, *args)

    monkeypatch.setattr(sr, "confined_run", swap)
    mode = "compare" if name == "bucket_compare.py" else "build1"
    rc = sr.main(seal_args(world, mode, "--only", "cv"))
    assert not sentinel.exists(), "a replaced sibling helper was executed"
    output = capsys.readouterr().out
    assert rc == expected_rc, output
    if expected_rc:
        assert "build1_vs_build2_differs" in output
    assert executed and all(p.is_relative_to(world["work"]) for p in executed)
    assert all(sr.sha256_file(p) == sr.sha256_file(TOOLS / name) for p in executed)
    report = json.loads(
        (
            world["work"]
            / f"seal_{world['sha'][:9]}"
            / mode
            / (
                "compare_cv_chartevents.json"
                if mode == "compare"
                else "cv_footer_scan.json"
            )
        ).read_text()
    )
    assert report["tool_sha256"] == sr.sha256_file(TOOLS / name)


def test_farm_acceptance_binds_its_dependency_receipt(world, capsys):
    farm = _build(world, capsys)
    root = _seal(world).target("cv")
    make_writable(root)
    inner_path = root / sr.IN_ROOT_RECEIPT
    inner = json.loads(inner_path.read_text())
    inner["different_receipt"] = True
    sr.write_json(inner_path, inner)
    sr.make_read_only(root, include_root=True)
    with pytest.raises(sr.SealError, match="farm_dependency_changed"):
        _accept(world, farm)


@pytest.mark.parametrize("helper", ["footer_scan.py", "bucket_compare.py"])
def test_a_success_exit_cannot_replace_a_pinned_helper_report(
    world, monkeypatch, capsys, helper
):
    if helper == "bucket_compare.py":
        for mode in ("build1", "build2"):
            assert sr.main(seal_args(world, mode, "--only", "cv")) == 0
        capsys.readouterr()
    original = sr.confined_run

    def wrong_report(seal, db, unit, argv, *args):
        result = original(seal, db, unit, argv, *args)
        if len(argv) > 2 and Path(argv[2]).name == helper:
            report_path = Path(argv[argv.index("--out") + 1])
            report = json.loads(report_path.read_text())
            report["tool_sha256"] = "0" * 64
            sr.write_json(report_path, report)
        return result

    monkeypatch.setattr(sr, "confined_run", wrong_report)
    mode = "compare" if helper == "bucket_compare.py" else "build1"
    assert sr.main(seal_args(world, mode, "--only", "cv")) == 1
    assert (
        "compare_report_unpinned" if mode == "compare" else "footer_scan_unpinned"
    ) in capsys.readouterr().out


@pytest.mark.parametrize("farm", [False, True])
def test_acceptance_verifies_the_receipts_validator_pin(world, capsys, farm):
    farm_root = _build(world, capsys)
    root = farm_root if farm else _seal(world).target("cv")
    make_writable(root)
    inner = json.loads((root / sr.IN_ROOT_RECEIPT).read_text())
    inner["validator"]["sha256"] = "0" * 64
    sr.write_json(root / sr.IN_ROOT_RECEIPT, inner)
    sr.make_read_only(root, include_root=True)
    external = _external(world, farm)
    ext = json.loads(external.read_text())
    ext["install"]["in_root_receipt_sha256"] = sr.sha256_file(root / sr.IN_ROOT_RECEIPT)
    sr.write_json(external, ext)
    with pytest.raises(sr.SealError, match="validator_unpinned"):
        sr.accept_root(root, external, world["sha"], "build1", "cv")


def _change_bucket_root(root, change, tmp_path):
    make_writable(root)
    if change == "extra_bucket":
        (root / "admissions_bucket").mkdir()
    elif change == "conversion_manifest":
        (root / sr.CONVERSION_MANIFEST).write_text('{"tampered":true}')
    else:
        link = root / "medication" / "1.parquet"
        source = link.resolve()
        if change == "raw_link":
            target = str(source.parent) + "/./" + source.name
        else:
            target = tmp_path / "same_content.parquet"
            shutil.copy2(source, target)
        link.unlink()
        os.symlink(target, link)
    sr.make_read_only(root, include_root=True)


@pytest.mark.parametrize("stage", ["before_freeze", "after_rename", "accept"])
@pytest.mark.parametrize(
    "change,reason",
    [
        ("extra_bucket", "bucket_root_entries_mismatch"),
        ("conversion_manifest", "bucket_conversion_manifest_changed"),
        ("raw_link", "bucket_links_changed"),
        ("resolved_link", "bucket_links_changed"),
    ],
)
def test_bucket_root_entries_manifest_and_links_are_bound_at_every_gate(
    world, tmp_path, capsys, monkeypatch, stage, change, reason
):
    # The alternate link target is allowed and has identical bytes/size/mtime. Containment and the
    # existing derived_from metadata checks alone therefore cannot detect the wrong source path.
    cfg = sr.load_config(world["config"])
    cfg["allowed_roots"].append(str(tmp_path))
    sr.write_json(world["config"], cfg)
    if stage == "before_freeze":
        original = sr.install

        def install(staging, target, inner, verify=None):
            _change_bucket_root(staging, change, tmp_path)
            # Let the real install write its receipt before freezing the root itself.
            staging.chmod(0o755)
            return original(staging, target, inner, verify)

        monkeypatch.setattr(sr, "install", install)
    elif stage == "after_rename":
        original = sr.rename_noreplace

        def rename(staging, target):
            result = original(staging, target)
            _change_bucket_root(target, change, tmp_path)
            return result

        monkeypatch.setattr(sr, "rename_noreplace", rename)
    rc = sr.main(seal_args(world, "build1", "--only", "sic"))
    root = sr.Seal(cfg, world["sha"], "build1", False, ["sic"], {}).target("sic")
    if stage == "accept":
        assert rc == 0, capsys.readouterr().out
        _change_bucket_root(root, change, tmp_path)
        with pytest.raises(sr.SealError, match=reason):
            sr.accept_root(
                root, _external(world, db="sic"), world["sha"], "build1", "sic"
            )
    else:
        assert rc == 1
        assert reason in capsys.readouterr().out
        assert (
            not root.exists()
            if stage == "before_freeze"
            else sr.refusal_marker(root).is_file()
        )


def test_an_unowned_operator_receipt_never_marks_the_target(world, capsys):
    farm = _build(world, capsys)
    sr.pending_marker(farm).write_text('{"verdict":"PENDING"}')
    external = _external(world, True)
    ext = json.loads(external.read_text())
    ext["db"] = "sic"
    sr.write_json(external, ext)
    with pytest.raises(sr.SealError, match="receipt_unbound"):
        sr.operator_commit(
            sr.load_config(world["config"]), world["sha"], "build1", "cv", farm=True
        )
    assert not sr.refusal_marker(farm).exists()
