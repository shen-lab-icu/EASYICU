"""delta57s: bound farm lineage uses the physical installation; synthetic sources only."""

from __future__ import annotations

import builtins
import json
import os
import shutil
import sys
from pathlib import Path

import duckdb
import pytest

from tests.support.seal_buckets import (
    make_bucket,
    make_world,
    make_writable,
    mock_flat_manifest_pin,
    seal_args,
)
from tools.seal_buckets import bucket_compare as bc
from tools.seal_buckets import seal_rebuild as sr


def _compare(monkeypatch, a, b, out):
    monkeypatch.setattr(
        sys, "argv", ["bucket_compare", str(a), str(b), "--out", str(out)]
    )
    with pytest.raises(SystemExit) as result:
        bc.main()
    return result.value.code, json.loads(out.read_text())


@pytest.fixture
def world(tmp_path, monkeypatch):
    mock_flat_manifest_pin(monkeypatch)
    w = make_world(tmp_path / "w")
    monkeypatch.setenv("SEAL_MIRROR", str(w["mirror"]))
    yield w
    make_writable(w["root"])


def _paths(w):
    seal = sr.Seal(sr.load_config(w["config"]), w["sha"], "build1", False, ["cv"], {})
    return (
        seal.farm_target("cv"),
        seal.target("cv"),
        seal.work / "build1/receipt_cv_farm.json",
    )


def _accept(w, farm, external):
    return sr.accept_root(farm, external, w["sha"], "build1", "cv")


def test_bound_bucket_only_farm_compares_and_accepts_without_easyicu(
    world, monkeypatch, tmp_path, capsys
):
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0
    farm, physical, external = _paths(world)
    inner = _accept(world, farm, external)
    assert inner["bucket_root_receipt_sha256"] == sr.sha256_file(
        physical / sr.IN_ROOT_RECEIPT
    )
    assert {"chartevents", "chartevents.parquet"}.isdisjoint(os.listdir(farm))
    real_import = builtins.__import__

    def no_reader(name, *args, **kwargs):
        assert name != "easyicu" and not name.startswith("easyicu.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_reader)
    # Another directory link ensures the anchor is realpath(bucket).parent, not realpath(farm).
    alias = tmp_path / "farm_alias"
    alias.symlink_to(farm, target_is_directory=True)
    code, result = _compare(
        monkeypatch,
        alias / "chartevents_bucket",
        physical / "chartevents_bucket",
        tmp_path / "compare.json",
    )
    assert code == 0 and result["verdict"] == "IDENTICAL"
    assert _accept(world, farm, external)["verdict"] == "PASS"
    capsys.readouterr()


@pytest.mark.parametrize("defect", ["missing", "size", "mtime"])
def test_comparator_ignores_a_matching_logical_source_decoy(
    tmp_path, monkeypatch, defect
):
    physical = tmp_path / "physical"
    bucket = make_bucket(physical)
    farm = tmp_path / "farm"
    farm.mkdir()
    (farm / "t_bucket").symlink_to(bucket, target_is_directory=True)
    # The old anchor accepts this byte/metadata-identical logical source even after the real source
    # has disappeared or changed. Never allow that decoy to supply the bucket's source identity.
    shutil.copytree(physical / "t", farm / "t", copy_function=shutil.copy2)
    source = physical / "t/1.parquet"
    if defect == "missing":
        source.unlink()
    elif defect == "size":
        source.write_bytes(source.read_bytes() + b"synthetic change")
    else:
        st = source.stat()
        os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    code, result = _compare(
        monkeypatch, farm / "t_bucket", farm / "t_bucket", tmp_path / "compare.json"
    )
    assert code == 1 and result["verdict"] == "INVALID"
    assert result["integrity"]["reason"] == (
        "bucket_derived_from_source_missing"
        if defect == "missing"
        else "bucket_derived_from_source_changed"
    )


def _damage(physical, defect, table="chartevents", source_path="chartevents.parquet"):
    make_writable(physical)
    source = physical / source_path
    if defect in {"absolute", "parent"}:
        manifest = physical / f"{table}_bucket/_BUCKET_MANIFEST.json"
        content = json.loads(manifest.read_text())
        content["derived_from"]["files"][0]["path"] = (
            str(source) if defect == "absolute" else "../outside.parquet"
        )
        sr.write_json(manifest, content)
    elif defect == "missing":
        source.unlink()
    else:
        target = source.resolve(strict=True)
        st = target.stat()
        target.chmod(0o600)
        if defect == "mtime":
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        elif defect == "size":
            target.write_bytes(target.read_bytes() + b"synthetic change")
        else:
            data = target.read_bytes()
            target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
        target.chmod(st.st_mode & 0o777)
    sr.make_read_only(physical, include_root=True)


@pytest.mark.parametrize("stage", ["before_freeze", "after_rename", "accept"])
@pytest.mark.parametrize(
    "defect", ["missing", "size", "mtime", "sha256", "absolute", "parent"]
)
def test_bound_farm_checks_physical_lineage_at_every_gate(
    world, monkeypatch, tmp_path, capsys, stage, defect
):
    farm, physical, external = _paths(world)
    reason = (
        "bucket_derived_from_invalid"
        if defect in {"absolute", "parent"}
        else "bucket_derived_from_source_missing"
        if defect == "missing"
        else "bucket_derived_from_source_changed"
    )
    frozen_farms = []
    exercised = []
    original_freeze = sr.make_read_only

    def freeze(root, include_root):
        if root.name == f".{farm.name}.seal_staging":
            frozen_farms.append(root)
        return original_freeze(root, include_root)

    monkeypatch.setattr(sr, "make_read_only", freeze)

    def exercise(root):
        # First prove this is the legitimate BOUND, bucket-only farm path. Every negative case
        # starts with the same valid linked install, not an unbound directory or a permissive stub.
        code, result = _compare(
            monkeypatch,
            root / "chartevents_bucket",
            physical / "chartevents_bucket",
            tmp_path / "good.json",
        )
        assert code == 0 and result["verdict"] == "IDENTICAL"
        _damage(physical, defect)
        exercised.append(stage)
        if (
            defect != "sha256"
        ):  # comparator retains its existing size/mtime identity contract
            code, result = _compare(
                monkeypatch,
                root / "chartevents_bucket",
                physical / "chartevents_bucket",
                tmp_path / "bad.json",
            )
            assert code == 1 and result["integrity"]["reason"] == reason

    if stage == "before_freeze":
        original = sr.install

        def install(staging, target, inner, verify=None):
            if inner.get("schema", "").startswith("lead_r13.seal_farm_receipt"):
                exercise(staging)
            return original(staging, target, inner, verify)

        monkeypatch.setattr(sr, "install", install)
    elif stage == "after_rename":
        original = sr.rename_noreplace

        def rename(staging, target):
            result = original(staging, target)
            if target == farm:
                exercise(target)
            return result

        monkeypatch.setattr(sr, "rename_noreplace", rename)
    rc = sr.main(seal_args(world, "build1", "--only", "cv"))
    if stage == "accept":
        assert rc == 0, capsys.readouterr().out
        assert _accept(world, farm, external)["verdict"] == "PASS"
        exercise(farm)
        with pytest.raises(sr.SealError) as exc:
            _accept(world, farm, external)
        assert exc.value.reason == reason
    else:
        assert rc == 1 and reason in capsys.readouterr().out
        if stage == "before_freeze":
            assert not frozen_farms and not farm.exists()
            assert not sr.pending_marker(farm).exists()
        else:
            assert json.loads(sr.refusal_marker(farm).read_text())["reason"] == reason
    assert exercised == [stage]


@pytest.mark.parametrize(
    "defect", ["missing", "size", "mtime", "sha256", "absolute", "parent", "unrecorded"]
)
def test_acceptance_checks_every_linked_bucket_not_only_the_dependency(
    world, capsys, defect
):
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0
    capsys.readouterr()
    farm, _, external = _paths(world)
    assert _accept(world, farm, external)["verdict"] == "PASS"
    extra = world["b6"] / "extra_install"
    bucket = make_bucket(extra)
    if defect != "unrecorded":
        _damage(extra, defect, table="t", source_path="t/1.parquet")
    sr.make_read_only(extra, include_root=True)
    make_writable(farm)
    (farm / "t_bucket").symlink_to(bucket, target_is_directory=True)
    inner = json.loads((farm / sr.IN_ROOT_RECEIPT).read_text())
    if defect != "unrecorded":
        inner["bucket_links"]["t_bucket"] = str(bucket)
    # Bind the complete snapshot so rejection must come from checking the extra bucket's own
    # lineage, rather than an unrelated stale receipt or a changed linked manifest's identity.
    inner["entries"] = sorted(n for n in os.listdir(farm) if n != sr.IN_ROOT_RECEIPT)
    inner["link_set"] = sr.link_set(farm, {sr.IN_ROOT_RECEIPT}, identities=True)
    digest = sr.write_json(farm / sr.IN_ROOT_RECEIPT, inner)
    sr.make_read_only(farm, include_root=True)
    receipt = json.loads(external.read_text())
    receipt["install"]["in_root_receipt_sha256"] = digest
    sr.write_json(external, receipt)
    with pytest.raises(sr.SealError) as exc:
        _accept(world, farm, external)
    assert exc.value.reason == (
        "farm_bucket_links_changed"
        if defect == "unrecorded"
        else "bucket_derived_from_invalid"
        if defect in {"absolute", "parent"}
        else "bucket_derived_from_source_missing"
        if defect == "missing"
        else "bucket_derived_from_source_changed"
    )


def test_carevue_origin_reraises_the_same_seal_error(world, monkeypatch):
    cfg = sr.load_config(world["config"])
    farm = Path(cfg["dbs"]["cv"]["current_root"])
    flat = Path(cfg["dbs"]["cv"]["build_input"])
    physical = world["b6"] / "mimic_cv"
    failure = sr.SealError("synthetic_origin_failure", "synthetic detail")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(duckdb, "connect", fail)
    with pytest.raises(sr.SealError) as exc:
        sr.compute_carevue_origin(farm, flat, physical)
    assert exc.value is failure and exc.value.reason == "synthetic_origin_failure"
    failure = RuntimeError("synthetic untyped failure")
    with pytest.raises(sr.SealError) as exc:
        sr.compute_carevue_origin(farm, flat, physical)
    assert exc.value.reason == "carevue_origin_query_failed"
