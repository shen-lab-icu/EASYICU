"""delta57s: bound farm lineage uses the physical installation; synthetic sources only."""

from __future__ import annotations

import builtins
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


def _pass_through(world, location="b6", relative=False, linked_source=False):
    """delta58s extra_probe: a complete source/bucket pair, with actual DuckDB hash partitions."""
    bucket = make_bucket(world[location] / "extra_install")
    source = bucket.parent / "t/1.parquet"
    flat = pa.table(
        {
            "k": pa.array(range(40), pa.int64()),
            "v": pa.array(np.where(np.arange(40) % 3 == 0, -0.0, 1.5), pa.float64()),
        }
    )
    pq.write_table(flat, source, use_dictionary=False)
    if linked_source:
        linked = world["databases"] / "extra_source.parquet"
        shutil.copy2(source, linked)
        linked.chmod(0o444)
        source.unlink()
        source.symlink_to(linked)
    manifest = json.loads((bucket / "_BUCKET_MANIFEST.json").read_text())
    inventory = []
    with duckdb.connect(":memory:", config={"temp_directory": "", "threads": 1}) as con:
        con.read_parquet(str(source)).create_view("f")
        for b in range(2):
            table = con.execute(
                "SELECT * FROM (SELECT *, row_number() OVER ()-1 AS _easyicu_source_ordinal "
                "FROM f) WHERE hash(k)%2=? ORDER BY k, _easyicu_source_ordinal",
                [b],
            ).fetch_arrow_table()
            part = bucket / f"bucket_id={b}/part.parquet"
            pq.write_table(table, part, use_dictionary=False)
            inventory.append(
                {
                    "relative_path": f"bucket_id={b}/part.parquet",
                    "row_count": table.num_rows,
                    "size": part.stat().st_size,
                }
            )
        con.read_parquet(
            [str(bucket / e["relative_path"]) for e in inventory]
        ).create_view("b")
        assert (
            con.execute(
                "SELECT count(*) FROM ((SELECT k,v FROM b EXCEPT ALL SELECT * FROM f) "
                "UNION ALL (SELECT * FROM f EXCEPT ALL SELECT k,v FROM b))"
            ).fetchone()[0]
            == 0
        )
        ordered = con.execute(
            "SELECT k,v FROM b ORDER BY _easyicu_source_ordinal"
        ).fetch_arrow_table()
        assert all(
            np.array_equal(
                ordered[n].to_numpy().view(np.uint64),
                flat[n].to_numpy().view(np.uint64),
            )
            for n in flat.column_names
        )
    manifest["inventory"] = inventory
    manifest["derived_from"]["files"] = [
        {
            "path": "t/1.parquet",
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "sha256": sr.sha256_file(source),
        }
    ]
    sr.write_json(bucket / "_BUCKET_MANIFEST.json", manifest)
    bc.validate_contract(bucket)
    sr.make_read_only(bucket.parent, include_root=True)
    current = Path(sr.load_config(world["config"])["dbs"]["cv"]["current_root"])
    target = os.path.relpath(bucket, current) if relative else str(bucket)
    (current / "t_bucket").symlink_to(target, target_is_directory=True)
    conversion = json.loads((current / sr.CONVERSION_MANIFEST).read_text())
    conversion["tables"].append({"table": "t"})
    sr.write_json(current / sr.CONVERSION_MANIFEST, conversion)
    return bucket, current, target


@pytest.mark.parametrize(
    "location,relative,linked_source",
    [
        ("b6", False, False),
        ("b6", True, False),
        ("databases", False, False),
        ("b6", False, True),
    ],
)
def test_pass_through_bucket_build_publish_and_standalone_accept(
    world, monkeypatch, tmp_path, capsys, location, relative, linked_source
):
    bucket, _, target = _pass_through(world, location, relative, linked_source)
    planned = []
    original = sr.plan_db

    def planning(seal, db):
        plan = original(seal, db)
        if not seal.dry_run:
            planned.append(plan)
        return plan

    monkeypatch.setattr(sr, "plan_db", planning)
    before = sr.link_set(bucket.parent, set(), identities=True)
    assert sr.main(seal_args(world, "build1", "--only", "cv", "--dry-run")) == 0
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
    farm, physical, external = _paths(world)
    inner = _accept(world, farm, external)
    bindings = inner["bucket_links"]
    assert set(bindings) == {"t_bucket", "chartevents_bucket"}
    assert bindings == planned[0]["bucket_links"]
    assert bindings["t_bucket"]["kind"] == "pass_through"
    assert bindings["t_bucket"]["target"] == target == os.readlink(farm / "t_bucket")
    assert bindings["t_bucket"]["source"] == str(bucket)
    assert bindings["t_bucket"]["contract"]["derived_from"]["files"][0]["sha256"]
    assert bindings["chartevents_bucket"]["kind"] == "rebuilt"
    assert bindings["chartevents_bucket"]["source"] == str(
        physical / "chartevents_bucket"
    )
    assert bindings["chartevents_bucket"]["contract"]["manifest_sha256"]
    assert json.loads(external.read_text())["bucket_links"] == bindings
    assert sr.link_set(bucket.parent, set(), identities=True) == before
    assert not sr.pending_marker(farm).exists() and not sr.refusal_marker(farm).exists()
    code, result = _compare(
        monkeypatch, farm / "t_bucket", bucket, tmp_path / "extra.json"
    )
    assert code == 0 and result["verdict"] == "IDENTICAL"
    assert _accept(world, farm, external)["verdict"] == "PASS"
    repo = Path(sr.__file__).resolve().parents[2]
    claimed_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    # A fresh CLI process uses the same sole production-data mock as the parent fixture. The
    # Git-bound tool bytes, validator pin, farm receipts and source contracts remain real checks.
    boot = """
import subprocess, sys
from pathlib import Path
import easyicu
origin = Path(easyicu.__file__).resolve()
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(f"easyicu_realpath={origin}\\nHEAD={head}", flush=True)
assert origin.is_relative_to(Path(sys.argv.pop(1)) / "src")
assert head == sys.argv.pop(1)
from tools.seal_buckets import seal_rebuild as sr
sr.CV_FLAT_MANIFEST_SHA256 = sys.argv.pop(1)
raise SystemExit(sr.main())
"""
    accepted = subprocess.run(
        [
            sys.executable,
            "-c",
            boot,
            str(repo),
            claimed_head,
            sr.CV_FLAT_MANIFEST_SHA256,
            world["sha"],
            "accept",
            "--accept-root",
            str(farm),
            "--external-receipt",
            str(external),
            "--accept-build",
            "build1",
            "--accept-db",
            "cv",
            "--config",
            str(world["config"]),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert accepted.returncode == 0 and "cv ACCEPTED PASS" in accepted.stdout, (
        accepted.stdout + accepted.stderr
    )


@pytest.mark.parametrize("stage", ["source_after_plan", "before_freeze", "accept"])
def test_pass_through_does_not_admit_unplanned_bucket_links(
    world, monkeypatch, capsys, stage
):
    bucket, current, _ = _pass_through(world)
    farm, _, external = _paths(world)
    exercised = []

    def add_link(root):
        (root / "unplanned_bucket").symlink_to(bucket, target_is_directory=True)
        exercised.append(stage)

    if stage == "source_after_plan":
        original = sr.plan_db

        def planning(seal, db):
            plan = original(seal, db)
            assert plan["bucket_links"]["t_bucket"]["kind"] == "pass_through"
            add_link(current)
            return plan

        monkeypatch.setattr(sr, "plan_db", planning)
    elif stage == "before_freeze":
        original = sr.install

        def install(staging, target, inner, verify=None):
            if target == farm:
                # Positive control: the planned extra bucket is valid before the unplanned addition.
                sr.check_farm_bucket_sources(staging, inner["bucket_links"])
                add_link(staging)
            return original(staging, target, inner, verify)

        monkeypatch.setattr(sr, "install", install)
    rc = sr.main(seal_args(world, "build1", "--only", "cv"))
    if stage == "accept":
        assert rc == 0, capsys.readouterr().out
        assert _accept(world, farm, external)["verdict"] == "PASS"
        make_writable(farm)
        add_link(farm)
        sr.make_read_only(farm, include_root=True)
        with pytest.raises(sr.SealError) as exc:
            _accept(world, farm, external)
        assert exc.value.reason == "farm_bucket_links_changed"
    else:
        assert rc == 1 and "farm_bucket_links_changed" in capsys.readouterr().out
        assert not farm.exists() and not sr.pending_marker(farm).exists()
    assert exercised == [stage]


@pytest.mark.parametrize("stage", ["before_freeze", "after_rename", "accept"])
@pytest.mark.parametrize("defect", ["size", "mtime", "sha256"])
def test_pass_through_sources_rechecked_at_every_gate(
    world, monkeypatch, capsys, stage, defect
):
    bucket, _, _ = _pass_through(world)
    farm, _, external = _paths(world)
    exercised = []

    def damage():
        _damage(bucket.parent, defect, table="t", source_path="t/1.parquet")
        exercised.append(stage)

    if stage == "before_freeze":
        original = sr.install

        def install(staging, target, inner, verify=None):
            if target == farm:
                sr.check_farm_bucket_sources(staging, inner["bucket_links"])
                damage()
            return original(staging, target, inner, verify)

        monkeypatch.setattr(sr, "install", install)
    elif stage == "after_rename":
        original = sr.rename_noreplace

        def rename(staging, target):
            result = original(staging, target)
            if target == farm:
                damage()
            return result

        monkeypatch.setattr(sr, "rename_noreplace", rename)
    rc = sr.main(seal_args(world, "build1", "--only", "cv"))
    reason = "bucket_derived_from_source_changed"
    if stage == "accept":
        assert rc == 0, capsys.readouterr().out
        assert _accept(world, farm, external)["verdict"] == "PASS"
        damage()
        with pytest.raises(sr.SealError) as exc:
            _accept(world, farm, external)
        assert exc.value.reason == reason
    else:
        assert rc == 1 and reason in capsys.readouterr().out
        if stage == "before_freeze":
            assert not farm.exists() and not sr.pending_marker(farm).exists()
        else:
            assert json.loads(sr.refusal_marker(farm).read_text())["reason"] == reason
    assert exercised == [stage]


@pytest.mark.parametrize(
    "defect", ["absolute", "parent", "missing", "directory", "contract", "writable"]
)
def test_pass_through_planning_verifies_sources_and_contract(world, capsys, defect):
    bucket, _, _ = _pass_through(world)
    if defect in {"absolute", "parent", "missing"}:
        _damage(bucket.parent, defect, table="t", source_path="t/1.parquet")
    else:
        make_writable(bucket.parent)
        if defect == "directory":
            source = bucket.parent / "t/1.parquet"
            source.unlink()
            source.mkdir()
        elif defect == "contract":
            (bucket / "bucket_id=0/part.parquet").unlink()
        if defect != "writable":
            sr.make_read_only(bucket.parent, include_root=True)
    reason = (
        "bucket_derived_from_invalid"
        if defect in {"absolute", "parent"}
        else "bucket_derived_from_source_missing"
        if defect in {"missing", "directory"}
        else "bucket_contract_invalid"
        if defect == "contract"
        else "farm_bucket_not_read_only"
    )
    assert sr.main(seal_args(world, "build1", "--only", "cv", "--dry-run")) == 1
    assert reason in capsys.readouterr().out
    farm, physical, _ = _paths(world)
    assert not farm.exists() and not physical.exists()


@pytest.mark.parametrize("change", ["manifest", "source_link"])
def test_pass_through_contract_is_bound_at_planning(world, monkeypatch, capsys, change):
    bucket, _, _ = _pass_through(world, linked_source=change == "source_link")
    original = sr.assemble_farm

    def assemble(seal, db, plan, bucket_root):
        make_writable(bucket.parent)
        if change == "manifest":
            manifest = bucket / "_BUCKET_MANIFEST.json"
            content = json.loads(manifest.read_text())
            content["note"] = "a valid but different contract after planning"
            sr.write_json(manifest, content)
        else:
            source = bucket.parent / "t/1.parquet"
            other = world["databases"] / "other_source.parquet"
            shutil.copy2(source, other)
            source.unlink()
            source.symlink_to(other)
        sr.make_read_only(bucket.parent, include_root=True)
        bc.validate_contract(bucket)
        return original(seal, db, plan, bucket_root)

    monkeypatch.setattr(sr, "assemble_farm", assemble)
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 1
    assert "farm_bucket_contract_changed" in capsys.readouterr().out


def test_pass_through_source_link_must_stay_inside_allowed_roots(world, capsys):
    bucket, _, _ = _pass_through(world)
    make_writable(bucket.parent)
    source = bucket.parent / "t/1.parquet"
    outside = world["root"] / "outside_allowed_roots.parquet"
    shutil.copy2(source, outside)
    source.unlink()
    source.symlink_to(outside)
    sr.make_read_only(bucket.parent, include_root=True)
    # The source is regular and byte/size/mtime-identical, but the link escapes the allowed roots.
    bc.validate_contract(bucket)
    assert sr.main(seal_args(world, "build1", "--only", "cv", "--dry-run")) == 1
    assert "farm_link_outside_allowed_roots" in capsys.readouterr().out
