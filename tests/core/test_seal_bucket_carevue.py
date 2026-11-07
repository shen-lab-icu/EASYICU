"""CareVue bucket-only farm ruling (2026-11-07): synthetic publication and acceptance regressions."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from tests.support.seal_buckets import (
    make_world,
    make_writable,
    mock_flat_manifest_pin,
    seal_args,
)
from tools.seal_buckets import seal_rebuild as sr


@pytest.fixture
def world(tmp_path, monkeypatch):
    mock_flat_manifest_pin(monkeypatch)
    w = make_world(tmp_path / "w")
    cfg = sr.load_config(w["config"])
    # Explicit even on the base: exercise the bucket-only contract independently of its default name.
    cfg["dbs"]["cv"]["farm"]["name"] = "mimic_cv_bkt"
    current = Path(cfg["dbs"]["cv"]["current_root"])
    reference = w["b6"] / "cv_reference"
    shutil.copytree(current, reference, symlinks=True)
    cfg["dbs"]["cv"]["farm"]["reference"] = str(reference)
    sr.write_json(w["config"], cfg)
    monkeypatch.setenv("SEAL_MIRROR", str(w["mirror"]))
    yield w
    make_writable(w["root"])


def _farm(w):
    return w["b6"] / f"mimic_cv_bkt_s{w['sha'][:9]}"


def _external(w):
    return w["work"] / f"seal_{w['sha'][:9]}" / "build1" / "receipt_cv_farm.json"


def _is_farm(inner):
    return inner.get("schema", "").startswith("lead_r13.seal_farm_receipt")


@pytest.mark.parametrize(
    "source_form", ["link", "file", "directory", "bare_file", "linked_manifest"]
)
def test_bucket_only_farm_accepts_and_preserves_reference_except_flat_chartevents(
    world, capsys, source_form
):
    cfg = sr.load_config(world["config"])
    current = Path(cfg["dbs"]["cv"]["current_root"])
    flat = current / "chartevents.parquet"
    if source_form == "file":
        flat.unlink()
        shutil.copyfile(world["view"] / "chartevents.parquet", flat)
    elif source_form == "directory":
        flat.unlink()
        (current / "chartevents").mkdir()
        os.symlink(
            world["view"] / "chartevents.parquet", current / "chartevents/part.parquet"
        )
    elif source_form == "bare_file":
        (current / "chartevents").write_text("synthetic flat entry")
    elif source_form == "linked_manifest":
        source = world["view"] / "farm_conversion.json"
        sr.write_json(
            source, {"tables": [{"table": "chartevents"}, {"table": "admissions"}]}
        )
        (current / sr.CONVERSION_MANIFEST).unlink()
        os.symlink(source, current / sr.CONVERSION_MANIFEST)
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
    farm = _farm(world)
    inner = sr.accept_root(farm, _external(world), world["sha"], "build1", "cv")
    assert inner["verdict"] == "PASS"
    assert {"chartevents", "chartevents.parquet"}.isdisjoint(os.listdir(farm))
    skip = {
        "chartevents",
        "chartevents.parquet",
        "chartevents_bucket",
        sr.IN_ROOT_RECEIPT,
    }
    assert sr.link_set(farm, skip) == sr.link_set(
        Path(cfg["dbs"]["cv"]["farm"]["reference"]), skip
    )
    manifest = json.loads((farm / sr.CONVERSION_MANIFEST).read_text())
    assert manifest["tables"] == [{"table": "admissions"}]
    assert manifest["data_path"] == str(farm)
    assert "chartevents is bucket-only" in manifest["derived_from"]["note"]
    if source_form == "linked_manifest":
        assert manifest["derived_from"]["path"] == str(source)
        assert manifest["derived_from"]["sha256"] == sr.sha256_file(source)
    bucket = world["b6"] / f"mimic_cv_bucket_s{world['sha'][:9]}"
    assert os.readlink(farm / "chartevents_bucket") == str(
        bucket / "chartevents_bucket"
    )
    # The builder's source and its provenance remain in the bucket root, outside the formal farm.
    assert (bucket / "chartevents.parquet").resolve() == (
        world["b6"] / "mimic_cv_flat_r13/chartevents.parquet"
    ).resolve()


def test_final_target_names_and_build2_suffix_are_exact(world):
    cfg = sr.load_config(world["config"])
    cfg["dbs"]["cv"]["farm"] = sr.PRODUCTION["dbs"]["cv"]["farm"]
    seal = sr.Seal(cfg, world["sha"], "build1", False, ["cv"], {})
    assert seal.farm_target("cv").name == f"mimic_cv_bkt_s{seal.s9}"
    assert seal.target("cv").name == f"mimic_cv_bucket_s{seal.s9}"
    assert seal.target("cv", "build2").name == f"mimic_cv_bucket_s{seal.s9}_build2"


def _add_flat(root, form, world):
    make_writable(root)
    if form.startswith("manifest"):
        manifest = root / sr.CONVERSION_MANIFEST
        content = json.loads(manifest.read_text())
        content["tables"].append(
            "chartevents" if form == "manifest_string" else {"table": "chartevents"}
        )
        sr.write_json(manifest, content)
    else:
        path = root / (
            "chartevents"
            if form in ("directory", "bare_file")
            else "chartevents.parquet"
        )
        if path.is_symlink() or path.is_file():
            path.unlink()
        if form in ("link", "dangling_link"):
            os.symlink(
                world["view"]
                / (
                    "missing.parquet"
                    if form == "dangling_link"
                    else "chartevents.parquet"
                ),
                path,
            )
        elif form == "directory":
            path.mkdir(exist_ok=True)
        else:
            path.write_text("synthetic flat entry")


def _bind_snapshot(root, inner):
    inner["entries"] = sorted(n for n in os.listdir(root) if n != sr.IN_ROOT_RECEIPT)
    inner["link_set"] = sr.link_set(root, {sr.IN_ROOT_RECEIPT}, identities=True)
    inner["conversion_manifest_sha256"] = sr.sha256_file(root / sr.CONVERSION_MANIFEST)


@pytest.mark.parametrize("stage", ["before_freeze", "after_rename", "accept"])
@pytest.mark.parametrize(
    "form",
    [
        "link",
        "dangling_link",
        "file",
        "directory",
        "bare_file",
        "manifest_dict",
        "manifest_string",
    ],
)
def test_flat_chartevents_is_refused_at_every_farm_gate(
    world, monkeypatch, capsys, stage, form
):
    if stage == "before_freeze":
        original = sr.install

        def install(staging, target, inner, verify=None):
            if _is_farm(inner):
                _add_flat(staging, form, world)
                # A self-consistent snapshot must still be refused by the bucket-only rule.
                if form != "dangling_link":
                    _bind_snapshot(staging, inner)
            return original(staging, target, inner, verify)

        monkeypatch.setattr(sr, "install", install)
    elif stage == "after_rename":
        original = sr.rename_noreplace

        def rename(staging, target):
            result = original(staging, target)
            if target == _farm(world):
                _add_flat(target, form, world)
                sr.make_read_only(target, include_root=True)
            return result

        monkeypatch.setattr(sr, "rename_noreplace", rename)
    rc = sr.main(seal_args(world, "build1", "--only", "cv"))
    farm = _farm(world)
    if stage == "accept":
        assert rc == 0, capsys.readouterr().out
        _add_flat(farm, form, world)
        inner = json.loads((farm / sr.IN_ROOT_RECEIPT).read_text())
        if form != "dangling_link":
            _bind_snapshot(farm, inner)
        digest = sr.write_json(farm / sr.IN_ROOT_RECEIPT, inner)
        sr.make_read_only(farm, include_root=True)
        ext = json.loads(_external(world).read_text())
        ext["install"]["in_root_receipt_sha256"] = digest
        sr.write_json(_external(world), ext)
        with pytest.raises(sr.SealError, match="farm_flat_chartevents"):
            sr.accept_root(farm, _external(world), world["sha"], "build1", "cv")
    else:
        assert rc == 1
        assert "farm_flat_chartevents" in capsys.readouterr().out
        if stage == "before_freeze":
            assert not farm.exists() and not sr.pending_marker(farm).exists()
            assert sr.writable_entries(farm.with_name(f".{farm.name}.seal_staging"))
        else:
            assert sr.refusal_marker(farm).is_file()


@pytest.mark.parametrize("mode", ["build1", "build2"])
@pytest.mark.parametrize("old_form", ["directory", "dangling_link"])
def test_old_name_collision_refuses_new_farm_without_reuse(
    world, mode, old_form, capsys
):
    old = world["b6"] / f"mimic_cv_dir_s{world['sha'][:9]}"
    if old_form == "directory":
        old.mkdir()
        (old / "sentinel").write_text("keep")
    else:
        os.symlink(world["b6"] / "missing", old)
    assert sr.main(seal_args(world, mode, "--only", "cv", "--dry-run")) == 1
    assert "farm_cross_name_collision" in capsys.readouterr().out
    assert (
        old.is_symlink()
        if old_form == "dangling_link"
        else (old / "sentinel").read_text() == "keep"
    )
    assert not _farm(world).exists()


@pytest.mark.parametrize("stage", ["before_freeze", "after_rename"])
def test_collision_appearing_during_publication_is_refused(
    world, monkeypatch, capsys, stage
):
    old = _farm(world).with_name(f"mimic_cv_dir_s{world['sha'][:9]}")
    if stage == "before_freeze":
        original = sr.install

        def install(staging, target, inner, verify=None):
            if _is_farm(inner):
                old.mkdir()
            return original(staging, target, inner, verify)

        monkeypatch.setattr(sr, "install", install)
    else:
        original = sr.rename_noreplace

        def rename(staging, target):
            result = original(staging, target)
            if target == _farm(world):
                old.mkdir()
            return result

        monkeypatch.setattr(sr, "rename_noreplace", rename)
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 1
    assert "farm_cross_name_collision" in capsys.readouterr().out
    assert old.is_dir()
    if stage == "before_freeze":
        assert not _farm(world).exists()
    else:
        assert sr.refusal_marker(_farm(world)).is_file()


def test_old_name_config_is_refused_even_when_new_farm_exists(world, capsys):
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
    cfg = sr.load_config(world["config"])
    cfg["dbs"]["cv"]["farm"]["name"] = "mimic_cv_dir"
    sr.write_json(world["config"], cfg)
    assert sr.main(seal_args(world, "build1", "--only", "cv", "--dry-run")) == 1
    assert "farm_name_mismatch" in capsys.readouterr().out
    assert _farm(world).is_dir()


@pytest.mark.parametrize("change", ["old_sibling", "old_target"])
def test_standalone_acceptance_refuses_cross_name_collision(world, capsys, change):
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
    farm = _farm(world)
    old = farm.with_name(f"mimic_cv_dir_s{world['sha'][:9]}")
    if change == "old_sibling":
        old.mkdir()
    else:
        farm.rename(old)
        farm = old
    with pytest.raises(
        sr.SealError, match="farm_cross_name_collision|farm_name_mismatch"
    ):
        sr.accept_root(farm, _external(world), world["sha"], "build1", "cv")


ORIGIN_KEYS = {
    "icustays_dbsource_counts",
    "carevue_stays",
    "other_source_stays",
    "rows_from_noncarevue_stays",
    "itemid_ge_220000_rows",
    "itemid_ge_220000_distinct_items",
    "flat_source_manifest_sha256",
    "flat_source_chartevents_sha256",
}


def test_origin_exact_fields_counts_and_both_external_receipts(world, capsys):
    source = world["view"] / "chartevents.parquet"
    table = pq.read_table(source)
    # Stay-level CareVue membership retains high item IDs; disclose them without filtering.
    values = table["itemid"].to_pylist()
    values[:12] = [220001] * 12
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("itemid"), "itemid", pa.array(values)
        ),
        source,
    )
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
    inner = sr.accept_root(_farm(world), _external(world), world["sha"], "build1", "cv")
    origin = inner["carevue_origin"]
    assert set(origin) == ORIGIN_KEYS
    assert origin == {
        "icustays_dbsource_counts": {"carevue": 19},
        "carevue_stays": 19,
        "other_source_stays": 0,
        "rows_from_noncarevue_stays": 0,
        "itemid_ge_220000_rows": 12,
        "itemid_ge_220000_distinct_items": 1,
        "flat_source_manifest_sha256": sr.CV_FLAT_MANIFEST_SHA256,
        "flat_source_chartevents_sha256": sr.sha256_file(source),
    }
    for name in ("receipt_cv.json", "receipt_cv_farm.json"):
        assert (
            json.loads(_external(world).with_name(name).read_text())["carevue_origin"]
            == origin
        )


@pytest.mark.parametrize(
    "violation",
    ["metavision_stay", "null_stay", "unknown_chart_stay", "null_chart_stay"],
)
def test_origin_noncarevue_stays_and_rows_are_refused(world, capsys, violation):
    if violation in ("metavision_stay", "null_stay"):
        pq.write_table(
            pa.table(
                {
                    "ICUSTAY_ID": range(1, 20),
                    "DBSOURCE": ["carevue"] * 18
                    + ([None] if violation == "null_stay" else ["metavision"]),
                }
            ),
            world["view"] / "icustays.parquet",
        )
        reason = "carevue_origin_other_source_stays"
    else:
        source = world["view"] / "chartevents.parquet"
        table = pq.read_table(source)
        values = table["icustay_id"].to_pylist()
        values[0] = None if violation == "null_chart_stay" else 99999
        pq.write_table(
            table.set_column(0, "icustay_id", pa.array(values, pa.int64())), source
        )
        reason = "carevue_origin_rows_from_noncarevue_stays"
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 1
    assert reason in capsys.readouterr().out
    assert not _farm(world).exists()


def test_pinned_flat_manifest_mismatch_is_typed_and_refused_before_build(world, capsys):
    (world["b6"] / "mimic_cv_flat_r13/conversion_manifest.json").write_text(
        '{"changed": true}'
    )
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 1
    assert "carevue_origin_flat_manifest_mismatch" in capsys.readouterr().out
    assert not _farm(world).exists()
    assert not (world["b6"] / f"mimic_cv_bucket_s{world['sha'][:9]}").exists()


def test_flat_chartevents_digest_must_match_builder_derived_from(
    world, monkeypatch, capsys
):
    original = sr.compute_carevue_origin

    def compute(farm, flat, bucket):
        # The flat input changes after the built bucket passes; never accept stale provenance.
        source = flat / "chartevents.parquet"
        with source.open("ab") as fh:
            fh.write(b"changed")
        return original(farm, flat, bucket)

    monkeypatch.setattr(sr, "compute_carevue_origin", compute)
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 1
    assert "carevue_origin_chartevents_sha_mismatch" in capsys.readouterr().out
    assert not _farm(world).exists()


@pytest.mark.parametrize("receipt", ["inner", "external", "bucket_external"])
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra_key",
        "other_source_stays",
        "rows_from_noncarevue_stays",
        "manifest_pin",
        "chartevents_sha",
    ],
)
def test_acceptance_rechecks_origin_fields_and_invariants(
    world, capsys, receipt, change
):
    assert sr.main(seal_args(world, "build1", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
    farm = _farm(world)
    inner = json.loads((farm / sr.IN_ROOT_RECEIPT).read_text())
    external = _external(world)
    if receipt == "bucket_external":
        farm = world["b6"] / f"mimic_cv_bucket_s{world['sha'][:9]}"
        external = external.with_name("receipt_cv.json")
    ext = json.loads(external.read_text())
    record = inner if receipt == "inner" else ext
    if change == "missing":
        record.pop("carevue_origin", None)
    else:
        origin = record.setdefault("carevue_origin", {})
        key = {
            "manifest_pin": "flat_source_manifest_sha256",
            "chartevents_sha": "flat_source_chartevents_sha256",
        }.get(change, change)
        origin[key] = "0" * 64 if change in ("manifest_pin", "chartevents_sha") else 1
    if receipt == "inner":
        make_writable(farm)
        ext["install"]["in_root_receipt_sha256"] = sr.write_json(
            farm / sr.IN_ROOT_RECEIPT, inner
        )
        sr.make_read_only(farm, include_root=True)
    sr.write_json(external, ext)
    with pytest.raises(sr.SealError, match="carevue_origin_"):
        sr.accept_root(farm, external, world["sha"], "build1", "cv")


@pytest.mark.parametrize("retired_lineage", [False, True])
def test_bucket_only_current_farm_builds_twins_recovers_accepts_and_compares(
    world, capsys, retired_lineage
):
    cfg = sr.load_config(world["config"])
    current = Path(cfg["dbs"]["cv"]["current_root"])
    (current / "chartevents.parquet").unlink()
    content = json.loads((current / sr.CONVERSION_MANIFEST).read_text())
    content["tables"] = [{"table": "admissions"}]
    if retired_lineage:
        content["derived_from"]["path"] = str(world["view"] / "retired_conversion.json")
    sr.write_json(current / sr.CONVERSION_MANIFEST, content)
    for mode in ("build1", "build2"):
        assert sr.main(seal_args(world, mode, "--only", "cv", "--dry-run")) == 0, (
            capsys.readouterr().out
        )
        assert sr.main(seal_args(world, mode, "--only", "cv")) == 0, (
            capsys.readouterr().out
        )
        bucket = (
            world["b6"]
            / f"mimic_cv_bucket_s{world['sha'][:9]}{'_build2' if mode == 'build2' else ''}"
        )
        external = _external(world).parent.parent / mode / "receipt_cv.json"
        sr.pending_marker(bucket).write_text("pending")
        assert sr.operator_commit(cfg, world["sha"], mode, "cv")["verdict"] == "PASS"
        assert (
            sr.accept_root(bucket, external, world["sha"], mode, "cv")["verdict"]
            == "PASS"
        )
    sr.pending_marker(_farm(world)).write_text("pending")
    assert (
        sr.operator_commit(cfg, world["sha"], "build1", "cv", farm=True)["verdict"]
        == "PASS"
    )
    inner = sr.accept_root(_farm(world), _external(world), world["sha"], "build1", "cv")
    manifest = json.loads((_farm(world) / sr.CONVERSION_MANIFEST).read_text())
    assert manifest["derived_from"]["path"] == content["derived_from"]["path"]
    assert set(inner["carevue_origin"]) == ORIGIN_KEYS
    assert sr.main(seal_args(world, "compare", "--only", "cv")) == 0, (
        capsys.readouterr().out
    )
