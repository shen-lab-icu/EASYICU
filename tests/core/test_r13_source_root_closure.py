"""Synthetic B64 closure and physical bound-farm integration regressions."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

from easyicu import formal_inputs as inputs
from easyicu import storage_binding as binding
from easyicu import storage_layout as layout
from easyicu import storage_seal_publication as publication
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource


def _world(tmp, kind):
    physical = tmp / "physical"
    bucket = physical / "chartevents_bucket"
    part = bucket / "bucket_id=0/part.parquet"
    part.parent.mkdir(parents=True)
    source_root = physical if kind == "default" else tmp / "source"
    source_root.mkdir(exist_ok=True)
    flat = source_root / "chartevents.parquet"
    frame = pd.DataFrame({"icustay_id": range(100, 120), "itemid": [220045] * 20,
                          "value": ["60"] * 20})
    frame.to_parquet(flat, index=False)
    frame.assign(**{layout.ORDINAL: range(len(frame))}).to_parquet(part, index=False)
    info = flat.stat()
    derived = {"table": "chartevents", "files": [{
        "path": flat.name, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
        "sha256": hashlib.sha256(flat.read_bytes()).hexdigest(),
    }]}
    if kind == "absolute":
        derived["source_root"] = str(source_root)
    elif kind == "relative":
        derived["source_root"] = "../source"
    elif kind == "link":
        (tmp / "source_link").symlink_to(source_root, target_is_directory=True)
        derived["source_root"] = "../source_link"
    manifest = {
        "schema": layout.SCHEMA, "scheme": "range", "key_column": "icustay_id",
        "key_type": "BIGINT", "range_breakpoints": [],
        "sort_keys": ["icustay_id", layout.ORDINAL], "source_order": layout.ORDINAL,
        "row_group_target": 65536, "builder_version": layout.BUILDER_VERSION,
        "inventory": [{"relative_path": "bucket_id=0/part.parquet",
                       "size": part.stat().st_size, "row_count": len(frame)}],
        "derived_from": derived,
    }
    (bucket / layout.MANIFEST).write_text(json.dumps(manifest))
    (bucket / layout.BUILD_RECEIPT).write_text("{}")
    (bucket / "_COMPLETE").write_text("synthetic")
    farm = tmp / "farm"
    farm.mkdir()
    (farm / bucket.name).symlink_to(bucket, target_is_directory=True)
    for root, outputs in ((source_root, [{"relative_path": flat.name}]), (farm, [])):
        (root / layout.CONVERSION_MANIFEST).write_text(json.dumps({
            "schema_version": layout.CONVERSION_MANIFEST_SCHEMA,
            "tables": [{"table": "chartevents", "row_count": len(frame), "outputs": outputs}],
        }))
    return farm, physical, source_root, flat, part, derived


def _audit(root):
    events, active = [], [True]

    def hook(event, args):
        if active[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.path.realpath(os.fsdecode(args[0])))
            if path.is_relative_to(root):
                events.append(path)

    sys.addaudithook(hook)
    return events, active


def _assert_source_manifest_visited(farm, root):
    events, active = _audit(root)
    try:
        publication.assert_roots_published([farm])
    finally:
        active[0] = False
    assert root / layout.CONVERSION_MANIFEST in events
    assert not any(path.suffix == ".parquet" for path in events)


@pytest.mark.parametrize("kind", ["absolute", "relative", "link"])
@pytest.mark.parametrize("suffix,reason", [
    (publication.PENDING_SUFFIX, publication.UNPUBLISHED),
    (publication.REFUSAL_SUFFIX, publication.REFUSED),
])
@pytest.mark.parametrize("entry", ["gate", "formal_plan"])
def test_source_root_marker_refused_before_any_metadata_open(tmp_path, kind, suffix, reason, entry):
    farm, _, root, _, _, _ = _world(tmp_path, kind)
    _assert_source_manifest_visited(farm, root)
    (root.parent / f".{root.name}{suffix}").write_text("{}")
    events, active = _audit(root)
    error = None
    try:
        if entry == "gate":
            publication.assert_roots_published([farm])
        else:
            inputs.plan_input_files("mimic", farm)
    except Exception as exc:
        error = exc
    finally:
        active[0] = False
    assert events == [], "source metadata was opened before publication refusal"
    assert isinstance(error, publication.SealPublicationError)
    assert error.reason == reason


@pytest.mark.parametrize("kind", ["default", "absolute", "relative", "link"])
def test_bound_bucket_only_farm_reader_gate_and_formal_plan_share_anchor(tmp_path, monkeypatch, kind):
    farm, physical, root, flat, part, derived = _world(tmp_path, kind)
    original = layout.resolve_derived_from_source
    calls = []

    def resolve(raw, **kwargs):
        path = original(raw, **kwargs)
        calls.append((raw, kwargs, path.resolve()))
        return path

    monkeypatch.setattr(layout, "resolve_derived_from_source", resolve)
    publication.assert_roots_published([farm])
    gate_paths = {path for raw, _, path in calls if raw == "."}
    assert gate_paths == {root}, "the gate must enqueue the actual source root"
    calls.clear()
    source = ICUDataSource(load_src_cfg("mimic"), base_path=farm, enable_cache=False)
    with binding.binding_unit():
        assert tuple(p.resolve() for p in source.resolve_target_files("chartevents")) == (part,)
        assert len(source._load_raw_frame("chartevents", ["icustay_id", "itemid", "value"])) == 20
        receipt = layout.bucket_source_receipt(farm / "chartevents_bucket")
        assert Path(receipt["derived_from"]["source_root"]).resolve() == root
        assert receipt["derived_from"]["files"][0]["realpath"] == str(flat)
    assert {path for raw, _, path in calls if raw == "."} == gate_paths
    assert original(flat.name, bucket_root=farm / "chartevents_bucket",
                    source_root=derived.get("source_root")).resolve() == flat
    assert inputs.plan_input_files("mimic", farm)["buckets"][0]["source_manifest"]
    if kind == "default":
        (physical.parent / f".{physical.name}{publication.PENDING_SUFFIX}").write_text("{}")
        with pytest.raises(publication.SealPublicationError) as caught:
            publication.assert_roots_published([farm])
        assert caught.value.reason == publication.UNPUBLISHED


@pytest.mark.parametrize("kind", ["absolute", "relative", "link"])
@pytest.mark.parametrize("config_kind", ["dict_overlay", "required_layout"])
def test_configuration_in_source_root_refused_before_metadata_but_work_root_allowed(
    tmp_path, kind, config_kind,
):
    farm, _, root, _, _, _ = _world(tmp_path, kind)
    _assert_source_manifest_visited(farm, root)
    config = root / ("dictionary.json" if config_kind == "dict_overlay" else "REQUIRED_LAYOUT.json")
    config.write_text("{}")
    outside = tmp_path / config.name
    outside.write_text("{}")
    if config_kind == "dict_overlay":
        repo = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("source_root_boot", repo / "tools/formal/formal_boot.py")
        boot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(boot)
        tools = tmp_path / "run/.formal_tools"
        tools.mkdir(parents=True)

        def check(path):
            return boot.bind_dictionary(str(tools), str(path), input_roots=[farm])

        error_type = boot.DictOverlayError
    else:
        def check(path):
            publication.assert_configuration_outside_roots(path, [farm])

        error_type = publication.SealPublicationError
    events, active = _audit(root)
    error = None
    try:
        check(config)
    except Exception as exc:
        error = exc
    finally:
        active[0] = False
    assert events == [], "source-root configuration or metadata opened before containment refusal"
    assert isinstance(error, error_type)
    assert error.reason == "formal_configuration_inside_input"
    check(outside)


@pytest.mark.parametrize("kind", ["absolute", "relative", "link"])
def test_sealed_comparator_source_root_semantics_match_reader(tmp_path, kind):
    from tools.seal_buckets import bucket_compare, seal_rebuild

    farm, _, _, flat, _, derived = _world(tmp_path, kind)
    bucket = farm / "chartevents_bucket"
    manifest = json.loads((bucket / layout.MANIFEST).read_text())
    bucket_compare._verify_sources(bucket, manifest)
    _, links = seal_rebuild.farm_bucket_sources(bucket)
    assert links[flat.name]["resolved_target"] == str(flat)
    assert layout.resolve_derived_from_source(
        flat.name, bucket_root=bucket, source_root=derived["source_root"],
    ).resolve() == flat
