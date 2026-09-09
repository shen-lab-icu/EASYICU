"""Definition-preserving Sepsis context across real worker/publisher paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import easyicu
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.api import extraction as api
from easyicu.api import derivation_context as ctx
from easyicu.research_agent.intake.export_package import (
    ExportPackageError,
    open_export_package,
)
from easyicu.scores.sepsis import sep3


def _inputs(id_col="stay_id"):
    # firstSI re-selection; lost baseline; clipped first onset; interior control;
    # exact lower/upper publication bounds; absent, false and missing SI.
    cases = [
        (1, [-60, -36, 0, 1], [0, 0, 0, 2], [-36, 0]),
        (2, [-36, -24, 0, 1], [0, 1, 2, 3], [0]),
        (3, [-48, -30, -24, 0], [0, 2, 0, 2], [0]),
        (4, [0, 1], [0, 2], [0]),
        (5, [-25, -24], [0, 2], [-24]),
        (6, [71, 72, 73], [0, 2, 3], [72]),
        (7, [0, 1], [0, 2], []),
        (8, [0, 1], [0, 2], []),
        (9, [0, 1], [None, None], [0]),
    ]
    score, suspicion = [], []
    for stay, times, values, infections in cases:
        score.extend((stay, float(t), v) for t, v in zip(times, values))
        suspicion.extend((stay, float(t), True) for t in infections)
    suspicion.extend([(8, 0.0, False), (8, 1.0, None)])
    return (
        pd.DataFrame(score, columns=[id_col, "charttime", "sofa"]),
        pd.DataFrame(suspicion, columns=[id_col, "charttime", "susp_inf"]),
    )


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root):
    return json.loads((root / "_manifest.json").read_text())


def _build(root, monkeypatch, *, streamed=True, backend="arrow", id_col="stay_id"):
    root.mkdir(exist_ok=True)
    worker = root / "worker"
    worker.mkdir()
    sofa, suspicion = _inputs(id_col)
    modules = {
        "sofa1_score": ["sofa", *api._SOFA1_COMPONENT_NAMES],
        "sepsis_shared": ["susp_inf"],
        "sepsis3_sofa1": ["sep3_sofa1"],
    }
    monkeypatch.setattr(api, "EXTRACT_MODULES", modules)
    monkeypatch.setattr(
        api, "_native_export_runtime_provenance", lambda: {"fixture": True}
    )
    merged = sofa.merge(suspicion, on=[id_col, "charttime"], how="outer")
    monkeypatch.setattr(easyicu, "load_concepts", lambda **kwargs: merged.copy())
    organs = sofa.copy()
    for component in api._SOFA1_COMPONENT_NAMES:
        organs[component] = organs.sofa if component == "sofa_resp" else 0.0
    table = pa.Table.from_pandas(organs, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            api._SOFA1_TIME_BASIS_KEY: api._SOFA1_TIME_BASIS,
        }
    )
    pq.write_table(table, root / "sofa1_score.parquet")
    suspicion.to_parquet(root / "sepsis_shared.parquet", index=False)
    pd.DataFrame({id_col: range(1, 10), "los_icu": 2.0}).to_parquet(
        root / "outcome.parquet", index=False
    )
    api._run_special_extraction(
        ["sepsis3_sofa1"],
        "miiv",
        "/synthetic",
        {id_col: list(range(1, 10))},
        1,
        str(worker),
        stream_output_batches=streamed,
        published_output_dir=str(root) if streamed else None,
    )
    worker_manifest = _manifest(worker)
    assert not worker_manifest["errors"]
    captured = worker_manifest["derivation_contexts"]["sep3_sofa1"]
    retained = ctx.transfer_context(worker, root, captured)
    shutil.move(worker / "sep3_sofa1.parquet", root / "sepsis3_sofa1.parquet")
    shutil.rmtree(worker)
    if backend == "pandas":
        monkeypatch.setattr(
            api, "_try_publish_native_export_arrow_fast_path", lambda **kwargs: None
        )
    api._publish_native_export_v2(
        database="miiv",
        data_path="/synthetic",
        output_dir=str(root),
        modules=list(modules),
        max_patients=None,
        result={
            "modules": {
                m: {
                    "errors": [],
                    "derivation_contexts": {"sep3_sofa1": retained}
                    if m == "sepsis3_sofa1"
                    else {},
                }
                for m in modules
            }
        },
        require_stay_time_bounds=True,
        expected_patient_ids=list(range(1, 10)),
    )
    return _hash(root / "_manifest.json")


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("backend", ["arrow", "pandas"])
@pytest.mark.parametrize("id_col", ["stay_id", "patientunitstayid"])
def test_actual_worker_to_native_replays_full_context(
    tmp_path, monkeypatch, streamed, backend, id_col
):
    manifest_hash = _build(
        tmp_path, monkeypatch, streamed=streamed, backend=backend, id_col=id_col
    )
    receipt = ctx.validate_native_derivation_context(
        tmp_path,
        expected_manifest_sha256=manifest_hash,
        expected_patient_ids=range(1, 10),
    )
    assert receipt["positive_stays"] == 4
    assert receipt["membership_or_onset_differences"] == 0
    actual = pd.read_parquet(tmp_path / "sepsis3_sofa1.parquet")
    assert dict(zip(actual.stay_id, actual.charttime)) == {2: 0, 4: 1, 5: -24, 6: 72}
    # Demonstrate why merely removing the old audit or replaying public views is wrong.
    sofa = pd.read_parquet(tmp_path / "sofa1_score.parquet")
    suspicion = pd.read_parquet(tmp_path / "sepsis_shared.parquet")
    clipped = sep3(
        sofa[["stay_id", "charttime", "sofa"]],
        suspicion,
        id_cols=["stay_id"],
        index_col="charttime",
    )
    clipped_events = dict(zip(clipped.stay_id, clipped.charttime))
    assert clipped_events[1] == 1
    assert clipped_events[2] == 1
    assert clipped_events[3] == 0
    assert 5 not in clipped_events
    manifest = _manifest(tmp_path)
    private = manifest["derivation_contexts"]["sep3_sofa1"]
    assert private["external_llm_allowed"] is False
    assert all(ctx.PRIVATE_DIRECTORY not in item["file"] for item in manifest["files"])
    with open_export_package(tmp_path) as package:
        assert all(
            ctx.PRIVATE_DIRECTORY not in item.relative_path for item in package.files
        )
        assert set(package.index_dict()) == {
            "sofa",
            *api._SOFA1_COMPONENT_NAMES,
            "susp_inf",
            "sep3_sofa1",
        }


@pytest.mark.parametrize(
    "damage", ["missing", "corrupt", "manifest", "cohort", "legacy"]
)
def test_missing_corrupt_or_drifted_authority_cannot_verify(
    tmp_path, monkeypatch, damage
):
    manifest_hash = _build(tmp_path, monkeypatch)
    manifest = _manifest(tmp_path)
    ref = manifest["derivation_contexts"]["sep3_sofa1"]["reference"]
    path = tmp_path / ref["file"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.chmod(0o600)
        path.write_bytes(b"corrupt")
    elif damage == "manifest":
        manifest["database"] = "eicu"
        (tmp_path / "_manifest.json").write_text(json.dumps(manifest))
    elif damage == "legacy":
        del manifest["derivation_contexts"]
        (tmp_path / "_manifest.json").write_text(json.dumps(manifest))
        manifest_hash = _hash(tmp_path / "_manifest.json")
    with pytest.raises(ctx.DerivationContextError):
        ctx.validate_native_derivation_context(
            tmp_path,
            expected_manifest_sha256=manifest_hash,
            expected_patient_ids=[1] if damage == "cohort" else None,
        )


@pytest.mark.parametrize(
    "damage",
    ["omit_batch", "duplicate_batch", "wrong_input_id", "clipped_input", "code"],
)
def test_resealed_inconsistent_context_is_rejected(tmp_path, monkeypatch, damage):
    _build(tmp_path, monkeypatch)
    manifest = _manifest(tmp_path)
    descriptor = manifest["derivation_contexts"]["sep3_sofa1"]
    envelope = ctx._read_json(tmp_path, descriptor["reference"])
    inputs = ctx._read_json(tmp_path, envelope["inputs"])
    if damage == "omit_batch":
        inputs["batches"].pop()
    elif damage == "duplicate_batch":
        inputs["batches"].append(inputs["batches"][0])
    elif damage == "code":
        inputs["code"]["scores/sepsis.py"] = "0" * 64
    else:
        batch = inputs["batches"][0]
        frame = ctx._read_frame(tmp_path, batch["suspicion"])
        if damage == "wrong_input_id":
            frame[batch["id_col"]] = 999
        else:
            frame = frame[frame.charttime >= -24]
        batch["suspicion"] = ctx._store(tmp_path, frame=frame)
    envelope["inputs"] = ctx._store(tmp_path, payload=inputs)
    descriptor["reference"] = ctx._store(tmp_path, payload=envelope)
    (tmp_path / "_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ctx.DerivationContextError):
        ctx.validate_native_derivation_context(
            tmp_path, expected_manifest_sha256=_hash(tmp_path / "_manifest.json")
        )


def test_context_cannot_be_promoted_to_research_file(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    manifest = _manifest(tmp_path)
    private = next((tmp_path / ctx.PRIVATE_DIRECTORY).glob("*.parquet"))
    manifest["files"][0]["file"] = private.relative_to(tmp_path).as_posix()
    (tmp_path / "_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ExportPackageError) as error:
        open_export_package(tmp_path)
    assert error.value.code == "private_derivation_context_as_data"


def test_transfer_is_byte_exact_and_snapshots_resist_live_mutation(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    sofa, suspicion = _inputs()
    recorder = ctx.SepsisContextRecorder(
        source, database="miiv", data_path="/synthetic"
    )
    recorder.derive(
        sofa, suspicion, ids=range(1, 10), id_col="stay_id", time_col="charttime"
    )
    reference = recorder.finish(expected_ids=range(1, 10))
    assert ctx.transfer_context(source, destination, reference) == reference
    for path in (source / ctx.PRIVATE_DIRECTORY).iterdir():
        assert (
            path.read_bytes()
            == (destination / ctx.PRIVATE_DIRECTORY / path.name).read_bytes()
        )
    payload = ctx._read_json(source, reference)
    ref = payload["batches"][0]["sofa"]
    path = source / ref["file"]
    original_read = pd.read_parquet

    def mutate_then_read(handle, **kwargs):
        path.chmod(0o600)
        path.write_bytes(b"corrupted after snapshot")
        return original_read(handle, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", mutate_then_read)
    pd.testing.assert_frame_equal(ctx._read_frame(source, ref), sofa)


def test_recorder_refuses_queue_drift_and_unknown_identity(tmp_path):
    sofa, suspicion = _inputs()
    recorder = ctx.SepsisContextRecorder(
        tmp_path, database="miiv", data_path="/synthetic"
    )
    with pytest.raises(ctx.DerivationContextError, match="cohort drift"):
        recorder.derive(
            sofa, suspicion, ids=[1], id_col="stay_id", time_col="charttime"
        )
    recorder.derive(
        sofa, suspicion, ids=range(1, 10), id_col="stay_id", time_col="charttime"
    )
    with pytest.raises(ctx.DerivationContextError, match="queue incomplete"):
        recorder.finish(expected_ids=range(1, 11))


def test_public_api_collector_retains_context_after_worker_cleanup(
    tmp_path, monkeypatch
):
    """Only dispatch/source IO are substituted; actual collector/publisher run."""
    root = tmp_path / "candidate"
    root.mkdir()
    sofa, suspicion = _inputs()
    for component in api._SOFA1_COMPONENT_NAMES:
        sofa[component] = sofa.sofa if component == "sofa_resp" else 0.0
    table = pa.Table.from_pandas(sofa, preserve_index=False)
    pq.write_table(
        table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                api._SOFA1_TIME_BASIS_KEY: api._SOFA1_TIME_BASIS,
            }
        ),
        root / "sofa1_score.parquet",
    )
    suspicion.to_parquet(root / "sepsis_shared.parquet", index=False)
    pd.DataFrame({"stay_id": range(1, 10), "los_icu": 2.0}).to_parquet(
        root / "outcome.parquet"
    )
    worker_directories = []

    class InlineProcess:
        exitcode = 0

        def __init__(self, *, target, args, daemon):
            self.args = args

        def start(self):
            args = self.args
            worker = Path(args[6]) / api._SPECIAL_OUTPUT_DIRNAME
            worker.mkdir()
            worker_directories.append(worker)
            api._run_special_extraction(
                *args[1:6],
                str(worker),
                use_sofa2=args[7],
                stream_output_batches=args[8],
                published_output_dir=args[9],
            )

        def join(self):
            pass

    monkeypatch.setattr(
        api,
        "_get_extraction_mp_context",
        lambda _: SimpleNamespace(Process=InlineProcess),
    )
    monkeypatch.setattr(
        api, "_native_export_runtime_provenance", lambda: {"fixture": True}
    )
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
    result = api.extract_database(
        "miiv",
        data_path=str(tmp_path),
        output_dir=str(root),
        modules=["sepsis3_sofa1"],
        patient_ids={"stay_id": list(range(1, 10))},
        batch_size=1,
        group_modules=False,
        stream_output_batches=True,
        verbose=False,
    )
    assert not result["modules"]["sepsis3_sofa1"]["errors"]
    assert all(not path.exists() for path in worker_directories)
    assert _manifest(root)["derivation_validation"]["status"] == "verified"
    assert result["native_export_v2"]["manifest_sha256"] == _hash(root / "_manifest.json")
    assert (
        ctx.validate_native_derivation_context(
            root, expected_manifest_sha256=_hash(root / "_manifest.json")
        )["positive_stays"]
        == 4
    )


def test_empty_one_shot_records_absence_without_fabricating_negatives(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(easyicu, "load_concepts", lambda **kwargs: pd.DataFrame())
    api._run_special_extraction(
        ["sepsis3_sofa1"], "miiv", "/synthetic", {"stay_id": [1, 2]}, 2, str(tmp_path)
    )
    manifest = _manifest(tmp_path)
    payload = ctx._read_json(tmp_path, manifest["derivation_contexts"]["sep3_sofa1"])
    assert payload["batches"][0]["action"] == "empty_si"
    assert payload["batches"][0]["sofa"] is None
    assert manifest["saved"] == {}


def test_object_si_argument_type_survives_parquet_roundtrip(tmp_path):
    sofa = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0.0, 1.0], "sofa": [0.0, 2.0]}
    )
    suspicion = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "susp_inf": pd.Series([-1], dtype="object"),
        }
    )
    recorder = ctx.SepsisContextRecorder(
        tmp_path, database="miiv", data_path="/synthetic"
    )
    original = recorder.derive(
        sofa, suspicion, ids=[1], id_col="stay_id", time_col="charttime"
    )
    payload = ctx._read_json(tmp_path, recorder.finish(expected_ids=[1]))
    batch = payload["batches"][0]
    restored = ctx._read_frame(tmp_path, batch["suspicion"]).astype(
        batch["suspicion_dtypes"]
    )
    replay = sep3(sofa, restored, id_cols=["stay_id"], index_col="charttime")
    pd.testing.assert_frame_equal(original, replay)
    assert len(replay) == 1  # Existing object branch, not a new clinical rule.


def test_one_shot_replay_reads_bounded_stay_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "MAX_CONTEXT_STAYS", 2)
    monkeypatch.setattr(ctx, "MAX_CONTEXT_ROWS", 10)
    manifest_hash = _build(tmp_path, monkeypatch, streamed=False)
    original_read = pd.read_parquet
    read_sizes = []

    def bounded_read(source, **kwargs):
        frame = original_read(source, **kwargs)
        read_sizes.append(len(frame))
        assert len(frame) <= 10
        assert frame.stay_id.nunique() <= 2
        return frame

    monkeypatch.setattr(pd, "read_parquet", bounded_read)
    receipt = ctx.validate_native_derivation_context(
        tmp_path, expected_manifest_sha256=manifest_hash
    )
    assert receipt["batches"] > 1 and receipt["positive_stays"] == 4
    assert read_sizes


def test_collector_copies_context_without_dataframe_reads(tmp_path, monkeypatch):
    sofa, suspicion = _inputs()
    recorder = ctx.SepsisContextRecorder(
        tmp_path, database="miiv", data_path="/synthetic"
    )
    recorder.derive(
        sofa, suspicion, ids=range(1, 10), id_col="stay_id", time_col="charttime"
    )
    reference = recorder.finish(expected_ids=range(1, 10))

    def refuse(*args, **kwargs):
        raise AssertionError(
            "collector must copy bounded bytes, not load context tables"
        )

    monkeypatch.setattr(pd, "read_parquet", refuse)
    assert ctx.transfer_context(tmp_path, tmp_path / "copy", reference) == reference


def test_oversize_stay_is_refused_without_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "MAX_CONTEXT_ROWS", 3)
    sofa, suspicion = _inputs()
    recorder = ctx.SepsisContextRecorder(
        tmp_path, database="miiv", data_path="/synthetic"
    )
    with pytest.raises(ctx.DerivationContextError, match="single-stay context exceeds"):
        recorder.derive(
            sofa, suspicion, ids=range(1, 10), id_col="stay_id", time_col="charttime"
        )
    assert not recorder.batches


def test_oversize_dependency_rejected_before_pandas_read(tmp_path, monkeypatch):
    frame = pd.DataFrame({"stay_id": [1] * 5, "charttime": range(5), "sofa": 0.0})
    ref = ctx._store(tmp_path, frame=frame)
    monkeypatch.setattr(ctx, "MAX_CONTEXT_ROWS", 4)

    def refuse(*args, **kwargs):
        raise AssertionError("oversize parquet was loaded")

    monkeypatch.setattr(pd, "read_parquet", refuse)
    with pytest.raises(ctx.DerivationContextError, match="exceeds read bound"):
        ctx._read_frame(tmp_path, ref, dependency=True)
