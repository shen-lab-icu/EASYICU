"""A streamed module resumes only the attempt it would itself have produced.

lead-r13 D3-RESUME-IDENTITY (Codex delta3): the stream ledger's identity was
the module, concepts, database, id column and cohort only, so an attempt
read from another installation, with other code or dictionaries, or with
other value-affecting load parameters was resumed as if it were this run's.
The identity now binds the source installation, the code and dictionary
digests and every load parameter except the execution ones (batch size,
workers). A changed publication closure refuses resume and preserves its
parts; other changed bindings start fresh and record why. Synthetic rows only.
"""

from __future__ import annotations

import hashlib
import json
import os

import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api

STAYS = list(range(1, 11))


def _frame(ids):
    return pd.DataFrame({
        "stay_id": ids, "charttime": [0.0] * len(ids),
        "test_signal": [float(v) * 1.5 for v in ids],
    })


def _run(out, monkeypatch, batch_size, load_kwargs, *, fail_on_call=None, calls=None):
    seen = calls if calls is not None else []

    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        seen.append(ids)
        if fail_on_call is not None and len(seen) == fail_on_call:
            raise MemoryError("simulated")
        return _frame(ids)

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)
    return api._stream_module_batches_to_parquet(
        "test_module", ["test_signal"], dict(load_kwargs), {"stay_id": STAYS},
        batch_size, str(out),
    )


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": {}}))
    (root / "table.parquet").write_bytes(b"synthetic")
    return root


def _kwargs(source, **extra):
    return {"database": "miiv", "concepts": ["test_signal"], "data_path": str(source),
            "merge": True, "use_sofa2": False, **extra}


def _interrupted(tmp_path, monkeypatch, kwargs):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(MemoryError):
        _run(out, monkeypatch, 4, kwargs, fail_on_call=2)
    assert (out / ".test_module.stream_ledger.json").exists()
    return out


def _resumed(out, monkeypatch, kwargs, batch_size=2):
    calls: list = []
    info = _run(out, monkeypatch, batch_size, kwargs, calls=calls)
    return info, sorted(v for ids in calls for v in ids)


def test_only_execution_parameters_may_change(tmp_path, monkeypatch, source):
    out = _interrupted(tmp_path, monkeypatch, _kwargs(source, concept_workers=1))
    info, extracted = _resumed(out, monkeypatch, _kwargs(source, concept_workers=4, verbose=True))
    assert extracted == STAYS[4:]  # resumed after the committed batch
    assert info["stream_resume_events"][0]["kind"] == "resume_after_worker_retry"


@pytest.mark.parametrize(
    ("change", "changed"),
    [
        (lambda src: _kwargs(src, use_sofa2=True), ["load_params_sha256"]),
        (lambda src: {**_kwargs(src), "interval": "2h"}, ["load_params_sha256"]),
    ],
)
def test_a_value_affecting_parameter_starts_fresh(tmp_path, monkeypatch, source, change, changed):
    out = _interrupted(tmp_path, monkeypatch, _kwargs(source))
    info, extracted = _resumed(out, monkeypatch, change(source))
    assert extracted == STAYS
    assert info["stream_resume_events"] == [
        {"kind": "resume_refused_identity_changed", "changed": changed, "discarded_batches": 1}
    ]


@pytest.mark.parametrize("change", ["rewrite", "touch"])
def test_an_edited_source_refuses_resume_and_preserves_parts(tmp_path, monkeypatch, source, change):
    from easyicu.storage_seal_publication import SealPublicationError, UNACCEPTED

    out = _interrupted(tmp_path, monkeypatch, _kwargs(source))
    table = source / "table.parquet"
    if change == "rewrite":
        table.write_bytes(b"synthetic, rebuilt")
    else:
        info = table.stat()
        os.utime(table, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(SealPublicationError) as caught:
        _resumed(out, monkeypatch, _kwargs(source))
    assert caught.value.reason == UNACCEPTED
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_another_installation_refuses_resume_and_preserves_parts(tmp_path, monkeypatch, source):
    from easyicu.storage_seal_publication import SealPublicationError, UNACCEPTED

    out = _interrupted(tmp_path, monkeypatch, _kwargs(source))
    other = tmp_path / "other"
    other.mkdir()
    (other / "conversion_manifest.json").write_text(json.dumps({"tables": {}}))
    ledger = out / ".test_module.stream_ledger.json"
    before = ledger.read_bytes()
    parts = set(out.glob("*.parquet"))
    with pytest.raises(SealPublicationError) as caught:
        _resumed(out, monkeypatch, _kwargs(other))
    assert caught.value.reason == UNACCEPTED
    assert ledger.read_bytes() == before
    assert set(out.glob("*.parquet")) == parts


def test_other_code_or_dictionaries_start_fresh(tmp_path, monkeypatch, source):
    out = _interrupted(tmp_path, monkeypatch, _kwargs(source))
    monkeypatch.setattr(api, "_stream_code_identity_sha256", lambda _db: "0" * 64)
    info, extracted = _resumed(out, monkeypatch, _kwargs(source))
    assert extracted == STAYS
    [event] = info["stream_resume_events"]
    assert event["changed"] == ["code_identity_sha256"]


def test_a_fresh_start_publishes_what_an_uninterrupted_run_publishes(tmp_path, monkeypatch, source):
    out = _interrupted(tmp_path, monkeypatch, _kwargs(source))
    info, _ = _resumed(out, monkeypatch, _kwargs(source, use_sofa2=True), batch_size=4)
    assert sorted(p.name for p in out.iterdir()) == ["test_module.parquet"]
    plain = tmp_path / "plain"
    plain.mkdir()
    reference = _run(plain, monkeypatch, 4, _kwargs(source, use_sofa2=True))

    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    assert sha(info["path"]) == sha(reference["path"])


def test_the_code_identity_binds_the_dictionaries(monkeypatch):
    api._stream_code_identity_sha256.cache_clear()
    base = api._native_export_runtime_provenance(source_name="miiv")
    monkeypatch.setattr(
        api, "_native_export_runtime_provenance",
        lambda source_name=None: {**base, "concept_dictionary_sha256": "f" * 64},
    )
    edited = api._stream_code_identity_sha256.__wrapped__("miiv")
    monkeypatch.undo()
    assert edited != api._stream_code_identity_sha256.__wrapped__("miiv")
