"""Exercise the real API dispatch without opening raw ICU tables or spawning."""

import inspect
import os
import tempfile
from types import SimpleNamespace

import pytest

from easyicu.api import extraction as api


class DispatchCaptured(Exception):
    pass


@pytest.fixture(autouse=True)
def preserve_process_temporary_settings():
    """The real API sets spill globals; do not leak them into other tests."""
    names = ("EASYICU_DUCKDB_TEMP_DIR", "TMPDIR")
    original_environment = {name: os.environ.get(name) for name in names}
    original_tempdir = tempfile.tempdir
    try:
        yield
    finally:
        tempfile.tempdir = original_tempdir
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.parametrize("module,expected_batch", [("respiratory", 5000), ("other_scores", 8000)])
@pytest.mark.parametrize("stream", [None, True])
def test_budgeted_api_dispatches_fixed_streaming_batches(monkeypatch, tmp_path, module, expected_batch, stream):
    captured = {}

    def process(*, target, args, daemon):
        captured.update(inspect.signature(target).bind(*args).arguments)
        raise DispatchCaptured

    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=process))
    with pytest.raises(DispatchCaptured):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=[module], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, stream_output_batches=stream, verbose=False,
        )
    assert captured["batch_size"] == expected_batch
    assert captured["stream_output_batches"] is True
    assert captured["adaptive_stream_batches"] is False
    assert captured["resource_budget_mb"] == 8192


def test_fixed_budget_rejects_explicit_adaptive_growth(tmp_path):
    with pytest.raises(ValueError, match="cannot override a fixed resource_budget_mb"):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=["respiratory"], patient_ids={"admissionid": [1]},
            stream_output_batches=True, adaptive_stream_batches=True,
            resource_budget_mb=8192, verbose=False,
        )


@pytest.mark.parametrize("has_output,stream", [(True, False), (False, None), (False, True)])
def test_fixed_batched_budget_cannot_collect_full_module_in_memory(tmp_path, has_output, stream):
    with pytest.raises(ValueError, match="requires an output_dir"):
        api.extract_database(
            "aumc", data_path=tmp_path,
            output_dir=tmp_path / "output" if has_output else None,
            modules=["respiratory"], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, stream_output_batches=stream, verbose=False,
        )


def test_adaptive_auto_stream_without_disk_output_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="requires stream_output_batches=True"):
        api.extract_database(
            "aumc", data_path=tmp_path, modules=["respiratory"],
            patient_ids={"admissionid": [1]}, adaptive_stream_batches=True,
            verbose=False,
        )


def test_budgeted_light_module_keeps_one_shot(monkeypatch, tmp_path):
    captured = {}

    def process(*, target, args, daemon):
        captured.update(inspect.signature(target).bind(*args).arguments)
        raise DispatchCaptured

    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=process))
    with pytest.raises(DispatchCaptured):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, verbose=False,
        )
    assert captured["batch_size"] == 23106
    assert captured["stream_output_batches"] is False
    assert captured["adaptive_stream_batches"] is False


@pytest.mark.parametrize(
    "database,total",
    [
        ("aumc", 23106), ("eicu", 200859), ("hirid", 33905),
        ("miiv", 94458), ("mimic", 61532), ("sic", 27386),
    ],
)
@pytest.mark.parametrize("module", api.EXTRACT_MODULE_ORDER)
def test_each_database_module_dispatch_preserves_its_own_plan(
    monkeypatch, tmp_path, database, total, module,
):
    """Check plan-to-worker wiring, not measured memory or optimal batch size."""
    captured = {}

    def process(*, target, args, daemon):
        captured.update(inspect.signature(target).bind(*args).arguments)
        raise DispatchCaptured

    def forbidden(*args, **kwargs):
        pytest.fail("dispatch contract tests must not load real ICU data")

    monkeypatch.setattr(api, "_get_all_patient_ids", forbidden)
    monkeypatch.setattr("easyicu.load_concepts", forbidden)
    monkeypatch.setattr("easyicu.api.load_concepts", forbidden)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=process))
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path))
    plan = api.plan_extraction_resources(
        database, [module], total, available_memory_mb=8192,
    )

    # Synthetic IDs only; capture before constructing or starting a worker.
    with pytest.raises(DispatchCaptured):
        api.extract_database(
            database, data_path=tmp_path, output_dir=tmp_path / "output",
            modules=[module], patient_ids=list(range(total)),
            resource_budget_mb=8192, verbose=False,
        )

    assert captured["batch_size"] == plan.batch_size
    assert captured["stream_output_batches"] is (plan.mode == "patient_batches")
    assert captured["adaptive_stream_batches"] is False
    assert captured["resource_budget_mb"] == 8192


@pytest.mark.parametrize("stream", [None, True])
def test_memory_killed_one_shot_retries_streamed_at_half_the_cohort(
    monkeypatch, tmp_path, stream
):
    calls = []
    oom_kills = [0]

    class KilledThenCaptured:
        def __init__(self, *, target, args, daemon):
            calls.append(inspect.signature(target).bind(*args).arguments)
            if len(calls) > 1:
                raise DispatchCaptured
            self.exitcode = None

        def start(self):
            # SIGKILL from the OOM killer: the cgroup records the event.
            oom_kills[0] += 1
            self.exitcode = -9

        def join(self):
            return None

    monkeypatch.setattr(
        api, "_get_extraction_mp_context",
        lambda _: SimpleNamespace(Process=KilledThenCaptured),
    )
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: oom_kills[0])
    with pytest.raises(DispatchCaptured):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, stream_output_batches=stream, verbose=False,
        )
    assert [call["stream_output_batches"] for call in calls] == [False, True]
    assert [call["batch_size"] for call in calls] == [23106, 11553]


def test_a_kill_without_oom_evidence_is_not_retried_as_memory(monkeypatch, tmp_path):
    """Exit -9 alone is not evidence (lead-r13 B-RETRY): no smaller retry."""
    calls = []

    class Killed:
        def __init__(self, *, target, args, daemon):
            calls.append(inspect.signature(target).bind(*args).arguments)
            self.exitcode = None

        def start(self):
            self.exitcode = -9

        def join(self):
            return None

    monkeypatch.setattr(
        api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=Killed)
    )
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: 0)
    with pytest.raises(api.NativeExportPublicationError):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, verbose=False,
        )
    assert len(calls) == 1


def test_memory_killed_one_shot_does_not_stream_when_the_caller_forbids_it(
    monkeypatch, tmp_path
):
    calls = []

    class Killed:
        def __init__(self, *, target, args, daemon):
            calls.append(inspect.signature(target).bind(*args).arguments)
            self.exitcode = None

        def start(self):
            self.exitcode = -9

        def join(self):
            return None

    monkeypatch.setattr(
        api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=Killed)
    )
    with pytest.raises(
        api.NativeExportPublicationError, match=r"failures=\['demographics'\]"
    ):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, stream_output_batches=False, verbose=False,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("error", "memory_exhausted", "retried"),
    [
        # The worker flags a memory exception by type (DuckDB
        # OutOfMemoryException); the text alone is not evidence.
        ("load_concepts(demographics): Out of Memory Error: could not allocate block", True, True),
        ("load_concepts(demographics): Out of Memory Error: could not allocate block", False, False),
        ("load_concepts(demographics): ValueError: bad unit", False, False),
    ],
)
def test_module_recording_a_duckdb_oom_retries_at_half_the_cohort(
    monkeypatch, tmp_path, error, memory_exhausted, retried
):
    import json as _json

    calls = []

    class RecordsModuleError:
        def __init__(self, *, target, args, daemon):
            bound = inspect.signature(target).bind(*args).arguments
            calls.append(bound)
            if len(calls) > 1:
                raise DispatchCaptured
            self.root = bound["output_root"]
            self.exitcode = None

        def start(self):
            # The worker survives a DuckDB OutOfMemoryException and records it.
            module_dir = os.path.join(self.root, "demographics")
            os.makedirs(module_dir, exist_ok=True)
            with open(os.path.join(module_dir, "_manifest.json"), "w") as handle:
                _json.dump({
                    "module": "demographics", "errors": [error], "saved": {},
                    "memory_exhausted": memory_exhausted,
                }, handle)
            self.exitcode = 0

        def join(self):
            return None

    monkeypatch.setattr(
        api, "_get_extraction_mp_context",
        lambda _: SimpleNamespace(Process=RecordsModuleError),
    )
    if retried:
        with pytest.raises(DispatchCaptured):
            api.extract_database(
                "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
                modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
                resource_budget_mb=8192, verbose=False,
            )
        assert [call["batch_size"] for call in calls] == [23106, 11553]
        assert [call["stream_output_batches"] for call in calls] == [False, True]
    else:
        with pytest.raises(api.NativeExportPublicationError):
            api.extract_database(
                "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
                modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
                resource_budget_mb=8192, verbose=False,
            )
        assert len(calls) == 1
