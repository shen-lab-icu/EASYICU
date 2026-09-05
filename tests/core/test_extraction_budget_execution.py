"""Exercise the real API dispatch without opening raw ICU tables or spawning."""

import inspect
from types import SimpleNamespace

import pytest

from easyicu.api import extraction as api


class DispatchCaptured(Exception):
    pass


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
