import inspect
import tempfile
from types import SimpleNamespace

import pytest

from easyicu.api import extraction as api


class GroupDispatched(Exception):
    pass


@pytest.mark.parametrize("stream_output_batches", [False, True])
@pytest.mark.parametrize(
    ("module", "isolate_shared", "expected_daemon"),
    [
        pytest.param("sepsis3_sofa2", False, False, id="isolated-score"),
        pytest.param("sepsis3_sofa1", False, True, id="no-isolated-dependency"),
        pytest.param("sepsis3_sofa1", True, False, id="isolated-shared"),
    ],
)
def test_special_dependency_controls_group_daemon(
    monkeypatch, tmp_path, stream_output_batches, module, isolate_shared, expected_daemon
):
    if isolate_shared:
        monkeypatch.setattr(
            api,
            "_ISOLATED_STREAM_BATCH_TARGETS",
            api._ISOLATED_STREAM_BATCH_TARGETS | {("eicu", "sepsis_shared")},
        )

    def process(*, target, args, daemon):
        assert target is api._extract_module_group_worker
        worker = inspect.signature(target).bind(*args).arguments
        assert worker["module_specs"] == []
        assert worker["special_modules"] == [module]
        assert worker["stream_output_batches"] is stream_output_batches
        assert daemon is expected_daemon
        raise GroupDispatched

    monkeypatch.setattr(
        api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=process)
    )
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(GroupDispatched):
        api.extract_database(
            "eicu",
            data_path=tmp_path,
            output_dir=tmp_path / "export",
            modules=[module],
            patient_ids={"patientunitstayid": [1, 2]},
            batch_size=2,
            native_export_v2=True,
            stream_output_batches=stream_output_batches,
            verbose=False,
        )
