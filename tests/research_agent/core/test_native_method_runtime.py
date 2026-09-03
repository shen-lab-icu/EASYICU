from types import SimpleNamespace

import pytest

from easyicu.research_agent.execution import native_method_runtime as runtime


@pytest.fixture(autouse=True)
def reset_cache():
    runtime._probe_survival.cache_clear()
    yield
    runtime._probe_survival.cache_clear()


def test_native_probe_uses_an_immutable_data_free_network_disabled_image(monkeypatch):
    calls = []
    image_id = "sha256:" + "a" * 64
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/opt/docker")

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] <= 15
        return SimpleNamespace(
            returncode=0, stdout=image_id if "inspect" in argv else "4.5.0|3.8.3"
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)
    result = runtime.probe_time_varying_native_runtime("test:mutable-tag")
    assert result.available
    assert result.image_id == image_id
    assert result.survival_version == "3.8.3"
    assert image_id in calls[1] and "test:mutable-tag" not in calls[1]
    assert calls[1][calls[1].index("--network") + 1] == "none"
    assert "--read-only" in calls[1] and "--tmpfs" in calls[1]
    assert "--mount" not in calls[1] and "-v" not in calls[1]


def test_native_probe_does_not_fall_back_to_host_r(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/opt/docker")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(
            returncode=0 if "inspect" in argv else 1,
            stdout="sha256:" + "b" * 64 if "inspect" in argv else "",
        ),
    )
    result = runtime.probe_time_varying_native_runtime("test:no-r")
    assert not result.available
    assert result.reason_code == "time_varying_r_survival_unavailable"
