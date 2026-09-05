"""Docker mount socket rejection must not depend on the default spill path."""

from pathlib import Path
import socket
import tempfile

import pandas as pd
import pytest


@pytest.mark.parametrize("long_default_tmp", [False, True])
def test_docker_runner_rejects_socket_inside_extra_mount_directory(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    long_default_tmp: bool,
):
    from easyicu.research_agent.execution import runner as runner_mod

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    if long_default_tmp:
        deep = tmp_path / ("nested-" * 20)
        deep.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(deep))
    # AF_UNIX addresses have a small OS limit. A prior extraction test or an
    # xdist worker may set a much longer default spill path; use a private
    # short directory while keeping the real unsafe-special-file assertion.
    with tempfile.TemporaryDirectory(prefix="easyicu-mount-", dir="/tmp") as directory:
        source = Path(directory).resolve()
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(source / "service.sock"))
            with pytest.raises(ValueError, match="unsafe special file"):
                ra.DockerRunner(
                    workdir=tmp_path / "run",
                    cohort_parquet=cohort,
                    extra_mounts=[(str(source), "/easyicu-extra/source", "ro")],
                )
