"""CodeRunner provenance details."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_runner_records_real_duration(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )
    result = runner.run(
        step_id="duration_probe",
        code="from pathlib import Path\nimport os\nPath(os.environ['STEP_OUT_DIR'], 'ok.txt').write_text('ok')\n",
    )

    assert result.succeeded
    assert 0 <= result.duration_seconds < 10
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "duration_seconds:" in log_text
