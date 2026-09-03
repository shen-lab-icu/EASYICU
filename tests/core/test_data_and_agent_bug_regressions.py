"""Cross-database conversion and research-agent execution bug regressions.

Origin: 2026-05-16 cross-database audit.

Each test pins a specific failure mode so the fix can't quietly regress:

1. eicu/respiratorycharting `sub_var` plumbing — ventilator SQL used
   `WHERE labname IN (...)` instead of `respchartvaluelabel` (`needs_real_data`).
2. CSV→Parquet converter MIXED_TYPE_COLUMNS override — mimic chartevents `value`
   column must stay VARCHAR or the GCS text values silently become NULL.
3. `load_bucketed_table_aggregated` VARCHAR bounds — eicu respiratorycharting
   value is string-typed, `value_min/max` filtering must `TRY_CAST` it
   (`needs_real_data`).
4. Truncated parquet detection — `DataConverter._has_parquet_footer` must
   reject a partial file so reconversion is triggered.
5. `ts_utils.expand` mixed numeric/datetime tolerance — when start_var is
   float (hours since admission) and end_var is datetime, the function must
   not raise; it falls back to a NA-only mask.

Where possible the tests use tiny synthetic fixtures so they run quickly
without `--run-real`. Tests that genuinely need a working eicu install carry
the `needs_real_data` marker and are skipped by default.
"""

from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

needs_real_data = pytest.mark.needs_real_data


def _authorized_local_openai_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    completions: object,
    max_retries: int,
):
    """Construct the real adapter through the reviewed local-provider factory."""

    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.llm import OpenAIClient

    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("EASYICU_LLM_MAX_RETRIES", str(max_retries))
    return build_provider_client(
        provider="openai",
        model="stub",
        base_url_override="http://127.0.0.1:8787/v1",
        request_timeout=1.0,
        title="EasyICU local retry regression",
        client_cls=OpenAIClient,
    )


# ---------------------------------------------------------------------------
# Bug #1 + #3 — eicu respiratorycharting end-to-end (sub_var + VARCHAR bounds)
# ---------------------------------------------------------------------------
@needs_real_data
def test_eicu_ventilator_peep_loads_rows():
    """Pre-fix: BinderException on `labname` column + DECIMAL/VARCHAR mismatch.

    Reproduces both bug #1 (sub_var not plumbed → wrong itemid column) and
    bug #3 (value bounds compared against VARCHAR without TRY_CAST). Either
    failure mode would raise; after the fix the call returns a DataFrame.
    """
    from easyicu.api import load_concepts

    df = load_concepts(
        "peep",
        database="eicu",
        max_patients=500,
        verbose=False,
    )
    # Don't assert non-empty — 500 patients may not all have PEEP — but the
    # call must succeed and return the expected schema.
    assert isinstance(df, pd.DataFrame)
    assert "peep" in df.columns


# ---------------------------------------------------------------------------
# Bug #2 — converter must pin mimic chartevents VALUE to a string type
# ---------------------------------------------------------------------------
def test_mimic_chartevents_value_stays_varchar(tmp_path):
    """Pre-fix: type inference sampled the first rows of CHARTEVENTS.csv.gz,
    saw an all-numeric VALUE, picked a numeric type, and silently dropped
    categorical GCS values like '4 Spontaneously' (they became NULL).

    This test feeds a tiny CSV.gz mimicking that schema: first 110 rows are
    pure numeric, the last row is the text value. Without the
    MIXED_TYPE_COLUMNS override the text row becomes NULL.
    """
    from easyicu.io.data_converter import ConversionStatus, DataConverter

    # Build tiny CSV.gz with the mimic-iii chartevents column shape
    csv_text = "row_id,subject_id,hadm_id,icustay_id,itemid,charttime,value,valuenum\n"
    # 110 numeric rows so type inference's first sample is "all numeric"
    for i in range(110):
        csv_text += f"{i+1},1,1,1,184,2020-01-01,{15+i*0.01},{15+i*0.01}\n"
    # Categorical row that should NOT be lost
    csv_text += "111,1,1,1,184,2020-01-01,4 Spontaneously,4\n"

    src = tmp_path / "CHARTEVENTS.csv.gz"
    with gzip.open(src, "wt") as f:
        f.write(csv_text)

    # 'mimic' has no chartevents partitioning config → flat parquet output.
    converter = DataConverter(str(tmp_path), database="mimic", verbose=False)
    result = converter.convert_all()["CHARTEVENTS.csv.gz"]
    assert result["status"] == ConversionStatus.COMPLETED, result.get("error")
    assert (
        result["row_count"] == 111
    ), "all rows including the text-value row must be kept"

    # Schema sanity: VALUE must be VARCHAR, the text value preserved.
    import duckdb

    out = tmp_path / "chartevents.parquet"
    con = duckdb.connect()
    types = {
        r[0]: r[1]
        for r in con.execute(
            f"DESCRIBE SELECT value FROM read_parquet('{out}') LIMIT 0"
        ).fetchall()
    }
    assert (
        types["value"].upper().startswith("VARCHAR")
    ), f"value column must stay VARCHAR, got {types['value']}"
    rows = con.execute(
        f"SELECT value FROM read_parquet('{out}') WHERE value LIKE '%Spontaneously%'"
    ).fetchall()
    assert rows and rows[0][0] == "4 Spontaneously", "categorical text row must survive"


# ---------------------------------------------------------------------------
# Bug #4 — DataConverter must detect truncated parquet shards
# ---------------------------------------------------------------------------
def test_truncated_shard_is_rejected(tmp_path):
    """Pre-2026-05 fix: shard scan only checked filename, so an interrupted
    conversion left `vitalperiodic/1.parquet` present-but-truncated and the
    scan happily marked the table as "converted". The footer check (last 4
    bytes == ``PAR1``) catches that.
    """
    from easyicu.io.data_converter import DataConverter

    # Build a real shard layout: <table>/1.parquet ... 3.parquet
    table_dir = tmp_path / "vitalperiodic"
    table_dir.mkdir()
    df = pd.DataFrame({"id": range(10), "v": range(10)})
    for i in (1, 2, 3):
        pq.write_table(pa.Table.from_pandas(df), table_dir / f"{i}.parquet")

    # Corrupt shard 2 by chopping its last 8 bytes (kills the magic footer)
    bad = table_dir / "2.parquet"
    raw = bad.read_bytes()
    bad.write_bytes(raw[:-8])

    # DataConverter expects a database directory; use tmp_path as data root
    converter = DataConverter(str(tmp_path), verbose=False)
    cache = converter._scan_shard_dirs()
    assert "vitalperiodic" not in cache, (
        "truncated shard must invalidate the cache so reconversion runs; "
        f"got cache={cache}"
    )


# ---------------------------------------------------------------------------
# Bug #5 — expand() must tolerate numeric/datetime column mismatch
# ---------------------------------------------------------------------------
def test_expand_tolerates_mixed_numeric_and_datetime():
    """Pre-fix: when start_var was float (hours since admission) but end_var
    was datetime64 (raw from inputevents), the
    ``data[start_var] <= data[end_col]`` comparison raised
    ``TypeError: Invalid comparison between dtype=datetime64[us] and ndarray``.

    After the fix the function falls back to a NA-only mask without raising
    (and the caller can keep raw rows for downstream use).
    """
    from easyicu.io.ts_utils import expand

    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            # numeric hours since admission
            "charttime": [2.0, 5.0, 1.0],
            # datetime end — incompatible type, used to crash the comparison
            "endtime": pd.to_datetime(
                ["2020-01-01 02:00", "2020-01-01 05:00", "2020-01-02 01:00"]
            ),
            "delirium_tx": [True, True, True],
        }
    )

    # Should not raise; should return a frame (may be empty after fallback).
    out = expand(
        df,
        start_var="charttime",
        end_var="endtime",
        step_size=pd.Timedelta(hours=1),
        id_cols=["stay_id"],
        keep_vars=["delirium_tx"],
    )
    assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Bug #6 (agent layer) — execution gate must count the deterministic 00_probe
# ---------------------------------------------------------------------------
def test_execution_gate_counts_deterministic_probe():
    """Pre-fix: ``reporting.readiness.execution_gate_status`` filtered ``00_probe``
    out of ``status_by_step`` while still keeping it in ``required_step_ids``,
    so the deterministic probe was always reported as a missing required step
    and ``execution_complete`` was forced to False. Pilot
    run_20260516T123840_cc32d5 surfaced this as `00_probe missing`."""
    from easyicu.research_agent.reporting.readiness import execution_gate_status
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="dummy question",
        steps=[
            AnalysisStep(step_id="00_probe", intent="probe distributions"),
            AnalysisStep(step_id="01_real_work", intent="do the real work"),
        ],
    )
    per_step = [
        {"step_id": "00_probe", "status": "ok"},
        {"step_id": "01_real_work", "status": "ok"},
    ]
    gate = execution_gate_status(plan=plan, per_step_records=per_step)
    assert gate["missing_steps"] == [], gate
    assert gate["failed_steps"] == [], gate
    assert gate["execution_complete"] is True, gate
    assert gate["completed_step_count"] == 2


# ---------------------------------------------------------------------------
# Bug #7 (agent layer) — OpenAIClient must retry transient JSONDecodeError
# ---------------------------------------------------------------------------
def test_openai_client_retries_json_decode_error(monkeypatch):
    """Pre-fix: a JSONDecodeError raised by the openai SDK (free-tier providers
    occasionally stream malformed chunks) was treated as fatal by the coder
    repair path, killing the whole step. After the fix the LLM client retries
    transient parse errors with short backoff."""
    import json
    from easyicu.research_agent.providers.llm import LLMMessage

    calls = {"n": 0}

    class _StubCreate:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                # First two calls fail with a JSON parse error mid-stream
                raise json.JSONDecodeError("Expecting value", "", 0)

            # Third call succeeds — return a minimal openai-shaped response
            class _Msg:
                content = "ok"

                def model_dump(self):
                    return {"role": "assistant", "content": "ok"}

            class _Choice:
                message = _Msg()
                finish_reason = "stop"

            class _Resp:
                choices = [_Choice()]
                usage = None

            return _Resp()

    client = _authorized_local_openai_client(
        monkeypatch,
        completions=_StubCreate(),
        max_retries=3,
    )

    # Monkeypatch time.sleep so the retries are instant in the test
    import time

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    out = client.complete(
        [LLMMessage(role="user", content="hi")],
        max_tokens=8,
        temperature=0.0,
    )
    assert out == "ok"
    assert calls["n"] == 3, f"expected 2 retries + 1 success, got {calls['n']} calls"


# ---------------------------------------------------------------------------
# Bug #8 (agent layer) — figure-contract must accept `skipped` summaries
# ---------------------------------------------------------------------------
def test_step_contract_accepts_skipped_figure_step():
    """Pre-fix: a figure-only step that legitimately reported
    ``"skipped": ["No SOFA-2 components available"]`` still failed the
    figure-output contract because the validator only looked for
    ``figure_path`` / ``figure_files`` keys. Pilot run_20260516T132601_ba6315
    blocked step 11_sofa2_component_figure on exactly this."""
    from easyicu.research_agent.plan_utils import _step_contract_findings
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="11_component_figure",
        intent="produce a publication-ready figure of component associations",
        expected_outputs=["figure:component_associations"],
    )
    skipped_summary = {
        "step_id": "11_component_figure",
        "analysis_type": "association",
        "skipped": ["No SOFA-2 components available in the dataset"],
        "figure_path": None,
    }
    findings = _step_contract_findings(step=step, step_summary=skipped_summary)
    figure_errors = [
        f
        for f in findings
        if "figure" in (f.message or "").lower() and f.severity == "error"
    ]
    assert (
        not figure_errors
    ), f"figure-only step that reported `skipped` must not be flagged; got {figure_errors}"


# ---------------------------------------------------------------------------
# Bug #9 (agent layer) — OpenAIClient must retry null-choices envelope
# ---------------------------------------------------------------------------
def test_openai_client_retries_null_choices(monkeypatch):
    """Pre-fix: free OpenRouter tier sometimes returns `choices=None` on partial
    failure; `resp.choices[0]` then raised the cryptic
    `'NoneType' object is not subscriptable` and killed the step. After the
    fix the LLM client treats null-choices/null-message as transient."""
    from easyicu.research_agent.providers.llm import LLMMessage

    calls = {"n": 0}

    class _Msg:
        content = "ok"

        def model_dump(self):
            return {"role": "assistant", "content": "ok"}

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _GoodResp:
        choices = [_Choice()]
        usage = None

    class _NullChoicesResp:
        choices = None
        usage = None
        finish_reason = "error"

    class _StubCreate:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _NullChoicesResp()
            return _GoodResp()

    client = _authorized_local_openai_client(
        monkeypatch,
        completions=_StubCreate(),
        max_retries=2,
    )

    import time

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    out = client.complete(
        [LLMMessage(role="user", content="hi")],
        max_tokens=8,
        temperature=0.0,
    )
    assert out == "ok"
    assert calls["n"] == 2, f"expected 1 retry + 1 success, got {calls['n']} calls"


# ---------------------------------------------------------------------------
# Bug #10 (agent layer) — deterministic runner repair must strip
#                         hallucinated easyicu.research_agent.* imports
# ---------------------------------------------------------------------------
def test_runner_repair_strips_fake_easyicu_import():
    """Pre-fix: when an LLM hallucinated
    `from easyicu.research_agent.rcs import restricted_cubic_spline`,
    repair attempts kept re-emitting the same bad import and the step died
    with `ModuleNotFoundError`. Pilot run_20260516T182501_329a01 lost step
    03 to exactly this. The runner repair now strips such imports and
    stubs the name with a clear NotImplementedError so the next repair
    attempt sees an actionable error."""
    from easyicu.research_agent.repairs.source import _deterministic_runner_repair

    bad_code = (
        "import pandas as pd\n"
        "from easyicu.research_agent.rcs import restricted_cubic_spline\n"
        "df = pd.read_parquet('cohort.parquet')\n"
        "spline = restricted_cubic_spline(df['sofa2'], df['death'])\n"
    )
    run_log = (
        "Traceback (most recent call last):\n"
        '  File "/run.py", line 2, in <module>\n'
        "    from easyicu.research_agent.rcs import restricted_cubic_spline\n"
        "ModuleNotFoundError: No module named 'easyicu.research_agent.rcs'\n"
    )
    result = _deterministic_runner_repair(code=bad_code, run_log=run_log)
    assert result is not None, "runner repair must fire for fake easyicu import"
    repair_name, repaired = result
    assert "strip_fake_easyicu_import" in repair_name
    # The original `from ... import ...` line must be gone (only the comment
    # marker referencing the module path remains for traceability).
    import re as _re

    assert not _re.search(
        r"^from\s+easyicu\.research_agent\.rcs\s+import",
        repaired,
        flags=_re.MULTILINE,
    ), repaired
    assert "restricted_cubic_spline" in repaired  # stub keeps the name defined
    assert "NotImplementedError" in repaired


def test_runner_repair_stops_when_reported_easyicu_module_exists(monkeypatch):
    """An import-loader failure is not authority to rewrite an existing module."""

    from easyicu.research_agent.repairs import runner_dispatch, source

    monkeypatch.setattr(
        runner_dispatch, "host_module_is_available", lambda _name: True
    )
    result = source._deterministic_runner_repair_candidate(
        code=(
            "from easyicu.research_agent.schema import AnalysisStep\n"
            "print(os.environ)\n"
        ),
        run_log=(
            "ModuleNotFoundError: No module named "
            "'easyicu.research_agent.schema'\n"
            "NameError: name 'os' is not defined\n"
        ),
    )

    assert result is None
