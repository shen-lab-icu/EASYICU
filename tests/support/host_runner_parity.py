"""Byte-level execution parity on synthetic inputs, without Docker or sockets."""

import hashlib
import json
import os
from pathlib import Path

from easyicu.research_agent.execution.host_runner import HostRunner
from easyicu.research_agent.execution.runner import CodeRunner
from easyicu.research_agent.execution.step_execution import CodeOrigin
from easyicu.research_agent.execution.runners.selection import resolve_standard_executor

_SELECTED_OWNERS = {}


def select_for_parity(step, **kwargs):
    decision = resolve_standard_executor(step, **kwargs)
    selected = decision.render_selection()
    if selected is not None:
        _SELECTED_OWNERS[hashlib.sha256(selected.code.encode()).hexdigest()] = (
            decision._executor.key
        )
    return selected


def table_binding(run_dir, key, frame):
    """Bind all rows of an invented upstream table to its content digest."""
    product = key.split(":")[1]
    path = run_dir / f"{product}.csv"
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "input_key": key,
        "product": product,
        "sha256": digest,
        "declared_kind": "table",
        "evidence_id": product,
        "produced_by_step": "synthetic_source",
    }
    return {
        **identity,
        "evidence_kind": "table",
        "relative_path": path.name,
        "identity_row": identity,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }


def assert_parity(workdir, cohort, step, code, manifest, extra_env):
    legacy = CodeRunner(
        workdir=workdir,
        cohort_parquet=cohort,
        allow_unsafe_host_fallback=True,
        extra_env=extra_env,
    )
    host = HostRunner(workdir=workdir, cohort_parquet=cohort, extra_env=extra_env)
    # Freeze Matplotlib's UUID salt and creation timestamps for the comparison;
    # both runners execute the exact renderer bytes with the same settings.
    for runner in (legacy, host):
        cache = runner._matplotlib_config_dir(workdir / "steps" / step.step_id)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "matplotlibrc").write_text("svg.hashsalt: synthetic-parity\n")
    # This test exercises the existing Python path, without invoking unshare or Docker.
    legacy.build_command = lambda *, script_path: [
        legacy.python_executable,
        str(script_path),
    ]
    first = legacy.run(step_id=step.step_id, code=code, resolved_inputs_path=manifest)
    assert first.succeeded, first.stderr
    expected = {
        path.relative_to(first.out_dir).as_posix(): path.read_bytes()
        for path in first.artefacts
    }
    second = host.run(
        step_id=step.step_id,
        code=code,
        resolved_inputs_path=manifest,
        code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR,
    )
    assert second.succeeded, second.stderr
    actual = {
        path.relative_to(second.out_dir).as_posix(): path.read_bytes()
        for path in second.artefacts
    }
    assert actual == expected
    assert actual
    if directory := os.environ.get("EASYICU_HOST_PARITY_COVERAGE_DIR"):
        digest = hashlib.sha256(code.encode()).hexdigest()
        owner = _SELECTED_OWNERS[digest]
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{owner}.json").write_text(
            json.dumps(
                {"owner": owner, "code_sha256": digest, "products": sorted(actual)}
            )
        )
