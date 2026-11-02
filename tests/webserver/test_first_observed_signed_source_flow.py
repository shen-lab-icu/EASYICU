"""A local report reopens the host-signed source flow without patient rereads."""

import os
import subprocess
import sys

from tests.support.first_observed_process import _PROCESS, _SOURCE_STUB
from tests.support.typed_export import typed_export
from easyicu.research_agent.intake.export_package import open_export_package


def test_signed_flow_reopens_without_raw_source_and_refuses_changed_attestation(
    tmp_path,
):
    export = typed_export(tmp_path / "synthetic-export", database="eicu")
    with open_export_package(export) as package:
        manifest = package.manifest_sha256
    (tmp_path / "sitecustomize.py").write_text(
        _SOURCE_STUB.replace('"b" * 64', repr(manifest))
    )
    env = os.environ.copy()
    env.update(
        EASYICU_HOME=str(tmp_path / "home"),
        EASYICU_TEST_FIRST_OBSERVED_SOURCE="1",
        EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
        PYTHONPATH=str(tmp_path) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    script = (
        _PROCESS.replace(
            'review_authority_sha256=request.authority_sha256, nonce="f"*64)',
            "review_authority_sha256=request.authority_sha256, nonce=canonical_sha256({"
            '"run_id": run_id, "study_context_id": study_id, "checkpoint_sha256": pending.checkpoint_sha256}))',
        )
        + r"""
from easyicu.webserver import agent_pipeline_runs, raw_source_authority
from easyicu.webserver.first_observed_deferred_execution import _load_signed_source_flow
(run / 'raw_preparation_checkpoint.json').write_text(approved.model_dump_json())
agent_pipeline_runs.raw_preparation_review_owner = lambda *_: (SimpleNamespace(run_dir=str(run), requests=hc.requests), {})
raw_source_authority.resolve_raw_source_identity_binding = lambda **_: (_ for _ in ()).throw(AssertionError('patient reread'))
recipe = {'host_binding': {'preparation_run_id':run_id, 'study_context_id':study_id,
                           'project_id':project, 'session_id':session}}
coordinate, flow, digest = _load_signed_source_flow(recipe)
assert coordinate.sha256 == resolved.sha256
assert flow['eligible_patients'] == 1 and flow['public_projection_authorized'] is False
assert len(digest) == 64
attestation = output.with_suffix('.json')
original = attestation.read_bytes(); attestation.write_bytes(original + b' ')
try:
    # Whitespace changes the attestation digest; signed semantics remain the same.
    _, _, changed = _load_signed_source_flow(recipe)
    assert changed != digest
finally:
    attestation.write_bytes(original)
document = json.loads(original); document['body']['private_source_flow']['eligible_patients'] = 2
attestation.write_text(json.dumps(document))
try:
    _load_signed_source_flow(recipe)
except (ValueError, RuntimeError): pass
else: raise AssertionError('tampered signed flow accepted')
print('signed local flow verified')
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "signed local flow verified" in result.stdout
