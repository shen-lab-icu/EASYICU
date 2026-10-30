"""Pre-review population proof uses real host receipts, with synthetic source data."""
from __future__ import annotations

import os
import subprocess
import sys

from easyicu.research_agent.intake.export_package import open_export_package
from tests.support.typed_export import typed_export
from tests.webserver.test_first_observed_guarded_process import _PROCESS, _SOURCE_STUB


def test_host_signed_preparation_proves_typed_rows_without_granting_analysis(tmp_path):
    export = typed_export(tmp_path / 'synthetic-export', database='eicu')
    with open_export_package(export) as package:
        manifest = package.manifest_sha256
    (tmp_path / 'sitecustomize.py').write_text(_SOURCE_STUB.replace('"b" * 64', repr(manifest)))
    env = os.environ.copy()
    env.update(EASYICU_HOME=str(tmp_path / 'home'), EASYICU_TEST_FIRST_OBSERVED_SOURCE='1',
               EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
               PYTHONPATH=str(tmp_path) + os.pathsep + env.get('PYTHONPATH', ''))
    process = _PROCESS.replace(
        'review_authority_sha256=request.authority_sha256, nonce="f"*64)',
        'review_authority_sha256=request.authority_sha256, nonce=canonical_sha256({'
        '"run_id": run_id, "study_context_id": study_id, "checkpoint_sha256": pending.checkpoint_sha256}))',
    )
    assert process != _PROCESS
    script = process + r'''
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver.first_observed_run_guard import verify_prepared_first_observed_population
(run / 'raw_preparation_checkpoint.json').write_text(approved.model_dump_json())
acquired = acquire_universe_for_question(export_dir=root/'synthetic-export', question='synthetic', llm=None,
    output_dir=root/'materialized', target_outcome='death', outcome_concepts=('death',),
    required_feature_concepts=('lact',), static_concepts=('age',), database='eicu',
    concept_selection_authority='host_exact', first_observed_icu_coordinate=resolved)
assert not acquired.blocked
criterion = spec.criteria[0].model_dump(mode='json')
source_identity = SourceIdentity.from_record(identity)
args = dict(cohort_path=acquired.universe_path, study_context_id=study_id,
            criterion=criterion, source_identity=source_identity)
proof = verify_prepared_first_observed_population(**args)
assert proof.coordinate_sha256 == resolved.sha256
assert proof.preparation_checkpoint_sha256 == approved.checkpoint_sha256
assert proof.preparation_run_id == run_id
for changed in (
    {'criterion': {**criterion, 'quote': 'not the reviewed criterion'}},
    {'source_identity': SourceIdentity(database='eicu_demo', registration_id='other', archive_sha256='a'*64)},
    {'study_context_id': 'other-study'},
):
    try:
        verify_prepared_first_observed_population(**{**args, **changed})
    except (ValueError, RuntimeError, KeyError):
        pass
    else:
        raise AssertionError('changed host population binding was accepted')
from dataclasses import replace
import pandas as pd
fake_path = root/'forged-coordinate.parquet'
pd.DataFrame({'stay_id': [1, 2, 3], FLAG_COLUMN: [True, True, True],
              REASON_COLUMN: ['kept', 'kept', 'kept']}).to_parquet(fake_path, index=False)
fake_coordinate = replace(resolved, path=fake_path, sha256=hashlib.sha256(fake_path.read_bytes()).hexdigest())
forged = acquire_universe_for_question(export_dir=root/'synthetic-export', question='synthetic', llm=None,
    output_dir=root/'forged-materialized', target_outcome='death', outcome_concepts=('death',),
    required_feature_concepts=('lact',), static_concepts=('age',), database='eicu',
    concept_selection_authority='host_exact', first_observed_icu_coordinate=fake_coordinate)
assert not forged.blocked
try:
    verify_prepared_first_observed_population(**{**args, 'cohort_path': forged.universe_path})
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError('self-sealed cohort with forged coordinate was accepted')
receipt_path = root/'home'/'.easyicu'/'private-preparation-decisions'/(choice.nonce+'.json')
original_receipt = receipt_path.read_bytes()
receipt = json.loads(original_receipt)
receipt['signature'] = '0'*64
receipt_path.write_text(json.dumps(receipt))
try:
    verify_prepared_first_observed_population(**args)
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError('tampered host signature was accepted')
receipt_path.write_bytes(original_receipt)
original_coordinate = output.read_bytes()
output.write_bytes(original_coordinate + b'changed')
try:
    verify_prepared_first_observed_population(**args)
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError('tampered coordinate was accepted')
print('host-signed pre-review population proof verified')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert 'host-signed pre-review population proof verified' in result.stdout
