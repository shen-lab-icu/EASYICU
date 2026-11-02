"""The first-observed worker uses real host receipts in a separate process.

Only the six-column synthetic data source is stubbed; this is not an eICU
Demo result, a real user approval, or research execution.
"""
import os
from pathlib import Path
import subprocess
import sys

from tests.support.first_observed_process import _PROCESS, _SOURCE_STUB






def test_independent_first_observed_worker_with_real_host_state(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(_SOURCE_STUB)
    env = os.environ.copy()
    env.update(EASYICU_HOME=str(tmp_path / "home"), EASYICU_TEST_FIRST_OBSERVED_SOURCE="1",
               EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
               PYTHONPATH=str(tmp_path) + os.pathsep + env.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", _PROCESS, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "guarded preparation and recovery passed" in result.stdout


def test_signed_host_coordinate_can_reach_actual_typed_acquisition(tmp_path):
    """Synthetic source only; a preparation grant never becomes analysis approval."""
    from easyicu.research_agent.intake.export_package import open_export_package
    from tests.support.typed_export import typed_export

    export = typed_export(tmp_path / "synthetic-export", database="eicu")
    with open_export_package(export) as package:
        manifest = package.manifest_sha256
    stub = _SOURCE_STUB.replace('"b" * 64', repr(manifest))
    (tmp_path / "sitecustomize.py").write_text(stub)
    env = os.environ.copy()
    env.update(EASYICU_HOME=str(tmp_path / "home"), EASYICU_TEST_FIRST_OBSERVED_SOURCE="1",
               EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
               PYTHONPATH=str(tmp_path) + os.pathsep + env.get("PYTHONPATH", ""))
    script = _PROCESS + r'''
import pandas as pd
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
acquired=acquire_universe_for_question(export_dir=root/'synthetic-export',question='synthetic',llm=None,
    output_dir=root/'materialized',target_outcome='death',outcome_concepts=('death',),
    required_feature_concepts=('lact',),static_concepts=('age',),database='eicu',
    concept_selection_authority='host_exact',first_observed_icu_coordinate=resolved)
assert not acquired.blocked
cohort=pd.read_parquet(acquired.universe_path); trajectory=pd.read_parquet(acquired.trajectory_path)
assert cohort['stay_id'].tolist()==[1] and set(trajectory['stay_id'])=={1}
owner=load_verified_materialized_cohort_authority(acquired.universe_path)
restriction=owner.authority.producer_parameters['first_observed_icu_restriction']
assert restriction['preparation_run_id']==run_id
assert restriction['preparation_binding_sha256']==resolved.preparation_binding_sha256
assert restriction['coordinate_sha256']==resolved.sha256
assert restriction['preparation_checkpoint_sha256']==approved.checkpoint_sha256
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
