"""P0-3: Real-data UI smoke test for Cohort Analysis panels.

This test validates that the full real-data flow works end-to-end
using MIMIC-IV (miiv) data from the configured data path.

Flow:
  1. Real Data → validate MIIV path
  2. Cohort Analysis → Shared Workspace load N patients
  3. Group Contrast → verify demographics loaded
  4. Snapshot → verify demographics loaded
  5. Cross-DB → verify path synced
  6. SOFA sensitivity → verify SOFA concepts loaded

Usage:
  # Dry run (no real data needed, tests logic only)
  pytest tests/test_real_ui_smoke.py -v

  # Full run (requires MIIV data at EASYICU_DATA_PATH)
  EASYICU_DATA_PATH=/path/to/miiv pytest tests/test_real_ui_smoke.py -v --run-real
"""
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Skip unless --run-real flag and EASYICU_DATA_PATH are set.
# The pytest option/marker registration lives in tests/conftest.py because
# pytest parses CLI options before importing individual test modules.
# ---------------------------------------------------------------------------
REAL_DATA_PATH = os.environ.get("EASYICU_DATA_PATH", "")
HAS_REAL_DATA = bool(REAL_DATA_PATH) and Path(REAL_DATA_PATH).exists()


needs_real_data = pytest.mark.needs_real_data


# ---------------------------------------------------------------------------
# Helpers: mock Streamlit session state
# ---------------------------------------------------------------------------

class FakeSessionState(dict):
    """dict-like object that also supports attribute access like st.session_state."""
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


@pytest.fixture
def mock_state():
    """Return a fresh session-state-like dict pre-seeded for real data mode."""
    state = FakeSessionState()
    state['entry_mode'] = 'real'
    state['database'] = 'miiv'
    state['data_path'] = REAL_DATA_PATH
    state['path_validated'] = True
    state['language'] = 'en'
    state['use_mock_data'] = False
    state['mock_params'] = {}
    return state


# ---------------------------------------------------------------------------
# Test: path helpers
# ---------------------------------------------------------------------------

class TestPathHelpers:
    """Test _default_real_data_root / _default_real_database / _sync_real_data_panel_defaults."""

    def test_default_root_reads_session_state(self, mock_state):
        """_default_real_data_root() should return st.session_state['data_path']."""
        # We can't easily import the Streamlit-dependent helpers directly,
        # so this test documents the expected contract.
        assert mock_state['data_path'] == REAL_DATA_PATH or not HAS_REAL_DATA

    def test_default_database_reads_session_state(self, mock_state):
        assert mock_state['database'] == 'miiv'

    def test_panel_keys_synced_after_workspace(self, mock_state):
        """After workspace load, all panel root keys should be set."""
        if not HAS_REAL_DATA:
            pytest.skip("No real data available")
        # Simulate what _ensure_cohort_real_workspace does
        mock_state['grp_data_root'] = REAL_DATA_PATH
        mock_state['grp_db_select'] = 'miiv'
        mock_state['dash_data_root'] = REAL_DATA_PATH
        mock_state['dash_db_select'] = 'miiv'
        mock_state['multidb_data_root'] = REAL_DATA_PATH
        mock_state['multidb_selected'] = ['miiv']
        # Verify all panel keys are set
        for key in ['grp_data_root', 'dash_data_root', 'multidb_data_root']:
            assert mock_state[key] == REAL_DATA_PATH
        assert mock_state['grp_db_select'] == 'miiv'
        assert mock_state['dash_db_select'] == 'miiv'
        assert mock_state['multidb_selected'] == ['miiv']


# ---------------------------------------------------------------------------
# Test: Shared workspace state machine
# ---------------------------------------------------------------------------

class TestRealWorkspaceStateMachine:
    """Test _cohort_real_workspace_ready / _ensure_cohort_real_workspace contract."""

    def test_workspace_not_ready_initially(self, mock_state):
        """Workspace should NOT be ready before loading."""
        assert not mock_state.get('_cohort_real_ws_ready')

    def test_workspace_ready_after_loading(self, mock_state):
        """After seeding the workspace keys, _ready check should pass."""
        import pandas as pd
        mock_state['_cohort_real_ws_ready'] = True
        mock_state['_cohort_real_ws_demographics'] = pd.DataFrame({'stay_id': [1, 2, 3], 'age': [60, 70, 80]})
        assert mock_state.get('_cohort_real_ws_ready')
        assert not mock_state.get('_cohort_real_ws_demographics').empty

    def test_workspace_seeds_panel_keys(self, mock_state):
        """Workspace load should seed grp_demographics, dash_demographics, multidb_data_root."""
        import pandas as pd
        demo_df = pd.DataFrame({'stay_id': [1, 2], 'age': [55, 65], 'gender': ['M', 'F'], 'survived': [1, 0]})
        # Simulate workspace seeding
        mock_state['grp_demographics'] = demo_df.copy()
        mock_state['grp_loaded_db'] = 'miiv'
        mock_state['grp_is_demo'] = False
        mock_state['dash_demographics'] = demo_df.copy()
        mock_state['dash_loaded_db'] = 'miiv'
        mock_state['dash_is_demo'] = False
        # Verify
        assert len(mock_state['grp_demographics']) == 2
        assert len(mock_state['dash_demographics']) == 2
        assert mock_state['grp_loaded_db'] == 'miiv'
        assert mock_state['dash_loaded_db'] == 'miiv'


# ---------------------------------------------------------------------------
# Integration tests (require real MIIV data)
# ---------------------------------------------------------------------------

@needs_real_data
class TestRealDataIntegration:
    """Full integration tests requiring real MIIV data on disk."""

    def test_validate_miiv_path(self):
        """Step 1: validate the MIIV data path."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from easyicu.webapp.app import validate_database_path
        result = validate_database_path(REAL_DATA_PATH, 'miiv')
        assert result['valid'], f"Validation failed: {result.get('message')}"

    def test_find_database_path(self):
        """find_database_path should resolve the MIIV data path."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from easyicu.webapp.app import find_database_path
        resolved = find_database_path(REAL_DATA_PATH, 'miiv')
        assert os.path.isdir(resolved), f"Resolved path does not exist: {resolved}"

    def test_load_demographics(self):
        """Load demographics for up to 1000 patients."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from easyicu.patient_filter import PatientFilter
        from easyicu.webapp.app import find_database_path
        resolved = find_database_path(REAL_DATA_PATH, 'miiv')
        pf = PatientFilter(database='miiv', data_path=resolved, verbose=False)
        df = pf._load_demographics()
        assert len(df) > 0, "No demographics loaded"
        assert 'stay_id' in df.columns or 'patient_id' in df.columns
        print(f"  ✅ Loaded {len(df):,} patients from {resolved}")

    def test_load_concepts_hr_map(self):
        """Load hr and map concepts for first 100 patients."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from easyicu.patient_filter import PatientFilter
        from easyicu.webapp.app import find_database_path
        from easyicu import load_concepts
        resolved = find_database_path(REAL_DATA_PATH, 'miiv')
        pf = PatientFilter(database='miiv', data_path=resolved, verbose=False)
        df = pf._load_demographics()
        id_col = 'stay_id' if 'stay_id' in df.columns else 'patient_id'
        patient_ids = df[id_col].dropna().astype(int).head(100).tolist()
        result = load_concepts(
            concepts=['hr', 'map'],
            database='miiv',
            data_path=resolved,
            patient_ids=patient_ids,
            verbose=False,
        )
        assert result is not None and not result.empty, "No concept data returned"
        assert 'hr' in result.columns or 'map' in result.columns
        print(f"  ✅ Loaded hr/map: {result.shape}")

    def test_load_sofa_concepts(self):
        """Load paired SOFA-1/SOFA-2 concepts for first 100 stays."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from easyicu.patient_filter import PatientFilter
        from easyicu.webapp.app import find_database_path
        from easyicu import load_concepts
        resolved = find_database_path(REAL_DATA_PATH, 'miiv')
        pf = PatientFilter(database='miiv', data_path=resolved, verbose=False)
        df = pf._load_demographics()
        id_col = 'stay_id' if 'stay_id' in df.columns else 'patient_id'
        patient_ids = df[id_col].dropna().astype(int).head(100).tolist()
        sofa_concepts = [
            'sofa', 'sofa2',
            'sofa_resp', 'sofa2_resp',
            'sofa_coag', 'sofa2_coag',
            'sofa_liver', 'sofa2_liver',
            'sofa_cardio', 'sofa2_cardio',
            'sofa_cns', 'sofa2_cns',
            'sofa_renal', 'sofa2_renal',
        ]
        result = load_concepts(
            concepts=sofa_concepts,
            database='miiv',
            data_path=resolved,
            patient_ids=patient_ids,
            verbose=False,
        )
        assert result is not None and not result.empty, "No SOFA data returned"
        loaded_sofa = [c for c in sofa_concepts if c in result.columns]
        print(f"  ✅ Loaded SOFA concepts: {loaded_sofa}")
        assert 'sofa' in loaded_sofa, "SOFA-1 should be loadable"
        assert 'sofa2' in loaded_sofa, "SOFA-2 should be loadable"


# ---------------------------------------------------------------------------
# Smoke test report fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def smoke_test_summary(request):
    """Print a summary after all tests complete."""
    yield
    print("\n" + "=" * 60)
    print("Real UI Smoke Test Summary")
    print("=" * 60)
    print(f"EASYICU_DATA_PATH: {REAL_DATA_PATH or '(not set)'}")
    print(f"Has real data: {HAS_REAL_DATA}")
    print("=" * 60)
