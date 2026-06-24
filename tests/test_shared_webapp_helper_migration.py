from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_non_streamlit_production_code_uses_shared_helpers() -> None:
    forbidden = (
        "easyicu.webapp.ai_optin",
        "easyicu.webapp.concept_catalog",
        "easyicu.webapp.data_paths",
        ".webapp.data_paths",
    )
    production_files = [
        "src/easyicu/webserver/provider_gate.py",
        "src/easyicu/webserver/catalog.py",
        "src/easyicu/webserver/dataio.py",
        "src/easyicu/api.py",
        "src/easyicu/cohort_visualization.py",
        "src/easyicu/comorbidity.py",
        "src/easyicu/outcomes.py",
        "scripts/full_export_modules.py",
    ]
    for rel_path in production_files:
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"{rel_path} still imports {marker}"


def test_concept_catalog_legacy_shim_points_to_shared_module() -> None:
    import easyicu.concept_catalog as shared
    import easyicu.webapp.concept_catalog as legacy

    assert legacy.CONCEPT_DICTIONARY is shared.CONCEPT_DICTIONARY
    assert legacy.CONCEPT_GROUPS_INTERNAL is shared.CONCEPT_GROUPS_INTERNAL
    assert legacy.CONCEPT_GROUP_NAMES is shared.CONCEPT_GROUP_NAMES
    assert legacy._get_patient_id_table_files is shared._get_patient_id_table_files


def test_shared_ai_optin_provider_gate_fails_closed() -> None:
    from easyicu.webserver.provider_gate import (
        CANONICAL_OPT_IN_SOURCE,
        ProviderGateError,
        resolve_provider_gate,
    )

    assert CANONICAL_OPT_IN_SOURCE == "easyicu.ai_optin.check_external_llm_opt_in"
    offline = resolve_provider_gate(
        run_type="full",
        llm_provider="mock",
        external_llm_opt_in=False,
        ai_enabled=False,
    )
    assert offline["client"] == "MockLLMClient"
    assert offline["client_constructed"] is False

    with pytest.raises(ProviderGateError) as excinfo:
        resolve_provider_gate(
            run_type="full",
            llm_provider="openai",
            external_llm_opt_in=True,
            ai_enabled=False,
            language="en",
        )
    detail = excinfo.value.detail
    assert detail["blocked_by"] == "canonical_ai_opt_in"
    assert detail["client_constructed"] is False
    assert detail["credentials_loaded"] is False


def test_shared_find_database_path_resolves_aliases(tmp_path: Path) -> None:
    from easyicu.data_paths import find_database_path

    versioned = tmp_path / "mimiciv" / "3.1"
    versioned.mkdir(parents=True)
    assert find_database_path(str(tmp_path), "miiv") == str(versioned)

    direct_flat = tmp_path / "eicu-crd"
    direct_flat.mkdir()
    (direct_flat / "patient.parquet").write_bytes(b"")
    assert find_database_path(str(direct_flat), "eicu") == str(direct_flat)

    fuzzy = tmp_path / "local_sicdb_snapshot" / "1.0.6"
    fuzzy.mkdir(parents=True)
    assert find_database_path(str(tmp_path), "sic") == str(fuzzy)
