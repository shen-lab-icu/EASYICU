"""Implementation-identity contracts for typed ResearchContext metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent import pipeline_cache
from easyicu.research_agent import run_input_capsule
from easyicu.research_agent.research_context.implementation_identity import (
    metadata_implementation_identity,
)
from tests.research_agent.test_materialized_column_metadata import _build_v2_context

_METADATA_IDENTITY_KEYS = (
    "metadata_projection_sha256",
    "metadata_sidecar_sha256",
    "icu_rules_sha256",
    "metadata_implementation_bundle_sha256",
)


def _selected_metadata_identity(identity: dict[str, object]) -> dict[str, object]:
    return {key: identity[key] for key in _METADATA_IDENTITY_KEYS}


def test_environment_and_cache_bind_the_same_metadata_implementation() -> None:
    expected = dict(metadata_implementation_identity())

    environment = run_input_capsule.build_environment_identity(
        llm_signature="identity-contract-test"
    )
    cache_identity = pipeline_cache.runtime_identity()

    assert _selected_metadata_identity(environment) == expected
    assert _selected_metadata_identity(cache_identity) == expected


def test_metadata_identity_hashes_sidecar_schema_bytes() -> None:
    import easyicu.concept.metadata_sidecar as metadata_sidecar

    sidecar_path = Path(metadata_sidecar.__file__).resolve()
    expected_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    identity = dict(metadata_implementation_identity())

    assert identity["metadata_sidecar_sha256"] == expected_sha256
    implementation_payload = {
        key: identity[key]
        for key in _METADATA_IDENTITY_KEYS
        if key != "metadata_implementation_bundle_sha256"
    }
    expected_bundle_sha256 = hashlib.sha256(
        json.dumps(
            implementation_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert identity["metadata_implementation_bundle_sha256"] == expected_bundle_sha256


def test_v2_context_binds_complete_metadata_implementation(tmp_path: Path) -> None:
    context = _build_v2_context(tmp_path)

    assert _selected_metadata_identity(
        context.materialized_inputs.cohort.model_dump(mode="python")
    ) == dict(metadata_implementation_identity())


def test_scientific_identity_excludes_metadata_implementation() -> None:
    scientific_identity = run_input_capsule.build_scientific_identity(
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2],
                "exposure": [0.0, 1.0],
                "outcome": [0, 1],
            }
        ),
        question="Is the declared exposure associated with the outcome?",
        cohort_name="identity_contract",
        database="generic_icu",
        target_outcome="outcome",
        primary_exposure="exposure",
        cross_database_validation=None,
        inclusion_criteria=None,
        exclusion_criteria=None,
        id_columns=("stay_id",),
        time_columns=None,
        outcome_columns=("outcome",),
        time_windows=None,
        concept_descriptions=None,
        user_preferences=None,
        notes=None,
        skill_key=None,
        experiment_spec=None,
        source_files=None,
        disable_icu_context=False,
    )

    assert not set(_METADATA_IDENTITY_KEYS).intersection(scientific_identity)


def test_metadata_sidecar_implementation_change_invalidates_runtime_identities(
    monkeypatch,
) -> None:
    run_input_capsule.engine_code_sha256.cache_clear()
    baseline_engine_sha = run_input_capsule.engine_code_sha256()
    baseline_environment = run_input_capsule.build_environment_identity(
        llm_signature="identity-contract-test"
    )
    baseline_cache_identity = pipeline_cache.runtime_identity()
    changed_identity = {
        **dict(metadata_implementation_identity()),
        "metadata_sidecar_sha256": "2" * 64,
    }

    with monkeypatch.context() as patch:
        patch.setattr(
            run_input_capsule,
            "metadata_implementation_identity",
            lambda: changed_identity,
        )
        patch.setattr(
            pipeline_cache,
            "metadata_implementation_identity",
            lambda: changed_identity,
        )
        run_input_capsule.engine_code_sha256.cache_clear()

        changed_engine_sha = run_input_capsule.engine_code_sha256()
        changed_environment = run_input_capsule.build_environment_identity(
            llm_signature="identity-contract-test"
        )
        changed_cache_identity = pipeline_cache.runtime_identity()

    run_input_capsule.engine_code_sha256.cache_clear()

    assert changed_engine_sha != baseline_engine_sha
    assert changed_environment != baseline_environment
    assert changed_cache_identity != baseline_cache_identity
    assert _selected_metadata_identity(changed_environment) == changed_identity
    assert _selected_metadata_identity(changed_cache_identity) == changed_identity
