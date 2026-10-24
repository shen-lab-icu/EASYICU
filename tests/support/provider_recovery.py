"""Shared synthetic fixtures; no recorded research inputs."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from easyicu.webserver import dataio
from easyicu.webserver.agent_review_recovery import WebReviewRecoveryRecord

def _limits(**overrides):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLimits,
    )

    values = {
        "max_provider_attempts_per_run": 3,
        "max_provider_attempts_per_batch": 6,
        "max_total_tokens_per_run": 1_000_000,
        "max_total_tokens_per_batch": 2_000_000,
        "max_estimated_cost_usd_per_batch": 10.0,
        "max_wall_clock_seconds_per_task": 60.0,
        "input_cost_usd_per_million_tokens": 1.0,
        "output_cost_usd_per_million_tokens": 2.0,
    }
    values.update(overrides)
    return ProviderHardStopLimits(**values)


def _prepared_package(root: Path) -> tuple[dict, dict]:
    export = root / "export"
    export.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        export / "demographics.parquet", index=False
    )
    (export / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt = dataio.validate_research_pipeline_source(
        str(export),
        database="miiv",
    )
    study = {
        "id": "study-a",
        "data_source": {"path": str(export), "database": "miiv"},
    }
    return study, receipt["binding"]


def _recoverable_package_record(tmp_path: Path) -> tuple[WebReviewRecoveryRecord, Path]:
    from easyicu.research_agent.orchestration.config import PipelineConfig

    wrapper = tmp_path / "wrapper"
    config = PipelineConfig(workdir=wrapper / "pipeline")
    study, package_binding = _prepared_package(tmp_path / "package-recovery")
    record = WebReviewRecoveryRecord.create(
        run_id="run-recovery-package",
        wrapper_dir=str(wrapper.resolve()),
        study=study,
        scientific_configuration_sha256="a" * 64,
        provider_meta={"provider": "openai", "external": True},
        provider_public={
            "provider": "openai",
            "model": "model-a",
            "credential_fingerprint": "c" * 16,
            "endpoint_fingerprint": "e" * 16,
            "credential_header": "authorization",
            "base_url_endpoint": "chat_completions",
        },
        credential_source="pi_verified",
        budget_mode="planner_canary",
        prepared_package_binding=package_binding,
        pipeline_config=config.recovery_payload(),
        pipeline_config_sha256=config.canonical_digest(),
        acquisition_projection={},
        hard_stop_ledger_path=str(wrapper / ".runtime" / "ledger.json"),
        hard_stop_task_id="web-job-recovery",
        hard_stop_declaration_sha256="b" * 64,
        created_at=1.0,
    )
    return record, Path(study["data_source"]["path"])
