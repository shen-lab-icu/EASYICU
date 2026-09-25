"""A failure lineage seeds a new plan from a Planner checkpoint at most once.

A continuation carries its source prefix forward. When a planning gate
rejects it, seeding again repeats that prefix and that rejection on every
"generate a fresh plan" click; the next attempt plans anew instead. A budget
or Provider stop keeps its explicit resume route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver import agent_review_recovery as recovery
from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import contracts, run_authority

_STUDY = {
    "id": "study-lineage",
    "revision": 1,
    "question": "Does an ICU measurement relate to in-hospital death?",
    "database": "miiv",
}
_DIGEST = study_contexts.scientific_configuration_sha256(_STUDY)


def _failed_prefix(root: Path, reason: str, *, continued: bool | None) -> dict:
    """One failed run with checkpoints; ``continued=None`` writes no seed."""

    wrapper = root / _STUDY["id"] / "run_prefix"
    wrapper.mkdir(parents=True)
    if continued is not None:
        seed = recovery.WebReviewRecoverySeed.create(
            wrapper_dir=str(wrapper.resolve()),
            study=_STUDY,
            scientific_configuration_sha256=_DIGEST,
            provider_meta={},
            provider_public={},
            credential_source="pi_verified",
            budget_mode="planner_canary",
            prepared_package_binding=None,
            pipeline_config={
                "development_progressive_resume_checkpoint_path": (
                    "/prior/run/progressive_planner_checkpoint_004.json"
                    if continued
                    else None
                )
            },
            pipeline_config_sha256="c" * 64,
            acquisition_projection={},
            hard_stop_ledger_path="",
            hard_stop_task_id="web-prior",
            hard_stop_declaration_sha256="d" * 64,
            created_at=1.0,
        )
        path = recovery.recovery_seed_path(wrapper)
        path.parent.mkdir()
        path.write_text(seed.model_dump_json())
    return {
        "run_id": "run_prefix",
        "study_id": _STUDY["id"],
        "run_status": "failed",
        "gate_reason": reason,
        "scientific_configuration_sha256": _DIGEST,
        "project_dir": str(wrapper),
        "development_planner_checkpoint_available": True,
    }


def _seed_source(root: Path, row: dict) -> str:
    return run_authority.resumable_planner_checkpoint_job_id(
        study=_STUDY,
        rows=[row],
        project_root=root,
    )


@pytest.mark.parametrize(
    "reason",
    [
        "research_pipeline_progressive_compile_failed",
        "research_pipeline_plan_contract_exhausted",
    ],
)
def test_a_rejected_continuation_does_not_seed_again(tmp_path, reason) -> None:
    row = _failed_prefix(tmp_path, reason, continued=True)

    assert _seed_source(tmp_path, row) == ""


@pytest.mark.parametrize("continued", [False, None])
def test_a_first_rejection_still_seeds_one_continuation(tmp_path, continued) -> None:
    row = _failed_prefix(
        tmp_path,
        "research_pipeline_progressive_compile_failed",
        continued=continued,
    )

    assert _seed_source(tmp_path, row) == "prefix"


def test_an_unreadable_seed_is_left_to_the_launch_scope_owner(tmp_path) -> None:
    row = _failed_prefix(
        tmp_path,
        "research_pipeline_progressive_compile_failed",
        continued=True,
    )
    recovery.recovery_seed_path(Path(row["project_dir"])).write_text("{}")

    # Selection does not guess; using the checkpoint then fails its scope check.
    assert _seed_source(tmp_path, row) == "prefix"


@pytest.mark.parametrize("reason", sorted(contracts.PLAN_RESUME_OFFER_GATE_REASONS))
def test_a_budget_or_provider_stop_keeps_its_resume_route(tmp_path, reason) -> None:
    row = _failed_prefix(tmp_path, reason, continued=True)

    assert _seed_source(tmp_path, row) == "prefix"
