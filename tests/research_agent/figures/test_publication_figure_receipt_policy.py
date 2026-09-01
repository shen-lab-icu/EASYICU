from __future__ import annotations

import inspect
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("repair_id", "host_sealed_renderer", "expected"),
    [
        (None, False, False),
        (None, True, True),
        ("registry_renderer_v1", False, True),
        ("registry_renderer_v1", True, True),
    ],
)
def test_code_seal_policy_distinguishes_host_and_registry_renderers(
    repair_id: str | None,
    host_sealed_renderer: bool,
    expected: bool,
) -> None:
    from easyicu.research_agent.execution.publication_figure import (
        SealedRendererState,
        sealed_renderer_code_seal_required,
    )

    state = SealedRendererState()
    state.repair_id = repair_id

    assert (
        sealed_renderer_code_seal_required(
            state=state,
            host_sealed_renderer=host_sealed_renderer,
        )
        is expected
    )


def _registry_state():
    from easyicu.research_agent.execution.publication_figure import (
        SealedRendererState,
    )

    state = SealedRendererState()
    state.repair_id = "registry_renderer_v1"
    state.implementation_sha256 = "implementation-sha"
    state.parent_digests = {"parent.csv": "parent-sha"}
    state.authorized_product_slots = {"figure:result": "result.png"}
    return state


def _matching_summary() -> dict[str, object]:
    return {
        "sealed_renderer_repair": "registry_renderer_v1",
        "sealed_renderer_implementation_sha256": "implementation-sha",
        "sealed_renderer_parent_digests": {"parent.csv": "parent-sha"},
        "planner_product_slot_bindings": {
            "figure:result": {"slot": "result.png"}
        },
    }


def test_ordinary_host_renderer_does_not_inherit_legacy_receipt_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import publication_figure

    def unexpected_parent_read(**_kwargs: object) -> None:
        raise AssertionError("ordinary host renderer must not read a legacy receipt")

    monkeypatch.setattr(
        publication_figure,
        "read_digest_bound_artifact_snapshot",
        unexpected_parent_read,
    )
    step_record: dict[str, object] = {}

    findings = publication_figure.validate_and_record_sealed_renderer_receipt(
        state=publication_figure.SealedRendererState(),
        authorized_code_sha256="host-code-sha",
        visual_step_summary={},
        run_dir=tmp_path,
        step_id="02_figure",
        step_record=step_record,
    )

    assert findings == ()
    assert step_record == {}


def test_registry_renderer_verifies_parent_and_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import publication_figure

    parent_reads: list[dict[str, object]] = []
    monkeypatch.setattr(
        publication_figure,
        "read_digest_bound_artifact_snapshot",
        lambda **kwargs: parent_reads.append(kwargs),
    )
    step_record: dict[str, object] = {}

    findings = publication_figure.validate_and_record_sealed_renderer_receipt(
        state=_registry_state(),
        authorized_code_sha256="authorized-code-sha",
        visual_step_summary=_matching_summary(),
        run_dir=tmp_path,
        step_id="02_figure",
        step_record=step_record,
    )

    assert findings == ()
    assert step_record["sealed_renderer_parent_receipt_verified"] is True
    assert parent_reads == [
        {
            "parent_out": tmp_path / "steps" / "02" / "outputs",
            "artifact_digests": {"parent.csv": "parent-sha"},
        }
    ]


def test_registry_renderer_parent_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import publication_figure

    def reject_parent(**_kwargs: object) -> None:
        raise ValueError("parent changed")

    monkeypatch.setattr(
        publication_figure,
        "read_digest_bound_artifact_snapshot",
        reject_parent,
    )
    step_record: dict[str, object] = {}

    findings = publication_figure.validate_and_record_sealed_renderer_receipt(
        state=_registry_state(),
        authorized_code_sha256="authorized-code-sha",
        visual_step_summary=_matching_summary(),
        run_dir=tmp_path,
        step_id="02_figure",
        step_record=step_record,
    )

    assert step_record["sealed_renderer_parent_receipt_verified"] is False
    assert [finding.validator for finding in findings] == [
        "sealed_renderer_authority"
    ]
    assert "direct-parent inputs changed" in findings[0].message


def test_registry_renderer_identity_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import publication_figure

    monkeypatch.setattr(
        publication_figure,
        "read_digest_bound_artifact_snapshot",
        lambda **_kwargs: None,
    )
    summary = _matching_summary()
    summary["sealed_renderer_implementation_sha256"] = "changed-sha"

    findings = publication_figure.validate_and_record_sealed_renderer_receipt(
        state=_registry_state(),
        authorized_code_sha256="authorized-code-sha",
        visual_step_summary=summary,
        run_dir=tmp_path,
        step_id="02_figure",
        step_record={},
    )

    assert [finding.validator for finding in findings] == [
        "sealed_renderer_authority"
    ]
    assert "exact sealed renderer identity" in findings[0].message
    assert findings[0].detail is not None
    assert findings[0].detail["reported_implementation_sha256"] == "changed-sha"


def test_candidate_loop_delegates_legacy_receipt_policy_to_figure_owner() -> None:
    from easyicu.research_agent.execution import candidate_loop, publication_figure

    assert (
        candidate_loop.validate_and_record_sealed_renderer_receipt
        is publication_figure.validate_and_record_sealed_renderer_receipt
    )
    source = inspect.getsource(candidate_loop._candidate_contract_setup_transition)
    assert "validate_and_record_sealed_renderer_receipt(" in source
    assert "legacy_sealed_renderer_receipt" not in source
