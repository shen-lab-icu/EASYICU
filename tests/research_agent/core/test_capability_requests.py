from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.resources import (
    CapabilityActivation,
    CapabilityApproval,
    ResourceCatalog,
    ResourceScheduler,
    ResourceSelectionPolicy,
    ResourceSelectionQuery,
    approved_capability_resource,
    build_capability_activation,
    build_capability_request,
    write_capability_request,
    verify_capability_activation,
)


def _request():
    return build_capability_request(
        method_name="competing-risks regression",
        package_name="scikit-survival",
        import_name="sksurv",
        version_spec=">=0.25,<0.26",
        purpose="Fit a declared Fine-Gray sensitivity model",
        analysis_families=("survival",),
        required_input_roles=("time_to_event", "event_type"),
        produced_output_roles=("effect_estimate",),
        license_spdx="GPL-3.0-only",
        upstream_source="https://github.com/sebp/scikit-survival",
        validation_test_refs=("tests/methods/test_fine_gray.py",),
        requested_by="planner:run-001",
        requested_at="2026-07-22T03:00:00Z",
        runtime_import_names=("pandas", "numpy", "statsmodels"),
    )


def _approval(request, *, decision="approved"):
    kwargs = {}
    if decision == "approved":
        kwargs = {
            "installed_version": "0.25.0",
            "image_reference": "easyicu/research-agent@sha256:abc",
            "image_digest": "sha256:" + "a" * 64,
            "validation_receipt_sha256": "b" * 64,
        }
    return CapabilityApproval(
        request_id=request.request_id,
        request_sha256=request.sha256,
        decision=decision,
        reviewer="maintainer",
        reviewed_at="2026-07-22T04:00:00Z",
        **kwargs,
    )


def test_unavailable_package_yields_non_executable_digest_bound_request() -> None:
    request = _request()
    assert request.runtime_install_allowed is False
    assert request.request_id.startswith("cap-")
    assert len(request.sha256) == 64

    tampered = request.model_dump(mode="python")
    tampered["purpose"] = "different purpose"
    with pytest.raises(ValidationError, match="does not bind"):
        type(request).model_validate(tampered)


def test_installed_package_does_not_generate_an_install_request() -> None:
    with pytest.raises(ValueError, match="must be registered"):
        build_capability_request(
            method_name="Cox model",
            package_name="lifelines",
            import_name="lifelines",
            version_spec=">=0.30,<0.31",
            purpose="Fit a Cox model",
            analysis_families=("survival",),
            license_spdx="MIT",
            upstream_source="https://github.com/CamDavidsonPilon/lifelines",
            validation_test_refs=("tests/test_survival.py",),
            requested_by="planner:run",
            requested_at="2026-07-22T03:00:00Z",
            runtime_import_names=("lifelines",),
        )


def test_request_file_is_write_once(tmp_path: Path) -> None:
    request = _request()
    path = tmp_path / "requests" / f"{request.request_id}.json"
    write_capability_request(path, request)
    write_capability_request(path, request)
    other = request.model_copy(
        update={"requested_at": "2026-07-22T05:00:00Z"}
    ).model_dump_json(indent=2)
    path.write_text(other, encoding="utf-8")
    with pytest.raises(FileExistsError, match="other bytes"):
        write_capability_request(path, request)


def test_approval_requires_image_version_and_test_receipt() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="requires version"):
        CapabilityApproval(
            request_id=request.request_id,
            request_sha256=request.sha256,
            decision="approved",
            reviewer="maintainer",
            reviewed_at="2026-07-22T04:00:00Z",
        )


def test_only_digest_bound_approval_enters_software_catalog() -> None:
    request = _request()
    approval = _approval(request)
    resource = approved_capability_resource(request, approval)
    assert resource.kind == "software"
    assert resource.permissions == ("coder_context", "sandbox_import")
    assert '"runtime_install_allowed":false' in resource.prompt_projection

    wrong = approval.model_copy(update={"request_sha256": "0" * 64})
    with pytest.raises(ValueError, match="does not bind"):
        approved_capability_resource(request, wrong)
    with pytest.raises(ValueError, match="rejected"):
        approved_capability_resource(request, _approval(request, decision="rejected"))


def test_approved_software_is_selected_inside_host_allowlist() -> None:
    request = _request()
    resource = approved_capability_resource(request, _approval(request))
    catalog = ResourceCatalog((resource,))
    query = ResourceSelectionQuery(
        purpose="coder",
        query="Fit a competing-risks survival regression",
        analysis_family="survival",
        available_input_roles=("time_to_event", "event_type"),
    )
    policy = ResourceSelectionPolicy(
        allowed_kinds=("software",),
        allowed_review_statuses=("approved",),
        allowed_permissions=("coder_context", "sandbox_import"),
        max_software=1,
    )
    selection = ResourceScheduler.select_resources(
        catalog=catalog,
        query=query,
        policy=policy,
        kind="software",
    )
    assert selection.resources == (resource,)
    assert selection.receipt.provider_calls == 0
    assert selection.receipt.selected[0].resource_id == resource.resource_id


def test_missing_typed_inputs_prevent_software_selection() -> None:
    request = _request()
    resource = approved_capability_resource(request, _approval(request))
    catalog = ResourceCatalog((resource,))
    query = ResourceSelectionQuery(
        purpose="coder",
        query="Fit competing risks",
        analysis_family="survival",
        available_input_roles=("time_to_event",),
    )
    policy = ResourceSelectionPolicy(
        allowed_kinds=("software",),
        allowed_review_statuses=("approved",),
        allowed_permissions=("coder_context", "sandbox_import"),
    )
    selection = ResourceScheduler.select_resources(
        catalog=catalog,
        query=query,
        policy=policy,
        kind="software",
    )
    assert selection.resources == ()


def test_activation_requires_new_profile_new_run_and_exact_image() -> None:
    request = _request()
    approval = _approval(request)
    activation = build_capability_activation(
        request=request,
        approval=approval,
        source_profile_ref="source/1",
        target_profile_ref="target/2",
    )
    resource = verify_capability_activation(
        request=request,
        approval=approval,
        activation=activation,
        current_profile_ref="target/2",
        expected_image_digest="sha256:" + "a" * 64,
        actual_image_digest="sha256:" + "a" * 64,
        runtime_import_names=("numpy", "sksurv"),
        is_resume=False,
    )
    assert resource.resource_id == "software:sksurv"
    assert activation.runtime_install_allowed is False
    assert activation.new_run_required is True

    with pytest.raises(ValueError, match="new run"):
        verify_capability_activation(
            request=request,
            approval=approval,
            activation=activation,
            current_profile_ref="target/2",
            expected_image_digest="sha256:" + "a" * 64,
            actual_image_digest="sha256:" + "a" * 64,
            runtime_import_names=("sksurv",),
            is_resume=True,
        )
    with pytest.raises(ValueError, match="runtime image"):
        verify_capability_activation(
            request=request,
            approval=approval,
            activation=activation,
            current_profile_ref="target/2",
            expected_image_digest="sha256:" + "a" * 64,
            actual_image_digest="sha256:" + "c" * 64,
            runtime_import_names=("sksurv",),
            is_resume=False,
        )


def test_activation_id_rejects_tampering() -> None:
    request = _request()
    activation = build_capability_activation(
        request=request,
        approval=_approval(request),
        source_profile_ref="source/1",
        target_profile_ref="target/2",
    )
    payload = activation.model_dump(mode="python")
    payload["target_profile_ref"] = "other/3"
    with pytest.raises(ValidationError, match="does not bind"):
        CapabilityActivation.model_validate(payload)
