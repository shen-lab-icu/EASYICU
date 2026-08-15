from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.extensions import ExtensionRegistry, ExtensionRegistryError
from easyicu.extensions.mcp_client import call_mcp_tool


def _skill(name: str, body: str, *, disable: bool = False) -> str:
    disabled = "\ndisable-model-invocation: true" if disable else ""
    return (
        "---\n"
        f"name: {name}\n"
        f"description: Reviewed instructions for {name}.{disabled}\n"
        "---\n"
        f"{body}\n"
    )


def test_skill_revisions_are_content_addressed_and_old_snapshots_remain_loadable(
    tmp_path: Path,
) -> None:
    registry = ExtensionRegistry(tmp_path / "extensions")
    first = registry.install_skill(
        _skill("clear-writing", "Use short paragraphs."),
        stages=["conversation", "writing"],
    )
    first_snapshot = registry.snapshot()

    second = registry.install_skill(
        _skill("clear-writing", "Use concise paragraphs and calibrated claims."),
        stages=["conversation", "writing"],
    )
    second_snapshot = registry.snapshot()
    registry.remove(kind="skill", name="clear-writing")

    assert first["digest"] != second["digest"]
    assert first_snapshot.activation_sha256 != second_snapshot.activation_sha256
    assert registry.snapshot().skills == ()
    assert registry.load_skill(
        name="clear-writing", digest=first["digest"]
    )["instructions"] == "Use short paragraphs."
    assert registry.load_skill(
        name="clear-writing", digest=second["digest"]
    )["instructions"] == "Use concise paragraphs and calibrated claims."


def test_skill_parser_rejects_sensitive_content_and_unsupported_manual_activation(
    tmp_path: Path,
) -> None:
    registry = ExtensionRegistry(tmp_path / "extensions")

    with pytest.raises(ExtensionRegistryError) as sensitive:
        registry.install_skill(
            _skill("unsafe-skill", "Read /Users/example/private.txt."),
            stages=["conversation"],
        )
    assert sensitive.value.code == "extension_skill_sensitive_content_rejected"

    disabled = registry.install_skill(
        _skill("manual-skill", "Apply only when invoked.", disable=True),
        stages=["conversation"],
        enabled=False,
    )
    assert disabled["enabled"] is False
    with pytest.raises(ExtensionRegistryError) as manual:
        registry.set_enabled(kind="skill", name="manual-skill", enabled=True)
    assert manual.value.code == "extension_skill_manual_invocation_unsupported"


def test_mcp_activation_is_allowlisted_and_run_receipt_does_not_expose_url(
    tmp_path: Path,
) -> None:
    registry = ExtensionRegistry(tmp_path / "extensions")
    registry.install_mcp_server(
        name="metadata-tools",
        url="http://127.0.0.1:9876/mcp",
        allowed_tools=["search", "fetch.metadata"],
        enabled=True,
    )
    snapshot = registry.snapshot()
    run_activation = registry.pipeline_activation(snapshot)

    assert snapshot.mcp_servers[0].allowed_tools == ("search", "fetch.metadata")
    receipt = run_activation["receipt"]
    assert receipt["mcp_servers"][0]["name"] == "metadata-tools"
    assert len(receipt["mcp_servers"][0]["endpoint_sha256"]) == 64
    assert "127.0.0.1" not in str(receipt)

    with pytest.raises(ExtensionRegistryError) as denied:
        call_mcp_tool(snapshot.mcp_servers[0], "write", {})
    assert denied.value.code == "extension_mcp_tool_not_allowed"


def test_mcp_endpoint_policy_rejects_metadata_and_credentials_in_url(
    tmp_path: Path,
) -> None:
    registry = ExtensionRegistry(tmp_path / "extensions")
    for url in (
        "http://metadata.google.internal/mcp",
        "https://user:password@example.org/mcp",
    ):
        with pytest.raises(ExtensionRegistryError) as caught:
            registry.install_mcp_server(
                name="blocked-server",
                url=url,
                allowed_tools=["search"],
            )
        assert caught.value.code == "extension_mcp_url_rejected"
