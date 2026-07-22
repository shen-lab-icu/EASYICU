"""Architecture contract for direct-parent artifact authority extraction."""

from __future__ import annotations

import ast
import inspect

from easyicu.research_agent import pipeline
from easyicu.research_agent.authority import parent_artifact
from easyicu.research_agent.figures import distribution_availability

PARENT_ARTIFACT_EXPORTS = (
    "_resolve_upstream_manifest_analysis_request",
    "_resolve_upstream_manifest_step",
    "_verified_direct_parent_artifact_digests",
    "_verified_direct_parent_table_names",
)


def _imported_module_leaves(module: object) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    leaves: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            leaves.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            leaves.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    return leaves


def test_pipeline_parent_artifact_exports_keep_object_identity() -> None:
    for name in PARENT_ARTIFACT_EXPORTS:
        assert getattr(pipeline, name) is getattr(parent_artifact, name)


def test_sealed_registry_distribution_seal_keeps_renderer_identity() -> None:
    """The registry adapter must dispatch to the exact figures seal object.

    The historical ``pipeline._distribution_availability_parent_digest_seal``
    re-export was retired when the distribution renderer moved into
    ``figures/sealed_registry.py`` (2026-07-22); the registry entry is now the
    single dispatch surface and must not wrap or fork the canonical seal.
    """

    from easyicu.research_agent.figures.sealed_registry import (
        sealed_renderer_adapter,
    )

    adapter = sealed_renderer_adapter(
        "distribution_availability_publication_bundle_from_parent_outputs_v1"
    )
    assert adapter is not None
    assert (
        adapter.seal
        is distribution_availability._distribution_availability_parent_digest_seal
    )
    assert not hasattr(pipeline, "_distribution_availability_parent_digest_seal")


def test_parent_artifact_authority_never_imports_pipeline_or_renderer() -> None:
    imported = _imported_module_leaves(parent_artifact)
    assert "pipeline" not in imported
    assert "distribution_availability" not in imported


def test_distribution_renderer_no_longer_imports_pipeline() -> None:
    assert "pipeline" not in _imported_module_leaves(distribution_availability)
