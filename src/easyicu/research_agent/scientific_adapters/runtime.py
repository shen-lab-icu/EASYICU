"""Approval-facing runtime contracts for optional scientific adapters.

Adapters are intentionally outside the normal Coder allow-list.  A package
being importable on one developer machine is not permission to offer it to an
agent or promote a scientific capability.  The existing capability-request /
approval / immutable-image activation flow remains the only route into an
execution image; this module makes the three first adapter candidates express
that flow consistently.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..resources.capability import CapabilityRequest, build_capability_request


@dataclass(frozen=True)
class ExternalAdapterSpec:
    """One dependency-neutral adapter boundary and its approval metadata."""

    adapter_id: str
    method_name: str
    package_name: str
    import_name: str
    version_spec: str
    purpose: str
    analysis_families: Tuple[str, ...]
    required_input_roles: Tuple[str, ...]
    produced_output_roles: Tuple[str, ...]
    license_spdx: str
    upstream_source: str
    validation_test_refs: Tuple[str, ...]


@dataclass(frozen=True)
class ExternalAdapterRuntime:
    """Truthful local observation, not approval or publication evidence."""

    adapter_id: str
    package_name: str
    import_name: str
    expected_version_spec: str
    status: Literal[
        "available",
        "unavailable",
        "incompatible_version",
        "distribution_unresolved",
    ]
    installed_version: Optional[str]
    issue_code: Optional[str]

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.external_adapter_runtime/2",
            "adapter_id": self.adapter_id,
            "package_name": self.package_name,
            "import_name": self.import_name,
            "expected_version_spec": self.expected_version_spec,
            "status": self.status,
            "installed_version": self.installed_version,
            "issue_code": self.issue_code,
        }


EXTERNAL_ADAPTER_SPECS: Tuple[ExternalAdapterSpec, ...] = (
    ExternalAdapterSpec(
        adapter_id="pandera_dataframe_contract_v1",
        method_name="Pandas dataframe contract validation",
        package_name="pandera",
        import_name="pandera.pandas",
        version_spec=">=0.28,<0.33",
        purpose=(
            "Validate generic dataframe columns, nullability, dtypes, and "
            "value domains without replacing EasyICU clinical semantics."
        ),
        analysis_families=("intake", "data_contract"),
        required_input_roles=("dataframe", "declared_schema"),
        produced_output_roles=("schema_validation_receipt",),
        license_spdx="MIT",
        upstream_source="https://pandera.readthedocs.io/en/stable/",
        validation_test_refs=(
            "tests/research_agent/core/test_scientific_adapters.py::test_pandera_adapter_builds_a_strict_non_coercing_schema",
        ),
    ),
    ExternalAdapterSpec(
        adapter_id="dowhy_identification_v1",
        method_name="Causal graph identification",
        package_name="dowhy",
        import_name="dowhy",
        version_spec=">=0.13,<0.14",
        purpose=(
            "Check identification for a declared causal graph without choosing "
            "a treatment, outcome, time zero, covariate set, or estimator."
        ),
        analysis_families=("causal_emulation",),
        required_input_roles=(
            "analysis_dataframe",
            "declared_causal_graph",
            "treatment",
            "outcome",
        ),
        produced_output_roles=("causal_identification_receipt",),
        license_spdx="MIT",
        upstream_source="https://www.pywhy.org/dowhy/v0.13/",
        validation_test_refs=(
            "tests/research_agent/core/test_scientific_adapters.py::test_dowhy_adapter_only_records_identification",
        ),
    ),
    ExternalAdapterSpec(
        adapter_id="sksurv_competing_risks_cif_v1",
        method_name="Competing-risks cumulative incidence",
        package_name="scikit-survival",
        import_name="sksurv",
        version_spec=">=0.28,<0.29",
        purpose=(
            "Estimate a declared competing-risks cumulative-incidence curve; "
            "never substitute a cause-naive Cox model."
        ),
        analysis_families=("time_to_event",),
        required_input_roles=("time_to_event", "event_type"),
        produced_output_roles=("cumulative_incidence_curve",),
        license_spdx="GPL-3.0-only",
        upstream_source="https://scikit-survival.readthedocs.io/en/stable/",
        validation_test_refs=(
            "tests/research_agent/core/test_scientific_adapters.py::test_sksurv_adapter_preserves_declared_event_codes",
        ),
    ),
)


def get_external_adapter_spec(adapter_id: str) -> ExternalAdapterSpec:
    """Return one declared adapter, rejecting unknown IDs fail-closed."""

    matches = [spec for spec in EXTERNAL_ADAPTER_SPECS if spec.adapter_id == adapter_id]
    if len(matches) != 1:
        raise ValueError(f"unknown external adapter: {adapter_id!r}")
    return matches[0]


def probe_external_adapter(adapter_id: str) -> ExternalAdapterRuntime:
    """Observe one optional dependency without importing or activating it."""

    spec = get_external_adapter_spec(adapter_id)
    try:
        present = importlib.util.find_spec(spec.import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        present = False
    if not present:
        return ExternalAdapterRuntime(
            adapter_id=spec.adapter_id,
            package_name=spec.package_name,
            import_name=spec.import_name,
            expected_version_spec=spec.version_spec,
            status="unavailable",
            installed_version=None,
            issue_code="external_adapter_dependency_unavailable",
        )
    try:
        version = importlib.metadata.version(spec.package_name)
    except importlib.metadata.PackageNotFoundError:
        # A namespace/import alias without distribution metadata cannot be bound
        # reproducibly into an approval or image receipt.
        return ExternalAdapterRuntime(
            adapter_id=spec.adapter_id,
            package_name=spec.package_name,
            import_name=spec.import_name,
            expected_version_spec=spec.version_spec,
            status="distribution_unresolved",
            installed_version=None,
            issue_code="external_adapter_distribution_unresolved",
        )
    try:
        compatible = Version(version) in SpecifierSet(spec.version_spec)
    except InvalidSpecifier:
        return ExternalAdapterRuntime(
            adapter_id=spec.adapter_id,
            package_name=spec.package_name,
            import_name=spec.import_name,
            expected_version_spec=spec.version_spec,
            status="incompatible_version",
            installed_version=version,
            issue_code="external_adapter_version_spec_invalid",
        )
    except InvalidVersion:
        return ExternalAdapterRuntime(
            adapter_id=spec.adapter_id,
            package_name=spec.package_name,
            import_name=spec.import_name,
            expected_version_spec=spec.version_spec,
            status="incompatible_version",
            installed_version=version,
            issue_code="external_adapter_installed_version_invalid",
        )
    if not compatible:
        return ExternalAdapterRuntime(
            adapter_id=spec.adapter_id,
            package_name=spec.package_name,
            import_name=spec.import_name,
            expected_version_spec=spec.version_spec,
            status="incompatible_version",
            installed_version=version,
            issue_code="external_adapter_version_incompatible",
        )
    return ExternalAdapterRuntime(
        adapter_id=spec.adapter_id,
        package_name=spec.package_name,
        import_name=spec.import_name,
        expected_version_spec=spec.version_spec,
        status="available",
        installed_version=version,
        issue_code=None,
    )


def build_external_adapter_request(
    *,
    adapter_id: str,
    requested_by: str,
    requested_at: str,
    runtime_import_names: Iterable[str],
) -> CapabilityRequest:
    """Create the existing immutable request for one adapter; never install it."""

    spec = get_external_adapter_spec(adapter_id)
    return build_capability_request(
        method_name=spec.method_name,
        package_name=spec.package_name,
        import_name=spec.import_name,
        version_spec=spec.version_spec,
        purpose=spec.purpose,
        analysis_families=spec.analysis_families,
        required_input_roles=spec.required_input_roles,
        produced_output_roles=spec.produced_output_roles,
        license_spdx=spec.license_spdx,
        upstream_source=spec.upstream_source,
        validation_test_refs=spec.validation_test_refs,
        requested_by=requested_by,
        requested_at=requested_at,
        runtime_import_names=runtime_import_names,
    )


__all__ = [
    "EXTERNAL_ADAPTER_SPECS",
    "ExternalAdapterRuntime",
    "ExternalAdapterSpec",
    "build_external_adapter_request",
    "get_external_adapter_spec",
    "probe_external_adapter",
]
