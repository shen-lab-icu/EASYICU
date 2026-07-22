"""Host-owned, deterministic resource catalog and selection boundary."""

from .catalog import ResourceCatalog, protocol_catalog_from_know_how
from .capability import (
    CapabilityApproval,
    CapabilityRequest,
    approved_capability_resource,
    build_capability_request,
    write_capability_request,
)
from .coder import (
    CODER_RESOURCE_BUNDLE_SCHEMA,
    CODER_RESOURCE_PROMPT_LIMIT_BYTES,
    CoderResourceBundle,
    CoderResourceIntegrityError,
    attach_coder_resources,
    attach_step_coder_input_authority,
    bind_materialized_coder_authority,
    bind_primary_cohort_role,
    build_coder_resource_bundle,
    persist_coder_resource_bundle,
)
from .context import (
    AssembledContext,
    BoundedContextAssembler,
    ContextAssemblyReceipt,
    ContextBudgetExceeded,
    ContextSegment,
    bounded_request_metrics,
)
from .scheduler import ProtocolResourceSelection, ResourceScheduler, ResourceSelection
from .schema import (
    ResourceDescriptor,
    ResourceSelectionPolicy,
    ResourceSelectionQuery,
    ResourceSelectionReceipt,
)

__all__ = [
    "ProtocolResourceSelection",
    "ResourceSelection",
    "AssembledContext",
    "BoundedContextAssembler",
    "ContextAssemblyReceipt",
    "ContextBudgetExceeded",
    "ContextSegment",
    "bounded_request_metrics",
    "ResourceCatalog",
    "ResourceDescriptor",
    "ResourceScheduler",
    "ResourceSelectionPolicy",
    "ResourceSelectionQuery",
    "ResourceSelectionReceipt",
    "protocol_catalog_from_know_how",
    "CapabilityApproval",
    "CapabilityRequest",
    "CODER_RESOURCE_BUNDLE_SCHEMA",
    "CODER_RESOURCE_PROMPT_LIMIT_BYTES",
    "CoderResourceBundle",
    "CoderResourceIntegrityError",
    "approved_capability_resource",
    "build_capability_request",
    "write_capability_request",
    "attach_coder_resources",
    "attach_step_coder_input_authority",
    "bind_materialized_coder_authority",
    "bind_primary_cohort_role",
    "build_coder_resource_bundle",
    "persist_coder_resource_bundle",
]
