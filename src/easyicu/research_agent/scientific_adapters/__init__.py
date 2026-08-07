"""Optional, receipt-first adapters for mature scientific libraries.

The package is not wired into the normal agent capability registry.  An
adapter becomes executable only after its request has passed the existing
approval and immutable-image activation procedure.  These modules therefore
provide implementation and test seams without silently upgrading a scientific
claim on developer-local imports.
"""

from .runtime import (
    EXTERNAL_ADAPTER_SPECS,
    ExternalAdapterRuntime,
    ExternalAdapterSpec,
    build_external_adapter_request,
    get_external_adapter_spec,
    probe_external_adapter,
)
from .pandera import (
    PanderaColumnContract,
    PanderaDataFrameContract,
    PanderaValidationReceipt,
    validate_dataframe_contract,
)
from .dowhy import (
    DoWhyIdentificationContract,
    DoWhyIdentificationReceipt,
    identify_declared_causal_effect,
)
from .sksurv import (
    CompetingRisksCIFContract,
    CompetingRisksCIFResult,
    estimate_declared_cumulative_incidence,
)

__all__ = [
    "EXTERNAL_ADAPTER_SPECS",
    "ExternalAdapterRuntime",
    "ExternalAdapterSpec",
    "build_external_adapter_request",
    "CompetingRisksCIFContract",
    "CompetingRisksCIFResult",
    "DoWhyIdentificationContract",
    "DoWhyIdentificationReceipt",
    "get_external_adapter_spec",
    "identify_declared_causal_effect",
    "estimate_declared_cumulative_incidence",
    "PanderaColumnContract",
    "PanderaDataFrameContract",
    "PanderaValidationReceipt",
    "probe_external_adapter",
    "validate_dataframe_contract",
]
