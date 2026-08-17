"""Compatibility adapter for the experimental prediction-validation owner.

This module remains outside production selection.  The dependency-neutral
owner is shared with the authority bridge so exact-source parsing and host
recomputation have one implementation.
"""

from ...prediction_validation_owner import (
    prediction_validation_receipt_findings,
    prediction_validation_result_findings,
    run_prediction_validation,
    run_prediction_validation_csv,
    seal_prediction_validation_receipt,
)

__all__ = [
    "prediction_validation_receipt_findings",
    "prediction_validation_result_findings",
    "run_prediction_validation",
    "run_prediction_validation_csv",
    "seal_prediction_validation_receipt",
]
