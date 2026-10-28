"""Executor wrappers and compatibility exports for verified typed inputs.

The shared artifact-consumption owner verifies bindings for both intake and
execution. Environment-based cohort wrappers remain on the execution side.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

__all__ = [
    "BINDING_REASON_CODES",
    "CLOSED_COHORT_PRODUCT_KEYS",
    "CLOSED_COHORT_PRODUCT_KIND",
    "BoundTypedInput",
    "TypedInputBindingError",
    "closed_cohort_product_vocabulary",
    "contained_regular_file",
    "is_closed_cohort_product_key",
    "load_step_cohort_frame",
    "load_typed_cohort",
    "load_typed_input",
    "read_frame",
    "run_dir_from_env",
    "sha256_file",
    "sole_typed_cohort_input",
]

# The closed-cohort product vocabulary now lives in
# ``contracts.cohort_product_keys`` so plan validation can read it without
# importing an executor.  It is re-exported here unchanged: every existing
# caller imports these five names from this module.
from ...contracts.cohort_product_keys import (  # noqa: E402
    CLOSED_COHORT_PRODUCT_KEYS,
    CLOSED_COHORT_PRODUCT_KIND,
    closed_cohort_product_vocabulary,
    is_closed_cohort_product_key,
    sole_typed_cohort_input,
)

from ...contracts.artifact_consumption import (  # noqa: E402
    BINDING_REASON_CODES,
    BoundTypedInput,
    TypedInputBindingError,
    contained_regular_file,
    load_typed_input,
    read_frame,
    sha256_file,
)


def load_typed_cohort(
    *,
    input_key: str,
    run_dir: Path,
    resolved_inputs_path: Path,
    require_consumption_contract: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Load a bound cohort frame, verifying the binding completely.

    The cohort capsule goes through the same owner as every other typed input.
    It is bound without ``exclusive`` because a cohort-consuming step may
    legitimately declare further inputs alongside it, and without a required
    ``step_id`` because the pre-typed cohort manifest predates carrying one;
    every capsule field that *is* present is checked either way.
    """

    bound = load_typed_input(
        input_key=input_key,
        run_dir=Path(run_dir),
        resolved_inputs=Path(resolved_inputs_path),
        require_consumption_contract=require_consumption_contract,
    )
    return bound.frame, bound.path


def run_dir_from_env() -> Path:
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    return Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2]).resolve()


def load_step_cohort_frame(
    *,
    typed_cohort_input: Optional[str],
    require_consumption_contract: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Load the step's bound cohort once, for every consumer of it.

    ``typed_cohort_input is None`` is the **pre-typed** path, where the runner
    hands the cohort over by environment variable instead of by binding. Only a
    caller that explicitly passes ``None`` reaches it. An executor that claims
    to own a step deterministically must not: a bare ``COHORT_PARQUET`` carries
    no digest, no product contract and no producer, so nothing computed from it
    can be bound back to the plan that asked for it.
    """

    if typed_cohort_input is None:
        cohort_path = Path(os.environ["COHORT_PARQUET"]).resolve()
        return read_frame(cohort_path), cohort_path
    return load_typed_cohort(
        input_key=typed_cohort_input,
        run_dir=run_dir_from_env(),
        resolved_inputs_path=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
        require_consumption_contract=require_consumption_contract,
    )
