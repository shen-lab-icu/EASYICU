"""Read the exact typed cohort a step was bound to, or refuse.

Owner of one question: *which bytes is this step allowed to read, and are they
the bytes the host promised?* Every deterministic executor that consumes
``artifact:analysis_cohort`` (or a ``cohort:`` product) asks it, and each
answer must be identical -- a step that reads a different frame from the one
the host sealed produces results nothing can bind.

The rules here are deliberately joined rather than separately optional:

* the path must resolve **inside** the run directory, with no symlink on any
  segment, so a binding cannot point outside the run;
* the bytes must hash to the digest recorded in the binding;
* the frame's columns and row count must equal the ``product_contract`` --
  a digest proves the file is unchanged, not that it is the table promised.

Checking only the digest, or only the contract, leaves a real gap, so neither
is offered alone.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

__all__ = [
    "TypedCohortBindingError",
    "contained_regular_file",
    "load_step_cohort_frame",
    "load_typed_cohort",
    "read_frame",
    "run_dir_from_env",
    "sha256_file",
]


class TypedCohortBindingError(RuntimeError):
    """A cohort binding could not be honoured exactly as recorded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_regular_file(path: Path, root: Path) -> Optional[Path]:
    """Return ``path`` only if it is a real file genuinely inside ``root``.

    Both the pre- and post-resolution containment checks are required: the
    first refuses a binding that names somewhere else, the second refuses one
    that reaches somewhere else through a link.
    """

    root = root.resolve()
    candidate = Path(path)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            return None
        parent = cursor.parent
        if parent == cursor:
            return None
        cursor = parent
    if not candidate.is_file():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise TypedCohortBindingError("Typed cohort table format is unsupported")


def load_typed_cohort(
    *,
    input_key: str,
    run_dir: Path,
    resolved_inputs_path: Path,
) -> tuple[pd.DataFrame, Path]:
    """Load exactly the frame recorded for ``input_key``, verifying it fully."""

    try:
        payload = json.loads(resolved_inputs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TypedCohortBindingError("Resolved input manifest is unreadable") from exc
    inputs = payload.get("inputs") if isinstance(payload, dict) else None
    binding = inputs.get(input_key) if isinstance(inputs, dict) else None
    if not isinstance(binding, dict):
        raise TypedCohortBindingError(
            f"Missing exact typed cohort binding: {input_key}"
        )
    relative_path = binding.get("relative_path")
    expected_sha256 = binding.get("sha256")
    contract = binding.get("product_contract")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not isinstance(contract, dict)
    ):
        raise TypedCohortBindingError("Typed cohort binding is incomplete")
    candidate = run_dir / relative_path
    cohort_path = contained_regular_file(candidate, run_dir)
    if cohort_path is None:
        raise TypedCohortBindingError(
            "Typed cohort binding is not a contained regular file"
        )
    if sha256_file(cohort_path) != expected_sha256:
        raise TypedCohortBindingError("Typed cohort digest verification failed")
    columns = contract.get("columns")
    row_count = contract.get("row_count")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(value, str) and value for value in columns)
        or len(columns) != len(set(columns))
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 0
    ):
        raise TypedCohortBindingError("Typed cohort product_contract is incomplete")
    frame = read_frame(cohort_path)
    if list(frame.columns) != columns:
        raise TypedCohortBindingError(
            "Typed cohort columns do not match product_contract"
        )
    if len(frame) != row_count:
        raise TypedCohortBindingError(
            "Typed cohort row count does not match product_contract"
        )
    return frame, cohort_path


def run_dir_from_env() -> Path:
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    return Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2]).resolve()


def load_step_cohort_frame(
    *,
    typed_cohort_input: str | None,
) -> tuple[pd.DataFrame, Path]:
    """Load the step's bound cohort once, for every consumer of it.

    ``typed_cohort_input is None`` is the pre-typed path, where the runner
    hands the cohort over by environment variable instead of by binding.
    """

    if typed_cohort_input is None:
        cohort_path = Path(os.environ["COHORT_PARQUET"]).resolve()
        return read_frame(cohort_path), cohort_path
    return load_typed_cohort(
        input_key=typed_cohort_input,
        run_dir=run_dir_from_env(),
        resolved_inputs_path=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
    )
