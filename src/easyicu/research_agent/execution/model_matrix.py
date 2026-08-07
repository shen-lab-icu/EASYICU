"""Compile Planner-declared model terms into a numeric design matrix.

This is the sole execution owner for variable coding shared by the adjusted
association and survival primary executors. It validates the observed domain
against the closed declaration and preserves missing values for the caller's
declared missing-data policy; it never chooses a reference or dtype-based
encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..contracts.model_terms import ModelTermSpec, level_spelling


class ModelTermCompilationError(ValueError):
    """A typed model-term declaration cannot be bound to its source frame."""

    owner = "easyicu.host.model_term_compiler_v1"
    phase = "model_matrix_compilation"

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


@dataclass(frozen=True, slots=True)
class CompiledModelMatrix:
    """Numeric design plus exact source lineage for every generated column."""

    design: Any
    exposure_columns: Tuple[str, ...]
    source_by_design_column: Mapping[str, str]


def _contrast_column(name: str, level: str) -> str:
    return f"{name}__is_{level}"


def _validate_observed_levels(series: Any, term: ModelTermSpec) -> Any:
    # Normalize pandas nullable scalars before using the dependency-neutral
    # spelling contract; missing values must never become an observed level.
    keys = series.astype("object").where(series.notna(), None).map(level_spelling)
    observed = {value for value in keys.unique().tolist() if value}
    declared = set(term.levels or ())
    unexpected = sorted(observed - declared)
    absent = sorted(declared - observed)
    if unexpected:
        raise ModelTermCompilationError(
            "model_term_observed_level_undeclared",
            f"{term.name!r} contains undeclared level(s): "
            + ", ".join(repr(item) for item in unexpected),
        )
    if absent:
        raise ModelTermCompilationError(
            "model_term_declared_level_absent",
            f"{term.name!r} has no observed row for declared level(s): "
            + ", ".join(repr(item) for item in absent),
        )
    return keys


def compile_model_terms(
    frame: Any,
    *,
    terms: Sequence[ModelTermSpec],
    exposure: str,
) -> CompiledModelMatrix:
    """Compile one exact term roster without inferring scientific choices."""

    import pandas as pd

    roster = tuple(
        item if isinstance(item, ModelTermSpec) else ModelTermSpec.model_validate(item)
        for item in terms
    )
    names = [item.name for item in roster]
    if len(names) != len(set(names)):
        raise ModelTermCompilationError(
            "model_term_source_repeated", "model term source names must be unique"
        )
    exposures = [item for item in roster if item.role == "exposure"]
    if len(exposures) != 1 or exposures[0].name != exposure:
        raise ModelTermCompilationError(
            "model_term_exposure_mismatch",
            "the compiled roster must contain one exact declared exposure",
        )
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ModelTermCompilationError(
            "model_term_source_missing",
            "declared model term column(s) are absent: " + ", ".join(missing),
        )

    design = pd.DataFrame(index=frame.index)
    source_by_column: Dict[str, str] = {}
    exposure_columns: list[str] = []
    for term in roster:
        source = frame[term.name]
        generated: list[str] = []
        if term.coding == "continuous":
            numeric = pd.to_numeric(source, errors="coerce")
            conversion_loss = source.notna() & numeric.isna()
            if bool(conversion_loss.any()):
                raise ModelTermCompilationError(
                    "model_term_numeric_conversion_loss",
                    f"continuous term {term.name!r} contains non-numeric values",
                )
            finite = numeric.dropna().map(lambda value: math.isfinite(float(value)))
            if not bool(finite.all()):
                raise ModelTermCompilationError(
                    "model_term_nonfinite",
                    f"continuous term {term.name!r} contains non-finite values",
                )
            design[term.name] = numeric.astype(float)
            generated = [term.name]
        elif term.coding == "ordinal_linear":
            keys = _validate_observed_levels(source, term)
            mapping = {
                level: float(index) for index, level in enumerate(term.levels or ())
            }
            encoded = keys.map(mapping)
            encoded = encoded.mask(keys.eq(""))
            design[term.name] = encoded.astype(float)
            generated = [term.name]
        else:
            keys = _validate_observed_levels(source, term)
            unobserved = keys.eq("")
            for level in term.contrast_levels:
                name = _contrast_column(term.name, level)
                design[name] = (keys == level).astype(float).mask(unobserved)
                generated.append(name)

        if not generated:
            raise ModelTermCompilationError(
                "model_term_generated_no_columns",
                f"term {term.name!r} generated no estimable design column",
            )
        for column in generated:
            source_by_column[column] = term.name
        if term.role == "exposure":
            exposure_columns.extend(generated)

    return CompiledModelMatrix(
        design=design,
        exposure_columns=tuple(exposure_columns),
        source_by_design_column=dict(source_by_column),
    )


__all__ = [
    "CompiledModelMatrix",
    "ModelTermCompilationError",
    "compile_model_terms",
]
