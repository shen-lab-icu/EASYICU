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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..contracts.model_retention import estimable_missing_category_rows
from ..contracts.model_terms import (
    ModelTermSpec,
    level_identity_class,
    level_spelling,
)


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
    values = series.astype("object").where(series.notna(), None)
    keys = values.map(level_spelling)

    # A spelling reached by more than one type class is two source categories
    # wearing one name. Treatment coding would merge them into a single
    # contrast and the closed-domain check above would see nothing wrong,
    # because it compares spellings to spellings.
    classes_by_spelling: Dict[str, set] = {}
    for raw in values.unique().tolist():
        spelling = level_spelling(raw)
        if not spelling:
            continue
        classes_by_spelling.setdefault(spelling, set()).add(level_identity_class(raw))
    ambiguous = sorted(
        spelling
        for spelling, classes in classes_by_spelling.items()
        if len(classes) > 1
    )
    if ambiguous:
        raise ModelTermCompilationError(
            "model_term_level_identity_ambiguous",
            f"{term.name!r} holds values of more than one type that share a "
            "declared level spelling, so distinct categories would be coded as "
            "one: "
            + ", ".join(
                f"{spelling!r} ({'/'.join(sorted(classes_by_spelling[spelling]))})"
                for spelling in ambiguous
            ),
        )

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


@dataclass(frozen=True, slots=True)
class MissingCategoryTerm:
    """One covariate whose unmeasured rows a model keeps as their own state."""

    covariate: str
    indicator: str
    coding: str
    #: The observed median a continuous or ordinal covariate is filled with;
    #: ``None`` for a binary or categorical one, whose contrasts are zero.
    fill_value: Optional[float]
    n_missing: int

    def public(self) -> Dict[str, Any]:
        return {
            "covariate": self.covariate,
            "term": self.indicator,
            "coding": self.coding,
            "fill_value": self.fill_value,
            "n_missing": self.n_missing,
        }


@dataclass(frozen=True, slots=True)
class UnmeasuredRowsDropped:
    """A listed covariate whose unmeasured state could not be estimated.

    Its unmeasured rows leave the fit, as under the default policy; the
    reason says which part of the estimability rule failed.
    """

    covariate: str
    n_missing: int
    reason_code: str

    def public(self) -> Dict[str, Any]:
        return {
            "covariate": self.covariate,
            "n_missing": self.n_missing,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class PrimaryModelRows:
    """The rows one declared model is fitted on, and their complete design."""

    #: Fitting rows only, indexed like the source frame; no value is missing.
    design: Any
    exposure_columns: Tuple[str, ...]
    source_by_design_column: Mapping[str, str]
    missing_category_terms: Tuple[MissingCategoryTerm, ...] = ()
    unmeasured_rows_dropped: Tuple[UnmeasuredRowsDropped, ...] = ()

    @property
    def availability_columns(self) -> Tuple[str, ...]:
        return tuple(item.indicator for item in self.missing_category_terms)


def _unmeasured_indicator(term: ModelTermSpec) -> str:
    # A binary or categorical covariate gains a level, spelled like its other
    # contrasts; a continuous or ordinal one gains an indicator beside it.
    if term.coding in {"binary", "categorical"}:
        return _contrast_column(term.name, "unmeasured")
    return f"{term.name}__unmeasured"


def primary_model_rows(
    frame: Any,
    *,
    terms: Sequence[ModelTermSpec],
    exposure: str,
    outcome: str,
    missing_category_covariates: Sequence[str] = (),
    term_groups: Optional[Mapping[str, str]] = None,
) -> PrimaryModelRows:
    """Select, and where declared fill, the rows one declared model uses.

    Without a declared policy these are the complete rows every fit already
    used: the outcome and every design column present.  A covariate listed in
    ``missing_category_covariates`` no longer removes its unmeasured rows.  A
    continuous or ordinal one is filled with its observed median over the
    fitting rows and gains an ``<name>__unmeasured`` indicator; a binary or
    categorical one gains an ``<name>__is_unmeasured`` level.  The exposure,
    the outcome and every unlisted covariate still have to be observed.  A
    state too small, or too uniform in its outcome, to be estimated is not
    kept: its rows leave the fit and ``unmeasured_rows_dropped`` says why
    (``contracts.model_retention.estimable_missing_category_rows``).

    ``term_groups`` maps generated terms (a spline basis) to the covariate
    they represent, so a listed covariate expanded into several continuous
    columns is kept as one state: its columns are filled with 0 there, which
    its ``<name>__unmeasured`` indicator absorbs, and the same rule decides.
    """

    roster = [
        item if isinstance(item, ModelTermSpec) else ModelTermSpec.model_validate(item)
        for item in terms
    ]
    compiled = compile_model_terms(frame, terms=roster, exposure=exposure)
    if outcome not in frame.columns:
        raise ModelTermCompilationError(
            "model_term_source_missing", f"the declared outcome {outcome!r} is absent"
        )
    groups = {str(key): str(value) for key, value in (term_groups or {}).items()}
    by_term = {item.name: item for item in roster}
    if any(
        key not in by_term
        or by_term[key].role != "covariate"
        or by_term[key].coding != "continuous"
        or value in by_term
        for key, value in groups.items()
    ):
        raise ModelTermCompilationError(
            "missing_category_group_invalid",
            "a term group must map continuous covariate terms to a covariate "
            "that is not itself a term",
        )
    members: Dict[str, list[ModelTermSpec]] = {}
    for item in roster:
        members.setdefault(groups.get(item.name, item.name), []).append(item)
    listed = list(dict.fromkeys(str(name).strip() for name in missing_category_covariates))
    for name in listed:
        if name not in members:
            raise ModelTermCompilationError(
                "missing_category_covariate_undeclared",
                f"{name!r} is not a declared model term",
            )
        if any(item.role == "exposure" for item in members[name]):
            raise ModelTermCompilationError(
                "missing_category_exposure_refused",
                f"the exposure {name!r} cannot be kept as unmeasured",
            )
    columns_by_source: Dict[str, list[str]] = {}
    for column, source in compiled.source_by_design_column.items():
        columns_by_source.setdefault(groups.get(source, source), []).append(column)

    design = compiled.design
    admitted = frame[outcome].notna()
    for source, columns in columns_by_source.items():
        if source not in listed:
            admitted &= design[columns].notna().all(axis=1)
    unmeasured_by_name = {
        name: design[columns_by_source[name]].isna().any(axis=1) for name in listed
    }
    n_admitted = int(admitted.sum())
    for name in listed:
        if n_admitted and int((unmeasured_by_name[name] & admitted).sum()) == n_admitted:
            raise ModelTermCompilationError(
                "missing_category_covariate_unobserved",
                f"{name!r} is unmeasured on every fitting row",
            )
    keep, not_estimable = estimable_missing_category_rows(
        rows=admitted, unmeasured=unmeasured_by_name, outcome=frame[outcome]
    )
    fitted = design.loc[keep].copy()
    source_by_design_column = dict(compiled.source_by_design_column)
    kept: list[MissingCategoryTerm] = []
    patterns: Dict[bytes, str] = {}
    for name in listed:
        if name in not_estimable:
            continue
        group = members[name]
        term = group[0]
        columns = columns_by_source[name]
        unmeasured = fitted[columns].isna().any(axis=1)
        n_missing = int(unmeasured.sum())
        if n_missing == 0:
            continue
        if n_missing == len(fitted):
            raise ModelTermCompilationError(
                "missing_category_covariate_unobserved",
                f"{name!r} is unmeasured on every fitting row",
            )
        indicator = (
            f"{name}__unmeasured" if len(group) > 1 else _unmeasured_indicator(term)
        )
        if indicator in fitted.columns or indicator in frame.columns:
            raise ModelTermCompilationError(
                "missing_category_level_collision"
                if term.coding in {"binary", "categorical"}
                else "missing_category_term_collision",
                f"{indicator!r} already names a column or declared level",
            )
        pattern = unmeasured.to_numpy().tobytes()
        if pattern in patterns:
            raise ModelTermCompilationError(
                "missing_category_indicator_collinear",
                f"{name!r} and {patterns[pattern]!r} are unmeasured on exactly "
                "the same rows, so their indicators cannot both be estimated",
            )
        patterns[pattern] = name
        fill: Optional[float] = None
        coding = str(term.coding)
        if len(group) > 1:
            # A basis is filled at 0; the indicator absorbs any constant.
            coding = "continuous_basis"
            fill = 0.0
            fitted.loc[unmeasured, columns] = 0.0
        elif term.coding in {"continuous", "ordinal_linear"}:
            (column,) = columns
            fill = float(fitted.loc[~unmeasured, column].median())
            fitted[column] = fitted[column].fillna(fill)
        else:
            fitted.loc[unmeasured, columns] = 0.0
        fitted[indicator] = unmeasured.astype(float)
        source_by_design_column[indicator] = name
        kept.append(
            MissingCategoryTerm(
                covariate=name,
                indicator=indicator,
                coding=coding,
                fill_value=fill,
                n_missing=n_missing,
            )
        )
    return PrimaryModelRows(
        design=fitted,
        exposure_columns=compiled.exposure_columns,
        source_by_design_column=source_by_design_column,
        missing_category_terms=tuple(kept),
        unmeasured_rows_dropped=tuple(
            UnmeasuredRowsDropped(covariate=name, n_missing=n_missing, reason_code=reason)
            for name, (n_missing, reason) in sorted(not_estimable.items())
        ),
    )


__all__ = [
    "CompiledModelMatrix",
    "MissingCategoryTerm",
    "ModelTermCompilationError",
    "PrimaryModelRows",
    "UnmeasuredRowsDropped",
    "compile_model_terms",
    "primary_model_rows",
]
