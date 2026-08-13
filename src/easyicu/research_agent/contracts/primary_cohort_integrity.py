"""Deterministic integrity gate for the host-locked primary analysis cohort.

The Planner owns cohort eligibility.  This module replays that typed definition
and verifies the Agent-produced cohort, denominator, and attrition products
without choosing scientific criteria itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..schema import AnalysisStep, ValidationFinding
from .primary_cohort import (
    _PRIMARY_ANALYSIS_COHORT_DATA_KINDS,
    _is_primary_analysis_cohort_flow_product,
    _primary_analysis_cohort_attrition_step,
    _primary_analysis_cohort_product_matches_plan,
    _primary_analysis_cohort_product_owner_finding,
)
from .product_files import contained_regular_output_file
from .product_identity import normalize_product_token as _normalise, typed_product


class RegisteredProductPathResolver(Protocol):
    """Resolve one typed product to output-local registered file paths."""

    def __call__(
        self,
        summary: Mapping[str, Any],
        *,
        product_name: str,
        allowed_kinds: frozenset[str],
        out_dir: Path,
    ) -> list[str]: ...


@dataclass(frozen=True)
class _FindingFactory:
    """Create stable fail-closed findings for one cohort-construction step."""

    step_id: str

    def __call__(
        self, issue: str, message: str, **detail: Any
    ) -> list[ValidationFinding]:
        return [
            ValidationFinding(
                validator="primary_analysis_cohort_integrity",
                severity="error",
                message=message,
                detail={"issue": issue, "step_id": self.step_id, **detail},
            )
        ]


@dataclass(frozen=True)
class _CohortReplayAuthority:
    """Immutable host replay facts consumed by downstream product validators."""

    universe: Any
    authoritative: Any
    replayed: Any
    raw_n: int
    locked_n: int
    identity_column: str
    authoritative_identity: Any
    expected_remaining_counts: tuple[int, ...]
    expected_exclusion_counts: tuple[int, ...]
    expected_criterion_ids: tuple[str, ...]
    accepted_criterion_ids: tuple[tuple[str, ...], ...]
    criterion_aliases: tuple[tuple[str, str], ...]

    def canonicalise_reported_rule_ids(
        self, reported: Sequence[str]
    ) -> list[str] | None:
        lookup = dict(self.criterion_aliases)
        canonical: list[str] = []
        for rule_id in reported:
            resolved = lookup.get(rule_id)
            if resolved is None:
                return None
            canonical.append(resolved)
        return canonical


def _integral_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


_PRIMARY_RAW_DENOMINATOR_FIELDS = (
    "n_universe",
    "universe_n",
    "n_input_universe",
)
_PRIMARY_FINAL_DENOMINATOR_FIELDS = (
    "n_final_analysis_cohort",
    "n_analysis_cohort",
    "n_final_cohort",
    "final_cohort_n",
)


def _coherent_integral_count_columns(
    table: Any,
    names: Sequence[str],
) -> tuple[str | None, dict[str, list[int | None]], str | None]:
    """Read synonymous structural-count columns without first-match masking."""

    present = [name for name in names if name in table.columns]
    values_by_column = {
        name: [_integral_count(value) for value in table[name].tolist()]
        for name in present
    }
    if any(value is None for values in values_by_column.values() for value in values):
        return (present[0] if present else None), values_by_column, "nonintegral"
    if len(present) > 1:
        reference = values_by_column[present[0]]
        if any(values_by_column[name] != reference for name in present[1:]):
            return present[0], values_by_column, "disagree"
    return (present[0] if present else None), values_by_column, None


def _coherent_integral_mapping_counts(
    mapping: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[int | None, dict[str, int | None], str | None]:
    """Read all present synonymous count fields from one declared mapping."""

    present = [name for name in names if name in mapping]
    values_by_field = {name: _integral_count(mapping[name]) for name in present}
    if any(value is None for value in values_by_field.values()):
        return None, values_by_field, "nonintegral"
    if len(present) > 1:
        reference = values_by_field[present[0]]
        if any(values_by_field[name] != reference for name in present[1:]):
            return reference, values_by_field, "disagree"
    return (values_by_field[present[0]] if present else None), values_by_field, None


def _assert_exact_frame_values_equal(left: Any, right: Any) -> None:
    """Compare host values exactly while ignoring storage-only dtype changes.

    Parquet round-trips may represent the same nullable flag as either
    ``boolean {False, True, NA}`` or ``float {0.0, 1.0, NaN}``.  Pandas'
    frame comparator treats those values as different even with dtype checks
    disabled.  Authority needs a narrower rule: column order and missingness
    must match exactly, and every nonmissing scalar must compare exactly after
    conversion from NumPy/Pandas scalar wrappers to Python scalars.  No numeric
    tolerance, string coercion, or scientific recoding is permitted.
    """

    if left.shape != right.shape:
        raise AssertionError(f"frame shapes differ: {left.shape} != {right.shape}")
    if list(left.columns) != list(right.columns):
        raise AssertionError("frame columns differ")

    def python_scalar(value: Any) -> Any:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                pass
        return value

    for position, column in enumerate(left.columns):
        left_series = left.iloc[:, position].reset_index(drop=True)
        right_series = right.iloc[:, position].reset_index(drop=True)
        left_missing = left_series.isna().tolist()
        right_missing = right_series.isna().tolist()
        if left_missing != right_missing:
            raise AssertionError(f"column {column!r} has different missingness")
        for row, (left_value, right_value, missing) in enumerate(
            zip(left_series.tolist(), right_series.tolist(), left_missing, strict=True)
        ):
            if missing:
                continue
            left_scalar = python_scalar(left_value)
            right_scalar = python_scalar(right_value)
            try:
                equal = bool(left_scalar == right_scalar)
            except (TypeError, ValueError):
                equal = False
            if not equal:
                raise AssertionError(
                    f"column {column!r} differs at row {row}: "
                    f"{left_scalar!r} != {right_scalar!r}"
                )


def _validate_attrition_products(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    out_dir: Path,
    registered_product_paths: RegisteredProductPathResolver,
    authority: _CohortReplayAuthority,
    finding: _FindingFactory,
) -> list[ValidationFinding]:
    """Validate every declared cohort-flow table against the host replay."""

    import pandas as pd

    expected_remaining_counts = list(authority.expected_remaining_counts)
    expected_exclusion_counts = list(authority.expected_exclusion_counts)
    expected_criterion_ids = list(authority.expected_criterion_ids)
    accepted_criterion_ids = list(authority.accepted_criterion_ids)

    declared_flow_products = {
        name
        for raw in (step.expected_outputs or [])
        if (parsed := typed_product(raw)) is not None
        for kind, name in (parsed,)
        if kind == "table" and _is_primary_analysis_cohort_flow_product(name)
    }
    for product in sorted(declared_flow_products):
        candidates = registered_product_paths(
            step_summary,
            product_name=product,
            allowed_kinds=frozenset({"table"}),
            out_dir=out_dir,
        )
        if len(candidates) != 1:
            return finding(
                "attrition_product_ambiguous",
                f"The declared {product} product must resolve to exactly one "
                "registered table.",
                product=product,
                candidates=candidates,
            )
        try:
            table_path = contained_regular_output_file(Path(out_dir), candidates[0])
            if table_path.suffix.lower() == ".parquet":
                table = pd.read_parquet(table_path)
            elif table_path.suffix.lower() == ".feather":
                table = pd.read_feather(table_path)
            else:
                table = pd.read_csv(
                    table_path,
                    sep="\t" if table_path.suffix.lower() == ".tsv" else ",",
                )
        except Exception as exc:
            return finding(
                "attrition_product_unreadable",
                f"The declared {product} table is unreadable or outside the step "
                "output directory.",
                product=product,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
        if table.empty:
            return finding(
                "attrition_counts_unverifiable",
                f"The declared {product} table has no verifiable count sequence.",
                product=product,
            )
        # Only typed identity columns carry an exact host-owned predicate ID.
        # Legacy ``criterion`` remains descriptive free text; its row order and
        # verified counts provide the compatibility identity without routing on
        # case-specific wording.
        canonical_identity_columns = [
            name
            for name in ("criterion_id", "attrition_category")
            if name in table.columns
        ]
        normalised_identity_by_column = {
            name: [_normalise(value) for value in table[name].tolist()]
            for name in canonical_identity_columns
        }
        if len(canonical_identity_columns) > 1:
            reference = normalised_identity_by_column[canonical_identity_columns[0]]
            if any(
                normalised_identity_by_column[name] != reference
                for name in canonical_identity_columns[1:]
            ):
                return finding(
                    "attrition_identity_columns_disagree",
                    f"The declared {product} table has contradictory canonical "
                    "Planner-predicate identity columns.",
                    product=product,
                    identity_columns=canonical_identity_columns,
                    identities_by_column=normalised_identity_by_column,
                )
        canonical_identity_column = (
            canonical_identity_columns[0] if canonical_identity_columns else None
        )

        remaining_column, remaining_by_column, remaining_issue = (
            _coherent_integral_count_columns(
                table,
                ("n_remaining", "n_remaining_rows"),
            )
        )
        partition_count_column, partition_by_column, partition_issue = (
            _coherent_integral_count_columns(table, ("n", "n_rows"))
        )
        if remaining_column is None and partition_count_column is None:
            denominator_row = table.iloc[0].to_dict()
            table_raw_n, table_raw_by_field, table_raw_issue = (
                _coherent_integral_mapping_counts(
                    denominator_row,
                    _PRIMARY_RAW_DENOMINATOR_FIELDS,
                )
            )
            table_final_n, table_final_by_field, table_final_issue = (
                _coherent_integral_mapping_counts(
                    denominator_row,
                    _PRIMARY_FINAL_DENOMINATOR_FIELDS,
                )
            )
            if table_raw_issue == "nonintegral" or table_final_issue == "nonintegral":
                return finding(
                    "cohort_denominator_fields_nonintegral",
                    f"The declared {product} table contains a non-integral "
                    "structural denominator in one of its synonymous fields.",
                    product=product,
                    raw_denominators_by_field=table_raw_by_field,
                    final_denominators_by_field=table_final_by_field,
                )
            if table_raw_issue == "disagree" or table_final_issue == "disagree":
                return finding(
                    "cohort_denominator_fields_disagree",
                    f"The declared {product} table contains contradictory "
                    "synonymous structural denominator fields.",
                    product=product,
                    raw_denominators_by_field=table_raw_by_field,
                    final_denominators_by_field=table_final_by_field,
                )
            if (
                product in {"cohort_denominator", "cohort_denominators"}
                and len(table) == 1
                and table_raw_n == authority.raw_n
                and table_final_n == authority.locked_n
            ):
                continue
            return finding(
                "attrition_counts_unverifiable",
                f"The declared {product} table has no verifiable count sequence.",
                product=product,
            )
        if remaining_column is not None:
            count_column = remaining_column
            count_values_by_column = remaining_by_column
            count_issue = remaining_issue
        else:
            count_column = partition_count_column
            count_values_by_column = partition_by_column
            count_issue = partition_issue
        assert count_column is not None
        if count_issue == "nonintegral":
            return finding(
                "attrition_counts_nonintegral",
                f"The declared {product} table contains invalid structural counts.",
                product=product,
                count_column=count_column,
                count_columns=list(count_values_by_column),
            )
        if count_issue == "disagree":
            return finding(
                "attrition_count_columns_disagree",
                f"The declared {product} table has contradictory synonymous "
                "structural-count columns.",
                product=product,
                count_columns=list(count_values_by_column),
                values_by_column=count_values_by_column,
            )
        integral_counts = [
            int(value)
            for value in count_values_by_column[count_column]
            if value is not None
        ]
        if any(value < 0 for value in integral_counts):
            return finding(
                "attrition_counts_negative",
                f"The declared {product} table contains negative structural counts.",
                product=product,
                count_column=count_column,
            )

        # Sequential flow schema: every row records the population remaining
        # after one rule.  When exclusions are explicit, each transition must
        # reconcile row by row; a matching grand total alone is insufficient.
        if remaining_column is not None:
            if integral_counts[0] != authority.raw_n or integral_counts[-1] != authority.locked_n:
                return finding(
                    "attrition_endpoints_mismatch",
                    f"The declared {product} table does not run from the raw "
                    "universe to the locked analysis cohort.",
                    product=product,
                    expected_universe_n=authority.raw_n,
                    reported_first_n=integral_counts[0],
                    expected_final_n=authority.locked_n,
                    reported_last_n=integral_counts[-1],
                )
            if any(
                current > previous
                for previous, current in zip(
                    integral_counts, integral_counts[1:], strict=False
                )
            ):
                return finding(
                    "attrition_counts_increase",
                    f"The declared {product} remaining-count sequence increases.",
                    product=product,
                )
            excluded_column, exclusions_by_column, exclusion_issue = (
                _coherent_integral_count_columns(
                    table,
                    (
                        "n_excluded_at_step",
                        "n_removed_from_prior_stage",
                        "n_excluded_rows",
                    ),
                )
            )
            if excluded_column is None:
                return finding(
                    "attrition_exclusions_unverifiable",
                    f"The declared {product} table does not report per-step "
                    "exclusions.",
                    product=product,
                )
            if exclusion_issue == "nonintegral":
                return finding(
                    "attrition_exclusions_nonintegral",
                    f"The declared {product} table contains invalid per-step "
                    "exclusion counts.",
                    product=product,
                    count_columns=list(exclusions_by_column),
                )
            if exclusion_issue == "disagree":
                return finding(
                    "attrition_count_columns_disagree",
                    f"The declared {product} table has contradictory synonymous "
                    "per-step exclusion-count columns.",
                    product=product,
                    count_columns=list(exclusions_by_column),
                    values_by_column=exclusions_by_column,
                )
            integral_exclusions = [
                int(value)
                for value in exclusions_by_column[excluded_column]
                if value is not None
            ]
            reported_remaining = list(integral_counts)
            reported_exclusions = list(integral_exclusions)
            has_terminal_row = False
            if (
                len(reported_remaining) == len(expected_remaining_counts) + 1
                and reported_remaining[-1] == authority.locked_n
                and reported_exclusions[-1] == 0
            ):
                has_terminal_row = True
                reported_remaining.pop()
                reported_exclusions.pop()
            if reported_remaining != expected_remaining_counts:
                return finding(
                    "attrition_stage_counts_mismatch",
                    f"The declared {product} remaining counts do not match the "
                    "ordered Planner-locked cohort predicates.",
                    product=product,
                    expected_remaining_counts=expected_remaining_counts,
                    reported_remaining_counts=reported_remaining,
                )
            if reported_exclusions != expected_exclusion_counts:
                return finding(
                    "attrition_transitions_do_not_conserve",
                    f"The declared {product} exclusions do not match each "
                    "ordered Planner-locked cohort predicate.",
                    product=product,
                    expected_exclusions=expected_exclusion_counts,
                    reported_exclusions=reported_exclusions,
                )
            if canonical_identity_column is not None:
                reported_rule_ids = normalised_identity_by_column[
                    canonical_identity_column
                ]
                expected_rule_ids = ["universe", *expected_criterion_ids]
                reported_predicate_ids = (
                    reported_rule_ids[:-1] if has_terminal_row else reported_rule_ids
                )
                reported_boundary_id = (
                    reported_predicate_ids[0] if reported_predicate_ids else None
                )
                canonical_reported_predicates = authority.canonicalise_reported_rule_ids(
                    reported_predicate_ids[1:]
                )
                allowed_terminal_ids = {
                    "analysis_cohort",
                    "final_analysis_cohort",
                    "final_cohort",
                    "primary_analysis_cohort",
                }
                terminal_id_valid = (
                    not has_terminal_row
                    or reported_rule_ids[-1] in allowed_terminal_ids
                )
                if (
                    reported_boundary_id != "universe"
                    or canonical_reported_predicates != expected_criterion_ids
                    or not terminal_id_valid
                ):
                    return finding(
                        "attrition_sequence_rule_ids_mismatch",
                        f"The declared {product} canonical identity sequence does "
                        "not bind each row to the corresponding Planner-locked "
                        "cohort predicate.",
                        product=product,
                        identity_column=canonical_identity_column,
                        expected_criterion_ids=expected_rule_ids,
                        accepted_criterion_ids=[
                            ["universe"],
                            *[list(aliases) for aliases in accepted_criterion_ids],
                        ],
                        reported_criterion_ids=reported_rule_ids,
                    )
            start_column = (
                "n_at_start_rows" if "n_at_start_rows" in table.columns else None
            )
            if start_column is not None:
                starts = [
                    _integral_count(value) for value in table[start_column].tolist()
                ]
                expected_starts = [authority.raw_n, *integral_counts[:-1]]
                if (
                    any(value is None for value in starts)
                    or [int(value) for value in starts if value is not None]
                    != expected_starts
                ):
                    return finding(
                        "attrition_start_counts_do_not_conserve",
                        f"The declared {product} start counts do not match the "
                        "preceding remaining denominators.",
                        product=product,
                        expected_start_counts=expected_starts,
                    )
            continue

        # Partition schema: one denominator row, one retained row, and zero or
        # more mutually exclusive excluded partitions.  A bare ``n`` column is
        # never treated as an ordered flow because that would make arbitrary
        # intermediate values unverifiable.
        assert partition_count_column is not None
        role_columns = [
            name
            for name in ("status", "partition_status", "row_role", "role")
            if name in table.columns
        ]
        normalised_roles_by_column = {
            name: [_normalise(value) for value in table[name].tolist()]
            for name in role_columns
        }
        if len(role_columns) > 1:
            reference = normalised_roles_by_column[role_columns[0]]
            if any(
                normalised_roles_by_column[name] != reference
                for name in role_columns[1:]
            ):
                return finding(
                    "attrition_role_columns_disagree",
                    f"The declared {product} partition table has contradictory "
                    "synonymous row-role columns.",
                    product=product,
                    role_columns=role_columns,
                    roles_by_column=normalised_roles_by_column,
                )
        role_column = role_columns[0] if role_columns else None
        if role_column is None:
            if partition_count_column == "n_rows":
                reported_remaining = list(integral_counts)
                if (
                    len(reported_remaining) == len(expected_remaining_counts) + 1
                    and reported_remaining[-1] == authority.locked_n
                ):
                    reported_remaining.pop()
                if reported_remaining == expected_remaining_counts:
                    continue
                return finding(
                    "attrition_stage_counts_mismatch",
                    f"The declared {product} row-count sequence does not match "
                    "the ordered Planner-locked cohort predicates.",
                    product=product,
                    expected_remaining_counts=expected_remaining_counts,
                    reported_remaining_counts=reported_remaining,
                )
            return finding(
                "attrition_partition_roles_unverifiable",
                f"The declared {product} partition table has no explicit row roles.",
                product=product,
                count_column=partition_count_column,
            )
        roles = normalised_roles_by_column[role_column]
        denominator_rows = [
            index for index, role in enumerate(roles) if role == "denominator"
        ]
        retained_rows = [
            index for index, role in enumerate(roles) if role == "retained"
        ]
        excluded_rows = [
            index for index, role in enumerate(roles) if role == "excluded"
        ]
        if (
            len(denominator_rows) != 1
            or len(retained_rows) != 1
            or len(denominator_rows) + len(retained_rows) + len(excluded_rows)
            != len(table)
        ):
            return finding(
                "attrition_partition_roles_invalid",
                f"The declared {product} partition table must contain exactly "
                "one denominator row, one retained row, and only explicit "
                "excluded partitions otherwise.",
                product=product,
                role_column=role_column,
            )
        reported_denominator = integral_counts[denominator_rows[0]]
        reported_retained = integral_counts[retained_rows[0]]
        reported_excluded_counts = [integral_counts[index] for index in excluded_rows]
        reported_excluded = sum(reported_excluded_counts)
        if (
            reported_denominator != authority.raw_n
            or reported_retained != authority.locked_n
            or reported_excluded != authority.raw_n - authority.locked_n
        ):
            return finding(
                "attrition_partitions_do_not_conserve",
                f"The declared {product} partitions do not reconcile the raw "
                "and final cohort denominators.",
                product=product,
                expected_universe_n=authority.raw_n,
                reported_universe_n=reported_denominator,
                expected_retained_n=authority.locked_n,
                reported_retained_n=reported_retained,
                expected_excluded_n=authority.raw_n - authority.locked_n,
                reported_excluded_n=reported_excluded,
            )
        category_column = canonical_identity_column
        if expected_criterion_ids and category_column is None:
            return finding(
                "attrition_partition_rule_ids_unverifiable",
                f"The declared {product} partition table does not identify the "
                "Planner-locked cohort rules attached to each exclusion.",
                product=product,
                expected_criterion_ids=expected_criterion_ids,
            )
        if expected_criterion_ids:
            assert category_column is not None
            denominator_id = _normalise(
                table.iloc[denominator_rows[0]][category_column]
            )
            retained_id = _normalise(table.iloc[retained_rows[0]][category_column])
            if denominator_id != "universe" or retained_id not in {
                "analysis_cohort",
                "final_analysis_cohort",
                "final_cohort",
                "primary_analysis_cohort",
            }:
                return finding(
                    "attrition_partition_boundary_ids_mismatch",
                    f"The declared {product} canonical identity column does not "
                    "identify the universe and retained analysis cohort exactly.",
                    product=product,
                    identity_column=category_column,
                    reported_denominator_id=denominator_id,
                    reported_retained_id=retained_id,
                )
            reported_rule_ids = [
                _normalise(table.iloc[index][category_column])
                for index in excluded_rows
            ]
            canonical_reported_rule_ids = authority.canonicalise_reported_rule_ids(
                reported_rule_ids
            )
            if (
                canonical_reported_rule_ids is None
                or len(canonical_reported_rule_ids)
                != len(set(canonical_reported_rule_ids))
                or set(canonical_reported_rule_ids) != set(expected_criterion_ids)
            ):
                return finding(
                    "attrition_partition_rule_ids_mismatch",
                    f"The declared {product} excluded partitions do not identify "
                    "each Planner-locked cohort predicate exactly once.",
                    product=product,
                    expected_criterion_ids=expected_criterion_ids,
                    accepted_criterion_ids=[
                        list(aliases) for aliases in accepted_criterion_ids
                    ],
                    reported_criterion_ids=reported_rule_ids,
                )
            reported_by_rule = dict(
                zip(
                    canonical_reported_rule_ids,
                    reported_excluded_counts,
                    strict=True,
                )
            )
            expected_by_rule = dict(
                zip(
                    expected_criterion_ids,
                    expected_exclusion_counts[1:],
                    strict=True,
                )
            )
        else:
            reported_by_rule = {}
            expected_by_rule = {}
        if reported_by_rule != expected_by_rule:
            return finding(
                "attrition_partition_rule_counts_mismatch",
                f"The declared {product} excluded partitions do not match the "
                "Planner-locked cohort predicates.",
                product=product,
                expected_by_rule=expected_by_rule,
                reported_by_rule=reported_by_rule,
            )

    return []

def primary_analysis_cohort_integrity_findings(
    *,
    step: AnalysisStep,
    plan: Any,
    step_summary: Mapping[str, Any],
    out_dir: Path,
    universe_path: Path,
    authoritative_cohort_path: Path,
    context: Any = None,
    registered_product_paths: RegisteredProductPathResolver,
) -> list[ValidationFinding]:
    """Verify one Agent-produced primary cohort against host-locked authority.

    This gate never chooses eligibility.  It replays the Planner-owned typed
    cohort definition on the raw universe, then checks the produced row identity
    and declared attrition accounting before the attempt may become current.
    """

    if not _primary_analysis_cohort_attrition_step(step):
        return []

    finding = _FindingFactory(step.step_id)

    owner_finding = _primary_analysis_cohort_product_owner_finding(
        step=step,
        plan=plan,
        validator="primary_analysis_cohort_integrity",
    )
    if owner_finding is not None:
        return [owner_finding]

    try:
        import pandas as pd

        from ..cohort.schema import (
            CohortDefinition,
            _planner_declared_context_column_bindings,
            _resolve_predicate_column,
            build_cohort,
            coerce_cohort_definition,
        )

        universe = pd.read_parquet(universe_path).reset_index(drop=True)
        authoritative = pd.read_parquet(authoritative_cohort_path).reset_index(
            drop=True
        )
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        replayed = universe.copy()
        expected_remaining_counts = [int(len(replayed))]
        expected_exclusion_counts = [0]
        expected_criterion_ids: list[str] = []
        accepted_criterion_ids: list[tuple[str, ...]] = []
        if definition is not None:
            column_bindings = _planner_declared_context_column_bindings(
                definition=definition,
                plan=plan,
                context=context,
                columns=replayed.columns,
            )
            ordered_predicates = [
                ("include", predicate) for predicate in definition.inclusion
            ] + [("exclude", predicate) for predicate in definition.exclusion]
            for order, (kind, predicate) in enumerate(ordered_predicates, start=1):
                before_n = int(len(replayed))
                resolved_column = _resolve_predicate_column(
                    replayed.columns,
                    predicate.concept_id,
                    predicate.aggregation,
                    column_bindings=column_bindings,
                )
                if resolved_column is None:
                    raise ValueError(
                        "locked cohort predicate has no host-resolved materialized "
                        f"column: {predicate.concept_id!r}"
                    )
                one = CohortDefinition(
                    name=f"criterion_{order}",
                    inclusion=(predicate,) if kind == "include" else (),
                    exclusion=(predicate,) if kind == "exclude" else (),
                )
                replayed = build_cohort(
                    one,
                    replayed,
                    column_bindings=column_bindings,
                ).reset_index(drop=True)
                after_n = int(len(replayed))
                concept = _normalise(predicate.concept_id)
                canonical_id = f"{kind}_{order:02d}_{concept}"
                resolved_id = f"{kind}_{order:02d}_{_normalise(resolved_column)}"
                expected_criterion_ids.append(canonical_id)
                accepted_criterion_ids.append(
                    tuple(dict.fromkeys((canonical_id, resolved_id)))
                )
                expected_remaining_counts.append(after_n)
                expected_exclusion_counts.append(before_n - after_n)
    except Exception as exc:
        return finding(
            "authority_replay_unavailable",
            "The primary analysis-cohort product could not be verified against "
            "the raw universe and Planner-locked cohort definition.",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )

    criterion_alias_to_canonical: dict[str, str] = {}
    for canonical_id, aliases in zip(
        expected_criterion_ids,
        accepted_criterion_ids,
        strict=True,
    ):
        for alias in aliases:
            existing = criterion_alias_to_canonical.setdefault(alias, canonical_id)
            if existing != canonical_id:
                return finding(
                    "attrition_rule_identity_ambiguous",
                    "Host-resolved cohort predicate columns do not provide a "
                    "unique attrition-rule identity.",
                    ambiguous_alias=alias,
                    canonical_criterion_ids=[existing, canonical_id],
                )

    raw_n = int(len(universe))
    locked_n = int(len(authoritative))
    if len(replayed) != locked_n:
        return finding(
            "locked_cohort_replay_count_mismatch",
            "The Planner-locked cohort replay does not match the authoritative "
            "materialised cohort denominator.",
            raw_universe_n=raw_n,
            replayed_cohort_n=int(len(replayed)),
            authoritative_cohort_n=locked_n,
        )

    identity_column = next(
        (
            candidate
            for candidate in ("stay_id", "encounter_id", "row_id")
            if all(
                candidate in frame.columns
                and not frame[candidate].isna().any()
                and not frame[candidate].duplicated().any()
                for frame in (replayed, authoritative)
            )
        ),
        None,
    )
    if identity_column is None:
        return finding(
            "row_identity_unverifiable",
            "The primary analysis cohort has no shared unique row identity; "
            "cohort membership cannot be sealed safely.",
        )

    expected_identity = (
        replayed[identity_column].astype("string").reset_index(drop=True)
    )
    authoritative_identity = (
        authoritative[identity_column].astype("string").reset_index(drop=True)
    )
    if not expected_identity.equals(authoritative_identity):
        return finding(
            "authoritative_cohort_identity_mismatch",
            "The materialised analysis cohort does not match the row identities "
            "obtained by replaying the Planner-locked definition.",
            identity_column=identity_column,
            raw_universe_n=raw_n,
            authoritative_cohort_n=locked_n,
        )
    try:
        _assert_exact_frame_values_equal(
            replayed.loc[:, authoritative.columns].reset_index(drop=True),
            authoritative,
        )
    except (AssertionError, KeyError) as exc:
        return finding(
            "authoritative_cohort_value_mismatch",
            "The materialised analysis cohort changes host columns relative to "
            "the Planner-locked cohort replay.",
            error=str(exc)[:300],
        )

    authority = _CohortReplayAuthority(
        universe=universe,
        authoritative=authoritative,
        replayed=replayed,
        raw_n=raw_n,
        locked_n=locked_n,
        identity_column=identity_column,
        authoritative_identity=authoritative_identity,
        expected_remaining_counts=tuple(expected_remaining_counts),
        expected_exclusion_counts=tuple(expected_exclusion_counts),
        expected_criterion_ids=tuple(expected_criterion_ids),
        accepted_criterion_ids=tuple(accepted_criterion_ids),
        criterion_aliases=tuple(criterion_alias_to_canonical.items()),
    )

    declared_cohort_products = {
        product[1]
        for raw in (step.expected_outputs or [])
        if (product := _primary_analysis_cohort_product_matches_plan(raw, plan=plan))
        is not None
    }
    if len(declared_cohort_products) != 1:
        return finding(
            "analysis_cohort_product_ambiguous",
            "The primary analysis-cohort product must have exactly one "
            "registered typed identity.",
            declared_products=sorted(declared_cohort_products),
        )
    cohort_product = next(iter(declared_cohort_products))
    cohort_candidates = registered_product_paths(
        step_summary,
        product_name=cohort_product,
        allowed_kinds=_PRIMARY_ANALYSIS_COHORT_DATA_KINDS,
        out_dir=out_dir,
    )
    if len(cohort_candidates) != 1:
        return finding(
            "analysis_cohort_product_ambiguous",
            "The declared primary analysis-cohort product must resolve to exactly one "
            "registered output file.",
            product=cohort_product,
            candidates=cohort_candidates,
        )
    try:
        produced_path = contained_regular_output_file(
            Path(out_dir), cohort_candidates[0]
        )
        if produced_path.suffix.lower() == ".parquet":
            produced = pd.read_parquet(produced_path).reset_index(drop=True)
        elif produced_path.suffix.lower() == ".feather":
            produced = pd.read_feather(produced_path).reset_index(drop=True)
        elif produced_path.suffix.lower() in {".csv", ".tsv"}:
            produced = pd.read_csv(
                produced_path,
                sep="\t" if produced_path.suffix.lower() == ".tsv" else ",",
            ).reset_index(drop=True)
        else:
            raise ValueError("unsupported analysis-cohort file type")
    except Exception as exc:
        return finding(
            "analysis_cohort_product_unreadable",
            "The declared analysis_cohort product is unreadable or outside the "
            "step output directory.",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
    if (
        identity_column not in produced.columns
        or produced[identity_column].isna().any()
        or produced[identity_column].duplicated().any()
        or not produced[identity_column]
        .astype("string")
        .reset_index(drop=True)
        .equals(authoritative_identity)
    ):
        return finding(
            "analysis_cohort_identity_mismatch",
            "The produced analysis_cohort does not preserve the exact ordered row "
            "identity of the Planner-locked cohort.",
            identity_column=identity_column,
            produced_cohort_n=int(len(produced)),
            authoritative_cohort_n=locked_n,
        )
    try:
        _assert_exact_frame_values_equal(
            produced.loc[:, authoritative.columns].reset_index(drop=True),
            authoritative,
        )
    except (AssertionError, KeyError) as exc:
        return finding(
            "analysis_cohort_value_mismatch",
            "The produced analysis_cohort changes or omits authoritative cohort "
            "columns; only additional derived columns are allowed.",
            error=str(exc)[:300],
        )

    reported_raw_n, raw_denominators_by_field, raw_denominator_issue = (
        _coherent_integral_mapping_counts(
            step_summary,
            _PRIMARY_RAW_DENOMINATOR_FIELDS,
        )
    )
    reported_final_n, final_denominators_by_field, final_denominator_issue = (
        _coherent_integral_mapping_counts(
            step_summary,
            _PRIMARY_FINAL_DENOMINATOR_FIELDS,
        )
    )
    if (
        raw_denominator_issue == "nonintegral"
        or final_denominator_issue == "nonintegral"
    ):
        return finding(
            "cohort_denominator_fields_nonintegral",
            "The primary cohort summary contains a non-integral structural "
            "denominator in one of its synonymous fields.",
            raw_denominators_by_field=raw_denominators_by_field,
            final_denominators_by_field=final_denominators_by_field,
        )
    if raw_denominator_issue == "disagree" or final_denominator_issue == "disagree":
        return finding(
            "cohort_denominator_fields_disagree",
            "The primary cohort summary contains contradictory synonymous "
            "structural denominator fields.",
            raw_denominators_by_field=raw_denominators_by_field,
            final_denominators_by_field=final_denominators_by_field,
        )
    if reported_raw_n != raw_n or reported_final_n != locked_n:
        return finding(
            "cohort_denominator_mismatch",
            "The primary cohort summary does not report the raw-universe and "
            "locked final denominators exactly.",
            expected_universe_n=raw_n,
            reported_universe_n=reported_raw_n,
            expected_final_n=locked_n,
            reported_final_n=reported_final_n,
        )

    return _validate_attrition_products(
        step=step,
        step_summary=step_summary,
        out_dir=out_dir,
        registered_product_paths=registered_product_paths,
        authority=authority,
        finding=finding,
    )

__all__ = [
    "RegisteredProductPathResolver",
    "primary_analysis_cohort_integrity_findings",
]
