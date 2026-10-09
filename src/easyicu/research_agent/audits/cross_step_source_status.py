"""Source-status denominators stay stable across completed run steps."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from ..schema import AnalysisStep, ValidationFinding


class CrossStepSourceStatusValidator:
    """Keep source-status denominators stable across completed run steps.

    A data-quality step may lock the number of source-consistent observed
    values for a measured concept.  Later descriptive/model steps are free to
    transform the value, but they must not silently redefine which rows were
    observed when they report the same concept on the same cohort.

    The gate is deliberately evidence-driven: it only compares explicit
    ``source_summary`` blocks against an earlier machine-readable
    ``missingness.source_status_counts`` block, and only when the category
    totals match.  Missing or ambiguous evidence is therefore skipped rather
    than guessed.
    """

    name = "cross_step_source_status"

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @classmethod
    def _is_valid_observed_label(cls, value: Any) -> bool:
        tokens = set(cls._normalise(value).split("_"))
        return (
            "invalid" not in tokens
            and "valid" in tokens
            and bool(tokens.intersection({"observed", "measured", "value", "level"}))
        )

    @classmethod
    def _status_role(cls, value: Any) -> Optional[str]:
        tokens = set(cls._normalise(value).split("_"))
        if cls._is_valid_observed_label(value):
            return "valid_observed"
        if "no" in tokens and tokens.intersection(
            {"source", "recorded", "observation"}
        ):
            return "no_source"
        if (
            tokens.intersection({"measured", "observed"})
            and "missing" in tokens
            and tokens.intersection({"summary", "value"})
        ):
            return "measured_summary_missing"
        if tokens.intersection({"contradictory", "inconsistent", "invalid"}):
            return "contradictory_invalid"
        return None

    @staticmethod
    def _as_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _flat_status_counts(
        cls, value: Any
    ) -> Optional[tuple[List[tuple[str, int]], Set[str]]]:
        """Parse one explicit four-role status mapping without guessing roles."""

        if not isinstance(value, dict):
            return None
        parsed = [
            (str(category), count)
            for category, raw_count in value.items()
            if (count := cls._as_count(raw_count)) is not None
        ]
        if not parsed or len(parsed) != len(value):
            return None
        present_roles = {
            role
            for category, _ in parsed
            if (role := cls._status_role(category)) is not None
        }
        return parsed, present_roles

    @classmethod
    def _declared_primary_source_summary(cls, summary: Dict[str, Any]) -> Optional[str]:
        """Return an explicitly declared primary summary column, if unique."""

        primary = summary.get("primary_exposure")
        if isinstance(primary, str) and primary.strip():
            return primary.strip()
        if isinstance(primary, dict):
            candidates = [
                str(primary.get(key) or "").strip()
                for key in ("column", "summary_variable", "source_summary")
            ]
            candidates = [value for value in candidates if value]
            if len(set(candidates)) == 1:
                return candidates[0]
        return None

    @classmethod
    def _prior_locks(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        locks: List[Dict[str, Any]] = []
        successful_statuses = {
            "ok",
            "complete",
            "completed",
            "repaired",
            "runner_repaired",
        }
        for record_index, record in enumerate(completed_step_records):
            status = str(record.get("status") or "").strip().lower()
            if status and status not in successful_statuses:
                continue
            summary = record.get("step_summary")
            if not isinstance(summary, dict):
                continue
            # A value-quality step may publish one exact top-level status map
            # bound to its explicit primary exposure.  This is the same closed
            # contract as the older nested ``missingness`` representation.
            source_summary = cls._declared_primary_source_summary(summary)
            flat = cls._flat_status_counts(summary.get("source_status_counts"))
            if source_summary and flat is not None:
                parsed, present_roles = flat
                required_roles = {
                    "valid_observed",
                    "no_source",
                    "measured_summary_missing",
                    "contradictory_invalid",
                }
                valid_counts = [
                    count
                    for category, count in parsed
                    if cls._is_valid_observed_label(category)
                ]
                if len(valid_counts) == 1 and required_roles <= present_roles:
                    role_counts = {
                        role: count
                        for category, count in parsed
                        if (role := cls._status_role(category)) is not None
                    }
                    locks.append(
                        {
                            "concept": cls._normalise(source_summary),
                            "source_summary": source_summary,
                            "scope": "step_summary.source_status_counts",
                            "total_n": sum(count for _, count in parsed),
                            "valid_observed_n": valid_counts[0],
                            "role_counts": role_counts,
                            "step_id": str(record.get("step_id") or "prior_step"),
                            "record_index": record_index,
                        }
                    )
            missingness = summary.get("missingness")
            if not isinstance(missingness, dict):
                continue
            source_counts = missingness.get("source_status_counts")
            if not isinstance(source_counts, dict):
                continue
            for scope, by_concept in source_counts.items():
                if not isinstance(by_concept, dict):
                    continue
                for concept, categories in by_concept.items():
                    if not isinstance(categories, dict):
                        continue
                    parsed = {
                        str(category): count
                        for category, raw_count in categories.items()
                        if (count := cls._as_count(raw_count)) is not None
                    }
                    valid_counts = [
                        count
                        for category, count in parsed.items()
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) != 1 or not parsed:
                        continue
                    locks.append(
                        {
                            "concept": cls._normalise(concept),
                            "source_summary": str(concept),
                            "scope": str(scope),
                            "total_n": sum(parsed.values()),
                            "valid_observed_n": valid_counts[0],
                            "step_id": str(record.get("step_id") or "prior_step"),
                            "record_index": record_index,
                        }
                    )
        return locks

    @classmethod
    def _current_status_blocks(cls, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        # Current descriptive steps may use ``source_status_schema`` for the
        # same four-category count contract.  Bind it only to an explicit
        # primary exposure; a free-standing map is intentionally ignored.
        source_summary = cls._declared_primary_source_summary(summary)
        flat = cls._flat_status_counts(summary.get("source_status_schema"))
        if source_summary and flat is not None:
            parsed, present_roles = flat
            valid_counts = [
                count
                for category, count in parsed
                if cls._is_valid_observed_label(category)
            ]
            required_roles = {
                "valid_observed",
                "no_source",
                "measured_summary_missing",
                "contradictory_invalid",
            }
            if len(valid_counts) == 1:
                role_counts = {
                    role: count
                    for category, count in parsed
                    if (role := cls._status_role(category)) is not None
                }
                blocks.append(
                    {
                        "concept": cls._normalise(source_summary),
                        "source_summary": source_summary,
                        "path": "source_status_schema",
                        "total_n": sum(count for _, count in parsed),
                        "valid_observed_n": valid_counts[0],
                        "role_counts": role_counts,
                        "missing_status_roles": sorted(required_roles - present_roles),
                    }
                )

        declarations: List[Dict[str, str]] = []

        def collect_declarations(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                summary_variable = value.get("summary_variable")
                if isinstance(summary_variable, str) and summary_variable.strip():
                    alias = cls._normalise(path[-1] if path else "")
                    alias = re.sub(r"_definition$", "", alias)
                    declarations.append(
                        {
                            "alias": alias,
                            "source_summary": summary_variable,
                            "base": re.sub(
                                r"_(?:first|max|min|mean|median)$",
                                "",
                                cls._normalise(summary_variable),
                            ),
                        }
                    )
                for key, child in value.items():
                    collect_declarations(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_declarations(child, (*path, str(index)))

        collect_declarations(summary)

        def declared_source_for(path: tuple[str, ...]) -> Optional[str]:
            hint = cls._normalise(path[-1] if path else "")
            hint = re.sub(r"_(?:measurement_)?status(?:_counts)?$", "", hint)
            exact = [
                declaration
                for declaration in declarations
                if declaration["alias"] == hint
            ]
            if len(exact) == 1:
                return exact[0]["source_summary"]
            semantic = [
                declaration
                for declaration in declarations
                if declaration["base"] == hint
                or declaration["base"].startswith(f"{hint}_")
                or hint.startswith(f"{declaration['base']}_")
            ]
            if len(semantic) == 1:
                return semantic[0]["source_summary"]
            return None

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                direct_counts = value.get("source_status_counts")
                source_columns = value.get("source_columns")
                if isinstance(direct_counts, dict) and isinstance(source_columns, list):
                    source_summary = next(
                        (
                            str(column)
                            for column in source_columns
                            if isinstance(column, str) and column.strip()
                        ),
                        None,
                    )
                    parsed_direct = [
                        (str(category), count)
                        for category, raw_count in direct_counts.items()
                        if (count := cls._as_count(raw_count)) is not None
                    ]
                    valid_counts = [
                        count
                        for category, count in parsed_direct
                        if cls._is_valid_observed_label(category)
                    ]
                    if source_summary and len(valid_counts) == 1 and parsed_direct:
                        present_roles = {
                            role
                            for category, _ in parsed_direct
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join((*path, "source_status_counts")),
                                "total_n": sum(count for _, count in parsed_direct),
                                "valid_observed_n": valid_counts[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                # Some reconciliation summaries store one concept per mapping
                # with ``counts`` and ``valid_observed_n`` rather than an
                # explicit source_columns list.  The concept key is still a
                # machine-readable source summary name, so preserve the same
                # four-category completeness and denominator lock.
                concept_counts = value.get("counts")
                concept_valid = cls._as_count(value.get("valid_observed_n"))
                if (
                    isinstance(concept_counts, dict)
                    and concept_valid is not None
                    and path
                ):
                    parsed_concept = [
                        (str(category), count)
                        for category, raw_count in concept_counts.items()
                        if (count := cls._as_count(raw_count)) is not None
                    ]
                    valid_counts = [
                        count
                        for category, count in parsed_concept
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) == 1 and parsed_concept:
                        present_roles = {
                            role
                            for category, _ in parsed_concept
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        source_summary = str(path[-1])
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join((*path, "counts")),
                                "total_n": sum(count for _, count in parsed_concept),
                                "valid_observed_n": valid_counts[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                if path and any(
                    "source_status_count" in cls._normalise(segment) for segment in path
                ):
                    parsed_nested = [
                        (str(category), count)
                        for category, raw in value.items()
                        if isinstance(raw, dict)
                        and (count := cls._as_count(raw.get("count", raw.get("n"))))
                        is not None
                    ]
                    valid_nested = [
                        count
                        for category, count in parsed_nested
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_nested) == 1 and parsed_nested:
                        present_roles = {
                            role
                            for category, _ in parsed_nested
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        source_summary = str(path[-1])
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(path),
                                "total_n": sum(count for _, count in parsed_nested),
                                "valid_observed_n": valid_nested[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                source_summary = value.get("source_summary")
                rows = value.get("measurement_status_counts")
                if not isinstance(rows, list):
                    rows = value.get("counts")
                if isinstance(source_summary, str) and isinstance(rows, list):
                    parsed: List[tuple[str, int]] = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        count = cls._as_count(row.get("count", row.get("n")))
                        category = row.get("category", row.get("status"))
                        if count is not None and category is not None:
                            parsed.append((str(category), count))
                    valid_counts = [
                        count
                        for category, count in parsed
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) == 1 and parsed:
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(path) or "step_summary",
                                "total_n": sum(count for _, count in parsed),
                                "valid_observed_n": valid_counts[0],
                            }
                        )
                # Newer descriptive summaries may expose the same contract as
                # scalar counts under ``missingness_and_measurement_status``
                # instead of a list of category rows.  Bind the status block to
                # an explicit nearby ``summary_variable`` declaration; never
                # guess from a human label alone.
                scalar_valid = cls._as_count(value.get("observed_valid_summary_n"))
                scalar_total = cls._as_count(value.get("denominator_n"))
                if scalar_valid is not None and scalar_total is not None:
                    scalar_source = value.get("source_summary") or value.get(
                        "summary_variable"
                    )
                    if not isinstance(scalar_source, str) or not scalar_source.strip():
                        scalar_source = declared_source_for(path)
                    if isinstance(scalar_source, str) and scalar_source.strip():
                        blocks.append(
                            {
                                "concept": cls._normalise(scalar_source),
                                "source_summary": scalar_source,
                                "path": ".".join(path) or "step_summary",
                                "total_n": scalar_total,
                                "valid_observed_n": scalar_valid,
                            }
                        )
                for key, child in value.items():
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return blocks

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        locks = self._prior_locks(completed_step_records)
        if not locks:
            return []

        findings: List[ValidationFinding] = []
        compared: Set[tuple[str, int, int]] = set()
        for current in self._current_status_blocks(step_summary):
            candidates = [
                lock
                for lock in locks
                if lock["concept"] == current["concept"]
                and lock["total_n"] == current["total_n"]
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda lock: (
                    "analytic" not in self._normalise(lock["scope"]),
                    -int(lock["record_index"]),
                )
            )
            expected = candidates[0]
            comparison_key = (
                current["concept"],
                current["total_n"],
                current["valid_observed_n"],
            )
            if comparison_key in compared:
                continue
            compared.add(comparison_key)
            missing_status_roles = current.get("missing_status_roles") or []
            if missing_status_roles:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Incomplete source-status schema for "
                            f"{current['source_summary']} in step {step.step_id}: "
                            f"missing categories {missing_status_roles}. Report "
                            "all four source-status categories explicitly, using "
                            "zero counts for supported zero-frequency strata "
                            "rather than omitting their machine-summary keys."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "summary_path": current["path"],
                            "source_summary": current["source_summary"],
                            "cohort_n": current["total_n"],
                            "missing_status_roles": missing_status_roles,
                            "expected_from_step": expected["step_id"],
                        },
                    )
                )
            expected_role_counts = expected.get("role_counts")
            current_role_counts = current.get("role_counts")
            if (
                isinstance(expected_role_counts, dict)
                and isinstance(current_role_counts, dict)
                and not missing_status_roles
                and current_role_counts != expected_role_counts
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Source-status category drift for "
                            f"{current['source_summary']}: step {step.step_id} "
                            "reallocated rows among observed, no-source, "
                            "measured-summary-missing, or contradictory states "
                            f"relative to completed step {expected['step_id']}. "
                            "Preserve the earlier closed source-status mapping."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "summary_path": current["path"],
                            "source_summary": current["source_summary"],
                            "cohort_n": current["total_n"],
                            "reported_status_counts": current_role_counts,
                            "expected_status_counts": expected_role_counts,
                            "expected_from_step": expected["step_id"],
                            "expected_scope": expected["scope"],
                        },
                    )
                )
                continue
            if current["valid_observed_n"] == expected["valid_observed_n"]:
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Source-status denominator drift for "
                        f"{current['source_summary']}: step {step.step_id} reports "
                        f"{current['valid_observed_n']} valid observed rows of "
                        f"{current['total_n']}, but completed step "
                        f"{expected['step_id']} locked "
                        f"{expected['valid_observed_n']} for the same concept and "
                        "cohort. Preserve the earlier source-status, variable-type, "
                        "and retain/flag range semantics instead of redefining "
                        "validity in this step."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "summary_path": current["path"],
                        "source_summary": current["source_summary"],
                        "cohort_n": current["total_n"],
                        "reported_valid_observed_n": current["valid_observed_n"],
                        "expected_valid_observed_n": expected["valid_observed_n"],
                        "expected_from_step": expected["step_id"],
                        "expected_scope": expected["scope"],
                    },
                )
            )
        return findings


__all__ = ["CrossStepSourceStatusValidator"]
