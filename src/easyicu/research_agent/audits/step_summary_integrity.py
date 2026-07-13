"""Host-authoritative integrity checks for generated step summaries.

This module validates input provenance, locked-cohort measurement-count
consistency, and explicit tabular subset reconciliations. It never chooses a
cohort, exposure, outcome, estimator, or scientific method.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..icu_rules import companion_count_column_for_measured
from ..schema import AnalysisStep, ValidationFinding


class StepSummaryIntegrityValidator:
    """Verify exact inputs, measurement provenance, and subset reconciliations.

    Generated code may describe one resolved artifact in several places in a
    step summary.  Those declarations are untrusted until they agree with one
    another and, for a checked subset reconciliation, with the host-resolved
    input tables. The gate is intentionally structural: it validates an
    explicit ``input_bindings`` list and claims of a checked subset
    reconciliation only when the block directly names typed artifacts or has
    a same-scope typed artifact declaration. Unrelated QC blocks that happen to
    use the word ``checked`` are outside this contract.
    """

    name = "step_summary_integrity"
    _ARTIFACT_FIELDS = (
        "artifact",
        "input_artifact",
        "reference_artifact",
        "parent_artifact",
        "upstream_artifact",
        "subset_artifact",
        "comparison_artifact",
    )
    _REFERENCE_ARTIFACT_FIELDS = (
        "reference_artifact",
        "parent_artifact",
        "upstream_artifact",
    )
    _SUBSET_ARTIFACT_FIELDS = ("subset_artifact", "comparison_artifact")
    _ROW_COUNT_FIELDS = ("row_count", "n_rows", "n")
    _TABULAR_SUFFIXES = {
        ".csv",
        ".feather",
        ".parquet",
        ".pq",
        ".tsv",
        ".xls",
        ".xlsx",
    }
    _AUXILIARY_OUTPUT_KINDS = {"log"}
    _PRESENTATION_OUTPUT_KINDS = {"chart", "fig", "figure", "heatmap", "plot"}
    _PRESENTATION_FILE_SUFFIXES = (".pdf", ".png", ".svg", ".tif", ".tiff")
    _PRESENTATION_BARE_SUFFIX_RE = re.compile(r"(?:^|_)(?:chart|figure|heatmap|plot)$")

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @staticmethod
    def _as_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _row_count_claim(cls, value: Mapping[str, Any]) -> tuple[Optional[int], str]:
        for field in cls._ROW_COUNT_FIELDS:
            if field not in value:
                continue
            raw = value.get(field)
            if raw is None:
                return None, field
            return cls._as_count(raw), field
        return None, ""

    @staticmethod
    def _table_row_count(path: Path) -> int:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            try:
                import pyarrow.parquet as pq

                return int(pq.ParquetFile(path).metadata.num_rows)
            except ImportError:
                return int(len(pd.read_parquet(path)))
        if suffix in {".csv", ".tsv"}:
            separator = "\t" if suffix == ".tsv" else ","
            header = pd.read_csv(path, sep=separator, nrows=0)
            if len(header.columns) == 0:
                return 0
            return int(
                len(pd.read_csv(path, sep=separator, usecols=[header.columns[0]]))
            )
        if suffix == ".feather":
            return int(len(pd.read_feather(path)))
        if suffix in {".xlsx", ".xls"}:
            return int(len(pd.read_excel(path)))
        raise ValueError(f"unsupported input table suffix: {suffix or '(none)'}")

    @classmethod
    def _is_tabular_binding(cls, binding: Mapping[str, Any]) -> bool:
        path = Path(str(binding.get("absolute_path") or ""))
        return path.suffix.lower() in cls._TABULAR_SUFFIXES

    @staticmethod
    def _read_table_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
        wanted = list(dict.fromkeys(str(column) for column in columns))
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path, columns=wanted)
        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(
                path,
                sep="\t" if suffix == ".tsv" else ",",
                usecols=wanted,
            )
        if suffix == ".feather":
            return pd.read_feather(path, columns=wanted)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, usecols=wanted)
        raise ValueError(f"unsupported input table suffix: {suffix or '(none)'}")

    @staticmethod
    def _table_column_names(path: Path) -> List[str]:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            try:
                import pyarrow.parquet as pq

                return [str(name) for name in pq.ParquetFile(path).schema.names]
            except ImportError:
                return [str(name) for name in pd.read_parquet(path).columns]
        if suffix in {".csv", ".tsv"}:
            separator = "\t" if suffix == ".tsv" else ","
            return [
                str(name) for name in pd.read_csv(path, sep=separator, nrows=0).columns
            ]
        if suffix == ".feather":
            return [str(name) for name in pd.read_feather(path).columns]
        if suffix in {".xlsx", ".xls"}:
            return [str(name) for name in pd.read_excel(path, nrows=0).columns]
        raise ValueError(f"unsupported input table suffix: {suffix or '(none)'}")

    @classmethod
    def _checked_reconciliation_blocks(
        cls, summary: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                path_name = cls._normalise(path[-1] if path else "")
                path_tokens = set(path_name.split("_"))
                if cls._normalise(
                    value.get("status")
                ) == "checked" and path_tokens.intersection(
                    {"subset", "reconciliation"}
                ):
                    blocks.append(
                        {
                            "path": ".".join(path),
                            "path_name": path_name,
                            "value": value,
                        }
                    )
                for key, child in value.items():
                    if not path and key == "input_bindings":
                        continue
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return blocks

    @classmethod
    def _artifact_declarations(cls, summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
        declarations: List[Dict[str, Any]] = []

        def scope_name(path: tuple[str, ...]) -> str:
            name = cls._normalise(path[-1] if path else "")
            return re.sub(
                r"_(?:input|artifact|subset|reconciliation|verification)$", "", name
            )

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                direct = [
                    (field, value.get(field))
                    for field in cls._ARTIFACT_FIELDS
                    if isinstance(value.get(field), str)
                ]
                for field, artifact in direct:
                    count, count_field = cls._row_count_claim(value)
                    declarations.append(
                        {
                            "artifact": artifact,
                            "artifact_field": field,
                            "path": ".".join(path) or "step_summary",
                            "scope": scope_name(path),
                            "loaded": (
                                value.get("loaded")
                                if isinstance(value.get("loaded"), bool)
                                else None
                            ),
                            "row_count": count if len(direct) == 1 else None,
                            "row_count_field": count_field if len(direct) == 1 else "",
                        }
                    )
                for key, child in value.items():
                    if not path and key == "input_bindings":
                        continue
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return declarations

    @classmethod
    def _named_artifact(
        cls, block: Mapping[str, Any], fields: Sequence[str]
    ) -> Optional[str]:
        values = [
            str(block[field])
            for field in fields
            if isinstance(block.get(field), str) and str(block[field]).strip()
        ]
        return values[0] if len(set(values)) == 1 else None

    @staticmethod
    def _string_columns(value: Any) -> Optional[List[str]]:
        if not isinstance(value, list) or not value:
            return None
        columns = [str(item).strip() for item in value if isinstance(item, str)]
        if len(columns) != len(value) or any(not column for column in columns):
            return None
        return list(dict.fromkeys(columns))

    @classmethod
    def _host_reconciliation_finding(
        cls,
        *,
        step: AnalysisStep,
        block_path: str,
        reference_artifact: str,
        subset_artifact: str,
        key_columns: Sequence[str],
        value_columns: Sequence[str],
        resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    ) -> Optional[ValidationFinding]:
        try:
            reference_path = Path(
                str(resolved_input_bindings[reference_artifact]["absolute_path"])
            )
            subset_path = Path(
                str(resolved_input_bindings[subset_artifact]["absolute_path"])
            )
            reference_column_names = set(cls._table_column_names(reference_path))
            subset_column_names = set(cls._table_column_names(subset_path))
            key_column_set = set(key_columns)
            shared_value_columns = sorted(
                (reference_column_names & subset_column_names) - key_column_set
            )
            declared_value_columns = set(value_columns)
            omitted_value_columns = sorted(
                set(shared_value_columns) - declared_value_columns
            )
            nonshared_value_columns = sorted(
                declared_value_columns - set(shared_value_columns)
            )
            if omitted_value_columns or nonshared_value_columns:
                return ValidationFinding(
                    validator=cls.name,
                    severity="error",
                    message=(
                        f"Host reconciliation scope is incomplete for step "
                        f"{step.step_id}: {block_path} must verify every shared "
                        "non-key column between the exact reference and subset "
                        "artifacts."
                    ),
                    detail={
                        "issue": "checked_reconciliation_value_scope_incomplete",
                        "step_id": step.step_id,
                        "summary_path": block_path,
                        "reference_artifact": reference_artifact,
                        "subset_artifact": subset_artifact,
                        "key_columns": list(key_columns),
                        "omitted_value_columns": omitted_value_columns,
                        "nonshared_value_columns": nonshared_value_columns,
                    },
                )
            columns = [*key_columns, *value_columns]
            reference = cls._read_table_columns(reference_path, columns)
            subset = cls._read_table_columns(subset_path, columns)
            if reference[list(key_columns)].isna().any(axis=None) or subset[
                list(key_columns)
            ].isna().any(axis=None):
                raise ValueError("key columns contain missing values")
            reference_duplicate_n = int(
                reference.duplicated(list(key_columns), keep=False).sum()
            )
            subset_duplicate_n = int(
                subset.duplicated(list(key_columns), keep=False).sum()
            )
            if reference_duplicate_n or subset_duplicate_n:
                raise ValueError(
                    "key columns are not unique "
                    f"(reference duplicate rows={reference_duplicate_n}, "
                    f"subset duplicate rows={subset_duplicate_n})"
                )

            reference_indexed = reference.set_index(list(key_columns))
            subset_indexed = subset.set_index(list(key_columns))
            missing_key_n = int(
                len(subset_indexed.index.difference(reference_indexed.index))
            )
            if missing_key_n:
                return ValidationFinding(
                    validator=cls.name,
                    severity="error",
                    message=(
                        f"Host reconciliation failed for step {step.step_id}: "
                        f"{block_path} has {missing_key_n} subset key(s) absent "
                        "from the reference artifact."
                    ),
                    detail={
                        "issue": "checked_reconciliation_host_key_mismatch",
                        "step_id": step.step_id,
                        "summary_path": block_path,
                        "reference_artifact": reference_artifact,
                        "subset_artifact": subset_artifact,
                        "key_columns": list(key_columns),
                        "missing_key_n": missing_key_n,
                    },
                )

            aligned_reference = reference_indexed.reindex(subset_indexed.index)
            mismatch_rows = pd.Series(False, index=range(len(subset_indexed)))
            mismatch_cell_n = 0
            for column in value_columns:
                subset_values = subset_indexed[column].reset_index(drop=True)
                reference_values = aligned_reference[column].reset_index(drop=True)
                equal = subset_values.eq(reference_values) | (
                    subset_values.isna() & reference_values.isna()
                )
                unequal = ~equal.fillna(False)
                mismatch_cell_n += int(unequal.sum())
                mismatch_rows |= unequal
            mismatch_row_n = int(mismatch_rows.sum())
            if mismatch_cell_n:
                return ValidationFinding(
                    validator=cls.name,
                    severity="error",
                    message=(
                        f"Host reconciliation failed for step {step.step_id}: "
                        f"{block_path} claims zero value mismatches, but exact "
                        f"resolved inputs disagree in {mismatch_cell_n} checked "
                        "cell(s)."
                    ),
                    detail={
                        "issue": "checked_reconciliation_host_value_mismatch",
                        "step_id": step.step_id,
                        "summary_path": block_path,
                        "reference_artifact": reference_artifact,
                        "subset_artifact": subset_artifact,
                        "key_columns": list(key_columns),
                        "value_columns_checked": list(value_columns),
                        "value_mismatch_cell_n": mismatch_cell_n,
                        "value_mismatch_row_n": mismatch_row_n,
                    },
                )
        except Exception as exc:
            return ValidationFinding(
                validator=cls.name,
                severity="error",
                message=(
                    f"Could not host-verify checked reconciliation "
                    f"{block_path} in step {step.step_id}: {exc}"
                ),
                detail={
                    "issue": "checked_reconciliation_host_verification_failed",
                    "step_id": step.step_id,
                    "summary_path": block_path,
                    "reference_artifact": reference_artifact,
                    "subset_artifact": subset_artifact,
                },
            )
        return None

    @classmethod
    def _planned_measurement_pairs(cls, step: AnalysisStep) -> Dict[str, str]:
        pairs: Dict[str, str] = {}
        for value in step.inputs or []:
            measured_column = str(value).strip()
            if ":" in measured_column:
                continue
            count_column = companion_count_column_for_measured(measured_column)
            if count_column is not None:
                pairs[measured_column] = count_column
        return dict(sorted(pairs.items()))

    @classmethod
    def _is_result_step(cls, step: AnalysisStep) -> bool:
        """Return whether a step owns a scientific result product.

        Figures and typed bookkeeping sidecars render or describe an existing
        result; they do not make the step a result owner.  Any other typed or
        bare output is conservatively treated as a result so a new scientific
        output kind cannot silently bypass integrity validation.
        """

        for raw_output in step.expected_outputs or []:
            output = str(raw_output).strip()
            if not output or cls._is_presentation_output(output):
                continue
            kind, separator, name = output.partition(":")
            if (
                separator
                and kind.strip().lower() in cls._AUXILIARY_OUTPUT_KINDS
                and name.strip()
            ):
                continue
            return True
        return False

    @classmethod
    def _is_presentation_output(cls, raw_output: Any) -> bool:
        """Recognise only structural presentation products.

        Bare words containing ``figure`` or ``plot`` are deliberately not
        enough: names such as ``figure_source_data`` and
        ``model_plot_statistics`` own scientific data and must remain subject
        to result integrity checks.
        """

        output = str(raw_output or "").strip().lower()
        if not output:
            return False
        kind, separator, name = output.partition(":")
        if separator:
            return kind.strip() in cls._PRESENTATION_OUTPUT_KINDS and bool(name.strip())
        return bool(cls._PRESENTATION_BARE_SUFFIX_RE.search(output)) or output.endswith(
            cls._PRESENTATION_FILE_SUFFIXES
        )

    @classmethod
    def _provenance_error(
        cls,
        *,
        step: AnalysisStep,
        issue: str,
        message: str,
        measured_column: Optional[str] = None,
        **detail: Any,
    ) -> ValidationFinding:
        return ValidationFinding(
            validator=cls.name,
            severity="error",
            message=message,
            detail={
                "issue": issue,
                "step_id": step.step_id,
                **(
                    {"measured_column": measured_column}
                    if measured_column is not None
                    else {}
                ),
                **detail,
            },
        )

    @staticmethod
    def _coerce_semantic_numeric(
        series: pd.Series,
        *,
        allow_boolean: bool,
    ) -> tuple[pd.Series, pd.Series]:
        """Coerce numeric values without laundering semantic dtypes."""

        def scalar_type_allowed(value: Any) -> bool:
            try:
                if bool(pd.isna(value)):
                    return True
            except (TypeError, ValueError):
                pass
            if pd.api.types.is_bool(value):
                return allow_boolean
            if isinstance(
                value, (date, datetime, timedelta, pd.Timestamp, pd.Timedelta)
            ):
                return False
            if pd.api.types.is_datetime64_dtype(type(value)):
                return False
            if pd.api.types.is_timedelta64_dtype(type(value)):
                return False
            return True

        semantic_valid = series.map(scalar_type_allowed).astype(bool)
        if pd.api.types.is_datetime64_any_dtype(
            series.dtype
        ) or pd.api.types.is_timedelta64_dtype(series.dtype):
            semantic_valid[:] = False
        elif pd.api.types.is_bool_dtype(series.dtype) and not allow_boolean:
            semantic_valid[:] = False
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric, semantic_valid

    @classmethod
    def _measurement_host_state(
        cls,
        *,
        step: AnalysisStep,
        cohort_path: Optional[Path],
    ) -> tuple[
        Dict[str, str],
        Dict[str, Dict[str, Any]],
        List[ValidationFinding],
    ]:
        """Replay immutable measurement facts independently of agent output."""

        measurement_pairs = cls._planned_measurement_pairs(step)
        measured_columns = list(measurement_pairs)
        if not measured_columns or not cls._is_result_step(step):
            return measurement_pairs, {}, []
        if cohort_path is None:
            return (
                measurement_pairs,
                {},
                [
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_host_source_missing",
                        message=(
                            f"Step {step.step_id} requires host verification of "
                            "planned measurement flags, but the locked "
                            "COHORT_PARQUET path was not supplied."
                        ),
                        planned_measured_columns=measured_columns,
                    )
                ],
            )

        locked_cohort = Path(cohort_path)
        try:
            cohort_columns = set(cls._table_column_names(locked_cohort))
        except Exception as exc:
            return (
                measurement_pairs,
                {},
                [
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_host_source_unreadable",
                        message=(
                            f"Could not inspect locked COHORT_PARQUET for step "
                            f"{step.step_id}: {exc}"
                        ),
                        planned_measured_columns=measured_columns,
                    )
                ],
            )

        replays: Dict[str, Dict[str, Any]] = {}
        findings: List[ValidationFinding] = []
        for measured_column, expected_count_column in measurement_pairs.items():
            replay: Dict[str, Any] = {
                "expected_count_column": expected_count_column,
                "resolved_count_column": None,
                "count_available": False,
            }
            replays[measured_column] = replay
            if measured_column not in cohort_columns:
                replay["state"] = "measured_column_missing"
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_measured_column_missing",
                        message=(
                            f"Locked COHORT_PARQUET lacks planned measurement flag "
                            f"{measured_column!r} for step {step.step_id}."
                        ),
                        measured_column=measured_column,
                    )
                )
                continue

            casefold_count_columns = sorted(
                column
                for column in cohort_columns
                if column.casefold() == expected_count_column.casefold()
            )
            count_ambiguous = False
            if expected_count_column in cohort_columns:
                resolved_count_column: Optional[str] = expected_count_column
            elif len(casefold_count_columns) == 1:
                resolved_count_column = casefold_count_columns[0]
            elif len(casefold_count_columns) > 1:
                resolved_count_column = None
                count_ambiguous = True
                replay["state"] = "count_column_ambiguous"
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_count_column_ambiguous",
                        message=(
                            f"Locked COHORT_PARQUET contains multiple case variants "
                            f"for the companion of {measured_column!r}; provenance "
                            "pairing is ambiguous."
                        ),
                        measured_column=measured_column,
                        expected_count_column=expected_count_column,
                        matching_count_columns=casefold_count_columns,
                    )
                )
            else:
                resolved_count_column = None

            replay["resolved_count_column"] = resolved_count_column
            replay["count_available"] = resolved_count_column is not None
            columns = [measured_column]
            if resolved_count_column is not None:
                columns.append(resolved_count_column)
            try:
                frame = cls._read_table_columns(locked_cohort, columns)
                measured, measured_semantic_valid = cls._coerce_semantic_numeric(
                    frame[measured_column],
                    allow_boolean=True,
                )
                valid_measured = measured_semantic_valid & measured.isin([0, 1])
                invalid_measured_n = int((~valid_measured).sum())
                replay["invalid_measured_n"] = invalid_measured_n
                if invalid_measured_n:
                    findings.append(
                        cls._provenance_error(
                            step=step,
                            issue="measurement_provenance_invalid_measured_values",
                            message=(
                                f"Locked-cohort measurement flag "
                                f"{measured_column!r} contains "
                                f"{invalid_measured_n} non-binary, missing, or "
                                "semantically invalid value(s)."
                            ),
                            measured_column=measured_column,
                            invalid_measured_n=invalid_measured_n,
                        )
                    )
                if count_ambiguous or resolved_count_column is None:
                    replay["state"] = replay.get("state") or "count_unavailable"
                    continue

                count, count_semantic_valid = cls._coerce_semantic_numeric(
                    frame[resolved_count_column],
                    allow_boolean=False,
                )
                valid_count = (
                    count_semantic_valid
                    & count.notna()
                    & count.ge(0)
                    & count.lt(float("inf"))
                    & count.mod(1).eq(0)
                )
                valid_pair = valid_measured & valid_count
                host = {
                    "comparison_n": int(valid_pair.sum()),
                    "invalid_pair_n": int((~valid_pair).sum()),
                    "discordant_n": int(
                        (
                            measured[valid_pair].astype(bool) != count[valid_pair].gt(0)
                        ).sum()
                    ),
                }
                replay.update({"state": "checked", "host": host})
                if host["invalid_pair_n"]:
                    findings.append(
                        cls._provenance_error(
                            step=step,
                            issue="measurement_provenance_invalid_pairs",
                            message=(
                                f"Locked-cohort measurement provenance for "
                                f"{measured_column!r} contains "
                                f"{host['invalid_pair_n']} row(s) with a "
                                "non-binary flag or invalid count."
                            ),
                            measured_column=measured_column,
                            expected_count_column=resolved_count_column,
                            invalid_pair_n=host["invalid_pair_n"],
                        )
                    )
                if host["discordant_n"]:
                    findings.append(
                        cls._provenance_error(
                            step=step,
                            issue=("measurement_provenance_count_flag_discordance"),
                            message=(
                                f"Locked-cohort measurement provenance for "
                                f"{measured_column!r} has "
                                f"{host['discordant_n']} row(s) where the "
                                "measured flag disagrees with (count > 0)."
                            ),
                            measured_column=measured_column,
                            expected_count_column=resolved_count_column,
                            discordant_n=host["discordant_n"],
                        )
                    )
            except Exception as exc:
                replay["state"] = "host_replay_failed"
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_host_replay_failed",
                        message=(
                            f"Could not replay locked-cohort measurement provenance "
                            f"for {measured_column!r} in step {step.step_id}: {exc}"
                        ),
                        measured_column=measured_column,
                    )
                )
        return measurement_pairs, replays, findings

    @classmethod
    def audit_locked_measurement_data_quality(
        cls,
        *,
        step: AnalysisStep,
        cohort_path: Optional[Path],
    ) -> List[ValidationFinding]:
        """Return only immutable host-data findings for pre-execution gating."""

        _, _, findings = cls._measurement_host_state(
            step=step,
            cohort_path=cohort_path,
        )
        return findings

    @classmethod
    def _measurement_provenance_findings(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        cohort_path: Optional[Path],
    ) -> List[ValidationFinding]:
        measurement_pairs, replays, findings = cls._measurement_host_state(
            step=step,
            cohort_path=cohort_path,
        )
        measured_columns = list(measurement_pairs)
        if not measured_columns or not cls._is_result_step(step):
            return []
        if not replays:
            return findings

        audit = step_summary.get("measurement_provenance_audit")
        checks = audit.get("checks") if isinstance(audit, Mapping) else None
        if not isinstance(audit, Mapping) or audit.get("source") != "COHORT_PARQUET":
            findings.append(
                cls._provenance_error(
                    step=step,
                    issue="measurement_provenance_source_invalid",
                    message=(
                        f"Step {step.step_id} must declare "
                        "measurement_provenance_audit.source='COHORT_PARQUET'; "
                        "agent-selected subsets are not provenance authority."
                    ),
                    reported_source=(
                        audit.get("source") if isinstance(audit, Mapping) else None
                    ),
                    planned_measured_columns=measured_columns,
                )
            )
            return findings
        if not isinstance(checks, list):
            findings.append(
                cls._provenance_error(
                    step=step,
                    issue="measurement_provenance_checks_missing",
                    message=(
                        f"Step {step.step_id} lacks a machine-readable "
                        "measurement_provenance_audit.checks list."
                    ),
                    planned_measured_columns=measured_columns,
                )
            )
            return findings

        checks_by_column: Dict[str, Mapping[str, Any]] = {}
        for index, raw in enumerate(checks):
            path = f"measurement_provenance_audit.checks.{index}"
            if not isinstance(raw, Mapping):
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_check_invalid",
                        message=f"{path} must be a machine-readable mapping.",
                        summary_path=path,
                    )
                )
                continue
            measured_column = raw.get("measured_column")
            if measured_column not in measured_columns:
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_check_unplanned",
                        message=(
                            f"{path} names unplanned measurement flag "
                            f"{measured_column!r}."
                        ),
                        measured_column=(
                            str(measured_column)
                            if isinstance(measured_column, str)
                            else None
                        ),
                        summary_path=path,
                        planned_measured_columns=measured_columns,
                    )
                )
                continue
            measured_column = str(measured_column)
            if measured_column in checks_by_column:
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_check_duplicate",
                        message=(
                            f"Step {step.step_id} declares more than one provenance "
                            f"check for {measured_column!r}."
                        ),
                        measured_column=measured_column,
                        summary_path=path,
                    )
                )
                continue
            checks_by_column[measured_column] = raw

        for measured_column, expected_count_column in measurement_pairs.items():
            replay = replays.get(measured_column, {})
            check = checks_by_column.get(measured_column)
            if check is None:
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_check_missing",
                        message=(
                            f"Step {step.step_id} lacks a locked-cohort provenance "
                            f"check for planned flag {measured_column!r}."
                        ),
                        measured_column=measured_column,
                        expected_count_column=expected_count_column,
                    )
                )
                continue
            if replay.get("state") in {
                "count_column_ambiguous",
                "host_replay_failed",
                "measured_column_missing",
            }:
                continue

            resolved_count_column = replay.get("resolved_count_column")
            count_available = bool(replay.get("count_available"))
            reported_count_column = (
                str(resolved_count_column) if count_available else expected_count_column
            )
            expected_status = "checked" if count_available else "unavailable"
            status = cls._normalise(check.get("status"))
            invalid_fields: List[str] = []
            if check.get("count_column") != reported_count_column:
                invalid_fields.append("count_column")
            if check.get("role") != "audit_only":
                invalid_fields.append("role")
            if status != expected_status:
                invalid_fields.append("status")
            if expected_status == "unavailable":
                reason = check.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    invalid_fields.append("reason")
                for field in ("comparison_n", "invalid_pair_n", "discordant_n"):
                    if field not in check or check.get(field) is not None:
                        invalid_fields.append(field)
            else:
                for field in ("comparison_n", "invalid_pair_n", "discordant_n"):
                    if cls._as_count(check.get(field)) is None:
                        invalid_fields.append(field)
            if invalid_fields:
                issue = (
                    "measurement_provenance_unavailable_contradicted"
                    if count_available and status == "unavailable"
                    else "measurement_provenance_check_invalid"
                )
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue=issue,
                        message=(
                            f"Measurement provenance for {measured_column!r} in "
                            f"step {step.step_id} has invalid fields: "
                            f"{', '.join(invalid_fields)}."
                        ),
                        measured_column=measured_column,
                        expected_count_column=reported_count_column,
                        expected_status=expected_status,
                        invalid_fields=invalid_fields,
                    )
                )
                continue
            if not count_available:
                continue

            reported = {
                field: cls._as_count(check.get(field))
                for field in ("comparison_n", "invalid_pair_n", "discordant_n")
            }
            host = replay.get("host")
            if reported != host:
                findings.append(
                    cls._provenance_error(
                        step=step,
                        issue="measurement_provenance_host_count_mismatch",
                        message=(
                            f"Host replay contradicts measurement provenance for "
                            f"{measured_column!r} in step {step.step_id}."
                        ),
                        measured_column=measured_column,
                        expected_count_column=reported_count_column,
                        reported=reported,
                        host=host,
                    )
                )
        return findings

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        resolved_input_bindings: Mapping[str, Mapping[str, Any]],
        cohort_path: Optional[Path] = None,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        host_bindings = {
            str(key): value
            for key, value in resolved_input_bindings.items()
            if isinstance(value, Mapping)
        }
        findings.extend(
            self._measurement_provenance_findings(
                step=step,
                step_summary=step_summary,
                cohort_path=cohort_path,
            )
        )
        summary_bindings: Dict[str, Dict[str, Any]] = {}
        raw_bindings = step_summary.get("input_bindings")
        if isinstance(raw_bindings, list):
            for index, raw in enumerate(raw_bindings):
                path = f"input_bindings.{index}"
                if not isinstance(raw, dict):
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"Input binding {path} in step {step.step_id} "
                                "must be a machine-readable mapping."
                            ),
                            detail={
                                "issue": "input_binding_invalid",
                                "step_id": step.step_id,
                                "summary_path": path,
                            },
                        )
                    )
                    continue
                input_key = raw.get("input_key")
                if not isinstance(input_key, str) or input_key not in host_bindings:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"Input binding {path} in step {step.step_id} "
                                "does not name an exact host-resolved input_key."
                            ),
                            detail={
                                "issue": "input_binding_key_unresolved",
                                "step_id": step.step_id,
                                "summary_path": path,
                                "input_key": input_key,
                                "resolved_input_keys": sorted(host_bindings),
                            },
                        )
                    )
                    continue
                if input_key in summary_bindings:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"Exact input_key {input_key!r} is declared more "
                                f"than once in step {step.step_id}."
                            ),
                            detail={
                                "issue": "input_binding_duplicate",
                                "step_id": step.step_id,
                                "summary_path": path,
                                "input_key": input_key,
                            },
                        )
                    )
                    continue
                summary_bindings[input_key] = raw
                loaded = raw.get("loaded")
                row_count, row_count_field = self._row_count_claim(raw)
                tabular_binding = self._is_tabular_binding(host_bindings[input_key])
                invalid_fields: List[str] = []
                if not isinstance(loaded, bool):
                    invalid_fields.append("loaded")
                if (
                    tabular_binding
                    and loaded is True
                    and (not row_count_field or row_count is None)
                ):
                    invalid_fields.append("row_count")
                if (
                    tabular_binding
                    and loaded is False
                    and row_count is not None
                    and row_count != 0
                ):
                    invalid_fields.append(row_count_field or "row_count")
                if invalid_fields:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"Input binding {input_key!r} in step "
                                f"{step.step_id} lacks a coherent loaded/row-count "
                                "declaration."
                            ),
                            detail={
                                "issue": "input_binding_load_contract_invalid",
                                "step_id": step.step_id,
                                "summary_path": path,
                                "input_key": input_key,
                                "invalid_fields": invalid_fields,
                            },
                        )
                    )
                if tabular_binding and loaded is True and row_count is not None:
                    try:
                        host_count = self._table_row_count(
                            Path(str(host_bindings[input_key]["absolute_path"]))
                        )
                    except Exception as exc:
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="error",
                                message=(
                                    f"Could not verify row count for exact input "
                                    f"{input_key!r} in step {step.step_id}: {exc}"
                                ),
                                detail={
                                    "issue": "input_binding_host_verification_failed",
                                    "step_id": step.step_id,
                                    "summary_path": path,
                                    "input_key": input_key,
                                },
                            )
                        )
                    else:
                        if row_count != host_count:
                            findings.append(
                                ValidationFinding(
                                    validator=self.name,
                                    severity="error",
                                    message=(
                                        f"Input binding row-count mismatch in step "
                                        f"{step.step_id}: {input_key!r} reports "
                                        f"{row_count}, but the exact host-resolved "
                                        f"artifact has {host_count} rows."
                                    ),
                                    detail={
                                        "issue": "input_binding_row_count_mismatch",
                                        "step_id": step.step_id,
                                        "summary_path": path,
                                        "input_key": input_key,
                                        "reported_row_count": row_count,
                                        "host_row_count": host_count,
                                    },
                                )
                            )
                for metadata_field in ("evidence_id", "sha256"):
                    reported = raw.get(metadata_field)
                    expected = host_bindings[input_key].get(metadata_field)
                    if reported is not None and str(reported) != str(expected):
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="error",
                                message=(
                                    f"Input binding {metadata_field} mismatch for "
                                    f"{input_key!r} in step {step.step_id}."
                                ),
                                detail={
                                    "issue": "input_binding_identity_mismatch",
                                    "step_id": step.step_id,
                                    "summary_path": path,
                                    "input_key": input_key,
                                    "field": metadata_field,
                                },
                            )
                        )

        declarations = self._artifact_declarations(step_summary)
        for declaration in declarations:
            artifact = str(declaration["artifact"])
            binding = summary_bindings.get(artifact)
            if binding is None:
                continue
            contradictions: List[str] = []
            if (
                declaration["loaded"] is not None
                and isinstance(binding.get("loaded"), bool)
                and declaration["loaded"] != binding["loaded"]
            ):
                contradictions.append("loaded")
            binding_count, _ = self._row_count_claim(binding)
            if (
                declaration["row_count"] is not None
                and binding_count is not None
                and declaration["row_count"] != binding_count
            ):
                contradictions.append(declaration["row_count_field"] or "row_count")
            if contradictions:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Contradictory declarations for exact input "
                            f"{artifact!r} in step {step.step_id}: "
                            f"{declaration['path']} disagrees with input_bindings."
                        ),
                        detail={
                            "issue": "nested_input_declaration_contradiction",
                            "step_id": step.step_id,
                            "summary_path": declaration["path"],
                            "input_key": artifact,
                            "contradictory_fields": contradictions,
                        },
                    )
                )

        for checked in self._checked_reconciliation_blocks(step_summary):
            block = checked["value"]
            block_path = str(checked["path"])
            reference_artifact = self._named_artifact(
                block, self._REFERENCE_ARTIFACT_FIELDS
            )
            subset_artifact = self._named_artifact(block, self._SUBSET_ARTIFACT_FIELDS)
            key_columns = self._string_columns(block.get("key_columns"))
            value_columns = self._string_columns(block.get("value_columns_checked"))
            value_mismatch_n = self._as_count(block.get("value_mismatch_n"))
            invalid_fields = []
            if reference_artifact not in host_bindings:
                invalid_fields.append("reference_artifact")
            if subset_artifact not in host_bindings:
                invalid_fields.append("subset_artifact")
            if reference_artifact is not None and reference_artifact == subset_artifact:
                invalid_fields.append("distinct_reference_and_subset_artifacts")
            if key_columns is None:
                invalid_fields.append("key_columns")
            if value_columns is None:
                invalid_fields.append("value_columns_checked")
            if value_mismatch_n != 0:
                invalid_fields.append("value_mismatch_n")

            related_artifacts = {reference_artifact, subset_artifact} - {None}
            scope = re.sub(
                r"_(?:subset|reconciliation|verification)$",
                "",
                str(checked["path_name"]),
            )
            scoped_typed_declarations = [
                declaration
                for declaration in declarations
                if scope
                and declaration["scope"] == scope
                and declaration["artifact"] in host_bindings
            ]
            has_direct_artifact_intent = any(
                isinstance(block.get(field), str)
                for field in (
                    *self._REFERENCE_ARTIFACT_FIELDS,
                    *self._SUBSET_ARTIFACT_FIELDS,
                )
            )
            if not has_direct_artifact_intent and not scoped_typed_declarations:
                continue
            falsely_unloaded = sorted(
                {
                    str(declaration["artifact"])
                    for declaration in declarations
                    if declaration["loaded"] is False
                    and (
                        declaration["artifact"] in related_artifacts
                        or (scope and declaration["scope"] == scope)
                    )
                }
            )
            if falsely_unloaded:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Checked reconciliation {block_path} in step "
                            f"{step.step_id} conflicts with loaded=false for "
                            f"{falsely_unloaded}."
                        ),
                        detail={
                            "issue": "checked_reconciliation_unloaded_input",
                            "step_id": step.step_id,
                            "summary_path": block_path,
                            "unloaded_artifacts": falsely_unloaded,
                        },
                    )
                )
            if invalid_fields:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Checked subset/reconciliation {block_path} in step "
                            f"{step.step_id} lacks host-verifiable machine fields. "
                            "Declare distinct reference_artifact and subset_artifact "
                            "inputs, non-empty "
                            "key_columns and value_columns_checked, and "
                            "value_mismatch_n=0."
                        ),
                        detail={
                            "issue": "checked_reconciliation_evidence_incomplete",
                            "step_id": step.step_id,
                            "summary_path": block_path,
                            "invalid_fields": invalid_fields,
                        },
                    )
                )
                continue
            host_finding = self._host_reconciliation_finding(
                step=step,
                block_path=block_path,
                reference_artifact=str(reference_artifact),
                subset_artifact=str(subset_artifact),
                key_columns=key_columns or [],
                value_columns=value_columns or [],
                resolved_input_bindings=host_bindings,
            )
            if host_finding is not None:
                findings.append(host_finding)
        return findings


__all__ = ["StepSummaryIntegrityValidator"]
