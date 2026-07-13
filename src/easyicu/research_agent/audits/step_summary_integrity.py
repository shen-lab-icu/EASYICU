"""Host-authoritative integrity checks for generated step summaries.

This module validates provenance declarations and explicit tabular subset
reconciliations. It never chooses a cohort, exposure, outcome, estimator, or
scientific method.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..schema import AnalysisStep, ValidationFinding


class StepSummaryIntegrityValidator:
    """Verify exact input claims and checked subset reconciliations.

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
                str(name)
                for name in pd.read_csv(path, sep=separator, nrows=0).columns
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
                if (
                    cls._normalise(value.get("status")) == "checked"
                    and path_tokens.intersection({"subset", "reconciliation"})
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
    def _artifact_declarations(
        cls, summary: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
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

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        host_bindings = {
            str(key): value
            for key, value in resolved_input_bindings.items()
            if isinstance(value, Mapping)
        }
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
            subset_artifact = self._named_artifact(
                block, self._SUBSET_ARTIFACT_FIELDS
            )
            key_columns = self._string_columns(block.get("key_columns"))
            value_columns = self._string_columns(block.get("value_columns_checked"))
            value_mismatch_n = self._as_count(block.get("value_mismatch_n"))
            invalid_fields = []
            if reference_artifact not in host_bindings:
                invalid_fields.append("reference_artifact")
            if subset_artifact not in host_bindings:
                invalid_fields.append("subset_artifact")
            if (
                reference_artifact is not None
                and reference_artifact == subset_artifact
            ):
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
