"""Strict replay-schema migration for completed legacy trajectory steps.

Older runs stored the representation and selected clustering contract across
large ``step_summary.json`` payloads and auxiliary files.  A resumed downstream
agent then had to rediscover those fields.  This module performs a narrow,
deterministic migration: it copies already-declared scientific choices into
typed manifests after verifying them against the registered producer outputs.
It never chooses a representation, method, cluster count, or assignment.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .evidence import EvidenceStore
from .runtime_artifacts import current_step_records, verified_run_evidence_path
from .schema import AnalysisPlan, ResearchContext, ValidationFinding
from .trajectory_plan_contract import evaluate_trajectory_plan_dag


_REPRESENTATION_SCHEMA = "trajectory_representation_schema"
_CANDIDATE_SCHEMA = "candidate_cluster_solution_schema"


class _LegacyTrajectorySchemaError(ValueError):
    """A legacy producer does not expose one unambiguous replay contract."""


def _source_name(record: Any, verified_path: Path) -> Optional[str]:
    evidence_id = str(getattr(record, "evidence_id", "") or "")
    prefix = f"{evidence_id}__"
    if not evidence_id or not verified_path.name.startswith(prefix):
        return None
    return verified_path.name[len(prefix) :] or None


def _active_file(
    *,
    record: Mapping[str, Any],
    evidence: EvidenceStore,
    run_dir: Path,
    stem: str,
) -> Tuple[Any, Path]:
    matches = _active_files(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem=stem,
    )
    if len(matches) != 1:
        raise _LegacyTrajectorySchemaError(
            f"expected one current {stem!r} output, found {len(matches)}"
        )
    return matches[0]


def _active_files(
    *,
    record: Mapping[str, Any],
    evidence: EvidenceStore,
    run_dir: Path,
    stem: str,
) -> List[Tuple[Any, Path]]:
    matches: List[Tuple[Any, Path]] = []
    active_ids = {
        str(value)
        for value in record.get("evidence_ids", [])
        if str(value).strip()
    }
    for evidence_record in evidence.records():
        if evidence_record.evidence_id not in active_ids:
            continue
        if evidence_record.produced_by_step != str(record.get("step_id") or ""):
            continue
        verified = verified_run_evidence_path(run_dir, evidence_record)
        if verified is None:
            continue
        source_name = _source_name(evidence_record, verified)
        if source_name is not None and Path(source_name).stem == stem:
            matches.append((evidence_record, verified))
    return matches


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _LegacyTrajectorySchemaError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise _LegacyTrajectorySchemaError(f"{label} must contain a JSON object")
    return payload


def _one_string(values: Sequence[Any], *, field: str) -> str:
    present = {
        str(value).strip()
        for value in values
        if isinstance(value, str) and str(value).strip()
    }
    if len(present) != 1:
        raise _LegacyTrajectorySchemaError(
            f"{field} must resolve to one explicit value; found {sorted(present)}"
        )
    return next(iter(present))


def _string_list(value: Any, *, field: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise _LegacyTrajectorySchemaError(f"{field} must be a non-empty list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(set(result)) != len(result):
        raise _LegacyTrajectorySchemaError(
            f"{field} must contain unique non-empty strings"
        )
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise _LegacyTrajectorySchemaError(f"{field} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise _LegacyTrajectorySchemaError(
            f"{field} must be a positive integer"
        ) from exc
    if numeric <= 0 or numeric != float(value):
        raise _LegacyTrajectorySchemaError(f"{field} must be a positive integer")
    return numeric


def _table_columns(path: Path) -> List[str]:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return list(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    elif suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return list(pd.read_csv(path, sep=separator, nrows=0).columns)
    raise _LegacyTrajectorySchemaError(
        f"unsupported legacy trajectory table format: {path.suffix}"
    )


def _table_columns_and_ids(path: Path, *, id_column: str) -> Tuple[List[str], List[Any]]:
    columns = _table_columns(path)
    if id_column not in columns:
        raise _LegacyTrajectorySchemaError(
            f"registered table does not contain explicit id_column={id_column!r}"
        )
    ids = _read_table(path, columns=[id_column])[id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise _LegacyTrajectorySchemaError(
            f"explicit id_column={id_column!r} must be complete and unique"
        )
    return columns, ids.tolist()


def _read_table(path: Path, *, columns: Optional[Sequence[str]] = None):
    import pandas as pd

    suffix = path.suffix.lower()
    selected = list(columns) if columns is not None else None
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=selected)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(
            path,
            sep="\t" if suffix == ".tsv" else ",",
            usecols=selected,
        )
    raise _LegacyTrajectorySchemaError(
        f"unsupported legacy trajectory table format: {path.suffix}"
    )


def _boolean_series(series: Any, *, field: str):
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower()
        )
    )
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    if normalized.isna().any() or not normalized.isin(mapping).all():
        raise _LegacyTrajectorySchemaError(f"{field} is not explicitly boolean")
    return normalized.map(mapping).astype(bool)


def _normalised_method_family(value: Any) -> Tuple[str, ...]:
    tokens = tuple(
        token
        for token in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
        if token
    )
    token_set = set(tokens)
    if {"gaussian", "mixture"} <= token_set:
        family = ["gaussian_mixture"]
        for modifier in ("bayesian", "diagonal", "latent", "class"):
            if modifier in token_set:
                family.append(modifier)
        return tuple(family)
    return tokens


def _schema_already_active(
    *,
    record: Mapping[str, Any],
    evidence: EvidenceStore,
    run_dir: Path,
    stem: str,
) -> bool:
    matches = _active_files(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem=stem,
    )
    if len(matches) > 1:
        raise _LegacyTrajectorySchemaError(
            f"multiple current {stem!r} outputs are active"
        )
    return len(matches) == 1


def _verified_step_summary(
    *,
    record: Mapping[str, Any],
    evidence: EvidenceStore,
    run_dir: Path,
) -> Tuple[Mapping[str, Any], str]:
    evidence_id = str(record.get("step_summary_evidence_id") or "")
    active_ids = {
        str(value)
        for value in record.get("evidence_ids", [])
        if str(value).strip()
    }
    if not evidence_id or evidence_id not in active_ids:
        raise _LegacyTrajectorySchemaError(
            "step_summary_evidence_id is not current for the resumed producer"
        )
    summary_record = evidence.get(evidence_id)
    step_id = str(record.get("step_id") or "")
    if summary_record is None or summary_record.produced_by_step != step_id:
        raise _LegacyTrajectorySchemaError(
            "step summary evidence is absent or belongs to another producer"
        )
    summary_path = verified_run_evidence_path(run_dir, summary_record)
    if summary_path is None:
        raise _LegacyTrajectorySchemaError(
            "step summary evidence failed path or digest verification"
        )
    disk_summary = _read_json(summary_path, label="registered step summary")
    embedded_summary = record.get("step_summary")
    if not isinstance(embedded_summary, Mapping) or dict(disk_summary) != dict(
        embedded_summary
    ):
        raise _LegacyTrajectorySchemaError(
            "outer record step_summary disagrees with digest-verified evidence"
        )
    return disk_summary, evidence_id


def _write_and_register_schema(
    *,
    payload: Mapping[str, Any],
    stem: str,
    record: Dict[str, Any],
    run_dir: Path,
    evidence: EvidenceStore,
    prompt_pack_version: str,
    input_evidence_ids: Sequence[str],
) -> str:
    step_id = str(record.get("step_id") or "")
    out_dir = run_dir / "resume_migrations" / step_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    registered = evidence.register_file(
        kind="log",
        description=(
            "Deterministically migrated trajectory replay schema from the "
            "completed legacy agent step."
        ),
        source_path=path,
        produced_by_step=step_id,
        inputs=list(dict.fromkeys(input_evidence_ids)) or None,
        producer="resume_schema_migration",
        generation_mode="system",
        prompt_pack_version=prompt_pack_version,
        metadata={
            "migration": "legacy_trajectory_replay_schema",
            "scientific_choices_changed": False,
        },
        on_sha_change="new_id",
    )
    evidence_ids = list(record.get("evidence_ids") or [])
    if registered.evidence_id not in evidence_ids:
        evidence_ids.append(registered.evidence_id)
    record["evidence_ids"] = evidence_ids
    migration_records = list(record.get("resume_schema_migrations") or [])
    migration_records.append(
        {
            "typed_product": f"manifest:{stem}",
            "evidence_id": registered.evidence_id,
            "relative_path": registered.relative_path,
        }
    )
    record["resume_schema_migrations"] = migration_records
    return registered.evidence_id


def _representation_schema_payload(
    *,
    record: Dict[str, Any],
    context: ResearchContext,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    summary, summary_evidence_id = _verified_step_summary(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
    )
    trajectory_source = summary.get("trajectory_source")
    if not isinstance(trajectory_source, Mapping):
        trajectory_source = {}
    representation_columns = _string_list(
        summary.get("representation_columns")
        or summary.get("scaled_representation_columns"),
        field="representation_columns",
    )
    observation_columns = _string_list(
        summary.get("ordered_observation_columns")
        or summary.get("observation_columns"),
        field="observation_columns",
    )
    profile_columns = _string_list(
        summary.get("profile_columns"), field="profile_columns"
    )
    profile_statistic = _one_string(
        [summary.get("profile_summary_statistic")],
        field="profile_summary_statistic",
    ).lower()
    if profile_statistic not in {"mean", "median"}:
        raise _LegacyTrajectorySchemaError(
            "profile_summary_statistic must be mean or median"
        )
    population_n = _positive_int(
        summary.get("representation_row_n") or summary.get("frozen_population_n"),
        field="frozen_population_n",
    )
    representation_record, representation_path = _active_file(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem="trajectory_representation",
    )
    membership_record, membership_path = _active_file(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem="trajectory_membership",
    )
    columns = _table_columns(representation_path)
    membership_columns = _table_columns(membership_path)
    explicit_ids = {
        str(value).strip()
        for value in (
            summary.get("id_column"),
            trajectory_source.get("source_id_column"),
        )
        if isinstance(value, str) and value.strip()
    }
    if len(explicit_ids) > 1:
        raise _LegacyTrajectorySchemaError("explicit legacy id_column values disagree")
    if explicit_ids:
        id_column = next(iter(explicit_ids))
        id_derivation = "explicit_legacy_metadata"
    else:
        shared = [column for column in columns if column in membership_columns]
        candidates: List[str] = []
        for column in shared:
            try:
                _, representation_ids = _table_columns_and_ids(
                    representation_path, id_column=column
                )
                _, membership_ids = _table_columns_and_ids(
                    membership_path, id_column=column
                )
            except _LegacyTrajectorySchemaError:
                continue
            if set(map(str, representation_ids)) == set(map(str, membership_ids)):
                candidates.append(column)
        if len(candidates) != 1:
            raise _LegacyTrajectorySchemaError(
                "legacy id_column lacks one complete, unique, equal-set shared column"
            )
        id_column = candidates[0]
        id_derivation = "unique_complete_equal_set_shared_column"
    _, ids = _table_columns_and_ids(representation_path, id_column=id_column)
    missing_coordinates = [
        column for column in representation_columns if column not in columns
    ]
    if missing_coordinates:
        raise _LegacyTrajectorySchemaError(
            "representation_columns are absent from the registered representation"
        )
    if len(ids) != population_n:
        raise _LegacyTrajectorySchemaError(
            "frozen_population_n disagrees with the registered representation"
        )
    missing_profiles = [column for column in profile_columns if column not in columns]
    if missing_profiles:
        raise _LegacyTrajectorySchemaError(
            "profile_columns are absent from the registered representation"
        )
    membership = _read_table(membership_path)
    required_membership = {
        id_column,
        "observed_window_count",
        "meets_min_observed_windows",
        "included_in_clustering",
    }
    if not required_membership <= set(membership.columns):
        raise _LegacyTrajectorySchemaError(
            "trajectory membership lacks the canonical replay fields"
        )
    if membership[id_column].isna().any() or membership[id_column].duplicated().any():
        raise _LegacyTrajectorySchemaError("membership id_column is not complete and unique")
    included = _boolean_series(
        membership["included_in_clustering"], field="included_in_clustering"
    )
    meets_minimum = _boolean_series(
        membership["meets_min_observed_windows"],
        field="meets_min_observed_windows",
    )
    import pandas as pd

    observed_counts = pd.to_numeric(
        membership["observed_window_count"], errors="coerce"
    )
    min_observed_windows = _positive_int(
        summary.get("min_observed_windows"), field="min_observed_windows"
    )
    if observed_counts.isna().any() or not (
        meets_minimum == (observed_counts >= min_observed_windows)
    ).all():
        raise _LegacyTrajectorySchemaError(
            "membership minimum-window flags do not replay the declared threshold"
        )
    if (included & ~meets_minimum).any():
        raise _LegacyTrajectorySchemaError(
            "legacy included population violates the declared minimum-window rule"
        )
    included_ids = membership.loc[included, id_column].map(str).tolist()
    if set(map(str, ids)) != set(included_ids) or len(included_ids) != population_n:
        raise _LegacyTrajectorySchemaError(
            "representation ids differ from the included membership population"
        )

    missingness_policy = summary.get("missingness_policy")
    if not isinstance(missingness_policy, Mapping) or (
        missingness_policy.get("no_zero_imputation") is not True
        or missingness_policy.get("no_value_imputation") is not True
    ):
        raise _LegacyTrajectorySchemaError(
            "legacy summary lacks explicit no-imputation boolean metadata"
        )
    indicator_columns = _string_list(
        summary.get("missingness_indicator_columns"),
        field="missingness_indicator_columns",
    )
    if len(indicator_columns) != len(representation_columns):
        raise _LegacyTrajectorySchemaError(
            "missingness indicators do not align one-to-one with model coordinates"
        )
    representation_audit = _read_table(
        representation_path,
        columns=[*representation_columns, *indicator_columns, *profile_columns],
    )
    for coordinate, indicator in zip(
        representation_columns, indicator_columns, strict=True
    ):
        observed = _boolean_series(
            representation_audit[indicator], field=f"indicator:{indicator}"
        )
        if representation_audit.loc[~observed, coordinate].notna().any():
            raise _LegacyTrajectorySchemaError(
                "an unobserved coordinate contains an imputed value"
            )

    profile_sources = summary.get("profile_window_columns")
    if not isinstance(profile_sources, Mapping):
        raise _LegacyTrajectorySchemaError("profile_window_columns is absent")
    expected_profile_columns = {f"profile__{key}" for key in profile_sources}
    if expected_profile_columns != set(profile_columns):
        raise _LegacyTrajectorySchemaError(
            "profile columns do not map exactly to profile_window_columns"
        )
    import numpy as np

    for family, raw_sources in profile_sources.items():
        sources = _string_list(raw_sources, field=f"profile_window_columns:{family}")
        profile_column = f"profile__{family}"
        if any(source not in representation_audit.columns for source in sources):
            raise _LegacyTrajectorySchemaError(
                "profile source columns are absent from the representation"
            )
        source_values = representation_audit[sources]
        replayed = (
            source_values.mean(axis=1, skipna=True)
            if profile_statistic == "mean"
            else source_values.median(axis=1, skipna=True)
        )
        if not np.allclose(
            replayed.to_numpy(dtype=float),
            representation_audit[profile_column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise _LegacyTrajectorySchemaError(
                "profile summaries do not replay the declared missing-aware statistic"
            )

    anchor = _one_string([summary.get("anchor")], field="anchor")
    task_anchors = {
        str(constraint.anchor_event).strip()
        for constraint in context.temporal_constraints
        if constraint.relation == "relative_to_anchor"
        and str(constraint.anchor_event).strip()
    }
    if len(task_anchors) == 1:
        task_anchor = next(iter(task_anchors))
        if anchor != task_anchor:
            raise _LegacyTrajectorySchemaError(
                "representation anchor disagrees with the task contract"
            )
        anchor_provenance = "task_contract"
        anchor_source = "temporal_constraints.relative_to_anchor"
    else:
        anchor_provenance = _one_string(
            [summary.get("anchor_provenance")], field="anchor_provenance"
        )
        if anchor_provenance != "agent_declared":
            raise _LegacyTrajectorySchemaError(
                "anchor without a unique task contract must be explicitly agent_declared"
            )
        anchor_source = _one_string(
            [summary.get("anchor_source")], field="anchor_source"
        )
    trailing_policy = {
        "zero_imputation": False,
        "eligibility_uses_observed_window_count": True,
        "profile_summaries_ignore_missing": True,
    }
    payload = {
        "schema_version": "easyicu.trajectory_representation_schema/1",
        "migration_source": "legacy_step_summary",
        "id_column": id_column,
        "id_column_derivation": id_derivation,
        "representation_columns": representation_columns,
        "observation_family": _one_string(
            [summary.get("observation_family")], field="observation_family"
        ),
        "observation_columns": observation_columns,
        "min_observed_windows": min_observed_windows,
        "profile_columns": profile_columns,
        "profile_summary_statistic": profile_statistic,
        "time_axis": _one_string([summary.get("time_axis")], field="time_axis"),
        "anchor": anchor,
        "anchor_provenance": anchor_provenance,
        "anchor_source": anchor_source,
        "trailing_na_policy": trailing_policy,
        "frozen_population_n": population_n,
        "representation_evidence_id": representation_record.evidence_id,
        "representation_sha256": representation_record.sha256,
        "membership_evidence_id": membership_record.evidence_id,
        "membership_sha256": membership_record.sha256,
        "step_summary_evidence_id": summary_evidence_id,
    }
    return payload, [
        summary_evidence_id,
        representation_record.evidence_id,
        membership_record.evidence_id,
    ]


def _candidate_schema_payload(
    *,
    record: Dict[str, Any],
    representation_schema: Mapping[str, Any],
    representation_schema_evidence_id: str,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    summary, summary_evidence_id = _verified_step_summary(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
    )
    model_record, model_path = _active_file(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem="candidate_cluster_models",
    )
    assignment_record, assignment_path = _active_file(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem="candidate_cluster_assignments",
    )
    selection_record, selection_path = _active_file(
        record=record,
        evidence=evidence,
        run_dir=run_dir,
        stem="cluster_selection",
    )
    models_payload = _read_json(model_path, label="candidate model artifact")
    selection = _read_json(selection_path, label="cluster selection artifact")
    if dict(selection) != summary.get("cluster_selection"):
        raise _LegacyTrajectorySchemaError(
            "candidate step summary and cluster-selection evidence disagree"
        )
    selected_k = _positive_int(
        selection.get("selected_n_clusters"), field="selected_n_clusters"
    )
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise _LegacyTrajectorySchemaError(
            "cluster selection must preserve at least two candidates"
        )
    selected_candidates = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and _positive_int(item.get("n_clusters"), field="candidate n_clusters")
        == selected_k
    ]
    if len(selected_candidates) != 1:
        raise _LegacyTrajectorySchemaError(
            "selected_n_clusters must resolve to one selection candidate"
        )
    selected_value = float(selected_candidates[0].get("criterion_value"))
    if not math.isfinite(selected_value):
        raise _LegacyTrajectorySchemaError("selected criterion value must be finite")

    model_family = _one_string(
        [models_payload.get("model_family")], field="model_family"
    )
    clustering_method = _one_string(
        [summary.get("clustering_method") or model_family],
        field="clustering_method",
    )
    if _normalised_method_family(clustering_method) != _normalised_method_family(
        model_family
    ):
        raise _LegacyTrajectorySchemaError(
            "candidate clustering_method and fitted model_family disagree"
        )
    models = models_payload.get("models")
    if not isinstance(models, list) or not models:
        raise _LegacyTrajectorySchemaError("candidate models are absent")
    fitted_models = [
        item
        for item in models
        if isinstance(item, Mapping)
        and (
            str(item.get("fit_status") or "").strip().lower() == "fitted"
            or item.get("converged") is True
        )
    ]
    record_families = {
        str(item.get("model_family") or "").strip() for item in fitted_models
    }
    if record_families != {model_family}:
        raise _LegacyTrajectorySchemaError(
            "fitted candidate records do not share the top-level model_family"
        )
    selected_models = [
        item
        for item in fitted_models
        if _positive_int(item.get("n_clusters"), field="model n_clusters")
        == selected_k
    ]
    if len(selected_models) != 1:
        raise _LegacyTrajectorySchemaError(
            "selected_n_clusters must resolve to one fitted candidate model"
        )
    selected_model = selected_models[0]
    model_value = float(
        selected_model.get("criterion_value", selected_model.get("bic"))
    )
    if not math.isfinite(model_value) or not math.isclose(
        model_value, selected_value, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise _LegacyTrajectorySchemaError(
            "selected model criterion disagrees with cluster selection"
        )
    representation_columns = _string_list(
        models_payload.get("representation_columns"),
        field="candidate representation_columns",
    )
    if representation_columns != list(
        representation_schema.get("representation_columns") or []
    ):
        raise _LegacyTrajectorySchemaError(
            "candidate representation_columns disagree with representation schema"
        )
    id_column = _one_string(
        [representation_schema.get("id_column")], field="id_column"
    )
    assignment_columns, assignment_ids = _table_columns_and_ids(
        assignment_path, id_column=id_column
    )
    representation_record = evidence.get(
        str(representation_schema.get("representation_evidence_id") or "")
    )
    if representation_record is None:
        raise _LegacyTrajectorySchemaError(
            "representation schema does not resolve to registered evidence"
        )
    representation_path = verified_run_evidence_path(run_dir, representation_record)
    if representation_path is None:
        raise _LegacyTrajectorySchemaError(
            "representation schema points to unverifiable evidence"
        )
    _, representation_ids = _table_columns_and_ids(
        representation_path, id_column=id_column
    )
    if set(map(str, assignment_ids)) != set(map(str, representation_ids)):
        raise _LegacyTrajectorySchemaError(
            "candidate assignments and representation have different id sets"
        )
    candidate_counts = [
        _positive_int(item.get("n_clusters"), field="candidate n_clusters")
        for item in candidates
        if isinstance(item, Mapping)
    ]
    expected_assignment_columns = {
        f"cluster_label_k{k}" for k in candidate_counts
    } | {f"max_posterior_k{k}" for k in candidate_counts} | {
        f"assignment_available_k{k}" for k in candidate_counts
    }
    if not expected_assignment_columns <= set(assignment_columns):
        raise _LegacyTrajectorySchemaError(
            "legacy candidate assignments lack one exact label/posterior/availability "
            "column set per selection candidate"
        )
    assignment_column = f"cluster_label_k{selected_k}"
    availability_column = f"assignment_available_k{selected_k}"
    assignment_frame = _read_table(
        assignment_path,
        columns=[id_column, assignment_column, availability_column],
    )
    availability = _boolean_series(
        assignment_frame[availability_column], field=availability_column
    )
    if not availability.all() or assignment_frame[assignment_column].isna().any():
        raise _LegacyTrajectorySchemaError(
            "selected candidate assignments are incomplete or unavailable"
        )
    explicit_model_ids = {
        str(value).strip()
        for value in (
            selection.get("selected_model_id"),
            selected_model.get("model_id"),
        )
        if isinstance(value, str) and value.strip()
    }
    if len(explicit_model_ids) > 1:
        raise _LegacyTrajectorySchemaError("selected model ids disagree")
    if explicit_model_ids:
        selected_model_id = next(iter(explicit_model_ids))
        selected_model_id_derivation = "explicit_legacy_metadata"
    else:
        selected_model_id = (
            f"{model_record.evidence_id}::n_clusters_{selected_k}"
        )
        selected_model_id_derivation = (
            "candidate_model_evidence_id_plus_selected_n_clusters"
        )
    payload = {
        "schema_version": "easyicu.candidate_cluster_solution_schema/2",
        "migration_source": "legacy_candidate_outputs",
        "id_column": id_column,
        "representation_columns": representation_columns,
        "clustering_method": clustering_method,
        "model_family": model_family,
        "fit_method": _one_string(
            [selected_model.get("fit_method")], field="fit_method"
        ),
        "covariance_type": _one_string(
            [selected_model.get("covariance_type")], field="covariance_type"
        ),
        "selected_n_clusters": selected_k,
        "selected_model_id": selected_model_id,
        "selected_model_id_derivation": selected_model_id_derivation,
        "assignment_column": assignment_column,
        "criterion": _one_string([selection.get("criterion")], field="criterion"),
        "selection_rule": _one_string(
            [selection.get("selection_rule")], field="selection_rule"
        ),
        "direction": _one_string([selection.get("direction")], field="direction"),
        "selected_criterion_value": selected_value,
        "representation_schema_evidence_id": representation_schema_evidence_id,
        "candidate_models_evidence_id": model_record.evidence_id,
        "candidate_assignments_evidence_id": assignment_record.evidence_id,
        "cluster_selection_evidence_id": selection_record.evidence_id,
        "step_summary_evidence_id": summary_evidence_id,
    }
    return payload, [
        summary_evidence_id,
        representation_schema_evidence_id,
        model_record.evidence_id,
        assignment_record.evidence_id,
        selection_record.evidence_id,
    ]


def materialize_legacy_trajectory_replay_schemas(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    run_dir: Path,
    evidence: EvidenceStore,
    per_step_records: List[Dict[str, Any]],
    prompt_pack_version: str,
) -> List[ValidationFinding]:
    """Materialize missing typed schemas for successful resumed producers."""

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)
    if not evaluation.applies or evaluation.findings:
        return []
    latest = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(per_step_records)
    }
    representation_id = evaluation.role_owners.get("representation")
    candidate_id = evaluation.role_owners.get("candidate_selection")
    if not representation_id or not candidate_id:
        return []
    representation_record = latest.get(representation_id)
    candidate_record = latest.get(candidate_id)
    representation_resumed = representation_record is not None and (
        str(representation_record.get("status") or "").lower() == "ok"
    )
    candidate_resumed = candidate_record is not None and (
        str(candidate_record.get("status") or "").lower() == "ok"
    )
    if not representation_resumed and not candidate_resumed:
        return []

    materialized: List[str] = []
    payload: Optional[Dict[str, Any]] = None
    representation_schema_evidence_id: Optional[str] = None
    try:
        if representation_resumed:
            assert representation_record is not None
            if not _schema_already_active(
                record=representation_record,
                evidence=evidence,
                run_dir=run_dir,
                stem=_REPRESENTATION_SCHEMA,
            ):
                payload, inputs = _representation_schema_payload(
                    record=representation_record,
                    context=context,
                    evidence=evidence,
                    run_dir=run_dir,
                )
                representation_schema_evidence_id = _write_and_register_schema(
                    payload=payload,
                    stem=_REPRESENTATION_SCHEMA,
                    record=representation_record,
                    run_dir=run_dir,
                    evidence=evidence,
                    prompt_pack_version=prompt_pack_version,
                    input_evidence_ids=inputs,
                )
                materialized.append(
                    f"manifest:{_REPRESENTATION_SCHEMA}"
                )
            else:
                schema_record, schema_path = _active_file(
                    record=representation_record,
                    evidence=evidence,
                    run_dir=run_dir,
                    stem=_REPRESENTATION_SCHEMA,
                )
                representation_schema_evidence_id = schema_record.evidence_id
                payload = dict(
                    _read_json(
                        schema_path, label="trajectory representation schema"
                    )
                )
        if candidate_resumed:
            assert candidate_record is not None
            if payload is None or representation_schema_evidence_id is None:
                raise _LegacyTrajectorySchemaError(
                    "a resumed candidate owner lacks a current representation schema"
                )
            candidate_schema_active = _schema_already_active(
                record=candidate_record,
                evidence=evidence,
                run_dir=run_dir,
                stem=_CANDIDATE_SCHEMA,
            )
            if candidate_schema_active:
                active_record, active_path = _active_file(
                    record=candidate_record,
                    evidence=evidence,
                    run_dir=run_dir,
                    stem=_CANDIDATE_SCHEMA,
                )
                active_payload = _read_json(
                    active_path, label="candidate cluster solution schema"
                )
                if active_payload.get("schema_version") != (
                    "easyicu.candidate_cluster_solution_schema/2"
                ):
                    candidate_record["evidence_ids"] = [
                        evidence_id
                        for evidence_id in candidate_record.get("evidence_ids", [])
                        if str(evidence_id) != active_record.evidence_id
                    ]
                    superseded = list(
                        candidate_record.get("resume_schema_superseded") or []
                    )
                    superseded.append(active_record.evidence_id)
                    candidate_record["resume_schema_superseded"] = superseded
                    candidate_schema_active = False
            if not candidate_schema_active:
                candidate_payload, inputs = _candidate_schema_payload(
                    record=candidate_record,
                    representation_schema=payload,
                    representation_schema_evidence_id=(
                        representation_schema_evidence_id
                    ),
                    evidence=evidence,
                    run_dir=run_dir,
                )
                _write_and_register_schema(
                    payload=candidate_payload,
                    stem=_CANDIDATE_SCHEMA,
                    record=candidate_record,
                    run_dir=run_dir,
                    evidence=evidence,
                    prompt_pack_version=prompt_pack_version,
                    input_evidence_ids=inputs,
                )
                materialized.append(f"manifest:{_CANDIDATE_SCHEMA}")
    except (_LegacyTrajectorySchemaError, TypeError, ValueError) as exc:
        return [
            ValidationFinding(
                validator="trajectory_resume_schema",
                severity="error",
                message=(
                    "Could not migrate completed legacy trajectory producers to "
                    "the typed replay-schema contract; downstream execution must "
                    "remain fail-closed."
                ),
                detail={"reason": str(exc)},
            )
        ]

    if not materialized:
        return []
    return [
        ValidationFinding(
            validator="trajectory_resume_schema",
            severity="info",
            message=(
                "Bound completed legacy trajectory choices into digest-verified "
                "typed replay schemas."
            ),
            detail={
                "kind": "legacy_trajectory_replay_schemas_materialized",
                "representation_step_id": representation_id,
                "candidate_step_id": candidate_id,
                "materialized_products": materialized,
                "scientific_choices_changed": False,
            },
        )
    ]


__all__ = ["materialize_legacy_trajectory_replay_schemas"]
