"""Bounded exploratory ICU landmark grouping for directly measured labs.

Only export-bound ICU-relative hours, known ICU follow-up and ICU death are
used. The measurement window is [0, landmark); risk begins at the landmark.
No identifiers or row-level measurements are returned in the public payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from easyicu.concept.metadata_sidecar import MetadataSidecarError, read_content_addressed_sidecar
from easyicu.webserver import entity_ids as entity_id_contract
from easyicu.webserver.source_cache_identity import source_file_identity

_LAB_MODULES = {"blood_gas", "chemistry", "hematology"}
_MAX_ROWS = 2_000_000
_MAX_ENTITIES = 50_000
_MAX_TIES = 128
_LANDMARK_SLOT = BoundedSemaphore(1)


def _batches(path: Path, columns: list[str]):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).iter_batches(batch_size=16_384, columns=columns)


def _blocked(reason: str) -> dict[str, Any]:
    return {"group": {"basis": "icu_landmark", "status": "blocked", "reason": reason,
                      "groups": [], "_members": [], "supported_outcomes": []},
            "vectors": ({}, {}), "details": {}}


def blocked(reason: str) -> dict[str, Any]:
    return _blocked(reason)


def _binding(path: Path, desc: dict, feature: dict, landmark_hours: int):
    """Verify the content-addressed time/unit contract for this physical file."""
    if feature.get("module") not in _LAB_MODULES:
        raise ValueError("Landmark grouping requires a directly measured laboratory feature.")
    manifest_path = path / "_manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 10_000_000:
        raise ValueError("Export time metadata is missing or invalid.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "easyicu_native_export_v2" or manifest.get("export_kind") != "web_module_extraction":
        raise ValueError("Export time metadata is missing or invalid.")
    # eICU's native producer uses ICU discharge status and discharge offset;
    # other databases can have a recorded death time distinct from ICU exit.
    if manifest.get("database") not in {"eicu", "eicu_demo"}:
        raise ValueError("ICU discharge-event timing is not verified for this source.")
    window = (manifest.get("cohort_contract") or {}).get("observation_window_hours")
    if isinstance(window, bool) or not isinstance(window, (int, float)) or not math.isfinite(window) or window < landmark_hours:
        raise ValueError("The export observation window does not cover the requested landmark.")
    ref = manifest.get("column_metadata") or {}
    name = ref.get("file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Export time metadata is missing or invalid.")
    sidecar = read_content_addressed_sidecar(path / name, expected_sha256=ref.get("sha256"), expected_size=ref.get("size"))
    if sidecar.source_database != str(manifest.get("database") or "").lower():
        raise ValueError("Export time metadata is missing or invalid.")
    items = [i for i in desc.get("files", []) if i.get("module") == feature.get("module")]
    if len(items) != 1:
        raise ValueError("The selected laboratory file is unavailable or ambiguous.")
    item = items[0]
    relative = item.get("file")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("The selected laboratory file is unavailable or ambiguous.")
    bindings = [b for b in sidecar.files if b.relative_path == relative and b.module == feature.get("module")]
    if len(bindings) != 1:
        raise ValueError("The selected laboratory file lacks verified time metadata.")
    binding = bindings[0]
    times = [t for t in binding.time_coordinates if t.column == "charttime" and t.origin == "icu_admission" and t.unit == "h"]
    if len(times) != 1:
        raise ValueError("The selected laboratory file lacks verified ICU-relative hours.")
    column = feature.get("column")
    value = binding.columns.get(column)
    if value is None or value.metadata.availability_basis != "direct_source" or value.metadata.aggregation is not None or value.metadata.derived_from_concepts or value.derivation_window is not None or value.representation_transform is not None:
        raise ValueError("Landmark grouping requires a directly measured laboratory feature.")
    unit = value.metadata.canonical_unit
    if not unit:
        raise ValueError("The measurement unit is not verified for this laboratory feature.")
    outcome_items = [b for b in sidecar.files if b.module == "outcome" and b.relative_path in {i.get("file") for i in desc.get("files", []) if i.get("module") == "outcome"}]
    if len(outcome_items) != 1:
        raise ValueError("ICU outcome metadata is missing or ambiguous.")
    out = outcome_items[0]
    los = out.columns.get("los_icu")
    event = out.columns.get("icu_death")
    if los is None or los.metadata.canonical_unit not in {"days", "day", "d"} or los.metadata.source_concept != "los_icu" or event is None or event.metadata.source_concept != "icu_death" or out.identity_column != binding.identity_column:
        raise ValueError("ICU death and ICU length-of-stay metadata are not verified.")
    if binding.identity_column not in item.get("columns", []) or column not in item.get("columns", []) or "charttime" not in item.get("columns", []):
        raise ValueError("The selected laboratory file lacks required physical columns.")
    if int(item.get("rows") or 0) > _MAX_ROWS:
        raise ValueError("The laboratory file exceeds the interactive landmark preview limit.")
    source_dir = manifest.get("data_path")
    if not isinstance(source_dir, str) or not source_dir:
        raise ValueError("The original ICU discharge source is unavailable for verification.")
    source_path = Path(source_dir).expanduser() / "patient.parquet"
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("The original ICU discharge source is unavailable for verification.")
    return item, binding.identity_column, unit, source_path, out.relative_path, name


def _verify_eicu_outcomes(path: Path, source_path: Path, outcome_file: str, cohort: list[str], output_id: str):
    """Read-only verification of the export against the declared native source.

    eICU's native status is a discharge status and its offset is measured in
    minutes from ICU admission. Unknown statuses stay unknown. Verification is
    required for every selected stay; neither raw values nor ids are returned.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    export_path = path / outcome_file
    if export_path.is_symlink() or not export_path.is_file() or pq.ParquetFile(source_path).metadata.num_rows > 1_000_000:
        raise ValueError("The original ICU discharge source is unavailable for verification.")
    wanted = set(cohort)
    def collect(file: Path, columns: list[str]):
        found = {}
        for batch in _batches(file, columns):
            rows = batch.to_pydict()
            for native_id, first, second in zip(*(rows[c] for c in columns)):
                sid = entity_id_contract.normalize_entity_id(native_id)
                if sid not in wanted:
                    continue
                if sid in found:
                    raise ValueError("The original ICU discharge source does not match the selected cohort.")
                found[sid] = (first, second)
        if set(found) != wanted:
            raise ValueError("The original ICU discharge source does not match the selected cohort.")
        return found
    raw = collect(source_path, ["patientunitstayid", "unitdischargestatus", "unitdischargeoffset"])
    out = collect(export_path, [output_id, "icu_death", "los_icu"])
    for sid in cohort:
        status, offset = raw[sid]
        event, los = out[sid]
        expected = {"alive": False, "expired": True}.get(str(status).strip().casefold()) if status is not None else None
        if event is not None and not isinstance(event, bool):
            raise ValueError("The exported ICU death status differs from the original discharge source.")
        if event != expected:
            raise ValueError("The exported ICU death status differs from the original discharge source.")
        expected_los = pd.to_numeric(offset, errors="coerce") / 1440.0 if offset is not None else float("nan")
        actual_los = pd.to_numeric(los, errors="coerce") if los is not None else float("nan")
        if not ((pd.isna(expected_los) and pd.isna(actual_los)) or (pd.notna(expected_los) and pd.notna(actual_los) and abs(expected_los - actual_los) <= 1e-5)):
            raise ValueError("The exported ICU follow-up differs from the original discharge source.")
    return ({sid: value[0] for sid, value in out.items() if value[0] is not None},
            {sid: float(value[1]) for sid, value in out.items() if value[1] is not None and pd.notna(value[1])})


def build_landmark_group(*, path: Path, desc: dict, entity_ids: list[str], outcome: Any,
                         feature: dict | None, threshold: float, landmark_hours: int) -> dict:
    if len(entity_ids) > _MAX_ENTITIES:
        return _blocked("The cohort exceeds the interactive landmark preview limit.")
    if not _LANDMARK_SLOT.acquire(blocking=False):
        return _blocked("Another landmark preview is running; try again after it finishes.")
    try:
        return _build_landmark_group(path=path, desc=desc, entity_ids=entity_ids, outcome=outcome,
                                     feature=feature, threshold=threshold, landmark_hours=landmark_hours)
    finally:
        _LANDMARK_SLOT.release()


def _build_landmark_group(*, path: Path, desc: dict, entity_ids: list[str], outcome: Any,
                          feature: dict | None, threshold: float, landmark_hours: int) -> dict:
    if not feature:
        return _blocked("The selected laboratory feature is unavailable.")
    try:
        manifest_identity = source_file_identity(path / "_manifest.json")
        initial_manifest = json.loads((path / "_manifest.json").read_text())
        initial_sidecar_name = (initial_manifest.get("column_metadata") or {}).get("file")
        if not isinstance(initial_sidecar_name, str) or Path(initial_sidecar_name).name != initial_sidecar_name:
            raise ValueError("Export time metadata is missing or invalid.")
        initial_sidecar_identity = source_file_identity(path / initial_sidecar_name)
        item, id_column, unit, source_path, outcome_file, sidecar_file = _binding(path, desc, feature, landmark_hours)
        lab_path = path / item["file"]
        if lab_path.is_symlink() or not lab_path.is_file():
            raise ValueError("The selected laboratory file is unavailable or ambiguous.")
        watched = [path / "_manifest.json", path / sidecar_file, source_path, path / outcome_file, lab_path]
        before = {str(p): source_file_identity(p) for p in watched}
        if (before[str(path / "_manifest.json")] != manifest_identity or sidecar_file != initial_sidecar_name
                or before[str(path / sidecar_file)] != initial_sidecar_identity):
            raise ValueError("The source files changed during landmark verification; try again.")
        events, times = _verify_eicu_outcomes(path, source_path, outcome_file, entity_ids, id_column)
        import pyarrow.parquet as pq
        import pyarrow.types as pat
        parquet = pq.ParquetFile(lab_path)
        if parquet.metadata.num_rows > _MAX_ROWS:
            raise ValueError("The laboratory file exceeds the interactive landmark preview limit.")
        schema = parquet.schema_arrow
        arrow_type = schema.field("charttime").type
        if not (pat.is_integer(arrow_type) or pat.is_floating(arrow_type) or pat.is_duration(arrow_type)):
            raise ValueError("The laboratory time column is not a verified relative-hours representation.")
        wanted, earliest = set(entity_ids), {}
        for batch in _batches(lab_path, [id_column, "charttime", feature["column"]]):
            rows = batch.to_pydict()
            for raw_id, raw_hour, raw_value in zip(rows[id_column], rows["charttime"], rows[feature["column"]]):
                sid = entity_id_contract.normalize_entity_id(raw_id)
                if sid not in wanted or raw_hour is None or raw_value is None:
                    continue
                hour = raw_hour.total_seconds() / 3600 if hasattr(raw_hour, "total_seconds") else raw_hour
                try:
                    hour, value = float(hour), float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(hour) and 0 <= hour < landmark_hours and math.isfinite(value)):
                    continue
                prior = earliest.get(sid)
                if prior is None or hour < prior[0]:
                    earliest[sid] = (hour, [value])
                elif hour == prior[0]:
                    if len(prior[1]) >= _MAX_TIES:
                        raise ValueError("Too many measurements share the earliest landmark timestamp.")
                    prior[1].append(value)
        import statistics
        first = {sid: statistics.median(values) for sid, (_, values) in earliest.items()}
        if any(source_file_identity(p) != before[str(p)] for p in watched):
            raise ValueError("The source files changed during landmark verification; try again.")
    except MetadataSidecarError:
        return _blocked("Export time metadata is missing or invalid.")
    except ValueError as exc:
        return _blocked(str(exc))
    except Exception:
        return _blocked("The export cannot verify or read the selected laboratory time and outcome metadata.")

    labels = [f"≤ {threshold:g}", f"> {threshold:g}"]
    members = [(labels[0], []), (labels[1], [])]
    counts = {"cohort_total": len(entity_ids), "missing_followup": 0, "missing_event": 0,
              "death_at_or_before_landmark": 0, "discharged_at_or_before_landmark": 0,
              "missing_measurement": 0, "included": 0}
    shifted_events, shifted_times = {}, {}
    for sid in entity_ids:
        duration = times.get(sid)
        if duration is None or not math.isfinite(duration) or duration < 0:
            counts["missing_followup"] += 1
        elif sid not in events:
            counts["missing_event"] += 1
        elif duration * 24 <= landmark_hours:
            key = "death_at_or_before_landmark" if events[sid] else "discharged_at_or_before_landmark"
            counts[key] += 1
        elif sid not in first:
            counts["missing_measurement"] += 1
        else:
            members[0 if first[sid] <= threshold else 1][1].append(sid)
            counts["included"] += 1
            shifted = duration - landmark_hours / 24
            shifted_events[sid] = bool(events[sid] and shifted <= 30)
            shifted_times[sid] = min(shifted, 30.0)
    ready = all(ids for _, ids in members)
    group = {"basis": "icu_landmark_first_measurement", "status": "ready" if ready else "blocked",
             "reason": None if ready else "The landmark risk set does not produce two non-empty groups.",
             "groups": [{"label": label, "count": len(ids)} for label, ids in members],
             "_members": members if ready else [], "supported_outcomes": ["icu_death"] if ready else [],
             "landmark_hours": landmark_hours, "measurement_unit": unit, "landmark_counts": counts}
    details = {"landmark_hours": landmark_hours, "measurement_window_hours": [0, landmark_hours],
               "measurement_window_end_inclusive": False, "measurement_aggregation": "first",
               "measurement_unit": unit, "landmark_counts": counts,
               "event_time_basis": "icu_discharge_status_and_exit_time_proxy",
               "supported_outcomes": ["icu_death"]}
    return {"group": group, "vectors": (shifted_events, shifted_times), "details": details}
