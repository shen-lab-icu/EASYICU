"""Bounded native event production and publication; no grid consolidation."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..concept.native_events import NATIVE_EVENT_MODULES


def run_event_extraction(*, module: str, database: str, data_path: str,
                         patient_ids: dict, batch_size: int | None, output_dir: str) -> None:
    from ..concept.fluid_administrations import fluid_source_receipt, load_event_batch
    from ..config import load_src_cfg
    from ..datasource import ICUDataSource
    from ..resources import load_dictionary

    started = time.monotonic()
    spec = NATIVE_EVENT_MODULES[module]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    available = database in {"aumc", "miiv", "sic"}
    receipt = {
        "schema": "easyicu.native_fluid_administrations/1",
        "status": "AVAILABLE" if available else "NOT_AVAILABLE",
        "reason": ("native_administration_records" if available else
                   "native_administrations_unavailable_counter_sources_are_not_events" if database == "hirid" else
                   "native_fluid_administration_source_not_validated"),
        **fluid_source_receipt(load_dictionary(), database),
        "mode_policy": "coded_evidence_only_no_duration_inference",
        "time_precision_minutes": 1,
        "row_grain": spec.row_grain,
    }
    saved = {}
    rows = 0
    if available:
        source = ICUDataSource(load_src_cfg(database), base_path=data_path, enable_cache=False)
        ident = source.config.id_configs["icustay"].id
        ids = list(dict.fromkeys(patient_ids[ident]))
        size = max(1, int(batch_size or 10_000))
        schema = pa.Table.from_pandas(spec.empty(), preserve_index=False).schema
        path = root / f"{module}.parquet"
        with pq.ParquetWriter(path, schema, compression="snappy") as writer:
            for offset in range(0, len(ids), size):
                frame = load_event_batch(source, database=database, patient_ids={ident: ids[offset:offset + size]})
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
                rows += len(frame)
        saved[module] = {"path": str(path), "rows": rows, "concepts": [],
                         "concept_meta": {}, "merge_keys": list(spec.primary_key)}
    manifest = {"saved": saved, "errors": [], "warnings": [], "event_receipt": receipt,
                "elapsed_sec": time.monotonic() - started}
    (root / "_manifest.json").write_text(json.dumps(manifest))


def publish_event_table(*, source_parquet: Path, temporary_parquet: Path, module: str,
                        stay_time_upper_bounds: dict[int, float]) -> dict:
    """Validate and copy event batches while preserving every native record."""
    from ..utils.time_units import ICU_TIME_FALLBACK_LIMIT_HOURS, ICU_TIME_PRE_ADMISSION_HOURS

    spec = NATIVE_EVENT_MODULES[module]
    schema_frame = spec.empty()
    schema = pa.Table.from_pandas(schema_frame, preserve_index=False).schema
    rows = 0
    excluded = 0
    batches = pq.ParquetFile(source_parquet).iter_batches(batch_size=100_000) if source_parquet.is_file() else ()
    with pq.ParquetWriter(temporary_parquet, schema, compression="snappy") as writer:
        for batch in batches:
            frame = spec.canonical(batch.to_pandas())
            if not np.isfinite(frame[["start", "end", "volume_ml"]]).all().all():
                raise ValueError("native event numeric fields must be finite")
            if (frame.end < frame.start).any() or (frame.volume_ml <= 0).any():
                raise ValueError("native event duration or volume is invalid")
            upper = frame.stay_id.map(stay_time_upper_bounds).fillna(ICU_TIME_FALLBACK_LIMIT_HOURS)
            keep = frame.start.ge(-ICU_TIME_PRE_ADMISSION_HOURS) & frame.start.le(upper)
            excluded += int((~keep).sum())
            frame = frame.loc[keep]
            rows += len(frame)
            writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
    return {
        "rows": rows, "schema_frame": schema_frame, "concept_non_null": {},
        "bounds_audit": {}, "semantic_audit": {},
        "time_axis_audit": {"policy": "native_event_start_in_icu_episode_with_24h_pre_post_allowance",
                            "excluded_rows": excluded, "end_policy": "native_end_preserved"},
        "row_grain_audit": {"primary_key": list(spec.primary_key), "row_grain": spec.row_grain,
                            "published_rows": rows, "publication_backend": "arrow_native_events",
                            "consolidated_rows": 0},
    }
