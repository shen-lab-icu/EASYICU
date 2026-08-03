"""Full-database extraction services.

This module owns worker isolation, bounds enforcement, grouped extraction, and
native-v2 export publication. The public API module only re-exports the stable
entry points.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..base import BaseICULoader, detect_database_type, get_default_data_path
from ..concept.catalog import CONCEPT_GROUPS_INTERNAL
from ..config import DATABASE_ID_CONFIG
from ..resources import load_dictionary
from .cohort import get_all_patient_ids_impl
from .concepts import (
    _concepts_need_sofa2,
    _normalize_patient_ids_for_db,
    _sample_patient_ids,
)


def _get_all_patient_ids(
    data_path: Union[str, Path],
    database: Optional[str] = None,
    max_patients: Optional[int] = None,
) -> tuple[List, str]:
    """Resolve extraction IDs through the fail-closed cohort service."""
    return get_all_patient_ids_impl(
        data_path,
        database_id_config=DATABASE_ID_CONFIG,
        detect_database_type_fn=detect_database_type,
        base_loader_cls=BaseICULoader,
        sample_patient_ids_fn=_sample_patient_ids,
        database=database,
        max_patients=max_patients,
    )


# ============================================================================
# 全库提取 API — 按模块子进程隔离与实测内存自适应
# ============================================================================

# Module definitions are derived from the shared web/export catalog so the
# public extract_database() API cannot drift from the 19-module full export.
EXTRACT_MODULES: Dict[str, List[str]] = {
    module: list(concepts) for module, concepts in CONCEPT_GROUPS_INTERNAL.items()
}

# Fast-to-slow preferred order. Unknown future modules are appended below.
_PREFERRED_EXTRACT_MODULE_ORDER: List[str] = [
    "vitals",
    "demographics",
    "outcome",
    "blood_gas",
    "chemistry",
    "hematology",
    "ventilator",
    "respiratory",
    "vasopressors",
    "medications",
    "neurological",
    "renal",
    "circulatory",
    "other_scores",
    "sepsis_shared",
    "sofa1_score",
    "sofa2_score",
    "sepsis3_sofa1",
    "sepsis3_sofa2",
]
EXTRACT_MODULE_ORDER: List[str] = [
    module for module in _PREFERRED_EXTRACT_MODULE_ORDER if module in EXTRACT_MODULES
] + [
    module
    for module in EXTRACT_MODULES
    if module not in _PREFERRED_EXTRACT_MODULE_ORDER
]

# Production cohort boundary used only to admit a calibrated one-shot fast
# path on sufficiently large-memory hosts.  Crossing it always uses streaming.
ONESHOT_MAX_PATIENTS = 150_000

# Streamed export planning is deliberately expressed as a continuous capacity
# model, not a handful of RAM tiers.  ``available`` is memory the OS says can
# be used now, so keep a meaningful reserve for the parent process, Arrow,
# DuckDB, and applications that remain open on a laptop.  The remaining
# working set is converted to stays with a conservative cross-module planning
# coefficient, rounded down, and capped at the production-proven 67k batch.
#
# At 8 GiB available the generic ceiling is 40k stays:
#   available 8192 - reserve 2048 = 6144 MiB
#   floor(6144 / 0.15 / 5000) * 5000 = 40,000
#
# A database-specific release calibration below may select a smaller initial
# pilot when the observed source/score working set is heavier; it is not a
# fixed RAM tier and later batches still adapt from measurement.  Explicit
# ``batch_size`` remains authoritative.
_STREAM_BATCH_RESERVE_FRACTION = 0.25
_STREAM_BATCH_MIN_RESERVE_MB = 2 * 1024
_STREAM_BATCH_MB_PER_STAY = 0.15
_STREAM_BATCH_QUANTUM = 5_000
_STREAM_BATCH_MIN = 5_000
_STREAM_BATCH_MAX = 67_000
_STREAM_BATCH_RETRY_FACTOR = 0.75
_STREAM_BATCH_MAX_RETRIES = 3

# A full-cohort one-shot is not admitted merely because the generic
# per-stay formula happens to cover the cohort.  The 2026-08-03 full-six run
# showed that source-table shape and score-grid construction dominate that
# simple model for several databases: MIMIC-III 61,532 stays reached about
# 16.83 GiB process-tree RSS, AUMC 23,106 stays reached about 29.31 GiB, and
# one 67k eICU batch reached about 15.6 GiB in ``other_scores``.  Use rounded-up
# release measurements as conservative *initial-pilot* references.  Every
# module can still grow after its first measured batch, so this does not create
# a fixed low-memory tier.
_STREAM_ONESHOT_MIN_AVAILABLE_MB = 24 * 1024
_STREAM_CALIBRATION_QUANTUM = 1_000
_STREAM_UNMEASURED_ONESHOT_GUARD_DATABASES = frozenset(
    {"mimic", "miiv", "aumc"}
)
_STREAM_CALIBRATED_REFERENCE = {
    # database: (observed stays, conservative process-tree peak MiB)
    "mimic": (61_532, 18 * 1024),
    "miiv": (94_458, 15 * 1024),
    "eicu": (67_000, 16 * 1024),
    "aumc": (23_106, 30 * 1024),
    "hirid": (33_905, 14 * 1024),
    "sic": (27_386, 10 * 1024),
}


def _normalise_stream_database(database: str) -> str:
    """Return the canonical stream-planning database name."""

    normalized = str(database).strip().lower()
    return {
        "mimiciii": "mimic",
        "mimic-iii": "mimic",
        "mimiciv": "miiv",
        "mimic-iv": "miiv",
    }.get(normalized, normalized)


def _stream_calibration(database: str) -> Optional[tuple[int, float]]:
    """Return the conservative release calibration for one database alias."""

    return _STREAM_CALIBRATED_REFERENCE.get(
        _normalise_stream_database(database)
    )


def _quantize_stream_capacity(capacity: float, quantum: int) -> int:
    """Round a positive stay capacity down without falling below 5k."""

    quantized = (max(0, int(capacity)) // int(quantum)) * int(quantum)
    return max(_STREAM_BATCH_MIN, min(_STREAM_BATCH_MAX, quantized))


def _process_tree_rss_mb() -> float:
    """Return current RSS for this process tree without making psutil mandatory."""
    try:
        import psutil

        root = psutil.Process(os.getpid())
        processes = [root, *root.children(recursive=True)]
        rss = 0
        seen = set()
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                rss += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss / (1024.0**2)
    except Exception:
        try:
            import resource
            import sys

            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return peak / (1024.0**2 if sys.platform == "darwin" else 1024.0)
        except Exception:
            return 0.0


def _available_memory_mb() -> float:
    try:
        from ..runtime.memory_manager import get_available_memory_mb

        return max(0.0, float(get_available_memory_mb()))
    except Exception:
        return 0.0


class _RSSPeakSampler:
    """Low-overhead module/batch RSS sampler used for adaptive planning."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.start_rss_mb = 0.0
        self.peak_rss_mb = 0.0
        self.available_memory_mb_at_start = 0.0
        self._stop_event = None
        self._thread = None

    def _sample(self) -> None:
        self.peak_rss_mb = max(self.peak_rss_mb, _process_tree_rss_mb())

    def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self):
        import threading

        self.start_rss_mb = _process_tree_rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self.available_memory_mb_at_start = _available_memory_mb()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="easyicu-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> Dict[str, float]:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval_seconds * 3))
        self._sample()
        return {
            "start_rss_mb": round(self.start_rss_mb, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_working_set_mb": round(
                max(0.0, self.peak_rss_mb - self.start_rss_mb),
                1,
            ),
            "available_memory_mb_at_start": round(
                self.available_memory_mb_at_start,
                1,
            ),
        }


def _adapt_stream_batch_size_from_first_batch(
    current_batch_size: int,
    *,
    observed_working_set_mb: float,
    available_memory_mb: float,
    remaining_patients: int,
) -> int:
    """Scale later batches from the first batch's measured working set.

    The same reserve contract used for launch planning remains in force.
    Growth is capped at the production-proven 67k and rounded to 5k, except
    that the exact 67k cap remains reachable.  A failed/too-small measurement
    leaves the original plan unchanged.
    """
    current = max(1, int(current_batch_size))
    remaining = max(0, int(remaining_patients))
    observed = max(0.0, float(observed_working_set_mb))
    available = max(0.0, float(available_memory_mb))
    if remaining <= 0 or observed < 64.0 or available <= 0:
        return min(current, remaining) if remaining else current

    reserve = max(
        float(_STREAM_BATCH_MIN_RESERVE_MB),
        available * _STREAM_BATCH_RESERVE_FRACTION,
    )
    usable = max(0.0, available - reserve)
    if usable <= 0:
        return min(current, remaining)

    measured_capacity = int(current * usable / observed)
    if measured_capacity >= _STREAM_BATCH_MAX:
        planned = _STREAM_BATCH_MAX
    else:
        planned = (
            measured_capacity // _STREAM_BATCH_QUANTUM
        ) * _STREAM_BATCH_QUANTUM
        planned = max(_STREAM_BATCH_MIN, planned)
    planned = min(_STREAM_BATCH_MAX, planned)
    return min(planned, remaining)


def _resolve_stream_batch_size(
    database: str,
    num_patients: int,
    requested_batch_size: Optional[int] = None,
    *,
    available_memory_mb: Optional[float] = None,
) -> int:
    """Choose a streamed-export batch from *currently available* memory.

    Total RAM is not a sufficient safety signal on a laptop: a nominal 16 GB
    machine may have only 8 GB available while an IDE, browser, or another
    analysis is open.  Explicit user choices always win.

    Automatic batches reserve 25% of currently available memory (at least
    2 GiB), then combine the generic 0.15-MiB-per-stay capacity with a
    conservative database calibration from the latest full-six run.  The
    initial pilot is rounded down, bounded to 5k--67k stays, and then resized
    from each module's first measured working set.  This preserves large
    batches where measurements support them without a fixed 10k tier.

    Below 24 GiB available memory, MIMIC-III, MIMIC-IV and AUMC cannot use an
    unmeasured one-shot fast path.  Lower-risk calibrated cohorts such as SIC
    and HiRID may remain one-shot when their conservative full-cohort peak fits
    the post-reserve budget.  When the high-risk guard alone requires a split,
    it starts from an even half rather than manufacturing a tiny residual
    batch.  Streamed cohorts are interleaved across the source-order range
    before chunking so a dense late eICU era is not concentrated into the
    final batch.  Explicit user choices remain authoritative.
    """

    total = int(num_patients)
    if total <= 0:
        raise ValueError("num_patients must be positive")

    if requested_batch_size is not None:
        requested = int(requested_batch_size)
        if requested <= 0:
            raise ValueError("batch_size must be positive")
        return min(requested, total)

    if available_memory_mb is None:
        from ..runtime.memory_manager import get_available_memory_mb

        available_memory_mb = get_available_memory_mb()
    available = max(0.0, float(available_memory_mb))

    reserve = max(
        float(_STREAM_BATCH_MIN_RESERVE_MB),
        available * _STREAM_BATCH_RESERVE_FRACTION,
    )
    usable = max(0.0, available - reserve)
    capacity = _quantize_stream_capacity(
        usable / _STREAM_BATCH_MB_PER_STAY,
        _STREAM_BATCH_QUANTUM,
    )

    calibration = _stream_calibration(database)
    calibrated_full_peak_mb = None
    if calibration is not None:
        reference_stays, reference_peak_mb = calibration
        calibrated_capacity = _quantize_stream_capacity(
            reference_stays * usable / max(1.0, float(reference_peak_mb)),
            _STREAM_CALIBRATION_QUANTUM,
        )
        capacity = min(capacity, calibrated_capacity)
        calibrated_full_peak_mb = (
            float(reference_peak_mb) * total / max(1, reference_stays)
        )

    guarded_unmeasured_one_shot = (
        available < _STREAM_ONESHOT_MIN_AVAILABLE_MB
        and _normalise_stream_database(database)
        in _STREAM_UNMEASURED_ONESHOT_GUARD_DATABASES
    )
    if total <= ONESHOT_MAX_PATIENTS and not guarded_unmeasured_one_shot:
        if calibrated_full_peak_mb is None:
            one_shot_fits = capacity >= total
        else:
            one_shot_fits = calibrated_full_peak_mb <= usable
        if one_shot_fits:
            return total

    if guarded_unmeasured_one_shot and capacity >= total:
        capacity = (total + 1) // 2
    return min(capacity, total)


def _next_stream_retry_batch_size(current_batch_size: int) -> int:
    """Return the next bounded batch after one adaptive worker crash."""

    current = max(_STREAM_BATCH_MIN, int(current_batch_size))
    proposed = int(current * _STREAM_BATCH_RETRY_FACTOR)
    proposed = (proposed // _STREAM_BATCH_QUANTUM) * _STREAM_BATCH_QUANTUM
    if proposed >= current:
        proposed = current - _STREAM_BATCH_QUANTUM
    return max(_STREAM_BATCH_MIN, proposed)


def _interleave_stream_patient_ids(
    patient_ids: List,
    batch_size: int,
) -> tuple[List, int]:
    """Deterministically spread source-order density across streamed batches.

    eICU event density increases materially across the source-ordered stay
    list: the last sequential ~67k respiratory stays exceeded a 14-GiB cgroup
    even in a fresh process, while three equally sized interleaved slices fit.
    Concatenating ``ids[offset::planned_batches]`` preserves every identifier
    exactly once and keeps the requested full batch size, but makes each large
    chunk sample the complete source-order range.  This avoids both a skewed
    final OOM and the much slower fallback to many tiny batches.

    Row order is not part of the native Parquet semantic contract; the order
    here is deterministic for a deterministic input cohort and is recorded in
    module telemetry.
    """

    ids = list(patient_ids)
    size = int(batch_size)
    if size < 1:
        raise ValueError("stream batch_size must be positive")
    planned_batches = (len(ids) + size - 1) // size if ids else 0
    if planned_batches <= 1:
        return ids, planned_batches
    interleaved = [
        patient_id
        for offset in range(planned_batches)
        for patient_id in ids[offset::planned_batches]
    ]
    return interleaved, planned_batches


def _get_extraction_mp_context(mp_module, *, platform_name: Optional[str] = None):
    """Resolve the extraction worker context with a cross-platform safe default.

    Extraction often follows Arrow/DuckDB conversion in the same Python
    process.  ``fork`` then copies native allocator and thread-pool state into
    the child: on a production eICU run this inherited about 30 GB of parent
    state, ran 32% slower, and produced a stale single-source MAP result for
    495,371 rows.  ``spawn`` starts every worker from a clean interpreter and
    is already the only supported Windows process model, so use it consistently
    on Windows, macOS, and Linux.

    ``EASYICU_MP_START_METHOD`` remains an explicit expert override (for
    example, a controlled Linux-only benchmark).  ``platform_name`` is retained
    for compatibility with callers/tests that exercise platform contracts.
    """

    del platform_name
    default_start_method = "spawn"
    start_method = (
        os.environ.get("EASYICU_MP_START_METHOD", default_start_method)
        .strip()
        .lower()
    )
    available = mp_module.get_all_start_methods()
    if start_method not in available:
        raise ValueError(
            f"unsupported EASYICU_MP_START_METHOD={start_method!r}; "
            f"available methods: {available}"
        )
    return mp_module.get_context(start_method)

# 特殊概念 — 需要专用加载函数而非 load_concepts
_SPECIAL_CONCEPT_MODULES = {"sepsis3_sofa1", "sepsis3_sofa2"}

# 已知数据库路径映射（可被 data_paths 参数或环境变量 EASYICU_DATA_PATH 覆盖）
# 默认使用环境变量中的数据根目录
_DEFAULT_DB_PATH_CACHE: Dict[str, str] = {}


def _get_default_db_path(database: str) -> Optional[str]:
    """惰性解析单个数据库的默认路径（按需，带缓存）。

    旧实现在 import api.py 时就为全部 6 个库递归扫描目录。
    在慢速 FUSE 挂载上，每个 os.listdir 要数秒，且每个提取子进程
    import 时都重复付出这笔开销。改为按需解析、只扫描真正用到的库。
    """
    if database in _DEFAULT_DB_PATH_CACHE:
        return _DEFAULT_DB_PATH_CACHE[database]
    _root = os.environ.get("EASYICU_DATA_PATH", "")
    if not _root:
        return None
    try:
        from easyicu.io.data_paths import find_database_path

        path = find_database_path(_root, database)
    except ImportError:
        path = os.path.join(_root, database)
    _DEFAULT_DB_PATH_CACHE[database] = path
    return path


def _build_default_db_paths() -> Dict[str, str]:
    """解析全部 6 个数据库的默认路径（仅 extract_all_databases 使用）。"""
    return {
        db: p
        for db in ["sic", "aumc", "hirid", "mimic", "miiv", "eicu"]
        if (p := _get_default_db_path(db)) is not None
    }


# 特殊模块（Sepsis-3）在分组临时目录下的输出子目录名
_SPECIAL_OUTPUT_DIRNAME = "_special"


def _extract_worker_env_setup(data_path: str) -> None:
    """提取子进程入口的共享环境准备。

    本 worker 已是隔离子进程：模块退出后 OS 完整回收内存，模块间无碎片累积。
    因此模块内部应一次性 in-process 加载，绝不要让 load_concepts 再启动“每批
    子进程 fork”——每次 fork 都会重读共享源表(chartevents/labevents…)，是数倍
    慢的根源。强制 in-process，让模块内单次扫表。
    """
    import os
    import sys

    os.environ.setdefault("EASYICU_DATA_PATH", data_path)
    os.environ.setdefault("EASYICU_FORCE_INPROCESS_BATCH", "1")
    _src_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Concept-bounds enforcement (physiological plausibility clamp)
# ─────────────────────────────────────────────────────────────────────────────
# R ricu applies `clamp_var` (out-of-range raw value → NA) BEFORE hourly
# aggregation, then `filter_bounds` after. EasyICU's DuckDB aggregation path
# (`load_bucketed_table_aggregated` / `_multi_aggregated` / `_wide_aggregated` in
# datasource.py) deliberately SKIPS the raw min/max WHERE-filter whenever a
# `value_transform` or an inline unit-convert is present (the raw column may be a
# different unit or VARCHAR — see datasource.py L3040-3049, 3108-3110), and the
# "post-agg filter_bounds handled in concept.py" step those comments defer to was
# never implemented (there is no concept.py and no filter_bounds anywhere in the
# package). `_filter_concept_data` (load_concepts.py:1142) enforces min/max but is
# only reached by the deprecated interactive loader, NOT the batch-export path.
# Net effect: declared concept `min`/`max` in concept-dict.json are NOT enforced
# for numeric concepts in `extract_database`, so gross source errors survive into
# the export (observed in mimiciv: hr 1e7, map 9e6, sbp 1e6, resp 7e6, spo2 9.9e6,
# peep 8.77e6, glu 1.28e6, wbc 1e6, lact 1.28e6). This is the single
# post-aggregation enforcement point for the LONG per-concept (`merge=False`)
# export: for each extracted concept it drops rows whose (post-conversion,
# target-unit) value lies outside the concept's declared [min, max]. NaN/missing
# and categorical (text-only) values are preserved. Idempotent — a no-op on data
# that is already within bounds.
_CONCEPT_BOUNDS_CACHE = None


_BOUNDS_METADATA_KEYS = (
    "rows_before",
    "bounds_dropped",
    "bounds_dropped_post_aggregation",
    "bounds_count_status",
    "bounds_raw_transformed_non_null",
    "bounds_bounded_transformed_non_null",
    "bounds_bounded_aggregate_non_null",
    "bounds_unit_suspect",
    "bounds_unbounded_retry",
    "bounds_skipped",
    "bounds_status",
)


def _load_concept_bounds_map():
    """Return ``{concept_name: (min, max)}`` from the active concept dictionary.

    Only concepts with at least one finite declared bound are included. Bounds are
    in the concept's declared (target) unit, matching the post-conversion value the
    aggregation path produces. Cached after first load.
    """
    global _CONCEPT_BOUNDS_CACHE
    if _CONCEPT_BOUNDS_CACHE is not None:
        return _CONCEPT_BOUNDS_CACHE
    import os as _os
    import json as _json

    bounds = {}
    data_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "data",
    )
    dict_paths = [
        _os.path.join(data_dir, "concept-dict.json"),
        _os.path.join(data_dir, "sofa2-dict.json"),
    ]
    try:
        for dict_path in dict_paths:
            with open(dict_path) as _f:
                _d = _json.load(_f)
            for _name, _entry in _d.items():
                if not isinstance(_entry, dict):
                    continue
                _mn = _entry.get("min")
                _mx = _entry.get("max")
                _mn = float(_mn) if _mn is not None else None
                _mx = float(_mx) if _mx is not None else None
                if _mn is not None or _mx is not None:
                    bounds[_name] = (_mn, _mx)
    except Exception as exc:
        import warnings as _warnings

        _warnings.warn(
            f"Could not load concept bounds from {dict_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        bounds = {}
    _CONCEPT_BOUNDS_CACHE = bounds
    return bounds


def _bounds_metadata_from_manifest_info(info):
    """Subset persisted concept-bound audit fields from a worker manifest entry."""
    if not isinstance(info, dict):
        return {}
    return {k: info[k] for k in _BOUNDS_METADATA_KEYS if k in info}


def _attach_bounds_metadata(df, info):
    """Attach bounds audit metadata to an in-memory concept DataFrame."""
    meta = _bounds_metadata_from_manifest_info(info)
    if meta and hasattr(df, "attrs"):
        df.attrs["easyicu_bounds"] = meta
        for key, value in meta.items():
            df.attrs[f"easyicu_{key}"] = value
    return meta


def _concept_result_info(path, info):
    """Build the public output-dir concept entry, preserving bounds audit data."""
    out = {"path": path, "rows": info.get("rows", 0)}
    out.update(_bounds_metadata_from_manifest_info(info))
    return out


def _enforce_concept_bounds(df, concept_name):
    """Drop rows whose numeric value for ``concept_name`` is outside its declared
    [min, max]. The per-concept extraction DataFrame holds the value in a column
    named after the concept. NaN/missing and non-numeric (categorical) values are
    preserved. Returns ``(df, n_dropped)``.
    """
    import pandas as _pd

    if not isinstance(df, _pd.DataFrame) or concept_name not in df.columns:
        return df, 0
    bnd = _load_concept_bounds_map().get(concept_name)
    if bnd is None:
        return df, 0
    loader_diagnostics = df.attrs.get("easyicu_bounds_loader", {})
    if isinstance(loader_diagnostics, dict) and loader_diagnostics.get(
        "bounds_unit_suspect"
    ):
        # The SQL fast path saw at least 100 transformed non-null values but
        # none within the declared bounds. It already retried without bounds,
        # so retain those recovered values and surface the existing -1 signal.
        return df, -1
    mn, mx = bnd
    v = _pd.to_numeric(df[concept_name], errors="coerce")
    numeric = v.notna()
    # UNIT-SAFETY GUARD: if a concept has BOTH bounds and its central value (median)
    # falls outside [min,max], the values are almost certainly in the wrong unit for
    # this database (e.g. temperature still in Fahrenheit, median ~98 vs bounds
    # [32,42]). Bound-dropping would then delete valid-but-mis-united data, so SKIP
    # enforcement and leave the concept untouched (surfaced upstream as a WARN). A
    # correctly-united physiological concept always has its median well within bounds,
    # so this never suppresses legitimate outlier removal. Requires enough data to
    # make the median meaningful.
    if mn is not None and mx is not None and int(numeric.sum()) >= 100:
        med = float(v[numeric].median())
        if med < mn or med > mx:
            return (
                df,
                -1,
            )  # sentinel: enforcement SKIPPED (unit-suspect), nothing dropped
    in_range = _pd.Series(True, index=df.index)
    if mn is not None:
        in_range &= v >= mn
    if mx is not None:
        in_range &= v <= mx
    # keep non-numeric/missing rows (NaN is "missing", not "out of range") and
    # numeric rows that are within [min, max]; drop only genuine out-of-range values.
    keep = (~numeric) | in_range
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return df, 0
    return df.loc[keep].reset_index(drop=True), n_drop


def _module_parquet_columns(columns, concepts, *, include_missing=False):
    """Return stable context + catalog column order without touching payloads."""
    requested = list(dict.fromkeys(concepts))
    requested_set = set(requested)
    context_columns = [column for column in columns if column not in requested_set]
    concept_columns = (
        requested
        if include_missing
        else [column for column in requested if column in columns]
    )
    return context_columns + concept_columns


@lru_cache(maxsize=512)
def _module_arrow_storage_kind(concept: str) -> str:
    return _native_export_storage_kind(
        concept,
        load_dictionary(include_sofa2=True),
    )


def _module_arrow_null_type(concept: str, pyarrow_module):
    """Return the non-lossy physical type for an unavailable concept column."""
    kind = _module_arrow_storage_kind(str(concept))
    if kind == "boolean":
        return pyarrow_module.bool_()
    if kind == "string":
        return pyarrow_module.string()
    # Use float64 for a concept whose first observed batch is empty.  A later
    # float32 value can widen losslessly; choosing float32 here could silently
    # downcast a later high-precision producer.
    return pyarrow_module.float64()


def _module_arrow_table(
    frame,
    concepts,
    pyarrow_module,
    *,
    module: str,
    schema=None,
):
    """Create a stable module table, adding structural nulls only in Arrow."""
    table = pyarrow_module.Table.from_pandas(frame, preserve_index=False)
    requested = list(dict.fromkeys(concepts))
    requested_set = set(requested)

    # Every non-demographics module is longitudinal in the native-v2 physical
    # contract, even when a particular patient batch has no timestamped event.
    # If the first streamed batch omitted ``charttime``, its Arrow schema used
    # to omit it too; later batches were then projected onto that first schema
    # and their real timestamps were silently discarded.  Establish the time
    # field before the first writer schema is frozen and keep it float64 in all
    # batches, including an all-null first batch.
    if module != "demographics":
        if "charttime" not in table.column_names:
            identity_positions = [
                table.column_names.index(column)
                for column in _NATIVE_EXPORT_ID_COLUMNS
                if column in table.column_names
            ]
            insert_at = max(identity_positions, default=-1) + 1
            table = table.add_column(
                insert_at,
                pyarrow_module.field("charttime", pyarrow_module.float64()),
                pyarrow_module.nulls(len(table), type=pyarrow_module.float64()),
            )
        elif table.schema.field("charttime").type != pyarrow_module.float64():
            charttime_index = table.column_names.index("charttime")
            try:
                charttime = table.column("charttime").cast(pyarrow_module.float64())
            except (TypeError, ValueError, pyarrow_module.ArrowInvalid) as exc:
                raise ValueError(
                    "module charttime must be numeric ICU-relative hours"
                ) from exc
            table = table.set_column(
                charttime_index,
                pyarrow_module.field("charttime", pyarrow_module.float64()),
                charttime,
            )
    context_columns = [
        column for column in table.column_names if column not in requested_set
    ]

    if schema is None:
        for concept in requested:
            if concept in table.column_names:
                continue
            table = table.append_column(
                concept,
                pyarrow_module.nulls(
                    len(table),
                    type=_module_arrow_null_type(concept, pyarrow_module),
                ),
            )
        return table.select(context_columns + requested)

    for field in schema:
        if field.name not in table.column_names:
            table = table.append_column(
                field,
                pyarrow_module.nulls(len(table), type=field.type),
            )
    table = table.select(schema.names)
    if table.schema != schema:
        table = table.cast(schema)
    return table


def _normalise_module_frame_for_parquet(result, concepts, *, reorder=True):
    """Return one module frame in a stable, parquet-writable representation.

    ``reorder=False`` lets the Arrow writer reorder column references without
    copying the dense pandas payload.  The default behavior remains a
    canonically ordered DataFrame for other callers.
    """
    import pandas as pd

    if not isinstance(result, pd.DataFrame) or result.empty:
        return None

    # This function sits at the peak-RSS boundary: ``result`` is the complete
    # dense module frame and Arrow is about to wrap it for parquet output.
    # An unconditional ``copy()`` used to keep two full pandas payloads alive
    # at exactly that point.  Export owns the frame and never reuses it, so
    # preserve the original object when its columns are already canonical.
    # Copy only for the uncommon duplicate/reorder cases.
    if result.columns.duplicated().any():
        result = result.loc[:, ~result.columns.duplicated()].copy()

    # Dedicated concept loaders are routed through sets internally
    # (KDIGO/circulatory/comorbidity/microbiology).  Their insertion order is
    # therefore affected by PYTHONHASHSEED and used to change the physical
    # Parquet schema across fresh processes and operating systems.  The module
    # catalog is the export contract: keep identifier/time columns in their
    # established order, then place requested concepts in catalog order.
    ordered_columns = _module_parquet_columns(result.columns, concepts)
    if reorder and list(result.columns) != ordered_columns:
        result = result.loc[:, ordered_columns].copy()

    # Indicator concepts can arrive as bool/float/NA object columns.  Arrow
    # cannot write that mixed representation, while genuine text columns must
    # stay text.
    for column in result.columns:
        if result[column].dtype == object:
            numeric = pd.to_numeric(result[column], errors="coerce")
            if bool((numeric.notna() | result[column].isna()).all()):
                result[column] = numeric
    return result


def _release_stream_batch_memory(
    pyarrow_module,
    *,
    trim_native_allocator: bool = True,
) -> None:
    """Return released batch buffers to the host before the next large batch.

    ``gc.collect()`` drops Python references but Arrow's memory pool and glibc
    may retain the freed pages. Across three eICU batches that made RSS look
    cumulative even though each batch was bounded. Arrow exposes a portable
    release hook; Linux additionally benefits from ``malloc_trim``. macOS and
    Windows use only the portable hook.
    """
    import gc
    import sys

    # DuckDB keeps an in-memory connection per worker thread.  Clearing the
    # Python loader drops DataFrames, but it does not release DuckDB's buffer
    # manager; on the third eICU batch that residual allocation can be the
    # difference between a bounded 67k batch and a cgroup OOM.  Closing here is
    # safe because this function is called only at explicit streamed-batch
    # boundaries; the next query lazily opens a fresh connection.
    try:
        from ..datasource import _close_duckdb_connections

        _close_duckdb_connections()
    except Exception:
        pass
    gc.collect()
    try:
        pyarrow_module.default_memory_pool().release_unused()
    except Exception:
        pass
    if trim_native_allocator and sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            pass


_VITAL_STREAM_DERIVED_CONCEPTS = (
    "pulse_pressure",
    "shock_index",
    "modified_shock_index",
    "diastolic_shock_index",
)


def _clear_stream_loader_caches(loader) -> None:
    if loader is None:
        # Streamed exports disable module grouping, so this helper normally
        # receives no explicitly shared loader.  ``load_concepts`` still owns
        # a process-global loader cache, however.  Returning here retained
        # cohort-scoped resolver state across patient batches; mixed
        # window/point concepts such as eICU ``dex`` then produced different
        # time grids for 45k versus 67k batches.
        from .concepts import clear_global_loader

        clear_global_loader()
        return
    try:
        loader.clear_cache()
        return
    except Exception:
        pass
    resolver = getattr(loader, "concept_resolver", None)
    if resolver is not None and hasattr(resolver, "drop_source_caches"):
        resolver.drop_source_caches()


def _attach_stream_derived_columns(base, addition, value_columns):
    """Attach derived values whose time keys are a subset of the base grid."""
    import numpy as np
    import pandas as pd

    if not isinstance(base, pd.DataFrame) or base.empty:
        return addition
    if not isinstance(addition, pd.DataFrame) or addition.empty:
        for column in value_columns:
            if column not in base.columns:
                base[column] = np.nan
        return base

    id_candidates = (
        "stay_id",
        "patientunitstayid",
        "icustay_id",
        "admissionid",
        "patientid",
        "CaseID",
    )
    key_columns = [
        column
        for column in (*id_candidates, "charttime")
        if column in base.columns and column in addition.columns
    ]
    if not key_columns or "charttime" not in key_columns:
        raise ValueError("streamed derived concept has no shared ICU time key")

    base_index = pd.MultiIndex.from_frame(base[key_columns])
    addition_index = pd.MultiIndex.from_frame(addition[key_columns])
    if not base_index.is_unique or not addition_index.is_unique:
        return base.merge(
            addition[key_columns + list(value_columns)],
            on=key_columns,
            how="outer",
            sort=False,
        )
    indexer = base_index.get_indexer(addition_index)
    if bool((indexer < 0).any()):
        return base.merge(
            addition[key_columns + list(value_columns)],
            on=key_columns,
            how="outer",
            sort=False,
        )

    for column in value_columns:
        values = np.full(len(base), np.nan, dtype="float64")
        if column in addition.columns:
            values[indexer] = pd.to_numeric(
                addition[column],
                errors="coerce",
            ).to_numpy()
        base[column] = values
    return base


def _load_stream_module_batch(
    load_concepts_fn,
    *,
    module_name: str,
    concepts: List[str],
    load_kwargs: Dict,
    patient_ids: Dict,
    loader,
    pyarrow_module,
):
    """Load one patient batch with bounded recursive-vitals intermediates."""
    requested_derived = [
        concept
        for concept in _VITAL_STREAM_DERIVED_CONCEPTS
        if concept in concepts
    ]
    if module_name != "vitals" or not requested_derived:
        return load_concepts_fn(**load_kwargs, patient_ids=patient_ids)

    base_concepts = [
        concept for concept in concepts if concept not in requested_derived
    ]
    base_kwargs = dict(load_kwargs)
    base_kwargs["concepts"] = base_concepts
    result = load_concepts_fn(**base_kwargs, patient_ids=patient_ids)

    for concept in requested_derived:
        _clear_stream_loader_caches(loader)
        _release_stream_batch_memory(pyarrow_module)
        derived_kwargs = dict(load_kwargs)
        derived_kwargs["concepts"] = [concept]
        derived = load_concepts_fn(
            **derived_kwargs,
            patient_ids=patient_ids,
        )
        result = _attach_stream_derived_columns(
            result,
            derived,
            [concept],
        )
        del derived
    _clear_stream_loader_caches(loader)
    _release_stream_batch_memory(pyarrow_module)
    return result


def _stream_module_batches_to_parquet(
    module_name: str,
    concepts: List[str],
    load_kwargs: Dict,
    patient_ids_filter: Dict,
    batch_size: int,
    output_dir: str,
    *,
    loader=None,
    adaptive_batch_growth: bool = False,
) -> Optional[Dict]:
    """Append bounded patient batches directly to one module parquet file.

    This is the constrained-host export path.  It deliberately trades repeated
    source scans for a hard resident-memory boundary: no full module DataFrame
    and no final ``concat`` are materialised in the worker.  The temporary
    partial file lives beside the eventual module output, so callers that put
    their output on an external disk never use the system volume for it.
    """
    import os
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    from easyicu import load_concepts as _lc

    if not patient_ids_filter or len(patient_ids_filter) != 1:
        raise ValueError("streamed module export requires one patient-id filter")
    id_col, all_ids = next(iter(patient_ids_filter.items()))
    if batch_size < 1:
        raise ValueError("streamed module export batch_size must be positive")
    all_ids, planned_partition_count = _interleave_stream_patient_ids(
        list(all_ids),
        int(batch_size),
    )

    destination = Path(output_dir) / f"{module_name}.parquet"
    partial = destination.with_name(f".{module_name}.partial.parquet")
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"refusing stale streamed module partial: {partial}")

    writer = None
    schema = None
    rows = 0
    produced_concepts = set()
    batch_telemetry = []
    current_batch_size = int(batch_size)
    batch_load_kwargs = dict(load_kwargs)
    batch_load_kwargs.pop("patient_ids", None)
    try:
        start = 0
        while start < len(all_ids):
            table = None
            batch_ids = all_ids[start : start + current_batch_size]
            # Keep the inner ``load_concepts`` boundary identical to the outer
            # writer boundary.  After first-batch adaptation, retaining the
            # original value here made an apparent 40k -> 67k growth execute as
            # hidden 40k + 27k inner loads and repeated the expensive source
            # scans that the larger outer batch was meant to avoid.
            batch_load_kwargs["batch_size"] = len(batch_ids)
            batch_sampler = _RSSPeakSampler().start()
            frame = None
            output_rows = 0
            try:
                batch = _load_stream_module_batch(
                    _lc,
                    module_name=module_name,
                    concepts=concepts,
                    load_kwargs=batch_load_kwargs,
                    patient_ids={id_col: batch_ids},
                    loader=loader,
                    pyarrow_module=pa,
                )
                frame = _normalise_module_frame_for_parquet(
                    batch,
                    concepts,
                    reorder=False,
                )
                if frame is not None:
                    produced_concepts.update(
                        concept for concept in concepts if concept in frame.columns
                    )
                    table = _module_arrow_table(
                        frame,
                        concepts,
                        pa,
                        module=module_name,
                        schema=schema,
                    )
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(
                            partial,
                            schema,
                            compression="snappy",
                        )
                    writer.write_table(table)
                    output_rows = len(frame)
                    rows += output_rows
            finally:
                batch_memory = batch_sampler.stop()

            batch_telemetry.append(
                {
                    "batch_index": len(batch_telemetry) + 1,
                    "start_offset": start,
                    "stays": len(batch_ids),
                    "inner_load_batch_size": int(
                        batch_load_kwargs["batch_size"]
                    ),
                    "output_rows": output_rows,
                    **batch_memory,
                }
            )
            del table
            del batch, frame
            _clear_stream_loader_caches(loader)
            _release_stream_batch_memory(pa)
            start += len(batch_ids)
            if (
                adaptive_batch_growth
                and len(batch_telemetry) == 1
                and start < len(all_ids)
            ):
                current_batch_size = _adapt_stream_batch_size_from_first_batch(
                    current_batch_size,
                    observed_working_set_mb=batch_memory["peak_working_set_mb"],
                    available_memory_mb=batch_memory[
                        "available_memory_mb_at_start"
                    ],
                    remaining_patients=len(all_ids) - start,
                )
        if writer is None:
            return None
        writer.close()
        writer = None
        os.replace(partial, destination)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise

    return {
        "path": str(destination),
        "rows": rows,
        "concepts": [name for name in concepts if name in produced_concepts],
        "stream_batches": batch_telemetry,
        "initial_batch_size": int(batch_size),
        "final_planned_batch_size": current_batch_size,
        "adaptive_batch_growth": bool(adaptive_batch_growth),
        "patient_partition_strategy": "source_order_interleaved_v1",
        "initial_planned_partition_count": planned_partition_count,
    }


def _run_module_extraction(
    module_name: str,
    concepts: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict],
    batch_size: Optional[int],
    output_dir: str,
    use_sofa2: bool = False,
    loader=None,
    stream_output_batches: bool = False,
    adaptive_stream_batches: bool = False,
) -> None:
    """加载一个模块的所有概念并写入 parquet + _manifest.json。

    在 worker 子进程内运行；``loader`` 由分组 worker 传入，用于 OOM 降级
    重试前先清掉共享缓存释放内存。
    """
    import json
    import os
    import time
    import traceback
    import pandas as pd
    from easyicu import load_concepts as _lc

    t0 = time.time()
    module_memory_sampler = _RSSPeakSampler().start()
    saved = {}
    errors = []
    stream_info = None

    # 构造 load_concepts 参数
    # use_sofa2 显式传入：分组模式下保持全组 loader 配置一致，
    # 避免 sofa2 自动检测切换字典时重建 loader、丢掉组内共享缓存。
    warnings = []
    kwargs = dict(
        data_path=data_path,
        database=database,
        concepts=concepts,
        verbose=False,
        merge=True,
        concept_workers=1,
        use_sofa2=use_sofa2,
        _defer_empty_columns_to_arrow=True,
    )
    if patient_ids_filter:
        kwargs["patient_ids"] = patient_ids_filter

    # ── 一个模块一次 load、合并成一个宽表、写一个 {module}.parquet（不重复 io）──
    # load_concepts 一次拿到该模块**所有概念**（chartevents/labevents 等共享源表只扫
    # 一次；内部若按患者分批也由它自己 concat，对外仍是一次调用、一次扫描）。
    #
    # 分批策略：**除超大队列外一律一次性**。只有患者数 > ONESHOT_MAX_PATIENTS（15万，
    # 实际只有 eICU ~20万命中）才让 auto_batch_size 以 ≤ MAX_EXTRACT_CHUNKS（默认 3）份
    # 启用。实测最重非 eICU 模块 miiv medications（49 概念 × 9.4万患者）merge=True 一次性
    # 峰值仅 5.44GB，远低于预算；旧内存估算器约 3-5× 高估会把这类模块误判成要分批（见
    # web 端 dataio.py:1657 的同款观察），故对 ≤15万 的库直接跳过估算、强制一次性。
    _n_ids = 0
    if patient_ids_filter:
        try:
            _n_ids = len(next(iter(patient_ids_filter.values())))
        except Exception:
            _n_ids = 0
    if _n_ids > ONESHOT_MAX_PATIENTS and (not batch_size or batch_size >= _n_ids):
        try:
            from easyicu.runtime.memory_manager import auto_batch_size as _auto_bs

            # 稳定预算：用物理总内存判定（而非波动的当前可用），避免后台程序临时吃内存
            # 把本可一次性的模块误判成分批。EASYICU_ONESHOT_BUDGET_MB 可覆盖此上限(MB)。
            _stable_avail_mb = None
            _env_budget = os.environ.get("EASYICU_ONESHOT_BUDGET_MB")
            if _env_budget:
                _stable_avail_mb = float(_env_budget) / 0.6
            else:
                try:
                    import psutil as _ps

                    _stable_avail_mb = _ps.virtual_memory().total / (1024 * 1024)
                except Exception:
                    _stable_avail_mb = None
            _safe_bs = _auto_bs(
                list(concepts), database, _n_ids, available_memory_mb=_stable_avail_mb
            )
            if _safe_bs and _safe_bs < _n_ids:
                batch_size = _safe_bs
        except Exception:
            pass

    if batch_size:
        kwargs["batch_size"] = batch_size

    streamed = False
    result = None
    try:
        if stream_output_batches:
            if not patient_ids_filter or not batch_size:
                raise ValueError(
                    "streamed module export requires patient_ids and batch_size"
                )
            streamed = True
            stream_info = _stream_module_batches_to_parquet(
                module_name,
                concepts,
                kwargs,
                patient_ids_filter,
                int(batch_size),
                output_dir,
                loader=loader,
                adaptive_batch_growth=adaptive_stream_batches,
            )
            if stream_info is not None:
                saved[module_name] = stream_info
        else:
            result = _lc(**kwargs)
    except MemoryError:
        traceback.print_exc()
        if streamed:
            errors.append(f"streamed module export exhausted memory: {module_name}")
            result = {}
        else:
            if loader is not None:
                try:
                    loader.concept_resolver.clear_table_cache()
                except Exception:
                    pass
            _n = 0
            try:
                _n = (
                    len(next(iter(patient_ids_filter.values())))
                    if patient_ids_filter
                    else 0
                )
            except Exception:
                _n = 0
            from easyicu.runtime.memory_manager import (
                MAX_EXTRACT_CHUNKS as _MAX_CH,
                _ceil_div as _cdiv,
            )

            fallback_bs = max(10000, _cdiv(_n, _MAX_CH)) if _n else 10000
            errors.append(
                f"{module_name}: one-shot OOM, retrying batched (batch_size={fallback_bs})"
            )
            kwargs["batch_size"] = fallback_bs
            try:
                result = _lc(**kwargs)
            except Exception as e:
                traceback.print_exc()
                errors.append(f"load_concepts({module_name}) batched: {e}")
                result = {}
    except Exception as e:
        traceback.print_exc()
        errors.append(
            f"{'streamed export' if streamed else 'load_concepts'}({module_name}): {e}"
        )
        result = {}

    # 写出：load_concepts(merge=True) 直接返回该模块宽表（id + time + 每概念一列），
    # 与 web 端(dataio.py)完全一致的成熟路径。**不再自造合并**——避免 endtime 列冲突、
    # 递归概念(oxygenation_index/adv_resp/ecmo…)一次性 load 爆内存、以及把含 numpy 的
    # 逐概念元数据塞进 manifest 导致 json.dump 崩溃等"手写合并"问题。生理边界在
    # load_concepts 内部按 filter_bounds 预聚合强制（与 web 端同一套）。
    if streamed:
        pass
    elif isinstance(result, pd.DataFrame) and len(result) > 0:
        try:
            result = _normalise_module_frame_for_parquet(
                result,
                concepts,
                reorder=False,
            )
            _cols = [c for c in concepts if c in result.columns]
            path = os.path.join(output_dir, f"{module_name}.parquet")
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = _module_arrow_table(
                result,
                concepts,
                pa,
                module=module_name,
            )
            pq.write_table(table, path, compression="snappy")
            saved[module_name] = {
                "path": path,
                "rows": len(result),
                "concepts": _cols,
            }
        except Exception as e:
            traceback.print_exc()
            errors.append(f"write({module_name}): {e}")
    elif isinstance(result, dict) and result:
        # merge=True 应始终返回 DataFrame；若意外返回 dict，大声记错而不静默丢数据。
        errors.append(
            f"{module_name}: merge=True returned a dict ({len(result)} concepts) unexpectedly; not written"
        )

    elapsed = time.time() - t0
    memory_stats = module_memory_sampler.stop()
    manifest = {
        "module": module_name,
        "saved": saved,
        "errors": errors,
        "warnings": warnings,
        "elapsed_sec": round(elapsed, 1),
        **memory_stats,
    }
    if stream_info is not None:
        manifest["stream_batches"] = stream_info.get("stream_batches", [])
        manifest["initial_batch_size"] = stream_info.get("initial_batch_size")
        manifest["final_planned_batch_size"] = stream_info.get(
            "final_planned_batch_size"
        )
        manifest["adaptive_batch_growth"] = stream_info.get(
            "adaptive_batch_growth",
            False,
        )
        manifest["patient_partition_strategy"] = stream_info.get(
            "patient_partition_strategy"
        )
        manifest["initial_planned_partition_count"] = stream_info.get(
            "initial_planned_partition_count"
        )
    with open(os.path.join(output_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f)


def _extract_module_worker(
    concepts: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    output_dir: str = "",
    module_name: str = "",
):
    """（兼容包装）单模块子进程入口。

    新的默认入口是 _extract_module_group_worker（组内共享源表扫描）；
    保留此包装以兼容仍按单模块 spawn 的旧调用方。
    """
    _extract_worker_env_setup(data_path)
    _run_module_extraction(
        module_name,
        concepts,
        database,
        data_path,
        patient_ids_filter,
        batch_size,
        output_dir,
    )


def _require_timed_positive_suspicion(
    frame,
    *,
    id_col: str,
    time_col: str,
    database: str,
) -> None:
    """Fail closed when a positive suspected-infection event has no time.

    Sepsis-3 applies a SOFA-delta window around the suspected-infection event,
    so a positive ``susp_inf`` value is not meaningful without an event time.
    Native-v2 always materialises a ``charttime`` field, including as an
    all-null structural column; checking column presence alone is therefore
    insufficient.  This validation is deliberately scoped to ``susp_inf`` and
    does not reject stay-level support fields such as ``infection_icd``.
    """
    if "susp_inf" not in frame.columns:
        raise ValueError("Sepsis dependency lacks susp_inf")

    positive = frame["susp_inf"].eq(True).fillna(False)
    if not bool(positive.any()):
        return
    if time_col not in frame.columns:
        raise ValueError(
            f"{database} Sepsis dependency has positive susp_inf rows but lacks "
            f"the required time column '{time_col}'"
        )

    missing_time = positive & frame[time_col].isna()
    if not bool(missing_time.any()):
        return
    sample_ids = (
        frame.loc[missing_time, id_col].drop_duplicates().head(5).tolist()
        if id_col in frame.columns
        else []
    )
    raise ValueError(
        f"{database} Sepsis dependency has {int(missing_time.sum())} positive "
        f"susp_inf rows with null {time_col}; sample {id_col}={sample_ids}. "
        "A timed Sepsis-3 window cannot be derived from stay-level positives."
    )


def _stream_special_extraction_batches(
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Dict,
    batch_size: int,
    output_dir: str,
    *,
    use_sofa2: bool,
    published_output_dir: str,
) -> None:
    """Derive Sepsis labels from already-streamed dependency module artifacts.

    The old path asked ``load_concepts`` to merge ``susp_inf``, ``sofa`` and
    ``sofa2`` together, which can form a huge time-indexed intermediate even
    for a tiny requested cohort.  In constrained mode the three dependency
    modules have already been published one at a time.  Filter each parquet to
    one patient batch, derive the patient-local labels, and append them.
    """
    import json
    import os
    import time

    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if not patient_ids_filter or len(patient_ids_filter) != 1:
        raise ValueError("streamed special export requires one patient-id filter")
    id_col, all_ids = next(iter(patient_ids_filter.items()))
    # These inputs are already projected dependency parquets (id, time, one
    # score/flag), not raw SOFA source tables.  The former hard 2,000-stay cap
    # caused up to 101 full parquet filter passes for eICU even when the outer
    # streamed batch had safely handled 40k--67k stays.  Reuse that measured
    # outer boundary.  Experts can still impose a smaller independent cap.
    safe_batch_size = int(batch_size)
    raw_sepsis_batch = os.environ.get("EASYICU_SEPSIS_BATCH_SIZE")
    if raw_sepsis_batch:
        try:
            safe_batch_size = min(
                safe_batch_size,
                max(1, int(raw_sepsis_batch)),
            )
        except ValueError:
            pass
    all_ids, planned_partition_count = _interleave_stream_patient_ids(
        list(all_ids),
        safe_batch_size,
    )
    concepts = [
        concept
        for module_name in special_modules
        for concept in EXTRACT_MODULES.get(module_name, [])
    ]
    need_sofa1 = "sep3_sofa1" in concepts
    need_sofa2 = "sep3_sofa2" in concepts
    writers = {}
    partials = {}
    rows = {concept: 0 for concept in concepts}
    errors = []
    started = time.time()
    module_memory_sampler = _RSSPeakSampler().start()
    source_root = Path(published_output_dir)

    def _read_dependency(module_name: str, ids: List) -> "pd.DataFrame":
        source = source_root / f"{module_name}.parquet"
        if not source.is_file():
            # A normal module can complete successfully with zero rows (for
            # example strict suspected infection is structurally unavailable
            # in SIC).  Its collector publishes an empty, error-free module
            # manifest but deliberately leaves parquet placeholder creation to
            # the final native publisher.  Accept only that explicit state;
            # an absent/failed dependency must remain fail-closed.
            manifest_path = source_root / f"{module_name}.manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                manifest = None
            if not (
                isinstance(manifest, dict)
                and manifest.get("module") == module_name
                and not manifest.get("errors")
                and not manifest.get("saved")
            ):
                raise FileNotFoundError(
                    f"missing streamed dependency module: {source}"
                )
            return pd.DataFrame(
                columns=[id_col, *EXTRACT_MODULES.get(module_name, [])]
            )
        return (
            ds.dataset(source, format="parquet")
            .to_table(filter=ds.field(id_col).isin(ids))
            .to_pandas()
        )

    def _append_frame(concept: str, frame) -> None:
        if frame is None or frame.empty:
            return
        table = __import__("pyarrow").Table.from_pandas(frame, preserve_index=False)
        if concept not in writers:
            partial = Path(output_dir) / f".{concept}.partial.parquet"
            partials[concept] = partial
            writers[concept] = pq.ParquetWriter(
                partial, table.schema, compression="snappy"
            )
        writers[concept].write_table(table)
        rows[concept] += len(frame)

    def _suspicion_timeline(susp, time_col: str):
        """Return validated, event-timed suspected-infection flags."""
        _require_timed_positive_suspicion(
            susp,
            id_col=id_col,
            time_col=time_col,
            database=database,
        )
        if time_col not in susp.columns:
            return pd.DataFrame(columns=[id_col, time_col, "susp_inf"])
        return susp[[id_col, time_col, "susp_inf"]]

    try:
        for start in range(0, len(all_ids), safe_batch_size):
            ids = all_ids[start : start + safe_batch_size]
            susp = _read_dependency("sepsis_shared", ids)
            # No strict infection evidence means Sepsis-3 is structurally
            # unavailable, not a cohort-wide negative label.  Leave both
            # derived modules empty so the native publisher can emit typed
            # structural placeholders.
            if susp.empty:
                continue
            sofa1 = _read_dependency("sofa1_score", ids) if need_sofa1 else None
            sofa2 = _read_dependency("sofa2_score", ids) if need_sofa2 else None
            if "susp_inf" not in susp.columns:
                errors.append(
                    "streamed Sepsis dependency sepsis_shared lacks susp_inf"
                )
                continue

            def _score_time_column(score):
                return next(
                    (
                        name
                        for name in (
                            "charttime",
                            "time",
                            "starttime",
                            "datetime",
                            "Offset",
                            "measuredat_minutes",
                            "measuredat",
                        )
                        if name in score.columns
                    ),
                    None,
                )

            if need_sofa1 and sofa1 is not None and "sofa" in sofa1.columns:
                from ..scores.sepsis import sep3 as _sep3

                time_col = _score_time_column(sofa1)
                if time_col is None:
                    errors.append("streamed SOFA-1 dependency lacks a time index")
                else:
                    susp1 = _suspicion_timeline(susp, time_col)
                    frame = _sep3(
                        sofa1[[id_col, time_col, "sofa"]],
                        susp1,
                        id_cols=[id_col],
                        index_col=time_col,
                    ).rename(columns={"sep3": "sep3_sofa1"})
                    if "sep3_sofa1" in frame.columns:
                        frame["sep3_sofa1"] = (
                            frame["sep3_sofa1"].fillna(0).astype(int)
                        )
                    _append_frame("sep3_sofa1", frame)
            if need_sofa2 and sofa2 is not None and "sofa2" in sofa2.columns:
                from ..scores.sepsis_sofa2 import sep3_sofa2 as _sep3_sofa2

                time_col = _score_time_column(sofa2)
                if time_col is None:
                    errors.append("streamed SOFA-2 dependency lacks a time index")
                else:
                    susp2 = _suspicion_timeline(susp, time_col)
                    frame = _sep3_sofa2(
                        sofa2[[id_col, time_col, "sofa2"]],
                        susp2,
                        id_cols=[id_col],
                        index_col=time_col,
                    )
                    if "sep3_sofa2" in frame.columns:
                        frame["sep3_sofa2"] = (
                            frame["sep3_sofa2"].fillna(0).astype(int)
                        )
                    _append_frame("sep3_sofa2", frame)

        saved = {}
        for concept, writer in writers.items():
            writer.close()
            destination = Path(output_dir) / f"{concept}.parquet"
            os.replace(partials[concept], destination)
            saved[concept] = {"path": str(destination), "rows": rows[concept]}
    except Exception:
        for writer in writers.values():
            writer.close()
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        module_memory_sampler.stop()
        raise

    manifest = {
        "module": "special_concepts",
        "saved": saved,
        "errors": errors,
        "elapsed_sec": round(time.time() - started, 1),
        "batch_size": safe_batch_size,
        "batch_count": (
            (len(all_ids) + safe_batch_size - 1) // safe_batch_size
            if all_ids
            else 0
        ),
        "patient_partition_strategy": "source_order_interleaved_v1",
        "initial_planned_partition_count": planned_partition_count,
        **module_memory_sampler.stop(),
    }
    with open(os.path.join(output_dir, "_manifest.json"), "w") as handle:
        json.dump(manifest, handle)


def _run_special_extraction(
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict],
    batch_size: Optional[int],
    output_dir: str,
    use_sofa2: bool = False,
    stream_output_batches: bool = False,
    published_output_dir: Optional[str] = None,
) -> None:
    """加载特殊概念（Sepsis-3 等）并写入 parquet + _manifest.json。

    sep3_sofa1/sep3_sofa2 不在 concept-dict 中，需要先加载 susp_inf + sofa/sofa2，
    然后通过 _load_sep3_diagnosis 逻辑计算 Sepsis-3 诊断。分组模式下与
    sofa1_score/sofa2_score 同进程运行，susp_inf/sofa/sofa2 直接命中组内缓存。
    """
    import json
    import os
    import time
    import traceback
    import pandas as pd
    from easyicu import load_concepts as _lc

    dependency_root = Path(published_output_dir or output_dir)
    required_dependency_modules = ["sepsis_shared"]
    if any("sep3_sofa1" in EXTRACT_MODULES.get(m, []) for m in special_modules):
        required_dependency_modules.append("sofa1_score")
    if any("sep3_sofa2" in EXTRACT_MODULES.get(m, []) for m in special_modules):
        required_dependency_modules.append("sofa2_score")
    published_dependencies_ready = bool(published_output_dir) and all(
        (dependency_root / f"{module}.parquet").is_file()
        for module in required_dependency_modules
    )

    if stream_output_batches or published_dependencies_ready:
        if not patient_ids_filter or not batch_size:
            raise ValueError(
                "streamed special export requires patient_ids and batch_size"
            )
        _stream_special_extraction_batches(
            special_modules,
            database,
            data_path,
            patient_ids_filter,
            int(batch_size),
            output_dir,
            use_sofa2=use_sofa2,
            published_output_dir=published_output_dir or output_dir,
        )
        return

    t0 = time.time()
    module_memory_sampler = _RSSPeakSampler().start()
    saved = {}
    errors = []

    # 构建公共加载参数（use_sofa2 显式传入以保持组内 loader 配置一致）
    load_kw = dict(
        data_path=data_path,
        database=database,
        verbose=False,
        merge=True,
        use_sofa2=use_sofa2,
    )
    if patient_ids_filter:
        load_kw["patient_ids"] = patient_ids_filter
    if batch_size:
        load_kw["batch_size"] = batch_size

    # 收集需要的概念: sep3_sofa1 需要 sofa, sep3_sofa2 需要 sofa2
    need_sofa1 = any(
        "sep3_sofa1" in EXTRACT_MODULES.get(m, []) for m in special_modules
    )
    need_sofa2 = any(
        "sep3_sofa2" in EXTRACT_MODULES.get(m, []) for m in special_modules
    )

    deps = ["susp_inf"]
    if need_sofa1:
        deps.append("sofa")
    if need_sofa2:
        deps.append("sofa2")

    try:
        merged = _lc(concepts=deps, **load_kw)
    except Exception:
        # sofa2 可能不可用，回退到仅 sofa
        try:
            merged = _lc(concepts=["susp_inf", "sofa"], **load_kw)
            need_sofa2 = False
        except Exception as e:
            traceback.print_exc()
            errors.append(f"Failed to load dependencies {deps}: {e}")
            merged = pd.DataFrame()

    if isinstance(merged, pd.DataFrame) and not merged.empty:
        # 检测 ID 和时间列
        id_col = next(
            (
                c
                for c in [
                    "stay_id",
                    "patientunitstayid",
                    "admissionid",
                    "patientid",
                    "icustay_id",
                    "CaseID",
                ]
                if c in merged.columns
            ),
            None,
        )
        time_col = next(
            (
                c
                for c in [
                    "charttime",
                    "time",
                    "starttime",
                    "datetime",
                    "Offset",
                    "measuredat_minutes",
                    "measuredat",
                ]
                if c in merged.columns
            ),
            None,
        )

        if id_col and time_col and "susp_inf" in merged.columns:
            _require_timed_positive_suspicion(
                merged,
                id_col=id_col,
                time_col=time_col,
                database=database,
            )
            # Sepsis-3 = a >=2-point SOFA increase WITHIN the suspected-infection
            # window (delta rule, R ricu sep3), NOT an absolute SOFA>=2. Use the
            # shared sep3()/sep3_sofa2() so both labels match load_sepsis3 and the
            # module export (unified to delta 2026-06-22).
            if need_sofa1 and "sofa" in merged.columns:
                from ..scores.sepsis import sep3 as _sep3

                result = _sep3(
                    merged[[id_col, time_col, "sofa"]],
                    merged[[id_col, time_col, "susp_inf"]],
                    id_cols=[id_col],
                    index_col=time_col,
                ).rename(columns={"sep3": "sep3_sofa1"})
                if "sep3_sofa1" in result.columns:
                    result["sep3_sofa1"] = result["sep3_sofa1"].fillna(0).astype(int)
                if len(result) > 0:
                    path = os.path.join(output_dir, "sep3_sofa1.parquet")
                    result.to_parquet(path, index=False, engine="pyarrow")
                    saved["sep3_sofa1"] = {"path": path, "rows": len(result)}

            if need_sofa2 and "sofa2" in merged.columns:
                from ..scores.sepsis_sofa2 import sep3_sofa2 as _sep3_sofa2

                result = _sep3_sofa2(
                    merged[[id_col, time_col, "sofa2"]],
                    merged[[id_col, time_col, "susp_inf"]],
                    id_cols=[id_col],
                    index_col=time_col,
                )
                if "sep3_sofa2" in result.columns:
                    result["sep3_sofa2"] = result["sep3_sofa2"].fillna(0).astype(int)
                if len(result) > 0:
                    path = os.path.join(output_dir, "sep3_sofa2.parquet")
                    result.to_parquet(path, index=False, engine="pyarrow")
                    saved["sep3_sofa2"] = {"path": path, "rows": len(result)}
        else:
            missing = []
            if not id_col:
                missing.append("id_col")
            if not time_col:
                missing.append("time_col")
            if "susp_inf" not in merged.columns:
                missing.append("susp_inf")
            # 🔧 FIX 2026-05-11: 对于 sic/hirid 等不支持 susp_inf 的数据库，
            # sep3_sofa1/sep3_sofa2 无法计算属正常情况，不应记为错误。
            # 只有当 id/time 列也缺失时才认为是真正的错误。
            if missing == ["susp_inf"]:
                pass  # 静默跳过：数据库不支持 susp_inf，sep3 概念不适用
            else:
                errors.append(
                    f"Missing columns: {missing}, available: {list(merged.columns)[:10]}"
                )

    elapsed = time.time() - t0
    manifest = {
        "module": "special_concepts",
        "saved": saved,
        "errors": errors,
        "elapsed_sec": round(elapsed, 1),
        **module_memory_sampler.stop(),
    }
    with open(os.path.join(output_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f)


def _extract_special_worker(
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    output_dir: str = "",
):
    """（兼容包装）特殊概念子进程入口 — 参见 _extract_module_group_worker。"""
    _extract_worker_env_setup(data_path)
    _run_special_extraction(
        special_modules,
        database,
        data_path,
        patient_ids_filter,
        batch_size,
        output_dir,
    )


def _extract_module_group_worker(
    module_specs: List[tuple],
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict],
    batch_size: Optional[int],
    output_root: str,
    use_sofa2: bool,
    stream_output_batches: bool = False,
    published_output_dir: Optional[str] = None,
    adaptive_stream_batches: bool = False,
):
    """在一个子进程中顺序提取一组共享源表的模块。

    keep_cache 让组内模块共享 raw/table 缓存（受 EASYICU_CACHE_BUDGET_MB
    字节预算约束），chartevents/labevents 等重表每组只扫一次，而不是每
    模块重扫一遍；子进程退出后 OS 仍完整回收内存。分组因此是“缓存复用”
    与“内存隔离”之间的折中：组内复用，组间隔离。

    module_specs: [(module_name, [concepts...]), ...]，每个模块写
    ``output_root/<module_name>/``；特殊模块写 ``output_root/_special/``。
    """
    import os
    import traceback

    _extract_worker_env_setup(data_path)
    from easyicu.api import keep_cache as _keep_cache

    with _keep_cache(
        database=database, data_path=data_path, use_sofa2=use_sofa2
    ) as _loader:
        for module_name, concepts in module_specs:
            out_dir = os.path.join(output_root, module_name)
            os.makedirs(out_dir, exist_ok=True)
            try:
                _run_module_extraction(
                    module_name,
                    concepts,
                    database,
                    data_path,
                    patient_ids_filter,
                    batch_size,
                    out_dir,
                    use_sofa2=use_sofa2,
                    loader=_loader,
                    stream_output_batches=stream_output_batches,
                    adaptive_stream_batches=adaptive_stream_batches,
                )
            except Exception:
                # _run_module_extraction 已内部捕获常规异常并写 manifest；
                # 这里兜底保证一个模块的意外崩溃不拖垮组内后续模块。
                traceback.print_exc()
        if special_modules:
            sp_dir = os.path.join(output_root, _SPECIAL_OUTPUT_DIRNAME)
            os.makedirs(sp_dir, exist_ok=True)
            try:
                _run_special_extraction(
                    special_modules,
                    database,
                    data_path,
                    patient_ids_filter,
                    batch_size,
                    sp_dir,
                    use_sofa2=use_sofa2,
                    stream_output_batches=stream_output_batches,
                    published_output_dir=published_output_dir,
                )
            except Exception:
                traceback.print_exc()


# 分组亲和表：同组模块共享同一批重源表（chartevents/labevents/inputevents
# 家族），或互为依赖（SOFA 闭包）。分组只影响“哪些模块共用一个子进程 +
# keep_cache”，不改变模块内容、输出布局或模块顺序语义。
_EXTRACT_MODULE_GROUP_AFFINITY: List[List[str]] = [
    # chartevents / nursecharting 家族
    ["vitals", "neurological", "respiratory", "ventilator"],
    # 入科级小表（icustays/admissions/patients）
    ["demographics", "outcome"],
    # labevents 家族
    ["blood_gas", "chemistry", "hematology", "renal"],
    # inputevents / prescriptions 家族
    ["vasopressors", "medications", "circulatory"],
    # 评分闭包：SOFA 组件被 sofa1/sofa2 共享，sep3_* 复用 susp_inf+sofa/sofa2
    ["other_scores", "sepsis_shared", "sofa1_score", "sofa2_score"],
]


def _group_modules_for_extraction(
    normal_modules: List[str],
    special_modules: List[str],
    group_modules: bool = True,
) -> List[Dict[str, List[str]]]:
    """把请求的模块划分为子进程组。

    返回 [{'modules': [...], 'special': [...]}, ...]。group_modules=False
    时退化为每模块一组（旧行为）。未出现在亲和表中的新模块各自成组。
    特殊模块（Sepsis-3）挂到评分组上（若本次请求包含评分组），使
    susp_inf/sofa/sofa2 命中组内缓存；否则单独成组。
    """
    if not group_modules:
        groups: List[Dict[str, List[str]]] = [
            {"modules": [m], "special": []} for m in normal_modules
        ]
        if special_modules:
            groups.append({"modules": [], "special": list(special_modules)})
        return groups

    groups = []
    assigned = set()
    for affinity in _EXTRACT_MODULE_GROUP_AFFINITY:
        members = [m for m in normal_modules if m in affinity]
        if members:
            groups.append({"modules": members, "special": []})
            assigned.update(members)
    for m in normal_modules:
        if m not in assigned:
            groups.append({"modules": [m], "special": []})

    if special_modules:
        target = next(
            (
                g
                for g in groups
                if any(
                    m in ("sofa1_score", "sofa2_score", "sepsis_shared")
                    for m in g["modules"]
                )
            ),
            None,
        )
        if target is None:
            groups.append({"modules": [], "special": list(special_modules)})
        else:
            target["special"] = list(special_modules)
    return groups


_GROUPING_MIN_HOST_MEMORY_MB = 24 * 1024
_GROUPING_MIN_CACHE_BUDGET_MB = 4096


def _resolve_extraction_grouping(
    group_modules: bool,
    stream_output_batches: bool,
    *,
    environment: Optional[Dict[str, str]] = None,
    total_memory_mb: Optional[float] = None,
) -> tuple[bool, str]:
    """Choose the safe grouping mode for the current memory contract.

    Sharing source-table caches across related modules is faster on a server,
    but real full-database profiling found 17--28 GiB process-tree peaks even
    with a bounded cache. A 16 GiB workstation must therefore isolate modules
    into separate subprocesses while still loading the full patient cohort
    once per module. Experts can force either mode with
    ``EASYICU_EXTRACT_GROUPING=1`` or ``=0``.
    """
    if not group_modules:
        return False, "disabled_by_argument"
    if stream_output_batches:
        return False, "streamed_batch_writer"

    env = os.environ if environment is None else environment
    raw_override = str(env.get("EASYICU_EXTRACT_GROUPING", "")).strip().lower()
    if raw_override in {"0", "off", "false", "no"}:
        return False, "disabled_by_environment"
    if raw_override in {"1", "on", "true", "yes"}:
        return True, "forced_by_environment"

    raw_cache_budget = env.get("EASYICU_CACHE_BUDGET_MB")
    if raw_cache_budget is not None:
        try:
            cache_budget_mb = float(raw_cache_budget)
        except (TypeError, ValueError):
            cache_budget_mb = None
        if (
            cache_budget_mb is not None
            and 0 < cache_budget_mb <= _GROUPING_MIN_CACHE_BUDGET_MB
        ):
            return False, "constrained_cache_budget"

    if total_memory_mb is None:
        try:
            import psutil

            total_memory_mb = psutil.virtual_memory().total / (1024**2)
        except Exception:
            # Fail safe when host capacity cannot be established.
            total_memory_mb = 16 * 1024
    if total_memory_mb <= _GROUPING_MIN_HOST_MEMORY_MB:
        return False, "low_memory_host"
    return True, "shared_cache_speed_path"


_NATIVE_EXPORT_SCHEMA_V2 = "easyicu_native_export_v2"
_NATIVE_EXPORT_ID_COLUMNS = (
    "stay_id",
    "patientunitstayid",
    "icustay_id",
    "admissionid",
    "patientid",
    "CaseID",
)


def _native_export_storage_kind(concept_id: str, dictionary) -> str:
    """Return the deterministic physical type family for one public concept."""
    from ..concept.catalog import CONCEPT_DICTIONARY

    definition = dictionary.get(concept_id)
    raw_class_name = getattr(definition, "class_name", None)
    if isinstance(raw_class_name, (list, tuple, set, frozenset)):
        class_names = {
            str(value).strip()
            for value in raw_class_name
            if str(value).strip()
        }
    elif raw_class_name is None:
        class_names = set()
    else:
        class_names = {str(raw_class_name).strip()}
    catalog_unit = CONCEPT_DICTIONARY.get(concept_id, ("", "", ""))[2]
    if "lgl_cncpt" in class_names or str(catalog_unit).strip().lower() == "boolean":
        return "boolean"
    if (
        class_names.intersection({"fct_cncpt", "chr_cncpt"})
        or str(catalog_unit).strip().lower() == "category"
        or concept_id == "avpu"
    ):
        return "string"
    return "float64"


def _canonicalise_native_export_frame(
    frame,
    *,
    module: str,
    requested_concepts: List[str],
    dictionary,
):
    """Project one native package file onto the cross-database physical schema.

    The extraction engine keeps source-native identifiers internally. Native-v2
    is the portable downstream contract, so it always exposes ``stay_id``,
    relative ``charttime`` (except the stay-level demographics module), and
    every requested concept in catalog order. Structurally unavailable concepts
    remain typed all-null placeholders and are separately marked unavailable in
    the manifest.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("native export canonicalisation requires a DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("native export frame has duplicate physical columns")

    candidates = [column for column in _NATIVE_EXPORT_ID_COLUMNS if column in frame]
    if not candidates:
        raise ValueError("native export frame has no ICU-stay identity column")
    identity = "stay_id" if "stay_id" in candidates else candidates[0]

    canonical = pd.DataFrame(index=frame.index)
    identity_values = pd.to_numeric(frame[identity], errors="coerce")
    if bool(identity_values.isna().any()):
        raise ValueError(
            f"native export identity '{identity}' contains missing or non-numeric values"
        )
    identity_float = identity_values.astype("float64")
    if bool((identity_float != np.floor(identity_float)).any()):
        raise ValueError(f"native export identity '{identity}' is not integer-valued")
    canonical["stay_id"] = identity_float.astype("int64")

    if module != "demographics":
        if "charttime" in frame:
            charttime = pd.to_numeric(frame["charttime"], errors="coerce")
            invalid = frame["charttime"].notna() & charttime.isna()
            if bool(invalid.any()):
                raise ValueError(
                    "native export charttime must be relative numeric ICU hours"
                )
            canonical["charttime"] = charttime.astype("float64")
        else:
            canonical["charttime"] = pd.Series(
                np.nan, index=frame.index, dtype="float64"
            )

    for concept in requested_concepts:
        kind = _native_export_storage_kind(concept, dictionary)
        if concept not in frame:
            if kind == "boolean":
                canonical[concept] = pd.Series(
                    pd.NA, index=frame.index, dtype="boolean"
                )
            elif kind == "string":
                canonical[concept] = pd.Series(
                    pd.NA, index=frame.index, dtype="string"
                )
            else:
                canonical[concept] = pd.Series(
                    np.nan, index=frame.index, dtype="float64"
                )
            continue

        source = frame[concept]
        if kind == "boolean":
            if pd.api.types.is_bool_dtype(source.dtype):
                canonical[concept] = source.astype("boolean")
                continue
            numeric = pd.to_numeric(source, errors="coerce")
            invalid = source.notna() & numeric.isna()
            invalid |= numeric.notna() & ~numeric.isin((0, 1))
            if bool(invalid.any()):
                raise ValueError(
                    f"native export logical concept '{concept}' contains values "
                    "outside {0, 1, missing}"
                )
            canonical[concept] = numeric.astype("boolean")
        elif kind == "string":
            canonical[concept] = source.astype("string")
        else:
            numeric = pd.to_numeric(source, errors="coerce")
            invalid = source.notna() & numeric.isna()
            if bool(invalid.any()):
                raise ValueError(
                    f"native export numeric concept '{concept}' contains non-numeric values"
                )
            canonical[concept] = numeric.astype("float64")

    return canonical


def _restore_native_export_storage_dtypes(
    frame,
    *,
    requested_concepts: List[str],
    dictionary,
):
    """Restore the exact native-v2 dtypes after row-grain aggregation."""
    import pandas as pd

    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="raise").astype(
        "int64"
    )
    if "charttime" in frame:
        frame["charttime"] = pd.to_numeric(
            frame["charttime"], errors="coerce"
        ).astype("float64")
    for concept in requested_concepts:
        kind = _native_export_storage_kind(concept, dictionary)
        if kind == "boolean":
            frame[concept] = frame[concept].astype("boolean")
        elif kind == "string":
            frame[concept] = frame[concept].astype("string")
        else:
            frame[concept] = pd.to_numeric(
                frame[concept], errors="coerce"
            ).astype("float64")
    return frame


def _consolidate_native_export_row_grain(
    frame,
    *,
    module: str,
    requested_concepts: List[str],
    dictionary,
    source_charttime=None,
):
    """Enforce one deterministic physical row per native-v2 primary key.

    Demographics is a stay-level table.  A source may nevertheless expose a
    static concept at several event times; each concept is therefore selected
    independently from its nearest non-null value to ICU admission (0 h), with
    stable source order as the tie-breaker.  BMI is recomputed from the selected
    height and weight rather than copied from a potentially different row.

    Every other module has the null-equal key ``(stay_id, charttime)``. Exact
    key collisions are consolidated by the physical type family: logical any,
    numeric median, and a single non-null string value. Conflicting strings are
    publication errors because silently choosing one would invent a category.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("native export grain consolidation requires a DataFrame")
    # Canonicalisation already owns a fresh frame.  Avoid another deep copy of
    # multi-million-row modules merely to inspect their key: this publication
    # guard must remain usable on the documented 16-GB profile.  Only normalize
    # a non-default index when an earlier time filter actually left one behind.
    working = frame
    if not isinstance(working.index, pd.RangeIndex) or (
        working.index.start != 0 or working.index.step != 1
    ):
        working = working.reset_index(drop=True)

    if module == "demographics":
        primary_key = ["stay_id"]
        duplicate_mask = working.duplicated(primary_key, keep=False)
        duplicate_excess = int(
            working.duplicated(primary_key, keep="first").sum()
        )
        duplicate_groups = int(
            working.loc[duplicate_mask, "stay_id"].nunique(dropna=False)
        )

        if source_charttime is None:
            source_time = pd.Series(np.nan, index=working.index, dtype="float64")
            source_time_present = False
            source_null_time_rows = None
        else:
            raw_time = pd.Series(source_charttime).reset_index(drop=True)
            if len(raw_time) != len(working):
                raise ValueError(
                    "demographics source charttime is not row-aligned with its frame"
                )
            source_time = pd.to_numeric(raw_time, errors="coerce")
            invalid_time = raw_time.notna() & source_time.isna()
            if bool(invalid_time.any()):
                raise ValueError(
                    "demographics source charttime must be numeric ICU-relative hours"
                )
            source_time = source_time.astype("float64")
            source_time_present = True
            source_null_time_rows = int(source_time.isna().sum())

        first_stays = working[["stay_id"]].drop_duplicates(
            "stay_id", keep="first"
        )
        consolidated = first_stays.reset_index(drop=True)
        conflict_groups: Dict[str, int] = {}
        for concept in requested_concepts:
            if concept == "bmi":
                continue
            candidates = pd.DataFrame(
                {
                    "stay_id": working["stay_id"],
                    concept: working[concept],
                    "_source_time": source_time,
                    "_row_order": np.arange(len(working), dtype="int64"),
                }
            )
            non_null = candidates.loc[candidates[concept].notna()].copy()
            if non_null.empty:
                selected = pd.Series(dtype=working[concept].dtype)
                conflicts = 0
            else:
                value_counts = non_null.groupby(
                    "stay_id", sort=False, dropna=False
                )[concept].nunique(dropna=True)
                conflicts = int((value_counts > 1).sum())
                non_null["_time_missing"] = non_null["_source_time"].isna()
                non_null["_abs_source_time"] = non_null["_source_time"].abs()
                non_null = non_null.sort_values(
                    [
                        "stay_id",
                        "_time_missing",
                        "_abs_source_time",
                        "_row_order",
                    ],
                    kind="mergesort",
                )
                selected = non_null.drop_duplicates(
                    "stay_id", keep="first"
                ).set_index("stay_id")[concept]
            conflict_groups[concept] = conflicts
            consolidated[concept] = consolidated["stay_id"].map(selected)

        recomputed_bmi_rows = 0
        if "bmi" in requested_concepts:
            bmi = pd.Series(np.nan, index=consolidated.index, dtype="float64")
            if {"height", "weight"}.issubset(consolidated.columns):
                height = pd.to_numeric(consolidated["height"], errors="coerce")
                weight = pd.to_numeric(consolidated["weight"], errors="coerce")
                bounds = _load_concept_bounds_map()
                height_min, height_max = bounds.get("height", (None, None))
                weight_min, weight_max = bounds.get("weight", (None, None))
                valid = height.notna() & weight.notna() & (height > 0) & (weight > 0)
                if height_min is not None:
                    valid &= height >= float(height_min)
                if height_max is not None:
                    valid &= height <= float(height_max)
                if weight_min is not None:
                    valid &= weight >= float(weight_min)
                if weight_max is not None:
                    valid &= weight <= float(weight_max)
                bmi.loc[valid] = weight.loc[valid] / (
                    height.loc[valid] / 100.0
                ) ** 2
                finite = np.isfinite(bmi.to_numpy(dtype="float64", na_value=np.nan))
                bmi.loc[~finite] = np.nan
                recomputed_bmi_rows = int(bmi.notna().sum())
            consolidated["bmi"] = bmi

        consolidated = consolidated[
            ["stay_id", *requested_concepts]
        ].reset_index(drop=True)
        consolidated = _restore_native_export_storage_dtypes(
            consolidated,
            requested_concepts=requested_concepts,
            dictionary=dictionary,
        )
        if bool(consolidated.duplicated(primary_key, keep=False).any()):
            raise RuntimeError("demographics row-grain consolidation was not unique")
        audit: Dict[str, object] = {
            "row_grain": "one_row_per_icu_stay",
            "primary_key": primary_key,
            "null_key_equality": "not_applicable",
            "source_rows": int(len(working)),
            "published_rows": int(len(consolidated)),
            "duplicate_key_rows_before": int(duplicate_mask.sum()),
            "duplicate_key_groups_before": duplicate_groups,
            "duplicate_excess_rows_before": duplicate_excess,
            "rows_consolidated": duplicate_excess,
            "duplicate_excess_rows_after": 0,
            "source_charttime_present": source_time_present,
            "source_null_charttime_rows": source_null_time_rows,
            "static_selection_policy": (
                "nearest_non_null_value_to_icu_admission_then_source_order"
            ),
            "conflicting_non_null_stay_groups_by_concept": conflict_groups,
            "bmi_policy": "recomputed_from_selected_weight_kg_and_height_cm",
            "bounded_source_bmi_non_null_rows_discarded": (
                int(working["bmi"].notna().sum())
                if "bmi" in requested_concepts
                else 0
            ),
            "recomputed_bmi_rows": recomputed_bmi_rows,
        }
        return consolidated, audit

    primary_key = ["stay_id", "charttime"]
    if "charttime" not in working:
        raise ValueError(
            f"native export module '{module}' has no canonical charttime column"
        )
    duplicate_mask = working.duplicated(primary_key, keep=False)
    duplicate_excess = int(
        working.duplicated(primary_key, keep="first").sum()
    )
    duplicate_rows = int(duplicate_mask.sum())
    records = []
    if duplicate_rows:
        duplicate_frame = working.loc[duplicate_mask]
        for _key, group in duplicate_frame.groupby(
            primary_key,
            sort=False,
            dropna=False,
        ):
            record = {
                "stay_id": group["stay_id"].iloc[0],
                "charttime": group["charttime"].iloc[0],
                "_row_order": int(group.index.min()),
            }
            for concept in requested_concepts:
                values = group[concept].dropna()
                kind = _native_export_storage_kind(concept, dictionary)
                if values.empty:
                    record[concept] = (
                        pd.NA if kind in {"boolean", "string"} else np.nan
                    )
                elif kind == "boolean":
                    record[concept] = bool(values.astype("boolean").any())
                elif kind == "string":
                    distinct = values.astype("string").drop_duplicates()
                    if len(distinct) > 1:
                        key_value = (
                            int(group["stay_id"].iloc[0]),
                            group["charttime"].iloc[0],
                        )
                        raise ValueError(
                            "native export cannot consolidate conflicting string "
                            f"concept '{concept}' at key {key_value!r}: "
                            f"{distinct.astype(str).tolist()!r}"
                        )
                    record[concept] = distinct.iloc[0]
                else:
                    record[concept] = float(
                        pd.to_numeric(values, errors="raise").median()
                    )
            records.append(record)

        unique_rows = working.loc[~duplicate_mask].copy()
        unique_rows["_row_order"] = unique_rows.index.astype("int64")
        aggregated = pd.DataFrame.from_records(records)
        consolidated = pd.concat(
            [unique_rows, aggregated], ignore_index=True, sort=False
        ).sort_values("_row_order", kind="mergesort")
        consolidated = consolidated.drop(columns="_row_order")
    else:
        consolidated = working

    consolidated = consolidated[
        ["stay_id", "charttime", *requested_concepts]
    ].reset_index(drop=True)
    consolidated = _restore_native_export_storage_dtypes(
        consolidated,
        requested_concepts=requested_concepts,
        dictionary=dictionary,
    )
    duplicate_after = int(
        consolidated.duplicated(primary_key, keep="first").sum()
    )
    if duplicate_after:
        raise RuntimeError(
            f"native export module '{module}' row-grain consolidation was not unique"
        )
    audit = {
        "row_grain": "one_row_per_icu_stay_relative_hour",
        "primary_key": primary_key,
        "null_key_equality": "nulls_equal",
        "source_rows": int(len(working)),
        "published_rows": int(len(consolidated)),
        "null_charttime_rows_before": int(working["charttime"].isna().sum()),
        "null_charttime_rows_after": int(consolidated["charttime"].isna().sum()),
        "duplicate_key_rows_before": duplicate_rows,
        "duplicate_key_groups_before": int(len(records)),
        "duplicate_excess_rows_before": duplicate_excess,
        "rows_consolidated": duplicate_excess,
        "duplicate_excess_rows_after": duplicate_after,
        "aggregation_policy": {
            "boolean": "any_non_null_preserving_all_null",
            "numeric": "median_non_null_preserving_all_null",
            "string": "single_non_null_value_or_fail_on_conflict",
        },
    }
    return consolidated, audit


def _native_export_file_sha256(path: Path) -> str:
    """Return the content digest sealed into a native-v2 file receipt."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_NATIVE_EXPORT_ARROW_BATCH_ROWS = 262_144
_NATIVE_EXPORT_DUCKDB_MEMORY_MB = 512
_NATIVE_EXPORT_PANDAS_FALLBACK_MAX_ROWS = 1_000_000
_NATIVE_EXPORT_PANDAS_FALLBACK_MAX_BYTES = 512 * 1024 * 1024
_NATIVE_EXPORT_PANDAS_FALLBACK_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


def _native_export_empty_schema_frame(
    *,
    module: str,
    requested_concepts: List[str],
    dictionary,
):
    """Return the zero-row pandas representation of the native-v2 schema.

    This frame is intentionally metadata-only.  The Arrow publisher uses it to
    preserve the same pandas/Parquet logical dtypes as the legacy
    ``DataFrame.to_parquet`` path without materialising a module payload in
    pandas.
    """
    import pandas as pd

    columns = {"stay_id": pd.Series([], dtype="int64")}
    if module != "demographics":
        columns["charttime"] = pd.Series([], dtype="float64")
    for concept in requested_concepts:
        kind = _native_export_storage_kind(concept, dictionary)
        if kind == "boolean":
            columns[concept] = pd.Series([], dtype="boolean")
        elif kind == "string":
            columns[concept] = pd.Series([], dtype="string")
        else:
            columns[concept] = pd.Series([], dtype="float64")
    return pd.DataFrame(columns)


def _native_export_arrow_batch_rows() -> int:
    """Resolve a bounded publisher batch without exposing an unbounded knob."""
    raw = os.environ.get("EASYICU_NATIVE_PUBLISH_BATCH_ROWS")
    if raw is None:
        return _NATIVE_EXPORT_ARROW_BATCH_ROWS
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        return _NATIVE_EXPORT_ARROW_BATCH_ROWS
    return min(1_000_000, max(16_384, requested))


def _native_export_duckdb_memory_mb() -> int:
    """Resolve the spillable uniqueness-audit memory ceiling."""
    raw = os.environ.get("EASYICU_NATIVE_PUBLISH_DUCKDB_MEMORY_MB")
    if raw is None:
        return _NATIVE_EXPORT_DUCKDB_MEMORY_MB
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        return _NATIVE_EXPORT_DUCKDB_MEMORY_MB
    return min(1_024, max(128, requested))


def _native_export_pandas_fallback_size(path: Path) -> Dict[str, int]:
    """Return physical bounds used before any full-frame fallback."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    return {
        "rows": int(metadata.num_rows),
        "parquet_bytes": int(path.stat().st_size),
        "uncompressed_parquet_bytes": int(
            sum(
                metadata.row_group(index).total_byte_size
                for index in range(metadata.num_row_groups)
            )
        ),
    }


def _native_export_pandas_fallback_is_bounded(size: Dict[str, int]) -> bool:
    return (
        size["rows"] <= _NATIVE_EXPORT_PANDAS_FALLBACK_MAX_ROWS
        and size["parquet_bytes"] <= _NATIVE_EXPORT_PANDAS_FALLBACK_MAX_BYTES
        and size["uncompressed_parquet_bytes"]
        <= _NATIVE_EXPORT_PANDAS_FALLBACK_MAX_UNCOMPRESSED_BYTES
    )


def _native_export_arrow_row_grain_audit(
    path: Path,
    *,
    module: str,
) -> Dict[str, object]:
    """Audit a canonical temporary Parquet with bounded, spillable DuckDB.

    Hashing all keys in Python would itself recreate a multi-gigabyte working
    set.  DuckDB is given an explicit memory ceiling and a temporary directory
    beside the output, so its global NULL-equal uniqueness proof can spill to
    the same user-selected volume as the export.
    """
    import tempfile

    import duckdb

    if module == "demographics":
        raise ValueError("Arrow row-grain audit is longitudinal-only")
    memory_mb = _native_export_duckdb_memory_mb()
    with tempfile.TemporaryDirectory(
        prefix=f".{module}.native-v2-grain-",
        dir=path.parent,
    ) as spill_dir:
        connection = duckdb.connect(
            database=":memory:",
            config={
                "memory_limit": f"{memory_mb}MB",
                "threads": "1",
                "temp_directory": spill_dir,
            },
        )
        try:
            row = connection.execute(
                """
                WITH key_counts AS (
                    SELECT stay_id, charttime, count(*)::BIGINT AS n
                    FROM read_parquet(?)
                    GROUP BY stay_id, charttime
                )
                SELECT
                    coalesce(sum(n), 0)::BIGINT AS source_rows,
                    coalesce(sum(n) FILTER (WHERE charttime IS NULL), 0)::BIGINT
                        AS null_charttime_rows,
                    coalesce(sum(n) FILTER (WHERE n > 1), 0)::BIGINT
                        AS duplicate_key_rows,
                    count(*) FILTER (WHERE n > 1)::BIGINT
                        AS duplicate_key_groups,
                    coalesce(sum(n - 1) FILTER (WHERE n > 1), 0)::BIGINT
                        AS duplicate_excess_rows
                FROM key_counts
                """,
                [str(path)],
            ).fetchone()
        finally:
            connection.close()
    if row is None:
        raise RuntimeError("native export DuckDB row-grain audit returned no result")
    source_rows, null_rows, duplicate_rows, duplicate_groups, duplicate_excess = (
        int(value) for value in row
    )
    return {
        "row_grain": "one_row_per_icu_stay_relative_hour",
        "primary_key": ["stay_id", "charttime"],
        "null_key_equality": "nulls_equal",
        "source_rows": source_rows,
        "published_rows": source_rows,
        "null_charttime_rows_before": null_rows,
        "null_charttime_rows_after": null_rows,
        "duplicate_key_rows_before": duplicate_rows,
        "duplicate_key_groups_before": duplicate_groups,
        "duplicate_excess_rows_before": duplicate_excess,
        "rows_consolidated": 0,
        "duplicate_excess_rows_after": duplicate_excess,
        "aggregation_policy": {
            "boolean": "any_non_null_preserving_all_null",
            "numeric": "median_non_null_preserving_all_null",
            "string": "single_non_null_value_or_fail_on_conflict",
        },
        "publication_backend": "pyarrow_record_batches",
        "uniqueness_backend": "duckdb_bounded_spillable_hash_aggregate",
        "uniqueness_memory_limit_mb": memory_mb,
    }


def _try_publish_native_export_arrow_fast_path(
    *,
    source_parquet: Path,
    temporary_parquet: Path,
    module: str,
    requested_concepts: List[str],
    dictionary,
    stay_time_upper_bounds: Dict[int, float],
) -> Optional[Dict[str, object]]:
    """Publish a unique longitudinal module without a full pandas payload.

    The temporary file is canonicalised and bounded batch-by-batch.  A global
    NULL-equal uniqueness audit is then performed under a fixed DuckDB memory
    budget.  Small duplicate-bearing modules return ``None`` for the exact
    pandas consolidation path; a large duplicate-bearing module fails closed
    rather than silently escaping the documented memory contract.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if module == "demographics":
        return None
    source_file = pq.ParquetFile(source_parquet)
    if source_file.metadata.num_rows == 0:
        return None
    source_schema = source_file.schema_arrow
    if len(set(source_schema.names)) != len(source_schema.names):
        raise ValueError("native export frame has duplicate physical columns")
    candidates = [
        column for column in _NATIVE_EXPORT_ID_COLUMNS if column in source_schema.names
    ]
    if not candidates:
        raise ValueError("native export frame has no ICU-stay identity column")
    identity = "stay_id" if "stay_id" in candidates else candidates[0]
    schema_frame = _native_export_empty_schema_frame(
        module=module,
        requested_concepts=requested_concepts,
        dictionary=dictionary,
    )
    target_schema = pa.Table.from_pandas(
        schema_frame,
        preserve_index=False,
    ).schema
    read_columns = list(
        dict.fromkeys(
            [
                identity,
                *(("charttime",) if "charttime" in source_schema.names else ()),
                *(
                    concept
                    for concept in requested_concepts
                    if concept in source_schema.names
                ),
            ]
        )
    )
    time_axis_audit: Optional[Dict[str, object]] = None
    bounds_audit: Optional[Dict[str, Dict[str, object]]] = None
    concept_non_null = {concept: 0 for concept in requested_concepts}
    writer = None
    try:
        writer = pq.ParquetWriter(
            temporary_parquet,
            target_schema,
            compression="snappy",
        )
        for batch in source_file.iter_batches(
            batch_size=_native_export_arrow_batch_rows(),
            columns=read_columns,
            use_threads=False,
        ):
            # Reuse the established validators on one bounded record batch.
            # This is deliberately a small pandas bridge, never a full-module
            # DataFrame; it minimizes semantic drift across pandas/Arrow types.
            frame = batch.to_pandas()
            frame = _canonicalise_native_export_frame(
                frame,
                module=module,
                requested_concepts=requested_concepts,
                dictionary=dictionary,
            )
            frame, batch_time_audit = _enforce_native_export_time_axis(
                frame,
                module=module,
                stay_time_upper_bounds=stay_time_upper_bounds,
            )
            batch_bounds_audit = _enforce_native_export_concept_bounds(
                frame,
                requested_concepts=requested_concepts,
                dictionary=dictionary,
            )
            if time_axis_audit is None:
                time_axis_audit = dict(batch_time_audit)
            else:
                for field in (
                    "excluded_rows",
                    "excluded_untimed_empty_rows",
                    "excluded_untimed_negative_rrt_criteria_rows",
                    "normalized_stay_level_rows",
                    "rows_with_los_bound",
                ):
                    if field in batch_time_audit:
                        time_axis_audit[field] = int(
                            time_axis_audit.get(field, 0)
                        ) + int(batch_time_audit[field])
            if bounds_audit is None:
                bounds_audit = {
                    concept: dict(record)
                    for concept, record in batch_bounds_audit.items()
                }
            else:
                for concept, record in batch_bounds_audit.items():
                    bounds_audit[concept]["excluded_out_of_bounds"] = int(
                        bounds_audit[concept]["excluded_out_of_bounds"]
                    ) + int(record["excluded_out_of_bounds"])
            for concept in requested_concepts:
                concept_non_null[concept] += int(frame[concept].notna().sum())
            output = pa.Table.from_pandas(
                frame,
                schema=target_schema,
                preserve_index=False,
                safe=True,
            )
            if len(output):
                writer.write_table(output)
            del output, frame
            _release_stream_batch_memory(
                pa,
                trim_native_allocator=False,
            )
        writer.close()
        writer = None
    except Exception:
        if writer is not None:
            writer.close()
        temporary_parquet.unlink(missing_ok=True)
        raise

    if time_axis_audit is None or bounds_audit is None:
        temporary_parquet.unlink(missing_ok=True)
        raise RuntimeError("native export Arrow publisher observed no source batches")

    row_grain_audit = _native_export_arrow_row_grain_audit(
        temporary_parquet,
        module=module,
    )
    duplicate_excess = int(row_grain_audit["duplicate_excess_rows_before"])
    if duplicate_excess:
        fallback_size = _native_export_pandas_fallback_size(source_parquet)
        temporary_parquet.unlink(missing_ok=True)
        if _native_export_pandas_fallback_is_bounded(fallback_size):
            return None
        raise ValueError(
            "native export row-grain consolidation exceeds the bounded pandas "
            f"fallback (module={module!r}, {fallback_size=}, "
            f"duplicate_excess_rows={duplicate_excess})"
        )

    return {
        "schema_frame": schema_frame,
        "rows": int(row_grain_audit["published_rows"]),
        "time_axis_audit": time_axis_audit,
        "bounds_audit": bounds_audit,
        "row_grain_audit": row_grain_audit,
        "concept_non_null": concept_non_null,
    }


def _native_export_stay_time_upper_bounds(outcome_frame) -> Dict[int, float]:
    """Return each stay's last plausible ICU-relative event hour.

    ``los_icu`` is a cross-database days concept.  A one-day post-discharge
    allowance preserves boundary measurements while preventing a single
    corrupt source timestamp from stretching hourly score grids for years.
    """
    import pandas as pd
    from ..utils.time_units import ICU_TIME_POST_DISCHARGE_HOURS

    if not isinstance(outcome_frame, pd.DataFrame) or "los_icu" not in outcome_frame:
        return {}
    identity = next(
        (
            column
            for column in _NATIVE_EXPORT_ID_COLUMNS
            if column in outcome_frame.columns
        ),
        None,
    )
    if identity is None:
        return {}
    stay_id = pd.to_numeric(outcome_frame[identity], errors="coerce")
    los_days = pd.to_numeric(outcome_frame["los_icu"], errors="coerce")
    valid = stay_id.notna() & los_days.notna() & (los_days >= 0)
    if not bool(valid.any()):
        return {}
    bounds = pd.DataFrame(
        {
            "stay_id": stay_id.loc[valid].astype("int64"),
            "upper": (
                los_days.loc[valid].astype("float64") * 24.0
                + ICU_TIME_POST_DISCHARGE_HOURS
            ),
        }
    )
    return {
        int(key): float(value)
        for key, value in bounds.groupby("stay_id", sort=False)["upper"].max().items()
    }


def _enforce_native_export_time_axis(
    frame,
    *,
    module: str,
    stay_time_upper_bounds: Dict[int, float],
):
    """Apply the ICU-relative time contract and return an auditable frame.

    Outcome values are stay-level endpoints with mixed follow-up windows, so
    their shared physical index is ICU admission (0 h).  Longitudinal modules
    retain only the ICU episode with a 24-hour pre/post allowance.  Stays
    without a usable LOS receive a conservative 366-day sanity fallback.
    """
    import numpy as np
    import pandas as pd
    from ..utils.time_units import (
        ICU_TIME_FALLBACK_LIMIT_HOURS,
        ICU_TIME_PRE_ADMISSION_HOURS,
    )

    audit: Dict[str, object] = {
        "policy": "icu_episode_with_24h_pre_post_allowance",
        "excluded_rows": 0,
        "excluded_untimed_empty_rows": 0,
        "excluded_untimed_negative_rrt_criteria_rows": 0,
        "normalized_stay_level_rows": 0,
        "fallback_upper_hours": ICU_TIME_FALLBACK_LIMIT_HOURS,
    }
    if "charttime" not in frame.columns:
        audit["policy"] = "not_applicable_stay_level_module"
        return frame, audit

    if module == "outcome":
        charttime = pd.to_numeric(frame["charttime"], errors="coerce")
        audit["policy"] = "stay_level_at_icu_admission"
        audit["normalized_stay_level_rows"] = int(
            (charttime.isna() | (charttime != 0.0)).sum()
        )
        frame = frame.copy()
        frame["charttime"] = 0.0
        return frame, audit

    charttime = pd.to_numeric(frame["charttime"], errors="coerce")
    upper = frame["stay_id"].map(stay_time_upper_bounds).astype("float64")
    upper = upper.fillna(ICU_TIME_FALLBACK_LIMIT_HOURS)
    invalid = charttime.notna() & (
        (charttime < -ICU_TIME_PRE_ADMISSION_HOURS) | (charttime > upper)
    )
    audit["excluded_rows"] = int(invalid.sum())
    audit["rows_with_los_bound"] = int(
        frame["stay_id"].isin(stay_time_upper_bounds).sum()
    )
    kept = frame.loc[~invalid].copy() if bool(invalid.any()) else frame

    # A full outer merge can retain a dependency-only row with no event time.
    # In particular, ``rrt_criteria`` is calculated as a boolean expression, so
    # an otherwise empty row becomes ``False`` instead of remaining missing.
    # It is not a negative observation at an unknown time; it is a merge
    # artifact.  The concept callback already removes this case, but the native
    # publication boundary repeats the guard because later module-wide joins
    # can recreate it.  Positive untimed criteria fail closed.
    null_time = kept["charttime"].isna()
    concept_columns = [
        column
        for column in kept.columns
        if column not in {*_NATIVE_EXPORT_ID_COLUMNS, "charttime"}
    ]
    # Fold one column at a time instead of materialising a dense
    # rows-by-concepts boolean DataFrame at the publication memory peak.
    any_concept_value = pd.Series(False, index=kept.index)
    for column in concept_columns:
        any_concept_value |= kept[column].notna()
    empty_untimed = null_time & ~any_concept_value
    audit["excluded_untimed_empty_rows"] = int(empty_untimed.sum())

    negative_rrt_artifact = pd.Series(False, index=kept.index)
    if module == "renal" and "rrt_criteria" in kept.columns:
        rrt_criteria = kept["rrt_criteria"].astype("boolean")
        positive_untimed = null_time & rrt_criteria.eq(True).fillna(False)
        if bool(positive_untimed.any()):
            sample_ids = (
                kept.loc[positive_untimed, "stay_id"]
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            raise ValueError(
                "native renal export contains positive rrt_criteria rows without "
                f"an event time; sample stay_id={sample_ids}"
            )
        other_concepts = [
            column for column in concept_columns if column != "rrt_criteria"
        ]
        other_value = pd.Series(False, index=kept.index)
        for column in other_concepts:
            other_value |= kept[column].notna()
        negative_rrt_artifact = (
            null_time
            & rrt_criteria.eq(False).fillna(False)
            & ~other_value
        )
        audit["excluded_untimed_negative_rrt_criteria_rows"] = int(
            negative_rrt_artifact.sum()
        )

    excluded_untimed = empty_untimed | negative_rrt_artifact
    if bool(excluded_untimed.any()):
        kept = kept.loc[~excluded_untimed].copy()
    elif kept is frame:
        return frame, audit

    # Preserve canonical numeric dtype even when every row was excluded.
    kept["charttime"] = pd.to_numeric(
        kept["charttime"], errors="coerce"
    ).astype(np.float64)
    return kept, audit


def _enforce_native_export_concept_bounds(
    frame,
    *,
    requested_concepts: List[str],
    dictionary,
) -> Dict[str, Dict[str, object]]:
    """Null values outside declared target-unit bounds in a canonical frame.

    Native-v2 is the final cross-database contract.  Earlier extraction paths
    enforce bounds where possible, but derived concepts and some wide-table
    routes can be created after that guard.  At publication time, keep the row
    and every other concept intact while replacing only the offending numeric
    cell with missing.  This deliberately does not use the loader's
    unit-suspect median escape hatch: a value outside the published target-unit
    contract must never survive in a sealed portable package.

    Returns one audit record per physical concept so the manifest can report
    exactly how many values the final contract excluded.
    """
    import numpy as np

    audit: Dict[str, Dict[str, object]] = {}
    bounds_map = _load_concept_bounds_map()
    for concept in requested_concepts:
        minimum, maximum = bounds_map.get(concept, (None, None))
        bounded = _native_export_storage_kind(concept, dictionary) == "float64" and (
            minimum is not None or maximum is not None
        )
        excluded = 0
        if bounded:
            out_of_bounds = frame[concept].notna()
            if minimum is not None:
                out_of_bounds &= frame[concept] < float(minimum)
            else:
                out_of_bounds &= False
            if maximum is not None:
                above_maximum = frame[concept].notna() & (
                    frame[concept] > float(maximum)
                )
                out_of_bounds |= above_maximum
            excluded = int(out_of_bounds.sum())
            if excluded:
                frame.loc[out_of_bounds, concept] = np.nan

        record: Dict[str, object] = {
            "excluded_out_of_bounds": excluded,
        }
        if bounded:
            record["declared_bounds"] = {
                "minimum": (None if minimum is None else float(minimum)),
                "maximum": (None if maximum is None else float(maximum)),
            }
        audit[concept] = record
    return audit


def _native_export_runtime_provenance() -> Dict[str, object]:
    """Capture enough runtime identity to reproduce or diagnose an export."""
    import hashlib
    import importlib.metadata
    import platform
    import subprocess
    import sys

    import duckdb
    import pyarrow

    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parents[1]
    catalog_files = [
        package_root / "data" / "concept-dict.json",
        package_root / "data" / "sofa2-dict.json",
    ]
    digest = hashlib.sha256()
    for path in catalog_files:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        git_commit = None
    try:
        git_status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        git_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        git_dirty = bool(git_status.strip())
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest() if git_dirty else None
    except Exception:
        git_dirty = None
        git_diff_sha256 = None
    try:
        package_version = importlib.metadata.version("easyicu")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "easyicu_version": package_version,
        "easyicu_git_commit": git_commit,
        "easyicu_git_dirty": git_dirty,
        "easyicu_git_diff_sha256": git_diff_sha256,
        "easyicu_import_path": str(package_root),
        "concept_catalog_sha256": digest.hexdigest(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "duckdb_version": duckdb.__version__,
    }


def _publish_native_export_v2(
    *,
    database: str,
    data_path: str,
    output_dir: str,
    modules: List[str],
    max_patients: Optional[int],
    result: Dict,
) -> Dict[str, object]:
    """Seal completed grouped-module files as one native-v2 package.

    The raw database has already been consumed by the grouped workers. This
    finalization reads only the newly written parquet files once, to validate
    their physical values and bind each column to producer-owned metadata. A
    partial extraction never gets a root native manifest.
    """
    import json
    from datetime import datetime, timezone

    from ..concept.export_metadata import (
        ExportMetadataError,
        build_export_file_metadata_binding,
        missing_primary_metadata_concepts,
    )
    from ..concept.metadata_sidecar import (
        EXPORT_PHYSICAL_SCOPE,
        ColumnMetadataSidecar,
        write_content_addressed_sidecar,
    )
    from ..config import load_src_cfg

    output_root = Path(output_dir)
    root_manifest = output_root / "_manifest.json"
    if root_manifest.exists() or root_manifest.is_symlink():
        raise ValueError(
            "native_export_v2 refuses an existing root _manifest.json; "
            "use a fresh output directory"
        )

    failures = {
        module: list((result.get("modules", {}).get(module) or {}).get("errors") or [])
        for module in modules
        if list((result.get("modules", {}).get(module) or {}).get("errors") or [])
    }
    missing_module_results = [
        module for module in modules if module not in result.get("modules", {})
    ]
    if failures or missing_module_results:
        raise ValueError(
            "native_export_v2 requires every requested module to finish before "
            f"publication (failures={sorted(failures)}, "
            f"missing_results={missing_module_results})"
        )

    # Structural non-availability (for example, a database without a Sepsis-3
    # source) is not an extraction failure. It has no physical file and must not
    # be silently given a typed binding; record it separately and select only
    # the files the producer actually materialized.
    unavailable_modules: List[Dict[str, object]] = []
    # Native-v2 promises one physical parquet per requested module. A database
    # may be structurally unable to produce an entire module (for example,
    # Sepsis-3 in a source without infection timestamps). Seal that absence as
    # a zero-row parquet with the same typed physical schema instead of making
    # downstream code branch on missing files.
    published_modules = list(modules)

    normalized_database = str(database).strip().lower()
    dictionary = load_dictionary(include_sofa2=True)
    source_config = load_src_cfg(normalized_database)
    class_prefixes = tuple(
        str(value).strip().lower()
        for value in source_config.class_prefix
        if str(value).strip()
    )
    requested_concept_plan = {
        module: list(EXTRACT_MODULES[module]) for module in published_modules
    }
    concept_plan: Dict[str, List[str]] = {}
    unavailable_concepts: List[Dict[str, str]] = []
    files: List[Dict[str, object]] = []
    file_bindings = []
    temporary_module_files: List[tuple[Path, Path]] = []
    outcome_source = output_root / "outcome.parquet"
    stay_time_upper_bounds: Dict[int, float] = {}
    if outcome_source.is_file() and not outcome_source.is_symlink():
        stay_time_upper_bounds = _native_export_stay_time_upper_bounds(
            pd.read_parquet(outcome_source)
        )

    for module in published_modules:
        relative_path = f"{module}.parquet"
        source_parquet = output_root / relative_path
        physical_output_missing = not source_parquet.is_file()
        source_rows = 0
        if physical_output_missing:
            original_columns = {"stay_id"}
        else:
            import pyarrow.parquet as _source_pq

            source_file = _source_pq.ParquetFile(source_parquet)
            source_rows = int(source_file.metadata.num_rows)
            source_names = list(source_file.schema_arrow.names)
            if len(set(source_names)) != len(source_names):
                raise ValueError("native export frame has duplicate physical columns")
            original_columns = set(source_names)
        produced_concepts: Optional[set[str]] = (
            set() if physical_output_missing else None
        )
        module_manifest_path = output_root / f"{module}.manifest.json"
        if (
            not physical_output_missing
            and module_manifest_path.is_file()
            and not module_manifest_path.is_symlink()
        ):
            module_manifest = json.loads(
                module_manifest_path.read_text(encoding="utf-8")
            )
            saved = module_manifest.get("saved")
            if isinstance(saved, dict):
                produced_concepts = set()
                for saved_name, record in saved.items():
                    if isinstance(saved_name, str):
                        produced_concepts.add(saved_name)
                    if isinstance(record, dict):
                        produced_concepts.update(
                            concept
                            for concept in (record.get("concepts") or [])
                            if isinstance(concept, str)
                        )
        # A sealed structural placeholder is a real zero-row parquet.  On a
        # later metadata-only republish, file existence alone must not turn it
        # into an "available" module.  The producer manifest's empty ``saved``
        # mapping is the authoritative evidence that no physical concept was
        # produced.  Preserve that status so native-v2 publication is
        # idempotent.
        whole_module_unavailable = physical_output_missing or (
            source_rows == 0 and produced_concepts == set()
        )
        if whole_module_unavailable:
            unavailable_modules.append(
                {
                    "module": module,
                    "reason": "producer_returned_no_physical_output",
                    "concept_ids": list(EXTRACT_MODULES[module]),
                }
            )
        structurally_unavailable = {
            concept
            for concept in requested_concept_plan[module]
            if produced_concepts is not None and concept not in produced_concepts
        }
        concept_plan[module] = [
            concept
            for concept in requested_concept_plan[module]
            if concept not in structurally_unavailable
        ]
        if not whole_module_unavailable:
            unavailable_concepts.extend(
                {
                    "module": module,
                    "concept": concept,
                    "reason": "producer_returned_no_physical_column",
                }
                for concept in requested_concept_plan[module]
                if concept in structurally_unavailable
            )
        # A module manifest is the only producer-owned evidence that a concept
        # is structurally unavailable. Without that evidence, a missing selected
        # physical column is a publication error, not an all-null placeholder.
        missing_selected_columns = [
            concept
            for concept in concept_plan[module]
            if concept not in original_columns
        ]
        if missing_selected_columns:
            for temporary, _destination in temporary_module_files:
                temporary.unlink(missing_ok=True)
            raise ValueError(
                "native_export_v2 cannot seal selected concepts without a primary "
                f"physical binding: {missing_selected_columns}"
            )

        temporary_parquet = None
        frame = None
        try:
            temporary_parquet = output_root / f".{module}.native-v2.tmp.parquet"
            if temporary_parquet.exists() or temporary_parquet.is_symlink():
                raise ValueError(
                    f"native_export_v2 refuses stale temporary file: "
                    f"{temporary_parquet}"
                )
            arrow_result = None
            if not physical_output_missing:
                arrow_result = _try_publish_native_export_arrow_fast_path(
                    source_parquet=source_parquet,
                    temporary_parquet=temporary_parquet,
                    module=module,
                    requested_concepts=requested_concept_plan[module],
                    dictionary=dictionary,
                    stay_time_upper_bounds=stay_time_upper_bounds,
                )
            if arrow_result is None:
                # Demographics has a concept-wise nearest-time selection policy;
                # small duplicate-bearing longitudinal modules also use this
                # exact fallback. No large table may enter it merely because
                # the Arrow path found keys that need consolidation.
                if physical_output_missing:
                    frame = pd.DataFrame(
                        {"stay_id": pd.Series([], dtype="int64")}
                    )
                else:
                    if module == "demographics":
                        fallback_size = _native_export_pandas_fallback_size(
                            source_parquet
                        )
                        if not _native_export_pandas_fallback_is_bounded(
                            fallback_size
                        ):
                            raise ValueError(
                                "native export demographics consolidation exceeds "
                                "the bounded pandas fallback "
                                f"({fallback_size=})"
                            )
                    frame = pd.read_parquet(source_parquet)
                source_charttime = (
                    frame["charttime"].copy()
                    if module == "demographics" and "charttime" in frame
                    else None
                )
                frame = _canonicalise_native_export_frame(
                    frame,
                    module=module,
                    requested_concepts=requested_concept_plan[module],
                    dictionary=dictionary,
                )
                frame, time_axis_audit = _enforce_native_export_time_axis(
                    frame,
                    module=module,
                    stay_time_upper_bounds=stay_time_upper_bounds,
                )
                if module == "demographics":
                    # Treat target-unit bound violations as missing before choosing
                    # the nearest static value. A corrupt +0.1 h height must not
                    # mask a valid -2 h height and then leave the stay empty.
                    bounds_audit = _enforce_native_export_concept_bounds(
                        frame,
                        requested_concepts=requested_concept_plan[module],
                        dictionary=dictionary,
                    )
                    frame, row_grain_audit = _consolidate_native_export_row_grain(
                        frame,
                        module=module,
                        requested_concepts=requested_concept_plan[module],
                        dictionary=dictionary,
                        source_charttime=source_charttime,
                    )
                    post_consolidation_bounds = _enforce_native_export_concept_bounds(
                        frame,
                        requested_concepts=requested_concept_plan[module],
                        dictionary=dictionary,
                    )
                    for concept, post_audit in post_consolidation_bounds.items():
                        bounds_audit[concept]["excluded_out_of_bounds"] += int(
                            post_audit["excluded_out_of_bounds"]
                        )
                else:
                    # Null physical bound violations before taking a duplicate-key
                    # median; otherwise two invalid source values could average to
                    # a plausible value and survive the publication contract.
                    bounds_audit = _enforce_native_export_concept_bounds(
                        frame,
                        requested_concepts=requested_concept_plan[module],
                        dictionary=dictionary,
                    )
                    frame, row_grain_audit = _consolidate_native_export_row_grain(
                        frame,
                        module=module,
                        requested_concepts=requested_concept_plan[module],
                        dictionary=dictionary,
                    )
                row_grain_audit["publication_backend"] = (
                    "pandas_bounded_row_grain_fallback"
                    if not physical_output_missing
                    else "pandas_structural_placeholder"
                )
                frame.to_parquet(
                    temporary_parquet,
                    index=False,
                    engine="pyarrow",
                    compression="snappy",
                )
                metadata_frame = frame
                published_rows = int(frame.shape[0])
                concept_non_null = {
                    concept: int(frame[concept].notna().sum())
                    for concept in requested_concept_plan[module]
                }
            else:
                time_axis_audit = arrow_result["time_axis_audit"]
                bounds_audit = arrow_result["bounds_audit"]
                row_grain_audit = arrow_result["row_grain_audit"]
                metadata_frame = arrow_result["schema_frame"]
                published_rows = int(arrow_result["rows"])
                concept_non_null = dict(arrow_result["concept_non_null"])
            binding = build_export_file_metadata_binding(
                relative_path=relative_path,
                module=module,
                frame=metadata_frame,
                concept_ids=concept_plan[module],
                database=normalized_database,
                database_class_prefixes=class_prefixes,
                dictionary=dictionary,
            )
        except ExportMetadataError as exc:
            for temporary, _destination in temporary_module_files:
                temporary.unlink(missing_ok=True)
            if temporary_parquet is not None:
                temporary_parquet.unlink(missing_ok=True)
            raise ValueError(
                f"native_export_v2 cannot seal {relative_path}: {exc.error}"
            ) from exc
        except Exception:
            for temporary, _destination in temporary_module_files:
                temporary.unlink(missing_ok=True)
            if temporary_parquet is not None:
                temporary_parquet.unlink(missing_ok=True)
            raise
        temporary_module_files.append(
            (temporary_parquet, output_root / relative_path)
        )
        file_bindings.append(binding)
        concept_status = {}
        for concept in requested_concept_plan[module]:
            non_null = int(concept_non_null[concept])
            if concept in structurally_unavailable:
                availability = "structurally_unavailable_placeholder"
            elif non_null == 0:
                availability = "produced_all_null"
            else:
                availability = "available"
            status = {
                "availability": availability,
                "non_null": non_null,
                "excluded_out_of_bounds": bounds_audit[concept][
                    "excluded_out_of_bounds"
                ],
            }
            if "declared_bounds" in bounds_audit[concept]:
                status["declared_bounds"] = bounds_audit[concept]["declared_bounds"]
            concept_status[concept] = status
        import pyarrow.parquet as _pq

        physical_schema = {
            field.name: str(field.type)
            for field in _pq.read_schema(temporary_parquet)
        }
        parquet_sha256 = _native_export_file_sha256(temporary_parquet)
        parquet_bytes = temporary_parquet.stat().st_size
        files.append(
            {
                "file": relative_path,
                "module": module,
                "availability": (
                    "structurally_unavailable"
                    if whole_module_unavailable
                    else "available"
                ),
                "concepts": len(concept_plan[module]),
                "concept_ids": concept_plan[module],
                "physical_concept_ids": requested_concept_plan[module],
                "rows": published_rows,
                "physical_schema": physical_schema,
                "parquet_sha256": parquet_sha256,
                "parquet_bytes": parquet_bytes,
                "primary_key": row_grain_audit["primary_key"],
                "row_grain": row_grain_audit["row_grain"],
                "row_grain_audit": row_grain_audit,
                "time_axis_audit": time_axis_audit,
                "concept_status": concept_status,
                "column_metadata_columns": list(binding.columns),
            }
        )
        # Prevent the previous fallback frame from overlapping the next
        # ``read_parquet`` RHS. Return unused Arrow pages as well so a 19-file
        # package does not appear cumulatively resident to the OS.
        frame = None
        metadata_frame = None
        try:
            import pyarrow as _pa

            _release_stream_batch_memory(_pa)
        except Exception:
            pass

    missing_primary = missing_primary_metadata_concepts(
        concept_plan=concept_plan,
        file_bindings=file_bindings,
    )
    if missing_primary:
        for temporary, _destination in temporary_module_files:
            temporary.unlink(missing_ok=True)
        raise ValueError(
            "native_export_v2 cannot seal selected concepts without a primary "
            f"physical binding: {missing_primary}"
        )

    # No final module is replaced until every file has passed canonical schema
    # projection and typed metadata binding.
    for temporary, destination in temporary_module_files:
        os.replace(temporary, destination)

    sidecar = ColumnMetadataSidecar(
        source_database=normalized_database,
        source_database_class_prefixes=class_prefixes,
        scope=EXPORT_PHYSICAL_SCOPE,
        files=tuple(file_bindings),
    )
    sidecar_ref = write_content_addressed_sidecar(output_root, sidecar)
    module_timings = {
        module: float((result.get("modules", {}).get(module) or {}).get("elapsed") or 0)
        for module in modules
    }
    module_peak_rss_mb = {
        module: float(
            (result.get("modules", {}).get(module) or {}).get("peak_rss_mb") or 0
        )
        for module in modules
    }
    module_peak_working_set_mb = {
        module: float(
            (result.get("modules", {}).get(module) or {}).get(
                "peak_working_set_mb"
            )
            or 0
        )
        for module in modules
    }
    manifest = {
        "schema_version": _NATIVE_EXPORT_SCHEMA_V2,
        "contract_revision": "native_v2_row_grain_sha256_size_20260803",
        "database": normalized_database,
        "data_path": str(data_path),
        "format": "parquet",
        "max_patients": max_patients,
        "generated": datetime.now(timezone.utc).isoformat(),
        "export_kind": "grouped_module_extraction",
        "canonical_physical_schema": {
            "identity_column": "stay_id",
            "time_column": "charttime",
            "time_origin": "icu_admission",
            "time_unit": "h",
            "time_window_policy": (
                "longitudinal modules: ICU episode with 24h pre/post allowance; "
                "outcome: stay-level at ICU admission"
            ),
            "concept_order": "module_catalog",
            "unavailable_representation": "typed_all_null_placeholder",
            "declared_bounds_policy": "out_of_range_to_null",
            "row_grain_contract": {
                "demographics": {
                    "row_grain": "one_row_per_icu_stay",
                    "primary_key": ["stay_id"],
                },
                "all_other_modules": {
                    "row_grain": "one_row_per_icu_stay_relative_hour",
                    "primary_key": ["stay_id", "charttime"],
                    "null_key_equality": "nulls_equal",
                },
            },
        },
        "module_timings_seconds": module_timings,
        "module_peak_rss_mb": module_peak_rss_mb,
        "module_peak_working_set_mb": module_peak_working_set_mb,
        "stream_retry_history": list(result.get("stream_retry_history", [])),
        "runtime_provenance": _native_export_runtime_provenance(),
        "unavailable_modules": unavailable_modules,
        "unavailable_concepts": unavailable_concepts,
        "concept_selection": {
            "mode": "all_in_selected_modules",
            "modules": concept_plan,
        },
        "files": files,
        "feature_definitions": {"included": False},
        "column_metadata": sidecar_ref.to_dict(),
    }
    temporary_manifest = output_root / ".native-export-v2-manifest.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_manifest, root_manifest)
    return {
        "manifest": str(root_manifest),
        "column_metadata": sidecar_ref.file,
        "column_metadata_sha256": sidecar_ref.sha256,
        "output_validation_reads": len(files),
    }


def extract_database(
    database: str,
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    modules: Optional[List[str]] = None,
    patient_ids: Optional[Union[List, Dict]] = None,
    max_patients: Optional[int] = None,
    batch_size: Optional[int] = None,
    group_modules: bool = True,
    native_export_v2: Optional[bool] = None,
    stream_output_batches: bool = False,
    verbose: bool = True,
    adaptive_stream_batches: Optional[bool] = None,
) -> Dict:
    """按 19 个模块分组、子进程隔离地提取整个数据库的全部特征。

    ★ 这是全量特征提取的推荐入口。 不要为了提取全量特征自己写
    `load_concepts` 循环——尤其不要按单概念或小批 patient_ids 循环，那会让
    共享源表(chartevents/labevents…)被反复重读，慢上数倍。

    工作原理与性能：
      * 概念按 19 个模块分组(EXTRACT_MODULE_ORDER)，每个模块一次性
        load_concepts(模块全部概念)，共享源表只扫一次。
      * 共享同族源表的模块进一步合并为分组(_EXTRACT_MODULE_GROUP_AFFINITY)，
        每组一个子进程、组内用 keep_cache 复用 raw/table 缓存：
        chartevents/labevents 等重表每组只扫一次，而不是每模块重扫一遍；
        SOFA 闭包只算一次并被 sofa1/sofa2/sep3_* 复用。缓存受
        EASYICU_CACHE_BUDGET_MB 字节预算约束（默认物理内存的 25%）；
        低内存整库安全性由下述 streamed pilot 与重试合同共同保证。
      * 每组在独立子进程中运行，组退出后 OS 完整回收内存（含 pymalloc
        arena 碎片），主进程 RSS 几乎不增长。group_modules=False 或环境变量
        EASYICU_EXTRACT_GROUPING=0 退回每模块一个子进程的旧行为。
      * 流式导出对低于 24 GiB 的 MIMIC-III、MIMIC-IV 和 AUMC 先运行按实测
        峰值校准的大 pilot，不再未经验证就 one-shot；SIC/HiRID 若保守峰值可
        放入预留后预算则保留 one-shot。每个模块再根据首批真实工作集调整后续
        批次，上限为 67,000 stays。
      * 参考实测：MIMIC-III 全量 61,532 stays 的 SOFA-2 六分量 ~6 分钟。

    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        data_path: 数据路径（None 则按数据库名自动解析）
        output_dir: 输出目录（None 则不写文件，仅返回 dict）
        modules: 要提取的模块列表（None = 全部 19 个模块）
        patient_ids: 患者 ID 列表或 dict（None = 全部患者）
        max_patients: 限制患者数量（与 patient_ids 互斥）
        batch_size: 模块内患者分批大小。None(默认) = 非流式提取时五个较小
            数据库一次性、完整 eICU 按稳定内存预算使用 1–3 个大批次；流式
            提取时按当前可用内存连续计算首批，并根据首批实测工作集调整后续
            批次（上限 67,000 stays）。仅在需要覆盖默认策略时显式传值。
        group_modules: True(默认) = 自动选择：内存充足的服务器将共享源表的
            模块合并为分组子进程；≤24GB 主机或 ≤4GB 显式缓存预算自动切换
            为每模块一个隔离子进程。False = 始终逐模块隔离。可用
            EASYICU_EXTRACT_GROUPING=1/0 强制覆盖。
        native_export_v2: 输出到磁盘时默认启用；基于刚写出的 parquet 建立
            跨库统一 schema 与 typed metadata sidecar，不会重读原始表。传
            False 可显式保留旧版未封装输出。若任何模块或 metadata 绑定失败，
            不会发布根 ``_manifest.json``。
        stream_output_batches: 将显式患者批次直接追加写入模块 parquet，不在
            worker 内合并整模块 DataFrame。用于本地磁盘/内存受限且输出位于
            外置盘的完整导出；会牺牲部分源表复用以换取稳定的峰值内存。
        verbose: 是否打印进度
        adaptive_stream_batches: ``None`` 保持公共默认：自动选择的流式 batch
            会自适应，用户显式 ``batch_size`` 固定不变。六库 launcher 会同时
            显式传入有 provenance 的首批计划和 ``True``，从而保证 plan 与实际
            首批一致，同时允许后续批次继续按实测增长或收缩。

    Note:
        提取 worker 在所有平台均使用 ``spawn`` 以隔离 Arrow/DuckDB 原生状态。
        从独立 Python 脚本调用本函数时，应和 Windows 的 multiprocessing
        要求一样，把入口放在 ``if __name__ == "__main__":`` 保护中。

    Returns:
        dict: {
            'database': str,
            'num_patients': int,
            'modules': {module_name: {'concepts': {name: DataFrame}, 'elapsed': float, 'errors': list}},
            'total_elapsed': float,
            'output_dir': str or None,
        }

    Examples:
        >>> # 提取 AUMC 全部特征到目录
        >>> result = extract_database('aumc', output_dir='/tmp/aumc_export')
        >>> print(f"共 {result['num_patients']} 患者, {result['total_elapsed']:.0f}s")

        >>> # 仅提取 vitals 和 demographics，返回 DataFrame
        >>> result = extract_database('miiv', modules=['vitals', 'demographics'])
        >>> hr_df = result['modules']['vitals']['concepts']['hr']
    """
    import multiprocessing as mp
    import tempfile
    import json
    import time
    import shutil

    from ..runtime.memory_manager import get_rss_mb

    t_start = time.time()

    # 确定数据路径
    if data_path is None:
        data_path = _get_default_db_path(database)
        if data_path is None:
            data_path = get_default_data_path()
    data_path = str(data_path)

    # 磁盘溢写 / 批处理中间文件的默认落点：**输出目录旁的 .easyicu_spill/**，而不是
    # 系统临时目录（常在快满的系统盘上）。输出目录通常在用户为数据特意选的大盘上，
    # 这样零配置即安全，调用方无需每次手设 TMPDIR / EASYICU_DUCKDB_TEMP_DIR。放在
    # 最前，确保后续所有 DuckDB 连接与 fork 出的 worker 子进程都继承此设置。
    # opt-out：显式把 EASYICU_DUCKDB_TEMP_DIR 指向别处（非 .easyicu_spill）则完全尊重。
    # 多库循环：每库各自重指向本库输出旁，故用 basename 判定"是否用户自定义"。
    if output_dir is not None:
        _cur_spill = os.environ.get("EASYICU_DUCKDB_TEMP_DIR")
        _user_spill = (
            _cur_spill is not None
            and os.path.basename(os.path.normpath(_cur_spill)) != ".easyicu_spill"
        )
        if not _user_spill:
            _spill_root = os.path.join(
                os.path.abspath(str(output_dir)), ".easyicu_spill"
            )
            try:
                os.makedirs(_spill_root, exist_ok=True)
                os.environ["EASYICU_DUCKDB_TEMP_DIR"] = _spill_root
                os.environ["TMPDIR"] = _spill_root
                tempfile.tempdir = _spill_root
            except Exception:
                pass

    # 获取患者 ID
    if patient_ids is None:
        all_ids, id_col = _get_all_patient_ids(data_path, database, max_patients)
        if not all_ids:
            raise ValueError(
                f"无法获取 {database} 的患者ID，请检查 data_path: {data_path}"
            )
        patient_ids_filter = {id_col: all_ids}
    else:
        patient_ids_filter = _normalize_patient_ids_for_db(database, patient_ids)
        id_col = list(patient_ids_filter.keys())[0]
        all_ids = list(patient_ids_filter.values())[0]

    num_patients = len(all_ids)

    # 流式导出先选择一个有证据的首批，而不是按数据库大小猜测 one-shot。
    # 2026-08-03 实测显示小队列也可能很重（AUMC 23,106 stays 约 29.31 GiB），
    # 所以低于 24 GiB 时高风险库先按数据库校准 pilot；SIC/HiRID 等较低风险库
    # 仅在保守峰值能放入预留后预算时保留 one-shot。每个模块再用首批 working-set
    # 自适应。显式用户 batch 仍保持固定；launcher 可以显式传计划值并单独打开
    # adaptive_stream_batches，使 provenance 与真实首批完全一致。
    if adaptive_stream_batches and not stream_output_batches:
        raise ValueError(
            "adaptive_stream_batches requires stream_output_batches=True"
        )

    if stream_output_batches:
        automatic_stream_batch = batch_size is None
        batch_size = _resolve_stream_batch_size(
            database,
            num_patients,
            batch_size,
        )
        _adaptive_stream_batches = (
            automatic_stream_batch
            if adaptive_stream_batches is None
            else bool(adaptive_stream_batches)
        )
        _auto_one_shot = False
    elif batch_size is None:
        _adaptive_stream_batches = False
        batch_size = max(num_patients + 1, 2_000_000)
        _auto_one_shot = True
    else:
        _adaptive_stream_batches = False
        _auto_one_shot = False

    # 确定要提取的模块
    if modules is None:
        modules = list(EXTRACT_MODULE_ORDER)
    else:
        # 保持用户指定顺序，但验证模块名
        for m in modules:
            if m not in EXTRACT_MODULES:
                raise ValueError(
                    f"未知模块 '{m}'，可选: {list(EXTRACT_MODULES.keys())}"
                )

    # 创建输出目录
    if native_export_v2 is None:
        native_export_v2 = output_dir is not None
    if output_dir is not None:
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    if native_export_v2 and output_dir is None:
        raise ValueError("native_export_v2 requires output_dir")
    if native_export_v2 and output_dir is not None:
        native_manifest = Path(output_dir) / "_manifest.json"
        if native_manifest.exists() or native_manifest.is_symlink():
            raise ValueError(
                "native_export_v2 refuses an existing root _manifest.json; "
                "use a fresh output directory"
            )

    if verbose:
        rss = get_rss_mb()
        print(f"{'='*60}")
        print(f"📊 extract_database: {database}")
        print(f"   患者数: {num_patients:,}, 模块数: {len(modules)}")
        if (
            _auto_one_shot
            and database == "eicu"
            and num_patients > ONESHOT_MAX_PATIENTS
        ):
            batch_description = "eICU 自适应 1–3 个大 batch（按模块内存估算）"
        elif _auto_one_shot:
            batch_description = "一次性 in-process"
        else:
            batch_description = f"batch_size={batch_size}"
        print(f"   批策略: {batch_description}")
        print(f"   RSS: {rss:.0f}MB, 输出: {output_dir or '仅内存'}")
        print(f"{'='*60}")

    result = {
        "database": database,
        "num_patients": num_patients,
        "batch_size": batch_size,
        "adaptive_stream_batches": _adaptive_stream_batches,
        "stream_retry_history": [],
        "stream_output_batches": stream_output_batches,
        "modules": {},
        "total_elapsed": 0,
        "output_dir": output_dir,
    }

    # 分离普通模块和特殊模块
    normal_modules = [m for m in modules if m not in _SPECIAL_CONCEPT_MODULES]
    special_modules = [m for m in modules if m in _SPECIAL_CONCEPT_MODULES]

    # Always start extraction from a clean interpreter.  Forking after Arrow or
    # DuckDB conversion can inherit allocator/thread-pool state and stale
    # loader caches from the parent.  The environment override remains for
    # controlled expert benchmarks.
    mp_ctx = _get_extraction_mp_context(mp)

    # ---- 模块分组：组内共享源表扫描（keep_cache），组间子进程隔离 ----
    group_flag, group_reason = _resolve_extraction_grouping(
        group_modules, stream_output_batches
    )

    groups = _group_modules_for_extraction(normal_modules, special_modules, group_flag)

    if verbose and group_flag:
        print(
            f"   分组: {len(groups)} 组（组内共享源表扫描；"
            f"EASYICU_EXTRACT_GROUPING=0 或 group_modules=False 关闭）"
        )
    elif verbose:
        print(f"   分组: 关闭（逐模块进程隔离；reason={group_reason}）")

    n_units_total = len(normal_modules) + len(special_modules)
    units_done = 0

    def _collect_module_result(tmp_mod_dir: str, mod_name: str) -> Dict:
        """读回单个模块 worker 的 manifest + parquet 输出。"""
        mod_result = {
            "concepts": {},
            "rows": 0,
            "elapsed": 0.0,
            "errors": [],
            "warnings": [],
            "bounds": {},
            "peak_rss_mb": 0.0,
            "peak_working_set_mb": 0.0,
            "stream_batches": [],
        }
        manifest_path = os.path.join(tmp_mod_dir, "_manifest.json")
        if not os.path.exists(manifest_path):
            mod_result["errors"] = [
                f"{mod_name}: worker produced no manifest (process may have died)"
            ]
            return mod_result
        with open(manifest_path) as f:
            manifest = json.load(f)
        mod_result["errors"] = manifest.get("errors", [])
        mod_result["warnings"] = manifest.get("warnings", [])
        mod_result["elapsed"] = manifest.get("elapsed_sec", 0.0)
        mod_result["peak_rss_mb"] = manifest.get("peak_rss_mb", 0.0)
        mod_result["peak_working_set_mb"] = manifest.get(
            "peak_working_set_mb",
            0.0,
        )
        mod_result["stream_batches"] = manifest.get("stream_batches", [])
        output_manifest = {
            "module": mod_name,
            "saved": {},
            "errors": mod_result["errors"],
            "warnings": mod_result["warnings"],
            "bounds": mod_result["bounds"],
            "elapsed_sec": mod_result["elapsed"],
            "start_rss_mb": manifest.get("start_rss_mb", 0.0),
            "peak_rss_mb": mod_result["peak_rss_mb"],
            "peak_working_set_mb": mod_result["peak_working_set_mb"],
            "available_memory_mb_at_start": manifest.get(
                "available_memory_mb_at_start",
                0.0,
            ),
            "stream_batches": mod_result["stream_batches"],
            "initial_batch_size": manifest.get("initial_batch_size"),
            "final_planned_batch_size": manifest.get(
                "final_planned_batch_size"
            ),
            "adaptive_batch_growth": manifest.get(
                "adaptive_batch_growth",
                False,
            ),
            "patient_partition_strategy": manifest.get(
                "patient_partition_strategy"
            ),
            "initial_planned_partition_count": manifest.get(
                "initial_planned_partition_count"
            ),
        }
        # 每个模块一个宽表 parquet：manifest["saved"] 只有一条（键=模块名），
        # info 里带 concepts（列名清单）+ concept_meta（逐概念 rows/bounds provenance）。
        for _saved_key, info in manifest.get("saved", {}).items():
            pq_path = info.get("path")
            if not pq_path or not os.path.exists(pq_path):
                continue
            module_rows = info.get("rows", 0)
            mod_result["rows"] += module_rows
            concept_meta = info.get("concept_meta", {}) or {}
            concept_names = info.get("concepts") or list(concept_meta.keys())
            # 逐概念 bounds 元数据（provenance）
            for cn, cmeta in concept_meta.items():
                bmeta = _bounds_metadata_from_manifest_info(cmeta)
                if bmeta:
                    mod_result["bounds"][cn] = bmeta
            if output_dir is not None:
                # flat：一个模块一个文件 output_dir/{module}.parquet（不重复 io）
                os.makedirs(output_dir, exist_ok=True)
                dst = os.path.join(output_dir, f"{mod_name}.parquet")
                shutil.move(pq_path, dst)
                module_info = {
                    "path": dst,
                    "rows": module_rows,
                    "concepts": concept_names,
                    "merge_keys": info.get("merge_keys", []),
                    "concept_meta": concept_meta,
                }
                output_manifest["saved"][mod_name] = module_info
                # 逐概念一条（path 都指向该模块宽表），供 summary CSV 保留每概念行数。
                for cn in concept_names:
                    cmeta = concept_meta.get(cn, {})
                    concept_info = {"path": dst, "rows": cmeta.get("rows", module_rows)}
                    for k, v in cmeta.items():
                        if k != "rows":
                            concept_info[k] = v
                    mod_result["concepts"][cn] = concept_info
            else:
                # 无输出目录：读回宽表 DataFrame 到内存（键=模块名）
                mod_result["concepts"][mod_name] = pd.read_parquet(pq_path)
        if output_dir is not None:
            with open(os.path.join(output_dir, f"{mod_name}.manifest.json"), "w") as f:
                json.dump(output_manifest, f)
        return mod_result

    def _count_rows(mod_result: Dict) -> int:
        """Count physical module rows, not rows repeated once per concept."""
        if "rows" in mod_result:
            return int(mod_result["rows"])

        n_rows = 0
        seen_paths = set()
        for v in mod_result["concepts"].values():
            if isinstance(v, dict):
                path = v.get("path")
                if path and path in seen_paths:
                    continue
                if path:
                    seen_paths.add(path)
                n_rows += v.get("rows", 0)
            elif isinstance(v, pd.DataFrame):
                n_rows += len(v)
        return n_rows

    def _collect_special_results(tmp_sp_dir: str, sp_modules: List[str]) -> None:
        """读回特殊模块（Sepsis-3）worker 输出到 result['modules']。"""
        nonlocal units_done
        manifest = None
        manifest_path = os.path.join(tmp_sp_dir, "_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        sp_elapsed = (manifest or {}).get("elapsed_sec", 0.0)
        for mod_name in sp_modules:
            concepts = EXTRACT_MODULES.get(mod_name, [])
            if manifest is None:
                mod_result = {
                    "concepts": {},
                    "rows": 0,
                    "elapsed": 0.0,
                    "errors": [
                        f"{mod_name}: worker produced no manifest (process may have died)"
                    ],
                    "warnings": [],
                    "bounds": {},
                    "peak_rss_mb": 0.0,
                    "peak_working_set_mb": 0.0,
                }
            else:
                mod_result = {
                    "concepts": {},
                    "rows": 0,
                    "elapsed": sp_elapsed,
                    "errors": manifest.get("errors", []),
                    "warnings": manifest.get("warnings", []),
                    "bounds": {},
                    "peak_rss_mb": manifest.get("peak_rss_mb", 0.0),
                    "peak_working_set_mb": manifest.get(
                        "peak_working_set_mb",
                        0.0,
                    ),
                }
                output_manifest = {
                    "module": mod_name,
                    "saved": {},
                    "errors": mod_result["errors"],
                    "warnings": mod_result["warnings"],
                    "bounds": mod_result["bounds"],
                    "elapsed_sec": sp_elapsed,
                    "start_rss_mb": manifest.get("start_rss_mb", 0.0),
                    "peak_rss_mb": manifest.get("peak_rss_mb", 0.0),
                    "peak_working_set_mb": manifest.get(
                        "peak_working_set_mb",
                        0.0,
                    ),
                    "available_memory_mb_at_start": manifest.get(
                        "available_memory_mb_at_start",
                        0.0,
                    ),
                    "batch_size": manifest.get("batch_size"),
                    "batch_count": manifest.get("batch_count"),
                    "patient_partition_strategy": manifest.get(
                        "patient_partition_strategy"
                    ),
                    "initial_planned_partition_count": manifest.get(
                        "initial_planned_partition_count"
                    ),
                }
                for c_name in concepts:
                    info = manifest.get("saved", {}).get(c_name)
                    if info and os.path.exists(info["path"]):
                        rows = info.get("rows", 0)
                        mod_result["rows"] += rows
                        meta = _bounds_metadata_from_manifest_info(info)
                        if meta:
                            mod_result["bounds"][c_name] = meta
                        if output_dir is not None:
                            # flat：派生模块（sepsis3_*）每模块单概念，与普通模块
                            # 统一写 output_dir/{module}.parquet，不再嵌套
                            # {module}/{concept}.parquet（否则 17 扁平 + 2 嵌套的
                            # 混合布局违反"每模块一个宽表"契约）。
                            os.makedirs(output_dir, exist_ok=True)
                            dst = os.path.join(output_dir, f"{mod_name}.parquet")
                            shutil.move(info["path"], dst)
                            concept_info = _concept_result_info(dst, info)
                            concept_info["rows"] = rows
                            mod_result["concepts"][c_name] = concept_info
                            output_manifest["saved"][c_name] = concept_info
                        else:
                            df = pd.read_parquet(info["path"])
                            _attach_bounds_metadata(df, info)
                            mod_result["concepts"][c_name] = df
                if output_dir is not None:
                    with open(
                        os.path.join(output_dir, f"{mod_name}.manifest.json"), "w"
                    ) as f:
                        json.dump(output_manifest, f)
            result["modules"][mod_name] = mod_result
            units_done += 1
            if verbose:
                print(
                    f"   {'✅' if not mod_result['errors'] else '⚠️'} "
                    f"[{units_done}/{n_units_total}] {mod_name}: "
                    f"{len(mod_result['concepts'])} concepts, "
                    f"{_count_rows(mod_result):,} rows, {sp_elapsed:.1f}s"
                )

    # ---- 逐组在子进程中加载 ----
    from collections import deque

    pending_groups = deque(groups)
    while pending_groups:
        group = pending_groups.popleft()
        group_batch_size = int(group.get("_batch_size", batch_size))
        stream_retry_attempt = int(group.get("_stream_retry_attempt", 0))
        group_mods = [m for m in group["modules"] if EXTRACT_MODULES.get(m)]
        group_special = list(group["special"])
        if not group_mods and not group_special:
            continue

        module_specs = [(m, EXTRACT_MODULES[m]) for m in group_mods]
        group_use_sofa2 = any(_concepts_need_sofa2(c) for _, c in module_specs) or any(
            "sofa2" in m for m in group_special
        )

        tmp_root = tempfile.mkdtemp(prefix="easyicu_grp_")
        if verbose:
            rss = get_rss_mb()
            label = " + ".join(group_mods + group_special)
            print(f"\n⏳ {label} ... RSS={rss:.0f}MB")

        proc = mp_ctx.Process(
            target=_extract_module_group_worker,
            args=(
                module_specs,
                group_special,
                database,
                data_path,
                patient_ids_filter,
                group_batch_size,
                tmp_root,
                group_use_sofa2,
                stream_output_batches,
                output_dir,
                _adaptive_stream_batches,
            ),
            daemon=True,
        )
        proc.start()
        proc.join()

        # 组 worker 硬崩溃（如 OOM kill）：已完成模块正常读回；未完成的
        # 模块拆成单模块组重试一次，避免一个组的失败拖垮整组输出。
        crashed = proc.exitcode not in (0, None)
        incomplete_mods = [
            m
            for m in group_mods
            if not os.path.exists(os.path.join(tmp_root, m, "_manifest.json"))
        ]
        special_incomplete = bool(group_special) and not os.path.exists(
            os.path.join(tmp_root, _SPECIAL_OUTPUT_DIRNAME, "_manifest.json")
        )
        can_split = len(group_mods) + (1 if group_special else 0) > 1
        incomplete = bool(incomplete_mods or special_incomplete)
        can_retry_smaller = (
            crashed
            and incomplete
            and _adaptive_stream_batches
            and stream_output_batches
            and stream_retry_attempt < _STREAM_BATCH_MAX_RETRIES
            and group_batch_size > _STREAM_BATCH_MIN
        )
        if can_retry_smaller and not can_split:
            retry_batch_size = _next_stream_retry_batch_size(group_batch_size)
            retry_modules = list(group_mods)
            retry_special = list(group_special)
            result["stream_retry_history"].append(
                {
                    "modules": retry_modules + retry_special,
                    "worker_exit_code": proc.exitcode,
                    "attempt": stream_retry_attempt + 1,
                    "previous_batch_size": group_batch_size,
                    "retry_batch_size": retry_batch_size,
                }
            )
            if verbose:
                print(
                    f"   ⚠️ worker exit={proc.exitcode}; adaptive memory retry "
                    f"{retry_modules + retry_special}: "
                    f"batch_size {group_batch_size} -> {retry_batch_size} "
                    f"(attempt {stream_retry_attempt + 1}/"
                    f"{_STREAM_BATCH_MAX_RETRIES})"
                )
            pending_groups.appendleft(
                {
                    "modules": retry_modules,
                    "special": retry_special,
                    "_batch_size": retry_batch_size,
                    "_stream_retry_attempt": stream_retry_attempt + 1,
                }
            )
            # Do not publish an error result from the killed attempt.  Its
            # private temporary directory is removed below; earlier completed
            # modules in the database output remain untouched.
            group_mods = []
            group_special = []
        elif crashed and can_split and incomplete:
            if verbose:
                retry_units = incomplete_mods + (
                    group_special if special_incomplete else []
                )
                print(
                    f"   ⚠️ group worker exit={proc.exitcode}; "
                    f"retrying individually: {retry_units}"
                )
            if special_incomplete:
                pending_groups.appendleft(
                    {
                        "modules": [],
                        "special": group_special,
                        "_batch_size": group_batch_size,
                        "_stream_retry_attempt": stream_retry_attempt,
                    }
                )
                group_special = []
            for m in reversed(incomplete_mods):
                pending_groups.appendleft(
                    {
                        "modules": [m],
                        "special": [],
                        "_batch_size": group_batch_size,
                        "_stream_retry_attempt": stream_retry_attempt,
                    }
                )
            group_mods = [m for m in group_mods if m not in incomplete_mods]

        for mod_name in group_mods:
            mod_result = _collect_module_result(
                os.path.join(tmp_root, mod_name), mod_name
            )
            result["modules"][mod_name] = mod_result
            units_done += 1
            if verbose:
                status = "✅" if not mod_result["errors"] else "⚠️"
                print(
                    f"   {status} [{units_done}/{n_units_total}] {mod_name}: "
                    f"{len(mod_result['concepts'])} concepts, "
                    f"{_count_rows(mod_result):,} rows, {mod_result['elapsed']:.1f}s"
                    + (
                        f" | errors: {mod_result['errors']}"
                        if mod_result["errors"]
                        else ""
                    )
                    + (
                        f" | warnings: {mod_result['warnings']}"
                        if mod_result.get("warnings")
                        else ""
                    )
                )

        if group_special:
            _collect_special_results(
                os.path.join(tmp_root, _SPECIAL_OUTPUT_DIRNAME), group_special
            )

        # 清理临时目录
        shutil.rmtree(tmp_root, ignore_errors=True)

    if native_export_v2:
        assert output_dir is not None
        result["native_export_v2"] = _publish_native_export_v2(
            database=database,
            data_path=data_path,
            output_dir=output_dir,
            modules=modules,
            max_patients=max_patients,
            result=result,
        )

    total_elapsed = time.time() - t_start
    result["total_elapsed"] = round(total_elapsed, 1)

    if verbose:
        rss = get_rss_mb()
        total_concepts = sum(len(m["concepts"]) for m in result["modules"].values())
        total_rows = sum(_count_rows(m) for m in result["modules"].values())
        all_errors = [e for m in result["modules"].values() for e in m["errors"]]
        all_warnings = [
            w for m in result["modules"].values() for w in m.get("warnings", [])
        ]
        print(f"\n{'='*60}")
        print(
            f"✅ {database} 完成: {total_concepts} concepts, "
            f"{total_rows:,} rows, {total_elapsed:.1f}s"
        )
        print(
            f"   RSS: {rss:.0f}MB" + (f"  |  输出: {output_dir}" if output_dir else "")
        )
        if all_errors:
            print(f"   ⚠️ {len(all_errors)} 错误: {all_errors[:5]}")
        if all_warnings:
            print(f"   ⚠️ {len(all_warnings)} 警告: {all_warnings[:5]}")
        print(f"{'='*60}")

    return result


def extract_all_databases(
    databases: Optional[List[str]] = None,
    data_paths: Optional[Dict[str, str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    modules: Optional[List[str]] = None,
    max_patients: Optional[int] = None,
    batch_size: Optional[int] = None,
    native_export_v2: Optional[bool] = None,
    verbose: bool = True,
) -> Dict:
    """逐库逐模块子进程隔离提取所有数据库的全部特征。

    每个模块运行在独立子进程中，主进程内存几乎不增长。
    适用于 16GB 内存环境。

    Args:
        databases: 要提取的数据库列表（None = 全部 6 个: sic, aumc, hirid, mimic, miiv, eicu）
        data_paths: {database: path} 覆盖默认路径
        output_dir: 输出根目录（每个库一个子目录）
        modules: 要提取的模块列表（None = 全部）
        max_patients: 每个库的患者数量限制
        batch_size: 子进程内患者分批大小
        native_export_v2: 输出到磁盘时默认为每个完整数据库发布 native-v2
            统一 schema 与 typed metadata package；False 显式关闭
        verbose: 是否打印进度

    Returns:
        dict: {database_name: extract_database() 返回值}

    Examples:
        >>> results = extract_all_databases(output_dir='/tmp/all_export')
        >>> for db, r in results.items():
        ...     print(f"{db}: {r['num_patients']:,} patients, {r['total_elapsed']:.0f}s")
    """
    import time

    if databases is None:
        databases = ["sic", "aumc", "hirid", "mimic", "miiv", "eicu"]

    merged_paths = _build_default_db_paths()
    if data_paths:
        merged_paths.update(data_paths)

    t_start = time.time()
    results = {}

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# extract_all_databases: {len(databases)} 个数据库")
        print(f"# 模块: {modules or '全部'}")
        print(f"# 输出: {output_dir or '仅内存'}")
        print(f"{'#'*60}")

    for db_idx, db in enumerate(databases):
        dp = merged_paths.get(db)
        if dp is None:
            if verbose:
                print(f"\n⚠️ 跳过 {db}: 无数据路径")
            continue

        if not os.path.isdir(dp):
            if verbose:
                print(f"\n⚠️ 跳过 {db}: 路径不存在 {dp}")
            continue

        db_output = None
        if output_dir is not None:
            db_output = os.path.join(str(output_dir), db)

        if verbose:
            print(f"\n{'━'*60}")
            print(f"  [{db_idx+1}/{len(databases)}] 🏥 {db.upper()}")
            print(f"{'━'*60}")

        try:
            r = extract_database(
                database=db,
                data_path=dp,
                output_dir=db_output,
                modules=modules,
                max_patients=max_patients,
                batch_size=batch_size,
                native_export_v2=native_export_v2,
                verbose=verbose,
            )
            results[db] = r
        except Exception as e:
            if verbose:
                print(f"  ❌ {db} 失败: {e}")
            results[db] = {"error": str(e)}

    total = time.time() - t_start

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# 全部完成: {total:.1f}s")
        for db, r in results.items():
            if "error" in r:
                print(f"#   {db}: ❌ {r['error']}")
            else:
                nc = sum(len(m["concepts"]) for m in r["modules"].values())
                nr = sum(
                    int(m.get("rows", 0))
                    if "rows" in m
                    else sum(
                        len(v) if hasattr(v, "__len__") else 0
                        for v in m["concepts"].values()
                    )
                    for m in r["modules"].values()
                )
                print(
                    f"#   {db}: {r['num_patients']:,} patients, "
                    f"{nc} concepts, {nr:,} rows, {r['total_elapsed']:.0f}s"
                )
        print(f"{'#'*60}")

    return results
