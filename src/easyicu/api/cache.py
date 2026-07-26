"""Explicit concept-result cache services for the public API."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd


def get_cache_key(concepts: List[str], source: str, **kwargs) -> str:
    """Return a full SHA-256 key for a canonical cache request."""
    payload = {
        "source": source,
        "concepts": sorted(concepts),
        "parameters": kwargs,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def data_path_fingerprint(
    data_path: Union[str, Path],
    *,
    exclude_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Fingerprint dataset identity and relevant file metadata."""
    root = Path(data_path).expanduser().resolve()
    excluded = Path(exclude_dir).expanduser().resolve() if exclude_dir else None
    digest = hashlib.sha256(str(root).encode())
    if root.is_file():
        stat = root.stat()
        digest.update(f"{root.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    suffixes = {".parquet", ".csv", ".gz", ".json"}
    files = (
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and (excluded is None or not path.is_relative_to(excluded))
    )
    for path in sorted(files, key=lambda item: str(item.relative_to(root))):
        stat = path.stat()
        relative = path.relative_to(root)
        digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def load_concept_cached_impl(
    concepts: Union[str, List[str]],
    source: str,
    data_path: Union[str, Path],
    *,
    get_cache_key_fn: Callable[..., str],
    data_path_fingerprint_fn: Callable[..., str],
    load_concepts_fn: Callable[..., Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    align_time_fn: Callable[..., object],
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    patient_ids: Optional[List] = None,
    merge: bool = True,
    align_time: bool = False,
    verbose: bool = True,
    use_pickle: bool = True,
    n_patients: Optional[int] = None,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load concept data through an explicit, dataset-isolated disk cache."""
    resolved_cache_dir = Path(cache_dir or (Path(data_path) / "cache"))
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    concept_list = [concepts] if isinstance(concepts, str) else list(concepts)
    cache_params = {
        "merge": merge,
        "align_time": align_time,
        "patient_ids": patient_ids,
        "n_patients": n_patients,
        "data_path": str(Path(data_path).expanduser().resolve()),
        "data_fingerprint": data_path_fingerprint_fn(
            data_path,
            exclude_dir=resolved_cache_dir,
        ),
        **kwargs,
    }
    cache_key = get_cache_key_fn(concept_list, source, **cache_params)
    cache_ext = "pkl" if use_pickle else "csv"
    cache_file = resolved_cache_dir / (
        f"{source}_{'_'.join(concept_list[:3])}_{cache_key[:32]}.{cache_ext}"
    )

    if not force_reload and cache_file.exists():
        if verbose:
            print(f"📦 从缓存加载: {cache_file.name}")
        try:
            if use_pickle:
                with cache_file.open("rb") as handle:
                    result = pickle.load(handle)
            else:
                result = pd.read_csv(cache_file, parse_dates=["charttime"])
            if verbose:
                size = len(result)
                unit = "行缓存数据" if isinstance(result, pd.DataFrame) else "个概念"
                print(f"✅ 成功加载 {size:,} {unit}")
            return result
        except Exception as exc:
            if verbose:
                print(f"⚠️  缓存加载失败（{type(exc).__name__}），重新提取...")

    result = load_concepts_fn(
        concepts=concept_list,
        patient_ids=patient_ids,
        database=source,
        data_path=data_path,
        merge=merge,
        verbose=verbose,
        n_patients=n_patients,
        **kwargs,
    )
    if align_time:
        result = align_time_fn(
            result,
            database=source,
            data_path=data_path,
            verbose=verbose,
        )
    try:
        if use_pickle:
            with cache_file.open("wb") as handle:
                pickle.dump(result, handle)
        elif isinstance(result, pd.DataFrame):
            result.to_csv(cache_file, index=False)
        if verbose:
            print(f"💾 缓存已保存: {cache_file.name}")
    except Exception as exc:
        if verbose:
            print(f"⚠️  缓存保存失败（{type(exc).__name__}）")
    return result


__all__ = [
    "data_path_fingerprint",
    "get_cache_key",
    "load_concept_cached_impl",
]
