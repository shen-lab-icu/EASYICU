"""Explicit concept-result cache services for the public API."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from easyicu.content_identity import file_content_receipt, verify_content_receipt


_CONTENT_RECEIPT_INDEX = ".easyicu_content_receipts.json"
_CONTENT_RECEIPT_LOCK = threading.RLock()


def _load_receipt_index(index_path: Optional[Path], root: Path) -> dict[str, dict]:
    if index_path is None or not index_path.is_file():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if payload.get("schema_version") != 1 or payload.get("root") != str(root):
        return {}
    files = payload.get("files")
    return files if isinstance(files, dict) else {}


def _save_receipt_index(
    index_path: Optional[Path], root: Path, receipts: dict[str, dict]
) -> None:
    if index_path is None:
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"schema_version": 1, "root": str(root), "files": receipts},
        sort_keys=True,
        separators=(",", ":"),
    )
    temporary = index_path.with_name(
        f"{index_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, index_path)


def _current_receipt(path: Path, previous: object) -> dict:
    matches, current = verify_content_receipt(path, previous)
    if matches and current is not None:
        return current
    return file_content_receipt(path)


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
    """Fingerprint dataset content with a persistent stat-to-digest index."""
    root = Path(data_path).expanduser().resolve()
    excluded = Path(exclude_dir).expanduser().resolve() if exclude_dir else None
    index_path = excluded / _CONTENT_RECEIPT_INDEX if excluded else None
    excluded_subtree = (
        excluded
        if excluded is not None
        and excluded != root
        and excluded.is_relative_to(root)
        else None
    )
    digest = hashlib.sha256(str(root).encode())

    suffixes = {".parquet", ".csv", ".gz", ".json"}
    if root.is_file():
        files = [root]
    else:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in suffixes
            and (index_path is None or path != index_path)
            and (
                excluded_subtree is None
                or not path.is_relative_to(excluded_subtree)
            )
        ]

    with _CONTENT_RECEIPT_LOCK:
        previous = _load_receipt_index(index_path, root)
        current_receipts: dict[str, dict] = {}
        for path in sorted(
            files,
            key=lambda item: item.name
            if root.is_file()
            else str(item.relative_to(root)),
        ):
            relative = path.name if root.is_file() else str(path.relative_to(root))
            receipt = _current_receipt(path, previous.get(relative))
            current_receipts[relative] = receipt
            digest.update(f"{relative}:{receipt['sha256']}\n".encode())
        _save_receipt_index(index_path, root, current_receipts)
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
    use_pickle: bool = False,
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
    cache_file = resolved_cache_dir / (
        f"{cache_key}.trusted.pkl" if use_pickle else f"{cache_key}.parquet"
    )

    if not force_reload and cache_file.exists():
        if verbose:
            print(f"📦 从缓存加载: {cache_file.name}")
        try:
            if use_pickle:
                with cache_file.open("rb") as handle:
                    result = pickle.load(handle)
            else:
                result = pd.read_parquet(cache_file)
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
            result.to_parquet(cache_file, index=False)
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
