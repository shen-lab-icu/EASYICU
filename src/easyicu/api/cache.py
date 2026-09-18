"""Explicit concept-result cache services for the public API."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from easyicu.content_identity import data_path_fingerprint

#: Environment variable carrying the secret for ``.trusted.pkl`` HMAC signing.
#: Kept as a module constant so the three ``use_pickle`` call sites document
#: the same gate. ``use_pickle`` defaults to ``False`` (parquet path) and stays
#: that way; enabling pickle without this key is refused fail-closed.
CACHE_HMAC_ENV_VAR = "EASYICU_CACHE_HMAC_KEY"


def _hmac_key() -> Optional[bytes]:
    """Return the configured HMAC key, or ``None`` when unset/empty."""
    raw = os.environ.get(CACHE_HMAC_ENV_VAR, "")
    raw = raw.strip() if isinstance(raw, str) else ""
    if not raw:
        return None
    return raw.encode("utf-8")


def _require_hmac_key() -> bytes:
    """Return the HMAC key or raise with an actionable opt-in message."""
    key = _hmac_key()
    if key is None:
        raise RuntimeError(
            "use_pickle=True requires a non-empty "
            f"{CACHE_HMAC_ENV_VAR} so .trusted.pkl is HMAC-signed "
            "(.trusted.pkl.hmac); refusing unsigned pickle cache. Set "
            f"{CACHE_HMAC_ENV_VAR} to a secret value in a fully controlled "
            "local environment."
        )
    return key


def _hmac_path(cache_file: Path) -> Path:
    """Return the sidecar path for a ``.trusted.pkl`` cache file."""
    return Path(str(cache_file) + ".hmac")


def _compute_hmac(payload: bytes, key: bytes, cache_name: str) -> str:
    """Bind the serialized payload to its cache request, not just its bytes."""
    signer = hmac.new(key, digestmod=hashlib.sha256)
    signer.update(b"easyicu.trusted-pickle/2\0")
    signer.update(cache_name.encode("utf-8"))
    signer.update(b"\0")
    signer.update(payload)
    return signer.hexdigest()


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
    """Load concept data through an explicit, dataset-isolated disk cache.

    ``use_pickle`` defaults to ``False`` (safe parquet path) and stays that
    way. ``use_pickle=True`` is a trusted-local compatibility opt-in that
    additionally requires ``EASYICU_CACHE_HMAC_KEY``: the pickle payload is
    HMAC-SHA256 signed into a ``.trusted.pkl.hmac`` sidecar on write and
    verified on read. Unsigned or tampered entries are deleted and recomputed.
    """
    resolved_cache_dir = Path(cache_dir or (Path(data_path) / "cache"))
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    # Fail closed before touching the cache: pickle must never be enabled
    # without an HMAC key, even when the entry would otherwise be a hit.
    hmac_key: Optional[bytes] = None
    if use_pickle:
        hmac_key = _require_hmac_key()
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
                assert hmac_key is not None  # guarded by _require_hmac_key above
                raw = cache_file.read_bytes()
                sidecar = _hmac_path(cache_file)
                if not sidecar.is_file():
                    raise ValueError(
                        "missing HMAC sidecar "
                        f"{sidecar.name}; refusing unsigned pickle cache"
                    )
                expected = sidecar.read_text(encoding="utf-8").strip().casefold()
                actual = _compute_hmac(raw, hmac_key, cache_file.name).casefold()
                if not hmac.compare_digest(expected, actual):
                    raise ValueError(
                        "HMAC mismatch for "
                        f"{cache_file.name}; possible tampered cache entry"
                    )
                result = pickle.loads(raw)
            else:
                result = pd.read_parquet(cache_file)
            if verbose:
                size = len(result)
                unit = "行缓存数据" if isinstance(result, pd.DataFrame) else "个概念"
                print(f"✅ 成功加载 {size:,} {unit}")
            return result
        except Exception as exc:
            if use_pickle:
                # Fail closed on tamper/unsigned entries: remove both files so
                # the next call cannot reuse them, then fall through to
                # recompute. The warning (not just verbose print) is what the
                # tamper regression test asserts on.
                for stale in (cache_file, _hmac_path(cache_file)):
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                warnings.warn(
                    "pickle cache HMAC verification failed "
                    f"({type(exc).__name__}: {exc}); cache entry deleted, "
                    "recomputing",
                    UserWarning,
                    stacklevel=2,
                )
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
            assert hmac_key is not None  # guarded by _require_hmac_key above
            # Sign the bytes produced by this process. Re-reading the shared
            # path before signing could authenticate a concurrent replacement.
            raw = pickle.dumps(result)
            cache_file.write_bytes(raw)
            _hmac_path(cache_file).write_text(
                _compute_hmac(raw, hmac_key, cache_file.name),
                encoding="utf-8",
            )
        elif isinstance(result, pd.DataFrame):
            result.to_parquet(cache_file, index=False)
        if verbose:
            print(f"💾 缓存已保存: {cache_file.name}")
    except Exception as exc:
        if verbose:
            print(f"⚠️  缓存保存失败（{type(exc).__name__}）")
    return result


__all__ = [
    "CACHE_HMAC_ENV_VAR",
    "data_path_fingerprint",
    "get_cache_key",
    "load_concept_cached_impl",
]
