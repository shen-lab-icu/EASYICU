"""Runtime defaults for loader execution profiles.

Expose helper utilities so scripts/tests can stay concise while pyricu
handles patient-scale heuristics and environment overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Dict, Any
import math
import os

@dataclass(frozen=True)
class LoaderDefaults:
    chunk_size: Optional[int]
    parallel_workers: int
    concept_workers: int
    parallel_backend: str
    profile: str
    source: Dict[str, str]

    def summary(self) -> str:
        chunk_label = self.chunk_size if self.chunk_size else "disabled"
        return (
            f"chunk_size={chunk_label} [{self.source['chunk']}], "
            f"parallel_workers={self.parallel_workers} [{self.source['parallel']}], "
            f"concept_workers={self.concept_workers} [{self.source['concept']}], "
            f"backend={self.parallel_backend} [{self.source['backend']}]"
        )

    def as_loader_kwargs(self, *, progress: bool = False) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "parallel_workers": self.parallel_workers,
            "concept_workers": self.concept_workers,
            "parallel_backend": self.parallel_backend,
            "progress": progress,
        }

_DEFAULT_ENV_KEYS: Dict[str, Sequence[str]] = {
    "chunk": ("PYRICU_CHUNK_SIZE",),
    "parallel": ("PYRICU_PARALLEL_WORKERS",),
    "concept": ("PYRICU_CONCEPT_WORKERS",),
    "backend": ("PYRICU_PARALLEL_BACKEND",),
}

_DEFAULT_TIERS = (
    # ⚡ 性能优化: 减少workers避免Python GIL和锁竞争导致的性能下降
    # 实测显示: 多线程在数据加载场景下反而会因为锁竞争变慢
    {"max_patients": 100, "profile": "tiny", "chunk": 100, "workers": 1, "concept_workers": 1, "backend": "thread"},
    {"max_patients": 500, "profile": "small", "chunk": 125, "workers": 1, "concept_workers": 1, "backend": "thread"},
    {"max_patients": 2000, "profile": "medium", "chunk": 200, "workers": 1, "concept_workers": 1, "backend": "thread"},
    {"max_patients": 8000, "profile": "large", "chunk": 400, "workers": 2, "concept_workers": 1, "backend": "thread"},
    {"max_patients": math.inf, "profile": "xlarge", "chunk": 800, "workers": 4, "concept_workers": 1, "backend": "thread"},
)

def resolve_loader_defaults(
    patient_goal: Optional[int],
    *,
    env: Optional[Mapping[str, str]] = None,
    env_keys: Optional[Dict[str, Sequence[str]]] = None,
    tiers: Sequence[Mapping[str, Any]] = _DEFAULT_TIERS,
    default_concept_workers: int = 1,
) -> LoaderDefaults:
    """Return loader defaults tuned to the requested patient volume."""

    env = dict(env or os.environ)
    merged_keys = {
        name: tuple(_dedupe_keys((env_keys or {}).get(name), _DEFAULT_ENV_KEYS.get(name)))
        for name in _DEFAULT_ENV_KEYS
    }

    target = patient_goal if patient_goal and patient_goal > 0 else None
    if not target:
        target = 5000

    profile = _select_tier(target, tiers)

    chunk_value, chunk_source = _read_int(merged_keys["chunk"], env, minimum=0)
    if chunk_value == 0:
        chunk_value = None
    if chunk_value is None:
        chunk_value = profile["chunk"]
        chunk_source = f"auto({profile['profile']})"

    parallel_value, parallel_source = _read_int(merged_keys["parallel"], env, minimum=1)
    if parallel_value is None:
        parallel_value = profile["workers"]
        parallel_source = f"auto({profile['profile']})"

    concept_value, concept_source = _read_int(merged_keys["concept"], env, minimum=1)
    if concept_value is None:
        # Use profile-specific concept_workers if available, otherwise fallback to default
        profile_concept_workers = profile.get("concept_workers", default_concept_workers)
        concept_value = max(1, profile_concept_workers)
        concept_source = f"auto({profile['profile']})"

    backend_value, backend_source = _read_choice(
        merged_keys["backend"],
        env,
        choices={"thread", "process", "auto"},
    )
    if backend_value is None:
        backend_value = profile["backend"]
        backend_source = f"auto({profile['profile']})"

    return LoaderDefaults(
        chunk_size=chunk_value,
        parallel_workers=parallel_value,
        concept_workers=concept_value,
        parallel_backend=backend_value,
        profile=profile["profile"],
        source={
            "chunk": chunk_source,
            "parallel": parallel_source,
            "concept": concept_source,
            "backend": backend_source,
        },
    )

def _select_tier(target: int, tiers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for tier in tiers:
        threshold = tier.get("max_patients", math.inf)
        if target <= threshold:
            return tier
    return tiers[-1]

def _dedupe_keys(custom: Optional[Sequence[str]], defaults: Optional[Sequence[str]]) -> Sequence[str]:
    seen = set()
    ordered = []
    for key_list in (custom or []), (defaults or []):
        for key in key_list or []:
            if key and key not in seen:
                seen.add(key)
                ordered.append(key)
    return tuple(ordered)

def _read_int(
    keys: Sequence[str],
    env: Mapping[str, str],
    *,
    minimum: int,
) -> tuple[Optional[int], Optional[str]]:
    for key in keys:
        raw = env.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= minimum:
            return value, f"env({key})"
    return None, None

def _read_choice(
    keys: Sequence[str],
    env: Mapping[str, str],
    *,
    choices: Sequence[str],
) -> tuple[Optional[str], Optional[str]]:
    allowed = {choice.lower(): choice for choice in choices}
    for key in keys:
        raw = env.get(key)
        if raw is None:
            continue
        normalized = raw.strip().lower()
        match = allowed.get(normalized)
        if match:
            return match, f"env({key})"
    return None, None
