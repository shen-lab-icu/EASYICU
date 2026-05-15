"""Pipeline cache: deterministic key derivation + on-disk index.

Extracted from :mod:`easyicu.research_agent.pipeline` so the cache
surface can be reasoned about (and tested) without dragging the
full ``ResearchAgentPipeline`` class in.

The cache key is a sha256 hex digest of a canonical JSON payload that
combines:

* the cohort bytes hash;
* normalised run knobs (question, target_outcome, skill, database, ...);
* a bag of caller-supplied flags (``enable_*``, ``max_*``, ``latex_*``,
  ``context_top_k``, ``disable_icu_context``, ...);
* a short signature of the configured LLM(s).

Cache hits are validated by checking that the manifest file referenced
by an index entry still exists; stale entries are silently evicted.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .llm import MockLLMClient
from .schema import PipelineResult


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def hash_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Streaming sha256 of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def llm_signature(llm: Any) -> str:
    """Return a short string identifying the configured LLM(s).

    Two runs with different LLMs *must* invalidate the cache because the
    bound manuscript / generated code / chosen plan could differ. The
    signature is canonicalised so router order doesn't matter.
    """
    if llm is None:
        return "unconfigured"
    if isinstance(llm, MockLLMClient):
        return "mock"
    if hasattr(llm, "iter_clients"):
        sigs = sorted(llm_signature(c) for c in llm.iter_clients())
        return "router(" + ",".join(sigs) + ")"
    model = getattr(llm, "_model", None)
    cls = getattr(llm, "name", llm.__class__.__name__)
    return f"{cls}:{model}" if model else str(cls)


def iter_mock_clients(llm: Any):
    """Yield every :class:`MockLLMClient` reachable through ``llm``.

    For a plain client this is just ``llm`` itself when it's a Mock;
    for an :class:`LLMRouter` we walk the router's child clients.
    Idempotent for non-Mock clients.
    """
    if llm is None:
        return
    if isinstance(llm, MockLLMClient):
        yield llm
        return
    if hasattr(llm, "iter_clients"):
        for child in llm.iter_clients():
            if isinstance(child, MockLLMClient):
                yield child


# ---------------------------------------------------------------------------
# PipelineCache class — wraps the on-disk index
# ---------------------------------------------------------------------------


class PipelineCache:
    """Encapsulates cache key generation and the on-disk cache index.

    The caller (typically the pipeline) constructs this with a target
    directory and then calls :meth:`compute_key` / :meth:`lookup` /
    :meth:`record_hit` directly. Configuration knobs that participate
    in the cache key are passed in as a ``flags`` mapping so the cache
    module stays decoupled from the pipeline's specific flag set.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)

    # -- index path -----------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self._cache_dir / "cache_index.json"

    # -- key derivation -------------------------------------------------

    def compute_key(
        self,
        *,
        cohort_path: Path,
        question: Optional[str],
        target_outcome: Optional[str],
        skill_key: Optional[str],
        database: Optional[str],
        llm: Any,
        stop_after_analysis: bool,
        manuscript_language: str,
        flags: Mapping[str, Any],
    ) -> str:
        """Compose the deterministic cache key for this run.

        ``flags`` is a free-form mapping of additional configuration
        knobs that should invalidate the cache when changed. Values are
        coerced through ``json.dumps(sort_keys=True)`` so they must be
        JSON-serialisable.
        """
        payload: Dict[str, Any] = {
            "cohort_sha256": hash_file(cohort_path),
            "question": (question or "").strip(),
            "target_outcome": (target_outcome or "").strip(),
            "skill": (skill_key or "").strip(),
            "database": (database or "").strip(),
            "stop_after_analysis": bool(stop_after_analysis),
            "manuscript_language": manuscript_language,
            "llm": llm_signature(llm),
        }
        for key, value in flags.items():
            if isinstance(value, bool):
                payload[key] = bool(value)
            else:
                payload[key] = value
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- index storage --------------------------------------------------

    def load_index(self) -> Dict[str, Dict[str, str]]:
        path = self.index_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}

    def save_index(self, index: Dict[str, Dict[str, str]]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    # -- query / record -------------------------------------------------

    def lookup(self, cache_key: str) -> Optional[PipelineResult]:
        """Return a previous :class:`PipelineResult` if the artefacts
        it points at still exist on disk; otherwise None.

        Stale entries whose run_dir or manifest is missing are silently
        evicted from the index.
        """
        index = self.load_index()
        entry = index.get(cache_key)
        if not entry:
            return None
        run_id = entry.get("run_id")
        workdir = entry.get("workdir")
        if not run_id or not workdir:
            return None
        run_dir = Path(workdir)
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            index.pop(cache_key, None)
            self.save_index(index)
            return None
        try:
            return PipelineResult(
                run_id=run_id,
                workdir=str(run_dir),
                context_path=str(run_dir / "research_context.json"),
                plan_path=str(run_dir / "analysis_plan.json"),
                manifest_path=str(manifest),
                report_path=str(run_dir / "results_report.md"),
                manuscript_path=str(run_dir / "manuscript_scaffold_bound.md"),
                evidence_count=int(entry.get("evidence_count") or 0),
                findings_count=int(entry.get("findings_count") or 0),
            )
        except Exception:
            return None

    def record_hit(self, cache_key: str, result: PipelineResult) -> None:
        index = self.load_index()
        index[cache_key] = {
            "run_id": result.run_id,
            "workdir": result.workdir,
            "evidence_count": str(result.evidence_count),
            "findings_count": str(result.findings_count),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_index(index)


__all__ = [
    "PipelineCache",
    "hash_file",
    "iter_mock_clients",
    "llm_signature",
]
