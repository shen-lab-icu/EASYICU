"""Persisted display verdicts; verify in the failed job or an isolated worker.

This record is never a launch receipt. Its identity uses bounded metadata reads
and file stats, so page projections cannot open cohort rows or rehearse a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
from easyicu.runtime.platform_compat import open_nofollow
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
from typing import Any

SCHEMA = "easyicu.development_resume_offer/1"
RECORD_NAME = "development_resume_offer.json"
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_WORK: Queue = Queue(maxsize=128)
_PENDING: dict[tuple[Path, str], Event] = {}
_LOCK = Lock()
_WORKER: Thread | None = None
_VERIFIER_TIMEOUT_SECONDS = 900
_LOGGER = logging.getLogger(__name__)


def _small_bytes(path: Path) -> bytes:
    fd = open_nofollow(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_METADATA_BYTES:
            raise ValueError("offer metadata is not a bounded regular file")
        raw = handle.read(_MAX_METADATA_BYTES + 1)
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("offer metadata exceeds its size bound")
    return raw


def _file_identity(path: Path) -> tuple:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return ("missing",)
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)
    # Only bounded JSON metadata gets a content digest. In particular no
    # parquet, row identities, package members or cohort bytes are opened.
    if stat.S_ISREG(info.st_mode) and path.suffix == ".json" and info.st_size <= _MAX_METADATA_BYTES:
        return (*identity, hashlib.sha256(_small_bytes(path)).hexdigest())
    return identity


def _directory_identity(directory: Path) -> list:
    if directory.is_symlink() or not directory.is_dir():
        return [(_file_identity(directory))]
    return [(path.name, _file_identity(path)) for path in sorted(directory.iterdir())]


def offer_key(wrapper: Path, seed: Any, study_id: str, config_digest: str) -> str:
    """Fingerprint display dependencies, without interpreting their authority.

    Include the small run/evidence records (checkpoint chain, capsule, grouping,
    context, grouped/source authorities and provenance), and stat every cohort
    and sidecar. Directory inventories also bind absence, replacement and added
    trajectories. The launch owner alone decides whether these inputs are valid.
    """
    pipeline = wrapper / "pipeline"
    inputs: dict[str, Any] = {
        "schema": SCHEMA, "study": study_id, "configuration": config_digest,
        "seed": _file_identity(wrapper / ".runtime/web_review_recovery_seed.json"),
        "pipeline": _directory_identity(pipeline),
    }
    if not pipeline.is_symlink() and pipeline.is_dir():
        runs = sorted(p for p in pipeline.iterdir() if p.name.startswith("run_") and p.is_dir() and not p.is_symlink())
        if len(runs) == 1:
            run = runs[0]
            inputs["run"] = _directory_identity(run)
            inputs["evidence"] = _directory_identity(run / "evidence")
            # Match the prepared-input owner's bounded sibling search. These
            # are coordinates for invalidation only, never restoration authority.
            try:
                capsule = json.loads(_small_bytes(run / "run_input_capsule.json"))
                reference = capsule["scientific_identity"]["materialized_cohort_authority_ref"]
                name = reference["file"]
                if Path(name).name != name:
                    raise ValueError("invalid source authority name")
                wrappers = [wrapper, *sorted(p for p in wrapper.parent.iterdir() if p != wrapper and p.name.startswith("run_"))]
                candidates = []
                for candidate in wrappers[:200]:
                    directory = candidate / "pipeline_input"
                    if candidate.is_symlink() or directory.is_symlink():
                        candidates.append((candidate.name, "symlink"))
                        continue
                    candidates.append((candidate.name, _file_identity(directory / name)))
                    if (directory / name).is_file():
                        inputs["source_input"] = _directory_identity(directory)
                        break
                inputs["source_candidates"] = candidates
            except (OSError, ValueError, KeyError, TypeError):
                # The capsule's bytes are already bound above. Invalid inputs
                # still get a stable false verdict from the authoritative check.
                pass
    if seed.prepared_package_binding:
        from easyicu.webserver.research_resume_package import prepared_package_identity

        source = seed.study.get("data_source") or {}
        path = str(source.get("path") or "")
        try:
            inputs["package"] = prepared_package_identity(path, seed.prepared_package_binding)
        except (OSError, ValueError, KeyError, TypeError):
            inputs["invalid_package"] = _directory_identity(Path(path)) if path else "missing"
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_verdict(wrapper: Path, key: str) -> bool | None:
    try:
        path = wrapper / RECORD_NAME
        if path.is_symlink():
            return None
        record = json.loads(_small_bytes(path))
        if record.get("schema_version") == SCHEMA and record.get("key") == key and type(record.get("verdict")) is bool:
            return record["verdict"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def _write_verdict(wrapper: Path, key: str, verdict: bool) -> None:
    # Same private temporary-file / fsync / atomic replace convention as the
    # Web recovery seed. No source paths, scientific payloads or rows are saved.
    fd, name = tempfile.mkstemp(prefix=".resume-offer-", dir=wrapper)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": SCHEMA, "key": key, "verdict": verdict}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, wrapper / RECORD_NAME)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _verify(wrapper: Path, seed: Any, study_id: str, digest: str, key: str) -> bool:
    from easyicu.webserver import agent_review_recovery, research_launch_resume

    verdict = False
    try:
        if offer_key(wrapper, seed, study_id, digest) != key:
            return False
        research_launch_resume.verify_development_resume_offer(wrapper_dir=wrapper, study=seed.study)
        verdict = True
    except Exception as exc:
        _LOGGER.warning("development_resume_offer_unavailable reason=%s exception_type=%s",
                        getattr(exc, "code", "development_resume_offer_error"), type(exc).__name__)
    # A changed seed or member during verification must not publish an old key.
    current = agent_review_recovery.load_recovery_seed(wrapper)
    if current is None or offer_key(wrapper, current, study_id, digest) != key:
        return False
    _write_verdict(wrapper, key, verdict)
    return verdict


def _finish(token: tuple[Path, str]) -> None:
    with _LOCK:
        event = _PENDING.pop(token, None)
        if event is not None:
            event.set()


def _verify_background(wrapper: Path, seed: Any, study_id: str, digest: str, key: str) -> None:
    # A CPU-heavy rehearsal in another Python thread can starve metadata reads
    # through the GIL. One child at a time keeps that work off the web interpreter
    # too. It reloads the seed and checks the queued key before touching rows.
    # A hung verifier must not hold the single worker: later offers would
    # never be verified. run() kills the child when the bound expires.
    try:
        result = subprocess.run(
            [sys.executable, "-m", __name__, str(wrapper), study_id, digest, key],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=_VERIFIER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _LOGGER.warning("development_resume_offer_worker_timeout seconds=%s", _VERIFIER_TIMEOUT_SECONDS)
        return
    if result.returncode:
        _LOGGER.warning("development_resume_offer_worker_failed exit_code=%s", result.returncode)


def _worker() -> None:
    while True:
        wrapper, seed, study_id, digest, key = _WORK.get()
        try:
            _verify_background(wrapper, seed, study_id, digest, key)
        except Exception as exc:
            # A failed cache write must neither kill the worker nor the page.
            _LOGGER.warning("development_resume_offer_cache_failed exception_type=%s", type(exc).__name__)
        finally:
            _finish((wrapper, key))
            _WORK.task_done()


def cached_offer(*, wrapper: Path, seed: Any, study_id: str, digest: str, verify_now: bool = False) -> bool:
    """Read a verdict, queue a miss, or verify synchronously in the failed job.

    The short lock protects only scheduling; page reads never wait for a
    rehearsal. At capacity a page simply hides the offer and retries next read.
    """
    global _WORKER
    key = offer_key(wrapper, seed, study_id, digest)
    verdict = _read_verdict(wrapper, key)
    if verdict is not None:
        return verdict
    token = (wrapper, key)
    with _LOCK:
        existing = _PENDING.get(token)
        if existing is None:
            _PENDING[token] = Event()
            if not verify_now:
                try:
                    _WORK.put_nowait((wrapper, seed, study_id, digest, key))
                except Full:
                    _PENDING.pop(token)
                else:
                    if _WORKER is None or not _WORKER.is_alive():
                        _WORKER = Thread(target=_worker, name="continuation-offers", daemon=True)
                        _WORKER.start()
    if not verify_now:
        return False
    if existing is not None:
        existing.wait()
        return _read_verdict(wrapper, key) is True
    try:
        return _verify(wrapper, seed, study_id, digest, key)
    finally:
        _finish(token)


if __name__ == "__main__":
    from easyicu.webserver.agent_review_recovery import load_recovery_seed

    wrapper, study_id, digest, key = sys.argv[1:]
    seed = load_recovery_seed(Path(wrapper))
    if seed is not None:
        _verify(Path(wrapper), seed, study_id, digest, key)
