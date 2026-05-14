"""Evidence store with content hashing and provenance index.

Every analytical artefact — table, figure, statistic, log, code —
is registered here with a SHA-256 hash of its bytes and a record of
which step produced it. The manuscript writer is only allowed to
cite ``evidence_id`` strings that resolve to a record in this store;
unbound citations are stripped during post-processing.

Why this exists:

* It is the single mechanism that makes the agent's claims
  *auditable*. A reviewer can recompute the hash of a figure and
  confirm it matches the manifest.
* It is what lets the manuscript scaffold be "honestly thin": the
  sentences are templates with placeholders, and the placeholders
  are guaranteed to point at real artefacts.

Alias resolution
----------------
Registration assigns a hash-suffixed evidence_id (e.g.
``table_table_one_8f3c19a4``) so that two artefacts with the same
filename stay distinct. The writer agent, however, prefers stable
semantic names like ``table_one`` or ``outcome_rate``. The store
therefore maintains an *alias* table — first-write-wins — so that
``{evidence:table_one}`` placeholders in the manuscript resolve to
the first registered evidence with that filename stem (or to an
explicit alias passed by the pipeline). Aliases are persisted
alongside the index so binding is reproducible.

The on-disk layout under ``<workdir>/evidence/`` is::

    evidence/
        <evidence_id>__<basename>.<ext>     # the artefact
        evidence_index.json                 # serialised list of EvidenceRecord
        evidence_aliases.json               # alias → evidence_id map
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import EvidenceRecord

logger = logging.getLogger(__name__)


def _quarantine_corrupt_index(path: Path, exc: Exception, kind: str) -> None:
    """Rename a corrupted index file aside and log a warning.

    Silently returning empty would erase the audit trail the writer agent
    relies on; instead we keep the broken bytes for forensic inspection
    and emit a warning so the operator knows evidence was lost.
    """
    if not path.exists():
        logger.warning("evidence %s missing; starting fresh: %s", kind, exc)
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(path.suffix + f".broken-{timestamp}")
    try:
        path.replace(backup)
    except OSError as rename_err:
        logger.warning(
            "evidence %s at %s is corrupt (%s); also failed to back up: %s",
            kind, path, exc, rename_err,
        )
        return
    logger.warning(
        "evidence %s at %s is corrupt (%s); moved to %s and starting fresh",
        kind, path, exc, backup,
    )


def sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceStore:
    """A directory-backed store of hashed artefacts.

    Each call to :meth:`register_file` copies the file into
    ``evidence/`` under a deterministic name (``<id>__<basename>``)
    and writes an :class:`EvidenceRecord` to the in-memory index. The
    store is persisted to ``evidence_index.json`` on every call so
    crashes don't lose evidence.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.dir = self.root / "evidence"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "evidence_index.json"
        self.aliases_path = self.dir / "evidence_aliases.json"
        self._records: List[EvidenceRecord] = self._load_records()
        self._aliases: Dict[str, str] = self._load_aliases()
        # T3.3 — concurrent step execution: every register / get / save
        # path runs under this lock so two worker threads can safely
        # call ``register_file`` simultaneously. Reentrant so that
        # methods that internally call ``register_file`` (e.g.
        # ``register_text`` → ``register_file``) don't self-deadlock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_records(self) -> List[EvidenceRecord]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [EvidenceRecord.model_validate(r) for r in data]
        except Exception as exc:
            _quarantine_corrupt_index(self.index_path, exc, kind="index")
            return []

    def _load_aliases(self) -> Dict[str, str]:
        if not self.aliases_path.exists():
            return {}
        try:
            data = json.loads(self.aliases_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _quarantine_corrupt_index(self.aliases_path, exc, kind="aliases")
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        logger.warning(
            "evidence aliases at %s is not a JSON object (got %s); ignoring",
            self.aliases_path, type(data).__name__,
        )
        _quarantine_corrupt_index(
            self.aliases_path,
            ValueError(f"unexpected type {type(data).__name__}"),
            kind="aliases",
        )
        return {}

    def _save(self) -> None:
        self.index_path.write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in self._records],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        self.aliases_path.write_text(
            json.dumps(self._aliases, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _add_alias(self, alias: str, evidence_id: str) -> None:
        """First-write-wins. We do not overwrite an existing alias because
        the manuscript scaffold is generated against the first artefact of
        each kind; later registrations with the same filename should not
        retroactively change what an earlier sentence pointed to."""
        if not alias:
            return
        if alias in self._aliases:
            return
        self._aliases[alias] = evidence_id

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _record_by_id(self, evidence_id: str) -> Optional[EvidenceRecord]:
        for record in self._records:
            if record.evidence_id == evidence_id:
                return record
        return None

    def _make_id(self, prefix: str, digest: str) -> str:
        return f"{prefix}_{digest[:8]}"

    def _register_target(
        self,
        *,
        evidence_id: str,
        kind: str,
        description: str,
        target: Path,
        sha256: str,
        produced_by_step: Optional[str],
        inputs: Optional[Sequence[str]],
        script_evidence_id: Optional[str],
        aliases: Optional[Sequence[str]],
        producer: Optional[str],
        generation_mode: Optional[str],
        prompt_pack_version: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> EvidenceRecord:
        existing = self._record_by_id(evidence_id)
        if existing is not None:
            if existing.sha256 != sha256:
                raise ValueError(
                    f"Evidence id collision for {evidence_id}: "
                    f"existing sha256={existing.sha256[:8]} new sha256={sha256[:8]}"
                )
            for alias in aliases or []:
                self._add_alias(alias, evidence_id)
            self._add_alias(_target_basename_stem(target, evidence_id), evidence_id)
            self._save()
            return existing

        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,  # type: ignore[arg-type]
            description=description,
            relative_path=str(target.relative_to(self.root)),
            sha256=sha256,
            produced_by_step=produced_by_step,
            inputs=list(inputs or []),
            script_evidence_id=script_evidence_id,
            producer=producer,
            generation_mode=generation_mode,
            prompt_pack_version=prompt_pack_version,
            metadata=dict(metadata or {}),
            created_at=datetime.now(timezone.utc),
        )
        self._records.append(record)

        for alias in aliases or []:
            self._add_alias(alias, evidence_id)
        self._add_alias(_target_basename_stem(target, evidence_id), evidence_id)
        self._add_alias(evidence_id, evidence_id)

        self._save()
        return record

    def register_file(
        self,
        *,
        kind: str,
        description: str,
        source_path: Path,
        produced_by_step: Optional[str] = None,
        inputs: Optional[Sequence[str]] = None,
        script_evidence_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot register missing file: {source_path}")
        # T3.3 — guard the entire critical section (id allocation, file
        # copy into evidence/, record append, alias update, persist).
        # The lock is reentrant so register_text → register_file
        # composition stays deadlock-free.
        with self._lock:
            source_digest = sha256_of_file(source_path)
            eid = evidence_id or self._make_id(
                _id_prefix(kind, source_path.stem), source_digest
            )
            target = self.dir / f"{eid}__{source_path.name}"
            if target.resolve() != source_path.resolve():
                shutil.copy2(source_path, target)
            digest = sha256_of_file(target)
            return self._register_target(
                evidence_id=eid,
                kind=kind,
                description=description,
                target=target,
                sha256=digest,
                produced_by_step=produced_by_step,
                inputs=inputs,
                script_evidence_id=script_evidence_id,
                aliases=aliases,
                producer=producer,
                generation_mode=generation_mode,
                prompt_pack_version=prompt_pack_version,
                metadata=metadata,
            )

    def register_text(
        self,
        *,
        kind: str,
        description: str,
        text: str,
        filename: str,
        produced_by_step: Optional[str] = None,
        inputs: Optional[Sequence[str]] = None,
        script_evidence_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        payload = text.encode("utf-8")
        digest = sha256_of_bytes(payload)
        eid = evidence_id or self._make_id(
            _id_prefix(kind, Path(filename).stem), digest
        )
        target = self.dir / f"{eid}__{filename}"
        target.write_bytes(payload)
        with self._lock:
            return self._register_target(
                evidence_id=eid,
                kind=kind,
                description=description,
                target=target,
                sha256=digest,
                produced_by_step=produced_by_step,
                inputs=inputs,
                script_evidence_id=script_evidence_id,
                aliases=aliases,
                producer=producer,
                generation_mode=generation_mode,
                prompt_pack_version=prompt_pack_version,
                metadata=metadata,
            )

    def register_json(
        self,
        *,
        kind: str,
        description: str,
        payload: Any,
        filename: str,
        produced_by_step: Optional[str] = None,
        inputs: Optional[Sequence[str]] = None,
        evidence_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return self.register_text(
            kind=kind,
            description=description,
            text=text,
            filename=filename,
            produced_by_step=produced_by_step,
            inputs=inputs,
            evidence_id=evidence_id,
            aliases=aliases,
            producer=producer,
            generation_mode=generation_mode,
            prompt_pack_version=prompt_pack_version,
            metadata=metadata,
        )

    def update_record(
        self,
        evidence_id: str,
        *,
        finding_severity: Optional[str] = None,
        finding_messages: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
    ) -> Optional[EvidenceRecord]:
        with self._lock:
            record = self.get(evidence_id)
            if record is None:
                return None
            if finding_severity is not None:
                record.finding_severity = finding_severity
            if finding_messages is not None:
                record.finding_messages = list(finding_messages)
            if metadata:
                merged = dict(record.metadata)
                merged.update(metadata)
                record.metadata = merged
            if producer is not None:
                record.producer = producer
            if generation_mode is not None:
                record.generation_mode = generation_mode
            if prompt_pack_version is not None:
                record.prompt_pack_version = prompt_pack_version
            self._save()
            return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def records(self) -> List[EvidenceRecord]:
        with self._lock:
            return list(self._records)

    def ids(self) -> List[str]:
        with self._lock:
            return [r.evidence_id for r in self._records]

    def aliases(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._aliases)

    def get(self, evidence_id_or_alias: str) -> Optional[EvidenceRecord]:
        with self._lock:
            # Direct id match first.
            for r in self._records:
                if r.evidence_id == evidence_id_or_alias:
                    return r
            # Then alias lookup. Resolve under the same lock so a
            # concurrent register_file can never invalidate the alias
            # → record path mid-lookup.
            eid = self._aliases.get(evidence_id_or_alias)
            if eid is not None:
                for r in self._records:
                    if r.evidence_id == eid:
                        return r
            # Hosted writers sometimes cite the stable basename of a
            # hash-suffixed artefact, e.g. ``figure_mortality`` for
            # ``figure_mortality_ab12cd34``. Accept this only when the
            # prefix is unique so we do not silently bind an ambiguous claim.
            prefix = f"{evidence_id_or_alias}_"
            candidate_ids = {
                r.evidence_id for r in self._records if r.evidence_id.startswith(prefix)
            }
            candidate_ids.update(
                eid for alias, eid in self._aliases.items() if alias.startswith(prefix)
            )
            if len(candidate_ids) == 1:
                only = next(iter(candidate_ids))
                for r in self._records:
                    if r.evidence_id == only:
                        return r
        return None

    def resolvable_names(self) -> List[str]:
        """Every name the binder will accept (evidence_ids + aliases)."""
        with self._lock:
            return sorted(
                set(r.evidence_id for r in self._records) | set(self._aliases)
            )

    # ------------------------------------------------------------------
    # Manuscript binding
    # ------------------------------------------------------------------

    def enforce_evidence_bound_scaffold(self, scaffold: str) -> tuple[str, List[str]]:
        """Drop result-like sentences that lack an explicit evidence placeholder.

        The writer is allowed to draft prose freely, but anything that looks like
        a numerical result or analytical claim must cite ``{evidence:<id>}``
        before it can enter the final manuscript. We keep headings, list items,
        and non-result narrative intact, and return the filtered scaffold plus a
        list of sentences that were removed.
        """
        removed: List[str] = []
        filtered_lines: List[str] = []
        for raw_line in scaffold.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "```", "-", "*", ">")):
                filtered_lines.append(line)
                continue
            sentences = _split_sentences(line)
            if len(sentences) == 1 and not _looks_result_like_sentence(sentences[0]):
                filtered_lines.append(line)
                continue
            kept: List[str] = []
            for sentence in sentences:
                if (
                    _looks_result_like_sentence(sentence)
                    and "{evidence:" not in sentence
                ):
                    removed.append(sentence.strip())
                    continue
                kept.append(sentence.strip())
            filtered_lines.append(" ".join(part for part in kept if part).strip())
        return "\n".join(filtered_lines).strip() + "\n", removed

    def bind_manuscript(self, scaffold: str, *, verbose: bool = False) -> str:
        """Replace ``{evidence:<id>}`` placeholders with provenance links.

        Default mode emits a compact markdown link
        ``[<id>](relative_path)`` that reads naturally inside a sentence
        ("see Table 1 [table_one](evidence/...)"). Set ``verbose=True``
        to keep the older ``[description | path | sha256=…]`` form for
        machine-readable tracking.

        Unbound placeholders are replaced with ``[evidence missing: id]``
        so a reviewer can immediately see what the writer expected to
        cite.
        """
        out: List[str] = []
        i = 0
        n = len(scaffold)
        while i < n:
            j = scaffold.find("{evidence:", i)
            if j < 0:
                out.append(scaffold[i:])
                break
            out.append(scaffold[i:j])
            k = scaffold.find("}", j)
            if k < 0:
                out.append(scaffold[j:])
                break
            eid = scaffold[j + len("{evidence:") : k]
            requested_ids = [
                _normalize_requested_evidence_id(item)
                for item in eid.split(",")
                if _normalize_requested_evidence_id(item)
            ]
            if not requested_ids:
                requested_ids = [eid]
            bound_parts: List[str] = []
            missing: List[str] = []
            for requested_id in requested_ids:
                rec = self.get(requested_id)
                if rec is None:
                    missing.append(requested_id)
                    continue
                if verbose:
                    suffix = _binding_caveat(rec, verbose=verbose)
                    bound_parts.append(
                        f"[{rec.description} | {rec.relative_path} | sha256={rec.sha256[:8]}]{suffix}"
                    )
                else:
                    bound_parts.append(
                        f'[{requested_id}]({rec.relative_path} "sha256={rec.sha256[:8]}")'
                        f"{_binding_caveat(rec, verbose=verbose)}"
                    )
            if missing:
                bound_parts.extend(f"[evidence missing: {item}]" for item in missing)
            if bound_parts:
                out.append("; ".join(bound_parts))
            elif verbose:
                out.append(f"[evidence missing: {eid}]")
            i = k + 1
        return "".join(out)


def _normalize_requested_evidence_id(value: str) -> str:
    """Normalize a manuscript placeholder item before alias lookup.

    Some writer models emit comma placeholders as
    ``{evidence:a, evidence:b}`` rather than ``{evidence:a, b}``.
    Treat the repeated prefix as harmless syntax noise.
    """

    item = (value or "").strip()
    if item.lower().startswith("evidence:"):
        item = item.split(":", 1)[1].strip()
    return item


def _target_basename_stem(target: Path, evidence_id: str) -> str:
    """Return the original filename stem from ``<evidence_id>__<filename>``.

    Evidence ids themselves may contain a doubled underscore when the id prefix
    ends with ``_``. Splitting on the first ``__`` then corrupts the alias.
    """

    prefix = f"{evidence_id}__"
    name = target.name
    if name.startswith(prefix):
        return Path(name[len(prefix) :]).stem
    return Path(name.split("__", 1)[-1]).stem


def _id_prefix(kind: str, stem: str) -> str:
    safe = "".join(c for c in stem if c.isalnum() or c in "_-").strip("_")[:32]
    return f"{kind}_{safe}" if safe else kind


def _binding_caveat(record: EvidenceRecord, *, verbose: bool = False) -> str:
    severity = record.finding_severity
    if severity in {"warning", "error"}:
        if verbose:
            return f" ({severity}: see manifest)"
        return f"<!-- {severity}: see manifest -->"
    return ""


_RESULT_TOKEN_RE = re.compile(
    r"(\bOR\b|\bHR\b|\bRR\b|\bAUC\b|\bAUROC\b|\bBrier\b|\bcalibration\b|"
    r"\bdiscrimination\b|\bperformance\b|\brobust(?:ness)?\b|"
    r"\boverfitting\b|\bmiscalibration\b|\bmissingness\b|\bgeneralisa(?:bility|ble)\b|"
    r"\bgeneraliza(?:bility|ble)\b|"
    r"\bmedian\b|\bmean\b|\bincidence\b|\bmortality\b|\bhazard\b|"
    r"\bconfidence interval\b|\bCI\b|\bp\s*[<=>]|%|\d)",
    re.I,
)


def _split_sentences(text: str) -> List[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+", text.strip())
        if part.strip()
    ]
    return parts or ([text.strip()] if text.strip() else [])


def _looks_result_like_sentence(sentence: str) -> bool:
    if "{evidence:" in sentence:
        return False
    return bool(_RESULT_TOKEN_RE.search(sentence))


__all__ = ["EvidenceStore", "sha256_of_file", "sha256_of_bytes"]
