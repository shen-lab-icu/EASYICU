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
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import EvidenceRecord


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
        except Exception:
            return []

    def _load_aliases(self) -> Dict[str, str]:
        if not self.aliases_path.exists():
            return {}
        try:
            data = json.loads(self.aliases_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
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
            for alias in (aliases or []):
                self._add_alias(alias, evidence_id)
            self._add_alias(target.stem.split("__", 1)[-1], evidence_id)
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

        for alias in (aliases or []):
            self._add_alias(alias, evidence_id)
        basename = target.name.split("__", 1)[-1]
        self._add_alias(Path(basename).stem, evidence_id)
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
        eid = evidence_id or self._make_id(_id_prefix(kind, Path(filename).stem), digest)
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
        return None

    def resolvable_names(self) -> List[str]:
        """Every name the binder will accept (evidence_ids + aliases)."""
        with self._lock:
            return sorted(set(r.evidence_id for r in self._records) | set(self._aliases))

    # ------------------------------------------------------------------
    # Manuscript binding
    # ------------------------------------------------------------------

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
            eid = scaffold[j + len("{evidence:"):k]
            rec = self.get(eid)
            if rec is None:
                out.append(f"[evidence missing: {eid}]")
            elif verbose:
                suffix = _binding_caveat(rec)
                out.append(
                    f"[{rec.description} | {rec.relative_path} | sha256={rec.sha256[:8]}]{suffix}"
                )
            else:
                # Compact in-line citation: link uses the requested
                # placeholder text (so the writer's own naming survives)
                # and the URL points at the registered file. Trailing
                # 8-char sha256 sits in the link title for hover.
                out.append(
                    f"[{eid}]({rec.relative_path} \"sha256={rec.sha256[:8]}\")"
                    f"{_binding_caveat(rec)}"
                )
            i = k + 1
        return "".join(out)


def _id_prefix(kind: str, stem: str) -> str:
    safe = "".join(c for c in stem if c.isalnum() or c in "_-").strip("_")[:32]
    return f"{kind}_{safe}" if safe else kind


def _binding_caveat(record: EvidenceRecord) -> str:
    severity = record.finding_severity
    if severity in {"warning", "error"}:
        return f" ({severity}: see manifest)"
    return ""


__all__ = ["EvidenceStore", "sha256_of_file", "sha256_of_bytes"]
