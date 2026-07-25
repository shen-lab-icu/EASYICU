"""Cross-run experience bank (Commit 3, Phase-1 widening, May 2026).

Inspired by HealthFlow's "self-evolving" multi-agent framework (Zhu et
al., 2025) which writes back four experience buckets — safeguards,
workflows, dataset anchors, code snippets — after every completed
task and retrieves them on subsequent runs. We start narrower with
two opinionated buckets that match incidents we have actually seen
in EasyICU pilots:

* ``concept_usage_hint`` — a positive lesson about the EasyICU
  concept layer. Example: "for a sepsis cohort, the cohort loader
  should be ``easyicu.api.load_sepsis3`` rather than a manual ICD-9
  regex; the dictionary callbacks already encode the Singer-2016
  definition." Emitted by the reflector after a successful run
  whose cohort step used an EasyICU concept loader.
* ``failure_counter_example`` — a negative lesson tied to a specific
  failure mode that recurred at least once. Example: "step
  ``03_complete_case_robustness`` failed with 502 after the
  reproducibility_envelope was registered; if you see ``Error code:
  502`` on a step that also writes the envelope, the resume path
  expects ``on_sha_change='new_id'`` (see ``authority/evidence_store.py``)." Emitted by
  the reflector after the supersession partition surfaces a
  previously-failing step that succeeded on retry.

Both are stored as a JSONL file with deterministic field order so
``git diff`` over the bank is human-readable.

Design choices that diverge from HealthFlow on purpose:

1. **No embeddings.** Retrieval is by lexical Jaccard similarity over
   the research question's normalised tokens. This keeps the bank
   functional without pulling in an embedding provider; it also keeps
   the rule deterministic so two runs with the same question
   retrieve the same experiences.
2. **Deterministic reflector.** The reflector is a rule-based
   function, not an LLM call. The set of triggers is hard-coded; the
   reflector either fires (and emits a record) or stays silent.
   Empirically the LLM-based "ask the model to summarise lessons
   learned" path in HealthFlow has a strong tendency to recommend
   tangential workflow tweaks; the deterministic path is narrower
   but never invents an experience that wasn't grounded in concrete
   pipeline state.
3. **Opt-in.** ``PipelineConfig.enable_experience_bank`` is False by
   default. Existing pilot runs never read or write the bank unless
   the flag is set. The npj DM submission run does not depend on
   experience-bank behaviour.
4. **Bounded retrieval.** Top-k defaults to 5 and a minimum
   similarity threshold filters out unrelated experiences. The
   retrieved hints are surfaced as a separate ``experience_hints``
   block in the planner prompt, never silently merged into the
   ICU-rules background.

The bank is a single JSONL file. Concurrent runs writing to the same
file are serialised by an exclusive ``flock`` held across the whole
read-modify-write, and the rewrite itself lands through a temp file
plus ``os.replace``, so a reader never observes a half-written bank.
The lock is per file, so different workdirs do not block each other.
On a platform without ``fcntl`` only the in-process lock applies; the
bank then carries the usual last-writer-wins risk across processes and
says so in a warning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover - POSIX in production and CI
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class ExperienceBankCorruptError(RuntimeError):
    """Raised when a rewrite would drop lines the loader could not parse."""


ExperienceKind = Literal["concept_usage_hint", "failure_counter_example"]


@dataclass
class ExperienceRecord:
    """One unit of cross-run lesson.

    Fields are flat (no nesting) so JSONL diffs and grep stay readable.

    * ``kind`` — bucket name; one of the literal values above.
    * ``research_question`` — the question text for the run that
      *produced* this record. The retrieval rule scores incoming
      questions against this field.
    * ``database`` — the ICU database short name
      (``mimic_iv`` / ``eicu`` / ...). When a future run targets the
      same database the record is favoured.
    * ``cohort_name`` — best-effort cohort label, e.g.
      ``sepsis3_aware``. Empty string if the producing run did not
      anchor a named cohort.
    * ``summary`` — one-sentence human-readable lesson. Surfaced
      verbatim to the planner; therefore must be self-contained.
    * ``detail`` — optional JSON-serialisable detail block. The
      reflector stores trigger context here so audits can trace why
      the record was emitted.
    * ``produced_at`` — ISO-8601 UTC timestamp.
    * ``producer_run_id`` — best-effort run identifier (workdir
      basename) of the producing run; for audit only.
    """

    kind: ExperienceKind
    research_question: str
    database: str
    cohort_name: str
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    produced_at: str = ""
    producer_run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep the order stable so JSONL diffs are readable.
        ordered_keys = [
            "kind",
            "research_question",
            "database",
            "cohort_name",
            "summary",
            "detail",
            "produced_at",
            "producer_run_id",
        ]
        return {k: d[k] for k in ordered_keys}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperienceRecord":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in payload.items() if k in known}
        return cls(**clean)


# ---------------------------------------------------------------------------
# Lexical similarity (deterministic, embedding-free)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "we",
        "this",
        "these",
        "those",
        "do",
        "does",
        "what",
        "which",
        "who",
        "how",
        "why",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lowercase + word-token + stopword-stripped token set.

    Used for both indexing and retrieval — must stay symmetric.
    """
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    """Token Jaccard over normalised text. Returns 0..1."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# ExperienceBank
# ---------------------------------------------------------------------------


class ExperienceBank:
    """JSONL-backed deterministic experience bank.

    A bank instance is cheap to construct; ``load()`` reads the file
    once into memory and ``save()`` rewrites it. For typical pilot
    cohorts of <100 records the in-memory cost is negligible. Each
    ``add()`` saves immediately — there is no batched write.
    """

    DEFAULT_FILENAME = "experience_bank.jsonl"

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Optional[Path] = Path(path) if path else None
        self._records: List[ExperienceRecord] = []
        self._lock = threading.Lock()
        #: Lines the last load could not parse. A mutation refuses to rewrite
        #: the file while this is non-zero, because the rewrite would drop
        #: them permanently.
        self.corrupt_lines: int = 0
        if self.path is not None and self.path.exists():
            self._load()

    # --- persistence -------------------------------------------------

    @contextmanager
    def _exclusive_file_lock(self):
        """Serialise the read-modify-write across processes, not just threads.

        The bank is explicitly documented as shareable between concurrent
        runs, and every mutation rewrites the whole file, so a thread lock
        alone loses records whenever two runs share a path.
        """

        if self.path is None or fcntl is None:
            if fcntl is None:
                logger.warning(
                    "experience-bank: fcntl is unavailable on this platform; "
                    "concurrent processes sharing %s can lose records",
                    self.path,
                )
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _load(self) -> None:
        assert self.path is not None
        records: List[ExperienceRecord] = []
        corrupt = 0
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    corrupt += 1
                    logger.error(
                        "experience-bank %s: skipping non-JSON line: %r",
                        self.path,
                        line[:80],
                    )
                    continue
                records.append(ExperienceRecord.from_dict(payload))
        except OSError as exc:
            # Returning here left ``self._records`` at its previous value while
            # the file on disk was never read. The next ``add()`` would then
            # rewrite the whole file from that stale in-memory state, silently
            # discarding whatever the unreadable file actually contained.
            raise ExperienceBankCorruptError(
                f"experience bank {self.path} could not be read: {exc}"
            ) from exc
        self._records = records
        self.corrupt_lines = corrupt

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=False)
            for r in self._records
        ]
        payload = "\n".join(lines) + ("\n" if lines else "")
        # Write-then-replace: a reader (or a crash) never sees a half-file.
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    # --- mutation ----------------------------------------------------

    def add(self, record: ExperienceRecord) -> None:
        """Add a record, deduplicating against ``(kind, summary)``.

        Re-adding an identical lesson refreshes ``produced_at`` /
        ``producer_run_id`` but does not duplicate the row.
        """
        with self._lock, self._exclusive_file_lock():
            # Re-read under the lock: another process may have appended since
            # this instance loaded, and a rewrite from the stale in-memory
            # list would silently drop its records.
            if self.path is not None and self.path.exists():
                self._load()
            if self.corrupt_lines:
                raise ExperienceBankCorruptError(
                    f"experience bank {self.path} has {self.corrupt_lines} "
                    "unparseable line(s); refusing to rewrite the file because "
                    "that would drop them permanently"
                )
            for existing in self._records:
                if existing.kind == record.kind and existing.summary == record.summary:
                    existing.produced_at = record.produced_at or existing.produced_at
                    existing.producer_run_id = (
                        record.producer_run_id or existing.producer_run_id
                    )
                    # Detail merge: incoming wins on overlap.
                    existing.detail = {**existing.detail, **record.detail}
                    self._save()
                    return
            self._records.append(record)
            self._save()

    def extend(self, records: Sequence[ExperienceRecord]) -> None:
        for r in records:
            self.add(r)

    def records(self) -> List[ExperienceRecord]:
        with self._lock:
            return list(self._records)

    # --- retrieval ---------------------------------------------------

    def retrieve(
        self,
        *,
        research_question: str,
        database: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.2,
    ) -> List[Tuple[ExperienceRecord, float]]:
        """Return top-k experiences ranked by Jaccard similarity.

        ``min_similarity`` filters out unrelated records before
        ranking. Matching the same database adds a small fixed boost
        (0.1) so a cross-database match never outranks a same-database
        match with similar lexical overlap.

        Raises :class:`ExperienceBankCorruptError` when the file held lines
        that could not be parsed. Serving the parseable remainder looks
        harmless — nothing crashes — but this output *steers the Planner*, and
        a silently truncated bank is an unprovable planning input: neither the
        run nor a reviewer can tell whether the record that would have changed
        the plan was one of the unreadable ones. The caller decides whether to
        continue without a bank; it must not continue with a partial one.
        """
        if self.corrupt_lines:
            raise ExperienceBankCorruptError(
                f"experience bank {self.path} has {self.corrupt_lines} "
                "unparseable line(s); refusing to serve a partial bank as a "
                "planning input"
            )
        scored: List[Tuple[ExperienceRecord, float]] = []
        with self._lock:
            corpus = list(self._records)
        for r in corpus:
            score = jaccard_similarity(research_question, r.research_question)
            if database and r.database and r.database == database:
                score += 0.1
            if score >= min_similarity:
                scored.append((r, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(0, int(top_k))]


# ---------------------------------------------------------------------------
# Deterministic reflector
# ---------------------------------------------------------------------------


def mine_experience_from_run(
    *,
    research_question: str,
    database: str,
    cohort_name: str,
    gates: Dict[str, Any],
    findings: Sequence[Dict[str, Any]],
    superseded_errors: Sequence[Dict[str, Any]] = (),
    plan_step_ids: Sequence[str] = (),
    producer_run_id: str = "",
) -> List[ExperienceRecord]:
    """Mine zero or more ExperienceRecords from a completed run.

    Triggers:

    * **concept_usage_hint**: the run passed every readiness gate AND
      its plan contained a step whose id mentions a concept-layer
      loader (``sepsis``, ``sofa``, ``kdigo``, ``susp_inf``,
      ``cohort_summary``). Records that the EasyICU loader-driven
      approach worked end-to-end for this question. Skipped if any
      validator finding has severity ``error``.
    * **failure_counter_example**: at least one step that was
      partitioned into ``superseded_errors`` (i.e. it failed and
      later succeeded). Records the exact failure-then-retry pattern
      so a future run hitting the same step pattern can plan around
      it.

    The reflector NEVER calls an LLM. The output is fully determined
    by its inputs, which makes the bank reproducible — re-running
    over the same run directory always yields the same records.
    """
    out: List[ExperienceRecord] = []
    now = datetime.now(timezone.utc).isoformat()

    # 1) concept_usage_hint
    error_findings = [
        f for f in findings if str(f.get("severity", "")).lower() == "error"
    ]
    all_gates_pass = (
        bool(gates.get("execution_complete"))
        and bool(gates.get("evidence_complete"))
        and bool(gates.get("numeric_verified"))
        and bool(gates.get("analysis_validated"))
    )
    concept_step_ids = [
        sid
        for sid in plan_step_ids
        if any(
            tok in sid.lower()
            for tok in ("sepsis", "sofa", "kdigo", "susp_inf", "cohort")
        )
    ]
    if all_gates_pass and not error_findings and concept_step_ids and cohort_name:
        out.append(
            ExperienceRecord(
                kind="concept_usage_hint",
                research_question=research_question,
                database=database,
                cohort_name=cohort_name,
                summary=(
                    f"For cohort '{cohort_name}' on {database}, the "
                    f"EasyICU concept-loader path "
                    f"({', '.join(sorted(set(concept_step_ids))[:3])}) "
                    f"closes every readiness gate end-to-end. Prefer it "
                    f"over manual ICD/regex cohort definitions."
                ),
                detail={
                    "concept_step_ids": sorted(set(concept_step_ids)),
                    "gates": {
                        k: bool(gates.get(k))
                        for k in (
                            "execution_complete",
                            "evidence_complete",
                            "numeric_verified",
                            "analysis_validated",
                        )
                    },
                },
                produced_at=now,
                producer_run_id=producer_run_id,
            )
        )

    # 2) failure_counter_example
    superseded_steps = {
        str(err.get("step_id") or err.get("detail", {}).get("step_id") or "")
        for err in superseded_errors
    }
    superseded_steps.discard("")
    for step_id in sorted(superseded_steps):
        # Pull one representative message for the audit detail.
        matching = next(
            (
                err
                for err in superseded_errors
                if (err.get("step_id") or err.get("detail", {}).get("step_id"))
                == step_id
            ),
            None,
        )
        msg = str((matching or {}).get("message") or "")[:280]
        out.append(
            ExperienceRecord(
                kind="failure_counter_example",
                research_question=research_question,
                database=database,
                cohort_name=cohort_name,
                summary=(
                    f"Step '{step_id}' on {database} ('{cohort_name}') failed "
                    f"then succeeded on retry; the original failure was "
                    f"superseded by the gate-readiness rules. If a future "
                    f"plan reuses this step shape, expect at least one "
                    f"transient retry before the gate closes."
                ),
                detail={
                    "step_id": step_id,
                    "original_failure_message": msg,
                },
                produced_at=now,
                producer_run_id=producer_run_id,
            )
        )

    return out


__all__ = [
    "ExperienceKind",
    "ExperienceRecord",
    "ExperienceBank",
    "jaccard_similarity",
    "mine_experience_from_run",
]
