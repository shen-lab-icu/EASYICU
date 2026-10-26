"""Durable delivery receipts for host-derived attention, with per-client ACKs.

The study, job and conversation owners remain authoritative. This journal
contains only immutable transition coordinates; it cannot advance a study or
approve a gate. SQLite serializes sequence assignment across concurrent runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any, Iterator, Mapping

from easyicu.webserver.attention_projection import (
    CLIENT_IDENTIFIER, MAX_SEQUENCE, AttentionTransition, job_transition, source_gate,
)
from easyicu.webserver.hosted import require_instance_path

MAX_PAGE = 100
SCHEMA_VERSION = "easyicu.attention/1"
_ITEM_COLUMNS = "sequence, project_id, conversation_id, run_id, kind, reason_code, timestamp"


class AttentionUnavailable(RuntimeError):
    """The host cannot persist or read attention delivery receipts."""


def _client(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(CLIENT_IDENTIFIER, value) is None:
        raise ValueError("attention_client_invalid")


class AttentionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # The existing hosted gateway selects a private per-user instance.
        # Neither HTTP headers nor request project IDs select a database.
        require_instance_path(self.path)
        connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                if self.path.is_symlink() or not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise AttentionUnavailable("attention_store_unsafe_path")
                if os.name != "nt":
                    os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            connection = sqlite3.connect(self.path, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version not in {0, 1}:
                    raise AttentionUnavailable("attention_store_version_invalid")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS attention ("
                    "sequence INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, "
                    "conversation_id TEXT NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL, "
                    "reason_code TEXT NOT NULL, timestamp TEXT NOT NULL, source_key TEXT NOT NULL, "
                    "UNIQUE(project_id, conversation_id, source_key))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS acknowledgements ("
                    "client_id TEXT NOT NULL, sequence INTEGER NOT NULL REFERENCES attention(sequence), "
                    "PRIMARY KEY(client_id, sequence))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS source_gate_cursors ("
                    "project_id TEXT NOT NULL, conversation_id TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                    "generation INTEGER NOT NULL, PRIMARY KEY(project_id, conversation_id))"
                )
                connection.execute("PRAGMA user_version = 1")
                yield connection
        except (OSError, sqlite3.Error) as exc:
            raise AttentionUnavailable("attention_store_unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def record(self, transitions: list[AttentionTransition]) -> None:
        if not transitions:
            return
        with self._connect() as connection:
            for transition in transitions:
                self._insert(connection, transition)

    @staticmethod
    def _insert(connection: sqlite3.Connection, transition: AttentionTransition) -> None:
        connection.execute(
            "INSERT INTO attention (project_id, conversation_id, run_id, kind, reason_code, timestamp, source_key) "
            "SELECT :project_id, :conversation_id, :run_id, :kind, :reason_code, :timestamp, :source_key "
            "WHERE NOT EXISTS (SELECT 1 FROM attention WHERE project_id=:project_id "
            "AND conversation_id=:conversation_id AND source_key=:source_key)", asdict(transition),
        )

    def observe_sessions(self, records: list[Mapping[str, Any]]) -> None:
        edges = [(record, edge) for record in records if (edge := source_gate(record)) is not None]
        if not edges:
            return
        with self._connect() as connection:
            for record, (fingerprint, transition) in edges:
                scope = record["project_id"], record["session_id"]
                previous = connection.execute(
                    "SELECT fingerprint, generation FROM source_gate_cursors WHERE project_id=? AND conversation_id=?", scope,
                ).fetchone()
                if previous is not None and previous["fingerprint"] == fingerprint:
                    continue
                generation = previous["generation"] + 1 if previous else 1
                connection.execute(
                    "INSERT OR REPLACE INTO source_gate_cursors VALUES (?, ?, ?, ?)", (*scope, fingerprint, generation),
                )
                if transition is not None:
                    self._insert(connection, replace(transition, source_key=f"source_gate:{generation}"))

    def publish_sessions(self, records: list[Mapping[str, Any]]) -> None:
        """A delivery failure must not roll back an authoritative session write."""
        try:
            self.observe_sessions(records)
        except AttentionUnavailable:
            logging.getLogger(__name__).error("attention_store_unavailable")

    def observe_replay(self, payload: Mapping[str, Any]) -> None:
        self.record([
            transition for job in payload.get("child_jobs", [])
            if isinstance(job, Mapping)
            and (transition := job_transition(payload.get("project_id"), payload.get("session_id"), job)) is not None
        ])

    def page(self, *, client_id: str, after: int = 0, limit: int = MAX_PAGE) -> dict[str, Any]:
        _client(client_id)
        if type(after) is not int or not 0 <= after <= MAX_SEQUENCE:
            raise ValueError("attention_cursor_invalid")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
            raise ValueError("attention_limit_invalid")
        with self._connect() as connection:
            unseen = "NOT EXISTS (SELECT 1 FROM acknowledgements a WHERE a.sequence=attention.sequence AND a.client_id=?)"
            rows = connection.execute(
                f"SELECT {_ITEM_COLUMNS} FROM attention WHERE sequence>? AND {unseen} ORDER BY sequence LIMIT ?",
                (after, client_id, limit + 1),
            ).fetchall()
            count = connection.execute(f"SELECT COUNT(*) FROM attention WHERE {unseen}", (client_id,)).fetchone()[0]
            high_water = connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM attention").fetchone()[0]
        items = [dict(row) for row in rows[:limit]]
        return {
            "schema_version": SCHEMA_VERSION, "items": items,
            "next_after": items[-1]["sequence"] if items else max(after, high_water),
            "has_more": len(rows) > limit, "unacknowledged_count": count,
        }

    def acknowledge(self, *, client_id: str, sequences: list[int]) -> dict[str, Any]:
        _client(client_id)
        if not 1 <= len(sequences) <= MAX_PAGE or any(type(seq) is not int or not 1 <= seq <= MAX_SEQUENCE for seq in sequences):
            raise ValueError("attention_sequences_invalid")
        with self._connect() as connection:
            for sequence in set(sequences):
                if connection.execute("SELECT 1 FROM attention WHERE sequence=?", (sequence,)).fetchone() is None:
                    raise ValueError("attention_sequence_unknown")
                connection.execute("INSERT OR IGNORE INTO acknowledgements VALUES (?, ?)", (client_id, sequence))
        return {"schema_version": SCHEMA_VERSION, "acknowledged": sorted(set(sequences))}
