"""Inspect registered aggregates and audited figures; never copy a run tree."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import struct
import zlib

from easyicu.research_agent.gates.figure_egress import (
    HOST_PRIVACY_AUDIT_BASIS,
    TRUSTED_FIGURE_PRODUCERS,
)
from easyicu.research_agent.gates.figure_privacy import (
    FIGURE_PRIVACY_RECEIPT_SCHEMA,
    TRUSTED_AUDIT_VERSIONS,
)

from .study_replay_io import (
    MAX_BUNDLE_BYTES,
    ReplayFiles,
    digest,
    identity,
    parse_json,
    read_snapshot,
    relative_parts,
)
from .study_replay_policy import (
    ReplayExportError,
    aggregate_table,
    public_text,
    scrub_secret_values,
)


def clean_png(raw: bytes) -> bytes:
    """Retain only pixel chunks; reject malformed, animated or oversized PNGs."""
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ReplayExportError("replay_figure_invalid")
    offset, chunks, pixels = 8, [], []
    width = height = channels = 0
    ended = False
    while offset < len(raw):
        if offset + 12 > len(raw) or ended:
            raise ReplayExportError("replay_figure_invalid")
        length = struct.unpack("!I", raw[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(raw):
            raise ReplayExportError("replay_figure_invalid")
        kind, data = raw[offset + 4 : offset + 8], raw[offset + 8 : end - 4]
        if zlib.crc32(kind + data) != struct.unpack("!I", raw[end - 4 : end])[0]:
            raise ReplayExportError("replay_figure_invalid")
        if not chunks:
            if kind != b"IHDR" or len(data) != 13:
                raise ReplayExportError("replay_figure_invalid")
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack("!2I5B", data)
            )
            channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color, 0)
            if (
                not width
                or not height
                or width * height > 4_000_000
                or depth != 8
                or not channels
                or compression
                or filtering
                or interlace
            ):
                raise ReplayExportError("replay_figure_invalid")
        elif kind == b"IHDR":
            raise ReplayExportError("replay_figure_invalid")
        if kind == b"IDAT":
            pixels.append(data)
        if kind == b"IEND":
            if data or not pixels:
                raise ReplayExportError("replay_figure_invalid")
            ended = True
        if kind in {b"IHDR", b"IDAT", b"IEND"}:
            chunks.append(raw[offset:end])
        elif kind in {b"acTL", b"fcTL", b"fdAT"} or not kind[0] & 32:
            raise ReplayExportError("replay_figure_invalid")
        offset = end
    if not ended:
        raise ReplayExportError("replay_figure_invalid")
    expected = height * (1 + width * channels)
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(b"".join(pixels), expected + 1)
        if (
            len(decoded) != expected
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            raise ValueError("invalid pixels")
    except (ValueError, zlib.error):
        raise ReplayExportError("replay_figure_invalid") from None
    return raw[:8] + b"".join(chunks)


class ReplayEvidence:
    """Resolve selected renderer coordinates through the immutable evidence index."""

    def __init__(self, files: ReplayFiles, run_id: str):
        self.files = files
        self.prefix = ""
        records = files.json("evidence/evidence_index.json", optional=True)
        if records is None:
            self.prefix = f"pipeline/{identity(run_id)}/"
            records = files.json(self.prefix + "evidence/evidence_index.json")
        if not isinstance(records, list) or len(records) > 10_000:
            raise ReplayExportError("replay_record_invalid")
        self.records: dict[str, dict] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ReplayExportError("replay_record_invalid")
            ident = identity(record.get("evidence_id"))
            if ident in self.records:
                raise ReplayExportError("replay_evidence_ambiguous")
            self.records[ident] = record
        self.snapshots: dict[str, bytes] = {}
        self.receipts: dict[str, dict] = {}
        self.tables: dict[str, dict] = {}
        self.redacted_tables: set[str] = set()

    def record(self, evidence_id: str, kind: str) -> dict:
        record = self.records.get(identity(evidence_id))
        if record is None:
            raise ReplayExportError("replay_evidence_missing")
        if record.get("kind") != kind:
            raise ReplayExportError("replay_evidence_class_forbidden")
        return record

    def raw(self, record: dict) -> bytes:
        ident = record["evidence_id"]
        if ident not in self.snapshots:
            relative = record.get("relative_path")
            if not isinstance(relative, str):
                raise ReplayExportError("replay_path_forbidden")
            # Validate before prefixing: an absolute path must not become relative.
            relative_parts(relative)
            raw = read_snapshot(self.files.root, self.prefix + relative)
            if sum(map(len, self.snapshots.values())) + len(raw) > MAX_BUNDLE_BYTES:
                raise ReplayExportError("replay_size_limit")
            if digest(raw) != record.get("sha256"):
                raise ReplayExportError("replay_digest_mismatch")
            self.snapshots[ident] = raw
            self.receipts[ident] = {
                "evidence_id": ident,
                "kind": record["kind"],
                "sha256": digest(raw),
            }
        return self.snapshots[ident]

    def table(self, evidence_id: str, *, expected: str | None = None) -> dict:
        record = self.record(evidence_id, "table")
        relative_parts(record.get("relative_path"))
        suffix = Path(str(record.get("relative_path"))).suffix.lower()
        if suffix not in {".csv", ".tsv", ".json"}:
            raise ReplayExportError("replay_evidence_class_forbidden")
        if not record.get("script_evidence_id") or record.get("produced_by_step") in {
            None,
            "",
            "cohort_definition",
        }:
            raise ReplayExportError("replay_aggregate_unproven")
        if expected is not None and record.get("sha256") != expected:
            raise ReplayExportError("replay_digest_mismatch")
        if evidence_id not in self.tables:
            raw = self.raw(record)
            try:
                if suffix == ".json":
                    value = parse_json(raw)
                    if not isinstance(value, dict) or set(value) != {"headers", "rows"}:
                        raise ReplayExportError("replay_table_invalid")
                    headers, rows = value["headers"], value["rows"]
                else:
                    reader = csv.reader(
                        io.StringIO(raw.decode("utf-8"), newline=""),
                        delimiter="\t" if suffix == ".tsv" else ",",
                        strict=True,
                    )
                    headers, rows = next(reader, []), []
                    for row in reader:
                        if len(rows) >= 1000:
                            raise ReplayExportError("replay_size_limit")
                        rows.append(row)
            except (UnicodeError, csv.Error):
                raise ReplayExportError("replay_table_invalid") from None
            if (
                not isinstance(headers, list)
                or not all(isinstance(x, str) for x in headers)
                or not isinstance(rows, list)
                or not all(isinstance(x, list) for x in rows)
            ):
                raise ReplayExportError("replay_table_invalid")
            projected = aggregate_table(headers, rows)
            if (
                scrub_secret_values(projected, self.files.secrets) != projected
                or headers != projected["headers"]
                or [[str(cell) for cell in row] for row in rows] != projected["rows"]
            ):
                self.redacted_tables.add(evidence_id)
            self.tables[evidence_id] = {
                "evidence_id": evidence_id,
                "sha256": record["sha256"],
                **projected,
            }
        return dict(self.tables[evidence_id])

    def figure_contract(self, metadata: dict) -> None:
        record = self.record(metadata.get("contract_evidence_id"), "log")
        contract = parse_json(self.raw(record))
        if (
            record.get("producer") not in TRUSTED_FIGURE_PRODUCERS
            or not isinstance(contract, dict)
            or contract.get("figure_id") != metadata.get("figure_id")
        ):
            raise ReplayExportError("replay_figure_uncleared")
        pending = [("", contract)]
        while pending:
            key, node = pending.pop()
            # Registry coordinates are not rendered text and may include hashes.
            if key.endswith(
                ("_evidence_id", "_evidence_ids", "_artifact_ids", "_sha256")
            ):
                continue
            if isinstance(node, dict):
                pending.extend(node.items())
            elif isinstance(node, list):
                pending.extend((key, item) for item in node)
            elif isinstance(node, str) and (
                public_text(node) != node
                or scrub_secret_values(node, self.files.secrets) != node
            ):
                # Sanitizing the JSON cannot erase text already drawn in pixels.
                raise ReplayExportError("replay_figure_text_unsafe")

    def figure(self, row: dict) -> tuple[dict, bytes]:
        matches = [
            record
            for record in self.records.values()
            if record.get("relative_path") == row.get("relative_path")
            and record.get("kind") == "figure"
        ]
        if len(matches) != 1:
            raise ReplayExportError("replay_figure_uncleared")
        record = matches[0]
        if Path(record["relative_path"]).suffix.lower() != ".png":
            raise ReplayExportError("replay_evidence_class_forbidden")
        metadata = record.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("aggregate_only") is not True
            or metadata.get("aggregate_only_basis") != HOST_PRIVACY_AUDIT_BASIS
            or record.get("producer") not in TRUSTED_FIGURE_PRODUCERS
            or metadata.get("aggregate_only_audit_version")
            not in TRUSTED_AUDIT_VERSIONS
        ):
            raise ReplayExportError("replay_figure_uncleared")
        audit = self.record(metadata.get("figure_privacy_audit_evidence_id"), "log")
        if audit.get("producer") not in TRUSTED_FIGURE_PRODUCERS:
            raise ReplayExportError("replay_figure_uncleared")
        receipt = parse_json(self.raw(audit))
        source_ids = metadata.get("source_evidence_ids")
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != FIGURE_PRIVACY_RECEIPT_SCHEMA
            or receipt.get("audit_version")
            != metadata.get("aggregate_only_audit_version")
            or receipt.get("aggregate_only") is not True
            or not metadata.get("figure_id")
            or receipt.get("figure_id") != metadata["figure_id"]
            or not isinstance(source_ids, list)
            or not source_ids
            or len(source_ids) != len(set(source_ids))
        ):
            raise ReplayExportError("replay_figure_uncleared")
        declared = metadata.get("source_evidence_sha256")
        cleared = receipt.get("source_sha256")
        if (
            not isinstance(declared, dict)
            or not isinstance(cleared, dict)
            or set(declared) != set(source_ids)
            or set(cleared) != set(source_ids)
        ):
            raise ReplayExportError("replay_figure_uncleared")
        for source_id in source_ids:
            source = self.record(source_id, "table")
            if source.get("sha256") != declared.get(source_id) or declared.get(
                source_id
            ) != cleared.get(source_id):
                raise ReplayExportError("replay_figure_uncleared")
            self.table(
                source_id
            )  # Reinspect actual bytes, not just the historical verdict.
            if source_id in self.redacted_tables:
                raise ReplayExportError("replay_figure_text_unsafe")
        self.figure_contract(metadata)
        raw = self.raw(record)
        if row.get("sha256", record["sha256"]) != record["sha256"]:
            raise ReplayExportError("replay_digest_mismatch")
        clean = clean_png(raw)
        return (
            {
                "evidence_id": record["evidence_id"],
                "source_sha256": record["sha256"],
                "path": f"figures/{record['evidence_id']}.png",
                "sha256": digest(clean),
                "label": public_text(row.get("label", "Figure")),
                "caption": public_text(row.get("caption", "")),
                "mark_count_verified": receipt.get("mark_count_verified") is True,
            },
            clean,
        )
