"""A small published native export and three ways to damage one member.

Built through the real ``extract_database(native_export_v2=True)`` path on a
synthetic HiRID source, in process (no worker processes).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATABASE = "hirid"
MODULES = ("demographics", "outcome")
DAMAGED_MODULE = "outcome"
DAMAGES = ("truncated", "flipped", "rewritten")


class _InProcess:
    def __init__(self, *, target, args, daemon):
        self.target, self.args = target, args
        self.exitcode = None

    def start(self):
        try:
            self.target(*self.args)
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1

    def join(self):
        return None


class _Context:
    @staticmethod
    def Process(*, target, args, daemon):
        return _InProcess(target=target, args=args, daemon=daemon)


def publish_synthetic_export(root: Path, monkeypatch) -> Path:
    """Generate a 6-stay source and publish a two-module native export."""

    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = root / "source" / DATABASE
    generate_source(source, DATABASE, stays=6)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: _Context())
    export = root / "export" / DATABASE
    try:
        api.extract_database(
            DATABASE, data_path=str(source), output_dir=str(export),
            modules=list(MODULES), native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()
    return export


def member_file(export: Path, module: str = DAMAGED_MODULE) -> str:
    manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
    return next(entry["file"] for entry in manifest["files"] if entry["module"] == module)


def damaged_copy(export: Path, destination: Path, damage: str) -> Path:
    """Copy ``export`` and damage its outcome member.

    ``truncated``: half the bytes (no footer). ``flipped``: one byte xor 1,
    footer intact. ``rewritten``: one numeric value +1, same schema, metadata
    and row count (a readable file holding different data).
    """

    shutil.copytree(export, destination)
    target = destination / member_file(export)
    data = target.read_bytes()
    if damage == "truncated":
        target.write_bytes(data[: len(data) // 2])
    elif damage == "flipped":
        flipped = bytearray(data)
        flipped[len(flipped) // 3] ^= 0x01
        target.write_bytes(bytes(flipped))
    elif damage == "rewritten":
        table = pq.read_table(target)
        for field in table.schema:
            if field.name in {"stay_id", "charttime"}:
                continue
            if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
                values = table.column(field.name).to_pylist()
                index = next((i for i, v in enumerate(values) if v is not None), None)
                if index is None:
                    continue
                values[index] += 1
                table = table.set_column(
                    table.schema.get_field_index(field.name), field,
                    pa.array(values, type=field.type),
                )
                break
        else:  # pragma: no cover - the synthetic outcome always has a value
            raise AssertionError("no numeric value to change")
        pq.write_table(table.replace_schema_metadata(table.schema.metadata), target)
        assert target.read_bytes() != data
    else:  # pragma: no cover
        raise ValueError(damage)
    return destination


def strip_digests(export: Path) -> None:
    """Make ``export`` look like a pre-digest manifest (listing only)."""

    path = export / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        entry.pop("parquet_sha256", None)
        entry.pop("parquet_bytes", None)
    path.write_text(json.dumps(manifest), encoding="utf-8")
