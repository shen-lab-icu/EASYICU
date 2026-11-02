"""Synthetic sentinel: metadata registry reads manifests, never source tables."""
import builtins
import io
import os
from pathlib import Path


def forbid_source_table_reads(monkeypatch, *, roots):
    import pandas as pd
    import pyarrow.parquet as pq

    roots = [Path(root).resolve() for root in roots]
    table_names = {path.name for root in roots for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}}
    accesses = []
    def reject_table(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value))
            if (path.suffix.lower() in {'.csv', '.parquet'}
                    and ((not path.is_absolute() and path.name in table_names)
                         or any(path.resolve().is_relative_to(root) for root in roots))):
                accesses.append(str(path))
                raise AssertionError('source table opened during metadata-only GET')
    def wrap(function):
        def guarded(path, *args, **kwargs):
            reject_table(path)
            return function(path, *args, **kwargs)
        return guarded
    monkeypatch.setattr(builtins, 'open', wrap(builtins.open))
    monkeypatch.setattr(io, 'open', wrap(io.open))
    monkeypatch.setattr(os, 'open', wrap(os.open))
    monkeypatch.setattr(pd, 'read_csv', wrap(pd.read_csv))
    monkeypatch.setattr(pd, 'read_parquet', wrap(pd.read_parquet))
    monkeypatch.setattr(pq, 'read_table', wrap(pq.read_table))
    monkeypatch.setattr(pq, 'ParquetFile', wrap(pq.ParquetFile))
    return accesses
