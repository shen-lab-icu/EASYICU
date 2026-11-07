"""Public storage-layout contract consumed by concept loading and inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

__all__ = ["ConceptDataSourceStorage"]


@runtime_checkable
class ConceptDataSourceStorage(Protocol):
    """The storage capabilities concept code may request from a data source."""

    def resolve_bucket_directory(self, table_name: str) -> Optional[Path]: ...

    def resolve_flat_parquet_directory(self, table_name: str) -> Optional[Path]: ...

    def resolve_loader_from_disk(
        self, table_name: str
    ) -> Optional[Callable[[], pd.DataFrame] | Path]: ...

    def resolve_target_files(
        self, table_name: str, selector_column: Optional[str] = None,
        selector_values: Optional[Iterable[object]] = None,
        *, loader: Optional[Path] = None,
        consume: bool = True,
    ) -> Tuple[Path, ...]: ...

    def get_bucket_files_for_ids(
        self,
        bucket_dir: Path,
        itemids: Iterable[object],
        duckdb_module: Any,
        *, selector_column: Optional[str] = None,
    ) -> Tuple[set[Any], int, Tuple[Path, ...]]: ...
