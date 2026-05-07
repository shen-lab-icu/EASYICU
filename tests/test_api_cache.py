from __future__ import annotations

import pandas as pd

from easyicu import api


def test_clear_global_loader_clears_datasource_cache() -> None:
    class Resolver:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class DataSource:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class Loader:
        def __init__(self) -> None:
            self.concept_resolver = Resolver()
            self.datasource = DataSource()

    loader = Loader()
    api._global_loader = loader
    api._loader_config = ("miiv", "/tmp/example", None, frozenset())

    api.clear_global_loader()

    assert loader.concept_resolver.cleared
    assert loader.datasource.cleared
    assert api._global_loader is None
    assert api._loader_config is None


def test_compress_dtypes_handles_table_wrappers() -> None:
    class Table:
        def __init__(self) -> None:
            self.data = pd.DataFrame({"value": [1.0, 2.0]})

    table = Table()

    compressed = api._compress_dtypes(table)

    assert compressed is table
    assert str(table.data["value"].dtype) == "Int8"
