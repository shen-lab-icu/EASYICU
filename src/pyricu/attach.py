"""Enhanced data source attachment with lazy loading capabilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from .config import DataSourceConfig, DataSourceRegistry
from .datasource import ICUDataSource

LOGGER = logging.getLogger(__name__)

class DataSourceEnvironment:
    """Environment container for attached data sources.

    This class mimics R's environment behavior, providing lazy loading
    of data sources and tables.
    """

    def __init__(self):
        self._sources: Dict[str, ICUDataSource] = {}
        self._configs: Dict[str, DataSourceConfig] = {}
        self._data_dirs: Dict[str, Path] = {}

    def attach_src(
        self,
        config: DataSourceConfig,
        data_dir: Path,
        *,
        force: bool = False,
    ) -> None:
        """Attach a data source to the environment.

        Args:
            config: Data source configuration
            data_dir: Directory containing the data files
            force: If True, re-attach even if already attached
        """
        name = config.name

        if name in self._sources and not force:
            LOGGER.info(f"Data source '{name}' already attached")
            return

        self._configs[name] = config
        self._data_dirs[name] = data_dir

        # Create lazy-loading data source
        source = ICUDataSource(
            config,
            base_path=data_dir,
            registry=None,
        )

        self._sources[name] = source
        LOGGER.info(f"Attached data source '{name}' from {data_dir}")

    def detach_src(self, name: str) -> None:
        """Detach a data source from the environment.

        Args:
            name: Name of the data source to detach
        """
        if name in self._sources:
            del self._sources[name]
            del self._configs[name]
            del self._data_dirs[name]
            LOGGER.info(f"Detached data source '{name}'")
        else:
            LOGGER.warning(f"Data source '{name}' not attached")

    def get_source(self, name: str) -> Optional[ICUDataSource]:
        """Get an attached data source.

        Args:
            name: Name of the data source

        Returns:
            ICUDataSource if attached, None otherwise
        """
        return self._sources.get(name)

    def list_sources(self) -> list[str]:
        """List all attached data sources.

        Returns:
            List of attached source names
        """
        return list(self._sources.keys())

    def is_attached(self, name: str) -> bool:
        """Check if a data source is attached.

        Args:
            name: Name of the data source

        Returns:
            True if attached, False otherwise
        """
        return name in self._sources

    def __contains__(self, name: str) -> bool:
        return self.is_attached(name)

    def __getitem__(self, name: str) -> ICUDataSource:
        source = self.get_source(name)
        if source is None:
            raise KeyError(f"Data source '{name}' not attached")
        return source

    def __repr__(self) -> str:
        sources = ", ".join(self.list_sources())
        return f"DataSourceEnvironment({sources})"

# Global data environment instance
data = DataSourceEnvironment()

def attach_src(
    source_name: str,
    registry: DataSourceRegistry,
    data_dir: Path | str,
    *,
    assign_to_global: bool = True,
) -> ICUDataSource:
    """Attach a data source for use.

    Args:
        source_name: Name of the data source
        registry: Registry containing configurations
        data_dir: Directory containing the data
        assign_to_global: If True, add to global data environment

    Returns:
        Attached ICUDataSource
    """
    config = registry.get(source_name)
    data_dir = Path(data_dir)

    if assign_to_global:
        data.attach_src(config, data_dir)
        return data[source_name]
    else:
        return ICUDataSource(config, base_path=data_dir)

def detach_src(source_name: str) -> None:
    """Detach a data source.

    Args:
        source_name: Name of the data source to detach
    """
    data.detach_src(source_name)

def list_attached() -> list[str]:
    """List all attached data sources.

    Returns:
        List of attached source names
    """
    return data.list_sources()

def src_data_avail(source_name: Optional[str] = None) -> Dict[str, bool]:
    """Check data availability for attached sources.

    Args:
        source_name: Specific source to check, or None for all

    Returns:
        Dictionary mapping source names to availability status
    """
    if source_name:
        sources = [source_name]
    else:
        sources = data.list_sources()

    availability = {}
    for name in sources:
        if not data.is_attached(name):
            availability[name] = False
            continue

        source = data[name]
        config = source.config

        # Check if data files exist
        all_available = True
        if source.base_path is not None:
            for table_name in config.tables.keys():
                table_file = source.base_path / f"{table_name}.parquet"
                if not table_file.exists():
                    all_available = False
                    break

        availability[name] = all_available

    return availability

def setup_src_data(
    source_name: str,
    registry: DataSourceRegistry,
    data_dir: Path | str,
    *,
    force: bool = False,
) -> None:
    """Complete data setup: download, import, and attach.

    Args:
        source_name: Name of the data source
        registry: Registry containing configurations
        data_dir: Directory for data storage
        force: If True, force re-download and re-import
    """
    from .download import download_src
    from .import_data import import_src

    config = registry.get(source_name)
    data_dir = Path(data_dir)

    LOGGER.info(f"Setting up data source '{source_name}'")

    # Download if needed
    try:
        download_src(config, data_dir, force=force)
    except Exception as e:
        LOGGER.warning(f"Download failed or skipped: {e}")

    # Import
    import_src(config, data_dir, force=force)

    # Attach
    attach_src(source_name, registry, data_dir)

    LOGGER.info(f"Successfully set up data source '{source_name}'")
