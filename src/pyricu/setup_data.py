"""Data setup orchestration module.

This module provides a high-level API for setting up ICU data sources,
handling the entire workflow of downloading, importing, and attaching data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .config import DataSourceRegistry
from .attach import attach_src, is_attached
from .download import download_src, download_demo
from .import_data import import_src
from .resources import load_data_sources

LOGGER = logging.getLogger(__name__)


def setup_data(
    source_name: str,
    data_dir: Union[str, Path],
    *,
    download: bool = True,
    import_data: bool = True,
    attach: bool = True,
    force_download: bool = False,
    force_import: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Set up an ICU data source.

    This function orchestrates the process of:
    1. Downloading raw data (optional)
    2. Importing/Converting data to efficient format (optional)
    3. Attaching the data source for use

    Args:
        source_name: Name of the data source (e.g., 'mimic_demo', 'miiv')
        data_dir: Directory where data will be stored
        download: Whether to download raw data
        import_data: Whether to import/convert data
        attach: Whether to attach the source after setup
        force_download: Force re-download of existing files
        force_import: Force re-import of existing tables
        username: PhysioNet username (for download)
        password: PhysioNet password (for download)
    """
    data_dir = Path(data_dir)
    registry = load_data_sources()
    
    try:
        config = registry.get(source_name)
    except KeyError:
        LOGGER.error(f"Data source '{source_name}' not found in registry")
        return

    LOGGER.info(f"Setting up data source '{source_name}' in {data_dir}")

    # 1. Download
    if download:
        LOGGER.info(f"Step 1/3: Downloading {source_name}...")
        try:
            if source_name.endswith('_demo'):
                download_demo(
                    data_dir, 
                    source=source_name, 
                    force=force_download,
                    username=username,
                    password=password
                )
            else:
                download_src(
                    config, 
                    data_dir, 
                    force=force_download,
                    username=username,
                    password=password
                )
        except Exception as e:
            LOGGER.error(f"Download failed: {e}")
            # Continue to import step as data might already exist
    
    # 2. Import
    if import_data:
        LOGGER.info(f"Step 2/3: Importing {source_name}...")
        try:
            import_src(config, data_dir, force=force_import)
        except Exception as e:
            LOGGER.error(f"Import failed: {e}")
            return

    # 3. Attach
    if attach:
        LOGGER.info(f"Step 3/3: Attaching {source_name}...")
        try:
            attach_src(source_name, registry, data_dir)
            LOGGER.info(f"Successfully attached {source_name}")
        except Exception as e:
            LOGGER.error(f"Attach failed: {e}")

    LOGGER.info(f"Setup for '{source_name}' completed.")
