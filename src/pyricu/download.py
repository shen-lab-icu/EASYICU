"""Data download utilities for PhysioNet and other ICU datasets.

This module provides functionality to download ICU datasets from various sources,
primarily PhysioNet. It handles authentication, download progress tracking,
and verification of downloaded files.
"""

from __future__ import annotations

import hashlib
import logging
import os
from getpass import getpass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from .config import DataSourceConfig, DataSourceRegistry

LOGGER = logging.getLogger(__name__)

class PhysioNetDownloader:
    """Handler for downloading data from PhysioNet."""

    BASE_URL = "https://physionet.org/files/"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.username = username or os.environ.get("RICU_PHYSIONET_USER")
        self.password = password or os.environ.get("RICU_PHYSIONET_PASS")
        self.session = requests.Session()

    def _ensure_credentials(self) -> None:
        """Prompt for credentials if not already set."""
        if not self.username:
            self.username = input("PhysioNet username: ")
        if not self.password:
            self.password = getpass("PhysioNet password: ")

    def download_file(
        self,
        url: str,
        destination: Path,
        *,
        verify_hash: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Download a single file with progress tracking.

        Args:
            url: URL of the file to download
            destination: Local path where the file will be saved
            verify_hash: Optional SHA256 hash to verify download
            force: If True, re-download even if file exists

        Returns:
            True if download was successful, False otherwise
        """
        if destination.exists() and not force:
            LOGGER.info(f"File {destination.name} already exists, skipping")
            return True

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self.session.get(url, stream=True, timeout=30)

            if response.status_code == 401:
                self._ensure_credentials()
                response = self.session.get(
                    url,
                    auth=(self.username, self.password),
                    stream=True,
                    timeout=30,
                )

            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with destination.open("wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            if verify_hash:
                if not self._verify_sha256(destination, verify_hash):
                    LOGGER.error(f"Hash verification failed for {destination.name}")
                    destination.unlink()
                    return False

            LOGGER.info(f"Successfully downloaded {destination.name}")
            return True

        except requests.RequestException as e:
            LOGGER.error(f"Failed to download {url}: {e}")
            if destination.exists():
                destination.unlink()
            return False

    def _verify_sha256(self, file_path: Path, expected_hash: str) -> bool:
        """Verify SHA256 hash of downloaded file."""
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        computed_hash = sha256_hash.hexdigest()
        return computed_hash.lower() == expected_hash.lower()

def download_src(
    config: DataSourceConfig,
    data_dir: Path,
    *,
    tables: Optional[Sequence[str]] = None,
    force: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Download data source tables.

    Args:
        config: Data source configuration
        data_dir: Directory where data will be downloaded
        tables: List of table names to download; None downloads all
        force: If True, re-download existing files
        username: PhysioNet username
        password: PhysioNet password
        verbose: If True, print progress information
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine which tables to download
    if tables is None:
        tables = list(config.tables.keys())

    # Check which files already exist
    if not force:
        existing = []
        for table_name in tables:
            table_cfg = config.get_table(table_name)
            file_path = table_cfg.first_file()
            if file_path and (data_dir / file_path).exists():
                existing.append(table_name)
        
        tables = [t for t in tables if t not in existing]
        if existing and verbose:
            LOGGER.info(f"Skipping {len(existing)} existing tables: {', '.join(existing)}")

    if not tables:
        LOGGER.info("All requested tables have already been downloaded")
        return

    # Get base URL from config or construct it
    base_url = config.extra.get("url", "")
    if not base_url:
        LOGGER.warning("No download URL found in configuration")
        return

    downloader = PhysioNetDownloader(username, password)

    # Download each table
    for table_name in tables:
        table_cfg = config.get_table(table_name)
        
        for file_entry in table_cfg.files:
            file_path = file_entry.get("path") or file_entry.get("file")
            if not file_path:
                continue

            url = urljoin(base_url, file_path)
            dest = data_dir / file_path

            if verbose:
                LOGGER.info(f"Downloading table {table_name}: {file_path}")

            success = downloader.download_file(url, dest, force=force)
            if not success:
                LOGGER.error(f"Failed to download {table_name}")

def download_sources(
    source_names: Iterable[str],
    registry: DataSourceRegistry,
    data_dirs: Sequence[Path | str],
    **kwargs,
) -> None:
    """Download multiple data sources.

    Args:
        source_names: List of data source names
        registry: Registry containing data source configurations
        data_dirs: Directories corresponding to each source
        **kwargs: Additional arguments passed to download_src
    """
    for source_name, data_dir in zip(source_names, data_dirs):
        try:
            config = registry.get(source_name)
            download_src(config, Path(data_dir), **kwargs)
        except Exception as e:
            LOGGER.error(f"Failed to download {source_name}: {e}")
