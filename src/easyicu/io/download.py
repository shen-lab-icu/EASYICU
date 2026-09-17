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

from ..config import DataSourceConfig, DataSourceRegistry

LOGGER = logging.getLogger(__name__)


class DownloadError(OSError):
    """Typed failure for a data-source download.

    Raised instead of returning ``False`` so callers can distinguish
    "download failed" from "nothing to do". Carries the URL and destination
    for source-identity attribution.
    """

    def __init__(self, message: str, *, url: str = "", destination: str = "") -> None:
        super().__init__(message)
        self.url = url
        self.destination = destination


class PhysioNetDownloader:
    """Handler for downloading data from PhysioNet."""

    BASE_URL = "https://physionet.org/files/"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.username = username or os.environ.get("EASYICU_PHYSIONET_USER")
        self.password = password or os.environ.get("EASYICU_PHYSIONET_PASS")
        self.session = requests.Session()

    def _ensure_credentials(self) -> None:
        """Prompt for credentials if not already set."""
        if not self.username:
            self.username = input("PhysioNet username: ")
        if not self.password:
            self.password = getpass("PhysioNet password: ")

    def _auth(self) -> Optional[tuple[str, str]]:
        """Return known credentials without prompting, if available."""
        if self.username and self.password:
            return (self.username, self.password)
        return None

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
            True if download was successful.

        Raises:
            DownloadError: On network failure or hash mismatch, with the URL
                and destination attached for attribution.
        """
        if destination.exists() and not force:
            if verify_hash:
                if self._verify_sha256(destination, verify_hash):
                    LOGGER.info(f"File {destination.name} already exists and verified, skipping")
                    return True
                else:
                    LOGGER.warning(f"File {destination.name} exists but hash mismatch, re-downloading")
            else:
                LOGGER.info(f"File {destination.name} already exists, skipping")
                return True

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Known authenticated sources (e.g. PhysioNet) must carry
            # credentials on the FIRST request, not only after a 401: the
            # initial anonymous request can trigger audit throttling or
            # redirect away from the authenticated flow.
            response = self.session.get(
                url, auth=self._auth(), stream=True, timeout=30
            )

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
                    destination.unlink(missing_ok=True)
                    raise DownloadError(
                        f"Hash verification failed for {destination.name}",
                        url=url,
                        destination=str(destination),
                    )

            LOGGER.info(f"Successfully downloaded {destination.name}")
            return True

        except DownloadError:
            raise
        except requests.RequestException as e:
            if destination.exists():
                try:
                    destination.unlink()
                except OSError:
                    pass
            raise DownloadError(
                f"Failed to download {url}: {e}",
                url=url,
                destination=str(destination),
            ) from e

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

    Raises:
        DownloadError: If any table file fails to download or verify.
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
        # Try to infer URL from source name if it's a known source
        if config.name == "mimic_demo":
            base_url = "https://physionet.org/files/mimic-iv-demo/2.2/"
        elif config.name == "eicu_demo":
            base_url = "https://physionet.org/files/eicu-crd-demo/2.0/"
        else:
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
            
            # Check for hash in file entry
            verify_hash = file_entry.get("hash")

            if verbose:
                LOGGER.info(f"Downloading table {table_name}: {file_path}")

            try:
                downloader.download_file(url, dest, verify_hash=verify_hash, force=force)
            except DownloadError as exc:
                raise DownloadError(
                    f"Failed to download table '{table_name}' ({file_path}): {exc}",
                    url=url,
                    destination=str(dest),
                ) from exc


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

    Raises:
        DownloadError: If any source fails to download.
        KeyError: If a source name is unknown.
    """
    for source_name, data_dir in zip(source_names, data_dirs):
        try:
            config = registry.get(source_name)
        except KeyError:
            raise
        except Exception as exc:
            raise DownloadError(
                f"Failed to resolve source '{source_name}': {exc}",
                url="",
                destination=str(data_dir),
            ) from exc
        try:
            download_src(config, Path(data_dir), **kwargs)
        except DownloadError as exc:
            raise DownloadError(
                f"Failed to download source '{source_name}': {exc}",
                url=exc.url,
                destination=exc.destination or str(data_dir),
            ) from exc


def download_demo(
    data_dir: Path | str,
    source: str = "mimic_demo",
    force: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Download demo data (MIMIC-IV Demo or eICU Demo).

    Args:
        data_dir: Directory where data will be downloaded
        source: 'mimic_demo' or 'eicu_demo'
        force: If True, re-download existing files
        username: PhysioNet username
        password: PhysioNet password

    Raises:
        KeyError: If the demo source is unknown.
        DownloadError: If the download fails.
    """
    from ..resources import load_data_sources

    registry = load_data_sources()
    config = registry.get(source)
    if config is None:
        raise KeyError(f"Demo source '{source}' not found in registry")

    download_src(
        config,
        Path(data_dir),
        force=force,
        username=username,
        password=password,
    )
