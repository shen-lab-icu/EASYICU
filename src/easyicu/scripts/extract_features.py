"""Command-line helper to extract ICU feature tables from local datasets.

This script mirrors the `extract_data()` routine in `ricu.R`, grouping
clinical concepts into thematic tables and exporting them as CSV files.
It relies on the packaged concept dictionary and data-source registry
provided by :mod:`easyicu`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .. import ConceptDictionary, ConceptResolver, DataSourceRegistry, ICUDataSource
from ..resources import load_data_sources, load_dictionary

LOGGER = logging.getLogger(__name__)

# Import optional modules
try:
    from ..download import download_src
    HAS_DOWNLOAD = True
except ImportError:
    HAS_DOWNLOAD = False

try:
    from ..import_data import import_src
    HAS_IMPORT = True
except ImportError:
    HAS_IMPORT = False

try:
    from ..attach import attach_src, data, setup_src_data
    HAS_ATTACH = True
except ImportError:
    HAS_ATTACH = False


DEFAULT_GROUPS: Mapping[str, Sequence[str]] = {
    "demo": ("age", "bmi", "height", "sex", "weight"),
    "outcome": (
        "death",
        "los_icu",
        "qsofa",
        "sirs",
        "sofa",
        "sofa_cardio",
        "sofa_cns",
        "sofa_coag",
        "sofa_liver",
        "sofa_renal",
        "sofa_resp",
    ),
    "vital": ("dbp", "etco2", "hr", "map", "sbp", "temp"),
    "neu": ("avpu", "egcs", "gcs", "mgcs", "rass", "vgcs"),
    "output": ("urine", "urine24"),
    "resp": ("ett_gcs", "mech_vent", "o2sat", "sao2", "pafi", "resp", "safi", "supp_o2", "vent_ind"),
    "lab": (
        "alb",
        "alp",
        "alt",
        "ast",
        "bicar",
        "bili",
        "bili_dir",
        "bun",
        "ca",
        "ck",
        "ckmb",
        "cl",
        "crea",
        "crp",
        "glu",
        "k",
        "mg",
        "na",
        "phos",
        "tnt",
    ),
    "blood": ("be", "cai", "fio2", "hbco", "lact", "methb", "pco2", "ph", "po2", "tco2"),
    "hematology": ("bnd", "esr", "fgn", "hgb", "inr_pt", "lymph", "mch", "mchc", "mcv", "neut", "plt", "ptt", "wbc"),
    "med": (
        "abx",
        "adh_rate",
        "cort",
        "dex",
        "dobu_dur",
        "dobu_rate",
        "dobu60",
        "epi_dur",
        "epi_rate",
        "ins",
        "norepi_dur",
        "norepi_equiv",
        "norepi_rate",
        "vaso_ind",
    ),
}


def extract_groups(
    resolver: ConceptResolver,
    datasource: ICUDataSource,
    groups: Mapping[str, Iterable[str]],
) -> Dict[str, pd.DataFrame]:
    """Resolve grouped concepts for a single data source."""

    results: Dict[str, pd.DataFrame] = {}
    for group_name, concepts in groups.items():
        try:
            LOGGER.info("Loading %s concepts: %s", group_name, ", ".join(concepts))
            frame = resolver.load_concepts(concepts, datasource)
            results[group_name] = frame
        except NotImplementedError as exc:
            LOGGER.warning("Concept group '%s' skipped (%s)", group_name, exc)
        except Exception:
            LOGGER.exception("Failed to load group '%s'", group_name)
            raise
    return results


def export_groups(
    frames: Mapping[str, pd.DataFrame],
    destination: Path,
    prefix: str,
) -> None:
    """Persist each concept group as CSV."""

    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        out_path = destination / f"{prefix}_{name}.csv"
        frame.to_csv(out_path, index=False)
        LOGGER.info("Wrote %s (%d rows)", out_path, len(frame))


def resolve_datasource(
    registry: DataSourceRegistry,
    source_name: str,
    data_dir: Path,
) -> ICUDataSource:
    """Instantiate an :class:`ICUDataSource` for ``source_name``."""

    config = registry.get(source_name)
    datasource = ICUDataSource(config, base_path=data_dir)
    return datasource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    copilot_parser = subparsers.add_parser(
        "copilot",
        help="Manage the governed Pi Copilot shell runtime",
    )
    copilot_subparsers = copilot_parser.add_subparsers(
        dest="copilot_command",
        required=True,
    )
    copilot_install_parser = copilot_subparsers.add_parser(
        "install",
        help="Install the exact pinned Pi runtime from the packaged lockfile",
    )
    copilot_install_parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Override the private versioned runtime directory.",
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract concept features from data sources",
    )
    extract_parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="List of data-source identifiers (e.g. mimic, eicu, hirid, aumc, miiv).",
    )
    extract_parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Filesystem directories matching the sources order.",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        default=Path("feature_exports"),
        help="Directory where CSV outputs will be written.",
    )
    extract_parser.add_argument(
        "--groups-json",
        type=Path,
        help="Optional JSON file overriding the default concept groups.",
    )
    extract_parser.add_argument(
        "--dictionary",
        type=Path,
        help="Optional path to a custom concept dictionary JSON.",
    )
    extract_parser.add_argument(
        "--registry",
        type=Path,
        help="Optional path to a custom data-sources JSON.",
    )

    # Download command
    if HAS_DOWNLOAD:
        download_parser = subparsers.add_parser(
            "download",
            help="Download data sources from PhysioNet",
        )
        download_parser.add_argument(
            "--sources",
            nargs="+",
            required=True,
            help="List of data sources to download",
        )
        download_parser.add_argument(
            "--data-dirs",
            nargs="+",
            required=True,
            help="Directories for storing downloaded data",
        )
        download_parser.add_argument(
            "--force",
            action="store_true",
            help="Force re-download of existing files",
        )
        download_parser.add_argument(
            "--username",
            help="PhysioNet username (or set EASYICU_PHYSIONET_USER)",
        )
        download_parser.add_argument(
            "--password",
            help="PhysioNet password (or set EASYICU_PHYSIONET_PASS)",
        )

    # Import command
    if HAS_IMPORT:
        import_parser = subparsers.add_parser(
            "import",
            help="Import CSV data to efficient formats",
        )
        import_parser.add_argument(
            "--sources",
            nargs="+",
            required=True,
            help="List of data sources to import",
        )
        import_parser.add_argument(
            "--data-dirs",
            nargs="+",
            required=True,
            help="Directories containing CSV files",
        )
        import_parser.add_argument(
            "--force",
            action="store_true",
            help="Force re-import of existing data",
        )
        import_parser.add_argument(
            "--cleanup",
            action="store_true",
            help="Delete CSV files after successful import",
        )

    # Setup command
    if HAS_DOWNLOAD and HAS_IMPORT and HAS_ATTACH:
        setup_parser = subparsers.add_parser(
            "setup",
            help="Complete setup: download, import, and attach",
        )
        setup_parser.add_argument(
            "--sources",
            nargs="+",
            required=True,
            help="List of data sources to set up",
        )
        setup_parser.add_argument(
            "--data-dirs",
            nargs="+",
            required=True,
            help="Directories for data storage",
        )
        setup_parser.add_argument(
            "--force",
            action="store_true",
            help="Force complete re-setup",
        )

    # Common arguments
    for subparser in [extract_parser] + ([download_parser] if HAS_DOWNLOAD else []) + ([import_parser] if HAS_IMPORT else []):
        subparser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging verbosity.",
        )
    
    return parser


def load_groups(groups_json: Path | None) -> Mapping[str, Iterable[str]]:
    if groups_json is None:
        return DEFAULT_GROUPS
    import json

    with groups_json.open("r", encoding="utf8") as handle:
        payload = json.load(handle)
    return {key: tuple(value) for key, value in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, 'command') or args.command is None:
        parser.print_help()
        return 1

    logging.basicConfig(level=getattr(logging, getattr(args, 'log_level', 'INFO')))

    # Handle different commands
    if args.command == "download":
        return handle_download(args)
    elif args.command == "import":
        return handle_import(args)
    elif args.command == "setup":
        return handle_setup(args)
    elif args.command == "extract":
        return handle_extract(args)
    elif args.command == "copilot":
        from easyicu.webserver.pi_copilot.install import install_runtime

        installed = install_runtime(destination=args.runtime_dir)
        LOGGER.info("Installed pinned Pi Copilot runtime at %s", installed)
        return 0
    else:
        parser.print_help()
        return 1


def handle_download(args) -> int:
    """Handle the download command."""
    if not HAS_DOWNLOAD:
        LOGGER.error("Download functionality not available. Install requests and tqdm.")
        return 1

    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return 1

    registry = load_data_sources(args.registry) if hasattr(args, 'registry') and args.registry else load_data_sources()

    from ..download import download_sources
    download_sources(
        args.sources,
        registry,
        args.data_dirs,
        force=args.force,
        username=getattr(args, 'username', None),
        password=getattr(args, 'password', None),
    )
    return 0


def handle_import(args) -> int:
    """Handle the import command."""
    if not HAS_IMPORT:
        LOGGER.error("Import functionality not available.")
        return 1

    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return 1

    registry = load_data_sources(args.registry) if hasattr(args, 'registry') and args.registry else load_data_sources()

    from ..import_data import import_sources
    import_sources(
        args.sources,
        registry,
        args.data_dirs,
        force=args.force,
        cleanup=getattr(args, 'cleanup', False),
    )
    return 0


def handle_setup(args) -> int:
    """Handle the setup command."""
    if not (HAS_DOWNLOAD and HAS_IMPORT and HAS_ATTACH):
        LOGGER.error("Setup functionality requires all modules installed.")
        return 1

    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return 1

    registry = load_data_sources()

    for source_name, data_dir in zip(args.sources, args.data_dirs):
        try:
            setup_src_data(source_name, registry, Path(data_dir), force=args.force)
        except Exception as e:
            LOGGER.error(f"Failed to setup {source_name}: {e}")
            return 1

    return 0


def handle_extract(args) -> int:
    """Handle the extract command."""
    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return 1

    registry = load_data_sources(args.registry) if args.registry else load_data_sources()
    dictionary = load_dictionary(args.dictionary) if args.dictionary else load_dictionary()

    resolver = ConceptResolver(dictionary)
    groups = load_groups(args.groups_json) if hasattr(args, 'groups_json') else load_groups(None)

    for source_name, dir_name in zip(args.sources, args.data_dirs):
        data_dir = Path(dir_name)
        if not data_dir.exists():
            LOGGER.error("Data directory %s does not exist; skipping %s", data_dir, source_name)
            continue

        LOGGER.info("Processing %s @ %s", source_name, data_dir)
        try:
            datasource = resolve_datasource(registry, source_name, data_dir)
        except KeyError:
            LOGGER.warning("No data-source configuration named '%s'; skipping.", source_name)
            continue

        frames = extract_groups(resolver, datasource, groups)
        if frames:
            export_groups(frames, args.output, prefix=source_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
