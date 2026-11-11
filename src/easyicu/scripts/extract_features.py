"""Command-line helper to extract ICU feature tables from local datasets.

Groups clinical concepts into thematic tables and exports them as CSV files.
It relies on the packaged concept dictionary and data-source registry
provided by :mod:`easyicu`.

Exit codes (consistent across extract, import, setup and download):

* ``0``: success.
* ``2``: user-input error: bad or mismatched arguments, an unknown, missing or
  unprepared data directory, a credentialed or unsupported download source,
  or an extraction the engine refuses to answer (the same code argparse uses
  for usage errors).
* ``1``: the operation itself failed (network or conversion failure).
* ``3``: ``--input-snapshot`` found changed inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from easyicu import ConceptDictionary, ConceptResolver, DataSourceRegistry, ICUDataSource
from easyicu.resources import load_data_sources, load_dictionary

LOGGER = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INPUT_CHANGED = 3

# Import optional modules (absolute easyicu.io paths; this file lives in the
# namespace package easyicu.scripts, so relative "..download" would resolve to
# a non-existent easyicu.download and silently disable these commands).
try:
    from easyicu.io.download import download_src
    HAS_DOWNLOAD = True
except ImportError:
    HAS_DOWNLOAD = False

try:
    from easyicu.io.import_data import import_src
    HAS_IMPORT = True
except ImportError:
    HAS_IMPORT = False

try:
    from easyicu.io.attach import attach_src, data, setup_src_data
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


def _known_input_errors() -> tuple:
    from easyicu.concept.errors import ConceptError
    from easyicu.databases.detection import DatabaseDetectionError

    return (DatabaseDetectionError, ConceptError)


KNOWN_INPUT_ERRORS = _known_input_errors()


def extract_groups(
    resolver: ConceptResolver,
    datasource: ICUDataSource,
    groups: Mapping[str, Iterable[str]],
) -> Dict[str, pd.DataFrame]:
    """Resolve grouped concepts for a single data source."""

    results: Dict[str, pd.DataFrame] = {}
    for group_name, concepts in groups.items():
        try:
            LOGGER.info("Loading concept group '%s': %s", group_name, ", ".join(concepts))
            frame = resolver.load_concepts(concepts, datasource)
            results[group_name] = frame
        except NotImplementedError as exc:
            LOGGER.warning("Concept group '%s' skipped (%s)", group_name, exc)
        except KNOWN_INPUT_ERRORS:
            # Reported once, as a single line, by the CLI; no traceback here.
            raise
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

    from easyicu.databases.detection import DatabaseDetectionError

    if not _has_table_files(data_dir):
        raise DatabaseDetectionError(
            "no_prepared_tables",
            f"No prepared database found in {data_dir}: run the converter "
            f"first (DataConverter(raw_dir, database='{source_name}')"
            ".convert_all())",
            data_path=data_dir,
        )
    config = registry.get(source_name)
    datasource = ICUDataSource(config, base_path=data_dir)
    return datasource


def _has_table_files(data_dir: Path) -> bool:
    """True when the directory holds any parquet/CSV table file."""
    data_dir = Path(data_dir)
    patterns = ("*.parquet", "*.csv", "*.csv.gz")
    for pattern in patterns:
        if next(data_dir.rglob(pattern), None) is not None:
            return True
    return False


INPUT_SNAPSHOT_HELP = (
    "Check that the listed input files are unchanged before and after the run "
    "(INPUT_SNAPSHOT.json and INPUT_SNAPSHOT_POST.json in --output; exit 3 if any "
    "changed). It does NOT publish a consumed set or a RUN_RECEIPT, and it does NOT "
    "prove that every file the run read was in the snapshot. The formal guarantee "
    "comes only from the governed full-workflow run with its RUN_RECEIPT."
)


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
    extract_parser.add_argument(
        "--input-snapshot",
        action="store_true",
        help=INPUT_SNAPSHOT_HELP,
    )

    # Download command
    if HAS_DOWNLOAD:
        download_parser = subparsers.add_parser(
            "download",
            help="Download the public demo datasets (mimic_demo, eicu_demo); credentialed databases must be downloaded manually",
        )
        download_parser.add_argument(
            "--sources",
            nargs="+",
            required=True,
            help="Public demo sources to download (mimic_demo, eicu_demo)",
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
            help="Complete setup: download (public demos only), import, and attach",
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
        return EXIT_USAGE

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
        # Lazy webserver import kept intentionally: this core CLI must import
        # without pulling the webserver stack; the copilot subcommand is the
        # only path that needs it.
        from easyicu.webserver.pi_copilot.install import install_runtime

        installed = install_runtime(destination=args.runtime_dir)
        LOGGER.info("Installed pinned Pi Copilot runtime at %s", installed)
        return 0
    else:
        parser.print_help()
        return EXIT_USAGE


def handle_download(args) -> int:
    """Handle the download command."""
    if not HAS_DOWNLOAD:
        LOGGER.error("Download functionality not available. Install requests and tqdm.")
        return EXIT_FAILURE

    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return EXIT_USAGE

    registry = load_data_sources(args.registry) if hasattr(args, 'registry') and args.registry else load_data_sources()

    from easyicu.io.download import (
        DownloadError,
        SourceNotDownloadableError,
        download_sources,
    )
    try:
        download_sources(
            args.sources,
            registry,
            args.data_dirs,
            force=args.force,
            username=getattr(args, 'username', None),
            password=getattr(args, 'password', None),
        )
    except SourceNotDownloadableError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    except KeyError as exc:
        LOGGER.error("No data-source configuration named %s", exc)
        return EXIT_USAGE
    except DownloadError as exc:
        LOGGER.error("Download failed: %s", exc)
        return EXIT_FAILURE
    return EXIT_OK


def handle_import(args) -> int:
    """Handle the import command."""
    if not HAS_IMPORT:
        LOGGER.error("Import functionality not available.")
        return EXIT_FAILURE

    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return EXIT_USAGE

    registry = load_data_sources(args.registry) if hasattr(args, 'registry') and args.registry else load_data_sources()

    from easyicu.io.import_data import TableImportError, import_sources
    try:
        import_sources(
            args.sources,
            registry,
            args.data_dirs,
            force=args.force,
            cleanup=getattr(args, 'cleanup', False),
        )
    except TableImportError as e:
        for name, err in e.failed.items():
            directory = dict(zip(args.sources, args.data_dirs)).get(name, "")
            if name in e.missing:
                LOGGER.error(
                    "%d tables missing under %s; is this the prepared directory?",
                    e.missing[name],
                    directory,
                )
            else:
                LOGGER.error("Failed to import %s: %s", name, err)
        return EXIT_USAGE if e.failed and set(e.failed) == set(e.missing) else EXIT_FAILURE
    return EXIT_OK


def handle_setup(args) -> int:
    """Handle the setup command."""
    if not (HAS_DOWNLOAD and HAS_IMPORT and HAS_ATTACH):
        LOGGER.error("Setup functionality requires all modules installed.")
        return EXIT_FAILURE

    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return EXIT_USAGE

    registry = load_data_sources()

    from easyicu.io.download import SourceNotDownloadableError
    from easyicu.io.import_data import TableImportError

    for source_name, data_dir in zip(args.sources, args.data_dirs):
        try:
            setup_src_data(source_name, registry, Path(data_dir), force=args.force)
        except SourceNotDownloadableError as e:
            LOGGER.error("%s", e)
            return EXIT_USAGE
        except KeyError as e:
            LOGGER.error("No data-source configuration named %s", e)
            return EXIT_USAGE
        except TableImportError as e:
            if e.missing:
                LOGGER.error(
                    "%d tables missing under %s; is this the prepared directory?",
                    sum(e.missing.values()),
                    data_dir,
                )
                return EXIT_USAGE
            LOGGER.error("Failed to set up %s: %s", source_name, e)
            return EXIT_FAILURE
        except Exception as e:
            LOGGER.error(f"Failed to setup {source_name}: {e}")
            return EXIT_FAILURE

    return EXIT_OK


def handle_extract(args) -> int:
    """Handle the extract command."""
    if len(args.sources) != len(args.data_dirs):
        LOGGER.error("Number of --sources must match number of --data-dirs.")
        return EXIT_USAGE

    try:
        registry = load_data_sources(args.registry) if args.registry else load_data_sources()
        dictionary = load_dictionary(args.dictionary) if args.dictionary else load_dictionary()
        groups = load_groups(args.groups_json) if hasattr(args, 'groups_json') else load_groups(None)
    except (OSError, ValueError) as exc:
        LOGGER.error("Cannot read the extract configuration: %s", exc)
        return EXIT_USAGE

    resolver = ConceptResolver(dictionary)

    snapshot = None
    if getattr(args, "input_snapshot", False):
        from ..formal_inputs import InputSnapshotError, take_snapshot, write_snapshot

        args.output.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = take_snapshot(
                f"{name}:{directory}" for name, directory in zip(args.sources, args.data_dirs)
            )
            write_snapshot(args.output / "INPUT_SNAPSHOT.json", snapshot)
        except InputSnapshotError as error:
            LOGGER.error("Input snapshot refused: %s", error)
            return EXIT_INPUT_CHANGED

    # S778: a requested source that is skipped or yields no rows makes the run
    # fail (exit 2) instead of reporting success with nothing written for it.
    incomplete = []
    for source_name, dir_name in zip(args.sources, args.data_dirs):
        data_dir = Path(dir_name)
        if not data_dir.exists():
            LOGGER.error("Data directory %s does not exist; skipping %s", data_dir, source_name)
            incomplete.append(f"{source_name}: data directory does not exist")
            continue

        try:
            datasource = resolve_datasource(registry, source_name, data_dir)
        except KeyError:
            LOGGER.error("No data-source configuration named '%s'; skipping.", source_name)
            incomplete.append(f"{source_name}: no data-source configuration")
            continue
        except KNOWN_INPUT_ERRORS as exc:
            LOGGER.error("%s", exc)
            incomplete.append(f"{source_name}: {exc}")
            continue

        LOGGER.info("Processing %s @ %s", source_name, data_dir)
        try:
            frames = extract_groups(resolver, datasource, groups)
        except KNOWN_INPUT_ERRORS as exc:
            LOGGER.error("Extraction refused for %s in %s: %s", source_name, data_dir, exc)
            incomplete.append(f"{source_name}: {exc}")
            continue
        if not frames or all(frame is None or len(frame) == 0 for frame in frames.values()):
            LOGGER.error("No concept rows extracted for %s; nothing written for it.", source_name)
            incomplete.append(f"{source_name}: no concept rows")
            continue
        export_groups(frames, args.output, prefix=source_name)

    if snapshot is not None:
        from ..formal_inputs import InputSnapshotError, verify_snapshot, write_snapshot

        try:
            write_snapshot(args.output / "INPUT_SNAPSHOT_POST.json", verify_snapshot(snapshot))
        except InputSnapshotError as error:
            LOGGER.error("Inputs changed during the run; the outputs are not valid: %s", error)
            return EXIT_INPUT_CHANGED
    # Every incomplete source has already been reported by its own error line.
    if incomplete:
        return EXIT_USAGE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
