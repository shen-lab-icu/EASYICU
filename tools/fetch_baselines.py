#!/usr/bin/env python
"""Fetch one or more research-agent baselines for local A/B.

The baseline registry lives in ``baselines/REGISTRY.md`` with a
machine-readable YAML block at the end. This script reads that block,
shallow-clones the requested baselines into ``baselines/_checkouts/``,
and pins to the configured ref so an A/B comparison always runs the
same commit.

Why not vendor:
* Licensing — every baseline has its own license; vendoring would
  require per-file audit.
* Size — the science-agent landscape is >10 GB if vendored.
* Freshness — pinning by ref avoids stale local forks.

Usage::

    # List entries
    python tools/fetch_baselines.py --list

    # Fetch one
    python tools/fetch_baselines.py --name healthflow

    # Fetch everything in a category
    python tools/fetch_baselines.py --category discovery-bench

    # Fetch all
    python tools/fetch_baselines.py --all

No third-party dependencies — we parse the YAML block with a tiny
handwritten reader, and call ``git`` through subprocess.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "baselines" / "REGISTRY.md"
CHECKOUT_DIR = REPO_ROOT / "baselines" / "_checkouts"


@dataclass
class BaselineEntry:
    name: str
    repo: str
    ref: str
    category: str
    axis: List[str]


# ---------------------------------------------------------------------------
# Tiny YAML reader (just enough for REGISTRY.md's flat block)
# ---------------------------------------------------------------------------


_YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.S)


def _parse_registry_yaml(text: str) -> List[BaselineEntry]:
    """Parse the `entries:` list from the YAML block in REGISTRY.md.

    We only support the exact shape the registry uses (a flat list of
    maps with simple scalar / list-of-string values). A deliberate
    non-dependency so this script runs on a fresh checkout.
    """
    m = _YAML_BLOCK_RE.search(text)
    if not m:
        raise RuntimeError("No ```yaml block found in REGISTRY.md")
    body = m.group(1)
    entries: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    current_indent = 0
    last_list_key: Optional[str] = None
    for raw_line in body.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        stripped = raw_line.lstrip()
        indent = len(raw_line) - len(stripped)
        if stripped.startswith("- "):
            current = {}
            entries.append(current)
            current_indent = indent
            last_list_key = None
            kv = stripped[2:].split(":", 1)
            if len(kv) == 2:
                key, val = kv[0].strip(), kv[1].strip()
                current[key] = _parse_scalar(val)
            continue
        if current is None:
            # top-level key ("entries:")
            continue
        if ":" not in stripped:
            if last_list_key and stripped.startswith("-"):
                value = stripped.lstrip("- ").strip()
                current.setdefault(last_list_key, []).append(_parse_scalar(value))
            continue
        key, _, raw_val = stripped.partition(":")
        key = key.strip()
        val = raw_val.strip()
        if val == "":
            # following lines under this key form a list
            last_list_key = key
            current[key] = []
            continue
        last_list_key = None
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            items = [_parse_scalar(p.strip()) for p in inner.split(",") if p.strip()]
            current[key] = items
        else:
            current[key] = _parse_scalar(val)
    parsed: List[BaselineEntry] = []
    for e in entries:
        try:
            parsed.append(
                BaselineEntry(
                    name=str(e["name"]),
                    repo=str(e["repo"]),
                    ref=str(e.get("ref", "main")),
                    category=str(e.get("category", "")),
                    axis=list(e.get("axis", []) or []),
                )
            )
        except KeyError as exc:
            print(f"Skipping malformed entry: {e!r} ({exc})", file=sys.stderr)
    return parsed


def _parse_scalar(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    return raw


def load_registry() -> List[BaselineEntry]:
    if not REGISTRY_PATH.exists():
        raise SystemExit(f"REGISTRY.md not found at {REGISTRY_PATH}")
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    return _parse_registry_yaml(text)


# ---------------------------------------------------------------------------
# Git fetch
# ---------------------------------------------------------------------------


def _run(cmd: List[str], *, cwd: Path) -> int:
    print(f"[baselines] $ {' '.join(cmd)}  (cwd={cwd})")
    return subprocess.call(cmd, cwd=str(cwd))


def fetch_one(entry: BaselineEntry, *, force: bool = False) -> Path:
    CHECKOUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKOUT_DIR / entry.name
    if dest.exists():
        if not force:
            print(f"[baselines] {entry.name} already present at {dest}. "
                  "Use --force to re-fetch.")
            return dest
        shutil.rmtree(dest)
    rc = _run(
        ["git", "clone", "--depth", "1", "--branch", entry.ref, entry.repo, entry.name],
        cwd=CHECKOUT_DIR,
    )
    if rc != 0:
        print(
            f"[baselines] shallow clone with branch {entry.ref!r} failed; "
            "retrying as full clone.",
            file=sys.stderr,
        )
        rc = _run(["git", "clone", entry.repo, entry.name], cwd=CHECKOUT_DIR)
        if rc == 0 and entry.ref not in {"", "main", "master"}:
            _run(["git", "checkout", entry.ref], cwd=dest)
    if rc != 0:
        raise SystemExit(
            f"Failed to clone {entry.name} from {entry.repo}; "
            "check network access."
        )
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_entries(entries: Iterable[BaselineEntry]) -> None:
    for e in entries:
        print(
            f"- {e.name:<26} [{e.category:<16}] axes={','.join(e.axis)}  →  {e.repo}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="List all entries and exit.")
    parser.add_argument("--name", help="Fetch a single entry by name.")
    parser.add_argument("--category", help="Fetch every entry in a category.")
    parser.add_argument("--axis", help="Fetch every entry tagged with this axis.")
    parser.add_argument("--all", action="store_true", help="Fetch every entry.")
    parser.add_argument("--force", action="store_true", help="Re-clone if present.")
    args = parser.parse_args()

    entries = load_registry()

    if args.list:
        _list_entries(entries)
        return 0

    selected: List[BaselineEntry] = []
    if args.all:
        selected = entries
    elif args.category:
        selected = [e for e in entries if e.category == args.category]
    elif args.axis:
        selected = [e for e in entries if args.axis in e.axis]
    elif args.name:
        selected = [e for e in entries if e.name == args.name]
        if not selected:
            print(f"No baseline named {args.name!r}. Use --list.", file=sys.stderr)
            return 2

    if not selected:
        print("Nothing to do. Pass --list / --name / --category / --axis / --all.",
              file=sys.stderr)
        return 2

    for e in selected:
        fetch_one(e, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
