#!/usr/bin/env python3
"""Regenerate the static web catalog data blocks from the Python SSOT.

The frontend bootstraps ``window.EU_CATALOG`` from
``webserver/static/js/data-catalog.js`` before ``api.js`` fetches the live
``/api/catalog`` (``webserver.catalog.build_catalog``).  Both must describe the
*same* feature set, but the static file was hand-transcribed and drifted
(missing 60 concepts as of 2026-07).  This script rewrites the four data blocks
(``groupConcepts``, ``dict``, ``cov``, ``desc``) directly from
``easyicu.concept.catalog`` so the bootstrap can never fall behind the backend.

The surrounding hand-written parts of the file (the ``groups`` display list,
``auditModules``, the coverage-computation JS, the IIFE wrapper) are left
untouched.  ``tests/test_concept_catalog_consistency.py`` locks the result:
if Python gains a concept and this script isn't re-run, the test fails.

Usage:
    python tools/generate_static_catalog.py           # rewrite in place
    python tools/generate_static_catalog.py --check    # exit 1 if out of sync
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from easyicu.concept import catalog as C  # noqa: E402

DATA_CATALOG_JS = (
    REPO_ROOT / "src" / "easyicu" / "webserver" / "static" / "js" / "data-catalog.js"
)

# Blocks this generator owns. Everything else in the file is hand-maintained.
_MANAGED_VARS = ("groupConcepts", "dict", "cov", "desc")


def _js_str(value: str) -> str:
    # Double-quoted JS string literal; json handles escaping of " and \.
    return json.dumps(value, ensure_ascii=False)


def _js_list(values) -> str:
    return "[" + ", ".join(_js_str(str(v)) for v in values) + "]"


def render_group_concepts() -> str:
    lines = ["const groupConcepts = {"]
    for module, concepts in C.CONCEPT_GROUPS_INTERNAL.items():
        lines.append(f"    {module}: {_js_list(concepts)},")
    lines.append("  }")
    return "\n".join(lines)


def render_dict() -> str:
    lines = ["const dict = {"]
    for key, meta in C.CONCEPT_DICTIONARY.items():
        lines.append(f"    {key}: {_js_list(meta)},")
    lines.append("  }")
    return "\n".join(lines)


def render_cov() -> str:
    lines = ["const cov = {"]
    for key, n in C.CONCEPT_DB_COVERAGE.items():
        lines.append(f"    {key}: {int(n)},")
    lines.append("  }")
    return "\n".join(lines)


def render_desc() -> str:
    lines = ["const desc = {"]
    for key, meta in C.CONCEPT_DESCRIPTIONS.items():
        lines.append(f"    {key}: {_js_list(meta)},")
    lines.append("  }")
    return "\n".join(lines)


_RENDERERS = {
    "groupConcepts": render_group_concepts,
    "dict": render_dict,
    "cov": render_cov,
    "desc": render_desc,
}


def _replace_block(source: str, var: str, replacement: str) -> str:
    """Replace ``const <var> = { ... };`` matching braces (no nested {} inside)."""
    marker = f"const {var} = {{"
    start = source.index(marker)
    brace_open = source.index("{", start)
    depth = 0
    i = brace_open
    while i < len(source):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # i points at the matching close brace; consume any trailing ';' run
    end = i + 1
    while source[end : end + 1] == ";":
        end += 1
    return source[:start] + replacement + ";" + source[end:]


def build(source: str) -> str:
    out = source
    for var in _MANAGED_VARS:
        out = _replace_block(out, var, _RENDERERS[var]())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if out of sync")
    args = ap.parse_args()

    current = DATA_CATALOG_JS.read_text(encoding="utf-8")
    updated = build(current)

    if args.check:
        if current != updated:
            print(
                "data-catalog.js is OUT OF SYNC with the Python catalog. "
                "Run: python tools/generate_static_catalog.py",
                file=sys.stderr,
            )
            return 1
        print("data-catalog.js is in sync with the Python catalog. ✅")
        return 0

    if current == updated:
        print("data-catalog.js already in sync — no change.")
        return 0
    DATA_CATALOG_JS.write_text(updated, encoding="utf-8")
    print(f"Rewrote {DATA_CATALOG_JS.relative_to(REPO_ROOT)} from the Python catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
