#!/usr/bin/env python3
"""Report functions that implement the same rule more than once.

Why this exists
---------------
An ad-hoc version of this scan was used to argue "duplication is only 0.26% of
the module". That number was not reproducible and, worse, the metric behind it
conflated *structure* with *meaning* in both directions:

* **False positives.** The normaliser mapped every Name, Attribute and Constant
  to a placeholder, so two functions with the same shape but different literals
  hashed equal. A gate keyed on ``"reason_a"`` and a gate keyed on ``"reason_b"``
  are not duplicates.
* **False negatives, the expensive kind.** Only byte-for-byte structural twins
  were found. The immutable-receipt write-once rule was implemented six times
  and the scan found two, because the other four spelled the same algorithm
  slightly differently. The rule with the most copies was the one the scan was
  worst at seeing.

So this tool reports two separate things and never adds them up:

``--mode structural`` (default)
    Exact structural twins, with constants and attribute names PRESERVED. This
    is high-precision copy-paste detection. A hit is almost always real.

``--mode shingle``
    Near-duplicates by token-shingle Jaccard similarity over the normalised
    body. This catches "same rule, different spelling". It is high-recall and
    low-precision: read every hit before believing it.

Neither mode can find a rule re-expressed with a different algorithm. A low
number here is evidence about copy-paste, not a certificate that the module has
no redundancy. Do not quote it as one.

Usage
-----
    python tools/find_duplicate_logic.py                       # structural
    python tools/find_duplicate_logic.py --mode shingle --min-similarity 0.85
    python tools/find_duplicate_logic.py --root src/easyicu/research_agent
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import itertools
import pathlib
import sys
from typing import Iterable, Iterator, NamedTuple

DEFAULT_ROOT = "src/easyicu/research_agent"
DEFAULT_MIN_LINES = 12
DEFAULT_MIN_SIMILARITY = 0.85
SHINGLE_SIZE = 5


class Fn(NamedTuple):
    path: str
    line: int
    name: str
    lines: int
    structure: str
    tokens: tuple[str, ...]


class _Shape(ast.NodeTransformer):
    """Erase identifiers but KEEP constants and attribute names.

    Parameter and local names are noise for copy-paste detection; a literal is
    not. ``raise FooError("a")`` and ``raise FooError("b")`` are two rules, and
    the earlier version of this scan could not tell them apart.
    """

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="_", annotation=None), node)


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _tokens(node: ast.AST) -> tuple[str, ...]:
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant):
            out.append(f"const:{child.value!r}")
        elif isinstance(child, ast.Attribute):
            out.append(f"attr:{child.attr}")
        elif isinstance(child, ast.Call):
            out.append("call")
        elif isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
            out.append(type(child).__name__.lower())
        elif isinstance(child, ast.Raise):
            out.append("raise")
    return tuple(out)


def _shingles(tokens: Iterable[str], size: int = SHINGLE_SIZE) -> frozenset[str]:
    seq = list(tokens)
    if len(seq) < size:
        return frozenset({"|".join(seq)}) if seq else frozenset()
    return frozenset(
        "|".join(seq[i : i + size]) for i in range(len(seq) - size + 1)
    )


def collect(root: pathlib.Path, min_lines: int) -> Iterator[Fn]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            if span < min_lines:
                continue
            body = ast.Module(body=_strip_docstring(node.body), type_ignores=[])
            try:
                shaped = _Shape().visit(body)
            except RecursionError:
                continue
            yield Fn(
                path=str(path),
                line=node.lineno,
                name=node.name,
                lines=span,
                structure=hashlib.sha256(ast.dump(shaped).encode()).hexdigest(),
                tokens=_tokens(body),
            )


def report_structural(fns: list[Fn]) -> int:
    groups = collections.defaultdict(list)
    for fn in fns:
        groups[fn.structure].append(fn)
    dups = [g for g in groups.values() if len(g) > 1]
    dups.sort(key=lambda g: -sum(f.lines for f in g[1:]))
    redundant = sum(sum(f.lines for f in g[1:]) for g in dups)
    print(f"exact structural twins (constants preserved): {len(dups)} groups, "
          f"{sum(len(g) for g in dups)} functions, {redundant} redundant lines")
    print("NOTE: this is copy-paste only. Same rule / different spelling is "
          "invisible here -- use --mode shingle.\n")
    for group in dups:
        print(f"  {len(group)}x {group[0].lines:4d} lines")
        for fn in group:
            print(f"        {fn.path}:{fn.line}  {fn.name}")
    return len(dups)


def report_shingle(fns: list[Fn], min_similarity: float) -> int:
    prepared = [(fn, _shingles(fn.tokens)) for fn in fns]
    prepared = [(fn, sh) for fn, sh in prepared if sh]
    hits = []
    for (a, sa), (b, sb) in itertools.combinations(prepared, 2):
        if a.structure == b.structure:
            continue  # already reported as an exact twin
        union = len(sa | sb)
        if not union:
            continue
        score = len(sa & sb) / union
        if score >= min_similarity:
            hits.append((score, a, b))
    hits.sort(key=lambda row: -row[0])
    print(f"near-duplicates at Jaccard >= {min_similarity}: {len(hits)} pairs")
    print("NOTE: high recall, low precision. Read each one; a shared idiom "
          "(guard + fsync + link) is not automatically a shared rule.\n")
    for score, a, b in hits:
        print(f"  {score:.2f}  {a.name} ({a.lines}L)  <->  {b.name} ({b.lines}L)")
        print(f"        {a.path}:{a.line}")
        print(f"        {b.path}:{b.line}")
    return len(hits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--mode", choices=("structural", "shingle"),
                        default="structural")
    parser.add_argument("--min-lines", type=int, default=DEFAULT_MIN_LINES)
    parser.add_argument("--min-similarity", type=float,
                        default=DEFAULT_MIN_SIMILARITY)
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    fns = list(collect(root, args.min_lines))
    print(f"scanned {root}: {len(fns)} functions >= {args.min_lines} lines\n")
    if args.mode == "structural":
        report_structural(fns)
    else:
        report_shingle(fns, args.min_similarity)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
