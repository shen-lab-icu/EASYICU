"""Detect accidental source-access drift in the frozen vocabulary set.

The runtime read guard and Rule 10 enforce source access beyond these checks.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
import runpy
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.util import resolve_name
from os import fsdecode
from pathlib import Path, PurePosixPath

import pytest

from tools import benchmark_question_sources
from tools.benchmark_question_sources import (
    HeldOutSourceRefused,
    require_allowed_question_source,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_MARKERS = ("heldout", "ho30")
_REFERENCE_PATTERN = "|".join(_SOURCE_MARKERS)
_EXCLUDES = (
    *(f":(exclude,glob,icase)**/*{marker}*" for marker in _SOURCE_MARKERS),
    ":!*_restricted*",
    ":!*D3.3*",
    ":!*D3.4*",
    ":!*S166b*",
)
_REVIEWED_DATA_PATHS = frozenset(
    {
        f"benchmarks/figure2_icu_agent_v2/{_SOURCE_MARKERS[0]}27_evaluation_rubric_v1.json",
        f"benchmarks/figure2_icu_agent_v2/{_SOURCE_MARKERS[0]}27_taskbank_v1.jsonl",
    }
)
_VOCABULARY_MANIFEST = "tests/governance/benchmark_vocabulary_only.tsv"
# These files define the guard, exercise it, or record the frozen review.
_GUARD_FILES = frozenset(
    {
        "tests/governance/test_held_out_sources_stay_closed.py",
        "tools/benchmark_question_sources.py",
        _VOCABULARY_MANIFEST,
    }
)
_REVIEWED_OWNERS = frozenset(
    {
        "benchmarks/figure2_icu_agent_v2/README.md",
        "benchmarks/figure2_icu_agent_v2/__init__.py",
        "benchmarks/figure2_icu_agent_v2/design_freeze_manifest_v2.json",
        "benchmarks/figure2_icu_agent_v2/experiment_protocol_v1.json",
        "benchmarks/figure2_icu_agent_v2/experiment_protocol_v2.json",
        "benchmarks/figure2_icu_agent_v2/freeze_v2.py",
        "benchmarks/figure2_icu_agent_v2/protocol.py",
        "benchmarks/figure2_icu_agent_v2/readiness.py",
        "tests/benchmarks/figure2_icu_agent_v2/test_freeze_v2.py",
        "tests/benchmarks/figure2_icu_agent_v2/test_protocol.py",
        "tests/benchmarks/figure2_icu_agent_v2/test_readiness.py",
        "tests/research_agent/core/test_supplement_inventory.py",
        "tests/research_agent/core/test_validity_signals.py",
        "tests/research_agent/gates/test_prediction_metric_integrity.py",
        "tests/research_agent/test_eval_harness_track4.py",
        "tools/import_canonical9_to_agent_projects.py",
    }
)
_EXPECTED_VOCABULARY = {
    "benchmarks/figure2_icu_agent_v2/data_platform_validation_protocol_v1.json": "VOCAB_ONLY",
    "benchmarks/figure2_icu_agent_v2/statistical_analysis_plan_v1.json": "VOCAB_ONLY",
    "docs/INDEPENDENT_EVAL_PROTOCOL.md": "VOCAB_ONLY",
    "docs/evidence/authority_followup_20260904/figure_exports/synthetic_sources/model_performance.csv": "VOCAB_ONLY",
    "src/easyicu/research_agent/__init__.py": "VOCAB_ONLY",
    "src/easyicu/research_agent/evaluation/independent_eval_harness.py": "VOCAB_ONLY",
    "src/easyicu/research_agent/figures/prediction.py": "ORDINARY_IO_NO_HO",
    "src/easyicu/research_agent/reporting/supplement_inventory.py": "ORDINARY_IO_NO_HO",
    "src/easyicu/research_agent/validity_signals.py": "ORDINARY_IO_NO_HO",
    "tools/arch_baselines/execution_phase.json": "VOCAB_ONLY",
    "tools/verify_figure_presentation.py": "ORDINARY_IO_NO_HO",
}
_PREREGISTRATION_BASE = "ef3472d8e1f4f32ed1c76a73eacf591d442ceb09"
_PREREGISTRATION_PATHS = (
    "benchmarks/figure2_icu_agent_v2/data_platform_validation_protocol_v1.json",
    "benchmarks/figure2_icu_agent_v2/statistical_analysis_plan_v1.json",
)
_BENCHMARK_DIRECTORY = "figure2_icu_agent_v2"
_DATA_FILENAMES = {PurePosixPath(path).name for path in _REVIEWED_DATA_PATHS}
_TEXT_TOKEN = re.compile(r"""[^\s"'`<>{}\[\](),;=|]+""")
_FILE_EXTENSION = re.compile(r"\.[a-z0-9]+$", re.IGNORECASE)
_PATH_TYPES = {
    "Path",
    "PosixPath",
    "WindowsPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
}
_IO_METHODS = {"open", "read_text", "read_bytes", "glob", "rglob", "iterdir"}
_IO_FUNCTIONS = {"os.listdir", "os.scandir", "json.load", "json.loads"}
_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
_VALUE_TRANSFORMS = {
    "absolute",
    "as_posix",
    "casefold",
    "expanduser",
    "format",
    "fspath",
    "get",
    "join",
    "lower",
    "lstrip",
    "normpath",
    "replace",
    "resolve",
    "rstrip",
    "strip",
    "upper",
    "with_name",
    "with_suffix",
}


@dataclass(frozen=True)
class _VocabularyReview:
    reason: str
    constants: dict[str, object] = field(default_factory=dict)
    literals: dict[str, int] = field(default_factory=dict)


@pytest.mark.parametrize(
    "source",
    [
        f"benchmarks/{_SOURCE_MARKERS[0]}27.jsonl",
        f"tests/fixtures/{_SOURCE_MARKERS[0].title()}27_sample/questions.jsonl",
        f"tests/fixtures/{_SOURCE_MARKERS[1].upper()}_sample/questions.jsonl",
        f"benchmarks/{_SOURCE_MARKERS[1]}.jsonl",
        f"tests/fixtures/local_{_SOURCE_MARKERS[0].title()}.json",
        f"tests/fixtures/local_{_SOURCE_MARKERS[1]}/questions.json",
        f"drafts/local-{_SOURCE_MARKERS[0]}.json",
    ],
    ids=[
        "held-out-file",
        "held-out-directory",
        "ho-directory",
        "ho-file",
        "embedded-filename",
        "embedded-directory",
        "embedded-unlisted",
    ],
)
def test_question_source_guard_refuses_held_out_paths(source: str) -> None:
    with pytest.raises(HeldOutSourceRefused) as error:
        require_allowed_question_source(source)
    assert error.value.reason_code == "held_out_source_refused"


@pytest.mark.parametrize(
    "source",
    [
        "benchmarks/meta_generalization/unlisted.jsonl",
        "tests/../benchmarks/meta_generalization/unlisted.jsonl",
        "other/tests/fixtures/questions.jsonl",
    ],
)
def test_question_source_guard_refuses_unlisted_paths(source: str) -> None:
    with pytest.raises(HeldOutSourceRefused) as error:
        require_allowed_question_source(source)
    assert error.value.reason_code == "question_source_not_allow_listed"


@pytest.mark.parametrize(
    "source",
    [
        "tests/fixtures/hosted_route_inventory.json",
        "benchmarks/figure2_icu_agent_v2/formal_safety12_taskbank_v1.jsonl",
        "benchmarks/meta_generalization/meta_benchmark.jsonl",
        "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json",
    ],
)
def test_question_source_guard_accepts_allow_listed_paths(source: str) -> None:
    require_allowed_question_source(source)
    require_allowed_question_source(str(_REPO_ROOT / source))


def test_question_source_guard_accepts_paths_from_a_symlinked_import(
    tmp_path: Path,
) -> None:
    checkout_alias = tmp_path / "checkout-alias"
    checkout_alias.symlink_to(_REPO_ROOT, target_is_directory=True)
    guard = runpy.run_path(str(checkout_alias / "tools/benchmark_question_sources.py"))[
        "require_allowed_question_source"
    ]
    guard(checkout_alias / "tests/fixtures/synthetic_questions.jsonl")
    guard(_REPO_ROOT / "tests/fixtures/synthetic_questions.jsonl")


@pytest.mark.parametrize("marker", _SOURCE_MARKERS, ids=["held-out", "ho"])
def test_question_source_guard_ignores_checkout_ancestors(
    monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    checkout = _REPO_ROOT.parent / marker / "checkout"
    monkeypatch.setattr(benchmark_question_sources, "_REPO_ROOT", checkout)
    require_allowed_question_source(
        checkout / "tests/fixtures/synthetic_questions.jsonl"
    )


@pytest.mark.parametrize(
    "reference",
    [
        f"drafts/{_SOURCE_MARKERS[0]}6.json",
        f"drafts/{_SOURCE_MARKERS[0].title()}7_entries.json",
        f"drafts/{_SOURCE_MARKERS[1]}.txt",
    ],
    ids=["no-underscore", "mixed-case", "lowercase-ho"],
)
def test_reference_scan_matches_case_and_suffix_variants(
    monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    def run_git(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        if command[1] == "ls-files":
            return subprocess.CompletedProcess(command, 0, stdout=b"")
        assert command[1] == "grep"
        flags = re.IGNORECASE if "-i" in command else 0
        matched = re.search(command[command.index("-E") + 1], reference, flags)
        stdout = b"tools/draft_reader.py\0" if matched else b""
        return subprocess.CompletedProcess(command, 0 if matched else 1, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", run_git)
    with pytest.raises(
        AssertionError, match="Unreviewed held-out source references.*draft_reader"
    ):
        test_only_reviewed_owners_reference_held_out_sources()


@pytest.mark.parametrize(
    "path",
    [
        f"tools/local_{_SOURCE_MARKERS[0]}_reader.py",
        f"scratch/legacy-{_SOURCE_MARKERS[1].upper()}/invoke.sh",
    ],
    ids=["script-filename", "script-parent-directory"],
)
def test_reference_scan_rejects_unreviewed_marker_paths_before_grep(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    def run_git(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[1] == "ls-files", (
            "Content scan ran before rejecting an unexpected path"
        )
        names = (
            b""
            if any(arg.startswith(":(exclude,") for arg in command)
            else path.encode() + b"\0"
        )
        return subprocess.CompletedProcess(command, 0, stdout=names)

    monkeypatch.setattr(subprocess, "run", run_git)
    with pytest.raises(AssertionError, match="Unexpected tracked held-out paths"):
        test_only_reviewed_owners_reference_held_out_sources()


def _git_paths(*args: str, allowed_exit_codes: tuple[int, ...] = (0,)) -> set[str]:
    result = subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True, check=False
    )
    assert result.returncode in allowed_exit_codes, fsdecode(result.stderr)
    return set(fsdecode(result.stdout).split("\0")) - {""}


def test_only_reviewed_owners_reference_held_out_sources() -> None:
    marker = re.compile(_REFERENCE_PATTERN, re.IGNORECASE)
    tracked = _git_paths("ls-files", "-z")
    unexpected = {
        path for path in tracked if marker.search(path)
    } - _REVIEWED_DATA_PATHS
    assert not unexpected, f"Unexpected tracked held-out paths: {sorted(unexpected)}"

    # Validate the same pathspec against the index before scanning file contents.
    scan_paths = _git_paths("ls-files", "-z", "--", ".", *_EXCLUDES)
    assert not any(marker.search(path) for path in scan_paths), (
        "The audit pathspec includes a held-out filename"
    )

    references = _git_paths(
        "grep",
        "-l",
        "-z",
        "-i",
        "-E",
        _REFERENCE_PATTERN,
        "--",
        ".",
        *_EXCLUDES,
        allowed_exit_codes=(0, 1),
    )
    unreviewed = (
        references - _REVIEWED_OWNERS - _GUARD_FILES - _read_vocabulary_review().keys()
    )
    assert not unreviewed, (
        f"Unreviewed held-out source references: {sorted(unreviewed)}"
    )


def _read_source(path: str) -> str:
    source = _REPO_ROOT / path
    assert source.resolve() == source, f"Reviewed path redirects elsewhere: {path}"
    return source.read_text(encoding="utf-8")


def _read_vocabulary_review() -> dict[str, _VocabularyReview]:
    entries = {}
    for line in _read_source(_VOCABULARY_MANIFEST).splitlines():
        if line and not line.startswith("#"):
            reason, path, *vocabulary = line.split("\t")
            assert path not in entries, f"Duplicate vocabulary review entry: {path}"
            assert reason in {"VOCAB_ONLY", "ORDINARY_IO_NO_HO"}
            assert len(vocabulary) <= 1
            entries[path] = _VocabularyReview(
                reason, **(ast.literal_eval(vocabulary[0]) if vocabulary else {})
            )
    return entries


def test_vocabulary_review_is_frozen() -> None:
    entries = _read_vocabulary_review()
    assert {
        path: entry.reason for path, entry in entries.items()
    } == _EXPECTED_VOCABULARY
    assert _REVIEWED_OWNERS.isdisjoint(entries)


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (
                    f"{node.module or ''}.{item.name}".lstrip(".")
                )
    return aliases


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_qualified_name(node.value, aliases)}.{node.attr}"
    return ""


def _is_io_call(name: str) -> bool:
    member = name.rsplit(".", 1)[-1]
    return (
        member in _IO_METHODS
        or name in _IO_FUNCTIONS
        or (name.startswith("pandas.") and member.startswith("read_"))
    )


def _has_source_marker(text: str) -> bool:
    return any(marker in text.casefold() for marker in _SOURCE_MARKERS)


def _path_tokens(text: str) -> list[str]:
    tokens = [
        token.rstrip(".:!?").replace("\\", "/") for token in _TEXT_TOKEN.findall(text)
    ]
    return [token for token in tokens if "/" in token or _FILE_EXTENSION.search(token)]


def _path_components(node: ast.expr, aliases: dict[str, str]) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [node.left, node.right]
    if isinstance(node, ast.Call):
        name = _qualified_name(node.func, aliases)
        if (
            name.rsplit(".", 1)[-1] in _PATH_TYPES | {"joinpath"}
            or name == "os.path.join"
        ):
            return [*node.args, *(keyword.value for keyword in node.keywords)]
    return []


def _assignments(node: ast.AST) -> Iterator[tuple[ast.AST, ast.expr]]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            yield target, node.value
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        if node.value is not None:
            yield node.target, node.value
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        yield node.target, node.iter
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        yield node.optional_vars, node.context_expr


def _parameters(args: ast.arguments) -> Iterator[tuple[str, list[ast.expr]]]:
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    for arg, default in zip(
        positional + args.kwonlyargs, defaults + args.kw_defaults, strict=True
    ):
        yield arg.arg, [] if default is None else [default]
    for arg in (args.vararg, args.kwarg):
        if arg is not None:
            yield arg.arg, []


def _subscript_value(container: ast.expr, key: object) -> ast.expr | None:
    if isinstance(container, (ast.Tuple, ast.List)) and isinstance(key, int):
        if -len(container.elts) <= key < len(container.elts):
            return container.elts[key]
    if isinstance(container, ast.Dict):
        for entry, value in zip(container.keys, container.values, strict=True):
            if isinstance(entry, ast.Constant) and entry.value == key:
                return value
    return None


def _value_inputs(node: ast.AST, aliases: dict[str, str]) -> Iterator[ast.AST]:
    if isinstance(node, ast.Call):
        name = _qualified_name(node.func, aliases)
        member = name.rsplit(".", 1)[-1]
        # Opaque calls may accept vocabulary as data while returning a value
        # unrelated to that argument, such as a record for a fixed filename.
        if (
            _path_components(node, aliases)
            or member in _VALUE_TRANSFORMS
            or name == "str"
        ):
            yield from node.args
            yield from (keyword.value for keyword in node.keywords)
            if isinstance(node.func, ast.Attribute):
                yield node.func.value
    elif isinstance(node, ast.IfExp):
        yield node.body
        yield node.orelse
    elif not isinstance(node, ast.Compare):
        yield from ast.iter_child_nodes(node)


def _documented_vocabulary(
    review: _VocabularyReview,
    expressions: list[ast.expr],
    module_values: dict[str, list[ast.expr]],
) -> set[ast.AST]:
    documented: set[ast.AST] = set()
    for selector, expected in review.constants.items():
        name, _, key = selector.partition(".")
        values = module_values.get(name, [])
        assert len(values) == 1, (
            f"Expected one assignment for vocabulary constant {name}"
        )
        value = values[0]
        if key:
            assert isinstance(value, ast.Dict), (
                f"Expected literal dictionary for {name}"
            )
            matches = [
                child
                for entry, child in zip(value.keys, value.values, strict=True)
                if isinstance(entry, ast.Constant) and entry.value == key
            ]
            assert len(matches) == 1, f"Expected one vocabulary entry for {selector}"
            value = matches[0]
        actual = ast.literal_eval(value)
        assert type(actual) is type(expected) and actual == expected, (
            f"Vocabulary constant changed: {selector}"
        )
        documented.update(
            node for node in ast.walk(value) if isinstance(node, ast.expr)
        )

    inline = {
        node
        for node in expressions
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in review.literals
        and node not in documented
    }
    assert Counter(node.value for node in inline) == review.literals, (
        "Documented inline vocabulary changed"
    )
    return documented | inline


def _module_name(path: str) -> str:
    return (
        path.removeprefix("src/")
        .removesuffix(".py")
        .removesuffix("/__init__")
        .replace("/", ".")
    )


_OWNER_MODULES = frozenset(
    _module_name(path) for path in _REVIEWED_OWNERS if path.endswith(".py")
) | frozenset(
    _module_name(path.removeprefix("tools/"))
    for path in _REVIEWED_OWNERS
    if path.startswith("tools/") and path.endswith(".py")
)
_QUESTION_SOURCE_MODULES = {
    "tools.benchmark_question_sources",
    "benchmark_question_sources",
}
_PATH_ONLY_GUARD_API = {
    "ALLOWED_QUESTION_SOURCES",
    "HeldOutSourceRefused",
    "require_allowed_question_source",
}


def _imports_source_loader(node: ast.Import | ast.ImportFrom, path: str) -> bool:
    if isinstance(node, ast.Import):
        modules = [item.name for item in node.names]
    else:
        base = node.module or ""
        if node.level:
            package = _module_name(path)
            if not path.endswith("/__init__.py"):
                package = package.rpartition(".")[0]
            base = resolve_name("." * node.level + base, package)
        if base in _QUESTION_SOURCE_MODULES:
            return any(item.name not in _PATH_ONLY_GUARD_API for item in node.names)
        modules = [base, *(f"{base}.{item.name}" for item in node.names)]
    forbidden = _OWNER_MODULES | _QUESTION_SOURCE_MODULES
    return any(
        module == owner or module.startswith(f"{owner}.")
        for module in modules
        for owner in forbidden
    )


def _is_bare_marker(value: str) -> bool:
    return _has_source_marker(value) and not any(char.isspace() for char in value)


def _names_benchmark(value: str) -> bool:
    candidates = [value, *_path_tokens(value)]
    return any(
        _BENCHMARK_DIRECTORY
        in PurePosixPath(candidate.casefold().replace("\\", "/")).parts
        for candidate in candidates
    )


def _python_vocabulary_violations(
    source: str, path: str, review: _VocabularyReview
) -> list[str]:
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    expressions = [node for node in nodes if isinstance(node, ast.expr)]
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    scopes = {
        node: _enclosing_scope(node, parents, include_comprehensions=True)
        for node in expressions
    }
    aliases = _import_aliases(tree)
    bindings: dict[ast.AST, dict[str, list[ast.expr]]] = defaultdict(
        lambda: defaultdict(list)
    )
    declarations: dict[ast.AST, dict[str, ast.Global | ast.Nonlocal]] = defaultdict(
        dict
    )
    member_writes: list[tuple[ast.Name, ast.expr]] = []
    for node in nodes:
        if isinstance(node, _FUNCTION_SCOPES):
            bindings[node].update(_parameters(node.args))
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            scope = _enclosing_scope(node, parents)
            declarations[scope].update((name, node) for name in node.names)
        for target, value in _assignments(node):
            for part in ast.walk(target):
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store):
                    bindings[scopes[part]][part.id].append(value)
                elif isinstance(part, (ast.Attribute, ast.Subscript)) and isinstance(
                    part.ctx, ast.Store
                ):
                    root = part.value
                    while isinstance(root, (ast.Attribute, ast.Subscript)):
                        root = root.value
                    if isinstance(root, ast.Name):
                        member_writes.append((root, value))

    def outer_scope(scope: ast.AST) -> ast.AST:
        scope = _enclosing_scope(scope, parents, include_comprehensions=True)
        # Class bodies have local names, but do not form enclosing lexical scopes.
        while isinstance(scope, ast.ClassDef):
            scope = _enclosing_scope(scope, parents, include_comprehensions=True)
        return scope

    def nonlocal_scope(scope: ast.AST, name: str) -> ast.AST:
        enclosing = outer_scope(scope)
        candidate = enclosing
        while candidate is not tree:
            if (
                isinstance(candidate, _FUNCTION_SCOPES)
                and name in bindings[candidate]
                and name not in declarations.get(candidate, {})
            ):
                return candidate
            candidate = outer_scope(candidate)
        return enclosing

    declared_scopes = {
        scope: {
            name: tree if isinstance(node, ast.Global) else nonlocal_scope(scope, name)
            for name, node in names.items()
        }
        for scope, names in declarations.items()
    }
    for scope, names in declared_scopes.items():
        for name, target_scope in names.items():
            if target_scope is not scope and name in bindings[scope]:
                bindings[target_scope][name].extend(bindings[scope].pop(name))

    def binding_scope(name: ast.Name) -> ast.AST:
        scope = scopes[name]
        while scope is not tree:
            declared = declared_scopes.get(scope, {}).get(name.id)
            if declared is not None:
                return declared
            if name.id in bindings[scope]:
                return scope
            scope = outer_scope(scope)
        return scope

    def assigned_value(name: ast.Name) -> ast.expr | None:
        values = bindings[binding_scope(name)].get(name.id, [])
        return values[0] if len(values) == 1 else None

    def resolve_literal(node: ast.expr, active: frozenset[ast.AST]) -> ast.expr | None:
        if node in active:
            return None
        active = active | {node}
        if isinstance(node, ast.Name):
            value = assigned_value(node)
            return resolve_literal(value, active) if value is not None else None
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                container = resolve_literal(node.value, active)
                if container is not None:
                    value = _subscript_value(container, node.slice.value)
                    if value is not None:
                        return resolve_literal(value, active)
            return None
        return node

    def string_value(node: ast.expr, active: frozenset[ast.AST] = frozenset()) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.Name, ast.Subscript)):
            value = resolve_literal(node, active)
            return string_value(value, active | {node}) if value is not None else ""
        active = active | {node}
        if isinstance(node, ast.JoinedStr):
            return "".join(
                (string_value(value.value, active) or "{value}")
                if isinstance(value, ast.FormattedValue)
                else string_value(value, active)
                for value in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = string_value(node.left, active)
            right = string_value(node.right, active)
            if left or right:
                return (left or "{value}") + (right or "{value}")
        return ""

    documented = _documented_vocabulary(review, expressions, bindings[tree])
    literals = {node: string_value(node) for node in expressions}
    path_literals = {
        node
        for node, value in literals.items()
        if any(_has_source_marker(token) for token in _path_tokens(value))
    }
    for node, value in literals.items():
        if (
            node not in documented
            and isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr))
            and _is_bare_marker(value)
        ) or any(
            _has_source_marker(literals[part])
            for part in _path_components(node, aliases)
        ):
            path_literals.add(node)

    # Propagate sources through assignments in the same lexical scope. Keep a
    # binding tainted if any assignment can supply the source, including branches.
    sources = (
        documented
        | path_literals
        | {
            node
            for node, value in literals.items()
            if value and (_has_source_marker(value) or _names_benchmark(value))
        }
    )
    tainted: set[tuple[ast.AST, str]] = set()

    def uses_source(expression: ast.expr) -> bool:
        pending: list[ast.AST] = [expression]
        while pending:
            node = pending.pop()
            if node in sources or (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and (binding_scope(node), node.id) in tainted
            ):
                return True
            pending.extend(_value_inputs(node, aliases))
        return False

    taint_inputs = {
        (scope, name): list(values)
        for scope, names in bindings.items()
        for name, values in names.items()
    }
    # A member store taints its root but does not replace the root's literal value.
    for root, value in member_writes:
        taint_inputs.setdefault((binding_scope(root), root.id), []).append(value)

    while True:
        new = {
            binding
            for binding, values in taint_inputs.items()
            if binding not in tainted and any(uses_source(value) for value in values)
        }
        if not new:
            break
        tainted.update(new)

    violations = {f"line {node.lineno}: path literal" for node in path_literals}
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and _imports_source_loader(
            node, path
        ):
            violations.add(f"line {node.lineno}: owner or loader import")
        if isinstance(node, ast.Call) and _is_io_call(
            _qualified_name(node.func, aliases)
        ):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if isinstance(node.func, ast.Attribute):
                arguments.append(node.func.value)
            if any(uses_source(argument) for argument in arguments):
                violations.add(f"line {node.lineno}: benchmark-derived IO")
    return sorted(violations)


def _json_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_strings(child)


def _text_vocabulary_violations(source: str, suffix: str) -> list[str]:
    strings = _json_strings(json.loads(source)) if suffix == ".json" else (source,)
    for value in strings:
        for token in _path_tokens(value):
            parts = PurePosixPath(posixpath.normpath(token.casefold())).parts
            if not parts:
                continue
            if (
                _BENCHMARK_DIRECTORY in parts
                or parts[-1] in _DATA_FILENAMES
                or ("benchmarks" in parts and _has_source_marker(token))
            ):
                return ["benchmark path literal"]
    return []


def _vocabulary_violations(
    source: str, path: str, review: _VocabularyReview
) -> list[str]:
    suffix = Path(path).suffix
    if suffix == ".py":
        return _python_vocabulary_violations(source, path, review)
    assert suffix in {".md", ".json", ".csv"}
    assert not review.constants and not review.literals
    return _text_vocabulary_violations(source, suffix)


@pytest.mark.parametrize("path", sorted(_EXPECTED_VOCABULARY))
def test_reviewed_vocabulary_has_no_source_access(path: str) -> None:
    violations = _vocabulary_violations(
        _read_source(path), path, _read_vocabulary_review()[path]
    )
    assert not violations, f"{path}: {violations}"


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        ("", "open(file=target)"),
        ("", "target.read_text()"),
        ("", "target.read_bytes()"),
        ("", "root.glob(target)"),
        ("", "root.rglob(target)"),
        ("", "target.iterdir()"),
        ("import os as fs\n", "fs.listdir(target)"),
        ("from os import scandir as scan\n", "scan(target)"),
        ("import pandas as pd\n", "pd.read_csv(target)"),
        ("from pandas import read_parquet as load_frame\n", "load_frame(target)"),
        ("import json as codec\n", "codec.load(target)"),
        ("from json import loads as parse\n", "parse(target)"),
    ],
)
def test_vocabulary_python_rejects_io_built_from_a_documented_token(
    imports: str, call: str
) -> None:
    tokens = (_SOURCE_MARKERS[0], "synthetic")
    source = (
        imports
        + f"TOKENS = {tokens!r}\n"
        + "def read(root):\n"
        + "    alias = TOKENS\n"
        + "    target = alias[0]\n"
        + f"    return {call}\n"
    )
    review = _VocabularyReview("ORDINARY_IO_NO_HO", constants={"TOKENS": tokens})
    assert _vocabulary_violations(source, "tools/example.py", review) == [
        f"line {len(source.splitlines())}: benchmark-derived IO"
    ]


def test_vocabulary_python_rejects_io_built_from_a_benchmark_directory() -> None:
    source = (
        "from pathlib import Path\n"
        f"ROOT = Path('benchmarks') / '{_BENCHMARK_DIRECTORY.upper()}'\n"
        "def read():\n"
        "    path = ROOT / 'synthetic.json'\n"
        "    return path.read_text()\n"
    )
    assert _vocabulary_violations(
        source, "tools/example.py", _VocabularyReview("ORDINARY_IO_NO_HO")
    ) == ["line 5: benchmark-derived IO"]


def test_vocabulary_python_rejects_a_benchmark_directory_default() -> None:
    source = (
        f"ROOT = '{_BENCHMARK_DIRECTORY}'\n"
        "def read(*, directory=ROOT):\n"
        "    return directory.iterdir()\n"
    )
    assert _vocabulary_violations(
        source, "tools/example.py", _VocabularyReview("ORDINARY_IO_NO_HO")
    ) == ["line 3: benchmark-derived IO"]


@pytest.mark.parametrize(
    ("root", "body", "line"),
    [
        (
            "directory",
            "class FixtureReader:\n"
            "    directory = 'staging'\n"
            "    def load(self):\n"
            "        return (directory / 'summary.csv').read_text()",
            5,
        ),
        (
            "directory",
            "class FixtureReader:\n"
            "    directory = 'staging'\n"
            "    rows = [(directory / name).read_text() for name in ['summary.csv']]",
            4,
        ),
        (
            "source_root",
            "directory = Path('staging')\n"
            "def configure():\n"
            "    global directory\n"
            "    directory = source_root\n"
            "def load():\n"
            "    return (directory / 'summary.csv').read_text()",
            7,
        ),
        (
            "source_root",
            "def outer():\n"
            "    directory = Path('staging')\n"
            "    def wrapper():\n"
            "        def configure():\n"
            "            nonlocal directory\n"
            "            directory = source_root\n"
            "    return (directory / 'summary.csv').read_text()",
            8,
        ),
    ],
    ids=["method", "class-comprehension", "global", "nonlocal"],
)
def test_vocabulary_python_resolves_source_bindings_across_scopes(
    root: str, body: str, line: int
) -> None:
    source = f"{root} = Path('benchmarks') / '{_BENCHMARK_DIRECTORY}'\n{body}"
    assert _vocabulary_violations(
        source, "tools/example.py", _VocabularyReview("ORDINARY_IO_NO_HO")
    ) == [f"line {line}: benchmark-derived IO"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            "open(staging_prefix + 'HeLd' + 'OuT_archive.csv')",
            ["path literal", "benchmark-derived IO"],
        ),
        (
            f"(Path('benchmarks') / ' {_SOURCE_MARKERS[1]} '.strip()).read_text()",
            ["benchmark-derived IO"],
        ),
    ],
    ids=["partial-concatenation", "trimmed-marker"],
)
def test_vocabulary_python_rejects_io_with_transformed_markers(
    expression: str, expected: list[str]
) -> None:
    assert _vocabulary_violations(
        expression, "tools/example.py", _VocabularyReview("ORDINARY_IO_NO_HO")
    ) == sorted(f"line 1: {violation}" for violation in expected)


@pytest.mark.parametrize(
    "body",
    [
        "reader.location = source_root / 'summary.csv'\n    return reader.location.read_text()",
        "settings['input'] = source_root / 'summary.csv'\n    return settings['input'].read_text()",
        "location = LABELS.get('staging')\n    return open(location)",
    ],
    ids=["attribute-store", "subscript-store", "dictionary-get"],
)
def test_vocabulary_python_propagates_sources_through_member_access(body: str) -> None:
    labels = {"staging": (_SOURCE_MARKERS[0], "synthetic")}
    source = (
        f"LABELS = {labels!r}\n"
        f"source_root = Path('benchmarks') / '{_BENCHMARK_DIRECTORY}'\n"
        f"def load(reader, settings):\n    {body}"
    )
    review = _VocabularyReview(
        "ORDINARY_IO_NO_HO", constants={"LABELS.staging": labels["staging"]}
    )
    assert _vocabulary_violations(source, "tools/example.py", review) == [
        "line 5: benchmark-derived IO"
    ]


@pytest.mark.parametrize(
    "source",
    [
        f"example = 'fixtures/{_SOURCE_MARKERS[0].title()}_demo.json'",
        f"example = '{_SOURCE_MARKERS[1].upper()}.csv'",
        f"from pathlib import Path as P\nexample = P('fixtures', '{_SOURCE_MARKERS[1]}')",
        f"example = Path('fixtures') / '{_SOURCE_MARKERS[0]}'",
        "PREFIX = 'fixtures/HeLd'\nSUFFIX = 'OuT.json'\nexample = PREFIX + SUFFIX",
    ],
)
def test_vocabulary_python_rejects_path_literals(source: str) -> None:
    violations = _vocabulary_violations(
        source, "tools/example.py", _VocabularyReview("VOCAB_ONLY")
    )
    assert violations and all(
        violation.endswith("path literal") for violation in violations
    )


@pytest.mark.parametrize("nested", [False, True], ids=["tuple", "nested-tuple"])
def test_vocabulary_python_rejects_a_documented_label_used_as_a_path(
    nested: bool,
) -> None:
    tokens = (_SOURCE_MARKERS[1].upper(),)
    if nested:
        source = f"LABELS = {{'staging': {tokens!r}}}\nexample = Path('fixtures') / LABELS['staging'][0]"
        selector = "LABELS.staging"
    else:
        source = f"TOKENS = {tokens!r}\nexample = Path('fixtures') / TOKENS[0]"
        selector = "TOKENS"
    review = _VocabularyReview("VOCAB_ONLY", constants={selector: tokens})
    assert _vocabulary_violations(source, "tools/example.py", review) == [
        "line 2: path literal"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "import benchmarks.figure2_icu_agent_v2.protocol as protocol",
        "from benchmarks import figure2_icu_agent_v2 as benchmark",
        "from tests.research_agent.core import test_validity_signals as owner",
        "from . import import_canonical9_to_agent_projects",
        pytest.param(
            "import import_canonical9_to_agent_projects as staging_loader",
            id="bare-tools-owner",
        ),
        "from tools.benchmark_question_sources import load_questions as load",
        "from .benchmark_question_sources import *",
        "from tools import benchmark_question_sources as sources",
        "import benchmark_question_sources as sources",
    ],
)
def test_vocabulary_python_rejects_owner_and_loader_imports(source: str) -> None:
    assert _vocabulary_violations(
        source, "tools/example.py", _VocabularyReview("VOCAB_ONLY")
    ) == ["line 1: owner or loader import"]


def test_vocabulary_python_allows_ordinary_io_and_keeps_function_scopes_separate() -> (
    None
):
    tokens = (_SOURCE_MARKERS[0],)
    source = (
        "from pathlib import Path\n"
        "from tools.benchmark_question_sources import require_allowed_question_source\n"
        f"TOKENS = {tokens!r}\n"
        "def describes_split(label):\n"
        "    target = TOKENS\n"
        "    return label in target\n"
        "def read_fixture():\n"
        "    target = Path('tests/fixtures/synthetic.json')\n"
        "    require_allowed_question_source(target)\n"
        "    return target.read_text()\n"
    )
    review = _VocabularyReview("ORDINARY_IO_NO_HO", constants={"TOKENS": tokens})
    assert _vocabulary_violations(source, "tools/example.py", review) == []


def test_vocabulary_python_allows_a_documented_label() -> None:
    tokens = (_SOURCE_MARKERS[0],)
    source = f"TOKENS = {tokens!r}\nlabel_matches = 'validation' in TOKENS"
    review = _VocabularyReview("VOCAB_ONLY", constants={"TOKENS": tokens})
    assert _vocabulary_violations(source, "tools/example.py", review) == []


def test_vocabulary_python_rejects_changed_documented_values() -> None:
    review = _VocabularyReview(
        "VOCAB_ONLY", constants={"TOKENS": (_SOURCE_MARKERS[0],)}
    )
    with pytest.raises(AssertionError, match="Vocabulary constant changed: TOKENS"):
        _vocabulary_violations("TOKENS = ('synthetic',)", "tools/example.py", review)


@pytest.mark.parametrize("suffix", [".md", ".json", ".csv"])
def test_vocabulary_text_rejects_benchmark_paths(suffix: str) -> None:
    path = (
        f"benchmarks/{_BENCHMARK_DIRECTORY}/synthetic-{_SOURCE_MARKERS[1].upper()}.json"
    )
    source = {
        ".md": f"[Synthetic example]({path})",
        ".json": json.dumps({"example": path}),
        ".csv": f"label,path\nsynthetic,{path}\n",
    }[suffix]
    assert _vocabulary_violations(
        source, f"synthetic{suffix}", _VocabularyReview("VOCAB_ONLY")
    ) == ["benchmark path literal"]


@pytest.mark.parametrize(
    "source",
    [
        f"--staging-input=benchmarks/{_SOURCE_MARKERS[0]}_synthetic.csv",
        f"--staging-rubric={next(name for name in sorted(_DATA_FILENAMES) if 'rubric' in name)}",
        f"|staging|benchmarks/{_SOURCE_MARKERS[1]}_synthetic.csv|",
    ],
    ids=["input-option", "rubric-option", "compact-table"],
)
def test_vocabulary_text_rejects_paths_next_to_option_and_table_delimiters(
    source: str,
) -> None:
    assert _vocabulary_violations(
        source, "synthetic.md", _VocabularyReview("VOCAB_ONLY")
    ) == ["benchmark path literal"]


@pytest.mark.parametrize("suffix", [".md", ".json", ".csv"])
def test_vocabulary_allows_plain_mentions(suffix: str) -> None:
    source = {
        ".md": f"Plain vocabulary mention of {_SOURCE_MARKERS[0].title()}27.",
        ".json": json.dumps({"branch": f"feat/review-{_SOURCE_MARKERS[1]}"}),
        ".csv": f"split\n{_SOURCE_MARKERS[0]}\n",
    }[suffix]
    assert (
        _vocabulary_violations(
            source, f"synthetic{suffix}", _VocabularyReview("VOCAB_ONLY")
        )
        == []
    )


def _enclosing_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    *,
    include_comprehensions: bool = False,
    include_arguments: bool = False,
) -> ast.AST:
    child = None
    while (parent := parents.get(node)) is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node in parent.body or (include_arguments and node is parent.args):
                return parent
        elif isinstance(parent, ast.ClassDef):
            if node in parent.body:
                return parent
        elif isinstance(parent, ast.Lambda) and node is parent.body:
            return parent
        elif include_comprehensions and isinstance(parent, _COMPREHENSIONS):
            # The outermost iterable is evaluated in the surrounding scope.
            first = parent.generators[0]
            if node is not first or child is not first.iter:
                return parent
        child, node = node, parent
    return node


def _item_construction_sites(source: str) -> list[tuple[int, tuple[int, ...]]]:
    tree = ast.parse(source)
    aliases = _import_aliases(tree)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    constructors = []
    io_by_scope: dict[ast.AST, list[int]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _qualified_name(node.func, aliases)
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
            node.ctx, ast.Load
        ):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and node is parent.func:
                continue
            name = _qualified_name(node, aliases)
            if (
                name.rsplit(".", 1)[-1] == "HeldOutItem"
                and isinstance(parent, ast.Attribute)
                and parent.attr == "from_dict"
            ):
                # Count the full factory reference once.
                continue
        else:
            continue
        scope = _enclosing_scope(node, parents, include_arguments=True)
        while isinstance(scope, ast.Lambda):
            scope = _enclosing_scope(scope, parents, include_arguments=True)
        if isinstance(node, ast.Call) and _is_io_call(name):
            io_by_scope[scope].append(node.lineno)
        owner = parents.get(scope)
        if (
            name.split(".")[-1] == "HeldOutItem"
            or name.split(".")[-2:] == ["HeldOutItem", "from_dict"]
            or (
                isinstance(node, ast.Call)
                and name == "cls"
                and isinstance(scope, _FUNCTION_SCOPES)
                and isinstance(owner, ast.ClassDef)
                and owner.name == "HeldOutItem"
            )
        ):
            constructors.append((node.lineno, scope))
    return sorted(
        (line, tuple(sorted(io_by_scope[scope]))) for line, scope in constructors
    )


def test_item_construction_outside_reader_owners_is_free_of_io() -> None:
    # The reference scan rejects the item identifier in any other tracked path.
    for path in sorted(_EXPECTED_VOCABULARY.keys() | _GUARD_FILES):
        if path.endswith(".py"):
            unsafe = [
                (line, io_lines)
                for line, io_lines in _item_construction_sites(_read_source(path))
                if io_lines
            ]
            assert not unsafe, (
                f"{path}: item creation and IO in the same scope: {unsafe}"
            )


@pytest.mark.parametrize("constructor", ["Item", "Item.from_dict"])
def test_item_construction_detects_io_in_the_same_function(constructor: str) -> None:
    source = (
        "from example import HeldOutItem as Item\n"
        "def build():\n"
        "    data = source.read_text()\n"
        f"    return {constructor}(data)\n"
    )
    assert _item_construction_sites(source) == [(4, (3,))]


@pytest.mark.parametrize(
    "factory",
    ["Record", "Record.from_dict", "lambda row: Record.from_dict(row)"],
    ids=["class-callable", "factory-callable", "lambda-factory"],
)
def test_item_construction_detects_callables_used_with_io(factory: str) -> None:
    source = (
        "from example import HeldOutItem as Record\n"
        "def load_rows():\n"
        "    rows = staging_source.read_text()\n"
        f"    return list(map({factory}, rows))\n"
    )
    assert _item_construction_sites(source) == [(4, (3,))]


def test_item_construction_detects_io_in_default_arguments() -> None:
    source = (
        "from example import HeldOutItem as Record\n"
        "def build_row(raw=staging_source.read_text()):\n"
        "    return Record.from_dict(raw)\n"
    )
    assert _item_construction_sites(source) == [(3, (2,))]


def test_item_construction_allows_io_in_a_different_function() -> None:
    source = (
        "def build(data):\n"
        "    def unrelated():\n"
        "        return source.read_text()\n"
        "    return HeldOutItem.from_dict(data)\n"
    )
    assert _item_construction_sites(source) == [(4, ())]


@pytest.mark.parametrize("path", _PREREGISTRATION_PATHS)
def test_preregistration_blobs_match_base(path: str) -> None:
    def git_oid(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=_REPO_ROOT).strip()

    source = _REPO_ROOT / path
    assert source.resolve() == source, f"Reviewed path redirects elsewhere: {path}"
    frozen = git_oid("rev-parse", f"{_PREREGISTRATION_BASE}:{path}")
    assert git_oid("rev-parse", f":{path}") == frozen
    assert git_oid("hash-object", "--no-filters", "--", path) == frozen
