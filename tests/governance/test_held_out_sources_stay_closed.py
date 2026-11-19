"""Keep benchmark question sources within the reviewed scope."""

from __future__ import annotations

import ast
import json
import posixpath
import re
import runpy
import subprocess
from collections import defaultdict
from collections.abc import Iterator
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
    "benchmarks/figure2_icu_agent_v2/data_platform_validation_protocol_v1.json": "preregistration_text",
    "benchmarks/figure2_icu_agent_v2/statistical_analysis_plan_v1.json": "preregistration_text",
    "docs/INDEPENDENT_EVAL_PROTOCOL.md": "heldout_item_type",
    "docs/evidence/authority_followup_20260904/figure_exports/synthetic_sources/model_performance.csv": "synthetic_split_label",
    "src/easyicu/research_agent/__init__.py": "heldout_item_type",
    "src/easyicu/research_agent/evaluation/independent_eval_harness.py": "heldout_item_type",
    "src/easyicu/research_agent/figures/prediction.py": "ml_split_vocabulary",
    "src/easyicu/research_agent/reporting/supplement_inventory.py": "ml_split_vocabulary",
    "src/easyicu/research_agent/validity_signals.py": "ml_split_vocabulary",
    "tools/arch_baselines/execution_phase.json": "branch_name_note",
    "tools/verify_figure_presentation.py": "ml_split_vocabulary",
}
_PREREGISTRATION_BASE = "ef3472d8e1f4f32ed1c76a73eacf591d442ceb09"
_BENCHMARK_DIRECTORY = "figure2_icu_agent_v2"
_DATA_FILENAMES = {PurePosixPath(path).name for path in _REVIEWED_DATA_PATHS}
_TEXT_TOKEN = re.compile(r"""[^\s"'`<>{}\[\](),;]+""")
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


def _read_vocabulary_review() -> dict[str, str]:
    entries = {}
    for line in _read_source(_VOCABULARY_MANIFEST).splitlines():
        if line and not line.startswith("#"):
            reason, path = line.split("\t")
            assert path not in entries, f"Duplicate vocabulary review entry: {path}"
            entries[path] = reason
    return entries


def test_vocabulary_review_is_frozen() -> None:
    entries = _read_vocabulary_review()
    assert entries == _EXPECTED_VOCABULARY
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


def _string_literal(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_string_literal(value) or "{value}" for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_literal(node.left), _string_literal(node.right)
        if left and right:
            return left + right
    return ""


def _is_path_expression(node: ast.AST, aliases: dict[str, str]) -> bool:
    if isinstance(node, ast.Call):
        return _qualified_name(node.func, aliases).rsplit(".", 1)[-1] in _PATH_TYPES
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _is_path_expression(node.left, aliases)
    )


def _python_vocabulary_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    aliases = _import_aliases(tree)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_io_call(
            _qualified_name(node.func, aliases)
        ):
            violations.append(f"line {node.lineno}: IO call")
        if _is_path_expression(node, aliases):
            parts = node.args if isinstance(node, ast.Call) else [node.right]
            if any(_has_source_marker(_string_literal(part)) for part in parts):
                violations.append(f"line {node.lineno}: path literal")
        literal = _string_literal(node)
        if any(_has_source_marker(token) for token in _path_tokens(literal)):
            violations.append(f"line {node.lineno}: path literal")
    return sorted(set(violations))


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


def _vocabulary_violations(source: str, suffix: str) -> list[str]:
    if suffix == ".py":
        return _python_vocabulary_violations(source)
    assert suffix in {".md", ".json", ".csv"}
    return _text_vocabulary_violations(source, suffix)


@pytest.mark.parametrize("path", sorted(_EXPECTED_VOCABULARY))
def test_reviewed_vocabulary_has_no_source_access(path: str) -> None:
    violations = _vocabulary_violations(_read_source(path), Path(path).suffix)
    assert not violations, f"{path}: {violations}"


@pytest.mark.parametrize(
    "source",
    [
        "open('synthetic.json')",
        "path.read_text()",
        "path.read_bytes()",
        "path.glob('*.json')",
        "path.rglob('*.json')",
        "path.iterdir()",
        "import os as fs\nfs.listdir('fixtures')",
        "from os import scandir as scan\nscan('fixtures')",
        "import pandas as pd\npd.read_csv('synthetic.csv')",
        "from pandas import read_parquet as load_frame\nload_frame('synthetic.parquet')",
        "import json as codec\ncodec.load(stream)",
        "from json import loads as parse\nparse('{}')",
    ],
)
def test_vocabulary_python_rejects_io(source: str) -> None:
    assert any(
        violation.endswith("IO call")
        for violation in _vocabulary_violations(source, ".py")
    )


@pytest.mark.parametrize(
    "source",
    [
        f"example = 'fixtures/{_SOURCE_MARKERS[0].title()}_demo.json'",
        f"example = '{_SOURCE_MARKERS[1].upper()}.csv'",
        f"from pathlib import Path as P\nexample = P('fixtures', '{_SOURCE_MARKERS[1]}')",
        f"example = Path('fixtures') / '{_SOURCE_MARKERS[0]}'",
    ],
)
def test_vocabulary_python_rejects_path_literals(source: str) -> None:
    assert any(
        violation.endswith("path literal")
        for violation in _vocabulary_violations(source, ".py")
    )


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
    assert _vocabulary_violations(source, suffix) == ["benchmark path literal"]


@pytest.mark.parametrize("suffix", [".py", ".md", ".json", ".csv"])
def test_vocabulary_allows_plain_mentions(suffix: str) -> None:
    source = {
        ".py": f"split = '{_SOURCE_MARKERS[0]}'",
        ".md": f"Plain vocabulary mention of {_SOURCE_MARKERS[0].title()}27.",
        ".json": json.dumps({"branch": f"feat/review-{_SOURCE_MARKERS[1]}"}),
        ".csv": f"split\n{_SOURCE_MARKERS[0]}\n",
    }[suffix]
    assert _vocabulary_violations(source, suffix) == []


def _enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    while (parent := parents.get(node)) is not None:
        if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node in parent.body:
                return parent
        elif isinstance(parent, ast.Lambda) and node is parent.body:
            return parent
        node = parent
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
        if not isinstance(node, ast.Call):
            continue
        scope = _enclosing_scope(node, parents)
        name = _qualified_name(node.func, aliases)
        if _is_io_call(name):
            io_by_scope[scope].append(node.lineno)
        owner = parents.get(scope)
        if (
            name.split(".")[-1] == "HeldOutItem"
            or name.split(".")[-2:] == ["HeldOutItem", "from_dict"]
            or (
                name == "cls"
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


def test_item_construction_allows_io_in_a_different_function() -> None:
    source = (
        "def build(data):\n"
        "    def unrelated():\n"
        "        return source.read_text()\n"
        "    return HeldOutItem.from_dict(data)\n"
    )
    assert _item_construction_sites(source) == [(4, ())]


@pytest.mark.parametrize(
    "path",
    [
        path
        for path, reason in _EXPECTED_VOCABULARY.items()
        if reason == "preregistration_text"
    ],
)
def test_preregistration_blobs_match_base(path: str) -> None:
    def git_oid(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=_REPO_ROOT).strip()

    source = _REPO_ROOT / path
    assert source.resolve() == source, f"Reviewed path redirects elsewhere: {path}"
    frozen = git_oid("rev-parse", f"{_PREREGISTRATION_BASE}:{path}")
    assert git_oid("rev-parse", f":{path}") == frozen
    assert git_oid("hash-object", "--no-filters", "--", path) == frozen
