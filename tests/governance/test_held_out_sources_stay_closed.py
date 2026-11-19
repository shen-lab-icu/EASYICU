"""Keep benchmark question sources within the reviewed scope."""

from __future__ import annotations

import re
import runpy
import subprocess
from os import fsdecode
from pathlib import Path

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
# The guard defines the marker tokens and the tests use them in synthetic paths.
_GUARD_FILES = frozenset(
    {
        "tests/governance/test_held_out_sources_stay_closed.py",
        "tools/benchmark_question_sources.py",
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
    unreviewed = references - _REVIEWED_OWNERS - _GUARD_FILES
    assert not unreviewed, (
        f"Unreviewed held-out source references: {sorted(unreviewed)}"
    )
