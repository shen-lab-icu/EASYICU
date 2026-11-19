"""Keep benchmark question sources within the reviewed scope."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tools.benchmark_question_sources import (
    HeldOutSourceRefused,
    require_allowed_question_source,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Build the uppercase marker at runtime so the audit does not match itself.
_HO_MARKER = "ho30".upper()
_EXCLUDES = (
    ":(exclude,glob,icase)**/*heldout*",
    f":(exclude,glob,icase)**/*{_HO_MARKER}*",
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
        "benchmarks/heldout27.jsonl",
        "tests/fixtures/HeLdOuT27_sample/questions.jsonl",
        f"tests/fixtures/{_HO_MARKER}_sample/questions.jsonl",
        "benchmarks/ho30.jsonl",
    ],
    ids=["held-out-file", "held-out-directory", "ho-directory", "ho-file"],
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


def test_only_reviewed_owners_reference_held_out_sources() -> None:
    # Check the same pathspec against filenames before scanning any contents.
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", ".", *_EXCLUDES],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    has_held_out_name = bool(re.search(rb"heldout|ho30", tracked, re.IGNORECASE))
    assert not has_held_out_name, "The audit pathspec includes a held-out filename"

    references = subprocess.run(
        ["git", "grep", "-l", "-E", rf"heldout[0-9]*_|{_HO_MARKER}", "--", ".", *_EXCLUDES],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert references.returncode in (0, 1), references.stderr
    unreviewed = set(references.stdout.splitlines()) - _REVIEWED_OWNERS
    assert not unreviewed, f"Unreviewed held-out source references: {sorted(unreviewed)}"
