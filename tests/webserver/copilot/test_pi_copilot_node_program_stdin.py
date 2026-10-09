"""A Node harness program reaches Node on stdin, however large it grows.

Linux caps one argument string at 128 KiB (MAX_ARG_STRLEN). Harnesses embed
whole owner sources in their program, so ``node --eval PROGRAM`` passed on
macOS while 21 Pi contract tests failed on CI with E2BIG.
``tests.support.node.run_node`` keeps the ``--eval`` argument layout.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    return node


def test_a_program_longer_than_one_linux_argument_keeps_the_eval_argv_layout() -> None:
    program = (
        "//"
        + "x" * 200_000
        + "\nprocess.stdout.write(JSON.stringify(process.argv.slice(1)));"
    )
    result = run_node(_node(), program, "first", "second", check=True)
    assert json.loads(result.stdout) == ["first", "second"]


def test_a_module_program_runs_as_an_es_module() -> None:
    program = (
        "import path from 'node:path';\n"
        "process.stdout.write(JSON.stringify([typeof path.join, process.argv.slice(1)]));"
    )
    result = run_node(_node(), program, "only", module=True, check=True)
    assert json.loads(result.stdout) == ["function", ["only"]]
