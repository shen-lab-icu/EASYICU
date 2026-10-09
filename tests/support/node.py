"""Run a JavaScript harness program under Node.

Harnesses embed whole owner sources in their program text. Linux caps one
argument string at 128 KiB (MAX_ARG_STRLEN), so ``node --eval PROGRAM`` that
runs on macOS fails on CI with E2BIG once the embedded sources outgrow it.
``run_node`` hands Node the program on standard input instead, which has no
such limit.
"""

from __future__ import annotations

import subprocess

# ``node -`` lists "-" as process.argv[1]. Dropping it before the program runs
# keeps the ``--eval`` layout harnesses read: process.argv[1] is the first
# argument after the program.
_EVAL_ARGV = "data:text/javascript,process.argv.splice(1,1)"


def run_node(
    node: str, program: str, *args: str, module: bool = False, **kwargs
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run([node, "--eval", program, *args])`` without the argv limit.

    ``module`` runs the program as an ES module (``--input-type=module``).
    The program and its output are UTF-8 text, as Node reads and writes them,
    whatever the locale. ``check``, ``cwd``, ``env`` and ``timeout`` pass
    through to :func:`subprocess.run`.

    The argv preload rides in ``process.execArgv``, which ``child_process.fork``
    and ``Worker`` inherit by default: a harness that forks must pass its own
    ``execArgv`` or the child loses its ``argv[1]``.
    """

    check = kwargs.pop("check", False)
    argv = [
        node,
        "--import",
        _EVAL_ARGV,
        *(["--input-type=module"] if module else []),
        "-",
        *args,
    ]
    result = subprocess.run(
        argv, input=program, capture_output=True, encoding="utf-8", **kwargs
    )
    if result.returncode and "bad option: --import" in result.stderr:
        raise RuntimeError(
            f"{node} has no --import (Node 18.19 or 20.6 and later); run_node needs it"
        )
    if check:
        result.check_returncode()
    return result
