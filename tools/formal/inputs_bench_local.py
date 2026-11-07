"""bench7v2's input-snapshot step: run the sealed checkout's easyicu.formal_inputs.

memgate.py counts a scope when a process in it has "bench_local.py" in its
command line, and the bench core checks match the same pattern. This file's
name ends in "bench_local.py", so the snapshot step is counted under its own
real script name (no argv rewriting, no change to memgate.py).
Usage: PYTHONPATH=<code>/src python inputs_bench_local.py snapshot|verify|plan ...
"""
import sys

from easyicu.formal_inputs import main

sys.exit(main(sys.argv[1:]))
