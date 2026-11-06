"""Run an extraction's worker processes in the calling thread (test double).

``extract_database`` starts module groups and isolated stream batches as
processes from ``_get_extraction_mp_context``. Replacing that context with
:func:`inprocess_mp_context` runs each target synchronously in the caller,
so a test can observe process-local state (the stay-table payload, the
binding units) with spies installed in this process.
"""

from __future__ import annotations


class _Run:
    def __init__(self, *, target, args, daemon):
        self.target, self.args, self.daemon, self.exitcode = target, args, daemon, None

    def start(self):
        self.target(*self.args)
        self.exitcode = 0

    def join(self, *_args, **_kwargs):
        return None

    def is_alive(self):
        return False


class _Context:
    @staticmethod
    def Process(*, target, args, daemon):
        return _Run(target=target, args=args, daemon=daemon)


def inprocess_mp_context(_mp=None):
    """A drop-in for ``api._get_extraction_mp_context``."""
    return _Context()
