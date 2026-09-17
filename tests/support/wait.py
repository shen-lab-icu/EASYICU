"""Polling helper for async/job tests (E-P2-6).

``wait_until`` replaces hand-rolled ``deadline = time.time() + N`` /
``while ...: time.sleep(0.01)`` loops in ``test_webserver_security_hardening``
and ``test_pi_copilot_gateway``.  A single helper keeps timeouts, poll
intervals, and failure messages consistent and makes flaky waits greppable.
"""

from __future__ import annotations

import time
from typing import Callable


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    interval: float = 0.01,
    message: str = "condition was not met before timeout",
) -> None:
    """Poll ``predicate`` until true or ``timeout`` seconds elapse.

    Raises ``AssertionError`` with ``message`` on timeout so failures read
    as assertion failures, not hangs.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(message)


__all__ = ["wait_until"]
