"""SIGUSR1 dumps every thread's stack and the process keeps running."""

import os
import signal
import threading
import time

import pytest

from easyicu.webserver import stack_dump


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signal")
def test_sigusr1_writes_all_thread_stacks(tmp_path):
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, name="stalled-worker", daemon=True)
    worker.start()
    out = open(tmp_path / "stacks.txt", "w+")
    try:
        assert stack_dump.install(out)
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.2)
        out.flush()
        text = (tmp_path / "stacks.txt").read_text()
    finally:
        import faulthandler

        faulthandler.unregister(signal.SIGUSR1)
        stop.set()
        out.close()
    assert "Current thread" in text
    assert text.count("hread 0x") >= 2  # The stalled worker is dumped too.


def test_the_web_app_installs_it_at_startup():
    from easyicu.webserver import app as web_app

    names = [handler.__name__ for handler in web_app.app.router.on_startup]
    # Second: the shared-socket refusal must stay first (its own contract).
    assert names[:2] == ["_refuse_a_shared_unix_socket", "_install_stack_dump_signal"]
