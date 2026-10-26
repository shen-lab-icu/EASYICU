"""Small fixed-command child process interface for the frozen desktop runtime.

PyInstaller's executable is the application, not a Python command line. This
entry point exposes only runtime probes and an exact, host-authorized step file;
it deliberately does not implement ``-c``, ``-m`` or arbitrary interpreter flags.
The execution owner must establish typed-code provenance before constructing the
command. The digest here binds the bytes read by the child to that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Sequence

WORKER_FLAG = "--easyicu-host-worker"


def frozen_worker_command(executable: str, action: str, *arguments: str) -> list[str]:
    if action not in {"capabilities", "identity", "execute", "replay", "supervise"}:
        raise ValueError("Unknown desktop worker action")
    return [executable, WORKER_FLAG, action, *arguments]


def _package_names(raw: str) -> list[str]:
    names = json.loads(raw)
    if (
        not isinstance(names, list) or not names or len(names) > 100
        or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name) for name in names)
    ):
        raise ValueError("Invalid runtime package names")
    return names


def _finish_descendants() -> None:
    """Reap owned background work while this worker is still its parent.

    The parent runner separately handles timeouts. Successful or failed normal
    completion must do this here: after this worker exits, detached children
    are reparented and cannot safely be rediscovered from its former PID.
    psutil's process handles check creation time before signalling, so PID reuse
    cannot turn cleanup into a signal to an unrelated process.
    """
    import psutil

    pending = psutil.Process().children()
    children = []
    seen = set()
    cleanup_failed = False
    while pending:
        child = pending.pop()
        try:
            identity = (child.pid, child.create_time())
            if identity in seen:
                continue
            seen.add(identity)
            children.append(child)
            # Freeze each spawning parent before enumerating its children.
            child.suspend()
            pending.extend(child.children())
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            cleanup_failed = True
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            cleanup_failed = True
    _, alive = psutil.wait_procs(children, timeout=2.0)
    for child in alive:
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                cleanup_failed = True
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            cleanup_failed = True
    if cleanup_failed:
        raise RuntimeError("Desktop worker could not stop a descendant")


def _execute(script: str, expected_sha256: str, *, replay: bool = False) -> None:
    path = Path(script)
    run_dir = Path(os.environ.get("EASYICU_RUN_DIR", ""))
    output_dir = Path(os.environ.get("STEP_OUT_DIR", ""))
    if (
        os.environ.get("EASYICU_HOST_RENDERED_ONLY") != "1"
        or not path.is_absolute() or path.is_symlink()
        or path.name != "analysis.py" or path.parent.parent.name != "steps"
        or not run_dir.is_absolute()
        or not output_dir.is_absolute()
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError("Desktop execution requires a host-owned step command")
    source_run = path.parent.parent.parent.resolve()
    if replay:
        # A registered sensitivity replay keeps the sealed primary script and
        # writes only inside a step's model_replays subtree of the same run.
        relative_output = output_dir.resolve().relative_to(source_run)
        parts = relative_output.parts
        valid = (
            len(parts) == 6 and parts[0] == "steps" and parts[2] == "outputs"
            and parts[3] == "model_replays" and parts[5] == "outputs"
            and run_dir.resolve() in {source_run, output_dir.parent.resolve()}
            and Path.cwd().resolve() == path.parent.resolve()
        )
    else:
        valid = (
            source_run == run_dir.resolve() and Path.cwd().resolve() == source_run
            and output_dir.resolve() == path.parent.joinpath("outputs").resolve()
        )
    if not valid:
        raise ValueError("Desktop execution coordinates do not match the authorized workspace")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Desktop step bytes changed after host authorization")
    # Execute the bytes we checked, never reopen a mutable script with runpy.
    previous_argv = sys.argv
    sys.argv = [str(path)]
    try:
        exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})  # noqa: S102 - exact bytes authorized by the typed host executor
    finally:
        try:
            _finish_descendants()
        finally:
            sys.argv = previous_argv


def _execute_registered(args: argparse.Namespace) -> None:
    """Run after admission, then remain alive for guardian-owned tree cleanup."""
    from easyicu.runtime.host_execution_lease import (
        complete_host_worker_cleanup,
        verify_host_worker,
    )

    # No user code, including its imports, is allowed before this handshake.
    authorized_sha256 = verify_host_worker(Path(args.lease_dir), args.nonce)
    source_run = Path(args.script).parent.parent.parent.resolve()
    if args.sha256 != authorized_sha256 or Path(args.lease_dir).resolve() != source_run / ".host-execution":
        raise ValueError("The worker lease does not bind this script and run")
    status = 0
    try:
        _execute(args.script, args.sha256, replay=args.action == "replay")
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        import traceback

        try:
            traceback.print_exc()
        except OSError:
            # A killed host can close its capture pipes before the guardian
            # reaches this worker. That must not prevent tree cleanup.
            pass
        status = 1
    try:
        # The acknowledgement reports logical completion, not quiescence.
        # Stay alive afterward so the guardian can freeze every worker thread,
        # enumerate descendants and stop the tree before releasing its lease.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except OSError:
                pass
        _finish_descendants()
        complete_host_worker_cleanup(Path(args.lease_dir), args.nonce, exitcode=status if 0 <= status <= 255 else 1)
    except BaseException:
        os._exit(125)
    while True:
        time.sleep(60.0)


def _supervise(args: argparse.Namespace) -> int:
    """Hold the execution lease until the exact child tree has stopped.

    A normal exception reports logical completion while its worker stays alive.
    The guardian owns final cleanup. Any independently exited worker remains
    unresolved: an orphan may have escaped a later process-tree lookup.
    """
    import math
    import subprocess

    from easyicu.runtime.host_execution_lease import claim_host_execution

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("A guardian timeout must be finite and positive")
    lease = claim_host_execution(Path(args.lease_dir), args.nonce)
    worker = None
    try:
        source_run = Path(args.script).parent.parent.parent.resolve()
        if args.sha256 != lease.script_sha256 or Path(args.lease_dir).resolve() != source_run / ".host-execution":
            lease.mark_quiesced("script_binding_mismatch", cleanup_proven=True)
            return 125
        if not lease.parent_alive() or lease.stop_requested():
            lease.mark_quiesced("cancelled_before_worker", cleanup_proven=True)
            return 125
        arguments = (
            "execute", "--script", args.script, "--sha256", args.sha256,
            "--lease-dir", args.lease_dir, "--nonce", args.nonce,
        )
        command = (
            frozen_worker_command(sys.executable, *arguments)
            if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "easyicu.runtime.host_worker", *arguments]
        )
        # stdout/stderr remain the parent's bounded capture pipes. No lock or
        # authority depends on their lifetime. The child waits for record_worker
        # before its script can execute, closing the Popen/receipt crash window.
        worker = subprocess.Popen(command, start_new_session=os.name == "posix")
        lease.record_worker(worker.pid)
        deadline = time.monotonic() + args.timeout
        while True:
            status = worker.poll()
            if status is not None:
                lease.mark_unresolved("worker_exit_without_cleanup")
                return 125
            completion = lease.worker_completion_status()
            reason = (
                ("worker_completed" if completion == 0 else "worker_failed") if completion is not None
                else "parent_lost" if not lease.parent_alive()
                else "stop_requested" if lease.stop_requested()
                else "timeout" if time.monotonic() >= deadline
                else None
            )
            if reason is not None:
                stopped = lease.terminate_worker_tree()
                if not stopped:
                    try:
                        worker.kill()
                    except OSError:
                        pass
                try:
                    worker.wait(timeout=3.0)
                except (OSError, subprocess.TimeoutExpired):
                    stopped = False
                if stopped:
                    lease.mark_quiesced(reason, cleanup_proven=True)
                else:
                    lease.mark_unresolved("worker_cleanup_unconfirmed")
                if completion is not None and stopped:
                    return completion
                return 124 if reason == "timeout" and stopped else 125
            time.sleep(0.05)
    except BaseException:
        # A failed publication must never turn a released kernel lock into a
        # claim that its child tree has exited. Preserve the unresolved receipt.
        stopped = worker is None
        if worker is not None:
            try:
                stopped = lease.terminate_worker_tree()
            except BaseException:
                stopped = False
            if not stopped:
                # Popen owns this direct child even if record_worker failed.
                # Killing it does not prove descendant cleanup, but prevents a
                # still-barriered, unregistered worker from sleeping forever.
                try:
                    worker.kill()
                except OSError:
                    pass
            try:
                worker.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                stopped = False
        try:
            if stopped:
                lease.mark_quiesced("guardian_failed", cleanup_proven=True)
            else:
                lease.mark_unresolved("guardian_cleanup_unconfirmed")
        except BaseException:
            pass
        return 125
    finally:
        lease.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EasyICU desktop analysis worker")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("capabilities", "identity"):
        commands.add_parser(action).add_argument("--names", required=True)
    for action in ("execute", "replay"):
        execute = commands.add_parser(action)
        execute.add_argument("--script", required=True)
        execute.add_argument("--sha256", required=True)
        execute.add_argument("--lease-dir")
        execute.add_argument("--nonce")
    supervise = commands.add_parser("supervise")
    supervise.add_argument("--script", required=True)
    supervise.add_argument("--sha256", required=True)
    supervise.add_argument("--lease-dir", required=True)
    supervise.add_argument("--nonce", required=True)
    supervise.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args(argv)
    if args.action == "supervise":
        return _supervise(args)
    if args.action in {"execute", "replay"}:
        if bool(args.lease_dir) != bool(args.nonce):
            raise ValueError("A worker lease requires its directory and nonce")
        if args.lease_dir:
            _execute_registered(args)
            raise AssertionError("registered workers exit without interpreter hooks")
        _execute(args.script, args.sha256, replay=args.action == "replay")
        return 0
    names = _package_names(args.names)
    if args.action == "capabilities":
        print(json.dumps([name for name in names if importlib.util.find_spec(name) is not None]))
        return 0
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    print(json.dumps({
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": versions,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
