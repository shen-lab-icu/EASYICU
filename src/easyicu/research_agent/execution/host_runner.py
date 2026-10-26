"""Desktop backend for exact host-rendered executor code.

This is a trusted-code runner, not a sandbox. Input immutability, output paths
and absence of network calls are properties of our typed executors. Model
candidates, repairs and restored scripts require a sandbox even when their
step originally had a deterministic owner.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from importlib import metadata
from functools import lru_cache
from typing import Mapping

from easyicu.runtime.host_execution_lease import (
    HOST_EXECUTION_SCHEMA,
    HostExecutionLeaseError,
    acknowledge_host_execution_parent,
    assert_host_execution_quiescent,
    prepare_host_execution,
    read_host_execution_receipt,
)

from ..canonical_json import canonical_sha256
from ..contracts.execution_result import RunnerFailureCode, RunResult
from .kernel_identity import (
    build_execution_kernel_identity,
    execution_kernel_relative_paths,
)
from .method_capabilities import set_runtime_capability_snapshot_provider
from .code_hygiene import reorder_forward_references
from .runner import (
    CodeRunner,
    _run_capturing_with_descendant_reaping,
    _safe_path_component,
)
from .step_execution import CodeOrigin, HostCodeRefused

HOST_RUNTIME_SCHEMA = "easyicu.host_runtime_preflight/1"
_LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _kernel_manifest(package_root: Path) -> tuple[str, ...]:
    # The running application owns the import graph. Rehash its source bytes
    # on adoption, without reparsing that graph for every step's runner.
    return execution_kernel_relative_paths(package_root)


class HostRunner(CodeRunner):
    """Run trusted executor bytes in the same Python environment as the app."""

    requires_code_origin = True
    manages_output_cleanup = True

    def __init__(
        self,
        *,
        workdir: Path,
        cohort_parquet: Path,
        timeout_seconds: float = 300.0,
        extra_env=None,
    ) -> None:
        super().__init__(
            workdir=workdir,
            cohort_parquet=cohort_parquet,
            timeout_seconds=timeout_seconds,
            extra_env=extra_env,
            python_executable=sys.executable,
            allow_unsafe_host_fallback=False,
        )
        self._runtime_lock = threading.RLock()
        self._runtime_bundle = None
        self._host_launch = None
        self._host_spawning = False
        self._host_run_lock = threading.Lock()

    def build_command(self, *, script_path: Path) -> list[str]:
        launch = getattr(self, "_host_launch", None)
        arguments = (
            "supervise" if launch is not None else "execute",
            "--script", str(script_path), "--sha256",
            hashlib.sha256(script_path.read_bytes()).hexdigest(),
        )
        if launch is not None:
            launch.bind_script(arguments[-1])
            arguments += (
                "--lease-dir", str(launch.path), "--nonce", launch.nonce,
                "--timeout", str(self.timeout_seconds),
            )
        if getattr(sys, "frozen", False):
            from easyicu.runtime.host_worker import frozen_worker_command

            return frozen_worker_command(self.python_executable, *arguments)
        # The worker cleans up descendants before its own normal exit. Once
        # a plain script exits, a detached child is reparented and a parent-side
        # process-tree walk can no longer establish that child's ownership.
        return [self.python_executable, "-m", "easyicu.runtime.host_worker", *arguments]

    def _runtime_probe_command(self, kind, names, source) -> list[str]:
        if getattr(sys, "frozen", False):
            from easyicu.runtime.host_worker import frozen_worker_command

            return frozen_worker_command(
                self.python_executable, kind, "--names", json.dumps(list(names)),
            )
        return super()._runtime_probe_command(kind, names, source)

    def _isolation_backend_for_cmd(self, cmd) -> str:
        return "host_rendered_executor"

    def _execution_cwd(self, step_dir: Path) -> Path:
        return self.workdir

    def _matplotlib_config_dir(self, step_dir: Path) -> Path:
        return self.workdir / ".matplotlib"

    def _configure_execution_env(self, env: dict[str, str]) -> None:
        env["EASYICU_HOST_RENDERED_ONLY"] = "1"
        if os.environ.get("TEMP"):
            env["TEMP"] = os.environ["TEMP"]

    def _execute_command(self, cmd, **kwargs):
        launch = self._host_launch
        if launch is None:
            raise RuntimeError("host_execution_lease_required")

        def spawned(proc):
            try:
                launch.bind_guardian(proc.pid)
            finally:
                launch.release()

        def stop_guardian(proc):
            if proc.poll() is None:
                from easyicu.runtime.host_execution_lease import request_host_execution_stop

                request_host_execution_stop(launch.path, launch.nonce)
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    # A hard-killed guardian never grants restart authority:
                    # its unresolved receipt survives even after locks close.
                    self._terminate_tree(proc)

        launch.mark_spawning()
        self._host_spawning = True
        timeout = kwargs.pop("timeout")
        result = _run_capturing_with_descendant_reaping(
            cmd, terminate_tree=stop_guardian, on_spawn=spawned,
            reap_process_group=False, timeout=timeout + 15.0, **kwargs,
        )
        # Do not collect files or write step receipts while child ownership is
        # uncertain. The guardian's exit code alone is not cleanup evidence.
        try:
            assert_host_execution_quiescent(
                launch.path, allow_missing=False, allow_current_parent=True,
                wait_timeout=2.0,
            )
        except HostExecutionLeaseError:
            # A guardian can exit before it claims the lease (for example,
            # during frozen-runtime startup). Retain that startup diagnostic
            # in the private backend log without writing into an unsettled
            # run. No worker could have run while its receipt is prepared.
            try:
                receipt = read_host_execution_receipt(launch.path)
            except (OSError, HostExecutionLeaseError):
                receipt = None
            if (
                receipt is not None and receipt.get("nonce") == launch.nonce
                and receipt.get("state") == "prepared"
                and receipt.get("worker") is None
            ):
                _LOGGER.error(
                    "Host guardian stopped before worker admission (exit=%s): %s",
                    result.returncode,
                    result.stderr.replace(launch.nonce, "[redacted]")[-4096:],
                )
            raise
        receipt = read_host_execution_receipt(launch.path)
        if receipt is None or receipt["nonce"] != launch.nonce:
            raise RuntimeError("host_execution_receipt_changed")
        if receipt.get("reason") == "timeout":
            raise subprocess.TimeoutExpired(
                cmd, timeout, output=result.stdout, stderr=result.stderr
            )
        return result

    @staticmethod
    def _terminate_tree(proc) -> None:
        """Stop Windows descendants and POSIX children that used setsid."""
        import psutil

        poll = getattr(proc, "poll", None)
        if callable(poll) and poll() is not None:
            return
        try:
            parent = psutil.Process(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        # Do not act on a new process that reused the direct child's PID
        # between the first poll and constructing the psutil identity handle.
        if callable(poll) and poll() is not None:
            return
        try:
            parent.suspend()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            children = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        for child in reversed(children):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def validate_runtime_capabilities(self) -> tuple[str, ...]:
        with self._runtime_lock:
            if self._runtime_bundle is None:
                if importlib.util.find_spec("psutil") is None:
                    raise RuntimeError("host_runtime_unavailable: psutil is required")
                capabilities = super().validate_runtime_capabilities()
                package_root = Path(__file__).resolve().parents[2]
                kernel = build_execution_kernel_identity(
                    package_root, relative_paths=_kernel_manifest(package_root)
                )
                requirements = "".join(
                    sorted(
                        {
                            f"{dist.metadata['Name']}=={dist.version}\n"
                            for dist in metadata.distributions()
                            if dist.metadata.get("Name")
                        }
                    )
                )
                provenance = {
                    "runtime": "host",
                    "python_executable": self.python_executable,
                    "python_version": sys.version,
                    "network": "unused_by_host_executors",
                    "code_policy": CodeOrigin.HOST_RENDERED_EXECUTOR.value,
                    "execution_lifecycle_schema": HOST_EXECUTION_SCHEMA,
                    "dependency_capture_method": "importlib.metadata.distributions",
                    "requirements_sha256": hashlib.sha256(
                        requirements.encode()
                    ).hexdigest(),
                    "execution_kernel_identity_schema": kernel.schema_version,
                    "execution_kernel_identity_sha256": kernel.identity_sha256,
                    "execution_kernel_source_sha256": kernel.source_sha256,
                    "execution_kernel_files_sha256": kernel.files_sha256,
                    "execution_kernel_file_count": kernel.file_count,
                    "runner_requirements_lock_sha256": kernel.requirements_lock_sha256,
                    "method_capabilities": list(capabilities),
                }
                self._runtime_bundle = {
                    "schema": HOST_RUNTIME_SCHEMA,
                    "provenance": provenance,
                    "requirements": requirements,
                }
            snapshot = tuple(self._runtime_bundle["provenance"]["method_capabilities"])
            set_runtime_capability_snapshot_provider(lambda: snapshot)
            return snapshot

    def export_validated_runtime_bundle(self) -> dict:
        self.validate_runtime_capabilities()
        return copy.deepcopy(self._runtime_bundle)

    def adopt_validated_runtime_bundle(self, bundle: Mapping[str, object]) -> None:
        # Rebuild on a new runner, including source and installed versions; a
        # receipt from another interpreter/runtime is never an authorization.
        if bundle != self.export_validated_runtime_bundle():
            raise ValueError(
                "Host runtime changed after preflight; refusing cached receipt"
            )

    @property
    def authority_identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "interpreter_and_inputs": super().authority_identity_sha256,
                "runtime": self.export_validated_runtime_bundle(),
            }
        )

    def run(
        self,
        *,
        step_id: str,
        code: str,
        resolved_inputs_path=None,
        code_origin: CodeOrigin = CodeOrigin.MODEL_GENERATED,
    ) -> RunResult:
        if not self._host_run_lock.acquire(blocking=False):
            raise HostExecutionLeaseError("host_execution_active")
        try:
            return self._run_host(
                step_id=step_id, code=code, resolved_inputs_path=resolved_inputs_path,
                code_origin=code_origin,
            )
        finally:
            self._host_run_lock.release()

    def _run_host(
        self, *, step_id: str, code: str, resolved_inputs_path,
        code_origin: CodeOrigin,
    ) -> RunResult:
        step_id = _safe_path_component(step_id, label="step_id")
        step_dir = self.workdir / "steps" / step_id
        out_dir = step_dir / "outputs"
        if code_origin != CodeOrigin.HOST_RENDERED_EXECUTOR:
            return RunResult(
                step_id=step_id,
                script_path=step_dir / "analysis.py",
                cwd=self.workdir,
                out_dir=out_dir,
                stdout="",
                stderr=HostCodeRefused.reason_code,
                returncode=126,
                duration_seconds=0,
                outputs_safe_to_collect=False,
                effective_isolation="blocked_host_code_policy",
                runner_failure_code=RunnerFailureCode.HOST_RUNNER_REFUSES_MODEL_CODE,
            )
        bundle = self.export_validated_runtime_bundle()
        # Reserve before deleting outputs, replacing the script or snapshotting
        # authority. A surviving guardian fences every step in this run.
        launch = prepare_host_execution(
            self.workdir / ".host-execution",
            script_sha256=hashlib.sha256(
                reorder_forward_references(code).encode("utf-8")
            ).hexdigest(),
        )
        self._host_launch = launch
        self._host_spawning = False
        try:
            if out_dir.is_symlink():
                out_dir.unlink()
            elif out_dir.exists():
                shutil.rmtree(out_dir)
            result = super().run(
                step_id=step_id, code=code, resolved_inputs_path=resolved_inputs_path
            )
            result.runtime_provenance = bundle["provenance"]
            return result
        finally:
            try:
                if not self._host_spawning:
                    launch.abort()
            finally:
                launch.release()
                self._host_launch = None
                if self._host_spawning:
                    acknowledge_host_execution_parent(launch.path, launch.nonce)
