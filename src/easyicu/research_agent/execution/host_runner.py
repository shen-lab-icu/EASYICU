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
import os
from pathlib import Path
import shutil
import sys
import threading
from importlib import metadata
from functools import lru_cache
from typing import Mapping

from ..canonical_json import canonical_sha256
from ..contracts.execution_result import RunnerFailureCode, RunResult
from .kernel_identity import (
    build_execution_kernel_identity,
    execution_kernel_relative_paths,
)
from .method_capabilities import set_runtime_capability_snapshot_provider
from .runner import (
    CodeRunner,
    _run_capturing_with_descendant_reaping,
    _safe_path_component,
)
from .step_execution import CodeOrigin, HostCodeRefused

HOST_RUNTIME_SCHEMA = "easyicu.host_runtime_preflight/1"


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

    def build_command(self, *, script_path: Path) -> list[str]:
        return [self.python_executable, str(script_path)]

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
        return _run_capturing_with_descendant_reaping(
            cmd, terminate_tree=self._terminate_tree, **kwargs
        )

    @staticmethod
    def _terminate_tree(proc) -> None:
        """Stop Windows descendants and POSIX children that used setsid."""
        import psutil

        try:
            parent = psutil.Process(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
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
        if out_dir.is_symlink():
            out_dir.unlink()
        elif out_dir.exists():
            shutil.rmtree(out_dir)
        result = super().run(
            step_id=step_id, code=code, resolved_inputs_path=resolved_inputs_path
        )
        result.runtime_provenance = bundle["provenance"]
        return result
