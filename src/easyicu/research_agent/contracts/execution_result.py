"""Dependency-neutral result returned by generated-code runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunResult:
    """Everything captured from one generated-code execution."""

    step_id: str
    script_path: Path
    cwd: Path
    out_dir: Path
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float
    artefacts: List[Path] = field(default_factory=list)
    timed_out: bool = False
    requested_network_policy: str = "none"
    effective_isolation: str = "unknown"
    isolation_degraded: bool = False
    isolation_degradation_reason: Optional[str] = None
    runtime_provenance: Dict[str, object] = field(default_factory=dict)
    # False means callers must not scan or hash anything under ``out_dir``.
    outputs_safe_to_collect: bool = True
    runner_log_path: Optional[Path] = None

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0 and not self.timed_out and self.outputs_safe_to_collect
        )


__all__ = ["RunResult"]
