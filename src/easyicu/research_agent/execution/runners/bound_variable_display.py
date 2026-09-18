"""Read variable labels and units from the host-bound context and plan bytes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...research_context.typed import parse_research_context_json
from .typed_input_binding import contained_regular_file, sha256_file


@dataclass(frozen=True)
class BoundVariableDisplay:
    column: str
    label: str
    unit: str | None
    context_sha256: str
    plan_sha256: str


def _bound_bytes(run_dir: Path, manifest: Mapping[str, Any], key: str) -> tuple[bytes, str]:
    binding = manifest.get(key)
    if not isinstance(binding, Mapping):
        raise ValueError(f"variable display requires a bound {key}")
    relative = binding.get("relative_path")
    digest = binding.get("sha256")
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or not isinstance(digest, str) or len(digest) != 64):
        raise ValueError(f"variable display {key} binding is incomplete")
    path = contained_regular_file(run_dir.resolve() / relative, run_dir.resolve())
    if path is None:
        raise ValueError(f"variable display {key} path is not contained")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or sha256_file(path) != digest:
        raise ValueError(f"variable display {key} digest mismatch")
    return raw, digest


def load_bound_variable_display(
    *, run_dir: Path, manifest: Mapping[str, Any], step_id: str, column: str,
) -> BoundVariableDisplay:
    """No clinical name, unit conversion, or scientific choice is inferred."""
    if manifest.get("step_id") != step_id:
        raise ValueError("variable display manifest belongs to another step")
    context_bytes, context_sha = _bound_bytes(run_dir, manifest, "context")
    plan_bytes, plan_sha = _bound_bytes(run_dir, manifest, "plan")
    context = parse_research_context_json(context_bytes)
    variables = [variable for variable in context.variables if variable.name == column]
    if len(variables) != 1:
        raise ValueError("variable display requires one exact context variable")
    variable = variables[0]
    plan = json.loads(plan_bytes)
    labels = plan.get("display_labels", {}) if isinstance(plan, dict) else None
    if not isinstance(labels, dict):
        raise ValueError("bound plan display_labels must be an object")
    label = labels.get(column) or variable.description or column
    if not isinstance(label, str) or not label.strip():
        raise ValueError("variable display label must be nonempty text")
    unit = variable.unit.strip() if variable.unit else None
    return BoundVariableDisplay(column, label.strip(), unit, context_sha, plan_sha)
