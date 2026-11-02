"""Lock CI's hand-maintained dependency floors to pyproject's declaration.

research_agent_ci.yml installs an explicit package list so a missing metadata
entry cannot hide an import error. That list is a second source of truth, and it
had already drifted: CI installed ``pyarrow>=14`` while the package declares
``pyarrow>=23``, so the agent suite could go green on a version the package
forbids. This test fails the moment the two disagree again.
"""

from __future__ import annotations

from pathlib import Path
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
AGENT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "research_agent_ci.yml"

_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?P<spec>.*)$"
)
_FLOOR_RE = re.compile(r">=\s*(?P<floor>[0-9][0-9A-Za-z.\-]*)")


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _declared_floors() -> dict[str, tuple[int, ...]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requirements: list[str] = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        requirements.extend(extra)

    floors: dict[str, tuple[int, ...]] = {}
    for requirement in requirements:
        match = _REQUIREMENT_RE.match(requirement.strip())
        if not match:
            continue
        floor_match = _FLOOR_RE.search(match.group("spec"))
        if not floor_match:
            continue
        name = match.group("name").lower().replace("_", "-")
        floors[name] = _version_tuple(floor_match.group("floor"))
    return floors


def _core_declared_floors() -> dict[str, tuple[int, ...]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    floors: dict[str, tuple[int, ...]] = {}
    for requirement in data["project"].get("dependencies", []):
        match = _REQUIREMENT_RE.match(requirement.strip())
        if not match:
            continue
        floor_match = _FLOOR_RE.search(match.group("spec"))
        if floor_match is None:
            continue
        name = match.group("name").lower().replace("_", "-")
        floors[name] = _version_tuple(floor_match.group("floor"))
    return floors


def _workflow_floors() -> dict[str, tuple[int, ...]]:
    text = AGENT_WORKFLOW.read_text(encoding="utf-8")
    floors: dict[str, tuple[int, ...]] = {}
    for quoted in re.findall(r'"([A-Za-z0-9._\-\[\]]+\s*[<>=!,. 0-9A-Za-z]*)"', text):
        match = _REQUIREMENT_RE.match(quoted.strip())
        if not match:
            continue
        floor_match = _FLOOR_RE.search(match.group("spec"))
        if not floor_match:
            continue
        name = match.group("name").lower().replace("_", "-")
        floors[name] = _version_tuple(floor_match.group("floor"))
    return floors


def test_agent_ci_floors_are_not_below_pyproject() -> None:
    declared = _declared_floors()
    workflow = _workflow_floors()

    assert workflow, "parsed no pinned requirements from research_agent_ci.yml"

    violations = [
        f"{name}: CI installs >={workflow[name]} but pyproject requires >={declared[name]}"
        for name in sorted(workflow)
        if name in declared and workflow[name] < declared[name]
    ]
    assert not violations, "CI dependency floors drifted below pyproject:\n" + "\n".join(
        violations
    )


def test_agent_ci_explicitly_installs_every_core_dependency() -> None:
    declared = _core_declared_floors()
    workflow = _workflow_floors()

    missing = sorted(set(declared) - set(workflow))
    assert not missing, (
        "research_agent CI uses --no-deps, so every core dependency must be "
        f"installed explicitly; missing: {missing}"
    )


def test_pyarrow_floor_is_locked_in_both_places() -> None:
    """The specific pair that had already drifted."""

    assert _declared_floors()["pyarrow"] == _workflow_floors()["pyarrow"]


def test_agent_ci_installs_the_mcp_transport_test_runtime() -> None:
    """The agent suite imports the optional official MCP SDK at collection."""

    declared = _declared_floors()
    workflow = _workflow_floors()

    assert workflow.get("mcp") == declared["mcp"], (
        "research_agent CI runs test_mcp_transport.py under --no-deps, so its "
        "explicit dependency list must install the declared MCP SDK floor"
    )


def test_agent_ci_installs_the_provider_boundary_test_runtime() -> None:
    """Provider ingress tests construct the real FastAPI application."""

    declared = _declared_floors()
    workflow = _workflow_floors()

    assert workflow.get("fastapi") == declared["fastapi"], (
        "research_agent CI runs provider-boundary TestClient tests under "
        "--no-deps, so its explicit list must install the FastAPI floor"
    )


def test_agent_ci_installs_the_webserver_owner_runtime() -> None:
    """Agent tests import WebApp owners whose Copilot gateway imports psutil."""

    declared = _declared_floors()
    workflow = _workflow_floors()

    assert workflow.get("psutil") == declared["psutil"], (
        "research_agent CI collects tests that import easyicu.webserver owners "
        "under --no-deps; the Copilot gateway imports psutil at module load, so "
        "the explicit list must install the declared psutil floor"
    )


def test_python_floor_matches_the_ci_matrices() -> None:
    """The package supports Python 3.11+, the version the desktop app and the
    runner image ship; CI must not spend a matrix leg on an unsupported 3.10."""

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.11"
    for workflow in (AGENT_WORKFLOW, REPO_ROOT / ".github" / "workflows" / "ci.yml"):
        assert '"3.10"' not in workflow.read_text(encoding="utf-8"), workflow.name
