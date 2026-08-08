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

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def test_no_test_module_imports_tomllib_without_a_310_fallback() -> None:
    """`requires-python = ">=3.10"` but tomllib is stdlib only from 3.11.

    Three test modules imported it unconditionally, so the entire Python 3.10
    matrix leg died at collection -- before a single test ran, on a version the
    package publicly claims to support. The failure is invisible locally
    (development runs 3.11+), which is exactly why it needs a test.
    """

    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "import tomllib" not in text:
            continue
        if "import tomli as tomllib" not in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "import tomllib inside try/except ModuleNotFoundError with a "
        "`import tomli as tomllib` fallback, or Python 3.10 CI dies at "
        f"collection: {offenders}"
    )


def test_the_tomli_fallback_is_actually_installed_on_310() -> None:
    """A fallback to a package nobody installs is not a fallback."""

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(
        req.startswith("tomli") and "python_version" in req for req in dev
    ), "dev extra must ship tomli for the Python versions lacking tomllib"

    workflow = AGENT_WORKFLOW.read_text(encoding="utf-8")
    assert "tomli" in workflow, (
        "research_agent CI installs an explicit list without extras, so the "
        "fallback has to be named there too"
    )


def test_optional_sksurv_adapter_does_not_break_python_310_resolution() -> None:
    """scikit-survival 0.28 requires Python 3.11, while core supports 3.10."""

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    adapters = data["project"]["optional-dependencies"]["scientific-adapters"]
    requirement = next(
        item for item in adapters if item.startswith("scikit-survival")
    )
    assert "python_version >= '3.11'" in requirement
