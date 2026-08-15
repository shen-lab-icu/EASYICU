"""User-installed, host-governed Skill and MCP extension boundary."""

from .contracts import (
    ExtensionActivationSnapshot,
    ExtensionRegistryError,
    McpServerActivation,
    SkillActivation,
)
from .registry import ExtensionRegistry

__all__ = [
    "ExtensionActivationSnapshot",
    "ExtensionRegistry",
    "ExtensionRegistryError",
    "McpServerActivation",
    "SkillActivation",
]
