"""Typed, case-neutral publication skills shared by Agent and Web hosts.

This module owns the small public contract for the publication layer.  The
executors remain in their existing owner modules:

* ``nature-figure`` is executed by the article figure strategy and the
  deterministic :class:`PublicationFigureSkill` renderer;
* ``nature-writing`` is executed by :class:`WriterAgent` and the downstream
  evidence, literature, numeric, and novelty audits.

Keeping the registry dependency-neutral lets Web settings describe and toggle
the same skills without importing renderer internals or copying policy logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


PUBLICATION_SKILL_REGISTRY_VERSION = "easyicu.publication-skills/1"
NATURE_FIGURE_SKILL_ID = "nature-figure"
NATURE_WRITING_SKILL_ID = "nature-writing"


@dataclass(frozen=True)
class PublicationSkillSpec:
    """Immutable public description of one built-in publication skill."""

    skill_id: str
    title: str
    version: str
    stage: str
    setting_key: str
    executor: str
    scope: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    invariants: Tuple[str, ...]
    default_enabled: bool = True

    def to_dict(self, *, enabled: bool | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.skill_id,
            "title": self.title,
            "version": self.version,
            "stage": self.stage,
            "setting_key": self.setting_key,
            "executor": self.executor,
            "scope": self.scope,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "invariants": list(self.invariants),
            "default_enabled": self.default_enabled,
        }
        if enabled is not None:
            payload["enabled"] = bool(enabled)
        return payload

    def workbench_card(self) -> Dict[str, Any]:
        """Return the stable subset rendered by Agent Science."""

        return {
            "id": self.skill_id,
            "title": self.title,
            "stage": self.stage,
            "route": "agent",
            "scope": self.scope,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "evidence": list(self.invariants),
            "setting_key": self.setting_key,
            "default_enabled": self.default_enabled,
            "version": self.version,
        }


NATURE_FIGURE_SKILL = PublicationSkillSpec(
    skill_id=NATURE_FIGURE_SKILL_ID,
    title="Nature-style evidence figure",
    version="easyicu.nature-figure/1",
    stage="Figure",
    setting_key="nature_figure_skill_enabled",
    executor=(
        "easyicu.research_agent.planning.figure_strategy + "
        "easyicu.research_agent.figures.skill.PublicationFigureSkill"
    ),
    scope=(
        "Plan a claim-first visual argument and render result-bearing figures "
        "from registered source data and code."
    ),
    inputs=(
        "article figure strategy",
        "registered result evidence",
        "source data and uncertainty definitions",
    ),
    outputs=(
        "claim-first FigureContract",
        "editable SVG/PDF plus PNG/TIFF exports",
        "source-data and visual-QA evidence",
    ),
    invariants=(
        "one-sentence core claim and panel evidence chain",
        "numeric figures are code-backed; free-form generation cannot alter results",
        "axis, unit, uncertainty, source-data, and export QA remain auditable",
    ),
)

NATURE_WRITING_SKILL = PublicationSkillSpec(
    skill_id=NATURE_WRITING_SKILL_ID,
    title="Nature-style evidence writing",
    version="easyicu.nature-writing/1",
    stage="Writing",
    setting_key="nature_writing_skill_enabled",
    executor="easyicu.research_agent.agents.core.WriterAgent",
    scope=(
        "Draft a broad-audience scientific argument with explicit paragraph "
        "roles, calibrated claims, and exact evidence/literature bindings."
    ),
    inputs=(
        "research context",
        "machine evidence digest",
        "run-bound literature digest",
    ),
    outputs=(
        "sectioned manuscript prose",
        "claim-evidence bindings",
        "missing-support, numeric, literature, and novelty audits",
    ),
    invariants=(
        "never invent results, references, methods, novelty, or statistics",
        "each paragraph has one reader-facing job and a supported claim",
        "terminology, causal language, and novelty wording remain calibrated",
    ),
)

PUBLICATION_SKILLS: Tuple[PublicationSkillSpec, ...] = (
    NATURE_FIGURE_SKILL,
    NATURE_WRITING_SKILL,
)


@dataclass(frozen=True)
class PublicationSkillActivation:
    """Run-bound activation receipt compiled from immutable config flags."""

    nature_figure_enabled: bool
    nature_writing_enabled: bool

    def enabled(self, skill: PublicationSkillSpec) -> bool:
        if skill.skill_id == NATURE_FIGURE_SKILL_ID:
            return self.nature_figure_enabled
        if skill.skill_id == NATURE_WRITING_SKILL_ID:
            return self.nature_writing_enabled
        raise KeyError(f"unknown publication skill: {skill.skill_id}")

    def to_dict(self) -> Dict[str, Any]:
        skills = [
            skill.to_dict(enabled=self.enabled(skill))
            for skill in PUBLICATION_SKILLS
        ]
        payload: Dict[str, Any] = {
            "schema_version": PUBLICATION_SKILL_REGISTRY_VERSION,
            "default_integration": True,
            "skills": skills,
            "active_skill_ids": [row["id"] for row in skills if row["enabled"]],
            "inactive_skill_ids": [
                row["id"] for row in skills if not row["enabled"]
            ],
        }
        payload["activation_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        return payload


def compile_publication_skill_activation(
    *,
    nature_figure_enabled: bool = True,
    nature_writing_enabled: bool = True,
) -> PublicationSkillActivation:
    """Compile the only value object publication consumers may inspect."""

    return PublicationSkillActivation(
        nature_figure_enabled=bool(nature_figure_enabled),
        nature_writing_enabled=bool(nature_writing_enabled),
    )


def publication_skill_flags_from_settings(
    settings: Mapping[str, Any],
) -> Dict[str, bool]:
    """Resolve Web's master switch plus per-skill switches once per run."""

    master = bool(settings.get("science_skills_enabled", True))
    return {
        "nature_figure_enabled": master
        and bool(settings.get(NATURE_FIGURE_SKILL.setting_key, True)),
        "nature_writing_enabled": master
        and bool(settings.get(NATURE_WRITING_SKILL.setting_key, True)),
    }


def publication_skill_workbench_cards() -> list[Dict[str, Any]]:
    return [skill.workbench_card() for skill in PUBLICATION_SKILLS]


__all__ = [
    "NATURE_FIGURE_SKILL",
    "NATURE_FIGURE_SKILL_ID",
    "NATURE_WRITING_SKILL",
    "NATURE_WRITING_SKILL_ID",
    "PUBLICATION_SKILLS",
    "PUBLICATION_SKILL_REGISTRY_VERSION",
    "PublicationSkillActivation",
    "PublicationSkillSpec",
    "compile_publication_skill_activation",
    "publication_skill_flags_from_settings",
    "publication_skill_workbench_cards",
]
