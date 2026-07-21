"""Architecture profile for the EasyICU research-agent runtime.

This module makes the package's intended design explicit and serialisable:
the system is not a generic "AI scientist", but an ICU-aware autonomous
research runtime with distinct safety-critical layers.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class SystemLayer(str, Enum):
    ICU_DATA_FOUNDATION = "icu_data_foundation"
    SAFE_ANALYTICAL_RUNTIME = "safe_analytical_runtime"
    AGENT_ORCHESTRATION = "agent_orchestration"
    SCIENTIFIC_DISCOVERY = "scientific_discovery"


class AgentRole(str, Enum):
    PLANNER = "planner"
    CLINICAL_SEMANTICS = "clinical_semantics"
    DATA_EXTRACTION = "data_extraction"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUALIZATION = "visualization"
    MANUSCRIPT = "manuscript"
    CRITIC = "critic"
    RUNTIME_SUPERVISOR = "runtime_supervisor"


class LayerComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    layer: SystemLayer
    responsibilities: List[str] = Field(default_factory=list)


class ArchitectureProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.research_architecture/1"
    title: str = "EasyICU research-agent architecture"
    design_goal: str = (
        "ICU-aware, evidence-bound, reproducible autonomous clinical research system"
    )
    principles: List[str] = Field(default_factory=list)
    components: List[LayerComponent] = Field(default_factory=list)


def default_architecture_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        principles=[
            "ICU-native semantics before agent autonomy.",
            "Deterministic execution and evidence binding before manuscript drafting.",
            "Clinical and statistical validators gate every generated analytical artefact.",
            "Scientific discovery hooks exist, but safety and provenance remain the primary runtime contract.",
        ],
        components=[
            LayerComponent(
                name="ConceptValidationLayer",
                layer=SystemLayer.ICU_DATA_FOUNDATION,
                responsibilities=[
                    "Validate concept metadata against the unified EasyICU registry.",
                    "Expose source tables, item ids, units, and caveats without allowing raw SQL access.",
                ],
            ),
            LayerComponent(
                name="TemporalAlignmentEngine",
                layer=SystemLayer.ICU_DATA_FOUNDATION,
                responsibilities=[
                    "Parse ICU time semantics into deterministic temporal constraints.",
                    "Resolve analysis windows relative to ICU admission or anchor events.",
                ],
            ),
            LayerComponent(
                name="ICUEpisodeResolver",
                layer=SystemLayer.ICU_DATA_FOUNDATION,
                responsibilities=[
                    "Resolve patient/stay/time/outcome columns into a stable cohort descriptor.",
                    "Carry cohort provenance into every downstream run.",
                ],
            ),
            LayerComponent(
                name="EvidenceStore + validators",
                layer=SystemLayer.SAFE_ANALYTICAL_RUNTIME,
                responsibilities=[
                    "Hash, register, and bind tables/figures/statistics/scripts to evidence ids.",
                    "Run cohort, concept-use, clinical-constraint, and statistical validation gates.",
                ],
            ),
            LayerComponent(
                name="RuntimeSupervisor",
                layer=SystemLayer.AGENT_ORCHESTRATION,
                responsibilities=[
                    "Coordinate plan → execute → critique → revise loops.",
                    "Track retries, runner repairs, and phase transitions in an audit log.",
                ],
            ),
            LayerComponent(
                name="Planner / Replanner / Coder / Critic / Writer agents",
                layer=SystemLayer.AGENT_ORCHESTRATION,
                responsibilities=[
                    "Route tasks through typed, evidence-aware agents.",
                    "Keep discussion/clinical interpretation in draft mode for human review.",
                ],
            ),
            LayerComponent(
                name="Research Know-How Registry (opt-in)",
                layer=SystemLayer.AGENT_ORCHESTRATION,
                responsibilities=[
                    "Retrieve a bounded set of versioned, source-backed ICU design candidates offline.",
                    "Expose unresolved concepts and preserve Planner adoption as typed, content-hashed references without changing cohort or estimand authority.",
                ],
            ),
            LayerComponent(
                name="ScientificDiscovery hooks",
                layer=SystemLayer.SCIENTIFIC_DISCOVERY,
                responsibilities=[
                    "Reserve interfaces for iterative hypothesis refinement and future discovery loops.",
                    "Do not weaken ICU safety or provenance guarantees.",
                ],
            ),
        ],
    )


def architecture_profile_markdown(profile: ArchitectureProfile | None = None) -> str:
    profile = profile or default_architecture_profile()
    lines = [
        f"# {profile.title}",
        "",
        f"- Goal: {profile.design_goal}",
        "",
        "## Principles",
        "",
    ]
    for principle in profile.principles:
        lines.append(f"- {principle}")
    lines.extend(["", "## Layered components", ""])
    for component in profile.components:
        lines.append(f"- **{component.layer.value} / {component.name}**")
        for item in component.responsibilities:
            lines.append(f"  - {item}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "SystemLayer",
    "AgentRole",
    "LayerComponent",
    "ArchitectureProfile",
    "default_architecture_profile",
    "architecture_profile_markdown",
]
