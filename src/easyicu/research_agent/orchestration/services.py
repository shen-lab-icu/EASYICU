"""Live collaborators injected into the research-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PipelineServices:
    """Non-serializable collaborators used by one pipeline instance.

    Configuration describes what a run is allowed to do. Services are the
    concrete objects that do it: provider clients, review control planes,
    runner factories, and optional plugin registries. Keeping these objects
    outside ``PipelineConfig`` prevents a live connection pool or lock from
    becoming part of a supposedly replayable configuration object.
    """

    llm: Optional[Any] = None
    vlm_client: Optional[Any] = None
    visual_qa_adapter: Optional[Any] = None
    llm_concept_auditor_client: Optional[Any] = None
    human_review_gate: Optional[Any] = None
    runner_factory: Optional[Any] = None
    case_plugin_registry: Optional[Any] = None
    provider_hard_stop: Optional[Any] = None

    PROVIDER_COLLABORATOR_FIELDS: ClassVar[Tuple[str, ...]] = (
        "llm",
        "vlm_client",
        "visual_qa_adapter",
        "llm_concept_auditor_client",
    )

    def map_provider_collaborators(
        self,
        mapper: Callable[[str, Any], Any],
    ) -> "PipelineServices":
        """Project every collaborator that may reach a model Provider.

        The services owner declares this complete set so constrained runtimes
        do not have to rediscover Provider-bearing fields independently.
        """

        return replace(
            self,
            **{
                name: mapper(name, getattr(self, name))
                for name in self.PROVIDER_COLLABORATOR_FIELDS
            },
        )

    @classmethod
    def split_legacy_kwargs(
        cls,
        kwargs: Mapping[str, Any],
        *,
        services: Optional["PipelineServices"] = None,
    ) -> Tuple["PipelineServices", Dict[str, Any]]:
        """Separate live collaborators from legacy flat constructor options."""
        remaining = dict(kwargs)
        supplied = {
            field.name: remaining.pop(field.name)
            for field in fields(cls)
            if field.name in remaining
        }
        if services is not None and supplied:
            names = ", ".join(sorted(supplied))
            raise TypeError(
                "Pass live collaborators through PipelineServices only; "
                f"received both services= and legacy option(s): {names}"
            )
        return services or cls(**supplied), remaining

    def canonical_payload(self) -> Dict[str, Optional[str]]:
        """Return type identities only; never serialize live object state."""
        return {
            field.name: (
                None
                if (value := getattr(self, field.name)) is None
                else f"{type(value).__module__}.{type(value).__qualname__}"
            )
            for field in fields(self)
        }


__all__ = ["PipelineServices"]
