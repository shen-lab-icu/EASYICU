"""Config-driven experiment specs for the research-agent runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..schema import TimeWindow


class CohortInputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cohort: str
    cohort_name: str = "cohort"
    database: str = "miiv"
    target_outcome: Optional[str] = None
    cross_database_validation: List[str] = Field(default_factory=list)
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    id_columns: List[str] = Field(default_factory=list)
    time_columns: List[str] = Field(default_factory=list)
    outcome_columns: List[str] = Field(default_factory=list)
    time_windows: List[TimeWindow] = Field(default_factory=list)
    concept_descriptions: Dict[str, str] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: str = "./research_output"
    # Kept in step with PipelineConfig.timeout_seconds; see the note there.
    timeout_seconds: float = 900.0
    standard_executor_timeout_seconds: float = 3_600.0
    manuscript_language: str = "en"
    stop_after_analysis: bool = False
    enable_literature: bool = True
    enable_visual_qa: bool = True
    enable_vlm_visual_qa: Optional[bool] = None
    enable_llm_concept_audit: Optional[bool] = None
    enable_memory: bool = True
    enable_latex: bool = True
    enable_probe_step: bool = True
    enable_replanning: bool = True
    enable_deterministic_code_fallback: bool = False
    enable_deterministic_planner_fallback: bool = False
    context_top_k: Optional[int] = None
    max_code_repair_attempts: int = 1
    max_concurrent_steps: int = 1


class ExperimentSpec(BaseModel):
    """Portable YAML/JSON experiment spec for a reproducible run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.research_experiment/1"
    question: Optional[str] = None
    skill: Optional[str] = None
    cohort: CohortInputSpec
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    manuscript_title: Optional[str] = None
    manuscript_authors: List[str] = Field(default_factory=list)

    def pipeline_kwargs(self) -> Dict[str, Any]:
        runtime = self.runtime
        return {
            "workdir": runtime.workdir,
            "timeout_seconds": runtime.timeout_seconds,
            "standard_executor_timeout_seconds": (
                runtime.standard_executor_timeout_seconds
            ),
            "manuscript_language": runtime.manuscript_language,
            "enable_literature": runtime.enable_literature,
            "enable_visual_qa": runtime.enable_visual_qa,
            "enable_vlm_visual_qa": runtime.enable_vlm_visual_qa,
            "enable_llm_concept_audit": runtime.enable_llm_concept_audit,
            "enable_memory": runtime.enable_memory,
            "enable_latex": runtime.enable_latex,
            "enable_probe_step": runtime.enable_probe_step,
            "enable_replanning": runtime.enable_replanning,
            "enable_deterministic_code_fallback": runtime.enable_deterministic_code_fallback,
            "enable_deterministic_planner_fallback": runtime.enable_deterministic_planner_fallback,
            "context_top_k": runtime.context_top_k,
            "max_code_repair_attempts": runtime.max_code_repair_attempts,
            "max_concurrent_steps": runtime.max_concurrent_steps,
        }

    def run_kwargs(self) -> Dict[str, Any]:
        cohort = self.cohort
        return {
            "question": self.question,
            "skill": self.skill,
            "cohort": cohort.cohort,
            "cohort_name": cohort.cohort_name,
            "database": cohort.database,
            "target_outcome": cohort.target_outcome,
            "cross_database_validation": cohort.cross_database_validation,
            "inclusion_criteria": cohort.inclusion_criteria,
            "exclusion_criteria": cohort.exclusion_criteria,
            "id_columns": cohort.id_columns or None,
            "time_columns": cohort.time_columns or None,
            "outcome_columns": cohort.outcome_columns or None,
            "time_windows": cohort.time_windows or None,
            "concept_descriptions": cohort.concept_descriptions or None,
            "user_preferences": cohort.user_preferences or None,
            "notes": cohort.notes,
            "manuscript_title": self.manuscript_title,
            "manuscript_authors": self.manuscript_authors or None,
            "manuscript_language": self.runtime.manuscript_language,
            "stop_after_analysis": self.runtime.stop_after_analysis,
        }


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    spec_path = Path(path)
    text = spec_path.read_text(encoding="utf-8")
    if spec_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    return ExperimentSpec.model_validate(payload)


def dump_experiment_spec(spec: ExperimentSpec, path: str | Path) -> Path:
    spec_path = Path(path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    if spec_path.suffix.lower() == ".json":
        spec_path.write_text(
            spec.model_dump_json(indent=2),
            encoding="utf-8",
        )
    else:
        spec_path.write_text(
            yaml.safe_dump(
                spec.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return spec_path


def build_pipeline_from_spec(
    spec: ExperimentSpec,
    *,
    llm: Any,
    pipeline_cls: Any,
) -> Tuple[Any, Dict[str, Any]]:
    pipeline = pipeline_cls(llm=llm, **spec.pipeline_kwargs())
    return pipeline, spec.run_kwargs()


__all__ = [
    "CohortInputSpec",
    "RuntimeSpec",
    "ExperimentSpec",
    "load_experiment_spec",
    "dump_experiment_spec",
    "build_pipeline_from_spec",
]
