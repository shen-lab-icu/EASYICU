# Research Know-How MVP

EasyICU has an opt-in, offline retrieval layer for source-backed ICU research
design candidates. It is disabled by default and does not add execution tools,
query external databases, or modify a cohort, time zero, exclusion rule, or
estimand automatically.

## Enable it

```python
from easyicu.research_agent import PipelineConfig, ResearchAgentPipeline

config = PipelineConfig(
    workdir="./research_output",
    enable_know_how=True,
    know_how_top_k=3,
    know_how_min_score=0.15,
)
pipeline = ResearchAgentPipeline.from_config(config)
```

`know_how_paths` may contain additional JSON files or directories. Card ids
must be unique across built-in and additional paths. Retrieval is deterministic
and uses the research question, inferred analysis family, database tag, and
available `ResearchContext` concepts. It does not call an LLM or the network.

## Runtime artifacts

- `know_how_retrieval.json` records the query, matching scores and reasons,
  unresolved concepts, card versions, citation ids, and source-file SHA-256.
- `know_how_prompt.md` is the exact bounded text supplied to Planner.
- `analysis_plan.json` contains `know_how_refs` only when Planner explicitly
  adopts one or more cards retrieved in that run.

Planner may cite only selected cards. Replanner and resume preserve adopted
references exactly. A changed source card or changed persisted retrieval
artifact fails closed.

## Built-in scope

The MVP ships eight `curated_mvp` cards: AKI onset prediction, sepsis
prognosis, lactate trajectories, vasopressor comparative effectiveness,
mechanical-ventilation liberation, ICU mortality prediction, longitudinal ICU
phenotyping, and cross-database external validation. `curated_mvp` means the
sources and structure were curated for this implementation; it does not claim
expert consensus review.
