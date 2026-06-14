# EasyICU examples

Runnable scripts that demonstrate the two layers of EasyICU. **Start with the
quickstart** — it shows the one rule that trips up most new API users: every
extraction API expects a **prepared (converted)** directory, never a raw
download.

Run any script from the repo root after `pip install -e ".[all]"`.

## Start here — core Python API (no LLM key needed)

| Script | What it shows |
|--------|---------------|
| [`quickstart_convert_and_load.py`](quickstart_convert_and_load.py) | The shortest correct path: **convert raw data → `load_concepts(...)`**. Read this first. |

## Research-agent layer (require an LLM API key)

These drive the optional evidence-bound research agent and call a real LLM, so
they need a provider key in the environment (e.g. `OPENROUTER_API_KEY` /
`OPENAI_API_KEY`). See each file's module docstring for the exact env vars.

| Script | What it shows |
|--------|---------------|
| [`clean_cohort_demo.py`](clean_cohort_demo.py) | Generates a clean synthetic cohort (no SOFA=0 artefact) and runs the full pipeline end-to-end. |
| [`research_agent_mortality_sofa.py`](research_agent_mortality_sofa.py) | Minimal end-to-end research-agent demo (mortality ~ SOFA). |
| [`research_agent_full_paper.py`](research_agent_full_paper.py) | Full paper run: WriterAgent produces a complete manuscript → PDF. |
| [`research_agent_freeform_cluster.py`](research_agent_freeform_cluster.py) | Free-form clustering: no skill, the agent decides everything. |
| [`research_agent_pattern_audit_demo.py`](research_agent_pattern_audit_demo.py) | The analysis-pattern auditor running on free-form clustering code. |
| [`research_agent_openrouter_paper.py`](research_agent_openrouter_paper.py) | Full paper via an OpenRouter free model. |
| [`research_agent_openhands.py`](research_agent_openhands.py) | Run the agent against an OpenHands-style runtime image. |

> The research-agent scripts are working examples kept close to ongoing
> development; flags and model names may change. The quickstart is the stable
> onboarding entry point.
