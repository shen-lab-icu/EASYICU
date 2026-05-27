# Research-agent baselines registry (O14)

This file tracks the external research/science-agent projects that
EasyICU's `research_agent` is positioned against in the paper's
Methods section. Entries are *not* vendored; the registry records
the repository URL, a pinned commit / release, the comparison axis,
and how to fetch the repo for a local A/B if needed.

**Usage:**

* This file is the source of truth. Keep entries short and factual.
* To fetch any entry locally run
  `python tools/fetch_baselines.py --name <name>`; the tool reads the
  same YAML block at the end of this file.
* Do not commit fetched baselines into the EasyICU repo. Fetches
  land under `baselines/_checkouts/` which is git-ignored.

---

## Legend

* **category** — where the project sits in the landscape:
  * `science-agent` — end-to-end idea → paper loops (AI-Scientist,
    ResearchAgent, AgentLaboratory).
  * `medical-agent` — ICU / biomedical focused agents (HealthFlow,
    OpenLens, M4, Biomni, MedAgents).
  * `literature` — retrieval + QA (PaperQA2, ScienceBeam).
  * `discovery-bench` — external scorecards for the research-agent
    paper (ScienceAgentBench, DiscoveryBench, CORE-Bench,
    MLE-Bench, DABStep, BLADE, EHRFlowBench).
  * `engineering` — runner / planner / orchestration baselines
    (SWE-agent, DataInterpreter, OpenHands).
  * `causal` — causal-inference libraries (not agents) we may pull
    into the CausalSkill stack (DoWhy, EconML, causallib, lifelines).

* **axis** — which part of EasyICU's contribution the baseline
  contrasts against:
  * `safety` — ICU pitfall rules + concept audit
  * `evidence` — SHA-256 evidence binding, manuscript provenance
  * `benchmark` — external scorecards
  * `planner` — analysis planning vs dynamic replanning
  * `reviewer` — reviewer / critic loop
  * `literature` — search / ground of claims
  * `causal` — identification strategy + sensitivity
  * `repro` — seed / prompt hash / lockfile

---

## Entries

### science-agent / safety, reviewer

#### SakanaAI/AI-Scientist (v1)

* **repo:** https://github.com/SakanaAI/AI-Scientist
* **pin:** `main @ 2024-10-01` (first release)
* **contrast:** AI-Scientist runs idea → experiment → paper → review.
  EasyICU uses the same three-role reviewer shape (O15) but keeps
  every generated number tied to a SHA-256 evidence record; AI-
  Scientist has no such binding and does not enforce ICU aggregation
  rules.
* **notes:** Does not run on medical tabular data out of the box.

#### SakanaAI/AI-Scientist-v2

* **repo:** https://github.com/SakanaAI/AI-Scientist-v2
* **pin:** `96bd51617cfdbb494a9fc283af00fe090edfae48`
* **contrast:** v2 adds a LangGraph reviewer loop and figure-
  reflection. We cite it as the direct inspiration for EasyICU's
  O15 (reviewer) and O3 (VLM visual QA) paths; EasyICU differs in
  that the reviewer is deterministic by default and the figure
  contract is claim-first.

#### Technion-Kishony-lab/data-to-paper

* **repo:** https://github.com/Technion-Kishony-lab/data-to-paper
* **pin:** `81df14c4b9600466e645c3b2b336cc54daa3df3a`
* **contrast:** End-to-end data analysis to manuscript with numeric
  hyperlinking. EasyICU's value-level NumericClaim registry and
  derived-claim API are compared against this mechanism while keeping
  ICU concept rails and deterministic gates.

#### KAIST-AILab/ResearchAgent

* **repo:** https://github.com/KAIST-AILab/ResearchAgent
* **pin:** `main @ 2024-05`
* **contrast:** Iterative hypothesis generation with peer review from
  an LLM panel. EasyICU's HypothesisBlueprint layer is inspired by
  this; we add ICU domain gates and concept-feasibility checks.

#### SamuelSchmidgall/AgentLaboratory

* **repo:** https://github.com/SamuelSchmidgall/AgentLaboratory
* **pin:** `main @ 2025`
* **contrast:** Human-in-the-loop science agent. We stay autonomous by
  default; HITL is an opt-in extension via the webapp page.

---

### medical-agent / safety, evidence

#### HealthFlow

* **repo:** https://github.com/ylab-open/HealthFlow
* **pin:** `45dab966959b6730ec20e8f3b3e22998735523ac`
* **contrast:** Self-evolving meta-planner with EHRFlowBench. EasyICU
  borrows the meta-planner digest (O10 / `RunMemory`) and targets
  EHRFlowBench via the external JSONL adapter (O12).

#### jarrycyx/openlens-ai

* **repo:** https://github.com/jarrycyx/openlens-ai
* **pin:** `ddb05b90e78517c1ee16885e711c9a61a73e7bf7`
* **contrast:** 5-agent pipeline (planner / coder / analyzer / writer
  / supervisor) with OpenHands, LaTeX export, Tavily search, VLM
  feedback. EasyICU mirrors the agent shape and adds the ICU
  concept context + SHA-256 evidence binding.

#### hannesill/m4

* **repo:** https://github.com/hannesill/m4
* **pin:** `312417196e970ea5678000bbeb0d2c7397eac63b`
* **contrast:** MCP server exposing MIMIC-IV / eICU skills. EasyICU's
  `mcp_server.py` (O2) and skill registry (`skills.py`) are M4-
  inspired; differs in the deterministic concept audit + evidence
  binding that sit in front of every tool.

#### snap-stanford/Biomni

* **repo:** https://github.com/snap-stanford/Biomni
* **pin:** `main @ 2025`
* **contrast:** Biomedical tool library with hundreds of skills. We
  target Biomni's skill-registry shape as inspiration for future
  ClinicalSkill marketplace work (O29).

---

### literature / literature

#### Future-House/paper-qa (PaperQA2)

* **repo:** https://github.com/Future-House/paper-qa
* **pin:** `v5 @ 2025`
* **contrast:** Retrieval-augmented QA over scientific PDFs with
  verification. EasyICU's `LiteratureAgent` is lighter but shares
  the "every claim binds to a citation" principle; we plan an
  integration hook (future).

---

### discovery-bench / benchmark

#### OSU-NLP-Group/ScienceAgentBench

* **repo:** https://github.com/OSU-NLP-Group/ScienceAgentBench
* **pin:** `main @ 2025`
* **contrast:** 102 real-paper tasks. Used as external scorecard
  against our O14 benchmark matrix.

#### allenai/discoverybench

* **repo:** https://github.com/allenai/discoverybench
* **pin:** `main @ 2024-06`
* **contrast:** Semi-structured discovery tasks. Same role in the
  matrix as ScienceAgentBench.

#### snap-stanford/core_bench

* **repo:** https://github.com/snap-stanford/core_bench
* **pin:** `main @ 2025`
* **contrast:** 270+ published-paper end-to-end reproduction. Medical
  subset is the most direct test of EasyICU's evidence binding.

#### openai/mle-bench

* **repo:** https://github.com/openai/mle-bench
* **pin:** `main @ 2024-11`
* **contrast:** Kaggle-style ML engineering baseline.

#### InfiAgent/DABench

* **repo:** https://github.com/InfiAgent/DAAgent
* **pin:** `main @ 2024-06`
* **contrast:** Data-analysis agent bench. Medical subset maps
  cleanly to our internal 6 items.

#### ColumbiaDSI/BLADE

* **repo:** https://github.com/columbia-dsi/blade
* **pin:** `main @ 2024-10`
* **contrast:** Data-analysis LLM benchmark from Columbia.

---

### engineering / planner, runner

#### All-Hands-AI/OpenHands

* **repo:** https://github.com/All-Hands-AI/OpenHands
* **pin:** `main @ 2026-03`
* **contrast:** Sandbox runner. EasyICU's `DockerRunner` and the
  `runner_factory` kwarg target OpenHands as a drop-in.

#### geekan/MetaGPT (DataInterpreter)

* **repo:** https://github.com/geekan/MetaGPT
* **pin:** `main @ 2025`
* **contrast:** Plan-as-graph with dynamic replan. EasyICU's
  ReplannerAgent is inspired by DataInterpreter's recovery path.

#### princeton-nlp/SWE-agent

* **repo:** https://github.com/princeton-nlp/SWE-agent
* **pin:** `main @ 2025`
* **contrast:** Software-engineering agent baseline.

---

### causal / causal

Not agents, but default components we may pull into the CausalSkill
stack (referenced by O18's `identification_strategy` metadata path):

* **py-why/dowhy** — https://github.com/py-why/dowhy
* **microsoft/EconML** — https://github.com/microsoft/EconML
* **BioMedIA/causallib** — https://github.com/BiomedSciAI/causallib
* **CamDavidsonPilon/lifelines** — https://github.com/CamDavidsonPilon/lifelines
* **raphaelvallat/pingouin** — https://github.com/raphaelvallat/pingouin

---

## Machine-readable index

Parsed by `tools/fetch_baselines.py`.

```yaml
entries:
  - name: ai-scientist-v1
    repo: https://github.com/SakanaAI/AI-Scientist
    ref: main
    category: science-agent
    axis: [safety, reviewer]
  - name: ai-scientist-v2
    repo: https://github.com/SakanaAI/AI-Scientist-v2
    ref: 96bd51617cfdbb494a9fc283af00fe090edfae48
    category: science-agent
    axis: [reviewer]
  - name: data-to-paper
    repo: https://github.com/Technion-Kishony-lab/data-to-paper
    ref: 81df14c4b9600466e645c3b2b336cc54daa3df3a
    category: science-agent
    axis: [evidence]
  - name: research-agent-kaist
    repo: https://github.com/KAIST-AILab/ResearchAgent
    ref: main
    category: science-agent
    axis: [planner, reviewer]
  - name: agent-laboratory
    repo: https://github.com/SamuelSchmidgall/AgentLaboratory
    ref: main
    category: science-agent
    axis: [reviewer]
  - name: healthflow
    repo: https://github.com/yhzhu99/HealthFlow
    ref: 45dab966959b6730ec20e8f3b3e22998735523ac
    category: medical-agent
    axis: [planner, benchmark]
  - name: openlens-ai
    repo: https://github.com/jarrycyx/openlens-ai
    ref: ddb05b90e78517c1ee16885e711c9a61a73e7bf7
    category: medical-agent
    axis: [planner, literature]
  - name: m4
    repo: https://github.com/hannesill/m4
    ref: 312417196e970ea5678000bbeb0d2c7397eac63b
    category: medical-agent
    axis: [safety, planner]
  - name: biomni
    repo: https://github.com/snap-stanford/Biomni
    ref: main
    category: medical-agent
    axis: [safety]
  - name: paper-qa
    repo: https://github.com/Future-House/paper-qa
    ref: main
    category: literature
    axis: [literature]
  - name: science-agent-bench
    repo: https://github.com/OSU-NLP-Group/ScienceAgentBench
    ref: main
    category: discovery-bench
    axis: [benchmark]
  - name: discoverybench
    repo: https://github.com/allenai/discoverybench
    ref: main
    category: discovery-bench
    axis: [benchmark]
  - name: core-bench
    repo: https://github.com/snap-stanford/core_bench
    ref: main
    category: discovery-bench
    axis: [benchmark]
  - name: mle-bench
    repo: https://github.com/openai/mle-bench
    ref: main
    category: discovery-bench
    axis: [benchmark]
  - name: dabench
    repo: https://github.com/InfiAgent/DAAgent
    ref: main
    category: discovery-bench
    axis: [benchmark]
  - name: blade
    repo: https://github.com/columbia-dsi/blade
    ref: main
    category: discovery-bench
    axis: [benchmark]
  - name: openhands
    repo: https://github.com/All-Hands-AI/OpenHands
    ref: main
    category: engineering
    axis: [planner]
  - name: metagpt
    repo: https://github.com/geekan/MetaGPT
    ref: main
    category: engineering
    axis: [planner]
  - name: swe-agent
    repo: https://github.com/princeton-nlp/SWE-agent
    ref: main
    category: engineering
    axis: [planner]
  - name: dowhy
    repo: https://github.com/py-why/dowhy
    ref: main
    category: causal
    axis: [causal]
  - name: econml
    repo: https://github.com/microsoft/EconML
    ref: main
    category: causal
    axis: [causal]
  - name: causallib
    repo: https://github.com/BiomedSciAI/causallib
    ref: main
    category: causal
    axis: [causal]
  - name: lifelines
    repo: https://github.com/CamDavidsonPilon/lifelines
    ref: main
    category: causal
    axis: [causal]
```
