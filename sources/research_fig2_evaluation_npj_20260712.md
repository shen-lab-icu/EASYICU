# Research sources: Figure 2 agent evaluation and npj Digital Medicine fit

> Search date: 2026-07-12
> Purpose: primary-source audit for EasyICU Figure 2 evaluation design.
> Scope: official journal guidance and original benchmark/evaluation papers only.

## Journal and reporting guidance

1. **npj Digital Medicine — Aims and scope**
   - URL: https://www.nature.com/npjdigitalmed/aims
   - Relevance: includes innovative AI and clinical informatics, but says the journal typically does not consider off-the-shelf AI studies, purely observational work, case studies, or small preliminary studies.

2. **npj Digital Medicine — Submission guidelines**
   - URL: https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines
   - Relevance: Nature Portfolio Reporting Summary, statistics/reproducibility details, code, data, uncertainty, multiple comparisons, and complete method reporting.

3. **npj Digital Medicine — Editorial process**
   - URL: https://www.nature.com/npjdigitalmed/for-authors-and-referees/editorial-process
   - Relevance: insufficient conceptual advance and major technical or interpretational problems are explicit rejection grounds.

4. **TRIPOD-LLM reporting guideline**
   - URL: https://www.nature.com/articles/s41591-024-03425-5
   - DOI: https://doi.org/10.1038/s41591-024-03425-5
   - Relevance: 19 main items and 50 subitems; emphasizes model/prompt/interface versions, evaluation setting, human oversight, task-specific performance, transparency and reproducibility.

5. **A framework for human evaluation of LLMs in healthcare**
   - URL: https://www.nature.com/articles/s41746-024-01258-7
   - DOI: https://doi.org/10.1038/s41746-024-01258-7
   - Relevance: recommends blinded assessment, evaluator training, inter-rater agreement, and four evaluators for medical-research applications; proposes adjudication toward kappa >= 0.7.

6. **A scalable framework for evaluating health language models**
   - URL: https://www.nature.com/articles/s41746-026-02492-x
   - DOI: https://doi.org/10.1038/s41746-026-02492-x
   - Relevance: granular Boolean criteria improve evaluator reliability and make failures actionable compared with broad Likert/composite scores; validates automatic ratings against human ratings.

## Scientific/data-analysis agent benchmarks

7. **BLADE: Benchmarking Language Model Agents for Data-Driven Science**
   - URL: https://doi.org/10.18653/v1/2024.findings-emnlp.815
   - Project: https://blade-bench.github.io/
   - Design: 12 open-ended questions/datasets, 536 acceptable analysis decisions, independent expert analyses and consensus, evaluator validation, repeated runs and bootstrap 95% confidence intervals.
   - Relevance: a small number of deep tasks can be defensible when task provenance, expert ground truth, multiple valid analyses, repeated runs and uncertainty are strong.

8. **ScienceAgentBench**
   - URL: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html
   - Design: 102 tasks from 44 peer-reviewed papers, nine subject-matter experts, multiple rounds of validation, five models x three frameworks, three attempts per task, program/execution/cost metrics and contamination mitigation.
   - Relevance: independently sourced tasks, expert validation, hidden evaluation and repeated attempts are central to a general scientific-agent claim.

9. **InfiAgent-DABench**
   - URL: https://proceedings.mlr.press/v235/hu24s.html
   - Design: 603 data-analysis questions from 124 CSV files; converts open-ended questions into closed-form outputs for scalable automatic evaluation and compares 34 LLMs.
   - Relevance: illustrates the scale normally used for a formal data-analysis benchmark and the trade-off between open-ended validity and closed-form scoring.

10. **MLAgentBench**
    - URL: https://proceedings.mlr.press/v235/huang24y.html
    - Design: 13 machine-learning experiment tasks, eight runs per agent-task, hidden test labels, explicit starter baselines and time/token reporting.
    - Relevance: small task count is acceptable when repetitions, hidden evaluation and baselines are strong.

## Nature Portfolio and npj clinical-agent evaluations

11. **AgentClinic**
    - URL: https://www.nature.com/articles/s41746-026-02674-7
    - DOI: https://doi.org/10.1038/s41746-026-02674-7
    - Design: hundreds of MedQA/MIMIC-IV/NEJM/specialty/language cases, 11 models, tool and bias perturbations, real EHR scenarios, clinician reader study and open evaluation materials.
    - Relevance: npj accepted a rich interactive agent benchmark with multiple conditions, human review and explicit limitations; it did not infer generality from nine single runs.

12. **Benchmarking LLM-based agent systems for clinical decision tasks**
    - URL: https://www.nature.com/articles/s41746-026-02443-6
    - DOI: https://doi.org/10.1038/s41746-026-02443-6
    - Design: three benchmark families, multiple baseline LLMs and agent variants, hundreds of cases, 862-item HARD set, 95% intervals, paired tests/multiplicity control, token/time/workflow-complexity and hallucination analysis.
    - Relevance: the closest npj standard for attributing value to an agent architecture rather than its backbone model.

13. **Tool-wielding language-model-based agent for clinical tabular data (ChatDA)**
    - URL: https://www.nature.com/articles/s44387-025-00070-2
    - DOI: https://doi.org/10.1038/s44387-025-00070-2
    - Design: 100 sequential questions on 10 public datasets plus 11 ML datasets, four comparator agents, ten replicates, human ground truth and 95% confidence intervals.
    - Relevance: directly comparable clinical data-analysis agent paper; explicitly acknowledges custom-benchmark bias.

14. **The evaluation illusion of LLMs in medicine**
    - URL: https://doi.org/10.1038/s41746-025-01963-x
    - Relevance: warns that task/metric choice can be misaligned with real clinical utility and that automatic metrics may not replace targeted human evaluation.

## Synthesis for EasyICU

- Nine tasks are not automatically too few: BLADE and MLAgentBench show that 12–13 deep tasks can be publishable.
- The decisive requirements are independent task provenance, a frozen evaluator-side oracle, repeated runs, uncertainty, baselines or bounded claims, human validation and complete reproducibility.
- EasyICU's current five concepts (plan, execution, result validity, evidence binding, safety) are defensible, but result validity and per-task safety must be genuinely scored rather than replaced by internal-validator success.
- Because the canonical nine have been used for architecture development, they must be described as a protocol-rich development/capability suite. A small untouched held-out suite is needed for a generalization claim.
- If no baseline/ablation is run, the paper may claim an auditable capability demonstration but cannot attribute improvements specifically to the EasyICU architecture versus the same backbone in a generic agent.
