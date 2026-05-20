[**中文版 README**](README_zh.md)

# EasyICU

> A reproducible infrastructure for cross-database ICU research, with standardized concept extraction, clinician-friendly web workflows, and scriptable Python APIs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/shen-lab-icu/easyicu)

EasyICU is a Python toolkit for intensive care unit (ICU) data analysis. It provides unified access to **6 major public ICU databases**, automated extraction of **200+ standardized clinical concepts** (the canonical web-side catalog exposes **217** — 204 dictionary concepts plus 13 special concepts: 10 KDIGO AKI staging outputs, 2 circulatory-failure indicators, and the Sepsis-3 SOFA-1 diagnosis, all loadable through the same `load_concepts(...)` call), and a **web-based interface** for cohort definition, feature review, visualization, and export.

## Why EasyICU

- **One concept layer across six public ICU databases**: EasyICU uses clinical concepts rather than database-specific variable lists, making cross-database analysis easier to write, review, and reuse.
- **Reproducible from both code and UI**: the same prepared data can be used in the web app and in Python scripts or notebooks.
- **Validated on clinically meaningful use cases**: the framework includes automated computation of **SOFA-2**, alongside standardized concepts, domain-specific loaders, and cohort analytics.

## Who This Repository Is For

- **Reviewers and clinical researchers** who need to quickly understand what the project contributes and how it can support ICU research workflows.
- **Web users** who want to validate data, define cohorts, inspect features, and export results without writing code.
- **Python users** who want to build scripted, reproducible extraction and feature engineering pipelines.

## Start Here

### Quick Lookup: "I want to ... → run ..."

| Goal | Entry point |
|------|-------------|
| Validate data / define cohorts / export features without writing Python | **Web app** — `easyicu-webapp` *(or `./start_easyicu.sh` / `start_easyicu.command`)* — see **Path A** |
| Build a reproducible extraction or feature pipeline in Python | **Python API** — `import easyicu` — see **Path B** |
| Extract features via CLI (scripted, no UI) | `easyicu` (the `extract_features` console script) |
| Run the research-agent on a question + cohort | `easyicu-research-agent` |
| Reproduce an external paper through the agent | `easyicu-research-replication` |
| Host the LLM proxy used by the research-agent | `easyicu-llm-server` |

All console scripts are declared in `pyproject.toml` under `[project.scripts]` and become available after `pip install -e ".[dev,webapp]"` (or `".[all]"`).

### Path A: Web Interface

Choose this path if you want to:
- launch EasyICU quickly
- validate and prepare data visually
- define cohorts and export features without writing Python

Recommended entry:
- Double-click `start_easyicu.bat` on Windows
- Double-click `start_easyicu.command` on macOS
- Run `./start_easyicu.sh` on Linux

First launch will create a local runtime under `.easyicu-runtime/` and
install the web dependencies automatically. Use Python 3.10+ for this
path.

Default local URL:

```text
http://127.0.0.1:8501
```

### Path B: Python API

Choose this path if you want to:
- call EasyICU from scripts or notebooks
- automate feature extraction
- build reproducible cohort pipelines in code

Minimal install:

Python 3.10+ is recommended for the current packaged dependencies.

```bash
git clone "https://github.com/shen-lab-icu/easyicu.git"
cd easyicu
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

Launch the web app manually if needed:

```bash
easyicu-webapp
```

## Reproducibility & Safety

- **Prepared data is the shared contract**: raw CSV / CSV.GZ / tar.gz dumps should be converted first, then reused by both the web interface and Python APIs.
- **AI assistant is opt-in**: the assistant starts disabled until a user explicitly enables it in the sidebar.
- **Human confirmation stays in the loop**: cohort, feature, conversion, and export actions still require explicit confirmation.
- **Automated checks are included**: `pytest` and GitHub Actions provide baseline repository and rendering checks for the packaged workflows.

## Paper, Citation & Reproducibility

- **Software citation**: GitHub citation metadata is provided in [CITATION.cff](CITATION.cff).
- **Repository entry point for reviewers**: this README is structured to show the project contribution, supported databases, and the shortest reproducible usage paths first.
- **Reproducible execution paths**: use the one-click launcher for the web workflow, or use the Python API examples below after running data preparation.
- **Manuscript link**: add the journal article or preprint URL here once it is public.

## Supported Public ICU Databases

| Database | URL |
|----------|-----|
| MIMIC-III | https://physionet.org/content/mimiciii/ |
| MIMIC-IV | https://physionet.org/content/mimiciv/ |
| eICU-CRD | https://physionet.org/content/eicu-crd/ |
| AmsterdamUMCdb | https://amsterdammedicaldatascience.nl/ |
| HiRID | https://hirid.intensivecare.ai/ |
| SICdb | https://physionet.org/content/sicdb/ |

## Web Workflow At A Glance

1. **Obtain ICU data** and place it in a local directory.
2. **Validate Data Path** in the web app.
3. If raw files are detected, use **Convert & Setup** to prepare the dataset.
4. Define the research cohort, select features, and export the result.
5. Use the built-in visualization and cohort analysis views for review.

### Mode Selection

On launch, EasyICU asks the user to pick a working mode. **Demo Mode** runs a guided tour against simulated ICU data with no tokens required; **Real Data Mode** connects to a local prepared dataset (or one of the supported public databases) and runs the full extraction-and-review workflow.

![Mode Selection](docs/images/01_mode_selection.jpg)

### Data Preparation

In Real Data Mode, EasyICU validates raw database directories and prepares them automatically for extraction. The web workflow detects unsupported raw layouts such as CSV / CSV.GZ / tar.gz, converts them to Parquet, applies database-specific optimizations, and prepares the structure needed by both the web interface and Python APIs.

### Patient Review — Modules and Features

The **Patient Review** tab loads concept tables by module (Vital Signs, Labs, SOFA, Sepsis, AKI, …) and lets reviewers inspect features, time series, per-patient summaries, and a built-in data-quality audit. Each module shows its mapped raw fields and concept-level definitions, with merged or single-feature preview modes.

![Patient Review](docs/images/02_patient_review.jpg)

### Time Series Review — Clinical Lanes

The **Time Series** sub-tab supports Clinical Lanes (multi-feature dashboard with reference thresholds), Single Patient, and Multi-Patient Comparison views. Each chart overlays clinically meaningful thresholds — e.g. Tachycardia / Bradycardia on heart rate, Fever / Hypothermia on temperature, Thrombocytopenia on platelets — so reviewers can sanity-check trends at a glance.

![Time Series — Clinical Lanes](docs/images/03_clinical_lanes.jpg)

### Cohort Statistics

The **Cohort Statistics** tab produces subgroup contrast tables (with p-values and SMD), coverage & eligibility audit, a one-page cohort snapshot, and SOFA-1 vs SOFA-2 sensitivity analysis — all powered by the prepared demo or real-data state. The Baseline Characteristics table below shows per-module values for the contrasted groups with significance flags (balanced / mild / large).

![Cohort Statistics](docs/images/04_cohort_statistics.jpg)

### Cross-Database Benchmark

The **Cross-DB Benchmark** tab harmonizes the same clinical concepts across all six supported ICU databases and overlays their feature distributions for direct comparison — a key sanity check when a study aims to generalize across cohorts.

![Cross-Database Benchmark](docs/images/06_cross_db_benchmark.jpg)

## Visualization & Analysis

EasyICU's main interface is organized as 5 top-level tabs:

- **Tutorial** — data-preparation workflow guide (data source → cohort → concepts → export) shown on the leftmost tab so first-time users can find it without leaving the main pane; also reachable via the sidebar "📚 Workflow Help" button.
- **Patient Review** — data tables, time-series review with clinical thresholds, per-patient overview, and data-quality audit (missingness / out-of-physio / temporal integrity).
- **Cohort Statistics** — subgroup contrast tables (with p-values and SMD), coverage & eligibility audit, cohort one-page snapshot, and SOFA-1 vs SOFA-2 sensitivity analysis.
- **Cross-DB Benchmark** — harmonized feature-distribution comparison across multiple ICU databases (kept separate because it needs raw schema for ≥ 2 databases).
- **Research Agent** — optional analysis-and-manuscript scaffolding driven by a research question; includes a built-in deterministic replication runner for paper reproduction.

The Research Agent layer turns a question + prepared EasyICU data into an evidence-bound research output via a 4-stage pipeline — **Plan → Build → Analyze → Gate** — and only drafts the manuscript after the evidence gate passes:

![Research Agent pipeline](docs/images/05_research_agent.jpg)

## Optional Research-Agent Layer

For advanced users, EasyICU also includes an optional
`easyicu.research_agent` layer for ICU-aware analysis planning,
evidence-bound reporting, manuscript scaffold generation, and
paper-aware replication mode for retrospective ICU studies. It is not
required for the standard web workflow or Python extraction APIs.
See [src/easyicu/research_agent/README.md](src/easyicu/research_agent/README.md)
for details.

## 🚀 Going Further (Developers / Advanced Users)

### Development & Testing

Create a local development environment and run the current automated checks:

Use Python 3.10+ for the development environment.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,webapp]"
pytest -q
```

GitHub Actions runs the same `pytest` suite on pushes and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for the expected workflow when proposing changes.

## 💻 Python API

Before calling any extraction API, make sure your database has already been prepared.
Raw CSV / CSV.GZ / tar.gz dumps are not the expected input for feature extraction APIs.
Use either the Web UI **Validate Data Path** -> **Convert & Setup** flow or the programmatic conversion step below first, then pass the prepared directory to `data_path`.

### API Prerequisite: Convert Data First

The web application can prepare the data for you automatically. You can also convert it in code:

```python
from easyicu.data_converter import DataConverter

converter = DataConverter('/path/to/raw/data', database='miiv')
converter.convert_all()
```

After conversion, use the prepared directory in all API examples below.

> **Tip — slow external storage (USB / network mounts)**: the default
> `parallel_workers=4` is tuned for local SSD and can deadlock on slow
> external storage during large sharded writes (PRESCRIPTIONS, CHARTEVENTS).
> Set `EASYICU_CONV_WORKERS=1` to force single-threaded conversion:
> ```bash
> EASYICU_CONV_WORKERS=1 python convert_my_data.py
> ```
> On a 90 GB AUMC numericitems on USB this trades ~30 % wall-clock for
> guaranteed completion.

### Minimal End-to-End Example

The example below shows the full workflow for API users: start from raw data, convert it, then extract standardized features.

```python
from easyicu.data_converter import DataConverter
from easyicu import load_concepts

database = 'miiv'
raw_data_path = '/path/to/mimic-iv-raw'

# Step 1: Convert raw files into the prepared format expected by EasyICU
converter = DataConverter(raw_data_path, database=database)
converter.convert_all()

# Step 2: Extract standardized concepts from the prepared dataset
vitals = load_concepts(
    concepts=['hr', 'map', 'resp', 'spo2'],
    database=database,
    data_path=raw_data_path,
    patient_ids=[30000123, 30000456],
    interval='1h',
    aggregate='mean',
)

print(vitals.head())

# Optional: save the extracted feature table
vitals.to_parquet('miiv_vitals_1h.parquet', index=False)
```

What this example assumes:
- `raw_data_path` points to your original downloaded database directory
- conversion prepares that same directory for EasyICU loading
- after conversion, pass the prepared directory to `data_path`

### Easy API — One-Liners

```python
from easyicu import load_sofa, load_sofa2, load_vitals, load_labs

# Load SOFA scores
sofa = load_sofa(
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123, 30000456]
)

# Load SOFA-2 (2025 revised criteria)
sofa2 = load_sofa2(
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123],
    keep_components=True  # retain organ sub-scores
)

# Load vital signs
vitals = load_vitals(database='miiv', data_path='/path/to/data')

# Load laboratory results
labs = load_labs(database='miiv', data_path='/path/to/data')
```

### Concept API — Flexible & Customizable

```python
from easyicu import load_concepts

# Batch-load multiple concepts
data = load_concepts(
    concepts=['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2'],
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123],
    interval='1h',       # align to 1-hour bins
    aggregate='mean',    # aggregate with mean
    verbose=True
)

# Load Sepsis-3 diagnosis
sepsis = load_concepts(
    'sep3',
    database='miiv',
    data_path='/path/to/data'
)

# Special concepts — KDIGO AKI staging and circulatory failure are
# computed by dedicated callbacks, but you can request them through the
# same `load_concepts(...)` call. The API will route to the right
# loader transparently.
aki_and_circ = load_concepts(
    concepts=['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo',
              'aki_stage_rrt', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
              'creat_low_past_48hr', 'creat_low_past_7day',
              'circ_failure', 'circ_event'],
    database='miiv',
    data_path='/path/to/data',
)

# Whole-DB / module-style batch extraction (fastest path)
# Pass the full list of concepts in one call so the resolver can share
# base-table reads (chartevents / labevents / inputevents bucket scans)
# across all of them. `merge=False` returns a `dict[concept -> DataFrame]`
# so the result stays small even on huge cohorts.
all_features = load_concepts(
    concepts=['hr', 'sbp', 'map', 'temp', 'spo2',
              'bili', 'crea', 'lact', 'plt', 'wbc',
              'sofa', 'sofa2', 'sep3',
              'aki', 'circ_failure'],
    database='miiv',
    data_path='/path/to/data',
    merge=False,           # return dict instead of one giant merged DataFrame
)
```

> **Note on full-cohort extraction**: when you don't pass `patient_ids`
> and `max_patients`, EasyICU loads every patient in the database. On a
> 16 GB machine with limited free memory, `load_concepts(...)` may
> auto-batch into subprocess workers — set the environment variable
> `EASYICU_BATCH_TIMEOUT_SEC` (default 3600) to bound each batch in
> case a worker hangs.

### Domain-Specific Loaders

```python
from easyicu import (
    load_demographics,      # Demographics
    load_outcomes,          # Clinical outcomes
    load_vitals_detailed,   # Detailed vital signs
    load_neurological,      # Neurological assessments
    load_output,            # Fluid output
    load_respiratory,       # Respiratory parameters
    load_lab_comprehensive, # Comprehensive lab panels
    load_blood_gas,         # Arterial blood gas
    load_hematology,        # Hematology
    load_medications,       # Medications
)

# Example: load demographics
demo = load_demographics(
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123]
)
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If EasyICU is helpful for your research, please give us a Star! ⭐**

Made with ❤️ for ICU researchers worldwide

</div>
