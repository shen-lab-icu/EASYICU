[**中文版 README**](README_zh.md)

# EasyICU

> A reproducible infrastructure for cross-database ICU research, with standardized concept extraction, clinician-friendly web workflows, and scriptable Python APIs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/shen-lab-icu/easyicu)

EasyICU is a Python toolkit for intensive care unit (ICU) data analysis. It provides unified access to **6 major public ICU databases**, automated extraction of **167 standardized clinical concepts**, and a **web-based interface** for cohort definition, feature review, visualization, and export.

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

### Data Preparation

EasyICU can validate raw database directories and prepare them automatically for extraction. The web workflow detects unsupported raw layouts such as CSV / CSV.GZ / tar.gz, converts them to Parquet, applies database-specific optimizations, and prepares the structure needed by both the web interface and Python APIs.

<img width="1931" height="956" alt="Data Conversion" src="https://github.com/user-attachments/assets/86ea826b-6a0f-491a-b967-c5a7ebdfaa5b" />

### Cohort Definition

Typical filters include:
- ICU length of stay
- age range
- first ICU admission only
- sex
- in-hospital mortality

<img width="1931" height="736" alt="Cohort Selection" src="https://github.com/user-attachments/assets/628caf50-bed3-4918-b36f-5930464e9fb7" />

### Feature Review And Export

Feature selection is organized by category, with concept definitions and mapping details available in the dictionary panel. Export formats include Parquet, CSV, and Excel.

<img width="1931" height="1018" alt="Feature Selection" src="https://github.com/user-attachments/assets/f37fc262-b0e8-4894-8a08-2614614f4f18" />

<img width="4249" height="2241" alt="Batch Export" src="https://github.com/user-attachments/assets/9575d396-14ef-4e02-a4ac-a2a6222b1776" />

## Visualization & Analysis

EasyICU includes interactive tools for:

- **Quick Visualization**: data tables, time-series review, patient overview, and data-quality assessment
- **Cohort Analysis**: subgroup contrast tables, cross-database distribution review, cohort snapshots, and SOFA-1/SOFA-2 sensitivity analysis

<img width="3051" height="1823" alt="Quick Visualization Example" src="https://github.com/user-attachments/assets/09c64137-9c6a-401e-a1d0-fe358ea458de" />

## Optional Research-Agent Layer

For advanced users, EasyICU also includes an optional
`easyicu.research_agent` layer for ICU-aware analysis planning,
evidence-bound reporting, manuscript scaffold generation, and
paper-aware replication mode for retrospective ICU studies. It is not
required for the standard web workflow or Python extraction APIs.
See [src/easyicu/research_agent/README.md](src/easyicu/research_agent/README.md)
for details.

## 🚀 Going Further (Developers / Advanced Users)

## Development & Testing

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
```

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
