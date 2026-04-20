[**中文版 README**](README_zh.md)

# EasyICU

> A unified, high-performance, clinician-friendly framework for data extraction and visualization across multiple public ICU databases.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/shen-lab-icu/easyicu)

EasyICU is a Python toolkit designed for intensive care unit (ICU) data analysis. Through a database abstraction layer, it provides unified access to **6 major public ICU databases**, supports automated extraction of **167 standardized clinical concepts**, and offers a **web-based visualization interface** — enabling clinical researchers without programming backgrounds to efficiently perform cohort definition, feature selection, and data quality review.

## ✨ Key Features

**🎯 Unified Multi-Database Clinical Concept Extraction** — EasyICU treats *clinical concepts* as the fundamental unit of feature engineering, replacing traditional static variable mappings with semantic modeling. The system extracts 167 standardized clinical concepts from six major public ICU databases — MIMIC-IV, MIMIC-III, eICU-CRD, AmsterdamUMCdb, HiRID, and SICdb — and is the first to implement automated computation of the **SOFA-2** score.

**🐍 Scriptable Python API for Reproducible Pipelines** — In addition to the web interface, EasyICU provides a Python API for loading concepts, organ scores, domain-specific modules, and full-database extractions inside scripts and notebooks. This makes it practical to build repeatable cohort pipelines and feature engineering workflows in code.

**🌐 Clinician-Oriented Visual Interface** — EasyICU integrates a web-based graphical interface designed to lower the technical barrier for EHR data analysis. Clinicians can perform cohort definition, feature selection, time-window configuration, and data quality review without writing code. The system consolidates patient time-series data into a unified view, supporting perspectives from individual case review to population-level analysis.

**🤖 Embedded AI Assistant for Workflow Guidance** — EasyICU includes an integrated AI assistant that helps users translate research questions into concrete EasyICU steps. It can explain which cohort filters, modules, concepts, and scores fit a task, assist with troubleshooting, and guide users through the current workflow with context-aware suggestions.

**🛠 One-Click Data Validation, Conversion, and Setup** — EasyICU can validate raw database directories and prepare them automatically for extraction. The web workflow detects unsupported raw layouts such as CSV / CSV.GZ / tar.gz, converts them to Parquet, applies database-specific optimizations, and prepares the structure needed by both the web interface and Python APIs.

**⚡ High-Performance Computing Optimizations** — Tailored for the high-frequency, high-dimensional, and sparse nature of ICU data, EasyICU incorporates multiple performance optimization strategies to ensure stable operation on machines with as little as **16 GB of RAM**.

---

## Quick Start Guide

### Choose Your Path

#### Path A: Web Interface Users

Choose this path if you want to:
- launch the EasyICU interface quickly
- validate data, convert raw files, and export features visually
- work without writing Python code

Start from [One-Click Launcher (Recommended)](#one-click-launcher-recommended).

#### Path B: Python API / Notebook / Script Users

Choose this path if you want to:
- call EasyICU from Python scripts or notebooks
- automate extraction pipelines
- build reproducible feature engineering workflows in code

Start from [Optional: Install for Python API / Development](#optional-install-for-python-api--development), then read [Python API](#-python-api).

### One-Click Launcher (Recommended)

If users only need to open the EasyICU web interface, they do not need Anaconda or VS Code first.
If this launcher already meets your needs, you can skip the Python/API installation section below.

Requirements:
- **Python 3.9+** installed
- Internet access on the first launch to download dependencies

Launch options:
- **Windows**: double-click `start_easyicu.bat`
- **macOS**: double-click `start_easyicu.command`
- **Linux**: run `./start_easyicu.sh`

The first run will automatically:
- create a local virtual environment in `.easyicu-runtime/venv`
- install the EasyICU web dependencies
- start the local service and open the browser

Default URL:

```text
http://127.0.0.1:8501
```

Notes:
- The first startup may take a few minutes
- On macOS, the first launch of `start_easyicu.command` may be blocked by Gatekeeper

macOS first-run note:
1. Double-click `start_easyicu.command` once.
2. If macOS shows a security warning, open `System Settings -> Privacy & Security`.
3. In the Security section, click `Open Anyway` for `start_easyicu.command`.
4. If needed, right-click the file and choose `Open` once to confirm the exception.

After this one-time approval, later launches should open normally.

### Optional: Install for Python API / Development

This section is only needed if you want to:
- use the Python API in scripts or notebooks
- install EasyICU into your own environment
- develop or modify EasyICU locally

Anaconda/Miniconda is optional. The one-click launcher above does not require it.

#### Option 1: Conda (optional)

1. **Download Anaconda**
   Visit the [Anaconda website](https://www.anaconda.com/download) to download the latest version.

   > 💡 **Lightweight alternative:** If disk space is limited, use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instead.

2. **Install Anaconda**
   - Run the installer.
   - (Optional) Click "Browse" to change the installation directory.
    - Prefer leaving the PATH checkbox unchanged and using **Anaconda Prompt**.
   - Click "Next" to complete the installation.

#### Option 2: Standard Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

#### Install EasyICU

Open an **Anaconda Prompt**, a terminal with conda activated, or a standard virtual environment, then run:

```bash
# Clone the repository (or download and extract the ZIP from GitHub)
git clone "https://github.com/shen-lab-icu/easyicu.git"

# Navigate into the project directory and install
cd easyicu
pip install -e ".[all]"
```

#### Launch the Web Application

```bash
easyicu-webapp
```

You should see output similar to:

```
You can now view your Streamlit app in your browser.
URL: http://localhost:8501
```

Open `http://localhost:8501` in your browser to access the EasyICU interface.

### Step 1: Obtain ICU Data

1. **Download ICU databases** (access credentials required):
   | Database | URL |
   |----------|-----|
   | MIMIC-III | https://physionet.org/content/mimiciii/ |
   | MIMIC-IV | https://physionet.org/content/mimiciv/ |
   | eICU-CRD | https://physionet.org/content/eicu-crd/ |
   | AmsterdamUMCdb | https://amsterdammedicaldatascience.nl/ |
   | HiRID | https://hirid.intensivecare.ai/ |
   | SICdb | https://physionet.org/content/sicdb/ |

2. **Extract the data** to a local directory.

### Step 2: Validate and Convert Data

1. Enter the path to your data directory in the web interface.
2. Click **Validate Data Path**.
3. EasyICU checks whether the database is already in a supported prepared format.
4. If raw **CSV / CSV.GZ / tar.gz** files are detected, the interface will offer **Convert & Setup**, which prepares the data in one click, including:
   - converting raw tables to **Parquet**
   - applying database-specific optimizations for large tables
   - preparing the layout needed by Web workflows and Python APIs
5. After setup finishes, load the prepared database from the same path.

<img width="1931" height="956" alt="Data Conversion" src="https://github.com/user-attachments/assets/86ea826b-6a0f-491a-b967-c5a7ebdfaa5b" />

---

### Step 3: Cohort Selection

1. Click **Cohort Selection** in the left sidebar.
2. Configure inclusion/exclusion criteria, for example:
   - **ICU length of stay** — e.g., ≥ 24 hours
   - **Age range** — e.g., 18–90 years
   - **First ICU admission only** — to exclude readmissions
   - **Sex**
   - **In-hospital mortality**
3. Click **Apply Filter**.
4. The interface displays the number of patients matching the criteria.

<img width="1931" height="736" alt="Cohort Selection" src="https://github.com/user-attachments/assets/628caf50-bed3-4918-b36f-5930464e9fb7" />

---

### Step 4: Feature Selection

1. Click **Select Features** in the left sidebar.
2. Check the desired clinical features grouped by category.
3. The dictionary panel on the right provides feature definitions and variable mapping details for reference.

<img width="1931" height="1018" alt="Feature Selection" src="https://github.com/user-attachments/assets/f37fc262-b0e8-4894-8a08-2614614f4f18" />

---

### Step 5: Batch Data Export

1. Click **Export Data** in the left sidebar.
2. Choose an export format and output path:

   | Format | Pros |
   |--------|------|
   | **Parquet** (recommended) | Small file size, fast I/O |
   | **CSV** | Universal, compatible with Excel and most statistics tools |
   | **Excel** | Opens directly in spreadsheet software; larger file size |

3. Set the number of patients to export.
4. Click **Start Export**.
5. Exported files are saved to the specified directory.

<img width="4249" height="2241" alt="Batch Export" src="https://github.com/user-attachments/assets/9575d396-14ef-4e02-a4ac-a2a6222b1776" />

---

### Step 6: Visualization & Analysis

#### Quick Visualization

The system provides interactive visualization tools for rapid data exploration:

- **Data Tables Explorer** — Browse loaded data by module with sorting and filtering.
- **Time Series Analysis** — Overlay multiple feature trends with interactive zoom and custom aggregation.
- **Patient Overview** — Comprehensive clinical trajectory for individual patients, highlighting key events and indicator changes.
- **Data Quality Assessment** — Missing-rate analysis, temporal coverage evaluation, and completeness statistics.

---

#### Cohort Analysis

The system supports statistical analysis of filtered research cohorts:

- **Group Comparison Analysis** — Multiple statistical tests available.
- **Multi-Database Feature Distribution** — Compare feature distributions across different ICU databases.
- **Cohort Dashboard** — Interactive display of demographics, clinical outcomes, and key indicators.

---

#### Visualization Example

<img width="3051" height="1823" alt="Quick Visualization Example" src="https://github.com/user-attachments/assets/09c64137-9c6a-401e-a1d0-fe358ea458de" />

---

## 🚀 Going Further (Developers / Advanced Users)

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
