[**中文版 README**](README_zh.md)

# EasyICU

> A unified, high-performance, clinician-friendly framework for data extraction and visualization across multiple public ICU databases.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/shen-lab-icu/pyricu)

EasyICU is a Python toolkit designed for intensive care unit (ICU) data analysis. Through a database abstraction layer, it provides unified access to **6 major public ICU databases**, supports automated extraction of **166 standardized clinical concepts**, and offers a **web-based visualization interface** — enabling clinical researchers without programming backgrounds to efficiently perform cohort definition, feature selection, and data quality review.

## ✨ Key Features

**🎯 Unified Multi-Database Clinical Concept Extraction** — EasyICU treats *clinical concepts* as the fundamental unit of feature engineering, replacing traditional static variable mappings with semantic modeling. The system extracts 166 standardized clinical concepts from six major public ICU databases — MIMIC-IV, MIMIC-III, eICU-CRD, AmsterdamUMCdb, HiRID, and SICdb — and is the first to implement automated computation of the **SOFA-2** score.

**🌐 Clinician-Oriented Visual Interface** — EasyICU integrates a web-based graphical interface designed to lower the technical barrier for EHR data analysis. Clinicians can perform cohort definition, feature selection, time-window configuration, and data quality review without writing code. The system consolidates patient time-series data into a unified view, supporting perspectives from individual case review to population-level analysis.

**⚡ High-Performance Computing Optimizations** — Tailored for the high-frequency, high-dimensional, and sparse nature of ICU data, EasyICU incorporates multiple performance optimization strategies to ensure stable operation on machines with as little as **16 GB of RAM**.

---

## Quick Start Guide

### Step 1: Install Anaconda

1. **Download Anaconda**
   Visit the [Anaconda website](https://www.anaconda.com/download) to download the latest version.

   > 💡 **Lightweight alternative:** If disk space is limited, use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instead.

2. **Install Anaconda**
   - Run the installer.
   - (Optional) Click "Browse" to change the installation directory.
   - Check "Add Anaconda to my PATH environment variable".
   - Click "Next" to complete the installation.


### Step 2: Install EasyICU

Open an **Anaconda Prompt** (or any terminal with conda activated) and run:

```bash
# Clone the repository (or download and extract the ZIP from GitHub)
git clone "https://github.com/shen-lab-icu/pyricu.git"

# Navigate into the project directory and install
cd pyricu
pip install -e ".[all]"
```

### Step 3: Launch the Web Application

```bash
pyricu-webapp
```

You should see output similar to:

```
You can now view your Streamlit app in your browser.
URL: http://localhost:8501
```

Open `http://localhost:8501` in your browser to access the EasyICU interface.

### Step 4: Obtain ICU Data

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

### Step 5: Data Conversion

1. Enter the path to your data directory in the web interface.
2. The system automatically detects the data format.
   - If the data is **not in Parquet format**, a conversion prompt will appear.
3. Click the **Convert** button. The system will:
   - Convert **CSV / CSV.GZ** files to **Parquet** format.
   - Apply read-performance optimizations for large tables (e.g., `chartevents`, `labevents`).
4. Refresh the page after conversion to load the new data.

<img width="1931" height="956" alt="Data Conversion" src="https://github.com/user-attachments/assets/86ea826b-6a0f-491a-b967-c5a7ebdfaa5b" />

---

### Step 6: Cohort Selection

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

### Step 7: Feature Selection

1. Click **Select Features** in the left sidebar.
2. Check the desired clinical features grouped by category.
3. The dictionary panel on the right provides feature definitions and variable mapping details for reference.

<img width="1931" height="1018" alt="Feature Selection" src="https://github.com/user-attachments/assets/f37fc262-b0e8-4894-8a08-2614614f4f18" />

---

### Step 8: Batch Data Export

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

### Step 9: Visualization & Analysis

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

### Easy API — One-Liners

```python
from pyricu import load_sofa, load_sofa2, load_vitals, load_labs

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
from pyricu import load_concepts

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
from pyricu import (
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

### Data Conversion (Programmatic)

The web application detects CSV data automatically and offers one-click conversion. You can also convert programmatically:

```python
from pyricu.data_converter import DataConverter

converter = DataConverter('/path/to/csv/data', database='miiv')
converter.convert_all()
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If EasyICU is helpful for your research, please give us a Star! ⭐**

Made with ❤️ for ICU researchers worldwide

</div>
