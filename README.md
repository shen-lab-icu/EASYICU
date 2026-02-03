# PyRICU

> 🏥 Python ICU 数据处理工具包

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/shen-lab-icu/pyricu)

PyRICU 是一个专为重症监护室 (ICU) 数据分析设计的 Python 工具包，支持多个主流 ICU 数据库。它提供统一的 API 来加载、处理和分析 ICU 临床数据。

## ✨ 核心特性

### 🎯 统一的多数据库支持 (6 个数据库)

| 数据库 | 版本 | 主键列 | 分桶优化表 | 状态 |
|--------|------|--------|-----------|------|
| **MIMIC-IV** | v3.1 | `stay_id` | chartevents, labevents, inputevents | ✅ 100% |
| **MIMIC-III** | v1.4 | `icustay_id` | chartevents, labevents | ✅ 100% |
| **eICU-CRD** | v2.0 | `patientunitstayid` | nursecharting, lab | ✅ 100% |
| **AmsterdamUMCdb** | v1.0.2 | `admissionid` | numericitems, listitems | ✅ 100% |
| **HiRID** | v1.1.1 | `patientid` | observations, pharma | ✅ 100% |
| **SICdb** | v1.0.6 | `CaseID` | data_float_h, laboratory | ✅ 100% |

> ⚠️ **注意**：不同数据库使用不同的患者 ID 列名，使用 `patient_ids` 参数时请确认对应数据库的主键列。

### 🌐 交互式 Web 应用
- **可视化数据浏览器** - 无需编程即可探索 ICU 数据
- **智能数据格式转换** - 自动检测 CSV/Parquet，一键转换
- **批量特征导出** - 支持 Parquet、CSV、Excel 格式
- **中英文双语界面** - 根据需要切换语言

### 📊 包含更丰富的临床评分系统
| 评分 | 描述 |
|------|------|
| **SOFA** | 器官衰竭序贯评估 |
| **SOFA-2** | 最新版本，纳入RRT、ECMO、机械循环支持 |
| **Sepsis-3** | 脓毒症诊断标准 |
| **qSOFA** | 快速 SOFA 评分 |
| **SIRS** | 全身炎症反应综合征 |

### 📋 特征分类 (145+ 概念)

| 分类 | 概念数 | 示例 |
|------|--------|------|
| ⭐ SOFA-2 评分 | 7 | sofa2, sofa2_resp, sofa2_coag... |
| 📊 SOFA-1 评分 | 7 | sofa, sofa_resp, sofa_coag... |
| 🦠 脓毒症相关 | 6 | sep3, sep3_sofa2, susp_inf, qsofa... |
| ❤️ 生命体征 | 7 | hr, sbp, dbp, map, temp, resp, spo2 |
| 🫁 呼吸支持 | 14 | fio2, pafi, safi, mech_vent, vent_ind... |
| 🌬️ 呼吸机参数 | 12 | peep, tidal_vol, pip, plateau_pres... |
| 🩸 血气分析 | 9 | po2, pco2, ph, lact, o2sat... |
| 🧪 生化检验 | 21 | bili, crea, glu, alb, bun... |
| 🔬 血液学 | 20 | hgb, plt, wbc, hct, inr_pt... |
| 💉 血管活性药物 | 17 | norepi_rate, dopa_rate, epi_rate... |
| 💊 其他药物 | 4 | abx, ins, dex, cort |
| 🚰 肾脏/尿量 | 15 | urine, urine24, crea, rrt... |
| 🧠 神经系统 | 11 | gcs, egcs, mgcs, vgcs, rass, avpu... |
| 🫀 循环支持 | 3 | ecmo, iabp, mech_circ_support |
| 👤 人口统计学 | 6 | age, sex, weight, height, bmi, adm |
| 📈 其他评分 | 4 | sirs, news, mews, apache_ii |
| 🎯 结局指标 | 3 | death, los_icu, los_hosp |

### ⚡ 高性能优化
- **智能缓存** - 自动缓存已加载的表，避免重复 I/O
- **Parquet 原生支持** - 列式存储，极速加载
- **并行处理** - 自动检测硬件资源，优化并行配置
- **增量计算** - 仅处理需要的时间窗口和患者
- **DuckDB 内存安全转换** 🆕 - 12GB 内存即可转换任意大小数据
- **分桶存储优化** - 大表按变量 ID 分桶，查询速度提升 10-50 倍

---

## 快速开始指南

如果您是第一次接触 Python，建议按照以下步骤操作：

### 第一步：安装 Anaconda

1. **下载 Anaconda**  
   访问 [Anaconda 官网](https://www.anaconda.com/download) 下载 Windows 版本（推荐 Python 3.11）  
   
   > 💡 **轻量替代方案**: 如果 C 盘空间紧张，可使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（仅 ~70MB，而 Anaconda 需要 ~3GB）

2. **安装 Anaconda**  
   - 双击下载的 `.exe` 文件
   - ⚠️ **重要：更改安装路径** - 点击 "Browse" 将安装目录改为 `D:\Anaconda3` 或其他非 C 盘路径
   - 勾选 "Add Anaconda to my PATH environment variable"
   - 点击 "Next" 直到完成

3. **验证安装**  
   打开 **Anaconda Prompt**，输入：
   ```bash
   python --version
   ```
   应该显示 Python 3.11.x 或更高版本

### 第二步：安装 PyRICU

在 **Anaconda Prompt** 中执行：

```bash
# 方式一：直接从 GitHub 安装（推荐）
pip install "pyricu[all] @ git+https://github.com/shen-lab-icu/pyricu.git"

# 方式二：如果网络慢，可先下载 ZIP 再安装
# 1. 访问 https://github.com/shen-lab-icu/pyricu
# 2. 点击绿色 "Code" 按钮 -> Download ZIP
# 3. 解压到 D:\pyricu (或其他目录)
# 4. 在 Anaconda Prompt 中进入该目录：
cd D:\pyricu
pip install -e ".[all]"
```

### 第三步：启动 Web 应用

```bash
# 在 Anaconda Prompt 中输入：
pyricu-webapp
```

会有以下显示：
You can now view your Streamlit app in your browser.

URL: http://localhost:8504

使用浏览器打开网址 `http://localhost:8504`，显示 PyRICU 界面。

### 第四步：准备数据

1. **下载 ICU 数据库**（需要先申请权限）
   - MIMIC-IV: https://physionet.org/content/mimiciv/
   - eICU: https://physionet.org/content/eicu-crd/
   - AmsterdamUMCdb: https://amsterdammedicaldatascience.nl/
   - HiRID: https://hirid.intensivecare.ai/
   - SICdb: https://physionet.org/content/sicdb/

2. **解压数据到本地目录**（如 `D:\icu_data\mimiciv`）

### 第五步：数据转换

1. **Web 界面转换（推荐）**
   - 点击左侧 **⚙️ 管理** 按钮进入数据管理模式
   - 输入数据目录路径（如 `D:\icu_data\mimiciv`）
   - 点击 **转换** 按钮，系统自动：
     - 将 CSV/CSV.GZ 转换为 Parquet 格式
     - 对大表（chartevents、labevents 等）进行分桶优化
   - 转换完成后刷新页面

2. **命令行转换（高级）**
   ```python
   from pyricu import DuckDBConverter
   conv = DuckDBConverter('/path/to/data', memory_limit_gb=8)
   conv.convert_all()  # 内存安全转换，峰值约 300MB
   ```

> 💡 **分桶优化说明**：MIMIC-IV chartevents（3亿行）等大表会自动按 itemid 分为 100 个桶，查询速度提升 10-50 倍。

### 第六步：队列选择 (Cohort Selection)

1. 在 Web 界面左侧选择 **🎯 队列**
2. 设置筛选条件：
   - **患者数量限制** - 设为 0 表示全部患者
   - **ICU 住院时长** - 如 ≥24 小时
   - **年龄范围** - 如 18-90 岁
   - **排除条件** - 如排除二次入院
3. 点击 **应用筛选** 查看符合条件的患者数

### 第七步：特征选择 (Select Features)

1. 在 Web 界面左侧选择 **📊 特征**
2. 按类别勾选需要的特征：
   - **生命体征** (hr, sbp, dbp, map, temp, resp, spo2)
   - **实验室检查** (bili, crea, glu, plt, wbc...)
   - **评分系统** (sofa, sofa2, qsofa, sirs, sep3...)
   - **血管活性药物** (norepi_rate, dopa_rate...)
3. 设置时间参数：
   - **时间间隔** - 如 1 小时
   - **聚合方式** - mean / median / first / last

### 第八步：批量导出

1. 在 Web 界面选择 **📤 导出**
2. 选择导出格式：
   - **Parquet** - 推荐，文件小、加载快
   - **CSV** - 通用格式，可用 Excel 打开
   - **Excel** - 直接用 Excel 打开，但文件较大
3. 点击 **开始导出**，文件保存到指定目录

### 第九步：可视化分析

1. **快速可视化 (Quick Visualization)**
   - 选择患者 ID 和特征
   - 查看时间序列图、分布直方图
   - 支持多特征叠加对比

2. **队列分析 (Cohort Analysis)**
   - 查看队列人口统计学特征
   - 生成特征相关性热图
   - 导出分析报告

### 💡 新手常见问题

**Q: 如何关闭应用？**  
A: 在 Anaconda Prompt 窗口按 `Ctrl + C`，或直接关闭窗口。

**Q: 如何再次启动？**  
A: 打开 Anaconda Prompt，输入 `pyricu-webapp`。

**Q: 转换数据需要多久？**  
A: MIMIC-IV 约 10-30 分钟，转换完成后下次加载只需几秒。

**Q: 需要多少内存？**  
A: **8GB 最低，12GB 推荐**。

**Q: 需要编程基础吗？**  
A: 使用 Web 应用**不需要**编程基础。如果需要定制分析，可以学习 Python API（见下文）。

### ⚠️ 常见问题排查

<details>
<summary><b>❌ 电脑卡死 / 内存不足</b></summary>

**原因**: 数据转换或加载时占用大量内存（MIMIC-IV chartevents 有 3 亿行）

**解决方案**:

1. **使用 DuckDB 转换（推荐，默认启用）** 🆕
   - 新版本默认使用 DuckDB 内存安全转换
   - 转换 3 亿行数据仅需 **300MB 内存**
   - 无需任何配置，开箱即用

2. **启动时使用低内存模式**
   ```bash
   pyricu-webapp --low-memory
   ```

3. **减少并行处理数**
   ```bash
   pyricu-webapp --workers 1
   ```

4. **只处理少量患者（用于测试）**
   - 在 Web 界面的「患者数量限制」中设置为 100-500

5. **命令行单表转换（极低内存）**
   ```python
   from pyricu import DuckDBConverter
   conv = DuckDBConverter('/path/to/data', memory_limit_gb=4)
   conv.convert_all()  # 内存峰值 < 500MB
   ```

6. **推荐配置**
   | 配置 | 最低要求 | 推荐配置 |
   |-----|---------|---------|
   | 内存 | **8GB** ✅ | 16GB+ |
   | 硬盘 | 50GB 可用 | 100GB+ SSD |
   | CPU | 4 核 | 8 核+ |

</details>

<details>
<summary><b>❌ 网络慢 / GitHub 下载失败</b></summary>

**解决方案**:

1. **使用国内 pip 镜像**
   ```bash
   pip install "pyricu[all] @ git+https://github.com/shen-lab-icu/pyricu.git" -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **手动下载安装**
   - 浏览器访问 https://github.com/shen-lab-icu/pyricu
   - 点击绿色 "Code" → "Download ZIP"
   - 解压到 `D:\pyricu`
   - 运行：`cd D:\pyricu && pip install -e ".[all]"`

</details>

### 📚 推荐工具（可选）

- **VS Code**: [下载链接](https://code.visualstudio.com/) - 用于查看和编辑 Python 代码
- **Git for Windows**: [下载链接](https://git-scm.com/download/win) - 用于更新 PyRICU 到最新版本

---

## 🚀 更进一步 (开发者 / 高级用户)

### 依赖包说明

| 安装选项 | 包含内容 |
|---------|---------|
| `pip install -e .` | 核心功能：pandas, numpy, pyarrow, pydantic |
| `pip install -e .[dev]` | 开发工具：pytest, black, ruff |
| `pip install -e .[viz]` | 可视化：plotly, kaleido |
| `pip install -e .[webapp]` | Web应用：streamlit, plotly, openpyxl, psutil |
| `pip install -e .[all]` | **全部功能** |


---

## 💻 Python API

### Easy API - 一行代码

```python
from pyricu import load_sofa, load_sofa2, load_vitals, load_labs

# 加载 SOFA 评分
sofa = load_sofa(
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123, 30000456]
)

# 加载 SOFA-2 (2025 新标准)
sofa2 = load_sofa2(
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123],
    keep_components=True  # 保留各器官分数
)

# 加载生命体征
vitals = load_vitals(database='miiv', data_path='/path/to/data')

# 加载实验室检查
labs = load_labs(database='miiv', data_path='/path/to/data')
```

### Concept API - 灵活自定义

```python
from pyricu import load_concepts

# 批量加载多个概念
data = load_concepts(
    concepts=['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2'],
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123],
    interval='1h',       # 1小时对齐
    aggregate='mean',    # 平均值聚合
    verbose=True
)

# 加载 Sepsis-3 诊断
sepsis = load_concepts(
    'sep3',
    database='miiv',
    data_path='/path/to/data'
)
```

### 专业模块函数

```python
from pyricu import (
    load_demographics,      # 人口统计学
    load_outcomes,          # 结局指标
    load_vitals_detailed,   # 详细生命体征
    load_neurological,      # 神经系统评估
    load_output,            # 输出量
    load_respiratory,       # 呼吸系统
    load_lab_comprehensive, # 全面实验室检查
    load_blood_gas,         # 血气分析
    load_hematology,        # 血液学检查
    load_medications,       # 药物治疗
)

# 示例：加载人口统计学
demo = load_demographics(
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123]
)
```

### 数据转换

Web 应用会自动检测数据格式。如果检测到 CSV 文件，会提示一键转换：

```python
# 命令行转换
from pyricu.data_converter import DataConverter

converter = DataConverter('/path/to/csv/data', database='miiv')
converter.convert_all()
```

---

## 🛠 开发指南

### 环境设置

```bash
git clone https://github.com/shen-lab-icu/pyricu.git
cd pyricu
pip install -e ".[dev]"
```

### 运行测试

```bash
# 快速测试
pytest -q tests/

# 带覆盖率
pytest --cov=pyricu --cov-report=term-missing
```

### 代码规范

```bash
# 格式化
black src/ tests/

# 检查
ruff check src/ tests/
```

---

## 📝 引用

如果在研究中使用 PyRICU，请引用：

```bibtex
@software{pyricu2026,
  title = {PyRICU: Python Toolkit for ICU Data Analysis},
  author = {Shen Lab ICU Analytics Team},
  year = {2026},
  url = {https://github.com/shen-lab-icu/pyricu},
  version = {0.3.0}
}
```

---

## ❓ 常见问题

<details>
<summary><b>Q: 如何提高大规模数据加载性能？</b></summary>

- ✅ 使用 Parquet 格式存储数据
- ✅ 使用 `patient_ids` 参数只加载需要的患者
- ✅ 批量加载多个概念（共享缓存）
- ✅ 合理设置 `interval` 和 `win_length`

</details>



<details>
<summary><b>Q: 可以用于临床实践吗？</b></summary>

⚠️ **PyRICU 仅供研究使用**。虽然我们努力确保准确性，但未经过临床验证，不应用于实际患者护理决策。

</details>

---

## 📄 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果 PyRICU 对您有帮助，请给我们一个 Star！⭐**

Made with ❤️ for ICU researchers worldwide

</div>
