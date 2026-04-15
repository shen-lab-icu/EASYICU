[**English README**](README.md)

# EasyICU

> 🏥 面向多公开 ICU 数据库的统一、高效、临床友好型数据提取与可视化框架

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/shen-lab-icu/easyicu)

EasyICU 是一个专为重症监护室（ICU）数据分析设计的 Python 工具包。它通过数据库抽象层统一处理 **6 个主流公开 ICU 数据库**，支持 **167 种**标准化临床概念的自动提取，并提供基于 **Web 的可视化界面**，使非编程背景的临床研究人员也能高效完成队列定义、特征筛选与数据质量审查。

## ✨ 核心特性

**🎯 统一的多数据库临床概念提取** — EasyICU 将「临床概念」作为特征工程的基本单位，以语义建模取代传统的静态变量映射。系统支持从 MIMIC-IV、MIMIC-III、eICU-CRD、AmsterdamUMCdb、HiRID、SICdb 六个主流公开 ICU 数据库中提取 167 种标准化临床概念，并率先实现了 **SOFA-2** 评分的自动化计算。

**🌐 面向临床用户的可视化交互界面** — EasyICU 集成了基于 Web 的图形化操作界面，旨在降低 EHR 数据分析的技术门槛。临床用户无需编程即可完成队列定义、特征选择、时间窗配置和数据质量审查，系统将患者时序数据整合为统一视图，支持从个体病例到群体分析的多维度审阅。

**⚡ 高性能计算优化** — 针对 ICU 数据高频、高维、稀疏的特点，EasyICU 引入了多种性能优化策略，确保在 **16 GB 内存**设备上即可稳定运行。

---

## 快速开始

### 一键启动（推荐）

如果用户只是想快速打开 EasyICU Web 界面，不需要先装 Anaconda，也不需要先打开 VS Code。

前提：
- 已安装 **Python 3.9+**
- 首次启动时可以联网下载依赖

启动方式：
- **Windows**：双击 `start_easyicu.bat`
- **macOS**：双击 `start_easyicu.command`
- **Linux**：运行 `./start_easyicu.sh`

首次运行会自动完成：
- 创建本地虚拟环境 `.easyicu-runtime/venv`
- 安装 EasyICU Web 所需依赖
- 启动本地服务并打开浏览器

默认地址：

```text
http://127.0.0.1:8501
```

说明：
- 首次启动通常会比后续启动慢几分钟
- macOS 首次双击脚本时，可能需要在系统安全提示里选择“仍要打开”

### 第一步：安装 Anaconda

1. **下载 Anaconda**
   访问 [Anaconda 官网](https://www.anaconda.com/download) 下载最新版本。

   > 💡 **轻量替代方案**：如果存储空间紧张，可使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

2. **安装 Anaconda**
   - 运行安装程序。
   - 可点击"Browse"修改安装目录。
   - 勾选"Add Anaconda to my PATH environment variable"。
   - 点击"Next"完成安装。

### 第二步：安装 EasyICU

在 **Anaconda Prompt**（或已激活 conda 的终端）中执行：

```bash
# 使用 Git 克隆仓库（也可直接从 GitHub 下载 ZIP 解压）
git clone "https://github.com/shen-lab-icu/easyicu.git"

# 进入项目目录并安装
cd easyicu
pip install -e ".[all]"
```

### 第三步：启动 Web 应用

```bash
easyicu-webapp
```

正常启动后会显示如下信息：

```
You can now view your Streamlit app in your browser.
URL: http://localhost:8501
```

用浏览器打开 `http://localhost:8501` 即可进入 EasyICU 界面。

### 第四步：准备数据

1. **下载 ICU 数据库**（需申请访问权限）：

   | 数据库 | 地址 |
   |--------|------|
   | MIMIC-III | https://physionet.org/content/mimiciii/ |
   | MIMIC-IV | https://physionet.org/content/mimiciv/ |
   | eICU-CRD | https://physionet.org/content/eicu-crd/ |
   | AmsterdamUMCdb | https://amsterdammedicaldatascience.nl/ |
   | HiRID | https://hirid.intensivecare.ai/ |
   | SICdb | https://physionet.org/content/sicdb/ |

2. **解压数据到本地目录**。

### 第五步：数据转换

1. 在 Web 界面中输入数据所在目录路径。
2. 系统自动检测数据格式：
   - 若数据**不是 Parquet 格式**，系统将提示需要转换。
3. 点击 **转换** 按钮，系统将自动执行：
   - 将 **CSV / CSV.GZ** 文件转换为 **Parquet** 格式。
   - 对大型数据表（如 `chartevents`、`labevents` 等）进行读取性能优化。
4. 转换完成后，刷新页面以加载转换后的数据。

<img width="1931" height="956" alt="数据转换" src="https://github.com/user-attachments/assets/86ea826b-6a0f-491a-b967-c5a7ebdfaa5b" />

---

### 第六步：队列选择

1. 在左侧导航栏点击 **队列选择**。
2. 设置筛选条件，例如：
   - **ICU 住院时长** — 如 ≥ 24 小时
   - **年龄范围** — 如 18–90 岁
   - **是否首次 ICU 入院** — 排除重复入院
   - **性别**
   - **院内死亡情况**
3. 点击 **应用筛选**。
4. 系统展示符合条件的患者数量。

<img width="1931" height="736" alt="队列选择" src="https://github.com/user-attachments/assets/628caf50-bed3-4918-b36f-5930464e9fb7" />

---

### 第七步：特征选择

1. 在左侧导航栏点击 **选择特征**。
2. 按分类勾选所需临床特征。
3. 右侧词典面板提供特征定义及变量映射说明，可作为选择参考。

<img width="1931" height="1018" alt="特征选择" src="https://github.com/user-attachments/assets/f37fc262-b0e8-4894-8a08-2614614f4f18" />

---

### 第八步：批量数据导出

1. 在左侧导航栏点击 **导出数据**。
2. 选择导出格式与保存路径：

   | 格式 | 特点 |
   |------|------|
   | **Parquet**（推荐） | 文件体积小，读取速度快 |
   | **CSV** | 通用格式，兼容 Excel 与多数统计软件 |
   | **Excel** | 可直接打开；文件体积较大 |

3. 设置导出的患者数量。
4. 点击 **开始导出**。
5. 导出文件保存至指定目录。

<img width="4249" height="2241" alt="批量导出" src="https://github.com/user-attachments/assets/9575d396-14ef-4e02-a4ac-a2a6222b1776" />

---

### 第九步：可视化分析

#### 快速可视化

系统提供多种交互式可视化工具，帮助用户快速理解数据结构与临床趋势：

- **数据表浏览器** — 按模块浏览数据，支持排序与筛选。
- **时间序列分析** — 多特征趋势叠加展示，支持交互缩放与自定义聚合方式。
- **患者概览** — 单患者综合临床轨迹，显示关键事件与指标变化。
- **数据质量评估** — 缺失率分析、时间覆盖评估、数据完整性统计。

---

#### 队列分析

系统支持对筛选后的研究队列进行统计学分析：

- **分组比较分析** — 支持多种统计检验方法。
- **跨数据库特征分布比较** — 对比不同 ICU 数据库的特征分布差异。
- **队列仪表盘** — 交互式展示人口学特征、临床结局与关键指标。

---

#### 可视化示例

<img width="3051" height="1823" alt="快速可视化示例" src="https://github.com/user-attachments/assets/09c64137-9c6a-401e-a1d0-fe358ea458de" />

---

## 🚀 进阶使用（开发者 / 高级用户）

## 💻 Python API

### Easy API — 一行代码

```python
from easyicu import load_sofa, load_sofa2, load_vitals, load_labs

# 加载 SOFA 评分
sofa = load_sofa(
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123, 30000456]
)

# 加载 SOFA-2（2025 修订标准）
sofa2 = load_sofa2(
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123],
    keep_components=True  # 保留各器官子分数
)

# 加载生命体征
vitals = load_vitals(database='miiv', data_path='/path/to/data')

# 加载实验室检查
labs = load_labs(database='miiv', data_path='/path/to/data')
```

### Concept API — 灵活自定义

```python
from easyicu import load_concepts

# 批量加载多个概念
data = load_concepts(
    concepts=['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2'],
    database='miiv',
    data_path='/path/to/mimic-iv',
    patient_ids=[30000123],
    interval='1h',       # 按 1 小时对齐
    aggregate='mean',    # 均值聚合
    verbose=True
)

# 加载 Sepsis-3 诊断
sepsis = load_concepts(
    'sep3',
    database='miiv',
    data_path='/path/to/data'
)
```

### 专业模块

```python
from easyicu import (
    load_demographics,      # 人口统计学
    load_outcomes,          # 结局指标
    load_vitals_detailed,   # 详细生命体征
    load_neurological,      # 神经系统评估
    load_output,            # 输出量
    load_respiratory,       # 呼吸系统参数
    load_lab_comprehensive, # 全面实验室检查
    load_blood_gas,         # 血气分析
    load_hematology,        # 血液学检查
    load_medications,       # 药物治疗
)

# 示例：加载人口统计学数据
demo = load_demographics(
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123]
)
```

### 程序化数据转换

Web 应用会自动检测数据格式并提供一键转换，也可通过代码调用：

```python
from easyicu.data_converter import DataConverter

converter = DataConverter('/path/to/csv/data', database='miiv')
converter.convert_all()
```

---

## 📄 许可证

本项目采用 **MIT 许可证**，详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果 EasyICU 对您的研究有所帮助，请给我们一个 Star！⭐**

Made with ❤️ for ICU researchers worldwide

</div>
