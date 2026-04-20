[**English README**](README.md)

# EasyICU

> 🏥 面向多公开 ICU 数据库的统一、高效、临床友好型数据提取与可视化框架

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/shen-lab-icu/easyicu)

EasyICU 是一个专为重症监护室（ICU）数据分析设计的 Python 工具包。它通过数据库抽象层统一处理 **6 个主流公开 ICU 数据库**，支持 **167 种**标准化临床概念的自动提取，并提供基于 **Web 的可视化界面**，使非编程背景的临床研究人员也能高效完成队列定义、特征筛选与数据质量审查。

## ✨ 核心特性

**🎯 统一的多数据库临床概念提取** — EasyICU 将「临床概念」作为特征工程的基本单位，以语义建模取代传统的静态变量映射。系统支持从 MIMIC-IV、MIMIC-III、eICU-CRD、AmsterdamUMCdb、HiRID、SICdb 六个主流公开 ICU 数据库中提取 167 种标准化临床概念，并率先实现了 **SOFA-2** 评分的自动化计算。

**🐍 面向脚本与 Notebook 的 Python API** — 除了 Web 界面，EasyICU 还提供可编程的 Python API，可在脚本和 notebook 中直接加载临床概念、器官评分、专业模块以及整库提取结果，便于构建可复现的队列筛选与特征工程流程。

**🌐 面向临床用户的可视化交互界面** — EasyICU 集成了基于 Web 的图形化操作界面，旨在降低 EHR 数据分析的技术门槛。临床用户无需编程即可完成队列定义、特征选择、时间窗配置和数据质量审查，系统将患者时序数据整合为统一视图，支持从个体病例到群体分析的多维度审阅。

**🤖 内置 AI 助手支持研究流程规划** — EasyICU 集成了上下文感知的 AI 助手，可以把研究问题映射成具体的 EasyICU 操作步骤。它能够帮助用户理解应该选择哪些队列筛选条件、模块、概念和评分，也可以辅助排查流程问题并给出当前页面相关的操作建议。

**🛠 一键式数据校验、转换与准备** — EasyICU 可以自动校验原始数据库目录，并将其准备成可直接提取的格式。Web 工作流能够识别 CSV / CSV.GZ / tar.gz 等原始布局，自动转换为 Parquet，执行数据库专用优化，并生成 Web 界面和 Python API 共用的数据准备结构。

**⚡ 高性能计算优化** — 针对 ICU 数据高频、高维、稀疏的特点，EasyICU 引入了多种性能优化策略，确保在 **16 GB 内存**设备上即可稳定运行。

---

## 快速开始

### 先选择你的使用路径

#### 路线 A：Web 界面用户

如果你想：
- 快速启动 EasyICU 图形界面
- 通过可视化方式完成数据校验、原始数据转换和特征导出
- 不写 Python 代码直接完成分析准备

从 [一键启动（推荐）](#一键启动推荐) 开始。

#### 路线 B：Python API / Notebook / 脚本用户

如果你想：
- 在 Python 脚本或 notebook 中调用 EasyICU
- 自动化批量提取流程
- 在代码里构建可复现的特征工程工作流

从 [可选：为 Python API / 开发环境安装 EasyICU](#可选为-python-api--开发环境安装-easyicu) 开始，然后阅读 [Python API](#-python-api)。

### 一键启动（推荐）

如果用户只是想快速打开 EasyICU Web 界面，不需要先装 Anaconda，也不需要先打开 VS Code。
如果一键启动已经满足需求，可以直接跳过下面的 Python/API 安装部分。

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
- macOS 首次运行 `start_easyicu.command` 时，可能会被系统安全机制拦截

macOS 首次运行说明：
1. 先双击一次 `start_easyicu.command`。
2. 如果系统弹出安全提示，打开 `系统设置 -> 隐私与安全性`。
3. 在“安全性”区域找到该脚本，点击“仍要打开”。
4. 如果仍被拦截，可对文件右键选择“打开”，再确认一次。

完成这一步后，后续再次启动通常就不需要重复授权了。

### 可选：为 Python API / 开发环境安装 EasyICU

这一部分只在以下场景需要：
- 你想在脚本或 notebook 里使用 Python API
- 你想把 EasyICU 安装到自己的 Python 环境中
- 你想本地开发或修改 EasyICU

Anaconda/Miniconda 是可选项，不是一键启动的前置要求。

#### 方式一：Conda（可选）

1. **下载 Anaconda**
   访问 [Anaconda 官网](https://www.anaconda.com/download) 下载最新版本。

   > 💡 **轻量替代方案**：如果存储空间紧张，可使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

2. **安装 Anaconda**
   - 运行安装程序。
   - 可点击"Browse"修改安装目录。
    - 一般不建议额外修改 PATH，直接使用 **Anaconda Prompt** 即可。
   - 点击"Next"完成安装。

#### 方式二：标准 Python 虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

#### 安装 EasyICU

在 **Anaconda Prompt**、已激活 conda 的终端，或普通 Python 虚拟环境中执行：

```bash
# 使用 Git 克隆仓库（也可直接从 GitHub 下载 ZIP 解压）
git clone "https://github.com/shen-lab-icu/easyicu.git"

# 进入项目目录并安装
cd easyicu
pip install -e ".[all]"
```

#### 启动 Web 应用

```bash
easyicu-webapp
```

正常启动后会显示如下信息：

```
You can now view your Streamlit app in your browser.
URL: http://localhost:8501
```

用浏览器打开 `http://localhost:8501` 即可进入 EasyICU 界面。

### 第一步：准备数据

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

### 第二步：验证并转换数据

1. 在 Web 界面中输入数据目录路径。
2. 点击 **Validate Data Path**。
3. EasyICU 会检查当前数据库是否已经是可直接读取的准备完成状态。
4. 如果检测到原始 **CSV / CSV.GZ / tar.gz** 数据，界面会继续提供 **Convert & Setup**，一键完成：
   - 转换为 **Parquet**
   - 对大表进行数据库专用优化
   - 准备 Web 工作流和 Python API 所需的数据布局
5. 准备完成后，从同一路径加载数据即可。

<img width="1931" height="956" alt="数据转换" src="https://github.com/user-attachments/assets/86ea826b-6a0f-491a-b967-c5a7ebdfaa5b" />

---

### 第三步：队列选择

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

### 第四步：特征选择

1. 在左侧导航栏点击 **选择特征**。
2. 按分类勾选所需临床特征。
3. 右侧词典面板提供特征定义及变量映射说明，可作为选择参考。

<img width="1931" height="1018" alt="特征选择" src="https://github.com/user-attachments/assets/f37fc262-b0e8-4894-8a08-2614614f4f18" />

---

### 第五步：批量数据导出

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

### 第六步：可视化分析

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

在调用任何提取 API 之前，请先确保数据库已经完成准备。
原始的 CSV / CSV.GZ / tar.gz 数据并不是特征提取 API 预期的直接输入。
请先通过 Web 界面的 **Validate Data Path** -> **Convert & Setup**，或先执行下面的程序化数据转换，然后再把准备好的目录传给 `data_path`。

### API 前置条件：先做数据转换

Web 应用可以自动完成数据准备，也可以在代码里手动转换：

```python
from easyicu.data_converter import DataConverter

converter = DataConverter('/path/to/raw/data', database='miiv')
converter.convert_all()
```

完成转换后，再使用下面的 API 示例。

### 最小端到端示例

下面这个示例演示了 API 用户的完整最小流程：从原始数据库目录开始，先转换，再提取标准化特征。

```python
from easyicu.data_converter import DataConverter
from easyicu import load_concepts

database = 'miiv'
raw_data_path = '/path/to/mimic-iv-raw'

# 第一步：把原始数据转换成 EasyICU 期望的准备完成格式
converter = DataConverter(raw_data_path, database=database)
converter.convert_all()

# 第二步：从准备完成的数据集中提取标准化临床概念
vitals = load_concepts(
    concepts=['hr', 'map', 'resp', 'spo2'],
    database=database,
    data_path=raw_data_path,
    patient_ids=[30000123, 30000456],
    interval='1h',
    aggregate='mean',
)

print(vitals.head())

# 可选：保存提取结果
vitals.to_parquet('miiv_vitals_1h.parquet', index=False)
```

这个示例默认你满足以下条件：
- `raw_data_path` 指向原始下载后的数据库目录
- 转换会把该目录准备成 EasyICU 可直接读取的格式
- 转换完成后，把这个准备好的目录继续传给 `data_path`

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

---

## 📄 许可证

本项目采用 **MIT 许可证**，详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果 EasyICU 对您的研究有所帮助，请给我们一个 Star！⭐**

Made with ❤️ for ICU researchers worldwide

</div>
