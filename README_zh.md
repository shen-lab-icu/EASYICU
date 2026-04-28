[**English README**](README.md)

# EasyICU

> 面向跨公开 ICU 数据库研究的可复现基础设施，提供标准化临床概念提取、面向临床用户的 Web 工作流，以及可编程的 Python API。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/shen-lab-icu/easyicu)

EasyICU 是一个面向重症监护室（ICU）数据分析的 Python 工具包。它统一接入 **6 个主流公开 ICU 数据库**，支持 **167 种标准化临床概念**的自动提取，并提供 **Web 可视化界面**，帮助用户完成队列定义、特征审阅、可视化分析与数据导出。

## 为什么是 EasyICU

- **用一套临床概念层覆盖六个公开 ICU 数据库**：EasyICU 以临床概念而不是数据库专属变量表作为核心抽象，更适合跨数据库研究、复用和同行审阅。
- **同时支持代码与图形界面的可复现工作流**：同一份准备完成的数据既可用于 Web 界面，也可用于 Python 脚本和 notebook。
- **围绕有临床意义的研究任务设计**：框架内置 **SOFA-2** 自动计算，并提供标准化概念提取、专业模块和队列分析能力。

## 这个仓库适合谁

- **审稿人和临床研究者**：希望快速理解项目贡献，以及它如何支持 ICU 研究工作流。
- **Web 用户**：希望不写代码就完成数据校验、队列定义、特征审阅和导出。
- **Python 用户**：希望通过脚本或 notebook 构建可复现的特征提取与队列分析流程。

## 从这里开始

### 路线 A：Web 界面

如果你想：
- 快速启动 EasyICU
- 用可视化方式完成数据准备
- 不写 Python 直接定义队列并导出特征

推荐入口：
- **Windows**：双击 `start_easyicu.bat`
- **macOS**：双击 `start_easyicu.command`
- **Linux**：运行 `./start_easyicu.sh`

默认本地地址：

```text
http://127.0.0.1:8501
```

### 路线 B：Python API

如果你想：
- 在脚本或 notebook 中调用 EasyICU
- 自动化特征提取流程
- 在代码里构建可复现的队列管线

最小安装方式：

```bash
git clone "https://github.com/shen-lab-icu/easyicu.git"
cd easyicu
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

如需手动启动 Web 应用：

```bash
easyicu-webapp
```

## 可复现性与安全说明

- **准备完成的数据是统一契约**：原始 CSV / CSV.GZ / tar.gz 数据需先转换，再供 Web 界面和 Python API 共用。
- **AI 助手默认关闭**：只有在用户显式启用后才会工作。
- **始终保留人工确认**：队列、特征、数据转换与导出等关键操作仍需用户确认。
- **仓库已包含自动化检查**：当前提供 `pytest` 与 GitHub Actions，覆盖基础仓库契约与界面渲染检查。

## 论文、引用与可复现

- **软件引用**：GitHub 引用元数据已写入 [CITATION.cff](CITATION.cff)。
- **给审稿人的仓库入口**：当前 README 已按“项目贡献、支持数据库、最短复现路径”的顺序组织。
- **可复现使用路径**：Web 用户可直接使用一键启动；Python 用户可在完成数据准备后复用下方 API 示例。
- **论文链接**：论文或预印本公开后，可在这里补充正式链接。

## 支持的公开 ICU 数据库

| 数据库 | 地址 |
|--------|------|
| MIMIC-III | https://physionet.org/content/mimiciii/ |
| MIMIC-IV | https://physionet.org/content/mimiciv/ |
| eICU-CRD | https://physionet.org/content/eicu-crd/ |
| AmsterdamUMCdb | https://amsterdammedicaldatascience.nl/ |
| HiRID | https://hirid.intensivecare.ai/ |
| SICdb | https://physionet.org/content/sicdb/ |

## Web 工作流总览

1. **准备 ICU 数据**并放到本地目录。
2. 在 Web 界面中执行 **Validate Data Path**。
3. 如检测到原始文件，使用 **Convert & Setup** 完成数据准备。
4. 定义研究队列、选择特征并导出结果。
5. 使用内置可视化与队列分析页面进行审阅。

### 数据准备

EasyICU 可以自动校验原始数据库目录，并将其准备成可直接提取的格式。Web 工作流会识别 CSV / CSV.GZ / tar.gz 等原始布局，将其转换为 Parquet，执行数据库专用优化，并准备 Web 界面和 Python API 共用的数据结构。

<img width="1931" height="956" alt="数据转换" src="https://github.com/user-attachments/assets/86ea826b-6a0f-491a-b967-c5a7ebdfaa5b" />

### 队列定义

典型筛选条件包括：
- ICU 住院时长
- 年龄范围
- 是否首次 ICU 入院
- 性别
- 院内死亡

<img width="1931" height="736" alt="队列选择" src="https://github.com/user-attachments/assets/628caf50-bed3-4918-b36f-5930464e9fb7" />

### 特征审阅与导出

特征按类别组织，右侧词典面板提供概念定义和变量映射说明。支持导出为 Parquet、CSV 和 Excel。

<img width="1931" height="1018" alt="特征选择" src="https://github.com/user-attachments/assets/f37fc262-b0e8-4894-8a08-2614614f4f18" />

<img width="4249" height="2241" alt="批量导出" src="https://github.com/user-attachments/assets/9575d396-14ef-4e02-a4ac-a2a6222b1776" />

## 可视化与分析

EasyICU 提供以下交互式工具：

- **快速可视化**：数据表浏览、时间序列审阅、单患者概览、数据质量评估
- **队列分析**：组间对照表、跨数据库分布对比、队列快照，以及 SOFA-1/SOFA-2 敏感性分析

<img width="3051" height="1823" alt="快速可视化示例" src="https://github.com/user-attachments/assets/09c64137-9c6a-401e-a1d0-fe358ea458de" />

## 🚀 进阶使用（开发者 / 高级用户）

## 开发与测试

建议为本项目创建独立开发环境，并运行当前自动化检查：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,webapp]"
pytest -q
```

仓库中的 GitHub Actions 会在 push 和 pull request 时运行同一套 `pytest` 检查。提交改动前可先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

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

**⭐ 如果 EasyICU 对你的研究有帮助，欢迎点一个 Star ⭐**

Made with ❤️ for ICU researchers worldwide

</div>
