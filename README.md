# PyRICU

> 🏥 Python ICU 数据处理工具包 - 基于 R 语言 ricu 包理念设计

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/shen-lab-icu/pyricu)

PyRICU 是一个专为重症监护室 (ICU) 数据分析设计的 Python 工具包，支持多个主流 ICU 数据库。它提供统一的 API 来加载、处理和分析 ICU 临床数据，让研究人员专注于数据科学而非数据工程。

## ✨ 核心特性

### 🎯 统一的多数据库支持
- **MIMIC-IV** - MIT 重症监护数据库 (推荐)
- **eICU-CRD** - Philips eICU 协作研究数据库
- **AmsterdamUMCdb** - 阿姆斯特丹大学医学中心数据库
- **HiRID** - 高分辨率 ICU 数据库

### 🌐 交互式 Web 应用
- **可视化数据浏览器** - 无需编程即可探索 ICU 数据
- **智能数据格式转换** - 自动检测 CSV/Parquet，一键转换
- **批量特征导出** - 支持 Parquet、CSV、Excel 格式
- **中英文双语界面** - 根据需要切换语言

### 📊 丰富的临床评分系统
| 评分 | 描述 |
|------|------|
| **SOFA** | 器官衰竭序贯评估 |
| **SOFA-2 (2025)** | 最新版本，支持 RRT、ECMO、机械循环支持 |
| **Sepsis-3** | 脓毒症诊断标准 |
| **qSOFA** | 快速 SOFA 评分 |
| **SIRS** | 全身炎症反应综合征 |
| **MEWS/NEWS** | 早期预警评分 |

### ⚡ 高性能优化
- **智能缓存** - 自动缓存已加载的表，避免重复 I/O
- **Parquet 原生支持** - 列式存储，极速加载
- **并行处理** - 自动检测硬件资源，优化并行配置
- **增量计算** - 仅处理需要的时间窗口和患者

---

## 🪟 Windows 用户快速指南 (临床医生推荐)

如果您是第一次接触 Python，建议按照以下步骤操作（总耗时约 15-20 分钟）：

### 第一步：安装 Anaconda (Python 环境)

1. **下载 Anaconda**  
   访问 [Anaconda 官网](https://www.anaconda.com/download) 下载 Windows 版本（推荐 Python 3.11）  
   国内镜像：[清华大学镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)（选择最新的 `Anaconda3-*-Windows-x86_64.exe`）
   
   > 💡 **轻量替代方案**: 如果 C 盘空间紧张，可使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（仅 ~70MB，而 Anaconda 需要 ~3GB）

2. **安装 Anaconda (避免 C 盘爆满)**  
   - 双击下载的 `.exe` 文件
   - ⚠️ **重要：更改安装路径** - 点击 "Browse" 将安装目录改为 `D:\Anaconda3` 或其他非 C 盘路径
   - 勾选 "Add Anaconda to my PATH environment variable"（添加到环境变量）
   - 点击 "Next" 直到完成
   
   > 💾 **空间需求**: Anaconda ~3GB, Miniconda ~400MB, PyRICU ~200MB

3. **验证安装**  
   打开 **Anaconda Prompt**（开始菜单搜索 "Anaconda Prompt"），输入：
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

浏览器会自动打开 `http://localhost:8501`，显示 PyRICU 界面。

### 第四步：准备数据

1. **下载 ICU 数据库**（需要先申请权限）
   - MIMIC-IV: https://physionet.org/content/mimiciv/
   - eICU: https://physionet.org/content/eicu-crd/

2. **解压数据到本地**（例如 `D:\mimic-iv\`）

3. **在 Web 界面中转换数据**
   - 左侧边栏选择数据库类型（如 MIMIC-IV）
   - 输入数据路径（如 `D:\mimic-iv\`）
   - 点击 "🔄 转换为 Parquet" 按钮

### 💡 常见问题

**Q: 如何关闭应用？**  
A: 在 Anaconda Prompt 窗口按 `Ctrl + C`，或直接关闭窗口。

**Q: 如何再次启动？**  
A: 打开 Anaconda Prompt，输入 `pyricu-webapp`。

**Q: 转换数据需要多久？**  
A: MIMIC-IV 约 30-60 分钟（取决于电脑配置），转换完成后下次加载只需几秒。

**Q: 需要编程基础吗？**  
A: 使用 Web 应用**不需要**编程基础。如果需要定制分析，可以学习 Python API（见下文）。

### ⚠️ 常见问题排查

<details>
<summary><b>❌ C 盘空间不足 / 磁盘爆满</b></summary>

**原因**: Anaconda 默认安装在 C 盘，占用 3-5GB

**解决方案**:

1. **使用 Miniconda 替代 Anaconda**（推荐）
   - 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（仅 70MB）
   - 安装时选择 D 盘：`D:\Miniconda3`
   - 安装完成后运行：`pip install "pyricu[all] @ git+https://github.com/shen-lab-icu/pyricu.git"`

2. **迁移已安装的 Anaconda**
   ```bash
   # 在 Anaconda Prompt 中
   conda config --add pkgs_dirs D:\conda_pkgs
   conda config --add envs_dirs D:\conda_envs
   ```

3. **清理缓存释放空间**
   ```bash
   conda clean --all -y
   pip cache purge
   ```

</details>

<details>
<summary><b>❌ 电脑卡死 / 内存不足</b></summary>

**原因**: 数据转换或加载时占用大量内存（MIMIC-IV chartevents 有 3 亿行）

**解决方案**:

1. **启动时使用低内存模式**
   ```bash
   pyricu-webapp --low-memory
   ```

2. **减少并行处理数**
   ```bash
   pyricu-webapp --workers 1
   ```

3. **只处理少量患者（用于测试）**
   - 在 Web 界面的「患者数量限制」中设置为 100-500

4. **转换大表时的建议**
   - 关闭其他程序（浏览器、Office 等）
   - 确保有 8GB+ 可用内存
   - 如果仍然卡死，尝试命令行单表转换：
   ```python
   from pyricu import DataConverter
   conv = DataConverter('/path/to/data', chunk_size=100000)  # 更小的块
   conv.convert_file('chartevents.csv')  # 单独转换一个表
   ```

5. **推荐配置**
   | 配置 | 最低要求 | 推荐配置 |
   |-----|---------|---------|
   | 内存 | 8GB | 16GB+ |
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

## 🚀 快速开始 (开发者 / 高级用户)

### 安装

```bash
# 基础安装
pip install git+https://github.com/shen-lab-icu/pyricu.git

# 包含 Web 应用
pip install "pyricu[webapp] @ git+https://github.com/shen-lab-icu/pyricu.git"

# 安装全部功能
pip install "pyricu[all] @ git+https://github.com/shen-lab-icu/pyricu.git"

# 或从源码安装
git clone https://github.com/shen-lab-icu/pyricu.git
cd pyricu
pip install -e ".[all]"
```

### 依赖包说明

| 安装选项 | 包含内容 |
|---------|---------|
| `pip install -e .` | 核心功能：pandas, numpy, pyarrow, pydantic |
| `pip install -e .[dev]` | 开发工具：pytest, black, ruff |
| `pip install -e .[viz]` | 可视化：plotly, kaleido |
| `pip install -e .[webapp]` | Web应用：streamlit, plotly, openpyxl, psutil |
| `pip install -e .[all]` | **全部功能** |

---

## 📦 数据准备 (首次使用必读)

PyRICU 使用 Parquet 格式存储数据，以获得最佳性能。如果您的原始数据是 CSV 格式，需要先进行转换。

### 转换方式

#### 方式一：使用 Web 应用 (推荐)

```bash
pyricu-webapp
```

在侧边栏：
1. 选择数据库类型 (如 MIMIC-IV)
2. 输入数据路径
3. 点击「🔄 转换为 Parquet」按钮

#### 方式二：使用 Python API

```python
from pyricu import DataConverter

# 创建转换器
converter = DataConverter(
    database='miiv',
    csv_path='/path/to/mimic-iv/csv',
    parquet_path='/path/to/mimic-iv/parquet'
)

# 转换所有表
converter.convert_all(parallel=True, n_jobs=4)
```

#### 方式三：使用命令行

```bash
pyricu-convert --database miiv --input /path/to/csv --output /path/to/parquet
```

### ⏱️ 转换时间估算

| 数据库 | 表数量 | 预估时间 | 内存需求 |
|-------|-------|---------|---------|
| MIMIC-IV | 30+ | 30-60 分钟 | 16GB+ |
| eICU-CRD | 20+ | 20-40 分钟 | 8GB+ |
| AmsterdamUMCdb | 15+ | 15-30 分钟 | 8GB+ |
| HiRID | 10+ | 10-20 分钟 | 8GB+ |

> ⚠️ **注意**: 大表 (如 chartevents、labevents) 会自动分片存储，以便支持更快的并行加载。

---

## 🌐 Web 应用 (推荐新手使用)

无需编写代码，通过图形界面探索 ICU 数据：

```bash
# 启动 Web 应用
pyricu-webapp

# 或
python -m pyricu.webapp
```

### Web 应用功能

1. **📂 数据路径配置** - 支持自动检测数据格式
2. **🔄 CSV → Parquet 转换** - 一键转换，加速后续加载
3. **🔧 特征选择** - 200+ 临床概念分类浏览
4. **📊 数据可视化** - 患者时间序列、SOFA 趋势图
5. **📤 批量导出** - Parquet/CSV/Excel 格式

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

---

## 📋 支持的临床概念 (200+)

### 生命体征
`hr` 心率 | `sbp/dbp/map` 血压 | `temp` 体温 | `resp` 呼吸 | `spo2` 血氧

### 实验室检查
`crea` 肌酐 | `bili` 胆红素 | `plt` 血小板 | `wbc` 白细胞 | `lactate` 乳酸 | `pafi` P/F比值

### 血气分析
`po2/pco2` 氧分压/二氧化碳分压 | `ph` 酸碱度 | `be` 碱剩余

### 血管活性药物
`norepi` 去甲肾上腺素 | `epi` 肾上腺素 | `dopa` 多巴胺 | `dobu` 多巴酚丁胺 | `vaso` 血管升压素

### 输入输出
`urine` 尿量 | `urine24` 24h尿量 | `fluid_in/out` 液体出入量

### 呼吸支持
`vent_ind` 机械通气 | `fio2` 吸氧浓度 | `peep` 呼气末正压 | `ecmo` ECMO

### 临床评分
`sofa/sofa2` SOFA评分 | `sep3` Sepsis-3 | `qsofa` qSOFA | `gcs` 格拉斯哥评分

> 💡 **查看完整概念列表**：`from pyricu import list_available_concepts; print(list_available_concepts())`

---

## 🔬 SOFA-2 (2025) 更新说明

SOFA-2 是 2025 年发布的器官衰竭评分更新版本：

| 系统 | SOFA-2 改进 |
|------|-------------|
| **呼吸** | P/F 阈值更新 (≤300/225/150/75)，需高级呼吸支持 |
| **凝血** | 血小板阈值放宽 (≤150/100/80/50) |
| **肝脏** | 1分阈值从 1.9 放宽至 ≤3.0 mg/dL |
| **心血管** | 联合 NE+Epi 剂量，支持机械循环支持检测 |
| **肾脏** | 支持 RRT 标准检测 (K+≥6.0 或 pH≤7.20) |
| **中枢神经** | 支持镇静前 GCS，谵妄治疗检测 |

---

## 📁 数据格式

### 支持的格式
- **Parquet** (推荐) - 列式存储，最佳性能
- **CSV/CSV.GZ** - 原始格式，自动检测并提示转换
- **FST** - R 语言兼容格式

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
@software{pyricu2024,
  title = {PyRICU: Python Toolkit for ICU Data Analysis},
  author = {ICU Analytics Team},
  year = {2024},
  url = {https://github.com/shen-lab-icu/pyricu},
  version = {0.2.0}
}
```

同时请引用原始 ricu 包：

```bibtex
@article{ricu2021,
  title={ricu: R Interface to Intensive Care Unit Datasets},
  author={Bennett, Nicolas and Moor, Michael and others},
  journal={Journal of Open Source Software},
  year={2021}
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
<summary><b>Q: SOFA-2 和 SOFA 有什么区别？</b></summary>

SOFA-2 是 2025 年更新版本，主要改进包括：
- 呼吸：P/F 阈值更新，强制要求高级呼吸支持
- 凝血：血小板阈值放宽
- 心血管：支持联合血管活性药剂量
- 肾脏：支持 RRT 标准检测
- 中枢神经：支持镇静前 GCS

</details>

<details>
<summary><b>Q: 可以用于临床实践吗？</b></summary>

⚠️ **PyRICU 仅供研究使用**。虽然我们努力确保准确性，但未经过临床验证，不应用于实际患者护理决策。

</details>

---

## 📞 联系方式

- **项目主页**: https://github.com/shen-lab-icu/pyricu
- **问题反馈**: https://github.com/shen-lab-icu/pyricu/issues

---

## 📄 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果 PyRICU 对您有帮助，请给我们一个 Star！⭐**

Made with ❤️ for ICU researchers worldwide

</div>
