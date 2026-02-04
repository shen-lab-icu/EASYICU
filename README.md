# EasyICU

> 🏥 面向多公开 ICU 数据库的统一、高效、临床友好型数据提取与可视化框架

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/shen-lab-icu/pyricu)

EasyICU 是一个专为重症监护室 (ICU) 数据分析设计的 Python 工具包。它通过数据库抽象层统一处理 6 个主流公开 ICU 数据库，支持 166种 标准化临床概念的自动提取，并提供基于 Web 的可视化界面，使非编程背景的临床研究人员也能高效完成队列定义、特征筛选与数据质量审查。

## ✨ 核心特性

**🎯 统一的多数据库临床概念提取** — EasyICU 将「临床概念」作为特征工程的基本单位，通过语义建模取代传统的静态变量映射。系统支持从 MIMIC-IV、MIMIC-III、eICU、AmsterdamUMCdb、HiRID、SICdb 六个主流公开 ICU 数据库中提取 166 标准化临床概念，并率先实现了 SOFA-2 的自动化计算。

**🌐 面向临床用户的可视化交互界面** — EasyICU 集成了基于 Web 的图形化操作界面，旨在降低 EHR 数据分析的技术门槛。临床用户无需编程即可完成队列定义、特征选择、时间窗配置和数据质量审查，系统将患者的时序数据整合为统一视图，支持从个体病例到群体分析的多维度审阅。

**⚡ 高性能计算优化** — 针对 ICU 数据高频、高维、稀疏的特点，EasyICU 引入多种性能优化策略，确保在 16GB 内存设备上稳定运行。

---

## 快速开始指南


### 第一步：安装 Anaconda

1. **下载 Anaconda**  
   访问 [Anaconda 官网](https://www.anaconda.com/download) 下载 Anaconda 最新版本
   
   > 💡 **轻量替代方案**: 如果 存储空间紧张，可使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. **安装 Anaconda**  
   - 打开安装包
   - 点击 "Browse" 修改安装目录
   - 勾选 "Add Anaconda to my PATH environment variable"
   - 点击 "Next" 直到完成


### 第二步：安装 PyRICU

在 **Anaconda Prompt** 中执行：

```bash
# 使用git下载pyciu (若没有git，可直接在github官网下载压缩包后解压)
git clone "https://github.com/shen-lab-icu/pyricu.git"

#在pyicu安装路径输入
pip install -e ".[all]"
```

### 第三步：启动 Web 应用

```bash
# 在 Anaconda Prompt 中输入：
pyricu-webapp
```

会有以下显示：
You can now view your Streamlit app in your browser.

URL: http://localhost:8501

使用浏览器打开网址 `http://localhost:8501`，显示 EasyICU 界面。

### 第四步：准备数据

1. **下载 ICU 数据库**（需要先申请权限）
   - MIMIC-III: https://physionet.org/content/mimiciii/
   - MIMIC-IV: https://physionet.org/content/mimiciv/
   - eICU: https://physionet.org/content/eicu-crd/
   - AmsterdamUMCdb: https://amsterdammedicaldatascience.nl/
   - HiRID: https://hirid.intensivecare.ai/
   - SICdb: https://physionet.org/content/sicdb/

2. **解压数据到本地目录**

### 第五步：数据转换

 **Web 界面转换**
   - 点击左侧 **⚙️ 管理** 按钮进入数据管理模式
   - 输入数据目录路径
   - 点击 **转换** 按钮，系统自动：
     - 将 CSV/CSV.GZ 转换为 Parquet 格式
     - 对大表（chartevents、labevents 等）的读取进行优化
   - 转换完成后刷新页面

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

---

## 🚀 更进一步 (开发者 / 高级用户)

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

## 📝 引用

如果在研究中使用 PyRICU，请引用：

```bibtex
@software{easyicu2026,
  title = {EasyICU: Python Toolkit for ICU Data Analysis},
  author = {Shen Lab ICU Analytics Team},
  year = {2026},
  url = {https://github.com/shen-lab-icu/pyricu},
  version = {0.3.0}
}
```


## 📄 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果 EasyICU 对您有帮助，请给我们一个 Star！⭐**

Made with ❤️ for ICU researchers worldwide

</div>