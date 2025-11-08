# pyricu

> Python 版 ICU 数据处理工具包 - 灵感来源于 R 语言的 ricu 包

pyricu 是一个专为重症监护室 (ICU) 数据分析设计的 Python 工具包，支持多个主流 ICU 数据库（MIMIC-IV、eICU、AUMC、HiRID 等）。它提供了统一的 API 来加载、处理和分析 ICU 临床数据，让研究人员能够专注于数据科学，而不是数据工程。

## ✨ 核心特性

### 🎯 统一的数据访问接口
- **多数据库支持**：MIMIC-IV, MIMIC-III, eICU, AUMC, HiRID
- **概念化抽象**：用统一的"概念"名称访问不同数据库中的相同临床指标
- **自动数据对齐**：自动处理时间序列对齐、单位转换、缺失值

### 🚀 两层 API 设计
1. **Easy API** - 预定义的便捷函数（`load_sofa`, `load_vitals`等）
2. **Concept API** - 灵活的主API（`load_concepts`）支持智能默认值和完全自定义

### 📊 丰富的临床评分系统
- **SOFA** (Sequential Organ Failure Assessment) - 器官衰竭评分
- **SOFA-2 (2025)** - 最新版本，支持 RRT 标准、ECMO、机械循环支持
- **Sepsis-3** - 脓毒症诊断标准
- **MEWS** - 改良早期预警评分
- **NEWS** - 国家早期预警评分
- **qSOFA** - 快速 SOFA 评分
- **SIRS** - 全身炎症反应综合征

### ⚡ 性能优化
- **智能缓存**：自动缓存已加载的表，避免重复 I/O
- **批量加载**：一次性加载多个概念，共享底层数据
- **增量计算**：仅处理需要的时间窗口和患者

### 🔧 灵活的扩展性
- **自定义概念字典**：轻松添加新的临床指标
- **回调函数机制**：实现复杂的数据转换逻辑
- **多字典合并**：组合不同来源的概念定义

## 🆚 pyricu vs ricu：为什么选择 Python？

| 特性 | pyricu (Python) | ricu (R) |
|------|----------------|----------|
| **语言生态** | Python - 机器学习/深度学习首选 | R - 传统统计分析 |
| **易用性** | 极简 API，一行代码即可 | 需要理解 R 数据结构 |
| **性能** | Pandas/Numpy 优化，支持大数据 | data.table 高效但语法复杂 |
| **集成性** | 无缝对接 scikit-learn, PyTorch, TensorFlow | 需要 reticulate 桥接 |
| **部署** | 易于打包和容器化部署 | 依赖 R 环境 |
| **协作** | Jupyter Notebook 友好 | RMarkdown |
| **SOFA-2 支持** | ✅ 原生支持最新 2025 版本 | ❌ 尚未更新 |
| **概念字典** | JSON 格式，易于编辑和扩展 | R 对象，修改较复杂 |

**pyricu 的独特优势：**
- ✅ **更现代的工作流**：Jupyter Notebook、VS Code、云端部署
- ✅ **更丰富的 ML 生态**：直接对接 XGBoost、LightGBM、深度学习框架
- ✅ **更快的迭代**：Python 动态特性，调试和原型开发更快
- ✅ **更好的可维护性**：类型提示、单元测试、持续集成

## 🚀 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/shen-lab-icu/pyricu.git
cd pyricu
pip install -e .

# 或者从 PyPI 安装（即将上线）
# pip install pyricu
```

### 5 分钟上手示例

#### 1. Easy API - 便捷函数

```python
from pyricu import load_sofa, load_vitals, load_labs

# 加载 SOFA 评分
sofa = load_sofa(
    database='miiv',
    data_path='/path/to/mimic-iv/data',
    patient_ids=[30000123, 30000456]
)

# 加载生命体征
vitals = load_vitals(
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123]
)

# 加载实验室检查
labs = load_labs(
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123]
)

print(sofa.head())
#    stay_id  charttime  sofa  sofa_resp  sofa_cardio  sofa_liver  sofa_coag  sofa_cns  sofa_renal
# 0  30000123       1.0   3.0        1.0          0.0         0.0        1.0       0.0         1.0
# 1  30000123       2.0   3.0        1.0          0.0         0.0        1.0       0.0         1.0
# 2  30000123       3.0   4.0        2.0          0.0         0.0        1.0       0.0         1.0
# (注: charttime 表示入 ICU 后的小时数)
```

#### 2. Concept API - 完全自定义

```python
from pyricu import load_concepts

# 批量加载多个概念
vitals = load_concepts(
    concepts=['hr', 'sbp', 'dbp', 'temp'],
    database='miiv',
    data_path='/path/to/mimic-iv/data',
    patient_ids=[30000123],
    interval='1h',  # 1小时对齐
    verbose=True
)

# 加载 SOFA-2 评分（2025新标准）
sofa2 = load_concepts(
    'sofa2',
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123, 30000456],
    interval='6h',           # 6小时间隔
    win_length='24h',        # 24小时窗口
    keep_components=True,    # 保留所有组件
    aggregate='max',         # 最大值聚合
    verbose=True
)

print(sofa2.columns)
# ['stay_id', 'charttime', 'sofa2', 'sofa2_resp', 'sofa2_coag', 
#  'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal']
```

## 📚 高级用法

### 批量加载多个概念

```python
from pyricu import load_concepts

# 一次加载多个概念（共享数据，性能更好）
concepts = ['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2']
vitals = load_concepts(
    concepts=concepts,
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123],
    interval='1h',
    aggregate={'hr': 'mean', 'sbp': 'max'}  # 每个概念不同聚合
)
```

### 字符串格式的时间参数

```python
from pyricu import load_concepts

# 支持便捷的字符串时间格式
data = load_concepts(
    concepts=['hr', 'map', 'spo2'],
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[123, 456],
    interval='30min',      # 30分钟间隔
    win_length='6h'        # 6小时窗口
)
```

### SOFA-2 (2025) 新版本评分

```python
from pyricu import load_sofa2

# 使用最新的 SOFA-2 评分系统
sofa2 = load_sofa2(
    database='miiv',
    data_path='/path/to/data',
    patient_ids=[30000123],
    interval='1h',
    win_length='24h',
    keep_components=True  # 保留所有子组件
)

print(sofa2.columns)
# ['stay_id', 'charttime', 'sofa2', 'sofa2_resp', 'sofa2_coag', 
#  'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal']
```

**SOFA-2 相比 SOFA-1 的改进：**
- ✅ **呼吸系统**：P/F 比值阈值更新（≤300/225/150/75），需要高级呼吸支持
- ✅ **凝血系统**：血小板阈值放宽（≤150/100/80/50）
- ✅ **肝脏系统**：1分阈值放宽至 ≤3.0 mg/dL
- ✅ **心血管系统**：联合去甲肾上腺素+肾上腺素剂量，支持机械循环支持检测
- ✅ **肾脏系统**：支持 RRT 标准检测（K+≥6.0 或 pH≤7.20），尿量单位改为 mL/kg/h
- ✅ **中枢神经**：支持镇静前 GCS 评分，谵妄治疗检测

### 自定义概念字典

```python
# 加载自定义概念字典
loader = ICUQuickLoader(
    data_path="/path/to/mimic-iv/data",
    database='miiv',
    dict_path="path/to/custom-concepts.json"  # 使用自定义字典
)

# 或者合并多个字典
loader = ICUQuickLoader(
    data_path="/path/to/mimic-iv/data",
    database='miiv',
    dict_path=[
        "path/to/concept-dict.json",
        "path/to/custom-concepts.json",
        "path/to/sofa2-dict.json"
    ]  # 后面的字典会覆盖前面的同名概念
)
```

### 时间窗口和聚合

```python
# 计算 24 小时滑动窗口的平均值
hr_24h = loader.load_concepts(
    'hr',
    patient_ids=[30000123],
    interval=pd.Timedelta(hours=1),
    win_length=pd.Timedelta(hours=24),
    aggregate='mean'  # 支持: mean, min, max, sum, first, last
)

# 计算每小时的最高血压
sbp_hourly = loader.load_concepts(
    'sbp',
    patient_ids=[30000123],
    interval=pd.Timedelta(hours=1),
    aggregate='max'
)
```

### 完整 API - 最大灵活性

```python
from pyricu.concept import ConceptDictionary, ConceptResolver
from pyricu.datasource import ICUDataSource
from pyricu.config import load_data_sources

# 1. 加载数据源配置
registry = load_data_sources()
datasource = ICUDataSource(
    config=registry.get('miiv'),
    base_path="/path/to/mimic-iv/data"
)

# 2. 加载概念字典
dictionary = ConceptDictionary.from_json("path/to/concept-dict.json")

# 3. 创建解析器
resolver = ConceptResolver(dictionary)

# 4. 加载概念（完全控制）
data = resolver.load_concepts(
    concept_names=['hr', 'sbp'],
    data_source=datasource,
    merge=True,
    aggregate={'hr': 'mean', 'sbp': 'max'},
    patient_ids=[30000123],
    verbose=True
)
```

## 📋 支持的临床概念

pyricu 内置了 **200+ 临床概念**，涵盖生命体征、实验室检查、用药、操作等多个类别。

### 生命体征 (Vital Signs)
- `hr` - 心率
- `sbp`, `dbp`, `map` - 收缩压、舒张压、平均动脉压
- `temp` - 体温
- `resp` - 呼吸频率
- `spo2` - 血氧饱和度
- `gcs` - 格拉斯哥昏迷评分

### 实验室检查 (Laboratory)
- `crea` - 肌酐
- `bili` - 胆红素
- `plt` - 血小板
- `wbc` - 白细胞
- `lactate` - 乳酸
- `pafi` - PaO2/FiO2 比值
- `po2`, `pco2` - 氧分压、二氧化碳分压
- `ph` - 酸碱度
- `potassium`, `sodium` - 钾、钠
- `glucose` - 血糖

### 用药 (Medications)
- `norepi` - 去甲肾上腺素
- `epi` - 肾上腺素
- `dopa` - 多巴胺
- `dobu` - 多巴酚丁胺
- `vaso` - 血管升压素
- `abx` - 抗生素

### 输入输出 (Intake/Output)
- `urine` - 尿量
- `urine24` - 24小时尿量
- `fluid_in` - 液体摄入
- `fluid_out` - 液体排出

### 呼吸支持 (Respiratory Support)
- `vent_ind` - 机械通气指征
- `fio2` - 吸氧浓度
- `peep` - 呼气末正压
- `adv_resp` - 高级呼吸支持
- `ecmo` - 体外膜肺氧合

### 肾脏替代治疗 (Renal Replacement Therapy)
- `rrt` - 肾脏替代治疗
- `rrt_criteria` - RRT 标准（SOFA-2）

### 临床评分 (Clinical Scores)
- `sofa` - SOFA 评分
- `sofa2` - SOFA-2 评分 (2025)
- `sep3` - Sepsis-3 诊断
- `mews` - MEWS 评分
- `news` - NEWS 评分
- `qsofa` - qSOFA 评分
- `sirs` - SIRS 评分

> 💡 **查看所有支持的概念**：运行 `loader.available_concepts()` 获取完整列表

## 🎓 实际应用案例

### 案例 1: 脓毒症患者队列研究

```python
from pyricu.easy import load_sepsis, load_sofa_score, load_labs
import pandas as pd

# 1. 加载所有疑似感染患者
sepsis_cohort = load_sepsis(
    data_path="/path/to/mimic-iv/data",
    database='miiv'
)

# 2. 筛选 Sepsis-3 阳性患者
sepsis_patients = sepsis_cohort[sepsis_cohort['sep3'] == True]['stay_id'].unique()
print(f"识别到 {len(sepsis_patients)} 名脓毒症患者")

# 3. 提取这些患者的详细数据
patient_data = load_sofa_score(
    data_path="/path/to/mimic-iv/data",
    patient_ids=sepsis_patients.tolist(),
    interval=pd.Timedelta(hours=1)
)

# 4. 计算最高 SOFA 评分
max_sofa = patient_data.groupby('stay_id')['sofa'].max()
print(f"平均最高 SOFA: {max_sofa.mean():.1f}")
```

### 案例 2: 机器学习特征工程

```python
from pyricu.easy import load_custom
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. 加载多模态特征
features = load_custom(
    data_path="/path/to/mimic-iv/data",
    concepts=[
        # 生命体征
        'hr', 'sbp', 'temp', 'resp', 'spo2',
        # 实验室
        'lactate', 'wbc', 'crea', 'bili', 'plt',
        # 评分
        'sofa', 'gcs'
    ],
    patient_ids=patient_list,
    interval=pd.Timedelta(hours=6)  # 每6小时采样
)

# 2. 准备训练数据
X = features[['hr', 'sbp', 'temp', 'lactate', 'sofa']]
y = outcomes  # 你的目标变量

# 3. 训练模型
model = RandomForestClassifier()
model.fit(X.dropna(), y)
```

### 案例 3: 时间序列分析

```python
from pyricu.quickstart import ICUQuickLoader
import matplotlib.pyplot as plt

loader = ICUQuickLoader("/path/to/mimic-iv/data", database='miiv')

# 加载某患者的完整时间序列
patient_id = 30000123
data = loader.load_concepts(
    ['hr', 'sbp', 'lactate', 'sofa'],
    patient_ids=[patient_id],
    interval=pd.Timedelta(hours=1),
    merge=True
)

# 可视化（charttime 表示入 ICU 后的小时数）
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
data.plot(x='charttime', y='hr', ax=axes[0], title='心率', xlabel='ICU 住院时间 (小时)')
data.plot(x='charttime', y='sbp', ax=axes[1], title='收缩压', xlabel='ICU 住院时间 (小时)')
data.plot(x='charttime', y='lactate', ax=axes[2], title='乳酸', xlabel='ICU 住院时间 (小时)')
data.plot(x='charttime', y='sofa', ax=axes[3], title='SOFA评分', xlabel='ICU 住院时间 (小时)')
plt.tight_layout()
plt.show()
```

## 🔧 配置和自定义

### 数据源配置

pyricu 支持多种 ICU 数据库，配置文件位于 `src/pyricu/extdata/config/data-sources.json`。

支持的数据库：
- **MIMIC-IV** (`miiv`) - 推荐
- **MIMIC-III** (`mimic`)
- **eICU** (`eicu`)
- **AmsterdamUMCdb** (`aumc`)
- **HiRID** (`hirid`)

### 自定义概念字典示例

创建自定义概念 JSON 文件（例如 `my-concepts.json`）：

```json
{
  "my_custom_concept": {
    "class": "num_cncpt",
    "description": "我的自定义指标",
    "category": "labs",
    "unit": "mg/dL",
    "sources": {
      "miiv": [
        {
          "table": "labevents",
          "sub_var": "itemid",
          "ids": [50912]
        }
      ]
    }
  }
}
```

使用自定义概念：

```python
loader = ICUQuickLoader(
    data_path="/path/to/mimic-iv/data",
    database='miiv',
    dict_path="my-concepts.json"
)

data = loader.load_concepts('my_custom_concept')
```

## ❓ 常见问题 (FAQ)

### Q1: pyricu 支持哪些数据格式？
A: pyricu 原生支持 **FST** 和 **Parquet** 格式（推荐），也支持 CSV。FST/Parquet 具有更好的压缩率和读取性能。

### Q2: 如何处理缺失值？
A: pyricu 会保留原始数据中的缺失值（NaN）。你可以在加载后使用 pandas 的方法处理：
```python
data = data.fillna(method='ffill')  # 前向填充
data = data.dropna()  # 删除缺失行
```

### Q3: 如何提高大规模数据加载性能？
A: 几个优化技巧：
- ✅ 使用 `patient_ids` 参数只加载需要的患者
- ✅ 批量加载多个概念（共享缓存）
- ✅ 使用 FST/Parquet 格式存储数据
- ✅ 合理设置 `interval` 和 `win_length`

### Q4: SOFA-2 和 SOFA-1 有什么区别？
A: SOFA-2 是 2025 年发布的更新版本，主要改进包括：
- 呼吸：P/F 阈值更新，强制要求高级呼吸支持
- 凝血：血小板阈值放宽
- 肝脏：1分阈值从 1.9 放宽至 3.0 mg/dL
- 心血管：支持联合血管活性药剂量，检测机械循环支持
- 肾脏：支持 RRT 标准检测，尿量单位改为 mL/kg/h
- 中枢神经：支持镇静前 GCS，检测谵妄治疗

### Q5: 如何获取帮助？
A: 
- 📖 查看示例代码：`examples/` 目录
- 🐛 报告问题：[GitHub Issues](https://github.com/shen-lab-icu/pyricu/issues)
- 💬 讨论交流：[GitHub Discussions](https://github.com/shen-lab-icu/pyricu/discussions)

### Q6: pyricu 可以用于临床实践吗？
A: ⚠️ **pyricu 仅供研究使用**。虽然我们努力确保准确性，但 pyricu 未经过临床验证，不应用于实际患者护理决策。

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **Fork 本仓库**
2. **创建功能分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **创建 Pull Request**

### 贡献方向

- 🆕 添加新的临床概念定义
- 🐛 修复 bug 和改进文档
- ⚡ 性能优化
- 🧪 添加单元测试
- 🌍 支持更多 ICU 数据库
- 📊 添加新的临床评分系统

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/shen-lab-icu/pyricu.git
cd pyricu

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black src/
ruff check src/
```

## 📝 引用

如果你在研究中使用了 pyricu，请引用：

```bibtex
@software{pyricu2024,
  title = {pyricu: Python Toolkit for ICU Data Analysis},
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

## 📄 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 [ricu](https://github.com/eth-mds/ricu) 项目提供的设计灵感
- 感谢 MIMIC、eICU、AUMC、HiRID 数据库团队提供的公开数据
- 感谢所有贡献者的支持

## 📞 联系方式

- **项目主页**: https://github.com/shen-lab-icu/pyricu
- **问题反馈**: https://github.com/shen-lab-icu/pyricu/issues
- **邮件**: icu-analytics@example.com

---

<div align="center">

**⭐ 如果 pyricu 对你有帮助，请给我们一个 Star！⭐**

Made with ❤️ for ICU researchers worldwide

</div>
