[**English README**](README.md)

# EasyICU

> 面向跨公开 ICU 数据库研究的可复现基础设施：标准化临床概念提取、面向临床用户的 Web 工作流、可编程的 Python API，以及一个**证据绑定研究 Agent**——追踪已覆盖、已注册的数值声明，并把无法核验的声明在稿件边界拦下。

[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](COPYING)
[![Version](https://img.shields.io/badge/version-0.1.3-green.svg)](https://github.com/shen-lab-icu/easyicu)
[![临床映射状态](https://img.shields.io/badge/clinical%20mappings-mapping--only-yellow.svg)](src/easyicu/data/clinical-contracts.json)

[![观看 3 分钟介绍视频](docs/images/video_overview_thumbnail.jpg)](https://github.com/shen-lab-icu/easyicu/releases/download/v0.1.3/easyicu-0.1.3-overview-zh.mp4)

▶ [观看介绍视频（中文，3:21）](https://github.com/shen-lab-icu/easyicu/releases/download/v0.1.3/easyicu-0.1.3-overview-zh.mp4) · [English](https://github.com/shen-lab-icu/easyicu/releases/download/v0.1.3/easyicu-0.1.3-overview-en.mp4)。视频画面中的数值仅为示意。

## EasyICU 能做什么

用自然语言提出一个 ICU 研究问题。EasyICU 把它对应到临床概念词典，在你本地的
ICU 数据上构建研究人群，先给出完整的分析计划供你审阅，再在你自己的电脑上计算
表格和图。每一个脚本、表格、统计量和图都以哈希登记进证据库；稿件只报告与已登记
结果绑定的数字，无法核验的内容会被拦下。

![EasyICU 工作流：问题、数据、队列、分析、稿件、PDF 与参考文献](docs/images/easyicu_workflow.svg)

**三种使用方式**

| | 适用情况 | 入口 |
|---|---|---|
| **桌面应用** | 不想安装 Python，直接点击完成一项研究。没有 Docker 时只给出表格和图形的首轮结果；论文需要 Docker（见下文）。 | [Releases 页面](https://github.com/shen-lab-icu/easyicu/releases)上的安装包；见[安装说明](docs/installation.md)。 |
| **源码运行的 Web 应用** | 从代码仓库启动，在本地浏览器中使用。 | `./start_easyicu.sh`（或 `start_easyicu.command` / `start_easyicu.bat`）；见下文「路线 A」。 |
| **Python API 与命令行** | 在脚本或 notebook 中编程完成提取与分析。 | 从本仓库或 release 的 wheel 安装，然后 `import easyicu`；见下文「路线 B」。 |

**如何得到论文。** 如果需要论文稿及其 PDF 和参考文献（.bib），请在提问时就写明，
例如「……，写成论文，并给出 PDF 和参考文献」。论文需要容器运行时：Docker 加 EasyICU
运行镜像，即源码运行的 Web 应用或服务器。没有 Docker 时，0.1.3 只给出表格和图形的首轮
结果，运行中会说明这一点。在 0.1.3 中，对已完成的首轮结果点击「扩展成完整论文」只补充
论文级分析，不生成稿件。

PhysioNet 官方公开的演示数据库（MIMIC-IV、MIMIC-III、eICU demo）可以在应用内下载，
接入自己的数据前就能完整试用每一步。

**哪些留在本机，模型会收到什么。** 抽取、审阅和分析都在你的电脑上运行，患者逐行记录留在本机。
开启模型调用后，已配置的模型可能收到你的消息、分析计划及其代码、变量名称、错误详情，
以及隐藏了极小计数的汇总统计。数据工具会阻止患者逐行记录进入模型请求；请勿在消息中粘贴患者记录。

EasyICU 是一个面向重症监护室（ICU）数据分析的 Python 工具包。它统一接入支持的公开 ICU 数据库，支持数百种标准化临床概念的自动提取，并提供 **Web 可视化界面**，帮助用户完成队列定义、特征审阅、可视化分析与数据导出。当前精确数量由已发布 registry 自动生成，见[目录摘要](docs/catalog_summary.md)，不再在说明文字里手工复制。

> **临床证据状态：**`0.1.3` 是软件/API 发布版本，不代表数据库特异性
> 临床验证已经完成。原有六个全概念数据库的映射仍均为 `mapping_only`；
> 独立临床复核与跨库输出可比性验证尚未完成。SOFA-2 已有算法级 golden
> 与失败关闭边界测试，但这些测试不能替代逐数据库映射验证。新增的 NWICU、
> Zhejiang eICU、CMAISE Jinhua ICU 和 Zigong 感染队列目前只完成首批核心概念接入，
> SOFA/SOFA-2 合同状态为 `not_assessed`。

> **呼吸来源修正（2026-10-08）：** MIMIC-IV 和 MIMIC-III 的 `spo2`/`o2sat` 已移除 SpO2 报警下限项目；新增 `peep_set`、`peep_total`、`fio2_chart`、`fio2_lab` 分源通道。旧 `peep`/`fio2` 保留原混合来源定义，既有导出不会自动重写。详见[来源与合成验证收据](docs/validation/respiratory_source_channel_repair.md)。

> **MAP 通道与利尿剂（2026-11-03）：** `map` 在同一 ICU 小时内有动脉导管值时取动脉值，否则取袖带值，不再混合汇总；新增 `map_invasive`、`map_nibp` 分通道概念，并映射 MIMIC-III 6702（Arterial BP Mean #2）。`furosemide` 覆盖所有给药途径；新增 `loop_diuretic`、`thiazide_diuretic`、`potassium_sparing_diuretic`、`acetazolamide` 与 `loop_diuretic_furosemide_eq`（静脉呋塞米当量 mg；Brater, N Engl J Med 1998）。既有导出不会自动重写。

新增可选的 MIIV [限定 stay 逐事件提取接口](docs/validation/miiv_scoped_source_events_v2.md)，返回小时值、私有原始事件追踪与来源收据；原 `load_concepts` 接口保持兼容。

## 为什么是 EasyICU

EasyICU 有两层，对应同一个问题的两半——*一个被报告的 ICU 结果有多可信？* **概念层**约束「产生这个数字的临床定义」，**证据绑定 Agent 层**约束「记录这个数字的证据链」。

- **用一套临床概念层注册十个公开 ICU 数据源**：六个数据库具有既有全概念映射，四个 community 数据源具有经审计的核心概念映射。EasyICU 以临床概念而不是数据库专属变量表作为核心抽象；映射存在本身不等于数据库特异性临床等价，也不证明跨库输出已经可比。
- **同时支持代码与图形界面的可复现工作流**：同一份准备完成的数据既可用于 Web 界面，也可用于 Python 脚本和 notebook。
- **证据绑定、可审计的分析 Agent**：可选的 research-agent 层把「问题 + 队列」变成可审计的分析——产物（脚本、日志、表格、统计量、图形）会以 SHA-256 登记进证据库。对于确定性验证器已覆盖的结构化产物和已注册数值声明，系统会核对不一致并在**稿件边界拦下**无法核验的声明；这是一种 *fail-closed* 设计，不是对任意分析代码的通用形式化证明。
- **用跨库复制验证可靠性**：同一个研究问题可作为 replication protocol 在多个数据库上各跑一遍，让一个结论的稳健性可以被*检查*，而不是被假定。
- **围绕有临床意义的研究任务测试**：框架内置 **SOFA-2** 操作化实现，并有算法级 golden 与负向边界测试；各数据库 SOFA-2 映射仍为 `mapping_only`，尚待独立临床验证。

## 这个仓库适合谁

- **审稿人和临床研究者**：希望快速理解项目贡献，以及它如何支持 ICU 研究工作流。
- **Web 用户**：希望不写代码就完成数据校验、队列定义、特征审阅和导出。
- **Python 用户**：希望通过脚本或 notebook 构建可复现的特征提取与队列分析流程。

## 从这里开始

[安装与启动指南](docs/installation.md) 区分桌面安装包、Python wheel 和源码工作区。桌面安装包自带运行环境；下面的源码启动器需要 Python。

> **调用任何 API 前的唯一铁律：** 所有提取 API 接收的都是**已转换（prepared）**的数据集，而不是原始下载包。如果你还没转换过这个数据库，请**先做转换**（Web 界面 *Validate Data Path → Convert & Setup*，或 `DataConverter(...).convert_all()` —— 见 [Python API](#-python-api)）。下文每个示例里的 `data_path` 指的都是*转换后的目录*。

### 快速查表:"我想…… → 运行……"

| 目标 | 入口 |
|------|------|
| 不写 Python,可视化校验数据 / 定义队列 / 导出特征 | **FastAPI 原生 Web 应用** — `easyicu-webapp` *(或 `./start_easyicu.sh` / `start_easyicu.command`)* —— 见 **路线 A** |
| 用 Python 构建可复现的特征/队列流水线 | **Python API** — `import easyicu` —— 见 **路线 B** |
| 用 CLI 跑特征提取(脚本化、无 UI) | `easyicu extract --sources miiv --data-dirs /path/to/prepared/mimic-iv --output out/`(对应 `extract_features` 控制台脚本;`easyicu --help` 列出 `extract`、`download`、`import`、`setup`、`copilot`) |
| 让 research-agent 跑一个研究问题 + 队列 | `easyicu-research-agent` |
| 用 agent 复现一篇外部论文 | `easyicu-research-replication` |
| 启动 research-agent 用的 LLM 代理服务 | `easyicu-llm-server` |
| 为每用户独立的 Unix socket 实例提供 HTTPS 登录入口 | `easyicu-hosted-gateway`（安装 `easyicu[webapp]`；账号与 TLS 配置由管理员维护） |
| 可直接复制运行的脚本 | [`examples/`](examples/) —— 从 [`quickstart_convert_and_load.py`](examples/quickstart_convert_and_load.py) 开始 |

所有控制台脚本都在 `pyproject.toml` 的 `[project.scripts]` 里声明,安装本包后即可使用 —— 安装方式（使用版 / 开发版）见 **[路线 B](#路线-bpython-api)**。

### 文档地图

本 README 是入口。每个主要分层都在自己代码旁维护一份聚焦的 README：

| 阅读 | 用于 |
|------|------|
| [产品目标与设计共识](docs/EASYICU_PRODUCT_AGREEMENT.md) | 接续 Research Agent 开发前阅读：用户目标、可靠性与灵活性的取舍、验收证据及待定标准。本文不代表能力已全部实现。 |
| [`src/easyicu/README.md`](src/easyicu/README.md) | 包级模块地图——概念、转换、API、评分、Web 和 Research Agent 的职责。代码贡献者从这里开始。 |
| [`docs/native_fastapi_webserver.md`](docs/native_fastapi_webserver.md) | 当前维护的 FastAPI 原生 WebApp 路径与本地 route/API QA 命令。 |
| [`src/easyicu/research_agent/README.md`](src/easyicu/research_agent/README.md) | 证据绑定的 research-agent 层:四层设计、就绪检查、跨库复现协议。 |
| [`src/easyicu/data/README.md`](src/easyicu/data/README.md) | 驱动跨库提取的概念字典(`concept-dict.json` 与 SOFA-2 overlay)。 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 提交改动时的预期工作流。 |

### 路线 A：FastAPI 原生 Web 界面

如果你想：
- 快速启动 EasyICU
- 用可视化方式完成数据准备
- 不写 Python 直接定义队列并导出特征

> **运行要求**
> - Python 3.11–3.12，至少 4 个 CPU 核心。
> - 全队列提取所需内存：MIMIC-IV 规模的数据库建议 16 GB 内存（4 核约 66–70 分钟；14 GB 也能完成，会自动重试一次）。eICU 需要高内存服务器（4 个 worker 峰值约 88 GiB）。
> - 各数据库的峰值内存与完整表格见 [docs/local_run_guide.md](docs/local_run_guide.md#memory--内存)。

推荐入口：
- **Windows**：双击 `start_easyicu.bat`
- **macOS**：双击 `start_easyicu.command`
- **Linux**：运行 `./start_easyicu.sh`

首次启动会在 `.easyicu-runtime/` 下自动创建本地运行环境，并安
装 Web 所需依赖。当前这条路径建议使用 Python 3.11–3.12。

默认本地地址：

```text
http://127.0.0.1:8765
```

### 路线 B：Python API

如果你想：
- 在脚本或 notebook 中调用 EasyICU
- 自动化特征提取流程
- 在代码里构建可复现的队列管线

当前打包依赖建议使用 Python 3.11–3.12。

**直接安装使用**（无需 clone）：

```bash
pip install "easyicu[webapp] @ git+https://github.com/shen-lab-icu/easyicu.git"
```

按需替换方括号里的 extra：

| 你想要…… | 安装 |
|----------|------|
| Python API —— 提取概念、SOFA / SOFA-2、sepsis-3、各类评分 | `easyicu` |
| FastAPI 原生 Web 应用（provider 状态工具默认休眠） | `easyicu[webapp]` |
| Plotly / Kaleido 图表导出 | `easyicu[viz]` |
| 托管 research-agent 的 LLM 代理 | `easyicu[llmserver]` |
| 可选的 LangGraph agent graph | `easyicu[agentic]` |
| 当前 active extras | `easyicu[all]` |

`easyicu[all]` 有意不包含 `easyicu[scientific-adapters]`。因果/生存分析
验证适配器需要单独、显式安装；安装本身不会赋予其科学结论权限。

**核心安装（`easyicu`）已内置 research-agent 的分析栈**（`scikit-learn`、`statsmodels`），
所以 Python API 和确定性 agent 路径开箱即用。research-agent CLI 另需一个 LLM 客户端
—— 装 `easyicu[webapp]`（内含 `openai`）或运行 `easyicu-llm-server` 即可。

**克隆开发**（可编辑安装）：

```bash
git clone "https://github.com/shen-lab-icu/easyicu.git"
cd easyicu
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

如需手动启动原生 Web 应用：

```bash
easyicu-webapp
```

旧 Streamlit 包已经从 active package boundary 删除。如需归档复核，只能从
Stage27 之前的 git 历史或本地 Stage27 archive patch 恢复。

## 可复现性与安全说明

- **准备完成的数据是统一契约**：原始 CSV / CSV.GZ / tar.gz 数据需先转换，再供 Web 界面和 Python API 共用。
- **AI 助手默认关闭**：只有在用户显式启用后才会工作。
- **始终保留人工确认**：队列、特征、数据转换与导出等关键操作仍需用户确认。
- **仓库已包含自动化检查**：Pull request 运行 Linux / Python 3.11 检查；完整矩阵由维护者对最终候选手动触发。当前维护的 Web UI gate 是 FastAPI 原生路径。

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
| NWICU | 本地受限 PhysioNet 发布包（0.1.0） |
| Zhejiang eICU | OMIX005817 |
| CMAISE Jinhua ICU | OMIX007493-02 |
| Zigong ICU Infection Cohort | PhysioNet 1.1；感染富集专病队列 |

后四个 community 数据源支持原始 ZIP/嵌套 CSV 转换、数据库识别、stay 索引，
以及经审计的人口学、结局、生命体征和部分化验映射。未映射概念会明确返回不可用。
Zhejiang 的 stay 是住院级代理，Jinhua 将一次住院中的多段 ICU 记录合并为一个窗口，
Zigong 不是无选择的通用 ICU 队列。三者的脱敏时间偏移分别按天（Zhejiang）、
分钟（Jinhua）和小时（Zigong）显式归一到 ICU 相对小时。完整限制会写入本地
`community_preparation_manifest.json`。

## Web 工作流总览

1. **选择数据源。** 在应用内下载官方公开演示数据库，或指向你本地的 ICU 数据导出。
2. **提出问题。** 在 Guided Copilot 对话中提问，Copilot 把问题对应到词典概念和研究人群。
3. **审阅完整计划**（人群、变量、方法）并批准。
4. **阅读结果。** 运行卡片给出表格、图，以及结果可以解读到什么程度。需要论文稿、PDF 和 .bib 时，请在提问时就写明。

前两张截图来自内置的只读演示，它回放一次已登记的 MIMIC-IV 运行，不调用模型。

### Copilot 对话

Copilot 会展示每一步在做什么，例如在准备计划之前，把问题中的变量对应到所选数据源。

![Copilot 把问题对应到数据源](docs/images/01_copilot_data_mapping.jpg)

### 结果与解读边界

每个结果都会说明这次运行算了什么、可以解读到什么程度；这里是仅描述性：计数和比例，没有调整估计、检验或置信区间。

![带解读边界的结果卡片](docs/images/02_copilot_claim_boundary.jpg)

### Skill Hub

Skill Hub 列出研究工作流、技能和方法模块。每个工作流都会说明需要哪些输入，以及批准后的计划可以产出哪些结果。

![Skill Hub](docs/images/03_skill_hub.jpg)

### 数据词典

数据词典列出每个临床概念的定义、单位以及在各数据库中声明的映射。声明的映射并不证明它在你的数据中具有等价的临床含义。

![数据词典](docs/images/04_data_dictionary.jpg)

### 跨库对照

**Cross-DB Benchmark（跨库对照）** Tab 把同一临床概念在已注册且具备该概念映射的 ICU 数据库间标准化对齐，并叠加其分布以直接对比。这里的 "Benchmark" 是 Web 界面中的探索性对照与质量检查入口，不代表正式模型排行榜或外部 benchmark 结论。

### 数据准备

EasyICU 自动校验原始数据库目录并完成提取前的准备：识别 CSV / CSV.GZ / tar.gz 等原始布局，将其转换为 Parquet，执行数据库专用优化，并准备 Web 界面与 Python API 共用的数据结构。

## 可视化与分析

在 **Guided Copilot** 中提出研究问题并选择数据源，执行前先审阅完整计划；对话中的运行卡片展示状态、文件和证据。提取工作区另外提供患者审阅、时间序列、队列统计和跨库对照，供专家直接进入。各视图复用同一份准备完成的数据和科学执行模块。

Research Agent 把"问题 + EasyICU 准备好的数据"通过 **Plan → Build → Analyze → Gate** 变成证据绑定的研究产物；只有在提问要求论文且 Evidence Gate 通过时才生成稿件。它不是全自动论文写作或自主科学发现系统，也不是临床决策支持工具。

## 证据绑定 Research-Agent 层

`easyicu.research_agent` 是一个可选层，把研究问题 + 已确认的队列导出变成**可审计**的分析。它不是标准 Web 工作流或 Python 提取 API 的必需部分。完整设计见
[src/easyicu/research_agent/README.md](src/easyicu/research_agent/README.md)。

**为什么它不只是「编排」。** 通用分析 agent 擅长规划和写代码，但弱于 ICU 语义——会把有序的 SOFA 子分当连续量平均、把缺失的 PaO₂ 静默填补、掉进 `SOFA==0` 高死亡率假象、把 ICU 死亡和院内死亡混用。EasyICU 用四层来补这个缺口：

1. **ICU 数据底座** —— 复用上面的概念字典，作为 agent 的 prompt 级数据视图（原始行不进入 prompt；变量使用由目录与确定性验证器约束）。
2. **安全分析运行时** —— SHA-256 `EvidenceStore`、数值声明注册表、确定性验证器、执行回放。
3. **Agent 编排** —— planner / replanner / coder / analyzer / writer / critic，每个 LLM 步骤之间都有确定性门。
4. **候选假设排序** —— 一个有界、人工策展的预规划阶段（**不是**自主「科学发现」系统）。

**Fail-closed，而非放任自流。** `ResearchContext` 把每个变量的角色、单位、允许聚合、时间窗、缺失语义和 ICU 陷阱同时带进 agent 循环*和*验证器。流水线产物被哈希登记；进入数值声明注册路径的数字会与其来源重新核对。四道 readiness gate ——**execution-complete / evidence-complete / numeric-verified / analysis-validated**——由代码机械计算（任何人复算得同一标签），把输出分成三态：**gate-reportable**、**analysis-only**、**diagnostic-only**。一个无法核验的声明（例如草稿里 `AUROC 0.8` 与登记的 `0.842` 不符）会在稿件边界被拦下，按类型路由到 rewrite / 代码重跑 / 人工复核——**重过同一套门**才能进入可报告稿件。

**用跨库复制验证可靠性。** 跨数据库工作默认走 replication protocol：同一问题在其它支持的数据库上重跑（`cross_database_validation=["eicu", "hirid"]`），让一个结论在 case-mix、覆盖度和缺失模式上的稳健性可被检查——而不是声称哪个库「更好」。

**确定性审计，而非 LLM 当评委。** 硬性检查（`concept_usage`、统计、因果、报告清单、多重比较、公平性）都是确定性、规则化的，系统不依赖一个 LLM 给另一个 LLM 打分。历史模块 `icu_agent_bench` 是**内部评估协议**，不是已冻结的公开 benchmark，应据此描述。

## 🚀 进阶使用（开发者 / 高级用户）



### 开发与测试

建议为本项目创建独立开发环境，并运行当前自动化检查：

开发环境建议使用 Python 3.11–3.12。

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,webapp]"
pytest -q
```

Pull request 运行 Linux / Python 3.11 检查；完整 Python / 操作系统矩阵仅对明确的最终候选手动触发，push main 不会自动启动。`pytest -q` 默认使用开发阶段的测试过滤。提交改动前可先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 💻 Python API

在调用任何提取 API 之前，请先确保数据库已经完成准备。
原始的 CSV / CSV.GZ / tar.gz 数据并不是特征提取 API 预期的直接输入。
请先通过 Web 界面的 **Validate Data Path** -> **Convert & Setup**，或先执行下面的程序化数据转换，然后再把准备好的目录传给 `data_path`。

### API 前置条件：先做数据转换

Web 应用可以自动完成数据准备，也可以在代码里手动转换：

```python
from easyicu import DataConverter

converter = DataConverter('/path/to/raw/data', database='miiv')
converter.convert_all()
```

完成转换后，再使用下面的 API 示例。

> **提示 — 外置慢速存储（USB / 网盘）**：默认 `parallel_workers=4`
> 是为本地 SSD 调优的，在外置慢速存储上转换 PRESCRIPTIONS、CHARTEVENTS
> 这类大表时可能出现 sharded 写入死锁。把环境变量
> `EASYICU_CONV_WORKERS=1` 设上强制单线程：
> ```bash
> EASYICU_CONV_WORKERS=1 python convert_my_data.py
> ```
> 在 90 GB 的 AUMC numericitems（USB 上）大约会换来 30% 的耗时增加，
> 但能保证不卡死、稳定跑完。

### 最小端到端示例

`data_path` 始终指向转换后的（准备完成的）目录；只有原始下载文件时，请先运行转换器（`DataConverter(...).convert_all()` 或 Web 界面的 **Convert & Setup**）。

下面这个示例演示了 API 用户的完整最小流程：从原始数据库目录开始，先转换，再提取标准化特征。

```python
from easyicu import DataConverter
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

> ⚠️ 下面的 `data_path` 必须指向**已转换/准备完成**的目录（见 [先做数据转换](#api-前置条件先做数据转换)）。传入原始下载包会报错。

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

> ⚠️ 下面的 `data_path` 必须指向**已转换/准备完成**的目录（见 [先做数据转换](#api-前置条件先做数据转换)）。

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

# 专用概念 —— KDIGO AKI 分期与循环衰竭由专门 callback 计算，
# 但你可以直接在同一个 `load_concepts(...)` 里请求它们，API 会
# 自动路由到对应的加载函数，对调用方完全透明。
aki_and_circ = load_concepts(
    concepts=['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo',
              'aki_stage_rrt', 'aki_assessable', 'aki_ascertainment',
              'observation_window_coverage',
              'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
              'creat_low_past_48hr', 'creat_low_past_7day',
              'circ_failure', 'circ_event'],
    database='miiv',
    data_path='/path/to/data',
)

# 整库 / 按模块批量提取（最快的方式）
# 把所有需要的概念一次性传进去，resolver 就能在所有概念之间共享
# base table（chartevents / labevents / inputevents 桶扫描）。
# `merge=False` 返回 `dict[concept -> DataFrame]`，避免把所有特征
# 合并成一个巨大的 DataFrame，更省内存。
all_features = load_concepts(
    concepts=['hr', 'sbp', 'map', 'temp', 'spo2',
              'bili', 'crea', 'lact', 'plt', 'wbc',
              'sofa', 'sofa2', 'sep3',
              'aki', 'circ_failure'],
    database='miiv',
    data_path='/path/to/data',
    merge=False,           # 返回 dict 而不是大合并 DataFrame
)
```

只有当 `aki_ascertainment == 'negative_complete'` 时，`aki_stage == 0`
才表示可以确定的无 AKI；`partial` 或 `indeterminate` 行不能作为无 AKI 对照。

> **关于全患者提取**：当你不传 `patient_ids` 和 `max_patients` 时，
> EasyICU 会加载整个数据库的所有患者。在内存紧张（< 6 GB 可用）的机器上
> 会自动分批到 subprocess。可以用环境变量
> `EASYICU_BATCH_TIMEOUT_SEC`（默认 3600）限制每个 batch 的最大时长，
> 子进程超时会被强制杀掉，避免父进程永远等下去。
> 对 `extract_database(..., stream_output_batches=True)` 的磁盘导出，
> 默认批量按**当前可用内存**连续计算：保留 25%（且至少保留 2 GiB），
> 再把连续容量估计与本轮六库实测峰值结合，随后让每个模块根据首批进程树
> 工作集调整后续批次（上限 67,000）。这不是固定 10,000 档：约 8 GiB
> 可用时，当前首批约为 MIMIC-III 20,000、MIMIC-IV 37,000、eICU 25,000；
> 数据异常稠密的 AUMC 则从 5,000 开始。可用内存低于 24 GiB 时，
> MIMIC-III、MIMIC-IV 和 AUMC 必须先提供首批实测，不能直接 one-shot；
> SIC/HiRID 等较低风险队列若保守峰值能放入预留后预算则仍可 one-shot。
> 若只是高风险保护要求拆批，首批按均衡的一半开始，不制造极小尾批。该保护来自
> 2026-08-03 全量证据：MIMIC-III one-shot 约 16.83 GiB、AUMC one-shot
> 约 29.31 GiB，eICU 67,000-stay 的 `other_scores` 批约 15.6 GiB。
> manifest 会记录模块及逐批峰值 RSS；Sepsis 标签派生复用相同外层批次，
> 不再暗中固定切成 2,000-stay 小批。用户显式传入的 `batch_size` 始终优先。
>
> **未完成的导出会被拒绝。** `extract_database` 或网页导出运行期间，导出目录中有
> `_IN_PROGRESS` 标记和尚未发布的原始模块文件；只有根 `_manifest.json` 写完后
> 才删除该标记。EasyICU 的读取方会拒绝带标记或缺少 `_manifest.json` 的目录
> （`native_export_unpublished`）。若目录中残留中断运行留下的标记，新的提取会被
> 拒绝，且不会删除其中任何文件：请选择新的输出目录，或在检查后把该目录移到别处。

### 专业模块

> ⚠️ 下面的 `data_path` 必须指向**已转换/准备完成**的目录（见 [先做数据转换](#api-前置条件先做数据转换)）。

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

EasyICU 的临床配置包含由 GPL-3 许可的
[ricu](https://github.com/eth-mds/ricu) 项目修改而来的材料，因此组合后的
EasyICU 发行物采用 **GPL-3.0-only**。许可说明见 [LICENSE](LICENSE)，完整条款见
[COPYING](COPYING)，来源、作者署名及修改说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。EasyICU 原创部分仍保留原始
MIT 授权，见 [LICENSES/MIT.txt](LICENSES/MIT.txt)。

---

<div align="center">

**⭐ 如果 EasyICU 对你的研究有帮助，欢迎点一个 Star ⭐**

Made with ❤️ for ICU researchers worldwide

</div>
