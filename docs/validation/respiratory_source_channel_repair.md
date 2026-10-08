# 呼吸测量与治疗分源修复（2026-10-08）

本批仅修软件；源码审查及逐文件核准的相关测试只用非患者字典和合成事件，未重导出、未拟合、未修改 sealed/current。一次范围未完全认证的全仓 fast 测试已中止，不能据其 marker/env 设置认证整个进程零患者读取，详见下文。历史封存导出及其研究结果尚未由此修复；实际报警值贡献多少仍未知。独立复核为另一智能体的数据库实现审查，不冒称人工临床验证。

## 来源与问题的区别

起点 Git 为 `06cd55d4dde46d842d608ab336b907a34d84d84d`。M3-A171 的非患者源码/元数据审计已记录在研究仓 Git `1aee87b97d567c9b51192cbbe4891b2428bbf162` 的 `results/evidence/M3-A171_联合动作来源零拟合审查/SOURCE_AUDIT.json`。该审计绑定 sealed EasyICU `97f508c6c94751a596481efafe68b41232e6bf42`，并非只审当前工作树。

- **SpO2 报警下限是明确的测量语义错误。** 226253 的字典标签是 SpO2 Desat Limit / Alarms，不能当实际氧饱和度。此前 `spo2` 和 `o2sat` 在 MIIV、MIMIC、MIMIC demo 共六条配置含该项；本次全部移除，保留其他原有测量来源。`o2sat` 仍包含其他测量方式，不因此变成纯脉搏血氧。
- **PEEP 是定义混合。** 220339 是设置值，224700 是总 PEEP。旧通用 `peep` 保持完整原对象；另增分源定义供研究选择。MIMIC CareVue 的 506 为 PEEP Set，505 仅标 PEEP，因此新设置通道含 506、不含 505。
- **FiO2 是记录来源混合。** chart 223835 与 lab 50816 属于不同记录源。新通道分离来源，未宣称同小时共现即同步床旁操作；MIMIC chart 的历史项目还含 set/analyzed/measured 等含义，因此 `fio2_chart` 不等于纯呼吸机设置。

官方 [MIMIC-IV vitalsign SQL](https://github.com/MIT-LCP/mimic-code/blob/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts/measurement/vitalsign.sql) 的 SpO2 来源仅 220277，支持排除报警项目。官方 [ventilator_setting SQL](https://github.com/MIT-LCP/mimic-code/blob/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts/measurement/ventilator_setting.sql) 仍合并两种 PEEP，而 FiO2 用 chart 223835，支持保留通用 PEEP 定义并另设分源通道。这里只交叉核映射含义，不声称其范围和聚合方法与 EasyICU 相同。固定 Git 与文件 SHA 见配套 JSON。

## 新通道合同

| 概念 | MIIV | MIMIC / MIMIC demo | 单位及既定范围 |
| --- | --- | --- | --- |
| `peep_set` | chart 220339 | chart 506、220339 | cmH2O，0–40 |
| `peep_total` | chart 224700 | chart 224700 | cmH2O，0–40 |
| `fio2_chart` | chart 223835 | 复制各自旧 chart 来源 | percent，21–100 |
| `fio2_lab` | lab 50816 | lab 50816 | percent，21–100 |

两种新 FiO2 通道均走 `percent_as_numeric`；小数与百分数经既定转换后聚合。MIMIC 新 lab 通道也明确配置该 callback，旧通用 `fio2` 未改。其他库无新映射，不把“无来源”伪装成可比较数据。范围沿既有合同，不能靠数值阈值恢复原导出已丢失的 item 身份。Python catalog 与生成的 Web catalog 同步加入四通道。

## 合成加载中的发现与修复

测试真正创建临时合成 Parquet，经 `ICUDataSource`、`ConceptResolver`、项目选择、单位 callback 和小时聚合；没有读取临床目录中的事件。

初版 8 例为 3 fail / 5 pass，暴露独立 MIIV lab 已归属 stay 却仍包装为 subject ID，以及 demo 使用错误 ICU 键。修复身份分支后，独立测试用不同 subject/hadm/stay 编号发现 demo 普通路径仍返回未选 stay；已修复真实过滤路径。共享变更仅限 MIIV lab 返回键和 MIMIC demo 沿其 MIMIC-III 原生 `icustay_id` 的身份处理。

最终 20 例覆盖三个配置、普通/DuckDB 两通路、报警 only 和报警/测量共存、设置/总 PEEP 与 chart/lab FiO2 隔离、同一住院的两次 ICU 入住、单 stay 选择及旧 `po2` 跨概念回归。合成 lab 不添加原生不存在的 stay 列。DuckDB 断言同时观测成功的单概念及多概念聚合，明确每个核心 chart 项目和 lab 50816 真正经过优化通路。观测器迭代曾把多概念调用遗漏、把失败回退尝试计为成功；保留失败记录，最终不是靠关闭优化获得通过。

独立审查还将旧通用和四个新通道同批加载，核查聚合未重新混池。测试仅证明这些合成合同，不证明现有导出已修复、无真实记录混入或全部原始表能安全按研究 allow-list 导出。后续候选导出仍须独立规定原始 schema、允许 stay 的最终返回边界、源事件溯源和发布流程。

## 验证范围

最终命令、结果、源 SHA 与独立审查状态在配套 JSON。全仓 fast suite 虽排除 `slow`、`needs_real_data`、`requires_corpus` 且三个 EASYICU 数据根指向不存在的临时路径，但未逐一核所有入口是否绕过这些配置。收到其他仓旧测试隐含临床读取的警示后，定点终止本进程 PID 4130283（SIGINT 后仍运行，再 SIGTERM，exit 143），保留到 26% 的日志；中止原因是数据范围未充分认证，不能声称全仓通过或全进程已证零患者读取。当前静态路径检索未证实本进程读取真实事件，但这也不是系统调用级证明。最终改跑逐文件审过的四个完整相关测试文件，显式 `-m ''`，无标记排除；输入仅打包元数据及临时合成表。四个线程库均限制为 1。全仓 ruff 与任务前基线完全相同的 16 项旧错误保持原样，未为本任务改无关脚本。

全仓中断前出现失败且没有完整 traceback；单跑社区目录测试定位到新增四概念后覆盖 JSON/Markdown 尚未生成及总数仍为 274。本次重新运行既有生成器，将合并数更新到 278、新四概念在其他社区库全部标 unavailable。没有增加其他库映射。完整相关测试的终态另列收据，不把中断的全仓运行改写成通过。
