# Patient Demo 再分发与真实覆盖审计

> 日期：2026-07-28  
> 任务：`WEBAPP-FASTAPI-NATIVE-QA` · `PATIENT-CROSSDB-VISUAL-PARITY`  
> 性质：只读审计；本轮没有上传数据、创建 GitHub Release 或修改患者可视化代码

## 结论

1. PhysioNet 的 **MIMIC-IV Clinical Database Demo 2.2** 与 **eICU-CRD Demo 2.0.1** 都是公开访问、按 **ODbL 1.0** 发布。ODbL 允许复制、再分发和制作衍生数据库，但公开分发时必须保留许可证与权利声明；衍生数据库还受 share-alike、机器可读副本或完整变更方法等条件约束。这里不等于可以再分发受 credential/DUA 约束的完整版 MIMIC-IV 或 eICU-CRD。
2. 不应把数据压进普通 Git 历史。GitHub 普通 Git 单文件上限为 100 MiB；eICU Demo ZIP 为 130.4 MB。GitHub Release 单资产上限为 2 GiB，适合承载原始不变归档或独立的 ODbL Demo Pack。
3. MIMIC 官方 Demo 的 EasyICU 数据链是全模块链，不是少量前端 fixture：
   - 32/32 原始表转换成功；
   - 19 个模块、151,373 行；
   - 281 个 feature definition；
   - 280 个 typed materialization；
   - 本地逐列复核得到 257 个至少有一个非空观测的概念、23 个已物化但在该 100-patient sample 中全空的概念、1 个未物化概念 `vent_free_days_28`；
   - 189 个有值数值概念、68 个有值非数值概念；174 个概念至少在一个 stay 中有两个数值点，具备患者轨迹条件。
4. **当前 Patient 浏览载荷还不全面**：
   - `patient_drilldown/__init__.py` 的 `_READ_MODULES` 只读 `demographics/outcome/sofa2_score/sepsis3_sofa2/vitals` 五个模块；
   - `_MAX_REVIEW_SIGNALS = 24`；
   - 当前激活 MIMIC Demo 的直接 API 复核只给选中病例返回 24 个 signal，质量分母只覆盖 35 个概念；其余已经在 Parquet 中有值的模块仍被 UI 当作 inventory/metadata。
5. 页面与既有日志中的 **83 numeric trajectory + 3 observed categorical + 195 metadata-only** 是 deterministic synthetic fallback 对 281 目录概念的覆盖口径，不是官方 MIMIC/eICU 的映射或非空覆盖证明。
6. eICU Demo 已有固定来源合同、安全下载/转换/全模块导出代码路径，`database="eicu_demo"` 也进入 canonical converter；但本机没有 eICU prepared cache 或同构 E2E receipt。因此目前只能说“链路已实现”，不能说“281 个字段已在官方 eICU Demo 全部完成映射并验证”。

## 官方证据

- MIMIC-IV Demo 2.2：<https://physionet.org/content/mimic-iv-demo/2.2/>
- MIMIC-IV Demo ODbL：<https://physionet.org/content/mimic-iv-demo/view-license/2.2/>
- eICU-CRD Demo 2.0.1：<https://physionet.org/content/eicu-crd-demo/2.0.1/>
- eICU-CRD Demo ODbL：<https://physionet.org/content/eicu-crd-demo/view-license/2.0.1/>
- GitHub 普通文件限制：<https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository>
- GitHub Release 限制：<https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases>

## 本地证据与复核口径

- 既有 MIMIC E2E receipt：`output/ui-qa/20260727_patient_echarts/mimic_iv_demo_e2e_receipt.json`
- prepared export：`~/.easyicu/demo_sources/mimic_iv_demo_v2_2/export/`
- export manifest：`_manifest.json`
- feature metadata：`feature_definitions.json`
- 逐列覆盖：按 manifest 的 281 个 concept id 读取 19 个 Parquet，统计真实 `notna()`；轨迹条件定义为至少一个 `stay_id` 内有两个非空数值点。
- 当前浏览载荷：直接调用 `patient_review_drilldown.patient_review_drilldown({})`，当前激活源为 MIMIC-IV Demo 2.2，得到 140 个 ICU stays、24 个 selected-entity signals、35 个 quality concepts。100 patients 与 140 stays 不矛盾，一个患者可有多个 ICU stay。

## 推荐实施顺序

1. 先修 Patient 的真实覆盖层：生成 281-feature × source 的 coverage index；模块清单读取全 19 模块的真实 observed/non-null 统计；曲线按“模块 → 特征 → 病例”懒加载，不把 281 列塞进 bootstrap，也不再用全局 24-signal cap 代表数据可用性。
2. 单独完成 eICU 130.4 MB E2E，生成与 MIMIC 同构的 receipt 和逐模块覆盖矩阵；没有这一步，不发布“双官方 Demo 已全面映射”的文案。
3. 再制作独立 GitHub Release：
   - 原始上游 ZIP 保持字节不变并附 upstream SHA-256；
   - EasyICU prepared pack 与软件 Apache 许可证隔离，明确按 ODbL 分发；
   - 内含 `LICENSE-ODbL.txt`、`NOTICE`、PhysioNet citation/DOI、来源版本、upstream URL、转换 receipt、feature coverage matrix 和校验和；
   - UI 优先尝试较小的 prepared pack，失败时回退 PhysioNet 的可续传原始下载。

## 发布前阻断项

- eICU 官方 Demo 尚无完整 E2E receipt。
- 当前 Patient UI 不能把“281 可查”描述为“281 都有真实曲线”。
- 当前 source summary 会把 `feature_definitions.csv` 当成额外模块，MIMIC API 报 20 modules；发布前应排除该元数据文件并保持 19 个临床模块语义。
- 正式公开再分发前应由项目负责人确认 ODbL notice/share-alike 包装；本审计不是法律意见。
