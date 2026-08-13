# B6-B typed column metadata sidecar handoff

> 日期：2026-07-18 EDT  
> 分支/HEAD：`refactor/agent-control-plane@7674814`  
> 范围：B6-B 五步中第 2 步；不含 concept-dict 科学内容修改，不改 Planner 的暴露、结局、队列、方法或 estimand 所有权。

## 结果

Web 原生导出与 research-agent intake 现在通过同一份内容寻址列级 metadata sidecar 闭合：

1. `easyicu.column_metadata_sidecar/1` 使用 canonical JSON、SHA-256、字节大小和 write-once temp+link+fsync 封存。
2. Web native export schema-v2 只在每个选定 concept 都有唯一 primary binding 时发布 sidecar + manifest；不完整导出不冒充 v2。
3. Agent intake 严格校验 sidecar filename/digest/size、package/file/module/concept identity、source database/resolution chain、time origin/unit 与物理列角色。
4. typed resolver 只允许 primary `{value,event_status}` 角色成为概念值；identifier/time/count/measurement/event-fraction 伴随列不得被任意数值列 fallback 升格。
5. native v2 缺失、孤儿、篡改、重复或不完整的 sidecar 全部 fail-closed；legacy v1 物理文件权威与固定 CSV SHA 保持不变。

## 提交

- `987bdc5 feat(concepts): seal typed column metadata sidecars`
- `7674814 feat(agent): bind native exports to typed column metadata`

## 验证

- metadata projection/sidecar、Web producer、native intake、Parquet/CSV/XLSX、catalog/cohort/replication 聚合：`303 passed, 1 deselected`。
- `test_meta_benchmark_spec.py` + capability registry：`28 passed`。
- Ruff（全部改动文件）、`git diff --check`、focused 反例全绿。
- 两路独立 adversarial review 最终 ACCEPT，无 HIGH/MED finding。

## 诚实边界

- sidecar 读取仍不是用同一 `O_NOFOLLOW` file descriptor 从头持有到尾；读取后的 digest/size 校验会阻断语义替换，这是后续可加深的 filesystem hardening，不是当前 metadata authority 的 fail-open。
- freeze gate 还需增加 v1 Parquet/XLSX/legacy-manifest 权威快照、Web 三格式端到端生产者、更广 SidecarRef/source-DB class-prefix 对抗矩阵。
- 当前 sidecar 权威只到 export-package intake；materialized cohort 和 ResearchContext 还没有封存并传播它，因此尚不可冻结 B6。

## 下一步

B6-B 第 3 步：materializer 从 verified export sidecar 生成 `materialized_cohort` scope 的 content-addressed metadata sidecar，ResearchContext v2 只添加 typed metadata/digest 坐标；旧 v1/封存 payload 继续可读，不能因新增 `None` 键改变旧科学 identity。
