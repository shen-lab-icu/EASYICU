# EasyICU Desktop v1（macOS Apple Silicon）隔离开发与验收

> 日期：2026-08-23
> 分支：`codex/easyicu-desktop-app-v1`
> worktree：`/Users/haibo/Documents/GitHub/EASYICU-desktop-app-v1`
> 基线：`1e5cda1657a524c4ced335d6950675f67f4fa72a`
> 状态：桌面封装和本机分发包完成；首轮全功能 UAT 的阻塞项已在同一隔离分支修复并完成受影响链路复验；未推送、未改写 `main`

## 结论

EasyICU 现可作为独立 macOS 应用启动。终端用户不需要源码目录、Python、Node 或 Git：Tauri 原生窗口负责应用生命周期，PyInstaller onedir 运行时承载现有 FastAPI WebApp，Node 运行时随包提供。科学执行、数据抽取、Idea Mining、Research Agent、EvidenceStore 和 publication gate 的 owner 均未复制到桌面层。

2026-08-23 的首轮全页面/全控件 UAT 发现 4 类可复现功能缺陷，其中 eICU 官方 demo 无法完成准备，直接阻塞真实 Cross-DB 双库链路。同日已在本隔离分支完成 owner-level 修复和受影响链路复验：eICU 官方 demo 可导出并注册，MIMIC-IV + eICU 官方双库聚合可达；Patient、Cross-DB 和两个 Copilot composer 的交互回归通过。首轮报告中的 Extraction `Extract again` 在当前源码与重建桌面包中均能正常返回推荐抽取页，未为无法复现的现象增加冗余兜底。

当前发行物可进入用户验收，但仍未做 Developer ID/notarization、Intel Mac/Windows、真实 Provider/E1 或 full exact-head CI，因此本任务不自行合并 `main`。

本轮只建立并验证 macOS Apple Silicon 内测发行物。当前 app 使用 ad-hoc 签名，尚未进行 Apple Developer ID 签名和 notarization，因此不能称为公开发行就绪；Windows 也尚未构建或验证。

## 实现边界

- 原生壳：`desktop/src-tauri/`；动态选择 `127.0.0.1` 非特权端口，显示启动页并在后端就绪后导航到 WebApp。
- Python 入口：`desktop/backend_entry.py`；应用状态和 runtime 写入 OS app-data，不覆盖真实 `HOME`。
- 桌面会话边界：`src/easyicu/webserver/desktop_session.py`；每次启动生成随机 token，探针使用私有 header，首页 bootstrap 将 token 换为 HttpOnly、SameSite=strict cookie，未授权请求返回 403。未设置桌面环境变量时，普通浏览器启动行为不变。
- 生命周期：原生退出主动终止子进程；Python parent-PID watcher 作为兜底，避免窗口退出后遗留后端。
- WebView 能力：启动页只有 Tauri core 默认能力，不向 FastAPI 页面暴露 shell 或 filesystem command。
- 构建：`desktop/scripts/build_macos.py` 创建隔离 venv，固定 PyInstaller `6.22.2`，安装 locked Node/Pi runtime，冻结 Python onedir runtime，构建 `.app`、ad-hoc codesign 并创建带 Applications 链接的 DMG。

## 验收证据

### 全页面 / 全控件 UAT（2026-08-23）

测试在独立 worktree 和两个临时 `EASYICU_HOME`/state root 中完成，不读取或覆盖正式用户状态。覆盖结果：

- 12 个主路由：Home/Entry、Guided Copilot、Idea Mining、Data Extraction、Patient Review、Cohort Statistics、Cross-DB、Project Monitor、Get Started、Data Dictionary、Settings、States。
- 6 个兼容/异常路由：`help`、`assistant`、`audit`、`sofareclass`、`icd`、未知 hash；均落到预期 owner 或安全首页。
- 148 个去重的页面级可见控件；另遍历 States 的 36 个组合、Dictionary 的 20 个分组按钮、Cohort 的 5 个视图和 6 种比较模式、Cross-DB 的 4 个结果页签、Settings 的 6 个页签、Project Monitor 的 5 个页签、Idea Mining 的 6 种来源、Guided 的 provider/source/local workflow 控件，以及 Extraction 的 8 个队列预设、4 个范围滑块、19 个模块、307 个概念、3 种格式和 2 种合并方式。
- 12 个主路由分别在 1280×720 与 1440×940 截图；页面水平溢出均为 0。证据目录：`output/playwright/desktop-full-qa/`（gitignored）。
- Demo 定制提取在 `/private/tmp/easyicu-custom-qa-20260823` 选择目标后完成；按设计只生成预览 ledger，目标目录保持空。
- 原生 `.app` 冷启动、Home→Patient→Extraction、推荐提取、About、File、View、Help 菜单和退出均实操；退出后 Tauri、冻结后端和动态端口全部释放。

#### 阻塞/高优先级发现

| 严重性 | 可复现发现 | 证据与边界 |
|---|---|---|
| P1 / 阻塞 | eICU 官方 demo（130.6 MB）下载和 parquet 转换成功，但“Continue preparation”稳定失败，错误 `demo_source_export_failed` → `DatabaseDetectionError` / `database_path_unrecognized`；catalog 停在 `converted`、`export_ready=false`、`registered=false`。 | MIMIC-IV 官方 demo（15.5 MB）同链路成功，得到 140 entities / 133,713 rows / 279 observed features；因此不是整条 demo 下载链路失效。eICU 失败使官方 Cross-DB 双库配置无法完成。测试 job：`339d3f6102ed`。 |
| P1 | 完成 Demo extraction 后 `Extract again` 按钮无动作；刷新、切换 Demo/Real、重启后复测仍可重现。 | 完成态发生 `repaint` 后按钮看似存在但未恢复有效 handler。推荐和定制 extraction 本身均可完成。 |
| P1 | Patient synthetic fallback 刚加载后，19 个模块按钮均可被点击/聚焦，但数据表持续停在 SOFA-2；官方 MIMIC 数据实际加载后 Lab Chemistry 切换正常。 | 同样指向状态重绘后的事件绑定缺口，而不是模块数据不可用。 |
| P1 | Cross-DB 搜索框接受 `lactate`，但 feature 计数和按钮集合不变（全部模块仍 307，Blood Gas 仍 9）。 | 模块过滤、Core/all mapped、数据库切换、Lactate 图和 4 个结果页签均正常，缺陷限定为搜索过滤。 |
| P2 | Guided local workflow 的 legacy composer 在 `Shift+Enter` 时直接发送；随后 Enter 又发送第二条，无法输入预期换行草稿。 | `screens-guided.js` 的 keydown handler 只判断 `e.key === 'Enter'`；Pi composer 已有 Shift/IME-safe 契约，但该 legacy local workflow 未覆盖。 |

上述发现本轮只记录，不在验收任务中顺手修改生产代码，以保持“测试结果”和“修复结果”可独立审计。

### UAT 缺陷修复与真实链路复验（2026-08-23）

| 首轮发现 | 根因 / 处理 | 复验证据 |
|---|---|---|
| eICU demo preparation 阻塞 | `BaseICULoader._setup_data_path()` 未给 `eicu_demo` 配置 eICU prepared-table markers；补同布局身份验证和 Python 回归。 | 复用已下载、已转换的官方 eICU v2.0.1 缓存，真实导出 `19` 个模块、`1,654,369` 行并注册成功。随后将既有官方 MIMIC-IV v2.2 导出登记到同一隔离状态，真实 Cross-DB 返回 `source_count=2`、`shared_modules=19`、`compatibility=compatible`、`raw_rows_returned=false`。 |
| Patient synthetic 模块切换回 SOFA-2 | demo 重绘时无条件 `reset()`；改用稳定 demo source key，只在来源变化时重置。 | Node owner 合同覆盖 demo `demographics → labs` 跨重绘保持；真实浏览器点击“实验室-生化”后 `aria-pressed=true` 且对应有界表格出现。 |
| Cross-DB search 输入后不筛选 | 仅监听 Enter/change/search；增加 160 ms `input` debounce，并在重绘后恢复搜索焦点。 | Node owner 合同改为只触发 `input`；真实浏览器输入 `oxygen`、不按 Enter，结果由 307 个映射特征缩为 1 个，Heart Rate 消失。 |
| legacy composer Shift+Enter 误发送 | legacy 与 Pi 各自实现键盘判断；新增 `composer-keyboard.js` 单一 owner，两个 composer 都复用 Enter/Shift/IME 契约。 | 可执行 JS + Python 静态合同覆盖 plain Enter、Shift+Enter、`isComposing`、legacy `keyCode 229` 和非 Enter；真实浏览器 Shift+Enter 后草稿未发送，plain Enter 后输入被提交清空。 |
| Extraction `Extract again` | 修复前源码 owner 已有正确直接绑定，本轮无法在清除测试干扰后复现；未添加第二套 handler。 | 真实浏览器完成推荐 demo extraction，点击“重新抽取”后推荐抽取 heading 恢复且重复按钮消失。 |

受影响页面 `extraction`、`patient`、`crossdb`、`guided` 在 1280×720 检查均 `scrollWidth == clientWidth`，无页面级横向溢出。

聚焦回归：

```text
tests/test_patient_filter_correctness.py + tests/test_pi_copilot_static.py +
tests/test_webserver_demo_sources.py
115 passed, 4 warnings

python tools/run_js_contracts.py
23/23 passed

desktop boundary tests
14 passed

Copilot data-source/concept/extraction/data-package/workbench contracts
6 passed
```

### Copilot 与数据工作台的当前能力审计

当前 Copilot 已经不是只能聊天的壳：它已有受治理工具可列出注册数据源、按 source/module/query 返回精确概念 ID、审阅绑定数据包的聚合分母/模块/概念可用性，并从对话中收集 StudyContext 后调用既有 Data Extraction owner。数据包可以作为 digest-bound、只读的嵌入式数据工作台显示在对话右侧；模型看不到主机路径或患者行。

但当前合同还没有两个专用只读工具：单来源任意特征的聚合分布，以及运行抽取前的 ICD 队列计数预览。现有 extraction owner 能在正式执行时计算 ICD cohort report，Cohort/Cross-DB owner 也能生成聚合图表；下一步应把这些 owner 增加为 bounded receipt 工具并把返回值渲染为 Copilot resource，而不是把 Data Workspace 的私有读取/筛选逻辑复制进 Copilot。

#### 已通过的主要功能链

- Guided：provider tabs、空配置不发送、故意不可达 endpoint 显示错误并清除 credential；本地项目/Idea/Prepare Data/Review Data/Run Project 的无 export 路径均 fail closed。
- Idea Mining：manual、URL、PDF、folder、Zotero、frontier 六种来源均渲染；空输入有明确校验；本地 manual idea 可产出 ledger。
- Patient/Cohort：synthetic fallback、官方 MIMIC 下载/转换/审阅、4 个 Patient tabs、5 个 Cohort views、6 种 comparison modes、load all modules 均通过（受上表 immediate synthetic module 切换缺陷限定）。
- Cross-DB：synthetic offline fallback 的数据库至少二选一门、运行、模块筛选、feature 图和 4 个结果页签通过；官方 pair 受 eICU 阻塞。
- Dictionary/Tutorial/Settings/States：搜索、多语言、分组、FAQ、诊断下载、显示密度、开关和 36 状态组合均通过。
- 安全边界：无 token 403、错误 token 403、正确 token 200、带 `Forwarded` 的请求 403。

### 聚焦合同

```text
python -m pytest -q \
  tests/test_webserver_desktop_session.py \
  tests/test_desktop_backend_entry.py \
  tests/test_desktop_distribution_contract.py
14 passed, 1 warning in 0.14s

cargo test --manifest-path desktop/src-tauri/Cargo.toml
3 passed; 0 failed
```

此前同一 patch 的 Ruff 聚焦检查通过；`desktop` npm audit 与随包 Node runtime 的 production audit 均为 0 vulnerabilities。该范围是桌面边界聚焦回归，不是 full CI，也不是 E1 科学验收。

扩大到当前 exact-head Web/desktop 相关矩阵：

```text
python -m pytest -q \
  tests/test_desktop_backend_entry.py \
  tests/test_desktop_distribution_contract.py \
  tests/test_webserver*.py \
  tests/test_static_frontend_ownership.py \
  tests/test_static_architecture_policy.py \
  tests/test_cohort_visualization_layout.py \
  tests/test_pi_copilot_static.py

785 passed, 2 failed, 5 warnings in 25.11s

python tools/run_js_contracts.py
22/22 passed
```

两项失败均位于桌面分支未修改的既有 Web 文件：Outcome fallback 计数 10 与后端 catalog 13 不一致；`screens-guided-pi.js` 1853 行超过 1787 行 ratchet。它们不是本桌面 patch 新引入，但属于当前 exact-head 未闭合基线，不能被聚焦通过数掩盖。

### 冻结后端与鉴权 smoke

- 无 token 的 `/api/catalog`：HTTP 403。
- 正确 `X-EasyICU-Desktop-Token`：HTTP 200。
- `/?desktop_token=...`：HTTP 303，并设置 HttpOnly、SameSite=strict cookie。
- cookie 后 `/`：HTTP 200，9,528 bytes。
- cookie 后 `/api/catalog`：HTTP 200，65,614 bytes。
- `/api/copilot/pi/status` 返回 `ok=true`、`pi_package_version=0.84.1`；运行阻塞仅为用户尚未配置 API key 与 EasyICU AI opt-in，桌面包没有绕过这些门。

### 原生桌面 UI 与退出

- Computer Use 从 `.app` 启动，先看到 `tauri://localhost` 本地启动页，再进入 `127.0.0.1:53754/` 的完整 `Welcome to EasyICU` 首页。
- 本机从启动调用到可访问首页约 `6,723 ms`；这是本机 smoke 计时，不是跨设备性能承诺。
- 1440×940 桌面窗口可见首页双栏、Guided Copilot、Classic Workspace、研究旅程和 Page guide；无可见横向裁切。
- `Cmd+Q` 后应用 `isRunning=false`；`pgrep` 无 `easyicu-backend`/`easyicu-desktop`；原动态端口返回连接失败、HTTP `000`。

### 分发物

```text
codesign --verify --deep --strict EasyICU.app
valid on disk; satisfies its Designated Requirement

hdiutil verify EasyICU_1.0.0_aarch64.dmg
checksum is VALID
```

- App：`desktop/src-tauri/target/release/bundle/macos/EasyICU.app`，本机约 980 MB。
- DMG：`desktop/src-tauri/target/release/bundle/dmg/EasyICU_1.0.0_aarch64.dmg`，本机约 438 MB。
- 修复后 DMG：456,690,588 bytes；SHA-256：`a8f5ae63a343accc6fdcb1762570160315b65026c0123f519836d44dcfc748a2`。
- 上述构建产物和 build/runtime 目录均被 `.gitignore` 排除，不进入源码提交。

## 尚未声称

- 未运行仓库 full exact-head CI；扩大 Web/desktop 矩阵为 785 passed / 2 pre-existing failed，不等于 release CI 通过。
- 未运行真实 Provider turn、普通 Web E1、完整 Research Agent 科学运行或正式科学实验；应用能启动不等于 Planner、临床、benchmark 或论文证据通过。
- 未使用真实 ChatGPT/API 凭据，未安装 extension/MCP，未点击会清空隔离设置的 `Reset to defaults`；这些有凭据、外部变更或破坏性的路径不伪装成已覆盖。
- 未做 Intel Mac、Windows、公开签名、notarization、自动更新或安装器升级测试。
- 未合并或推送此分支；`main` 的现有 dirty 工作保持隔离。

## 合并判定与下一步

当前判定：首轮阻塞项已归零，分支和修复后 Apple Silicon 内测包可进入用户验收；**本任务不自行合并或推送**。公开发行前仍需 Developer ID/notarization、目标 macOS 矩阵、full exact-head CI；科学就绪仍需单独的真实 Provider/E1 证据。
