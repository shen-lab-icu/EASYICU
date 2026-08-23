# EasyICU Desktop v1（macOS Apple Silicon）隔离开发与验收

> 日期：2026-08-23
> 分支：`codex/easyicu-desktop-app-v1`
> worktree：`/Users/haibo/Documents/GitHub/EASYICU-desktop-app-v1`
> 基线：`1e5cda1657a524c4ced335d6950675f67f4fa72a`
> 状态：功能和本机分发包完成；未合并、未推送、未改写 `main`

## 结论

EasyICU 现可作为独立 macOS 应用启动。终端用户不需要源码目录、Python、Node 或 Git：Tauri 原生窗口负责应用生命周期，PyInstaller onedir 运行时承载现有 FastAPI WebApp，Node 运行时随包提供。科学执行、数据抽取、Idea Mining、Research Agent、EvidenceStore 和 publication gate 的 owner 均未复制到桌面层。

本轮只建立并验证 macOS Apple Silicon 内测发行物。当前 app 使用 ad-hoc 签名，尚未进行 Apple Developer ID 签名和 notarization，因此不能称为公开发行就绪；Windows 也尚未构建或验证。

## 实现边界

- 原生壳：`desktop/src-tauri/`；动态选择 `127.0.0.1` 非特权端口，显示启动页并在后端就绪后导航到 WebApp。
- Python 入口：`desktop/backend_entry.py`；应用状态和 runtime 写入 OS app-data，不覆盖真实 `HOME`。
- 桌面会话边界：`src/easyicu/webserver/desktop_session.py`；每次启动生成随机 token，探针使用私有 header，首页 bootstrap 将 token 换为 HttpOnly、SameSite=strict cookie，未授权请求返回 403。未设置桌面环境变量时，普通浏览器启动行为不变。
- 生命周期：原生退出主动终止子进程；Python parent-PID watcher 作为兜底，避免窗口退出后遗留后端。
- WebView 能力：启动页只有 Tauri core 默认能力，不向 FastAPI 页面暴露 shell 或 filesystem command。
- 构建：`desktop/scripts/build_macos.py` 创建隔离 venv，固定 PyInstaller `6.22.2`，安装 locked Node/Pi runtime，冻结 Python onedir runtime，构建 `.app`、ad-hoc codesign 并创建带 Applications 链接的 DMG。

## 验收证据

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
- DMG SHA-256：`41904be5fbb92f59f1b178db8f99e9ef3d6eacbe5fc10754266839e7bae9f7ef`。
- 上述构建产物和 build/runtime 目录均被 `.gitignore` 排除，不进入源码提交。

## 尚未声称

- 未运行 full exact-head CI；按当前 Web 开发策略，只在 E1 11/11 或冻结/合并/发布检查点运行。
- 未运行真实 Provider turn、普通 Web E1 或正式科学实验；应用能启动不等于 Planner、临床、benchmark 或论文证据通过。
- 未做 Intel Mac、Windows、公开签名、notarization、自动更新或安装器升级测试。
- 未合并或推送此分支；`main` 的现有 dirty 工作保持隔离。
