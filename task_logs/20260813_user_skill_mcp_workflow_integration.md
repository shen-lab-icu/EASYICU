# USER-EXTENSIONS-WORKFLOW-20260813

## 目标

为 EasyICU/PiAgent 提供用户可操作的 Skill 与 MCP 扩展管理，并让启用内容以可审计、不可漂移的方式进入后续新建的 Pi 会话和 Research Agent 运行。保留 Nature Figure/Writing 两个默认内置技能；不扩大论文证据权威。

## 已完成

- 新增 `easyicu.extensions` owner：标准 `SKILL.md` 解析、内容寻址版本对象、安装/更新、启停、移除、并发安全 registry、不可变激活快照和 path-free run receipt。
- 新增只读 Streamable HTTP MCP client：安装时必须声明 1–32 个工具白名单；调用前再次校验 endpoint、参数大小/深度、凭据/患者标识/宿主路径，并关闭 redirect。返回值经过有界投影，信任级别固定为 `untrusted_external_metadata`。
- 把通用出站 URL 策略下沉到 `easyicu.outbound_url_security`；只允许公共 HTTPS 或 loopback HTTP，拒绝 URL 凭据、query/fragment、云 metadata、private/link-local/reserved/multicast 地址。
- 新增 `/api/extensions` 管理 API：列表、Skill 安装、MCP 握手测试、MCP 安装、启停和移除；Pydantic request 均 `extra="forbid"`。
- Settings 新增独立 owner `screens-settings-extensions.js`，在“技能”和“MCP 工具”页提供粘贴/上传、阶段选择、握手列举、显式工具白名单、启停和移除。样式仅进入 `settings.css`，未写入 catch-all CSS/JS。
- Pi 新会话创建时冻结 `ExtensionActivationSnapshot`，重开会话继续使用原快照；registry 后续更新不会改变旧会话。Node sidecar 不读取用户文件路径，只能调用宿主的 `easyicu_list_extensions`、`easyicu_load_skill`、`easyicu_call_mcp_tool`。
- MCP 调用同时要求 Settings 总开关、会话冻结 server/tool 白名单和当轮 `mcp_read` grant；结果不能进入 scientific evidence authority。
- Web Research Agent 提交时冻结同一 registry；写作 Skill 只作为标明“不可信建议数据”的 user-prompt advisory 进入 Writer，不进入 system prompt。MCP 只写 path-free descriptor receipt，不直接进入 Writer。
- Nature Figure/Writing 继续默认开启并由既有 publication skill registry 驱动；用户 Skill 是额外可插拔层，不替代 figure source-data/code、证据绑定、隐私和 readiness 规则。

## 验证

### 聚焦回归

```text
.venv/bin/python -m pytest -q \
  tests/test_extension_registry.py \
  tests/test_webserver_extensions.py \
  tests/test_pi_copilot_extensions.py \
  tests/test_pi_copilot_contract.py \
  tests/test_pi_copilot_gateway.py \
  tests/test_pi_copilot_static.py \
  tests/test_webserver_static_routes.py \
  tests/test_pi_copilot_provider_config.py \
  tests/research_agent/test_user_extensions.py \
  tests/research_agent/test_pipeline_config_contract.py \
  tests/research_agent/test_publication_skills.py

206 passed, 1 warning
```

- `ruff check`（扩展、Pi、pipeline、Web route 与新增测试）通过。
- `node --check`（API、Settings owner、Pi UI、Node sidecar）通过。
- `python -m compileall` 通过；`git diff --check` 通过。

### 真实 MCP 互操作

在 loopback 启动 EasyICU 官方 Streamable HTTP MCP server，使用新 client 完成 initialize → list tools → allowlisted call：

```json
{"call_ok": true, "listed": true, "server": "local-easyicu", "tool_count": 14, "trust": "untrusted_external_metadata"}
```

验证结束后测试 MCP server 已关闭。

### 浏览器 QA

- 临时 extension home 下完成 Skill 粘贴安装、conversation/writing 阶段选择、启用和停用。
- MCP 握手成功列出 14 个工具；显式白名单 `research_agent.list_skills` 后安装成功。
- `/api/extensions/skills/install`、`/state`、`/mcp/test`、`/mcp/install` 均返回 200。
- 1440×900 与 1280×800：document/body 无水平 overflow，Settings tabs 全部在视口内，console 0 error。
- 截图：`output/playwright/settings-extensions-1440x900.png`、`output/playwright/settings-extensions-1280x800.png`。
- 浏览器、临时 Web server 和 MCP server 均已关闭；未触碰原有 8765 进程。

## 安全与产品边界

- v1 不保存 MCP token、自定义 header 或认证配置；只支持 Streamable HTTP。
- 用户 Skill 仅支持 `conversation` / `writing` 两个低权限阶段；未开放自定义可执行代码、Shell、宿主文件或网络权限。
- `disable-model-invocation: true` 的 Skill 可以安装但不能启用，因为当前产品没有独立的人工 slash-command 调用面。
- Plugin bundle 的签名、依赖、升级/回滚和权限清单尚未实现；本轮完成的是 Skill + MCP 管理器，不表述为开放插件市场。
- 未启动 Provider、正式 Canonical9 batch 或 full exact-head CI；这项工程完成不改变 paper authority 4/9。
