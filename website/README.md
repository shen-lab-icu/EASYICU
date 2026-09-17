# EasyICU 项目官网

本地多页静态官网，独立于 `127.0.0.1:8765` 研究工作台。当前预览：<http://127.0.0.1:54591/>。本轮完成页面与交互；尚未部署或公开发布。

## 页面与功能

| 页面 | 内容 |
| --- | --- |
| `dist/index.html` | 产品定位、真实工作台截图、基础能力、研究流程、成果阅读入口 |
| `dist/demos.html` | 两种当前界面切换、两步截图导览和原图预览 |
| `dist/catalog.html` | 6 个数据库、274 项临床概念、19 个模块、15 项研究能力；搜索、组合筛选、分页和方法说明 |
| `dist/guide.html` | 安装与启动、数据准备、研究流程、外部模型配置、8 项 FAQ；按系统切换与复制命令 |
| `dist/about.html` | 项目说明、设计原则、GitHub 反馈入口和可复制的软件引用 |

团队姓名、机构和联系邮箱由用户明确暂不确定，因此页面使用公开项目入口，没有增加团队身份或机构背书。自动播放是基于真实截图的步骤导览，不是操作录像，也不会触发分析。

## 维护与本地预览

无需 Node 包安装或前端构建工具。仅向静态服务器提供 `dist/`，不要把仓库根目录或本目录的来源记录暴露出去。

在仓库根目录执行：

```sh
# 用仓库环境更新目录快照（读取临床概念和研究能力登记）。
.venv/bin/python website/scripts/export_catalog.py
# 从共享模板生成五个 HTML 页面。
python3 website/scripts/build_site.py
# 仅在没有现有预览服务时启动。
python3 -m http.server 54591 --bind 127.0.0.1 --directory website/dist
```

内容和 HTML 模板位于 `scripts/build_site.py`；目录导出逻辑位于 `scripts/export_catalog.py`。`dist/styles.css` 和 `dist/site.js` 是直接维护的源文件，构建不会覆盖它们。目录快照 `dist/catalog-data.js` 由导出脚本生成。没有后端、分析服务、遥测、外部字体或模型请求。

## 内容与素材来源

- 页面依据仓库 README、产品共识、概念字典、能力登记和 2026-09-16 当前进度编写。目录数量代表登记数量，不是跨库等价性或所有方法的独立验证结果。
- 工作台及并排阅读截图：`outputs/easyicu_ui_review_20260916/07-workspace-refined-chrome.png` 与 `10-reader-refined-chrome.png`；已与 2026-09-16 当前 Chrome 工作台核对。屏幕展示的是已有研究项目。来源截图位于 gitignored 的 `outputs/` 未入库，可用本地工作台按 `asset-provenance.json` 中的 SHA-256 重新核对。
- 已删除官网内早期队列截图及改版前的两份工作台截图，移除对应标签与导览步骤。后续选用截图须先与当前运行 UI 核对，不按“同日截图”推定其为最终版本。
- 未发表的 E2 `analysis_only` 研究图件已从官网源文件、构建产物和素材清单中移除，不进入 Git 历史。
- `asset-provenance.json` 保存 2 项界面截图的逻辑来源和 SHA-256；`catalog-provenance.json` 保存目录输入及哈希。这些维护记录不在被服务的 `dist/` 中。
- 尚未上传任何素材到托管平台。公开上线前需确定发布目的地并补充已确定的团队信息；未确定的团队信息可以继续省略。

## focused 验证

检查基线：现有开发 checkout `codex/dev9-web-acceptance-20260906`，HEAD `1802174b13faeccd41f336f8bedbf5cdfb603e33` 加本次 `website/` 未提交内容。其他任务的工作台与后端改动保留，本轮未重启研究服务。

已通过 HTML ID、页面锚点和本地素材完整性检查，五页 HTTP 200，两个 JavaScript 文件语法检查。公开 GitHub 与数据库介绍链接结果保存在 `link-check.json`（含 checked_at/head/branch 绑定头，本地 fallback 锚点已在 dist 内验证）。

浏览器检查包括：375 / 390 px 手机与 1280 / 1552 px 桌面视口、五页无页面级横向溢出、手机菜单及 Escape 关闭、截图与键盘切换、原图预览及 Escape 关闭、两步导览及自动播放结束、概念搜索/组合筛选/空结果/重置/分页、方法筛选、安装系统切换与实际剪贴板内容、FAQ 展开、软件引用复制。控制台无 error / warn。检查范围和最终文件哈希见 `verification.json`。

这是静态官网 focused 验收，不是完整应用 CI、研究方法验证或发表授权。
