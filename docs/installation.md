# Install and run / 安装与启动

| 使用方式 | 需要下载 | 本机要求 |
|---|---|---|
| 桌面界面 | 对应平台的 EasyICU 安装包 | 安装包说明中的操作系统版本；自带 Python 与 Node |
| Python API / 浏览器界面 | EasyICU wheel | Python 3.10+；浏览器界面需 `webapp` 依赖 |
| 开发或复现检查 | Git 源码仓库 / sdist | Python 与对应开发依赖 |

软件发行以 [GitHub Releases](https://github.com/shen-lab-icu/EASYICU/releases)
中具体版本的资产与说明为准。演示数据 Release 提供数据，不能作为应用安装包使用。

## macOS 桌面包

当前桌面构建目标为 Apple Silicon。打开提供的 `.dmg`，把 `EasyICU.app`
拖入 Applications，再启动应用。安装包自带运行环境，不需要 Git 或源码目录。
本地测试构建只有 ad-hoc 签名；公开分发需要完成 Developer ID 签名与公证。

工作区旁的旧 `EasyICU.app` 是本机源码启动器，它通过 `project-path.txt`
定位源码；桌面构建产物则来自 `desktop/src-tauri/target/release/bundle/`。
维护者的构建与依赖更新流程见 [desktop/README.md](../desktop/README.md)。

## Python wheel

以下命令在包含 wheel 的目录中执行；将文件名替换为收到的版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "./easyicu-1.0.0-py3-none-any.whl[webapp]"
python -m pip check
easyicu-webapp
```

Windows 激活命令为 `.venv\Scripts\activate`。仅使用 Python API 时可以去掉
`[webapp]`。启动后打开命令输出的本地地址；默认是 `http://127.0.0.1:8765`。
关闭终端中的服务即可停止浏览器版本。

如尚未获得 wheel，可直接从代码仓库安装，机器需有 Git：

```bash
python -m pip install "easyicu[webapp] @ git+https://github.com/shen-lab-icu/EASYICU.git"
```

该命令跟随主线开发状态；需要固定环境时，在 Git URL 后添加选定提交或标签，
并保存本次环境依赖快照。

## 首次使用

启动后选择演示数据，或连接自己有访问权限的数据目录。原始 CSV / CSV.GZ /
tar.gz 必须先完成转换；Python 提取 API 接受准备完成的数据目录。
研究流程从问题与数据源开始，审阅完整计划后执行，再查看数据、图表和证据。
外部模型功能按应用中的明确启用与配置流程使用。

个人数据、模型凭据和研究输出保存在用户自己的目录，不需加入 Git。
升级前备份研究目录和应用状态，保留此前使用的安装包与版本信息。
科学结论与临床映射状态以产品声明和对应验证证据为准。

## Source builds and verification

Build a source distribution from a clean Git checkout with `python -m build`.
The sdist includes tracked tests, fixtures, tools, and documentation through the
setuptools-scm file finder. A wheel contains the installable product resources.
Git is needed for the original source inventory, but installing an existing
wheel or sdist does not require the development checkout.

Contributors should follow [CONTRIBUTING.md](../CONTRIBUTING.md). The macOS
desktop dependency lock has its own target and does not restrict the Python
library's supported version range.
