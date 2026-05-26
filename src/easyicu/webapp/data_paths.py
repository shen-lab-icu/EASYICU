"""Real-data path resolution and directory input helpers for the EasyICU webapp."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


def _get_database_download_info(database: str, lang: str = 'en') -> dict | None:
    """Return the official download/access page for a database."""
    download_map = {
        'miiv': {
            'name': 'MIMIC-IV',
            'url': 'https://physionet.org/content/mimiciv/',
        },
        'mimic': {
            'name': 'MIMIC-III',
            'url': 'https://physionet.org/content/mimiciii/',
        },
        'eicu': {
            'name': 'eICU-CRD',
            'url': 'https://physionet.org/content/eicu-crd/',
        },
        'aumc': {
            'name': 'AmsterdamUMCdb',
            'url': 'https://amsterdammedicaldatascience.nl/amsterdamumcdb/',
        },
        'hirid': {
            'name': 'HiRID',
            'url': 'https://hirid.intensivecare.ai/',
        },
        'sic': {
            'name': 'SICdb',
            'url': 'https://physionet.org/content/sicdb/',
        },
    }
    info = download_map.get(database)
    if not info:
        return None
    return {
        'name': info['name'],
        'url': info['url'],
        'label': (
            f"Open {info['name']} download page"
            if lang == 'en' else
            f"打开 {info['name']} 下载页"
        ),
        'note': (
            'Some databases require credentialed access or data use approval before download.'
            if lang == 'en' else
            '部分数据库需要先申请访问权限或完成数据使用审批后才能下载。'
        ),
    }


def _choose_directory_dialog(initial_dir: str = "") -> str | None:
    """Try to open a native folder picker for local desktop usage."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    start_dir = initial_dir or os.path.expanduser("~")
    try:
        start_dir = str(Path(start_dir).expanduser())
    except Exception:
        start_dir = os.path.expanduser("~")
    if not Path(start_dir).exists():
        start_dir = str(Path(start_dir).parent)

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(initialdir=start_dir)
        return selected or None
    except Exception:
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _closest_existing_dir(path_str: str, fallback: str = "") -> Path:
    candidate = (path_str or fallback or os.path.expanduser("~")).strip()
    try:
        path = Path(candidate).expanduser()
    except Exception:
        path = Path(os.path.expanduser("~"))
    if path.is_file():
        path = path.parent
    while not path.exists() and path != path.parent:
        path = path.parent
    if not path.exists():
        path = Path(os.path.expanduser("~"))
    return path


def _keep_directory_browser_open(browser_open_key: str) -> None:
    st.session_state[browser_open_key] = True


def _set_directory_browser_cwd(
    browser_open_key: str,
    browser_cwd_key: str,
    target_dir: str,
) -> None:
    st.session_state[browser_open_key] = True
    st.session_state[browser_cwd_key] = target_dir


def _select_directory_browser_cwd(
    browser_open_key: str,
    pending_input_key: str,
    target_dir: str,
) -> None:
    st.session_state[pending_input_key] = target_dir
    st.session_state[browser_open_key] = False


def _close_directory_browser(browser_open_key: str) -> None:
    st.session_state[browser_open_key] = False


def _render_directory_browser_dialog(
    *,
    input_key: str,
    button_key: str,
    value: str = "",
) -> None:
    lang = st.session_state.get("language", "en")
    browser_open_key = f"{button_key}_open"
    browser_cwd_key = f"{button_key}_cwd"
    browser_filter_key = f"{button_key}_filter"
    browser_show_hidden_key = f"{button_key}_show_hidden"
    browser_new_folder_key = f"{button_key}_new_folder"
    pending_input_key = f"{input_key}__pending_value"

    # Streamlit 1.45's native dialog "X" has no dismissal callback. Render the
    # browser as an inline, app-controlled panel and keep the open flag one-shot:
    # passive reruns close it, while in-panel navigation explicitly reopens it.
    st.session_state[browser_open_key] = False

    current_dir = _closest_existing_dir(
        st.session_state.get(browser_cwd_key, ""),
        st.session_state.get(input_key, value or ""),
    )
    st.session_state[browser_cwd_key] = str(current_dir)
    if browser_filter_key not in st.session_state:
        st.session_state[browser_filter_key] = ""
    if browser_show_hidden_key not in st.session_state:
        st.session_state[browser_show_hidden_key] = False
    if browser_new_folder_key not in st.session_state:
        st.session_state[browser_new_folder_key] = ""

    panel_title = (
        "Browse server folders"
        if lang == "en" else
        "浏览服务器目录"
    )
    header_hint = (
        "Choose a folder on the server running EasyICU."
        if lang == "en" else
        "选择运行 EasyICU 的服务器上的目录。"
    )
    st.markdown(
        f'<div class="server-browser-inline-title">{panel_title}</div>',
        unsafe_allow_html=True,
    )
    st.caption(header_hint)
    st.markdown(f'<div class="server-browser-path">{current_dir}</div>', unsafe_allow_html=True)

    nav_cols = st.columns([1, 1, 1.4, 1.4])
    with nav_cols[0]:
        up_label = "⬆ Up" if lang == "en" else "⬆ 上级"
        up_target = str(current_dir.parent if current_dir != current_dir.parent else current_dir)
        if st.button(
            up_label,
            key=f"{button_key}_dlg_up",
            use_container_width=True,
            on_click=_set_directory_browser_cwd,
            args=(browser_open_key, browser_cwd_key, up_target),
        ):
            st.session_state[browser_open_key] = True
            st.session_state[browser_cwd_key] = up_target
            st.rerun()
    with nav_cols[1]:
        home_label = "🏠 Home" if lang == "en" else "🏠 主目录"
        if st.button(
            home_label,
            key=f"{button_key}_dlg_home",
            use_container_width=True,
            on_click=_set_directory_browser_cwd,
            args=(browser_open_key, browser_cwd_key, str(Path.home())),
        ):
            st.session_state[browser_open_key] = True
            st.session_state[browser_cwd_key] = str(Path.home())
            st.rerun()
    with nav_cols[2]:
        select_label = "✅ Use This Folder" if lang == "en" else "✅ 使用当前目录"
        if st.button(
            select_label,
            key=f"{button_key}_dlg_select",
            use_container_width=True,
            on_click=_select_directory_browser_cwd,
            args=(browser_open_key, pending_input_key, str(current_dir)),
        ):
            st.session_state[pending_input_key] = str(current_dir)
            st.session_state[browser_open_key] = False
            st.rerun()
    with nav_cols[3]:
        close_label = "✕ Close" if lang == "en" else "✕ 关闭"
        if st.button(
            close_label,
            key=f"{button_key}_dlg_close",
            use_container_width=True,
            on_click=_close_directory_browser,
            args=(browser_open_key,),
        ):
            st.session_state[browser_open_key] = False
            st.rerun()

    tools_col1, tools_col2 = st.columns([1.4, 2.2])
    with tools_col1:
        st.checkbox(
            "Show hidden folders" if lang == "en" else "显示隐藏目录",
            key=browser_show_hidden_key,
            on_change=_keep_directory_browser_open,
            args=(browser_open_key,),
            help="Hidden folders starting with '.' are hidden by default." if lang == "en" else "默认隐藏以 . 开头的目录。",
        )
    with tools_col2:
        create_cols = st.columns([2.4, 1])
        with create_cols[0]:
            st.text_input(
                "New folder name" if lang == "en" else "新建文件夹名称",
                key=browser_new_folder_key,
                placeholder="e.g. exports_20260415" if lang == "en" else "例如 exports_20260415",
                label_visibility="collapsed",
                on_change=_keep_directory_browser_open,
                args=(browser_open_key,),
            )
        with create_cols[1]:
            create_label = "📁 Create" if lang == "en" else "📁 创建"
            if st.button(
                create_label,
                key=f"{button_key}_dlg_create",
                use_container_width=True,
                on_click=_keep_directory_browser_open,
                args=(browser_open_key,),
            ):
                new_folder_name = str(st.session_state.get(browser_new_folder_key, "")).strip()
                if not new_folder_name:
                    st.warning("Please enter a folder name first." if lang == "en" else "请先输入文件夹名称。")
                elif any(sep in new_folder_name for sep in ('/', '\\')) or new_folder_name in {'.', '..'}:
                    st.warning("Folder name cannot contain path separators." if lang == "en" else "文件夹名称不能包含路径分隔符。")
                else:
                    try:
                        target_dir = current_dir / new_folder_name
                        target_dir.mkdir(parents=False, exist_ok=False)
                        st.session_state[browser_open_key] = True
                        st.session_state[browser_cwd_key] = str(target_dir)
                        st.session_state[pending_input_key] = str(target_dir)
                        st.session_state[browser_new_folder_key] = ""
                        st.success(f"Created folder: {new_folder_name}" if lang == "en" else f"已创建文件夹：{new_folder_name}")
                        st.rerun()
                    except FileExistsError:
                        st.warning(f"Folder already exists: {new_folder_name}" if lang == "en" else f"文件夹已存在：{new_folder_name}")
                    except Exception as exc:
                        st.error(f"Create folder failed: {exc}" if lang == "en" else f"创建文件夹失败：{exc}")

    st.text_input(
        "Directory Filter" if lang == "en" else "目录筛选",
        key=browser_filter_key,
        placeholder="Filter subfolders..." if lang == "en" else "筛选子目录...",
        on_change=_keep_directory_browser_open,
        args=(browser_open_key,),
    )
    dir_filter = st.session_state.get(browser_filter_key, "").strip().lower()
    show_hidden = bool(st.session_state.get(browser_show_hidden_key, False))

    try:
        subdirs = [p for p in sorted(current_dir.iterdir(), key=lambda p: p.name.lower()) if p.is_dir()]
    except Exception as exc:
        st.error(f"Browse error: {exc}")
        subdirs = []

    if not show_hidden:
        subdirs = [p for p in subdirs if not p.name.startswith('.')]

    if dir_filter:
        subdirs = [p for p in subdirs if dir_filter in p.name.lower()]

    browser_list = st.container(height=460, border=True)
    with browser_list:
        if not subdirs:
            empty_msg = "No subdirectories found" if lang == "en" else "没有可用子目录"
            st.caption(empty_msg)
        else:
            shown_subdirs = subdirs[:120]
            for subdir in shown_subdirs:
                if st.button(
                    f"📁 {subdir.name}",
                    key=f"{button_key}_dlg_dir_{hash(str(subdir))}",
                    use_container_width=True,
                    on_click=_set_directory_browser_cwd,
                    args=(browser_open_key, browser_cwd_key, str(subdir)),
                ):
                    st.session_state[browser_open_key] = True
                    st.session_state[browser_cwd_key] = str(subdir)
                    st.rerun()
            if len(subdirs) > len(shown_subdirs):
                more_msg = (
                    f"Showing first {len(shown_subdirs)} directories. Narrow with the filter above."
                    if lang == "en" else
                    f"当前仅显示前 {len(shown_subdirs)} 个目录。可用上方筛选缩小范围。"
                )
                st.caption(more_msg)


def _directory_input(
    label: str,
    *,
    input_key: str,
    button_key: str,
    value: str = "",
    placeholder: str = "",
    help: str | None = None,
    label_visibility: str = "visible",
    show_value: bool = True,
) -> str:
    """Text input with a modal server-side directory browser."""
    lang = st.session_state.get("language", "en")
    browse_label = "📂"
    pending_input_key = f"{input_key}__pending_value"

    if pending_input_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_input_key)
    input_has_state = input_key in st.session_state

    browser_open_key = f"{button_key}_open"
    browser_cwd_key = f"{button_key}_cwd"
    browser_filter_key = f"{button_key}_filter"

    col_input, col_button = st.columns([8, 1.2])
    with col_input:
        text_input_kwargs = {
            "key": input_key,
            "placeholder": placeholder,
            "help": help,
            "label_visibility": label_visibility,
        }
        if not input_has_state and show_value:
            text_input_kwargs["value"] = value or ""
        typed_value = st.text_input(
            label,
            **text_input_kwargs,
        ).strip()
    with col_button:
        if label_visibility == "visible":
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        browse_help = (
            "Browse directories on the server running EasyICU"
            if lang == "en" else
            "浏览运行 EasyICU 的服务器目录"
        )
        if st.button(browse_label, key=button_key, use_container_width=True, help=browse_help):
            st.session_state[browser_open_key] = True
            st.session_state[browser_cwd_key] = str(_closest_existing_dir(typed_value, value))
            st.rerun()

    if st.session_state.get(browser_open_key, False):
        with st.container(border=True):
            _render_directory_browser_dialog(input_key=input_key, button_key=button_key, value=value)

    return typed_value


def _path_looks_like_database(path: str) -> bool:
    """检查路径是否看起来像数据库目录（包含 parquet/csv 文件或已知子目录）"""
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    entries_lower = [e.lower() for e in entries]
    # 包含 parquet 文件
    if any(e.endswith('.parquet') for e in entries_lower):
        return True
    # 包含已知子目录（MIMIC hosp/icu, eICU 表名, HiRID 分桶等）
    known_dirs = {'hosp', 'icu', 'observations_bucket', 'pharma_bucket',
                  'observation_tables', 'pharma_records', 'reference_data'}
    if known_dirs & set(entries_lower):
        return True
    # 包含 csv/csv.gz 文件
    if any(e.endswith('.csv') or e.endswith('.csv.gz') for e in entries_lower):
        return True
    # 包含以 _bucket 结尾的子目录（分桶数据）
    if any(e.endswith('_bucket') for e in entries_lower):
        return True
    return False


def find_database_path(root: str, db_name: str) -> str:
    """智能检测数据库路径，支持多种目录命名方式

    支持以下场景:
    - root=根目录, db_name=数据库 → root/alias[/version]
    - root=数据库目录本身 → 直接返回 root（当 root 目录名匹配别名或包含数据文件）
    - root=版本目录 → 直接返回 root（当 root 目录包含 parquet/csv 文件）

    Args:
        root: ICU数据根目录，或直接的数据库路径
        db_name: 数据库名称（miiv, eicu, aumc, hirid, mimic, sic）

    Returns:
        完整的数据库路径
    """
    # 定义每个数据库可能的目录名称和版本号
    db_aliases = {
        'miiv': ['mimiciv', 'mimic-iv', 'miiv', 'mimic_iv', 'mimic-iv-3.1'],
        'eicu': ['eicu', 'eicu-crd', 'eicu_crd'],
        'aumc': ['aumc', 'amsterdamumc', 'amsterdam'],
        'hirid': ['hirid', 'hi-rid'],
        'mimic': ['mimiciii', 'mimic-iii', 'mimic3', 'mimic_iii'],
        'sic': ['sicdb', 'sic', 'sic-db'],
    }

    aliases = db_aliases.get(db_name, [db_name])

    # ===== 优先检查: root 本身就是数据库目录 =====
    if os.path.isdir(root):
        root_basename = os.path.basename(os.path.normpath(root)).lower()
        # 1) root 目录名精确匹配或包含数据库别名
        matched = root_basename in aliases
        if not matched:
            for alias in aliases:
                if alias in root_basename or root_basename.startswith(alias):
                    matched = True
                    break
        if matched:
            # 可能还有版本子目录
            try:
                subdirs = [d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d))
                           and d[0].isdigit()]
            except OSError:
                subdirs = []
            if subdirs:
                subdirs.sort(reverse=True)
                return os.path.join(root, subdirs[0])
            return root
        # 2) root 目录包含数据文件（用户直接指向了版本目录或扁平数据库目录）
        if _path_looks_like_database(root):
            return root

    # ===== 常规搜索: root 是父目录 =====
    for alias in aliases:
        # 尝试直接目录
        direct_path = os.path.join(root, alias)
        if os.path.isdir(direct_path):
            # 检查是否有版本子目录
            try:
                subdirs = [d for d in os.listdir(direct_path)
                           if os.path.isdir(os.path.join(direct_path, d))
                           and d[0].isdigit()]
            except OSError:
                subdirs = []
            if subdirs:
                subdirs.sort(reverse=True)
                return os.path.join(direct_path, subdirs[0])
            else:
                return direct_path

        # 尝试带版本的固定路径
        default_versions = {
            'mimiciv': '3.1', 'mimic-iv': '3.1', 'miiv': '3.1',
            'eicu': '2.0.1', 'eicu-crd': '2.0.1',
            'aumc': '1.0.2',
            'hirid': '1.1.1',
            'mimiciii': '1.4', 'mimic-iii': '1.4',
            'sicdb': '1.0.6', 'sic': '1.0.6',
        }
        if alias in default_versions:
            versioned_path = os.path.join(root, alias, default_versions[alias])
            if os.path.isdir(versioned_path):
                return versioned_path

    # ===== 模糊匹配: 扫描 root 下目录名是否部分匹配 =====
    if os.path.isdir(root):
        try:
            for entry in os.listdir(root):
                entry_path = os.path.join(root, entry)
                if not os.path.isdir(entry_path):
                    continue
                entry_lower = entry.lower()
                # 检查目录名是否包含任何别名（如 "my_sic_data" 包含 "sic"）
                for alias in aliases:
                    if alias in entry_lower:
                        # 再检查版本子目录
                        try:
                            subdirs = [d for d in os.listdir(entry_path)
                                       if os.path.isdir(os.path.join(entry_path, d))
                                       and d[0].isdigit()]
                        except OSError:
                            subdirs = []
                        if subdirs:
                            subdirs.sort(reverse=True)
                            return os.path.join(entry_path, subdirs[0])
                        return entry_path
        except OSError:
            pass

    # 回退：返回 root 本身（而非拼接不存在的路径）
    return root


def _default_real_data_root() -> str:
    """Prefer the already validated sidebar data path for real-data analysis panels."""
    current_path = st.session_state.get('data_path')
    if current_path:
        return str(current_path)
    return os.environ.get('EASYICU_DATA_PATH', '')


def _default_real_database() -> str:
    """Return the current sidebar database selection when available."""
    current_db = st.session_state.get('database')
    return current_db if current_db in {'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic'} else 'miiv'


def _sync_real_data_panel_defaults(
    *,
    root_key: str,
    db_key: str | None = None,
    multi_db_key: str | None = None,
) -> None:
    """Seed cohort-analysis widgets from the validated sidebar real-data setup.

    Streamlit keeps widget values in session_state once a widget key exists. This
    means a cohort subpanel opened before Step 1 validation can keep an empty
    path even after the sidebar path is validated. Values that came from the
    sidebar keep following the sidebar; values manually changed inside a subpanel
    are preserved.
    """
    default_root = _default_real_data_root()
    root_sync_key = f"_{root_key}_synced_from_sidebar"
    current_root = st.session_state.get(root_key)
    previous_synced_root = st.session_state.get(root_sync_key)
    if default_root and (not current_root or current_root == previous_synced_root):
        st.session_state[root_key] = default_root
        st.session_state[root_sync_key] = default_root

    default_db = _default_real_database()
    valid_dbs = {'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic'}
    if db_key:
        db_sync_key = f"_{db_key}_synced_from_sidebar"
        current_db = st.session_state.get(db_key)
        previous_synced_db = st.session_state.get(db_sync_key)
        if current_db not in valid_dbs or current_db == previous_synced_db:
            st.session_state[db_key] = default_db
            st.session_state[db_sync_key] = default_db
    if multi_db_key:
        multi_sync_key = f"_{multi_db_key}_synced_from_sidebar"
        current_multi = st.session_state.get(multi_db_key)
        previous_synced_multi = st.session_state.get(multi_sync_key)
        if not current_multi or current_multi == previous_synced_multi:
            st.session_state[multi_db_key] = [default_db]
            st.session_state[multi_sync_key] = [default_db]


def render_directory_structure_guide(lang: str = 'en'):
    """渲染目录结构指南弹窗"""
    with st.popover("📂 " + ("Directory Structure Guide" if lang == 'en' else "目录结构指南")):
        struct_info = """
**Expected directory structure:**

```
icudb/                    ← Your ICU Data Root
├── mimiciv/              ← or mimic-iv/, miiv/
│   └── 3.1/              ← version folder (optional)
├── eicu/
│   └── 2.0.1/
├── aumc/
│   └── 1.0.2/
├── hirid/
│   └── 1.1.1/
├── mimiciii/             ← or mimic-iii/, mimic/
│   └── 1.4/
└── sicdb/                ← or sic/
    └── 1.0.6/
```

**Tips:**
- Version folders (3.1, 2.0.1, etc.) are optional
- Database folder names can vary (mimiciv, mimic-iv, miiv)
- System will auto-detect the correct path
""" if lang == 'en' else """
**期望的目录结构：**

```
icudb/                    ← 你的ICU数据根目录
├── mimiciv/              ← 或 mimic-iv/, miiv/
│   └── 3.1/              ← 版本文件夹（可选）
├── eicu/
│   └── 2.0.1/
├── aumc/
│   └── 1.0.2/
├── hirid/
│   └── 1.1.1/
├── mimiciii/             ← 或 mimic-iii/, mimic/
│   └── 1.4/
└── sicdb/                ← 或 sic/
    └── 1.0.6/
```

**提示：**
- 版本文件夹 (3.1, 2.0.1 等) 是可选的
- 数据库文件夹名称可以变化 (mimiciv, mimic-iv, miiv)
- 系统会自动检测正确的路径
"""
        st.markdown(struct_info)
