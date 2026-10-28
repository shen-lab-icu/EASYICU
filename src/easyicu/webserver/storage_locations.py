"""Visible storage defaults and typed validation shared by local workflows.

Settings owns persistence. This owner performs bounded filesystem probes only;
it never opens a database or moves existing user data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from easyicu.webserver import hosted, state_paths

PATH_KEYS = frozenset({"data_dir", "project_dir"})
STORAGE_KEYS = PATH_KEYS | {"data_dir_confirmed", "data_dir_history"}


class StorageLocationError(ValueError):
    def __init__(self, code: str, reason: str, **detail: Any) -> None:
        self.code = code
        self.reason = reason
        self.detail = {"reason": reason, **detail}
        super().__init__(reason)


def visible_project_default() -> Path:
    if hosted.is_hosted():
        return state_paths.projects_root()
    home = Path.home() if hosted.deployment_kind() == "desktop" else state_paths.user_home()
    documents = home / "Documents"
    return (documents if documents.is_dir() else home) / "EasyICU"


def defaults() -> dict[str, Any]:
    parent = visible_project_default()
    return {"data_dir": str(parent / "Data"), "project_dir": str(parent)}


def validate_directory(value: Any, *, required_bytes: int = 0) -> Path:
    """Accept an absolute existing or creatable directory with a write probe."""
    text = str(value or "").strip()
    try:
        path = Path(text).expanduser()
        if not text or not path.is_absolute() or any(ord(ch) < 32 for ch in text):
            raise StorageLocationError("storage_path_absolute_required", "请输入完整的文件夹路径。")
        path = path.resolve()
        if path.exists() and not path.is_dir():
            raise StorageLocationError("storage_path_not_directory", "此路径不是文件夹。")
        ancestor = path
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise StorageLocationError("storage_path_not_directory", "上级路径不是文件夹。")
        if not os.access(ancestor, os.W_OK | os.X_OK):
            raise StorageLocationError("storage_path_not_writable", "没有此文件夹的写入权限。")
        free = shutil.disk_usage(ancestor).free
        if free < required_bytes:
            raise StorageLocationError("storage_insufficient_space", "此文件夹的可用空间不足。",
                                       required_bytes=required_bytes, available_bytes=free)
        existed = path.exists()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise StorageLocationError("storage_path_not_creatable", "无法创建此文件夹。") from None
        try:
            with tempfile.TemporaryFile(dir=path) as handle:
                handle.write(b"EasyICU storage check")
                handle.flush()
        except OSError:
            code = "storage_path_not_writable" if existed else "storage_path_not_creatable"
            raise StorageLocationError(code, "无法写入此文件夹。") from None
        return path
    except StorageLocationError:
        raise
    except (OSError, ValueError, RuntimeError):
        raise StorageLocationError("storage_path_unavailable", "无法访问此文件夹。") from None


def prepare_settings_patch(patch: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    from easyicu.webserver.settings import SettingsValidationError

    result = dict(patch)
    for key in patch.keys() & STORAGE_KEYS:
        try:
            if hosted.is_hosted():
                raise StorageLocationError("storage_managed_by_administrator", "存储位置由管理员设置。")
            if key == "data_dir_history":
                raise StorageLocationError("storage_history_read_only", "已有数据的位置由应用保留。")
            if key in PATH_KEYS:
                result[key] = str(validate_directory(patch[key]))
            elif key == "data_dir_confirmed" and patch[key]:
                validate_directory(patch.get("data_dir", current["data_dir"]))
        except StorageLocationError as exc:
            raise SettingsValidationError([{"key": key, "code": exc.code, "reason": exc.reason}]) from None
    if "data_dir" in result and result["data_dir"] != current["data_dir"]:
        result.setdefault("data_dir_confirmed", False)
        result["data_dir_history"] = list(dict.fromkeys([
            *current.get("data_dir_history", []), current["data_dir"],
        ]))
    return result


def data_directory() -> Path:
    from easyicu.webserver.settings import load_settings

    return Path(load_settings()["data_dir"])


def project_directory() -> Path:
    from easyicu.webserver.settings import load_settings

    return Path(load_settings()["project_dir"])


def previous_data_directories() -> tuple[Path, ...]:
    from easyicu.webserver.settings import load_settings

    return tuple(Path(value) for value in load_settings().get("data_dir_history", []) if isinstance(value, str))


def data_directory_confirmed() -> bool:
    from easyicu.webserver.settings import load_settings

    return bool(load_settings()["data_dir_confirmed"])
