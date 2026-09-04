from datetime import date
from pathlib import Path

from tools.lint_progress import lint_current, lint_root


def _current_text(*references: str) -> str:
    return "\n".join(
        [
            "更新: 2026-08-13",
            "## 🎯 当前真相（一句话）",
            "Current.",
            "## 📍 基线",
            "baseline",
            "## 🔨 正在做",
            "work",
            "## ✅ 已完成",
            "done",
            "## ⏭️ 下一步",
            "next",
            "## ⚠️ 不要做",
            "avoid",
            "## 📚 证据",
            *references,
        ]
    )


def test_progress_lint_resolves_each_task_log_in_its_declared_owner(
    tmp_path: Path,
) -> None:
    current = tmp_path / "项目进度" / "agent" / "CURRENT.md"
    current.parent.mkdir(parents=True)
    current.write_text(
        _current_text(
            "`EASYICU/task_logs/repo.md`",
            "`task_logs/workspace.md`",
        ),
        encoding="utf-8",
    )
    (tmp_path / "EASYICU" / "task_logs").mkdir(parents=True)
    (tmp_path / "EASYICU" / "task_logs" / "repo.md").write_text(
        "repo", encoding="utf-8"
    )
    (tmp_path / "task_logs").mkdir()
    (tmp_path / "task_logs" / "workspace.md").write_text("workspace", encoding="utf-8")

    errors, _warnings = lint_current(
        current,
        date(2026, 8, 13),
        21,
        workspace_root=tmp_path,
    )

    assert errors == []


def test_progress_lint_fails_on_a_missing_current_evidence_pointer(
    tmp_path: Path,
) -> None:
    current = tmp_path / "项目进度" / "web" / "CURRENT.md"
    current.parent.mkdir(parents=True)
    current.write_text(
        _current_text("`EASYICU/task_logs/missing.md`"),
        encoding="utf-8",
    )

    errors, _warnings = lint_current(
        current,
        date(2026, 8, 13),
        21,
        workspace_root=tmp_path,
    )

    assert errors == ["web/CURRENT.md:15: 缺失仓库证据 `EASYICU/task_logs/missing.md`"]


def test_progress_lint_rejects_duplicate_current_files_in_module_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "项目进度"
    current = root / "agent" / "CURRENT.md"
    current.parent.mkdir(parents=True)
    current.write_text(_current_text(), encoding="utf-8")
    (current.parent / "CURRENT 2.md").write_text(_current_text(), encoding="utf-8")
    (root / "README.md").write_text(
        "| [agent](agent/CURRENT.md) | current | in_progress | 2026-08-13 | X |\n",
        encoding="utf-8",
    )

    errors, _warnings = lint_root(root, today=date(2026, 8, 13), stale_days=21)

    assert errors == [
        "agent/CURRENT 2.md: 模块根目录存在非标准 CURRENT 副本 —— "
        "当前真相只能叫 CURRENT.md；历史快照移入 history/"
    ]


def test_progress_lint_allows_current_snapshots_under_history(tmp_path: Path) -> None:
    root = tmp_path / "项目进度"
    current = root / "agent" / "CURRENT.md"
    current.parent.mkdir(parents=True)
    current.write_text(_current_text(), encoding="utf-8")
    history = current.parent / "history" / "CURRENT_20260813.md"
    history.parent.mkdir()
    history.write_text(_current_text(), encoding="utf-8")
    (root / "README.md").write_text(
        "| [agent](agent/CURRENT.md) | current | in_progress | 2026-08-13 | X |\n",
        encoding="utf-8",
    )

    errors, _warnings = lint_root(root, today=date(2026, 8, 13), stale_days=21)

    assert errors == []
