from datetime import date
from pathlib import Path

from tools.lint_progress import lint_current


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
