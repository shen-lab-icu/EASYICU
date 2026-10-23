"""Tiny synthetic Node packages for offline desktop staging tests."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DESKTOP / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_files(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def node_app(root: Path) -> Path:
    write_files(root, {
        "package.json": '{"type":"module"}',
        "src/main.mjs": "export const gateway = true;\n",
        "src/skills/web-prototype/SKILL.md": "Synthetic skill\n",
    })
    for name in ("@earendil-works/pi-coding-agent", "@earendil-works/pi-ai", "typebox"):
        write_files(root / "node_modules" / name, {
            "package.json": json.dumps({"name": name, "version": "1.0.0", "type": "module", "exports": "./index.js"}),
            "index.js": "export const synthetic = true;\n",
            "index.d.ts": "export declare const synthetic: boolean;\n",
            "index.js.map": "{}\n",
            "dist-types/unused.js": "// never executed\n",
            "docs/guide.md": "Synthetic documentation\n",
            "tests/unused.js": "throw new Error('test only');\n",
            "examples/unused.js": "// synthetic example\n",
        })
    write_files(root / "node_modules/@earendil-works/pi-ai", {
        "index.js": 'export const loadProvider = () => import("./providers/openai-completions.js");\n',
        **{f"providers/{name}.js": "export const synthetic = true;\n" for name in (
            "anthropic", "google", "openai-completions", "openai-responses", "openai-codex-responses",
        )},
    })
    return root


def node_package(root: Path, *, name="synthetic", version="1.0.0", **manifest) -> Path:
    write_files(root, {
        "package.json": json.dumps({"name": name, "version": version, "type": "module",
                                    "exports": "./index.js", **manifest}),
        "index.js": "export const value = 1;\n",
    })
    return root
