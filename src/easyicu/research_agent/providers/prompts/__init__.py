"""Versioned prompt-pack loader for the research-agent layer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


PROMPT_PACK_VERSION = "easyicu-research-agent-prompts/v1"
_PROMPT_ROOT = Path(__file__).resolve().parent / "v1"
_PROMPT_FILES = {
    "system": "system.txt",
    "coder": "coder.txt",
    "replanner": "replanner.txt",
    "writer": "writer.txt",
    "nature_writing": "nature_writing.txt",
}


def load_prompt_pack() -> Dict[str, str]:
    pack: Dict[str, str] = {}
    for key, rel in _PROMPT_FILES.items():
        path = _PROMPT_ROOT / rel
        pack[key] = path.read_text(encoding="utf-8").strip()
    return pack


def prompt_pack_files() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, rel in _PROMPT_FILES.items():
        path = _PROMPT_ROOT / rel
        out[f"{PROMPT_PACK_VERSION}/{rel}"] = _sha256(path)
    return out


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


__all__ = [
    "PROMPT_PACK_VERSION",
    "load_prompt_pack",
    "prompt_pack_files",
]
