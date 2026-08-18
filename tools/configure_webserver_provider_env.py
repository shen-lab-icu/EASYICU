#!/usr/bin/env python3
"""Create EasyICU's private provider env file without echoing secrets."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Mapping


DEFAULT_ENV_PATH = Path.home() / ".easyicu" / "provider.env"


def provider_env_names(provider: str) -> Dict[str, str]:
    text = str(provider or "openai").strip().lower()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from easyicu.research_agent.providers.capabilities import provider_profile

    profile = provider_profile(text)
    if profile is None:
        raise ValueError(f"unsupported provider: {provider!r}")
    return {
        "api_key": profile.api_key_env_names[0],
        "base_url": profile.base_url_env_names[0],
        "model": profile.model_env_names[0],
    }


def write_provider_env(
    path: Path,
    entries: Mapping[str, str],
    *,
    force: bool = False,
) -> Dict[str, object]:
    """Atomically write a 0600 env file and return sanitized metadata."""
    target = Path(path).expanduser()
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to overwrite")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    lines = [
        "# EasyICU private external-provider config",
        "# Created by tools/configure_webserver_provider_env.py",
        "# Keep this file mode 0600. Do not commit it.",
    ]
    for key, value in entries.items():
        clean_key = str(key).strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", clean_key):
            raise ValueError(f"invalid env key: {clean_key!r}")
        clean_value = str(value or "").strip()
        if clean_value:
            lines.append(f"{clean_key}={_quote_env_value(clean_value)}")
    payload = "\n".join(lines) + "\n"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(tmp, 0o600)
    os.replace(tmp, target)
    os.chmod(target, 0o600)
    return {
        "path": str(target),
        "mode": "0600",
        "keys": sorted(
            key for key, value in entries.items() if str(value or "").strip()
        ),
        "secrets_returned": False,
    }


def collect_entries(args: argparse.Namespace) -> Dict[str, str]:
    names = provider_env_names(args.provider)
    from easyicu.research_agent.providers.capabilities import provider_profile

    profile = provider_profile(args.provider)
    assert profile is not None  # argparse/provider_env_names already validated it
    api_key = getpass.getpass(f"{names['api_key']}: ").strip()
    base_url = str(args.base_url or "").strip()
    if not base_url:
        base_url = str(profile.default_base_url or "")
    if not base_url:
        transport_hint = (
            "Anthropic API base, e.g. https://api.anthropic.com"
            if profile.transport == "anthropic_messages"
            else "OpenAI-compatible base, e.g. http://127.0.0.1:8787/v1"
        )
        base_url = input(f"{names['base_url']} ({transport_hint}): ").strip()
    model = str(args.model or "").strip()
    if not model:
        model = input(f"{names['model']}: ").strip()
    max_tokens = str(args.max_tokens or "").strip()
    json_format_style = str(args.json_format_style or "").strip()
    entries = {
        names["api_key"]: api_key,
        names["base_url"]: base_url,
        names["model"]: model,
    }
    if max_tokens:
        entries["EASYICU_LLM_MAX_TOKENS"] = max_tokens
    if json_format_style:
        entries["EASYICU_LLM_JSON_FORMAT_STYLE"] = json_format_style
    missing = [
        key
        for key, value in entries.items()
        if key != "EASYICU_LLM_MAX_TOKENS" and not value
    ]
    if missing:
        raise ValueError(f"missing required values for: {', '.join(missing)}")
    return entries


def _quote_env_value(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:@+=,-]+", value):
        return value
    return json.dumps(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from easyicu.research_agent.providers.capabilities import SUPPORTED_PROVIDER_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default="openai",
        choices=list(SUPPORTED_PROVIDER_NAMES),
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", default="")
    parser.add_argument(
        "--json-format-style", choices=["chat", "responses", "both"], default=""
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        meta = write_provider_env(args.path, collect_entries(args), force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "secrets_returned": False}, indent=2
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"ok": True, **meta}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
