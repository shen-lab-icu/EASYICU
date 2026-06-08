"""Shared LLM configuration helpers for EasyICU web UI surfaces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import streamlit as st


ProviderInfo = tuple[str, str, str, bool, str, str]
INTERNAL_PROVIDER_KEYS = {"easyicu_hosted"}
_LOCAL_ENV_LOADED = False


def _load_local_env_file() -> None:
    """Load repo-local LLM defaults without exposing or overriding secrets."""
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    _LOCAL_ENV_LOADED = True
    for root in (Path.cwd(), *Path.cwd().parents):
        env_path = root / ".env.local"
        if not env_path.exists() or not env_path.is_file():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        return


def hosted_base_url() -> str:
    """Return the configured EasyICU hosted relay URL, if any."""
    _load_local_env_file()
    return (
        os.getenv("EASYICU_HOSTED_BASE_URL")
        or os.getenv("EASYICU_LLM_BASE_URL")
        or "http://47.241.42.236/v1"
    ).strip()


def default_provider_key() -> str:
    """Prefer hosted mode when a hosted relay URL is configured."""
    return "easyicu_hosted" if hosted_base_url() else "openrouter"


def public_default_provider_key() -> str:
    """Default provider exposed in end-user UI surfaces."""
    return default_provider_key()


PROVIDERS: Dict[str, ProviderInfo] = {
    "easyicu_hosted": (
        "EasyICU Hosted",
        hosted_base_url(),
        "hosted-default",
        False,
        "Use the EasyICU managed relay. No user API key required.",
        "使用 EasyICU 托管代理，无需用户自己填写 API Key。",
    ),
    "huggingface_free": (
        "HuggingFace",
        "https://router.huggingface.co/v1",
        "deepseek-ai/DeepSeek-R1:fastest",
        True,
        "Requires your own Hugging Face token.",
        "需要用户自己提供 Hugging Face token。",
    ),
    "openai": (
        "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4o",
        True,
        "GPT-4o, GPT-4o-mini, o1, etc.",
        "GPT-4o、GPT-4o-mini、o1 等。",
    ),
    "deepseek": (
        "DeepSeek",
        "https://api.deepseek.com",
        "deepseek-chat",
        True,
        "DeepSeek-V3, DeepSeek-R1. Very affordable.",
        "DeepSeek-V3、DeepSeek-R1，价格极低。",
    ),
    "anthropic": (
        "Anthropic",
        "https://api.anthropic.com/v1",
        "claude-sonnet-4-20250514",
        True,
        "Claude Sonnet, Opus, Haiku.",
        "Claude Sonnet、Opus、Haiku。",
    ),
    "openrouter": (
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "openai/gpt-oss-120b:free",
        True,
        "Aggregator with free & paid models. Get key at openrouter.ai",
        "模型聚合平台，有免费和付费模型。在 openrouter.ai 获取 Key。",
    ),
    "together": (
        "Together AI",
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        True,
        "Llama, Mistral, Qwen. Free tier available (signup).",
        "Llama、Mistral、Qwen。注册即有免费额度。",
    ),
    "groq": (
        "Groq",
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
        True,
        "Ultra-fast inference. Free tier with rate limits.",
        "超低延迟推理。免费套餐有速率限制。",
    ),
    "siliconflow": (
        "SiliconFlow (硅基流动)",
        "https://api.siliconflow.cn/v1",
        "deepseek-ai/DeepSeek-V3",
        True,
        "China-based, DeepSeek/Qwen. Free tier available.",
        "国内平台，DeepSeek/Qwen 等模型，注册赠送额度。",
    ),
    "custom": (
        "Custom / Compatible",
        "",
        "",
        True,
        "Any OpenAI-compatible endpoint.",
        "任意 OpenAI 兼容接口。",
    ),
}


def ensure_llm_config_state() -> None:
    """Ensure the shared LLM keys exist in Streamlit session state."""
    _load_local_env_file()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    openrouter_model = os.getenv("EASYICU_OPENROUTER_MODEL", "").strip()
    has_openrouter_env = bool(openrouter_key)
    provider_default = "openrouter" if has_openrouter_env else default_provider_key()
    defaults = {
        "llm_enabled": False,
        "llm_provider": provider_default,
        "llm_api_key": openrouter_key if has_openrouter_env else "",
        "llm_model": openrouter_model if has_openrouter_env else "",
        "llm_base_url": openrouter_base_url if has_openrouter_env else "",
        "llm_configured": has_openrouter_env,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def needs_api_key(provider: str) -> bool:
    return PROVIDERS.get(provider, PROVIDERS["custom"])[3]


def provider_defaults(provider: str) -> ProviderInfo:
    return PROVIDERS.get(provider, PROVIDERS["custom"])


def is_internal_provider(provider: str) -> bool:
    return provider in INTERNAL_PROVIDER_KEYS


def public_provider_keys() -> list[str]:
    return list(PROVIDERS.keys())


def coerce_public_provider(provider: str) -> str:
    return provider if provider in PROVIDERS else "custom"


def public_provider_defaults(provider: str) -> ProviderInfo:
    return provider_defaults(coerce_public_provider(provider))


def is_configured() -> bool:
    ensure_llm_config_state()
    provider = st.session_state.get("llm_provider", default_provider_key())
    _, default_url, _default_model, _needs_key, _desc_en, _desc_zh = provider_defaults(provider)
    if needs_api_key(provider) and not st.session_state.get("llm_api_key", "").strip():
        return False
    return bool((st.session_state.get("llm_base_url", "") or default_url).strip())


@dataclass(frozen=True)
class AgentLLMConfig:
    """Research-agent compatible LLM settings."""

    choice: str
    api_key: str
    model: str
    base_url: Optional[str]
    extra_headers: Optional[dict[str, str]]
    source_label: str


def agent_config_from_shared_settings() -> AgentLLMConfig:
    """Convert the sidebar LLM store into a research-agent client config."""
    ensure_llm_config_state()
    provider = st.session_state.get("llm_provider", default_provider_key())
    display, default_url, default_model, _needs_key, _desc_en, _desc_zh = provider_defaults(provider)
    api_key = st.session_state.get("llm_api_key", "").strip()
    model = st.session_state.get("llm_model", "").strip() or default_model
    base_url = st.session_state.get("llm_base_url", "").strip() or default_url or None

    if provider == "openai":
        choice = "OpenAI"
        base_url = None if base_url == "https://api.openai.com/v1" else base_url
        extra_headers = None
    elif provider == "openrouter":
        choice = "OpenRouter"
        extra_headers = {
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": "EasyICU research-agent webapp",
        }
    else:
        choice = "Custom OpenAI-compatible"
        extra_headers = None

    if provider == "easyicu_hosted" and not api_key:
        api_key = os.getenv("EASYICU_HOSTED_CLIENT_TOKEN", "easyicu-hosted")

    return AgentLLMConfig(
        choice=choice,
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_headers=extra_headers,
        source_label=display,
    )
