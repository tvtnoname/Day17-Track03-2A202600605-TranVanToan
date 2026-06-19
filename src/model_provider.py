from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "anthropic-ai": "anthropic",
    "open-ai": "openai",
    "open_ai": "openai",
    "gpt": "openai",
    "google": "gemini",
    "google-generativeai": "gemini",
    "openrouter-ai": "openrouter",
}


def normalize_provider(value: str) -> str:
    v = value.strip().lower()
    return _ALIASES.get(v, v)


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unsupported provider: {config.provider!r}")
