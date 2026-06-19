from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    api_key: str | None = None
    base_url: str | None = None

    p = provider.lower()
    if p in ("openai", "custom"):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL") if p == "custom" else None
    elif p == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif p == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif p == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif p == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model_cfg,
        judge_model=model_cfg,
    )
