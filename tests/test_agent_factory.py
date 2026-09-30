"""FinAjan ajan fabrikası (agent_factory) testleri."""

import pandas as pd
import pytest

from src.agent import FinanceAgent
from src.agent_factory import PROVIDER_PRESETS, build_agent, default_config_from_env
from src.demo_llm import DemoLLM
from src.llm_client import LLMClient


@pytest.fixture
def empty_df():
    return pd.DataFrame()


def test_build_agent_demo_mode(empty_df):
    """Demo modunda DemoLLM kullanan FinanceAgent üretildiğini doğrular."""
    agent = build_agent(mode="demo", df=empty_df)
    assert isinstance(agent, FinanceAgent)
    assert isinstance(agent.llm, DemoLLM)
    assert agent.llm.is_available()


def test_build_agent_llm_mode(empty_df):
    """LLM modunda LLMClient kullanan FinanceAgent üretildiğini doğrular."""
    agent = build_agent(
        mode="llm",
        df=empty_df,
        api_key="dummy_key",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
    )
    assert isinstance(agent, FinanceAgent)
    assert isinstance(agent.llm, LLMClient)
    assert agent.llm.api_key == "dummy_key"
    assert agent.llm.model == "openai/gpt-oss-120b"
    assert agent.llm.is_available()


def test_build_agent_invalid_mode(empty_df):
    """Geçersiz mod parametresi verildiğinde ValueError fırlatıldığını doğrular."""
    with pytest.raises(ValueError):
        build_agent(mode="unknown_mode", df=empty_df)


def test_provider_presets_structure():
    """Sağlayıcı ön ayarlarının gerekli alanları taşıdığını doğrular."""
    assert "Groq" in PROVIDER_PRESETS
    assert "Gemini (Google)" in PROVIDER_PRESETS
    assert "OpenRouter" in PROVIDER_PRESETS
    assert "Özel Sağlayıcı" in PROVIDER_PRESETS

    for name, p in PROVIDER_PRESETS.items():
        assert "base_url" in p
        assert "default_model" in p
        assert "description" in p


def test_default_config_from_env_does_not_reveal_key(monkeypatch):
    """default_config_from_env fonksiyonunun anahtar değerini döndürmediğini test eder."""
    monkeypatch.setenv("LLM_API_KEY", "super_secret_key_123")
    monkeypatch.setenv("LLM_MODEL", "my-custom-model")

    cfg = default_config_from_env()
    assert cfg["configured"] is True
    assert cfg["has_key"] is True
    assert cfg["model"] == "my-custom-model"
    # Dönen sözlükte anahtar değeri bulunmamalıdır
    assert "super_secret_key_123" not in str(cfg)
    assert "api_key" not in cfg
