"""FinAjan ajan fabrikası (agent factory) modülü.

Mod seçimine (Demo vs LLM) göre uygun istemci ve FinanceAgent örneğini kurar.
Sağlayıcı ön ayarlarını (presets) ve ortam değişkeni kontrollerini yönetir.
"""

import os
from typing import Any, Literal
import pandas as pd
from dotenv import load_dotenv

from src.agent import FinanceAgent
from src.demo_llm import DemoLLM
from src.llm_client import LLMClient

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
        "description": "Groq Cloud (Hızlı, OpenAI uyumlu)",
    },
    "Gemini (Google)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "description": "Google Gemini OpenAI Uyumlu Uç Noktası",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "description": "OpenRouter API Ağ Geçidi",
    },
    "Özel Sağlayıcı": {
        "base_url": "",
        "default_model": "",
        "description": "Özel veya yerel (LocalAI, Ollama vb.) OpenAI uyumlu API",
    },
}


def default_config_from_env() -> dict[str, Any]:
    """Ortam değişkenlerinde veya .env dosyasında tanımlı hazır ayarları tespit eder.

    Güvenlik: API anahtarının değerini ASLA arayüze sızdırmaz; yalnızca 'has_key'
    boolean bayrağı olarak varlığını bildirir.

    Returns:
        dict: Yapılandırma durumu, base_url, model ve anahtar varlık bilgisi.
    """
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()

    has_key = bool(api_key)
    has_model = bool(model)
    configured = has_key and has_model

    return {
        "configured": configured,
        "has_key": has_key,
        "base_url": base_url,
        "model": model or "openai/gpt-oss-120b",
    }


def build_agent(
    mode: Literal["demo", "llm"],
    df: pd.DataFrame,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> FinanceAgent:
    """Belirtilen moda göre FinanceAgent örneği inşa eder.

    Args:
        mode: 'demo' (çevrimdışı kurallı) veya 'llm' (gerçek model).
        df: Finansal işlem verisi DataFrame'i.
        api_key: LLM modu için API anahtarı.
        base_url: LLM modu için API adresi.
        model: LLM modu için model adı.

    Returns:
        FinanceAgent: Kullanıma hazır ajan örneği.
    """
    if mode == "demo":
        llm = DemoLLM()
    elif mode == "llm":
        # load_env=False ile dışarıdan parametreyle gelen anahtar kullanılır
        # Parametre verilmemişse env'den okunur
        llm = LLMClient(
            api_key=api_key or os.getenv("LLM_API_KEY"),
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            model=model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            load_env=False,
        )
    else:
        raise ValueError(f"Geçersiz ajan modu: '{mode}'. 'demo' veya 'llm' olmalıdır.")

    return FinanceAgent(llm=llm, df=df)
