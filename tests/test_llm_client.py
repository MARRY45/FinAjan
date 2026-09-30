"""FinAjan OpenAI Uyumlu LLM Client birim testleri (sahte istemci ve mock ile)."""

import json
from unittest.mock import MagicMock
import openai
import pytest

from src.llm_client import (
    LLMClient,
    LLMConfigError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)


def test_missing_api_key_raises_config_error(monkeypatch):
    """API anahtarı bulunmadığında LLMConfigError fırlatıldığını doğrular."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = LLMClient(api_key=None, load_env=False)
    assert not client.is_available()

    with pytest.raises(LLMConfigError) as exc_info:
        client.create(messages=[{"role": "user", "content": "Merhaba"}])
    assert "tanımlı değil" in str(exc_info.value).lower()


def test_missing_model_raises_config_error(monkeypatch):
    """Model adı tanımlı olmadığında LLMConfigError fırlatıldığını doğrular."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(LLMConfigError) as exc_info:
        LLMClient(api_key="dummy_key", model="", load_env=False)
    assert "model" in str(exc_info.value).lower()


def test_model_name_from_env(monkeypatch):
    """Model adının ortam değişkeninden başarıyla okunduğunu doğrular."""
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    client = LLMClient(api_key="dummy_key", load_env=False)
    assert client.model == "llama-3.3-70b-versatile"


def test_base_url_default_and_custom(monkeypatch):
    """Varsayılan ve özel base_url değerlerinin doğruluğunu test eder."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    client_default = LLMClient(api_key="dummy_key", load_env=False)
    assert client_default.base_url == "https://api.groq.com/openai/v1"

    client_custom = LLMClient(
        api_key="dummy_key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        load_env=False,
    )
    assert client_custom.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_text_response_parsing():
    """Metin yanıtının, usage verisinin ve finish_reason alanının doğruluğunu test eder."""
    client = LLMClient(api_key="dummy_key", load_env=False)

    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "Toplam gideriniz 42.000 TL."
    mock_choice.message.tool_calls = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 25
    mock_response.usage.completion_tokens = 15
    mock_response.usage.total_tokens = 40

    client._client = MagicMock()
    client._client.chat.completions.create.return_value = mock_response

    resp = client.create(messages=[{"role": "user", "content": "Özet"}], system="Sistem promptu")

    assert resp.text == "Toplam gideriniz 42.000 TL."
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40}
    assert resp.message == {"role": "assistant", "content": "Toplam gideriniz 42.000 TL."}

    # System mesajının messages listesinin en başına eklendiğini doğrula
    call_args = client._client.chat.completions.create.call_args[1]
    assert call_args["messages"][0] == {"role": "system", "content": "Sistem promptu"}
    assert call_args["messages"][1] == {"role": "user", "content": "Özet"}


def test_tool_calls_parsing():
    """Tool çağrılarının id, function.name ve JSON argümanlarına doğru ayrıştırıldığını test eder."""
    client = LLMClient(api_key="dummy_key", load_env=False)

    mock_tc = MagicMock()
    mock_tc.id = "call_abc123"
    mock_tc.function.name = "get_summary"
    mock_tc.function.arguments = json.dumps({"start_date": "2026-01-01"})

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 30
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 40

    client._client = MagicMock()
    client._client.chat.completions.create.return_value = mock_response

    resp = client.create(messages=[{"role": "user", "content": "Özet ver"}])

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0] == {
        "id": "call_abc123",
        "name": "get_summary",
        "arguments": {"start_date": "2026-01-01"},
    }
    assert resp.finish_reason == "tool_calls"
    assert "tool_calls" in resp.message
    assert resp.message["tool_calls"][0]["id"] == "call_abc123"


def test_tool_calls_json_parse_error():
    """Bozuk JSON argümanı geldiğinde _parse_error bayrağı eklendiğini ve çökmediğini test eder."""
    client = LLMClient(api_key="dummy_key", load_env=False)

    mock_tc = MagicMock()
    mock_tc.id = "call_broken"
    mock_tc.function.name = "get_summary"
    mock_tc.function.arguments = "{start_date: invalid json...}"

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    client._client = MagicMock()
    client._client.chat.completions.create.return_value = mock_response

    resp = client.create(messages=[{"role": "user", "content": "Özet ver"}])

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["arguments"].get("_parse_error") is True
    assert resp.tool_calls[0]["arguments"].get("raw") == "{start_date: invalid json...}"


def test_finish_reason_length_raises_response_error():
    """Yanıt uzunluk sınırında kesildiğinde LLMResponseError fırlatıldığını test eder."""
    client = LLMClient(api_key="dummy_key", load_env=False)

    mock_choice = MagicMock()
    mock_choice.finish_reason = "length"
    mock_choice.message.content = "Yarım kalan met..."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    client._client = MagicMock()
    client._client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMResponseError) as exc_info:
        client.create(messages=[{"role": "user", "content": "Test"}])
    assert "uzunluk sınırında kesildi" in str(exc_info.value).lower()


def test_reasoning_effort_passed_via_extra_body(monkeypatch):
    """LLM_REASONING_EFFORT tanımlıysa extra_body ile iletildiğini, boşsa iletilmediğini test eder."""
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    client_with_effort = LLMClient(api_key="dummy_key", load_env=False)
    client_with_effort._client = MagicMock()

    mock_choice = MagicMock(finish_reason="stop", message=MagicMock(content="Cevap", tool_calls=None))
    client_with_effort._client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    client_with_effort.create(messages=[{"role": "user", "content": "Test"}])
    call_kwargs = client_with_effort._client.chat.completions.create.call_args[1]
    assert call_kwargs.get("extra_body") == {"reasoning_effort": "medium"}

    # Boş durum testi
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    client_without_effort = LLMClient(api_key="dummy_key", load_env=False)
    client_without_effort._client = MagicMock()
    client_without_effort._client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    client_without_effort.create(messages=[{"role": "user", "content": "Test"}])
    call_kwargs2 = client_without_effort._client.chat.completions.create.call_args[1]
    assert "extra_body" not in call_kwargs2


def test_assistant_message_sanitization():
    """Dönen yanıtta reasoning veya reasoning_content olsa dahi response.message'a dahil edilmediğini test eder."""
    client = LLMClient(api_key="dummy_key", load_env=False)

    mock_msg = MagicMock()
    mock_msg.content = "Temiz cevap metni"
    mock_msg.tool_calls = None
    mock_msg.reasoning = "Dahili düşünce süreci..."
    mock_msg.reasoning_content = "Gizli reasoning içeriği..."

    mock_choice = MagicMock(finish_reason="stop", message=mock_msg)
    mock_response = MagicMock(choices=[mock_choice])

    client._client = MagicMock()
    client._client.chat.completions.create.return_value = mock_response

    resp = client.create(messages=[{"role": "user", "content": "Test"}])

    assert "reasoning" not in resp.message
    assert "reasoning_content" not in resp.message
    assert resp.message == {"role": "assistant", "content": "Temiz cevap metni"}


def test_error_transformations():
    """OpenAI hata sınıflarının FinAjan hata sınıflarına doğru dönüştürüldüğünü test eder."""
    client = LLMClient(api_key="dummy_key", load_env=False)
    client._client = MagicMock()

    # 1. APIConnectionError -> LLMConnectionError
    client._client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())
    with pytest.raises(LLMConnectionError) as exc_conn:
        client.create(messages=[{"role": "user", "content": "Test"}])
    assert "bağlanılamadı" in str(exc_conn.value).lower()

    # 2. RateLimitError -> LLMRateLimitError
    dummy_resp = MagicMock(status_code=429)
    client._client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded", response=dummy_resp, body=None
    )
    with pytest.raises(LLMRateLimitError) as exc_rate:
        client.create(messages=[{"role": "user", "content": "Test"}])
    assert "ücretsiz kullanım sınırına ulaşıldı" in str(exc_rate.value).lower()

    # 3. APIStatusError (400 vb.) -> LLMResponseError
    client._client.chat.completions.create.side_effect = openai.APIStatusError(
        message="Bad Request", response=MagicMock(status_code=400), body=None
    )
    with pytest.raises(LLMResponseError) as exc_status:
        client.create(messages=[{"role": "user", "content": "Test"}])
    assert "hata oluştu" in str(exc_status.value).lower()
