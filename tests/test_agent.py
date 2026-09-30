"""FinAjan FinanceAgent orkestrasyon ve tool-use testleri (senaryolu sahte LLM ile)."""

import json
from typing import Any
from unittest.mock import MagicMock
import pandas as pd
import pytest

from data.generate_data import generate
from src.agent import FinanceAgent
from src.llm_client import (
    LLMClient,
    LLMConnectionError,
    LLMResponse,
)


class MockLLMClient:
    """Senaryo tabanlı testler için sahte LLM istemcisi."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.call_history: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def create(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: Any = None,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        self.call_history.append({
            "messages": [dict(m) for m in messages],
            "system": system,
            "tools": tools,
        })
        if not self.responses:
            raise RuntimeError("MockLLMClient için tanımlı yanıt kalmadı.")
        return self.responses.pop(0)


def make_tool_response(tool_calls: list[dict[str, Any]], text: str = "") -> LLMResponse:
    """Yardımcı: Tool çağrısı içeren LLMResponse üretir."""
    sanitized_tcs = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                if isinstance(tc["arguments"], dict)
                else str(tc["arguments"]),
            },
        }
        for tc in tool_calls
    ]
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        message={"role": "assistant", "content": text, "tool_calls": sanitized_tcs},
    )


def make_text_response(text: str) -> LLMResponse:
    """Yardımcı: Düz metin yanıtı içeren LLMResponse üretir."""
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        message={"role": "assistant", "content": text},
    )


@pytest.fixture(scope="module")
def full_dataset():
    """Tüm sentetik veri setini ve ground truth sözlüğünü yükler/üretir."""
    return generate(seed=42)


def test_plain_text_response_without_tools(full_dataset):
    """Herhangi bir araç çağırmadan doğrudan metin döndürme senaryosunu test eder."""
    df, _ = full_dataset

    resp = make_text_response("Merhaba! Size nasıl yardımcı olabilirim?")
    mock_llm = MockLLMClient([resp])
    agent = FinanceAgent(llm=mock_llm, df=df)

    result = agent.ask("Selam")

    assert result.error is None
    assert result.answer == "Merhaba! Size nasıl yardımcı olabilirim?"
    assert result.steps == []
    assert len(mock_llm.call_history) == 1


def test_single_tool_call_flow(full_dataset):
    """Tek bir araç çağrısı ve ardından yanıt akışını doğrular."""
    df, _ = full_dataset

    # 1. Yanıt: Tool çağrısı
    resp1 = make_tool_response([{"id": "call_1", "name": "get_summary", "arguments": {}}])
    # 2. Yanıt: Nihai cevap (gerçek veri seti toplam gideri 573.395,68 TL)
    resp2 = make_text_response("Toplam gideriniz 573.395,68 TL'dir.")

    mock_llm = MockLLMClient([resp1, resp2])
    agent = FinanceAgent(llm=mock_llm, df=df)

    result = agent.ask("Genel harcama özetim nedir?")

    assert result.error is None
    assert "573.395,68" in result.answer
    assert len(result.steps) == 1
    assert result.steps[0]["tool"] == "get_summary"
    assert not result.steps[0]["hata_mi"]

    # OpenAI protokolü kontrolü: 2. çağrıda messages listesinde assistant ve tool rolleri yer almalı
    second_call_msgs = mock_llm.call_history[1]["messages"]
    assistant_msg = second_call_msgs[-2]
    tool_msg = second_call_msgs[-1]

    assert assistant_msg["role"] == "assistant"
    assert "tool_calls" in assistant_msg
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "toplam_gider" in tool_msg["content"]


def test_multi_step_tool_flow(full_dataset):
    """Ardışık çok adımlı araç çağrı akışını doğrular."""
    df, _ = full_dataset

    # 1. Adım: get_category_summary
    resp1 = make_tool_response([{"id": "c1", "name": "get_category_summary", "arguments": {}}])
    # 2. Adım: list_transactions
    resp2 = make_tool_response([{"id": "c2", "name": "list_transactions", "arguments": {"category": "Market", "limit": 3}}])
    # 3. Adım: Cevap
    resp3 = make_text_response("En çok harcama Market kategorisinde yapılmıştır.")

    mock_llm = MockLLMClient([resp1, resp2, resp3])
    agent = FinanceAgent(llm=mock_llm, df=df)

    result = agent.ask("En çok nereye harcadım ve son market işlemlerim neler?")

    assert len(result.steps) == 2
    assert result.steps[0]["tool"] == "get_category_summary"
    assert result.steps[1]["tool"] == "list_transactions"
    assert "Market" in result.answer


def test_parallel_tool_calls_separate_tool_messages(full_dataset):
    """Tek bir yanıtta modelin 2 aracı paralel çağırmasını ve her birinin AYRI 'tool' mesajı olarak iletildiğini test eder."""
    df, _ = full_dataset

    resp1 = make_tool_response([
        {"id": "p1", "name": "get_summary", "arguments": {}},
        {"id": "p2", "name": "get_subscriptions", "arguments": {}},
    ])
    resp2 = make_text_response("Özet ve abonelikler incelendi.")

    mock_llm = MockLLMClient([resp1, resp2])
    agent = FinanceAgent(llm=mock_llm, df=df)

    result = agent.ask("Genel durum ve aboneliklerimi söyle.")

    assert len(result.steps) == 2
    # 2. çağrıda LLM'e iletilen son mesajları incele
    second_call_messages = mock_llm.call_history[1]["messages"]

    # Son 3 mesaj: assistant (with tool_calls), tool p1, tool p2
    assistant_msg = second_call_messages[-3]
    tool_msg_1 = second_call_messages[-2]
    tool_msg_2 = second_call_messages[-1]

    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 2

    assert tool_msg_1["role"] == "tool"
    assert tool_msg_1["tool_call_id"] == "p1"

    assert tool_msg_2["role"] == "tool"
    assert tool_msg_2["tool_call_id"] == "p2"


def test_tool_json_parse_error_handling(full_dataset):
    """Araç argümanında _parse_error olduğunda ajanın hata döndürüp döngüye devam ettiğini test eder."""
    df, _ = full_dataset

    resp1 = make_tool_response([
        {"id": "err_p", "name": "get_summary", "arguments": {"_parse_error": True, "raw": "{bad json"}}
    ])
    resp2 = make_text_response("JSON ayrıştırılamadı, işlem yapılamadı.")

    mock_llm = MockLLMClient([resp1, resp2])
    agent = FinanceAgent(llm=mock_llm, df=df)

    result = agent.ask("Özet")

    assert len(result.steps) == 1
    assert result.steps[0]["hata_mi"] is True
    assert "JSON formatında çözülemedi" in result.steps[0]["output_ozeti"]


def test_max_steps_limit(full_dataset):
    """Model sürekli araç çağırırsa max_steps sınırında durulduğunu ve nazik mesaj dönüldüğünü test eder."""
    df, _ = full_dataset

    infinite_tool_resps = [
        make_tool_response([{"id": f"call_{i}", "name": "get_dataset_info", "arguments": {}}])
        for i in range(10)
    ]

    mock_llm = MockLLMClient(infinite_tool_resps)
    agent = FinanceAgent(llm=mock_llm, df=df, max_steps=3)

    result = agent.ask("Sonsuz döngü sorusu")

    assert len(result.steps) == 3
    assert "çok fazla adım" in result.answer.lower()


def test_hallucination_guard_triggers_retry_on_invented_number(full_dataset):
    """Model uydurma bir sayı kullandığında guard'ın tetiklenip 1 kez retry yaptığını test eder."""
    df, _ = full_dataset

    # 1. Adım: get_summary çağrısı
    resp1 = make_tool_response([{"id": "s1", "name": "get_summary", "arguments": {}}])

    # 2. Adım: Model uydurma bir sayı (999.888,00 TL) yazıyor
    resp2 = make_text_response("Toplam gideriniz 999.888,00 TL olarak gerçekleşti.")

    # 3. Adım: Guard uyarısından sonra model doğru sayıyı (573.395,68 TL) yazıyor
    resp3 = make_text_response("Düzeltiyorum, toplam gideriniz 573.395,68 TL.")

    mock_llm = MockLLMClient([resp1, resp2, resp3])
    agent = FinanceAgent(llm=mock_llm, df=df)

    result = agent.ask("Toplam giderim ne kadar?")

    # Toplam 3 model çağrısı yapılmış olmalı (tool_use -> uydurma stop -> guard uyarısı -> doğru stop)
    assert len(mock_llm.call_history) == 3
    # 3. çağrıda guard retry uyarısı user mesajı olarak gönderilmiş olmalı
    last_user_msg = mock_llm.call_history[2]["messages"][-1]["content"]
    assert "doğrulanamadı" in last_user_msg.lower()
    # Nihai cevapta doğru sayı yer almalı ve unverified_numbers boş olmalı
    assert "573.395,68" in result.answer
    assert result.unverified_numbers == []


def test_multi_turn_memory_keeps_only_plain_text(full_dataset):
    """Turlar arası hafızada ara tool_calls mesajlarının tutulmadığını,
    yalnızca düz metin (soru, cevap) çiftlerinin korunduğunu test eder.
    """
    df, _ = full_dataset

    # 1. Tur: Tool use içeren akış
    resp1_1 = make_tool_response([{"id": "m1", "name": "get_dataset_info", "arguments": {}}])
    resp1_2 = make_text_response("Veri seti 760 işlem içermektedir.")

    # 2. Tur: Doğrudan yanıt
    resp2_1 = make_text_response("İkinci soru cevabı.")

    mock_llm = MockLLMClient([resp1_1, resp1_2, resp2_1])
    agent = FinanceAgent(llm=mock_llm, df=df)

    # 1. Soru
    agent.ask("1. Soru: Kaç işlem var?")
    assert len(agent.history) == 2

    # 2. Soru
    agent.ask("2. Soru: Teşekkürler.")

    # 2. sorunun LLM çağrısındaki mesaj geçmişini kontrol et
    turn2_initial_messages = mock_llm.call_history[2]["messages"]
    # Sırasıyla: user (1. soru), assistant (1. cevap), user (2. soru)
    assert len(turn2_initial_messages) == 3
    assert turn2_initial_messages[0] == {"role": "user", "content": "1. Soru: Kaç işlem var?"}
    assert turn2_initial_messages[1] == {"role": "assistant", "content": "Veri seti 760 işlem içermektedir."}
    assert turn2_initial_messages[2] == {"role": "user", "content": "2. Soru: Teşekkürler."}


def test_llm_connection_error_gracefully_handled(full_dataset):
    """LLM bağlantı hatası verdiğinde ajan uygulamasının çökmediğini ve AgentResult.error döndürdüğünü test eder."""
    df, _ = full_dataset

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.create.side_effect = LLMConnectionError("Bağlantı koptu.")

    agent = FinanceAgent(llm=mock_llm, df=df)
    result = agent.ask("Bakiye nedir?")

    assert result.error is not None
    assert "Bağlantı koptu" in result.error
    assert "sorun oluştu" in result.answer.lower()
