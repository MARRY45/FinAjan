"""FinAjan DemoLLM uçtan uca (FinanceAgent + DemoLLM + gerçek veri) testleri."""

import pytest
from data.generate_data import generate
from src.agent import FinanceAgent
from src.demo_llm import DemoLLM


@pytest.fixture(scope="module")
def dataset():
    df, gt = generate(seed=42)
    return df, gt


def test_demo_overall_summary_intent(dataset):
    """Niyet 1: Toplam gelir/gider/bakiye sorusu."""
    df, gt = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Toplam ne kadar harcama yaptım?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    assert res.steps[0]["tool"] == "get_summary"
    assert not res.steps[0]["hata_mi"]
    # 573.395,68 TL beklenen toplam gider
    assert "573.395,68" in res.answer
    assert res.unverified_numbers == []


def test_demo_monthly_summary_intent(dataset):
    """Niyet 2: Bu ay / güncel ay harcama durumu."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Bu ayki harcama durumum nedir?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    assert res.steps[0]["tool"] == "get_summary"
    assert "2026-09" in res.answer
    assert res.unverified_numbers == []


def test_demo_subscriptions_intent(dataset):
    """Niyet 3: Abonelikler ve düzenli ödemeler."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Hangi aboneliklerim var?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    assert res.steps[0]["tool"] == "get_subscriptions"
    assert "Netflix" in res.answer or "Spotify" in res.answer or "Ahmet" in res.answer
    assert res.unverified_numbers == []


def test_demo_suspicious_intent(dataset):
    """Niyet 4: Şüpheli, anomali ve mükerrer işlemler."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Şüpheli veya mükerrer işlem var mı?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    tool_names = [s["tool"] for s in res.steps]
    assert "find_duplicates" in tool_names or "detect_large_transactions" in tool_names
    assert res.unverified_numbers == []


def test_demo_category_distribution_intent(dataset):
    """Niyet 5: En çok hangi kategoride harcama yapıldığı."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("En çok hangi kategoride harcama yaptım?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    assert res.steps[0]["tool"] == "get_category_summary"
    assert "Kira" in res.answer or "Teknoloji" in res.answer or "Market" in res.answer
    assert res.unverified_numbers == []


def test_demo_unrecognized_question_does_not_crash(dataset):
    """Tanımlı senaryolar dışındaki soruda ajanın çökmediğini ve öneri sunduğunu test eder."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Yarın hava durumu nasıl olacak?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert res.steps == []
    assert "örnek soruları" in res.answer
    assert res.unverified_numbers == []


def test_demo_savings_intent(dataset):
    """Niyet 6: Tasarruf ve bütçe optimizasyonu fırsatları."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Nereden tasarruf edebilirim?")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    assert res.steps[0]["tool"] == "get_savings_opportunities"
    assert "Tasarruf Fırsatları" in res.answer
    assert "Abonelikler" in res.answer
    assert res.unverified_numbers == []


def test_demo_categorization_intent(dataset):
    """Niyet 7: Bilinmeyen işlemleri kategorilendirme önerileri."""
    df, _ = dataset
    agent = FinanceAgent(llm=DemoLLM(), df=df)

    res = agent.ask("Bilinmeyen işlemleri kategorilendir")
    assert res.error is None
    assert "🧪 Demo modu (LLM yok)" in res.answer
    assert len(res.steps) >= 1
    assert res.steps[0]["tool"] == "categorize_unknown_transactions"
    assert "Kategori Önerileri" in res.answer
    assert res.unverified_numbers == []

