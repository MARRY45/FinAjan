"""FinAjan OpenAI uyumlu LLM canlı API entegrasyon testi.

Bu test varsayılan olarak API anahtarı yoksa ATLANIR (SKIP).
Yalnızca geçerli bir LLM_API_KEY ve LLM_MODEL yapılandırıldığında çalıştırılır.
"""

import pytest

from src.agent import FinanceAgent
from src.llm_client import LLMClient
from src.tools import load_transactions


def _is_llm_available() -> bool:
    try:
        return LLMClient().is_available()
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _is_llm_available(),
    reason="LLM API anahtarı veya modeli tanımlı değil, canlı test atlandı",
)
def test_live_llm_integration():
    """Canlı LLM kabul testi: 4 temel finansal soru sırayla sınanır."""
    llm = LLMClient()
    df = load_transactions("data/transactions.csv")
    agent = FinanceAgent(llm=llm, df=df)

    # Soru 1: Toplam harcama
    r1 = agent.ask("Toplam ne kadar harcama yaptım?")
    assert r1.error is None
    assert len(r1.steps) >= 1
    assert "TL" in r1.answer or "573" in r1.answer

    # Soru 2: En çok harcama yapılan kategori
    r2 = agent.ask("En çok hangi kategoride harcama yaptım?")
    assert r2.error is None
    assert len(r2.steps) >= 1
    assert "Kira" in r2.answer or "Market" in r2.answer

    # Soru 3: Şüpheli veya mükerrer işlem var mı?
    r3 = agent.ask("Şüpheli veya mükerrer işlem var mı?")
    assert r3.error is None
    assert len(r3.steps) >= 1

    # Soru 4: Hangi aboneliklerim var?
    r4 = agent.ask("Hangi aboneliklerim var?")
    assert r4.error is None
    assert len(r4.steps) >= 1
