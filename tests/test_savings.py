"""FinAjan tasarruf fırsatları (savings) birim testleri."""

import pandas as pd
import pytest

from data.generate_data import generate
from src.savings import find_savings_opportunities, get_savings_opportunities


@pytest.fixture(scope="module")
def dataset():
    df, gt = generate(seed=42)
    return df, gt


def test_find_savings_opportunities_on_synthetic_data(dataset):
    """Sentetik veri setinde tasarruf fırsatları analizinin doğruluğunu test eder."""
    df, gt = dataset
    res = find_savings_opportunities(df)

    assert isinstance(res, dict)
    assert "abonelikler" in res
    assert "yinelenen_cekimler" in res
    assert "yuksek_sapma" in res
    # Asla yapay bir toplam potansiyel_tasarruf rakamı üretilmemelidir
    assert "potansiyel_tasarruf" not in res

    # 1. Abonelikler kontrolü: yıllık = aylık x 12
    subs = res["abonelikler"]
    assert subs["toplam_aylik_maliyet"] > 0
    assert abs(subs["toplam_yillik_maliyet"] - (subs["toplam_aylik_maliyet"] * 12.0)) < 0.05
    for item in subs["kalemler"]:
        assert item["yillik_tutar"] == round(item["aylik_tutar"] * 12.0, 2)
        assert "değerlendirebilirsiniz" in item["aciklama"].lower()

    # 2. Yinelenen çekimler kontrolü: 2 grup, toplam fazla çekim 527.50 TL
    # 342.50 (BİM) + 185.00 (Starbucks) = 527.50 TL
    dups = res["yinelenen_cekimler"]
    assert len(dups["kalemler"]) == 2
    assert dups["toplam_fazla_cekim_tutari"] == 527.50
    for item in dups["kalemler"]:
        assert item["fazla_adet"] >= 1
        assert "iade talep" in item["aciklama"].lower()
        assert "değerlendirebilirsiniz" in item["aciklama"].lower()

    # 3. Yüksek sapma kontrolü
    sapma = res["yuksek_sapma"]
    assert isinstance(sapma["kalemler"], list)
    for s_item in sapma["kalemler"]:
        assert s_item["artis_orani"] >= 0.30
        assert s_item["fark_tutari"] > 0
        assert "değerlendirebilirsiniz" in s_item["aciklama"].lower()


def test_find_savings_opportunities_empty():
    """Boş DataFrame ile fonksiyonun çökmeden güvenli varsayılanlar döndürdüğünü test eder."""
    res = find_savings_opportunities(pd.DataFrame())
    assert res["abonelikler"]["toplam_aylik_maliyet"] == 0.0
    assert res["yinelenen_cekimler"]["toplam_fazla_cekim_tutari"] == 0.0
    assert res["yuksek_sapma"]["kalemler"] == []


def test_get_savings_opportunities_tool_wrapper(dataset):
    """get_savings_opportunities araç sarmalayıcısının JSON uyumlu sözlük döndürdüğünü test eder."""
    df, _ = dataset
    res = get_savings_opportunities(df)
    assert isinstance(res, dict)
    assert "ozet" in res
