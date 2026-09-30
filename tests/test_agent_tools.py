"""FinAjan ajan araçları (agent_tools) katmanı testleri."""

import json
import pandas as pd
import pytest

from data.generate_data import generate
from src.agent_tools import (
    DISPATCHER,
    TOOL_DEFINITIONS,
    execute_tool,
)


@pytest.fixture(scope="module")
def full_dataset():
    """Tüm sentetik veri setini ve ground truth sözlüğünü yükler/üretir."""
    return generate(seed=42)


def test_tool_definitions_validity():
    """Tüm araç tanımlarının OpenAI function şema standartlarına uygunluğunu ve 11 aracın varlığını test eder."""
    expected_tools = {
        "get_dataset_info",
        "get_summary",
        "get_monthly_summary",
        "get_category_summary",
        "get_category_trend",
        "list_transactions",
        "find_duplicates",
        "detect_large_transactions",
        "get_subscriptions",
        "categorize_unknown_transactions",
        "get_savings_opportunities",
    }
    assert len(TOOL_DEFINITIONS) == 11
    found_names = set()

    for item in TOOL_DEFINITIONS:
        assert item.get("type") == "function"
        fn = item.get("function")
        assert isinstance(fn, dict)
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert params.get("type") == "object"
        assert "properties" in params
        found_names.add(fn["name"])

    assert found_names == expected_tools


def test_all_tools_return_json_serializable(full_dataset):
    """Her bir aracın çıktısının json.dumps ile başarıyla serileştirilebildiğini doğrular."""
    df, _ = full_dataset

    tool_calls_to_test = [
        ("get_dataset_info", {}),
        ("get_summary", {"start_date": "2026-01-01", "end_date": "2026-03-31"}),
        ("get_monthly_summary", {"start_month": "2026-01", "end_month": "2026-03"}),
        ("get_category_summary", {"start_date": "2026-01-01"}),
        ("get_category_trend", {"category": "Market", "months": 3}),
        ("list_transactions", {"category": "Market", "limit": 5}),
        ("find_duplicates", {}),
        ("detect_large_transactions", {}),
        ("get_subscriptions", {}),
        ("categorize_unknown_transactions", {"limit": 10}),
        ("get_savings_opportunities", {}),
    ]

    for name, args in tool_calls_to_test:
        res = execute_tool(name, args, df)
        assert isinstance(res, dict)
        assert "error" not in res, f"Araç '{name}' hata döndürdü: {res.get('error')}"

        # json.dumps hata fırlatmamalı
        serialized = json.dumps(res, ensure_ascii=False)
        assert isinstance(serialized, str)
        assert len(serialized) > 0


def test_invalid_category_returns_error_dict(full_dataset):
    """Geçersiz kategori parametresi verildiğinde istisna fırlatmayıp error sözlüğü döndürdüğünü test eder."""
    df, _ = full_dataset
    res = execute_tool("get_category_trend", {"category": "UzaySeyahati"}, df)

    assert "error" in res
    assert "gecerli_degerler" in res
    assert "Market" in res["gecerli_degerler"]


def test_invalid_date_returns_error_dict(full_dataset):
    """Geçersiz tarih parametresi verildiğinde istisna fırlatmayıp error sözlüğü döndürdüğünü test eder."""
    df, _ = full_dataset
    res = execute_tool("get_summary", {"start_date": "01-01-2026"}, df)

    assert "error" in res
    assert "geçersiz tarih biçimi" in res["error"].lower()


def test_list_transactions_limit_cap(full_dataset):
    """list_transactions aracının limit argümanının azami 50 ile sınırlandırıldığını test eder."""
    df, _ = full_dataset
    res = execute_tool("list_transactions", {"limit": 100}, df)

    assert "islemler" in res
    assert len(res["islemler"]) <= 50


def test_unknown_tool_returns_error_dict(full_dataset):
    """Tanımlı olmayan bir araç çağrıldığında dispatcher'ın hata sözlüğü döndürdüğünü test eder."""
    df, _ = full_dataset
    res = execute_tool("olmayan_arac", {}, df)

    assert "error" in res
    assert "bilinmeyen araç adı" in res["error"].lower()
    assert "gecerli_araclar" in res


def test_detect_large_transactions_has_seviye(full_dataset):
    """detect_large_transactions çıktısında 'seviye' alanı olduğunu ve 3 kasıtlı anomalinin 'yuksek' çıktığını doğrular."""
    df, _ = full_dataset
    res = execute_tool("detect_large_transactions", {}, df)

    assert "anomaliler" in res
    anomalies = res["anomaliler"]

    # Her kayıtta seviye olmalı
    for a in anomalies:
        assert a["seviye"] in ("yuksek", "dusuk")

    # Üretilen veride enjekte edilen 3 dev anomalinin seviyesi 'yuksek' olmalı (skor >= 10.0)
    yuksek_anomaliler = [a for a in anomalies if a["seviye"] == "yuksek"]
    assert len(yuksek_anomaliler) == 3


def test_get_category_trend_returns_requested_months(full_dataset):
    """get_category_trend fonksiyonunun talep edilen ay sayısı kadar (3 ay) trend döndürdüğünü doğrular."""
    df, _ = full_dataset
    res = execute_tool("get_category_trend", {"category": "Restoran", "months": 3}, df)

    assert res["kategori"] == "Restoran"
    assert res["ay_sayisi"] == 3
    assert len(res["aylik_trend"]) == 3
