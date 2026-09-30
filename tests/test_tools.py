"""FinAjan finansal araçlar ve saf fonksiyon testleri."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate
from src.constants import (
    CATEGORY_ABONELIK,
    CATEGORY_BILINMIYOR,
    CATEGORY_GELIR,
    CATEGORY_KIRA,
    CATEGORY_MARKET,
    CATEGORY_RESTORAN,
    COLUMNS,
)
from src.tools import (
    DataLoadError,
    calculate_balance,
    calculate_expenses,
    calculate_income,
    category_monthly_totals,
    detect_large_transactions,
    filter_period,
    find_duplicates,
    find_recurring_transactions,
    get_reference_date,
    group_by_category,
    load_transactions,
    monthly_summary,
    search_transactions,
    summarize_transactions,
)


@pytest.fixture
def sample_df():
    """Elle tanımlanmış, sonuçları bilinen küçük test veri seti."""
    data = [
        {"tarih": "2026-01-01", "tutar": 1000.0, "aciklama": "Maaş", "isyeri": "ABC A.Ş.", "kategori": "Gelir", "hesap": "Vadesiz Hesap"},
        {"tarih": "2026-01-05", "tutar": -200.0, "aciklama": "Market Alışverişi", "isyeri": "Migros", "kategori": "Market", "hesap": "Kredi Kartı"},
        {"tarih": "2026-01-10", "tutar": -300.0, "aciklama": "Kahve", "isyeri": "Starbucks", "kategori": "Restoran", "hesap": "Kredi Kartı"},
        {"tarih": "2026-02-01", "tutar": 1000.0, "aciklama": "Maaş", "isyeri": "ABC A.Ş.", "kategori": "Gelir", "hesap": "Vadesiz Hesap"},
        {"tarih": "2026-02-15", "tutar": -400.0, "aciklama": "Gıda", "isyeri": "Migros", "kategori": "Market", "hesap": "Kredi Kartı"},
    ]
    df = pd.DataFrame(data)[COLUMNS]
    df["tarih"] = pd.to_datetime(df["tarih"])
    return df


@pytest.fixture(scope="module")
def full_dataset():
    """Tüm sentetik veri setini ve ground truth sözlüğünü yükler/üretir."""
    return generate(seed=42)


# --- 1. Elle Yazılmış Küçük DataFrame ile Matematiksel Doğruluk ---

def test_calculate_income_and_expenses(sample_df):
    """Gelir, gider ve bakiye hesaplamalarını doğrular."""
    assert calculate_income(sample_df) == 2000.0
    assert calculate_expenses(sample_df) == 900.0
    assert calculate_balance(sample_df) == 1100.0


def test_summarize_transactions(sample_df):
    """Özet sözlüğünün alanlarını ve değerlerini doğrular."""
    summary = summarize_transactions(sample_df)
    assert summary["satir_sayisi"] == 5
    assert summary["gelir"] == 2000.0
    assert summary["gider"] == 900.0
    assert summary["net"] == 1100.0
    assert summary["bilinmiyor_orani"] == 0.0
    assert "2026-01-01" in summary["tarih_araligi"]


def test_group_by_category(sample_df):
    """Kategori bazında harcama dağılımının ve oranlarının toplamla uyumunu doğrular."""
    cat_df = group_by_category(sample_df, expenses_only=True)
    assert len(cat_df) == 2
    assert set(cat_df["kategori"]) == {"Market", "Restoran"}

    market_row = cat_df[cat_df["kategori"] == "Market"].iloc[0]
    assert market_row["toplam"] == 600.0
    assert market_row["islem_sayisi"] == 2

    restoran_row = cat_df[cat_df["kategori"] == "Restoran"].iloc[0]
    assert restoran_row["toplam"] == 300.0
    assert restoran_row["islem_sayisi"] == 1

    # Oranların toplamı 1.0 olmalı
    assert round(cat_df["oran"].sum(), 2) == 1.0


def test_monthly_summary_totals(sample_df):
    """Aylık özet tablosunun toplamlarının genel toplamlarla eşit olduğunu doğrular."""
    m_df = monthly_summary(sample_df)
    assert len(m_df) == 2
    assert list(m_df["ay"]) == ["2026-01", "2026-02"]

    # Aylık toplamların genel toplama eşitliği
    assert m_df["gelir"].sum() == calculate_income(sample_df)
    assert m_df["gider"].sum() == calculate_expenses(sample_df)
    assert m_df["net"].sum() == calculate_balance(sample_df)
    assert m_df["islem_sayisi"].sum() == len(sample_df)


def test_filter_period(sample_df):
    """Tarih aralığı filtresinin sınır değerleri doğru içerdiğini test eder."""
    filtered = filter_period(sample_df, "2026-01-01", "2026-01-31")
    assert len(filtered) == 3
    assert (filtered["tarih"] <= pd.Timestamp("2026-01-31")).all()


# --- 2. Kenar Durumlar (Boş DF, Hata Sınıfı, MAD=0) ---

def test_empty_dataframe_handling():
    """Boş DataFrame verildiğinde fonksiyonların çökmeden güvenli varsayılan değerler döndürdüğünü test eder."""
    empty_df = pd.DataFrame(columns=COLUMNS)
    assert calculate_income(empty_df) == 0.0
    assert calculate_expenses(empty_df) == 0.0
    assert calculate_balance(empty_df) == 0.0

    summary = summarize_transactions(empty_df)
    assert summary["satir_sayisi"] == 0
    assert summary["tarih_araligi"] is None

    assert group_by_category(empty_df).empty
    assert monthly_summary(empty_df).empty
    assert find_duplicates(empty_df).empty
    assert find_recurring_transactions(empty_df).empty
    assert detect_large_transactions(empty_df).empty


def test_data_load_error_on_missing_file():
    """Mevcut olmayan bir dosya yolunda DataLoadError fırlatıldığını doğrular."""
    with pytest.raises(DataLoadError) as exc_info:
        load_transactions("non_existent_path_xyz123.csv")
    assert "bulunamadı" in str(exc_info.value).lower()


def test_data_load_error_on_invalid_schema():
    """Zorunlu kolonları eksik olan CSV dosyalarında DataLoadError fırlatıldığını doğrular."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".csv") as tmp:
        tmp.write("tarih,tutar,isyeri\n2026-01-01,-100.0,Migros\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(DataLoadError) as exc_info:
            load_transactions(tmp_path)
        assert "zorunlu sütunlar eksik" in str(exc_info.value).lower()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_mad_zero_case():
    """MAD (Median Absolute Deviation) 0 olduğunda sıfıra bölme hatası olmadan IQR yöntemine düştüğünü test eder."""
    # 7 adet -100.0 (medyan = 100, MAD = 0), 3 adet -200.0 (Q75=200, IQR=100 > 0) ve 1 adet -1500.0 büyük harcama
    rows = [
        {"tarih": f"2026-01-{i:02d}", "tutar": -100.0, "aciklama": "Giyim", "isyeri": "Zara", "kategori": "Giyim", "hesap": "Kredi Kartı"}
        for i in range(1, 8)
    ]
    rows += [
        {"tarih": f"2026-01-{i:02d}", "tutar": -200.0, "aciklama": "Giyim", "isyeri": "Zara", "kategori": "Giyim", "hesap": "Kredi Kartı"}
        for i in range(8, 11)
    ]
    rows.append({
        "tarih": "2026-01-20", "tutar": -1500.0, "aciklama": "Zara Mont", "isyeri": "Zara", "kategori": "Giyim", "hesap": "Kredi Kartı"
    })
    test_df = pd.DataFrame(rows)

    # Sıfıra bölme hatası vermemeli ve IQR ile anomalisi tespit edilmeli
    res = detect_large_transactions(test_df, esik=3.5, min_ornek=8)
    assert not res.empty
    assert res.iloc[0]["tutar"] == -1500.0


# --- 3. Üretilen Gerçek Veriyle Doğrulamalar ---

def test_find_duplicates_matches_ground_truth(full_dataset):
    """find_duplicates fonksiyonunun ground truth'taki kasıtlı duplicate kayıtlarını eksiksiz bulduğunu doğrular."""
    df, gt = full_dataset
    dups = find_duplicates(df)

    # Tam 2 olay x 2 kayıt = 4 satır dönmeli
    assert len(dups) == 4

    # Her iki duplicate grubunun ground truth anahtarlarıyla uyuştuğunu kontrol et
    found_keys = {
        f"{row['tarih']}|{row['isyeri']}|{row['tutar']:.2f}|{row['hesap']}"
        for _, row in dups.iterrows()
    }
    gt_keys = {item["row_key"] for item in gt["duplicates"]}
    assert found_keys == gt_keys


def test_detect_large_transactions_catches_all_anomalies(full_dataset):
    """detect_large_transactions fonksiyonunun enjekte edilen 3 anomalinin 3'ünü de yakaladığını doğrular."""
    df, gt = full_dataset
    detected = detect_large_transactions(df, esik=3.5, min_ornek=8)

    detected_keys = {
        f"{row['tarih']}|{row['isyeri']}|{row['tutar']:.2f}"
        for _, row in detected.iterrows()
    }

    # Ground truth'taki 3 anomalinin tamamı yakalanmalı
    for a in gt["anomalies"]:
        key = f"{a['tarih']}|{a['isyeri']}|{a['tutar']:.2f}"
        assert key in detected_keys, f"Anomali tespit edilemedi: {key}"

    # Toplam tespit edilen anomali sayısı <= 3 enjekte + en fazla 5 yanlış alarm
    assert len(detected) <= 8


def test_find_recurring_finds_expected_subscriptions(full_dataset):
    """find_recurring_transactions fonksiyonunun Netflix, Spotify, YouTube Premium, MacFit ve kirayı bulduğunu,
    değişken harcama işyerlerini (market/restoran) bulmadığını doğrular.
    """
    df, _ = full_dataset
    recurring = find_recurring_transactions(df)

    found_merchants = set(recurring["isyeri"])

    # Beklenen 5 düzenli işlem bulunmalı
    expected = {"Netflix", "Spotify", "YouTube Premium", "MacFit", "Ev Sahibi Ahmet Yılmaz"}
    for exp in expected:
        assert exp in found_merchants, f"Tekrarlayan işlem bulunamadı: {exp}"

    # Market veya restoran işyerleri bulunmamalı
    unwanted = {"Migros", "BİM", "A101", "Şok", "Starbucks", "Burger King", "Yemeksepeti", "GetirYemek"}
    for unw in unwanted:
        assert unw not in found_merchants, f"İstenmeyen işyeri tekrarlayan bulundu: {unw}"


def test_group_by_category_expenses_only_excludes_income(full_dataset):
    """group_by_category varsayılanında (expenses_only=True) Gelir kategorisinin olmadığını
    ve oran sütununun toplamının 1.0 olduğunu doğrular.
    """
    df, _ = full_dataset
    cat_df = group_by_category(df, expenses_only=True)

    # Gelir kategorisi kesinlikle olmamalı
    assert CATEGORY_GELIR not in cat_df["kategori"].values

    # Oranların toplamı 1.0 olmalı (yuvarlama toleransı ile)
    assert abs(cat_df["oran"].sum() - 1.0) < 1e-3


def test_search_transactions_filters_and_turkish_case(full_dataset):
    """search_transactions fonksiyonunun Türkçe karakter ve büyük/küçük harf duyarsız arama,
    limit üst sınırı (50) ve azalan tarih sıralamasını doğrular.
    """
    df, _ = full_dataset

    # "BİM" araması: "bim", "BİM", "BIM", "bım" hepsi eşleşmeli
    res1 = search_transactions(df, merchant="bim")
    assert not res1.empty
    assert (res1["isyeri"] == "BİM").all()

    res2 = search_transactions(df, merchant="BİM")
    assert len(res1) == len(res2)

    # Limit üst sınırı 50 kontrolü
    res_large = search_transactions(df, limit=100)
    assert len(res_large) <= 50

    # Tarihe göre azalan sıralı olmalı
    tarihler = pd.to_datetime(res1["tarih"])
    assert tarihler.is_monotonic_decreasing


def test_category_monthly_totals_fills_zero_months(full_dataset):
    """category_monthly_totals fonksiyonunun harcama olmayan ayları 0 ile doldurduğunu ve
    tüm 12 ayı sıralı döndürdüğünü doğrular.
    """
    df, _ = full_dataset

    trend_market = category_monthly_totals(df, "Market")
    assert len(trend_market) == 12
    assert (trend_market["toplam"] > 0).all()

    # Hiç harcama olmayan hayali kategori
    trend_empty = category_monthly_totals(df, "OlmayanKategori")
    assert len(trend_empty) == 12
    assert (trend_empty["toplam"] == 0.0).all()
    assert (trend_empty["islem_sayisi"] == 0).all()
