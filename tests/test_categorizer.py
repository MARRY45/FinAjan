"""FinAjan işlem kategorileme (categorizer) birim testleri."""

import pandas as pd
import pytest

from data.generate_data import generate
from src.categorizer import (
    categorize_transactions,
    categorize_unknown_transactions,
    evaluate_categorizer,
    normalize_text,
)


@pytest.fixture(scope="module")
def dataset():
    df, gt = generate(seed=42)
    return df, gt


def test_normalize_text_turkish_characters():
    """Türkçe karakterlerin ve büyük-küçük harflerin doğru normalize edildiğini test eder."""
    assert normalize_text("İSTANBULKART") == "istanbulkart"
    assert normalize_text("IŞIK KIRTASİYE") == "isik kirtasiye"
    assert normalize_text("Şok Market") == "sok market"
    assert normalize_text("Çiçek Sepeti") == "cicek sepeti"
    assert normalize_text("Özlem Döner") == "ozlem doner"


def test_input_dataframe_not_mutated(dataset):
    """categorize_transactions fonksiyonunun girdi DataFrame'ini mutasyona uğratmadığını test eder."""
    df, _ = dataset
    original_cols = list(df.columns)
    original_len = len(df)
    original_kategori_counts = df["kategori"].value_counts().to_dict()

    res_df = categorize_transactions(df)

    assert list(df.columns) == original_cols
    assert len(df) == original_len
    assert df["kategori"].value_counts().to_dict() == original_kategori_counts
    # Yeni kolonlar sadece sonuç DataFrame'inde bulunmalı
    assert "onerilen_kategori" in res_df.columns
    assert "guven" in res_df.columns
    assert "yontem" in res_df.columns


def test_known_merchant_and_dictionary_matching():
    """Bilinen geçmiş veriden ve sözlükten doğru kategorilerin önerildiğini test eder."""
    sample_data = pd.DataFrame([
        # Geçmişte bilinen işlem
        {"tarih": "2026-01-01", "tutar": -100.0, "aciklama": "BIM MARKET", "isyeri": "BİM", "kategori": "Market", "hesap": "Kredi Kartı"},
        # Bilinmeyen işlem ama aynı işyeri (Yöntem 1: veri_temelli)
        {"tarih": "2026-01-02", "tutar": -50.0, "aciklama": "BIM HARCAMA", "isyeri": "BİM", "kategori": "Bilinmiyor", "hesap": "Kredi Kartı"},
        # Geçmişte olmayan ama sözlükte olan (Yöntem 2: sozluk)
        {"tarih": "2026-01-03", "tutar": -80.0, "aciklama": "KAHVE HARCAMA", "isyeri": "Starbucks Coffee", "kategori": "Bilinmiyor", "hesap": "Kredi Kartı"},
        # Sözlükte de olmayan tamamen belirsiz işlem
        {"tarih": "2026-01-04", "tutar": -30.0, "aciklama": "XYZ ABC LTD", "isyeri": "XYZ İşletmesi", "kategori": "Bilinmiyor", "hesap": "Kredi Kartı"},
    ])

    res = categorize_transactions(sample_data)

    # 1. satır: Zaten Market
    assert res.loc[0, "onerilen_kategori"] == "Market"
    assert res.loc[0, "yontem"] == "mevcut"

    # 2. satır: BİM geçmişten dolayı Market (yüksek güven)
    assert res.loc[1, "onerilen_kategori"] == "Market"
    assert res.loc[1, "guven"] == "yuksek"
    assert res.loc[1, "yontem"] == "veri_temelli"

    # 3. satır: Starbucks sözlükten dolayı Restoran (orta güven)
    assert res.loc[2, "onerilen_kategori"] == "Restoran"
    assert res.loc[2, "guven"] == "orta"
    assert res.loc[2, "yontem"] == "sozluk"

    # 4. satır: Belirsiz -> Bilinmiyor (düşük güven)
    assert res.loc[3, "onerilen_kategori"] == "Bilinmiyor"
    assert res.loc[3, "guven"] == "dusuk"
    assert res.loc[3, "yontem"] == "yok"


def test_evaluate_categorizer_accuracy_threshold(dataset):
    """Ground truth ile değerlendirmede doğruluk oranının >= 0.90 olduğunu doğrular."""
    df, gt = dataset
    eval_res = evaluate_categorizer(df, gt)

    assert eval_res["toplam_bilinmiyor"] > 0
    assert eval_res["kapsama"] >= 0.90
    assert eval_res["dogruluk"] >= 0.90
    assert eval_res["hatali_sayisi"] >= 0
    assert isinstance(eval_res["kategori_bazinda"], dict)


def test_categorize_unknown_transactions_tool(dataset):
    """Ajan araç fonksiyonunun doğru özet ve sınırlandırılmış liste döndürdüğünü test eder."""
    df, _ = dataset
    res = categorize_unknown_transactions(df, limit=10)

    assert isinstance(res, dict)
    assert res["bilinmiyor_sayisi"] == 114
    assert len(res["oneriler"]) == 10
    assert "özet" in res or "ozet" in res
    assert res["yuksek_guven_sayisi"] > 0

    first_item = res["oneriler"][0]
    assert "tarih" in first_item
    assert "isyeri" in first_item
    assert "onerilen_kategori" in first_item
    assert "guven" in first_item


def test_holdout_merchants_evaluation():
    """Sözlükte ve veri setinde bulunmayan 10 gerçekçi Türk işyeriyle hold-out testini yürütür.

    Sözlükte doğrudan adı geçmeyen işyerlerinde eşleşenlerin doğru kategorilendirildiğini,
    eşleşmeyenlerin ise yanlış tahmin yapılmadan 'Bilinmiyor'/'yok' (öneri yok) olarak bırakıldığını doğrular.
    """
    holdout_data = pd.DataFrame([
        {"tarih": "2026-05-01", "isyeri": "Kahve Dünyası", "aciklama": "KAHVE DUNYASI KADIKOY", "tutar": -85.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Restoran"},
        {"tarih": "2026-05-02", "isyeri": "Hakmar Ekspres", "aciklama": "HAKMAR SUBE 123 GIDA", "tutar": -210.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Market"},
        {"tarih": "2026-05-03", "isyeri": "Sephora Kozmetik", "aciklama": "SEPHORA NISANTASI", "tutar": -450.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Sağlık"},
        {"tarih": "2026-05-04", "isyeri": "Total Petrol", "aciklama": "TOTAL PETROL AKARYAKIT", "tutar": -900.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Ulaşım"},
        {"tarih": "2026-05-05", "isyeri": "Tavuk Dünyası", "aciklama": "TAVUK DUNYASI LEVENT", "tutar": -175.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Restoran"},
        {"tarih": "2026-05-06", "isyeri": "Mudo Concept", "aciklama": "MUDO CITY ZORLU", "tutar": -620.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Giyim"},
        {"tarih": "2026-05-07", "isyeri": "Bebek Kasabı", "aciklama": "BEBEK KASAP ET GIDA", "tutar": -350.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Market"},
        {"tarih": "2026-05-08", "isyeri": "Simit Sarayı", "aciklama": "SIMIT SARAYI BESIKTAS", "tutar": -45.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Restoran"},
        {"tarih": "2026-05-09", "isyeri": "Decathlon", "aciklama": "DECATHLON MALTEPE", "tutar": -780.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Diğer"},
        {"tarih": "2026-05-10", "isyeri": "Cinetech", "aciklama": "CINETECH TORUNLAR", "tutar": -120.0, "kategori": "Bilinmiyor", "hesap": "Kredi Kartı", "gercek": "Eğlence"},
    ])

    res = categorize_transactions(holdout_data)

    # 1. Eşleşen 4 işyeri doğru önerilmiş olmalıdır
    assert res.loc[0, "onerilen_kategori"] == "Restoran"  # Kahve Dünyası
    assert res.loc[3, "onerilen_kategori"] == "Ulaşım"    # Total Petrol
    assert res.loc[6, "onerilen_kategori"] == "Market"    # Bebek Kasabı
    assert res.loc[7, "onerilen_kategori"] == "Restoran"  # Simit Sarayı

    # 2. Eşleşmeyen 6 işyeri için yanlış tahmin yapılmamalı, öneri yok ('Bilinmiyor' / 'yok') dönmelidir
    unmatched_indices = [1, 2, 4, 5, 8, 9]
    for idx in unmatched_indices:
        assert res.loc[idx, "onerilen_kategori"] == "Bilinmiyor"
        assert res.loc[idx, "guven"] == "dusuk"
        assert res.loc[idx, "yontem"] == "yok"

