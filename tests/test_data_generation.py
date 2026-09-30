"""FinAjan sentetik veri üretimi testleri."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from data.generate_data import generate
from src.constants import (
    CATEGORY_ABONELIK,
    CATEGORY_BILINMIYOR,
    CATEGORY_FATURA,
    CATEGORY_GELIR,
    CATEGORY_KIRA,
    COLUMNS,
    END_DATE,
    START_DATE,
)


@pytest.fixture(scope="module")
def generated_dataset():
    """Testler için sentetik veriyi ve ground truth sözlüğünü üretir."""
    return generate(seed=42)


def test_schema_and_columns(generated_dataset):
    """CSV şemasının, kolon isimlerinin ve veri tiplerinin sözleşmeye uygunluğunu doğrular."""
    df, _ = generated_dataset

    # Kolonlar tam olarak eşleşmeli
    assert list(df.columns) == COLUMNS

    # Boş (NaN/Null) değer olmamalı
    assert df.isna().sum().sum() == 0

    # Satır sayısı anlamlı büyüklükte olmalı (45-70 işlem/ay * 12 ay + sabitler)
    assert len(df) >= 600

    # Tarih formatı ve tipler
    tarih_dt = pd.to_datetime(df["tarih"], errors="coerce")
    assert tarih_dt.notna().all()

    # Tutar float olmalı
    assert pd.api.types.is_float_dtype(df["tutar"]) or pd.api.types.is_numeric_dtype(df["tutar"])


def test_date_range(generated_dataset):
    """Tarih aralığının tam olarak 12 aylık dönemi kapsadığını doğrular."""
    df, _ = generated_dataset

    min_date = df["tarih"].min()
    max_date = df["tarih"].max()

    assert min_date == str(START_DATE)
    assert max_date <= str(END_DATE)

    # 12 farklı ay olmalı
    aylar = pd.to_datetime(df["tarih"]).dt.strftime("%Y-%m").unique()
    assert len(aylar) == 12


def test_amount_signs(generated_dataset):
    """Gelir pozitif, diğer tüm kategorilerin negatif tutarlı olduğunu doğrular."""
    df, _ = generated_dataset

    gelirler = df[df["kategori"] == CATEGORY_GELIR]
    assert (gelirler["tutar"] > 0).all()

    giderler = df[df["kategori"] != CATEGORY_GELIR]
    assert (giderler["tutar"] < 0).all()


def test_determinism():
    """Aynı seed ile iki kez generate() çağrıldığında sonuçların byte byte aynı olduğunu doğrular."""
    df1, gt1 = generate(seed=42)
    df2, gt2 = generate(seed=42)

    # DataFrame eşitliği
    pd.testing.assert_frame_equal(df1, df2)

    # CSV hash eşitliği
    csv1 = df1.to_csv(index=False, encoding="utf-8")
    csv2 = df2.to_csv(index=False, encoding="utf-8")
    hash1 = hashlib.sha256(csv1.encode("utf-8")).hexdigest()
    hash2 = hashlib.sha256(csv2.encode("utf-8")).hexdigest()
    assert hash1 == hash2

    # Ground truth eşitliği
    assert gt1 == gt2


def test_monthly_fixed_transactions(generated_dataset):
    """Her ayda tam 1 maaş, 1 kira, 3 fatura ve 4 abonelik olduğunu doğrular."""
    df, _ = generated_dataset

    df_copy = df.copy()
    df_copy["ay"] = pd.to_datetime(df_copy["tarih"]).dt.strftime("%Y-%m")
    aylar = sorted(df_copy["ay"].unique())
    assert len(aylar) == 12

    for ay in aylar:
        month_df = df_copy[df_copy["ay"] == ay]

        # 1 adet maaş
        maas = month_df[(month_df["kategori"] == CATEGORY_GELIR) & (month_df["tutar"] == 65000.0)]
        assert len(maas) == 1

        # 1 adet kira
        kira = month_df[(month_df["kategori"] == CATEGORY_KIRA) & (month_df["tutar"] == -18500.0)]
        assert len(kira) == 1

        # 3 adet fatura
        faturalar = month_df[month_df["kategori"] == CATEGORY_FATURA]
        assert len(faturalar) == 3

        # 4 adet abonelik
        abonelikler = month_df[month_df["kategori"] == CATEGORY_ABONELIK]
        assert len(abonelikler) == 4


def test_unknown_ratio(generated_dataset):
    """Bilinmiyor kategorisinin yalnızca değişken harcamalarda ve %12-18 bandında olduğunu doğrular."""
    df, gt = generated_dataset

    unk_count = (df["kategori"] == CATEGORY_BILINMIYOR).sum()
    unk_ratio = unk_count / len(df)

    # %12 - %18 aralığında olmalı
    assert 0.12 <= unk_ratio <= 0.18

    # Yalnızca Kredi Kartı (değişken harcamalar) bilinmiyor olmalı
    unk_df = df[df["kategori"] == CATEGORY_BILINMIYOR]
    assert (unk_df["hesap"] == "Kredi Kartı").all()

    # Ground truth eşleşmeli
    assert len(gt["unknown_categories"]) == unk_count


def test_duplicate_events_independent(generated_dataset):
    """CSV'den bağımsız olarak 4 kolona göre gruplama yapıldığında tam 2 adet duplicate olayının bulunduğunu doğrular."""
    df, gt = generated_dataset

    subset = ["tarih", "isyeri", "tutar", "hesap"]
    counts = df.groupby(subset).size()
    duplicate_groups = counts[counts >= 2]

    # Tam olarak 2 grup olmalı
    assert len(duplicate_groups) == 2

    # Her grupta tam olarak 2 kayıt olmalı
    for count in duplicate_groups:
        assert count == 2

    # Ground truth ile eşleşme
    gt_keys = {item["row_key"] for item in gt["duplicates"]}
    found_keys = {
        f"{idx[0]}|{idx[1]}|{idx[2]:.2f}|{idx[3]}" for idx in duplicate_groups.index
    }
    assert found_keys == gt_keys
