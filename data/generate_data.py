"""FinAjan sentetik işlem verisi üreticisi.

SEED=42 ile tamamen deterministik sentetik finansal veri seti üretir.
Çıktılar:
- data/transactions.csv
- data/ground_truth.json
"""

import calendar
import json
import math
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faker import Faker
import numpy as np
import pandas as pd

from src.constants import (
    ACCOUNT_KREDI_KARTI,
    ACCOUNT_VADESIZ,
    CATEGORY_ABONELIK,
    CATEGORY_BILINMIYOR,
    CATEGORY_DIGER,
    CATEGORY_EGLENCE,
    CATEGORY_FATURA,
    CATEGORY_GELIR,
    CATEGORY_GIYIM,
    CATEGORY_KIRA,
    CATEGORY_MARKET,
    CATEGORY_RESTORAN,
    CATEGORY_SAGLIK,
    CATEGORY_TEKNOLOJI,
    CATEGORY_ULASIM,
    COLUMNS,
    END_DATE,
    RANDOM_SEED,
    START_DATE,
)

# Gerçekçi Türk İşyerleri ve Banka Açıklama Kalıpları
MERCHANTS = {
    CATEGORY_MARKET: [
        ("Migros", "POS ALISVERIS MIGROS"),
        ("BİM", "POS HARCAMA BIM MARKET"),
        ("A101", "A101 YENIMAHALLE SUBESI"),
        ("Şok", "SOK MARKETLER T.A.S."),
        ("CarrefourSA", "CARREFOURSA MINI KART"),
    ],
    CATEGORY_RESTORAN: [
        ("Yemeksepeti", "YEMEKSEPETI SIPARIS"),
        ("GetirYemek", "GETIRYEMEK ODEME"),
        ("Starbucks", "STARBUCKS COFFEE"),
        ("Burger King", "BURGER KING SUBE"),
        ("Köfteci Yusuf", "KOFTECI YUSUF ET"),
        ("EspressoLab", "ESPRESSOLAB KAHVE"),
    ],
    CATEGORY_ULASIM: [
        ("İstanbulkart", "ISTANBULKART YUKLEME BELBIM"),
        ("Shell", "SHELL AKARYAKIT ISTASYONU"),
        ("Opet", "OPET PETROL OTOMOTIV"),
        ("BP", "BP PETROL OTOYOL"),
        ("Uber", "UBER TRIP YOLCULUK"),
        ("TCDD Taşımacılık", "TCDD ONLINE BILET"),
    ],
    CATEGORY_GIYIM: [
        ("LC Waikiki", "LC WAIKIKI MAGAZACILIK"),
        ("Zara", "ZARA GIYIM MAGAZA"),
        ("DeFacto", "DEFACTO PERAKENDE"),
        ("Mango", "MANGO TURKIYE GIYIM"),
        ("Boyner", "BOYNER BUYUK MAGAZACILIK"),
    ],
    CATEGORY_EGLENCE: [
        ("Biletix", "BILETIX ETKINLIK BILETI"),
        ("Cinemaximum", "CINEMAXIMUM SINEMA BILET"),
        ("Steam Games", "STEAM PURCHASE VALVE"),
        ("PlayStation Network", "PLAYSTATION NETWORK GIK"),
        ("D&R", "D&R MAGAZALARI KITAP"),
    ],
    CATEGORY_SAGLIK: [
        ("Eczane", "ECZANE ILAC VE MEDIKAL"),
        ("Acıbadem Sağlık", "ACIBADEM POLIKLINIK"),
        ("Memorial Hastanesi", "MEMORIAL HASTANESI LAB"),
        ("Gratis", "GRATIS KISISEL BAKIM"),
    ],
    CATEGORY_TEKNOLOJI: [
        ("Trendyol", "TRENDYOL.COM ELEKTRONIK"),
        ("Hepsiburada", "HEPSIBURADA PAZARYERI"),
        ("Amazon Türkiye", "AMAZON TURKEY PERAKENDE"),
        ("MediaMarkt", "MEDIAMARKT TEKNOLOJI"),
        ("Teknosa", "TEKNOSA ELEKTRONIK"),
        ("Apple Store", "APPLE STORE ZORLU"),
    ],
    CATEGORY_DIGER: [
        ("PTT", "PTT KARGO GONDERIM"),
        ("Kırtasiye", "SEMT KIRTASIYESI VE OFIS"),
        ("Oto Yıkama", "OTO YIKAMA VE TEMIZLIK"),
        ("Terzi", "TERZI TADILAT ISLERI"),
    ],
}

# Lognormal Dağılım Parametreleri (Değişken harcama büyüklükleri)
LOGNORMAL_PARAMS = {
    CATEGORY_MARKET: (5.85, 0.45),      # ~350 TL medyan
    CATEGORY_RESTORAN: (5.50, 0.50),    # ~245 TL medyan
    CATEGORY_ULASIM: (4.90, 0.55),      # ~135 TL medyan
    CATEGORY_GIYIM: (6.80, 0.40),       # ~900 TL medyan
    CATEGORY_EGLENCE: (5.70, 0.55),     # ~300 TL medyan
    CATEGORY_SAGLIK: (5.55, 0.60),      # ~260 TL medyan
    CATEGORY_TEKNOLOJI: (7.25, 0.50),   # ~1400 TL medyan
    CATEGORY_DIGER: (5.20, 0.50),       # ~180 TL medyan
}

VARIABLE_PROBS = {
    CATEGORY_MARKET: 0.32,
    CATEGORY_RESTORAN: 0.24,
    CATEGORY_ULASIM: 0.16,
    CATEGORY_GIYIM: 0.08,
    CATEGORY_EGLENCE: 0.08,
    CATEGORY_SAGLIK: 0.05,
    CATEGORY_TEKNOLOJI: 0.04,
    CATEGORY_DIGER: 0.03,
}


def _get_months(start_date: date, end_date: date):
    """Başlangıç ve bitiş tarihleri arasındaki tüm (yıl, ay) çiftlerini döndürür."""
    months = []
    y, m = start_date.year, start_date.month
    while True:
        months.append((y, m))
        if y == end_date.year and m == end_date.month:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def generate(seed: int = RANDOM_SEED):
    """Sentetik işlem verilerini ve ground truth metadatasını üretir.
    
    Returns:
        tuple[pd.DataFrame, dict]: (transactions_df, ground_truth_dict)
    """
    # 1. Tohumları ayarla (determinizm garantisi)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    _fake = Faker("tr_TR")

    months = _get_months(START_DATE, END_DATE)
    records = []
    seen_natural_keys = set()

    # 2. Aylık Sabit İşlemler (Maaş, Kira, Faturalar, Abonelikler)
    for year, month in months:
        # 2.1 Maaş (Gelir, Vadesiz Hesap, Ayın 1'i)
        records.append({
            "tarih": f"{year}-{month:02d}-01",
            "tutar": 65000.00,
            "aciklama": "MAAŞ ÖDEMESİ - ABC TEKNOLOJİ A.Ş.",
            "isyeri": "ABC Teknoloji A.Ş.",
            "kategori": CATEGORY_GELIR,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })

        # 2.2 Kira (Kira, Vadesiz Hesap, Ayın 3'ü)
        records.append({
            "tarih": f"{year}-{month:02d}-03",
            "tutar": -18500.00,
            "aciklama": "KİRA ÖDEMESİ - AHMET YILMAZ",
            "isyeri": "Ev Sahibi Ahmet Yılmaz",
            "kategori": CATEGORY_KIRA,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })

        # 2.3 Faturalar (3 Adet: İnternet, Elektrik, Su)
        # İnternet: Sabit
        records.append({
            "tarih": f"{year}-{month:02d}-10",
            "tutar": -390.00,
            "aciklama": "FATURA ÖDEMESİ TÜRK TELEKOM",
            "isyeri": "Türk Telekom",
            "kategori": CATEGORY_FATURA,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })
        # Elektrik: Mevsimsel ±%25 (Kış ve yaz aylarında artış)
        elek_factor = 1.0 + 0.22 * math.cos((month - 1) * 2 * math.pi / 12)
        records.append({
            "tarih": f"{year}-{month:02d}-15",
            "tutar": -round(850.00 * elek_factor, 2),
            "aciklama": "OTOMATİK FATURA ENERJİSA",
            "isyeri": "Enerjisa",
            "kategori": CATEGORY_FATURA,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })
        # Su: Mevsimsel ±%15
        su_factor = 1.0 + 0.12 * math.sin((month - 1) * 2 * math.pi / 12)
        records.append({
            "tarih": f"{year}-{month:02d}-20",
            "tutar": -round(320.00 * su_factor, 2),
            "aciklama": "OTOMATİK FATURA İSKİ",
            "isyeri": "İSKİ",
            "kategori": CATEGORY_FATURA,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })

        # 2.4 Abonelikler (4 Adet: Netflix, Spotify, YouTube Premium, MacFit)
        # Netflix (Ayın 12'si ±1 gün)
        day_netflix = 12 + (month % 3 - 1)
        records.append({
            "tarih": f"{year}-{month:02d}-{day_netflix:02d}",
            "tutar": -229.99,
            "aciklama": "NETFLIX.COM ABONELİK",
            "isyeri": "Netflix",
            "kategori": CATEGORY_ABONELIK,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })
        # Spotify (Ayın 18'i ±1 gün)
        day_spotify = 18 + ((month + 1) % 3 - 1)
        records.append({
            "tarih": f"{year}-{month:02d}-{day_spotify:02d}",
            "tutar": -59.99,
            "aciklama": "SPOTIFY PREMIUM",
            "isyeri": "Spotify",
            "kategori": CATEGORY_ABONELIK,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })
        # YouTube Premium (Ayın 24'ü ±1 gün)
        day_yt = 24 + ((month + 2) % 3 - 1)
        records.append({
            "tarih": f"{year}-{month:02d}-{day_yt:02d}",
            "tutar": -79.99,
            "aciklama": "GOOGLE YOUTUBE PREMIUM",
            "isyeri": "YouTube Premium",
            "kategori": CATEGORY_ABONELIK,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })
        # Spor Salonu (Ayın 5'i ±1 gün)
        day_macfit = 5 + (month % 3 - 1)
        records.append({
            "tarih": f"{year}-{month:02d}-{day_macfit:02d}",
            "tutar": -1250.00,
            "aciklama": "MACFIT AYLIK ÜYELİK",
            "isyeri": "MacFit",
            "kategori": CATEGORY_ABONELIK,
            "hesap": ACCOUNT_VADESIZ,
            "is_variable": False,
        })

    # Sabit kayıtların anahtarlarını collision setine ekle
    for r in records:
        seen_natural_keys.add((r["tarih"], r["isyeri"], r["tutar"], r["hesap"]))

    # 3. Değişken Günlük Harcamalar (Kredi Kartı)
    var_categories = list(VARIABLE_PROBS.keys())
    var_weights = [VARIABLE_PROBS[c] for c in var_categories]

    for year, month in months:
        days_in_month = calendar.monthrange(year, month)[1]
        # Ayda yaklaşık 45-70 işlem
        n_transactions = rng.integers(50, 65)

        for _ in range(n_transactions):
            # Kategori seç
            cat = rng.choice(var_categories, p=var_weights)
            merchants_list = MERCHANTS[cat]
            merchant_idx = rng.integers(0, len(merchants_list))
            isyeri, aciklama_base = merchants_list[merchant_idx]

            # Gün seçimi (Restoran ve Eğlence hafta sonu ağırlıklı)
            if cat in (CATEGORY_RESTORAN, CATEGORY_EGLENCE):
                day_weights = []
                for d in range(1, days_in_month + 1):
                    weekday = date(year, month, d).weekday()  # 5=Cumartesi, 6=Pazar
                    day_weights.append(3.0 if weekday in (5, 6) else 1.0)
                day_weights = np.array(day_weights) / sum(day_weights)
                day = rng.choice(range(1, days_in_month + 1), p=day_weights)
            else:
                day = rng.integers(1, days_in_month + 1)

            tarih_str = f"{year}-{month:02d}-{day:02d}"

            # Tutar çekimi (lognormal dağılım) ve collision önleme
            # Doğal çakışmaları ve aşırı uç değerleri (sahte anomali oluşumunu) engelle
            mu, sigma = LOGNORMAL_PARAMS[cat]
            max_normal_amt = math.exp(mu + 2.0 * sigma)
            attempts = 0
            while True:
                raw_amt = rng.lognormal(mean=mu, sigma=sigma)
                if raw_amt > max_normal_amt:
                    continue
                tutar = -round(float(raw_amt), 2)
                if tutar > -1.00:
                    tutar = -1.00
                key = (tarih_str, isyeri, tutar, ACCOUNT_KREDI_KARTI)
                if key not in seen_natural_keys:
                    seen_natural_keys.add(key)
                    break
                attempts += 1
                if attempts > 50:
                    tutar = round(tutar - 0.01 * attempts, 2)
                    seen_natural_keys.add((tarih_str, isyeri, tutar, ACCOUNT_KREDI_KARTI))
                    break

            records.append({
                "tarih": tarih_str,
                "tutar": tutar,
                "aciklama": aciklama_base,
                "isyeri": isyeri,
                "kategori": cat,
                "hesap": ACCOUNT_KREDI_KARTI,
                "is_variable": True,
            })

    # 4. Kasıtlı Anomali Enjeksiyonu (Tam 3 adet, farklı aylarda, farklı kategorilerde)
    # Kategoriler: Teknoloji, Giyim, Restoran
    # Medyanın en az 8 katı büyüklükte tutar
    anomalies = [
        {
            "tarih": "2025-12-18",
            "tutar": -38499.00,
            "aciklama": "POS ALISVERIS APPLE STORE ZORLU",
            "isyeri": "Apple Store",
            "kategori": CATEGORY_TEKNOLOJI,
            "hesap": ACCOUNT_KREDI_KARTI,
            "is_variable": True,
            "is_anomaly": True,
        },
        {
            "tarih": "2026-04-14",
            "tutar": -19850.00,
            "aciklama": "BOYNER BUYUK MAGAZACILIK",
            "isyeri": "Boyner",
            "kategori": CATEGORY_GIYIM,
            "hesap": ACCOUNT_KREDI_KARTI,
            "is_variable": True,
            "is_anomaly": True,
        },
        {
            "tarih": "2026-07-22",
            "tutar": -6950.00,
            "aciklama": "KOFTECI YUSUF ET",
            "isyeri": "Köfteci Yusuf",
            "kategori": CATEGORY_RESTORAN,
            "hesap": ACCOUNT_KREDI_KARTI,
            "is_variable": True,
            "is_anomaly": True,
        },
    ]

    for a in anomalies:
        records.append(a)
        seen_natural_keys.add((a["tarih"], a["isyeri"], a["tutar"], a["hesap"]))

    # 5. Kasıtlı Duplicate Enjeksiyonu (Tam 2 olay, farklı aylarda, değişken harcamalardan)
    # Olay 1: 2025-11 (Kasım)
    dup_event_1 = {
        "tarih": "2025-11-14",
        "tutar": -342.50,
        "aciklama": "POS HARCAMA BIM MARKET",
        "isyeri": "BİM",
        "kategori": CATEGORY_MARKET,
        "hesap": ACCOUNT_KREDI_KARTI,
        "is_variable": True,
        "is_duplicate": True,
    }
    # Olay 2: 2026-05 (Mayıs)
    dup_event_2 = {
        "tarih": "2026-05-19",
        "tutar": -185.00,
        "aciklama": "STARBUCKS COFFEE",
        "isyeri": "Starbucks",
        "kategori": CATEGORY_RESTORAN,
        "hesap": ACCOUNT_KREDI_KARTI,
        "is_variable": True,
        "is_duplicate": True,
    }

    # Her olay için tam 2 kayıt ekle
    records.append(dup_event_1.copy())
    records.append(dup_event_1.copy())
    records.append(dup_event_2.copy())
    records.append(dup_event_2.copy())

    # 6. "Bilinmiyor" Kategorisi Dönüşümü
    # Yalnızca DEĞİŞKEN harcamalarda, tüm işlemlerin %13-17'si
    # Anomali veya duplicate kayıtları bilinmiyor YAPMA
    total_tx_count = len(records)
    target_unknown_count = int(round(total_tx_count * 0.15))

    eligible_indices = [
        i for i, r in enumerate(records)
        if r.get("is_variable", False)
        and not r.get("is_anomaly", False)
        and not r.get("is_duplicate", False)
    ]

    chosen_unknown_indices = rng.choice(
        eligible_indices, size=target_unknown_count, replace=False
    )

    unknown_categories_map = {}
    for idx in chosen_unknown_indices:
        r = records[idx]
        key = f"{r['tarih']}|{r['isyeri']}|{r['tutar']:.2f}|{r['hesap']}"
        unknown_categories_map[key] = r["kategori"]
        r["kategori"] = CATEGORY_BILINMIYOR

    # Ground truth verisini hazırla
    ground_truth = {
        "anomalies": [
            {
                "tarih": a["tarih"],
                "isyeri": a["isyeri"],
                "tutar": a["tutar"],
                "kategori": a["kategori"],
                "hesap": a["hesap"],
                "row_key": f"{a['tarih']}|{a['isyeri']}|{a['tutar']:.2f}|{a['hesap']}",
            }
            for a in anomalies
        ],
        "duplicates": [
            {
                "event_id": 1,
                "row_key": f"{dup_event_1['tarih']}|{dup_event_1['isyeri']}|{dup_event_1['tutar']:.2f}|{dup_event_1['hesap']}",
                "records": [
                    {k: v for k, v in dup_event_1.items() if k in COLUMNS},
                    {k: v for k, v in dup_event_1.items() if k in COLUMNS},
                ],
            },
            {
                "event_id": 2,
                "row_key": f"{dup_event_2['tarih']}|{dup_event_2['isyeri']}|{dup_event_2['tutar']:.2f}|{dup_event_2['hesap']}",
                "records": [
                    {k: v for k, v in dup_event_2.items() if k in COLUMNS},
                    {k: v for k, v in dup_event_2.items() if k in COLUMNS},
                ],
            },
        ],
        "unknown_categories": unknown_categories_map,
    }

    # Temiz DataFrame oluştur ve gerekli kolonları seç
    df = pd.DataFrame(records)[COLUMNS]

    # Sıralama: tarih, hesap, isyeri, tutar, aciklama
    df = df.sort_values(
        by=["tarih", "hesap", "isyeri", "tutar", "aciklama"]
    ).reset_index(drop=True)

    return df, ground_truth


def main():
    """CSV ve ground_truth.json dosyalarını data/ dizini altına yazar."""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df, ground_truth = generate(seed=RANDOM_SEED)

    csv_path = data_dir / "transactions.csv"
    gt_path = data_dir / "ground_truth.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"Veri üretimi başarıyla tamamlandı:")
    print(f"- Toplam satır sayısı: {len(df)}")
    print(f"- CSV dosyası: {csv_path}")
    print(f"- Ground truth dosyası: {gt_path}")


if __name__ == "__main__":
    main()
