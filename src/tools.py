"""FinAjan finansal analiz araçları ve hesaplama modülü.

ÖNEMLİ VERİ SÖZLEŞMESİ VE KARAR:
- GELİR POZİTİF (örn. maaş: +65000.00 TL)
- GİDER NEGATİF (örn. harcama: -150.00 TL)
Hesaplama ve özet fonksiyonlarında gider toplamları pozitif büyüklük (büyüklük/mutlak değer)
olarak raporlanır; bakiye ise gelir + gider (cebirsel toplam) olarak hesaplanır.

Bu modül Streamlit, LLM veya dosya sistemi (load_transactions hariç) bağımlılığı içermez.
Saf Python ve Pandas fonksiyonları sunar; girdi DataFrame'lerini asla mutasyona uğratmaz.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from src.constants import (
    ANOMALY_ESIK,
    ANOMALY_EXCLUDED_CATEGORIES,
    ANOMALY_MIN_ORNEK,
    CATEGORY_BILINMIYOR,
    COLUMNS,
    END_DATE,
    RECURRING_INTERVAL_MAX,
    RECURRING_INTERVAL_MIN,
    RECURRING_MAX_CV,
    RECURRING_MIN_MONTHS,
)


class DataLoadError(Exception):
    """Veri yükleme, dosya okuma veya şema doğrulama hataları için özel istisna sınıfı.
    
    Güvenlik ilkesi gereği iç yol veya hassas sistem izleri (stack trace) sızdırmaz.
    """
    pass


def load_transactions(path: Union[str, Path]) -> pd.DataFrame:
    """Belirtilen yoldaki CSV dosyasını okur, şemasını doğrular ve tipleri dönüştürür.

    Args:
        path: CSV dosyasının dosya yolu.

    Returns:
        pd.DataFrame: Doğrulanmış ve temizlenmiş işlem veri tablosu.

    Raises:
        DataLoadError: Dosya bulunamazsa, boşsa veya şema geçersizse fırlatılır.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError("İşlem verisi dosyası belirtilen yolda bulunamadı.")

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception:
        raise DataLoadError("İşlem verisi dosyası okunamadı veya biçimi geçersiz.")

    if df.empty:
        raise DataLoadError("İşlem verisi dosyası boş.")

    # Zorunlu sütun kontrolü
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataLoadError("Veri dosyasında zorunlu sütunlar eksik.")

    df_clean = df.copy()

    # Tip dönüşümleri
    try:
        df_clean["tarih"] = pd.to_datetime(df_clean["tarih"])
    except Exception:
        raise DataLoadError("Tarih sütununda geçersiz format tespit edildi.")

    try:
        df_clean["tutar"] = pd.to_numeric(df_clean["tutar"], errors="raise").astype(float)
    except Exception:
        raise DataLoadError("Tutar sütununda sayısal olmayan değerler tespit edildi.")

    return df_clean[COLUMNS].copy()


def get_reference_date(df: pd.DataFrame) -> pd.Timestamp:
    """Veri setindeki son işlem tarihini referans tarihi olarak döndürür.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        pd.Timestamp: Veri setindeki en büyük tarih veya varsayılan END_DATE.
    """
    if df.empty or "tarih" not in df.columns:
        return pd.Timestamp(END_DATE)
    max_date = pd.to_datetime(df["tarih"]).max()
    if pd.isna(max_date):
        return pd.Timestamp(END_DATE)
    return max_date


def filter_period(
    df: pd.DataFrame,
    start: Union[str, date, datetime, pd.Timestamp],
    end: Union[str, date, datetime, pd.Timestamp],
) -> pd.DataFrame:
    """İşlem verilerini belirli bir başlangıç ve bitiş tarihi aralığına göre filtreler.

    Args:
        df: İşlem verilerini içeren DataFrame.
        start: Başlangıç tarihi (dahil).
        end: Bitiş tarihi (dahil).

    Returns:
        pd.DataFrame: Filtrelenmiş yeni DataFrame kopyası.
    """
    if df.empty or "tarih" not in df.columns:
        return df.copy()

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    df_copy = df.copy()
    tarih_series = pd.to_datetime(df_copy["tarih"])
    mask = (tarih_series >= start_ts) & (tarih_series <= end_ts)
    return df_copy[mask].copy().reset_index(drop=True)


def calculate_income(df: pd.DataFrame) -> float:
    """Veri setindeki pozitif tutarlı (gelir) işlemlerin toplamını döndürür.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        float: Gelirlerin toplam tutarı (pozitif).
    """
    if df.empty or "tutar" not in df.columns:
        return 0.0
    income = df[df["tutar"] > 0]["tutar"].sum()
    return float(round(income, 2))


def calculate_expenses(df: pd.DataFrame) -> float:
    """Veri setindeki negatif tutarlı (gider) işlemlerin pozitif büyüklük toplamını döndürür.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        float: Toplam harcama tutarı (pozitif büyüklük olarak).
    """
    if df.empty or "tutar" not in df.columns:
        return 0.0
    expenses = abs(df[df["tutar"] < 0]["tutar"].sum())
    return float(round(expenses, 2))


def calculate_balance(df: pd.DataFrame) -> float:
    """Gelirler ile giderlerin cebirsel farkını (net bakiye) hesaplar.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        float: Net bakiye (Gelir - Gider).
    """
    if df.empty or "tutar" not in df.columns:
        return 0.0
    balance = df["tutar"].sum()
    return float(round(balance, 2))


def summarize_transactions(df: pd.DataFrame) -> dict[str, Any]:
    """İşlem verilerinin genel özet istatistiklerini sözlük olarak döndürür.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        dict: Satır sayısı, tarih aralığı, gelir, gider, net ve bilinmiyor oranı.
    """
    if df.empty or "tutar" not in df.columns or "tarih" not in df.columns:
        return {
            "satir_sayisi": 0,
            "tarih_araligi": None,
            "gelir": 0.0,
            "gider": 0.0,
            "net": 0.0,
            "bilinmiyor_orani": 0.0,
        }

    tarih_series = pd.to_datetime(df["tarih"])
    min_date = str(tarih_series.min().date())
    max_date = str(tarih_series.max().date())
    date_range = f"{min_date} - {max_date}"

    unk_count = (df["kategori"] == CATEGORY_BILINMIYOR).sum() if "kategori" in df.columns else 0
    unk_ratio = float(round(unk_count / len(df), 4)) if len(df) > 0 else 0.0

    return {
        "satir_sayisi": int(len(df)),
        "tarih_araligi": date_range,
        "gelir": calculate_income(df),
        "gider": calculate_expenses(df),
        "net": calculate_balance(df),
        "bilinmiyor_orani": unk_ratio,
    }


def group_by_category(df: pd.DataFrame, expenses_only: bool = True) -> pd.DataFrame:
    """İşlemleri kategorilere göre gruplayıp toplam tutar, işlem sayısı ve oranları hesaplar.

    Args:
        df: İşlem verilerini içeren DataFrame.
        expenses_only: True ise yalnızca giderler (<0) analiz edilir.

    Returns:
        pd.DataFrame: [kategori, toplam, islem_sayisi, oran] kolonlarına sahip özet tablo.
    """
    cols = ["kategori", "toplam", "islem_sayisi", "oran"]
    if df.empty or "kategori" not in df.columns or "tutar" not in df.columns:
        return pd.DataFrame(columns=cols)

    df_copy = df.copy()
    if expenses_only:
        df_filtered = df_copy[df_copy["tutar"] < 0].copy()
    else:
        df_filtered = df_copy.copy()

    if df_filtered.empty:
        return pd.DataFrame(columns=cols)

    df_filtered["mutlak_tutar"] = df_filtered["tutar"].abs()
    grouped = (
        df_filtered.groupby("kategori")
        .agg(
            toplam=("mutlak_tutar", "sum"),
            islem_sayisi=("mutlak_tutar", "count"),
        )
        .reset_index()
    )

    grouped["toplam"] = grouped["toplam"].round(2)
    grand_total = grouped["toplam"].sum()
    if grand_total > 0:
        grouped["oran"] = (grouped["toplam"] / grand_total).round(4)
    else:
        grouped["oran"] = 0.0

    grouped = grouped.sort_values(by="toplam", ascending=False).reset_index(drop=True)
    return grouped[cols]


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """İşlemleri aylara (YYYY-MM) göre gruplayarak aylık gelir, gider, net ve işlem sayısını özetler.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        pd.DataFrame: [ay, gelir, gider, net, islem_sayisi] kolonlarına sahip özet tablo.
    """
    cols = ["ay", "gelir", "gider", "net", "islem_sayisi"]
    if df.empty or "tarih" not in df.columns or "tutar" not in df.columns:
        return pd.DataFrame(columns=cols)

    df_copy = df.copy()
    df_copy["ay"] = pd.to_datetime(df_copy["tarih"]).dt.strftime("%Y-%m")

    records = []
    for month_str, grp in df_copy.groupby("ay"):
        gelir = calculate_income(grp)
        gider = calculate_expenses(grp)
        net = calculate_balance(grp)
        records.append({
            "ay": month_str,
            "gelir": gelir,
            "gider": gider,
            "net": net,
            "islem_sayisi": int(len(grp)),
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="ay", ascending=True).reset_index(drop=True)
    else:
        summary_df = pd.DataFrame(columns=cols)

    return summary_df[cols]


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Aynı tarih, işyeri, tutar ve hesap bilgisine sahip yinelenen (duplicate) kayıtları bulur.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        pd.DataFrame: 2 veya daha fazla kez tekrarlanan tüm satırlar.
    """
    if df.empty:
        return df.copy()

    subset_cols = ["tarih", "isyeri", "tutar", "hesap"]
    for c in subset_cols:
        if c not in df.columns:
            return df.iloc[0:0].copy()

    dups = df[df.duplicated(subset=subset_cols, keep=False)].copy()
    dups = dups.sort_values(by=subset_cols).reset_index(drop=True)
    return dups


def find_recurring_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Düzenli tekrarlayan gider harcamalarını (abonelik, kira vb.) tespit eder.

    Kriterler:
    - Yalnızca gider satırları (<0)
    - Aynı gün tekrarları önce tekilleştirilir
    - En az 3 farklı ayda gerçekleşmiş olmalı
    - Ardışık işlemler arasındaki medyan aralık 25-35 gün olmalı
    - Tutar değişim katsayısı (std / mean) <= %5 olmalı

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        pd.DataFrame: [isyeri, kategori, ortalama_tutar, adet, medyan_aralik_gun, tutar_sapma_orani] tablosu.
    """
    result_cols = [
        "isyeri",
        "kategori",
        "ortalama_tutar",
        "adet",
        "medyan_aralik_gun",
        "tutar_sapma_orani",
    ]
    if df.empty or "tutar" not in df.columns or "isyeri" not in df.columns:
        return pd.DataFrame(columns=result_cols)

    # 1. Yalnızca gider satırları
    df_gider = df[df["tutar"] < 0].copy()
    if df_gider.empty:
        return pd.DataFrame(columns=result_cols)

    # 2. Aynı gün tekrarları tekilleştir
    df_unique = df_gider.drop_duplicates(subset=["isyeri", "tarih"]).copy()
    df_unique["tarih_dt"] = pd.to_datetime(df_unique["tarih"])
    df_unique["ay"] = df_unique["tarih_dt"].dt.to_period("M")

    recurring_list = []
    # İşyeri bazında grupla
    for (isyeri, kat), grp in df_unique.groupby(["isyeri", "kategori"]):
        # En az 3 farklı ay
        unique_months = grp["ay"].nunique()
        if unique_months < RECURRING_MIN_MONTHS:
            continue

        sorted_dates = grp["tarih_dt"].sort_values()
        date_diffs = sorted_dates.diff().dropna().dt.days
        if date_diffs.empty:
            continue

        medyan_aralik = float(date_diffs.median())
        # Medyan aralık 25-35 gün kontrolü
        if not (RECURRING_INTERVAL_MIN <= medyan_aralik <= RECURRING_INTERVAL_MAX):
            continue

        # Tutar sapma oranı (değişim katsayısı: std / mean)
        amts = grp["tutar"].abs()
        mean_amt = float(amts.mean())
        if mean_amt == 0:
            continue
        std_amt = float(amts.std(ddof=0)) if len(amts) > 1 else 0.0
        cv = std_amt / mean_amt

        # Tutar sapma oranı <= %5
        if cv > RECURRING_MAX_CV:
            continue

        recurring_list.append({
            "isyeri": isyeri,
            "kategori": kat,
            "ortalama_tutar": round(mean_amt, 2),
            "adet": int(len(grp)),
            "medyan_aralik_gun": round(medyan_aralik, 1),
            "tutar_sapma_orani": round(cv, 4),
        })

    if not recurring_list:
        return pd.DataFrame(columns=result_cols)

    res_df = pd.DataFrame(recurring_list)
    res_df = res_df.sort_values(by="ortalama_tutar", ascending=False).reset_index(drop=True)
    return res_df[result_cols]


def detect_large_transactions(
    df: pd.DataFrame,
    esik: float = ANOMALY_ESIK,
    min_ornek: int = ANOMALY_MIN_ORNEK,
) -> pd.DataFrame:
    """Kategori bazında modified z-score kullanarak sıra dışı büyük harcamaları tespit eder.

    Kriterler:
    - Yalnızca gider satırları (<0)
    - Kira, Fatura, Abonelik, Gelir ve Bilinmiyor kategorileri hariç tutulur
    - Kategori bazında modified z-score = 0.6745 * (x - medyan) / MAD
    - MAD == 0 ise IQR yöntemine (IQR / 1.349) başvurulur
    - Örnek sayısı < min_ornek olan kategoriler atlanır

    Args:
        df: İşlem verilerini içeren DataFrame.
        esik: Anomali eşik skoru (varsayılan: 3.5).
        min_ornek: Kategori başına gereken asgari işlem adedi (varsayılan: 8).

    Returns:
        pd.DataFrame: [tarih, isyeri, kategori, tutar, skor] tablosu.
    """
    result_cols = ["tarih", "isyeri", "kategori", "tutar", "skor"]
    if df.empty or "tutar" not in df.columns or "kategori" not in df.columns:
        return pd.DataFrame(columns=result_cols)

    # Yalnızca giderler ve hariç tutulan kategoriler dışındakiler
    df_gider = df[df["tutar"] < 0].copy()
    df_clean = df_gider[~df_gider["kategori"].isin(ANOMALY_EXCLUDED_CATEGORIES)].copy()

    if df_clean.empty:
        return pd.DataFrame(columns=result_cols)

    anomaly_rows = []
    for cat, grp in df_clean.groupby("kategori"):
        if len(grp) < min_ornek:
            continue

        x = grp["tutar"].abs().values
        med = float(np.median(x))
        abs_diff = np.abs(x - med)
        mad = float(np.median(abs_diff))

        if mad > 0:
            scores = 0.6745 * (x - med) / mad
        else:
            # MAD == 0 ise IQR'a düş
            q75, q25 = np.percentile(x, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                scale = iqr / 1.349
                scores = 0.6745 * (x - med) / scale
            else:
                std = float(np.std(x))
                if std > 0:
                    scores = 0.6745 * (x - med) / std
                else:
                    scores = np.zeros(len(x))

        mask = scores >= esik
        if np.any(mask):
            flagged = grp.iloc[mask].copy()
            flagged["skor"] = np.round(scores[mask], 2)
            for _, row in flagged.iterrows():
                anomaly_rows.append({
                    "tarih": str(pd.to_datetime(row["tarih"]).date()),
                    "isyeri": row["isyeri"],
                    "kategori": row["kategori"],
                    "tutar": float(round(row["tutar"], 2)),
                    "skor": float(row["skor"]),
                })

    if not anomaly_rows:
        return pd.DataFrame(columns=result_cols)

    out_df = pd.DataFrame(anomaly_rows)
    out_df = out_df.sort_values(by="skor", ascending=False).reset_index(drop=True)
    return out_df[result_cols]


def _normalize_tr(text: str) -> str:
    """Türkçe karakterleri ve büyük/küçük harfleri duyarsızlaştırmak için normalize eder."""
    if not isinstance(text, str):
        return ""
    # İ->i, I->i, ı->i dönüşümleri ile büyük/küçük harf duyarsızlığı
    res = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    return res.lower()


def search_transactions(
    df: pd.DataFrame,
    category: str | None = None,
    merchant: str | None = None,
    start: Union[str, date, datetime, pd.Timestamp, None] = None,
    end: Union[str, date, datetime, pd.Timestamp, None] = None,
    min_amount: float | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """İşlemleri filtreler, arar ve en yeni işlem önce gelecek şekilde döndürür.

    Args:
        df: İşlem verilerini içeren DataFrame.
        category: İsteğe bağlı kategori filtresi.
        merchant: İsteğe bağlı işyeri adı (büyük/küçük harf ve Türkçe İ/ı duyarsız 'içerir').
        start: İsteğe bağlı başlangıç tarihi (dahil).
        end: İsteğe bağlı bitiş tarihi (dahil).
        min_amount: İsteğe bağlı asgari mutlak tutar eşiği (|tutar| >= min_amount).
        limit: Döndürülecek azami kayıt sayısı (varsayılan: 20, üst sınır: 50).

    Returns:
        pd.DataFrame: Filtrelenmiş ve tarihe göre azalan sıralanmış işlemler.
    """
    if df.empty:
        return df.copy()

    df_filtered = df.copy()

    # Kategori filtresi
    if category:
        df_filtered = df_filtered[
            df_filtered["kategori"].str.lower() == category.strip().lower()
        ]

    # İşyeri filtresi (Türkçe İ/ı duyarsız içerir eşleşmesi)
    if merchant:
        norm_merchant = _normalize_tr(merchant.strip())
        is_match = df_filtered["isyeri"].apply(lambda x: norm_merchant in _normalize_tr(str(x)))
        df_filtered = df_filtered[is_match]

    # Tarih aralığı filtresi
    if start:
        df_filtered = df_filtered[pd.to_datetime(df_filtered["tarih"]) >= pd.to_datetime(start)]
    if end:
        df_filtered = df_filtered[pd.to_datetime(df_filtered["tarih"]) <= pd.to_datetime(end)]

    # Asgari mutlak tutar filtresi
    if min_amount is not None:
        df_filtered = df_filtered[df_filtered["tutar"].abs() >= float(min_amount)]

    # Limit üst sınırı 50
    effective_limit = min(max(1, int(limit)), 50)

    # En yeni işlem önce
    df_sorted = df_filtered.sort_values(by="tarih", ascending=False).reset_index(drop=True)
    return df_sorted.head(effective_limit).copy()


def category_monthly_totals(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Belirtilen bir kategorinin tüm aylardaki harcama toplamını ve işlem sayısını döndürür.
    
    Harcama olmayan aylar 0.0 tutar ve 0 işlem sayısı ile doldurulur.

    Args:
        df: İşlem verilerini içeren DataFrame.
        category: İncelenecek harcama kategorisi.

    Returns:
        pd.DataFrame: [ay, toplam, islem_sayisi] tablosu (ay sıralı).
    """
    result_cols = ["ay", "toplam", "islem_sayisi"]
    if df.empty or "tarih" not in df.columns:
        return pd.DataFrame(columns=result_cols)

    df_copy = df.copy()
    df_copy["ay"] = pd.to_datetime(df_copy["tarih"]).dt.strftime("%Y-%m")
    all_months = sorted(df_copy["ay"].unique())

    # Kategori filtresi (harcamalar pozitif büyüklük olarak)
    cat_mask = (df_copy["kategori"].str.lower() == category.strip().lower()) & (df_copy["tutar"] < 0)
    df_cat = df_copy[cat_mask].copy()

    records = []
    if df_cat.empty:
        for m in all_months:
            records.append({"ay": m, "toplam": 0.0, "islem_sayisi": 0})
    else:
        grouped = df_cat.groupby("ay").agg(
            toplam=("tutar", lambda s: round(float(abs(s).sum()), 2)),
            islem_sayisi=("tutar", "count"),
        ).to_dict(orient="index")

        for m in all_months:
            if m in grouped:
                records.append({
                    "ay": m,
                    "toplam": grouped[m]["toplam"],
                    "islem_sayisi": int(grouped[m]["islem_sayisi"]),
                })
            else:
                records.append({"ay": m, "toplam": 0.0, "islem_sayisi": 0})

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df = out_df.sort_values(by="ay", ascending=True).reset_index(drop=True)
    else:
        out_df = pd.DataFrame(columns=result_cols)

    return out_df[result_cols]
