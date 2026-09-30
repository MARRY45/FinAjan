"""FinAjan ajan araçları ve yürütme (dispatcher) katmanı.

Ajan yalnızca bu modüldeki araçları kullanabilir; dosya sistemine veya
ham DataFrame'e doğrudan erişimi yoktur.
Tüm araçlar JSON'a serileştirilebilir sözlükler döndürür.
"""

import json
from datetime import date, datetime
from typing import Any, Union

import pandas as pd

from src.constants import (
    ACCOUNTS,
    ANOMALY_HIGH_SCORE,
    CATEGORIES,
    START_DATE,
)
from src import tools


# --- OpenAI Function Şemaları ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_info",
            "description": "Veri setinin başlangıç ve bitiş tarihlerini, referans ayını ('bu ay'), toplam işlem sayısını ve geçerli kategorileri döndürür.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Belirli bir tarih aralığında veya tüm veri setinde toplam gelir, toplam gider, net bakiye, işlem sayısı ve bilinmeyen kategori oranını özetler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Başlangıç tarihi (YYYY-MM-DD formatında). Belirtilmezse veri setinin başlangıcı alınır.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Bitiş tarihi (YYYY-MM-DD formatında). Belirtilmezse veri setinin sonu alınır.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_summary",
            "description": "Aylık bazda (YYYY-MM) gelir, gider, net bakiye ve işlem sayısı dökümünü döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_month": {
                        "type": "string",
                        "description": "Başlangıç ayı (YYYY-MM formatında, örn. '2025-10').",
                    },
                    "end_month": {
                        "type": "string",
                        "description": "Bitiş ayı (YYYY-MM formatında, örn. '2026-09').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_summary",
            "description": "Kategorilere göre harcama dağılımını, toplam harcama tutarlarını, işlem sayılarını ve genel harcamadaki oranlarını döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Başlangıç tarihi (YYYY-MM-DD).",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Bitiş tarihi (YYYY-MM-DD).",
                    },
                    "expenses_only": {
                        "type": "boolean",
                        "description": "Yalnızca gider harcamalarını analiz etmek için true (varsayılan: true).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_trend",
            "description": "Belirtilen bir harcama kategorisinin son N aydaki (varsayılan: son 3 ay) aylık harcama trendini ve işlem sayılarını getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "İncelenecek harcama kategorisi (örn. 'Market', 'Restoran', 'Giyim').",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Kaç aylık trend getirileceği (varsayılan: 3).",
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "İşlemleri işyeri adına (büyük/küçük harf ve Türkçe karakter duyarsız arama), kategoriye, tarihe veya asgari tutara göre listeler. En yeni işlemler önce listelenir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "İsteğe bağlı kategori filtresi.",
                    },
                    "merchant": {
                        "type": "string",
                        "description": "İşyeri veya kurum adı (örn. 'Migros', 'Starbucks', 'BİM').",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Başlangıç tarihi (YYYY-MM-DD).",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Bitiş tarihi (YYYY-MM-DD).",
                    },
                    "min_amount": {
                        "type": "number",
                        "description": "Asgari mutlak işlem tutarı (örn. 500.0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Listelenecek azami işlem adedi (varsayılan: 20, üst sınır: 50).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_duplicates",
            "description": "Aynı gün, aynı işyeri, aynı hesap ve aynı tutarda mükerrer çekilmiş şüpheli kopya (duplicate) işlemleri bulur.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_large_transactions",
            "description": "Kategori bazında istatistiksel modified z-score yöntemiyle olağandışı büyük harcamaları tespit eder. Skor >= 10.0 olanlar 'yuksek' seviye, diğerleri 'dusuk' seviye olarak etiketlenir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Başlangıç tarihi (YYYY-MM-DD).",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Bitiş tarihi (YYYY-MM-DD).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subscriptions",
            "description": "Düzenli aralıklarla tekrarlanan sabit ödemeleri (Netflix, Spotify vb. abonelikler, kira ve sabit faturalar) tespit eder. Her kaydın kategorisi belirtilir.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "categorize_unknown_transactions",
            "description": "Kategorisi 'Bilinmiyor' olan harcama işlemlerini geçmiş işlem hafızası ve işyeri kural motoruyla analiz ederek önerilen kategorileri ve güven skorlarını döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Döndürülecek azami öneri adedi (varsayılan: 50).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_savings_opportunities",
            "description": "Abonelikler ve yıllık maliyetleri, mükerrer çekimler ve potansiyel iade tutarları ile önceki aylara göre yüksek artış gösteren harcama kategorilerini analiz eder.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# --- Yardımcı Doğrulama ve Kırpma Fonksiyonları ---

def _validate_date(d_str: str | None) -> tuple[bool, str | None]:
    """Tarih dizesinin YYYY-MM-DD formatında olup olmadığını doğrular."""
    if not d_str:
        return True, None
    try:
        datetime.strptime(d_str.strip(), "%Y-%m-%d")
        return True, None
    except ValueError:
        return False, f"Geçersiz tarih biçimi: '{d_str}'. Beklenen format: YYYY-MM-DD."


def _validate_month(m_str: str | None) -> tuple[bool, str | None]:
    """Ay dizesinin YYYY-MM formatında olup olmadığını doğrular."""
    if not m_str:
        return True, None
    try:
        datetime.strptime(m_str.strip(), "%Y-%m")
        return True, None
    except ValueError:
        return False, f"Geçersiz ay biçimi: '{m_str}'. Beklenen format: YYYY-MM."


def _validate_category(cat_str: str | None) -> tuple[bool, str | None]:
    """Kategorinin geçerli kategorilerden biri olup olmadığını doğrular."""
    if not cat_str:
        return True, None
    matched = [c for c in CATEGORIES if c.lower() == cat_str.strip().lower()]
    if matched:
        return True, matched[0]
    return False, None


def _truncate_output_if_needed(result: dict[str, Any]) -> dict[str, Any]:
    """Çıktı JSON'u 8000 karakteri aşıyorsa satırları kırpar ve 'kirpildi': true ekler."""
    json_str = json.dumps(result, ensure_ascii=False)
    if len(json_str) <= 8000:
        return result

    result_copy = dict(result)
    result_copy["kirpildi"] = True

    # Listeleri küçülterek 8000 karakterin altına çek
    for key, val in list(result_copy.items()):
        if isinstance(val, list) and len(val) > 1:
            while len(val) > 1 and len(json.dumps(result_copy, ensure_ascii=False)) > 8000:
                val.pop()

    return result_copy


# --- Araç Gerçekleştirmeleri ---

def tool_get_dataset_info(df: pd.DataFrame, **kwargs) -> dict[str, Any]:
    ref_date = tools.get_reference_date(df)
    tarih_series = pd.to_datetime(df["tarih"]) if not df.empty else pd.Series()
    min_date_str = str(tarih_series.min().date()) if not df.empty else str(START_DATE)
    max_date_str = str(ref_date.date())

    return {
        "baslangic_tarihi": min_date_str,
        "bitis_tarihi": max_date_str,
        "son_ay": ref_date.strftime("%Y-%m"),
        "toplam_islem_sayisi": int(len(df)),
        "gecerli_kategoriler": CATEGORIES,
        "gecerli_hesaplar": ACCOUNTS,
        "para_birimi": "TRY",
    }


def tool_get_summary(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    v_ok1, err1 = _validate_date(start_date)
    if not v_ok1:
        return {"error": err1, "ornek": "2026-01-01"}
    v_ok2, err2 = _validate_date(end_date)
    if not v_ok2:
        return {"error": err2, "ornek": "2026-09-30"}

    df_filtered = df
    if start_date or end_date:
        s = start_date or str(pd.to_datetime(df["tarih"]).min().date())
        e = end_date or str(pd.to_datetime(df["tarih"]).max().date())
        df_filtered = tools.filter_period(df, s, e)

    s_dict = tools.summarize_transactions(df_filtered)
    return {
        "tarih_araligi": s_dict["tarih_araligi"],
        "islem_sayisi": s_dict["satir_sayisi"],
        "toplam_gelir": s_dict["gelir"],
        "toplam_gider": s_dict["gider"],
        "net_bakiye": s_dict["net"],
        "bilinmeyen_kategori_orani": s_dict["bilinmiyor_orani"],
        "para_birimi": "TRY",
    }


def tool_get_monthly_summary(
    df: pd.DataFrame,
    start_month: str | None = None,
    end_month: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    v_ok1, err1 = _validate_month(start_month)
    if not v_ok1:
        return {"error": err1, "ornek": "2026-01"}
    v_ok2, err2 = _validate_month(end_month)
    if not v_ok2:
        return {"error": err2, "ornek": "2026-09"}

    m_df = tools.monthly_summary(df)
    if start_month:
        m_df = m_df[m_df["ay"] >= start_month]
    if end_month:
        m_df = m_df[m_df["ay"] <= end_month]

    records = m_df.to_dict(orient="records")
    return {
        "aylik_ozetler": records,
        "para_birimi": "TRY",
    }


def tool_get_category_summary(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    expenses_only: bool = True,
    **kwargs,
) -> dict[str, Any]:
    v_ok1, err1 = _validate_date(start_date)
    if not v_ok1:
        return {"error": err1, "ornek": "2026-01-01"}
    v_ok2, err2 = _validate_date(end_date)
    if not v_ok2:
        return {"error": err2, "ornek": "2026-09-30"}

    df_filtered = df
    if start_date or end_date:
        s = start_date or str(pd.to_datetime(df["tarih"]).min().date())
        e = end_date or str(pd.to_datetime(df["tarih"]).max().date())
        df_filtered = tools.filter_period(df, s, e)

    cat_df = tools.group_by_category(df_filtered, expenses_only=expenses_only)
    records = cat_df.to_dict(orient="records")
    return {
        "kategori_dagilimi": records,
        "gider_toplami": float(round(cat_df["toplam"].sum(), 2)),
        "para_birimi": "TRY",
    }


def tool_get_category_trend(
    df: pd.DataFrame,
    category: str,
    months: int = 3,
    **kwargs,
) -> dict[str, Any]:
    v_ok, actual_cat = _validate_category(category)
    if not v_ok or not actual_cat:
        return {
            "error": f"'{category}' geçerli bir kategori değil.",
            "gecerli_degerler": CATEGORIES,
        }

    trend_df = tools.category_monthly_totals(df, actual_cat)
    effective_months = max(1, int(months))
    subset = trend_df.tail(effective_months)
    records = subset.to_dict(orient="records")

    return {
        "kategori": actual_cat,
        "ay_sayisi": len(records),
        "aylik_trend": records,
        "para_birimi": "TRY",
    }


def tool_list_transactions(
    df: pd.DataFrame,
    category: str | None = None,
    merchant: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_amount: float | None = None,
    limit: int = 20,
    **kwargs,
) -> dict[str, Any]:
    actual_cat = None
    if category:
        v_ok, actual_cat = _validate_category(category)
        if not v_ok:
            return {
                "error": f"'{category}' geçerli bir kategori değil.",
                "gecerli_degerler": CATEGORIES,
            }

    v_ok1, err1 = _validate_date(start_date)
    if not v_ok1:
        return {"error": err1, "ornek": "2026-01-01"}
    v_ok2, err2 = _validate_date(end_date)
    if not v_ok2:
        return {"error": err2, "ornek": "2026-09-30"}

    res_df = tools.search_transactions(
        df=df,
        category=actual_cat,
        merchant=merchant,
        start=start_date,
        end=end_date,
        min_amount=min_amount,
        limit=limit,
    )

    records = []
    for _, row in res_df.iterrows():
        records.append({
            "tarih": str(pd.to_datetime(row["tarih"]).date()),
            "tutar": float(round(row["tutar"], 2)),
            "isyeri": row["isyeri"],
            "kategori": row["kategori"],
            "hesap": row["hesap"],
            "aciklama": row["aciklama"],
        })

    return {
        "islem_sayisi": len(records),
        "islemler": records,
        "para_birimi": "TRY",
    }


def tool_find_duplicates(df: pd.DataFrame, **kwargs) -> dict[str, Any]:
    dups_df = tools.find_duplicates(df)
    if dups_df.empty:
        return {
            "duplicate_grup_sayisi": 0,
            "toplam_kayit_sayisi": 0,
            "gruplar": [],
            "para_birimi": "TRY",
        }

    subset = ["tarih", "isyeri", "tutar", "hesap"]
    groups_list = []
    for (t, isyeri, tutar, hesap), grp in dups_df.groupby(subset):
        groups_list.append({
            "tarih": str(pd.to_datetime(t).date()),
            "isyeri": isyeri,
            "tutar": float(round(abs(tutar), 2)),
            "hesap": hesap,
            "adet": int(len(grp)),
            "aciklama": grp["aciklama"].iloc[0],
        })

    return {
        "duplicate_grup_sayisi": len(groups_list),
        "toplam_kayit_sayisi": len(dups_df),
        "gruplar": groups_list,
        "para_birimi": "TRY",
    }


def tool_detect_large_transactions(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    v_ok1, err1 = _validate_date(start_date)
    if not v_ok1:
        return {"error": err1, "ornek": "2026-01-01"}
    v_ok2, err2 = _validate_date(end_date)
    if not v_ok2:
        return {"error": err2, "ornek": "2026-09-30"}

    df_filtered = df
    if start_date or end_date:
        s = start_date or str(pd.to_datetime(df["tarih"]).min().date())
        e = end_date or str(pd.to_datetime(df["tarih"]).max().date())
        df_filtered = tools.filter_period(df, s, e)

    anomalies_df = tools.detect_large_transactions(df_filtered)
    records = []
    for _, row in anomalies_df.iterrows():
        skor = float(row["skor"])
        seviye = "yuksek" if skor >= ANOMALY_HIGH_SCORE else "dusuk"
        records.append({
            "tarih": str(row["tarih"]),
            "isyeri": row["isyeri"],
            "kategori": row["kategori"],
            "tutar": float(round(abs(row["tutar"]), 2)),
            "skor": skor,
            "seviye": seviye,
        })

    return {
        "anomali_sayisi": len(records),
        "anomaliler": records,
        "para_birimi": "TRY",
    }


def tool_get_subscriptions(df: pd.DataFrame, **kwargs) -> dict[str, Any]:
    rec_df = tools.find_recurring_transactions(df)
    records = []
    for _, row in rec_df.iterrows():
        records.append({
            "isyeri": row["isyeri"],
            "kategori": row["kategori"],
            "ortalama_tutar": float(row["ortalama_tutar"]),
            "islem_adeti": int(row["adet"]),
            "medyan_aralik_gun": float(row["medyan_aralik_gun"]),
            "tutar_sapma_orani": float(row["tutar_sapma_orani"]),
        })

    return {
        "duzenli_islem_sayisi": len(records),
        "duzenli_islemler": records,
        "para_birimi": "TRY",
    }


def tool_categorize_unknown_transactions(df: pd.DataFrame, limit: int = 50, **kwargs) -> dict[str, Any]:
    from src.categorizer import categorize_unknown_transactions
    return categorize_unknown_transactions(df, limit=limit)


def tool_get_savings_opportunities(df: pd.DataFrame, **kwargs) -> dict[str, Any]:
    from src.savings import get_savings_opportunities
    return get_savings_opportunities(df)


# --- Araç Yürütücü (Dispatcher) ---

DISPATCHER = {
    "get_dataset_info": tool_get_dataset_info,
    "get_summary": tool_get_summary,
    "get_monthly_summary": tool_get_monthly_summary,
    "get_category_summary": tool_get_category_summary,
    "get_category_trend": tool_get_category_trend,
    "list_transactions": tool_list_transactions,
    "find_duplicates": tool_find_duplicates,
    "detect_large_transactions": tool_detect_large_transactions,
    "get_subscriptions": tool_get_subscriptions,
    "categorize_unknown_transactions": tool_categorize_unknown_transactions,
    "get_savings_opportunities": tool_get_savings_opportunities,
}


def execute_tool(name: str, args: dict[str, Any] | None, df: pd.DataFrame) -> dict[str, Any]:
    """Belirtilen aracı verilen argümanlar ve DataFrame ile çalıştırır.

    Args:
        name: Çağrılacak aracın adı.
        args: Araca iletilecek parametreler sözlüğü.
        df: İşlem verilerini içeren DataFrame.

    Returns:
        dict: Aracın JSON serileştirilebilir çıktı sözlüğü (hata veya veri).
    """
    if name not in DISPATCHER:
        return {
            "error": f"Bilinmeyen araç adı: '{name}'.",
            "gecerli_araclar": list(DISPATCHER.keys()),
        }

    func = DISPATCHER[name]
    call_args = args or {}

    try:
        raw_result = func(df=df, **call_args)
    except Exception as e:
        return {
            "error": f"Araç çalıştırılırken dahili hata oluştu: {type(e).__name__}",
        }

    return _truncate_output_if_needed(raw_result)
