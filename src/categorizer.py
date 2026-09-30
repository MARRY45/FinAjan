"""FinAjan akıllı işlem kategorileme modülü.

Kategorisi 'Bilinmiyor' olarak işaretlenmiş finansal hareketleri, geçmiş işlem verilerinden
ve Türkçe işyeri sözlüğünden yararlanarak deterministik kurallarla kategorilere önerir.
Girdi DataFrame'ini mutasyona uğratmaz; orijinal veriyi değiştirmez, sadece öneri kolonları ekler.
"""

from typing import Any
import pandas as pd

from src.constants import (
    CATEGORY_ABONELIK,
    CATEGORY_BILINMIYOR,
    CATEGORY_DIGER,
    CATEGORY_EGLENCE,
    CATEGORY_FATURA,
    CATEGORY_GIYIM,
    CATEGORY_KIRA,
    CATEGORY_MARKET,
    CATEGORY_RESTORAN,
    CATEGORY_SAGLIK,
    CATEGORY_TEKNOLOJI,
    CATEGORY_ULASIM,
)

# Gerçek hayatta yaygın Türk işyeri ve anahtar kelime sözlüğü
KEYWORD_DICTIONARY: dict[str, list[str]] = {
    CATEGORY_MARKET: [
        "migros", "a101", "bim", "sok", "carrefour", "file", "macrocenter",
        "tarim kredi", "pazar", "bakkal", "manav", "kasap", "sarkuteri", "market",
    ],
    CATEGORY_RESTORAN: [
        "starbucks", "kahve", "burger", "kofte", "doner", "pizza", "restoran",
        "lokanta", "yemeksepeti", "kebap", "mcdonalds", "pide", "pastane", "cafe",
        "kafe", "firinci", "meyhane", "tatli", "lahmacun", "durum", "simit",
    ],
    CATEGORY_GIYIM: [
        "zara", "lcw", "waikiki", "mango", "defacto", "boyner", "koton", "mavi",
        "ipekyol", "vakko", "derimod", "network", "ayakkabi", "butik", "giyim",
        "tekstil", "kundura", "flo", "penti",
    ],
    CATEGORY_TEKNOLOJI: [
        "apple", "vatan", "teknosa", "mediamarkt", "samsung", "bilgisayar",
        "elektronik", "dijital", "teknoloji", "monster", "itopya",
    ],
    CATEGORY_ULASIM: [
        "shell", "opet", "bp", "petrol ofisi", "istanbulkart", "marmaray",
        "metro", "taksi", "uber", "bilet", "havalimani", "otogar", "akaryakit",
        "benzin", "otobus", "pegasus", "thy", "marti", "binbin", "otopark",
    ],
    CATEGORY_FATURA: [
        "turk telekom", "vodafone", "iski", "igdas", "ck bogazici",
        "enerjisa", "fatura", "elektrik", "dogalgaz", "su faturasi", "turksat", "telekom",
    ],
    CATEGORY_ABONELIK: [
        "netflix", "spotify", "youtube", "disney", "macfit", "blutv", "prime",
        "spor salonu", "gym", "fitness", "abonelik", "exxen", "gain",
    ],
    CATEGORY_SAGLIK: [
        "eczane", "hastane", "klinik", "medikal", "laboratuvar", "saglik",
        "doktor", "dis", "optik", "gratis", "watsons", "acibadem", "medicana",
        "liv hospital", "florence", "medical park",
    ],
    CATEGORY_KIRA: [
        "kira", "ev sahibi", "emlak", "aidat", "apartman", "konut", "rezidans",
    ],
    CATEGORY_EGLENCE: [
        "sinema", "tiyatro", "konser", "biletix", "steam", "playstation",
        "lunapark", "muze", "bovling", "paribu cineverse", "sinemax",
    ],
    CATEGORY_DIGER: [
        "ptt", "kargo", "kirtasiye", "oto yikama", "terzi", "kuru temizleme", "noter",
    ],
}


def normalize_text(text: Any) -> str:
    """Türkçe karakterleri ve büyük/küçük harf duyarlılığını normalize eder.

    Args:
        text: Normalize edilecek metin nesnesi.

    Returns:
        str: Küçük harfli ve Türkçe karakterleri dönüştürülmüş metin.
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()
    tr_map = {
        "İ": "i", "I": "ı", "ı": "i", "ç": "c", "Ç": "c",
        "ğ": "g", "Ğ": "g", "ö": "o", "Ö": "o", "ş": "s",
        "Ş": "s", "ü": "u", "Ü": "u",
    }
    for tr_char, eng_char in tr_map.items():
        t = t.replace(tr_char, eng_char)

    return t.lower()


def _build_historical_merchant_map(df: pd.DataFrame) -> dict[str, str]:
    """Kategorisi bilinen geçmiş işlemlerden işyeri -> kategori eşleme tablosu çıkarır.

    Eğer bir işyeri yalnızca tek bir kategoride harcama görmüşse yüksek güvenle eşlenir.
    """
    known_df = df[df["kategori"] != CATEGORY_BILINMIYOR]
    if known_df.empty:
        return {}

    merchant_map: dict[str, str] = {}
    for merchant, grp in known_df.groupby("isyeri"):
        norm_m = normalize_text(merchant)
        cats = grp["kategori"].unique()
        if len(cats) == 1:
            merchant_map[norm_m] = str(cats[0])

    return merchant_map


def _match_dictionary_category(norm_merchant: str, norm_desc: str) -> str | None:
    """Türkçe işyeri sözlüğünden anahtar kelime eşleşmesi arar."""
    combined_text = f"{norm_merchant} {norm_desc}"

    for category, keywords in KEYWORD_DICTIONARY.items():
        for kw in keywords:
            # Kelime sınırıyla veya tam içerik kontrolü
            if kw in norm_merchant or kw in combined_text:
                return category

    return None


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """İşlem tablosundaki 'Bilinmiyor' kategorili satırlara kategori önerileri ekler.

    Girdi tablosunu değiştirmez. Yeni kolonlar:
    - onerilen_kategori: Önerilen standart kategori
    - guven: 'yuksek', 'orta', 'dusuk'
    - yontem: 'veri_temelli', 'sozluk', 'mevcut', 'yok'

    Args:
        df: Finansal işlem DataFrame'i.

    Returns:
        pd.DataFrame: Öneri kolonları eklenmiş yeni DataFrame kopyası.
    """
    if df is None or df.empty:
        res = pd.DataFrame(columns=list(df.columns) + ["onerilen_kategori", "guven", "yontem"]) if df is not None else pd.DataFrame()
        return res

    result_df = df.copy()

    # 1. Yöntem hazırlığı: Geçmiş veri tabanlı eşleşme
    historical_map = _build_historical_merchant_map(result_df)

    onerilen_list: list[str] = []
    guven_list: list[str] = []
    yontem_list: list[str] = []

    for _, row in result_df.iterrows():
        current_cat = row.get("kategori", CATEGORY_BILINMIYOR)

        # Kategorisi zaten bilinen işlemler
        if current_cat != CATEGORY_BILINMIYOR:
            onerilen_list.append(current_cat)
            guven_list.append("yuksek")
            yontem_list.append("mevcut")
            continue

        norm_m = normalize_text(row.get("isyeri", ""))
        norm_d = normalize_text(row.get("aciklama", ""))

        # Yöntem 1: Geçmiş veride aynı işyeri tek bir kategoriye sahip mi?
        if norm_m in historical_map:
            onerilen_list.append(historical_map[norm_m])
            guven_list.append("yuksek")
            yontem_list.append("veri_temelli")
            continue

        # Yöntem 2: Genel Türkçe işyeri sözlüğüyle anahtar kelime eşleşmesi
        dict_cat = _match_dictionary_category(norm_m, norm_d)
        if dict_cat:
            onerilen_list.append(dict_cat)
            guven_list.append("orta")
            yontem_list.append("sozluk")
            continue

        # Eşleşme bulunamadı
        onerilen_list.append(CATEGORY_BILINMIYOR)
        guven_list.append("dusuk")
        yontem_list.append("yok")

    result_df["onerilen_kategori"] = onerilen_list
    result_df["guven"] = guven_list
    result_df["yontem"] = yontem_list

    return result_df


def evaluate_categorizer(df: pd.DataFrame, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Kategorileyicinin başarısını ground_truth verisiyle nesnel olarak değerlendirir.

    Args:
        df: İşlem verileri DataFrame'i.
        ground_truth: Sentetik veri üretiminde saklanan gerçek değerler sözlüğü.

    Returns:
        dict: dogruluk, kapsama, hatali_sayisi ve kategori_bazinda metrikleri.
    """
    if df is None or df.empty or "unknown_categories" not in ground_truth:
        return {
            "toplam_bilinmiyor": 0,
            "onerilen_sayisi": 0,
            "dogru_sayisi": 0,
            "hatali_sayisi": 0,
            "dogruluk": 0.0,
            "kapsama": 0.0,
            "kategori_bazinda": {},
        }

    unknown_map: dict[str, str] = ground_truth["unknown_categories"]
    categorized_df = categorize_transactions(df)

    unknown_mask = categorized_df["kategori"] == CATEGORY_BILINMIYOR
    eval_rows = categorized_df[unknown_mask]

    toplam_bilinmiyor = len(eval_rows)
    onerilen_sayisi = 0
    dogru_sayisi = 0
    hatali_sayisi = 0
    kategori_bazinda: dict[str, dict[str, Any]] = {}

    for _, row in eval_rows.iterrows():
        rk = f"{row['tarih']}|{row['isyeri']}|{row['tutar']:.2f}|{row['hesap']}"
        true_cat = str(unknown_map.get(rk, ""))
        pred_cat = row["onerilen_kategori"]

        if true_cat not in kategori_bazinda:
            kategori_bazinda[true_cat] = {"toplam": 0, "dogru": 0, "dogruluk": 0.0}
        kategori_bazinda[true_cat]["toplam"] += 1

        if pred_cat != CATEGORY_BILINMIYOR:
            onerilen_sayisi += 1
            if pred_cat == true_cat:
                dogru_sayisi += 1
                kategori_bazinda[true_cat]["dogru"] += 1
            else:
                hatali_sayisi += 1

    for cat_data in kategori_bazinda.values():
        if cat_data["toplam"] > 0:
            cat_data["dogruluk"] = round(cat_data["dogru"] / cat_data["toplam"], 4)

    dogruluk = round(dogru_sayisi / onerilen_sayisi, 4) if onerilen_sayisi > 0 else 0.0
    kapsama = round(onerilen_sayisi / toplam_bilinmiyor, 4) if toplam_bilinmiyor > 0 else 0.0

    return {
        "toplam_bilinmiyor": toplam_bilinmiyor,
        "onerilen_sayisi": onerilen_sayisi,
        "dogru_sayisi": dogru_sayisi,
        "hatali_sayisi": hatali_sayisi,
        "dogruluk": dogruluk,
        "kapsama": kapsama,
        "kategori_bazinda": kategori_bazinda,
    }


def categorize_unknown_transactions(df: pd.DataFrame, limit: int = 20) -> dict[str, Any]:
    """Ajan için araç fonksiyonu: Bilinmeyen işlemleri kategoriler ve özet sunar.

    Args:
        df: İşlem verileri DataFrame'i.
        limit: Döndürülecek azami öneri adedi (varsayılan: 20, en fazla 50).

    Returns:
        dict: JSON serileştirilebilir öneri listesi ve istatistik özeti.
    """
    if df is None or df.empty:
        return {
            "bilinmiyor_sayisi": 0,
            "oneriler": [],
            "ozet": "İşlem tablosu boş veya veri bulunamadı.",
        }

    actual_limit = min(max(1, limit), 50)
    categorized_df = categorize_transactions(df)

    unknown_mask = categorized_df["kategori"] == CATEGORY_BILINMIYOR
    unknown_df = categorized_df[unknown_mask]

    toplam_bilinmiyor = len(unknown_df)
    if toplam_bilinmiyor == 0:
        return {
            "bilinmiyor_sayisi": 0,
            "yuksek_guven_sayisi": 0,
            "orta_guven_sayisi": 0,
            "dusuk_guven_sayisi": 0,
            "oneriler": [],
            "ozet": "Kategorisi 'Bilinmiyor' olan hiçbir işlem bulunamadı. Tüm işlemler kategorilendirilmiş.",
        }

    yuksek = int((unknown_df["guven"] == "yuksek").sum())
    orta = int((unknown_df["guven"] == "orta").sum())
    dusuk = int((unknown_df["guven"] == "dusuk").sum())

    sample_rows = unknown_df.head(actual_limit)
    oneriler: list[dict[str, Any]] = []

    for _, r in sample_rows.iterrows():
        oneriler.append({
            "tarih": str(r["tarih"]),
            "isyeri": str(r["isyeri"]),
            "tutar": float(r["tutar"]),
            "aciklama": str(r["aciklama"]),
            "onerilen_kategori": str(r["onerilen_kategori"]),
            "guven": str(r["guven"]),
            "yontem": str(r["yontem"]),
        })

    ozet = (
        f"Toplam {toplam_bilinmiyor} adet bilinmeyen işlem incelendi. "
        f"{yuksek} adedine yüksek güvenle, {orta} adedine orta güvenle kategori önerildi."
    )

    return {
        "bilinmiyor_sayisi": toplam_bilinmiyor,
        "yuksek_guven_sayisi": yuksek,
        "orta_guven_sayisi": orta,
        "dusuk_guven_sayisi": dusuk,
        "oneriler": oneriler,
        "ozet": ozet,
    }
