"""FinAjan tasarruf fırsatları ve bütçe optimizasyonu modülü.

Deterministik saf Python fonksiyonları:
- Dijital aboneliklerin aylık ve yıllık kümülatif maliyetini hesaplar.
- Mükerrer (duplicate) çekimlerden kaynaklanan fazla tahsilatları ve potansiyel iade tutarlarını belirler.
- Son ay harcamalarında geçmiş 3+ ay ortalamasına kıyasla %30'dan fazla artış gösteren değişken kategorileri tespit eder.
- Asla yapay bir 'toplam potansiyel tasarruf' sayısı uydurmaz; bulguları bağımsız kalemler ve öneri diliyle sunar.
"""

from typing import Any
import numpy as np
import pandas as pd

from src.constants import (
    CATEGORY_ABONELIK,
    SAVINGS_ANOMALY_THRESHOLD,
    SAVINGS_EXCLUDED_CATEGORIES,
    SAVINGS_MIN_PRIOR_MONTHS,
)
from src import tools
from src.ui_helpers import format_pct, format_try


def find_savings_opportunities(df: pd.DataFrame) -> dict[str, Any]:
    """İşlem tablosundan deterministik tasarruf ve optimizasyon fırsatlarını çıkarır.

    Args:
        df: Finansal işlem verileri DataFrame'i.

    Returns:
        dict: abonelikler, yinelenen_cekimler ve yuksek_sapma dökümlerini içeren sözlük.
    """
    if df is None or df.empty:
        return {
            "abonelikler": {
                "toplam_aylik_maliyet": 0.0,
                "toplam_yillik_maliyet": 0.0,
                "kalemler": [],
            },
            "yinelenen_cekimler": {
                "toplam_fazla_cekim_tutari": 0.0,
                "kalemler": [],
            },
            "yuksek_sapma": {
                "kalemler": [],
            },
            "ozet": "İşlem tablosu boş veya veri bulunamadı.",
        }

    # 1. Abonelikler Analizi
    rec_df = tools.find_recurring_transactions(df)
    sub_items: list[dict[str, Any]] = []

    if not rec_df.empty:
        sub_mask = rec_df["kategori"].str.lower() == CATEGORY_ABONELIK.lower()
        sub_df = rec_df[sub_mask]

        for _, row in sub_df.iterrows():
            aylik = round(float(row["ortalama_tutar"]), 2)
            yillik = round(aylik * 12.0, 2)
            isyeri = str(row["isyeri"])
            sub_items.append({
                "isyeri": isyeri,
                "kategori": str(row["kategori"]),
                "aylik_tutar": aylik,
                "yillik_maliyet": yillik,
                "yillik_tutar": yillik,
                "islem_adeti": int(row["adet"]),
                "aciklama": (
                    f"{isyeri} düzenli abonelik ödemesi (aylık {format_try(aylik)}, yıllık {format_try(yillik)}); "
                    "aktif kullanım durumunuza göre iptal etmeyi veya daha uygun bir plana geçmeyi değerlendirebilirsiniz."
                ),
            })

    toplam_aylik_sub = round(sum(item["aylik_tutar"] for item in sub_items), 2)
    toplam_yillik_sub = round(toplam_aylik_sub * 12.0, 2)

    # 2. Yinelenen (Duplicate) Çekimler Analizi
    dup_df = tools.find_duplicates(df)
    dup_items: list[dict[str, Any]] = []

    if not dup_df.empty:
        # Aynı gün, işyeri, hesap ve tutara göre grupla
        group_cols = ["tarih", "isyeri", "hesap", "tutar"]
        for _, grp in dup_df.groupby(group_cols):
            adet = len(grp)
            if adet > 1:
                fazla_adet = adet - 1
                tutar = abs(float(grp["tutar"].iloc[0]))
                fazla_tutar = round(fazla_adet * tutar, 2)
                isyeri = str(grp["isyeri"].iloc[0])
                tarih = str(grp["tarih"].iloc[0])
                hesap = str(grp["hesap"].iloc[0])
                aciklama_orj = str(grp["aciklama"].iloc[0])

                dup_items.append({
                    "tarih": tarih,
                    "isyeri": isyeri,
                    "hesap": hesap,
                    "adet": adet,
                    "birim_tutar": tutar,
                    "tekil_tutar": tutar,
                    "fazla_adet": fazla_adet,
                    "fazla_tutar": fazla_tutar,
                    "fazla_cekim_tutari": fazla_tutar,
                    "aciklama": (
                        f"{tarih} tarihinde {isyeri} için aynı tutarda ({format_try(tutar)}) {fazla_adet} adet fazla mükerrer "
                        f"çekim tespit edildi ({format_try(fazla_tutar)} fazla tahsilat). "
                        "Bankanız veya ilgili işyeriyle iletişime geçerek iade talep etmeyi değerlendirebilirsiniz."
                    ),
                })

    toplam_fazla_cekim = round(sum(item["fazla_tutar"] for item in dup_items), 2)

    # 3. Yüksek Kategori Sapması Analizi
    # Son ay harcaması önceki 3+ ay ortalamasından %30 fazla olan değişken kategoriler
    sapma_items: list[dict[str, Any]] = []

    df_clean = df.copy()
    df_clean["ay"] = pd.to_datetime(df_clean["tarih"]).dt.strftime("%Y-%m")

    # Yalnızca harcamalar (tutar < 0) ve hariç tutulan kategoriler dışındakiler
    exp_mask = (df_clean["tutar"] < 0) & (~df_clean["kategori"].isin(SAVINGS_EXCLUDED_CATEGORIES))
    exp_df = df_clean[exp_mask]

    all_months = sorted(df_clean["ay"].unique().tolist())
    if len(all_months) >= (SAVINGS_MIN_PRIOR_MONTHS + 1):
        latest_month = all_months[-1]
        prior_months = all_months[:-1]

        # Her kategori için aylık toplamları hesapla
        for cat, cat_grp in exp_df.groupby("kategori"):
            latest_sum = abs(float(cat_grp[cat_grp["ay"] == latest_month]["tutar"].sum()))

            prior_sums = [
                abs(float(cat_grp[cat_grp["ay"] == m]["tutar"].sum()))
                for m in prior_months
            ]

            if len(prior_sums) >= SAVINGS_MIN_PRIOR_MONTHS:
                prior_avg = float(np.mean(prior_sums))
                if prior_avg > 0:
                    artis_orani = (latest_sum - prior_avg) / prior_avg
                    if artis_orani >= SAVINGS_ANOMALY_THRESHOLD:
                        fark = round(latest_sum - prior_avg, 2)
                        sapma_items.append({
                            "kategori": str(cat),
                            "son_ay": latest_month,
                            "son_ay_tutari": round(latest_sum, 2),
                            "onceki_aylar_ortalamasi": round(prior_avg, 2),
                            "artis_orani": round(artis_orani, 4),
                            "fark_tutari": fark,
                            "aciklama": (
                                f"{cat} kategorisi son ay ({latest_month}) harcamanız ({format_try(latest_sum)}), "
                                f"önceki dönem ortalamasının ({format_try(prior_avg)}) {format_pct(artis_orani)} üzerinde "
                                f"({format_try(fark)} artış). Harcamalarınızı gözden geçirmeyi değerlendirebilirsiniz."
                            ),
                        })

    ozet_lines = [
        f"Tespit edilen fırsatlar: {len(sub_items)} adet düzenli abonelik (aylık {format_try(toplam_aylik_sub)}, yıllık {format_try(toplam_yillik_sub)}), "
        f"{len(dup_items)} adet mükerrer çekim grubu (toplam {format_try(toplam_fazla_cekim)} potansiyel iade) "
        f"ve son ayda ortalamanın üzerinde artan {len(sapma_items)} harcama kategorisi."
    ]

    return {
        "abonelikler": {
            "toplam_aylik_maliyet": toplam_aylik_sub,
            "toplam_yillik_maliyet": toplam_yillik_sub,
            "kalemler": sub_items,
        },
        "yinelenen_cekimler": {
            "toplam_fazla_cekim_tutari": toplam_fazla_cekim,
            "kalemler": dup_items,
        },
        "yuksek_sapma": {
            "kalemler": sapma_items,
        },
        "ozet": " ".join(ozet_lines),
    }


def get_savings_opportunities(df: pd.DataFrame) -> dict[str, Any]:
    """Ajan için araç fonksiyonu: Tasarruf fırsatları analiz sözlüğünü döndürür."""
    return find_savings_opportunities(df)
