"""FinAjan Faz 1 doğrulama ve metrik yazdırma betiği."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.tools import (
    calculate_expenses,
    calculate_income,
    detect_large_transactions,
    find_duplicates,
    find_recurring_transactions,
    group_by_category,
    load_transactions,
    monthly_summary,
    summarize_transactions,
)


def run_verification():
    csv_path = PROJECT_ROOT / "data" / "transactions.csv"
    df = load_transactions(csv_path)

    print("=== 1. İLK 10 SATIR ===")
    print(df.head(10).to_string(index=False))

    print("\n=== 2. TOPLAM SATIR SAYISI ===")
    print(f"Toplam Satır: {len(df)}")

    print("\n=== 3. KATEGORİ DAĞILIMI ===")
    cat_summary = group_by_category(df, expenses_only=False)
    print(cat_summary.to_string(index=False))

    print("\n=== 4. AYLARA GÖRE İŞLEM SAYISI ===")
    m_summary = monthly_summary(df)
    print(m_summary[["ay", "islem_sayisi", "gelir", "gider", "net"]].to_string(index=False))

    print("\n=== 5. GELİR VE GİDER TOPLAMLARI ===")
    gelir = calculate_income(df)
    gider = calculate_expenses(df)
    print(f"Gelir Toplamı : {gelir:,.2f} TL")
    print(f"Gider Toplamı : {gider:,.2f} TL")
    print(f"Net Bakiye    : {(gelir - gider):,.2f} TL")

    print("\n=== 6. DUPLICATE GRUP SAYISI ===")
    dups = find_duplicates(df)
    dup_groups = dups.groupby(["tarih", "isyeri", "tutar", "hesap"]).size()
    print(f"Duplicate Grup Sayısı: {len(dup_groups)} (Toplam yinelenen kayıt: {len(dups)})")
    for idx, count in dup_groups.items():
        print(f"  - {idx[0]} | {idx[1]} | {idx[2]:.2f} TL | {idx[3]}: {count} adet")

    print("\n=== 7. DETECT_LARGE_TRANSACTIONS (BULUNAN İŞLEMLER) ===")
    anomalies = detect_large_transactions(df, esik=3.5, min_ornek=8)
    print(f"Tespit Edilen Büyük Harcama Sayısı: {len(anomalies)}")
    print(anomalies.to_string(index=False))

    print("\n=== 8. BULUNAN ABONELİKLER VE TEKRARLAYAN HARCAMALAR ===")
    recurring = find_recurring_transactions(df)
    print(f"Tespit Edilen Tekrarlayan Harcama Sayısı: {len(recurring)}")
    print(recurring.to_string(index=False))


if __name__ == "__main__":
    run_verification()
