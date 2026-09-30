"""FinAjan projesi için merkezi sabitler ve yapılandırmalar."""

from datetime import date

# Rastgelelik tohumu (determinizm için)
RANDOM_SEED = 42

# Referans Tarih Aralığı (date.today() kullanılmaz, determinizm esastır)
END_DATE = date(2026, 9, 30)
START_DATE = date(2025, 10, 1)

# Veri Sözleşmesi Kolonları
COLUMNS = ["tarih", "tutar", "aciklama", "isyeri", "kategori", "hesap"]

# Standart Hesap Türleri
ACCOUNT_VADESIZ = "Vadesiz Hesap"
ACCOUNT_KREDI_KARTI = "Kredi Kartı"
ACCOUNTS = [ACCOUNT_VADESIZ, ACCOUNT_KREDI_KARTI]

# Standart Kategori Listesi
CATEGORY_GELIR = "Gelir"
CATEGORY_KIRA = "Kira"
CATEGORY_FATURA = "Fatura"
CATEGORY_ABONELIK = "Abonelik"
CATEGORY_MARKET = "Market"
CATEGORY_RESTORAN = "Restoran"
CATEGORY_ULASIM = "Ulaşım"
CATEGORY_GIYIM = "Giyim"
CATEGORY_EGLENCE = "Eğlence"
CATEGORY_SAGLIK = "Sağlık"
CATEGORY_TEKNOLOJI = "Teknoloji"
CATEGORY_DIGER = "Diğer"
CATEGORY_BILINMIYOR = "Bilinmiyor"

CATEGORIES = [
    CATEGORY_GELIR,
    CATEGORY_KIRA,
    CATEGORY_FATURA,
    CATEGORY_ABONELIK,
    CATEGORY_MARKET,
    CATEGORY_RESTORAN,
    CATEGORY_ULASIM,
    CATEGORY_GIYIM,
    CATEGORY_EGLENCE,
    CATEGORY_SAGLIK,
    CATEGORY_TEKNOLOJI,
    CATEGORY_DIGER,
    CATEGORY_BILINMIYOR,
]

# Değişken Gider Kategorileri (günlük harcamalar)
VARIABLE_EXPENSE_CATEGORIES = [
    CATEGORY_MARKET,
    CATEGORY_RESTORAN,
    CATEGORY_ULASIM,
    CATEGORY_GIYIM,
    CATEGORY_EGLENCE,
    CATEGORY_SAGLIK,
    CATEGORY_TEKNOLOJI,
    CATEGORY_DIGER,
]

# Anomali Tespiti Sabitleri
ANOMALY_ESIK = 3.5
ANOMALY_MIN_ORNEK = 8
ANOMALY_HIGH_SCORE = 10.0  # Bu skor ve üzeri 'yuksek' seviye anomali sayılır
# Büyük harcama analizinde hariç tutulan kategoriler
ANOMALY_EXCLUDED_CATEGORIES = [
    CATEGORY_KIRA,
    CATEGORY_FATURA,
    CATEGORY_ABONELIK,
    CATEGORY_GELIR,
    CATEGORY_BILINMIYOR,
]

# Tekrarlayan Harcama Sabitleri
RECURRING_MIN_MONTHS = 3
RECURRING_INTERVAL_MIN = 25
RECURRING_INTERVAL_MAX = 35
RECURRING_MAX_CV = 0.05  # Tutar değişim katsayısı (std / mean) <= %5

# Bilinmiyor Kategori Oranı Aralıkları (Değişken harcamalar üzerinden)
UNKNOWN_RATIO_MIN = 0.13
UNKNOWN_RATIO_MAX = 0.17

# Tasarruf Fırsatları Eşikleri
SAVINGS_ANOMALY_THRESHOLD = 0.30  # Son ay ortalamadan %30 fazla ise
SAVINGS_MIN_PRIOR_MONTHS = 3       # Karşılaştırma için gereken asgari önceki ay sayısı
SAVINGS_EXCLUDED_CATEGORIES = [
    CATEGORY_KIRA,
    CATEGORY_FATURA,
    CATEGORY_ABONELIK,
    CATEGORY_GELIR,
    CATEGORY_BILINMIYOR,
]

