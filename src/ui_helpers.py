"""FinAjan kullanıcı arayüzü biçimlendirme ve yardımcı işlevler modülü."""

from typing import Union

TURKISH_SHORT_MONTHS = {
    "01": "Oca",
    "02": "Şub",
    "03": "Mar",
    "04": "Nis",
    "05": "May",
    "06": "Haz",
    "07": "Tem",
    "08": "Ağu",
    "09": "Eyl",
    "10": "Eki",
    "11": "Kas",
    "12": "Ara",
}


def format_try(x: Union[float, int, None]) -> str:
    """Sayısal tutarı Türkçe para formatına (42.315,50 TL) dönüştürür.

    Args:
        x: Biçimlendirilecek tutar (None ise '0,00 TL' döner).

    Returns:
        str: Türkçe binlik ayracı (nokta), ondalık ayracı (virgül) ve ' TL' soneki.
    """
    if x is None:
        return "0,00 TL"

    try:
        val = float(x)
    except (ValueError, TypeError):
        return "0,00 TL"

    is_neg = val < 0
    abs_val = abs(val)

    # US formatında 2 ondalıklı üretip noktalı ve virgüllü yerleri değiştir
    us_formatted = f"{abs_val:,.2f}"
    tr_formatted = us_formatted.replace(",", "X").replace(".", ",").replace("X", ".")

    prefix = "-" if is_neg else ""
    return f"{prefix}{tr_formatted} TL"


def format_pct(x: Union[float, int, None], decimals: int = 1, is_ratio: bool = True) -> str:
    """Yüzde değerini Türkçe formatına (%12,5) dönüştürür.

    Args:
        x: Yüzde değeri (oran veya 100 bazlı).
        decimals: Ondalık basamak sayısı (varsayılan: 1).
        is_ratio: True ise değeri 100 ile çarpar (0.125 -> %12,5).

    Returns:
        str: Türkçe '%' öneki ve virgüllü ondalık ayracı.
    """
    if x is None:
        return f"%0,{ '0' * decimals }"

    try:
        val = float(x)
    except (ValueError, TypeError):
        return f"%0,{ '0' * decimals }"

    if is_ratio:
        val = val * 100

    formatted = f"{val:.{decimals}f}".replace(".", ",")
    return f"%{formatted}"


def month_label(month_str: str) -> str:
    """YYYY-MM formatındaki ay bilgisini Türkçe kısa ay adına çevirir (örn. '2026-09' -> 'Eyl 2026').

    Args:
        month_str: YYYY-MM biçiminde tarih dizesi.

    Returns:
        str: Türkçe ay ve yıl etiketi (geçersiz biçimde orijinal dizeyi döner).
    """
    if not isinstance(month_str, str) or "-" not in month_str:
        return str(month_str)

    parts = month_str.strip().split("-")
    if len(parts) >= 2:
        year, month = parts[0], parts[1]
        month_name = TURKISH_SHORT_MONTHS.get(month)
        if month_name:
            return f"{month_name} {year}"

    return month_str
