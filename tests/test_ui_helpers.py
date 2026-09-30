"""FinAjan UI yardımcıları (ui_helpers) test modülü."""

import pytest
from src.ui_helpers import format_pct, format_try, month_label


def test_format_try():
    """format_try fonksiyonunun Türkçe para biçimlendirmesini test eder."""
    assert format_try(42315.5) == "42.315,50 TL"
    assert format_try(42315.50) == "42.315,50 TL"
    assert format_try(0) == "0,00 TL"
    assert format_try(-18500.0) == "-18.500,00 TL"
    assert format_try(-150.25) == "-150,25 TL"
    assert format_try(1234567.89) == "1.234.567,89 TL"
    assert format_try(None) == "0,00 TL"


def test_format_pct():
    """format_pct fonksiyonunun Türkçe yüzde biçimlendirmesini test eder."""
    assert format_pct(0.125) == "%12,5"
    assert format_pct(0.0) == "%0,0"
    assert format_pct(0.0553, decimals=2) == "%5,53"
    assert format_pct(15.2, is_ratio=False) == "%15,2"
    assert format_pct(None) == "%0,0"


def test_month_label():
    """month_label fonksiyonunun YYYY-MM değerlerini Türkçe etiketlere çevirdiğini test eder."""
    assert month_label("2026-09") == "Eyl 2026"
    assert month_label("2025-10") == "Eki 2025"
    assert month_label("2026-01") == "Oca 2026"
    assert month_label("2026-12") == "Ara 2026"
    assert month_label("gecersiz") == "gecersiz"
