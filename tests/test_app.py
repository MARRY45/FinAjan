"""FinAjan Streamlit arayüzü (app.py) AppTest birim testleri."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from data.generate_data import generate
from src import tools
from src.ui_helpers import format_try

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def dataset():
    df, gt = generate(seed=42)
    return df, gt


def test_app_opens_without_exception():
    """Uygulamanın varsayılan (demo) modda hiçbir istisna fırlatmadan açıldığını test eder."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert len(at.exception) == 0, f"Uygulama açılırken hata verdi: {at.exception}"


def test_app_tabs_structure():
    """Uygulamada tam olarak 5 ana sekmenin bulunduğunu test eder."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert len(at.tabs) == 5
    tab_labels = [t.label for t in at.tabs]
    assert any("Genel Bakış" in lbl for lbl in tab_labels)
    assert any("Harcama Analizi" in lbl for lbl in tab_labels)
    assert any("Anormallikler" in lbl for lbl in tab_labels)
    assert any("Tasarruf" in lbl for lbl in tab_labels)
    assert any("AI Asistan" in lbl for lbl in tab_labels)


def test_app_kpi_metrics_consistency(dataset):
    """Genel bakış sekmesindeki KPI metriklerinin tools.py hesaplamalarıyla tutarlılığını test eder."""
    df, _ = dataset
    summary = tools.summarize_transactions(df)

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert len(at.metric) >= 4

    metrics_dict = {m.label: m.value for m in at.metric}

    expected_gelir = format_try(summary["gelir"])
    expected_gider = format_try(summary["gider"])
    expected_net = format_try(summary["net"])
    expected_islem = f"{summary['satir_sayisi']} adet"

    assert metrics_dict["Toplam Gelir"] == expected_gelir
    assert metrics_dict["Toplam Gider"] == expected_gider
    assert metrics_dict["Net Nakit Akışı"] == expected_net
    assert metrics_dict["Toplam İşlem"] == expected_islem


def test_app_ai_assistant_button_interaction():
    """AI Asistan sekmesinde örnek soru butonuna tıklandığında cevap üretildiğini test eder."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert len(at.button) >= 5

    # 1. Buton: "Toplam ne kadar harcama yaptım?"
    at.button[0].click().run()

    assert len(at.exception) == 0
    # En az bir kullanıcı ve bir asistan mesajı olmalıdır
    assert len(at.chat_message) >= 2

    assistant_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(assistant_msgs) >= 1

    last_assistant_text = assistant_msgs[-1].markdown[0].value
    assert "Demo modu" in last_assistant_text
    assert "573.395,68" in last_assistant_text
