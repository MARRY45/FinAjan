"""FinAjan Streamlit ekran görüntülerini otomatik kaydeden Playwright betiği."""

from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url: str, timeout: int = 20) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print("[1/5] Streamlit sunucusu başlatılıyor...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        print("[2/5] Sunucu sağlık kontrolü bekleniyor...")
        if not wait_for_server("http://localhost:8501/_stcore/health", timeout=20):
            print("HATA: Streamlit sunucusu zamanında başlatılamadı.")
            return

        print("[3/5] Playwright Chromium başlatılıyor...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            print("[4/5] Sayfaya gidiliyor...")
            page.goto("http://localhost:8501", wait_until="networkidle")
            time.sleep(2)

            # 1. Genel Bakış Ekran Görüntüsü
            print("Ekran görüntüsü alınıyor: 01_genel_bakis.png")
            page.screenshot(path=str(SCREENSHOTS_DIR / "01_genel_bakis.png"), full_page=True)

            # 2. Harcama Analizi
            print("Ekran görüntüsü alınıyor: 02_harcama_analizi.png")
            page.get_by_role("tab", name="Harcama Analizi").click()
            time.sleep(1.5)
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_harcama_analizi.png"), full_page=True)

            # 3. Anormallikler
            print("Ekran görüntüsü alınıyor: 03_anormallikler.png")
            page.get_by_role("tab", name="Anormallikler").click()
            time.sleep(1.5)
            page.screenshot(path=str(SCREENSHOTS_DIR / "03_anormallikler.png"), full_page=True)

            # 4. Tasarruf
            print("Ekran görüntüsü alınıyor: 04_tasarruf.png")
            page.get_by_role("tab", name="Tasarruf").click()
            time.sleep(1.5)
            page.screenshot(path=str(SCREENSHOTS_DIR / "04_tasarruf.png"), full_page=True)

            # 5. AI Asistan
            print("Ekran görüntüsü alınıyor: 05_ai_asistan.png")
            page.get_by_role("tab", name="AI Asistan").click()
            time.sleep(1.5)
            # 1. Örnek soru butonuna tıkla
            buttons = page.get_by_role("button", name="Toplam ne kadar harcama yaptım?")
            if buttons.count() > 0:
                buttons.first.click()
                time.sleep(3)

            page.screenshot(path=str(SCREENSHOTS_DIR / "05_ai_asistan.png"), full_page=True)
            browser.close()

        print("[5/5] Ekran görüntüleri başarıyla kaydedildi -> docs/screenshots/")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Streamlit sunucusu sonlandırıldı.")


if __name__ == "__main__":
    main()
