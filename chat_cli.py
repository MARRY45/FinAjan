"""FinAjan Terminal Sohbet Arayüzü (CLI).

Kullanıcıların komut satırından finansal sorular sorup yanıt almasını sağlar.
Kullanım:
    python chat_cli.py
"""

import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.agent import FinanceAgent
from src.llm_client import LLMClient
from src.tools import DataLoadError, load_transactions


def main():
    load_dotenv()
    print("=" * 60)
    print("  FinAjan — Kişisel Finans Asistanı (CLI)")
    print("=" * 60)

    # 1. LLM İstemcisini kontrol et
    try:
        llm = LLMClient()
    except Exception as e:
        print(f"\n[HATA] LLM yapılandırma hatası: {e}")
        print("Lütfen projenin kök dizinindeki .env dosyasını kontrol edin.\n")
        sys.exit(1)

    if not llm.is_available():
        print("\n[HATA] LLM API anahtarı veya modeli eksik. Lütfen .env dosyasını kontrol edin.")
        print("Örnek yapılandırma için .env.example dosyasına göz atabilirsiniz.\n")
        sys.exit(1)

    print(f"[BİLGİ] Sağlayıcı: {llm.base_url} | Model: {llm.model}")

    # 2. Veri setini yükle
    csv_path = PROJECT_ROOT / "data" / "transactions.csv"
    try:
        df = load_transactions(csv_path)
        print(f"[BİLGİ] {len(df)} adet işlem verisi yüklendi ({csv_path.name}).")
    except DataLoadError as e:
        print(f"\n[HATA] Veri yüklenemedi: {e}")
        print("Lütfen önce 'python data/generate_data.py' komutunu çalıştırın.\n")
        sys.exit(1)

    # 3. Ajanı ilklendir
    agent = FinanceAgent(llm=llm, df=df)
    print("\nFinAjan hazır! Çıkmak için 'exit', 'quit' veya 'q' yazabilirsiniz.\n")

    while True:
        try:
            user_input = input("FinAjan> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nÇıkış yapılıyor...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Görüşmek üzere!")
            break

        print("\n[FinAjan düşünüyor...]")
        result = agent.ask(user_input)

        # Adımları göster
        if result.steps:
            print("\n--- Kullanılan Araçlar ---")
            for step in result.steps:
                tool_name = step["tool"]
                params = step["input"]
                sure = step["sure_ms"]
                status_icon = "❌" if step["hata_mi"] else "🔧"
                print(f"{status_icon} {tool_name} {params} ({sure}ms)")
            print("---------------------------\n")

        # Yanıtı yazdır
        print(result.answer)

        # Doğrulanamayan sayılar uyarısı
        if result.unverified_numbers:
            print(f"\n⚠️  Uyarı: Cevaptaki şu sayılar araç verileriyle doğrulanamadı: {result.unverified_numbers}")

        if result.error:
            print(f"\n⚠️  Hata Bildirimi: {result.error}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
