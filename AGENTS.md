# AGENTS.md — FinAjan Geliştirme Kuralları ve Prensipleri

Bu belge, FinAjan projesinde görev alacak tüm geliştirici ve ajanlar için bağlayıcı kuralları içerir.

## 1. Çalışma Prensipleri ve Otonomi
- **Tam Otonomi:** Kullanıcıya onay sormadan tüm görev baştan sona kendi kararınla tamamlanır.
- **Karar Alma:** Belirsiz noktalarda en makul ve sağlam seçenek tercih edilir; ne seçildiği görev sonu raporunda açıkça belirtilir.
- **İzin İstememe:** Dosya oluşturma/düzenleme, paket kurma, terminal komutu çalıştırma, test yazma ve çalıştırma için izin istenmez.
- **Hata Yönetimi (Döngü Sınırı):** Bir adım hata verirse durulmaz; kök neden bulunur, düzeltilir ve tekrar denenir (en fazla 3 deneme). Sorun çözülmezse hata ve denenen adımlar raporlanıp kalan işlere devam edilir.
- **Gerçek Çıktı Kanıtı:** Her faz ve görev sonunda raporlar, varsayımlara değil doğrudan gerçek komut ve test çıktılarına dayandırılır.

## 2. Mimari ve Kodlama Standartları
- **Model Çağrı İzolasyonu:** Model çağrıları yalnızca `src/llm_client.py` içinde yapılır; projenin başka hiçbir yerinde doğrudan LLM çağrısı bulunamaz.
- **Hesaplama Güvenilirliği:** LLM sayı üretmez ve matematiksel/finansal hesaplama yapmaz; tüm hesaplamalar `src/tools.py` içindeki deterministik saf Python fonksiyonları tarafından gerçekleştirilir.
- **Saf Fonksiyonlar (`src/tools.py`):** Streamlit, LLM veya dosya sistemine bağımlı olamaz (yalnızca `load_transactions` dosya okur). Girdi DataFrame'lerini mutasyona uğratmaz (`copy` kullanılır). Type hint ve Türkçe docstring zorunludur. Hata sınıfı `DataLoadError` olup iç yol veya hassas stack trace sızdırmaz.
- **Windows / UTF-8 Uyumluluğu:** Tüm dosya okuma ve yazma işlemlerinde `encoding="utf-8"` parametresi açıkça verilir. PowerShell oturumlarında `$env:PYTHONUTF8="1"` kullanılır.

## 3. Güvenlik ve Sınırlar
- **Dizin Sınırı:** Yalnızca proje klasörü (`finajan`) sınırları içinde çalışılır; dışındaki dosyalara müdahale edilmez.
- **Secret İzolasyonu:** API anahtarları asla koda, dokümanlara, loglara veya commit geçmişine yazılmaz; yalnızca `.env` dosyasında tutulur. Anahtar hiçbir çıktıya yazılmaz; .env dosyası içeriği gösterilmez.
- **Model Standardı:** LLM çağrıları OpenAI uyumlu API formatında (`openai` SDK) ve yalnızca `src/llm_client.py` üzerinden izole olarak yürütülür.
- **Git ve Dosya Güvenliği:** `git push --force` ve geri alınamaz toplu silme işlemleri kesinlikle yasaktır.
- **Sentetik Veri Şeffaflığı:** Proje sentetik veri ile çalıştığından `data/*.csv` gitignore edilmez; repoyu klonlayan herkes demoyu hemen çalıştırabilmelidir. `.agent/` dizini repoda korunur.
