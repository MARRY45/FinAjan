"""FinAjan çevrimdışı kural tabanlı Demo LLM modülü.

Bu modül gerçek bir dil modeli değildir; API anahtarı veya internet bağlantısı
olmadan uygulamanın tüm araç döngüsünü ve UI yeteneklerini sergileyen
deterministik bir senaryo yürütücüsüdür.
LLMClient ile birebir aynı arayüze (is_available, create) sahiptir.
"""

import json
from typing import Any, Sequence

from src.llm_client import LLMResponse
from src.ui_helpers import format_pct, format_try


class DemoLLM:
    """Çevrimdışı senaryoları oynatan sahte/kural tabanlı LLM istemcisi."""

    def __init__(self, *args, **kwargs):
        pass

    def is_available(self) -> bool:
        """Demo modu her zaman kullanılabilir durumdadır."""
        return True

    def create(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        """Kullanıcı mesajına göre uygun araç çağrısını tetikler veya araç çıktısını şablonlar.

        Args:
            messages: Mesaj geçmişi listesi.
            system: Sistem istemi (prompt).
            tools: Araç tanımları listesi.
            max_tokens: Azami token sayısı.

        Returns:
            LLMResponse: Ajan döngüsüyle uyumlu yanıt nesnesi.
        """
        # 1. Aşama Kontrolü: Geçmişte çalıştırılmış araç sonuçları ('role': 'tool') var mı?
        tool_messages = [m for m in messages if m.get("role") == "tool"]

        if tool_messages:
            # Araçlar çalıştırılmış; sonuçları toplayıp doğrulanmış şablon yanıtı üret
            return self._build_templated_response(tool_messages)

        # 2. Aşama: İlk tur. Kullanıcı sorusunu analiz et ve uygun araçları çağır
        user_messages = [m for m in messages if m.get("role") == "user"]
        query = user_messages[-1]["content"].lower() if user_messages else ""

        return self._route_intent(query)

    def _route_intent(self, q: str) -> LLMResponse:
        """Kullanıcı sorgusundan niyet çıkarıp ilgili araç çağrılarını döndürür."""
        # Niyet 1: Bu ay / aylık özet
        if any(k in q for k in ["bu ay", "son ay", "eylül", "aylık durum", "bu ayki", "bu ayın"]):
            tool_calls = [{
                "id": "demo_month",
                "name": "get_summary",
                "arguments": {"start_date": "2026-09-01", "end_date": "2026-09-30"},
            }]
            return self._make_tool_call_response(tool_calls)

        # Niyet 2: Tasarruf fırsatları ve bütçe optimizasyonu
        if any(k in q for k in ["tasarruf", "bütçe kısma", "harcama optimizasyonu", "nereden kısabilirim", "fırsat"]):
            tool_calls = [{
                "id": "demo_savings",
                "name": "get_savings_opportunities",
                "arguments": {},
            }]
            return self._make_tool_call_response(tool_calls)

        # Niyet 3: Bilinmeyen işlemleri kategorileme
        if any(k in q for k in ["kategorilendir", "kategori öner", "bilinmeyen", "bilinmiyor"]):
            tool_calls = [{
                "id": "demo_categorize",
                "name": "categorize_unknown_transactions",
                "arguments": {"limit": 10},
            }]
            return self._make_tool_call_response(tool_calls)

        # Niyet 4: Abonelikler ve düzenli ödemeler
        if any(k in q for k in ["abonelik", "düzenli ödeme", "sabit ödeme", "fatura", "netflix", "spotify"]):
            tool_calls = [{
                "id": "demo_sub",
                "name": "get_subscriptions",
                "arguments": {},
            }]
            return self._make_tool_call_response(tool_calls)

        # Niyet 5: Şüpheli, anomali veya mükerrer işlemler
        if any(k in q for k in ["şüpheli", "mükerrer", "duplicate", "anormal", "büyük işlem", "anomali", "olağandışı"]):
            tool_calls = [
                {"id": "demo_dup", "name": "find_duplicates", "arguments": {}},
                {"id": "demo_large", "name": "detect_large_transactions", "arguments": {}},
            ]
            return self._make_tool_call_response(tool_calls)

        # Niyet 6: Kategori dağılımı / en çok harcama yapılan yer
        if any(k in q for k in ["kategori", "en çok", "dağılım", "nereye", "hangi kategori", "harcama dağılımı"]):
            tool_calls = [{
                "id": "demo_cat",
                "name": "get_category_summary",
                "arguments": {},
            }]
            return self._make_tool_call_response(tool_calls)

        # Niyet 7: Genel bakiye, toplam gelir, toplam gider
        if any(k in q for k in ["toplam", "gelir", "gider", "bakiye", "özet", "harcama", "finansal durum", "hesap durumu"]):
            tool_calls = [{
                "id": "demo_sum",
                "name": "get_summary",
                "arguments": {},
            }]
            return self._make_tool_call_response(tool_calls)

        # Niyet dışı / Anlaşılmayan soru
        fallback_text = (
            "🧪 Demo modu (LLM yok)\n\n"
            "Sorunuzu tam olarak eşleştiremedim. Demo modunda şu örnek soruları sorabilirsiniz:\n\n"
            "1. 'Toplam ne kadar harcama yaptım?' (Genel bakiye ve harcama özeti)\n"
            "2. 'Bu ayki harcama durumum nedir?' (Güncel referans ayı özeti)\n"
            "3. 'En çok hangi kategoride harcama yaptım?' (Kategori dağılımı)\n"
            "4. 'Hangi aboneliklerim ve düzenli ödemelerim var?' (Abonelik tespiti)\n"
            "5. 'Şüpheli veya mükerrer işlem var mı?' (Anomali ve kopya harcama tespiti)\n"
            "6. 'Nereden tasarruf edebilirim?' (Tasarruf ve bütçe optimizasyonu fırsatları)\n"
            "7. 'Bilinmeyen işlemleri kategorilendir' (İşyeri kural motoruyla kategori önerileri)\n\n"
            "💡 Gerçek yapay zeka deneyimi için sol panelden kendi API anahtarınızı tanımlayabilirsiniz."
        )
        return LLMResponse(
            text=fallback_text,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            message={"role": "assistant", "content": fallback_text},
        )

    def _make_tool_call_response(self, tool_calls: list[dict[str, Any]]) -> LLMResponse:
        """Araç çağrısı üreten LLMResponse nesnesi oluşturur."""
        sanitized_tcs = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ]
        return LLMResponse(
            text="",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
            usage={"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            message={"role": "assistant", "content": "", "tool_calls": sanitized_tcs},
        )

    def _build_templated_response(self, tool_messages: list[dict[str, Any]]) -> LLMResponse:
        """Araç çıktılarından halüsinasyonsuz, doğrulanmış Türkçe yanıt metni üretir."""
        parsed_outputs: list[dict[str, Any]] = []
        for tm in tool_messages:
            try:
                parsed_outputs.append(json.loads(tm.get("content", "{}")))
            except Exception:
                continue

        lines: list[str] = ["🧪 Demo modu (LLM yok)\n"]

        for data in parsed_outputs:
            # 1. Tasarruf fırsatları çıktısı
            if "abonelikler" in data and "yinelenen_cekimler" in data and "yuksek_sapma" in data:
                lines.append("💡 Tespit Edilen Tasarruf Fırsatları:\n")
                subs = data.get("abonelikler", {})
                dups = data.get("yinelenen_cekimler", {})
                devs = data.get("yuksek_sapma", {}).get("kalemler", [])

                sub_items = subs.get("kalemler", [])
                lines.append(f"• Düzenli Abonelikler ({len(sub_items)} adet): Aylık {format_try(subs.get('toplam_aylik_maliyet', 0.0))}, Yıllık {format_try(subs.get('toplam_yillik_maliyet', 0.0))}")
                for s in sub_items[:3]:
                    lines.append(f"  - {s['isyeri']} ({s['kategori']}): {format_try(s['aylik_tutar'])} / ay (Yıllık: {format_try(s['yillik_maliyet'])})")

                dup_items = dups.get("kalemler", [])
                lines.append(f"\n• Mükerrer Çekim İadesi: {len(dup_items)} işlem grubunda toplam {format_try(dups.get('toplam_fazla_cekim_tutari', 0.0))} potansiyel iade imkanı")
                for d in dup_items:
                    lines.append(f"  - {d['tarih']} {d['isyeri']}: {format_try(d['fazla_cekim_tutari'])} fazla çekim ({d['hesap']})")

                if devs:
                    lines.append(f"\n• Yüksek Harcama Artışı ({len(devs)} kategori):")
                    for dev in devs:
                        lines.append(f"  - {dev['kategori']}: Son ay {format_try(dev['son_ay_tutari'])} (Önceki aylar ortalaması: {format_try(dev['onceki_aylar_ortalamasi'])})")

            # 2. Kategori önerileri çıktısı
            elif "bilinmiyor_sayisi" in data and "oneriler" in data:
                b_sayisi = data.get("bilinmiyor_sayisi", 0)
                y_sayisi = data.get("yuksek_guven_sayisi", 0)
                lines.append("🏷️ Bilinmeyen İşlemler İçin Kategori Önerileri:\n")
                lines.append(f"• Toplam {b_sayisi} adet bilinmeyen işlemden {y_sayisi} adedine yüksek güvenle kategori önerildi.")
                oneriler = data.get("oneriler", [])[:5]
                if oneriler:
                    lines.append("\nÖrnek öneriler:")
                    for oneri in oneriler:
                        lines.append(f"• {oneri['isyeri']} ({format_try(oneri['tutar'])}) -> {oneri['onerilen_kategori']} [Güven: {oneri['guven']}, Yöntem: {oneri['yontem']}]")

            # 3. Abonelikler çıktısı
            elif "duzenli_islemler" in data:
                items = data.get("duzenli_islemler", [])
                lines.append(f"Tespit edilen düzenli ödeme ve abonelikleriniz ({len(items)} adet):")
                for it in items:
                    lines.append(f"• {it['isyeri']} ({it['kategori']}): {format_try(it['ortalama_tutar'])}")

            # 2. Mükerrer çekimler çıktısı
            elif "duplicate_grup_sayisi" in data:
                groups = data.get("gruplar", [])
                lines.append(f"Mükerrer (Duplicate) Şüpheli İşlemler ({len(groups)} grup):")
                if not groups:
                    lines.append("• Mükerrer işlem tespit edilmedi.")
                for g in groups:
                    lines.append(f"• {g['tarih']} - {g['isyeri']}: {format_try(g['tutar'])} ({g['hesap']})")

            # 3. Anomali çıktıları
            elif "anomali_sayisi" in data:
                anomalies = [a for a in data.get("anomaliler", []) if a.get("seviye") == "yuksek"]
                lines.append(f"\nYüksek Seviye Olağandışı Harcamalar ({len(anomalies)} adet):")
                if not anomalies:
                    lines.append("• Yüksek riskli anomali bulunamadı.")
                for a in anomalies:
                    lines.append(f"• {a['tarih']} - {a['isyeri']} ({a['kategori']}): {format_try(a['tutar'])}")

            # 4. Kategori dağılımı çıktısı
            elif "kategori_dagilimi" in data:
                cats = data.get("kategori_dagilimi", [])
                total_exp = data.get("gider_toplami", 0.0)
                lines.append(f"Kategorilere Göre Harcama Dağılımı (Toplam: {format_try(total_exp)}):")
                if cats:
                    top_c = cats[0]
                    lines.append(f"En çok harcama yapılan kategori: {top_c['kategori']} ({format_try(top_c['toplam'])})")
                    lines.append("\nÖne çıkan kategoriler:")
                    for c in cats[:5]:
                        lines.append(f"• {c['kategori']}: {format_try(c['toplam'])}")

            # 5. Genel veya Aylık Özet çıktısı
            elif "toplam_gelir" in data and "toplam_gider" in data:
                t_range = data.get("tarih_araligi", "İncelenen dönem")
                gelir = data.get("toplam_gelir")
                gider = data.get("toplam_gider")
                net = data.get("net_bakiye")
                adet = data.get("islem_sayisi")
                lines.append(f"{t_range} Dönemi Finansal Özeti:")
                lines.append(f"• Toplam Gelir: {format_try(gelir)}")
                lines.append(f"• Toplam Gider: {format_try(gider)}")
                lines.append(f"• Net Bakiye: {format_try(net)}")
                lines.append(f"• Toplam İşlem Adeti: {adet}")

        answer = "\n".join(lines).strip()
        return LLMResponse(
            text=answer,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60},
            message={"role": "assistant", "content": answer},
        )
