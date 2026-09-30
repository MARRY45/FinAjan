"""FinAjan — Kişisel Finans Asistanı Streamlit Uygulaması."""

from pathlib import Path
import streamlit as st
import pandas as pd

from data.generate_data import generate
from src import charts
from src import tools
from src.agent_factory import PROVIDER_PRESETS, build_agent, default_config_from_env
from src.categorizer import categorize_transactions
from src.constants import CATEGORY_BILINMIYOR
from src.savings import find_savings_opportunities
from src.ui_helpers import format_pct, format_try, month_label


@st.cache_data
def load_data() -> pd.DataFrame:
    """İşlem verilerini önbelleğe alarak yükler; dosya yoksa sentetik veri üretir."""
    csv_path = Path("data/transactions.csv")
    if not csv_path.exists():
        df_gen, _ = generate(seed=42)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_gen.to_csv(csv_path, index=False, encoding="utf-8")
    return tools.load_transactions(csv_path)


def main():
    st.set_page_config(
        page_title="FinAjan — Kişisel Finans Asistanı",
        page_icon="💰",
        layout="wide",
    )

    # 1. Veri Yükleme
    try:
        raw_df = load_data()
        df = raw_df.copy()
    except Exception as e:
        st.error(f"Veri seti yüklenirken bir hata oluştu: {e}")
        st.stop()

    # 2. Yan Panel (Sidebar) — Yapay Zeka Ayarları
    st.sidebar.title("⚙️ Yapay Zeka Ayarları")

    env_cfg = default_config_from_env()
    default_mode_idx = 1 if env_cfg["configured"] else 0

    mode_selection = st.sidebar.radio(
        "Çalışma Modu",
        options=["Demo (Çevrimdışı)", "Kendi API Anahtarım"],
        index=default_mode_idx,
        help="Demo modunda kural tabanlı senaryo oynatıcı kullanılır; internet veya anahtar gerekmez.",
    )

    mode = "demo" if "Demo" in mode_selection else "llm"
    api_key: str | None = None
    base_url: str | None = None
    model_name: str | None = None

    if mode == "llm":
        provider_name = st.sidebar.selectbox(
            "Sağlayıcı Seçin",
            options=list(PROVIDER_PRESETS.keys()),
            index=0,
        )
        preset = PROVIDER_PRESETS[provider_name]

        default_base_url = env_cfg["base_url"] if env_cfg["configured"] else preset["base_url"]
        default_model = env_cfg["model"] if env_cfg["configured"] else preset["default_model"]

        base_url = st.sidebar.text_input("API Base URL", value=default_base_url)
        model_name = st.sidebar.text_input("Model Adı", value=default_model)

        api_key = st.sidebar.text_input(
            "API Anahtarı",
            type="password",
            placeholder="Anahtarınızı buraya girin..." if not env_cfg["has_key"] else "(.env içinde tanımlı)",
            help="Anahtarınız yalnızca mevcut oturum belleğinde tutulur, hiçbir yere kaydedilmez.",
        )
        if not api_key and env_cfg["has_key"]:
            import os
            api_key = os.getenv("LLM_API_KEY")

        st.sidebar.caption("🔒 Güvenlik: API anahtarı oturum bittiğinde silinir.")
    else:
        provider_name = "Demo"
        st.sidebar.info("🧪 Çevrimdışı Demo modu aktif. Gerçek LLM çağrısı yapılmaz, kota harcanmaz.")

    # Oturum ve Ajan Yönetimi
    config_signature = f"{mode}_{provider_name}_{base_url}_{model_name}_{bool(api_key)}"
    if "current_config" not in st.session_state or st.session_state.current_config != config_signature:
        st.session_state.current_config = config_signature
        st.session_state.agent = build_agent(
            mode=mode,
            df=df,
            api_key=api_key,
            base_url=base_url,
            model=model_name,
        )
        st.session_state.messages = []

    agent = st.session_state.agent

    # 3. Ana Başlık
    st.title("💰 FinAjan — Kişisel Finans Asistanı")
    st.caption("Banka ve kredi kartı işlem hareketlerini analiz eden otonom kişisel finans paneli.")

    # 4. Sekmeler (Tabs)
    tab_overview, tab_analysis, tab_anomalies, tab_savings, tab_assistant = st.tabs([
        "📊 Genel Bakış",
        "📈 Harcama Analizi",
        "🚨 Anormallikler",
        "💡 Tasarruf",
        "🤖 AI Asistan",
    ])

    # --- SEKME 1: GENEL BAKIŞ ---
    with tab_overview:
        st.subheader("Finansal Durum Özeti")
        summary = tools.summarize_transactions(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toplam Gelir", format_try(summary["gelir"]))
        with col2:
            st.metric("Toplam Gider", format_try(summary["gider"]))
        with col3:
            net_val = summary["net"]
            st.metric("Net Nakit Akışı", format_try(net_val))
        with col4:
            st.metric("Toplam İşlem", f"{summary['satir_sayisi']} adet")

        st.markdown("---")
        st.plotly_chart(charts.monthly_trend(df), use_container_width=True)

    # --- SEKME 2: HARCAMA ANALİZİ ---
    with tab_analysis:
        st.subheader("Kategori ve Trend Analizi")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.category_bar(df), use_container_width=True)
        with c2:
            st.plotly_chart(charts.income_vs_expense(df), use_container_width=True)

        st.markdown("---")
        st.subheader("Tekil Kategori İnceleme")
        all_cats = [c for c in tools.group_by_category(df, expenses_only=True)["kategori"]]
        selected_cat = st.selectbox("İncelemek İstediğiniz Harcama Kategorisi:", options=all_cats, index=0)
        st.plotly_chart(charts.category_trend(df, selected_cat), use_container_width=True)

    # --- SEKME 3: ANORMALLİKLER ---
    with tab_anomalies:
        st.subheader("Olağandışı Harcamalar ve Anomali Tespiti")
        st.markdown(
            "*İstatistiksel modified z-score yöntemiyle normal harcama alışkanlıklarının "
            "belirgin şekilde dışına çıkan olağandışı harcamalar listelenir.*"
        )
        anomalies_df = tools.detect_large_transactions(df)
        if not anomalies_df.empty:
            anom_view = anomalies_df.copy()
            anom_view["seviye"] = anom_view["skor"].apply(lambda s: "Yüksek" if s >= 10.0 else "Düşük")
            anom_view["tutar"] = anom_view["tutar"].apply(format_try)
            anom_view = anom_view[["tarih", "isyeri", "kategori", "tutar", "skor", "seviye"]]
            anom_view.columns = ["Tarih", "İşyeri", "Kategori", "Tutar", "Sapma Skoru", "Risk Seviyesi"]
            st.dataframe(anom_view, use_container_width=True)
        else:
            st.info("Olağandışı harcama tespit edilmedi.")

        st.markdown("---")
        st.subheader("Mükerrer (Duplicate) Çekim Şüphesi")
        st.markdown(
            "*Aynı gün, aynı işyeri, aynı hesap ve aynı tutarda mükerrer çekilmiş "
            "şüpheli kopya işlemler incelenir.*"
        )
        dup_df = tools.find_duplicates(df)
        if not dup_df.empty:
            dup_view = dup_df[["tarih", "isyeri", "tutar", "hesap", "aciklama"]].copy()
            dup_view["tutar"] = dup_view["tutar"].apply(format_try)
            dup_view.columns = ["Tarih", "İşyeri", "Tutar", "Hesap Türü", "Açıklama"]
            st.dataframe(dup_view, use_container_width=True)
        else:
            st.info("Mükerrer çekilmiş işlem bulunamadı.")

        st.markdown("---")
        st.subheader("Tekrarlayan Sabit Ödemeler ve Abonelikler")
        st.markdown(
            "*Aylık düzenli aralıklarla tekrarlanan sabit faturalar, kira ve dijital "
            "abonelik ödemeleri tespit edilir.*"
        )
        rec_df = tools.find_recurring_transactions(df)
        if not rec_df.empty:
            rec_view = rec_df.copy()
            rec_view["ortalama_tutar"] = rec_view["ortalama_tutar"].apply(format_try)
            rec_view = rec_view[["isyeri", "kategori", "ortalama_tutar", "adet", "medyan_aralik_gun"]]
            rec_view.columns = ["Kurum / İşyeri", "Kategori", "Ortalama Tutar", "İşlem Adeti", "Medyan Aralık (Gün)"]
            st.dataframe(rec_view, use_container_width=True)
        else:
            st.info("Tekrarlayan abonelik veya düzenli ödeme bulunamadı.")

        st.markdown("---")
        st.subheader("Kategori Önerileri (Bilinmeyen İşlemler)")
        st.markdown(
            "*Geçmiş işlem hafızası ve işyeri kural motoruyla kategorisi 'Bilinmiyor' "
            "olan işlemler için üretilen kategori önerileri.*"
        )
        cat_recs = categorize_transactions(df)
        unknown_recs = cat_recs[cat_recs["kategori"] == CATEGORY_BILINMIYOR].copy()
        if not unknown_recs.empty:
            cat_view = unknown_recs[["tarih", "isyeri", "tutar", "onerilen_kategori", "guven", "yontem", "aciklama"]].copy()
            cat_view["tutar"] = cat_view["tutar"].apply(format_try)
            cat_view.columns = ["Tarih", "İşyeri", "Tutar", "Önerilen Kategori", "Güven Seviyesi", "Yöntem", "Açıklama"]
            st.dataframe(cat_view, use_container_width=True)
        else:
            st.info("Kategorisi bilinmeyen işlem bulunamadı.")

    # --- SEKME 4: TASARRUF FIRSATLARI ---
    with tab_savings:
        st.subheader("💡 Tasarruf ve Bütçe İyileştirme Fırsatları")
        st.markdown(
            "*Abonelik maliyetleri, olası mükerrer çekim iadeleri ve geçmiş dönem ortalamasına "
            "göre olağandışı artan harcamalar deterministik olarak analiz edilir.*"
        )
        savings_data = find_savings_opportunities(df)
        subs = savings_data["abonelikler"]
        dups = savings_data["yinelenen_cekimler"]
        devs = savings_data["yuksek_sapma"]["kalemler"]

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.metric("Yıllık Abonelik Maliyeti", format_try(subs["toplam_yillik_maliyet"]))
        with s_col2:
            st.metric("Mükerrer Çekim İade Potansiyeli", format_try(dups["toplam_fazla_cekim_tutari"]))
        with s_col3:
            st.metric("Yüksek Artış Gösteren Kategori", f"{len(devs)} adet")

        st.markdown("---")
        st.subheader("Düzenli Abonelikler ve Yıllık Maliyetleri")
        if subs["kalemler"]:
            subs_df = pd.DataFrame(subs["kalemler"])
            subs_df["aylik_tutar"] = subs_df["aylik_tutar"].apply(format_try)
            subs_df["yillik_maliyet"] = subs_df["yillik_maliyet"].apply(format_try)
            subs_df = subs_df[["isyeri", "kategori", "aylik_tutar", "yillik_maliyet", "aciklama"]]
            subs_df.columns = ["İşyeri / Kurum", "Kategori", "Aylık Tutar", "Yıllık Maliyet", "Detay"]
            st.dataframe(subs_df, use_container_width=True)
        else:
            st.info("Düzenli abonelik tespit edilmedi.")

        st.markdown("---")
        st.subheader("Mükerrer (Duplicate) Çekimler ve İade Potansiyeli")
        if dups["kalemler"]:
            dups_df = pd.DataFrame(dups["kalemler"])
            dups_df["tekil_tutar"] = dups_df["tekil_tutar"].apply(format_try)
            dups_df["fazla_cekim_tutari"] = dups_df["fazla_cekim_tutari"].apply(format_try)
            dups_df = dups_df[["tarih", "isyeri", "adet", "tekil_tutar", "fazla_cekim_tutari", "hesap"]]
            dups_df.columns = ["Tarih", "İşyeri", "Çekim Adeti", "Tekil Tutar", "Fazla Çekim (İade)", "Hesap"]
            st.dataframe(dups_df, use_container_width=True)
        else:
            st.info("Mükerrer çekim tespit edilmedi.")

        st.markdown("---")
        st.subheader("Önceki Döneme Göre Yüksek Artış Gösteren Harcamalar")
        if devs:
            devs_df = pd.DataFrame(devs)
            devs_df["son_ay_tutari"] = devs_df["son_ay_tutari"].apply(format_try)
            devs_df["onceki_aylar_ortalamasi"] = devs_df["onceki_aylar_ortalamasi"].apply(format_try)
            devs_df["artis_orani"] = devs_df["artis_orani"].apply(format_pct)
            devs_df["fark_tutari"] = devs_df["fark_tutari"].apply(format_try)
            devs_df = devs_df[["kategori", "son_ay", "son_ay_tutari", "onceki_aylar_ortalamasi", "artis_orani", "fark_tutari"]]
            devs_df.columns = ["Kategori", "İncelenen Ay", "Bu Ayki Harcama", "Geçmiş Ortalama", "Artış Oranı", "Artış Tutarı"]
            st.dataframe(devs_df, use_container_width=True)
        else:
            st.info("Son ayda önceki dönem ortalamasına göre belirgin artış gösteren kategori bulunamadı.")

    # --- SEKME 5: AI ASİSTAN ---
    with tab_assistant:
        st.subheader("FinAjan Finansal Sohbet Asistanı")

        if mode == "demo":
            st.info(
                "🧪 **Demo Modu Aktif:** Cevaplar çevrimdışı kural tabanlı senaryo oynatıcı tarafından üretilir. "
                "Gerçek yapay zeka için sol panelden kendi API anahtarınızı tanımlayabilirsiniz."
            )

        # Örnek Soru Butonları
        st.write("**Hızlı Sorular:**")
        sample_questions = [
            "Toplam ne kadar harcama yaptım?",
            "Bu ayki harcama durumum nedir?",
            "En çok hangi kategoride harcama yaptım?",
            "Hangi aboneliklerim ve düzenli ödemelerim var?",
            "Şüpheli veya mükerrer işlem var mı?",
            "💡 Nereden tasarruf edebilirim?",
            "🏷️ Bilinmeyenleri kategorilendir",
        ]

        q_cols = st.columns(len(sample_questions))
        selected_prompt = None

        for idx, sq in enumerate(sample_questions):
            with q_cols[idx]:
                if st.button(sq, key=f"sq_btn_{idx}", use_container_width=True):
                    selected_prompt = sq

        # Mesaj Geçmişini Göster
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("steps"):
                    with st.expander("🔍 Adımları göster"):
                        for step in msg["steps"]:
                            icon = "❌" if step["hata_mi"] else "🔧"
                            st.write(f"{icon} **{step['tool']}** ({step['sure_ms']}ms)")
                            st.json(step["input"])
                if msg.get("unverified"):
                    st.warning(f"⚠️ Cevaptaki şu sayılar veri seti araçlarıyla doğrulanamadı: {msg['unverified']}")
                if msg.get("error"):
                    st.error(f"⚠️ Hata: {msg['error']}")

        # Sohbet Girişi
        user_input = st.chat_input("Finansal bir soru sorun...")
        prompt_to_send = selected_prompt or user_input

        if prompt_to_send:
            # Kullanıcı mesajını kaydet ve göster
            st.session_state.messages.append({"role": "user", "content": prompt_to_send})
            with st.chat_message("user"):
                st.markdown(prompt_to_send)

            # Ajanı çalıştır
            with st.chat_message("assistant"):
                with st.spinner("FinAjan verileri analiz ediyor..."):
                    result = agent.ask(prompt_to_send)
                st.markdown(result.answer)

                if result.steps:
                    with st.expander("🔍 Adımları göster"):
                        for step in result.steps:
                            icon = "❌" if step["hata_mi"] else "🔧"
                            st.write(f"{icon} **{step['tool']}** ({step['sure_ms']}ms)")
                            st.json(step["input"])

                if result.unverified_numbers:
                    st.warning(f"⚠️ Cevaptaki şu sayılar veri seti araçlarıyla doğrulanamadı: {result.unverified_numbers}")

                if result.error:
                    st.error(f"⚠️ Hata: {result.error}")

            # Asistan yanıtını geçmişe ekle
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "steps": result.steps,
                "unverified": result.unverified_numbers,
                "error": result.error,
            })


if __name__ == "__main__":
    main()
