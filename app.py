# =============================================================================
# TELCO CHURN TAHMİNİ - STREAMLİT WEB UYGULAMASI
# İstatistik Bölümü Lisans Tezi
# Yazar: Mehmet Can Kara
# =============================================================================

# Gerekli kütüphaneleri içe aktarıyoruz
import streamlit as st  # Web arayüzü için ana kütüphane
import pandas as pd  # Veri işleme için
import numpy as np  # Sayısal hesaplamalar için
import joblib  # Kaydedilmiş modelleri yüklemek için
import os  # Dosya yolu kontrolü için

# -----------------------------------------------------------------------------
# SAYFA AYARLARI — tarayıcı sekmesinde görünen başlık ve ikon
# layout="wide" sayesinde ekranın tamamını kullanıyoruz
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Tahmini",
    layout="wide",
    page_icon="📊"
)

# -----------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR — tekrar eden işleri fonksiyonlara topluyoruz
# -----------------------------------------------------------------------------

@st.cache_data  # Veriyi bir kez okuyup önbelleğe alıyoruz, her seferinde diskten okumuyoruz
def load_csv(path):
    """CSV dosyasını güvenli şekilde yükler; dosya yoksa None döndürür."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource  # Model gibi büyük nesneleri bir kez yükleyip bellekte tutuyoruz
def load_model(path):
    """Pickle dosyasından model/scaler yükler; dosya yoksa None döndürür."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def show_image(path, caption=""):
    """PNG görselini güvenli şekilde gösterir; dosya yoksa uyarı verir."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        # Dosya henüz oluşturulmamışsa kullanıcıyı bilgilendiriyoruz
        st.info(f"Görsel bulunamadı: {path}")


# -----------------------------------------------------------------------------
# SOL MENÜ (SIDEBAR) — kullanıcının sayfalar arasında geçiş yapması için
# -----------------------------------------------------------------------------
page = st.sidebar.radio("Sayfa", [
    "🏠 Ana Sayfa",
    "📊 Veri Analizi",
    "🤖 Churn Tahmini",
    "📈 Model Karşılaştırma",
    "👥 Müşteri Segmentleri",
    "ℹ️ Hakkında"
])

# Sidebar'ın altına proje bilgisi ekliyoruz
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Telco Churn Tez Projesi**\n\n"
    "İstatistik Bölümü — 2026\n\n"
    "Mehmet Can Kara"
)


# =============================================================================
# SAYFA 1 — ANA SAYFA
# =============================================================================
if page == "🏠 Ana Sayfa":
    st.title("📊 Telco Müşteri Kayıp Tahmini")
    st.markdown(
        "Bu uygulama, IBM Telco Customer Churn veri seti kullanılarak "
        "geliştirilen **makine öğrenmesi modellerini** interaktif olarak "
        "sunar. 7 043 müşterinin demografik, abonelik ve finansal verileri "
        "analiz edilerek kayıp (churn) riski tahmin edilmektedir."
    )

    st.markdown("---")

    # 4 metrik kartı yan yana gösteriyoruz
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Toplam müşteri sayısı — veri setinde 7043 satır var
        st.metric(label="Toplam Müşteri", value="7 043")
    with col2:
        # Churn oranı — 1869/7043 = %26.5
        st.metric(label="Churn Oranı", value="%26.5")
    with col3:
        # En iyi model — model_training.py sonuçlarına göre Random Forest
        st.metric(label="En İyi Model", value="Random Forest")
    with col4:
        # F1 skoru — thesis_summary_table.csv'den Random Forest F1 değeri
        st.metric(label="F1 Skoru", value="0.5739")

    st.markdown("---")
    st.subheader("Veri Setinin İlk 5 Satırı")

    # Ham veriyi yükleyip ilk 5 satırını tabloda gösteriyoruz
    try:
        df_raw = load_csv("data/Telco-Customer-Churn.csv")
        if df_raw is not None:
            st.dataframe(df_raw.head(), use_container_width=True)
        else:
            st.warning("Veri seti bulunamadı: data/Telco-Customer-Churn.csv")
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")


# =============================================================================
# SAYFA 2 — VERİ ANALİZİ
# =============================================================================
elif page == "📊 Veri Analizi":
    st.title("📊 Keşifsel Veri Analizi (EDA)")
    st.markdown(
        "Aşağıdaki sekmeler, veri setinin farklı boyutlarına ait "
        "görselleştirmeleri içerir. Bu grafikler `01_eda.ipynb` "
        "notebook'unda üretilmiştir."
    )

    # 4 sekme oluşturuyoruz — her biri farklı bir analiz kategorisine ait
    tab_genel, tab_demo, tab_servis, tab_finans = st.tabs([
        "Genel Bakış", "Demografik", "Servisler", "Finansal"
    ])

    # ---- Genel Bakış sekmesi ----
    with tab_genel:
        st.subheader("Eksik Veri ve Churn Dağılımı")
        col1, col2 = st.columns(2)
        with col1:
            show_image("results/figures/01_eksik_veri.png", "Eksik Veri Analizi")
        with col2:
            show_image("results/figures/02_churn_dagilimi.png", "Churn Dağılımı")

        st.markdown(
            "**Yorum:** TotalCharges sütununda 11 eksik değer tespit edilmiş "
            "ve medyan ile doldurulmuştur. Churn dağılımı dengesizdir: "
            "müşterilerin %73.5'i kalmış, %26.5'i ayrılmıştır."
        )

    # ---- Demografik sekmesi ----
    with tab_demo:
        st.subheader("Demografik Değişkenler ve Churn")
        show_image(
            "results/figures/07_onemli_kategorikler_yigilik.png",
            "Önemli Kategorik Değişkenler — Yığılık Grafik"
        )
        st.markdown(
            "**Yorum:** Cinsiyet, churn ile anlamlı ilişki göstermezken; "
            "Partner ve Dependents durumu churn oranını etkiler."
        )

    # ---- Servisler sekmesi ----
    with tab_servis:
        st.subheader("Servis Değişkenleri ve Churn")
        show_image(
            "results/figures/06_kategorik_churn_oranlari.png",
            "Kategorik Değişkenler — Churn Oranları"
        )
        st.markdown(
            "**Yorum:** Fiber optik internet kullanan müşterilerde churn oranı "
            "belirgin şekilde yüksektir. Online güvenlik ve teknik destek "
            "hizmeti alan müşterilerin churn oranı daha düşüktür."
        )

    # ---- Finansal sekmesi ----
    with tab_finans:
        st.subheader("Sayısal Değişkenler — Dağılımlar ve Korelasyon")
        col1, col2 = st.columns(2)
        with col1:
            show_image(
                "results/figures/03_sayisal_histogram_kde.png",
                "Sayısal Değişkenler — Histogram ve KDE"
            )
        with col2:
            show_image(
                "results/figures/05_korelasyon_heatmap.png",
                "Korelasyon Isı Haritası"
            )
        st.markdown(
            "**Yorum:** tenure ile TotalCharges arasında güçlü pozitif korelasyon "
            "vardır (beklenen, çünkü uzun süreli müşteriler daha fazla öder). "
            "Aylık ödeme miktarı yüksek olan müşterilerde churn eğilimi artar."
        )


# =============================================================================
# SAYFA 3 — CHURN TAHMİNİ (KRİTİK SAYFA)
# =============================================================================
elif page == "🤖 Churn Tahmini":
    st.title("🤖 Müşteri Churn Tahmini")
    st.markdown(
        "Aşağıdaki formu doldurarak bir müşterinin **kayıp riskini** "
        "tahmin edebilirsiniz. Model, eğitim sırasında öğrendiği örüntülere "
        "göre olasılık üretir."
    )

    # Sayfayı iki kolona bölüyoruz: solda form, sağda sonuç
    col_form, col_result = st.columns([2, 1])

    with col_form:
        st.subheader("Müşteri Bilgileri")

        # Her satırda 3 alan olsun diye kolonlar oluşturuyoruz
        # ---- DEMOGRAFİK BİLGİLER ----
        st.markdown("**Demografik Bilgiler**")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            # Cinsiyet — Male veya Female
            gender = st.selectbox("Cinsiyet", ["Male", "Female"])
        with d2:
            # 65 yaş üstü mü? 0=Hayır, 1=Evet — zaten 0/1 olduğu için encode etmiyoruz
            senior_citizen = st.selectbox("Yaşlı Müşteri (65+)", [0, 1])
        with d3:
            # Eşi/partneri var mı?
            partner = st.selectbox("Partner", ["Yes", "No"])
        with d4:
            # Bağımlısı (çocuk vb.) var mı?
            dependents = st.selectbox("Bağımlı", ["Yes", "No"])

        # ---- ABONELİK BİLGİLERİ ----
        st.markdown("**Abonelik Bilgileri**")
        a1, a2, a3 = st.columns(3)
        with a1:
            # Müşterinin kaç aydır abone olduğu — 0 = 1 aydan az (yeni müşteri)
            tenure = st.slider("Abonelik Süresi (ay)", 0, 72, 12)
        with a2:
            # Sözleşme tipi — aylık olanlar daha çok churn ediyor
            contract = st.selectbox("Sözleşme", [
                "Month-to-month", "One year", "Two year"
            ])
        with a3:
            # Kağıtsız faturalandırma — churn ile ilişkili bulundu
            paperless_billing = st.selectbox("Kağıtsız Fatura", ["Yes", "No"])

        a4, a5 = st.columns(2)
        with a4:
            # Ödeme yöntemi — Electronic check kullananlarda churn yüksek
            payment_method = st.selectbox("Ödeme Yöntemi", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
        with a5:
            pass  # Boş kolon — hizalama için

        # ---- SERVİS BİLGİLERİ ----
        st.markdown("**Servis Bilgileri**")
        s1, s2, s3 = st.columns(3)
        with s1:
            # Telefon hizmeti var mı?
            phone_service = st.selectbox("Telefon Hizmeti", ["Yes", "No"])
        with s2:
            # Birden fazla hat — PhoneService=No ise "No phone service" seçilmeli
            multiple_lines = st.selectbox("Birden Fazla Hat", [
                "Yes", "No", "No phone service"
            ])
        with s3:
            # İnternet hizmet tipi — Fiber optic kullananlarda churn yüksek
            internet_service = st.selectbox("İnternet Hizmeti", [
                "DSL", "Fiber optic", "No"
            ])

        s4, s5, s6 = st.columns(3)
        with s4:
            # Online güvenlik — almayanlarda churn daha yüksek
            online_security = st.selectbox("Online Güvenlik", [
                "Yes", "No", "No internet service"
            ])
        with s5:
            # Online yedekleme hizmeti
            online_backup = st.selectbox("Online Yedekleme", [
                "Yes", "No", "No internet service"
            ])
        with s6:
            # Cihaz koruma planı
            device_protection = st.selectbox("Cihaz Koruması", [
                "Yes", "No", "No internet service"
            ])

        s7, s8, s9 = st.columns(3)
        with s7:
            # Teknik destek — almayanlarda churn daha yüksek
            tech_support = st.selectbox("Teknik Destek", [
                "Yes", "No", "No internet service"
            ])
        with s8:
            # TV yayın hizmeti
            streaming_tv = st.selectbox("Streaming TV", [
                "Yes", "No", "No internet service"
            ])
        with s9:
            # Film yayın hizmeti
            streaming_movies = st.selectbox("Streaming Film", [
                "Yes", "No", "No internet service"
            ])

        # ---- FİNANSAL BİLGİLER ----
        st.markdown("**Finansal Bilgiler**")
        f1, f2 = st.columns(2)
        with f1:
            # Aylık ödeme tutarı (USD) — veri setinde 18.25 ile 118.75 arasında
            monthly_charges = st.number_input(
                "Aylık Ücret ($)", min_value=18.0, max_value=120.0, value=65.0
            )
        with f2:
            # Toplam ödeme tutarı (USD) — tenure * MonthlyCharges civarında olmalı
            total_charges = st.number_input(
                "Toplam Ücret ($)", min_value=0.0, max_value=9000.0, value=1000.0
            )

        # ---- TAHMİN BUTONU ----
        predict_button = st.button("🔮 Tahmin Et", use_container_width=True, type="primary")

    # ---- SONUÇ ALANI ----
    with col_result:
        st.subheader("Tahmin Sonucu")

        if predict_button:
            try:
                # 1) Kaydedilmiş modeli ve scaler'ı yüklüyoruz
                model = load_model("models/best_model.pkl")
                scaler = load_model("models/scaler.pkl")

                # feature_names.csv header=False ile yazıldığı için
                # header=None ile okuyoruz — yoksa ilk satır (SeniorCitizen)
                # sütun başlığı sanılır ve listeden kaybolur
                fn_path = "results/tables/feature_names.csv"
                if os.path.exists(fn_path):
                    feature_names = pd.read_csv(
                        fn_path, header=None
                    )[0].tolist()
                else:
                    feature_names = None

                # Herhangi biri eksikse kullanıcıyı bilgilendiriyoruz
                if model is None or scaler is None or feature_names is None:
                    st.error(
                        "Model dosyaları bulunamadı! Önce model eğitimi "
                        "scriptlerini çalıştırmalısınız:\n\n"
                        "```\npython preprocessing.py\npython model_training.py\n```"
                    )
                else:
                    # 2) Kullanıcı girişlerini bir sözlüğe topluyoruz
                    # SeniorCitizen int olarak kalmalı — model 0/1 integer bekliyor
                    input_data = {
                        "gender": gender,
                        "SeniorCitizen": int(senior_citizen),
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                    }

                    # 3) Sözlüğü tek satırlık DataFrame'e dönüştürüyoruz
                    input_df = pd.DataFrame([input_data])

                    # 4) is_new_customer flag ekliyoruz — tenure 0 ise yeni müşteri
                    # preprocessing.py'de de aynı flag oluşturulmuştu
                    input_df["is_new_customer"] = (input_df["tenure"] == 0).astype(int)

                    # 5) Kategorik değişkenleri one-hot encode ediyoruz
                    # SeniorCitizen zaten 0/1 integer — listede YOK, encode edilmeyecek
                    # is_new_customer da 0/1 flag — o da listede yok
                    # preprocessing.py satır 136-140 ile birebir aynı liste
                    categorical_cols = [
                        "gender", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod",
                    ]
                    input_encoded = pd.get_dummies(
                        input_df, columns=categorical_cols, drop_first=True
                    )

                    # 6) Eğitimde kullanılan tüm sütunların mevcut olmasını sağlıyoruz
                    # Tek bir müşteri formu tüm kategorileri içeremez, eksik sütunları
                    # 0 ile dolduruyoruz (o kategori bu müşteride yok demek)
                    for col in feature_names:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0

                    # 7) Sütun sırasını eğitimdeki sırayla eşleştiriyoruz
                    # Model, sütunları tam bu sırada bekliyor
                    input_encoded = input_encoded[feature_names]

                    # 8) Sayısal sütunları scaler ile ölçeklendiriyoruz
                    # preprocessing.py'deki numerical_cols ile aynı 3 sütun
                    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
                    input_encoded[numerical_cols] = scaler.transform(
                        input_encoded[numerical_cols]
                    )

                    # 9) Model ile tahmin yapıyoruz
                    prediction = model.predict(input_encoded)[0]
                    # predict_proba iki sütun döndürür: [P(No), P(Yes)]
                    proba = model.predict_proba(input_encoded)[0]
                    # Churn olma olasılığı = P(Yes) = ikinci eleman
                    churn_probability = proba[1] * 100

                    # 10) Sonucu görselle gösteriyoruz
                    st.markdown("---")

                    # Risk seviyesini olasılığa göre belirliyoruz
                    if churn_probability < 30:
                        risk_label = "Düşük Risk ✅"
                        risk_color = "green"
                    elif churn_probability < 70:
                        risk_label = "Orta Risk ⚠️"
                        risk_color = "orange"
                    else:
                        risk_label = "Yüksek Risk 🚨"
                        risk_color = "red"

                    # Büyük yazıyla churn olasılığını gösteriyoruz
                    st.markdown(
                        f"<h1 style='text-align:center; color:{risk_color};'>"
                        f"%{churn_probability:.1f}</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h3 style='text-align:center; color:{risk_color};'>"
                        f"{risk_label}</h3>",
                        unsafe_allow_html=True,
                    )

                    # Progress bar ile gauge görünümü oluşturuyoruz
                    st.progress(churn_probability / 100)

                    st.markdown("---")

                    # Tahmin detaylarını gösteriyoruz
                    if prediction == 1:
                        st.error(
                            "**Tahmin: Müşteri AYRILACAK** — Bu müşteri için "
                            "elde tutma stratejileri uygulanmalıdır."
                        )
                    else:
                        st.success(
                            "**Tahmin: Müşteri KALACAK** — Bu müşteri "
                            "mevcut durumda stabil görünmektedir."
                        )

                    # Olasılık dağılımını gösteriyoruz
                    st.markdown(
                        f"- Kalma olasılığı: **%{proba[0]*100:.1f}**\n"
                        f"- Ayrılma olasılığı: **%{proba[1]*100:.1f}**"
                    )

            except Exception as e:
                st.error(f"Tahmin sırasında hata oluştu: {e}")
        else:
            # Kullanıcı henüz butona basmadıysa bilgi mesajı gösteriyoruz
            st.info(
                "Soldaki formu doldurup **Tahmin Et** butonuna basın. "
                "Model, müşterinin churn olasılığını hesaplayacaktır."
            )


# =============================================================================
# SAYFA 4 — MODEL KARŞILAŞTIRMA
# =============================================================================
elif page == "📈 Model Karşılaştırma":
    st.title("📈 Model Karşılaştırma")
    st.markdown(
        "5 farklı makine öğrenmesi modeli eğitilmiş ve değerlendirilmiştir. "
        "Aşağıda tüm modellerin performans metrikleri, ROC eğrileri ve "
        "istatistiksel karşılaştırmaları yer almaktadır."
    )

    # ---- PERFORMANS TABLOSU ----
    st.subheader("Performans Metrikleri")
    try:
        df_summary = load_csv("results/tables/thesis_summary_table.csv")
        if df_summary is not None:
            # En iyi modeli (Random Forest) vurgulamak için stil fonksiyonu
            def highlight_best(row):
                """Random Forest satırını sarı arka planla vurgular."""
                if row["Model"] == "Random Forest":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            # Tabloyu biçimlendirilmiş şekilde gösteriyoruz
            st.dataframe(
                df_summary.style.apply(highlight_best, axis=1).format(
                    {col: "{:.4f}" for col in df_summary.columns if col != "Model"}
                ),
                use_container_width=True,
            )
            st.caption("🏆 Sarı satır: En iyi model (Random Forest)")
        else:
            st.warning("Tablo bulunamadı: results/tables/thesis_summary_table.csv")
    except Exception as e:
        st.error(f"Tablo yüklenirken hata: {e}")

    # ---- DETAYLI METRİKLER ----
    st.subheader("Detaylı Metrikler")
    try:
        df_full = load_csv("results/tables/model_full_metrics.csv")
        if df_full is not None:
            st.dataframe(df_full, use_container_width=True)
        else:
            st.warning("Tablo bulunamadı: results/tables/model_full_metrics.csv")
    except Exception as e:
        st.error(f"Tablo yüklenirken hata: {e}")

    # ---- GÖRSELLEŞTİRMELER ----
    st.subheader("ROC Eğrileri ve Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        show_image("results/figures/09_roc_curves.png", "ROC Eğrileri — 5 Model")
    with col2:
        show_image(
            "results/figures/08_confusion_matrices.png", "Confusion Matrix — 5 Model"
        )

    # ---- McNEMAR TESTİ ----
    st.subheader("İstatistiksel Karşılaştırma — McNemar Testi")
    try:
        df_mcnemar = load_csv("results/tables/mcnemar_test_results.csv")
        if df_mcnemar is not None:
            st.dataframe(df_mcnemar, use_container_width=True)
            st.markdown(
                "**Sonuç:** Random Forest ile XGBoost arasındaki performans farkı "
                "istatistiksel olarak **anlamlı değildir** (p = 1.0 ≥ α = 0.05). "
                "Yani iki modelin tahmin gücü arasında anlamlı bir fark yoktur."
            )
        else:
            st.warning("Tablo bulunamadı: results/tables/mcnemar_test_results.csv")
    except Exception as e:
        st.error(f"Tablo yüklenirken hata: {e}")


# =============================================================================
# SAYFA 5 — MÜŞTERİ SEGMENTLERİ
# =============================================================================
elif page == "👥 Müşteri Segmentleri":
    st.title("👥 Müşteri Segmentasyonu")
    st.markdown(
        "K-Means kümeleme algoritması ile müşteriler 4 segmente ayrılmıştır. "
        "Segmentleme için **tenure**, **MonthlyCharges** ve **TotalCharges** "
        "değişkenleri kullanılmıştır (RFM benzeri yaklaşım)."
    )

    # ---- SEGMENT PROFİLLERİ ----
    try:
        df_segments = load_csv("results/tables/segment_profiles.csv")
        if df_segments is not None:
            # Segment isimlerini daha anlaşılır hale getiriyoruz
            segment_names = {
                0: "Sadık Yüksek Değerli",
                1: "Premium Aktif",
                2: "Düşük Aktiviteli",
                3: "Yeni Risk Altında",
            }

            # Her segment için bir kart oluşturuyoruz
            st.subheader("Segment Profilleri")
            cols = st.columns(4)

            for idx, row in df_segments.iterrows():
                seg_id = int(row["Segment"])
                seg_name = segment_names.get(seg_id, f"Segment {seg_id}")
                churn_rate = row["Churn Oranı (%)"]
                customer_count = int(row["Müşteri Sayısı"])

                # Churn oranına göre renk belirliyoruz
                if churn_rate < 15:
                    color = "🟢"  # Düşük churn — yeşil
                elif churn_rate < 30:
                    color = "🟡"  # Orta churn — sarı
                else:
                    color = "🔴"  # Yüksek churn — kırmızı

                with cols[idx % 4]:
                    st.markdown(
                        f"### {color} {seg_name}\n\n"
                        f"- **Müşteri:** {customer_count:,}\n"
                        f"- **Churn:** %{churn_rate:.1f}\n"
                        f"- **Ort. Tenure:** {row['Ort. Tenure (ay)']:.0f} ay\n"
                        f"- **Ort. Aylık:** ${row['Ort. MonthlyCharges ($)']:.0f}\n"
                        f"- **Sözleşme:** {row['En Sık Contract']}"
                    )

            st.markdown("---")

            # Segment profil tablosunu tam olarak gösteriyoruz
            st.subheader("Segment Detay Tablosu")
            st.dataframe(df_segments, use_container_width=True)
        else:
            st.warning("Tablo bulunamadı: results/tables/segment_profiles.csv")
    except Exception as e:
        st.error(f"Segment verileri yüklenirken hata: {e}")

    # ---- GÖRSELLEŞTIRME ----
    st.subheader("Segment Dağılımı (2D)")
    show_image("results/figures/17_segments_2d.png", "K-Means — 2D Segment Dağılımı")


# =============================================================================
# SAYFA 6 — HAKKINDA
# =============================================================================
elif page == "ℹ️ Hakkında":
    st.title("ℹ️ Proje Hakkında")

    st.markdown(
        """
        ## Proje Amacı

        Bu proje, bir telekomünikasyon şirketinin müşteri kayıp (churn) 
        problemini **makine öğrenmesi** yöntemleriyle analiz etmeyi ve 
        tahmin etmeyi amaçlamaktadır. İstatistiksel testler ve model 
        karşılaştırmaları ile akademik bir çerçeve sunulmaktadır.

        ## Veri Seti

        - **Kaynak:** IBM Telco Customer Churn
        - **Boyut:** 7 043 müşteri, 21 değişken
        - **Hedef:** Churn (Yes / No) — ikili sınıflandırma
        - **Dengesizlik:** %73.5 No / %26.5 Yes → SMOTE ile dengelenmiştir

        ## Kullanılan Algoritmalar

        5 farklı makine öğrenmesi modeli eğitilmiş ve karşılaştırılmıştır:

        | # | Model | Açıklama |
        |---|-------|----------|
        | 1 | Logistic Regression | Temel (baseline) model |
        | 2 | Random Forest | Ağaç tabanlı topluluk modeli — **en iyi model** |
        | 3 | XGBoost | Gradient boosting modeli |
        | 4 | SVM | Destek vektör makinesi |
        | 5 | KNN | K-en yakın komşu |

        ## İstatistiksel Analizler

        - **Chi-square testi:** Kategorik değişkenlerin churn ile ilişkisi
        - **Pearson korelasyonu:** Sayısal değişkenler arası ilişki
        - **McNemar testi:** Modeller arası performans farkının anlamlılığı
        - **Bootstrap:** %95 güven aralığı hesaplama

        ## Müşteri Segmentasyonu

        - **Yöntem:** K-Means kümeleme (k=4)
        - **Değişkenler:** tenure, MonthlyCharges, TotalCharges (RFM benzeri)
        - **Doğrulama:** Elbow method + Silhouette score

        ---

        ## Yazar

        **Mehmet Can Kara**  
        İstatistik Bölümü — Lisans Bitirme Tezi  
        2026

        ---

        *Bu uygulama Streamlit ile geliştirilmiştir.*
        """
    )
