# PROJE İLERLEME DURUMU

Son güncelleme: 17 Mayıs 2026

## TAMAMLANANLAR

- [x] Faz 1: EDA — 01_eda.ipynb oluşturuldu (8 adım, 7 grafik, chi-square tablosu)

- [x] Faz 2: Ön İşleme — preprocessing.py oluşturuldu (11 adım, SMOTE uygulandı, train/test kaydedildi)

- [x] Faz 3: Model Eğitimi — model_training.py oluşturuldu ve çalıştırıldı (5 model, top 3 GridSearchCV)

- [ ] Faz 4: Değerlendirme — model_evaluation.py oluşturuldu (12 adım, henüz çalıştırılmadı)

- [x] Faz 5: Segmentasyon — segmentation.py oluşturuldu (12 adım, K-Means k=4, RFM benzeri)

- [x] Faz 6: Streamlit — app.py oluşturuldu (6 sayfa, tahmin formu, model karşılaştırma, segmentasyon)

- [ ] Faz 7: Deployment — Devam ediyor (GitHub push bekleniyor)

## ŞU ANKİ DURUM

Faz 6 tamamlandı. Faz 7 devam ediyor:
- [x] .gitignore oluşturuldu
- [x] requirements.txt Streamlit Cloud için güncellendi (13 kütüphane, sabit versiyonlar)
- [x] README.md finalize edildi (banner, kurulum, sonuçlar, MIT lisansı)
- [ ] GitHub repository oluşturulacak ve push edilecek
- [ ] Streamlit Cloud deployment yapılacak

## FAZ 6 ÇIKTILARI

Dosya: app.py
Yapı: Çok sayfalı Streamlit uygulaması (sidebar navigation)
Sayfalar:
  1. Ana Sayfa — proje özeti, 4 metrik kart, veri seti önizleme
  2. Veri Analizi — 4 sekmeli EDA görselleri (Genel, Demografik, Servisler, Finansal)
  3. Churn Tahmini — 19 değişkenli form, model ile canlı tahmin, risk göstergesi
  4. Model Karşılaştırma — performans tablosu, ROC eğrileri, McNemar testi
  5. Müşteri Segmentleri — 4 segment kartı, profil tablosu, 2D görsel
  6. Hakkında — proje bilgileri, kullanılan algoritmalar, yazar
Bağımlılıklar: requirements.txt oluşturuldu
Çalıştırma: streamlit run app.py

## FAZ 5 ÇIKTILARI

Script: segmentation.py
Yöntem: K-Means Kümeleme (RFM benzeri — tenure, MonthlyCharges, TotalCharges)
Optimal k: 4 (Elbow + Silhouette analizi)
Segmentler:
  - Sadık Yüksek Değerli Müşteriler (yüksek tenure, yüksek TotalCharges)
  - Yeni Risk Altındaki Müşteriler (düşük tenure, yüksek churn)
  - Düşük Aktiviteli Müşteriler (düşük charges)
  - Premium Aktif Müşteriler (yüksek monthly, orta tenure)
Kaydedilen dosyalar:
  - results/figures/15_elbow_method.png
  - results/figures/16_silhouette_scores.png
  - results/figures/17_segments_2d.png
  - results/figures/18_segments_3d.html
  - results/figures/19_segment_churn_rates.png
  - results/tables/segment_profiles.csv
  - results/tables/customers_with_segments.csv
  - models/kmeans_model.pkl
  - models/segmentation_scaler.pkl

## FAZ 2 ÇIKTILARI

Script: preprocessing.py
Kaydedilen dosyalar:
  - results/tables/X_train.csv (SMOTE sonrası eğitim özellikleri)
  - results/tables/X_test.csv (test özellikleri)
  - results/tables/y_train.csv (SMOTE sonrası eğitim etiketleri)
  - results/tables/y_test.csv (test etiketleri)
  - results/tables/feature_names.csv (özellik isimleri)
  - models/scaler.pkl (StandardScaler)

## FAZ 3 ÇIKTILARI

Script: model_training.py
5 model: Logistic Regression, Random Forest, XGBoost, SVM, KNN
En iyi 3 model için GridSearchCV hiperparametre optimizasyonu
En iyi model: Random Forest (Tuned CV F1 = 0.848)
Kaydedilen dosyalar:
  - models/baseline_models.pkl (5 baseline model)
  - models/tuned_models.pkl (3 optimize edilmiş model)
  - models/best_model.pkl (Random Forest)
  - results/tables/model_baseline_results.csv
  - results/tables/model_tuned_results.csv
  - results/tables/best_model_info.csv

## FAZ 4 HAZIRLANAN DOSYA

Script: model_evaluation.py (12 adım)
İçerik:
  - 5 model için kapsamlı metrikler (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, NPV)
  - Confusion matrix görselleştirme (2x3 subplot)
  - ROC eğrileri (5 model tek grafikte)
  - Precision-Recall eğrileri (5 model tek grafikte)
  - McNemar's test (Random Forest vs XGBoost — istatistiksel anlamlılık)
  - Bootstrap %95 güven aralığı (en iyi model F1 için)
  - Feature Importance (top 15 özellik)
  - SHAP analizi (beeswarm + bar plot)
  - Tez özet tablosu (CSV + Markdown)
Çalıştırma: python model_evaluation.py (SHAP nedeniyle 3-5 dakika sürer)

## FAZ 1 ÇIKTILARI

Notebook: notebooks/01_eda.ipynb
Grafikler (results/figures/):
  - 01_eksik_veri.png
  - 02_churn_dagilimi.png
  - 03_sayisal_histogram_kde.png
  - 04_sayisal_boxplot.png
  - 05_korelasyon_heatmap.png
  - 06_kategorik_churn_oranlari.png
  - 07_onemli_kategorikler_yigilik.png
Tablolar (results/tables/):
  - chi_square_sonuclari.csv

## NOTLAR

- TotalCharges sütunu string olarak geliyor; Faz 2'de pd.to_numeric + medyan imputation uygulanacak
- Sınıf dengesizliği: %73.5 No / %26.5 Yes — Faz 2'de SMOTE uygulanacak
- tenure ile TotalCharges arasında yüksek korelasyon bekleniyor — VIF kontrolü Faz 4'te yapılacak
- gender değişkeni Churn ile anlamlı ilişki göstermeyebilir (chi-square p>0.05)