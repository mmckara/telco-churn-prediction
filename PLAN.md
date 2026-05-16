# 14 GÜNLÜK PROJE PLANI

## FAZ 1: KURULUM VE VERİ KEŞFİ (Gün 1-2)

**Notebook:** 01_eda.ipynb

- Sanal ortam kurulumu, kütüphane yükleme
- Veri setini yükleme, ilk inceleme (head, info, describe)
- Eksik veri analizi (özellikle TotalCharges)
- Sınıf dağılımı analizi (Churn = Yes/No oranı)
- Tek değişkenli analiz (histogramlar, bar plotlar)
- İki değişkenli analiz (Churn ile her değişkenin ilişkisi)
- Korelasyon analizi (sayısallar için Pearson, kategorikler için Chi-square)
- EDA bulgularının özet tablosu

**Çıktılar:** 8-10 görselleştirme, EDA bulgular dökümü

## FAZ 2: VERİ ÖN İŞLEME (Gün 3-4)

**Notebook:** 02_preprocessing.ipynb

- TotalCharges sütununu düzeltme (boşlukları NaN'a, sayısala çevirme)
- Eksik veri doldurma (median imputation)
- customerID sütununu kaldırma
- Hedef değişkeni encode etme (Churn: Yes=1, No=0)
- Kategorik değişkenleri encode etme (One-Hot Encoding)
- Train/test split (80/20, stratify=y, random_state=42)
- Sayısal değişkenleri ölçekleme (StandardScaler)
- Sınıf dengesizliğini çözme (SMOTE)
- Scaler ve encoder'ı kaydetme (.pkl)

**Çıktılar:** Hazır X_train, X_test, y_train, y_test + kaydedilmiş scaler

## FAZ 3: MODEL EĞİTİMİ (Gün 5-7)

**Notebook:** 03_model_training.ipynb

- 5 model eğitimi:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - K-Nearest Neighbors
- Her model için 5-fold cross-validation
- En iyi 2-3 model için GridSearchCV ile hiperparametre optimizasyonu
- Final modeli .pkl olarak kaydetme

**Çıktılar:** 5 model + kaydedilmiş best_model.pkl

## FAZ 4: MODEL DEĞERLENDİRME VE İSTATİSTİKSEL ANALİZ (Gün 8-9)

**Notebook:** 04_model_evaluation.ipynb

- Her model için: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion matrix görselleştirme
- ROC eğrileri (tek grafikte tüm modeller)
- Precision-Recall eğrileri
- McNemar's test (modeller arası anlamlılık)
- Bootstrap %95 güven aralıkları
- Özellik önemi (Feature Importance) - top 10
- SHAP analizi (en iyi model için)
- Sonuç karşılaştırma tablosu (tezdeki ana tablo)

**Çıktılar:** Tezdeki tüm tablolar ve grafikler

## FAZ 5: MÜŞTERİ SEGMENTASYONU (Gün 10)

**Notebook:** 05_segmentation.ipynb

- K-Means için özellik seçimi (tenure, MonthlyCharges, TotalCharges)
- Elbow method ile optimal k bulma
- Silhouette score analizi
- K-Means uygulama (genelde k=4)
- Segment profilleri çıkarma (her segmentin özellikleri)
- Segmentlerin churn oranları
- Görselleştirme (2D/3D scatter plotlar)

**Çıktılar:** Müşteri segmentleri + her birinin profili

## FAZ 6: STREAMLIT UYGULAMASI (Gün 11-12)

**Dosya:** [app.py](http://app.py)

- Çok sayfalı yapı (sidebar navigation)
- Sayfa 1: Ana Sayfa (proje tanıtımı)
- Sayfa 2: Veri Analizi (EDA görselleri)
- Sayfa 3: Tahmin (form + tahmin + olasılık)
- Sayfa 4: Model Karşılaştırma (tablolar, ROC eğrileri)
- Sayfa 5: Segmentasyon (interaktif grafik)
- Sayfa 6: Hakkında

**Çıktılar:** Çalışan Streamlit uygulaması

## FAZ 7: DEPLOYMENT VE FİNAL (Gün 13-14)

- GitHub repository oluşturma, kodu yükleme
- Streamlit Community Cloud'a deploy etme
- [README.md](http://README.md) final hali (canlı link dahil)
- requirements.txt'i pip freeze ile oluşturma
- Tüm dosyaların son kontrolü
- Tez yazımı için tüm grafikleri /results/figures/ klasörüne export etme