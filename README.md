# 📡 Telekomünikasyon Müşteri Kaybı (Churn) Tahmini ve Segmentasyonu

> **Lisans Bitirme Tezi** — İstatistik Bölümü, 2026

---

## 🖼️ Uygulama Ekran Görüntüsü

[Buraya screenshot eklenecek]

##  Canlı Uygulama

https://telco-churn-prediction-glgke5fgmt8474audr58vs.streamlit.app

---

## 📋 Proje Açıklaması

Bu çalışma, IBM Telco Customer Churn veri seti üzerinde 5 farklı makine öğrenmesi algoritmasını karşılaştırarak müşteri kaybını tahmin etmeyi ve K-Means kümeleme ile müşteri segmentasyonu yapmayı amaçlamaktadır.

**En iyi model:** Random Forest — F1 Score: 0.5739 | Accuracy: %75.4

---

## 📊 Veri Seti

- **Kaynak:** IBM Cognos Analytics Sample Data
- **URL:** [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Boyut:** 7.043 müşteri, 21 değişken
- **Hedef değişken:** `Churn` (Yes = 1 / No = 0) — müşterinin aboneliği iptal edip etmeyeceği
- **Sınıf dağılımı:** %73 No (kalan) / %27 Yes (kaybedilen) — dengesiz veri seti

---

## 🔬 Kullanılan Yöntemler

### Sınıflandırma Algoritmaları (Karşılaştırmalı)

1. **Logistic Regression** — Baseline model
2. **Random Forest Classifier** — En iyi model (GridSearchCV ile optimize edildi)
3. **XGBoost Classifier** — Gradient boosting
4. **Support Vector Machine (SVM)** — Kernel tabanlı sınıflandırma
5. **K-Nearest Neighbors (KNN)** — Örnek tabanlı öğrenme

### Ön İşleme ve Dengeleme

- `TotalCharges` eksik değer temizleme ve tip dönüşümü
- One-Hot Encoding (kategorik değişkenler)
- StandardScaler (sayısal değişkenler)
- **SMOTE** (Synthetic Minority Over-sampling Technique) — sınıf dengesizliği giderme

### Müşteri Segmentasyonu

- **K-Means Clustering** (RFM benzeri: tenure, MonthlyCharges, TotalCharges)
- Optimal küme sayısı: **k = 4** (Elbow + Silhouette analizi)

### İstatistiksel Değerlendirme

- Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity, NPV
- **McNemar's Test** — modeller arası istatistiksel anlamlılık
- **Bootstrap %95 Güven Aralığı** — en iyi model F1 skoru
- **Chi-Square Testi** — kategorik değişkenlerin Churn ile ilişkisi
- **SHAP Değerleri** — model yorumlanabilirliği

---

## 📁 Klasör Yapısı

```
telco-churn-tez/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_segmentation.ipynb
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
├── results/
│   ├── figures/
│   └── tables/
├── app.py
├── requirements.txt
├── README.md
├── PLAN.md
├── PROGRESS.md
└── .cursorrules
```

---

## ⚙️ Kurulum

```powershell
# 1. Projeyi klonla
git clone https://github.com/[KULLANICI_ADIN]/telco-churn-prediction.git
cd telco-churn-prediction

# 2. Sanal ortam oluştur ve aktive et
python -m venv venv
.\venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## ▶️ Streamlit Uygulamasını Çalıştırma

```powershell
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` adresi açılır.

---

## 📈 Sonuçlar Özeti

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | - | - | - | - | - |
| **Random Forest** | **%75.4** | - | - | **0.5739** | - |
| XGBoost | - | - | - | - | - |
| SVM | - | - | - | - | - |
| KNN | - | - | - | - | - |

> Not: Tablo model_evaluation.py çalıştırıldıktan sonra tamamlanacaktır.

**En iyi model:** Random Forest (GridSearchCV ile optimize edildi)
- Test seti F1 Score: **0.5739**
- Test seti Accuracy: **%75.4**
- Sınıf dengesizliği SMOTE ile giderildi

---

## 👤 Yazar

**Mehmet Can Kara**  
İstatistik Bölümü — Lisans Bitirme Tezi  
2026

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
