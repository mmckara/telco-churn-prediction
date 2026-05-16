# =============================================================================
# FAZ 2: VERİ ÖN İŞLEME (PREPROCESSING)
# Proje: Telekomünikasyon Müşteri Kaybı (Churn) Tahmini
# Çalıştırma: python preprocessing.py  (venv aktif olmalı)
# =============================================================================


# ── ADIM 1: KÜTÜPHANELERİ IMPORT ET ─────────────────────────────────────────

# pandas: veriyi okumak ve tablo işlemleri için
import pandas as pd

# numpy: sayısal hesaplamalar için
import numpy as np

# train_test_split: veriyi eğitim ve test olarak bölmek için
from sklearn.model_selection import train_test_split

# StandardScaler: sayısal değişkenleri standartlaştırmak için (ortalama=0, std=1)
# Scaler train setinde fit edilecek, test setinde sadece transform uygulanacak
from sklearn.preprocessing import StandardScaler

# SMOTE: azınlık sınıfını (Churn=Yes) sentetik örneklerle çoğaltmak için
# Sadece train setine uygulanacak — test setine ASLA dokunmayacak
from imblearn.over_sampling import SMOTE

# joblib: eğitilmiş scaler ve modelleri .pkl dosyasına kaydetmek için
import joblib

# os: klasör oluşturma işlemleri için
import os

# Kaydedilecek klasörleri tanımlıyoruz
MODELS_DIR = 'models/'
TABLES_DIR = 'results/tables/'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 60)
print("ADIM 1 TAMAMLANDI: Kütüphaneler yüklendi.")
print("=" * 60)


# ── ADIM 2: VERİYİ YÜKLE ─────────────────────────────────────────────────────

# CSV dosyasını pandas DataFrame olarak okuyoruz
# preprocessing.py proje kökünde, data/ klasörü de kök dizinde
df = pd.read_csv('data/Telco-Customer-Churn.csv')

print("\n" + "=" * 60)
print("ADIM 2: VERİ YÜKLEME")
print("=" * 60)
print(f"Veri seti boyutu (satır, sütun): {df.shape}")
print(f"Beklenen                        : (7043, 21)")


# ── ADIM 3: TOTALCHARGES DÜZELT ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 3: TOTALCHARGES DÜZELTME")
print("=" * 60)

# TotalCharges sütunundaki boşlukları NaN'a çeviriyoruz
# errors='coerce': sayıya dönüştürülemeyen değerleri (boşluk dahil) NaN yapar
# EDA'da gördüğümüz 11 boş satır tenure=0 olan yeni müşterilerdi
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

nan_sayisi = df['TotalCharges'].isnull().sum()
print(f"\nDönüşüm sonrası TotalCharges NaN sayısı: {nan_sayisi}")
print(f"Bu {nan_sayisi} satır tenure=0 olan yeni müşterilerdir.")

# is_new_customer flag ekliyoruz: tenure=0 olanları 1, diğerlerini 0 yapıyoruz
# Bu sayede "yeni müşteri" bilgisini kaybetmeden NaN'ı doldurabileceğiz
# Tezde bu flag'in churn üzerindeki etkisi ayrıca tartışılabilir
df['is_new_customer'] = (df['tenure'] == 0).astype(int)
print(f"\n'is_new_customer' sütunu eklendi.")
print(f"Yeni müşteri sayısı (tenure=0): {df['is_new_customer'].sum()}")

# Medyan ile dolduruyoruz: medyan, ortalamadan daha robust bir merkezi eğilim ölçüsüdür
# Aykırı değerlerden etkilenmediği için tercih ediyoruz
totalcharges_medyan = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(totalcharges_medyan)

print(f"\nTotalCharges medyan değeri     : {totalcharges_medyan:.2f}")
print(f"Doldurma sonrası NaN sayısı    : {df['TotalCharges'].isnull().sum()}")
print("Tüm NaN değerler medyan ile dolduruldu.")


# ── ADIM 4: CUSTOMERID SÜTUNUNU KALDIR ───────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 4: CUSTOMERID SÜTUNUNU KALDIR")
print("=" * 60)

# customerID sadece tanımlayıcı bir kimlik numarası — modelde kullanmıyoruz
# DATA_DICTIONARY.md'ye göre bu sütun kesinlikle modele alınmamalı
df = df.drop('customerID', axis=1)

print(f"customerID sütunu kaldırıldı.")
print(f"Güncel veri seti boyutu: {df.shape}  (beklenen: 7043 satır, 20 sütun)")


# ── ADIM 5: HEDEF DEĞİŞKENİ AYIR VE ENCODE ET ───────────────────────────────

print("\n" + "=" * 60)
print("ADIM 5: HEDEF DEĞİŞKENİ AYIR VE ENCODE ET")
print("=" * 60)

# Churn sütununu sayısala çeviriyoruz: Yes → 1 (ayrıldı), No → 0 (kalmış)
# Bu dönüşüm binary classification için zorunlu — model sayısal etiket bekler
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Churn sütununu özellikler matrisinden (X) çıkarıyoruz
X = df.drop('Churn', axis=1)

print(f"X (özellik matrisi) boyutu : {X.shape}")
print(f"y (hedef değişken) boyutu  : {y.shape}")
print(f"\nSınıf dağılımı:")
print(f"  Churn=0 (Kalmış) : {(y==0).sum()} kişi (%{(y==0).sum()/len(y)*100:.2f})")
print(f"  Churn=1 (Ayrılmış): {(y==1).sum()} kişi (%{(y==1).sum()/len(y)*100:.2f})")


# ── ADIM 6: SAYISAL VE KATEGORİK SÜTUNLARI BELİRLE ──────────────────────────

print("\n" + "=" * 60)
print("ADIM 6: SÜTUN TİPLERİNİ BELİRLE")
print("=" * 60)

# Sayısal sütunlar: doğrudan sayısal değer taşıyanlar
# SeniorCitizen zaten 0/1 olduğu için ayrıca encode gerekmez — integer olarak kalır
# is_new_customer da 0/1 flag, encode etmiyoruz
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Kategorik sütunlar: X'teki tüm object tipindeki sütunlar
# SeniorCitizen ve is_new_customer sayısal/binary olduğundan hariç tutuyoruz
categorical_cols = [
    col for col in X.columns
    if col not in numerical_cols
    and col not in ['SeniorCitizen', 'is_new_customer']
]

print(f"Sayısal sütunlar ({len(numerical_cols)} adet)   : {numerical_cols}")
print(f"Kategorik sütunlar ({len(categorical_cols)} adet): {categorical_cols}")
print(f"Binary sütunlar (encode edilmeyecek) : ['SeniorCitizen', 'is_new_customer']")


# ── ADIM 7: KATEGORİK DEĞİŞKENLERİ ONE-HOT ENCODE ET ────────────────────────

print("\n" + "=" * 60)
print("ADIM 7: ONE-HOT ENCODING")
print("=" * 60)

# One-Hot Encoding: kategorik değerleri 0/1 sütunlarına dönüştürür
# drop_first=True: ilk kategoriyi dummy tuzağından kaçınmak için düşürüyoruz
#   Örn: gender sütununda Male/Female varsa → gender_Male sütunu oluşur, Female zımnen anlaşılır
# Bu işlem veri sızıntısı (data leakage) yaratmaz çünkü sadece sütun yapısına bakıyor,
# istatistik hesaplamıyor — train/test split'ten önce yapılması güvenlidir
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Encoding öncesi X boyutu : {X.shape}")
print(f"Encoding sonrası X boyutu: {X_encoded.shape}")
print(f"Yeni sütun sayısı        : {X_encoded.shape[1]} (önceki: {X.shape[1]})")
print(f"\nTüm sütunlar:")
for col in X_encoded.columns:
    print(f"  - {col}  [{X_encoded[col].dtype}]")


# ── ADIM 8: TRAIN / TEST SPLIT ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 8: TRAIN / TEST SPLIT")
print("=" * 60)

# Veriyi 80/20 oranında bölüyoruz
# 80% train: modelin öğrenmesi için yeterli örnek gerekiyor (>5600 satır)
# 20% test : güvenilir değerlendirme için en az ~1400 örnek yeterli
# stratify=y: train ve test setlerinde Churn oranı aynı kalır (%73/%27)
# random_state=42: sonuçlar her çalıştırmada aynı olsun (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"X_train boyutu : {X_train.shape}")
print(f"X_test  boyutu : {X_test.shape}")
print(f"y_train boyutu : {y_train.shape}")
print(f"y_test  boyutu : {y_test.shape}")
print(f"\nTrain seti sınıf dağılımı:")
print(f"  Churn=0: {(y_train==0).sum()} (%{(y_train==0).sum()/len(y_train)*100:.2f})")
print(f"  Churn=1: {(y_train==1).sum()} (%{(y_train==1).sum()/len(y_train)*100:.2f})")
print(f"\nTest seti sınıf dağılımı:")
print(f"  Churn=0: {(y_test==0).sum()} (%{(y_test==0).sum()/len(y_test)*100:.2f})")
print(f"  Churn=1: {(y_test==1).sum()} (%{(y_test==1).sum()/len(y_test)*100:.2f})")
print("\nstratify=y sayesinde her iki sette de oran korundu.")


# ── ADIM 9: STANDARTLAŞTIRMA (StandardScaler) ────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 9: SAYISAL DEĞİŞKENLERİ STANDARTLAŞTIR")
print("=" * 60)

# StandardScaler: her sayısal sütunu ortalama=0, standart sapma=1 yapar
# NEDEN: tenure 0-72, MonthlyCharges 18-119, TotalCharges 0-8685 aralığında
# Bu farklı ölçekler Logistic Regression, SVM, KNN gibi algoritmaları yanıltır
# Ağaç tabanlı modeller (Random Forest, XGBoost) bunu gerektirmez ama zarar da vermez
#
# KRİTİK KURAL — Data Leakage Önleme:
# scaler.fit_transform() → SADECE train setine uygula
# scaler.transform()     → test setine SADECE transform uygula (fit değil!)
# Test seti, train setini "görmemiş" bir dış veri gibi işlenmelidir
scaler = StandardScaler()

# Scaler'ı sadece train setindeki sayısal sütunlarla eğitiyoruz
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Test setine sadece transform uyguluyoruz — train istatistikleriyle ölçekliyoruz
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Standartlaştırılan sütunlar: {numerical_cols}")
print(f"\nScaler parametreleri (train setinden öğrenildi):")
for col, mean, std in zip(numerical_cols, scaler.mean_, scaler.scale_):
    print(f"  {col:>20}: ortalama={mean:.2f}, std={std:.2f}")
print(f"\nX_train sayısal sütun istatistikleri (0'a yakın olmalı):")
print(X_train[numerical_cols].describe().round(3).to_string())

# Scaler'ı kaydediyoruz — Streamlit uygulamasında yeni müşteri verisi için kullanılacak
joblib.dump(scaler, f'{MODELS_DIR}scaler.pkl')
print(f"\nScaler kaydedildi: {MODELS_DIR}scaler.pkl")


# ── ADIM 10: SMOTE ile SINIF DENGESİZLİĞİNİ GİDER ───────────────────────────

print("\n" + "=" * 60)
print("ADIM 10: SMOTE UYGULA (SINIF DENGESİZLİĞİ)")
print("=" * 60)

# SMOTE (Synthetic Minority Over-sampling Technique):
# Azınlık sınıfına (Churn=1) sentetik yeni örnekler üretir
# K-nearest neighbors kullanarak gerçekçi ara noktalar türetir
# SONUÇ: Churn=0 ve Churn=1 sayıları eşitlenir → model her iki sınıfı iyi öğrenir
#
# UYARI: SMOTE SADECE train setine uygulanır
# Test seti gerçek dünya dağılımını korumalıdır — ASLA SMOTE uygulanmaz
smote = SMOTE(random_state=42)

print(f"SMOTE öncesi train seti:")
print(f"  Churn=0: {(y_train==0).sum()}")
print(f"  Churn=1: {(y_train==1).sum()}")

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nSMOTE sonrası train seti:")
print(f"  Churn=0: {(y_train_smote==0).sum()}")
print(f"  Churn=1: {(y_train_smote==1).sum()}")
print(f"\nToplam train satırı: {len(y_train_smote)} (önceki: {len(y_train)})")
print("Her iki sınıf eşitlendi — model dengeli veriyle eğitilecek.")


# ── ADIM 11: HAZIR VERİYİ KAYDET ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 11: İŞLENMİŞ VERİYİ KAYDET")
print("=" * 60)

# İşlenmiş train ve test setlerini CSV olarak kaydediyoruz
# Faz 3'te (model eğitimi) bu dosyaları okuyacağız
X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_encoded.columns)
X_test_df        = pd.DataFrame(X_test.values, columns=X_encoded.columns)
y_train_smote_df = pd.Series(y_train_smote, name='Churn')
y_test_df        = pd.Series(y_test.values, name='Churn')

X_train_smote_df.to_csv(f'{TABLES_DIR}X_train.csv', index=False)
X_test_df.to_csv(f'{TABLES_DIR}X_test.csv', index=False)
y_train_smote_df.to_csv(f'{TABLES_DIR}y_train.csv', index=False)
y_test_df.to_csv(f'{TABLES_DIR}y_test.csv', index=False)

# Sütun isimlerini de kaydediyoruz — Streamlit uygulamasında gerekecek
sutun_listesi = pd.Series(X_encoded.columns.tolist())
sutun_listesi.to_csv(f'{TABLES_DIR}feature_names.csv', index=False, header=False)

print(f"Kaydedilen dosyalar ({TABLES_DIR}):")
print(f"  X_train.csv       : {X_train_smote_df.shape}")
print(f"  X_test.csv        : {X_test_df.shape}")
print(f"  y_train.csv       : {y_train_smote_df.shape}")
print(f"  y_test.csv        : {y_test_df.shape}")
print(f"  feature_names.csv : {len(sutun_listesi)} sütun")
print(f"\nScaler  : {MODELS_DIR}scaler.pkl")

print("\n" + "=" * 60)
print("FAZ 2 ÖN İŞLEME TAMAMLANDI.")
print("=" * 60)
print(f"""
ÖZET:
  Ham veri         : 7043 satır, 21 sütun
  Encoding sonrası : {X_encoded.shape[1]} sütun
  Train seti       : {X_train_smote_df.shape[0]} satır (SMOTE sonrası)
  Test  seti       : {X_test_df.shape[0]} satır (orijinal dağılım)
  Scaler           : models/scaler.pkl
  
Sonraki adım: Faz 3 - Model Eğitimi (model_training.py)
""")
