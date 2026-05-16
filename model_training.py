# =============================================================================
# FAZ 3: MODEL EĞİTİMİ (MODEL TRAINING)
# Proje: Telekomünikasyon Müşteri Kaybı (Churn) Tahmini
# Çalıştırma: python model_training.py  (venv aktif olmalı)
# DİKKAT: Bu script 5-15 dakika sürebilir (özellikle SVM yavaştır).
# =============================================================================


# ── ADIM 1: KÜTÜPHANELERİ IMPORT ET ─────────────────────────────────────────

# pandas: CSV dosyalarını okumak ve sonuç tablolarını oluşturmak için
import pandas as pd

# numpy: sayısal hesaplamalar ve ortalama/standart sapma hesapları için
import numpy as np

# joblib: eğitilmiş modelleri .pkl dosyasına kaydetmek için
# sklearn modelleri pickle uyumlu, joblib daha verimli
import joblib

# os: klasör varlık kontrolü ve oluşturma işlemleri için
import os

# time: her modelin eğitim süresini saniye cinsinden ölçmek için
import time

# LogisticRegression: temel istatistiksel sınıflandırma modeli (baseline)
# Doğrusal ilişkileri öğrenir, katsayılar yorumlanabilir — tez için ideal
from sklearn.linear_model import LogisticRegression

# RandomForestClassifier: birden fazla karar ağacını bir araya getiren ensemble model
# Aşırı öğrenmeye (overfitting) karşı dayanıklı, özellik önemi verir
from sklearn.ensemble import RandomForestClassifier

# SVC: Support Vector Classifier — sınıflar arasına en geniş boşluğu bulan model
# probability=True ile olasılık tahmini de yapabilir (ROC eğrisi için gerekli)
from sklearn.svm import SVC

# KNeighborsClassifier: en yakın K komşuya bakarak sınıf tahmin eder
# Basit ama etkili — özellikle küçük veri setlerinde iyi çalışır
from sklearn.neighbors import KNeighborsClassifier

# XGBClassifier: Gradient Boosting'in optimize edilmiş hali
# Kaggle yarışmalarında sıkça birinci olan güçlü bir ensemble yöntem
from xgboost import XGBClassifier

# StratifiedKFold: veriyi K parçaya bölerken sınıf oranlarını korur
# Dengesiz veri setlerinde (bizde %73/%27) bu çok önemli
from sklearn.model_selection import StratifiedKFold

# cross_val_score: modeli K-fold cross-validation ile değerlendirir
# Her fold'da farklı bir parça test, geri kalanı eğitim olarak kullanılır
from sklearn.model_selection import cross_val_score

# GridSearchCV: tüm hiperparametre kombinasyonlarını deneyerek en iyisini bulur
# Brute-force yaklaşım — yeni başlayanlar için en anlaşılır yöntem
from sklearn.model_selection import GridSearchCV

# Çıktı dosyalarının kaydedileceği klasörler
MODELS_DIR = 'models/'
TABLES_DIR = 'results/tables/'

# Klasörler yoksa oluşturuyoruz (preprocessing.py zaten oluşturmuş olabilir)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("FAZ 3: MODEL EĞİTİMİ")
print("=" * 70)
print("ADIM 1 TAMAMLANDI: Kütüphaneler yüklendi.")


# ── ADIM 2: HAZIRLANMIŞ VERİYİ YÜKLE ─────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 2: HAZIRLANMIŞ VERİYİ YÜKLE")
print("=" * 70)

# preprocessing.py tarafından kaydedilen CSV dosyalarını okuyoruz
# X_train: SMOTE uygulanmış eğitim özellikleri (dengeli sınıflar)
# X_test: orijinal dağılımdaki test özellikleri
# y_train: SMOTE uygulanmış eğitim etiketleri
# y_test: orijinal test etiketleri
X_train = pd.read_csv(f'{TABLES_DIR}X_train.csv')
X_test = pd.read_csv(f'{TABLES_DIR}X_test.csv')
y_train = pd.read_csv(f'{TABLES_DIR}y_train.csv').squeeze()
y_test = pd.read_csv(f'{TABLES_DIR}y_test.csv').squeeze()

# .squeeze() ile DataFrame'i Series'e çeviriyoruz çünkü y tek sütunlu —
# sklearn modelleri hedef değişkeni 1D array olarak bekler

print(f"X_train boyutu : {X_train.shape}")
print(f"X_test  boyutu : {X_test.shape}")
print(f"y_train boyutu : {y_train.shape}")
print(f"y_test  boyutu : {y_test.shape}")

print(f"\ny_train sınıf dağılımı (SMOTE sonrası — dengeli olmalı):")
print(f"  Churn=0: {(y_train == 0).sum()}")
print(f"  Churn=1: {(y_train == 1).sum()}")

print(f"\ny_test sınıf dağılımı (orijinal dağılım — dengesiz olmalı):")
print(f"  Churn=0: {(y_test == 0).sum()}")
print(f"  Churn=1: {(y_test == 1).sum()}")


# ── ADIM 3: 5 MODEL TANIMLA ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 3: 5 MODEL TANIMLA")
print("=" * 70)

# Modelleri bir sözlükte (dict) tutuyoruz — döngüyle hepsini eğitmek kolay olacak
# random_state=42: her çalıştırmada aynı sonuçlar — tezde tekrarlanabilirlik şart
models = {
    # Logistic Regression: lineer sınıflandırıcı (tez baseline modeli)
    # max_iter=1000: yakınsaması garantilensin diye iterasyon limitini artırıyoruz
    # Varsayılan 100 bazı veri setlerinde yetersiz kalabilir
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    # Random Forest: 100 karar ağacının çoğunluk oylamasıyla karar verir
    # n_jobs=-1: tüm CPU çekirdeklerini kullan — eğitim hızlanır
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    # XGBoost: sıralı ağaçlar önceki ağacın hatalarından öğrenir (boosting)
    # eval_metric='logloss': binary classification için standart kayıp fonksiyonu
    # use_label_encoder=False: eski XGBoost uyarılarını bastırır
    "XGBoost": XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    ),

    # SVM: veri noktalarını ayıran en optimum hiperdüzlemi bulur
    # probability=True: predict_proba() çalışsın diye — ROC eğrisi için gerekli
    # DİKKAT: SVM büyük veri setlerinde YAVAŞTIR (O(n²) veya O(n³))
    "SVM": SVC(
        random_state=42,
        probability=True
    ),

    # KNN: yeni bir örneğin en yakın K komşusuna bakarak sınıf belirler
    # n_jobs=-1: mesafe hesaplamalarını paralel yap — hız kazancı
    "KNN": KNeighborsClassifier(
        n_jobs=-1
    ),
}

print(f"{len(models)} model tanımlandı:")
for model_name in models:
    print(f"  - {model_name}")


# ── ADIM 4: HER MODEL İÇİN CROSS-VALIDATION VE EĞİTİM ──────────────────────

print("\n" + "=" * 70)
print("ADIM 4: CROSS-VALIDATION VE EĞİTİM")
print("=" * 70)

# 5-fold Stratified Cross-Validation ayarı
# StratifiedKFold: her fold'da sınıf oranları korunur (SMOTE sonrası %50/%50)
# shuffle=True: veriyi karıştırır — sıralama etkisini kaldırır
# random_state=42: karıştırma her seferinde aynı olsun
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Her modelin sonuçlarını tutacak listeler
results_list = []
# Eğitilmiş baseline modelleri saklamak için sözlük
baseline_models = {}

for model_name, model in models.items():
    print(f"\n{'─' * 50}")
    print(f"Model: {model_name}")
    print(f"{'─' * 50}")

    # Eğitim süresini ölçmek için başlangıç zamanını kaydediyoruz
    start_time = time.time()

    # SVM büyük veri setlerinde çok yavaş olabilir — kullanıcıyı bilgilendiriyoruz
    if model_name == "SVM":
        print("⚠ UYARI: SVM eğitimi yavaş olabilir (1-5 dakika). Sabırlı olun...")

    # 5-fold cross-validation: modeli 5 kez farklı train/test bölümleriyle değerlendirir
    # scoring='f1': dengesiz sınıflarda accuracy yanıltıcıdır, F1 daha güvenilir
    # F1 = 2 * (precision * recall) / (precision + recall)
    # cv=cv: yukarıda tanımladığımız StratifiedKFold nesnesini kullanıyoruz
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

    # Modeli TÜM train seti üzerinde eğitiyoruz
    # CV sadece performans tahmini verir — gerçek model tüm veriyle eğitilmeli
    model.fit(X_train, y_train)

    # Eğitim süresini hesapla (CV + fit toplam süresi)
    elapsed_time = time.time() - start_time

    # Sonuçları yazdır
    print(f"  CV F1 Skorları (5 fold): {np.round(cv_scores, 4)}")
    print(f"  CV F1 Ortalama ± Std   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Eğitim süresi          : {elapsed_time:.2f} saniye")

    # Sonuçları tabloya ekle
    results_list.append({
        'Model': model_name,
        'CV_F1_Mean': round(cv_scores.mean(), 4),
        'CV_F1_Std': round(cv_scores.std(), 4),
        'Training_Time_Sec': round(elapsed_time, 2)
    })

    # Eğitilmiş modeli sözlüğe kaydet
    baseline_models[model_name] = model


# ── ADIM 5: SONUÇLARI DATAFRAME OLARAK TOPLA VE KAYDET ───────────────────────

print("\n" + "=" * 70)
print("ADIM 5: BASELINE SONUÇLAR TABLOSU")
print("=" * 70)

# Sonuçları DataFrame'e çeviriyoruz — tablo olarak görüntülemek için
baseline_results_df = pd.DataFrame(results_list)

# CV F1 skoruna göre azalan sırada sıralıyoruz — en iyi model üstte olsun
baseline_results_df = baseline_results_df.sort_values('CV_F1_Mean', ascending=False)

# Index'i sıfırlıyoruz — sıralama sonrası düzgün görünsün
baseline_results_df = baseline_results_df.reset_index(drop=True)

print("\n5 Modelin Baseline Sonuçları (CV F1 skoruna göre sıralı):")
print(baseline_results_df.to_string(index=False))

# CSV olarak kaydediyoruz — tezde tablo olarak kullanılacak
baseline_results_df.to_csv(f'{TABLES_DIR}model_baseline_results.csv', index=False)
print(f"\nSonuçlar kaydedildi: {TABLES_DIR}model_baseline_results.csv")


# ── ADIM 6: EN İYİ 3 MODEL İÇİN GRIDSEARCHCV ────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 6: EN İYİ 3 MODEL İÇİN HİPERPARAMETRE OPTİMİZASYONU")
print("=" * 70)

# En yüksek CV F1 skoruna sahip 3 modeli seçiyoruz
# Zaten sıralı olduğu için ilk 3 satır en iyi modeller
top_3_names = baseline_results_df['Model'].head(3).tolist()

print(f"\nOptimize edilecek 3 model: {top_3_names}")

# Her model için denenecek hiperparametre ızgaraları (grid)
# Bu değerler literatürde yaygın kullanılan aralıklardan seçildi
param_grids = {
    "Logistic Regression": {
        # C: düzenlileştirme (regularization) gücü — düşük C = daha güçlü düzenlileştirme
        # Aşırı öğrenmeyi önlemek için farklı C değerlerini deniyoruz
        'C': [0.01, 0.1, 1, 10],
        # L2 penalty: katsayıların karelerini cezalandırır (Ridge)
        'penalty': ['l2'],
        # LBFGS solver: L2 penalty ile uyumlu, çok sınıflı problemlerde de çalışır
        'solver': ['lbfgs'],
    },

    "Random Forest": {
        # n_estimators: ormandaki ağaç sayısı — fazlası genelde daha iyi ama yavaş
        'n_estimators': [100, 200],
        # max_depth: ağacın maksimum derinliği — derin ağaçlar overfit edebilir
        # None: sınırsız derinlik (yaprak saf olana kadar büyür)
        'max_depth': [10, 20, None],
        # min_samples_split: bir düğümü bölmek için gereken minimum örnek sayısı
        # Yüksek değer = daha basit ağaç = overfitting riski azalır
        'min_samples_split': [2, 5],
    },

    "XGBoost": {
        # n_estimators: boosting turlarının sayısı — her turda bir ağaç eklenir
        'n_estimators': [100, 200],
        # max_depth: her ağacın maksimum derinliği — XGBoost'ta 3-7 arası yaygın
        'max_depth': [3, 5, 7],
        # learning_rate: her ağacın katkısının ağırlığı — düşük = daha yavaş öğrenme ama genellikle daha iyi
        'learning_rate': [0.01, 0.1],
    },

    "SVM": {
        # C: hata toleransı — büyük C = daha az hata toleransı (overfitting riski)
        'C': [0.1, 1, 10],
        # RBF kernel: doğrusal olmayan sınırları öğrenebilir
        'kernel': ['rbf'],
        # gamma='scale': 1/(n_features * X.var()) — veri boyutuna otomatik uyum sağlar
        'gamma': ['scale'],
    },

    "KNN": {
        # n_neighbors: kaç komşuya bakılacak — tek sayı tercih edilir (beraberlik olmasın)
        'n_neighbors': [3, 5, 7, 11],
        # weights: komşuların oy ağırlığı
        # 'uniform': her komşunun oyu eşit
        # 'distance': yakın komşunun oyu daha ağır basar
        'weights': ['uniform', 'distance'],
    },
}

# Optimize edilmiş modelleri saklamak için sözlük
tuned_models = {}
tuned_results_list = []

for model_name in top_3_names:
    print(f"\n{'─' * 50}")
    print(f"Optimizasyon: {model_name}")
    print(f"{'─' * 50}")

    # SVM optimize ediliyorsa kullanıcıyı uyarıyoruz — GridSearchCV ile SVM çok yavaş
    if model_name == "SVM":
        print("⚠ UYARI: SVM + GridSearchCV çok yavaş olabilir (5-10 dakika). Sabırlı olun...")

    # Bu model için tanımlı hiperparametre ızgarasını alıyoruz
    param_grid = param_grids[model_name]
    print(f"Denenecek parametreler: {param_grid}")

    # Kaç farklı kombinasyon denenecek hesaplayalım — kullanıcı süreyi tahmin edebilsin
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    print(f"Toplam kombinasyon sayısı: {n_combinations}")
    print(f"Her kombinasyon 5-fold CV ile test edilecek → {n_combinations * 5} model eğitimi")

    # Eğitim süresini ölçmek için başlangıç zamanı
    start_time = time.time()

    # GridSearchCV: tüm parametre kombinasyonlarını dener, en iyi skoru verenini seçer
    # cv=5: 5-fold cross-validation
    # scoring='f1': F1 skorunu optimize ediyoruz (dengesiz veri için accuracy yanıltıcı)
    # n_jobs=-1: tüm CPU çekirdeklerini kullan
    # refit=True (varsayılan): en iyi parametrelerle modeli tüm train verisinde yeniden eğitir
    grid_search = GridSearchCV(
        estimator=models[model_name],
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        refit=True
    )

    # GridSearchCV'yi çalıştırıyoruz — tüm kombinasyonlar deneniyor
    grid_search.fit(X_train, y_train)

    # Geçen süreyi hesapla
    elapsed_time = time.time() - start_time

    # En iyi parametreleri ve skoru yazdır
    print(f"\n  En iyi parametreler : {grid_search.best_params_}")
    print(f"  En iyi CV F1 skoru  : {grid_search.best_score_:.4f}")
    print(f"  Optimizasyon süresi : {elapsed_time:.2f} saniye")

    # Optimize edilmiş modeli sakla (refit=True sayesinde zaten en iyi parametrelerle eğitilmiş)
    tuned_models[model_name] = grid_search.best_estimator_

    # Sonuçları tabloya ekle
    tuned_results_list.append({
        'Model': model_name,
        'Best_Params': str(grid_search.best_params_),
        'Tuned_CV_F1': round(grid_search.best_score_, 4),
        'Tuning_Time_Sec': round(elapsed_time, 2)
    })

# Optimize sonuçlarını DataFrame'e çevir
tuned_results_df = pd.DataFrame(tuned_results_list)

# CV F1 skoruna göre sırala
tuned_results_df = tuned_results_df.sort_values('Tuned_CV_F1', ascending=False)
tuned_results_df = tuned_results_df.reset_index(drop=True)

print(f"\n{'─' * 50}")
print("Optimize Edilmiş Modellerin Sonuçları:")
print(tuned_results_df.to_string(index=False))

# CSV olarak kaydet — tezde karşılaştırma tablosu olarak kullanılacak
tuned_results_df.to_csv(f'{TABLES_DIR}model_tuned_results.csv', index=False)
print(f"\nSonuçlar kaydedildi: {TABLES_DIR}model_tuned_results.csv")


# ── ADIM 7: FİNAL MODELLERİNİ KAYDET ────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 7: MODELLERİ KAYDET")
print("=" * 70)

# 5 baseline modeli tek bir dosyaya kaydediyoruz (dict olarak)
# Bu modeller varsayılan parametrelerle eğitildi — karşılaştırma referansı
joblib.dump(baseline_models, f'{MODELS_DIR}baseline_models.pkl')
print(f"5 baseline model kaydedildi : {MODELS_DIR}baseline_models.pkl")

# 3 optimize edilmiş modeli tek bir dosyaya kaydediyoruz
# Bu modeller GridSearchCV ile bulunan en iyi parametrelerle eğitildi
joblib.dump(tuned_models, f'{MODELS_DIR}tuned_models.pkl')
print(f"3 optimize model kaydedildi : {MODELS_DIR}tuned_models.pkl")

# En yüksek CV F1 skoruna sahip modeli "best_model" olarak ayrıca kaydediyoruz
# Streamlit uygulaması bu modeli kullanacak
best_model_name = tuned_results_df.iloc[0]['Model']
best_model_score = tuned_results_df.iloc[0]['Tuned_CV_F1']
best_model = tuned_models[best_model_name]

joblib.dump(best_model, f'{MODELS_DIR}best_model.pkl')
print(f"\nEn iyi model kaydedildi     : {MODELS_DIR}best_model.pkl")
print(f"  Model adı : {best_model_name}")
print(f"  CV F1 skoru: {best_model_score:.4f}")

# En iyi modelin adını da metin dosyası olarak kaydediyoruz
# Faz 4 (değerlendirme) ve Streamlit uygulaması bu bilgiyi kullanacak
best_model_info = pd.DataFrame([{
    'Model': best_model_name,
    'CV_F1': best_model_score
}])
best_model_info.to_csv(f'{TABLES_DIR}best_model_info.csv', index=False)
print(f"  Model bilgisi: {TABLES_DIR}best_model_info.csv")


# ── ADIM 8: ÖZET ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 8: FAZ 3 ÖZET")
print("=" * 70)

print("\n┌──────────────────────────────────────────────────────────────┐")
print("│                5 MODELİN BASELINE SONUÇLARI                 │")
print("└──────────────────────────────────────────────────────────────┘")
print(baseline_results_df.to_string(index=False))

print("\n┌──────────────────────────────────────────────────────────────┐")
print("│           3 OPTİMİZE EDİLMİŞ MODELİN SONUÇLARI             │")
print("└──────────────────────────────────────────────────────────────┘")
print(tuned_results_df.to_string(index=False))

print(f"\n★ En iyi model: {best_model_name} (CV F1 = {best_model_score:.4f})")

print(f"""
Kaydedilen dosyalar:
  {MODELS_DIR}baseline_models.pkl     — 5 baseline model
  {MODELS_DIR}tuned_models.pkl        — 3 optimize edilmiş model
  {MODELS_DIR}best_model.pkl          — En iyi model ({best_model_name})
  {TABLES_DIR}model_baseline_results.csv — Baseline sonuç tablosu
  {TABLES_DIR}model_tuned_results.csv    — Optimizasyon sonuç tablosu
  {TABLES_DIR}best_model_info.csv        — En iyi model bilgisi
""")

print("=" * 70)
print("FAZ 3 TAMAMLANDI.")
print("=" * 70)
print("\nSonraki adım: Faz 4 — Model Değerlendirme ve İstatistiksel Analiz")
