# =============================================================================
# FAZ 4: MODEL DEĞERLENDİRME VE İSTATİSTİKSEL ANALİZ
# Proje: Telekomünikasyon Müşteri Kaybı (Churn) Tahmini
# Çalıştırma: python model_evaluation.py  (venv aktif olmalı)
# DİKKAT: SHAP analizi 3-5 dakika sürebilir, sabırlı olun.
# ÖNEMLİ: Bu scripti çalıştırmadan önce model_training.py çalıştırılmış olmalı!
# =============================================================================


# ── ADIM 1: KÜTÜPHANELERİ IMPORT ET ─────────────────────────────────────────

# matplotlib backend'ini dosyaya kaydetme moduna alıyoruz
# 'Agg': ekrana çizim yapmaz, sadece dosyaya kaydeder — sunucu/script ortamları için ideal
import matplotlib
matplotlib.use('Agg')

# pandas: tablo işlemleri, CSV okuma/yazma
import pandas as pd

# numpy: sayısal hesaplamalar, dizi işlemleri
import numpy as np

# joblib: kaydedilmiş .pkl model dosyalarını yüklemek için
import joblib

# os: dosya/klasör varlık kontrolü ve oluşturma
import os

# matplotlib.pyplot: grafik çizimi ve kaydetme
import matplotlib.pyplot as plt

# seaborn: istatistiksel görselleştirme (heatmap, renk paletleri)
import seaborn as sns

# sklearn.metrics: model performans metrikleri
# accuracy_score: doğru tahmin oranı (TP+TN)/(TP+TN+FP+FN)
# precision_score: pozitif tahminlerin doğruluk oranı TP/(TP+FP)
# recall_score: gerçek pozitifleri yakalama oranı TP/(TP+FN)
# f1_score: precision ve recall'un harmonik ortalaması
# roc_auc_score: ROC eğrisi altındaki alan (ayırt etme gücü)
# average_precision_score: PR eğrisi altındaki alan
# confusion_matrix: tahmin vs gerçek karşılaştırma tablosu
# roc_curve: ROC eğrisi koordinatları (FPR, TPR)
# precision_recall_curve: PR eğrisi koordinatları
# classification_report: tüm metriklerin özet raporu
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)

# McNemar's test: iki sınıflandırıcı arasındaki performans farkının
# istatistiksel anlamlılığını test eder — tez jürisi için kritik
from statsmodels.stats.contingency_tables import mcnemar

# SHAP (SHapley Additive exPlanations): modelin kararlarını açıklamak için
# Oyun teorisinden gelen Shapley değerlerini kullanarak her özelliğin
# tahmine katkısını hesaplar — kara kutu modelleri yorumlanabilir yapar
import shap

# Tekrarlanabilirlik için rastgele tohum değeri sabitliyoruz
# Tüm projede 42 kullanıyoruz — tezde aynı sonuçları üretmek şart
np.random.seed(42)

# Görselleştirme ayarları — akademik yayın standartları
# 'whitegrid': arka plan beyaz, ızgara çizgileri var — verileri okumak kolay
sns.set_style('whitegrid')
# 'colorblind': renk körü dostu palet — akademik standartta tercih edilir
sns.set_palette('colorblind')
# Font boyutu 12: tez baskısında okunabilir olması için minimum gerekli boyut
plt.rcParams.update({'font.size': 12})

# Çıktı klasör yolları — tüm dosyalar buraya kaydedilecek
MODELS_DIR = 'models/'
FIGURES_DIR = 'results/figures/'
TABLES_DIR = 'results/tables/'

# Klasörler yoksa oluştur (önceki fazlar zaten oluşturmuş olabilir)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("FAZ 4: MODEL DEĞERLENDİRME VE İSTATİSTİKSEL ANALİZ")
print("=" * 70)
print("ADIM 1 TAMAMLANDI: Kütüphaneler yüklendi.")


# ── ADIM 2: VERİ VE MODELLERİ YÜKLE ─────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 2: VERİ VE MODELLERİ YÜKLE")
print("=" * 70)

# Gerekli dosyaların var olup olmadığını kontrol ediyoruz
# model_training.py çalıştırılmamışsa bu dosyalar eksik olacak
required_files = [
    f'{MODELS_DIR}baseline_models.pkl',
    f'{MODELS_DIR}tuned_models.pkl',
    f'{TABLES_DIR}X_test.csv',
    f'{TABLES_DIR}y_test.csv',
    f'{TABLES_DIR}model_baseline_results.csv',
    f'{TABLES_DIR}model_tuned_results.csv',
]

# Her dosyayı tek tek kontrol et — eksik olan varsa hata ver ve dur
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print("\nHATA: Aşağıdaki dosyalar bulunamadı:")
    for f in missing:
        print(f"  ✗ {f}")
    print("\nÇÖZÜM: Önce model_training.py çalıştırın:")
    print("  python model_training.py")
    exit(1)

# Test verisini yüklüyoruz — modelleri değerlendirmek için
# X_test: preprocessing.py tarafından kaydedilen test özellikleri (orijinal dağılım)
X_test = pd.read_csv(f'{TABLES_DIR}X_test.csv')

# y_test: test etiketleri — .squeeze() ile DataFrame'i Series'e çeviriyoruz
# sklearn metrikleri 1D array bekler, DataFrame değil
y_test = pd.read_csv(f'{TABLES_DIR}y_test.csv').squeeze()

print(f"X_test boyutu : {X_test.shape}")
print(f"y_test boyutu : {y_test.shape}")
print(f"Test seti sınıf dağılımı:")
print(f"  Churn=0 (Kalmış)  : {(y_test == 0).sum()}")
print(f"  Churn=1 (Ayrılmış): {(y_test == 1).sum()}")

# 5 baseline modeli yüklüyoruz — varsayılan parametrelerle eğitilmişler
# Bu dict şu anahtarları içerir: "Logistic Regression", "Random Forest",
# "XGBoost", "SVM", "KNN"
baseline_models = joblib.load(f'{MODELS_DIR}baseline_models.pkl')
print(f"\n5 baseline model yüklendi: {list(baseline_models.keys())}")

# 3 optimize edilmiş modeli yüklüyoruz — GridSearchCV ile en iyi parametreler bulunmuş
# Hangi 3 modelin optimize edildiği CV F1 skoruna göre belirlenmişti
tuned_models = joblib.load(f'{MODELS_DIR}tuned_models.pkl')
print(f"3 tuned model yüklendi  : {list(tuned_models.keys())}")

# Final modeller sözlüğü: her model için mevcut en iyi versiyonu kullanıyoruz
# Tuned versiyon varsa onu al, yoksa baseline versiyonu kullan
# Bu şekilde 5 modelin her biri en iyi performansıyla değerlendirilecek
all_models = {}
for name, model in baseline_models.items():
    if name in tuned_models:
        # Bu model GridSearchCV ile optimize edilmiş — tuned versiyonu kullan
        all_models[name] = tuned_models[name]
        print(f"  {name}: TUNED versiyon kullanılıyor")
    else:
        # Bu model optimize edilmemiş — baseline versiyonu kullan
        all_models[name] = model
        print(f"  {name}: BASELINE versiyon kullanılıyor")

# Modellerin değerlendirme sırası — grafikler ve tablolarda tutarlı olacak
model_order = ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "KNN"]

# Cross-validation sonuçlarını da yüklüyoruz — thesis summary tablosunda lazım olacak
baseline_results = pd.read_csv(f'{TABLES_DIR}model_baseline_results.csv')
tuned_results = pd.read_csv(f'{TABLES_DIR}model_tuned_results.csv')

print("\nADIM 2 TAMAMLANDI: Veri ve modeller yüklendi.")


# ── ADIM 3: HER MODEL İÇİN KAPSAMLI METRİKLER ──────────────────────────────

print("\n" + "=" * 70)
print("ADIM 3: KAPSAMLI METRİK HESAPLAMA")
print("=" * 70)

# Tüm modellerin tahminlerini ve olasılıklarını saklayacağız
# Sonraki adımlarda (ROC, PR, McNemar) tekrar kullanmak için
predictions = {}
probabilities = {}
metrics_list = []

for model_name in model_order:
    # Modeli sözlükten al
    model = all_models[model_name]

    # predict(): 0/1 sınıf tahmini — confusion matrix ve F1 için
    y_pred = model.predict(X_test)

    # predict_proba()[:, 1]: Churn=1 olma olasılığı (0-1 arası sürekli değer)
    # ROC-AUC ve PR-AUC hesabında sınıf olasılıkları gerekli
    # [:, 1] → ikinci sütun = pozitif sınıf (Churn=1) olasılığı
    y_prob = model.predict_proba(X_test)[:, 1]

    # Tahminleri sakla — McNemar, ROC, PR eğrileri için tekrar kullanacağız
    predictions[model_name] = y_pred
    probabilities[model_name] = y_prob

    # Confusion matrix'ten TP, TN, FP, FN değerlerini çıkarıyoruz
    # .ravel() 2x2 matrixi düzleştirip 4 değer döndürür: TN, FP, FN, TP sırasıyla
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Temel metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Olasılık tabanlı metrikler (threshold-bağımsız, daha güvenilir)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Specificity (Özgüllük): Gerçek negatifleri doğru tahmin etme oranı
    # TN / (TN + FP) → "Kalmış müşterilerin kaçını doğru bulduk?"
    # Yüksek specificity = gereksiz müdahaleleri azaltır
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # NPV (Negative Predictive Value): Negatif tahminlerin doğruluk oranı
    # TN / (TN + FN) → "Kalmış dediğimiz müşterilerden kaçı gerçekten kalmış?"
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Sonuçları listeye ekle — sonra DataFrame'e çevireceğiz
    metrics_list.append({
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC': round(pr_auc, 4),
        'Specificity': round(specificity, 4),
        'NPV': round(npv, 4),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
    })

    # Her model için sonuçları konsola yazdır
    print(f"\n{'─' * 50}")
    print(f"Model: {model_name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  NPV         : {npv:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# Sonuçları DataFrame'e çevirip F1-Score'a göre sıralıyoruz
metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

# CSV olarak kaydet — tezin bulgular bölümünde kullanılacak
metrics_df.to_csv(f'{TABLES_DIR}model_full_metrics.csv', index=False)
print(f"\nMetrik tablosu kaydedildi: {TABLES_DIR}model_full_metrics.csv")

print("\n5 Modelin Kapsamlı Metrikleri (F1-Score'a göre sıralı):")
print(metrics_df.to_string(index=False))

print("\nADIM 3 TAMAMLANDI: Tüm metrikler hesaplandı.")


# ── ADIM 4: CONFUSION MATRICES ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 4: CONFUSION MATRIX GÖRSELLEŞTİRME")
print("=" * 70)

# 2 satır × 3 sütun = 6 hücre, 5 model + 1 boş hücre
# figsize=(18, 10): geniş alan, heatmap etiketleri rahat okunabilsin
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 2D axes dizisini 1D'ye düzleştiriyoruz — döngüde kolay erişim için
axes_flat = axes.flatten()

# Türkçe etiketler: 0 = Ayrılmadı (kalmış müşteri), 1 = Ayrıldı (churn)
labels = ['Ayrılmadı', 'Ayrıldı']

for idx, model_name in enumerate(model_order):
    ax = axes_flat[idx]

    # confusion_matrix: [[TN, FP], [FN, TP]] formatında 2x2 matris döndürür
    cm = confusion_matrix(y_test, predictions[model_name])

    # seaborn heatmap: renk yoğunluğu ile değerleri görselleştir
    # annot=True: hücrelerin içine sayıları yaz
    # fmt='d': tam sayı formatı (ondalık olmasın)
    # cmap='Blues': mavi tonları — akademik yayınlarda yaygın
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar=False,
                annot_kws={'size': 14, 'fontweight': 'bold'})

    # Her subplot'a model adını başlık olarak yaz
    ax.set_title(model_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen', fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=12)

# 6. hücre boş — 5 modelimiz var, subplot grid 2×3 = 6 hücre
axes_flat[5].axis('off')

# Ana başlık: tüm subplot'ların üstünde ortak başlık
fig.suptitle('Confusion Matrix - Tüm Modeller', fontsize=16, fontweight='bold', y=1.02)

# tight_layout: subplot'lar arasındaki boşlukları optimize eder
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}08_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Confusion matrix grafiği kaydedildi: {FIGURES_DIR}08_confusion_matrices.png")
print("ADIM 4 TAMAMLANDI.")


# ── ADIM 5: ROC EĞRİLERİ ────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 5: ROC EĞRİLERİ")
print("=" * 70)

# Tek grafikte 5 modelin ROC eğrisini karşılaştıracağız
fig, ax = plt.subplots(figsize=(10, 8))

# Her model için farklı renk — colorblind palette'ten 5 renk alıyoruz
colors = sns.color_palette('colorblind', n_colors=5)

for idx, model_name in enumerate(model_order):
    # roc_curve: FPR (yanlış pozitif oranı) ve TPR (doğru pozitif oranı) döndürür
    # Farklı threshold değerleri için bu oranlar hesaplanır
    fpr, tpr, _ = roc_curve(y_test, probabilities[model_name])

    # AUC değerini hesapla — legend'de göstereceğiz
    auc_val = roc_auc_score(y_test, probabilities[model_name])

    # Eğriyi çiz — kalın çizgi (lw=2), model adı ve AUC değeri legend'de
    ax.plot(fpr, tpr, color=colors[idx], lw=2,
            label=f'{model_name} (AUC = {auc_val:.4f})')

# Diagonal kesikli çizgi: rastgele tahmin edici (AUC = 0.50)
# Bir model bu çizginin altındaysa rastgele tahminden bile kötü demektir
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7,
        label='Rastgele Tahmin (AUC = 0.50)')

# Eksen etiketleri ve başlık (Türkçe)
ax.set_xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=13)
ax.set_ylabel('Doğru Pozitif Oranı (TPR)', fontsize=13)
ax.set_title('ROC Eğrileri - Model Karşılaştırması', fontsize=15, fontweight='bold')

# Legend sağ altta — eğrileri kapatmamak için en uygun konum
ax.legend(loc='lower right', fontsize=11)

# Eksen aralığı: 0-1 arası (olasılık değerleri)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}09_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"ROC eğrileri kaydedildi: {FIGURES_DIR}09_roc_curves.png")
print("ADIM 5 TAMAMLANDI.")


# ── ADIM 6: PRECISION-RECALL EĞRİLERİ ───────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 6: PRECISION-RECALL EĞRİLERİ")
print("=" * 70)

# Dengesiz veri setlerinde (bizde %73/%27) ROC eğrisi yanıltıcı olabilir
# PR eğrisi azınlık sınıfına (Churn=Yes) odaklanır — daha güvenilir değerlendirme
fig, ax = plt.subplots(figsize=(10, 8))

for idx, model_name in enumerate(model_order):
    # precision_recall_curve: farklı threshold'larda precision ve recall döndürür
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probabilities[model_name])

    # PR-AUC (Average Precision): eğri altındaki alan — tek sayıda özet
    pr_auc_val = average_precision_score(y_test, probabilities[model_name])

    # Eğriyi çiz — recall x ekseni, precision y ekseni
    ax.plot(rec_arr, prec_arr, color=colors[idx], lw=2,
            label=f'{model_name} (PR-AUC = {pr_auc_val:.4f})')

# Baseline çizgisi: pozitif sınıf oranı (%26.5 civarı)
# Rastgele tahmin edici bu seviyede kalır
baseline_rate = y_test.mean()
ax.axhline(y=baseline_rate, color='gray', linestyle='--', lw=1.5, alpha=0.7,
           label=f'Rastgele Tahmin ({baseline_rate:.2f})')

ax.set_xlabel('Recall (Duyarlılık)', fontsize=13)
ax.set_ylabel('Precision (Kesinlik)', fontsize=13)
ax.set_title('Precision-Recall Eğrileri - Model Karşılaştırması',
             fontsize=15, fontweight='bold')

# Legend sol altta — eğrileri kapatmamak için uygun konum
ax.legend(loc='lower left', fontsize=11)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}10_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"PR eğrileri kaydedildi: {FIGURES_DIR}10_pr_curves.png")
print("ADIM 6 TAMAMLANDI.")


# ── ADIM 7: McNEMAR'S TEST (İSTATİSTİKSEL ANLAMLILIK) ──────────────────────

print("\n" + "=" * 70)
print("ADIM 7: McNEMAR'S TEST — İSTATİSTİKSEL ANLAMLILIK")
print("=" * 70)

# McNemar testi: iki sınıflandırıcının tahmin HATALARI arasındaki farkı test eder
# H0: İki model aynı hata oranına sahiptir (fark yoktur)
# H1: İki model farklı hata oranlarına sahiptir (anlamlı fark vardır)
# Bu test eşleştirilmiş (paired) veriler üzerinde çalışır — her iki model
# aynı test setinde değerlendirildiği için uygulanabilir

# Random Forest ve XGBoost en güçlü 2 model — aralarındaki farkı test edeceğiz
model1_name = "Random Forest"
model2_name = "XGBoost"

print(f"\nKarşılaştırılan modeller: {model1_name} vs {model2_name}")

# Her modelin tahminlerini alıyoruz (ADIM 3'te hesaplamıştık)
y_pred_m1 = predictions[model1_name]
y_pred_m2 = predictions[model2_name]

# Her tahmin için doğru mu yanlış mı kontrolü — boolean diziler oluşturuyoruz
# y_test.values: pandas Series'i numpy array'e çeviriyoruz — indeks sorununu önlemek için
correct_m1 = (y_pred_m1 == y_test.values)
correct_m2 = (y_pred_m2 == y_test.values)

# McNemar contingency (olasılık) tablosu:
#                     Model2 Doğru    Model2 Yanlış
# Model1 Doğru          a               b
# Model1 Yanlış         c               d
#
# a: Her iki model de doğru tahmin etmiş
# b: Model1 doğru, Model2 yanlış → Model1'in avantajlı olduğu örnekler
# c: Model1 yanlış, Model2 doğru → Model2'nin avantajlı olduğu örnekler
# d: Her iki model de yanlış tahmin etmiş
#
# McNemar testi SADECE uyumsuz çiftlere (b ve c) bakar
# b ≈ c ise modeller arasında anlamlı fark yoktur
a = np.sum(correct_m1 & correct_m2)
b = np.sum(correct_m1 & ~correct_m2)
c = np.sum(~correct_m1 & correct_m2)
d = np.sum(~correct_m1 & ~correct_m2)

print(f"\nContingency (Olasılık) Tablosu:")
print(f"  Her ikisi de doğru (a)          : {a}")
print(f"  {model1_name} doğru, {model2_name} yanlış (b): {b}")
print(f"  {model1_name} yanlış, {model2_name} doğru (c): {c}")
print(f"  Her ikisi de yanlış (d)         : {d}")
print(f"  Toplam test örneği              : {a + b + c + d}")

# 2x2 numpy array olarak tabloyu oluşturuyoruz — statsmodels bu formatı bekler
contingency_table = np.array([[a, b], [c, d]])

# McNemar testini uyguluyoruz
# exact=True: binom dağılımı ile kesin test — küçük örneklemlerde bile güvenilir
# Chi-kare yaklaşımından (exact=False) daha doğru sonuç verir
mcnemar_result = mcnemar(contingency_table, exact=True)

# Test sonuçlarını yazdır
test_stat = mcnemar_result.statistic
p_value = mcnemar_result.pvalue

print(f"\nMcNemar Test Sonuçları:")
print(f"  Test istatistiği : {test_stat:.4f}")
print(f"  p-değeri         : {p_value:.6f}")

# Anlamlılık düzeyi: alpha = 0.05 (sosyal bilimlerde standart eşik)
alpha = 0.05
print(f"  Anlamlılık düzeyi: α = {alpha}")

# Yorumlama: p-değerini alpha ile karşılaştır
if p_value < alpha:
    yorum = (f"İki model ({model1_name} vs {model2_name}) arasındaki performans farkı "
             f"istatistiksel olarak ANLAMLIDIR (p={p_value:.6f} < α={alpha}).")
    karar = "H0 RED — Modeller arasında anlamlı fark VAR"
else:
    yorum = (f"İki model ({model1_name} vs {model2_name}) arasındaki performans farkı "
             f"istatistiksel olarak ANLAMLI DEĞİLDİR (p={p_value:.6f} >= α={alpha}).")
    karar = "H0 KABUL — Modeller arasında anlamlı fark YOK"

print(f"\n  YORUM: {yorum}")
print(f"  KARAR: {karar}")

# McNemar sonuçlarını CSV olarak kaydet — tezin istatistiksel analiz bölümü için
mcnemar_df = pd.DataFrame([{
    'Model_1': model1_name,
    'Model_2': model2_name,
    'Both_Correct_a': int(a),
    'M1_Correct_M2_Wrong_b': int(b),
    'M1_Wrong_M2_Correct_c': int(c),
    'Both_Wrong_d': int(d),
    'Test_Statistic': round(test_stat, 4),
    'P_Value': round(p_value, 6),
    'Alpha': alpha,
    'Significant': p_value < alpha,
    'Interpretation': yorum,
}])

mcnemar_df.to_csv(f'{TABLES_DIR}mcnemar_test_results.csv', index=False)
print(f"\nMcNemar sonuçları kaydedildi: {TABLES_DIR}mcnemar_test_results.csv")
print("ADIM 7 TAMAMLANDI.")


# ── ADIM 8: BOOTSTRAP %95 GÜVEN ARALIKLARI ──────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 8: BOOTSTRAP %95 GÜVEN ARALIKLARI")
print("=" * 70)

# En iyi model Faz 3'te belirlendi ve best_model.pkl olarak kaydedildi
# Random Forest: CV F1 = 0.848 ile en yüksek skoru aldı
# Test F1'e göre değil, CV F1'e göre seçim yapıyoruz — çünkü CV daha güvenilir
best_model_name = "Random Forest"
best_model = joblib.load(f'{MODELS_DIR}best_model.pkl')

print(f"En iyi model (Faz 3'ten): {best_model_name}")

# En iyi modelin test seti tahminlerini alıyoruz
y_pred_best = predictions[best_model_name]

# y_test'i numpy array'e çeviriyoruz — bootstrap örneklemede indeks sorunu olmasın
y_test_arr = y_test.values

# Bootstrap yöntemi:
# Test setinden aynı boyutta "yerine koyarak" (with replacement) alt örneklemler çekiyoruz
# Her örneklem için F1 hesaplıyoruz
# 1000 F1 değerinin %2.5 ve %97.5 percentile'ları güven aralığını verir
# Bu yöntem dağılım varsayımı gerektirmez — non-parametrik güven aralığı

# Tekrarlanabilirlik için seed'i tekrar sabitleyelim
np.random.seed(42)

# 1000 iterasyon: istatistiksel olarak yeterli, hesaplama süresi makul
n_iterations = 1000
n_samples = len(y_test_arr)
boot_f1_scores = []

print(f"Bootstrap iterasyon sayısı: {n_iterations}")
print(f"Örneklem boyutu (her iterasyonda): {n_samples}")
print("Bootstrap hesaplanıyor...")

for i in range(n_iterations):
    # Yerine koyarak rastgele indeksler seçiyoruz
    # Bazı örnekler birden fazla seçilir, bazıları hiç seçilmez — bu bootstrap'ın özü
    indices = np.random.choice(n_samples, size=n_samples, replace=True)

    # Seçilen indekslerdeki gerçek ve tahmin değerleriyle F1 hesapla
    # zero_division=0: eğer bootstrap örneklemde sadece bir sınıf varsa hata vermesin
    boot_f1 = f1_score(y_test_arr[indices], y_pred_best[indices], zero_division=0)
    boot_f1_scores.append(boot_f1)

# Bootstrap F1 dağılımının istatistikleri
boot_f1_scores = np.array(boot_f1_scores)
f1_mean = np.mean(boot_f1_scores)
f1_std = np.std(boot_f1_scores)

# %95 güven aralığı: 2.5. ve 97.5. yüzdelikler
# "%95 olasılıkla gerçek F1 değeri bu aralıktadır" anlamına gelir
ci_lower = np.percentile(boot_f1_scores, 2.5)
ci_upper = np.percentile(boot_f1_scores, 97.5)

# Orijinal (bootstrap olmadan) F1 skoru
original_f1 = f1_score(y_test_arr, y_pred_best)

print(f"\nBootstrap Sonuçları ({best_model_name}):")
print(f"  Orijinal F1 skoru     : {original_f1:.4f}")
print(f"  Bootstrap F1 ortalama : {f1_mean:.4f}")
print(f"  Bootstrap F1 std      : {f1_std:.4f}")
print(f"  %95 Güven Aralığı     : [{ci_lower:.4f} - {ci_upper:.4f}]")
print(f"\n  → {best_model_name} F1 = {original_f1:.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")

# Bootstrap dağılım histogramını çiziyoruz
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram: 1000 bootstrap F1 değerinin dağılımı
# bins=50: yeterli çözünürlük, normal dağılıma yakınlık gözlemlenebilir
# edgecolor='black': barlar arası sınır çizgisi — okunabilirlik
ax.hist(boot_f1_scores, bins=50, color=sns.color_palette('colorblind')[0],
        edgecolor='black', alpha=0.7)

# Orijinal F1 değerini dikey kırmızı çizgiyle göster
ax.axvline(x=original_f1, color='red', linestyle='-', lw=2,
           label=f'Orijinal F1 = {original_f1:.4f}')

# %95 güven aralığı sınırlarını kesikli dikey çizgilerle göster
ax.axvline(x=ci_lower, color='orange', linestyle='--', lw=2,
           label=f'%2.5 = {ci_lower:.4f}')
ax.axvline(x=ci_upper, color='orange', linestyle='--', lw=2,
           label=f'%97.5 = {ci_upper:.4f}')

ax.set_xlabel('F1 Skoru', fontsize=13)
ax.set_ylabel('Frekans', fontsize=13)
ax.set_title(f'{best_model_name} - Bootstrap F1 Dağılımı (%95 Güven Aralığı)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}11_bootstrap_f1.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nBootstrap histogramı kaydedildi: {FIGURES_DIR}11_bootstrap_f1.png")
print("ADIM 8 TAMAMLANDI.")


# ── ADIM 9: FEATURE IMPORTANCE (ÖZELLİK ÖNEMİ) ─────────────────────────────

print("\n" + "=" * 70)
print("ADIM 9: FEATURE IMPORTANCE (ÖZELLİK ÖNEMİ)")
print("=" * 70)

# feature_importances_: Random Forest ve XGBoost gibi ağaç tabanlı modellerde
# her özelliğin sınıflandırmaya ne kadar katkıda bulunduğunu ölçer
# Gini impurity azalmasına (MDI) dayanır — özellik ne kadar çok
# saf düğümler oluşturuyorsa o kadar önemlidir
print(f"En iyi model: {best_model_name}")

importances = best_model.feature_importances_
feature_names = X_test.columns.tolist()

# Özellik önemlerini DataFrame'e çeviriyoruz — sıralama ve kaydetme kolay olsun
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Önem skoruna göre azalan sıralama — en etkili özellik üstte
importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

# CSV olarak kaydet — tezde tablo olarak kullanılacak
importance_df.to_csv(f'{TABLES_DIR}feature_importance.csv', index=False)
print(f"\nTüm özellik önemleri kaydedildi: {TABLES_DIR}feature_importance.csv")

# En önemli 15 özelliği seçiyoruz — grafik okunabilirliği için
top15 = importance_df.head(15).copy()

print(f"\nEn Önemli 15 Özellik ({best_model_name}):")
for idx, row in top15.iterrows():
    print(f"  {idx+1:2d}. {row['Feature']:40s} : {row['Importance']:.4f}")

# Yatay bar grafiği — büyükten küçüğe (üstte en önemli)
fig, ax = plt.subplots(figsize=(10, 8))

# iloc[::-1]: sırayı tersine çeviriyoruz çünkü barh alt'tan üste çizer
# En önemli özelliğin grafiğin üstünde olmasını istiyoruz
top15_reversed = top15.iloc[::-1]

ax.barh(top15_reversed['Feature'], top15_reversed['Importance'],
        color=sns.color_palette('colorblind')[0], edgecolor='black', alpha=0.8)

ax.set_xlabel('Önem Skoru (Gini Impurity Azalması)', fontsize=13)
ax.set_title(f'{best_model_name} - En Önemli 15 Özellik',
             fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}12_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFeature importance grafiği kaydedildi: {FIGURES_DIR}12_feature_importance.png")
print("ADIM 9 TAMAMLANDI.")


# ── ADIM 10: SHAP ANALİZİ ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 10: SHAP ANALİZİ (Bu adım 3-5 dakika sürebilir)")
print("=" * 70)

# SHAP (SHapley Additive exPlanations):
# Oyun teorisinden gelen Shapley değerlerini makine öğrenmesine uyarlar
# Her özelliğin her bir tahmin üzerindeki katkısını hesaplar
# Feature Importance'tan farkı: SHAP yön (pozitif/negatif etki) de gösterir
# Örn: "Yüksek MonthlyCharges churn olasılığını ARTIRIR" bilgisini verir

# TreeExplainer: ağaç tabanlı modeller (RF, XGBoost, LightGBM) için
# optimize edilmiş SHAP hesaplayıcı — çok daha hızlı
print(f"\nSHAP hesaplanıyor: {best_model_name}")
print("Bu işlem birkaç dakika sürebilir, sabırlı olun...")

explainer = shap.TreeExplainer(best_model)

# shap_values: her test örneği için her özelliğin SHAP değerini hesaplar
# Binary sınıflandırmada liste döner: [sınıf_0_değerleri, sınıf_1_değerleri]
# Biz sınıf 1'i (Churn=Yes) kullanacağız
shap_values = explainer.shap_values(X_test)

# SHAP API versiyonuna göre çıktı formatı değişebilir
# Eski versiyon: list of 2 arrays → [class_0, class_1]
# Yeni versiyon: tek array döndürebilir
if isinstance(shap_values, list):
    # İki sınıf var — pozitif sınıf (Churn=1) SHAP değerlerini alıyoruz
    shap_vals = shap_values[1]
    print("SHAP değerleri hesaplandı (ikili sınıf formatı).")
else:
    # Tek array dönmüş — doğrudan kullanıyoruz
    shap_vals = shap_values
    print("SHAP değerleri hesaplandı (tek array formatı).")

print(f"SHAP values boyutu: {shap_vals.shape}")

# --- SHAP Beeswarm (Summary) Plot ---
# Her nokta bir test örneğini temsil eder
# X ekseni: SHAP değeri (tahmine katkı miktarı, pozitif = churn artırır)
# Y ekseni: özellikler (önem sırasına göre)
# Renk: özelliğin gerçek değeri (kırmızı = yüksek, mavi = düşük)
# Bu grafik hem özellik önemini hem de etkinin yönünü gösterir
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_test, show=False)
plt.title(f'{best_model_name} - SHAP Özet Grafiği (Beeswarm)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}13_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close('all')

print(f"SHAP beeswarm grafiği kaydedildi: {FIGURES_DIR}13_shap_summary.png")

# --- SHAP Bar Plot ---
# Her özelliğin ortalama |SHAP| değerini bar grafiği olarak gösterir
# Beeswarm'dan daha basit — tez sunumunda kullanmak için ideal
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_test, plot_type='bar', show=False)
plt.title(f'{best_model_name} - SHAP Özellik Önemi (Ortalama |SHAP|)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}14_shap_bar.png', dpi=300, bbox_inches='tight')
plt.close('all')

print(f"SHAP bar grafiği kaydedildi: {FIGURES_DIR}14_shap_bar.png")
print("ADIM 10 TAMAMLANDI.")


# ── ADIM 11: TEZ ÖZET TABLOSU ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 11: TEZ ÖZET TABLOSU")
print("=" * 70)

# Tezin "Bulgular" bölümüne doğrudan gidecek karşılaştırma tablosu
# Test seti metriklerini CV sonuçlarıyla birleştiriyoruz

# Her model için CV F1 bilgisini topluyoruz
# Baseline CV sonuçlarını sözlüğe alıyoruz — tüm 5 model için var
cv_info = {}
for _, row in baseline_results.iterrows():
    cv_info[row['Model']] = {
        'CV_F1_Mean': row['CV_F1_Mean'],
        'CV_F1_Std': row['CV_F1_Std']
    }

# Tuned modeller için CV F1 ortalamasını güncelliyoruz
# GridSearchCV'nin best_score_ değeri daha iyi (optimize edilmiş) CV F1 sonucudur
for _, row in tuned_results.iterrows():
    if row['Model'] in cv_info:
        cv_info[row['Model']]['CV_F1_Mean'] = row['Tuned_CV_F1']

# Test metrikleri ve CV bilgilerini birleştirip tek tablo oluşturuyoruz
summary_list = []
for _, row in metrics_df.iterrows():
    model_name = row['Model']
    summary_list.append({
        'Model': model_name,
        'Accuracy': row['Accuracy'],
        'Precision': row['Precision'],
        'Recall': row['Recall'],
        'F1': row['F1-Score'],
        'ROC-AUC': row['ROC-AUC'],
        'CV_F1_Mean': cv_info[model_name]['CV_F1_Mean'],
        'CV_F1_Std': cv_info[model_name]['CV_F1_Std'],
    })

# DataFrame oluştur ve F1'e göre sırala
summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values('F1', ascending=False).reset_index(drop=True)

# CSV olarak kaydet
summary_df.to_csv(f'{TABLES_DIR}thesis_summary_table.csv', index=False)
print(f"Tez özet tablosu (CSV) kaydedildi: {TABLES_DIR}thesis_summary_table.csv")

# Markdown formatında kaydet — tez yazımında doğrudan kullanılabilir
with open(f'{TABLES_DIR}thesis_summary_table.md', 'w', encoding='utf-8') as f:
    f.write('# Model Karşılaştırma Tablosu\n\n')
    f.write('| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 ± std |\n')
    f.write('|-------|----------|-----------|--------|------|---------|-------------|\n')
    for _, row in summary_df.iterrows():
        # CV F1 ± std formatı: "0.8480 ± 0.0054"
        cv_str = f"{row['CV_F1_Mean']:.4f} ± {row['CV_F1_Std']:.4f}"
        f.write(
            f"| {row['Model']} "
            f"| {row['Accuracy']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['F1']:.4f} "
            f"| {row['ROC-AUC']:.4f} "
            f"| {cv_str} |\n"
        )

print(f"Tez özet tablosu (MD) kaydedildi : {TABLES_DIR}thesis_summary_table.md")

# Tabloyu konsola da yazdıralım — kontrol amaçlı
print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
print("│              TEZ BULGULAR TABLOSU — MODEL KARŞILAŞTIRMASI                  │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print(f"\n{'Model':<22} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'ROC-AUC':>8} {'CV F1 ± std':>16}")
print("─" * 80)
for _, row in summary_df.iterrows():
    cv_str = f"{row['CV_F1_Mean']:.4f}±{row['CV_F1_Std']:.4f}"
    print(f"{row['Model']:<22} {row['Accuracy']:>8.4f} {row['Precision']:>9.4f} "
          f"{row['Recall']:>7.4f} {row['F1']:>7.4f} {row['ROC-AUC']:>8.4f} {cv_str:>16}")

print("\nADIM 11 TAMAMLANDI.")


# ── ADIM 12: ÖZET ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADIM 12: FAZ 4 ÖZET")
print("=" * 70)

# En iyi model (Random Forest) bilgilerini metrics_df'ten alıyoruz
best_row = metrics_df[metrics_df['Model'] == best_model_name].iloc[0]
print(f"\n★ EN İYİ MODEL: {best_model_name}")
print(f"  Accuracy  : {best_row['Accuracy']:.4f}")
print(f"  Precision : {best_row['Precision']:.4f}")
print(f"  Recall    : {best_row['Recall']:.4f}")
print(f"  F1-Score  : {best_row['F1-Score']:.4f}")
print(f"  ROC-AUC   : {best_row['ROC-AUC']:.4f}")
print(f"  PR-AUC    : {best_row['PR-AUC']:.4f}")

# McNemar sonucu
print(f"\n★ McNEMAR'S TEST ({model1_name} vs {model2_name}):")
print(f"  p-değeri : {p_value:.6f}")
print(f"  {yorum}")

# Bootstrap CI
print(f"\n★ BOOTSTRAP %95 GÜVEN ARALIĞI ({best_model_name}):")
print(f"  F1 = {original_f1:.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")

# Kaydedilen dosyalar
print(f"""
Kaydedilen dosyalar:
  TABLOLAR:
    {TABLES_DIR}model_full_metrics.csv       — 5 modelin kapsamlı metrikleri
    {TABLES_DIR}mcnemar_test_results.csv     — McNemar test sonuçları
    {TABLES_DIR}feature_importance.csv       — Özellik önem sıralaması
    {TABLES_DIR}thesis_summary_table.csv     — Tez özet tablosu (CSV)
    {TABLES_DIR}thesis_summary_table.md      — Tez özet tablosu (Markdown)
  
  GRAFİKLER:
    {FIGURES_DIR}08_confusion_matrices.png   — 5 model confusion matrix
    {FIGURES_DIR}09_roc_curves.png           — ROC eğrileri karşılaştırması
    {FIGURES_DIR}10_pr_curves.png            — PR eğrileri karşılaştırması
    {FIGURES_DIR}11_bootstrap_f1.png         — Bootstrap F1 dağılımı
    {FIGURES_DIR}12_feature_importance.png   — En önemli 15 özellik
    {FIGURES_DIR}13_shap_summary.png         — SHAP beeswarm grafiği
    {FIGURES_DIR}14_shap_bar.png             — SHAP bar grafiği
""")

print("=" * 70)
print("FAZ 4 TAMAMLANDI — Tüm istatistiksel analizler bitti.")
print("=" * 70)
print("\nSonraki: Faz 5 — Segmentasyon (segmentation.py)")
