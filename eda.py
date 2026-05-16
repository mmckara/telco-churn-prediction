# =============================================================================
# FAZ 1: KEŞİFÇİ VERİ ANALİZİ (EDA)
# Proje: Telekomünikasyon Müşteri Kaybı (Churn) Tahmini
# Veri Seti: IBM Telco Customer Churn (7043 müşteri, 21 değişken)
# Çalıştırma: python eda.py  (venv aktif olmalı)
# =============================================================================

# ── ADIM 1: KÜTÜPHANELERİ YÜKLE ─────────────────────────────────────────────

# pandas: veriyi tablo (DataFrame) formatında okumak ve işlemek için
import pandas as pd

# numpy: sayısal hesaplamalar için
import numpy as np

# matplotlib: temel grafik çizim kütüphanesi
import matplotlib.pyplot as plt

# seaborn: istatistiksel görselleştirme; daha güzel grafikler için
import seaborn as sns

# scipy.stats: Chi-square ve Mann-Whitney U gibi istatistiksel testler için
from scipy import stats

# warnings: gereksiz uyarı mesajlarını gizlemek için
import warnings
warnings.filterwarnings('ignore')

# os: klasör oluşturma işlemleri için
import os

# Tüm grafikler için renk körü dostu palet kullanıyoruz (akademik standart)
sns.set_palette('colorblind')

# Grafiklerin font büyüklüğünü ve kaydetme kalitesini tez baskısına uygun ayarlıyoruz
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Grafiklerin ve tabloların kaydedileceği klasör yollarını tanımlıyoruz
FIGURES_DIR = 'results/figures/'
TABLES_DIR  = 'results/tables/'

# Klasörler yoksa oluşturuyoruz (exist_ok=True hata vermemesini sağlar)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 60)
print("ADIM 1 TAMAMLANDI: Kütüphaneler yüklendi.")
print(f"Grafikler klasörü: {os.path.abspath(FIGURES_DIR)}")
print("=" * 60)


# ── ADIM 2: VERİYİ YÜKLE VE İLK İNCELEME ────────────────────────────────────

# CSV dosyasını pandas DataFrame olarak okuyoruz
# eda.py proje kökünde, data/ klasörü de kök dizinde, bu yüzden doğrudan yol yazıyoruz
df = pd.read_csv('data/Telco-Customer-Churn.csv')

print("\n" + "=" * 60)
print("ADIM 2: VERİ YÜKLEME VE İLK İNCELEME")
print("=" * 60)

# Kaç satır ve kaç sütun olduğunu kontrol ediyoruz; beklenen: (7043, 21)
print(f"\nVeri Seti Boyutu (satır, sütun): {df.shape}")

# İlk 5 satırı yazdırıyoruz - verinin genel görünümünü tanımak için
print("\nİlk 5 Satır:")
print(df.head().to_string())

# Her sütunun adını, kaç null içerdiğini ve veri tipini yazdırıyoruz
# DİKKAT: TotalCharges burada 'object' (string) görünecek - bu bir veri kalitesi sorunudur
print("\nSütun Bilgileri (df.info()):")
df.info()

# Sayısal sütunlar için temel istatistikleri hesaplıyoruz
print("\nTemel İstatistikler (sayısal sütunlar):")
print(df.describe().to_string())

# Kategorik sütunların özet istatistikleri
print("\nKategorik Sütun İstatistikleri:")
print(df.describe(include='object').to_string())

print("\nADIM 2 TAMAMLANDI.")


# ── ADIM 3: TOTALCHARGES SÜTUNUNUN TİP KONTROLÜ ──────────────────────────────

print("\n" + "=" * 60)
print("ADIM 3: TOTALCHARGES TİP KONTROLÜ")
print("=" * 60)

# TotalCharges sütununun gerçek veri tipini kontrol ediyoruz
# DATA_DICTIONARY.md'ye göre bu sütun sayısal olmasına rağmen STRING olarak geliyor
# Sebebi: bazı satırlarda sayı yerine BOŞLUK karakteri var
# Python tüm sütunda boşluk gördüğünde onu sayısal değil 'object' (string) olarak işaretliyor
print(f"\nTotalCharges sütununun veri tipi: {df['TotalCharges'].dtype}")
print("Beklenen: float64 veya int64")
print("Gerçek   : object (string)")
print("NEDEN: Bazı satırlarda sayı yerine boşluk karakteri (' ') var.")

# str.strip() ile baştaki/sondaki boşlukları temizleyip boş string kalanları buluyoruz
# DATA_DICTIONARY.md'ye göre bu 11 satır tenure=0 olan YENİ müşterilere aittir
bos_satirlar = df[df['TotalCharges'].str.strip() == '']
print(f"\nBoşluk içeren satır sayısı: {len(bos_satirlar)}")
print("\nBu satırların içeriği (customerID, tenure, MonthlyCharges, TotalCharges, Churn):")
print(bos_satirlar[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].to_string())
print("\nSONUÇ: Tüm boş satırlarda tenure=0 — bu müşteriler henüz 1 ayını doldurmamış.")
print("Bu satırları SİLMEYECEĞİZ; Faz 2'de medyan ile dolduracağız.")

print("\nADIM 3 TAMAMLANDI.")


# ── ADIM 4: EKSİK VERİ ANALİZİ ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 4: EKSİK VERİ ANALİZİ")
print("=" * 60)

# Analiz için TotalCharges'ı geçici olarak sayısala çeviriyoruz
# errors='coerce': dönüştürülemeyen değerleri (boşlukları) NaN'a çevirir
# NOT: Bu KALICI bir değişiklik değil — asıl dönüşüm Faz 2'de yapılacak
df_analiz = df.copy()
df_analiz['TotalCharges'] = pd.to_numeric(df_analiz['TotalCharges'], errors='coerce')

# Her sütundaki NaN (eksik) değerleri sayıyoruz ve oranını hesaplıyoruz
eksik_veri  = df_analiz.isnull().sum()
eksik_yuzde = (df_analiz.isnull().sum() / len(df_analiz) * 100).round(4)

eksik_tablo = pd.DataFrame({
    'Eksik Sayi': eksik_veri,
    'Eksik Yuzde (%)': eksik_yuzde
})

print("\nEksik Veri Özeti (eksik içeren sütunlar):")
eksik_olan = eksik_tablo[eksik_tablo['Eksik Sayi'] > 0]
print(eksik_olan.to_string() if len(eksik_olan) > 0 else "  Hiç eksik veri yok!")
print(f"\nToplam sütun sayısı           : {df_analiz.shape[1]}")
print(f"Eksik veri içeren sütun sayısı : {(eksik_tablo['Eksik Sayi'] > 0).sum()}")

# Eksik veri dağılımını bar chart ile görselleştiriyoruz
# Sadece 1 sütunda eksik veri var (TotalCharges), bu yüzden grafik sade olacak
fig, ax = plt.subplots(figsize=(8, 4))
eksik_plot = eksik_tablo[eksik_tablo['Eksik Sayi'] > 0]
bars = ax.bar(eksik_plot.index, eksik_plot['Eksik Sayi'],
              color=sns.color_palette('colorblind')[0], edgecolor='black', linewidth=0.5)
ax.set_title('Sutunlara Gore Eksik Veri Sayisi', fontsize=14)
ax.set_xlabel('Sutun Adi', fontsize=12)
ax.set_ylabel('Eksik Veri Sayisi', fontsize=12)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}01_eksik_veri.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nGrafik kaydedildi: {FIGURES_DIR}01_eksik_veri.png")

print("\nADIM 4 TAMAMLANDI.")


# ── ADIM 5: HEDEF DEĞİŞKEN (CHURN) DAĞILIMI ──────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 5: HEDEF DEĞİŞKEN (CHURN) DAĞILIMI")
print("=" * 60)

# Churn sütunundaki Yes/No değerlerini sayıyoruz ve yüzdelerini hesaplıyoruz
churn_sayim = df['Churn'].value_counts()
churn_yuzde = df['Churn'].value_counts(normalize=True) * 100

print(f"\nChurn = No  (müşteri kalmış):   {churn_sayim['No']:>5} kişi  (%{churn_yuzde['No']:.2f})")
print(f"Churn = Yes (müşteri ayrılmış): {churn_sayim['Yes']:>5} kişi  (%{churn_yuzde['Yes']:.2f})")
print(f"\nÇoğunluk / Azınlık oranı: {churn_sayim['No'] / churn_sayim['Yes']:.2f}x")
print("ÖNEMLI: Veri seti DENGESİZDİR! Faz 2'de SMOTE uygulanacak.")

# Churn dağılımını bar chart ve pasta grafik olarak yan yana çiziyoruz
# Bar chart sayıyı, pasta oranı gösterir — her ikisi de tezde işe yarar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
renkler = sns.color_palette('colorblind', 2)

# Sol grafik: mutlak sayı
bars = ax1.bar(churn_sayim.index, churn_sayim.values, color=renkler,
               edgecolor='black', linewidth=0.5)
ax1.set_title('Musteri Kaybi Dagilimi - Sayi', fontsize=14)
ax1.set_xlabel('Churn Durumu', fontsize=12)
ax1.set_ylabel('Musteri Sayisi', fontsize=12)
ax1.set_ylim(0, max(churn_sayim.values) * 1.2)
for bar, (label, sayi) in zip(bars, churn_sayim.items()):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f'{sayi}\n(%{churn_yuzde[label]:.1f})',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Sağ grafik: pasta (oransal görünüm)
ax2.pie(churn_sayim.values,
        labels=['Kalmis (No)', 'Ayrilmis (Yes)'],
        autopct='%1.1f%%', colors=renkler, startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
ax2.set_title('Musteri Kaybi Dagilimi - Oran', fontsize=14)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}02_churn_dagilimi.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafik kaydedildi: {FIGURES_DIR}02_churn_dagilimi.png")

print("\nADIM 5 TAMAMLANDI.")


# ── ADIM 6: SAYISAL DEĞİŞKEN ANALİZİ ─────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 6: SAYISAL DEĞİŞKEN ANALİZİ")
print("=" * 60)

# TotalCharges'ı sayısala çeviriyoruz (bu noktadan itibaren kalıcı olarak)
# errors='coerce': boşluklar NaN'a dönüşür; Faz 2'de bunlar medyan ile doldurulacak
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Sayısal sütunları ve Türkçe etiketlerini tanımlıyoruz
sayisal_sutunlar = ['tenure', 'MonthlyCharges', 'TotalCharges']
sutun_etiketleri = {
    'tenure': 'Musteri Suresi (Ay)',
    'MonthlyCharges': 'Aylik Ucret (USD)',
    'TotalCharges': 'Toplam Ucret (USD)'
}

print("\nGruplar arası ortalama karşılaştırması:")
for s in sayisal_sutunlar:
    ort_no  = df[df['Churn'] == 'No'][s].mean()
    ort_yes = df[df['Churn'] == 'Yes'][s].mean()
    print(f"  {s:>20}: Kalmis={ort_no:.1f}, Ayrilmis={ort_yes:.1f}")

# ── 6a: Histogram + KDE ──────────────────────────────────────────────────────
# Her sayısal değişkenin dağılımını Churn durumuna göre karşılaştırıyoruz
# KDE (Kernel Density Estimate): histogramın yumuşatılmış hali
# İki grubu üst üste bindiriyoruz — gruplar ne kadar ayrışırsa değişken o kadar güçlüdür
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Sayisal Degiskenlerin Churn Durumuna Gore Dagilimi (Histogram + KDE)', fontsize=14)
for ax, sutun in zip(axes, sayisal_sutunlar):
    sns.histplot(data=df[df['Churn'] == 'No'], x=sutun, kde=True, ax=ax,
                 label='Kalmis (No)', alpha=0.6, color=sns.color_palette('colorblind')[0])
    sns.histplot(data=df[df['Churn'] == 'Yes'], x=sutun, kde=True, ax=ax,
                 label='Ayrilmis (Yes)', alpha=0.6, color=sns.color_palette('colorblind')[1])
    ax.set_xlabel(sutun_etiketleri[sutun], fontsize=12)
    ax.set_ylabel('Frekans', fontsize=12)
    ax.set_title(sutun_etiketleri[sutun], fontsize=13)
    ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}03_sayisal_histogram_kde.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nGrafik kaydedildi: {FIGURES_DIR}03_sayisal_histogram_kde.png")

# ── 6b: Boxplot (Churn'a göre) ───────────────────────────────────────────────
# Boxplot: medyan, çeyrekler ve aykırı değerleri aynı anda gösterir
# İki kutunun pozisyonu birbirinden ne kadar ayrıysa, değişken o kadar iyi tahmin edicidir
# Başlığa Mann-Whitney U p-değerini ekliyoruz — istatistiksel anlam hemen görülsün
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Sayisal Degiskenler: Churn Durumuna Gore Boxplot', fontsize=14)
for ax, sutun in zip(axes, sayisal_sutunlar):
    sns.boxplot(data=df, x='Churn', y=sutun, ax=ax,
                palette='colorblind', order=['No', 'Yes'])
    ax.set_xlabel('Churn Durumu', fontsize=12)
    ax.set_ylabel(sutun_etiketleri[sutun], fontsize=12)
    ax.set_xticklabels(['Kalmis (No)', 'Ayrilmis (Yes)'], fontsize=11)
    # Mann-Whitney U testi: normal dağılım varsayımı gerektirmez
    # H0: iki grubun dağılımı aynıdır; p < 0.05 ise gruplar istatistiksel olarak farklıdır
    grup_no  = df[df['Churn'] == 'No'][sutun].dropna()
    grup_yes = df[df['Churn'] == 'Yes'][sutun].dropna()
    _, p_val = stats.mannwhitneyu(grup_no, grup_yes, alternative='two-sided')
    ax.set_title(f'{sutun_etiketleri[sutun]}\n(Mann-Whitney p={p_val:.2e})', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}04_sayisal_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafik kaydedildi: {FIGURES_DIR}04_sayisal_boxplot.png")

# ── 6c: Korelasyon Heatmap ───────────────────────────────────────────────────
# Pearson korelasyon matrisi: sayısal değişkenler arasındaki doğrusal ilişkiyi ölçer
# |r| > 0.8 → multicollinearity riski (Logistic Regression'da sorun çıkarabilir)
# Churn'ü de 0/1 olarak dahil ediyoruz — hedefle hangi değişkenin en çok ilişkili olduğunu görmek için
df_kor = df[sayisal_sutunlar].copy()
df_kor['Churn (0/1)'] = (df['Churn'] == 'Yes').astype(int)
korelasyon_matrisi = df_kor.corr(method='pearson')

print("\nPearson Korelasyon Matrisi:")
print(korelasyon_matrisi.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(korelasyon_matrisi,
            annot=True, fmt='.3f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, ax=ax,
            cbar_kws={'label': 'Pearson Korelasyon Katsayisi'})
ax.set_title('Sayisal Degiskenler Arasi Pearson Korelasyon Matrisi', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}05_korelasyon_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafik kaydedildi: {FIGURES_DIR}05_korelasyon_heatmap.png")

print("\nADIM 6 TAMAMLANDI.")


# ── ADIM 7: KATEGORİK DEĞİŞKEN ANALİZİ ──────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 7: KATEGORİK DEĞİŞKEN ANALİZİ")
print("=" * 60)

# customerID (sadece tanımlayıcı) ve SeniorCitizen (0/1 sayısal) listeye dahil edilmiyor
kategorik_sutunlar = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# ── 7a: Chi-square Bağımsızlık Testi ────────────────────────────────────────
# Her kategorik değişken için Chi-square testi uyguluyoruz
# H0 (null hipotez): değişken ile Churn BAĞIMSIZ (aralarında ilişki yok)
# p < 0.05 → H0 reddedilir → anlamlı ilişki VAR
# Chi-square istatistiği ne kadar büyükse, ilişki o kadar güçlüdür
chi_square_sonuclar = []
for sutun in kategorik_sutunlar:
    capraz_tablo = pd.crosstab(df[sutun], df['Churn'])
    chi2, p_degeri, sd, _ = stats.chi2_contingency(capraz_tablo)
    chi_square_sonuclar.append({
        'Degisken': sutun,
        'Chi2': round(chi2, 2),
        'p_degeri': round(p_degeri, 6),
        'SerbestlikDerecesi': sd,
        'Anlamli_mi': 'EVET' if p_degeri < 0.05 else 'HAYIR'
    })

# Chi-square değerine göre büyükten küçüğe sıralıyoruz
chi_df = pd.DataFrame(chi_square_sonuclar).sort_values('Chi2', ascending=False).reset_index(drop=True)

print("\nChi-Square Bağımsızlık Testi Sonuçları (Churn ile İlişki Gücüne Göre):")
print(chi_df.to_string(index=False))

# Tabloyu CSV olarak kaydediyoruz — tezde tablo olarak kullanılacak
chi_df.to_csv(f'{TABLES_DIR}chi_square_sonuclari.csv', index=False, encoding='utf-8-sig')
print(f"\nTablo kaydedildi: {TABLES_DIR}chi_square_sonuclari.csv")

# ── 7b: Kategorik Churn Oranları Bar Chart ───────────────────────────────────
# Her kategorik değişken için Churn=Yes oranını gösteren bar chart çiziyoruz
# 15 değişkeni 5 satır x 3 sütunluk ızgaraya yerleştiriyoruz
fig, axes = plt.subplots(5, 3, figsize=(18, 26))
fig.suptitle('Kategorik Degiskenlere Gore Churn Oranlari (%)', fontsize=16, y=1.005)
axes = axes.flatten()
for idx, sutun in enumerate(kategorik_sutunlar):
    # Her kategorideki Churn=Yes yüzdesini hesaplıyoruz
    churn_oran = df.groupby(sutun)['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).reset_index()
    churn_oran.columns = [sutun, 'Churn_Orani']
    churn_oran = churn_oran.sort_values('Churn_Orani', ascending=False)
    bars = axes[idx].bar(range(len(churn_oran)), churn_oran['Churn_Orani'],
                         color=sns.color_palette('colorblind', len(churn_oran)),
                         edgecolor='black', linewidth=0.3)
    axes[idx].set_xticks(range(len(churn_oran)))
    axes[idx].set_xticklabels(churn_oran[sutun], rotation=25, ha='right', fontsize=9)
    # p-değerini başlığa ekliyoruz — okuyucu istatistiksel önemi hemen görsün
    p_val = chi_df[chi_df['Degisken'] == sutun]['p_degeri'].values[0]
    axes[idx].set_title(f'{sutun}  (p={p_val:.4f})', fontsize=11)
    axes[idx].set_ylabel('Churn Orani (%)', fontsize=10)
    axes[idx].set_ylim(0, 105)
    for bar in bars:
        axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                       f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}06_kategorik_churn_oranlari.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafik kaydedildi: {FIGURES_DIR}06_kategorik_churn_oranlari.png")

# ── 7c: En Etkili 3 Değişken — Yığılmış Bar Chart ───────────────────────────
# Yığılmış bar: hem No hem Yes oranını aynı anda gösterir; her kategorinin toplamı %100'dür
onemli_sutunlar = ['Contract', 'InternetService', 'PaymentMethod']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('En Etkili 3 Kategorik Degisken: Yigilik Churn Dagilimi', fontsize=14)
for ax, sutun in zip(axes, onemli_sutunlar):
    # normalize='index': her kategorinin kendi toplamı %100 olacak şekilde normalleştirir
    ct = pd.crosstab(df[sutun], df['Churn'], normalize='index') * 100
    ct[['No', 'Yes']].plot(kind='bar', stacked=True, ax=ax,
                           color=sns.color_palette('colorblind', 2),
                           edgecolor='black', linewidth=0.5)
    ax.set_title(sutun, fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('Oran (%)', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=10)
    ax.legend(['Kalmis (No)', 'Ayrilmis (Yes)'], fontsize=10)
    ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}07_onemli_kategorikler_yigilik.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafik kaydedildi: {FIGURES_DIR}07_onemli_kategorikler_yigilik.png")

print("\nADIM 7 TAMAMLANDI.")


# ── ADIM 8: EDA BULGULARI ÖZETİ ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 8: EDA BULGULARI ÖZETİ")
print("=" * 60)

churn_no_sayi  = churn_sayim['No']
churn_yes_sayi = churn_sayim['Yes']
toplam         = len(df)

print(f"""
BULGU 1 — Sinif Dengesizligi:
  Churn=No  : {churn_no_sayi} kisi (%{churn_no_sayi/toplam*100:.1f})
  Churn=Yes : {churn_yes_sayi} kisi (%{churn_yes_sayi/toplam*100:.1f})
  Oran      : {churn_no_sayi/churn_yes_sayi:.2f}x — SMOTE gerekli!

BULGU 2 — TotalCharges Veri Kalitesi:
  11 satirda bossluk var (tenure=0 olan yeni musteriler)
  Cozum: medyan ile doldurma (Faz 2)

BULGU 3 — En Kritik Kategorik Degisken:
  Contract (Chi2={chi_df[chi_df['Degisken']=='Contract']['Chi2'].values[0]})
  Month-to-month ~%42 churn, Two year ~%3 churn

BULGU 4 — tenure Guclu Ayrim Yapar:
  Ayrilmis ortalama tenure : {df[df['Churn']=='Yes']['tenure'].mean():.1f} ay
  Kalmis  ortalama tenure  : {df[df['Churn']=='No']['tenure'].mean():.1f} ay
  (Mann-Whitney p<0.001)

BULGU 5 — Fiber Optik Yuksek Churn:
  ~%42 churn (DSL ~%19, internet yok ~%7)

BULGU 6 — Multicollinearity Riski:
  TotalCharges ile tenure arasi Pearson r = {korelasyon_matrisi.loc['TotalCharges','tenure']:.3f}
  Logistic Regression'da VIF kontrolu yapilmali

BULGU 7 — MonthlyCharges:
  Ayrilmis medyan: {df[df['Churn']=='Yes']['MonthlyCharges'].median():.1f} USD
  Kalmis  medyan : {df[df['Churn']=='No']['MonthlyCharges'].median():.1f} USD

BULGU 8 — Elektronik Cek Odeme → Yuksek Churn

BULGU 9 — TechSupport ve OnlineSecurity Koruyucu Etki Gosteriyor

BULGU 10 — gender Churn ile Anlamsiz:
  p = {chi_df[chi_df['Degisken']=='gender']['p_degeri'].values[0]} (p>0.05 → ilişki yok)
""")

print("=" * 60)
print("FAZ 1 EDA TAMAMLANDI.")
print(f"Kaydedilen grafikler ({FIGURES_DIR}):")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  - {f}")
print(f"\nKaydedilen tablolar ({TABLES_DIR}):")
for f in sorted(os.listdir(TABLES_DIR)):
    print(f"  - {f}")
print("=" * 60)
print("Sonraki adim: Faz 2 - Veri On Isleme (preprocessing.py)")
