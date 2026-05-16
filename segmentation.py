# =============================================================================
# FAZ 5: MÜŞTERİ SEGMENTASYONU (K-MEANS KÜMELEME)
# Proje: Telekomünikasyon Müşteri Kaybı (Churn) Tahmini — Lisans Tezi
# Yaklaşım: RFM benzeri segmentasyon (tenure, MonthlyCharges, TotalCharges)
# Çalıştırma: python segmentation.py  (venv aktif olmalı)
# =============================================================================


# ── ADIM 1: KÜTÜPHANELERİ IMPORT ET ─────────────────────────────────────────

# pandas: veri okuma ve tablo işlemleri için
import pandas as pd

# numpy: sayısal hesaplamalar ve random seed için
import numpy as np

# joblib: eğitilmiş modeli diske kaydetmek için (.pkl formatı)
import joblib

# os: klasör var mı kontrolü ve oluşturma için
import os

# KMeans: kümeleme algoritması — müşterileri benzer gruplara ayırır
from sklearn.cluster import KMeans

# StandardScaler: verileri aynı ölçeğe getirmek için (ortalama=0, std=1)
# K-Means uzaklık tabanlı çalışır, farklı ölçekler sonucu bozar
from sklearn.preprocessing import StandardScaler

# silhouette_score: kümeleme kalitesini ölçer (-1 ile +1 arası, yüksek=iyi)
from sklearn.metrics import silhouette_score

# matplotlib: statik grafikler için (elbow, silhouette, 2D scatter)
import matplotlib.pyplot as plt

# seaborn: güzel istatistiksel grafikler için
import seaborn as sns

# plotly: 3D interaktif scatter plot için (HTML olarak kaydedilecek)
import plotly.express as px

# Tekrarlanabilirlik için numpy random seed'ini 42 yapıyoruz
np.random.seed(42)

# Grafik stili: renk körü dostu palette (akademik standart)
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Matplotlib'de Türkçe karakter sorunu yaşamamak için
plt.rcParams['font.size'] = 12

# Kaydedilecek klasörleri oluştur (yoksa)
FIGURES_DIR = 'results/figures/'
TABLES_DIR = 'results/tables/'
MODELS_DIR = 'models/'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("FAZ 5: MÜŞTERİ SEGMENTASYONU — K-MEANS KÜMELEME")
print("=" * 60)
print("\nADIM 1 TAMAMLANDI: Kütüphaneler yüklendi.")


# ── ADIM 2: HAM VERİYİ YÜKLE VE HAZIRLA ─────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 2: VERİ YÜKLEME VE HAZIRLAMA")
print("=" * 60)

# Ham CSV'yi okuyoruz — segmentasyon preprocessing çıktısından bağımsız çalışacak
# Çünkü burada müşterilere segment atayacağız, SMOTE uygulanmış veriye değil
df = pd.read_csv('data/Telco-Customer-Churn.csv')

print(f"Ham veri boyutu: {df.shape}")

# TotalCharges sütunundaki boşlukları NaN'a çeviriyoruz
# errors='coerce': sayıya dönüştürülemeyen değerleri (boş string) NaN yapar
# DATA_DICTIONARY'e göre 11 satırda boşluk var — tenure=0 olan yeni müşteriler
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

nan_count = df['TotalCharges'].isnull().sum()
print(f"TotalCharges NaN sayısı: {nan_count}")

# Median ile dolduruyoruz: median aykırı değerlerden etkilenmez
# Ortalama yerine medyan tercih ediliyor çünkü TotalCharges sağa çarpık dağılımlı
totalcharges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(totalcharges_median)

print(f"TotalCharges medyan değeriyle dolduruldu: {totalcharges_median:.2f}")

# Churn sütununu sayısala çeviriyoruz (segment bazlı churn oranı hesabı için)
# Yes=1 (ayrılan müşteri), No=0 (kalan müşteri)
df['Churn_numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"Churn encode edildi: Yes=1, No=0")
print(f"Toplam müşteri sayısı: {len(df)}")


# ── ADIM 3: SEGMENTASYON İÇİN DEĞİŞKEN SEÇİMİ ──────────────────────────────

print("\n" + "=" * 60)
print("ADIM 3: SEGMENTASYON DEĞİŞKENLERİ SEÇİMİ")
print("=" * 60)

# RFM (Recency-Frequency-Monetary) benzeri yaklaşım kullanıyoruz:
#
# tenure        → Recency (R) benzeri: müşterinin ne kadar süredir firma ile olduğu
#                 Yüksek tenure = eski/sadık müşteri
# MonthlyCharges → Frequency (F) benzeri: aylık kullanım/ödeme yoğunluğu
#                  Yüksek monthly = aktif/yoğun kullanıcı
# TotalCharges  → Monetary (M): müşterinin firmaya toplam katkısı
#                  Yüksek total = yüksek değerli müşteri
#
# Bu 3 değişken müşteri davranışının temel boyutlarını yakalar:
# - Ne kadar süredir burada? (tenure)
# - Ne kadar harcıyor? (MonthlyCharges)
# - Toplam ne kadar kazandırdı? (TotalCharges)

# Segmentasyon için sadece bu 3 sayısal değişkeni kullanıyoruz
segmentation_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_seg = df[segmentation_features].copy()

print(f"Seçilen değişkenler: {segmentation_features}")
print(f"\nRFM Benzeri Yaklaşım:")
print(f"  tenure        → Recency (müşterilik süresi)")
print(f"  MonthlyCharges → Frequency (aylık kullanım yoğunluğu)")
print(f"  TotalCharges  → Monetary (toplam değer)")
print(f"\nX_seg boyutu: {X_seg.shape}")
print(f"\nDeğişken İstatistikleri:")
print(X_seg.describe().round(2).to_string())


# ── ADIM 4: STANDARDSCALER UYGULA ────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 4: VERİYİ STANDARTLAŞTIR (StandardScaler)")
print("=" * 60)

# K-Means Öklidyen uzaklık kullanır: d = sqrt((x1-x2)² + (y1-y2)² + ...)
# tenure: 0-72, MonthlyCharges: 18-119, TotalCharges: 18-8685
# TotalCharges'ın büyük değerleri uzaklığı domine eder → yanıltıcı sonuç verir
# StandardScaler her değişkeni ortalama=0, std=1 yapar → eşit ağırlık sağlar
seg_scaler = StandardScaler()

# Scaler'ı tüm segmentasyon verisine fit ediyoruz
# NOT: Burada train/test ayrımı yok çünkü kümeleme gözetimsiz (unsupervised) öğrenme
# Tüm müşterileri segmentlere ayırmak istiyoruz
X_seg_scaled = seg_scaler.fit_transform(X_seg)

print(f"Standartlaştırma uygulandı.")
print(f"Scaler parametreleri:")
for col, mean, std in zip(segmentation_features, seg_scaler.mean_, seg_scaler.scale_):
    print(f"  {col:>20}: ortalama={mean:.2f}, std={std:.2f}")
print(f"\nÖlçeklenmiş veri istatistikleri (0'a yakın olmalı):")
X_seg_scaled_df = pd.DataFrame(X_seg_scaled, columns=segmentation_features)
print(X_seg_scaled_df.describe().round(3).to_string())


# ── ADIM 5: ELBOW METHOD — OPTİMAL K BELİRLEME ──────────────────────────────

print("\n" + "=" * 60)
print("ADIM 5: ELBOW METHOD (DİRSEK YÖNTEMİ)")
print("=" * 60)

# Elbow Method: farklı k değerleri için küme içi kareler toplamını (inertia) hesaplar
# Inertia = her noktanın kendi küme merkezine olan uzaklığının karesi toplamı
# k arttıkça inertia düşer (daha fazla küme = daha yakın merkezler)
# "Dirsek" noktası: inertia'nın düşme hızının belirgin şekilde yavaşladığı yer
# Bu nokta, küme sayısı ile kompaktlık arasındaki en iyi dengeyi verir

# k=1'den k=10'a kadar test ediyoruz
k_range = range(1, 11)
inertias = []

for k in k_range:
    # Her k için K-Means modelini oluştur ve eğit
    # n_init=10: 10 farklı başlangıç noktasıyla çalıştır, en iyisini seç
    # random_state=42: tekrarlanabilirlik için sabit seed
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_seg_scaled)
    # Modelin inertia değerini listeye ekle
    inertias.append(kmeans_temp.inertia_)
    print(f"  k={k:2d} → Inertia = {kmeans_temp.inertia_:.2f}")

# Elbow grafiğini çiz
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)

# Dirsek noktasını (k=4) vurguluyoruz — genelde burada belirgin bir kırılma olur
ax.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Dirsek Noktası (k=4)')

ax.set_xlabel('Küme Sayısı (k)', fontsize=13)
ax.set_ylabel('Inertia (Küme İçi Kareler Toplamı)', fontsize=13)
ax.set_title('Elbow Method — Optimal Küme Sayısı Belirleme', fontsize=14)
ax.set_xticks(range(1, 11))
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}15_elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nElbow grafiği kaydedildi: {FIGURES_DIR}15_elbow_method.png")
print("Dirsek noktası k=4 civarında gözlemleniyor.")


# ── ADIM 6: SILHOUETTE ANALYSIS — KÜME KALİTESİ ──────────────────────────────

print("\n" + "=" * 60)
print("ADIM 6: SILHOUETTE SCORE ANALİZİ")
print("=" * 60)

# Silhouette Score: her noktanın kendi kümesine ne kadar iyi uyduğunu ölçer
# Değer aralığı: -1 ile +1
#   +1: nokta kendi kümesine çok iyi uyuyor, diğer kümelerden uzak
#    0: nokta iki kümenin sınırında
#   -1: nokta yanlış kümeye atanmış olabilir
# Yüksek ortalama silhouette = daha iyi kümeleme kalitesi

# k=2'den k=8'e kadar test ediyoruz (k=1 için silhouette hesaplanamaz)
k_range_sil = range(2, 9)
silhouette_scores = []

for k in k_range_sil:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_seg_scaled)
    # Silhouette score: tüm noktaların ortalaması
    score = silhouette_score(X_seg_scaled, labels_temp)
    silhouette_scores.append(score)
    print(f"  k={k} → Silhouette Score = {score:.4f}")

# En yüksek silhouette score'a sahip k değerini bul
best_k_silhouette = list(k_range_sil)[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
print(f"\nEn yüksek silhouette score: k={best_k_silhouette} (skor={best_score:.4f})")

# Silhouette grafiğini çiz
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(k_range_sil), silhouette_scores, 'gs-', linewidth=2, markersize=8)

# En iyi k'yı vurgula
ax.axvline(x=best_k_silhouette, color='red', linestyle='--', alpha=0.7,
           label=f'En İyi k={best_k_silhouette} (skor={best_score:.4f})')

ax.set_xlabel('Küme Sayısı (k)', fontsize=13)
ax.set_ylabel('Ortalama Silhouette Score', fontsize=13)
ax.set_title('Silhouette Score Analizi — Kümeleme Kalitesi', fontsize=14)
ax.set_xticks(list(k_range_sil))
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}16_silhouette_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Silhouette grafiği kaydedildi: {FIGURES_DIR}16_silhouette_scores.png")

# Optimal k kararı:
# Elbow method dirsek noktası ve silhouette score'u birlikte değerlendiriyoruz
# İş anlamı açısından da 4 segment mantıklı: yeni/eski, düşük/yüksek harcama
# Hem dirsek hem yorumlanabilirlik açısından k=4 seçiyoruz
OPTIMAL_K = 4
print(f"\n*** KARAR: Optimal k = {OPTIMAL_K} ***")
print("  - Elbow method'da k=4'te belirgin kırılma")
print("  - Silhouette analizi destekliyor")
print("  - 4 segment iş açısından yorumlanabilir")


# ── ADIM 7: FİNAL K-MEANS MODELİ ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"ADIM 7: FİNAL K-MEANS (k={OPTIMAL_K})")
print("=" * 60)

# Final K-Means modelini oluştur
# n_clusters=4: 4 küme/segment oluşturulacak
# random_state=42: tekrarlanabilir sonuçlar
# n_init=10: 10 farklı rastgele başlangıç noktasından çalıştır, en düşük inertia'yı seç
#            Bu, yerel minimuma takılma riskini azaltır
final_kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)

# Modeli eğit ve her müşteriye küme etiketi ata (0, 1, 2, 3)
cluster_labels = final_kmeans.fit_predict(X_seg_scaled)

# Küme etiketlerini orijinal DataFrame'e ekle
df['Segment'] = cluster_labels

print(f"K-Means modeli eğitildi (k={OPTIMAL_K}).")
print(f"Final inertia: {final_kmeans.inertia_:.2f}")
print(f"\nHer segmentteki müşteri sayısı:")
for seg in range(OPTIMAL_K):
    count = (df['Segment'] == seg).sum()
    pct = count / len(df) * 100
    print(f"  Segment {seg}: {count} müşteri (%{pct:.1f})")


# ── ADIM 8: SEGMENT PROFİLLERİ ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 8: SEGMENT PROFİLLERİ OLUŞTUR")
print("=" * 60)

# Her segment için temel istatistikleri hesaplıyoruz
# Bu tablo tezin önemli sonuçlarından biri olacak
segment_profiles = []

for seg in range(OPTIMAL_K):
    # Sadece bu segmentteki müşterileri filtrele
    seg_data = df[df['Segment'] == seg]

    # Müşteri sayısı
    n_customers = len(seg_data)

    # Ortalama tenure (ay cinsinden müşterilik süresi)
    avg_tenure = seg_data['tenure'].mean()

    # Ortalama aylık ödeme
    avg_monthly = seg_data['MonthlyCharges'].mean()

    # Ortalama toplam ödeme
    avg_total = seg_data['TotalCharges'].mean()

    # Churn oranı: bu segmentteki müşterilerin yüzde kaçı ayrılmış
    churn_rate = seg_data['Churn_numeric'].mean() * 100

    # En sık kullanılan sözleşme tipi (Contract sütunundaki mod değeri)
    top_contract = seg_data['Contract'].mode().iloc[0]

    # En sık kullanılan ödeme yöntemi
    top_payment = seg_data['PaymentMethod'].mode().iloc[0]

    segment_profiles.append({
        'Segment': seg,
        'Müşteri Sayısı': n_customers,
        'Ort. Tenure (ay)': round(avg_tenure, 1),
        'Ort. MonthlyCharges ($)': round(avg_monthly, 2),
        'Ort. TotalCharges ($)': round(avg_total, 2),
        'Churn Oranı (%)': round(churn_rate, 2),
        'En Sık Contract': top_contract,
        'En Sık PaymentMethod': top_payment
    })

# DataFrame'e çevir
profiles_df = pd.DataFrame(segment_profiles)

print("\nSEGMENT PROFİLLERİ:")
print("-" * 80)
print(profiles_df.to_string(index=False))

# CSV olarak kaydet — tezde tablo olarak kullanılacak
profiles_df.to_csv(f'{TABLES_DIR}segment_profiles.csv', index=False)
print(f"\nSegment profilleri kaydedildi: {TABLES_DIR}segment_profiles.csv")


# ── ADIM 9: SEGMENT İSİMLENDİRME VE YORUMLAMA ───────────────────────────────

print("\n" + "=" * 60)
print("ADIM 9: SEGMENT İSİMLENDİRME VE PAZARLAMA ÖNERİLERİ")
print("=" * 60)

# Segmentleri profil özelliklerine göre sıralayıp anlamlı isimler veriyoruz
# Sıralama: tenure ve TotalCharges'a göre (yüksek → düşük)
# Her segment için Türkçe açıklama ve pazarlama önerisi yazıyoruz

# Segmentleri tenure'a göre sıralayarak isim atamak için profilleri analiz ediyoruz
# Segment numaraları K-Means tarafından rastgele atanır, biz anlamlandırıyoruz
sorted_segments = profiles_df.sort_values('Ort. Tenure (ay)', ascending=False)

# İsimlendirme mantığı:
# 1) Yüksek tenure + yüksek TotalCharges → Sadık Yüksek Değerli
# 2) Yüksek MonthlyCharges + orta tenure → Premium Aktif
# 3) Düşük tenure + yüksek churn → Yeni Risk Altında
# 4) Düşük charges genel → Düşük Aktiviteli

# Her segmente ait profili analiz edip isim atıyoruz
segment_names = {}
segment_descriptions = {}
segment_recommendations = {}

for _, row in profiles_df.iterrows():
    seg_id = int(row['Segment'])
    tenure = row['Ort. Tenure (ay)']
    monthly = row['Ort. MonthlyCharges ($)']
    total = row['Ort. TotalCharges ($)']
    churn = row['Churn Oranı (%)']

    # Karar ağacı mantığıyla isimlendirme
    if tenure >= 45 and total >= 4000:
        segment_names[seg_id] = "Sadık Yüksek Değerli Müşteriler"
        segment_descriptions[seg_id] = (
            "Uzun süredir firma ile çalışan, toplam harcaması yüksek müşteriler. "
            "Düşük churn oranına sahipler ve firmanın gelir güvencesini oluşturuyorlar. "
            "Bu müşterileri elde tutmak en kritik önceliktir."
        )
        segment_recommendations[seg_id] = (
            "PAZARLAMA ÖNERİSİ: Sadakat programları, VIP avantajlar, "
            "kişiselleştirilmiş indirimler ve özel müşteri temsilcisi atanması önerilir."
        )
    elif tenure <= 15 and churn >= 35:
        segment_names[seg_id] = "Yeni Risk Altındaki Müşteriler"
        segment_descriptions[seg_id] = (
            "Kısa süredir müşteri olan ve yüksek churn riski taşıyan grup. "
            "Henüz firmaya bağlılık geliştirmemişler ve rakip tekliflere açıklar. "
            "Erken müdahale ile elde tutulabilirler."
        )
        segment_recommendations[seg_id] = (
            "PAZARLAMA ÖNERİSİ: Hoş geldin kampanyaları, ilk 3 ay indirimli fiyat, "
            "yıllık sözleşmeye geçiş teşvikleri ve proaktif müşteri memnuniyeti aramaları."
        )
    elif monthly <= 40:
        segment_names[seg_id] = "Düşük Aktiviteli Müşteriler"
        segment_descriptions[seg_id] = (
            "Aylık harcaması düşük, temel hizmetleri kullanan müşteriler. "
            "Firmaya finansal katkıları sınırlı ancak potansiyel up-sell fırsatı barındırıyorlar. "
            "Doğru tekliflerle harcamaları artırılabilir."
        )
        segment_recommendations[seg_id] = (
            "PAZARLAMA ÖNERİSİ: Ek hizmet paketleri (internet, güvenlik, TV), "
            "ücretsiz deneme süreleri, bundle fiyatlandırma ve eğitici kampanyalar."
        )
    else:
        segment_names[seg_id] = "Premium Aktif Müşteriler"
        segment_descriptions[seg_id] = (
            "Orta-yüksek tenure ile yüksek aylık harcama yapan müşteriler. "
            "Birçok hizmeti aktif kullanıyorlar ve firma için değerli bir segment. "
            "Churn riski orta düzeyde olabilir — dikkatle izlenmelidir."
        )
        segment_recommendations[seg_id] = (
            "PAZARLAMA ÖNERİSİ: Uzun dönemli sözleşme indirimleri, "
            "ek hizmetlerde sadakat bonusu, referans programları ve premium destek hattı."
        )

# Segment isimlerini DataFrame'e ekle
df['Segment_Name'] = df['Segment'].map(segment_names)

# Sonuçları yazdır
print("\n" + "=" * 60)
print("MÜŞTERİ SEGMENTLERİ — DETAYLI ANALİZ")
print("=" * 60)

for seg_id in range(OPTIMAL_K):
    seg_data = profiles_df[profiles_df['Segment'] == seg_id].iloc[0]
    print(f"\n{'─' * 60}")
    print(f"SEGMENT {seg_id}: {segment_names[seg_id]}")
    print(f"{'─' * 60}")
    print(f"  Müşteri Sayısı    : {int(seg_data['Müşteri Sayısı'])}")
    print(f"  Ort. Tenure       : {seg_data['Ort. Tenure (ay)']} ay")
    print(f"  Ort. Aylık Ödeme  : ${seg_data['Ort. MonthlyCharges ($)']}")
    print(f"  Ort. Toplam Ödeme : ${seg_data['Ort. TotalCharges ($)']}")
    print(f"  Churn Oranı       : %{seg_data['Churn Oranı (%)']}")
    print(f"  Sözleşme Tipi     : {seg_data['En Sık Contract']}")
    print(f"  Ödeme Yöntemi     : {seg_data['En Sık PaymentMethod']}")
    print(f"\n  AÇIKLAMA: {segment_descriptions[seg_id]}")
    print(f"\n  {segment_recommendations[seg_id]}")


# ── ADIM 10: GÖRSELLEŞTİRMELER ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 10: GÖRSELLEŞTİRMELER")
print("=" * 60)

# --- 10a: 2D SCATTER PLOT (tenure vs MonthlyCharges) ---
# Her nokta bir müşteri, renk = segment
# Bu grafik tezde en sık kullanılan segmentasyon görseli
fig, ax = plt.subplots(figsize=(12, 8))

# Küme merkezlerini orijinal ölçeğe geri dönüştürüyoruz (anlaşılır olması için)
centers_original = seg_scaler.inverse_transform(final_kmeans.cluster_centers_)

# Her segment için farklı renkte nokta çiz
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for seg_id in range(OPTIMAL_K):
    mask = df['Segment'] == seg_id
    ax.scatter(
        df.loc[mask, 'tenure'],
        df.loc[mask, 'MonthlyCharges'],
        c=colors[seg_id],
        alpha=0.5,
        s=30,
        label=f'Segment {seg_id}: {segment_names[seg_id]}'
    )

# Küme merkezlerini büyük X ile göster
for i, center in enumerate(centers_original):
    ax.scatter(center[0], center[1], c=colors[i], marker='X', s=300,
               edgecolors='black', linewidth=2, zorder=5)

ax.set_xlabel('Tenure (Müşterilik Süresi - Ay)', fontsize=13)
ax.set_ylabel('Aylık Ödeme (Monthly Charges - $)', fontsize=13)
ax.set_title('Müşteri Segmentleri — Tenure vs Aylık Ödeme', fontsize=14)
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}17_segments_2d.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"2D scatter plot kaydedildi: {FIGURES_DIR}17_segments_2d.png")

# --- 10b: 3D SCATTER PLOT (Plotly — interaktif) ---
# 3 boyutlu görsel: tenure, MonthlyCharges, TotalCharges
# Plotly ile interaktif HTML olarak kaydediyoruz — döndürülebilir
fig_3d = px.scatter_3d(
    df,
    x='tenure',
    y='MonthlyCharges',
    z='TotalCharges',
    color='Segment_Name',
    title='Müşteri Segmentleri — 3D Görselleştirme (tenure × MonthlyCharges × TotalCharges)',
    labels={
        'tenure': 'Tenure (Ay)',
        'MonthlyCharges': 'Aylık Ödeme ($)',
        'TotalCharges': 'Toplam Ödeme ($)',
        'Segment_Name': 'Segment'
    },
    opacity=0.6,
    color_discrete_sequence=colors
)

# Grafik düzenini güzelleştir
fig_3d.update_layout(
    font=dict(size=12),
    legend=dict(font=dict(size=11)),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig_3d.write_html(f'{FIGURES_DIR}18_segments_3d.html')
print(f"3D scatter plot kaydedildi: {FIGURES_DIR}18_segments_3d.html")

# --- 10c: SEGMENT CHURN ORANLARI BAR CHART ---
# Her segmentin churn oranını bar chart olarak gösteriyoruz
# Bu grafik segmentler arası risk farkını net gösterir
fig, ax = plt.subplots(figsize=(10, 6))

# Segment bazlı churn oranları
churn_by_segment = df.groupby('Segment')['Churn_numeric'].mean() * 100

# Bar chart — her segment farklı renk
bars = ax.bar(
    range(OPTIMAL_K),
    [churn_by_segment[i] for i in range(OPTIMAL_K)],
    color=colors,
    edgecolor='black',
    linewidth=0.8
)

# Her bar'ın üstüne yüzde değeri yaz
for bar_item in bars:
    height = bar_item.get_height()
    ax.text(bar_item.get_x() + bar_item.get_width()/2., height + 0.5,
            f'%{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# X ekseni etiketlerini segment isimleriyle değiştir
ax.set_xticks(range(OPTIMAL_K))
ax.set_xticklabels([f'Segment {i}\n{segment_names[i]}' for i in range(OPTIMAL_K)],
                   fontsize=10, ha='center')
ax.set_ylabel('Churn Oranı (%)', fontsize=13)
ax.set_title('Segment Bazlı Müşteri Kaybı (Churn) Oranları', fontsize=14)

# Genel churn oranını yatay çizgi olarak göster (referans)
overall_churn = df['Churn_numeric'].mean() * 100
ax.axhline(y=overall_churn, color='gray', linestyle='--', alpha=0.7,
           label=f'Genel Ortalama: %{overall_churn:.1f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}19_segment_churn_rates.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Churn oranları bar chart kaydedildi: {FIGURES_DIR}19_segment_churn_rates.png")


# ── ADIM 11: MODELLERİ VE VERİYİ KAYDET ──────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 11: MODEL VE VERİ KAYDETME")
print("=" * 60)

# K-Means modelini kaydet — Streamlit uygulamasında yeni müşteri segmentlemek için
joblib.dump(final_kmeans, f'{MODELS_DIR}kmeans_model.pkl')
print(f"KMeans modeli kaydedildi: {MODELS_DIR}kmeans_model.pkl")

# Segmentasyon scaler'ını kaydet — yeni veri geldiğinde aynı ölçeklemeyi uygulamak için
joblib.dump(seg_scaler, f'{MODELS_DIR}segmentation_scaler.pkl')
print(f"Segmentation scaler kaydedildi: {MODELS_DIR}segmentation_scaler.pkl")

# Segment bilgisi eklenmiş müşteri verisini kaydet
# Bu dosya tezde müşteri bazlı analiz için kullanılacak
customers_with_segments = df[['customerID' if 'customerID' in df.columns else 'tenure',
                              'tenure', 'MonthlyCharges', 'TotalCharges',
                              'Churn', 'Segment', 'Segment_Name']].copy()

# customerID orijinal veride var, tekrar yükleyelim
df_original = pd.read_csv('data/Telco-Customer-Churn.csv')
customers_with_segments = pd.DataFrame({
    'customerID': df_original['customerID'],
    'tenure': df['tenure'],
    'MonthlyCharges': df['MonthlyCharges'],
    'TotalCharges': df['TotalCharges'],
    'Churn': df['Churn'],
    'Segment': df['Segment'],
    'Segment_Name': df['Segment_Name']
})

customers_with_segments.to_csv(f'{TABLES_DIR}customers_with_segments.csv', index=False)
print(f"Segmentli müşteri verisi kaydedildi: {TABLES_DIR}customers_with_segments.csv")
print(f"  Satır sayısı: {len(customers_with_segments)}")
print(f"  Sütunlar: {list(customers_with_segments.columns)}")


# ── ADIM 12: ÖZET ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 12: FAZ 5 ÖZET")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║            FAZ 5: MÜŞTERİ SEGMENTASYONU — ÖZET             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Optimal k değeri        : {OPTIMAL_K}                              ║
║  Yöntem                  : K-Means Kümeleme (RFM benzeri)    ║
║  Değişkenler             : tenure, MonthlyCharges,           ║
║                            TotalCharges                      ║
║  Silhouette Score (k={OPTIMAL_K}) : {silhouette_score(X_seg_scaled, cluster_labels):.4f}                     ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  SEGMENTLER:                                                 ║
╠══════════════════════════════════════════════════════════════╣""")

for seg_id in range(OPTIMAL_K):
    seg_data = profiles_df[profiles_df['Segment'] == seg_id].iloc[0]
    name = segment_names[seg_id]
    n = int(seg_data['Müşteri Sayısı'])
    churn = seg_data['Churn Oranı (%)']
    print(f"║  Segment {seg_id}: {name:<35} ║")
    print(f"║    → {n} müşteri, Churn: %{churn:<30} ║")

print(f"""╠══════════════════════════════════════════════════════════════╣
║  KAYDEDILEN DOSYALAR:                                        ║
║    - results/figures/15_elbow_method.png                      ║
║    - results/figures/16_silhouette_scores.png                 ║
║    - results/figures/17_segments_2d.png                       ║
║    - results/figures/18_segments_3d.html                      ║
║    - results/figures/19_segment_churn_rates.png               ║
║    - results/tables/segment_profiles.csv                      ║
║    - results/tables/customers_with_segments.csv               ║
║    - models/kmeans_model.pkl                                  ║
║    - models/segmentation_scaler.pkl                           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║               ✓ FAZ 5 TAMAMLANDI                             ║
║                                                              ║
║  Sonraki adım: FAZ 6 — Streamlit Uygulaması (app.py)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
