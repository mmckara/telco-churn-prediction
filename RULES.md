# TEKNİK KURALLAR VE STANDARTLAR

## KOD STİLİ

- Python 3.11 kullanılacak

- PEP-8 stiline uyulacak

- Tüm değişken/fonksiyon adları İngilizce, snake_case

- Türkçe karakter SADECE yorumlarda

## REPRODUCIBILITY (TEKRARLANABİLİRLİK)

- Her random işlemde random_state=42

- Numpy random.seed(42) kullan

- Train/test split'ten önce her zaman shuffle

## VERİ İŞLEME KURALLARI

- Train'de görmediğin veriyi test'te fit etme (data leakage!)

- Scaler train'de fit, test'te sadece transform

- SMOTE SADECE train setine uygulanır, test'e ASLA

- Cross-validation içinde SMOTE uygula (Pipeline kullan)

## DOSYA KAYDETME KURALLARI

- Tüm modeller models/ klasörüne, .pkl uzantısı ile

- Tüm grafikler results/figures/ klasörüne, .png olarak, dpi=300

- Tablolar results/tables/ klasörüne, .csv olarak

## YORUM KURALLARI

- Her fonksiyonun docstring'i olacak

- Her satır kodun üstünde yorum olacak (yeni başlayan öğrenci için)

- Yorum dili: Türkçe

- Karmaşık matematiksel işlemlerde formül de yaz

## GÖRSELLEŞTİRME STANDARTLARI

- Tüm grafiklerde: başlık (TR), eksen etiketleri (TR), legend

- Font boyutu en az 12

- dpi=300 ile kaydet (tez baskısı için)

- Renk paleti: seaborn 'colorblind' (renk körü dostu, akademik standart)

- Tek model için: tek grafik. Karşılaştırma için: aynı grafikte üst üste/yan yana

## CROSS-VALIDATION

- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

- Stratify=y her zaman (dengesiz sınıflar için)

- scoring='f1' (accuracy değil!) - dengesiz veri yüzünden

## HİPERPARAMETRE OPTİMİZASYONU

- GridSearchCV kullan (yeni başlayan için RandomizedSearchCV'den daha öğretici)

- cv=5, scoring='f1', n_jobs=-1

- Sadece en iyi 2-3 model için yap (zaman tasarrufu)

## NOTEBOOK STANDARTLARI

- Her notebook bir markdown başlıkla başlasın

- Bölümler markdown hücreleriyle ayrılsın

- Çıktıları temizleme - jüri için temiz görünüm