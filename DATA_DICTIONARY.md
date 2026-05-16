# IBM TELCO CHURN VERİ SETİ - DEĞİŞKEN SÖZLÜĞÜ

Toplam 7043 satır, 21 sütun.

## TANIMLAYICI DEĞİŞKEN

| Sütun | Tip | Açıklama | MODELDE KULLAN? |

|-------|-----|----------|-----------------|

| customerID | String | Benzersiz müşteri kimliği (örn: 7590-VHVEG) | HAYIR - sadece referans |

## DEMOGRAFİK DEĞİŞKENLER

| Sütun | Tip | Değerler | Açıklama |

|-------|-----|----------|----------|

| gender | Kategorik | Male, Female | Müşteri cinsiyeti |

| SeniorCitizen | İkili (0/1) | 0, 1 | 65 yaş üstü mü? **ZATEN 0/1, encode etme** |

| Partner | Kategorik | Yes, No | Eşi/partneri var mı? |

| Dependents | Kategorik | Yes, No | Bağımlısı (çocuk vs.) var mı? |

## ABONELİK DEĞİŞKENLERİ

| Sütun | Tip | Değerler | Açıklama |

|-------|-----|----------|----------|

| tenure | Sayısal | 0-72 | Müşterinin firma ile kaç AY çalıştığı. **0 = 1 aydan az müşteri, eksik veri DEĞİL** |

| Contract | Kategorik | Month-to-month, One year, Two year | Sözleşme tipi |

| PaperlessBilling | Kategorik | Yes, No | Kağıtsız faturalandırma |

| PaymentMethod | Kategorik | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Ödeme yöntemi |

## SERVİS DEĞİŞKENLERİ

| Sütun | Tip | Değerler | Açıklama |

|-------|-----|----------|----------|

| PhoneService | Kategorik | Yes, No | Telefon hizmeti var mı? |

| MultipleLines | Kategorik | Yes, No, No phone service | Birden fazla hat |

| InternetService | Kategorik | DSL, Fiber optic, No | İnternet hizmet tipi |

| OnlineSecurity | Kategorik | Yes, No, No internet service | Çevrimiçi güvenlik |

| OnlineBackup | Kategorik | Yes, No, No internet service | Çevrimiçi yedekleme |

| DeviceProtection | Kategorik | Yes, No, No internet service | Cihaz koruması |

| TechSupport | Kategorik | Yes, No, No internet service | Teknik destek |

| StreamingTV | Kategorik | Yes, No, No internet service | TV yayın hizmeti |

| StreamingMovies | Kategorik | Yes, No, No internet service | Film yayın hizmeti |

## FİNANSAL DEĞİŞKENLER

| Sütun | Tip | Değerler | Açıklama |

|-------|-----|----------|----------|

| MonthlyCharges | Sayısal | 18.25 - 118.75 | Aylık ödeme tutarı (USD) |

| TotalCharges | Sayısal (AMA STRING gelir!) | 18.80 - 8684.80 | Toplam ödeme. **DİKKAT: 11 satırda BOŞLUK var, NaN'a çevrilip sayısala dönüştürülmeli. Boş olanlar tenure=0 olan müşteriler** |

## HEDEF DEĞİŞKEN

| Sütun | Tip | Değerler | Açıklama |

|-------|-----|----------|----------|

| Churn | Kategorik | Yes, No | Müşteri ayrıldı mı? **MODELLEMEDE: Yes=1, No=0** |

## KRİTİK NOTLAR

### Sınıf Dengesizliği

Churn=No: 5174 müşteri (%73.46)  

Churn=Yes: 1869 müşteri (%26.54)  

→ SMOTE veya class_weight='balanced' MUTLAKA kullanılmalı.

### "No internet service" Değeri

6 sütunda görülür (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies). Bu değer aslında "InternetService=No" durumunu yansıtır. Encoding sırasında "No" ile aynı muamele edilebilir, ya da ayrı kategori bırakılabilir. **Tezde her iki yaklaşımı da karşılaştırmak iyi bir tartışma sağlar.**

### "No phone service" Değeri

MultipleLines sütununda. PhoneService=No olan müşterilerde görülür.

### TotalCharges Eksik Veri

11 satırda boşluk var (tenure=0 olan yeni müşteriler). Bu satırlar:

- Silinmemeli (önemli bilgi içerebilir)

- MonthlyCharges × tenure formülüyle hesaplanmamalı (tenure=0)

- 0 ile doldurulabilir VEYA median ile doldurulabilir

- **Tezde: median ile doldurma + ayrı bir "yeni müşteri" flag'i** önerilir

### Çoklu Bağlantı (Multicollinearity) Uyarısı

TotalCharges = MonthlyCharges × tenure ilişkisi var. VIF (Variance Inflation Factor) kontrolü yapılmalı. Logistic Regression için TotalCharges veya tenure düşürülebilir.