# AI-Financial-Risk-Analyzer
Türkiye'deki halka açık şirketlerin finansal risklerini XGBoost kullanarak tahmin eden makine öğrenmesi projesi.


Bu proje, Türkiye'deki halka açık şirketlerin finansal oranlarını ve göstergelerini analiz ederek, şirketlerin **finansal risk seviyelerini tahmin etmeyi** amaçlayan uçtan uca bir Makine Öğrenmesi projesidir. Projede, bağımsız denetçi görüşlerinden yola çıkılarak şirketlerin finansal sıkıntı yaşama ihtimalleri modellenmiş ve Yapay Zeka teknikleriyle bu kararların arkasındaki nedenler ortaya konmuştur.

## İş Problemi ve Amacı
Finans sektöründe bir şirketin kriz durumuna gireceğini önceden tahmin etmek (Risk Yönetimi) hayati önem taşır. Geleneksel analiz yöntemleri çok zaman alır ve karmaşık finansal ilişkileri gözden kaçırabilir. 

Bu projenin temel amacı:
* Makine öğrenmesi kullanarak şirketlerin risk durumlarını otomatize edilmiş bir şekilde tahmin etmek.
* Finansal krizleri gözden kaçırmanın maliyeti çok yüksek olduğu için modelin **Duyarlılık (Recall)** oranını maksimize etmek.
* Hangi finansal göstergelerin (Cari Oran, Özkaynak vb.) riski tetiklediğini matematiksel olarak kanıtlamak.

## Makine Öğrenmesi Yaklaşımı ve Geliştirme Adımları
Proje, standart bir modelleme çalışmasından ziyade dengesiz veri setleriyle (Imbalanced Data) başa çıkmaya ve iş hedeflerine (Business Objectives) uygun olarak optimize edilmiştir.

1. **Veri Hazırlığı ve EDA:** 10.751 satırlık veri seti temizlenmiş, `Görüş Tipi` değişkeni hedef değişkene (`Risk: 1, Risksiz: 0`) dönüştürülmüştür. Sınıf dengesizliği (%90 Risksiz, %10 Riskli) tespit edilmiştir.
2. **Özellik Seçimi (Feature Selection):** 50'den fazla finansal metrik arasından, modele en çok katkı sağlayan en iyi 20 değişken (Top 20 Features) seçilerek gürültü azaltılmıştır.
3. **Hiperparametre Optimizasyonu:** XGBoost algoritması kullanılarak `RandomizedSearchCV` ile model karmaşıklığı, öğrenme oranı ve ağaç derinliği optimize edilmiştir. Sınıf dengesizliği `scale_pos_weight` parametresi ile çözülmüştür.
4. **Eşik Değeri (Threshold) Optimizasyonu:** Finansal bir felaketi gözden kaçırmamak adına, Recall değerini Precision'dan daha çok önemseyen **F2-Score** hesaplanmış ve en optimal karar eşiği (0.30 civarı) matematiksel olarak belirlenmiştir.

## Model Performansı (Gerçekleşen Sonuçlar)
Eşik değeri optimizasyonu sonrası modelimiz, gerçekte riskli olan şirketlerin **%81'ini önceden tespit etmeyi başarmıştır.** 
| Sınıf           | Precision (Kesinlik)  | Recall (Duyarlılık) | F1-Score |
| :---            | :---:                 | :---:               | :---:    |
| **0 (Risksiz)** | 0.97                  | 0.85                | 0.91     |
| **1 (Riskli)**  | **0.39**              | **0.81**            | **0.53** |
| *Accuracy*      | -                     | -                   | *0.85*   |

*Not: Finansal risk yönetiminde "Yanlış Alarm" (False Positive) vermenin maliyeti, batmakta olan bir şirketi gözden kaçırmanın (False Negative) maliyetinden çok daha düşüktür. Bu sebeple %81 Recall oranı, bu iş problemi için çok başarılı bir sonuçtur.*

## Finansal İçgörüler ve Açıklanabilirlik
Modelimizin keşfettiği en önemli 2 finansal içgörü şunlardır:

1. **Ana Ortaklığa Ait Özkaynaklar (Equity):** Model, özkaynağı erimiş veya eksiye düşmüş şirketleri yüksek riskli (1) olarak sınıflandırmayı matematiksel olarak kendi kendine öğrenmiştir. Şirketin finansal şoklara karşı tamponu kalmaması en büyük risk faktörü olarak öne çıkmıştır.
2. **L Model Skoru:** Tekil finansal oranlar yerine, kompozit bir endeks olan iflas/sıkıntı skorunun (L Skoru) karar sürecinde en belirleyici metriklerden biri olduğu saptanmıştır.
