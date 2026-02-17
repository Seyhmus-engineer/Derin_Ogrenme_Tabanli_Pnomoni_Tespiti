# 🫁 AI-Rad: Derin Öğrenme ile Pnömoni (Zatürre) Teşhis Sistemi

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-green)
![Teknofest](https://img.shields.io/badge/Status-Teknofest%20Projesi-red)

> **"Hekimler için güvenilir bir ikinci görüş."**

AI-Rad, Teknofest Sağlıkta Yapay Zeka kategorisi kapsamında
geliştirilmiş; akciğer röntgen (X-Ray) görüntülerinden **Pnömoni
(Zatürre)** tespiti yapan ve karar sürecini destekleyen bir Yapay Zeka
sistemidir.

Bu sistem yalnızca sınıflandırma yapmakla kalmaz; **Akciğer
Segmentasyonu + Grad-CAM** teknolojileri ile modelin karar verirken
hangi bölgelere odaklandığını hekime görsel olarak sunar.

⚠️ Not: Bu sistem klinik teşhis koymaz. Karar destek ve tarama
(screening) amaçlıdır.

------------------------------------------------------------------------

## 🎯 Proje Özeti ve Yenilikçi Yönler

Bu projede standart derin öğrenme yaklaşımının ötesine geçilerek
**Klinik Ön İşleme Boru Hattı (Clinical Preprocessing Pipeline)**
geliştirilmiştir.

### 🔍 Akciğer Odaklanması (Lung ROI Crop)

-   torchxrayvision PSPNet ile akciğer segmentasyonu
-   Boyun, omuz, siyah padding alanlarının çıkarılması
-   Shortcut learning riskinin azaltılması

### ⚖️ Kontrast Standardizasyonu (CLAHE)

-   Farklı cihazlardan gelen röntgenlerin ışık dengesinin normalize
    edilmesi
-   Klinik tutarlılığın artırılması

### 🚫 Shortcut Learning Engelleme

Modelin: - Kemik yapılarına - Cihaz kablolarına - R/L marker
etiketlerine

odaklanmasını engelleyerek yalnızca **akciğer parankimine** dikkat
etmesi sağlanmıştır.

------------------------------------------------------------------------

## 🏆 Başarı Metrikleri

  -----------------------------------------------------------------------
  Metrik             Değer            Klinik Anlamı
  ------------------ ---------------- -----------------------------------
  **Recall           **%98.2**        Hasta vakaları kaçırma oranı
  (Duyarlılık)**                      minimuma indirildi

  **Accuracy         **%93.0**        Genel teşhis başarısı
  (Doğruluk)**                        

  **F1-Score**       **%94.6**        Dengeli ve güvenilir performans
  -----------------------------------------------------------------------

🎯 Özellikle Recall yüksek tutulmuştur (Yanlış negatifleri azaltmak
için).

------------------------------------------------------------------------

## 🏗️ Model Mimarisi

### 1️⃣ Ön İşleme (Preprocessing)

-   PSPNet ile Akciğer Segmentasyonu
-   CLAHE Kontrast Eşitleme
-   Otomatik ROI Crop
-   Border Removal
-   RandomResizedCrop
-   RandomErasing

### 2️⃣ Sınıflandırma (Classification)

-   Backbone: EfficientNet-B0 (ImageNet Pretrained)
-   Optimizer: AdamW (lr=2e-4)
-   Scheduler: Linear Warmup + Cosine Decay
-   Loss: Weighted CrossEntropy (Class Imbalance için)
-   AMP (Mixed Precision): Aktif
-   Early Stopping: Aktif

### 3️⃣ Açıklanabilirlik (Explainable AI - XAI)

-   Grad-CAM
-   Masked Heatmap Overlay
-   Klinik uyumlu görselleştirme paneli

------------------------------------------------------------------------

## 🔥 Grad-CAM Açıklaması

Model, zatürreyi noktasal bir lezyon olarak değil; çoğunlukla **bölgesel
yoğunlaşma (diffüz opasite)** olarak tespit eder.

Bu yaklaşım zatürrenin klinik doğası ile uyumludur.

Grad-CAM çıktıları: - Yalnızca akciğer maskesi içinde gösterilir - Blur
sonrası tekrar maskelenir - Yanlış yorumlamaların önüne geçilir

------------------------------------------------------------------------

## 🖥️ Arayüz (Streamlit Demo)

Proje, Streamlit ile geliştirilmiş kullanıcı dostu bir arayüze sahiptir.

Özellikler:

-   📂 Röntgen Yükleme (.jpg, .png)
-   🧠 Otomatik Analiz
-   🔥 Grad-CAM Görselleştirme
-   📝 Akıllı Klinik Ön Rapor (Normal / Şüpheli / Yüksek Risk)
-   📥 Rapor İndirme

Çalıştırmak için:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🚀 Kurulum

``` bash
git clone https://github.com/KULLANICI_ADIN/AI-Rad-Pneumonia.git
cd AI-Rad-Pneumonia

pip install torch torchvision
pip install torchxrayvision
pip install streamlit
pip install opencv-python
pip install tqdm
```

------------------------------------------------------------------------

## ▶️ Model Eğitimi

``` bash
python train_single.py
```

## ▶️ Grad-CAM Üretimi

``` bash
python gradcam_lungmasked_batch_final.py
```

------------------------------------------------------------------------

## 🔬 Gelecek Geliştirmeler

-   YOLO tabanlı lezyon tespiti
-   U-Net ile gerçek segmentasyon
-   Multi-class sınıflandırma (Bacterial / Viral / COVID)
-   DICOM desteği
-   PACS entegrasyonu

------------------------------------------------------------------------

## ⚠️ Yasal Uyarı

Bu proje:

-   Klinik teşhis koymaz
-   Radyolog yerine geçmez
-   Araştırma ve eğitim amaçlı geliştirilmiştir

------------------------------------------------------------------------

## 👨‍💻 Geliştirici

Şeyhmus Elik\
Computer Engineering\
Medical AI & Deep Learning

## 📜 Lisans

This project is licensed under the MIT License.
