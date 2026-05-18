# 🎨 Real-Time Neural Style Transfer (Gerçek Zamanlı Sanatsal Stil Aktarımı)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=flat&logo=opencv)

## 📌 Proje Hakkında
[cite_start]Bu proje, bir fotoğrafın içeriğini koruyarak üzerine belirli bir sanatsal stili gerçek zamanlı olarak uygulayabilen bir derin öğrenme modelidir[cite: 2]. [cite_start]Klasik yöntemlerin aksine, bu sistem bir kez eğitildikten sonra herhangi bir yeni fotoğrafı saniyenin onda biri gibi kısa bir sürede sanatsal bir esere dönüştürür[cite: 3]. [cite_start]Kullanıcı fotoğrafını yüklediği anda, eğitilmiş Transformer ağı stili anında uygular ve sonucu web arayüzünde gösterir[cite: 4].

## 🚀 Temel Özellikler ve Çözülen Problemler
* [cite_start]**Hız ve Verimlilik:** Klasik yöntemlerin dakikalar süren işlem süresi, milisaniyelere indirilerek gerçek zamanlı video işleme veya anlık fotoğraf filtreleme mümkün kılınır[cite: 14].
* [cite_start]**Akıllı Stil Aktarımı:** Transformer mimarisi sayesinde, stil aktarılırken fotoğrafın semantik bütünlüğü bozulmaz, nesneler tanınmaya devam eder[cite: 18].
* [cite_start]**Öğrenme ve Genelleme:** Sistem sadece iki görsel arasındaki piksel farkını değil, bir stilin genel karakteristiklerini öğrenir[cite: 15, 16].
* [cite_start]**Üretken Yapay Zeka Deneyimi:** Sadece veri tahmini değil, "Generative AI" mantığıyla yeni ve sanatsal içerikler üretilir[cite: 17].

## ⚙️ Teknik Mimari ve Yaklaşım
[cite_start]Bu projede, her görüntü için ayrı bir optimizasyon yapmak yerine, bir Görüntü Dönüştürücü Ağ (Image Transformation Network) eğitilir[cite: 6]. Sistem iki temel mimari üzerinden çalışır:

### 1. Transformer Network (Dönüştürücü Ağ)
* [cite_start]Binlerce resimden oluşan bir veri seti üzerinde eğitilir[cite: 7].
* [cite_start]Amacı, girdi olarak aldığı herhangi bir resmi, hedeflediği sanat tarzına (örneğin Van Gogh) dönüştürmeyi "öğrenmektir"[cite: 8].

### 2. Loss Network (Kayıp Ağı - VGG19)
[cite_start]Eğitim aşamasında modelin performansını ölçmek için dondurulmuş bir VGG19 modeli kullanılır[cite: 9]. [cite_start]Bu ağ eğitilmez, sadece "öğretmen" görevi görerek şu hesaplamaları yapar[cite: 10]:
* [cite_start]**İçerik Kaybı (Content Loss):** Üretilen görselin orijinal nesne yapısını koruyup korumadığını denetler[cite: 11].
* [cite_start]**Stil Kaybı (Style Loss):** Hedef tablonun renk ve doku karakteristiklerinin ne kadar aktarıldığını hesaplar[cite: 12].








