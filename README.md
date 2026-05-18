## Gerçek Zamanlı Sanatsal Stil Aktarımı

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=flat&logo=opencv)

## Proje Hakkında 
Bu proje, derin öğrenme (Deep Learning) ve Evrişimsel Sinir Ağları (CNN) kullanarak, bir fotoğrafın orijinal içeriğini ve nesne yapısını korurken seçilen sanatsal bir stili anında uygulayan bir **Gerçek Zamanlı Stil Aktarımı** sistemidir. 

Kullanıcının yüklediği fotoğraflar saniyenin onda biri gibi bir sürede, semantik bütünlüğü bozulmadan dönüştürülerek web arayüzünde gösterilir. Proje, "Generative AI" mantığıyla sadece iki görsel arasındaki piksel farkını değil, seçilen stilin genel karakteristiklerini öğrenen bir mimari üzerine kuruludur.

##  Beklenen Çıktı ve Kullanım Alanları
Zaman alıcı ve teknik bilgi gerektiren fotoğraf düzenleme işlemlerini saniyelere indiren bu sistem;
* Sosyal medya uygulamaları ve kamera filtreleri,
* Dijital sanat ve tasarım platformları,
* Reklam ve medya sektöründe kullanılmak üzere tasarlanmıştır.

##  Kullanılan Teknolojiler 
* **Programlama Dili:** Python
* **Derin Öğrenme Framework'ü:** PyTorch / TensorFlow 
* **Görüntü İşleme:** OpenCV, NumPy
* **Yöntemler:** CNN, Transfer Learning, Style Transfer Algorithms

##  Teknik Mimari ve Veri Akışı
Projenin görüntü işleme ve dönüşüm aşamaları şu temel adımlardan oluşur:

1. **`load_image()`:** Ham görüntünün dizi (array) formatında sisteme yüklenmesi.
2. **`preprocess_image()`:** Görüntünün modelin beklediği çok boyutlu tensör (tensor) formatına dönüştürülmesi ve normalize edilmesi.
3. **`transform_image()`:** Önceden eğitilmiş ağ üzerinden stil aktarımı işleminin gerçekleştirilmesi.
4. **`calculate_loss()`:** İçerik (content) ve stil (style) kayıplarının hesaplanarak optimizasyonun sağlanması.
5. **`save_output()`:** Çıktı tensörlerinin tekrar NumPy matrislerine dönüştürülerek görselleştirilmesi ve kaydedilmesi.






