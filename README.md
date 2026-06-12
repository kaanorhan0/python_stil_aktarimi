# Gerçek Zamanlı Yapay Zeka Stil Aktarımı (Neural Style Transfer)

Bu proje, VGG19 derin öğrenme modeli kullanılarak kullanıcıların yüklediği fotoğrafların ünlü ressamların (Van Gogh, Picasso vb.) sanatsal stillerine dönüştürüldüğü bir görüntü işleme motorudur.

## 🛠️ Kullanılan Kütüphaneler ve Teknolojiler
- *Python (3.8+)*: Ana programlama dili.
- *PyTorch & Torchvision*: Derin öğrenme altyapısı, tensör işlemleri ve önceden eğitilmiş VGG19 modelinin (Feature Extractor) kullanımı için.
- *Streamlit*: Kullanıcı dostu ve hızlı web arayüzünün oluşturulması için.

## ⚙️ Donanım ve Yazılım Gereksinimleri
- *İşletim Sistemi*: Windows 10/11, macOS veya Linux
- *RAM*: Minimum 8 GB (Modelin tensör hesaplamaları için önerilir)
- *Ekran Kartı (GPU)*: NVIDIA CUDA destekli bir GPU işlemleri hızlandıracaktır ancak zorunlu değildir. Sistem otomatik olarak CPU/GPU tespiti yapmaktadır.
- *Python*: 3.8 veya üzeri bir sürüm yüklü olmalıdır.

## 📂 Klasör Yapısı
- main.py : Streamlit arayüzünü başlatan ve kullanıcı etkileşimini yöneten ana tetikleyici dosya.
- model_motoru.py : VGG19 modelinin transfer öğrenme ile entegre edildiği, içerik ve stil kayıplarının hesaplandığı derin öğrenme motoru.
- requirements.txt : Projenin çalışması için gereken kütüphanelerin listesi.
- gece.jpeg, monet.jpg, picasso.jpg : Sistemin test edilmesi için klasöre eklenmiş örnek (sample) stil ve içerik verileri.

## 🚀 Kurulum Kılavuzu

*1. Projeyi Bilgisayarınıza İndirin (Clone)*
Terminali açın ve aşağıdaki komutu girerek projeyi bilgisayarınıza klonlayın:
git clone https://github.com/kaanorphan0/python_stil_aktarimi.git

*2. Proje Klasörüne Girin*
cd python_stil_aktarimi

*3. Gerekli Kütüphaneleri Yükleyin*
Projenin ihtiyaç duyduğu kütüphaneleri kurmak için terminale şu komutu yazın:
pip install -r requirements.txt

## 🎯 Çalıştırma Komutu
Kurulum tamamlandıktan sonra projeyi başlatmak için terminale aşağıdaki komutu girmeniz yeterlidir:

streamlit run main.py

Komutu girdikten sonra varsayılan tarayıcınızda (localhost:8501) proje arayüzü otomatik olarak açılacaktır.

## 📊 Veri Seti, Veritabanı ve Model Bilgisi
- *Veritabanı (.env, dump vb.):* Bu proje doğrudan RAM üzerinde tensör işlemleri yaptığı için herhangi bir harici SQL/NoSQL veritabanı veya .env dosyası gerektirmez.
- *Veri Seti:* Özel bir veri seti kullanılmamıştır. PyTorch üzerinden önceden ImageNet ile eğitilmiş VGG19 modeli anlık olarak (pretrained=True) indirilip kullanılmaktadır. Ekstra bir model dosyası indirmeye gerek yoktur.

## 🎥 Proje Demo Videosu
Projenin bilgisayarımızda kurulum ve çalışma adımlarını gösteren demo videosuna aşağıdan ulaşabilirsiniz:
[Videoyu İzlemek İçin Tıklayın (Google Drive)](https://drive.google.com/file/d/1N3KXMXddARQ2UzYi8O09-00IJFK_Awci/view?usp=sharing)









