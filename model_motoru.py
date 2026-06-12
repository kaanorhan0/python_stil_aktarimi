# --- KÜTÜPHANELER ---
import torch                          # PyTorch'un ana kütüphanesi (Tensör işlemleri için)
import torch.nn as nn                 # Sinir ağları (Neural Networks) modülü
import torchvision.models as models   # VGG19 gibi hazır eğitilmiş modelleri çekmek için
import torch.optim as optim           # Optimizasyon algoritmaları (Adam vb.) için

# 1. GRAM MATRİSİ FONKSİYONU (Stil Çıkarımı)

def gram_matrix(tensor):
    """
    Tensörün Gram matrisini hesaplar.
    Gram matrisi, bir görüntüdeki renklerin ve dokuların (fırça darbeleri vb.)
    birbiriyle olan ilişkisini matematiksel olarak yakalamamızı sağlar.
    """
    # _, d (derinlik/kanal sayısı), h (yükseklik), w (genişlik) değerlerini alıyoruz.
    _, d, h, w = tensor.size()
    
    # 3 boyutlu tensörü (d, h, w), 2 boyutlu düz bir matrise (d, h*w) çeviriyoruz.
    tensor = tensor.view(d, h * w)
    
    # Matrisi kendi transpozu (devriği) ile çarpıyoruz. İşte bu Gram matrisidir!
    gram = torch.mm(tensor, tensor.t())
    
    return gram

# 2. ÖZELLİK ÇIKARMA FONKSİYONU (VGG19'u Kesip Biçme)

def ozellikleri_cikar(image, model):
    """
    Görüntüyü VGG19 ağından geçirir ancak sonuna kadar gitmez.
    Sadece belirlediğimiz duraklarda (katmanlarda) inip o anki özellikleri toplar.
    """
    # VGG19'un katman numaraları ve onlara verdiğimiz isimler.
    # Derin katmanlar (conv4_2) nesneyi (içerik) anlar.
    # Sığ katmanlar (conv1_1 vb.) renkleri (stil) anlar.
    layers = {
        '0': 'conv1_1',   # Stil için
        '5': 'conv2_1',   # Stil için
        '10': 'conv3_1',  # Stil için
        '19': 'conv4_1',  # Stil için
        '21': 'conv4_2',  # İÇERİK (Nesne yapısı) için
        '28': 'conv5_1'   # Stil için
    }
    
    features = {}
    x = image # İşlenecek görüntüyü x değişkenine atıyoruz
    
    # VGG19'un içindeki katmanları sırayla dönüyoruz
    for name, layer in model._modules.items():
        x = layer(x) # Görüntüyü bir sonraki katmandan geçir
        # Eğer geldiğimiz katman bizim listemizdeyse, o anki görüntüyü kaydet
        if name in layers:
            features[layers[name]] = x
            
    return features

# 3. ANA MOTOR: STİL AKTARIM DÖNGÜSÜ

# Sadece stili_aktar fonksiyonunu aşağıdakiyle değiştirin, diğer yerler (gram_matrix vb) aynı kalsın.

def stili_aktar(icerik_tensoru, stil_tensoru, vgg_model, adim_sayisi=50):
    """
    İçerik ve Stil tensörlerini alarak optimizasyon döngüsünü çalıştırır.
    NOT: vgg_model artık dışarıdan (main.py'den) hazır olarak geliyor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    icerik_tensoru = icerik_tensoru.to(device)
    stil_tensoru = stil_tensoru.to(device)
    
    # Orijinal resmi kopyalayıp, değiştirilebilir (requires_grad=True) yapıyoruz
    hedef_tensor = icerik_tensoru.clone().requires_grad_(True).to(device)
    
    stil_ozellikleri = ozellikleri_cikar(stil_tensoru, vgg_model)
    icerik_ozellikleri = ozellikleri_cikar(icerik_tensoru, vgg_model)
    
    stil_gramlari = {katman: gram_matrix(stil_ozellikleri[katman]) for katman in stil_ozellikleri}
    stil_agirliklari = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
    
    icerik_carpani = 1      
    stil_carpani = 1e6      
    
    optimizer = optim.Adam([hedef_tensor], lr=0.03)
    
    for adim in range(adim_sayisi):
        hedef_ozellikleri = ozellikleri_cikar(hedef_tensor, vgg_model)
        
        icerik_kaybi = torch.mean((hedef_ozellikleri['conv4_2'] - icerik_ozellikleri['conv4_2'])**2)
        
        stil_kaybi = 0
        for katman in stil_agirliklari:
            hedef_ozellik = hedef_ozellikleri[katman]
            hedef_gram = gram_matrix(hedef_ozellik)
            _, d, h, w = hedef_ozellik.shape 
            stil_gram = stil_gramlari[katman]
            katman_stil_kaybi = stil_agirliklari[katman] * torch.mean((hedef_gram - stil_gram)**2)
            stil_kaybi += katman_stil_kaybi / (d * h * w)
            
        toplam_kayip = (icerik_carpani * icerik_kaybi) + (stil_carpani * stil_kaybi)
        
        optimizer.zero_grad()      
        toplam_kayip.backward()    
        optimizer.step()           
        
    return hedef_tensor.detach()