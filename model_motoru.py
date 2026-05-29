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

def stili_aktar(icerik_tensoru, stil_tensoru, adim_sayisi=150):
    """
    İçerik (kullanıcı fotoğrafı) ve Stil (sanat eseri) tensörlerini alarak
    yapay zeka optimizasyon döngüsünü çalıştıran ana fonksiyondur.
    """
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # VGG19 modelini internetten indir (pretrained=True), cihazın belleğine yükle.
    # .eval() modu ile "eğitim" modunu kapatıyoruz çünkü modelin kendisini değil, resmi eğiteceğiz.
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # Modelin kendi ağırlıklarını donduruyoruz (Boşuna işlem gücü harcamaması için)
    for param in vgg.parameters():
        param.requires_grad = False 
        
    # Kişi 2'den gelen tensörleri cihaza gönderiyoruz
    icerik_tensoru = icerik_tensoru.to(device)
    stil_tensoru = stil_tensoru.to(device)
    
    # ÜRETECEĞİMİZ YENİ RESİM: Başlangıçta orijinal resmin birebir kopyasıdır.
    # requires_grad_(True) diyoruz çünkü adım adım bu resmin piksellerini değiştireceğiz.
    hedef_tensor = icerik_tensoru.clone().requires_grad_(True).to(device)
    
    # Orijinal resmin ve sanat eserinin özelliklerini sadece BİR KERE baştan hesaplıyoruz.
    stil_ozellikleri = ozellikleri_cikar(stil_tensoru, vgg)
    icerik_ozellikleri = ozellikleri_cikar(icerik_tensoru, vgg)
    
    # Sanat eserinin tüm katmanları için Gram matrislerini önceden hesaplayıp sözlüğe kaydediyoruz
    stil_gramlari = {katman: gram_matrix(stil_ozellikleri[katman]) for katman in stil_ozellikleri}
    
    # Stil katmanlarının önem dereceleri (İlk katmanlar dokuyu, son katmanlar genel renkleri verir)
    stil_agirliklari = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
    
    # İki kayıp (loss) arasındaki dengeyi kuran çarpanlar (alpha ve beta)
    icerik_carpani = 1      # İçeriği koruma gücü
    stil_carpani = 1e6      # Stili uygulama gücü (1 milyon)
    
    # Optimizatör: Hedef tensörün (yeni resmin) piksellerini güncelleyecek matematiksel motor.
    # lr=0.03 (Learning Rate), piksellerin her adımda ne kadar hızlı değişeceğidir.
    optimizer = optim.Adam([hedef_tensor], lr=0.03)
    
    # --- YAPAY ZEKA DÖNGÜSÜ BAŞLIYOR ---
    for adim in range(adim_sayisi):
        # 1. Adım: Yeni oluşan resmin o anki özelliklerini çıkar
        hedef_ozellikleri = ozellikleri_cikar(hedef_tensor, vgg)
        
        # 2. İÇERİK KAYBI: Yeni resim ile orijinal resmin 'conv4_2' katmanındaki farkı (MSE)
        icerik_kaybi = torch.mean((hedef_ozellikleri['conv4_2'] - icerik_ozellikleri['conv4_2'])**2)
        
        # 3. STİL KAYBI: Yeni resim ile sanat eserinin Gram matrisleri arasındaki fark
        stil_kaybi = 0
        for katman in stil_agirliklari:
            hedef_ozellik = hedef_ozellikleri[katman]
            hedef_gram = gram_matrix(hedef_ozellik)
            _, d, h, w = hedef_ozellik.shape # Boyutları alıyoruz (Normalize etmek için)
            
            stil_gram = stil_gramlari[katman]
            
            # Her katman için farkı bul, ağırlığıyla çarp ve matris boyutuna bölerek normalize et
            katman_stil_kaybi = stil_agirliklari[katman] * torch.mean((hedef_gram - stil_gram)**2)
            stil_kaybi += katman_stil_kaybi / (d * h * w)
            
        # 4. TOPLAM HATA (KAYIP): Modelin amacı bu toplamı sıfıra yaklaştırmaktır
        toplam_kayip = (icerik_carpani * icerik_kaybi) + (stil_carpani * stil_kaybi)
        
        # 5. GERİ YAYILIM VE PİKSEL GÜNCELLEME (Backpropagation)
        optimizer.zero_grad()      # Eski türevleri sıfırla
        toplam_kayip.backward()    # Yeni hatanın türevini hesapla (Pikselleri ne yöne değiştirelim?)
        optimizer.step()           # Pikselleri yeni yöne doğru güncelle
        
    # Döngü bittiğinde, pikselleri tamamen değişmiş yeni sanat eserini Kişi 2'ye yolla
    return hedef_tensor.detach()