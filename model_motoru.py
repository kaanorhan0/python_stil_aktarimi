import torch
import torch.nn as nn
import torchvision.models as models

class StilTransferModeli(nn.Module):
    def __init__(self):
        super(StilTransferModeli, self).__init__()
        
        # 1. Önceden eğitilmiş VGG19 modelini yükle (Sadece özellik çıkaran kısmını alıyoruz)
        vgg = models.vgg19(pretrained=True).features
        
        # 2. Modelin ağırlıklarını dondur (Modeli eğitmeyeceğiz, sadece özellik çıkaracağız)
        for param in vgg.parameters():
            param.requires_grad = False
            
        # 3. İhtiyacımız olan katmanları seçiyoruz
        self.model = vgg[:29] 
        
        # Stil ve İçerik için Gatys makalesine uygun katman numaraları
        self.stil_katmanlari = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        self.icerik_katmani = {'21': 'conv4_2'}

    def forward(self, x):
        ozellikler = {}
        for isim, katman in self.model._modules.items():
            x = katman(x)
            if isim in self.stil_katmanlari:
                ozellikler[self.stil_katmanlari[isim]] = x
            if isim in self.icerik_katmani:
                ozellikler[self.icerik_katmani[isim]] = x
        return ozellikler

def gram_matrisi(tensor):
    """Stil kaybı için Gram Matrisini (doku ve renk korelasyonu) hesaplar."""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

if __name__ == "__main__":
    model = StilTransferModeli()
    print("Harika! VGG19 Modeli başarıyla yüklendi ve katmanlar ayrıştırıldı.")