import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_data, img_size=512):
    """
    Kullanıcının Streamlit üzerinden yüklediği resmi alır, VGG19 ve Transformer
    modelinin yiyebileceği tensör formatına (ve boyuta) getirir.
    """
    # 1. Resmi aç ve garanti olsun diye RGB'ye çevir (PNG'lerdeki saydamlık kanalı modeli bozar)
    image = Image.open(image_data).convert('RGB')
    
    # 2. Resme uygulanacak dönüşümleri tanımla
    transform = transforms.Compose([
        transforms.Resize(img_size),          # Görüntünün kısa kenarını yeniden boyutlandır
        transforms.CenterCrop(img_size),      # Tam ortadan kare olacak şekilde kırp
        transforms.ToTensor(),                # Piksel değerlerini [0, 1] arasına çek ve Tensöre çevir
        
        # NOT: Kişi 1'in modeline göre gerekirse aşağıdaki satırı açarsınız. 
        # VGG19 genellikle ImageNet istatistikleriyle normalize edilmiş veri ister.
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Dönüşümü uygula
    image_tensor = transform(image)
    
    # 4. PyTorch modelleri verileri "Batch" (yığın) halinde bekler. 
    # (Channel, Height, Width) formatını -> (1, Channel, Height, Width) formatına sokuyoruz.
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def tensor_to_image(tensor):
    """
    Stil aktarımı bittikten sonra modelin tükürdüğü tensörü, 
    Streamlit arayüzünde gösterebilmek için tekrar normal bir görsele çevirir.
    """
    # 1. Tensörü hesaplama grafiğinden ayır (detach) ve CPU belleğine al
    image_tensor = tensor.detach().cpu().clone()
    
    # 2. Eklediğimiz Batch boyutunu kaldırıyoruz: (1, C, H, W) -> (C, H, W)
    image_tensor = image_tensor.squeeze(0)
    
    # 3. Değerlerin kesinlikle 0 ile 1 arasında olduğundan emin ol (patlak pikselleri önler)
    image_tensor = image_tensor.clamp(0, 1)
    
    # 4. PyTorch (Kanal, Yükseklik, Genişlik) kullanır. Bizim bunu (Yükseklik, Genişlik, Kanal) yapmamız lazım.
    image_numpy = image_tensor.numpy().transpose(1, 2, 0)
    
    # 5. [0, 1] aralığındaki değerleri [0, 255] aralığına genişlet ve uint8 veri tipine çevir
    image_numpy = (image_numpy * 255).astype(np.uint8)
    
    # 6. Numpy dizisini Streamlit'in basabileceği PIL objesine geri çevir
    return Image.fromarray(image_numpy)