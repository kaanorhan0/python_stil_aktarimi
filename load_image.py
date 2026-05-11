
import torch
import torchvision.transforms as transforms
from PIL import Image

def resmi_yukle_ve_hazirla(image_file, max_size=400):
    """Kullanicidan gelen resmi PyTorch tensörüne çevirir."""
    image = Image.open(image_file).convert('RGB')
    
  
    size = max_size if max(image.size) > max_size else max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def tensoru_resme_cevir(tensor):
    """Modelden çikan matematiksel tensörü tekrar fotoğrafa dönüştürür."""
    image = tensor.cpu().clone().detach().squeeze(0)
    
    
    unloader = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)),
        transforms.ToPILImage()
    ])
    image = unloader(image)
    return image