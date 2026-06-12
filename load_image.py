# load_image.py dosyasının tamamını bununla değiştirin:
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_data, img_size=512):
    image = Image.open(image_data).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_size),          
        transforms.CenterCrop(img_size),      
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # NORMALİZASYON AÇILDI
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def tensor_to_image(tensor):
    image_tensor = tensor.detach().cpu().clone().squeeze(0)
    
    # DENORMALİZASYON İŞLEMİ (Renkleri eski haline getirme)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    
    image_tensor = image_tensor.clamp(0, 1)
    image_numpy = image_tensor.numpy().transpose(1, 2, 0)
    image_numpy = (image_numpy * 255).astype(np.uint8)
    return Image.fromarray(image_numpy)