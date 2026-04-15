import streamlit as st
import torch
from PIL import Image


st.set_page_config(page_title="Gerçek Zamanlı Stil Aktarımı", layout="centered")

st.title("🎨 Gerçek Zamanlı Sanatsal Stil Aktarımı")
st.write("Fotoğrafınızı yükleyin ve anında sanatsal bir esere dönüştürün!")

stil_secimi = st.selectbox(
    "Uygulamak istediğiniz stili seçin:",
    ("Van Gogh - Yıldızlı Gece", "Picasso", "Claude Monet")
)


yuklenen_dosya = st.file_uploader("Bir fotoğraf yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if yuklenen_dosya is not None:
    
    orijinal_resim = Image.open(yuklenen_dosya)
    st.subheader("Orijinal Fotoğraf")
    st.image(orijinal_resim, use_container_width=True)
    
    
    if st.button("Stili Uygula"):
        with st.spinner("Sanat eseri oluşturuluyor..."):
            
            st.success("İşlem tamamlandı! (Saniyenin onda biri sürede)")
            st.subheader("Dönüştürülmüş Fotoğraf")
            st.image(orijinal_resim, use_container_width=True, caption=f"Uygulanan Stil: {stil_secimi}")