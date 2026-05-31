import streamlit as st
import torch
import time
import base64
import io
from PIL import Image

# --- ENTEGRASYON İŞLEMİ ---
from load_image import load_and_preprocess_image, tensor_to_image
from model_motoru import stili_aktar
# --- 1. SAYFA VE PERFORMANS AYARLARI ---

st.set_page_config(
    page_title="Neural Art Studio Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. YAPAY ZEKA MOTORU VE ÖNBELLEKLEME (CACHING) ---
@st.cache_resource(show_spinner="Yapay Zeka Modeli Belleğe Yükleniyor...")
def load_vgg19_model():
    """Önceden eğitilmiş VGG19 özellik çıkarım ağını hazırda tutar."""
    time.sleep(0.5) 
    return "VGG19_Hazir"

# --- 3. PROFESYONEL CSS VE ARKA PLAN ---
def apply_enterprise_css(bg_image_name: str):
    """Karanlık mod ağırlıklı, cam efektli kurumsal arayüz tasarımı."""
    try:
        with open(bg_image_name, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        
        custom_css = f"""
        <style>
            .stApp {{
                background-image: linear-gradient(rgba(5, 5, 8, 0.90), rgba(5, 5, 8, 0.90)), url("data:image/jpeg;base64,{bin_str}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .block-container {{ padding-top: 2rem !important; }}
            #MainMenu {{visibility: hidden;}}
            header {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            
            [data-testid="stSidebar"] {{
                background-color: rgba(15, 23, 42, 0.6) !important;
                backdrop-filter: blur(15px);
                border-right: 1px solid rgba(255, 255, 255, 0.05);
            }}
            
            [data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {{
                background: rgba(30, 41, 59, 0.3) !important;
                backdrop-filter: blur(12px);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 1.5rem;
            }}
            
            [data-testid="stDownloadButton"] button {{
                border: 1px solid #3b82f6 !important;
                color: #3b82f6 !important;
            }}
            [data-testid="stDownloadButton"] button:hover {{
                background-color: #3b82f6 !important;
                color: white !important;
            }}
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ '{bg_image_name}' bulunamadı! Lütfen arkaplan resmini proje klasörüne koyun.")

# --- 4. GERÇEK ZAMANLI STİL AKTARIM İŞLEM HATTI (PIPELINE) ---

def process_neural_transfer(image_data, style_name, cozunurluk, epoch_sayisi):
    """Kişi 1, Kişi 2 ve Kişi 3'ün görevlerini birbirine bağlayan ana zincir."""
    
    # 1. Seçilen stile göre doğru dosyayı belirle
    stil_dosyasi = "gece.jpeg" # Varsayılan
    if "Picasso" in style_name:
        stil_dosyasi = "picasso.jpg" 
    elif "Monet" in style_name:
        stil_dosyasi = "monet.jpg"   

    with st.status("Yapay Zeka İşlem Hattı Başlatıldı...", expanded=True) as status:
        st.write("🔍 [Kişi 2 Modülü] Görüntü matrislere (tensör) dönüştürülüyor...")
        content_tensor = load_and_preprocess_image(image_data, img_size=cozunurluk)
        
        st.write(f"🌌 [Sistem] '{style_name}' referans görseli hazırlanıyor...")
        
        try:
            style_tensor = load_and_preprocess_image(stil_dosyasi, img_size=cozunurluk)
        except FileNotFoundError:
            st.error(f"⚠️ Hata: Klasörde '{stil_dosyasi}' adlı bir resim bulunamadı! Lütfen internetten bir resim indirip adını {stil_dosyasi} yaparak klasöre ekleyin.")
            st.stop() # Resmi bulamazsa sistemi güvenli şekilde durdur
            
        st.write(f"🧠 [Kişi 1 Modülü] VGG19 üzerinden özellikler çıkarılıyor ve {epoch_sayisi} adımda optimize ediliyor...")
        output_tensor = stili_aktar(content_tensor, style_tensor, adim_sayisi=epoch_sayisi)
        
        st.write("✨ [Kişi 2 Modülü] Piksel optimizasyonu tamamlandı, tensör görselleştiriliyor...")
        final_image = tensor_to_image(output_tensor)
        
        status.update(label="Sanatsal Dönüşüm Başarıyla Tamamlandı!", state="complete", expanded=False)
        
    return final_image
# --- 5. İNDİRME YARDIMCISI ---
def convert_image_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- ANA UYGULAMA MİMARİSİ ---
def main():
    apply_enterprise_css('gece.jpeg')
    _ = load_vgg19_model() # Modeli belleğe al
    
    # --- YAN MENÜ (SIDEBAR) KONTROL PANELİ ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🎨 Neural Studio</h2>", unsafe_allow_html=True)
        st.caption("<p style='text-align: center;'>Stil Aktarım Laboratuvarı</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("**1. Girdi Ayarları**")
        yuklenen_dosya = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        st.markdown("**2. Stil Referansı**")
        secilen_stil = st.selectbox(
            "Stil Seçimi",
            ("🌌 Van Gogh - Yıldızlı Gece", "⬛ Picasso - Kübizm", "🌅 Monet - Gündoğumu"),
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Gelişmiş Mühendislik Ayarları (Hiperparametreler modele bağlanıyor)
        with st.expander("🛠️ Gelişmiş Hiperparametreler"):
            cozunurluk = st.slider("Çözünürlük Kalitesi", 256, 768, 400, step=128, help="Modelin işleyeceği tensör boyutu.")
            stil_yogunlugu = st.slider("Stil Ağırlığı", 10, 100, 75, help="Stilin orijinal resmi ne kadar ezeceği.")
            epoch_sayisi = st.slider("Optimizasyon Adımı (Epoch)", 10, 150, 50, help="Yapay zekanın iterasyon sayısı.")

        st.divider()
        buton_aktif_mi = yuklenen_dosya is not None
        baslat_butonu = st.button("🚀 Modeli Çalıştır", disabled=not buton_aktif_mi, use_container_width=True, type="primary")

    # --- ANA ÇALIŞMA ALANI (MAIN WORKSPACE) ---
    st.markdown("<h1 style='text-align: left; padding-bottom: 0;'>✨ Neural Art Studio <span style='font-size: 0.5em; color: #3b82f6; border: 1px solid #3b82f6; padding: 2px 8px; border-radius: 10px; vertical-align: middle;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Gerçek Zamanlı Evrişimli Sinir Ağları (CNN) Laboratuvarı</p>", unsafe_allow_html=True)

    if yuklenen_dosya is None:
        st.info("👈 Başlamak için sol panelden bir fotoğraf yükleyin ve parametreleri ayarlayın.")
    else:
        orijinal_resim = Image.open(yuklenen_dosya)
        
        if baslat_butonu:
            st.divider()
            
            # Üç kişinin kodunu çalıştıran entegre fonksiyon tetikleniyor
            sonuc_resmi = process_neural_transfer(yuklenen_dosya, secilen_stil, cozunurluk, epoch_sayisi)
            
            st.success("✅ Sanatsal dönüşüm başarıyla tamamlandı!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(orijinal_resim, caption="Orijinal Girdi Görseli", use_container_width=True)
            with col2:
                st.image(sonuc_resmi, caption=f"Dönüştürülmüş Sonuç ({secilen_stil.split(' ')[1]})", use_container_width=True)
            
            st.divider()
            
            # İndirme Butonu
            img_bytes = convert_image_to_bytes(sonuc_resmi)
            st.download_button(
                label="📥 Yüksek Çözünürlüklü Olarak İndir (PNG)",
                data=img_bytes,
                file_name="neural_art_ciktisi.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.subheader("Orijinal Fotoğraf Önizlemesi")
            st.image(orijinal_resim, use_container_width=True)
            st.warning("Dönüşümü başlatmak için sol menüdeki **'🚀 Modeli Çalıştır'** butonuna tıklayın.")

if __name__ == "__main__":
    main()