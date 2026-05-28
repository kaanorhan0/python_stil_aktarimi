import streamlit as st
from PIL import Image
import time
import base64
import io

# --- 1. SAYFA VE PERFORMANS AYARLARI ---
st.set_page_config(
    page_title="Neural Art Studio Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. YAPAY ZEKA MOTORU VE ÖNBELLEKLEME (CACHING) ---
# Gerçek model eklendiğinde sistemin çökmemesi için modeli RAM'de tutar
@st.cache_resource(show_spinner="Yapay Zeka Modeli Belleğe Yükleniyor...")
def load_vgg19_model():
    """Önceden eğitilmiş VGG19 özellik çıkarım ağını başlatır."""
    time.sleep(1) # Simülasyon
    return "VGG19_Model_Loaded"

# --- 3. PROFESYONEL CSS ---
def apply_enterprise_css(bg_image_name: str):
    """Karanlık mod ağırlıklı, cam efektli kurumsal arayüz tasarımı."""
    try:
        with open(bg_image_name, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        
        custom_css = f"""
        <style>
            /* Derin Karartılmış Arka Plan */
            .stApp {{
                background-image: linear-gradient(rgba(5, 5, 8, 0.90), rgba(5, 5, 8, 0.90)), url("data:image/jpeg;base64,{bin_str}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Üstteki gereksiz boşlukları ve menüleri sil */
            .block-container {{ padding-top: 2rem !important; }}
            #MainMenu {{visibility: hidden;}}
            header {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            
            /* Modern Panel Tasarımı */
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
            
            /* İndirme Butonu Vurgusu */
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
        st.error(f"⚠️ '{bg_image_name}' bulunamadı!")

# --- 4. GÖRÜNTÜ İŞLEME SİMÜLASYONU ---
def process_neural_transfer(image: Image.Image, style: str, style_weight: int) -> Image.Image:
    """Gerçek zamanlı stil aktarım işlem hattı (Pipeline)."""
    
    # Modern Durum Bildirici (st.status)
    with st.status("Yapay Zeka İşlem Hattı Başlatıldı...", expanded=True) as status:
        st.write("🔍 Görüntü matrislere (tensör) dönüştürülüyor...")
        time.sleep(0.5)
        
        st.write("🧠 VGG19 üzerinden içerik özellikleri (content features) çıkarılıyor...")
        time.sleep(0.8)
        
        st.write(f"🎨 '{style}' stili için Gram matrisleri hesaplanıyor...")
        st.write(f"⚙️ Stil yoğunluğu: %{style_weight} olarak uygulandı.")
        time.sleep(0.8)
        
        st.write("✨ Piksel optimizasyonu tamamlanıyor...")
        time.sleep(0.5)
        
        status.update(label="Sanatsal Dönüşüm Başarıyla Tamamlandı!", state="complete", expanded=False)
        
    return image # Gerçekte modelin çıktısı olacak

# --- 5. İNDİRME YARDIMCISI ---
def convert_image_to_bytes(img: Image.Image):
    """Görüntüyü indirilebilir formata (bytes) çevirir."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- ANA UYGULAMA MİMARİSİ ---
def main():
    apply_enterprise_css('gece.jpeg')
    _ = load_vgg19_model() # Arka planda modeli hazırla

    # --- YAN MENÜ (SIDEBAR) KONTROL PANELİ ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2000/2000494.png", width=50) # Şık bir ikon
        st.markdown("## ⚙️ Sistem Paneli")
        st.markdown("Yapay zeka parametrelerini buradan yapılandırın.")
        st.divider()
        
        st.markdown("**1. Girdi Ayarları**")
        yuklenen_dosya = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png"], label_visibility="collapsed")
        
        st.markdown("**2. Stil Referansı**")
        secilen_stil = st.selectbox(
            "Stil Seçimi",
            ("🌌 Van Gogh - Yıldızlı Gece", "⬛ Picasso - Kübizm", "🌅 Monet - Gündoğumu"),
            label_visibility="collapsed"
        )
        
        st.divider()
        # Gelişmiş Mühendislik Ayarları
        with st.expander("🛠️ Gelişmiş Hiperparametreler"):
            st.slider("Çözünürlük Kalitesi", 256, 1024, 512, step=128, help="Modelin işleyeceği tensör boyutu.")
            stil_yogunlugu = st.slider("Stil Ağırlığı (Style Weight)", 10, 100, 75, help="Uygulanan stilin orijinal fotoğrafı ne kadar ezeceğini belirler.")
            st.slider("Optimizasyon Adımı (Epoch)", 10, 100, 30, help="L-BFGS iterasyon sayısı.")

    # --- ANA ÇALIŞMA ALANI (MAIN WORKSPACE) ---
    st.markdown("<h1 style='text-align: left; padding-bottom: 0;'>✨ Neural Art Studio <span style='font-size: 0.5em; color: #3b82f6; border: 1px solid #3b82f6; padding: 2px 8px; border-radius: 10px; vertical-align: middle;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Gerçek Zamanlı Evrişimli Sinir Ağları (CNN) Laboratuvarı</p>", unsafe_allow_html=True)
    
    if yuklenen_dosya is None:
        # Boş Durum
        st.write("")
        st.info("Sistemi başlatmak için sol panelden bir fotoğraf yükleyin ve parametreleri ayarlayın.")
    else:
        orijinal_resim = Image.open(yuklenen_dosya)
        
        col_img, col_action = st.columns([3, 1])
        with col_img:
            st.image(orijinal_resim, caption="Orijinal Girdi Görseli", use_container_width=True)
        with col_action:
            st.markdown("### İşlem Merkezi")
            st.caption("GPU Kullanımı: **Hazır**")
            baslat_butonu = st.button("🚀 Modeli Çalıştır", use_container_width=True, type="primary")

        if baslat_butonu:
            st.divider()
            
            # İşlem Hattı
            sonuc_resmi = process_neural_transfer(orijinal_resim, secilen_stil, stil_yogunlugu)
            
            # Sonuç Gösterimi ve İndirme
            st.markdown(f"### 🎨 Dönüştürülmüş Sonuç ({secilen_stil.split(' ')[1]})")
            st.image(sonuc_resmi, use_container_width=True)
            
            # İndirme Butonu
            img_bytes = convert_image_to_bytes(sonuc_resmi)
            st.download_button(
                label="📥 Yüksek Çözünürlüklü Olarak İndir (PNG)",
                data=img_bytes,
                file_name="neural_art_ciktisi.png",
                mime="image/png",
                use_container_width=True
            )

if __name__ == "__main__":
    main()