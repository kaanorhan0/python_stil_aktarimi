import streamlit as st
import time
import io
from PIL import Image, ImageFilter, ImageEnhance


st.set_page_config(
    page_title="Sanatsal Stil Aktarımı",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def css_ekle():
    st.markdown("""
    <style>
    /* Karanlık tema arka planı ve metin renkleri */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sağ üstteki gereksiz menüleri gizle */
    #MainMenu, header, footer {visibility: hidden;}

    /* Buton tasarımı (Vurgulu ve şık) */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 10px;
    }
    .stButton > button:hover {
        filter: brightness(1.2);
    }
    </style>
    """, unsafe_allow_html=True)

def resmi_indirilebilir_yap(img: Image.Image) -> bytes:
    """Üretilen fotoğrafı indirme butonuna uygun formata çevirir."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def yapay_zeka_simulasyonu(icerik_resmi: Image.Image, stil_adi: str):
    """Gerçek PyTorch modeli eklenene kadar çalışan, ilerleme çubuklu simülasyon."""
    ilerleme_cubugu = st.progress(0, text="Yapay zeka modeli başlatılıyor...")

    # Aşama 1
    time.sleep(0.5)
    ilerleme_cubugu.progress(25, text="Fotoğraf matrislere (tensör) dönüştürülüyor...")

    # Aşama 2
    time.sleep(0.8)
    ilerleme_cubugu.progress(60, text="VGG19 modeli üzerinden özellikler çıkarılıyor...")

    # Aşama 3
    time.sleep(0.7)
    ilerleme_cubugu.progress(90, text=f"{stil_adi} dokuları hesaplanıyor...")

    
    sonuc = icerik_resmi.copy().convert("RGB")
    if "Van Gogh" in stil_adi:
        sonuc = sonuc.filter(ImageFilter.GaussianBlur(radius=1.5))
        sonuc = ImageEnhance.Color(sonuc).enhance(1.8)
    elif "Picasso" in stil_adi:
        sonuc = sonuc.filter(ImageFilter.FIND_EDGES)
    elif "Monet" in stil_adi:
        sonuc = sonuc.filter(ImageFilter.SMOOTH_MORE)
        sonuc = ImageEnhance.Color(sonuc).enhance(1.2)

    time.sleep(0.5)
    ilerleme_cubugu.progress(100, text="İşlem tamamlandı!")
    time.sleep(0.4)
    ilerleme_cubugu.empty()

    return sonuc


def main():
    css_ekle()

    
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🎨 Neural Studio</h2>", unsafe_allow_html=True)
        st.caption("<p style='text-align: center;'>Stil Aktarım Laboratuvarı</p>", unsafe_allow_html=True)
        st.divider()

        st.subheader("1. Girdi Görseli")
        yuklenen_dosya = st.file_uploader("Fotoğraf yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        st.write("") 

        st.subheader("2. Sanatsal Stil")
        stiller = ["🌌 Van Gogh - Yıldızlı Gece", "⬛ Picasso - Kübizm", "🌅 Monet - Gündoğumu"]
        secilen_stil = st.selectbox("Stil seçin", stiller, label_visibility="collapsed")

        st.divider()

        
        buton_aktif_mi = yuklenen_dosya is not None
        baslat_butonu = st.button("🚀 Dönüşümü Başlat", disabled=not buton_aktif_mi, use_container_width=True)

    
    st.title("✨ Gerçek Zamanlı Sanatsal Stil Aktarımı")
    st.write("Yüklediğiniz fotoğrafı derin öğrenme (CNN) modelleriyle ünlü bir tabloya dönüştürün.")

    if yuklenen_dosya is None:
        st.info("👈 Başlamak için sol panelden bir fotoğraf yükleyin.")
    else:
        orijinal_resim = Image.open(yuklenen_dosya)

        
        if baslat_butonu:
            st.divider()
            
            
            sonuc_resmi = yapay_zeka_simulasyonu(orijinal_resim, secilen_stil)

            st.success("✅ Sanatsal dönüşüm başarıyla tamamlandı!")

            
            col1, col2 = st.columns(2)
            with col1:
                st.image(orijinal_resim, caption="Orijinal Fotoğraf", use_container_width=True)
            with col2:
                st.image(sonuc_resmi, caption=f"Sonuç ({secilen_stil.split(' - ')[0][1:]})", use_container_width=True)

            
            st.divider()
            st.download_button(
                label="📥 Yüksek Çözünürlüklü Olarak İndir (PNG)",
                data=resmi_indirilebilir_yap(sonuc_resmi),
                file_name="sanatsal_sonuc.png",
                mime="image/png"
            )

        
        else:
            st.subheader("Orijinal Fotoğraf Önizlemesi")
            st.image(orijinal_resim, use_container_width=True)
            st.warning("Dönüşümü başlatmak için sol menüdeki **'Dönüşümü Başlat'** butonuna tıklayın.")

if __name__ == "__main__":
    main()