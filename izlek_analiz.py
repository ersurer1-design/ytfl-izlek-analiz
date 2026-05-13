import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import random
import time
import base64
import requests
import re

# --- 0. YARDIMCI FONKSİYONLAR ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# --- 1. METİN GEÇERLİLİK VE ANLAMSIZLIK KONTROLÜ ---
def metin_gecerli_mi(text):
    text = text.strip()
    if len(text) < 50:
        return False, "Hata: Analiz için en az 50 karakterlik bir haber metni girmelisiniz."
    
    benzersiz_karakterler = set(text.lower())
    # HATA BURADAYDI: len() fonksiyonu bir sayı üzerinde kullanılamaz. 
    # Doğrudan set'in uzunluğuna bakıyoruz.
    if len(benzersiz_karakterler) < 8:
        return False, "Hata: Metin anlamsız görünüyor (karakter çeşitliliği çok düşük)."
    
    sesli_harfler = re.findall(r'[aeıioöuüAEIİOÖUÜ]', text)
    if len(sesli_harfler) / len(text) < 0.15:
        return False, "Hata: Metin doğal bir dil yapısına sahip görünmüyor (sesli harf oranı düşük)."

    rakamlar = re.findall(r'[0-9]', text)
    if len(rakamlar) / len(text) > 0.4:
        return False, "Hata: Metin çok fazla sayı içeriyor, haber metni niteliği taşımıyor."

    return True, ""

# --- 2. TÜRKÇE META VERİ SÖZLÜĞÜ ---
EXIF_TR = {
    'Make': 'Cihaz Üreticisi', 'Model': 'Cihaz Modeli', 'Software': 'Düzenleme Yazılımı',
    'DateTime': 'Oluşturulma Tarihi', 'ExifImageWidth': 'Genişlik (px)',
    'ExifImageHeight': 'Yükseklik (px)', 'XResolution': 'Yatay Çözünürlük',
    'YResolution': 'Dikey Çözünürlük', 'ResolutionUnit': 'Çözünürlük Birimi',
    'Orientation': 'Görsel Yönü', 'ExposureMode': 'Pozlama Modu', 'Flash': 'Flaş Durumu',
    'FocalLength': 'Odak Uzaklığı', 'ISOSpeedRatings': 'ISO Hızı', 'ExposureTime': 'Pozlama Süresi',
    'FNumber': 'Diyafram Değeri (F)', 'SceneType': 'Sahne Tipi', 'ColorSpace': 'Renk Uzayı'
}

# --- 3. AI TESPİT API (SIGHTENGINE) ---
def ai_kontrol_api(image_path):
    try:
        if "api_user" not in st.secrets or "api_secret" not in st.secrets:
            st.error("Hata: Secrets panelinde API anahtarları tanımlanmamış!")
            return None
            
        params = {
            'models': 'genai',
            'api_user': st.secrets["api_user"], 
            'api_secret': st.secrets["api_secret"]
        }
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        output = response.json()
        
        if output.get('status') == 'success':
            return output.get('type', {}).get('ai_generated', 0)
        else:
            err_msg = output.get('error', {}).get('message', 'Bilinmeyen API hatası')
            st.error(f"API Hatası: {err_msg}")
            return None
    except Exception as e:
        st.error(f"Bağlantı Hatası: {str(e)}")
        return None

# --- 4. VERİ SETİ VE MODEL EĞİTİMİ (NLP) ---
@st.cache_resource
def izlek_beyin_egit():
    dosya = "nlp_egitim_veri_seti.csv"
    if not os.path.exists(dosya):
        data = []
        g_kalip = ["Resmi makamlarca yapılan açıklamada {0} {1}.", "{0} tarafından duyurulan yeni kararla {1} onaylandı."]
        s_kalip = ["ŞOK İDDİA! {0} aslında {1} olduğu gizleniyor!", "{0} hakkında skandal görüntü: {1} gerçeği şok etti!"]
        ozneler = ["Bakanlık", "TÜBİTAK", "Valilik", "Emniyet Müdürlüğü"]
        eylemler = ["yeni çalışma başlattı", "başarıya ulaştı", "açıklama yaptı"]
        for _ in range(3000):
            data.append([random.choice(g_kalip).format(random.choice(ozneler), random.choice(eylemler)), 0])
            data.append([random.choice(s_kalip).format(random.choice(ozneler), "manipüle edilmiş"), 1])
        pd.DataFrame(data, columns=['text', 'label']).to_csv(dosya, index=False, encoding='utf-8-sig')
    
    df = pd.read_csv(dosya)
    v = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X = v.fit_transform(df['text'].astype(str))
    m = MultinomialNB(alpha=0.1)
    m.fit(X, df['label'])
    return v, m

vectorizer, model = izlek_beyin_egit()

# --- 5. SAYFA AYARLARI VE CSS (TASARIM KORUNDU) ---
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url, initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Dinamik Kaydırma Çubuğu Ayarı */
    .main, .stApp {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* Sidebar Kapatma Butonunu Gizle (Sabit yapmak için) */
    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }

    /* Üst menü ve footer gizleme */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp { background-image: url('https://www.transparenttextures.com/patterns/carbon-fibre.png'); }
    .header-bar { background-color: #FFD700; padding: 15px; border-radius: 10px; display: flex; align-items: center; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
    .header-bar h1 { color: black !important; margin:0; font-size: 24px; }
    .header-bar img { height: 40px; margin-right: 20px; }
    .kunya-box { background-color: #f0f7ff; border-left: 5px solid #1e3c72; padding: 15px; border-radius: 5px; font-size: 14px; color: #1e3c72 !important; line-height: 1.6; }
    .meta-card { background-color: #ffffff !important; color: #1e3c72 !important; border: 1px solid #e0e0e0; padding: 12px; border-radius: 8px; margin-bottom: 8px; font-size: 0.9em; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .meta-card b { color: #d92323 !important; }
    .sidebar-logo-box { display: flex; justify-content: center; padding: 20px; background-color: white; border-radius: 10px; margin-bottom: 20px; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.2em; margin-top: 20px; border: 2px solid; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. YAN PANEL (SIDEBAR) ---
with st.sidebar:
    logo_b64 = get_base64_image("YTFL LOGO.jpg")
    if logo_b64:
        st.markdown(f'<div class="sidebar-logo-box"><img src="data:image/jpeg;base64,{logo_b64}" width="150"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Proje Künyesi")
    st.markdown(f'''<div class="kunya-box">
        <b>Proje:</b> Sahte Haber ve Görsel Tespiti<br>
        <b>Danışman:</b> Hasan ERSÜRER<br>
        <hr style="margin: 8px 0; border:0; border-top: 1px solid #d1d1d1;">
        <b>Öğrenciler:</b><br>
        • Abdullah ELŞAHAP<br>
        • Harun Buğra ŞANVERDİ<br>
        • Hasan Kayra GÜLLÜ<br>
        <hr style="margin: 8px 0; border:0; border-top: 1px solid #d1d1d1;">
        <b>Okul:</b> Yahya Turan Fen Lisesi<br>
        <b>Teknoloji:</b> NLP & ELA & EXIF Analizi
    </div>''', unsafe_allow_html=True)
    st.write(""); st.success("Sistem Durumu: Hazır ✅")

# --- 7. ANA SAYFA ---
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}">
        <h1>YTFL İzlek: Dijital Doğrulama Sistemi</h1>
    </div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Metin Analizi", "🖼️ Görsel Analiz (ELA)", "📂 Meta Veri (EXIF)"])

with tab1:
    st.subheader("Haber Doğrulama (NLP)")
    metin = st.text_area("Analiz edilecek haber metnini giriniz:", height=150, placeholder="En az 50 karakterlik bir metin girin...")
    if st.button("Analizi Başlat"):
        gecerli, mesaj = metin_gecerli_mi(metin)
        if not gecerli:
            st.warning(mesaj)
        else:
            with st.spinner("Yapay zeka örüntüleri tarıyor..."):
                time.sleep(0.8)
                vec = vectorizer.transform([metin])
                tahmin = model.predict(vec)[0]
                olasilik = model.predict_proba(vec)[0]
                if tahmin == 1:
                    st.error(f"🚨 SONUÇ: ŞÜPHELİ (%{olasilik[1]*100:.1f})")
                else:
                    st.success(f"✅ SONUÇ: GÜVENİLİR (%{olasilik[0]*100:.1f})")

with tab2:
    st.subheader("Görsel Manipülasyon Analizi (ELA)")
    yukle = st.file_uploader("Görsel yükleyin (JPG/JPEG):", type=['jpg', 'jpeg'])
    if yukle:
        with open("temp.jpg", "wb") as f: f.write(yukle.getbuffer())
        c1, c2 = st.columns(2)
        c1.image(yukle, caption="Orijinal Görsel", width='stretch')
        
        # ELA Analizi
        im = Image.open("temp.jpg").convert('RGB')
        im.save("resaved.jpg", 'JPEG', quality=90)
        resaved = Image.open("resaved.jpg")
        ela = ImageChops.difference(im, resaved)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        ela_viz = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        c2.image(ela_viz, caption="ELA Analizi Sonucu", width='stretch')
        st.info("💡 **ELA Analizi İpucu:** Sağdaki görselde aşırı parlak görünen alanlar, görselin o kısımlarıyla dijital olarak oynanmış olabileceğini gösterir.")

with tab3:
    st.subheader("Dijital İz Analizi (Metadata)")
    if 'yukle' in locals() and yukle:
        try:
            img = Image.open("temp.jpg")
            exif_data = img._getexif()
            if exif_data:
                st.write("Görselin içerisine gömülü teknik veriler (Türkçe):")
                cols = st.columns(3)
                for i, (tag, value) in enumerate(exif_data.items()):
                    etiket_ing = TAGS.get(tag, tag)
                    etiket_tr = EXIF_TR.get(etiket_ing, etiket_ing)
                    if isinstance(value, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{etiket_tr}:</b><br>{value}</div>', unsafe_allow_html=True)
            else: st.warning("Bu görselde teknik iz (meta veri) bulunamadı.")
        except: st.error("Teknik veri okuma hatası oluştu.")
        
        st.divider()
        if st.button("Nihai AI Analizini Gerçekleştir"):
            with st.spinner("Yapay zekâ modelleri görselin DNA'sını inceliyor..."):
                olasılık = ai_kontrol_api("temp.jpg")
                if olasılık is not None:
                    if olasılık > 0.5:
                        st.markdown(f'<div class="result-box" style="background-color: #ffebee; color: #d32f2f; border-color: #d32f2f;">Analiz Sonucu: %{olasılık*100:.1f} Olasılıkla Yapay Zekâ Ürünü 🚨</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box" style="background-color: #e8f5e9; color: #2e7d32; border-color: #2e7d32;">Analiz Sonucu: %{(1-olasılık)*100:.1f} Olasılıkla Gerçek Çekim ✅</div>', unsafe_allow_html=True)
                else:
                    st.warning("Analiz sonucu alınamadı. Lütfen API anahtarlarınızı ve internet bağlantısını kontrol edin.")
    else: st.info("Lütfen Görsel Analiz sekmesinden bir fotoğraf yükleyin.")

# --- 8. FOOTER ---
meb_b64 = get_base64_image("meb.png")
tubitak_b64 = get_base64_image("tubitak.png")
st.markdown(f'''
    <div style="background-color: white; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; align-items: center; gap: 10px; margin-top: 30px;">
        <div style="display: flex; gap: 40px;">
            <img src="data:image/png;base64,{meb_b64}" height="50">
            <img src="data:image/png;base64,{tubitak_b64}" height="50">
        </div>
        <div style="color: #666; font-size: 0.8em; text-align:center;">
            © 2026 - Reyhanlı Yahya Turan Fen Lisesi TÜBİTAK 4006 Projesi
        </div>
    </div>
''', unsafe_allow_html=True)
