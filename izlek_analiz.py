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
        return False, "⚠️ Hata: Analiz için en az 50 karakterlik bir haber metni girmelisiniz."
    
    benzersiz_karakterler = set(text.lower())
    if len(benzersiz_karakterler) < 8:
        return False, "⚠️ Hata: Metin anlamsız görünüyor (karakter çeşitliliği çok düşük)."
    
    sesli_harfler = re.findall(r'[aeıioöuüAEIİOÖUÜ]', text)
    if len(sesli_harfler) / len(text) < 0.15:
        return False, "⚠️ Hata: Metin doğal bir dil yapısına sahip görünmüyor."

    rakamlar = re.findall(r'[0-9]', text)
    if len(rakamlar) / len(text) > 0.4:
        return False, "⚠️ Hata: Metin çok fazla sayı içeriyor, haber metni niteliği taşımıyor."

    return True, ""

# --- 2. TÜRKÇE META VERİ SÖZLÜĞÜ ---
EXIF_TR = {
    'Make': 'Cihaz Üreticisi', 'Model': 'Cihaz Modeli', 'Software': 'Yazılım/Editör',
    'DateTime': 'Oluşturulma Tarihi', 'ExifImageWidth': 'Genişlik (px)',
    'ExifImageHeight': 'Yükseklik (px)', 'XResolution': 'Yatay Çözünürlük',
    'YResolution': 'Dikey Çözünürlük', 'ResolutionUnit': 'Birim',
    'Orientation': 'Yönelim', 'ExposureMode': 'Pozlama Modu', 'Flash': 'Flaş',
    'FocalLength': 'Odak Uzaklığı', 'ISOSpeedRatings': 'ISO', 'ExposureTime': 'Pozlama Süresi',
    'FNumber': 'Diyafram (F)', 'SceneType': 'Sahne Tipi', 'ColorSpace': 'Renk Uzayı'
}

# --- 3. AI TESPİT API (SIGHTENGINE) ---
def ai_kontrol_api(image_path):
    try:
        if "api_user" not in st.secrets or "api_secret" not in st.secrets:
            st.error("❌ Hata: API anahtarları tanımlanmamış!")
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
            err_msg = output.get('error', {}).get('message', 'Bilinmeyen hata')
            st.error(f"API Hatası: {err_msg}")
            return None
    except Exception as e:
        st.error(f"Bağlantı Hatası: {str(e)}")
        return None

# --- 4. VERİ SETİ VE MODEL EĞİTİMİ ---
@st.cache_resource
def izlek_beyin_egit():
    dosya = "nlp_egitim_veri_seti.csv"
    if not os.path.exists(dosya):
        data = []
        g_kalip = ["Resmi makamlarca yapılan açıklamada {0} {1}.", "{0} tarafından onaylandı."]
        s_kalip = ["ŞOK İDDİA! {0} aslında {1}!", "{0} gerçeği şok etti!"]
        ozneler = ["Bakanlık", "TÜBİTAK", "Valilik"]
        eylemler = ["yeni çalışma başlattı", "açıklama yaptı"]
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

# --- 5. SAYFA AYARLARI VE GELİŞMİŞ CSS ---
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url)

st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Üst Bar */
    .header-bar {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.5);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        margin-bottom: 30px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
    }
    
    /* Künye Kutusu */
    .kunya-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00d2ff;
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 10px;
        font-size: 14px;
        color: #e0e0e0 !important;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Meta Veri Kartları */
    .meta-card {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff !important;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        transition: transform 0.3s;
    }
    .meta-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: #00d2ff;
    }
    .meta-card b { color: #00d2ff !important; }
    
    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d2ff !important;
        color: black !important;
    }

    /* Butonlar */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 20px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #00d2ff;
        transform: scale(1.02);
    }

    /* Sidebar Logo */
    .sidebar-logo-box {
        background: white;
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(255,255,255,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. YAN PANEL ---
with st.sidebar:
    logo_b64 = get_base64_image("YTFL LOGO.jpg")
    if logo_b64:
        st.markdown(f'<div class="sidebar-logo-box"><img src="data:image/jpeg;base64,{logo_b64}" width="100%"></div>', unsafe_allow_html=True)
    
    st.markdown("### 💠 Sistem Künyesi")
    st.markdown(f'''<div class="kunya-box">
        <span style="color:#00d2ff; font-weight:bold;">PROJE:</span> Sahte Haber ve Görsel Tespiti<br>
        <span style="color:#00d2ff; font-weight:bold;">DANIŞMAN:</span> Hasan ERSÜRER<br>
        <hr style="opacity:0.2;">
        <b>ÖĞRENCİLER:</b><br>
        ⚡ Abdullah ELŞAHAP<br>
        ⚡ Harun Buğra ŞANVERDİ<br>
        ⚡ Hasan Kayra GÜLLÜ<br>
        <hr style="opacity:0.2;">
        <b>OKUL:</b> Yahya Turan Fen Lisesi<br>
        <b>TEKNOLOJİ:</b> NLP • ELA • EXIF
    </div>''', unsafe_allow_html=True)
    st.write("")
    st.success("Sistem Aktif: Tarama Hazır ✅")

# --- 7. ANA SAYFA ---
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}" style="border-radius:5px; box-shadow: 0 0 10px rgba(255,255,255,0.3);">
        <h1 style="color: #FFD700; margin:0; font-size: 26px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">YTFL İZLEK: Dijital Doğrulama Merkezi</h1>
    </div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡 METİN ANALİZİ", "⚡ GÖRSEL ANALİZ (ELA)", "🔍 DİJİTAL İZ (EXIF)"])

with tab1:
    st.markdown("### 🤖 Haber Doğrulama Ünitesi")
    metin = st.text_area("Analiz edilecek veri girişini yapın:", height=150, placeholder="Haber metnini buraya yapıştırın...")
    if st.button("SİSTEMİ ÇALIŞTIR"):
        gecerli, mesaj = metin_gecerli_mi(metin)
        if not gecerli:
            st.warning(mesaj)
        else:
            with st.spinner("Yapay Zeka nöral ağları taranıyor..."):
                time.sleep(1)
                vec = vectorizer.transform([metin])
                tahmin = model.predict(vec)[0]
                olasilik = model.predict_proba(vec)[0]
                if tahmin == 1:
                    st.error(f"🚨 TESPİT: MANİPÜLE EDİLMİŞ İÇERİK (%{olasilik[1]*100:.1f})")
                else:
                    st.success(f"✅ TESPİT: DOĞRULANMIŞ İÇERİK (%{olasilik[0]*100:.1f})")

with tab2:
    st.markdown("### 📸 Görsel Manipülasyon Filtresi")
    yukle = st.file_uploader("Görsel yükle (JPG/JPEG):", type=['jpg', 'jpeg'])
    if yukle:
        with open("temp.jpg", "wb") as f: f.write(yukle.getbuffer())
        c1, c2 = st.columns(2)
        c1.image(yukle, caption="ORİJİNAL VERİ", width='stretch')
        
        im = Image.open("temp.jpg").convert('RGB')
        im.save("resaved.jpg", 'JPEG', quality=90)
        resaved = Image.open("resaved.jpg")
        ela = ImageChops.difference(im, resaved)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        ela_viz = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        c2.image(ela_viz, caption="HATA SEVİYESİ ANALİZİ (ELA)", width='stretch')
        st.info("ℹ️ **Analiz Notu:** ELA sonucunda homojen olmayan parlaklıklar, piksellerin yeniden kaydedildiğini veya değiştirildiğini gösterir.")

with tab3:
    st.markdown("### 📂 Dijital Parmak İzi Analizi")
    if 'yukle' in locals() and yukle:
        try:
            img = Image.open("temp.jpg")
            exif_data = img._getexif()
            if exif_data:
                st.write("🛰️ Gömülü Meta Veriler Çözümlendi:")
                cols = st.columns(3)
                for i, (tag, value) in enumerate(exif_data.items()):
                    etiket_ing = TAGS.get(tag, tag)
                    etiket_tr = EXIF_TR.get(etiket_ing, etiket_ing)
                    if isinstance(value, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{etiket_tr}</b><br>{value}</div>', unsafe_allow_html=True)
            else: st.warning("⚠️ Bu görselde teknik bir iz tespit edilemedi.")
        except: st.error("❌ Veri okuma hatası.")
        
        st.divider()
        if st.button("NİHAİ AI KARARINI ÜRET"):
            with st.spinner("Derin öğrenme modelleri görseli test ediyor..."):
                olasılık = ai_kontrol_api("temp.jpg")
                if olasılık is not None:
                    if olasılık > 0.5:
                        st.markdown(f'<div class="result-box" style="background-color: rgba(211, 47, 47, 0.2); color: #ff5f5f; border-color: #d32f2f;">KRİTİK: %{olasılık*100:.1f} Olasılıkla AI Tarafından Üretilmiş! 🚨</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box" style="background-color: rgba(46, 125, 50, 0.2); color: #66bb6a; border-color: #2e7d32;">GÜVENLİ: %{(1-olasılık)*100:.1f} Olasılıkla Gerçek Çekim ✅</div>', unsafe_allow_html=True)
                else:
                    st.warning("📡 API bağlantısı başarısız. Lütfen konfigürasyonu kontrol edin.")
    else: st.info("🔍 Lütfen önce bir görsel yükleyin.")

# --- 8. FOOTER ---
meb_b64 = get_base64_image("meb.png")
tubitak_b64 = get_base64_image("tubitak.png")
st.markdown(f'''
    <div style="background: rgba(255,255,255,0.03); padding: 30px; border-radius: 20px; margin-top: 50px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
        <div style="display: flex; justify-content: center; gap: 50px; margin-bottom: 15px;">
            <img src="data:image/png;base64,{meb_b64}" height="60" style="filter: drop-shadow(0 0 5px white);">
            <img src="data:image/png;base64,{tubitak_b64}" height="60" style="filter: drop-shadow(0 0 5px white);">
        </div>
        <p style="color: #888; font-size: 0.9em; letter-spacing: 1px;">
            REYHANLI YAHYA TURAN FEN LİSESİ • TÜBİTAK 4006 BİLİM FUARI • 2026
        </p>
    </div>
''', unsafe_allow_html=True)
