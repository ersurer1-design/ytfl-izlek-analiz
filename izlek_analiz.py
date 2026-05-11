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

# --- 1. METİN ANALİZ FONKSİYONLARI ---
def metin_gecerli_mi(text):
    text = text.strip()
    if len(text) < 50:
        return False, "Hata: Analiz için en az 50 karakter giriniz."
    benzersiz_karakterler = set(text.lower())
    if len(benzersiz_karakterler) < 8:
        return False, "Hata: Metin anlamsız (karakter çeşitliliği düşük)."
    return True, ""

# --- 2. TÜRKÇE META VERİ SÖZLÜĞÜ ---
EXIF_TR = {
    'Make': 'Üretici', 'Model': 'Model', 'Software': 'Yazılım',
    'DateTime': 'Tarih/Saat', 'ExifImageWidth': 'Genişlik',
    'ExifImageHeight': 'Yükseklik', 'Orientation': 'Yön',
    'ISOSpeedRatings': 'ISO', 'FNumber': 'Diyafram',
    'ExposureTime': 'Pozlama', 'FocalLength': 'Odak Uzaklığı'
}

# --- 3. AI TESPİT API ---
def ai_kontrol_api(image_path):
    try:
        if "api_user" not in st.secrets: return None
        params = {'models': 'genai', 'api_user': st.secrets["api_user"], 'api_secret': st.secrets["api_secret"]}
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        output = response.json()
        return output.get('type', {}).get('ai_generated', 0) if output.get('status') == 'success' else None
    except: return None

# --- 4. NLP MODEL EĞİTİMİ ---
@st.cache_resource
def izlek_beyin_egit():
    dosya = "nlp_egitim_veri_seti.csv"
    if not os.path.exists(dosya):
        data = []
        for _ in range(3000):
            data.append(["Resmi makamlarca yapılan açıklama onaylandı.", 0])
            data.append(["ŞOK İDDİA! Gizli bilgiler sızdırıldı!", 1])
        pd.DataFrame(data, columns=['text', 'label']).to_csv(dosya, index=False, encoding='utf-8-sig')
    df = pd.read_csv(dosya)
    v = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X = v.fit_transform(df['text'].astype(str))
    m = MultinomialNB(alpha=0.1).fit(X, df['label'])
    return v, m

vectorizer, model = izlek_beyin_egit()

# --- 5. GELİŞMİŞ SADE & KOYU TEMA (CSS) ---
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek", layout="wide", page_icon=bayrak_url)

st.markdown("""
    <style>
    /* Global Koyu Arka Plan */
    .stApp {
        background-color: #0b0d10;
        color: #e0e0e0;
    }
    
    /* Sade Başlık Çubuğu */
    .header-bar {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px 25px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        margin-bottom: 25px;
    }
    
    /* Künye ve Kart Tasarımı */
    .kunya-box, .meta-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 6px;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .meta-card b { color: #58a6ff; } /* Teknik Mavi */

    /* Tab Tasarımı */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }

    /* Buton Tasarımı */
    .stButton>button {
        background-color: #21262d;
        color: #58a6ff;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 20px;
        transition: 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }

    /* Sidebar düzeni */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("### [ YTFL ANALİZ ]")
    st.markdown(f'''<div class="kunya-box">
        <b style="color:#58a6ff;">PROJE:</b> Sahte Haber Tespiti<br>
        <b style="color:#58a6ff;">DANIŞMAN:</b> Hasan ERSÜRER<br>
        <hr style="border-top: 1px solid #30363d;">
        <b>ÖĞRENCİLER:</b><br>
        > Abdullah ELŞAHAP<br>
        > Harun Buğra ŞANVERDİ<br>
        > Hasan Kayra GÜLLÜ<br>
        <hr style="border-top: 1px solid #30363d;">
        <b>TEKNOLOJİ:</b> NLP & ELA
    </div>''', unsafe_allow_html=True)
    st.write("")
    st.code("Sistem: Çevrimiçi\nStatus: OK", language="bash")

# --- 7. ANA EKRAN ---
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}" width="35" style="margin-right:15px; opacity:0.8;">
        <h2 style="color: #c9d1d9; margin:0; font-family: sans-serif; font-size: 20px; letter-spacing: 1px;">
            YTFL İZLEK // DİJİTAL DOĞRULAMA SİSTEMİ
        </h2>
    </div>
''', unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["[ 01 ] METİN ANALİZİ", "[ 02 ] GÖRSEL ANALİZ", "[ 03 ] DİJİTAL İZ"])

with t1:
    st.markdown("#### Metin Veri Girişi")
    txt = st.text_area("Analiz edilecek haber metni:", height=150)
    if st.button("ANALİZİ BAŞLAT"):
        gecerli, msg = metin_gecerli_mi(txt)
        if not gecerli: st.warning(msg)
        else:
            with st.spinner("Taranıyor..."):
                time.sleep(0.5)
                vec = vectorizer.transform([txt])
                prob = model.predict_proba(vec)[0]
                if model.predict(vec)[0] == 1:
                    st.error(f"DURUM: ŞÜPHELİ // GÜVEN SKORU: %{prob[1]*100:.1f}")
                else:
                    st.success(f"DURUM: GÜVENİLİR // GÜVEN SKORU: %{prob[0]*100:.1f}")

with t2:
    st.markdown("#### Görsel Hata Seviyesi Analizi (ELA)")
    img_file = st.file_uploader("Görsel Yükle (JPG/JPEG)", type=['jpg', 'jpeg'])
    if img_file:
        with open("temp.jpg", "wb") as f: f.write(img_file.getbuffer())
        c1, c2 = st.columns(2)
        c1.image(img_file, caption="HAM VERİ", width='stretch')
        im = Image.open("temp.jpg").convert('RGB')
        im.save("resaved.jpg", 'JPEG', quality=90)
        ela = ImageChops.difference(im, Image.open("resaved.jpg"))
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        ela_viz = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        c2.image(ela_viz, caption="ELA TARAMASI", width='stretch')

with t3:
    st.markdown("#### Metadata & Yapay Zeka Taraması")
    if 'img_file' in locals() and img_file:
        try:
            exif = Image.open("temp.jpg")._getexif()
            if exif:
                cols = st.columns(3)
                for i, (tag, val) in enumerate(exif.items()):
                    t_tr = EXIF_TR.get(TAGS.get(tag, tag), TAGS.get(tag, tag))
                    if isinstance(val, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{t_tr}</b><br>{val}</div>', unsafe_allow_html=True)
            else: st.info("Görselde meta veri bulunamadı.")
        except: st.error("Veri hatası.")
        
        st.divider()
        if st.button("NİHAİ YAPAY ZEKA KONTROLÜ"):
            with st.spinner("İşleniyor..."):
                res = ai_kontrol_api("temp.jpg")
                if res is not None:
                    color = "#f85149" if res > 0.5 else "#238636"
                    label = "YAPAY ZEKA ÜRÜNÜ" if res > 0.5 else "GERÇEK ÇEKİM"
                    st.markdown(f'<div style="padding:20px; border-radius:6px; border:1px solid {color}; color:{color}; text-align:center; font-weight:bold;">SONUÇ: {label} (%{res*100:.1f})</div>', unsafe_allow_html=True)

# --- 8. FOOTER ---
st.markdown(f'''
    <div style="margin-top: 50px; padding: 20px; border-top: 1px solid #30363d; text-align: center; opacity: 0.6; font-size: 0.8em;">
        REYHANLI YAHYA TURAN FEN LİSESİ • 2026 • TÜBİTAK 4006
    </div>
''', unsafe_allow_html=True)
