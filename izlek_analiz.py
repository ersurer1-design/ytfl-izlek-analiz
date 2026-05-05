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

# --- 1. METİN GEÇERLİLİK KONTROLÜ ---
def metin_gecerli_mi(text):
    text = text.strip()
    if len(text) < 50:
        return False, "Hata: Analiz için en az 50 karakterlik bir haber metni girmelisiniz."
    benzersiz_karakterler = set(text.lower())
    if len(benzersiz_karakterler) < 8:
        return False, "Hata: Metin anlamsız görünüyor (karakter çeşitliliği düşük)."
    sesli_harfler = re.findall(r'[aeıioöuüAEIİOÖUÜ]', text)
    if len(sesli_harfler) / len(text) < 0.15:
        return False, "Hata: Metin doğal bir dil yapısına sahip görünmüyor."
    return True, ""

# --- 2. TÜRKÇE META VERİ SÖZLÜĞÜ ---
EXIF_TR = {
    'Make': 'Cihaz Üreticisi', 'Model': 'Cihaz Modeli', 'Software': 'Yazılım',
    'DateTime': 'Tarih/Saat', 'ExifImageWidth': 'Genişlik', 'ExifImageHeight': 'Yükseklik',
    'XResolution': 'Yatay Çözünürlük', 'YResolution': 'Dikey Çözünürlük',
    'Orientation': 'Yön', 'ExposureMode': 'Pozlama', 'Flash': 'Flaş',
    'ISOSpeedRatings': 'ISO', 'FNumber': 'Diyafram'
}

# --- 3. AI TESPİT API ---
def ai_kontrol_api(image_path):
    try:
        params = {'models': 'genai', 'api_user': st.secrets["api_user"], 'api_secret': st.secrets["api_secret"]}
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        output = response.json()
        if output['status'] == 'success': return output['type']['ai_generated']
        return None
    except: return None

# --- 4. MODEL EĞİTİMİ (NLP) ---
@st.cache_resource
def izlek_beyin_egit():
    dosya = "nlp_egitim_veri_seti.csv"
    if not os.path.exists(dosya):
        data = []
        g_kalip = ["Resmi makamlarca yapılan açıklamada {0} {1}.", "{0} tarafından duyurulan kararla {1} onaylandı."]
        s_kalip = ["ŞOK İDDİA! {0} aslında {1} olduğu gizleniyor!", "{0} hakkında skandal görüntü şok etti!"]
        ozneler = ["Bakanlık", "TÜBİTAK", "Valilik"]
        eylemler = ["yeni çalışma başlattı", "başarıya ulaştı", "açıklama yaptı"]
        for _ in range(3000):
            data.append([random.choice(g_kalip).format(random.choice(ozneler), random.choice(eylemler)), 0])
            data.append([random.choice(s_kalip).format(random.choice(ozneler), "manipüle edildi"), 1])
        pd.DataFrame(data, columns=['text', 'label']).to_csv(dosya, index=False, encoding='utf-8-sig')
    df = pd.read_csv(dosya)
    v = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X = v.fit_transform(df['text'].astype(str))
    m = MultinomialNB(alpha=0.1); m.fit(X, df['label'])
    return v, m

vectorizer, model = izlek_beyin_egit()

# --- 5. SAYFA AYARLARI VE CSS ---
# Bayrak URL'sini daha küçük ve stabil bir sürümle değiştirdik
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url)

st.markdown("""
    <style>
    .stApp { background-image: url('https://www.transparenttextures.com/patterns/carbon-fibre.png'); }
    .header-bar { 
        background-color: #FFD700; padding: 15px; border-radius: 10px; 
        display: flex; align-items: center; margin-bottom: 25px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .header-bar img { height: 40px; width: auto; margin-right: 20px; object-fit: contain; }
    .kunya-box { background-color: #f0f7ff; border-left: 5px solid #1e3c72; padding: 15px; border-radius: 5px; font-size: 14px; color: #1e3c72 !important; }
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
        • Urva SALAMA<br>
        <hr style="margin: 8px 0; border:0; border-top: 1px solid #d1d1d1;">
        <b>Okul:</b> Yahya Turan Fen Lisesi<br>
        <b>Teknoloji:</b> NLP & ELA Analizi
    </div>''', unsafe_allow_html=True)
    st.write(""); st.success("Sistem Durumu: Hazır ✅")

# --- 7. ANA SAYFA ---
# Header kısmındaki bayrağın parçalı görünmesini engellemek için CSS'i sabitledik
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}">
        <h1 style="color: black; margin:0; font-size: 24px; font-family: sans-serif;">YTFL İzlek: Dijital Doğrulama Sistemi</h1>
    </div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Metin Analizi", "🖼️ Görsel Analiz (ELA)", "📂 Meta Veri (EXIF)"])

with tab1:
    st.subheader("Haber Doğrulama (NLP)")
    metin = st.text_area("Haber metni:", height=150, placeholder="Analiz için en az 50 karakter girin...")
    if st.button("Analizi Başlat"):
        gecerli, mesaj = metin_gecerli_mi(metin)
        if not gecerli: st.warning(mesaj)
        else:
            with st.spinner("İnceleniyor..."):
                time.sleep(0.5); vec = vectorizer.transform([metin])
                tahmin = model.predict(vec)[0]; olasilik = model.predict_proba(vec)[0]
                if tahmin == 1: st.error(f"🚨 ŞÜPHELİ (%{olasilik[1]*100:.1f})")
                else: st.success(f"✅ GÜVENİLİR (%{olasilik[0]*100:.1f})")

with tab2:
    st.subheader("Görsel Analiz (ELA)")
    yukle = st.file_uploader("Görsel Yükleyin (JPG):", type=['jpg', 'jpeg'])
    if yukle:
        with open("temp.jpg", "wb") as f: f.write(yukle.getbuffer())
        c1, c2 = st.columns(2)
        c1.image(yukle, caption="Orijinal", use_container_width=True)
        im = Image.open("temp.jpg").convert('RGB'); im.save("resaved.jpg", 'JPEG', quality=90)
        resaved = Image.open("resaved.jpg"); ela = ImageChops.difference(im, resaved)
        extrema = ela.getextrema(); max_diff = max([ex[1] for ex in extrema]) or 1
        ela_viz = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        c2.image(ela_viz, caption="ELA Analizi", use_container_width=True)
        st.info("💡 **ELA Analizi:** Parlak alanlar dijital manipülasyon izlerini gösterebilir.")

with tab3:
    st.subheader("Dijital İz Analizi (Metadata)")
    if 'yukle' in locals() and yukle:
        try:
            img = Image.open("temp.jpg"); exif_data = img._getexif()
            if exif_data:
                cols = st.columns(3)
                for i, (tag, value) in enumerate(exif_data.items()):
                    tr_etiket = EXIF_TR.get(TAGS.get(tag, tag), TAGS.get(tag, tag))
                    if isinstance(value, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{tr_etiket}:</b><br>{value}</div>', unsafe_allow_html=True)
            else: st.warning("Teknik veri bulunamadı.")
        except: st.error("Hata oluştu.")
        
        st.divider()
        if st.button("Nihai AI Analizini Yap"):
            with st.spinner("AI Analiz ediliyor..."):
                olasılık = ai_kontrol_api("temp.jpg")
                if olasılık is not None:
                    if olasılık > 0.5:
                        st.markdown(f'<div class="result-box" style="background-color: #ffebee; color: #d32f2f; border-color: #d32f2f;">Analiz Sonucu: %{olasılık*100:.1f} Olasılıkla Yapay Zekâ Ürünü 🚨</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box" style="background-color: #e8f5e9; color: #2e7d32; border-color: #2e7d32;">Analiz Sonucu: %{(1-olasılık)*100:.1f} Olasılıkla Gerçek Çekim ✅</div>', unsafe_allow_html=True)

# --- 8. FOOTER ---
meb_b64 = get_base64_image("meb.png"); tubitak_b64 = get_base64_image("tubitak.png")
st.markdown(f'''
    <div style="background-color: white; padding: 15px; border-radius: 10px; display: flex; flex-direction: column; align-items: center; gap: 10px; margin-top: 30px;">
        <div style="display: flex; gap: 40px;"><img src="data:image/png;base64,{meb_b64}" height="50"><img src="data:image/png;base64,{tubitak_b64}" height="50"></div>
        <div style="color: #666; font-size: 0.8em;">© 2026 - Reyhanlı Yahya Turan Fen Lisesi TÜBİTAK 4006 Projesi</div>
    </div>
''', unsafe_allow_html=True)