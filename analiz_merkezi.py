import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import random
import re
import time
import base64
import requests

# --- 0. YARDIMCI FONKSİYONLAR ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# --- 1. TEKNİK VERİ HAVUZU (NLP) ---
def veriyi_hazirla():
    dosya = "nlp_egitim_veri_seti.csv"
    if not os.path.exists(dosya):
        g_ozneler = ["Cumhurbaşkanı Erdoğan", "Bakanlık", "TÜBİTAK", "Milli Takım", "Kenan Yıldız"]
        g_eylemler = ["açıkladı", "belirtildi", "duyurdu", "vurguladı"]
        s_ozneler = ["Şok iddia", "WhatsApp grupları", "Sosyal medya", "Gizli kaynaklar"]
        s_eylemler = ["iddia edildi", "görüldü", "şok etti", "ortaya atıldı"]
        data = []
        for _ in range(5000):
            go, ge = random.choice(g_ozneler), random.choice(g_eylemler)
            data.append([f"{go} tarafından yapılan açıklamada yeni bir gelişme {ge}.", 0])
            so, se = random.choice(s_ozneler), random.choice(s_eylemler)
            data.append([f"ŞOK! {so} tarafından paylaşılan videoda gerçekler {se}!", 1])
        pd.DataFrame(data, columns=['text', 'label']).to_csv(dosya, index=False, encoding='utf-8-sig')

veriyi_hazirla()

# --- 2. GÜVENLİK VE METİN KONTROLÜ (DÜZELTİLEN YAPI) ---
def is_valid_input(text):
    text = text.strip()
    # 1. Kural: Çok kısa metin engelleme
    if len(text) < 30: 
        return False, "⚠️ Hata: Analiz için en az 30 karakterlik anlamlı bir cümle girmelisiniz."
    # 2. Kural: Sadece sayı ve özel karakter engelleme (Harf kontrolü)
    if not any(c.isalpha() for c in text):
        return False, "⚠️ Hata: Giriş sadece sayı veya işaretlerden oluşamaz, lütfen metin girin."
    # 3. Kural: Karakter çeşitliliği (Anlamsız 'aaaaaa' gibi girişleri engeller)
    if len(set(text.lower())) < 6:
        return False, "⚠️ Hata: Girdiğiniz metin anlamsız görünüyor. Lütfen gerçek bir haber metni girin."
    return True, ""

# --- 3. AI TESPİT API (GÜVENLİ SÜRÜM) ---
def ai_kontrol_api(image_path):
    try:
        params = {
            'models': 'genai',
            'api_user': st.secrets["api_user"], 
            'api_secret': st.secrets["api_secret"]
        }
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        output = response.json()
        if output['status'] == 'success':
            return output['type']['ai_generated']
        return None
    except: return None

# --- 4. SAYFA AYARLARI VE CSS (ORTALANMIŞ LOGO) ---
bayrak_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Flag_of_Turkey.svg/1200px-Flag_of_Turkey.svg.png"
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url, initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    .stAppDeployButton, #stDecoration, header {{ display: none !important; }}
    .block-container {{ padding-top: 0rem !important; }}
    
    .stApp {{ 
        background-image: url('https://www.transparenttextures.com/patterns/carbon-fibre.png'); 
        background-attachment: fixed; 
    }}
    
    .header-bar {{ 
        background-color: #FFD700; padding: 20px; border-radius: 0 0 10px 10px; 
        display: flex; align-items: center; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .header-bar h1 {{ color: #000000 !important; margin: 0; font-weight: bold; font-size: calc(1.2rem + 1vw); }}
    
    /* SIDEBAR LOGO KUTUSU - TAM ORTALANDI */
    .sidebar-logo-box {{
        display: flex; 
        justify-content: center; /* SOLA YASLIYDI, ORTALANDI */
        align-items: center; 
        width: 100%;
        padding: 20px 0; 
        background-color: white; 
        border-bottom: 1px solid #f0f0f0; 
        margin-bottom: 10px;
    }}
    .sidebar-logo-box img {{ max-width: 160px; height: auto; }}
    
    [data-testid="stSidebar"] {{ background-color: #ffffff !important; border-right: 1px solid #e0e0e0; }}
    .kunya-box {{ background-color: #e1f5fe; border: 2px solid #0288d1; padding: 12px; border-radius: 10px; color: #01579b; font-size: 14px; }}
    .stButton>button {{ border-radius: 8px; background-color: #d92323; color: white; font-weight: bold; width: 100%; }}
    
    /* FOOTER VE MOBİL UYUM */
    .footer-white-bar {{
        background-color: white; width: 100%; padding: 20px; margin-top: 30px;
        border-radius: 10px; display: flex; flex-direction: column; align-items: center;
    }}
    .logo-container {{ display: flex; gap: 30px; flex-wrap: wrap; justify-content: center; align-items: center; }}
    .logo-container img {{ height: 60px; width: auto; }}

    @media (max-width: 768px) {{
        .header-bar h1 {{ font-size: 1.1rem !important; }}
        .logo-container img {{ height: 45px; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. MODEL EĞİTİMİ ---
@st.cache_resource
def izlek_beyin_egit():
    dataset_yolu = "nlp_egitim_veri_seti.csv"
    if os.path.exists(dataset_yolu):
        df = pd.read_csv(dataset_yolu, encoding='utf-8-sig')
        v = CountVectorizer(ngram_range=(1, 2), min_df=2)
        X = v.fit_transform(df['text'].astype(str))
        m = MultinomialNB(alpha=0.5)
        m.fit(X, df['label'])
        return v, m
    return None, None

vectorizer, model = izlek_beyin_egit()

# --- 6. GÖRSEL ANALİZ (ELA) ---
def compute_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    temporary = Image.open(temp_path)
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(255.0 / max_diff)
    return ela_image

# --- 7. YAN PANEL (SIDEBAR) ---
with st.sidebar:
    logo_b64 = get_base64_image("YTFL LOGO.jpg")
    if logo_b64:
        # LOGO BURADA DIV İÇERİSİNDE ORTALANIYOR
        st.markdown(f'<div class="sidebar-logo-box"><img src="data:image/jpeg;base64,{logo_b64}"></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1e3c72; text-align: center;'>Proje Künyesi</h3>", unsafe_allow_html=True)
    st.markdown(f'''<div class="kunya-box">
        <b>Proje:</b> Sahte Haber ve Görsel Tespiti<br>
        <b>Danışman:</b> Hasan ERSÜRER<br>
        <b>Okul:</b> Reyhanlı Yahya Turan Fen Lisesi
    </div>''', unsafe_allow_html=True)
    st.write(""); st.success("Sistem Hazır ✅")

# --- 8. ANA SAYFA ---
st.markdown(f'<div class="header-bar"><img src="{bayrak_url}" width="40" style="margin-right: 12px;"><h1>YTFL İzlek Analiz</h1></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Metin Analizi", "🖼️ Görsel Analiz"])

with tab1:
    st.subheader("Haber Doğrulama Modülü")
    metin = st.text_area("Analiz edilecek metni girin:", height=150)
    if st.button("Analizi Başlat"):
        # GÜVENLİK FİLTRESİ BURADA ÇALIŞIYOR
        valid, mesaj = is_valid_input(metin)
        if not valid:
            st.warning(mesaj)
        elif vectorizer and model:
            bar = st.progress(0)
            status = st.empty()
            for p in range(101):
                time.sleep(0.01)
                bar.progress(p)
                status.caption("Analiz ediliyor...")
            tahmin = model.predict(vectorizer.transform([metin]))[0]
            olasilik = model.predict_proba(vectorizer.transform([metin]))[0]
            bar.empty(); status.empty()
            if tahmin == 1:
                st.error(f"🚨 SONUÇ: ŞÜPHELİ (Risk: %{olasilik[1]*100:.1f})")
            else:
                st.success(f"✅ SONUÇ: GÜVENİLİR (Güven: %{olasilik[0]*100:.1f})")

with tab2:
    st.subheader("Görsel Manipülasyon Tespiti")
    yukle = st.file_uploader("Fotoğraf seçin:", type=['jpg', 'jpeg'])
    if yukle:
        with open("img.jpg", "wb") as f: f.write(yukle.getbuffer())
        ca, cb = st.columns(2)
        ca.image(yukle, caption="Orijinal Resim", use_container_width=True) 
        cb.image(compute_ela("img.jpg"), caption="ELA Analizi", use_container_width=True)
        
        st.divider()
        if st.button("Yapay Zeka (AI) Doğrulaması"):
            with st.spinner("Modeller taranıyor..."):
                olasılık = ai_kontrol_api("img.jpg")
                if olasılık is not None:
                    if olasılık > 0.6: st.error(f"🚨 ANALİZ: YAPAY ZEKA ÜRÜNÜ (%{olasılık*100:.1f})")
                    else: st.success(f"✅ ANALİZ: GERÇEK ÇEKİM (%{(1-olasılık)*100:.1f})")
                else: st.warning("API Hatası: Anahtarları kontrol edin.")

# --- 9. FOOTER ---
meb_b64 = get_base64_image("meb.png")
tubitak_b64 = get_base64_image("tubitak.png")
st.markdown(f'''
    <div class="footer-white-bar">
        <div class="logo-container">
            <img src="data:image/png;base64,{meb_b64}">
            <img src="data:image/png;base64,{tubitak_b64}">
        </div>
        <div style="color: #666; font-size: 0.9em; margin-top: 10px;">© 2026 - Yahya Turan Fen Lisesi TÜBİTAK 4006</div>
    </div>
''', unsafe_allow_html=True)
