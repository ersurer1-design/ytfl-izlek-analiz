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
    'DateTime': 'Oluşturulma Tarihi', 'ExifImageWidth': 'Görsel Genişliği (Piksel)',
    'ExifImageHeight': 'Görsel Yüksekliği (Piksel)', 'XResolution': 'Yatay Çözünürlük',
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

# --- 5. SAYFA AYARLARI VE CSS (CYBER-TECH TEMA) ---
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url)

st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%);
        color: #ffffff;
    }
    
    /* Üst Başlık Çubuğu */
    .header-bar {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.4);
        padding: 20px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
    }
    .header-bar img { height: 45px; margin-right: 20px; }
    
    /* Künye Kutusu */
    .kunya-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00d2ff;
        padding: 20px;
        border-radius: 10px;
        font-size: 14px;
        color: #e0e0e0 !important;
        backdrop-filter: blur(5px);
    }
    
    /* Meta Veri Kartları */
    .meta-card {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 0.9em;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .meta-card b { color: #00d2ff !important; }
    
    /* Sidebar */
    .sidebar-logo-box {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Butonlar */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 10px #00d2ff;
        transform: scale(1.02);
    }

    /* Sonuç Kutuları */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
        margin-top: 20px;
        border: 2px solid;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. YAN PANEL (SIDEBAR) ---
with st.sidebar:
    logo_b64 = get_base64_image("YTFL LOGO.jpg")
    if logo_b64:
        st.markdown(f'<div class="sidebar-logo-box"><img src="data:image/jpeg;base64,{logo_b64}" width="140"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Proje Künyesi")
    st.markdown(f'''<div class="kunya-box">
        <b style="color:#00d2ff;">PROJE:</b> Sahte Haber ve Görsel Tespiti<br>
        <b>Danışman:</b> Hasan ERSÜRER<br>
        <hr style="opacity:0.2;">
        <b>Öğrenciler:</b><br>
        • Abdullah ELŞAHAP<br>
        • Harun Buğra ŞANVERDİ<br>
        • Hasan Kayra GÜLLÜ<br>
        <hr style="opacity:0.2;">
        <b>Okul:</b> Yahya Turan Fen Lisesi<br>
        <b>Teknoloji:</b> NLP & ELA & AI Analizi
    </div>''', unsafe_allow_html=True)
    st.write("")
    st.success("Sistem Durumu: Aktif ✅")

# --- 7. ANA SAYFA ---
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}">
        <h1 style="color: #FFD700; margin:0; font-size: 24px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">YTFL İZLEK: Dijital Doğrulama Sistemi</h1>
    </div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Metin Analizi", "🖼️ Görsel Analiz (ELA)", "📂 Meta Veri (EXIF)"])

with tab1:
    st.subheader("Haber Doğrulama (NLP)")
    metin = st.text_area("Analiz edilecek haber metnini giriniz:", height=150, placeholder="En az 50 karakterlik bir metin girin...")
    if st.button("ANALİZİ BAŞLAT"):
        gecerli, mesaj = metin_gecerli_mi(metin)
        if not gecerli:
            st.warning(mesaj)
        else:
            with st.spinner("Yapay zeka örüntüleri tarıyor..."):
                time.sleep(0.8)
                vec = vectorizer.transform([metin])
                tahmin = model.predict(vec)[0]
                olasilik = model.predict_proba(vec)[0]
                if tahmin == 1: st.error(f"🚨 SONUÇ: ŞÜPHELİ (%{olasilik[1]*100:.1f})")
                else: st.success(f"✅ SONUÇ: GÜVENİLİR (%{olasilik[0]*100:.1f})")

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
        st.info("💡 **İpucu:** Sağdaki görselde parlak alanlar dijital müdahale izlerini temsil eder.")

with tab3:
    st.subheader("Dijital İz Analizi (Metadata)")
    if 'yukle' in locals() and yukle:
        try:
            img = Image.open("temp.jpg")
            exif_data = img._getexif()
            if exif_data:
                st.write("Görsel teknik verileri (Analiz Edildi):")
                cols = st.columns(3)
                for i, (tag, value) in enumerate(exif_data.items()):
                    etiket_ing = TAGS.get(tag, tag)
                    etiket_tr = EXIF_TR.get(etiket_ing, etiket_ing)
                    if isinstance(value, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{etiket_tr}:</b><br>{value}</div>', unsafe_allow_html=True)
            else: st.warning("Bu görselde meta veri bulunamadı.")
        except: st.error("Teknik veri okuma hatası!")
        
        st.divider()
        if st.button("NİHAİ AI KONTROLÜNÜ YAP"):
            with st.spinner("AI Modelleri görseli inceliyor..."):
                olasılık = ai_kontrol_api("temp.jpg")
                if olasılık is not None:
                    if olasılık > 0.5:
                        st.markdown(f'<div class="result-box" style="background-color: rgba(211, 47, 47, 0.2); color: #ff4b4b; border-color: #ff4b4b;">Analiz Sonucu: %{olasılık*100:.1f} Olasılıkla Yapay Zekâ Ürünü 🚨</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box" style="background-color: rgba(46, 125, 50, 0.2); color: #00ff7f; border-color: #00ff7f;">Analiz Sonucu: %{(1-olasılık)*100:.1f} Olasılıkla Gerçek Çekim ✅</div>', unsafe_allow_html=True)
                else:
                    st.warning("Analiz sonucu alınamadı. API/Secrets kontrolü yapın.")
    else: st.info("Lütfen Görsel Analiz sekmesinden fotoğraf yükleyin.")

# --- 8. FOOTER ---
meb_b64 = get_base64_image("meb.png")
tubitak_b64 = get_base64_image("tubitak.png")
st.markdown(f'''
    <div style="background-color: rgba(255,255,255,0.05); padding: 25px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); display: flex; flex-direction: column; align-items: center; gap: 15px; margin-top: 40px;">
        <div style="display: flex; gap: 50px;">
            <img src="data:image/png;base64,{meb_b64}" height="55" style="filter: brightness(0) invert(1);">
            <img src="data:image/png;base64,{tubitak_b64}" height="55" style="filter: brightness(0) invert(1);">
        </div>
        <div style="color: #aaa; font-size: 0.85em; text-align:center;">
            © 2026 - REYHANLI YAHYA TURAN FEN LİSESİ<br>TÜBİTAK 4006 BİLİM FUARI PROJESİ
        </div>
    </div>
''', unsafe_allow_html=True)
