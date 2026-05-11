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
    'DateTime': 'Oluşturulma Tarihi', 'ExifImageWidth': 'Genişlik (px)',
    'ExifImageHeight': 'Yükseklik (px)', 'XResolution': 'Yatay Çözünürlük',
    'YResolution': 'Dikey Çözünürlük', 'ResolutionUnit': 'Birim',
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

# --- 5. SAYFA AYARLARI VE CSS (İSTEKLERE GÖRE DÜZENLENDİ) ---
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url)

st.markdown("""
    <style>
    /* GitHub ve Share Butonlarını Gizle */
    header [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    
    /* Manage App Butonunu Koru */
    /* Manage app butonu sahibi olduğunuzda sağ altta çıkar, CSS ile ezmiyoruz */
    
    /* Dinamik Scroll Bar Ayarı */
    /* İçerik sığmıyorsa scroll bar çıkar, sığıyorsa kaybolur */
    .main, .stApp {
        overflow-y: auto !important;
    }

    /* Koyu Sade Tekno Tema */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    .header-bar {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-bar h1 { color: #58a6ff !important; margin:0; font-size: 22px; font-family: sans-serif; }
    .header-bar img { height: 35px; margin-right: 15px; }

    .kunya-box {
        background-color: #010409;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        padding: 15px;
        border-radius: 6px;
        font-size: 14px;
        color: #8b949e !important;
    }
    
    .meta-card {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.9em;
    }
    .meta-card b { color: #58a6ff !important; }

    /* Butonlar */
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

    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab"] { color: #8b949e !important; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. YAN PANEL (SIDEBAR) ---
with st.sidebar:
    logo_b64 = get_base64_image("YTFL LOGO.jpg")
    if logo_b64:
        st.markdown(f'<div style="text-align:center; padding:10px; background:white; border-radius:10px; margin-bottom:20px;"><img src="data:image/jpeg;base64,{logo_b64}" width="130"></div>', unsafe_allow_html=True)
    
    st.markdown("### [ YTFL ANALİZ ]")
    st.markdown(f'''<div class="kunya-box">
        <b style="color:#58a6ff;">PROJE:</b> Sahte Haber Tespiti<br>
        <b>Danışman:</b> Hasan ERSÜRER<br>
        <hr style="border-top:1px solid #30363d;">
        <b>Öğrenciler:</b><br>
        > Abdullah ELŞAHAP<br>
        > Harun Buğra ŞANVERDİ<br>
        > Hasan Kayra GÜLLÜ<br>
        <hr style="border-top:1px solid #30363d;">
        <b>Okul:</b> Yahya Turan Fen Lisesi<br>
        <b>Teknoloji:</b> NLP & ELA Analizi
    </div>''', unsafe_allow_html=True)
    st.write(""); st.success("Sistem Durumu: Aktif ✅")

# --- 7. ANA SAYFA ---
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}">
        <h1>YTFL İZLEK // Dijital Doğrulama Merkezi</h1>
    </div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["[ 01 ] METİN ANALİZİ", "[ 02 ] GÖRSEL ANALİZ", "[ 03 ] DİJİTAL İZ"])

with tab1:
    st.subheader("Haber Doğrulama (NLP)")
    metin = st.text_area("Analiz edilecek haber metnini giriniz:", height=150)
    if st.button("TARAMAYI BAŞLAT"):
        gecerli, mesaj = metin_gecerli_mi(metin)
        if not gecerli:
            st.warning(mesaj)
        else:
            with st.spinner("Vektörel analiz yapılıyor..."):
                time.sleep(0.8)
                vec = vectorizer.transform([metin])
                tahmin = model.predict(vec)[0]
                olasilik = model.predict_proba(vec)[0]
                if tahmin == 1:
                    st.error(f"🚨 TESPİT: ŞÜPHELİ (%{olasilik[1]*100:.1f})")
                else:
                    st.success(f"✅ TESPİT: GÜVENİLİR (%{olasilik[0]*100:.1f})")

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
        c2.image(ela_viz, caption="Hata Seviyesi Analizi", width='stretch')

with tab3:
    st.subheader("Metadata & AI Analizi")
    if 'yukle' in locals() and yukle:
        try:
            img = Image.open("temp.jpg")
            exif_data = img._getexif()
            if exif_data:
                cols = st.columns(3)
                for i, (tag, value) in enumerate(exif_data.items()):
                    etiket_ing = TAGS.get(tag, tag)
                    etiket_tr = EXIF_TR.get(etiket_ing, etiket_ing)
                    if isinstance(value, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{etiket_tr}:</b><br>{value}</div>', unsafe_allow_html=True)
            else: st.warning("Bu görselde teknik iz bulunamadı.")
        except: st.error("Teknik veri okuma hatası.")
        
        st.divider()
        if st.button("NİHAİ AI KONTROLÜNÜ ÇALIŞTIR"):
            with st.spinner("AI Modelleri DNA taraması yapıyor..."):
                olasılık = ai_kontrol_api("temp.jpg")
                if olasılık is not None:
                    if olasılık > 0.5:
                        st.markdown(f'<div style="padding:20px; border-radius:6px; border:1px solid #f85149; color:#f85149; text-align:center; font-weight:bold;">SONUÇ: %{olasılık*100:.1f} Olasılıkla Yapay Zekâ Ürünü 🚨</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="padding:20px; border-radius:6px; border:1px solid #238636; color:#238636; text-align:center; font-weight:bold;">SONUÇ: %{(1-olasılık)*100:.1f} Olasılıkla Gerçek Çekim ✅</div>', unsafe_allow_html=True)
    else: st.info("Lütfen önce Görsel Analiz sekmesinden bir fotoğraf yükleyin.")

# --- 8. FOOTER ---
st.markdown('''
    <div style="margin-top: 50px; padding: 20px; border-top: 1px solid #30363d; text-align: center; opacity: 0.6; font-size: 0.8em;">
        REYHANLI YAHYA TURAN FEN LİSESİ • 2026 • TÜBİTAK 4006
    </div>
''', unsafe_allow_html=True)
