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

# --- 5. SAYFA AYARLARI VE CSS (TEKNOLOJİK TEMA) ---
bayrak_url = "https://flagcdn.com/w80/tr.png" 
st.set_page_config(page_title="YTFL İzlek Analiz", layout="wide", page_icon=bayrak_url)

st.markdown(f"""
    <style>
    /* Ana Arka Plan */
    .stApp {{
        background: linear-gradient(135deg, #0a0e14 0%, #161b22 100%);
        color: #e6edf3;
    }}
    
    /* Header Bar */
    .header-bar {{
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.3);
        padding: 20px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        margin-bottom: 30px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.1);
    }}
    
    /* Künye Kutusu */
    .kunya-box {{
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-left: 5px solid #58a6ff;
        padding: 20px;
        border-radius: 10px;
        font-size: 14px;
        color: #c9d1d9 !important;
        line-height: 1.6;
    }}
    
    /* Meta Veri Kartları */
    .meta-card {{
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 0.9em;
        transition: transform 0.2s;
    }}
    .meta-card:hover {{
        transform: scale(1.02);
        border-color: #58a6ff !important;
    }}
    .meta-card b {{ color: #58a6ff !important; }}
    
    /* Sidebar Tasarımı */
    [data-testid="stSidebar"] {{
        background-color: #010409;
        border-right: 1px solid #30363d;
    }}
    .sidebar-logo-box {{
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }}

    /* Butonlar */
    .stButton>button {{
        background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }}
    .stButton>button:hover {{
        box-shadow: 0 0 15px rgba(56, 139, 253, 0.4);
        transform: translateY(-2px);
    }}

    /* Sonuç Kutuları */
    .result-box {{
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3em;
        margin-top: 25px;
        border: 2px dashed;
        backdrop-filter: blur(5px);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 6. YAN PANEL (SIDEBAR) ---
with st.sidebar:
    logo_b64 = get_base64_image("YTFL LOGO.jpg")
    if logo_b64:
        st.markdown(f'<div class="sidebar-logo-box"><img src="data:image/jpeg;base64,{logo_b64}" width="140"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Analiz Panel")
    st.markdown(f'''<div class="kunya-box">
        <b style="color:#58a6ff;">PROJE:</b> Sahte Haber ve Görsel Tespiti<br>
        <b>Danışman:</b> Hasan ERSÜRER<br>
        <hr style="border-color:#30363d;">
        <b>Öğrenciler:</b><br>
        • Abdullah ELŞAHAP<br>
        • Harun Buğra ŞANVERDİ<br>
        • Hasan Kayra GÜLLÜ<br>
        <hr style="border-color:#30363d;">
        <b>Konum:</b> Yahya Turan Fen Lisesi<br>
        <b>Teknoloji:</b> NLP & ELA & AI
    </div>''', unsafe_allow_html=True)
    st.write("")
    st.success("Sistem Durumu: Çevrimiçi ✅")

# --- 7. ANA SAYFA ---
st.markdown(f'''
    <div class="header-bar">
        <img src="{bayrak_url}" style="border-radius:4px;">
        <h1 style="color: #FFD700; margin:0; font-size: 26px; letter-spacing: 1px;">YTFL İZLEK: Dijital Doğrulama Merkezi</h1>
    </div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Metin Analizi", "🖼️ Görsel Analiz (ELA)", "📂 Meta Veri (EXIF)"])

with tab1:
    st.subheader("Haber Doğrulama Ünitesi (NLP)")
    metin = st.text_area("Analiz edilecek haber metnini giriniz:", height=180, placeholder="Veriyi buraya yapıştırın...")
    if st.button("TARAMAYI BAŞLAT"):
        gecerli, mesaj = metin_gecerli_mi(metin)
        if not gecerli:
            st.warning(mesaj)
        else:
            with st.spinner("Yapay zeka örüntüleri tarıyor..."):
                time.sleep(1)
                vec = vectorizer.transform([metin])
                tahmin = model.predict(vec)[0]
                olasilik = model.predict_proba(vec)[0]
                if tahmin == 1:
                    st.error(f"🚨 TESPİT: ŞÜPHELİ İÇERİK (%{olasilik[1]*100:.1f})")
                else:
                    st.success(f"✅ TESPİT: GÜVENİLİR İÇERİK (%{olasilik[0]*100:.1f})")

with tab2:
    st.subheader("Görsel Manipülasyon Filtresi (ELA)")
    yukle = st.file_uploader("Görsel yükleyin (JPG/JPEG):", type=['jpg', 'jpeg'])
    if yukle:
        with open("temp.jpg", "wb") as f: f.write(yukle.getbuffer())
        c1, c2 = st.columns(2)
        c1.image(yukle, caption="Ham Veri", width='stretch')
        
        im = Image.open("temp.jpg").convert('RGB')
        im.save("resaved.jpg", 'JPEG', quality=90)
        resaved = Image.open("resaved.jpg")
        ela = ImageChops.difference(im, resaved)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        ela_viz = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        c2.image(ela_viz, caption="ELA Tarama Sonucu", width='stretch')
        st.info("💡 **Hata Seviyesi Analizi:** Sağdaki görselde homojen olmayan parlak alanlar, dijital müdahale izlerini temsil eder.")

with tab3:
    st.subheader("Metadata & AI Taraması")
    if 'yukle' in locals() and yukle:
        try:
            img = Image.open("temp.jpg")
            exif_data = img._getexif()
            if exif_data:
                st.markdown("##### 🛰️ Gömülü Teknik Veriler:")
                cols = st.columns(3)
                for i, (tag, value) in enumerate(exif_data.items()):
                    etiket_ing = TAGS.get(tag, tag)
                    etiket_tr = EXIF_TR.get(etiket_ing, etiket_ing)
                    if isinstance(value, (str, int, float)):
                        cols[i % 3].markdown(f'<div class="meta-card"><b>{etiket_tr}:</b><br>{value}</div>', unsafe_allow_html=True)
            else: st.warning("Bu görselde meta veri izi bulunamadı.")
        except: st.error("Veri okuma hatası!")
        
        st.divider()
        if st.button("NİHAİ AI KARARINI ÜRET"):
            with st.spinner("AI Modelleri görselin DNA'sını inceliyor..."):
                olasılık = ai_kontrol_api("temp.jpg")
                if olasılık is not None:
                    if olasılık > 0.5:
                        st.markdown(f'<div class="result-box" style="background-color: rgba(211, 47, 47, 0.1); color: #ff7b72; border-color: #f85149;">SONUÇ: %{olasılık*100:.1f} Olasılıkla Yapay Zekâ Ürünü 🚨</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box" style="background-color: rgba(46, 125, 50, 0.1); color: #7ee787; border-color: #3fb950;">SONUÇ: %{(1-olasılık)*100:.1f} Olasılıkla Gerçek Çekim ✅</div>', unsafe_allow_html=True)
                else:
                    st.warning("Analiz başarısız. API/İnternet bağlantısını kontrol edin.")
    else: st.info("Lütfen Görsel Analiz sekmesinden veri yükleyin.")

# --- 8. FOOTER ---
meb_b64 = get_base64_image("meb.png")
tubitak_b64 = get_base64_image("tubitak.png")
st.markdown(f'''
    <div style="background-color: rgba(255,255,255,0.02); padding: 30px; border-radius: 15px; border: 1px solid #30363d; display: flex; flex-direction: column; align-items: center; gap: 15px; margin-top: 40px;">
        <div style="display: flex; gap: 50px;">
            <img src="data:image/png;base64,{meb_b64}" height="55" style="filter: brightness(0) invert(1);">
            <img src="data:image/png;base64,{tubitak_b64}" height="55" style="filter: brightness(0) invert(1);">
        </div>
        <div style="color: #8b949e; font-size: 0.85em; text-align:center;">
            <b>REYHANLI YAHYA TURAN FEN LİSESİ</b><br>TÜBİTAK 4006 BİLİM FUARI - 2026
        </div>
    </div>
''', unsafe_allow_html=True)
