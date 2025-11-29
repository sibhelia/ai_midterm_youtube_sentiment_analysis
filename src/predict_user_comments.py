# ==============================================================================
# src/predict_user_comments.py
# FİNAL: Kendi YouTube yorumlarımızı tahmin etme modülü.
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ------------------------------------------------------------------------------
# A. CONFIGURATION (YAPILANDIRMA)
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Gerekli Dosyalar
USER_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_metadata.csv')
WORD2VEC_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'word2vec_model.bin')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib')

# EN İYİ MODELİMİZİN ADI (Klasördekiyle birebir aynı olmalı)
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'Model_1_Genis_ve_Kontrollu.joblib')

# Çıktı Dosyası
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_predicted.csv')

# Word2Vec Ayarı (Eğitimdekiyle AYNI olmalı)
VECTOR_SIZE = 300 

# ------------------------------------------------------------------------------
# B. PREPROCESSING (Eğitimdeki "Duygu Analizi Özel" Versiyonuyla AYNI)
# ------------------------------------------------------------------------------
# NLTK Kontrolü
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def clean_and_tokenize(text):
    """Metni temizler ve token'lara ayırır. STOPWORDS SİLİNMEZ!"""
    if not isinstance(text, str):
        return []
    
    # 1. Temizlik
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[^\w\s]', ' ', text) # Noktalamaları boşluğa çevir
    text = re.sub(r'\d+', '', text)   
    
    # 2. Tokenize
    try:
        tokens = word_tokenize(text, language='turkish')
    except LookupError:
        tokens = word_tokenize(text, language='turkish')
        
    # 3. Filtreleme (Sadece tek harflileri atıyoruz, "ama", "değil" kalıyor!)
    tokens = [word for word in tokens if len(word) > 1]
    
    return tokens

def get_sentence_vector(text_tokens, model):
    vec = np.zeros(VECTOR_SIZE)
    count = 0
    for word in text_tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec

# ------------------------------------------------------------------------------
# C. MAIN EXECUTION (ANA ÇALIŞTIRMA)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ Tahmin İşlemi Başlatılıyor...")
    
    # 1. Dosya Kontrolleri
    if not os.path.exists(USER_DATA_PATH):
        print(f"❌ Hata: Yorum dosyası yok: {USER_DATA_PATH}")
        print("Lütfen önce data_acquisition.py dosyasını çalıştırıp veri çekin.")
        exit()
        
    if not os.path.exists(MLP_MODEL_PATH):
        print(f"❌ Hata: Model dosyası yok: {MLP_MODEL_PATH}")
        print("Lütfen önce mlp_classifier.py dosyasını çalıştırın.")
        exit()

    # 2. Modelleri Yükle
    print("📥 Modeller yükleniyor (Word2Vec, Scaler, MLP)...")
    try:
        w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        mlp_model = joblib.load(MLP_MODEL_PATH)
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        exit()
    
    # 3. Veriyi Yükle
    df = pd.read_csv(USER_DATA_PATH)
    print(f"✅ {len(df)} adet yorum yüklendi.")
    
    # Sütun adı bulma (Bizim kodumuz 'Yorum' olarak kaydediyor ama garantilemek için)
    col_name = None
    possible_names = ['Yorum', 'Yorum_Metni', 'Comment_Text']
    for name in possible_names:
        if name in df.columns:
            col_name = name
            break
            
    if col_name is None:
        print(f"❌ Hata: Yorum sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")
        exit()
        
    # 4. Özellik Çıkarımı (Vectorization)
    print("⚙️  Yorumlar vektöre dönüştürülüyor...")
    df['tokens'] = df[col_name].apply(clean_and_tokenize)
    
    features = []
    for tokens in df['tokens']:
        vec = get_sentence_vector(tokens, w2v_model)
        features.append(vec)
    X_user = np.array(features)
    
    # 5. Scaling (Normalizasyon - ÇOK ÖNEMLİ)
    # Eğitimde kullandığımız scaler ile aynı dönüşümü yapıyoruz
    X_user = scaler.transform(X_user)
    
    # 6. Tahmin Yapma
    print("🔮 Model tahmin yapıyor...")
    predictions_encoded = mlp_model.predict(X_user)
    
    # 7. Sonuçları Etiketleme
    # Alfabetik Sıra: 0=Nötr, 1=Olumlu, 2=Olumsuz
    label_map = {0: 'Nötr', 1: 'Olumlu', 2: 'Olumsuz'}
    
    df['Tahmin_Edilen_Duygu'] = [label_map.get(p, "Bilinmiyor") for p in predictions_encoded]
    
    # Sadece gerekli sütunları kaydet
    df_result = df[[col_name, 'Tahmin_Edilen_Duygu']]
    df_result.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*60)
    print("📋 ÖRNEK TAHMİNLER (İlk 15 Yorum)")
    print("="*60)
    pd.set_option('display.max_colwidth', 80) # Yorumların tamamını görelim
    print(df_result.head(15).to_string(index=False))
    print("\n" + "="*60)
    print(f"✅ BİTTİ! Tüm tahminler şuraya kaydedildi:\n   -> {OUTPUT_PATH}")