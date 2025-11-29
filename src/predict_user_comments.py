"""
BMM4101 Yapay Zeka Teknikleri Dersi - Vize Projesi
Öğrenci: Sibel Akkurt
Dosya: predict_user_comments.py
Açıklama: Eğitilen model ile yeni yorumların duygu durumunu tahmin eden modül.
"""

import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

USER_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_metadata.csv')
WORD2VEC_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'word2vec_model.bin')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib')
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'Model_1_Genis_ve_Kontrollu.joblib')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_predicted.csv')

VECTOR_SIZE = 300 
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[^\w\s]', ' ', text) 
    text = re.sub(r'\d+', '', text)   
    try:
        tokens = word_tokenize(text, language='turkish')
    except LookupError:
        tokens = word_tokenize(text, language='turkish')
    # Stopwords korunur
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

if __name__ == "__main__":
    print("Tahmin İşlemi Başlatılıyor...")
    
    if not os.path.exists(USER_DATA_PATH) or not os.path.exists(MLP_MODEL_PATH):
        print("Gerekli dosyalar bulunamadı.")
        exit()

    print("Modeller yükleniyor...")
    w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mlp_model = joblib.load(MLP_MODEL_PATH)
    
    df = pd.read_csv(USER_DATA_PATH)
    print(f"{len(df)} yorum işleniyor.")
 
    col_name = None
    for name in ['Yorum', 'Yorum_Metni', 'Comment_Text', 'textDisplay']:
        if name in df.columns:
            col_name = name
            break
    
    if not col_name:
        print("Yorum sütunu bulunamadı.")
        exit()

    df['tokens'] = df[col_name].apply(clean_and_tokenize)
    features = [get_sentence_vector(tokens, w2v_model) for tokens in df['tokens']]
    X_user = np.array(features)

    X_user = scaler.transform(X_user)
    predictions = mlp_model.predict(X_user)
    
    label_map = {0: 'Nötr', 1: 'Olumlu', 2: 'Olumsuz'}
    df['Tahmin_Edilen_Duygu'] = [label_map.get(p, "Bilinmiyor") for p in predictions]

    if 'tokens' in df.columns:
        del df['tokens']
        
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Sonuçlar kaydedildi: {OUTPUT_PATH}")