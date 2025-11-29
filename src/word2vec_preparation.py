# ==============================================================================
# src/word2vec_preparation.py
# DUYGU ANALİZİ ÖZEL VERSİYON: Stopwords SİLİNMİYOR + Skip-Gram Kullanılıyor
# ==============================================================================

import pandas as pd
import nltk
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import logging
import os

# Günlükleme ayarları
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 0. NLTK EKSİKLERİNİ GİDERME
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ------------------------------------------------------------------------------
# A. CONFIGURATION
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              

TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'training_data_set.csv')
USER_COMMENTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_metadata.csv')

COMMENT_COLUMN_NAME = 'Yorum'
LABEL_COLUMN_NAME = 'Durum'

# --- KRİTİK DEĞİŞİKLİKLER ---
VECTOR_SIZE = 300    # Detaylı vektörler
WINDOW = 10          # Pencereyi genişlettik (Cümlenin genelini görsün)
MIN_COUNT = 2        # Nadir kelimeleri de öğrensin (Örn: "berbat" az geçse bile önemli)
SG = 1               # SKIP-GRAM: Küçük veri setlerinde çok daha başarılıdır!
EPOCHS = 50          # Daha uzun eğitim
WORKERS = 4          

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'word2vec_model.bin')
X_TRAIN_SAVE_PATH = os.path.join(PROJECT_ROOT, 'data', 'X_train_features.npy') 
Y_TRAIN_SAVE_PATH = os.path.join(PROJECT_ROOT, 'data', 'y_train_labels.csv')

# ------------------------------------------------------------------------------
# B. PREPROCESSING (STOPWORDS SİLMEK YOK!)
# ------------------------------------------------------------------------------
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []
    
    # 1. Küçült
    text = text.lower()
    
    # 2. Linkleri temizle
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    
    # 3. Sadece harfleri tut (Noktalama işaretlerini kaldır ama kelimeleri bitişik yazma)
    # "güzel." -> "güzel" olsun diye boşluklu değiştiriyoruz
    text = re.sub(r'[^\w\s]', ' ', text) 
    
    # 4. Sayıları kaldır
    text = re.sub(r'\d+', '', text)   
    
    # 5. Tokenize et
    try:
        tokens = word_tokenize(text, language='turkish')
    except LookupError:
        nltk.download('punkt_tab')
        tokens = word_tokenize(text, language='turkish')
        
    # --- KRİTİK NOKTA: STOPWORDS SİLMİYORUZ! ---
    # "değil", "hiç", "ama" gibi kelimeler duygu analizi için ÇOK ÖNEMLİDİR.
    # Sadece 1 harfli anlamsız karakterleri atıyoruz.
    tokens = [word for word in tokens if len(word) > 1]
    
    return tokens

# ------------------------------------------------------------------------------
# C. TRAINING & VECTORIZATION
# ------------------------------------------------------------------------------
def train_word2vec_model(sentences, path):
    print("\n⏳ Word2Vec modeli eğitiliyor (Skip-Gram Modu)...")
    model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, sg=SG, workers=WORKERS)
    model.train(sentences, total_examples=len(sentences), epochs=EPOCHS)
    model.save(path)
    print(f"✅ Word2Vec Modeli Kaydedildi: {path}")
    return model

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

def create_feature_vectors(df, comment_col, model):
    df['tokens'] = df[comment_col].apply(clean_and_tokenize)
    feature_vectors = [get_sentence_vector(tokens, model) for tokens in df['tokens']]
    return np.array(feature_vectors)

# ------------------------------------------------------------------------------
# D. MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📂 Çalışma Dizini: {PROJECT_ROOT}")
    print("⏳ Veri Yükleniyor...")
    
    try:
        if not os.path.exists(TRAINING_DATA_PATH):
            print(f"❌ HATA: Dosya bulunamadı -> {TRAINING_DATA_PATH}")
            exit()

        try:
             df_train = pd.read_csv(TRAINING_DATA_PATH, encoding='utf-8')
        except UnicodeDecodeError:
             df_train = pd.read_csv(TRAINING_DATA_PATH, encoding='utf-16')

        print("🛠️  Veri Standardizasyonu...")
        if 'Görüş' in df_train.columns:
            df_train.rename(columns={'Görüş': 'Yorum'}, inplace=True)
        if 'Durum' in df_train.columns and 'Tarafsız' in df_train['Durum'].values:
            df_train['Durum'] = df_train['Durum'].replace('Tarafsız', 'Nötr')

        print(f"✅ Eğitim Seti: {len(df_train)} satır.")

        user_texts = []
        if os.path.exists(USER_COMMENTS_PATH):
            try:
                df_user = pd.read_csv(USER_COMMENTS_PATH)
                if not df_user.empty and COMMENT_COLUMN_NAME in df_user.columns:
                    user_texts = df_user[COMMENT_COLUMN_NAME].dropna().tolist()
            except: pass

        all_texts = df_train[COMMENT_COLUMN_NAME].dropna().tolist() + user_texts
        all_sentences = [clean_and_tokenize(str(text)) for text in all_texts]

        if not all_sentences:
            exit()

    except Exception as e:
        print(f"\n❌ Hata: {e}")
        exit()

    w2v_model = train_word2vec_model(all_sentences, MODEL_PATH)
    X_train = create_feature_vectors(df_train, COMMENT_COLUMN_NAME, w2v_model)
    y_train_df = df_train[[LABEL_COLUMN_NAME]]
    
    np.save(X_TRAIN_SAVE_PATH, X_train)
    y_train_df.to_csv(Y_TRAIN_SAVE_PATH, index=False)
    
    print(f"\n✅ ÖZELLİKLER GÜNCELLENDİ (Boyut: {X_train.shape}).")
    print("👉 Şimdi 'python src/mlp_classifier.py' komutunu çalıştırın!")