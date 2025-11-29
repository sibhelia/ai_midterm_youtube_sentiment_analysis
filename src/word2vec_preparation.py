# ==============================================================================
# BMM4101 Yapay Zeka Teknikleri Dersi - Vize Projesi
# Öğrenci: Sibel Akkurt | No: 202213709048
# Dosya: word2vec_preparation.py
# Açıklama: Ham metin verilerinin ön işlenmesi (preprocessing), temizlenmesi,
#           Gensim kütüphanesi ile Word2Vec modelinin eğitilmesi ve metinlerin
#           sayısal vektörlere dönüştürülmesi işlemlerini gerçekleştirir.
# ==============================================================================

import pandas as pd
import nltk
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              

TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'training_data_set.csv')
USER_COMMENTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_metadata.csv')

COMMENT_COLUMN_NAME = 'Yorum'
LABEL_COLUMN_NAME = 'Durum'


VECTOR_SIZE = 300   
WINDOW = 10         
MIN_COUNT = 2       
SG = 1              
EPOCHS = 50         
WORKERS = 4         

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'word2vec_model.bin')
X_TRAIN_SAVE_PATH = os.path.join(PROJECT_ROOT, 'data', 'X_train_features.npy') 
Y_TRAIN_SAVE_PATH = os.path.join(PROJECT_ROOT, 'data', 'y_train_labels.csv')


try:
    stop_words = set(stopwords.words('turkish'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('turkish'))


sentiment_critical_words = {
    'ama', 'fakat', 'lakin', 'ancak',     
    'değil', 'yok', 'hayır', 'hiç', 'ne', 
    'çok', 'daha', 'en', 'pek',           
    'asla', 'sakın'                       
}

final_stop_words = stop_words - sentiment_critical_words

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
        nltk.download('punkt_tab')
        tokens = word_tokenize(text, language='turkish')
        
    tokens = [word for word in tokens if len(word) > 1 and word not in final_stop_words]
    
    return tokens

def train_word2vec_model(sentences, path):
    print("\nWord2Vec modeli eğitiliyor (Skip-Gram Mimarisi)...")
    model = Word2Vec(
        sentences, 
        vector_size=VECTOR_SIZE, 
        window=WINDOW, 
        min_count=MIN_COUNT, 
        sg=SG, 
        workers=WORKERS
    )
    model.train(sentences, total_examples=len(sentences), epochs=EPOCHS)
    
    model.save(path)
    print(f"Word2Vec Modeli Başarıyla Kaydedildi: {path}")
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

if __name__ == "__main__":
    print(f"Çalışma Dizini: {PROJECT_ROOT}")
    print("Veri Yükleme ve Ön İşleme Başlatılıyor...")
    
    try:
        if not os.path.exists(TRAINING_DATA_PATH):
            print(f"HATA: Eğitim veri seti bulunamadı -> {TRAINING_DATA_PATH}")
            exit()

        try:
             df_train = pd.read_csv(TRAINING_DATA_PATH, encoding='utf-8')
        except UnicodeDecodeError:
             df_train = pd.read_csv(TRAINING_DATA_PATH, encoding='utf-16')

        print("Veri Standardizasyonu...")
        if 'Görüş' in df_train.columns:
            df_train.rename(columns={'Görüş': 'Yorum'}, inplace=True)
        if 'Durum' in df_train.columns and 'Tarafsız' in df_train['Durum'].values:
            df_train['Durum'] = df_train['Durum'].replace('Tarafsız', 'Nötr')

        print(f"Eğitim Seti Yüklendi: {len(df_train)} satır.")

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
            print("İşlenecek metin bulunamadı.")
            exit()

    except Exception as e:
        print(f"Hata: {e}")
        exit()

    w2v_model = train_word2vec_model(all_sentences, MODEL_PATH)
 
    X_train = create_feature_vectors(df_train, COMMENT_COLUMN_NAME, w2v_model)
    y_train_df = df_train[[LABEL_COLUMN_NAME]]
 
    np.save(X_TRAIN_SAVE_PATH, X_train)
    y_train_df.to_csv(Y_TRAIN_SAVE_PATH, index=False)
    
    print(f"\n İşlem Tamamlandı.")
    print(f"Özellik Matrisi Kaydedildi: {X_TRAIN_SAVE_PATH}")