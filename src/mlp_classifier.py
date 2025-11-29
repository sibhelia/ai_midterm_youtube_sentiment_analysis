# ==============================================================================
# BMM4101 Yapay Zeka Teknikleri Dersi - Vize Projesi
# Öğrenci: Sibel Akkurt | No: 202213709048
# Dosya: mlp_classifier.py
# Açıklama: Word2Vec ile elde edilen özellik vektörleri kullanılarak Çok Katmanlı
#           Algılayıcı (MLP) modellerinin eğitilmesi, hiperparametre optimizasyonu
#           ile karşılaştırılması ve en başarılı modelin kaydedilmesi işlemlerini içerir.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

X_PATH = os.path.join(PROJECT_ROOT, 'data', 'X_train_features.npy')
Y_PATH = os.path.join(PROJECT_ROOT, 'data', 'y_train_labels.csv')

REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# MODEL 1 EĞİTİMİ
MODEL_CONFIGS = {
    "Model_1_Genis_ve_Kontrollu": {
        # MİMARİ 1: Geniş Tek Katmanlı Yapı (Wide Network)
        # Amaç: Yüksek boyutlu (300) girdi vektörlerini geniş bir nöron katmanıyla 
        # işleyerek, aşırı öğrenmeyi (overfitting) yüksek regülasyon ile engellemek.
        'hidden_layer_sizes': (500,), 
        'activation': 'relu',     
        'solver': 'adam',         
        'alpha': 0.05,            
        'batch_size': 256,        
        'learning_rate': 'adaptive',
        'max_iter': 1000,       
        'early_stopping': True, 
        'validation_fraction': 0.1,
        'random_state': 42     
    },
    # MODEL 2 EĞİTİMİ
    "Model_2_SGD_Optimize": {
        # MİMARİ 2: Derin ve Daralan Yapı (Deep Narrow Network)
        # Amaç: Stokastik Gradyan İnişi (SGD) optimizasyonu ile yerel minimumlardan
        # kaçınarak daha iyi genelleme performansı elde etmek.
        'hidden_layer_sizes': (300, 150), 
        'activation': 'tanh',     
        'solver': 'sgd',          
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.01, 
        'momentum': 0.9,         
        'alpha': 0.01,           
        'max_iter': 2000,        
        'early_stopping': False, 
        'random_state': 123
    }
}

def evaluate_model(y_true, y_pred, model_name, class_labels):
   
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n--- {model_name} Performans Analizi ---")
    print(f"Doğruluk (Accuracy)  : {acc:.4f}")
    print(f"Kesinlik (Precision) : {prec:.4f}")
    print(f"Duyarlılık (Recall)  : {rec:.4f}")
    print(f"F1 Skoru (F1-Score)  : {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Hata Matrisi: {model_name}')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close() 
    
    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

if __name__ == "__main__":
    print("--- MLP Model Eğitimi ve Karşılaştırma Süreci Başlatıldı ---")
    
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("HATA: Eğitim verileri bulunamadı. Lütfen önce 'word2vec_preparation.py' çalıştırınız.")
        exit()
        
    X = np.load(X_PATH)
    y_df = pd.read_csv(Y_PATH)
    y = y_df.iloc[:, 0].values 
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # Veri Setinin Bölünmesi (Hold-out Yöntemi)
    # %80 Eğitim, %20 Test olarak ayrılmıştır. Stratify ile sınıf dengesi korunmuştur.
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Veri Normalizasyonu (Standardizasyon)
    # Yapay sinir ağlarının yakınsama hızını ve performansını artırmak için uygulanır.
    print("Veri Normalizasyonu (StandardScaler) uygulanıyor...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Scaler modelinin kaydedilmesi 
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    results = []
    
    # Modellerin sırayla eğitilmesi ve test edilmesi
    for name, config in MODEL_CONFIGS.items():
        print(f"\n  Model Eğitimi Başlatılıyor: {name} ...")
        
        mlp = MLPClassifier(**config)
        mlp.fit(X_train, y_train)
        
        y_pred = mlp.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred, name, class_names)
        results.append(metrics)
        
        joblib.dump(mlp, os.path.join(MODELS_DIR, f'{name}.joblib'))
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print(" MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*60)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(REPORTS_DIR, 'model_comparison_results.csv'), index=False)
    print(f"\nTüm süreç tamamlandı. Sonuçlar ve modeller '{REPORTS_DIR}' dizinine kaydedildi.")