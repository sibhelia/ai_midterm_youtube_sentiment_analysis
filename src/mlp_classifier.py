# ==============================================================================
# src/mlp_classifier.py
# STRATEJİK FİNAL: Overfitting Önleyici ve Farklı Algoritmalar
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

# ------------------------------------------------------------------------------
# A. CONFIGURATION
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

X_PATH = os.path.join(PROJECT_ROOT, 'data', 'X_train_features.npy')
Y_PATH = os.path.join(PROJECT_ROOT, 'data', 'y_train_labels.csv')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# B. MODEL CONFIGURATIONS
# ------------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Model_1_Genis_ve_Kontrollu": {
        # STRATEJİ 1: Derinleştirmek yerine genişletiyoruz (Tek katman 500 nöron).
        # Alpha (Ceza) artırıldı: 0.0001 -> 0.05. Bu, ezberlemeyi (overfitting) engeller.
        'hidden_layer_sizes': (500,), 
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.05,            # <-- KRİTİK: Ezber bozucu parametre
        'batch_size': 256,        # Daha genel bakış için büyük batch
        'learning_rate': 'adaptive',
        'max_iter': 1000,
        'early_stopping': True,   # Model kötüleşirse durdur
        'validation_fraction': 0.1,
        'random_state': 42
    },
    "Model_2_SGD_Optimize": {
        # STRATEJİ 2: 'adam' yerine 'sgd' kullanıyoruz.
        # SGD genelde daha zor eğitilir ama "local minima"dan kaçıp daha iyi genelleme yapabilir.
        'hidden_layer_sizes': (300, 150), 
        'activation': 'tanh',
        'solver': 'sgd',          # <-- KRİTİK: Farklı matematiksel yaklaşım
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.01, # Biraz daha agresif başlangıç
        'momentum': 0.9,
        'alpha': 0.01,            # Hafif ceza
        'max_iter': 2000,         # SGD yavaştır, süre tanıyalım
        'early_stopping': False, 
        'random_state': 123
    }
}

# ------------------------------------------------------------------------------
# C. EVALUATION FUNCTION
# ------------------------------------------------------------------------------
def evaluate_model(y_true, y_pred, model_name, class_labels):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n--- {model_name} Performans Sonuçları ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    
    save_path = os.path.join(REPORTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    
    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

# ------------------------------------------------------------------------------
# D. MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ MLP Eğitimi (Stratejik Mod) Başlatılıyor...")
    
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("❌ Hata: Veri dosyaları bulunamadı!")
        exit()
        
    X = np.load(X_PATH)
    y_df = pd.read_csv(Y_PATH)
    y = y_df.iloc[:, 0].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # Veriyi Böl
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # --- SCALING ---
    print("⚖️  Veri Normalize Ediliyor...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    results = []
    
    for name, config in MODEL_CONFIGS.items():
        print(f"\n⚙️  Model Eğitiliyor: {name} ...")
        
        mlp = MLPClassifier(**config)
        mlp.fit(X_train, y_train)
        
        y_pred = mlp.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, name, class_names)
        results.append(metrics)
        
        joblib.dump(mlp, os.path.join(MODELS_DIR, f'{name}.joblib'))
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*50)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(REPORTS_DIR, 'model_comparison_results.csv'), index=False)
    print(f"\n✅ Sonuçlar kaydedildi.")