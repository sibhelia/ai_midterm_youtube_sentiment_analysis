# YouTube Sentiment Analysis

![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Durum-Tamamlandı-success)
![Focus](https://img.shields.io/badge/Odak-NLP%20%26%20Machine%20Learning-orange)

---

##  Proje Hakkında

Bu çalışma, **BMM4101 Yapay Zeka Teknikleri** dersi kapsamında geliştirilmiştir. Projenin temel amacı, YouTube üzerindeki teknoloji/yazılım videolarına yapılan Türkçe yorumları analiz ederek, izleyici tepkilerini **Yapay Sinir Ağları (YSA/MLP)** ve **Word2Vec** teknolojileriyle otomatik olarak sınıflandırmaktır.

**Temel Görev:** Yorumları **"Olumlu"**, **"Olumsuz"** veya **"Nötr"** olarak 3 sınıfa ayırarak etiketlemek.

---

##  Mimari ve Kullanılan Teknolojiler

Proje, modern Doğal Dil İşleme (NLP) boru hattı (pipeline) üzerine kurulmuştur:

| Teknoloji | Kullanım Amacı | Detay |
|:---|:---|:---|
| **Gensim Word2Vec** | Özellik Çıkarımı | Kelimeleri 300 boyutlu vektörlere dönüştürme (Skip-Gram). Tüm tokenlerden model oluşturulmuştur. |
| **Scikit-Learn MLP** | Sınıflandırma | "Geniş ve Kontrollü" Yapay Sinir Ağı ile sınıflandırma. |
| **NLTK** | Ön İşleme | Metin temizliği, tokenization işlemleri. |
| **YouTube Data API** | Veri Toplama | Video ve yorum verilerinin (metadata) otomatik çekilmesi. |
| **Tkinter** | Arayüz | Sonuçların görselleştirilmesi için masaüstü GUI. |

---

## Dosya Yapısı

Proje dizinleri, sürdürülebilirlik ve düzen için modüler ayrılmıştır:

```text
ai_midterm_youtube_sentiment_analysis/
├── 📂 data/               
│   ├── training_data_set.csv   
│   ├── user_comments_metadata.csv
│   └── user_comments_predicted.csv 
│
├── 📂 models/               
│   ├── word2vec_model.bin       
│   ├── Model_1_Genis_ve_Kontrollu.joblib 
│   └── scaler.joblib             
│
├── 📂 reports/             
│   ├── model_comparison_results.csv 
│   └── *_confusion_matrix.png       
│
├── 📂 src/                
│   ├── data_acquisition.py       
│   ├── word2vec_preparation.py  
│   ├── mlp_classifier.py        
│   ├── predict_user_comments.py
│   └── gui_visualization.py      
│
└── 📄 README.md            
```

---

## Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla izleyin.

### 1. Gerekli Kütüphaneler

Terminal veya komut satırında şu komutu çalıştırarak bağımlılıkları yükleyin:

```bash
pip install pandas numpy scikit-learn gensim nltk matplotlib seaborn google-api-python-client
```

### 2. Adım Adım Çalıştırma Rehberi

#### Adım 1: Veri Çekme
YouTube API kullanarak yorumları ve meta verileri indirin.

```bash
python src/data_acquisition.py
```

#### Adım 2: Model Eğitimi (Word2Vec)
Metinleri ön işler (NLTK) ve sayısal vektörlere dönüştürür (Word2Vec).

```bash
python src/word2vec_preparation.py
```

#### Adım 3: Sınıflandırma Eğitimi (MLP)
Yapay sinir ağını eğitir, 2 farklı modeli karşılaştırır ve performans metriklerini (Accuracy, F1 vb.) hesaplar.

```bash
python src/mlp_classifier.py
```

#### Adım 4: Tahmin
Kendi çektiğimiz 40-50+ yorumu eğitilen model ile sınıflandırır.

```bash
python src/predict_user_comments.py
```

#### Adım 5: Sonuçları Gör (Arayüz)
Analiz sonuçlarını görsel arayüzde inceleyin.

```bash
python src/gui_visualization.py
```

---



Bu proje akademik amaçla hazırlanmıştır ve BMM4101 dersi vize ödevi gereksinimlerini karşılamaktadır.
