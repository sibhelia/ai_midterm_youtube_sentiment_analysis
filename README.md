# 🧠 YouTube Türkçe Duygu Analizi (Sentiment Analysis)
### 🎓 Yapay Zeka Teknikleri | Vize Projesi

![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Durum-Tamamlandı-success)
![Focus](https://img.shields.io/badge/Odak-NLP%20%26%20Machine%20Learning-orange)

---

## 📋 Proje Hakkında

Bu çalışma, **BMM4101 Yapay Zeka Teknikleri** dersi kapsamında geliştirilmiştir. Projenin temel amacı, YouTube üzerindeki teknoloji/yazılım videolarına yapılan Türkçe yorumları analiz ederek, izleyici tepkilerini **Yapay Sinir Ağları (YSA/MLP)** ve **Word2Vec** teknolojileriyle otomatik olarak sınıflandırmaktır.

**🔍 Temel Görev:** Yorumları **"Olumlu"**, **"Olumsuz"** veya **"Nötr"** olarak 3 sınıfa ayırarak etiketlemek.

---

## 🏗️ Mimari ve Kullanılan Teknolojiler

Proje, modern Doğal Dil İşleme (NLP) boru hattı (pipeline) üzerine kurulmuştur:

| Teknoloji | Kullanım Amacı | Detay |
|:---|:---|:---|
| **Gensim Word2Vec** | Özellik Çıkarımı | Kelimeleri 300 boyutlu vektörlere dönüştürme (Skip-Gram). Tüm tokenlerden model oluşturulmuştur. |
| **Scikit-Learn MLP** | Sınıflandırma | "Geniş ve Kontrollü" Yapay Sinir Ağı ile sınıflandırma. |
| **NLTK** | Ön İşleme | Metin temizliği, tokenization işlemleri. |
| **YouTube Data API** | Veri Toplama | Video ve yorum verilerinin (metadata) otomatik çekilmesi. |
| **Tkinter** | Arayüz | Sonuçların görselleştirilmesi için masaüstü GUI. |

---

## 📂 Dosya Yapısı

Proje dizinleri, sürdürülebilirlik ve düzen için modüler ayrılmıştır:

```text
ai_midterm_youtube_sentiment_analysis/
├── 📂 data/                 # Veri Merkezi
│   ├── training_data_set.csv     # Eğitim Veri Seti (Etiketli)
│   ├── user_comments_metadata.csv # YouTube'dan çekilen ham yorumlar
│   └── user_comments_predicted.csv # Tahmin Sonuçları (Çıktı)
│
├── 📂 models/               # Yapay Zeka Beyni
│   ├── word2vec_model.bin        # Eğitilmiş Kelime Vektörleri
│   ├── Model_1_Genis_ve_Kontrollu.joblib # Final MLP Modeli
│   └── scaler.joblib             # Normalizasyon Aracı
│
├── 📂 reports/              # Raporlama
│   ├── model_comparison_results.csv # Model karşılaştırma tablosu
│   └── *_confusion_matrix.png       # Hata matrisi görselleri
│
├── 📂 src/                  # Kaynak Kodlar
│   ├── data_acquisition.py       # 📥 Veri Çekme (YouTube API)
│   ├── word2vec_preparation.py   # ⚙️ Ön İşleme ve Vektörleştirme
│   ├── mlp_classifier.py         # 🧠 Model Eğitimi ve Karşılaştırma
│   ├── predict_user_comments.py  # 🔮 Tahminleme (Kendi verimiz)
│   └── gui_visualization.py      # 🖥️ Arayüz
│
└── 📄 README.md             # Proje Dokümantasyonu
```

---

## 🚀 Kurulum ve Çalıştırma

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

## 📊 Performans Sonuçları

Geliştirilen modellerin karşılaştırmalı başarı oranları aşağıdadır:

| Model Adı | Mimari | Accuracy (Doğruluk) | F1-Score |
|:---|:---|:---:|:---:|
| **Model 1 (Final)** | Geniş Katman (500 Nöron) + Regularization | **%XX.XX** 🏆 | **AA.AA** |
| **Model 2 (Alternatif)** | SGD Optimizasyonu + Tanh Aktivasyonu | %YY.YY | BB.BB |

**Analiz:** Yapılan deneylerde, Türkçe gibi eklemeli dillerde ve kısa sosyal medya yorumlarında; çok derin ağlar yerine geniş ve iyi regüle edilmiş (alpha=0.05) ağların daha iyi genelleme yaptığı ve ezberlemeyi (overfitting) engellediği görülmüştür. Ayrıca Stopwords temizliği yapılmaması başarıyı artırmıştır.

---

## 👤 Hazırlayan

**Ad Soyad:** Sibel Akkurt
**Numara:** 202213709048 
**Bölüm:** Bilgisayar Mühendisliği

Bu proje akademik amaçla hazırlanmıştır ve BMM4101 dersi vize ödevi gereksinimlerini karşılamaktadır.
