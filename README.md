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
| :--- | :--- | :--- |
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


