📊 YouTube Türkçe Duygu Analizi (Sentiment Analysis)BMM4101 Yapay Zeka Teknikleri Dersi - Vize ÖdeviHazırlayan: Sibel Akkurt - 202213709048 Bu proje, YouTube videolarına yapılan Türkçe yorumların Word2Vec ve Çok Katmanlı Algılayıcı (MLP - Multi-Layer Perceptron) kullanılarak Olumlu, Olumsuz veya Nötr olarak sınıflandırılmasını sağlar.📁 Proje YapısıProje, modüler bir yapıda tasarlanmış olup aşağıdaki klasör hiyerarşisine sahiptir:ai_midterm_youtube_sentiment_analysis/
│
├── data/                   # Veri setleri ve özellik matrisleri
│   ├── training_data_set.csv     # Eğitim verisi (Etiketli)
│   ├── user_comments_metadata.csv # YouTube'dan çekilen yorumlar
│   ├── X_train_features.npy       # Word2Vec ile sayısallaştırılmış veriler
│   └── y_train_labels.csv         # Eğitim etiketleri
│
├── models/                 # Eğitilmiş modeller ve araçlar
│   ├── word2vec_model.bin         # Eğitilmiş Word2Vec modeli
│   ├── scaler.joblib              # Veri normalizasyon ölçekleyici
│   └── Model_1_Genis_ve_Kontrollu.joblib # En iyi performans gösteren MLP modeli
│
├── reports/                # Performans raporları ve grafikler
│   ├── model_comparison_results.csv # Model karşılaştırma tablosu
│   └── *_confusion_matrix.png       # Hata matrisi görselleri
│
├── src/                    # Kaynak kodlar
│   ├── data_acquisition.py        # YouTube API ile veri çekme
│   ├── word2vec_preparation.py    # Ön işleme ve özellik çıkarımı
│   ├── mlp_classifier.py          # Model eğitimi ve karşılaştırma
│   ├── predict_user_comments.py   # Yeni yorumların tahmini
│   └── gui_visualization.py       # Arayüz (GUI)
│
└── README.md               # Proje dokümantasyonu
🚀 Kurulum ve HazırlıkProjeyi çalıştırmadan önce gerekli Python kütüphanelerini yüklemeniz gerekmektedir:Bashpip install pandas numpy scikit-learn gensim nltk matplotlib seaborn google-api-python-client
⚙️ Kullanılan Yöntemler ve AlgoritmalarBu projede metin sınıflandırma için hibrit bir yaklaşım benimsenmiştir:1. Veri Ön İşleme (Preprocessing)Temizlik: URL, sayı ve noktalama işaretleri temizlendi.Tokenization: NLTK kütüphanesi kullanılarak metinler parçalandı.Stopwords Stratejisi: Duygu analizinde anlam kaymasını önlemek için ("değil", "hiç" vb.) etkisiz kelimeler silinmemiştir.2. Özellik Çıkarımı (Feature Extraction) - Word2VecMetinleri sayısal vektörlere dönüştürmek için Gensim Word2Vec kullanılmıştır.Algoritma: Skip-Gram (sg=1) - Küçük veri setlerinde nadir kelimeleri daha iyi yakalar.Vektör Boyutu: 300Pencere (Window): 10Cümle Temsili: Her yorumdaki kelime vektörlerinin ortalaması alınarak cümle vektörü oluşturulmuştur.3. Sınıflandırma (Classification) - MLPScikit-Learn kütüphanesi ile Yapay Sinir Ağları (ANN/MLP) eğitilmiştir. İki farklı mimari karşılaştırılmıştır:Model 1 (Geniş ve Kontrollü): Tek gizli katmanda 500 nöron, yüksek regülasyon (Alpha=0.05).Model 2 (SGD Optimize): Stokastik Gradyan İnişi ile optimize edilmiş derin yapı.En Başarılı Model: Model_1_Genis_ve_Kontrollu (%70.82 Başarı Oranı)🖥️ Nasıl Çalıştırılır?Projeyi sıfırdan çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız:Adım 1: Veri Çekme (API Key gerektirir)Bashpython src/data_acquisition.py
YouTube videosundaki yorumları çeker ve kaydeder.Adım 2: Word2Vec Eğitimi ve VektörleştirmeBashpython src/word2vec_preparation.py
Metinleri ön işler, Word2Vec modelini eğitir ve özellik matrislerini (.npy) oluşturur.Adım 3: MLP Model Eğitimi ve KarşılaştırmaBashpython src/mlp_classifier.py
Modelleri eğitir, karşılaştırır ve sonuçları reports/ klasörüne kaydeder.Adım 4: Tahmin YapmaBashpython src/predict_user_comments.py
Çekilen YouTube yorumlarını eğitilen model ile analiz eder ve etiketler.Adım 5: Arayüzü BaşlatmaBashpython src/gui_visualization.py
Sonuçları görüntülemek ve filtrelemek için kullanıcı arayüzünü açar.📊 SonuçlarYapılan testler sonucunda elde edilen performans metrikleri:ModelAccuracyPrecisionRecallF1-ScoreModel 1 (Geniş)0.70820.70020.70820.7032Model 2 (SGD)0.67540.66760.67540.6707Bu doküman, BMM4101 Yapay Zeka Teknikleri dersi vize ödevi kapsamında hazırlanmıştır.