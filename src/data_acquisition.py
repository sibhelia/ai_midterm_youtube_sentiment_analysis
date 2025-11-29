# ==============================================================================
# src/data_acquisition.py
# Bu dosya, YouTube API kullanarak verdiğiniz LINKTEKI videonun yorumlarını çeker.
# ==============================================================================

import pandas as pd
import os
import re
from googleapiclient.discovery import build

# ------------------------------------------------------------------------------
# A. AYARLAR (BURAYI DÜZENLEYİN)
# ------------------------------------------------------------------------------

# 1. YouTube API Anahtarınızı tırnak içine yapıştırın:
API_KEY = "BURAYA_API_KEYINIZI_YAZIN"

# 2. Analiz etmek istediğiniz videonun TAM LİNKİNİ buraya yapıştırın:
# (Örnek: "https://www.youtube.com/watch?v=7foCbOktTZM" veya "https://youtu.be/..." olabilir)
VIDEO_LINK = "https://www.youtube.com/watch?v=7foCbOktTZM"

# ------------------------------------------------------------------------------
# B. YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------------------------

def extract_video_id(url):
    """Verilen tam YouTube linkinden sadece Video ID'sini çekip çıkarır."""
    # Eğer kullanıcı direkt ID girdiyse (11 karakterli), olduğu gibi döndür
    if len(url) == 11 and "http" not in url:
        return url
        
    # Standart link (youtube.com/watch?v=ID)
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    
    # Kısa link (youtu.be/ID)
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
        
    else:
        return None

# Dosya Yolları
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_metadata.csv')

def get_video_details(youtube, video_id):
    """Videonun başlık, kanal, tarih ve beğeni bilgilerini çeker."""
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()

    if not response['items']:
        return None

    item = response['items'][0]
    return {
        'Video_Basligi': item['snippet']['title'],
        'Kanal_Adi': item['snippet']['channelTitle'],
        'Video_Yayin_Tarihi': item['snippet']['publishedAt'],
        'Video_Begeni': item['statistics'].get('likeCount', 0),
        'Video_Izlenme': item['statistics'].get('viewCount', 0)
    }

def get_comments(youtube, video_id, max_results=100):
    """Videodaki yorumları çeker."""
    comments_data = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100, 
        textFormat="plainText"
    )

    try:
        while request and len(comments_data) < max_results:
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                
                entry = {
                    'Yorum': comment['textDisplay'],  
                    'Yorum_Tarihi': comment['publishedAt'],
                    'Kullanici_Adi': comment['authorDisplayName'],
                    'Begeni_Sayisi': comment['likeCount'],
                    'Profil_Resmi': comment['authorProfileImageUrl'],
                    'Kanal_URL': comment['authorChannelUrl']
                }
                comments_data.append(entry)

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                )
            else:
                break
                
    except Exception as e:
        print(f"⚠️ Yorum çekerken uyarı: {e}")

    return comments_data

# ------------------------------------------------------------------------------
# C. ANA ÇALIŞTIRMA BLOĞU
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ YouTube Veri Çekici Başlatılıyor...")
    
    if "BURAYA" in API_KEY:
        print("❌ HATA: Lütfen kodun 16. satırına kendi API Key'inizi yapıştırın!")
        exit()

    # Linkten ID ayıklama
    video_id = extract_video_id(VIDEO_LINK)
    if not video_id:
        print(f"❌ HATA: Geçersiz YouTube Linki: {VIDEO_LINK}")
        exit()

    print(f"🔗 Algılanan Video ID: {video_id}")

    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        
        # 1. Video Bilgileri
        print("📥 Video bilgileri sorgulanıyor...")
        video_info = get_video_details(youtube, video_id)
        
        if not video_info:
            print("❌ Video bulunamadı! Linki kontrol edin veya video gizli olabilir.")
            exit()
            
        print(f"   Video: {video_info['Video_Basligi']}")
        print(f"   Kanal: {video_info['Kanal_Adi']}")
        
        # 2. Yorumlar
        print("📥 Yorumlar indiriliyor...")
        comments = get_comments(youtube, video_id, max_results=100)
        
        if not comments:
            print("❌ Bu videoda hiç yorum yok veya yorumlar kapalı.")
            exit()
            
        # 3. Birleştirme ve Kayıt
        full_data = []
        for c in comments:
            merged = {**video_info, **c} 
            full_data.append(merged)
            
        df = pd.DataFrame(full_data)
        
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        
        print(f"\n✅ İŞLEM BAŞARILI!")
        print(f"   Toplam {len(df)} yorum çekildi.")
        print(f"   Dosya kaydedildi: {OUTPUT_PATH}")
        print("\n👉 Sıradaki Adım: 'python src/predict_user_comments.py' komutunu çalıştırarak bu yorumların duygu analizini yapın.")

    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")