# ==============================================================================
# BMM4101 Yapay Zeka Teknikleri - Veri Toplama Modülü
# Öğrenci: Sibel Akkurt | No: 202213709048
# Açıklama: YouTube Data API v3 kullanılarak video yorumlarının çekilmesi.
# ==============================================================================

import pandas as pd
import os
from googleapiclient.discovery import build

API_KEY = "AIzaSyCCWYW4dsD0oEjARL4xa5cQbu_XWKUudEI"

VIDEO_LINK = "https://www.youtube.com/watch?v=7foCbOktTZM"

MAX_COMMENTS = 50

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_metadata.csv')

def extract_video_id(url):
    if len(url) == 11 and "http" not in url:
        return url
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    else:
        return None

def get_video_details(youtube, video_id):
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

def get_comments(youtube, video_id, max_results=50):
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
                if len(comments_data) >= max_results:
                    break
                    
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

            if 'nextPageToken' in response and len(comments_data) < max_results:
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
        print(f"Hata: {e}")

    return comments_data

if __name__ == "__main__":
    print("--- Veri Toplama İşlemi Başlatılıyor ---")
    
    video_id = extract_video_id(VIDEO_LINK)
    if not video_id:
        print("HATA: Geçersiz Video Linki.")
        exit()

    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
       
        video_info = get_video_details(youtube, video_id)
        if not video_info:
            print("Video bulunamadı.")
            exit()
            
        print(f"Video: {video_info['Video_Basligi']}")
        
        comments = get_comments(youtube, video_id, max_results=MAX_COMMENTS)
        
        full_data = []
        for c in comments:
            merged = {**video_info, **c} 
            full_data.append(merged)
            
        df = pd.DataFrame(full_data)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        
        print(f"İşlem Tamamlandı. Toplam {len(df)} yorum kaydedildi.")
        print(f"Dosya Yolu: {OUTPUT_PATH}")

    except Exception as e:
        print(f"Beklenmeyen Hata: {e}")