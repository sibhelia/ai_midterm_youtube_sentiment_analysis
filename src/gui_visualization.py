# ==============================================================================
# BMM4101 Yapay Zeka Teknikleri Dersi - Vize Projesi
# Öğrenci: Sibel Akkurt | No: 202213709048
# Dosya: gui_visualization.py
# Açıklama: Analiz sonuçlarının görselleştirilmesi, filtrelenmesi ve raporlanması
#           için CustomTkinter kütüphanesi ile geliştirilmiş grafiksel kullanıcı arayüzü.
# ==============================================================================

import customtkinter as ctk
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_comments_predicted.csv')

COLORS = {
    'Olumlu': '#2ecc71', 
    'Olumsuz': '#e74c3c', 
    'Nötr': '#f1c40f',    
    'Dark_Bg': '#2b2b2b',
    'Light_Bg': '#ececec'
}

class SentimentApp(ctk.CTk):
   
    def __init__(self):
        super().__init__()

        self.title("Yapay Zeka Duygu Analizi Dashboard")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        self.df = self.load_data()
        
        self.bind("<Configure>", self.on_window_resize)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.header_frame.grid_columnconfigure(0, weight=3)
        self.header_frame.grid_columnconfigure(1, weight=1)

        self.create_video_card()
        self.create_student_card()

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.content_frame.grid_columnconfigure(0, weight=2)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        if self.df is not None:
            self.create_comments_area()
            self.create_charts_area()
        else:
            ctk.CTkLabel(self.content_frame, text="Veri dosyası bulunamadı. Lütfen analiz işlemini tamamlayınız.", 
                       font=("Arial", 20), text_color="red").grid(row=0, column=0, columnspan=2)

    def load_data(self):
       
        if os.path.exists(DATA_PATH):
            try:
                df = pd.read_csv(DATA_PATH)
                if 'Yorum' in df.columns: self.col_name = 'Yorum'
                elif 'Yorum_Metni' in df.columns: self.col_name = 'Yorum_Metni'
                else: self.col_name = df.columns[0]
                return df
            except Exception: return None
        return None

    def on_window_resize(self, event):
       
        if event.widget == self:
            width = self.winfo_width()
            if width < 1000:
                self.content_frame.grid_columnconfigure(0, weight=1)
                self.content_frame.grid_columnconfigure(1, weight=0)
            else:
                self.content_frame.grid_columnconfigure(0, weight=2)
                self.content_frame.grid_columnconfigure(1, weight=1)

    def toggle_theme(self):
       
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
            self.theme_switch.configure(text="☀️ Light Mod")
        else:
            ctk.set_appearance_mode("Dark")
            self.theme_switch.configure(text="🌙 Dark Mod")
        
        if self.df is not None:
            self.update_idletasks()
            self.draw_charts()

    def create_video_card(self):
      
        card = ctk.CTkFrame(self.header_frame, corner_radius=10)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        card.bind("<Enter>", lambda e: card.configure(border_width=2, border_color="#3498db"))
        card.bind("<Leave>", lambda e: card.configure(border_width=0))
        
        try:
            row = self.df.iloc[0]
            title = row.get('Video_Basligi', 'Bilinmiyor')
            channel = row.get('Kanal_Adi', '-')
            likes = row.get('Video_Begeni', '-')
            date = str(row.get('Video_Yayin_Tarihi', '-')).split('T')[0]
            count = len(self.df)
        except:
            title, channel, likes, date, count = ("Veri Yüklenemedi", "-", "-", "-", 0)

        self.video_title = ctk.CTkLabel(card, text=f"📺 {title}", font=("Arial", 18, "bold"), wraplength=600, justify="left")
        self.video_title.pack(anchor="w", padx=20, pady=(15, 5))
        
        stats = f"👤 {channel}    📅 {date}    👍 {likes} Beğeni    💬 {count} Yorum Analiz Edildi"
        self.video_stats = ctk.CTkLabel(card, text=stats, font=("Arial", 12), text_color="gray")
        self.video_stats.pack(anchor="w", padx=20, pady=(0, 15))
        
        card.bind("<Configure>", lambda e: self.update_video_wraplength())

    def update_video_wraplength(self):
        if hasattr(self, 'video_title'):
            width = self.header_frame.winfo_width()
            self.video_title.configure(wraplength=max(300, int(width * 0.6)))

    def create_student_card(self):
        card = ctk.CTkFrame(self.header_frame, corner_radius=10)
        card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        card.bind("<Enter>", lambda e: card.configure(border_width=2, border_color="#9b59b6"))
        card.bind("<Leave>", lambda e: card.configure(border_width=0))
        
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(info_frame, text="BMM4101 Yapay Zeka Teknikleri Dersi Vize Ödevi", font=("Arial", 12, "bold")).pack(anchor="w")
        ctk.CTkLabel(info_frame, text="Hazırlayan: Sibel Akkurt", font=("Arial", 11)).pack(anchor="w")
        
        self.theme_switch = ctk.CTkSwitch(card, text="🌙 Dark Mod", command=self.toggle_theme, onvalue="on", offvalue="off")
        self.theme_switch.select()
        self.theme_switch.pack(anchor="e", padx=15, pady=(5, 10))

    def create_comments_area(self):
        left_panel = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        search_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        search_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(search_frame, text="📝 Yorum Listesi", font=("Arial", 16, "bold")).pack(side="left")
        
        search_box_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        search_box_frame.pack(side="right", fill="x", expand=True, padx=(20, 0))
        
        ctk.CTkLabel(search_box_frame, text="🔍", font=("Arial", 16)).pack(side="left", padx=(0, 5))
        self.search_entry = ctk.CTkEntry(search_box_frame, placeholder_text="Ara: kelime veya duygu")
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", self.filter_comments)

        self.scroll_frame = ctk.CTkScrollableFrame(left_panel, label_text="")
        self.scroll_frame.pack(fill="both", expand=True)
        
        try:
            self.scroll_frame._parent_canvas.configure(yscrollincrement=15)
        except Exception:
            pass
        
        self.display_comments(self.df)

    def create_charts_area(self):
        self.right_panel = ctk.CTkScrollableFrame(self.content_frame, label_text="📊 Analiz Grafikleri")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.draw_charts()

    def draw_charts(self):
        for widget in self.right_panel.winfo_children():
            widget.destroy()

        counts = self.df['Tahmin_Edilen_Duygu'].value_counts()
        for label in ['Olumlu', 'Olumsuz', 'Nötr']:
            if label not in counts: counts[label] = 0
            
        labels = ['Olumlu', 'Olumsuz', 'Nötr']
        sizes = [counts['Olumlu'], counts['Olumsuz'], counts['Nötr']]
        colors = [COLORS['Olumlu'], COLORS['Olumsuz'], COLORS['Nötr']]

        is_dark = ctk.get_appearance_mode() == "Dark"
        bg_color = COLORS['Dark_Bg'] if is_dark else COLORS['Light_Bg']
        text_color = 'white' if is_dark else 'black'

        fig1, ax1 = plt.subplots(figsize=(4.5, 3.5), facecolor=bg_color, dpi=80)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': text_color, 'fontsize': 9})
        ax1.set_title("Duygu Dağılımı (%)", color=text_color, fontsize=11)
        plt.tight_layout(pad=0.5)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=self.right_panel)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="x", pady=5, padx=10)

        fig2, ax2 = plt.subplots(figsize=(4, 4.5), facecolor=bg_color, dpi=80)
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_title("Yorum Sayıları", color=text_color, fontsize=11)
        ax2.set_facecolor(bg_color)
        ax2.tick_params(axis='x', colors=text_color, labelsize=9)
        ax2.tick_params(axis='y', colors=text_color, labelsize=9)
        ax2.spines['bottom'].set_color(text_color)
        ax2.spines['left'].set_color(text_color)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', color=text_color, fontsize=9)

        plt.tight_layout(pad=0.5)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.right_panel)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="x", pady=(20, 20), padx=10)
        
        plt.close(fig1)
        plt.close(fig2)

    def display_comments(self, data):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
            
        for _, row in data.iterrows():
            sentiment = row.get('Tahmin_Edilen_Duygu', 'Nötr')
            user = row.get('Kullanici_Adi', 'Anonim')
            text = row.get(self.col_name, '')
            likes = row.get('Begeni_Sayisi', 0)
            
            border = COLORS.get(sentiment, 'gray')
            
            # Kart Yapısı
            card = ctk.CTkFrame(self.scroll_frame, border_width=2, border_color=border)
            card.pack(fill="x", pady=3)
            
            header = ctk.CTkFrame(card, fg_color="transparent", height=20)
            header.pack(fill="x", padx=8, pady=2)
            
            ctk.CTkLabel(header, text=f"👤 {user}", font=("Arial", 11, "bold")).pack(side="left")
            
            right_text = f"👍 {likes}   |   {sentiment}"
            ctk.CTkLabel(header, text=right_text, text_color=border, font=("Arial", 11, "bold")).pack(side="right")
            
            ctk.CTkLabel(card, text=text, anchor="w", justify="left", wraplength=600, font=("Arial", 12)).pack(fill="x", padx=8, pady=(0,5))

    def filter_comments(self, event):
        query = self.search_entry.get().lower().strip()
        if not query:
            self.display_comments(self.df)
            return
        
        try:
            sentiment_map = {'olumlu': 'Olumlu', 'olumsuz': 'Olumsuz', 'nötr': 'Nötr', 'notr': 'Nötr'}
            if query in sentiment_map:
                filtered = self.df[self.df['Tahmin_Edilen_Duygu'] == sentiment_map[query]]
            else:
                text_mask = self.df[self.col_name].astype(str).str.lower().str.contains(query, na=False)
                sentiment_mask = self.df['Tahmin_Edilen_Duygu'].astype(str).str.lower().str.contains(query, na=False)
                filtered = self.df[text_mask | sentiment_mask]
            
            self.display_comments(filtered)
        except Exception:
            self.display_comments(self.df)

if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()