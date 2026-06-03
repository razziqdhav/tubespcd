import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# 1. FUNGSI BACKEND (IMAGE PROCESSING & FITUR)
# ==========================================

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Segmentasi Thresholding Otsu
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Konversi ke HSV
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    
    if not np.any(mask > 0):
        return [0] * 12
    
    # --- EKSTRAKSI FITUR 1: WARNA (9 Fitur HSV) ---
    h_mean = np.mean(h_channel[mask > 0])
    s_mean = np.mean(s_channel[mask > 0])
    v_mean = np.mean(v_channel[mask > 0])
    
    h_std = np.std(h_channel[mask > 0])
    s_std = np.std(s_channel[mask > 0])
    v_std = np.std(v_channel[mask > 0])
    
    h_med = np.median(h_channel[mask > 0])
    s_med = np.median(s_channel[mask > 0])
    v_med = np.median(v_channel[mask > 0])
    
    warna_features = [h_mean, s_mean, v_mean, h_std, s_std, v_std, h_med, s_med, v_med]
    
    # --- EKSTRAKSI FITUR 2: TEKSTUR (1 Fitur Canny Edge Density) ---
    edges = cv2.Canny(gray, 100, 200)
    mango_area = np.sum(mask > 0)
    edge_density = np.sum(edges[mask > 0] > 0) / mango_area if mango_area > 0 else 0
    
    # --- EKSTRAKSI FITUR 3: CACAT / BINTIK HITAM (2 Fitur) ---
    # Filter tingkat kegelapan diperketat (V < 30) agar benar-benar mengambil bercak busuk pekat
    dark_pixels = (v_channel < 30) & (mask > 0)
    
    # Erosi masker luar sebanyak 5 piksel untuk membuang noise bayangan tepi kulit
    kernel_edge = np.ones((5,5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_edge, iterations=1)
    
    dark_pixels_clean = (v_channel < 30) & (eroded_mask > 0)
    dark_area_percentage = np.sum(dark_pixels_clean) / mango_area if mango_area > 0 else 0
    
    # Pembersihan Morfologi (Opening) gumpalan bintik
    dark_mask_clean = dark_pixels_clean.astype(np.uint8) * 255
    kernel_morph = np.ones((3,3), np.uint8)
    dark_mask_clean = cv2.morphologyEx(dark_mask_clean, cv2.MORPH_OPEN, kernel_morph)
    
    # Hitung kontur bintik yang luasnya signifikan (> 15 piksel agar bintik busuk kecil tetap tertangkap)
    contours, _ = cv2.findContours(dark_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jumlah_bintik = len([c for c in contours if cv2.contourArea(c) > 15])
    jumlah_bintik_norm = min(jumlah_bintik / 20, 1.0)
    
    fitur_bintik = [dark_area_percentage, jumlah_bintik_norm]
    
    return warna_features + [edge_density] + fitur_bintik


def process_and_visualize(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, (180, 180))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    original = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    edges = cv2.Canny(blurred, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    segmented = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_channel = cv2.cvtColor(hsv[:,:,0], cv2.COLOR_GRAY2RGB)
    s_channel = cv2.cvtColor(hsv[:,:,1], cv2.COLOR_GRAY2RGB)
    v_channel = cv2.cvtColor(hsv[:,:,2], cv2.COLOR_GRAY2RGB)
    
    v = hsv[:,:,2]
    kernel_edge = np.ones((5,5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_edge, iterations=1)
    dark_spots = (v < 30) & (eroded_mask > 0)
    
    dark_vis = np.zeros_like(img_resized)
    dark_vis[dark_spots] = [0, 0, 255]
    dark_overlay = cv2.addWeighted(img_resized, 0.7, dark_vis, 0.3, 0)
    dark_overlay_rgb = cv2.cvtColor(dark_overlay, cv2.COLOR_BGR2RGB)
    
    return {
        'original': original, 'grayscale': grayscale, 'blurred': blurred_rgb,
        'mask': mask_rgb, 'edges': edges_rgb, 'segmented': segmented_rgb,
        'hue': h_channel, 'saturation': s_channel, 'value': v_channel,
        'dark_spots': dark_overlay_rgb
    }


# ==========================================
# 2. CLASS APLIKASI GUI
# ==========================================

class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Kematangan Mangga - Rule-Based Final Super Akurat")
        self.root.geometry("1400x850")
        self.root.configure(bg="#1E1E2E")
        self.root.resizable(True, True)
        
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        BG = "#1E1E2E"
        CARD = "#313244"
        ACCENT = "#89B4FA"
        SUCCESS = "#A6E3A1"
        DANGER = "#F38BA8"
        TEXT = "#CDD6F4"
        
        self.root.configure(bg=BG)
        
        header = tk.Frame(self.root, bg=ACCENT, height=45)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title = tk.Label(header, text="DETERMINISTIC MANGO DETECTOR | Unified Rules Version", 
                        font=("Segoe UI", 14, "bold"), bg=ACCENT, fg=BG)
        title.pack(pady=8)
        
        main_paned = tk.PanedWindow(self.root, bg=BG, sashwidth=5, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=15, text=None)
        
        left_panel = tk.Frame(main_paned, bg=CARD, relief="flat", bd=0, width=320)
        main_paned.add(left_panel, width=320)
        left_panel.pack_propagate(False)
        
        right_panel = tk.Frame(main_paned, bg=CARD, relief="flat", bd=0)
        main_paned.add(right_panel, width=1000)
        
        left_canvas = tk.Canvas(left_panel, bg=CARD, highlightthickness=0)
        scrollbar = tk.Scrollbar(left_panel, orient="vertical", command=left_canvas.yview)
        left_scrollable = tk.Frame(left_canvas, bg=CARD)
        
        left_canvas.configure(yscrollcommand=scrollbar.set)
        left_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas_window = left_canvas.create_window((0, 0), window=left_scrollable, anchor="nw", width=300)
        
        def configure_scroll_region(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        
        def configure_canvas_width(event):
            left_canvas.itemconfig(canvas_window, width=event.width)
        
        left_scrollable.bind("<Configure>", configure_scroll_region)
        left_canvas.bind("<Configure>", configure_canvas_width)
        
        info_frame = tk.LabelFrame(left_scrollable, text="ℹ️ Informasi Sistem", 
                               bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold"))
        info_frame.pack(fill="x", padx=10, pady=10)
        
        lbl_info = tk.Label(info_frame, text="Sistem Aturan Terpadu.\nAturan dirancang bertingkat untuk\nmemastikan presisi tinggi.", 
                            bg=CARD, fg=ACCENT, font=("Segoe UI", 9, "italic"), justify="left")
        lbl_info.pack(pady=10, padx=5)
        
        upload_frame = tk.LabelFrame(left_scrollable, text="📸 Upload Gambar", 
                                    bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold"))
        upload_frame.pack(fill="x", padx=10, pady=10)
        
        self.btn_upload = tk.Button(upload_frame, text="📁 Pilih Gambar", 
                                   bg="#45475A", fg=TEXT, font=("Segoe UI", 10),
                                   relief="flat", cursor="hand2", padx=10, pady=5,
                                   command=self.upload_image)
        self.btn_upload.pack(pady=5)
        
        self.preview_frame = tk.Frame(upload_frame, bg="#1E1E2E", width=260, height=180)
        self.preview_frame.pack(pady=10)
        self.preview_frame.pack_propagate(False)
        
        self.lbl_preview = tk.Label(self.preview_frame, bg="#1E1E2E", fg=TEXT, 
                                   text="🖼️ Preview akan muncul disini", font=("Segoe UI", 9))
        self.lbl_preview.pack(fill="both", expand=True)
        
        self.btn_predict = tk.Button(upload_frame, text="🔍 DETEKSI KEMATANGAN", 
                                    bg=SUCCESS, fg=BG, font=("Segoe UI", 11, "bold"),
                                    relief="flat", cursor="hand2", padx=15, pady=8,
                                    command=self.predict_image)
        self.btn_predict.pack(pady=10)
        
        result_frame = tk.LabelFrame(left_scrollable, text="📋 HASIL DETEKSI", 
                                    bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold"))
        result_frame.pack(fill="x", padx=10, pady=10)
        
        self.lbl_result = tk.Label(result_frame, text="—", 
                                  bg=CARD, fg=TEXT, font=("Segoe UI", 16, "bold"),
                                  pady=20)
        self.lbl_result.pack()
        
        self.lbl_confidence = tk.Label(result_frame, text="", 
                                      bg=CARD, fg=TEXT, font=("Segoe UI", 10))
        self.lbl_confidence.pack(pady=(0, 10))
        
        fitur_frame = tk.LabelFrame(left_scrollable, text="🔧 Fitur yang Digunakan", 
                                   bg=CARD, fg=ACCENT, font=("Segoe UI", 10, "bold"))
        fitur_frame.pack(fill="x", padx=10, pady=10)
        
        fitur_text = """
        ┌─────────────────────────────────┐
        │  🎨 FITUR WARNA (9)             │
        │    • Hue Mean, Std, Median      │
        │    • Saturation Mean, Std, Med  │
        │    • Value Mean, Std, Median    │
        ├─────────────────────────────────┤
        │  📐 FITUR CANNY EDGE (1)         │
        │    • Edge Density               │
        ├─────────────────────────────────┤
        │  ⚫ FITUR BINTIK HITAM (2)       │
        │    • Dark Area Percentage       │
        │    • Jumlah Bintik (Normalized) │
        ├─────────────────────────────────┤
        │  📊 TOTAL: 12 FITUR             │
        └─────────────────────────────────┘
        """
        
        fitur_label = tk.Label(fitur_frame, text=fitur_text, 
                              bg=CARD, fg=TEXT, font=("Consolas", 9), 
                              justify="left", anchor="w")
        fitur_label.pack(pady=8, padx=5)
        
        vis_title = tk.Label(right_panel, text="🔄 TAHAPAN PREPROCESSING & DETEKSI BINTIK", 
                            bg=CARD, fg=ACCENT, font=("Segoe UI", 12, "bold"))
        vis_title.pack(pady=10)
        
        self.fig = plt.Figure(figsize=(11, 7), facecolor=CARD, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_vis_status = tk.Label(right_panel, text="⬅️ Upload gambar untuk melihat preprocessing", 
                                      bg=CARD, fg=TEXT, font=("Segoe UI", 10))
        self.lbl_vis_status.pack(pady=5)

    def update_visualization(self):
        if not self.image_path:
            return
        
        try:
            result = process_and_visualize(self.image_path)
            if result is None:
                return
            
            self.fig.clear()
            
            images = [
                ("Original", result['original']),
                ("Grayscale", result['grayscale']),
                ("Gaussian Blur", result['blurred']),
                ("Otsu Mask", result['mask']),
                ("Canny Edge", result['edges']),
                ("Segmented", result['segmented']),
                ("Hue", result['hue']),
                ("Saturation", result['saturation']),
                ("Value", result['value']),
                ("🔴 Dark Spots", result['dark_spots'])
            ]
            
            for i, (title, img) in enumerate(images):
                ax = self.fig.add_subplot(2, 5, i+1)
                ax.imshow(img)
                ax.set_title(title, fontsize=9, color='white')
                ax.axis('off')
                ax.set_facecolor('#313244')
            
            self.fig.tight_layout(pad=1.5)
            self.fig.patch.set_facecolor('#313244')
            
            for ax in self.fig.axes:
                ax.title.set_color('white')
            
            self.canvas.draw()
            self.lbl_vis_status.config(text="✅ Preprocessing selesai! Area merah = bintik hitam (ciri busuk)", fg="#A6E3A1")
            
        except Exception as e:
            self.lbl_vis_status.config(text=f"❌ Error: {str(e)}", fg="#F38BA8")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Mangga", 
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        self.image_path = file_path
        
        try:
            img = Image.open(file_path)
            img = img.resize((240, 170), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=self.preview_image, text="")
            self.lbl_preview.image = self.preview_image
            
            self.update_visualization()
            
            self.lbl_result.config(text="⏳ Klik Deteksi", fg="#CDD6F4")
            self.lbl_confidence.config(text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuka gambar\n{str(e)}")

    # ==========================================
    # 3. ATURAN LOGIKA FINAL TERPADU (SUPER PRESISI)
    # ==========================================
    def predict_image(self):
        try:
            if not self.image_path: 
                messagebox.showwarning("Peringatan", "Upload gambar dahulu!")
                return
            
            self.lbl_result.config(text="🔄 Memproses Aturan...", fg="#F9E2AF")
            self.lbl_confidence.config(text="")
            self.root.update()

            fitur = extract_features(self.image_path)
            
            if fitur is None:
                self.lbl_result.config(text="❌ Gagal", fg="#F38BA8")
                return
                
            h_mean = fitur[0]
            edge_density = fitur[9]          
            dark_area_percentage = fitur[10] 
            jumlah_bintik_norm = fitur[11]   
            
            # --- BLOK ATURAN LOGIKA BERTINGKAT (NESTED RULE-BASED) ---
            
            # ATURAN UTAMA 1: TANGKAP INDIKASI BUSUK DULUAN
            # Jika terdeteksi bintik hitam pekat di atas 1% (0.01) atau gumpalan bintik terkonfirmasi, langsung divonis Busuk.
            # Ini akan memotong semua noise gambar ramai maupun tunggal karena sifatnya mutlak.
            if dark_area_percentage > 0.01 or jumlah_bintik_norm > 0.0:
                prediksi = "Busuk"
                indikator = f"Bercak Pembusukan Terdeteksi ({dark_area_percentage*100:.1f}%)"
            
            # ATURAN UTAMA 2: JIKA LOLOS SENSOR BUSUK, CEK APAKAH MULTI-OBJEK RAMAI
            elif edge_density > 0.12:
                if h_mean > 35:
                    prediksi = "Mentah"
                    indikator = f"Tumpukan Buah | Dominan Hijau"
                else:
                    prediksi = "Matang"
                    indikator = f"Tumpukan Buah | Dominan Kuning"
            
            # ATURAN UTAMA 3: KONDISI NORMAL / OBJEK TUNGGAL SEPI
            else:
                if h_mean > 35:
                    prediksi = "Mentah"
                    indikator = f"Objek Tunggal | Warna Hijau (Hue: {h_mean:.1f})"
                else:
                    prediksi = "Matang"
                    indikator = f"Objek Tunggal | Warna Kuning (Hue: {h_mean:.1f})"
            
            # --- OUTPUT TAMPILAN KE COMPONENT GUI ---
            if prediksi == "Matang":
                self.lbl_result.config(text="🥭 MANGGA MATANG ✓", fg="#A6E3A1")
            elif prediksi == "Mentah":
                self.lbl_result.config(text="🥭 MANGGA MENTAH", fg="#F9E2AF")
            else:
                self.lbl_result.config(text="🥭 MANGGA BUSUK ✗", fg="#F38BA8")
            
            self.lbl_confidence.config(text=f"Metode: Rule-Based Terpadu | Pemicu: {indikator}")
            
            print(f"\n=== PREDIKSI DETEKSI DETERMINISTIK ===")
            print(f"Hasil Akhir: {prediksi}")
            print(f"Detail Fitur -> Hue Mean: {h_mean:.2f} | Kerapatan Tepi: {edge_density:.4f} | Area Bintik: {dark_area_percentage:.4f} | Jml Bintik Norm: {jumlah_bintik_norm:.4f}")

        except Exception as e:
            messagebox.showerror("Error", f"Proses prediksi gagal!\n{str(e)}")


# ==========================================
# 4. RUNNING INTERFACE JENDELA UTAMA
# ==========================================

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectorApp(root)
    root.mainloop()