import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. FUNGSI BACKEND (PENGOLAHAN CITRA & ML)
# ==========================================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    img_resized = cv2.resize(img, (100, 100))
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if not np.any(mask > 0): return [0]*9 
        
    h_channel, s_channel, v_channel = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    
    h_mean, s_mean, v_mean = np.mean(h_channel[mask > 0]), np.mean(s_channel[mask > 0]), np.mean(v_channel[mask > 0])
    h_std, s_std, v_std = np.std(h_channel[mask > 0]), np.std(s_channel[mask > 0]), np.std(v_channel[mask > 0])
    h_med, s_med, v_med = np.median(h_channel[mask > 0]), np.median(s_channel[mask > 0]), np.median(v_channel[mask > 0])
    
    return [h_mean, s_mean, v_mean, h_std, s_std, v_std, h_med, s_med, v_med]

def load_dataset_mangga(folder_path):
    X, y = [], []
    if not os.path.exists(folder_path): return np.array(X), np.array(y)
    
    for folder_name in os.listdir(folder_path):
        name_lower = folder_name.lower()
        if name_lower == 'unripe': label = 'Mentah'
        elif name_lower == 'ripe': label = 'Matang'
        # [UPDATE] Menangkap typo 'overipe' (R nya 1) dan 'overripe' (R nya 2)
        elif name_lower in ['overripe', 'overipe']: label = 'Busuk'
        else: continue
            
        class_folder = os.path.join(folder_path, folder_name)
        if not os.path.isdir(class_folder): continue
        
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

# ==========================================
# 2. CLASS APLIKASI GUI MODERN
# ==========================================
class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Kematangan Mangga AI")
        # [UPDATE] Memanjangkan window ke bawah agar teks hasil tidak terpotong
        self.root.geometry("950x850")
        self.root.configure(bg="#F4F6F9") 
        self.root.resizable(False, False)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.model_knn = None
        self.scaler = None
        self.batas_toleransi = 0
        self.train_dir = "KematanganBuahMangga/Train"
        self.test_dir = "KematanganBuahMangga/Test"
        self.image_path = None
        self.preview_image = None
        
        self.setup_ui()

    def setup_ui(self):
        # =========================
        # WARNA TEMA
        # =========================
        BG_COLOR = "#E0F2FE"
        CARD_COLOR = "#FFFFFF"
        BLUE = "#0EA5E9" 
        SECONDARY = "#38BDF8"
        TEXT = "#0F172A"
        SUBTEXT = "#475569"
        
        self.root.configure(bg=BG_COLOR)
        # =========================
        # HEADER / NAVBAR
        # =========================
        navbar = tk.Frame(self.root, bg="#0EA5E9", height=70)
        navbar.pack(fill="x")
        logo = tk.Label(navbar,
                        text="🍋 Mango AI Detector",
                        font=("Segoe UI", 18, "bold"),
                        bg="#0EA5E9",
                        fg=TEXT)
        logo.pack(side="left", padx=25, pady=15)
        
        nav_btn = tk.Button(navbar,
                            text="AI Detection",
                            bg=SECONDARY,
                            fg=TEXT,
                            relief="flat",
                            font=("Segoe UI", 10, "bold"),
                            padx=20,
                            pady=8,
                            cursor="hand2")
        nav_btn.pack(side="right", padx=25)
        
        # =========================
        # HERO SECTION
        # =========================
        hero = tk.Frame(self.root, bg=BG_COLOR)
        hero.pack(pady=25)
        
        badge = tk.Label(hero,
                         text="✨ AI Powered Mango Classification",
                         bg="#BAE6FD",
                         fg="#0369A1",
                         font=("Segoe UI", 10),
                         padx=15,
                         pady=5)
        badge.pack(pady=10)
        
        title = tk.Label(hero,
                         text="Detect Mango Ripeness\nQuickly & Accurately",
                         bg=BG_COLOR,
                         fg=TEXT,
                         font=("Segoe UI", 28, "bold"),
                         justify="center")
        title.pack()
        
        desc = tk.Label(hero,
                        text="Unggah gambar mangga dan biarkan AI menentukan\napakah mangga matang, mentah, atau busuk.",
                        bg=BG_COLOR,
                        fg=SUBTEXT,
                        font=("Segoe UI", 12),
                        justify="center")
        desc.pack(pady=15)
        
        # =========================
        # MAIN CARD
        # =========================
        main_card = tk.Frame(self.root,
                             bg=CARD_COLOR,
                             bd=0,
                             highlightthickness=0)
        main_card.pack(padx=25, pady=10, fill="both", expand=True)
        
        # =========================
        # TRAIN SECTION
        # =========================
        train_title = tk.Label(main_card,
                               text="Train AI Model",
                               bg=CARD_COLOR,
                               fg=TEXT,
                               font=("Segoe UI", 16, "bold"))
        train_title.pack(pady=(20, 10))
        
        self.btn_train = tk.Button(main_card,
                                   text="Mulai Training",
                                   bg=SECONDARY,
                                   fg=TEXT,
                                   activebackground="#6D28D9",
                                   activeforeground="white",
                                   relief="flat",
                                   cursor="hand2",
                                   font=("Segoe UI", 11, "bold"),
                                   padx=20,
                                   pady=10,
                                   command=self.train_model)
        self.btn_train.pack()
        
        self.lbl_status = tk.Label(main_card,
                                   text="Model belum dilatih",
                                   bg=CARD_COLOR,
                                   fg="#FCA5A5",
                                   font=("Segoe UI", 10))
        self.lbl_status.pack(pady=10)
        
        # =========================
        # IMAGE PREVIEW
        # =========================
        self.img_frame = tk.Frame(main_card,
                                  bg="#0EA5E9",
                                  width=320,
                                  height=240)
        self.img_frame.pack(pady=20)
        
        self.img_frame.pack_propagate(False)
        
        self.lbl_image = tk.Label(self.img_frame, 
                                  bg="#BAE6FD",
                                  fg="#0369A1",
                                  text="Preview Gambar",
                                  font=("Segoe UI", 12))
        self.lbl_image.pack(fill="both", expand=True)
        
        button_container = tk.Frame(main_card, bg=CARD_COLOR, height=100)
        button_container.pack(pady=20)
        button_container.pack_propagate(False)

        # =========================
        # BUTTON UPLOAD
        # # =========================
        self.btn_upload = tk.Button(button_container, text="📁 Upload Gambar",
                                    bg="#0EA5E9", fg=TEXT,
                                    activebackground="#6D28D9",
                                    activeforeground="white",
                                    relief="raised", bd=2, cursor="hand2",
                                    font=("Segoe UI", 11, "bold"),
                                    width=18, height=2, command=self.predict_image)
        self.btn_upload.pack(side="left", padx=15)
        
        # =========================
        # BUTTON DETEKSI
        # =========================
        self.btn_predict = tk.Button(button_container,
                                     text="🔍 Deteksi",
                                     bg="#38BDF8",
                                     fg=TEXT,
                                     activebackground="#6D28D9",
                                     activeforeground="white",
                                     relief="flat",
                                     cursor="hand2",
                                     font=("Segoe UI", 11, "bold"),
                                     padx=20,
                                     pady=10,
                                     command=self.predict_image)
        self.btn_predict.pack(side="left", padx=15)
        
        # =========================
        # RESULT BOX
        # =========================
        result_frame = tk.Frame(main_card,
                                bg="#E0F2FE",
                                padx=20,
                                pady=20)
        result_frame.pack(pady=(10, 25), padx=20, fill="x")
        
        self.lbl_result = tk.Label(result_frame,
                                   text="HASIL DETEKSI",
                                   bg="#1E293B",
                                   fg="#0369A1",
                                   font=("Segoe UI", 20, "bold"))
        self.lbl_result.pack()

    def train_model(self):
        try:
            self.lbl_status.config(text="Status: Sedang memproses dataset Train & Test...", fg="#2980B9")
            self.root.update() 

            X_train, y_train = load_dataset_mangga(self.train_dir)
            X_test, y_test = load_dataset_mangga(self.test_dir)
            
            if len(X_train) == 0:
                messagebox.showerror("Error Folder Train", f"Tidak ada gambar valid di dalam {self.train_dir}!")
                self.lbl_status.config(text="Status: Gagal (Data Latih Kosong)", fg="#E74C3C")
                return

            k_value = 5 if len(X_train) >= 5 else len(X_train)

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            self.model_knn = KNeighborsClassifier(n_neighbors=k_value)
            self.model_knn.fit(X_train_scaled, y_train)

            akurasi_teks = "(Akurasi: Tidak Teruji)"
            if len(X_test) > 0:
                X_test_scaled = self.scaler.transform(X_test)
                akurasi = accuracy_score(y_test, self.model_knn.predict(X_test_scaled)) * 100
                akurasi_teks = f"(Akurasi: {akurasi:.2f}%)"

            jarak_training, _ = self.model_knn.kneighbors(X_train_scaled)
            rata_jarak_tiap_data = np.mean(jarak_training, axis=1)
            self.batas_toleransi = np.mean(rata_jarak_tiap_data) + (2 * np.std(rata_jarak_tiap_data))

            self.lbl_status.config(text=f"Status: Model Siap! {akurasi_teks}", fg="#27AE60")
            self.btn_train.config(text="Latih Ulang Model", bg="#95A5A6")
            self.btn_upload.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error Sistem Training", f"Gagal Melatih Model!\nPesan Error: {str(e)}")
            self.lbl_status.config(text="Status: Error Sistem", fg="#E74C3C")

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        
        if not file_path:
            return
        
        self.image_path = file_path
        
        try:
            img = Image.open(file_path)
            img = img.resize((280, 220))
            self.preview_image = ImageTk.PhotoImage(img)
            
            self.lbl_image.config(image=self.preview_image, text="")
            
            self.lbl_image.image = self.preview_image
            
            self.lbl_result.config(text="✅ Gambar berhasil diupload", fg="#38BDF8")
            
        except Exception as e:
            messagebox.showerror("Error Upload", f"Gagal membuka gambar\n\n{str(e)}")

    def predict_image(self):
        try:
            if not self.model_knn or not self.image_path: 
                return
            
            self.lbl_result.config(text="MEMPROSES...", fg="#E67E22")
            self.root.update()

            fitur = extract_features(self.image_path)
            
            if fitur is None:
                messagebox.showerror("Error Gambar", "Gambar rusak atau tidak bisa dibaca.")
                self.lbl_result.config(text="GAGAL", fg="#E74C3C")
                return
                
            fitur_scaled = self.scaler.transform([fitur])
            
            jarak_tes, _ = self.model_knn.kneighbors(fitur_scaled)
            jarak_rata = np.mean(jarak_tes[0])
            
            # [UPDATE] Semua proses print di terminal sudah dihapus
            
            if jarak_rata > self.batas_toleransi:
                self.lbl_result.config(text="⚠️ BUKAN MANGGA!", fg="#E74C3C")
            else:
                prediksi = self.model_knn.predict(fitur_scaled)[0]
                
                if prediksi == "Matang": warna = "#27AE60"
                elif prediksi == "Mentah": warna = "#F1C40F"
                else: warna = "#8E44AD"
                    
                # [UPDATE] Output disesuaikan dengan permintaan: "Mangga ini matang", dll.
                self.lbl_result.config(text=f"Mangga ini {prediksi.lower()}", fg=warna)

        except Exception as e:
             messagebox.showerror("Error Sistem Deteksi", f"Proses prediksi terhenti!\nPesan Error: {str(e)}")
             self.lbl_result.config(text="ERROR", fg="#E74C3C")

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectorApp(root)
    root.mainloop()