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
        self.root.geometry("650x780")
        self.root.configure(bg="#F8F9FA") 
        self.root.resizable(False, False)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.model_knn = None
        self.scaler = None
        self.batas_toleransi = 0
        self.train_dir = "KematanganBuahMangga/Train"
        self.test_dir = "KematanganBuahMangga/Test"
        self.image_path = None
        
        self.setup_ui()

    def setup_ui(self):
        header_frame = tk.Frame(self.root, bg="#2C3E50", pady=15)
        header_frame.pack(fill='x')
        tk.Label(header_frame, text="Deteksi Kualitas Mangga", font=("Segoe UI", 18, "bold"), fg="white", bg="#2C3E50").pack()
        tk.Label(header_frame, text="Kenali mangga Mentah, Matang, atau Busuk dengan AI", font=("Segoe UI", 10), fg="#BDC3C7", bg="#2C3E50").pack()

        train_frame = tk.Frame(self.root, bg="#FFFFFF", bd=1, relief="ridge")
        train_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(train_frame, text="1. Persiapan Sistem (Training)", font=("Segoe UI", 12, "bold"), bg="#FFFFFF", fg="#34495E").pack(pady=(10, 5))
        
        self.btn_train = tk.Button(train_frame, text="Mulai Pelatihan Model AI", bg="#27AE60", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", padx=15, pady=5, command=self.train_model)
        self.btn_train.pack(pady=5)

        self.lbl_status = tk.Label(train_frame, text="Status: Model belum dilatih", fg="#E74C3C", bg="#FFFFFF", font=("Segoe UI", 10, "italic"))
        self.lbl_status.pack(pady=(0, 10))

        test_frame = tk.Frame(self.root, bg="#FFFFFF", bd=1, relief="ridge")
        test_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        tk.Label(test_frame, text="2. Pengujian Gambar", font=("Segoe UI", 12, "bold"), bg="#FFFFFF", fg="#34495E").pack(pady=(15, 10))

        self.btn_upload = tk.Button(test_frame, text="Unggah Gambar Mangga", bg="#2980B9", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", padx=15, pady=5, command=self.upload_image, state=tk.DISABLED)
        self.btn_upload.pack()

        self.img_frame = tk.Frame(test_frame, bg="#ECF0F1", width=250, height=250, bd=2, relief="groove")
        self.img_frame.pack(pady=15)
        self.img_frame.pack_propagate(False) 
        
        self.lbl_image = tk.Label(self.img_frame, text="Preview Gambar\n(Kosong)", bg="#ECF0F1", fg="#7F8C8D", font=("Segoe UI", 10))
        self.lbl_image.pack(expand=True, fill='both')

        self.btn_predict = tk.Button(test_frame, text="🔍 Deteksi Kematangan", bg="#E67E22", fg="white", font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2", padx=20, pady=8, command=self.predict_image, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)

        # [UPDATE] Membesarkan font hasil agar lebih jelas
        self.lbl_result = tk.Label(test_frame, text="HASIL AKAN TAMPIL DI SINI", font=("Segoe UI", 18, "bold"), fg="#95A5A6", bg="#FFFFFF")
        self.lbl_result.pack(pady=(5, 20))

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
        if not file_path: return
        
        self.image_path = file_path
        img = Image.open(self.image_path)
        img.thumbnail((240, 240)) 
        
        img_tk = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=img_tk, text="")
        self.lbl_image.image = img_tk 
        
        self.btn_predict.config(state=tk.NORMAL)
        self.lbl_result.config(text="SIAP DIDETEKSI", fg="#7F8C8D")

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