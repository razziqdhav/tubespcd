import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. FUNGSI BACKEND (PENGOLAHAN CITRA & ML)
# ==========================================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img_resized = cv2.resize(img, (100, 100))
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv_img[:, :, 0])
    s_mean = np.mean(hsv_img[:, :, 1])
    v_mean = np.mean(hsv_img[:, :, 2])
    return [h_mean, s_mean, v_mean]

def load_dataset(dataset_root, fruit_keyword):
    X, y = [], []
    if not os.path.exists(dataset_root): return np.array(X), np.array(y)
    for folder_name in os.listdir(dataset_root):
        if fruit_keyword in folder_name.lower():
            label = 'Mentah' if 'unripe' in folder_name.lower() else 'Matang' if 'ripe' in folder_name.lower() else None
            if not label: continue
            class_folder = os.path.join(dataset_root, folder_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# ==========================================
# 2. CLASS APLIKASI GUI
# ==========================================
class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Deteksi Kematangan Buah")
        self.root.geometry("600x650")
        self.root.configure(bg="#f0f0f0")
        
        # Variabel Global untuk Aplikasi
        self.model_knn = None
        self.dataset_root = "Ripe & Unripe Fruits"
        self.image_path = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- Judul ---
        tk.Label(self.root, text="Deteksi Kematangan Buah (PCD)", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=15)

        # --- Frame Pilih Buah & Training ---
        frame_top = tk.Frame(self.root, bg="#f0f0f0")
        frame_top.pack(pady=10)

        tk.Label(frame_top, text="Pilih Buah:", bg="#f0f0f0", font=("Helvetica", 10)).grid(row=0, column=0, padx=5)
        
        self.fruit_combobox = ttk.Combobox(frame_top, values=["Pisang (banana)", "Mangga (mango)", "Pepaya (papaya)", "Apel (apple)"], state="readonly", width=20)
        self.fruit_combobox.current(0)
        self.fruit_combobox.grid(row=0, column=1, padx=5)

        self.btn_train = tk.Button(frame_top, text="Latih Model (Train)", bg="#4CAF50", fg="white", command=self.train_model)
        self.btn_train.grid(row=0, column=2, padx=10)

        self.lbl_status = tk.Label(self.root, text="Status: Model belum dilatih", fg="red", bg="#f0f0f0", font=("Helvetica", 10, "italic"))
        self.lbl_status.pack(pady=5)

        # --- Pemisah ---
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=15, padx=20)

        # --- Frame Upload Gambar ---
        self.btn_upload = tk.Button(self.root, text="Pilih Gambar untuk Diuji", bg="#2196F3", fg="white", command=self.upload_image, state=tk.DISABLED)
        self.btn_upload.pack(pady=10)

        # Area Tampil Gambar (Ukurannya dibiarkan fleksibel)
        self.lbl_image = tk.Label(self.root, bg="#e0e0e0", text="[Preview Gambar]", width=40, height=15)
        self.lbl_image.pack(pady=10)

        # --- Tombol Prediksi ---
        self.btn_predict = tk.Button(self.root, text="Deteksi Kematangan", bg="#FF9800", fg="white", font=("Helvetica", 12, "bold"), command=self.predict_image, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)

        # Area Tampil Hasil
        self.lbl_result = tk.Label(self.root, text="-", font=("Helvetica", 20, "bold"), fg="#333333", bg="#f0f0f0")
        self.lbl_result.pack(pady=10)

    # ==========================================
    # 3. FUNGSI AKSI TOMBOL
    # ==========================================
    def train_model(self):
        selection = self.fruit_combobox.get()
        keyword = selection.split("(")[1].split(")")[0]

        self.lbl_status.config(text=f"Status: Sedang melatih model untuk {keyword}... Mohon tunggu.", fg="blue")
        self.root.update() 

        X, y = load_dataset(self.dataset_root, keyword)
        
        if len(X) == 0:
            messagebox.showerror("Error", f"Dataset untuk '{keyword}' tidak ditemukan di folder {self.dataset_root}!")
            self.lbl_status.config(text="Status: Gagal melatih model", fg="red")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_knn = KNeighborsClassifier(n_neighbors=3)
        self.model_knn.fit(X_train, y_train)

        akurasi = accuracy_score(y_test, self.model_knn.predict(X_test)) * 100
        
        self.lbl_status.config(text=f"Status: Model siap! (Akurasi: {akurasi:.2f}%)", fg="green")
        
        self.btn_upload.config(state=tk.NORMAL)
        self.lbl_result.config(text="-", fg="#333333")
        self.image_path = None
        self.lbl_image.config(image='', text="[Preview Gambar]", width=40, height=15) # Kembalikan ukuran label default

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path: return
        
        self.image_path = file_path
        
        # Tampilkan gambar di UI dengan proporsi yang benar
        img = Image.open(self.image_path)
        
        # Menggunakan thumbnail agar rasio asli terjaga (tidak gepeng/terpotong)
        # Batas maksimal yang ditampilkan di layar GUI adalah 300x300 piksel
        img.thumbnail((300, 300)) 
        
        img_tk = ImageTk.PhotoImage(img)
        
        # Menghapus pengaturan width/height kaku dari label agar menyesuaikan ukuran gambar proporsional
        self.lbl_image.config(image=img_tk, text="", width=0, height=0)
        self.lbl_image.image = img_tk 
        
        self.btn_predict.config(state=tk.NORMAL)
        self.lbl_result.config(text="...", fg="#333333")

    def predict_image(self):
        if not self.model_knn or not self.image_path: return
        
        fitur = extract_features(self.image_path)
        if fitur is None:
            messagebox.showerror("Error", "Gagal membaca ciri gambar.")
            return
            
        prediksi = self.model_knn.predict([fitur])[0]
        
        warna = "#4CAF50" if prediksi == "Matang" else "#F44336" 
        
        self.lbl_result.config(text=f"Hasil: {prediksi.upper()}", fg=warna)

# ==========================================
# JALANKAN APLIKASI
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectorApp(root)
    root.mainloop()