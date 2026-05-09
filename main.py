# ==========================================
# IMPORT LIBRARY YANG DIBUTUHKAN
# ==========================================
import cv2          # OpenCV: Library utama untuk pengolahan citra (baca gambar, resize, ubah warna)
import numpy as np  # NumPy: Digunakan untuk operasi matematika dan manipulasi matriks/array gambar
import os           # OS: Digunakan untuk membaca direktori/folder dan file yang ada di komputer
from sklearn.model_selection import train_test_split # Membagi data menjadi data latih (train) dan data uji (test)
from sklearn.neighbors import KNeighborsClassifier   # Algoritma Machine Learning K-Nearest Neighbors (KNN)
from sklearn.metrics import accuracy_score           # Untuk menghitung persentase akurasi model

# ==========================================
# 1. TAHAP PRE-PROCESSING & EKSTRAKSI CIRI
# ==========================================
def extract_features(image_path):
    """
    Fungsi ini bertugas mengambil satu gambar mentah, memprosesnya, 
    dan menghasilkan 3 angka penting (nilai rata-rata Hue, Saturation, Value).
    """
    # Membaca gambar dari lokasi file (path)
    img = cv2.imread(image_path)
    
    # Jika gambar rusak atau tidak terbaca, lewati dan kembalikan nilai kosong (None)
    if img is None:
        return None
    
    # [Pre-processing 1] Resizing: Mengubah ukuran semua gambar menjadi seragam (100x100 piksel)
    # Ini penting agar komputasi program menjadi ringan dan cepat
    img_resized = cv2.resize(img, (100, 100))
    
    # [Pre-processing 2] Konversi Ruang Warna: Mengubah BGR (bawaan OpenCV) menjadi HSV
    # HSV (Hue, Saturation, Value) jauh lebih baik untuk deteksi warna buah karena tidak terlalu 
    # terpengaruh oleh bayangan atau cahaya dibandingkan RGB.
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # [Ekstraksi Ciri] Mengambil nilai rata-rata dari masing-masing saluran/channel (H, S, dan V)
    # Channel 0 adalah Hue (Jenis warna: merah, hijau, kuning, dll)
    h_mean = np.mean(hsv_img[:, :, 0])
    # Channel 1 adalah Saturation (Kepekatan warna)
    s_mean = np.mean(hsv_img[:, :, 1])
    # Channel 2 adalah Value (Tingkat kecerahan)
    v_mean = np.mean(hsv_img[:, :, 2])
    
    # Mengembalikan hasil ekstraksi berupa array (daftar) berisi 3 nilai warna tersebut
    return [h_mean, s_mean, v_mean]

# ==========================================
# 2. TAHAP PEMUATAN DATA (LOAD DATASET)
# ==========================================
def load_dataset(dataset_root, fruit_keyword):
    """
    Fungsi ini membuka folder dataset utama, mencari buah yang diminta,
    mengekstrak fitur semua gambarnya, dan memberinya label (Mentah/Matang).
    """
    X = [] # List X: Tempat menyimpan fitur (kumpulan nilai rata-rata HSV tadi)
    y = [] # List y: Tempat menyimpan label/jawaban (contoh: "Mentah", "Matang")
    
    # Mengecek apakah folder utama (Ripe & Unripe Fruits) ada di komputer
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Folder utama '{dataset_root}' tidak ditemukan!")
        return np.array(X), np.array(y)
        
    print(f"[INFO] Mengekstrak ciri warna untuk '{fruit_keyword}' di dalam {dataset_root}...")
    
    # Membaca satu per satu folder yang ada di dalam "Ripe & Unripe Fruits"
    for folder_name in os.listdir(dataset_root):
        
        # Mengecek apakah nama folder saat ini sesuai dengan buah yang dicari user (misal: "banana")
        if fruit_keyword in folder_name.lower():
            
            # [Pelabelan] Memberikan label berdasarkan nama foldernya
            if 'unripe' in folder_name.lower():
                label = 'Mentah'
            elif 'ripe' in folder_name.lower():
                label = 'Matang'
            else:
                continue # Jika tidak ada kata ripe/unripe, lewati folder ini
                
            # Menggabungkan nama folder utama dan sub-folder
            class_folder = os.path.join(dataset_root, folder_name)
            
            # Membaca satu per satu gambar di dalam folder buah tersebut
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                
                # Memanggil fungsi ekstraksi ciri di atas untuk memproses gambar ini
                features = extract_features(file_path)
                
                # Jika gambarnya berhasil diekstrak warnanya
                if features is not None:
                    X.append(features) # Masukkan nilai warnanya ke dalam list X
                    y.append(label)    # Masukkan labelnya (Mentah/Matang) ke list y
                    
    # Mengubah list Python biasa menjadi array NumPy agar bisa dibaca oleh algoritma Machine Learning
    return np.array(X), np.array(y)

# ==========================================
# 3. ANTARMUKA PROGRAM UTAMA (MENU)
# ==========================================
print("="*40)
print(" PROGRAM DETEKSI KEMATANGAN BUAH (PCD)")
print("="*40)
print("Pilih jenis buah yang ingin diuji:")
print("1. Pisang (Banana)")
print("2. Mangga (Mango)")
print("3. Pepaya (Papaya)")
print("4. Apel (Apple)")

# Meminta user memasukkan angka pilihan
pilihan = input("Masukkan angka pilihan (1-4): ")

# Kamus (dictionary) untuk menerjemahkan angka pilihan ke nama buah dan kata kunci foldernya
daftar_buah = {
    '1': ('Pisang', 'banana'), 
    '2': ('Mangga', 'mango'), 
    '3': ('Pepaya', 'papaya'),
    '4': ('Apel', 'apple')
}

# Jika user memasukkan angka selain 1, 2, 3, atau 4
if pilihan not in daftar_buah:
    print("Pilihan tidak valid! Program dihentikan.")
    exit()

# Menyimpan pilihan user ke dalam variabel
nama_buah_id, keyword_en = daftar_buah[pilihan]

# Mendefinisikan nama folder tempat gambar disimpan
folder_utama_dataset = "Ripe & Unripe Fruits" 

# Menjalankan fungsi pemuatan data dan menyimpan hasilnya di variabel X dan y
X, y = load_dataset(folder_utama_dataset, keyword_en)

# Jika ternyata gambarnya tidak ada
if len(X) == 0:
    print(f"[ERROR] Tidak ada gambar yang berhasil dimuat untuk buah {nama_buah_id}.")
    exit()

print(f"[INFO] Berhasil memuat dan mengekstrak ciri dari {len(X)} gambar {nama_buah_id}.")

# ==========================================
# 4. TAHAP MACHINE LEARNING (KLASIFIKASI KNN)
# ==========================================
# Memecah data: 80% data digunakan model untuk belajar (train), 20% digunakan untuk ujian (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan algoritma K-Nearest Neighbors (KNN) dengan melihat 3 tetangga warna terdekat (k=3)
model_knn = KNeighborsClassifier(n_neighbors=3)

# Meminta model KNN untuk "belajar" mencocokkan ciri warna (X_train) dengan jawabannya (y_train)
model_knn.fit(X_train, y_train)

# Menguji kepintaran model menggunakan data ujian (X_test) dan menghitung persentase kebenarannya
akurasi = accuracy_score(y_test, model_knn.predict(X_test)) * 100
print(f"[INFO] Model selesai dilatih. Akurasi sistem pengenalan: {akurasi:.2f}%")

# ==========================================
# 5. TAHAP PENGUJIAN GAMBAR BARU (INFERENCE)
# ==========================================
print("\n" + "="*40)
print("Contoh format path: Ripe & Unripe Fruits/ripe banana/gambar1.jpg")

# Meminta user memasukkan lokasi gambar yang mau ditebak kematangannya
gambar_tes = input(f"Masukkan path gambar {nama_buah_id} yang ingin diuji: ")

# Membersihkan karakter tanda kutip (" atau ') jika user melakukan drag-and-drop file gambar ke terminal
gambar_tes = gambar_tes.strip('"').strip("'") 

# Memeriksa apakah file gambar tersebut benar-benar ada di komputer
if os.path.exists(gambar_tes):
    
    # Tahap 1: Ekstrak dulu ciri warna gambar barunya
    fitur_tes = extract_features(gambar_tes)
    
    # Tahap 2: Suruh model KNN menebak berdasarkan ciri warna tersebut
    prediksi = model_knn.predict([fitur_tes])
    
    # Tahap 3: Tampilkan hasil tebakannya ke layar
    print("\n" + "="*40)
    print(f">>> HASIL DETEKSI: {nama_buah_id} ini terdeteksi {prediksi[0].upper()} <<<")
    print("="*40)
else:
    print("\n[ERROR] Gambar tidak ditemukan. Pastikan path yang dimasukkan benar.")