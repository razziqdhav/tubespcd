import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. TAHAP PRE-PROCESSING & EKSTRAKSI CIRI
# ==========================================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, (100, 100))
    
    # [UPDATE] Menghaluskan gambar untuk menghilangkan noise/pantulan cahaya (flash)
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Masking Background (Memisahkan objek dari latar belakang)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Jika gagal melakukan masking (gambar hitam semua), kembalikan nilai 0 untuk 9 fitur
    if not np.any(mask > 0):
        return [0] * 9
    
    h_channel = hsv_img[:, :, 0]
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    
    # [UPDATE] Ekstraksi 9 Fitur (Mean, Standar Deviasi, dan Median)
    h_mean = np.mean(h_channel[mask > 0])
    s_mean = np.mean(s_channel[mask > 0])
    v_mean = np.mean(v_channel[mask > 0])
    
    h_std = np.std(h_channel[mask > 0])
    s_std = np.std(s_channel[mask > 0])
    v_std = np.std(v_channel[mask > 0])
    
    h_med = np.median(h_channel[mask > 0])
    s_med = np.median(s_channel[mask > 0])
    v_med = np.median(v_channel[mask > 0])
    
    return [h_mean, s_mean, v_mean, h_std, s_std, v_std, h_med, s_med, v_med]

# ==========================================
# 2. TAHAP PEMUATAN DATA (KHUSUS MANGGA 3 KELAS)
# ==========================================
def load_dataset_mangga(dataset_root):
    X, y = [], []
    
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Folder utama '{dataset_root}' tidak ditemukan!")
        return np.array(X), np.array(y)
        
    print(f"[INFO] Memuat dataset KHUSUS MANGGA (Mentah, Matang, Busuk)...")
    
    for folder_name in os.listdir(dataset_root):
        # Kunci: Hanya proses folder yang ada kata "mango"
        if 'mango' not in folder_name.lower():
            continue
            
        # Pelabelan 3 Kelas
        if 'unripe' in folder_name.lower():
            label = 'Mentah'
        elif 'ripe' in folder_name.lower():
            label = 'Matang'
        elif 'rotten' in folder_name.lower() or 'busuk' in folder_name.lower():
            label = 'Busuk'
        else:
            continue
            
        class_folder = os.path.join(dataset_root, folder_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            features = extract_features(file_path)
            
            if features is not None:
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

# ==========================================
# 3. PROGRAM UTAMA & PELATIHAN MODEL
# ==========================================
print("="*50)
print(" SISTEM DETEKSI KEMATANGAN MANGGA AI (CLI) ")
print("="*50)

folder_utama_dataset = "Ripe & Unripe Fruits" # Pastikan nama foldermu benar
X, y = load_dataset_mangga(folder_utama_dataset)

if len(X) == 0:
    print("[ERROR] Dataset mangga tidak ditemukan. Pastikan nama foldernya ada kata 'mango' dan 'unripe/ripe/rotten'.")
    exit()

# Membagi dan menskalakan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Latih KNN dengan K=5
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)

akurasi = accuracy_score(y_test, model_knn.predict(X_test_scaled)) * 100
print(f"[INFO] Berhasil memuat {len(X)} gambar mangga.")
print(f"[INFO] Akurasi pengenalan model: {akurasi:.2f}%")

# ==========================================
# 4. MENGHITUNG BATAS TOLERANSI "BUKAN MANGGA"
# ==========================================
# [UPDATE KETAT] Batas Toleransi menggunakan Mean + (2 * Standar Deviasi)
jarak_training, _ = model_knn.kneighbors(X_train_scaled)
rata_jarak_tiap_data = np.mean(jarak_training, axis=1)

mean_jarak = np.mean(rata_jarak_tiap_data)
std_jarak = np.std(rata_jarak_tiap_data)

# Radius maksimal ciri fisik mangga
batas_toleransi = mean_jarak + (2 * std_jarak)
print(f"[INFO] Sistem Keamanan Aktif (Batas Anomali: {batas_toleransi:.2f})\n")

# ==========================================
# 5. PENGUJIAN GAMBAR BARU (INFERENCE L00P)
# ==========================================
while True:
    print("="*50)
    print("Ketik 'exit' untuk keluar dari program.")
    gambar_tes = input("Masukkan path gambar yang ingin diuji: ").strip('"').strip("'") 
    
    if gambar_tes.lower() == 'exit':
        print("Program dihentikan. Terima kasih!")
        break

    if os.path.exists(gambar_tes):
        # Ekstrak 9 Fitur
        fitur_tes = extract_features(gambar_tes)
        
        # Scaling
        fitur_tes_scaled = scaler.transform([fitur_tes])
        
        # Cek Jarak gambar baru ke data mangga yang ada di memori
        jarak_tes, _ = model_knn.kneighbors(fitur_tes_scaled)
        jarak_rata_rata_tes = np.mean(jarak_tes[0])
        
        print("\n" + "-"*50)
        # Filter Anomali (Apakah ini benar-benar Mangga?)
        if jarak_rata_rata_tes > batas_toleransi:
            print(f">>> ⚠️ HASIL: BUKAN MANGGA! (Objek Ditolak) <<<")
            print(f"(Alasan: Jarak fitur {jarak_rata_rata_tes:.2f} melebihi batas toleransi mangga {batas_toleransi:.2f})")
        else:
            prediksi = model_knn.predict(fitur_tes_scaled)
            print(f">>> ✅ HASIL DETEKSI: Mangga ini terdeteksi {prediksi[0].upper()} <<<")
        print("-"*50 + "\n")
    else:
        print("\n[ERROR] Gambar tidak ditemukan. Pastikan path yang dimasukkan benar.\n")