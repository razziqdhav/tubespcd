import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. TAHAP PRE-PROCESSING & EKSTRAKSI CIRI
# ==========================================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    img_resized = cv2.resize(img, (100, 100))
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if not np.any(mask > 0): return [0] * 9
    
    h_channel, s_channel, v_channel = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    
    h_mean, s_mean, v_mean = np.mean(h_channel[mask > 0]), np.mean(s_channel[mask > 0]), np.mean(v_channel[mask > 0])
    h_std, s_std, v_std = np.std(h_channel[mask > 0]), np.std(s_channel[mask > 0]), np.std(v_channel[mask > 0])
    h_med, s_med, v_med = np.median(h_channel[mask > 0]), np.median(s_channel[mask > 0]), np.median(v_channel[mask > 0])
    
    return [h_mean, s_mean, v_mean, h_std, s_std, v_std, h_med, s_med, v_med]

# ==========================================
# 2. TAHAP PEMUATAN DATA (SESUAI FOLDER BARU)
# ==========================================
def load_dataset_mangga(folder_path):
    X, y = [], []
    if not os.path.exists(folder_path): 
        print(f"[ERROR] Folder '{folder_path}' tidak ditemukan!")
        return np.array(X), np.array(y)
        
    for folder_name in os.listdir(folder_path):
        name_lower = folder_name.lower()
        
        # Penyesuaian dengan nama folder Overripe, Ripe, Unripe
        if name_lower == 'unripe': label = 'Mentah'
        elif name_lower == 'ripe': label = 'Matang'
        elif name_lower == 'overripe': label = 'Busuk'
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
# 3. PROGRAM UTAMA & PELATIHAN MODEL
# ==========================================
print("="*50)
print(" SISTEM DETEKSI KEMATANGAN MANGGA BERBASIS KNN ")
print("="*50)

folder_train = "KematanganBuahMangga/Train"
folder_test = "KematanganBuahMangga/Test"

print("[INFO] Memuat data Train...")
X_train, y_train = load_dataset_mangga(folder_train)
print("[INFO] Memuat data Test...")
X_test, y_test = load_dataset_mangga(folder_test)

if len(X_train) == 0 or len(X_test) == 0:
    print("[ERROR] Data Train atau Test kosong. Pastikan path folder benar.")
    exit()

# Scaling Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Latih KNN
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)

akurasi = accuracy_score(y_test, model_knn.predict(X_test_scaled)) * 100
print(f"[INFO] Berhasil memuat {len(X_train)} gambar latih & {len(X_test)} gambar uji.")
print(f"[INFO] Akurasi pengenalan model: {akurasi:.2f}%")

# ==========================================
# 4. BATAS TOLERANSI ANOMALI
# ==========================================
jarak_training, _ = model_knn.kneighbors(X_train_scaled)
rata_jarak_tiap_data = np.mean(jarak_training, axis=1)
batas_toleransi = np.mean(rata_jarak_tiap_data) + (2 * np.std(rata_jarak_tiap_data))

print(f"[INFO] Sistem Keamanan Aktif (Batas Anomali: {batas_toleransi:.2f})\n")

# ==========================================
# 5. PENGUJIAN GAMBAR BARU
# ==========================================
while True:
    print("="*50)
    print("Contoh: KematanganBuahMangga/Test/Ripe/gambar1.jpg")
    gambar_tes = input("Masukkan path gambar yang ingin diuji (ketik 'exit' untuk keluar): ").strip('"').strip("'") 
    
    if gambar_tes.lower() == 'exit': break

    if os.path.exists(gambar_tes):
        fitur_tes = extract_features(gambar_tes)
        fitur_tes_scaled = scaler.transform([fitur_tes])
        
        jarak_tes, _ = model_knn.kneighbors(fitur_tes_scaled)
        jarak_rata_rata_tes = np.mean(jarak_tes[0])
        
        print("\n" + "-"*50)
        if jarak_rata_rata_tes > batas_toleransi:
            print(f">>> ⚠️ HASIL: BUKAN MANGGA! (Objek Ditolak)")
            print(f"(Alasan: Jarak fitur {jarak_rata_rata_tes:.2f} > batas {batas_toleransi:.2f})")
        else:
            prediksi = model_knn.predict(fitur_tes_scaled)
            print(f">>> ✅ HASIL DETEKSI: Mangga ini terdeteksi {prediksi[0].upper()} <<<")
        print("-"*50 + "\n")
    else:
        print("\n[ERROR] Gambar tidak ditemukan.\n")