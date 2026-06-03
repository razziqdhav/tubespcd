# 🥭 Sistem Deteksi Kematangan Mangga - Tugas Besar PCD

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-blue.svg)

## 👨‍💻 Kelompok D5

1. 152023091 Razziq Dhavino Rafadhillah  
2. 152024134 Az-Zahra Agustina  
3. 152024188 Putri Amelina Rahmawati  
4. 152024189 Devarasya Rizky Amelia Putri  
5. 152024192 Mila Siti Nabila  

**Tugas Besar Pengolahan Citra Digital (PCD)**  
**Tahun**: 2025/2

---


## 📋 Deskripsi Proyek

**Sistem Deteksi Kematangan Mangga** adalah aplikasi berbasis komputer vision yang menggunakan teknik pemrosesan citra digital untuk mengklasifikasikan tingkat kematangan buah mangga secara otomatis. Sistem ini menggunakan pendekatan **rule-based deterministic** dengan ekstraksi fitur berbasis warna (HSV), tekstur, dan deteksi cacat buah.

Aplikasi ini dilengkapi dengan **Graphical User Interface (GUI)** yang memudahkan pengguna untuk menganalisis gambar mangga dan mendapatkan hasil klasifikasi beserta visualisasi proses pemrosesan citra.

### Kategori Klasifikasi
- **Unripe (Mentah)**: Mangga dengan warna hijau yang solid, tekstur halus, tanpa cacat
- **Ripe (Matang)**: Mangga dengan warna kuning/merah yang seimbang, tingkat kematangan optimal
- **Overripe (Terlalu Matang)**: Mangga dengan warna gelap, bintik hitam, kemungkinan busuk

---

## ✨ Fitur Utama

### 1. **Deteksi dan Segmentasi Citra**
- Segmentasi menggunakan Otsu Thresholding
- Deteksi tepi menggunakan algoritma Canny
- Penghapusan noise dan artefak tepi

### 2. **Ekstraksi Fitur Komprehensif**
- **Fitur Warna (9 features)**: Mean, Std, Median dari H, S, V channels
- **Fitur Tekstur (1 feature)**: Edge density menggunakan Canny Edge Detection
- **Fitur Cacat (2 features)**: Persentase area gelap dan jumlah bintik hitam terdeteksi
- **Total: 12 fitur** untuk klasifikasi

### 3. **GUI Interaktif**
- Interface modern dengan tema dark
- Upload gambar dari file explorer
- Visualisasi 9 tahap pemrosesan citra

### 4. **Dataset Training-Testing**
- Struktur folder terorganisir untuk Train/Test sets
- 3 kategori: Unripe, Ripe, Overripe
- Dataset siap untuk machine learning atau rule-based validation

---

## 📁 Struktur Proyek

```
tubespcd/
├── main.py                          # Main application & GUI
├── README.md                        # Dokumentasi proyek
└── KematanganBuahMangga/            # Dataset
    ├── Train/                       # Training set
    │   ├── Unripe/                  # Gambar mangga mentah
    │   ├── Ripe/                    # Gambar mangga matang
    │   └── Overripe/                # Gambar mangga terlalu matang
    └── Test/                        # Testing set
        ├── Unripe/
        ├── Ripe/
        └── Overripe/
```
---

## 🔬 Metodologi Pemrosesan Citra

### 1. Pre-processing
- Resize gambar ke 100x100 piksel untuk konsistensi
- Konversi BGR ke Grayscale untuk analisis tekstur
- Gaussian Blur dengan kernel 5x5 untuk mengurangi noise

### 2. Segmentasi
- **Otsu Thresholding**: Automatic threshold untuk memisahkan mangga dari background
- **Morphological Operations**: Eroding untuk menghilangkan artifact tepi

### 3. Ekstraksi Fitur
- **Warna**: Analisis HSV channels untuk menentukan tingkat kematangan
- **Tekstur**: Edge Detection menggunakan Canny untuk mengukur roughness
- **Cacat**: Deteksi dark spots dengan morphological operations dan contour analysis

### 4. Klasifikasi
- **Rule-Based Decision Tree**: Kombinasi logis dari fitur untuk keputusan akhir
- **Confidence Scoring**: Mengukur seberapa yakin sistem pada prediksi

---

## 📚 Dataset Information

### Struktur Dataset
- **Total Sampel**: Train + Test sets dengan 3 kategori
- **Format File**: JPG, PNG
- **Ukuran Gambar**: Bervariasi (di-resize ke 100x100 dalam preprocessing)
- **Organizer**: Folder terpisah per kategori untuk kemudahan labeling

### Cara Menambah Dataset
1. Tempatkan gambar baru di folder kategori yang sesuai
2. Nama file: `mangga_[kategori]_[nomor].[ext]`
3. Contoh: `mangga_ripe_001.jpg`

---

## 🐛 Troubleshooting

### Error: "Image not found"
- Pastikan file gambar ada di path yang benar
- Format gambar harus JPG atau PNG

### Error: "No valid features extracted"
- Gambar mungkin terlalu gelap atau terang
- Coba dengan gambar mangga yang lebih jelas

### GUI tidak responsive
- Pastikan semua dependencies terinstall dengan benar
- Coba update OpenCV: `pip install --upgrade opencv-python`