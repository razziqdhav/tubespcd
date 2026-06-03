# 🥭 Sistem Deteksi Kematangan Mangga - Tugas Besar PCD

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

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

### 3. **Sistem Rule-Based Deterministic**
- Aturan bertingkat untuk akurasi tinggi
- Logika fuzzy terintegrasi untuk keputusan yang robust
- Confidence score untuk setiap prediksi

### 4. **GUI Interaktif**
- Interface modern dengan tema dark
- Upload gambar dari file explorer
- Real-time visualization dari proses pemrosesan
- Menampilkan hasil klasifikasi dengan confidence score
- Visualisasi 9 tahap pemrosesan citra

### 5. **Dataset Training-Testing**
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

## 🚀 Instalasi dan Setup

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. **Clone Repository**
```bash
git clone https://github.com/username/tubespcd.git
cd tubespcd
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Atau install secara manual:
```bash
pip install opencv-python numpy pillow matplotlib scikit-image
```

3. **Jalankan Aplikasi**
```bash
python main.py
```

---

## 💻 Penggunaan Aplikasi

### Langkah-Langkah Penggunaan

1. **Buka Aplikasi**: Jalankan `python main.py`
2. **Upload Gambar**: Klik tombol "📁 Pilih Gambar" untuk memilih gambar mangga
3. **Lihat Hasil**: Sistem secara otomatis akan:
   - Memproses gambar
   - Mengekstraksi fitur
   - Menampilkan klasifikasi (Unripe/Ripe/Overripe)
   - Menampilkan confidence score
4. **Analisis Visualisasi**: Lihat 9 tahap pemrosesan citra untuk debugging

### Contoh Output
```
Hasil Klasifikasi: RIPE
Confidence Score: 0.94
Warna (H,S,V): (12.3, 245.6, 198.4)
Edge Density: 0.156
Dark Area %: 2.3%
Jumlah Bintik: 0
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

## 📊 Performa Model

| Kategori | Deskripsi | Fitur Utama |
|----------|-----------|-----------|
| **Unripe** | Mangga masih mentah | HSV value rendah, edge density rendah, tanpa bintik |
| **Ripe** | Mangga siap panen | HSV balance optimal, edge density sedang, minimal bintik |
| **Overripe** | Mangga terlalu matang | HSV value tinggi, banyak bintik gelap, edge density tinggi |

---

## 🛠️ Teknologi yang Digunakan

- **OpenCV**: Computer Vision processing
- **NumPy**: Numerical computation
- **Tkinter**: GUI framework
- **Pillow (PIL)**: Image manipulation
- **Matplotlib**: Data visualization
- **scikit-image**: Image processing algorithms

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

## 🎯 Fitur Advanced

### Logging & Analysis
- Sistem mencatat semua fitur yang diekstraksi
- Debugging visualization untuk 9 tahap pemrosesan

### Extensibility
- Mudah untuk menambah fitur baru
- Rule-based system dapat di-customize
- GUI dapat dikembangkan dengan fitur tambahan

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

---

## 📝 Struktur File main.py

```
main.py
├── extract_features()          # Ekstraksi 12 fitur dari gambar
├── process_and_visualize()     # Proses citra & buat visualisasi 9 tahap
└── FruitDetectorApp (Class)    # GUI Application utama
    ├── __init__()              # Inisialisasi aplikasi
    ├── setup_ui()              # Setup interface
    ├── upload_image()          # Upload gambar
    ├── classify()              # Klasifikasi hasil
    └── display_results()       # Tampilkan hasil & visualisasi
```

---

## 🔍 Penelitian Lanjutan

Proyek ini dapat dikembangkan lebih lanjut dengan:
- **Machine Learning**: Implementasi SVM, Random Forest, atau Deep Learning
- **Real-time Detection**: Integrasi dengan kamera untuk deteksi real-time
- **Mobile App**: Port ke aplikasi mobile (Flutter/React Native)
- **Augmented Reality**: AR visualization untuk quality control di lapangan

---

## 👨‍💻 Developer

**Nama Proyek**: Tugas Besar Pengolahan Citra Digital (PCD)
**Tahun**: 2024-2025
**Status**: Complete

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** - lihat file LICENSE untuk detail.

---

## 📧 Kontak & Dukungan

Untuk pertanyaan atau saran, silakan buat issue di repository ini atau hubungi developer melalui GitHub.

---

## 🙏 Acknowledgments

- OpenCV community untuk dokumentasi lengkap
- Dataset dari penelitian kematangan buah
- Terinspirasi dari penelitian computer vision untuk agriculture

---

**Happy Mangga Detecting! 🥭✨**