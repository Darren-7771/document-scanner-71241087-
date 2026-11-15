# Smart Document Scanner — Tugas Final CVIA

Aplikasi desktop pemindai dokumen otomatis berbasis Python dan OpenCV tanpa GPU maupun model *deep learning*.  
Disusun oleh **Darren Malvino Gunawan (71241087)** — Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Kristen Duta Wacana (UKDW) Yogyakarta.

## Fitur Utama

- **Preprocessing:** Resize ke tinggi 500 piksel (menjaga rasio aspek), konversi *grayscale*, *Morphological Closing* `15x15`, dan *Gaussian Blur* `5x5`.
- **Deteksi Tepi & 4 Sudut:** *Canny Edge Detection* (`60`, `160`), dilasi `3x3`, serta pencarian kontur gabungan (*Canny* + *Otsu mask*) dengan *Convex Hull* dan aproksimasi poligon *Douglas-Peucker*.
- **Pelurusan Perspektif:** Transformasi homografi `3x3` (`cv2.getPerspectiveTransform` dan `cv2.warpPerspective`).
- **Binarisasi & Ekspor PDF:** *Adaptive Gaussian Thresholding* (`block=21`, `C=10`) serta pembanding *Global Otsu*, lalu ekspor hasil akhir ke PDF `150 DPI`.
- **Antarmuka Desktop Tkinter:** Tampilan 2 panel untuk melihat foto asli, pratinjau tiap tahap (termasuk ringkasan 6 tahap), serta menyimpan hasil gambar dan PDF ke `output_tahapan/`.

## Struktur Folder

```text
Tugas Final CVIA/
├── scanner_core.py          # Mesin pemrosesan citra (5 tahap) & ekspor PDF
├── gui_scanner.py           # Antarmuka desktop Tkinter + Pillow
├── requirements.txt         # Dependensi Python
├── .gitignore
├── input_dokumen/           # Dataset uji utama (asli, miring 15° & 30°)
└── input tambahan/          # Foto uji tambahan tulisan tangan
```

## Cara Menjalankan

1. **Instal dependensi** (disarankan menggunakan Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan aplikasi GUI**:

   ```bash
   python gui_scanner.py
   ```

3. Klik tombol **Ambil Foto Dokumen**, pilih foto dari folder `input_dokumen/` atau `input tambahan/`, lalu jelajahi mode tampilan (`1`–`7`) atau klik **Simpan PDF & Semua Tahap**.
