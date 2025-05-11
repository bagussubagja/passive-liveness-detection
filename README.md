# Passive Liveness Detection

![Liveness Detection](https://img.shields.io/badge/AI-Liveness%20Detection-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

Implementasi sederhana dari sistem **Passive Liveness Detection** menggunakan Python dan TensorFlow. Sistem ini berfungsi mendeteksi apakah wajah pada sebuah gambar adalah **asli (real)** atau **palsu (fake)** berdasarkan pelatihan model CNN.

## 📦 Fitur

- Melatih model klasifikasi gambar dengan TensorFlow
- Deteksi dua kelas: `real` dan `fake`
- Menggunakan dataset dari Kaggle
- Output berupa model `.h5` dan `.tflite`
- Fungsi prediksi liveness untuk satu gambar

## 📁 Struktur Dataset

Dataset yang digunakan berasal dari Kaggle:

**Dataset**: [Liveness Detection Dataset by anhnguynthch](https://www.kaggle.com/datasets/anhnguynthch/liveness-detection-dataset)

Ekstrak dataset dan atur strukturnya seperti berikut:

```
.
├── dataset/
│   ├── fake/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── real/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
├── test_images/
│   ├── sample1.jpg
│   └── sample2.jpg
├── face-predict.py
├── generate-liveness-model.py
├── liveness_model.h5      # ← hasil training
└── liveness_model.tflite  # ← model untuk mobile
```

## 🛠 Instalasi & Setup

### 1. Buat Virtual Environment

```bash
python3 -m venv venv310
source venv310/bin/activate
pip install tensorflow pillow numpy
```

## 🚀 Menjalankan Training

**File**: `generate-liveness-model.py`

Jalankan script berikut untuk:
- Melatih model CNN sederhana
- Menyimpan model sebagai `.h5`
- Mengonversi ke `.tflite`

```bash
python3 generate-liveness-model.py
```

## 🔍 Prediksi Gambar Tunggal

**File**: `face-predict.py`

Gunakan untuk memprediksi apakah wajah dalam gambar adalah asli atau palsu:

```bash
python3 face-predict.py path/to/image.jpg
```

**Contoh output**:
```
Confidence: 0.9771, Decision: fake
```

## 🧠 Arsitektur Model CNN

Model sederhana menggunakan struktur berikut:

1. Conv2D → ReLU → MaxPooling
2. Conv2D → ReLU → MaxPooling
3. Flatten → Dense → Dropout → Output Sigmoid