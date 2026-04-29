# 🌋 Earthquake Tectonic: Hybrid Intelligence for Seismic Analysis

**Earthquake Tectonic** adalah platform analisis seismik tingkat lanjut yang menggabungkan kekuatan *Deep Learning* dan *Metaheuristic Optimization* untuk memetakan, menganalisis, dan memprediksi pola gempa tektonik di Indonesia. Proyek ini tidak hanya melakukan klasterisasi, tetapi juga pemodelan prediktif untuk estimasi risiko gempa di masa depan.

---

## 🚀 Fitur Unggulan

- **Hybrid Prediction Models**: Menggunakan arsitektur **CNN** untuk pengenalan pola spasial dan **LSTM** untuk analisis deret waktu (time-series) guna mendeteksi anomali.
- **Metaheuristic Optimization**: Implementasi **Ant Colony Optimization (ACO)** untuk analisis zonasi dan **Genetic Algorithm (GA)** untuk optimasi parameter prediksi.
- **Advanced Mapping**: Visualisasi vektor arah pergerakan lempeng dan peta zonasi dampak melalui output HTML interaktif.
- **Comprehensive Evaluation**: Sistem penilaian performa model yang ketat menggunakan metrik sistem yang komprehensif (`system_metrics.json`).
- **Multi-Platform Dashboard**: Visualisasi data yang kaya menggunakan Python-based dashboard dan integrasi framework Tectonic (C#/.NET).

---

## 🛠️ Tech Stack & Architecture

### Intelligence Core
- **Deep Learning**: CNN (Convolutional Neural Networks), LSTM (Long Short-Term Memory), & Hybrid Transformers.
- **Optimization**: Genetic Algorithm (GA) & Ant Colony Optimization (ACO).
- **Processing**: Python (Scikit-learn, Keras, TensorFlow, Pandas).

### Frontend & Visualization
- **Python Dashboard**: Menggunakan komponen kustom untuk plot arah dan peta mitigasi.
- **Tectonic UI**: Pengembangan antarmuka berbasis C# (XAML) untuk performa desktop yang stabil.

---

## 🗂️ Struktur Repositori

Berdasarkan arsitektur sistem, proyek dibagi menjadi beberapa modul utama:

```text
├── dashboard/              # Antarmuka visual berbasis Python
│   ├── components/         # Plot arah, peta, dan summary cards
│   └── pages/              # Analisis ACO, GA, CNN, LSTM, & Evaluation
├── data/                   # Dataset historis & live history (2025-2026)
├── output/                 # Hasil pemrosesan model
│   ├── aco_results/        # Brain state, epicenters, & impact zones
│   ├── cnn_results/        # Model prediksi & training logs
│   ├── ga_results/         # Best chromosome & vector maps
│   └── lstm_results/       # Direction deviation & hybrid transformers
└── Tectonic/               # Modul aplikasi berbasis C#/.NET (XAML)
```
📊 Alur Kerja Model
Preprocessing: Data mentah dari data/ diolah dan disimpan dalam datacache.

Optimization (ACO & GA): Algoritma mencari jalur epicenter paling kritis dan mengoptimalkan parameter fitur.

Training (CNN & LSTM): Model Deep Learning dilatih untuk mengenali pola tektonik yang kompleks.

Output Generation: Hasil berupa file .keras, .pld, dan laporan .txt di simpan di folder output.

Visualization: Semua metrik dan peta interaktif disajikan melalui Dashboard.

⚙️ Instalasi
Prasyarat
Python 3.9+

.NET SDK (untuk modul Tectonic)

Langkah-langkah
Clone Repository:

```
git clone [https://github.com/Felix-ryn/gempatektonik.git](https://github.com/Felix-ryn/gempatektonik.git)
```
Install Python Dependencies:
```
pip install -r requirements.txt
```
Run Dashboard:
```
python dashboard/app.py
```
