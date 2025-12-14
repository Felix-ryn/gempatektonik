import pandas as pd
import logging
import os

def load_data(path: str) -> pd.DataFrame | None:
    """
    Memuat data gempa, membersihkan nama kolom, memvalidasi tipe data,
    dan mengurutkan berdasarkan waktu (Penting untuk analisis Time-Series/LSTM).
    """
    logger = logging.getLogger(__name__)

    # 1. Cek apakah file ada
    if not os.path.exists(path):
        logger.critical(f"FILE TIDAK DITEMUKAN: '{path}'. Cek konfigurasi path di config.py")
        return None

    try:
        # 2. Deteksi Format dan Load Data
        _, extension = os.path.splitext(path)
        ext = extension.lower()

        if ext == '.xlsx':
            df = pd.read_excel(path, engine='openpyxl')
        elif ext == '.csv':
            # Coba membaca dengan pemisah desimal: '.' dan jika gagal, coba dengan ','
            try:
                 df = pd.read_csv(path, encoding='utf-8', decimal='.')
            except:
                 df = pd.read_csv(path, encoding='utf-8', decimal=',')

        # 3. Standardisasi Nama Kolom
        # Memastikan nama kolom sesuai dengan variabel yang dipakai di Engine (ACO/GA/CNN)
        rename_map = {
            'Tanggal': 'Tanggal', 
            'Date time': 'Tanggal',           # Tambahkan alias
            'Lintang': 'Lintang',
            'Bujur': 'Bujur',
            'Magnitudo': 'Magnitudo',
            'Kedalaman (km)': 'Kedalaman_km',
            'Depth (Km)': 'Kedalaman_km',      # Tambahkan alias
            'Lokasi (Jatim)': 'Lokasi',
            'Location (Jatim)': 'Lokasi',      # Tambahkan alias
            'Kedalaman': 'Kedalaman_km',
        }
        df.rename(columns=rename_map, inplace=True)

        # 4. Validasi Kolom Wajib
        # Jika kolom ini tidak ada, Engine pasti error. Fail-fast di sini.
        required_cols = ['Tanggal', 'Lintang', 'Bujur', 'Magnitudo', 'Kedalaman_km']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.critical(f"STRUKTUR DATA SALAH. Kolom hilang: {missing}")
            return None

        # 5. Konversi Tipe Data & Cleaning
        # Tanggal ke Datetime
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        
        # Numerik (Hapus baris jika ada huruf di kolom angka)
        numeric_cols = ['Lintang', 'Bujur', 'Magnitudo', 'Kedalaman_km']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Pembersihan Teks Lokasi (Agar rapi di Peta Dashboard)
        if 'Lokasi' in df.columns:
            df['Lokasi'] = df['Lokasi'].astype(str).str.title().str.strip()
        else:
            df['Lokasi'] = "Lokasi Tidak Diketahui"

        # 6. Hapus Data Rusak (NaN)
        initial_count = len(df)
        df.dropna(subset=required_cols, inplace=True)
        final_count = len(df)

        if final_count < initial_count:
            logger.warning(f"Dihapus {initial_count - final_count} baris data yang rusak/kosong.")

        # 7. Pengurutan Waktu (SANGAT PENTING untuk LSTM)
        # Data harus urut dari masa lalu ke masa depan agar LSTM bisa memprediksi tren
        df = df.sort_values(by='Tanggal', ascending=True).reset_index(drop=True)

        logger.info(f"Data berhasil dimuat & dibersihkan. Total baris valid: {len(df)}")
        return df

    except Exception as e:
        logger.critical(f"Error tak terduga saat memuat data: {e}")
        return None