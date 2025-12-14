"""
TectonicAI - Realtime Pipeline Orchestrator
Arsitektur:
1. Load data BMKG (historical + realtime)
2. (Opsional) Jalankan ACO & GA untuk update konteks spasial/vektor
3. Jalankan LSTM Hybrid untuk analisis temporal + anomaly detection
"""

import logging
import os
import pandas as pd
import time
from datetime import datetime, timedelta
# Perbaikan Impor: Tambah Tuple untuk type hinting
from typing import Tuple 
# Import critical dari library
from sklearn.model_selection import train_test_split 

# Import critical dari modul internal TectonicAI
from TectonicAI.config import CONFIG
from TectonicAI.processing.data_handler import load_data
from TectonicAI.features.feature_engineer import FeatureEngineer
from TectonicAI.services.inaTEWS_api_client import InaTEWSApiClient
from TectonicAI.engines.aco_engine import ACOEngine
from TectonicAI.engines.ga_engine import GAEngine
from TectonicAI.engines.lstm_engine import LSTMEngine
from TectonicAI.engines.evaluation_engine import EvaluationEngine

# -----------------------------------------------------------
# SETUP LOGGER GLOBAL
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TectonicAI_Realtime_Pipeline")


# ==========================================================
# 1. HELPER FUNCTIONS (Perbaikan Scope: Diperlukan oleh run_cycle)
# ==========================================================

# ----------------------------------------------------------
# INJECTION LOADER
# ----------------------------------------------------------
def load_injection_file(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        # Menangani path yang mungkin tidak ada di CONFIG
        df = pd.read_excel(path) 
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

        cutoff = datetime.now() - timedelta(days=90)
        df = df[df["Tanggal"] >= cutoff]
        logger.info(f"Injection loaded: {len(df)} rows")
        return df
    except:
        return pd.DataFrame()


# ----------------------------------------------------------
# ARCHIVE >90 DAYS
# ----------------------------------------------------------
def archive_old_data(df, base_dir):
    cutoff = datetime.now() - timedelta(days=90)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    old = df[df["Tanggal"] < cutoff]
    recent = df[df["Tanggal"] >= cutoff].copy()

    if not old.empty:
        archive_file = os.path.join(base_dir, "archive/tectonic_history_archive.csv")
        os.makedirs(os.path.dirname(archive_file), exist_ok=True)
        old.to_csv(archive_file, mode="a", index=False, header=not os.path.exists(archive_file))
        logger.info(f"Archived {len(old)} rows")

    return recent.reset_index(drop=True)


# ----------------------------------------------------------
# DATASET UPDATE (Mengganti fungsi update_dataset yang hilang)
# ----------------------------------------------------------
def update_dataset(df_old, save_path, injection_path):
    try:
        sensor = InaTEWSApiClient() # Memerlukan import InaTEWSApiClient
        df_new = sensor.fetch_recent_data()
    except:
        df_new = pd.DataFrame()

    df_inj = load_injection_file(injection_path)

    for df_temp in [df_old, df_new, df_inj]: 
        if df_temp is not None and "Tanggal" in df_temp.columns:
            df_temp["Tanggal"] = pd.to_datetime(df_temp["Tanggal"], errors="coerce")

    df = pd.concat([d for d in [df_old, df_new, df_inj] if d is not None], ignore_index=True)
    df.drop_duplicates(subset=["Tanggal", "Lintang", "Bujur"], inplace=True)
    df.sort_values("Tanggal", inplace=True)

    df = archive_old_data(df, os.path.dirname(save_path))

    # Save logic yang stabil
    if save_path.endswith(".xlsx"):
        df.to_excel(save_path, index=False, engine='openpyxl') # Tambah engine untuk stabilitas
    else:
        df.to_csv(save_path, index=False)
        
    return df


# ----------------------------------------------------------
# TEXT REPORT (Mengganti fungsi generate_text_report yang hilang)
# ----------------------------------------------------------
def generate_text_report(row, config):
    # Asumsi row adalah baris terakhir (df.iloc[-1])
    path = config["output_paths"]["latest_report_txt"]

    content = f"""
========================= TECTONIC AI REPORT =========================
Waktu       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Lokasi      : {row.get('Lokasi','-')}
Magnitude   : {row.get('Magnitudo',0)}
Kedalaman   : {row.get('Kedalaman_km',0)} km

Zonasi ACO  : {row.get('Status_Zona','UNKNOWN')}
LSTM        : {'ANOMALI' if row.get('is_Anomaly') else 'NORMAL'}
CNN Impact  : {row.get('Proporsi_Grid_Terdampak_CNN',0)*100:.2f} %
Final Prob  : {row.get('Final_Prob_Impact',0)*100:.1f} %
======================================================================
"""

    with open(path, "w") as f:
        f.write(content)


# -----------------------------------------------------------
# HELPER: SPLIT TRAIN / FULL (UNTUK LSTM) - Kode Asli
# -----------------------------------------------------------
def split_train_from_full(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ... (Kode Asli) ...
    if len(df) < 50:
        logger.warning("Data sangat sedikit, seluruh data akan dipakai sebagai train & full.")
        return df.copy(), df.copy()

    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train].copy()
    full_df = df.copy()
    return train_df, full_df


# ----------------------------------------------------------
# 5. FULL ORCHESTRATOR (1 cycle only)
# ----------------------------------------------------------
def run_cycle():
    # ... (inisialisasi CONFIG, logger, summary) ...
    input_path = CONFIG["paths"]["input_data"]
    injection_path = CONFIG["paths"]["injection_file"]
    split_cfg = CONFIG["data_split"]
    summary = {}

    # 1. LOAD DATA DASAR (Historis Awal)
    df_raw = load_data(input_path) 
    if df_raw is None:
        logger.error("Database statis tidak ditemukan!")
        return None
    
    # 2. UPDATE DATASET PENUH (Historis + Realtime)
    df_full = update_dataset(df_raw, input_path, injection_path)

    # 3. FEATURE ENGINEERING PADA DATA PENUH (Diperlukan untuk LSTM/CNN)
    fe = FeatureEngineer(CONFIG)
    df_full_fe = fe.transform(df_full)

    # 4. ISOLASI DATA STATIS/HISTORIS untuk ACO/GA
    # Kita asumsikan data Statis adalah SEMUA data KECUALI data yang masuk di siklus ini.
    # Karena kita sudah menjalankan update_dataset, data Realtime sudah ada di df_full_fe.
    
    # Pendekatan Aman: Pisahkan Train (Historis) dan Test (Realtime) berdasarkan Index
    # Asumsi: Train Set adalah data yang sudah ada sebelum data terbaru masuk.
    
    # Dapatkan jumlah data awal (historis)
    n_static = len(df_raw) 
    
    # Tentukan train/static set dan test/realtime set
    if n_static < len(df_full):
        # Jika ada data baru yang masuk (realtime)
        df_static_aco_ga = df_full_fe.iloc[:n_static].copy()
        df_realtime_test = df_full_fe.iloc[n_static:].copy()
        
        # Jika data realtime terlalu kecil, gunakan baris terakhir dari df_full
        if df_realtime_test.empty:
            df_realtime_test = df_full_fe.iloc[-1:].copy()
        
    else:
        # Jika tidak ada data baru (data lama sudah dibersihkan)
        df_static_aco_ga = df_full_fe.copy()
        df_realtime_test = pd.DataFrame() # Test set kosong

    # --- JALUR STATIS (ACO & GA) ---
    # ACO dan GA hanya memproses data historis/statis
    
    # 5. ACO (PROSES HANYA PADA DATA STATIS)
    try:
        aco = ACOEngine(CONFIG["aco_path_model"])
        # ACO berjalan HANYA pada data Statis/Historis
        df_static_aco_ga, aco_meta = aco.run(df_static_aco_ga) 
        summary['aco_results'] = aco_meta
    except Exception as e:
        logger.error(f"ACO Engine GAGAL: {e}")
        summary['aco_results'] = {}

    # 6. GA (PROSES HANYA PADA DATA STATIS)
    try:
        ga = GAEngine(CONFIG["ga_vector_model"])
        # GA berjalan HANYA pada data Statis/Historis
        df_static_aco_ga, ga_meta = ga.run(df_static_aco_ga)
        summary['ga_results'] = ga_meta
    except Exception as e:
        logger.error(f"GA Engine GAGAL: {e}")
        summary['ga_results'] = {}

    # 7. RE-FUSION KONTEKS STATIS (ACO/GA) UNTUK LSTM/CNN
    # Hasil ACO/GA (yaitu kolom Pheromone_Score, Status_Zona, dll.)
    # hanya ada di df_static_aco_ga. Kita harus menyatukan kembali hasilnya ke df_full_fe.
    
    # Dapatkan kolom hasil ACO/GA yang dibutuhkan LSTM/CNN (Context Features)
    context_cols = [c for c in df_static_aco_ga.columns if 'Pheromone' in c or 'Status_Zona' in c or 'Radius_Visual' in c]
    
    # Gabungkan kolom konteks ke DF utama (df_full_fe)
    for col in context_cols:
        if col in df_static_aco_ga.columns:
            # Transfer kolom konteks dari Statis ke Full DF
            df_full_fe[col] = df_full_fe.index.map(df_static_aco_ga[col].to_dict()).fillna(0)


    # --- JALUR REALTIME (LSTM, CNN, EVALUATION) ---
    # LSTM/CNN menganalisis df_full_fe, menggunakan df_static_aco_ga sebagai Train Set
    
    train_idx = df_static_aco_ga.index
    test_idx = df_full_fe.index[len(df_static_aco_ga):]

    # 8. LSTM (REALTIME ANALYSIS)
    try:
        from TectonicAI.engines.lstm_engine import LSTMEngine
        lstm = LSTMEngine(CONFIG["lstm_anomaly_model"])
        # LSTM berjalan di DF PENUH (Hybrid), tetapi HANYA training pada data Statis
        df_full_fe, lstm_meta = lstm.run(df_full_fe, df_static_aco_ga) 
        summary['lstm_results'] = lstm_meta
    except Exception as e:
        logger.error(f"LSTM Engine GAGAL: {e}")
        df_full_fe["Temporal_Risk_Factor"] = 0.0
        summary['lstm_results'] = {}

    # 9. CNN (REALTIME ANALYSIS)
    try:
        from TectonicAI.engines.cnn_engine import CNNEngine
        cnn = CNNEngine(CONFIG["cnn_hybrid_model"])
        # CNN berjalan di DF PENUH (Hybrid)
        df_full_fe = cnn.train_and_predict(df_full_fe, train_idx, test_idx)
    except Exception as e:
        logger.error(f"CNN Engine GAGAL: {e}")
        df_full_fe["Proporsi_Grid_Terdampak_CNN"] = 0.0

    # 10. EVALUATION & REPORTING (FIX CRITICAL)
    evaluator = EvaluationEngine(CONFIG)
    df_full_fe = evaluator.run(df_full_fe, train_idx, test_idx)
    summary['eval_results'] = {} 
    
    if not df_full_fe.empty:
        generate_text_report(df_full_fe.iloc[-1], CONFIG)
    
    return df_full_fe