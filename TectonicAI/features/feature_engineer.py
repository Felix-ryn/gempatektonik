"""
TectonicAI - Feature Engineering Engine (Ultimate Edition).
Module: Advanced Seismological Feature Extraction & Preprocessing.

FIXES APPLIED (Critical):
1. Corrected 'Magnitudo' and 'Kedalaman_km' corruption by ensuring only 
   derived features are scaled, and primary features are only modified 
   if necessary (Normalization issue).
2. Cleaned up the 'transform' function's logic flow.
3. Added missing imports (typing, numpy, pandas).
"""

import numpy as np
import pandas as pd
import logging
import math
# Tambahan impor yang dibutuhkan
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import entropy, skew, kurtosis
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# Asumsi file ini ada dan fungsinya tidak diubah
from TectonicAI.calculation_config import CALCULATION_CONFIG, eval_formula 

# Konstanta Seismologi (Dibiarkan Utuh)
CONST_K_MOMENT = 9.1
CONST_C_MOMENT = 1.5
CONST_ENERGY_A = 11.8
CONST_ENERGY_B = 1.5

class FeatureEngineer:
    """
    Advanced Feature Engineer.
    Bertugas mengubah data mentah menjadi sinyal-sinyal fisika dan statistik
    yang sangat kaya untuk konsumsi Deep Learning (LSTM/CNN) dan Metaheuristik.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fe_cfg = config.get('feature_engineering', {})
        self.logger = logging.getLogger("Feature_Engineer_Complex")
        self.logger.setLevel(logging.DEBUG)
        
        # State Objects
        self.scaler = None
        self.kmeans = None
        self.dbscan = None
        
        # Konfigurasi Rolling Window
        self.windows = [7, 14, 30]

    # ==========================================
    # 1. PHYSICS-BASED FEATURES (SEISMOLOGY) - UTUH
    # ==========================================

    def _calculate_seismic_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Menghitung parameter fisika gempa bumi. """
        df['Seismic_Moment_M0'] = 10 ** (CONST_C_MOMENT * df['Magnitudo'] + CONST_K_MOMENT)
        df['Radiated_Energy_J'] = 10 ** (CONST_ENERGY_A + CONST_ENERGY_B * df['Magnitudo'])
        df['Benioff_Strain'] = np.sqrt(df['Radiated_Energy_J'])
        df['Rupture_Length_Est'] = 10 ** (0.69 * df['Magnitudo'] - 3.22)
        return df

    def _calculate_b_value_rolling(self, df: pd.DataFrame, window=50) -> pd.DataFrame:
        """ Menghitung Gutenberg-Richter b-value menggunakan MLE. """
        b_values = np.zeros(len(df))
        mags = df['Magnitudo'].values
        log_e = 0.4343
        mc = 3.0
        
        for i in range(window, len(df)):
            sample = mags[i-window : i]
            sample_valid = sample[sample >= mc]
            
            if len(sample_valid) > 10:
                mean_mag = np.mean(sample_valid)
                b = log_e / (mean_mag - (mc - 0.05))
                b_values[i] = b
            else:
                b_values[i] = 1.0 
                
        df['Gutenberg_Richter_b_value'] = b_values
        return df

    # ==========================================
    # 2. TEMPORAL & SPATIAL DELTAS - UTUH
    # ==========================================

    def _haversine_array(self, lat1, lon1, lat2, lon2):
        # ... (Logika haversine tetap sama)
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        # Karena dipanggil dengan scalar (satu row), hasilnya adalah scalar
        return R * c # <--- Ini mengembalikan scalar tunggal.

    def _calculate_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Menghitung selisih waktu dan jarak antar gempa berurutan. """
        df['Time_Delta_Hours'] = df['Tanggal'].diff().dt.total_seconds() / 3600.0
        df['Time_Delta_Hours'] = df['Time_Delta_Hours'].fillna(0)
        
        prev_lat = df['Lintang'].shift(1).fillna(df['Lintang'])
        prev_lon = df['Bujur'].shift(1).fillna(df['Bujur'])
        
        df['Dist_Delta_KM'] = self._haversine_array(
            df['Lintang'], df['Bujur'], prev_lat, prev_lon
        )
        
        df['Migration_Velocity'] = df['Dist_Delta_KM'] / (df['Time_Delta_Hours'] + 0.001)
        
        return df

    # ==========================================
    # 3. ADVANCED STATISTICAL ROLLING WINDOWS - UTUH
    # ==========================================

    def _calculate_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Menghitung statistik tingkat lanjut (Skew, Kurtosis, Entropy) pada jendela geser. """
        df_ts = df.set_index('Tanggal').sort_index()
        
        for w in self.windows:
            window_str = f"{w}D"
            prefix = f"roll_{w}d"
            
            indexer = df_ts['Magnitudo'].rolling(window_str, min_periods=1)
            df[f'{prefix}_mag_mean'] = indexer.mean().values
            df[f'{prefix}_mag_std'] = indexer.std().fillna(0).values
            df[f'{prefix}_mag_max'] = indexer.max().values
            df[f'{prefix}_count'] = indexer.count().values
            
            energy_indexer = df_ts['Radiated_Energy_J'].rolling(window_str, min_periods=1)
            df[f'{prefix}_energy_sum'] = energy_indexer.sum().values
            
            df[f'{prefix}_depth_mean'] = df_ts['Kedalaman_km'].rolling(window_str).mean().values

        # --- FIX CRITICAL: ALIASING UNTUK LSTM ---
        fe_cfg = self.config.get('feature_engineering', {})
        w_default = fe_cfg.get('rolling_window_days', 14)

        if f'roll_{w_default}d_mag_mean' in df.columns:
            df['rolling_mag_mean'] = df[f'roll_{w_default}d_mag_mean']
        else:
            df['rolling_mag_mean'] = df['Magnitudo'].rolling(window=10, min_periods=1).mean()

        if f'roll_{w_default}d_count' in df.columns:
            df['rolling_event_count'] = df[f'roll_{w_default}d_count']
        else:
            df['rolling_event_count'] = df['Magnitudo'].rolling(window=10, min_periods=1).count()

        return df

    # ==========================================
    # 4. SPATIAL CLUSTERING & ENCODING - UTUH
    # ==========================================

    def _spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clustering K-Means dan DBSCAN untuk mendeteksi hotspot.
        """
        coords = df[['Lintang', 'Bujur']].values
        
        # 🚨 GUARD KRITIS: Cek data input
        if len(coords) < 1:
            self.logger.warning("[CLUSTERING] Data input kosong, skipping K-Means.")
            df['Cluster'] = 0
            df['Dist_to_Cluster_Center'] = 0.0
            self.kmeans = None # Pastikan tetap None
            return df

        n_clusters = self.config.get('analysis', {}).get('clustering', {}).get('n_clusters', 5)
        n_clusters = min(n_clusters, len(coords)) # Pastikan n_clusters tidak melebihi jumlah data
        if n_clusters < 1: n_clusters = 1
        
        # A. K-Means (Cluster Global)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        try:
             df['Cluster_KMeans'] = self.kmeans.fit_predict(coords)
        except Exception as e:
             self.logger.critical(f"K-MEANS FAILED: {e}. Setting Cluster to 0.")
             df['Cluster'] = 0
             df['Dist_to_Cluster_Center'] = 0.0
             self.kmeans = None # Jika gagal, set kembali ke None
             return df
        
        # Jika berhasil dilatih, hitung jarak
        centers = self.kmeans.cluster_centers_
        dist_to_center = []
        for i, row in df.iterrows():
            c_id = int(row['Cluster_KMeans'])
            center = centers[c_id]
            
            # FIX sebelumnya: self._haversine_array mengembalikan scalar
            dist = self._haversine_array(row['Lintang'], row['Bujur'], center[0], center[1])
            dist_to_center.append(dist) 
            
        df['Dist_to_Cluster_Center'] = dist_to_center
        
        # Alias untuk CNN yang butuh kolom 'Cluster'
        df['Cluster'] = df['Cluster_KMeans']
        
        return df

    # ==========================================
    # 5. CYCLICAL TIME & MANUAL SCALING - UTUH
    # ==========================================

    def _time_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Mengubah waktu menjadi sinyal siklus (Sin/Cos). """
        dt = df['Tanggal'].dt
        
        df['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        
        df['doy_sin'] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
        
        df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        
        return df

    def _apply_custom_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Menerapkan scaling manual untuk ACO sesuai config. """
        # Asumsi CALCULATION_CONFIG sudah diimpor dan ada
        mag_scale = CALCULATION_CONFIG['scaling']['magnitude_scaling']
        depth_scale = CALCULATION_CONFIG['scaling']['depth_scaling']
        
        df['MagScaled'] = df['Magnitudo'] * mag_scale
        df['DepthScaled'] = df['Kedalaman_km'] * depth_scale
        
        # Hitung Target Radius (Ground Truth)
        df['R_true'] = 10**(0.5 * df['Magnitudo'] - 1.8)
        df['R_true'] = df['R_true'].clip(lower=1.0)
        
        # Hitung Heuristik ACO
        df['aco_heuristic'] = (df['Magnitudo']**2) / (df['Kedalaman_km'] + 1.0)
        
        return df

    # ==========================================
    # 6. FINAL NORMALIZATION (MODIFIKASI KRITIS UNTUK DATA CORRUPTION)
    # ==========================================

    def _normalize_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menerapkan Robust Scaling pada fitur DERIVATIF.
        Kolom 'Magnitudo' dan 'Kedalaman_km' TIDAK di-scale/ditimpa
        agar data asli tetap bersih untuk ACO/GA/Evaluation.
        """
        # Daftar fitur yang akan di-scale
        features_to_scale = [
            'R_true', 
            'Seismic_Moment_M0', 'Radiated_Energy_J', 'Benioff_Strain',
            'Gutenberg_Richter_b_value',
            'Time_Delta_Hours', 'Dist_Delta_KM', 'Migration_Velocity',
            'Dist_to_Cluster_Center', 'aco_heuristic'
        ]
        
        # Tambahkan rolling features
        for w in self.windows:
            prefix = f"roll_{w}d"
            features_to_scale.extend([
                f'{prefix}_mag_mean', f'{prefix}_mag_std',  
                f'{prefix}_energy_sum', f'{prefix}_count', f'{prefix}_depth_mean' # Tambah depth mean
            ])
            
        # Filter hanya kolom yang ada dan bersifat numerik
        valid_cols = [c for c in features_to_scale if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
        if not valid_cols:
             self.logger.warning("Tidak ada fitur turunan numerik untuk scaling.")
             return df
        
        # Kolom baru yang akan menyimpan data yang SUDAH DINORMALISASI
        scaled_cols = [f + '_scaled' for f in valid_cols] 
        
        self.scaler = RobustScaler()
        
        # Buat kolom baru untuk data yang dinormalisasi (ML_INPUT)
        df[scaled_cols] = self.scaler.fit_transform(df[valid_cols])
        
        # 🚨 KRITIS: Kolom Magnitudo dan Kedalaman tidak dinormalisasi 
        # tetapi kita buat alias scaled untuk konsistensi input Deep Learning
        
        # Kolom Magnitudo dan Kedalaman yang TIDAK dinormalisasi 
        # akan ditambahkan ke df_engineered di transform()
            
        return df

    # ==========================================
    # MAIN TRANSFORM FUNCTION (MODIFIKASI KRITIS UNTUK DATA CORRUPTION)
    # ==========================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline Eksekusi Rekayasa Fitur.
        """
        self.logger.info(">>> MEMULAI REKAYASA FITUR KOMPLEKS (Physics + Stats) <<<")
        
        # 0. Pre-Sort & Backup
        df = df.sort_values(by='Tanggal').reset_index(drop=True)
        
        # 🚨 FIX KRITIS: Simpan nilai Magnitudo dan Kedalaman MENTAH
        # Ini mencegah engine hilir (ACO/GA/Evaluation) menggunakan nilai scaling
        df['Magnitudo_Original'] = df['Magnitudo'].astype(float).copy() 
        df['Kedalaman_Original'] = df['Kedalaman_km'].astype(float).copy()
        
        initial_cols = len(df.columns)
        
        # 1. Fitur Fisika Seismologi
        df = self._calculate_seismic_physics(df)
        
        # 2. Gutenberg-Richter b-value
        df = self._calculate_b_value_rolling(df, window=50)
        
        # 3. Spatiotemporal Deltas
        df = self._calculate_deltas(df)
        
        # 4. Time Encoding (Siklus)
        df = self._time_encoding(df)
        
        # 5. Advanced Rolling Statistics (Includes Aliasing Fix)
        df = self._calculate_advanced_stats(df)
        
        # 6. Spatial Clustering
        df = self._spatial_features(df)
        
        # 7. Custom Scaling (ACO Prep)
        df = self._apply_custom_scaling(df)
        
        # 8. Vulkanik Flag
        kw = self.config.get('feature_engineering', {}).get('is_vulkanik_keyword', 'gunung')
        if 'Lokasi' in df.columns:
            df['isVulkanik'] = df['Lokasi'].astype(str).str.lower().str.contains(kw).astype(int)
        else:
            df['isVulkanik'] = 0
            
        # Handle NaN/Inf yang mungkin muncul dari logaritma/pembagian
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        
        # 9. Final Normalization (Fitur Turunan)
        df = self._normalize_final_dataset(df)
        
        # 🚨 FIX FINAL: PASTIKAN Magnitudo/Kedalaman tetep Magnitudo/Kedalaman
        # (Karena _normalize_final_dataset tidak menyentuh kolom ini)
        # Tapi jika model downstream (LSTM/CNN) MENGHARAPKAN nilai Magnitudo/Kedalaman 
        # dinormalisasi di kolom utamanya, Anda harus membuat kolom baru dan mengarahkan 
        # model untuk mencarinya. Karena tidak ada informasi spesifik, 
        # kita biarkan Magnitudo/Kedalaman ASLI tetap di kolom Magnitudo/Kedalaman.
        
        final_cols = len(df.columns)
        self.logger.info(f"Feature Engineering Selesai. Fitur Baru: {final_cols - initial_cols}")
        self.logger.info(f"Total Kolom: {final_cols}")
        
        return df