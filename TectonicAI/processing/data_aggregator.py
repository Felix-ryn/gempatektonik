# ============================================================
# TectonicAI – Data Aggregator (SAFE EDITION)
# File: TectonicAI/processing/data_aggregator.py
#
# Fungsi:
# - Menggabungkan RAW, PROCESSED, dan INJECTION
# - Menjamin kolom konsisten
# - Menangani kasus kosong atau format tidak lengkap
# - Mengembalikan DataFrame final yang siap dipakai seluruh engine
#
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional

class DataAggregator:
    """
    Menggabungkan 3 sumber data:
    1. RAW realtime
    2. PROCESSED (hasil perhitungan sistem)
    3. INJECTION (manual input ≤ 90 hari)
    """

    REQUIRED_COLS = [
        "Tanggal",
        "Lintang",
        "Bujur",
        "Magnitudo",
        "Kedalaman_km",
        "Lokasi"
    ]

    OPTIONAL_DEFAULTS = {
        "rolling_mag_mean": 0.0,
        "rolling_event_count": 0.0,
        "Cluster": 0,
        "Pheromone_Score": 0.0,
        "Context_Risk_Pheromone": 0.0,
        "Context_Zone_IsRed": 0.0,
        "Context_Impact_Radius": 0.0,
        "Temporal_Risk_Factor": 0.0,
        "LSTM_Error": 0.0,
        "Model_Uncertainty": 0.0,
        "Proporsi_Grid_Terdampak_CNN": 0.0,
    }

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Pastikan kolom wajib ada dan tipe data aman."""

        df = df.copy()

        # Konversi tanggal aman
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

        numeric_cols = ["Lintang", "Bujur", "Magnitudo", "Kedalaman_km"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Tanggal"])       # tanggal wajib
        df = df.dropna(subset=numeric_cols)       # numeric wajib

        # Tambahkan kolom opsional jika belum ada
        for col, val in DataAggregator.OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = val
            df[col] = df[col].fillna(val)

        return df

    @staticmethod
    def merge(raw_df: pd.DataFrame,
              processed_df: Optional[pd.DataFrame],
              injection_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Menggabungkan ketiga data:
        1. RAW
        2. PROCESSED
        3. INJECTION (≤90 hari)

        Aturan:
        - Jika ada duplikasi tanggal & koordinat → injection menang
        - Jika processed ada tambahan kolom → tetap dipertahankan
        - Semua data kemudian diurutkan berdasarkan tanggal
        """

        logger = logging.getLogger("DataAggregator")
        logger.setLevel(logging.INFO)

        # Pastikan DataFrame tidak None
        raw = raw_df.copy() if raw_df is not None else pd.DataFrame()
        proc = processed_df.copy() if processed_df is not None else pd.DataFrame()
        inj = injection_df.copy() if injection_df is not None else pd.DataFrame()

        # Normalisasi semua dataset
        frames = []
        for name, df in [("RAW", raw), ("PROCESSED", proc), ("INJECTION", inj)]:
            if df is not None and not df.empty:
                df_n = DataAggregator._normalize(df)
                frames.append(df_n)
                logger.info(f"[Aggregator] {name}: {len(df_n)} baris dimuat.")
            else:
                logger.info(f"[Aggregator] {name}: kosong.")

        if not frames:
            logger.warning("[Aggregator] Semua sumber data kosong. Menghasilkan DF kosong.")
            return pd.DataFrame(columns=DataAggregator.REQUIRED_COLS)

        # Gabungkan semua data
        full_df = pd.concat(frames, ignore_index=True)

        # Hapus duplikasi berdasarkan Tanggal + Koordinat
        full_df = full_df.sort_values("Tanggal")
        full_df = full_df.drop_duplicates(
            subset=["Tanggal", "Lintang", "Bujur"],
            keep="last"   # injection selalu menang
        )

        full_df = full_df.reset_index(drop=True)

        logger.info(f"[Aggregator] Total final: {len(full_df)} baris.")

        return full_df
