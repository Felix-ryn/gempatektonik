# ============================================================
# TectonicAI – Injection Loader (SAFE EDITION)
# File: TectonicAI/processing/injection_loader.py
#
# Fungsi:
# - Membaca file suntikan Excel/CSV
# - Validasi & normalisasi kolom
# - Only load data <= 90 hari terakhir (auto-filter)
# - Auto-clean missing/invalid values
# - Tidak mengubah RAW dataset
#
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Optional

class InjectionLoader:
    """
    Membaca data suntikan manual yang digunakan untuk:
    - Simulasi gempa
    - Data prediksi manual
    - Input manual ketika tidak ada data realtime

    Aturan bisnis:
    ----------------------
    1. Hanya data <= 90 hari terakhir yang diterima
    2. Jika file hilang → return DataFrame kosong
    3. Nama kolom wajib:
        - Tanggal
        - Lintang
        - Bujur
        - Magnitudo
        - Kedalaman_km
        - Lokasi
    """

    REQUIRED_COLS = [
        "Tanggal",
        "Lintang",
        "Bujur",
        "Magnitudo",
        "Kedalaman_km",
        "Lokasi"
    ]

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        logger = logging.getLogger("InjectionLoader")
        logger.setLevel(logging.INFO)

        # -------------------------------
        # 1. CEK FILE EXISTS
        # -------------------------------
        if not os.path.exists(file_path):
            logger.warning(f"[Injection] File suntikan tidak ditemukan: {file_path}")
            return pd.DataFrame()

        # -------------------------------
        # 2. LOAD FILE EXCEL/CSV
        # -------------------------------
        try:
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"[Injection] Gagal membaca file suntikan: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning("[Injection] File suntikan kosong.")
            return pd.DataFrame()

        # -------------------------------
        # 3. VALIDASI KOLUMNYA
        # -------------------------------
        for col in InjectionLoader.REQUIRED_COLS:
            if col not in df.columns:
                logger.error(f"[Injection] Kolom wajib hilang: {col}")
                return pd.DataFrame()

        # -------------------------------
        # 4. BERSIHKAN DATA
        # -------------------------------
        df = df.copy()

        # Konversi tanggal aman
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        df = df.dropna(subset=["Tanggal"])

        numeric_cols = ["Lintang", "Bujur", "Magnitudo", "Kedalaman_km"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        # -------------------------------
        # 5. FILTER 90 HARI TERAKHIR
        # -------------------------------
        today = datetime.now().date()
        cutoff = today - timedelta(days=90)

        df_90 = df[df["Tanggal"].dt.date >= cutoff]

        if df_90.empty:
            logger.info("[Injection] Semua data suntikan lebih tua dari 90 hari → diabaikan.")
            return pd.DataFrame()

        df_90 = df_90.sort_values("Tanggal").reset_index(drop=True)

        logger.info(f"[Injection] {len(df_90)} baris data suntikan valid dimuat (<=90 hari).")

        return df_90