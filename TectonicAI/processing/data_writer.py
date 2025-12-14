# ============================================================
# TectonicAI – DATA WRITER (SAFE EDITION)
# File: TectonicAI/processing/data_writer.py
#
# Fungsi:
# - Menyimpan RAW realtime
# - Menyimpan PROCESSED (hasil pipeline)
# - Backup harian otomatis
# - Anti-corrupt file
# - Anti-duplikasi (Tanggal + Koordinat)
# ============================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional

class DataWriter:

    def __init__(self, base_path: str):
        """
        base_path biasanya:
        TectonicAI/output/realtime/
        """
        self.base = base_path
        self.raw_path = os.path.join(base_path, "data_raw.csv")
        self.proc_path = os.path.join(base_path, "data_processed.csv")
        self.archive_dir = os.path.join(base_path, "archive")

        os.makedirs(self.base, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

        self.logger = logging.getLogger("DataWriter")
        self.logger.setLevel(logging.INFO)

    # ============================================================
    # HELPER: Sanitize DataFrame
    # ============================================================

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Convert Tanggal
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        df = df.dropna(subset=["Tanggal"])

        # Numeric columns
        numeric_cols = ["Lintang", "Bujur", "Magnitudo", "Kedalaman_km"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        # Remove inf
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        return df

    # ============================================================
    # WRITE RAW DATA
    # ============================================================

    def write_raw(self, df_new: pd.DataFrame):
        """
        Menyimpan data RAW realtime secara kumulatif tanpa kehilangan histori.
        """
        df_new = self._clean_df(df_new)

        # Load existing
        if os.path.exists(self.raw_path):
            try:
                df_old = pd.read_csv(self.raw_path)
                df_old = self._clean_df(df_old)
            except:
                self.logger.error("RAW corrupted → creating new file.")
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        df_all = pd.concat([df_old, df_new], ignore_index=True)

        df_all = df_all.drop_duplicates(
            subset=["Tanggal", "Lintang", "Bujur"], keep="last"
        ).sort_values("Tanggal")

        try:
            df_all.to_csv(self.raw_path, index=False)
            self.logger.info(f"RAW updated: {len(df_all)} rows.")
        except Exception as e:
            self.logger.error(f"RAW write failed: {e}")

    # ============================================================
    # WRITE PROCESSED DATA (90 HARI)
    # ============================================================

    def write_processed(self, df_final: pd.DataFrame):
        """
        Menyimpan data processed hasil pipeline:
        - 90 hari terakhir
        - Menjaga backup harian
        """
        df_final = self._clean_df(df_final)

        # 1. Filter hanya 90 hari terakhir
        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=90)
        df_90 = df_final[df_final["Tanggal"] >= cutoff].copy()

        # 2. Load old processed
        if os.path.exists(self.proc_path):
            try:
                old = pd.read_csv(self.proc_path)
                old["Tanggal"] = pd.to_datetime(old["Tanggal"], errors="coerce")
            except:
                old = pd.DataFrame()
        else:
            old = pd.DataFrame()

        # 3. Merge old + new
        df_all = pd.concat([old, df_90], ignore_index=True)
        df_all = df_all.drop_duplicates(
            subset=["Tanggal", "Lintang", "Bujur"], keep="last"
        ).sort_values("Tanggal")

        # 4. Backup ke folder archive dengan nama tanggal
        archive_name = os.path.join(
            self.archive_dir,
            f"{datetime.now().strftime('%Y%m%d')}_processed.csv"
        )
        try:
            df_all.to_csv(archive_name, index=False)
            self.logger.info(f"Backup saved: {archive_name}")
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")

        # 5. Save main processed
        try:
            df_all.to_csv(self.proc_path, index=False)
            self.logger.info(f"Processed updated: {len(df_all)} rows.")
        except Exception as e:
            self.logger.error(f"Processed write failed: {e}")