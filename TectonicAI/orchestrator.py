# ============================================================
# TectonicAI Orchestrator – Full Single-Cycle Pipeline
# File: TectonicAI/orchestrator.py
# PERAN:
#   - Menjalankan 1x siklus penuh, memisahkan data Statis (Context) dan Historys (Train/Live).
#   - Mengembalikan:
#     df_final (DataFrame HANYA data live/dinamis terbaru)
#     summary (dict ringkasan untuk dashboard/monitoring)
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

# ============================================================
# IMPORT INTERNAL MODULES (Asumsi path sudah diatur)
# ============================================================
from TectonicAI.config import CONFIG
from TectonicAI.processing.data_handler import load_data
from TectonicAI.services.inaTEWS_api_client import InaTEWSApiClient

try:
    from TectonicAI.features.feature_engineer import FeatureEngineer
    from TectonicAI.engines.aco_engine import ACOEngine
    from TectonicAI.engines.ga_engine import GAEngine
    from TectonicAI.engines.lstm_engine import LSTMEngine
    from TectonicAI.engines.cnn_engine import CNNEngine
    from TectonicAI.engines.evaluation_engine import EvaluationEngine
except ImportError as e:
    print(f"[CRITICAL ERROR] Engine import gagal: {e}")
    # Placeholder classes jika modul belum ada saat testing isolasi
    print(f"[CRITICAL ERROR] Import Modules Gagal: {e}")
    # Menggunakan placeholder CONFIG jika import gagal
    CONFIG = {"paths": {"live_history_data": "data/Tectonic_Earthquake_live_history.csv", 
                        "static_baseline_data": "data/Tectonic_Earthquake_data.csv",
                        "injection_file": "data/data_injection.xlsx"}} 
    
# ------------------------------------------------------------
# LOGGER SETUP
# ------------------------------------------------------------
logger = logging.getLogger("TECTONIC_ORCHESTRATOR")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )

# ============================================================
# 1. HELPER FUNCTIONS (DATA MANAGEMENT)
# ============================================================

def _load_injection_file(path: str) -> pd.DataFrame:
    """Memuat data injeksi manual dan filter <= 90 hari."""
    if not path or not os.path.exists(path): return pd.DataFrame()
    try:
        # Gunakan pandas untuk membaca Excel/CSV
        if path.endswith('.xlsx'): df = pd.read_excel(path)
        else: df = pd.read_csv(path)
        
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
            cutoff = datetime.now() - timedelta(days=90)
            df = df[df["Tanggal"] >= cutoff]
            df.sort_values("Tanggal", inplace=True)
            
        logger.info(f"[INJECTION] Loaded {len(df)} record(s) from injection file (<=90 hari).")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"[INJECTION] Failed reading injection file: {e}")
        return pd.DataFrame()

def _archive_old_data(df: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    """Memindahkan data lawas (> 90 hari) ke file archive CSV."""
    if df.empty or "Tanggal" not in df.columns: return df.copy()
    
    archive_dir = os.path.join(base_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    df_proc = df.copy()
    df_proc["Tanggal"] = pd.to_datetime(df_proc["Tanggal"], errors="coerce")
    cutoff = datetime.now() - timedelta(days=90)
    mask_old = df_proc["Tanggal"] < cutoff
    df_old = df_proc[mask_old]
    df_recent = df_proc[~mask_old].copy() 
    
    if not df_old.empty:
        archive_file = os.path.join(archive_dir, "tectonic_history_archive.csv")
        try:
            header_status = not os.path.exists(archive_file)
            df_old.to_csv(archive_file, index=False, mode="a", header=header_status)
            logger.info(f"[ARCHIVE] {len(df_old)} record(s) moved to archive (>90 hari).")
        except Exception as e:
            logger.error(f"[ARCHIVE] Failed to archive data: {e}")
            
    df_recent.reset_index(drop=True, inplace=True)
    return df_recent

def _manage_live_history_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Menggabungkan Realtime (BMKG) dan Injection ke Live History Buffer.
    Ini adalah sumber data TRAINING UTAMA (Historys) dan LIVE TEST.
    """
    live_history_path = config["paths"]["live_history_data"]
    
    # 1. Load data Live History yang sudah ada (Buffer Historys)
    df_history_old = load_data(live_history_path) 
    if df_history_old is None: df_history_old = pd.DataFrame()
    
    # 2. Fetch data Dinamis (Realtime + Injection)
    df_realtime = InaTEWSApiClient().fetch_recent_data() 
    df_injection = _load_injection_file(config["paths"].get("injection_file", ""))
    
    # 3. Gabung dan Bersihkan
    df_list = [d for d in [df_history_old, df_realtime, df_injection] if d is not None and not d.empty]
    if not df_list: return pd.DataFrame()
        
    df_combined = pd.concat(df_list, ignore_index=True)

    if not df_combined.empty:
        df_combined["Tanggal"] = pd.to_datetime(df_combined["Tanggal"], errors="coerce")
        subset_cols = ["Tanggal", "Lintang", "Bujur"]
        existing_subset = [c for c in subset_cols if c in df_combined.columns]
        if existing_subset:
            df_combined.drop_duplicates(subset=existing_subset, keep="last", inplace=True)
        df_combined.sort_values("Tanggal", inplace=True)
        
    # Archive data > 90 hari dan filter data lama
    base_dir = os.path.dirname(live_history_path)
    df_clean_history = _archive_old_data(df_combined, base_dir)

    # 4. Simpan kembali Historys (Live Buffer) - KRITIS: ERROR HANDLING
    try:
        # Coba simpan file gabungan yang baru
        df_clean_history.to_csv(live_history_path, index=False)
        logger.info(f"[HISTORYS] Live History Data successfully SAVED. Rows: {len(df_clean_history)}")
        return df_clean_history.reset_index(drop=True)
        
    except Exception as e:
        # Jika gagal simpan (Permission Denied), log error dan gunakan data dari memory
        logger.error(f"[HISTORYS] Failed to save Live History: {e}")
        logger.warning("[HISTORYS] Save GAGAL. Menggunakan data dari memory untuk siklus ini. (Rekomendasi: Tutup file CSV)")
        return df_clean_history.reset_index(drop=True) # Return dari memory jika save gagal

def _load_static_context_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Memuat data Statis (Baseline/Context) untuk konteks ACO/GA/FE."""
    static_path = config["paths"]["static_baseline_data"]
    df_static = load_data(static_path)
    if df_static is None:
        logger.warning("[STATIC] Static Context data not found.")
        return pd.DataFrame()
    logger.info(f"[STATIC] Historical Data loaded from {static_path}. Rows: {len(df_static)}")
    return df_static

# ============================================================
# 2. CLASS TECTONIC ORCHESTRATOR
# ============================================================

class TectonicOrchestrator:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("TectonicOrchestrator")

    def run_full_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        summary = {"pipeline_status": "started", "start_time": str(datetime.now())}
        
        self.logger.info("=== STARTING TECTONIC AI HYBRID PIPELINE ===")

        # ======================================================
        # PHASE 1 – DATA LOAD, FUSION & SPLIT (Historys vs. Static Context)
        # ======================================================
        
        # 1. HISTORYS (LIVE BUFFER) - SUMBER UTAMA UNTUK TRAINING/FE
        df_historys_train = _manage_live_history_data(self.config)
        
        # 2. DATA STATIS (BASELINE/CONTEXT) - UNTUK ACO/GA KONTEKS DASAR
        df_static_context = _load_static_context_data(self.config)
        
        if df_historys_train.empty:
            self.logger.critical("[ORCH] Historys (Live Buffer) kosong. Pipeline Aborted.")
            summary["pipeline_status"] = "failed_no_live_data"
            return pd.DataFrame(), summary
            
        # --- Fusion untuk Feature Engineering (Memastikan Konteks FE Lengkap) ---
        df_static_context['Data_Source'] = 'STATIC_CONTEXT'
        df_historys_train['Data_Source'] = 'LIVE_HISTORYS'
        
        # Concat Statis + Historys (Statis sebagai konteks lama)
        df_full_combined = pd.concat([df_static_context, df_historys_train], ignore_index=True)
        # Hati-hati: Duplikasi harus dilakukan setelah tagging
        df_full_combined.drop_duplicates(subset=["Tanggal", "Lintang", "Bujur"], keep="last", inplace=True)
        df_full_combined.sort_values("Tanggal", inplace=True)
        df_full_combined.reset_index(drop=True, inplace=True)
        df_dynamic = df_full_combined.copy() # Data yang akan diproses Engines

        # 3. Feature Engineering di SELURUH DATA
        try:
            fe = FeatureEngineer(self.config)
            df_dynamic = fe.transform(df_dynamic)
            self.logger.info(f"[ORCH] Feature Engineering Done. Cols: {df_dynamic.shape[1]}")
        except Exception as e:
            self.logger.critical(f"[ORCH] FE Failed: {e}")
            summary["pipeline_status"] = "failed_fe"
            return df_dynamic, summary


        # 4. Tentukan Train/Test Split (Berdasarkan Tag 'LIVE_HISTORYS')

        df_history_only = df_dynamic[df_dynamic['Data_Source'] == 'LIVE_HISTORYS'].copy()

        n_history_live = len(df_history_only)

        # 1. Tentukan Test Index (Selalu baris terakhir dari data Historys)
        if n_history_live == 0:
            test_idx = pd.Index([])
        elif n_history_live == 1:
            test_idx = df_history_only.index
        else:
            test_idx = df_history_only.index[-1:]

        # 2. Train Index: Semua Index yang BUKAN Test Index
        all_indices = df_dynamic.index
        train_idx = all_indices.difference(test_idx)

        # ======================================================
        # [FIX KRITIS] Konversi index LABEL → POSITIONAL INDEX
        # ======================================================

        # 🔧 SIMPAN mapping SEBELUM reset_index
        label_to_pos = {idx: pos for pos, idx in enumerate(df_dynamic.index)}

        # Reset agar iloc aman
        df_dynamic = df_dynamic.reset_index(drop=True)

        # Konversi ke numpy positional index
        train_idx = np.array(
            [label_to_pos[i] for i in train_idx if i in label_to_pos],
            dtype=int
        )

        test_idx = np.array(
            [label_to_pos[i] for i in test_idx if i in label_to_pos],
            dtype=int
        )

        # ======================================================

        n_static_train = len(train_idx)
        n_dynamic_live = len(test_idx)

        self.logger.info(
            f"[ORCH] Split Context: Total={len(df_dynamic)} | "
            f"Train(Historys+Static)={n_static_train} | Test(Live)={n_dynamic_live}"
        )

        if n_static_train == 0:
            self.logger.critical(
                "[ORCH] Train Set (Historis/Context) kosong. Model tidak akan dilatih."
            )
    
        # ======================================================
        # PHASE 2-5: ENGINE EXECUTION
        # ======================================================
        
        # --- A. ACO (Metaheuristik Spasial) ---
        try:
            aco = ACOEngine(self.config.get("aco_path_model", {}))
            # Hati-hati: ACO/GA dijalankan pada df_dynamic (data gabungan)
            df_dynamic, aco_meta = aco.run(df_dynamic.copy())
            summary["aco_meta"] = aco_meta
            self.logger.info("[ORCH] ACO finished.")
        except Exception as e:
            self.logger.error(f"[ORCH] ACO Error: {e}")
            if "Pheromone_Score" not in df_dynamic.columns: df_dynamic["Pheromone_Score"] = 0.0

        # --- B. GA (Metaheuristik Vektor) ---
        try:
            ga = GAEngine(self.config.get("ga_vector_model", {}))
            df_dynamic, ga_meta = ga.run(df_dynamic.copy())
            summary["ga_meta"] = ga_meta
            self.logger.info("[ORCH] GA finished.")
        except Exception as e:
            self.logger.error(f"[ORCH] GA Error: {e}")

        # --- C. LSTM (Temporal Anomaly) ---
        try:
            lstm_path = self.config.get("lstm_anomaly_model", {})
            lstm = LSTMEngine(lstm_path)
            train_context = df_dynamic.loc[train_idx] if not train_idx.empty else pd.DataFrame()
            df_dynamic, lstm_res = lstm.run(df_dynamic, train_context)
            summary["lstm_anomalies_count"] = len(lstm_res.get("anomalies", []))
            self.logger.info("[ORCH] LSTM finished.")
        except Exception as e:
            self.logger.error(f"[ORCH] LSTM Error: {e}")
            for col in ["Temporal_Risk_Factor", "Model_Uncertainty", "is_Anomaly"]:
                if col not in df_dynamic.columns: df_dynamic[col] = 0.0 if col != "is_Anomaly" else False

        # --- D. CNN (Spatial Analysis) ---
        try:
            cnn_path = self.config.get("cnn_hybrid_model", {})
            cnn = CNNEngine(cnn_path)
            df_dynamic = cnn.train_and_predict(df_dynamic, train_idx, test_idx)
            self.logger.info("[ORCH] CNN finished.")
        except Exception as e:
            self.logger.error(f"[ORCH] CNN Error: {e}")
            if "Proporsi_Grid_Terdampak_CNN" not in df_dynamic.columns: df_dynamic["Proporsi_Grid_Terdampak_CNN"] = 0.0

        # --- E. Evaluation (Final Ensembling) ---
        try:
            evaluator = EvaluationEngine(self.config)
            df_dynamic = evaluator.run(df_dynamic, train_idx, test_idx)
            self.logger.info("[ORCH] Evaluation & Probability Fusion finished.")
        except Exception as e:
            self.logger.critical(f"[ORCH] Evaluation Engine Error: {e}")
            if "Final_Prob_Impact" not in df_dynamic.columns: df_dynamic["Final_Prob_Impact"] = 0.0


        # ======================================================
        # PHASE 6: FINAL OUTPUT (Hanya Data Live Terbaru)
        # ======================================================
        
        df_final_live = df_dynamic.loc[test_idx].copy()
        
        last_ts = "N/A"
        if "Tanggal" in df_final_live.columns and not df_final_live.empty:
            last_ts = str(df_final_live["Tanggal"].iloc[-1])
            
        summary["last_data_timestamp"] = last_ts
        summary["total_processed_rows"] = len(df_final_live)
        summary["pipeline_status"] = "success"
        
        # Tambahkan df_dynamic FULL ke summary untuk Dashboard Historis
        summary["df_dynamic_full"] = df_dynamic 

        self.logger.info(f"=== PIPELINE COMPLETE. Returning {len(df_final_live)} live records. ===")
        
        return df_final_live, summary


# ============================================================
# 3. DIRECT EXECUTION (ENTRY POINT)
# ============================================================

def run_full_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Wrapper function untuk memanggil orkestrator dari luar tanpa inisialisasi manual.
    """
    # Menggunakan asumsi CONFIG loaded
    orchestrator = TectonicOrchestrator(CONFIG)
    return orchestrator.run_full_pipeline()

if __name__ == "__main__":
    # Test Run
    df_result, res_summary = run_full_pipeline()
    print("Execution Summary:", res_summary)