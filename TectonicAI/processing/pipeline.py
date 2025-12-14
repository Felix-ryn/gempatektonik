# ============================================================
#  TECTONIC AI – MASTER PIPELINE (Titanium Orchestrator Edition)
#  File: TectonicAI/pipeline.py  (atau main.py)
#
#  PERAN:
#   - Menjalankan orkestrator utama
#   - Mengelola log global
#   - Menghasilkan dashboard output
#
#  Catatan:
#   Pipeline lama (FeatureEngineer, train_test_split, dll) tidak dipakai lagi.
#   Semua alur sekarang menggunakan sistem RAW+REALTIME+INJECTION → Orchestrator.
# ============================================================

import logging
import os
import json
import pandas as pd

# CONFIG
from TectonicAI.config import CONFIG

# ORCHESTRATOR UTAMA
from orchestrator import TectonicOrchestrator

# Dashboard generator (sudah ada dalam project kamu)
from TectonicAI.reporting.dashboard_generator import generate_dashboard


# ============================================================
# GLOBAL LOGGER
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("PIPELINE_MASTER")


# ============================================================
#  FUNGSI UTAMA PIPELINE
# ============================================================

def run_tectonic_system():

    logger.info("==============================================================")
    logger.info(">>> MEMULAI SISTEM TECTONIC AI — TITANIUM ORCHESTRATOR <<<")
    logger.info("==============================================================\n")

    # ----------------------------------------------------------
    # 1. Inisialisasi Orchestrator
    # ----------------------------------------------------------
    orchestrator = TectonicOrchestrator(CONFIG)

    # ----------------------------------------------------------
    # 2. Jalankan FULL PIPELINE
    # ----------------------------------------------------------
    df_final, summary = orchestrator.run_full_pipeline()

    if df_final.empty:
        logger.critical("Pipeline selesai TANPA DATA (empty dataframe).")
        return

    logger.info(f"[SUMMARY] Total baris final = {len(df_final)}")
    logger.info(f"[SUMMARY] Timestamp terakhir = {summary.get('last_timestamp')}")

    # ----------------------------------------------------------
    # 3. Simpan FULL CSV FINAL (untuk dashboard)
    # ----------------------------------------------------------
    out_csv = CONFIG['output_paths']['full_csv']
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df_final.to_csv(out_csv, index=False)
    logger.info(f"✔ Data Final disimpan ke: {out_csv}")

    # ----------------------------------------------------------
    # 4. Generate Dashboard
    # ----------------------------------------------------------
    logger.info(">>> MEMBANGUN DASHBOARD SYSTEM...")

    try:
        generate_dashboard(
            df=df_final,
            config=CONFIG,
            aco_results=summary.get("aco_state"),
            ga_results=summary.get("ga_vector"),
            lstm_results=summary.get("n_lstm_anomalies"),
            eval_results={}  # evaluation metrics JSON sudah disimpan terpisah
        )

        logger.info(f"✔ Dashboard selesai dihasilkan: {CONFIG['output_paths']['dashboard']}")
    except Exception as e:
        logger.error(f"[DASHBOARD ERROR] {e}")

    logger.info("\n==============================================================")
    logger.info(">>> SISTEM TECTONIC AI SELESAI (FULL SUCCESS) <<<")
    logger.info("==============================================================")

