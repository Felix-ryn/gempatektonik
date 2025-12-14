# ============================================================
# TectonicAI Main System (Daemon + Full Orchestration)
# FINAL & CLEAN VERSION — Integrasi Dashboard Ganda
# ============================================================

import logging
import time
import os
from datetime import datetime
import pandas as pd

from TectonicAI.config import CONFIG
from TectonicAI import orchestrator

# Import Dashboard Generators
# Asumsi modul-modul ini sudah di-save di direktori TectonicAI/reporting/
from TectonicAI.reporting.live_monitor_generator import generate_live_monitor
from TectonicAI.reporting.historical_analyzer import generate_historical_dashboard 

# ----------------------------------------------------------
# LOGGER
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MAIN_SYSTEM")

# ----------------------------------------------------------
# FUNGSI HELPER: TEXT REPORT 
# ----------------------------------------------------------
def generate_text_report(df_result: pd.DataFrame, config: dict):
    """
    Menghasilkan laporan teks cepat dari baris data terakhir.
    """
    if df_result.empty:
        logger.warning("Tidak ada data untuk laporan teks.")
        return

    row = df_result.iloc[-1]
    # Menggunakan output_paths dari config
    path = config["output_paths"].get("latest_report_txt", "output/LATEST_EARTHQUAKE_REPORT.txt") 
    
    content = f"""
========================= TECTONIC AI REPORT =========================
Waktu Pemrosesan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tanggal Gp    : {row.get('Tanggal', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}
Lokasi        : {row.get('Lokasi','-')}
Magnitude     : {row.get('Magnitudo',0):.2f}
Kedalaman     : {row.get('Kedalaman_km',0):.1f} km

Zonasi ACO    : {row.get('Status_Zona','UNKNOWN')}
LSTM          : {'ANOMALI' if row.get('is_Anomaly') else 'NORMAL'}
CNN Impact    : {row.get('Proporsi_Grid_Terdampak_CNN',0)*100:.2f} %
Final Prob    : {row.get('Final_Prob_Impact',0)*100:.1f} %
======================================================================
"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Report teks berhasil disimpan ke: {path}")
    except Exception as e:
        logger.error(f"Gagal menyimpan report teks: {e}")


# ----------------------------------------------------------
# 5. FULL ORCHESTRATOR WRAPPER (1 cycle only)
# ----------------------------------------------------------
def run_cycle_wrapper():
    """
    Memanggil Orkestrator Utama (TectonicOrchestrator) yang berisi 
    semua logika pipeline, dan memicu pembuatan dashboard.
    """
    # 1. Jalankan Orchestrator (Mengembalikan df_live dan summary PENUH)
    # df_live = HANYA 1 BARIS TERBARU
    # summary = Metadata dari semua engine + df_dynamic_full
    df_live, summary = orchestrator.run_full_pipeline() 

    if df_live.empty:
        logger.warning("Orkestrator mengembalikan DataFrame kosong.")
        return None

    # 2. Tambahkan Logika Text Report
    generate_text_report(df_live, CONFIG)
    
    # 3. Panggil Dashboard Live Monitor (Fokus pada 1 data terbaru)
    try:
        # Perbaiki panggilan dengan get() untuk menghindari KeyError jika kunci config hilang
        generate_live_monitor(df_live, CONFIG, summary) 
        logger.info("Dashboard Live Monitor berhasil diperbarui.")
    except Exception as e:
        # Log spesifik untuk debugging KeyError pada konfigurasi
        logger.error(f"Gagal membuat Live Monitor: {e}")
        
    # 4. Panggil Dashboard Historis (Menggunakan df_dynamic_full dari summary)
    df_dynamic_full = summary.get("df_dynamic_full")
    if df_dynamic_full is not None and not df_dynamic_full.empty:
        try:
            generate_historical_dashboard(
                df_dynamic_full, 
                CONFIG, 
                summary, 
                eval_results=summary.get("eval_metrics", {})
            )
            logger.info("Dashboard Historis berhasil diperbarui.")
        except Exception as e:
            logger.error(f"Gagal membuat Dashboard Historis: {e}")
    
    return df_live


# ----------------------------------------------------------
# 6. REALTIME LOOP (DAEMON)
# ----------------------------------------------------------
def main():
    logger.info(">>> TectonicAI Realtime Mode Started <<<")

    while True:
        start = time.time()

        try:
            result = run_cycle_wrapper() 
            if result is not None:
                logger.info("Cycle OK.")
        except Exception as e:
            logger.error(f"Cycle failed: {e}")

        dur = time.time() - start
        
        cycle_time = CONFIG.get("realtime_config", {}).get("cycle_time_sec", 60)
        min_sleep = CONFIG.get("realtime_config", {}).get("min_sleep_sec", 5)
        
        sleep = max(min_sleep, cycle_time - int(dur)) 
        logger.info(f"Cycle time {dur:.1f}s — sleeping {sleep}s")
        time.sleep(sleep)

if __name__ == "__main__":
    main()