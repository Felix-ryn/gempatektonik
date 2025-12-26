# ============================================================
# TectonicAI Main System (Daemon + Full Orchestration)
# FINAL & CLEAN VERSION — Integrasi Dashboard Ganda
# ============================================================

import logging            # Modul standar Python untuk sistem logging (INFO, WARNING, ERROR)
import time               # Digunakan untuk delay (sleep) dan menghitung durasi cycle
import os                 # Digunakan untuk operasi file & folder (mkdir, path)
from datetime import datetime  # Untuk manipulasi waktu & timestamp laporan
import pandas as pd       # Library utama untuk DataFrame (output pipeline, laporan, dashboard)

from TectonicAI.config import CONFIG      # Import konfigurasi global (path output, realtime config, dll)
from TectonicAI import orchestrator        # Import modul orchestrator sebagai pengendali seluruh pipeline

# Import Dashboard Generators
# Asumsi modul-modul ini sudah di-save di direktori TectonicAI/reporting/
from TectonicAI.reporting.live_monitor_generator import generate_live_monitor        # Fungsi pembuat dashboard realtime (1 data terbaru)
from TectonicAI.reporting.historical_analyzer import generate_historical_dashboard    # Fungsi pembuat dashboard historis (semua data)

# ----------------------------------------------------------
# LOGGER
# ----------------------------------------------------------
logging.basicConfig(       # Inisialisasi sistem logging global
    level=logging.INFO,    # Level logging: INFO (menampilkan info, warning, error)
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Format pesan log
    handlers=[logging.StreamHandler()]  # Output log ke terminal
)
logger = logging.getLogger("MAIN_SYSTEM")  # Logger khusus untuk file main system

# ----------------------------------------------------------
# FUNGSI HELPER: TEXT REPORT 
# ----------------------------------------------------------
def generate_text_report(df_result: pd.DataFrame, config: dict):  # Fungsi pembuat laporan teks singkat dari data terakhir
    """
    Menghasilkan laporan teks cepat dari baris data terakhir.  # Docstring penjelasan fungsi
    """
    if df_result.empty:                    # Cek jika DataFrame kosong
        logger.warning("Tidak ada data untuk laporan teks.")  # Warning jika tidak ada data
        return                             # Hentikan fungsi

    row = df_result.iloc[-1]               # Ambil 1 baris terakhir (data gempa terbaru)
     # Menggunakan output_paths dari config
    path = config["output_paths"].get("latest_report_txt", "output/LATEST_EARTHQUAKE_REPORT.txt")  # Ambil path file laporan

    content = f"""                          # Membuat isi laporan teks menggunakan f-string
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
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Membuat folder output jika belum ada
        with open(path, "w") as f:                          # Membuka file laporan
            f.write(content)                                # Menulis isi laporan
        logger.info(f"Report teks berhasil disimpan ke: {path}")  # Log sukses
    except Exception as e:
        logger.error(f"Gagal menyimpan report teks: {e}")   # Log error jika gagal


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
def main():                                                   # Fungsi utama daemon
    logger.info(">>> TectonicAI Realtime Mode Started <<<")  # Log saat sistem mulai

    while True:                                              # Loop realtime (daemon)
        start = time.time()                                  # Catat waktu mulai cycle

        try:
            result = run_cycle_wrapper()                     # Jalankan 1 cycle pipeline
            if result is not None:
                logger.info("Cycle OK.")
        except Exception as e:
            logger.error(f"Cycle failed: {e}")

        dur = time.time() - start                            # Hitung durasi cycle
        
        cycle_time = CONFIG.get("realtime_config", {}).get("cycle_time_sec", 60)  # Interval ideal
        min_sleep = CONFIG.get("realtime_config", {}).get("min_sleep_sec", 5)     # Minimal sleep
        
        sleep = max(min_sleep, cycle_time - int(dur))        # Hitung waktu sleep aman
        logger.info(f"Cycle time {dur:.1f}s — sleeping {sleep}s")
        time.sleep(sleep)                                    # Delay sebelum cycle berikutnya

if __name__ == "__main__":                                   # Entry point Python
    main()