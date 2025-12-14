# TectonicAI/config.py
import os
from pathlib import Path

# Menentukan root project (asumsi config.py ada di TectonicAI/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder Utama
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "TectonicAI" / "assets"

# Sub-folder Output (Wajib ada untuk kerapian file output engine)
ACO_RESULTS_DIR = OUTPUT_DIR / "aco_results"
GA_RESULTS_DIR = OUTPUT_DIR / "ga_results"
LSTM_RESULTS_DIR = OUTPUT_DIR / "lstm_results"
CNN_RESULTS_DIR = OUTPUT_DIR / "cnn_results"

CONFIG = {
    # --- PATHS INPUT ---
    "paths": {
        "static_baseline_data": str(DATA_DIR / "Tectonic_Earthquake_data.csv"),
        "live_history_data": str(DATA_DIR / "Tectonic_Earthquake_live_history.csv"),
        "injection_file": str(DATA_DIR / "data_injection.xlsx"),
        "assets_dir": str(ASSETS_DIR),
        "output_base": str(OUTPUT_DIR)
    },

    # --- OUTPUT PATHS (Jembatan Data & Visualisasi) ---
    "output_paths": {
        # Dashboard Final & Laporan
        "dashboard": str(OUTPUT_DIR / "STRATEGIC_RISK_DASHBOARD.html"),
        "full_csv": str(OUTPUT_DIR / "full_analysis_results.csv"),
        "metrics_json": str(OUTPUT_DIR / "system_metrics.json"),
        "latest_report_txt": str(OUTPUT_DIR / "LATEST_EARTHQUAKE_REPORT.txt"),
        "dashboard_live_monitor": "output/dashboards/TectonicAI_Monitor_LIVE.html", 
        "dashboard_historical": "output/dashboards/TectonicAI_Historical_Context.html",
        
        # ACO Output (Zoning)
        "aco_zoning_excel": str(ACO_RESULTS_DIR / "aco_zoning_data_for_lstm.xlsx"),
        "aco_state_file": str(ACO_RESULTS_DIR / "aco_brain_state.pkl"), # Memori Semut (Incremental)
        
        # GA Output (Evolution)
        "ga_prediction_excel": str(GA_RESULTS_DIR / "ga_prediction_data_for_lstm.xlsx"),
        "ga_state": str(GA_RESULTS_DIR / "ga_state.pkl"),
        
        # Deep Learning Output
        "lstm_to_cnn": str(LSTM_RESULTS_DIR / "lstm_data_for_cnn.xlsx"), # Jembatan LSTM ke CNN
    },

    # --- DATA PREPARATION ---
    "data_split": {
        "test_size": 0.3,
        "random_state": 42,
        "shuffle_data": False # CRITICAL: False agar urutan waktu tidak rusak
    },

    "feature_engineering": {
        "is_vulkanik_keyword": "gunung",
        "rolling_window_days": 14 
    },

    "realtime_config": {
        # Waktu siklus maksimal (detik) untuk Daemon/main.py. Diatur 60 detik (1 menit).
        "cycle_time_sec": 60,
        # Minimum waktu tidur (detik)
        "min_sleep_sec": 5 
    },

    # --- KONFIGURASI ENGINE AI (Sesuai parameter Anda) ---
    "aco_path_model": {
        "n_ants": 50, "n_iterations": 50, "n_epicenters": 15,
        "alpha": 1.5, "beta": 2.5, "evaporation_rate": 0.4,
        "pheromone_deposit": 100.0, "risk_threshold": 0.75
    },

    "ga_vector_model": {
        "population_size": 100, "generations": 150, "mutation_rate": 0.015,
        "magnitude_threshold": 4.5, "n_islands": 4, "migration_rate": 0.1
    },

    "lstm_anomaly_model": {
        "sequence_length": 15, "epochs": 60, "units": 128,
        "anomaly_threshold_std": 3.0, "mc_samples": 20
    },

    "cnn_hybrid_model": {
        "grid_size": 64, "epochs": 40, "batch_size": 16
    },

    "evaluation_model": {
        "models": ["RandomForestClassifier", "NaiveBayes"],
        "primary_model": "RandomForestClassifier"
    },

    "dashboard": {
        "map_center": [-7.5, 112.5],
        "zoom_start": 7,
    }
}