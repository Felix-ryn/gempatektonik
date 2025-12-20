import streamlit as st
import importlib
import sys
from pathlib import Path

# --- 1. KONFIGURASI PATH (Sangat Penting) ---
# Memastikan folder dashboard terbaca sebagai root modul
FILE_PATH = Path(__file__).resolve()
DASHBOARD_DIR = FILE_PATH.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.append(str(DASHBOARD_DIR))

# --- 2. CONFIG HALAMAN UTAMA ---
st.set_page_config(
    page_title="TectonicAI Dashboard",
    layout="wide",
    page_icon="🌋",
    initial_sidebar_state="expanded"
)

# --- 3. FUNGSI LOAD PAGE YANG AMAN (SOLUSI EROR ANDA) ---
def load_page(page_name):
    """
    Memuat halaman secara dinamis.
    Memperbaiki error 'parent pages not in sys.modules' dengan cek sys.modules dulu.
    """
    # Nama module lengkap, misal: pages.Overview
    module_path = f"pages.{page_name}"

    try:
        if module_path in sys.modules:
            # Jika sudah ada di memori, reload agar perubahan code terbaca
            return importlib.reload(sys.modules[module_path])
        else:
            # Jika belum ada, import baru
            return importlib.import_module(module_path)
    except ImportError as e:
        st.error(f"❌ Gagal memuat halaman **{page_name}**.")
        st.error(f"Detail Error: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan tidak terduga pada {page_name}: {e}")
        return None

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("🌋 TectonicAI")
st.sidebar.markdown("---")

# Daftar Halaman yang tersedia (sesuaikan nama file di folder pages tanpa .py)
PAGES = {
    "🏠 Overview": "Overview",
    "📉 LSTM Anomaly": "LSTM_Anomaly",
    "🐜 ACO Analysis": "ACO_Analysis",
    "🧬 GA Optimization": "GA_Analysis", 
    "🎯 CNN Prediction": "CNN_Prediction",
    "✅ Evaluation": "Evaluation"
}

selection = st.sidebar.radio("Navigasi", list(PAGES.keys()))
page_file = PAGES[selection]

# --- 5. EKSEKUSI HALAMAN ---
# Load modul halaman yang dipilih
page_module = load_page(page_file)

# Jalankan fungsi main() dari halaman tersebut jika berhasil di-load
if page_module:
    if hasattr(page_module, 'main'):
        page_module.main()
    else:
        st.warning(f"File `pages/{page_file}.py` tidak memiliki fungsi `main()`.")

# Footer Sidebar
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 TectonicAI Project")