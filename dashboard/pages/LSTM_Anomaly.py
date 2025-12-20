import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Setup Path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
DASHBOARD_DIR = FILE_PATH.parent.parent
sys.path.append(str(DASHBOARD_DIR))

# Import Component
from components import render_anomaly_chart

st.set_page_config(page_title="LSTM Analysis", layout="wide", page_icon="📉")

DATA_PATH = PROJECT_ROOT / "output" / "lstm_results" / "lstm_output_2years.csv"

def main():
    st.title("📉 LSTM Temporal Anomaly Detection")

    if not DATA_PATH.exists():
        st.error(f"Data tidak ditemukan: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])

    # Gunakan Component Anomaly Chart
    st.subheader("Visualisasi Anomali")
    render_anomaly_chart(
        df, 
        date_col_candidates=['Tanggal'],
        risk_col_candidates=['Temporal_Risk_Factor'],
        anomaly_col_candidates=['is_Anomaly']
    )

    st.subheader("📋 Detail Data Anomali")
    anomalies = df[df['is_Anomaly'] == True]
    if anomalies.empty:
        st.info("Tidak ada anomali terdeteksi.")
    else:
        st.dataframe(anomalies[['Tanggal', 'Temporal_Risk_Factor', 'Context_Risk_Pheromone', 'Global_Tectonic_Stress']], use_container_width=True)

if __name__ == "__main__":
    main()