# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def render_summary_cards(df, df_recent=None):
    # Menggunakan kode Unicode \U0001F4CA pengganti emoji grafik biar tidak '??'
    st.subheader("\U0001F4CA System Summary")
    
    # Hitung Metrics Utama
    total_records = len(df) if df is not None else 0
    anomaly_rate = 0.0
    avg_risk = 0.0
    
    if df is not None and not df.empty:
        # Konversi ke numeric untuk mencegah error perhitungan
        if 'is_Anomaly' in df.columns:
            anom_series = pd.to_numeric(df['is_Anomaly'], errors='coerce').fillna(0)
            anomaly_rate = anom_series.mean() * 100
            
        if 'Temporal_Risk_Factor' in df.columns:
            avg_risk = pd.to_numeric(df['Temporal_Risk_Factor'], errors='coerce').mean()

    # Hitung Metrics Recent
    recent_risk = 0.0
    if df_recent is not None and not df_recent.empty:
        if 'Temporal_Risk_Factor' in df_recent.columns:
            recent_risk = pd.to_numeric(df_recent['Temporal_Risk_Factor'], errors='coerce').mean()

    # Tampilkan Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data (2Y)", f"{total_records:,}")
    c2.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    c3.metric("Avg Risk (All)", f"{avg_risk:.3f}")
    c4.metric("Avg Risk (15d)", f"{recent_risk:.3f}", 
              delta=f"{recent_risk - avg_risk:.3f}", delta_color="inverse")