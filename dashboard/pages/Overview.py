import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import sys
import os
from pathlib import Path

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="TectonicAI Command Center", layout="wide", page_icon="🌋")

# --- 1. SETUP PATH SYSTEM ---
FILE_PATH = Path(__file__).resolve()
ROOT_A = FILE_PATH.parent.parent.parent 
ROOT_B = Path(r"C:\Project Joki\gempatektonik")

if (ROOT_A / "output").exists():
    PROJECT_ROOT = ROOT_A
elif (ROOT_B / "output").exists():
    PROJECT_ROOT = ROOT_B
else:
    PROJECT_ROOT = Path(os.getcwd())

sys.path.append(str(PROJECT_ROOT))

# --- 2. DEFINISI FILE PATH ---
BASE_OUT = PROJECT_ROOT / "output"

FILES = {
    # ACO Files
    "aco_map": BASE_OUT / "aco_results" / "aco_impact_zones.html",
    "aco_json": BASE_OUT / "aco_results" / "aco_summary.json",
    # GA Files
    "ga_map": BASE_OUT / "ga_results" / "ga_vector_map.html",
    "ga_csv": BASE_OUT / "ga_results" / "ga_best_chromosome.csv",
    # LSTM Files
    "lstm_csv": BASE_OUT / "lstm_results" / "lstm_output_2years.csv",
    "lstm_recent": BASE_OUT / "lstm_results" / "lstm_recent_15_days.csv",
    # CNN Files
    "cnn_csv": BASE_OUT / "cnn_results" / "cnn_next_earthquake_prediction.csv"
}

# --- 3. FUNGSI LOAD DATA ---
def load_dashboard_data():
    data = {
        "aco_html": None, "aco_stats": {},
        "ga_html": None, "ga_stats": {},
        "lstm_df": None, "recent_risk": 0.0,
        "cnn_stats": {}, 
        "debug_log": []
    }

    # A. LOAD ACO MAP
    if FILES["aco_map"].exists():
        try:
            with open(FILES["aco_map"], 'r', encoding='utf-8') as f:
                data["aco_html"] = f.read()
        except:
            try: 
                with open(FILES["aco_map"], 'r', encoding='latin-1') as f:
                    data["aco_html"] = f.read()
            except Exception as e:
                data["debug_log"].append(f"Gagal baca ACO Map: {e}")

    # B. LOAD ACO STATS
    if FILES["aco_json"].exists():
        try:
            with open(FILES["aco_json"], 'r') as f:
                data["aco_stats"] = json.load(f)
        except: pass

    # C. LOAD GA MAP
    if FILES["ga_map"].exists():
        try:
            with open(FILES["ga_map"], 'r', encoding='utf-8') as f:
                data["ga_html"] = f.read()
        except:
            try:
                with open(FILES["ga_map"], 'r', encoding='latin-1') as f:
                    data["ga_html"] = f.read()
            except: pass

    # D. LOAD GA DATA
    if FILES["ga_csv"].exists():
        try:
            df = pd.read_csv(FILES["ga_csv"])
            if not df.empty:
                latest = df.iloc[-1]
                data["ga_stats"] = {
                    "angle": float(latest.get("Angle_Deg", 0)),
                    "distance": float(latest.get("Distance", 0)),
                    "fitness": float(latest.get("Fitness", 0))
                }
        except: pass

    # E. LOAD LSTM
    if FILES["lstm_csv"].exists():
        try:
            df = pd.read_csv(FILES["lstm_csv"])
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            data["lstm_df"] = df
        except: pass

    if FILES["lstm_recent"].exists():
        try:
            df = pd.read_csv(FILES["lstm_recent"])
            if 'Temporal_Risk_Factor' in df.columns:
                data["recent_risk"] = df['Temporal_Risk_Factor'].mean()
        except: pass

    # F. LOAD CNN (Membaca output real dari Simple CNN Engine)
    if FILES["cnn_csv"].exists():
        try:
            df = pd.read_csv(FILES["cnn_csv"])
            if not df.empty:
                latest = df.iloc[-1] # Ambil prediksi paling baru
                data["cnn_stats"] = {
                    "arah_str": latest.get("arah_prediksi", "Unknown"),
                    "sudut": float(latest.get("arah_derajat", 0.0)),
                    "conf": float(latest.get("confidence_scalar", 0.0))
                }
        except Exception as e:
            data["debug_log"].append(f"Gagal baca CNN CSV: {e}")

    return data

def main():
    st.title("🌋 TectonicAI: Integrated Command Center")
    st.markdown("Dashboard ini menyatukan hasil analisis **Spasial (ACO)**, **Vektor (GA)**, dan **Temporal (LSTM/CNN)** dalam satu tampilan terpadu.")
    
    if st.button("🔄 Refresh Semua Data"):
        st.rerun()

    d = load_dashboard_data()

    # --- DEBUGGING ---
    if d["debug_log"]:
        with st.expander("⚠️ System Path Debugging"):
            for log in d["debug_log"]:
                st.error(log)

    # --- KPI GLOBAL ---
    st.subheader("📊 Status Sistem Real-time")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # ACO KPI
    radius = d['aco_stats'].get('impact_radius_km', 0)
    kpi1.metric("Radius Dampak (ACO)", f"{radius:.1f} KM", delta="Zona Bahaya")
    
    # GA KPI
    ga_angle = d['ga_stats'].get('angle', 0)
    kpi2.metric("Arah Optimasi (GA)", f"{ga_angle:.1f}°", delta="Vektor Optimal")
    
    # LSTM KPI
    risk = d['recent_risk']
    risk_label = "CRITICAL" if risk > 0.6 else ("WARNING" if risk > 0.3 else "STABLE")
    kpi3.metric("Risiko Temporal (15 Hari)", f"{risk:.4f}", delta=risk_label, delta_color="inverse")
    
    # CNN KPI (UPDATE)
    cnn_data = d.get('cnn_stats', {})
    cnn_dir = cnn_data.get('arah_str', '-')
    kpi4.metric("Prediksi CNN", cnn_dir, help="Arah perambatan selanjutnya")

    st.markdown("---")

    # =========================================================================
    # BAGIAN 1: ANALISIS SPASIAL (ACO)
    # =========================================================================
    st.header("1. Analisis Zona Bahaya (ACO)")
    col_aco_map, col_aco_info = st.columns([2, 1])
    
    with col_aco_map:
        if d['aco_html']:
            components.html(d['aco_html'], height=500, scrolling=True)
        else:
            st.error("Peta ACO tidak dapat dimuat.")

    with col_aco_info:
        st.subheader("📋 Detail Zona")
        center_lat = d['aco_stats'].get('center_latitude', 0)
        center_lon = d['aco_stats'].get('center_longitude', 0)
        st.write(f"**Titik Pusat (Centroid):**")
        st.code(f"Lat: {center_lat:.5f}\nLon: {center_lon:.5f}")
        st.write(f"**Luas Area Terdampak:**")
        st.metric("Area (KM²)", f"{d['aco_stats'].get('impact_area_km2', 0):,.0f}")
        st.success(f"Total Events: **{d['aco_stats'].get('total_events', 0)}**")

    st.markdown("---")

    # =========================================================================
    # BAGIAN 2: ANALISIS VEKTOR (GA)
    # =========================================================================
    st.header("2. Optimasi Arah Pergerakan (GA)")
    col_ga_map, col_ga_compass = st.columns([2, 1])

    with col_ga_map:
        if d['ga_html']:
            components.html(d['ga_html'], height=500, scrolling=True)
        else:
            st.error("Peta GA tidak dapat dimuat.")

    with col_ga_compass:
        st.subheader("🧭 Visualisasi Arah (GA)")
        fig_ga = go.Figure(go.Scatterpolar(
            r=[0, 1], theta=[0, ga_angle],
            mode='lines+markers',
            marker=dict(symbol="arrow-bar-up", size=20, color="blue"),
            line=dict(color="blue", width=5),
            name="GA Vector"
        ))
        fig_ga.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(direction="clockwise", rotation=90)
            ),
            showlegend=False, height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            title=f"Azimuth: {ga_angle:.2f}°"
        )
        st.plotly_chart(fig_ga, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # BAGIAN 3: TEMPORAL & PREDIKSI (LSTM & CNN) - UPDATED
    # =========================================================================
    st.header("3. Tren Waktu (LSTM) & Prediksi Future (CNN)")
    col_lstm, col_cnn = st.columns([2, 1])

    with col_lstm:
        st.subheader("📈 Tren Risiko (2 Tahun Terakhir)")
        if d['lstm_df'] is not None:
            df_chart = d['lstm_df'].sort_values('Tanggal')
            fig_lstm = px.line(df_chart, x='Tanggal', y='Temporal_Risk_Factor', title="Indeks Risiko Seismik")
            anomalies = df_chart[df_chart['is_Anomaly'] == 1]
            if not anomalies.empty:
                fig_lstm.add_scatter(
                    x=anomalies['Tanggal'], y=anomalies['Temporal_Risk_Factor'],
                    mode='markers', name='ANOMALI', marker=dict(color='red', size=10, symbol='x')
                )
            if df_chart['Temporal_Risk_Factor'].max() < 0.1:
                fig_lstm.update_layout(yaxis_range=[-0.05, 1.0])
            st.plotly_chart(fig_lstm, use_container_width=True)
        else:
            st.warning("Data LSTM belum tersedia.")

    # [BAGIAN YANG ANDA MINTA DIMASUKKAN ADA DI SINI]
    with col_cnn:
        st.markdown("### 🧠 CNN Forecast (Spatio-Temporal)")
        
        # Load data CNN dari dictionary 'd' yang sudah diload di atas
        cnn_data = d.get('cnn_stats', {})
        
        if cnn_data:
            cnn_angle = cnn_data.get('sudut', 0.0)
            cnn_dir_str = cnn_data.get('arah_str', "Menunggu Data")
            cnn_conf = cnn_data.get('conf', 0.0)
            
            # Metric Display
            st.metric(
                label="Prediksi Arah Gempa",
                value=f"{cnn_dir_str}",
                delta=f"{cnn_angle:.1f}°"
            )
            
            # Simple Arrow Visualization
            fig_cnn = go.Figure(go.Scatterpolar(
                r=[0, 1], theta=[0, cnn_angle],
                mode='lines+markers',
                marker=dict(symbol="arrow-bar-up", size=15, color="#FF4B4B"),
                line=dict(color="#FF4B4B", width=4),
                name="Arah"
            ))
            fig_cnn.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False),
                    angularaxis=dict(
                        tickvals=[0, 90, 180, 270],
                        ticktext=['U', 'T', 'S', 'B'],
                        rotation=90, direction="clockwise"
                    )
                ),
                showlegend=False, height=200,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_cnn, use_container_width=True)
            
            st.caption(f"Confidence: {cnn_conf:.1%} | Input: H-1 & H Vectors")
            
        else:
            st.warning("Data CNN belum tersedia.")
            st.info("Jalankan engine untuk memicu prediksi.")

if __name__ == "__main__":
    main()